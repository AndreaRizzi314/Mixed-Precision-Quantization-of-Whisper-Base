"""Per-matrix packed weight export; activations remain floating point."""
import shutil
from pathlib import Path

BIT_MODES = {2: "INT2_SYM", 3: "INT3_SYM", 4: "INT4_ASYM", 8: "INT8_ASYM"}
LOCKED_COMPONENTS = {"encoder_conv1", "encoder_conv2", "decoder_token_embedding", "decoder_output_projection"}
GRAPH_FILES = {"encoder": "openvino_encoder_model.xml", "decoder": "openvino_decoder_model.xml"}


def operations(inventory: dict) -> list[dict]:
    rows = [{"graph": node["graph"], "name": node["name"], "component": component,
             "locked": component in LOCKED_COMPONENTS,
             "parameter": parameter_name(node)}
            for component, nodes in inventory["groups"].items() for node in nodes]
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicated OpenVINO operation names")
    return sorted(rows, key=lambda row: (row["graph"], row["name"]))

def parameter_name(node: dict) -> str:
    prefix = node["name"].split("/", 1)[0]
    if not prefix.startswith("__module."):
        raise ValueError(f"Unexpected OpenVINO node name: {prefix}")
    tail = prefix.removeprefix("__module.")
    if tail == "proj_out":
        return "model.decoder.embed_tokens.weight"  # Tied output projection.
    if node["graph"] == "encoder":
        tail = "model.encoder." + tail
    return tail + ".weight"

def fake_quantize(weight, bits: int, group_size: int = 64):
    """Approximate NNCF's grouped round-to-nearest error for sensitivity ranking."""
    import torch

    original_shape = weight.shape
    matrix = weight.detach().float().reshape(original_shape[0], -1)
    width = matrix.shape[1]
    pad = (-width) % group_size
    if pad:
        matrix = torch.nn.functional.pad(matrix, (0, pad))
    blocks = matrix.reshape(matrix.shape[0], -1, group_size)
    if bits == 8:
        blocks = matrix.reshape(matrix.shape[0], 1, -1)
    if bits == 4 or bits == 8:
        lo = blocks.amin(-1, keepdim=True)
        hi = blocks.amax(-1, keepdim=True)
        scale = ((hi - lo) / ((1 << bits) - 1)).clamp_min(1e-10)
        restored = ((blocks - lo) / scale).round().clamp(0, (1 << bits) - 1) * scale + lo
    else:
        qmax = (1 << (bits - 1)) - 1
        scale = (blocks.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-10)
        restored = (blocks / scale).round().clamp(-qmax - 1, qmax) * scale
    return restored.reshape(matrix.shape[0], -1)[:, :width].reshape(original_shape)

def export_assignment(source: Path, target: Path, inventory: dict,
                      assignments: dict[str, int], group_size: int = 64, *,
                      four_mode: str = "INT4_ASYM", method: str | None = None,
                      calibration_datasets: dict | None = None) -> dict:
    """Compress separate sets of named operations at 2/3/4/8 bits."""
    import nncf
    import openvino as ov

    if four_mode not in ("INT4_ASYM", "INT4_SYM") or method not in (None, "awq", "gptq"):
        raise ValueError("Unsupported INT4 quantizer or correction method")
    if method and not calibration_datasets:
        raise ValueError("Data-aware compression requires graph calibration datasets")

    rows = operations(inventory)
    expected = {row["name"] for row in rows if not row["locked"]}
    if set(assignments) != expected or set(assignments.values()) - set(BIT_MODES):
        raise ValueError("Bit assignment does not cover exactly the tunable operations")
    desired = {row["name"]: (8 if row["locked"] else assignments[row["name"]]) for row in rows}
    target.mkdir(parents=True, exist_ok=True)
    base = source / "models" / "ov_fp32"
    for file in base.glob("*.json"):
        shutil.copy2(file, target / file.name)
    core = ov.Core()
    for graph, filename in GRAPH_FILES.items():
        model = core.read_model(str(base / filename))
        graph_names = {row["name"] for row in rows if row["graph"] == graph}
        for bits in (4, 8):
            names = {name for name in graph_names if desired[name] == bits}
            if not names:
                continue
            ignored = sorted(graph_names - names)
            kwargs = {"mode": getattr(nncf.CompressWeightsMode, four_mode if bits == 4 else BIT_MODES[bits]),
                      "ignored_scope": nncf.IgnoredScope(names=ignored, validate=True)
                      if ignored else None}
            if bits != 8:
                kwargs.update(group_size=group_size, ratio=1.0, all_layers=True)
            if bits == 4 and method:
                data = calibration_datasets[graph]
                kwargs.update(dataset=nncf.Dataset(data), subset_size=len(data))
                kwargs[method] = True
            model = nncf.compress_weights(model, **kwargs)
        low_precision = {}
        for bits in (3, 2):
            names = {name for name in graph_names if desired[name] == bits}
            if not names:
                continue
            ignored = sorted(graph_names - names)
            low_precision[bits] = nncf.compress_weights(
                model.clone(), mode=getattr(nncf.CompressWeightsMode, BIT_MODES[bits]),
                group_size=group_size, ratio=1.0, all_layers=True,
                ignored_scope=nncf.IgnoredScope(names=ignored, validate=True))
        if len(low_precision) == 2:
            # NNCF cannot parse a graph already containing packed u3 or u2 data.
            # Create the two disjoint sets from the same 4/8-bit graph and graft
            # each 2-bit weight expression onto the otherwise 3-bit graph.
            model = low_precision[3]
            base_nodes = {node.get_friendly_name(): node for node in model.get_ops()}
            two_nodes = {node.get_friendly_name(): node for node in low_precision[2].get_ops()}
            for name in graph_names:
                if desired[name] != 2:
                    continue
                left, right = base_nodes[name], two_nodes[name]
                weight_index = 0 if left.get_type_name() == "Gather" else 1
                left.input(weight_index).replace_source_output(
                    right.input(weight_index).get_source_output())
            model.validate_nodes_and_infer_types()
        elif low_precision:
            model = next(iter(low_precision.values()))
        ov.save_model(model, str(target / filename), compress_to_fp16=False)
    from .graphs import serialized_bytes
    observed = effective_bits(target, rows)
    for row in rows:
        actual = observed[row["name"]]
        wanted = desired[row["name"]]
        if actual != wanted:
            raise RuntimeError(f"{row['name']} requested {wanted} bits, saved {actual} bits")
    return {"serialized_bytes": serialized_bytes(target), "effective_bits": observed,
            "bit_counts": {str(bit): sum(x == bit for x in observed.values()) for bit in BIT_MODES}}

def effective_bits(target: Path, rows: list[dict]) -> dict[str, int | None]:
    import openvino as ov

    core = ov.Core()
    result = {}
    for graph, filename in GRAPH_FILES.items():
        model = core.read_model(str(target / filename))
        named = {node.get_friendly_name(): node for node in model.get_ops()}
        for row in rows:
            if row["graph"] != graph:
                continue
            node = named[row["name"]]
            source_input = 0 if node.get_type_name() == "Gather" else 1
            visited = set()
            producer = node.input(source_input).get_source_output().get_node()
            for _ in range(14):
                if producer.get_type_name() == "Constant":
                    kind = producer.output(0).get_element_type().get_type_name()
                    result[row["name"]] = next((bit for bit in BIT_MODES
                                               if kind in (f"u{bit}", f"i{bit}")), None)
                    break
                if producer.get_friendly_name() in visited or producer.get_input_size() == 0:
                    break
                visited.add(producer.get_friendly_name())
                producer = producer.input(0).get_source_output().get_node()
            if row["name"] not in result:
                raise RuntimeError(f"Could not trace packed weight for {row['name']}")
    return result
