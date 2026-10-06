"""Whisper graph inventory, FP32 export, and uniform INT8 control."""
import json
import shutil
from pathlib import Path

COMPONENTS = (
    "encoder_conv1", "encoder_conv2", "encoder_attention_qkv", "encoder_attention_output",
    "encoder_ffn_expand", "encoder_ffn_contract",
    "decoder_self_qkv", "decoder_self_output", "decoder_cross_qkv", "decoder_cross_output",
    "decoder_ffn_expand", "decoder_ffn_contract", "decoder_token_embedding",
    "decoder_output_projection",
)
GRAPH_FILES = {"encoder": "openvino_encoder_model.xml", "decoder": "openvino_decoder_model.xml"}


def component_for_op(graph: str, op_type: str, name: str) -> str | None:
    if graph == "encoder":
        if op_type == "Convolution":
            if "__module.conv1/" in name:
                return "encoder_conv1"
            if "__module.conv2/" in name:
                return "encoder_conv2"
        if op_type == "MatMul":
            if ".self_attn." in name:
                return ("encoder_attention_output" if ".out_proj/" in name
                        else "encoder_attention_qkv" if any(f".{p}_proj/" in name for p in "qkv") else None)
            if ".fc1/" in name:
                return "encoder_ffn_expand"
            if ".fc2/" in name:
                return "encoder_ffn_contract"
    if graph == "decoder":
        if op_type == "MatMul":
            if ".self_attn." in name or ".encoder_attn." in name:
                prefix = "decoder_self" if ".self_attn." in name else "decoder_cross"
                if ".out_proj/" in name:
                    return f"{prefix}_output"
                if any(f".{p}_proj/" in name for p in "qkv"):
                    return f"{prefix}_qkv"
            if ".fc1/" in name:
                return "decoder_ffn_expand"
            if ".fc2/" in name:
                return "decoder_ffn_contract"
            if "__module.proj_out/" in name:
                return "decoder_output_projection"
        if op_type == "Gather" and ".embed_tokens/" in name:
            return "decoder_token_embedding"
    return None

def graph_inventory(base_dir: Path) -> dict:
    import openvino as ov

    result = {component: [] for component in COMPONENTS}
    seen = []
    core = ov.Core()
    for graph, filename in GRAPH_FILES.items():
        model = core.read_model(str(base_dir / filename))
        for node in model.get_ops():
            op_type = node.get_type_name()
            name = node.get_friendly_name()
            group = component_for_op(graph, op_type, name)
            if group:
                result[group].append({"graph": graph, "op_type": op_type, "name": name})
                seen.append(name)
    empty = [g for g, nodes in result.items() if not nodes]
    if empty or len(seen) != len(set(seen)):
        raise RuntimeError(f"Unmatched or duplicate graph component nodes: empty={empty}")
    return {"groups": result, "node_count": len(seen),
            "group_node_count": {g: len(nodes) for g, nodes in result.items()}}

def serialized_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and p.name != "variant.json")

def export_fp32(config: dict, root: Path) -> dict:
    from optimum.intel.openvino import OVModelForSpeechSeq2Seq
    env = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    target = root / "models" / "ov_fp32"
    if not (target / GRAPH_FILES["encoder"]).exists():
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            config["model_id"], revision=env["model_revision"], export=True, compile=False)
        target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(target)
    inventory = graph_inventory(target)
    (root / "graph_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return {"path": str(target), "serialized_bytes": serialized_bytes(target), "inventory": inventory}

def compress_variant(root: Path, label: str, preserve: set[str], *,
                     mode: str = "INT8_ASYM", group_size: int | None = None,
                     ratio: float | None = None,
                     output_root: Path | None = None) -> dict:
    """Compress eligible weights except nodes in `preserve`; save real IR."""
    import nncf
    import openvino as ov

    unknown = preserve - set(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown components: {sorted(unknown)}")
    if not hasattr(nncf.CompressWeightsMode, mode):
        raise ValueError(f"Unknown weight compression mode: {mode}")
    base_dir = root / "models" / "ov_fp32"
    target = (output_root or root) / "models" / label
    metadata_path = target / "variant.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (set(metadata["preserved_components"]) != preserve or
                metadata.get("mode", "INT8_ASYM") != mode or
                metadata.get("group_size") != group_size or
                metadata.get("ratio") != ratio):
            raise RuntimeError(f"Existing variant {label} has different compression settings")
        return metadata
    target.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((root / "graph_inventory.json").read_text(encoding="utf-8"))
    for item in base_dir.iterdir():
        if item.is_file() and item.suffix == ".json":
            shutil.copy2(item, target / item.name)
    core = ov.Core()
    for graph, filename in GRAPH_FILES.items():
        model = core.read_model(str(base_dir / filename))
        ignored = [node["name"] for group in preserve for node in inventory["groups"][group]
                   if node["graph"] == graph]
        scope = nncf.IgnoredScope(names=ignored, validate=True) if ignored else None
        compressed = nncf.compress_weights(
            model, mode=getattr(nncf.CompressWeightsMode, mode),
            group_size=group_size, ratio=(ratio if ratio is not None else
                                          1.0 if mode != "INT8_ASYM" else None),
            ignored_scope=scope)
        ov.save_model(compressed, str(target / filename), compress_to_fp16=False)
    metadata = {"label": label, "preserved_components": sorted(preserve),
                "serialized_bytes": serialized_bytes(target), "path": str(target),
                "mode": mode, "group_size": group_size, "ratio": ratio,
                "compression": f"NNCF {mode} weight-only with named node exclusions",
                "excluded_nodes": [node for group in preserve for node in inventory["groups"][group]]}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
