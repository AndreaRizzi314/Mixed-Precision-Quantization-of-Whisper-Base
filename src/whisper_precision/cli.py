"""Search saved Whisper Base weight precision and confirm two frozen candidates.

Commands resume from immutable records. Set up source models and development
audio with `python -m whisper_precision.prepare` first. Outputs stay under artifacts/.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import random
import statistics
import traceback
from collections import Counter
from pathlib import Path

from whisper_precision.io import checksum, read_json
from whisper_precision.data import choose_rows_by_duration, duration_band
from whisper_precision.quantization import export_assignment, operations
from whisper_precision.search import (ablations, aggregate, assert_disjoint, beam_proposals,
    candidate_id, choose_finalists, evaluate_cached, file_digest, immutable, model_digest,
    passes, save, shortlist, specification, subset_result)
from whisper_precision.graphs import serialized_bytes
from whisper_precision.paths import repository_root


@contextlib.contextmanager
def export_log(path: Path):
    import logging
    import nncf  # Initialize NNCF's logger before redirecting stdout.
    logger = logging.getLogger("nncf")
    previous = logger.handlers[:]
    with path.open("w", encoding="utf-8") as handle:
        handler = logging.StreamHandler(handle)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
        logger.handlers = [handler]
        try:
            with contextlib.redirect_stdout(handle):
                yield
        finally:
            logger.handlers = previous


def context(config: dict) -> tuple[Path, Path, dict]:
    root, source = Path(config["output_dir"]), Path(config["source_root"])
    root.mkdir(parents=True, exist_ok=True)
    env = read_json(source / "environment.json")
    if env["model_revision"] != config["model_revision"]:
        raise RuntimeError("The source model revision differs from the pinned checkpoint")
    inventory = read_json(source / "graph_inventory.json")
    immutable(root / "config.json", config)
    provenance = {"config_sha256": checksum(config), "source_model_sha256":
                  model_digest(source / "models" / "ov_fp32"), "int8_model_sha256":
                  model_digest(source / "models" / "uniform_int8")}
    immutable(root / "provenance.json", provenance)
    return root, source, inventory


def prepare(config: dict, root: Path) -> None:
    """Reserve both tests in one deterministic pass over pinned test-clean."""
    if (root / "reservations.json").exists():
        verify_reservations(config, root)
        return
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset
    from whisper_precision.inference import environment

    development = [read_json(Path(p)) for p in config["development_manifests"]]
    assert_disjoint(*development)
    clips = sorted((x for m in development for x in m["clips"]), key=lambda x: x["id"])
    if len(clips) != 180:
        raise RuntimeError("Expected exactly 180 previously evaluated development clips")
    for clip in clips:
        if not Path(clip["audio"]).exists():
            raise FileNotFoundError(clip["audio"])
    dev = {"name": "development", "clips": clips, "count": len(clips),
           "role": "all source samples are now development, including earlier test clips",
           "source_manifests": config["development_manifests"]}
    screen = {"name": "screen", "clips": development[0]["clips"], "count": 72,
              "source_manifest": config["development_manifests"][0]}
    immutable(root / "development_manifest.json", dev)
    immutable(root / "screen_manifest.json", screen)
    exclusions = [read_json(Path(p)) for p in config["clean_exclusion_manifests"]]
    excluded = {x["id"] for m in exclusions for x in m["clips"]}
    targets = {band: config["test_a_targets"][band] + config["test_b_targets"][band]
               for band in ("short", "medium", "long")}
    if not (root / "test_a_manifest.json").exists() or not (root / "test_b_manifest.json").exists():
        data = load_dataset(config["dataset_id"], "clean", split="test", streaming=True,
                            revision=config["dataset_revision"], cache_dir=str(root / "cache")).decode(False)
        rows, scanned, available = choose_rows_by_duration(data, targets, config["seed"], excluded)
        groups = {band: [] for band in targets}
        for row in rows:
            band = duration_band(sf.info(io.BytesIO(row["audio"]["bytes"])).duration)
            groups[band].append(row)
        rng = random.Random(config["seed"] + 17)
        separated = {"test_a": [], "test_b": []}
        for band, band_rows in groups.items():
            rng.shuffle(band_rows)
            n = config["test_a_targets"][band]
            separated["test_a"].extend(band_rows[:n])
            separated["test_b"].extend(band_rows[n:])
        for name, chosen in separated.items():
            selected = []
            for row in sorted(chosen, key=lambda x: x["id"]):
                samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                if sr != 16000 or samples.ndim != 1:
                    raise RuntimeError("Expected mono 16 kHz LibriSpeech audio")
                target = root / "data" / name / f"{row['id']}.flac"
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(target, samples, sr)
                selected.append({"id": str(row["id"]), "speaker_id": row["speaker_id"],
                    "text": row["text"], "audio": str(target), "seconds": len(samples)/sr,
                    "duration_band": duration_band(len(samples)/sr), "audio_sha256": file_digest(target)})
            manifest = {"name": name, "dataset_id": config["dataset_id"],
                "dataset_revision": config["dataset_revision"], "dataset_config": "clean",
                "split": "test", "count": len(selected), "clips": selected,
                "duration_targets": config[f"{name}_targets"], "rows_scanned": scanned,
                "available_by_band": available, "excluded_manifests": config["clean_exclusion_manifests"],
                "excluded_id_count": len(excluded)}
            immutable(root / f"{name}_manifest.json", manifest)
            print(f"Reserved {name}: {len(selected)} unused clips", flush=True)
    a, b = (read_json(root / f"{n}_manifest.json") for n in ("test_a", "test_b"))
    assert_disjoint(dev, a, b)
    if excluded.intersection(x["id"] for m in (a, b) for x in m["clips"]):
        raise RuntimeError("A reserved clean set overlaps previous test clips")
    immutable(root / "reservations.json", {"config_sha256": checksum(config),
        "development_sha256": checksum(dev), "screen_sha256": checksum(screen),
        "test_a_sha256": checksum(a), "test_b_sha256": checksum(b),
        "status": "reserved_before_phase5_search", "clean_only_gate": True})
    if not (root / "environment.json").exists():
        fresh_environment = environment(config)
        save(root / "environment.json", fresh_environment)
    verify_reservations(config, root)


def verify_reservations(config: dict, root: Path, *, check_audio: bool = False) -> dict:
    frozen = read_json(root / "reservations.json")
    if frozen["config_sha256"] != checksum(config):
        raise RuntimeError("Reserved study configuration changed")
    manifests = {}
    for name in ("development", "screen", "test_a", "test_b"):
        value = read_json(root / f"{name}_manifest.json")
        if checksum(value) != frozen[f"{name}_sha256"]:
            raise RuntimeError(f"Reserved {name} changed")
        manifests[name] = value
        if check_audio:
            for clip in value["clips"]:
                if "audio_sha256" in clip and file_digest(Path(clip["audio"])) != clip["audio_sha256"]:
                    raise RuntimeError(f"Reserved audio changed: {clip['id']}")
    assert_disjoint(manifests["development"], manifests["test_a"], manifests["test_b"])
    integrity_path = root / "audio_integrity.json"
    if integrity_path.exists() and check_audio:
        for audio, expected in read_json(integrity_path)["files"].items():
            if file_digest(Path(audio)) != expected:
                raise RuntimeError(f"Frozen audio contents changed: {audio}")
    return manifests


def freeze_audio(root: Path) -> None:
    path = root / "audio_integrity.json"
    if not path.exists():
        manifests = [read_json(root / f"{n}_manifest.json") for n in ("development", "test_a", "test_b")]
        immutable(path, {"files": {c["audio"]: file_digest(Path(c["audio"]))
                    for m in manifests for c in m["clips"]},
            "reservations_sha256": checksum(read_json(root / "reservations.json"))})


def calibration(config: dict, root: Path, source: Path) -> dict:
    """Actual FP32 encoder outputs and teacher-forced decoder prefixes, dev only."""
    import numpy as np
    import openvino as ov
    import soundfile as sf
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(config["model_id"],
        revision=config["model_revision"], local_files_only=True)
    processor.tokenizer.set_prefix_tokens(language="english", task="transcribe")
    clips = read_json(root / "screen_manifest.json")["clips"]
    selected = [next(x for x in clips if x["duration_band"] == band) for band in ("short", "long")]
    core = ov.Core()
    encoder = core.compile_model(str(source / "models" / "ov_fp32" / "openvino_encoder_model.xml"), "CPU")
    dataset = {"encoder": [], "decoder": []}
    for clip in selected:
        audio, sr = sf.read(clip["audio"], dtype="float32")
        features = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="np").input_features
        hidden = encoder({"input_features": features})[encoder.output(0)]
        labels = processor.tokenizer(clip["text"], return_tensors="np").input_ids
        # Whisper shifts labels right with decoder_start_token_id, retaining prompts.
        prefix = np.concatenate([np.array([[50258]], dtype=np.int64), labels[:, :-1]], axis=1)
        dataset["encoder"].append({"input_features": features})
        dataset["decoder"].append({"input_ids": prefix, "encoder_hidden_states": hidden,
                                   "beam_idx": np.array([0], dtype=np.int32)})
    save(root / "calibration.json", {"clip_ids": [x["id"] for x in selected],
        "method": "FP32 hidden states and teacher-forced decoder inputs; development only",
        "screen_manifest_sha256": checksum(read_json(root / "screen_manifest.json"))})
    return dataset


def activation_ranking(config: dict, root: Path) -> None:
    """Diagonal reconstruction Hessian 2*E[x^2], not a full task Hessian."""
    path = root / "activation_scores.json"
    if path.exists():
        return
    import torch
    import soundfile as sf
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    from whisper_precision.quantization import fake_quantize
    torch.set_num_threads(8)
    prior = read_json(Path("artifacts/layerwise/sensitivity.json"))
    processor = AutoProcessor.from_pretrained(config["model_id"], revision=config["model_revision"], local_files_only=True)
    processor.tokenizer.set_prefix_tokens(language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(config["model_id"], revision=config["model_revision"], local_files_only=True)
    model.eval()
    inputs_squared, counts, handles = {}, {}, []
    parameters = {r["parameter"] for r in prior["operations"].values()}
    def hook(name):
        def collect(module, values, result):
            x = values[0].detach().float().reshape(-1, values[0].shape[-1])
            square = x.square().sum(0)
            inputs_squared[name] = inputs_squared.get(name, torch.zeros_like(square)) + square
            counts[name] = counts.get(name, 0) + x.shape[0]
        return collect
    for name, module in model.named_modules():
        if name + ".weight" in parameters:
            handles.append(module.register_forward_hook(hook(name + ".weight")))
    clips = read_json(root / "screen_manifest.json")["clips"]
    chosen = [next(c for c in clips if c["duration_band"] == b) for b in ("short", "long")]
    with torch.inference_mode():
        for clip in chosen:
            audio, sr = sf.read(clip["audio"], dtype="float32")
            features = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features
            labels = processor.tokenizer(clip["text"], return_tensors="pt").input_ids
            model(input_features=features, labels=labels)
        weights = dict(model.named_parameters())
        rows = {}
        for name, old in prior["operations"].items():
            param = old["parameter"]
            diagonal = 2 * inputs_squared[param] / counts[param]
            weight = weights[param].detach()
            options = {str(bit): {**old["options"][str(bit)], "proxy_loss":
                      float(((fake_quantize(weight, bit) - weight.float()).square() * diagonal[None, :]).sum())}
                      for bit in (2, 3, 4, 8)}
            rows[name] = {**old, "options": options}
    for handle in handles:
        handle.remove()
    save(path, {"method": "input-activation diagonal reconstruction Hessian times weight error",
        "definition": "2 E[x^2] per input channel; local linear reconstruction, not a full task Hessian",
        "clip_ids": [c["id"] for c in chosen], "screen_manifest_sha256": checksum(read_json(root / "screen_manifest.json")),
        "operations": rows})
    print("Calibrated activation-weighted matrix ranking on two development clips", flush=True)


def preflight(config: dict, root: Path, source: Path, inventory: dict) -> None:
    verify_reservations(config, root)
    path = root / "preflight.json"
    status = read_json(path) if path.exists() else {}
    # Saving an unchanged graph can alter generated XML node names. Compare the
    # actual BIN payloads, not XML hashes, to detect a correction silently skipped.
    for method in ("awq", "gptq"):
        if status.get(method, {}).get("supported") and status.get("plain", {}).get("supported"):
            plain = root / "preflight_models" / "plain"
            alternate = root / "preflight_models" / method
            if all(file_digest(plain / name) == file_digest(alternate / name)
                   for name in ("openvino_encoder_model.bin", "openvino_decoder_model.bin")):
                status[method].update({"supported": False,
                    "error": f"{method.upper()} was skipped: both weight payloads equal plain INT4"})
                save(path, status)
    movable = {r["name"]: r for r in operations(inventory) if not r["locked"]}
    assignment = {name: 4 for name in movable}
    manifest = read_json(root / "screen_manifest.json")
    clips = [next(x for x in manifest["clips"] if x["duration_band"] == band) for band in ("short", "long")]
    probe = {"name": "preflight", "clips": clips, "count": 2}
    data = calibration(config, root, source)
    activation_ranking(config, root)
    for method in ("plain", "awq", "gptq"):
        if method in status:
            continue
        target = root / "preflight_models" / method
        log = root / "logs" / f"preflight_{method}.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with export_log(log):
                meta = export_assignment(source, target, inventory, assignment,
                    method=None if method == "plain" else method,
                    calibration_datasets=None if method == "plain" else data)
            sha = model_digest(target)
            if method != "plain" and all(file_digest(target / name) == file_digest(root / "preflight_models" / "plain" / name)
                                        for name in ("openvino_encoder_model.bin", "openvino_decoder_model.bin")):
                raise RuntimeError(f"{method.upper()} produced no model changes; calibration refinement was skipped")
            result = evaluate_cached(config, root, f"preflight_{method}", probe, target, sha)
            status[method] = {"supported": True, "metadata": meta, "model_sha256": sha,
                              "smoke_wer": result["wer"], "prediction_count": len(result["predictions"])}
            print(f"Preflight {method}: export, packed types, and generation passed", flush=True)
        except Exception as error:
            status[method] = {"supported": False, "error": f"{type(error).__name__}: {error}",
                              "traceback": traceback.format_exc(), "log": str(log)}
            print(f"Preflight {method}: {status[method]['error']}", flush=True)
            if method == "plain":
                save(path, status)
                raise
        save(path, status)


def assert_tests_unread(root: Path) -> None:
    if any((root / name).exists() for name in ("test_results.json", "outcome.json")) or any(
        (root / "results" / split).exists() and any((root / "results" / split).glob("*.json*"))
        for split in ("test_a", "test_b")
    ):
        raise RuntimeError("Fresh test predictions already exist without a frozen selection; do not revise this search")


class Study:
    def __init__(self, config: dict, root: Path, source: Path, inventory: dict):
        self.config, self.root, self.source, self.inventory = config, root, source, inventory
        self.manifests = verify_reservations(config, root)
        freeze_audio(root)
        self.provenance = read_json(root / "provenance.json")
        self.catalog_path = root / "candidates.json"
        self.catalog = read_json(self.catalog_path) if self.catalog_path.exists() else {}
        self.screen_path = root / "screen_results.json"
        self.screen = read_json(self.screen_path) if self.screen_path.exists() else {}
        self.full_path = root / "development_results.json"
        self.full = read_json(self.full_path) if self.full_path.exists() else {}
        self.movable = {r["name"]: r for r in operations(inventory) if not r["locked"]}
        self.sens = read_json(root / "activation_scores.json")["operations"]
        self.datasets = None
        self.limit = min(config["maximum_bytes"], serialized_bytes(source / "models" / "uniform_int8"))

    def controls(self) -> None:
        for label in ("ov_fp32", "uniform_int8"):
            if label not in self.full:
                print(f"Development control: {label}, 180 clips", flush=True)
                sha = self.provenance["source_model_sha256" if label == "ov_fp32" else "int8_model_sha256"]
                self.full[label] = evaluate_cached(self.config, self.root, label,
                    self.manifests["development"], self.source / "models" / label, sha)
                save(self.full_path, self.full)
            self.screen[label] = subset_result(self.full[label], self.manifests["screen"]["clips"])
        save(self.screen_path, self.screen)

    def add(self, family: str, reason: str, spec: dict) -> str | None:
        label = candidate_id(spec)
        if label in self.catalog:
            return None
        if len(self.catalog) >= self.config["maximum_candidates"]:
            raise RuntimeError("The registered 96-candidate cap was reached")
        self.catalog[label] = {"family": family, "reason": reason, "spec": spec,
                               "spec_sha256": checksum(spec), "status": "pending"}
        save(self.catalog_path, self.catalog)
        return label

    def run(self, label: str) -> None:
        record = self.catalog[label]
        if record["status"] in ("measured", "failed", "too_large"):
            return
        target = self.root / "models" / label
        metadata_path = self.root / "allocations" / f"{label}.json"
        if not metadata_path.exists():
            import shutil
            target.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(target.parent).free < 150_000_000:
                raise OSError("Insufficient model-export disk space: need at least 150 MB free; candidate remains pending")
        try:
            spec = record["spec"]
            if metadata_path.exists():
                meta = read_json(metadata_path)
                if meta["spec_sha256"] != checksum(spec) or meta["source_sha256"] != self.provenance["source_model_sha256"]:
                    raise RuntimeError("Export metadata is inconsistent")
                if model_digest(target) != meta["model_sha256"]:
                    raise RuntimeError("Saved candidate model changed")
            else:
                log = self.root / "logs" / f"{label}.txt"
                log.parent.mkdir(parents=True, exist_ok=True)
                if spec["method"] and self.datasets is None:
                    self.datasets = calibration(self.config, self.root, self.source)
                with export_log(log):
                    meta = export_assignment(self.source, target, self.inventory,
                        spec["assignments"], spec["group_size"], four_mode=spec["four_mode"],
                        method=spec["method"], calibration_datasets=self.datasets if spec["method"] else None)
                meta.update({"spec_sha256": checksum(spec), "source_sha256": self.provenance["source_model_sha256"],
                             "model_sha256": model_digest(target)})
                save(metadata_path, meta)
            record["metadata"] = meta
            if meta["serialized_bytes"] >= self.limit:
                record["status"] = "too_large"
                record["error"] = "Actual saved size is not smaller than uniform INT8"
            else:
                value = evaluate_cached(self.config, self.root, label, self.manifests["screen"],
                                        target, meta["model_sha256"])
                self.screen[label] = value
                save(self.screen_path, self.screen)
                record["status"] = "measured"
                print(f"{len([r for r in self.catalog.values() if r['status']=='measured'])}/96 {record['family']} "
                      f"{label}: {meta['serialized_bytes']/1e6:.2f} MB, {100*value['wer']:.2f}% screen WER", flush=True)
        except Exception as error:
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            record["traceback"] = traceback.format_exc()
            print(f"Candidate {label} failed: {record['error']}", flush=True)
        save(self.catalog_path, self.catalog)

    def family_count(self, family: str) -> int:
        return sum(r["family"] == family for r in self.catalog.values())

    def proposals(self, family: str, proposed: list[tuple[str, dict]]) -> None:
        for reason, spec in proposed:
            if self.family_count(family) >= self.config["family_limits"][family]:
                break
            label = self.add(family, reason, spec)
            if label:
                self.run(label)

    def rows(self) -> dict:
        return {k: v for k, v in self.screen.items() if k in self.catalog and
                self.catalog[k]["status"] == "measured"}

    def top(self, count: int) -> list[str]:
        return shortlist(self.rows(), count, self.screen["ov_fp32"]["wer"])

    def search(self) -> None:
        if (self.root / "selection.json").exists():
            verify_selection(self.config, self.root)
            return
        assert_tests_unread(self.root)
        self.controls()
        for label in list(self.catalog):
            if self.catalog[label]["status"] == "pending":
                self.run(label)
        self.proposals("ablation", ablations(self.movable))
        batches_path = self.root / "beam_batches.json"
        batches = read_json(batches_path) if batches_path.exists() else {"batches": []}
        while self.family_count("beam") < 30:
            pending = next((b for b in batches["batches"] if not b["complete"]), None)
            if pending is None:
                proposals = beam_proposals(self.catalog, self.rows(), self.movable,
                    set(self.catalog), self.screen["ov_fp32"]["wer"], min(6, 30-self.family_count("beam")))
                if not proposals:
                    raise RuntimeError("Component search exhausted unexpectedly")
                pending = {"complete": False, "proposals": proposals,
                           "measured_screen_sha256": checksum(self.screen)}
                batches["batches"].append(pending)
                save(batches_path, batches)
            self.proposals("beam", pending["proposals"])
            pending["complete"] = True
            save(batches_path, batches)
        anchors_path = self.root / "matrix_anchors.json"
        if not anchors_path.exists():
            save(anchors_path, {"labels": self.top(3)})
        anchors = read_json(anchors_path)["labels"]
        proposals = []
        for label in anchors:
            spec = self.catalog[label]["spec"]
            assignment = spec["assignments"]
            up, down = [], []
            for name, bits in assignment.items():
                options = self.sens[name]["options"]
                cost = self.sens[name]["parameters"] * 4 / 8
                change = (options["4"]["proxy_loss"] - options["8"]["proxy_loss"]) / max(cost, 1)
                (up if bits == 4 else down).append((name, change))
            # Three rescues and three downgrades; fill from the available side.
            ranked_up = sorted(up, key=lambda x: (-x[1], x[0]))
            ranked_down = sorted(down, key=lambda x: (x[1], x[0]))
            chosen = ranked_up[:3] + ranked_down[:3]
            for item in ranked_up[3:] + ranked_down[3:]:
                if len(chosen) >= 6:
                    break
                chosen.append(item)
            for name, _ in chosen:
                mutated = dict(assignment); mutated[name] = 8 if assignment[name] == 4 else 4
                proposals.append((f"single_matrix_flip from {label}: {name}", specification(mutated)))
        self.proposals("matrix", proposals)
        self.fill_family("matrix")
        anchors_path = self.root / "quantizer_anchors.json"
        if not anchors_path.exists():
            save(anchors_path, {"labels": self.top(4)})
        preflight = read_json(self.root / "preflight.json")
        variants = [(32, "INT4_ASYM", None), (16, "INT4_ASYM", None),
                    (64, "INT4_ASYM", "awq") if preflight["awq"]["supported"] else (64, "INT4_SYM", None),
                    (64, "INT4_ASYM", "gptq") if preflight["gptq"]["supported"] else (32, "INT4_SYM", None)]
        proposals = [(f"quantizer refinement of {label}", specification(self.catalog[label]["spec"]["assignments"], g, m, method))
                     for label in read_json(anchors_path)["labels"] for g, m, method in variants]
        self.proposals("quantizer", proposals)
        self.fill_family("quantizer")
        anchors_path = self.root / "lowbit_anchor.json"
        if not anchors_path.exists():
            ranked = sorted(self.rows(), key=lambda k: (max(0, self.screen[k]["wer"] / self.screen["ov_fp32"]["wer"] - 1.02),
                                                       self.screen[k]["serialized_bytes"], self.screen[k]["wer"], k))
            # Plain asymmetric graph keeps INT2/3 grafting independent of correction.
            anchor = next(k for k in ranked if self.catalog[k]["spec"]["method"] is None and
                          self.catalog[k]["spec"]["four_mode"] == "INT4_ASYM")
            save(anchors_path, {"label": anchor})
        label = read_json(anchors_path)["label"]
        spec = self.catalog[label]["spec"]
        proposals = []
        for bit, count in ((3, 8), (2, 4)):
            ranked = sorted(spec["assignments"], key=lambda n: (
                (self.sens[n]["options"][str(bit)]["proxy_loss"] - self.sens[n]["options"][str(spec['assignments'][n])]["proxy_loss"]) /
                max(1, self.sens[n]["parameters"] * (spec["assignments"][n] - bit) / 8), n))
            for name in ranked[:count]:
                mutated = dict(spec["assignments"]); mutated[name] = bit
                proposals.append((f"isolated {bit}-bit probe from {label}: {name}", specification(mutated, spec["group_size"])))
        self.proposals("lowbit", proposals)
        self.fill_family("lowbit")
        if len(self.catalog) != 96:
            raise RuntimeError(f"Search contains {len(self.catalog)} rather than 96 candidate records")
        full_selection = self.root / "development_shortlist.json"
        if not full_selection.exists():
            save(full_selection, {"labels": self.top(self.config["full_development_candidates"]),
                 "screen_results_sha256": checksum(self.screen), "candidates_sha256": checksum(self.catalog)})
        chosen = read_json(full_selection)
        if chosen["screen_results_sha256"] != checksum(self.screen) or chosen["candidates_sha256"] != checksum(self.catalog):
            raise RuntimeError("The recorded development shortlist changed")
        for label in chosen["labels"]:
            if label not in self.full:
                print(f"Full development evaluation: {label}, 180 clips", flush=True)
                meta = self.catalog[label]["metadata"]
                self.full[label] = evaluate_cached(self.config, self.root, label,
                    self.manifests["development"], self.root / "models" / label, meta["model_sha256"])
                save(self.full_path, self.full)
        save(self.root / "search_complete.json", {"candidates_sha256": checksum(self.catalog),
            "screen_results_sha256": checksum(self.screen), "development_results_sha256": checksum(self.full),
            "family_counts": dict(Counter(r["family"] for r in self.catalog.values())),
            "status_counts": dict(Counter(r["status"] for r in self.catalog.values()))})

    def fill_family(self, family: str) -> None:
        # Deduplication or unavailable anchors can reduce a family. Fill with new
        # deterministic single-matrix flips, keeping its technique and logging why.
        for anchor in self.top(6):
            spec = self.catalog[anchor]["spec"]
            for name in sorted(spec["assignments"]):
                if self.family_count(family) >= self.config["family_limits"][family]:
                    return
                assignment = dict(spec["assignments"])
                assignment[name] = (3 if family == "lowbit" else 8 if assignment[name] == 4 else 4)
                g, mode = (32, "INT4_SYM") if family == "quantizer" else (64, "INT4_ASYM")
                self.proposals(family, [(f"deterministic deduplication fill from {anchor}: {name}", specification(assignment, g, mode))])
        if self.family_count(family) < self.config["family_limits"][family]:
            raise RuntimeError(f"Could not fill {family} search budget")


def select(config: dict, root: Path) -> None:
    verify_reservations(config, root)
    path = root / "selection.json"
    if path.exists():
        verify_selection(config, root)
        return
    assert_tests_unread(root)
    complete = read_json(root / "search_complete.json")
    catalog, results = read_json(root / "candidates.json"), read_json(root / "development_results.json")
    candidates = {k: v for k, v in results.items() if k in catalog and catalog[k]["status"] == "measured"}
    labels, rule = choose_finalists(candidates, results["ov_fp32"])
    choice = {"status": "frozen_before_any_phase5_heldout_predictions", "ordered_labels": labels,
              "rule": rule, "primary": labels[0], "runner_up": labels[1] if len(labels) > 1 else None,
              "primary_dev_wer": candidates[labels[0]]["wer"], "fp32_dev_wer": results["ov_fp32"]["wer"],
              "config_sha256": checksum(config), "reservations_sha256": checksum(read_json(root / "reservations.json")),
              "candidates_sha256": checksum(catalog), "screen_results_sha256": checksum(read_json(root / "screen_results.json")),
              "development_results_sha256": checksum(results), "search_complete_sha256": checksum(complete)}
    for name in ("preflight", "activation_scores"):
        choice[f"{name}_sha256"] = checksum(read_json(root / f"{name}.json"))
    freeze_audio(root)
    choice["audio_integrity_sha256"] = checksum(read_json(root / "audio_integrity.json"))
    choice["algorithm_sources_sha256"] = {p: file_digest(Path(p)) for p in
        ("src/whisper_precision/cli.py", "src/whisper_precision/search.py", "src/whisper_precision/quantization.py",
         "src/whisper_precision/inference.py", "src/whisper_precision/core.py", "src/whisper_precision/graphs.py",
         "src/whisper_precision/data.py", "src/whisper_precision/io.py", "src/whisper_precision/paths.py")}
    immutable(path, choice)
    print(f"Frozen finalists: {labels}; selection rule {rule}", flush=True)


def verify_selection(config: dict, root: Path) -> dict:
    verify_reservations(config, root)
    choice = read_json(root / "selection.json")
    if choice["config_sha256"] != checksum(config):
        raise RuntimeError("Frozen configuration changed")
    for name in ("reservations", "candidates", "screen_results", "development_results", "search_complete", "preflight", "activation_scores"):
        if choice[f"{name}_sha256"] != checksum(read_json(root / f"{name}.json")):
            raise RuntimeError(f"Frozen {name} changed")
    if choice["audio_integrity_sha256"] != checksum(read_json(root / "audio_integrity.json")):
        raise RuntimeError("Frozen audio integrity record changed")
    for path, digest in choice.get("algorithm_sources_sha256", {}).items():
        if file_digest(Path(path)) != digest:
            raise RuntimeError(f"Frozen evaluation source changed: {path}")
    return choice


def model_reference(root: Path, source: Path, label: str) -> tuple[Path, str]:
    if label in ("ov_fp32", "uniform_int8"):
        path = source / "models" / label
        key = "source_model_sha256" if label == "ov_fp32" else "int8_model_sha256"
        sha = read_json(root / "provenance.json")[key]
    else:
        path = root / "models" / label
        sha = read_json(root / "allocations" / f"{label}.json")["model_sha256"]
    if model_digest(path) != sha:
        raise RuntimeError(f"Frozen model changed: {label}")
    return path, sha


def final(config: dict, root: Path, source: Path) -> None:
    choice = verify_selection(config, root)
    manifests = verify_reservations(config, root, check_audio=True)
    path = root / "test_results.json"
    results = read_json(path) if path.exists() else {"test_a": {}, "test_b": {}}
    for split in ("test_a", "test_b"):
        labels = ["ov_fp32", "uniform_int8"] + [label for label in choice["ordered_labels"]
            if split == "test_a" or passes(results["test_a"][label], results["test_a"]["ov_fp32"], config["relative_wer_tolerance"])]
        for label in labels:
            if label in results[split]:
                continue
            print(f"Independent {split}: {label}, 150 clips", flush=True)
            model_path, sha = model_reference(root, source, label)
            results[split][label] = evaluate_cached(config, root, label, manifests[split], model_path, sha)
            save(path, results)
    outcome = {}
    catalog = read_json(root / "candidates.json")
    for label in choice["ordered_labels"]:
        a = passes(results["test_a"][label], results["test_a"]["ov_fp32"], config["relative_wer_tolerance"])
        b = label in results["test_b"] and passes(results["test_b"][label], results["test_b"]["ov_fp32"], config["relative_wer_tolerance"])
        outcome[label] = {"passes_a": a, "passes_b": b, "test_b_evaluated": label in results["test_b"],
                          "meets_goal": a and b and catalog[label]["metadata"]["serialized_bytes"] <
                          min(config["maximum_bytes"], results["test_a"]["uniform_int8"]["serialized_bytes"])}
    passed = [x for x in outcome if outcome[x]["meets_goal"]]
    best = min(passed, key=lambda k: catalog[k]["metadata"]["serialized_bytes"], default=None)
    save(root / "outcome.json", {"candidates": outcome, "smallest_new_passing": best,
        "selection_sha256": checksum(choice), "test_results_sha256": checksum(results),
        "criterion": "smaller than INT8 and <=1.05 times paired FP32 WER on both clean sets"})
    print(f"Independent outcome: {outcome}", flush=True)


def diagnostics(config: dict, root: Path, source: Path) -> None:
    import hashlib
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset
    choice = verify_selection(config, root)
    if not (root / "outcome.json").exists():
        raise RuntimeError("Finish the independent decision before diagnostic scoring")
    path = root / "other_manifest.json"
    if not path.exists():
        excluded = {x["id"] for p in config["other_exclusion_manifests"] for x in read_json(Path(p))["clips"]}
        dataset = load_dataset(config["dataset_id"], "other", split="test", streaming=True,
            revision=config["dataset_revision"], cache_dir=str(root / "cache")).decode(False)
        rows, scanned, available = choose_rows_by_duration(dataset, config["diagnostic_other_targets"], config["seed"]+2, excluded)
        clips = []
        for row in rows:
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if sr != 16000 or audio.ndim != 1:
                raise RuntimeError("Expected mono 16 kHz diagnostic audio")
            target = root / "data" / "other" / f"{row['id']}.flac"
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, audio, sr)
            clips.append({"id": row["id"], "speaker_id": row["speaker_id"], "text": row["text"],
                "audio": str(target), "seconds": len(audio)/sr, "duration_band": duration_band(len(audio)/sr)})
        immutable(path, {"name": "other", "clips": clips, "count": len(clips),
            "dataset_revision": config["dataset_revision"], "dataset_config": "other", "split": "test",
            "rows_scanned": scanned, "available_by_band": available,
            "duration_targets": config["diagnostic_other_targets"], "excluded_manifests": config["other_exclusion_manifests"],
            "role": "secondary diagnostic; no clean accuracy gate"})
    a = read_json(root / "test_a_manifest.json")
    path = root / "noise_manifest.json"
    if not path.exists():
        noisy = []
        for clip in a["clips"]:
            audio, sr = sf.read(clip["audio"], dtype="float32")
            seed = int(hashlib.sha256(f"{config['seed']}:{clip['id']}".encode()).hexdigest()[:16], 16)
            noise = np.random.default_rng(seed).standard_normal(len(audio)).astype("float32")
            noise *= np.sqrt(np.mean(audio.astype("float64") ** 2)) / (
                10 ** (config["noise_snr_db"]/20) * np.sqrt(np.mean(noise.astype("float64") ** 2)))
            target = root / "data" / "noise" / f"{clip['id']}.wav"
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, audio + noise, sr, subtype="FLOAT")
            noisy.append({**clip, "audio": str(target), "audio_sha256": file_digest(target)})
        immutable(path, {"name": "noise", "clips": noisy, "count": len(noisy),
            "source_manifest_sha256": checksum(a), "method": "deterministic white Gaussian noise",
            "snr_db": config["noise_snr_db"], "role": "paired synthetic stress, not real noisy speech"})
    labels = ["ov_fp32", "uniform_int8", *choice["ordered_labels"]]
    results_path = root / "diagnostic_results.json"
    results = read_json(results_path) if results_path.exists() else {"other": {}, "noise": {}}
    for name in ("other", "noise"):
        manifest = read_json(root / f"{name}_manifest.json")
        for label in labels:
            if label in results[name]:
                continue
            print(f"Diagnostic {name}: {label}", flush=True)
            model_path, sha = model_reference(root, source, label)
            results[name][label] = evaluate_cached(config, root, label, manifest, model_path, sha)
            save(results_path, results)
    timing_path = root / "timing.json"
    timing = read_json(timing_path) if timing_path.exists() else {}
    clips = [x for band in ("short", "medium", "long") for x in
             [c for c in a["clips"] if c["duration_band"] == band][:4]]
    manifest = {"name": "timing", "clips": clips, "count": len(clips)}
    for label in labels:
        if label in timing:
            continue
        model_path, sha = model_reference(root, source, label)
        value = evaluate_cached(config, root, label, manifest, model_path, sha,
                                repeats=config["timing_benchmark_repeats"]+1)
        samples = [t for p in value["predictions"] for t in p["latency_s"][1:]]
        timing[label] = {"median_seconds": statistics.median(samples), "mean_seconds": statistics.mean(samples),
            "clip_ids": [x["id"] for x in clips], "repeats": config["timing_benchmark_repeats"], "warmups": 1}
        save(timing_path, timing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(repository_root() / "configs" / "search.json"))
    parser.add_argument("command", choices=("prepare", "preflight", "search", "select", "final", "diagnostics", "all"))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    os.chdir(repository_root())
    if args.command == "all":
        import subprocess
        import sys
        for command in ("prepare", "preflight", "search", "select", "final", "diagnostics"):
            subprocess.run([sys.executable, "-m", "whisper_precision.cli", "--config", str(config_path), command], check=True)
        return
    if args.command in ("preflight", "search", "select", "final"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = read_json(config_path)
    root, source, inventory = context(config)
    if args.command in ("prepare", "all"):
        prepare(config, root)
    if args.command in ("preflight", "all"):
        preflight(config, root, source, inventory)
    if args.command in ("search", "all"):
        if not (root / "preflight.json").exists():
            raise RuntimeError("Run preflight before search")
        Study(config, root, source, inventory).search()
    if args.command in ("select", "all"):
        select(config, root)
    if args.command in ("final", "all"):
        final(config, root, source)
    if args.command in ("diagnostics", "all"):
        diagnostics(config, root, source)


if __name__ == "__main__":
    main()
