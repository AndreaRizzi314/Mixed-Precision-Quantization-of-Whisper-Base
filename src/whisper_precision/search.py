"""Measured-WER search helpers, immutable manifests, and resumable evaluations."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

from .io import checksum, read_json
from .core import corpus_wer


def atomic_replace(temporary: Path, path: Path) -> None:
    # Windows sync/indexing can briefly hold a destination open.
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.1 * 2**attempt, .8))


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")
    atomic_replace(temporary, path)


def immutable(path: Path, value: dict) -> None:
    if path.exists():
        if checksum(read_json(path)) != checksum(value):
            raise RuntimeError(f"Immutable record changed: {path}")
    else:
        save(path, value)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_digest(path: Path) -> str:
    return checksum({p.name: file_digest(p) for p in sorted(path.iterdir())
                     if p.suffix in (".xml", ".bin", ".json") and p.name != "variant.json"})


def assert_disjoint(*manifests: dict) -> None:
    seen = set()
    for manifest in manifests:
        ids = [x["id"] for x in manifest["clips"]]
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("Duplicate or overlapping clip IDs")
        seen.update(ids)


def passes(row: dict, baseline: dict, tolerance: float = .05) -> bool:
    # Compare integer edit counts when the same reference-word total is available.
    if row["reference_words"] != baseline["reference_words"]:
        raise ValueError("Models were scored on different reference-word totals")
    return row["errors"] <= baseline["errors"] * (1 + tolerance)


def aggregate(predictions: list[dict], serialized_bytes: int) -> dict:
    score = corpus_wer([(x["reference"], x["hypothesis"]) for x in predictions])
    return {**score, "predictions": predictions, "serialized_bytes": serialized_bytes,
            "backend": "openvino"}


def subset_result(row: dict, clips: list[dict]) -> dict:
    mapped = {x["id"]: x for x in row["predictions"]}
    ordered = []
    for clip in clips:
        prediction = mapped[clip["id"]]
        if prediction["reference"] != clip["text"]:
            raise RuntimeError("Cached reference differs from the manifest")
        ordered.append(prediction)
    return aggregate(ordered, row["serialized_bytes"])


def evaluation_key(config: dict, manifest: dict, model_sha: str, repeats: int = 1) -> str:
    # A changed scorer or generator must never reuse an earlier prediction cache.
    package = Path(__file__).resolve().parent
    sources = {name: file_digest(package / name) for name in
               ("core.py", "inference.py", "search.py", "io.py", "graphs.py")}
    return checksum({"version": 2, "evaluation_sources": sources,
                     "manifest": checksum(manifest), "model": model_sha,
                     "model_revision": config["model_revision"],
                     "max_new_tokens": config["max_new_tokens"], "repeats": repeats,
                     "decode": "english_transcribe_greedy_t0_no_previous_tokens",
                     "scorer": "NFKC_upper_punctuation_removed_word_levenshtein"})


def cached_features(config: dict, root: Path, processor, clip: dict) -> dict:
    """Cache identical processor tensors; model inference settings are unchanged."""
    import numpy as np
    import torch
    from .inference import _audio_features
    audio_sha = clip.get("audio_sha256") or file_digest(Path(clip["audio"]))
    key = checksum({"audio_sha256": audio_sha, "processor_revision": config["model_revision"],
                    "format": "processor_pt_attention_mask_v1"})
    path = root / "features" / f"{key}.npz"
    if not path.exists():
        inputs = _audio_features(processor, clip)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **{k: v.detach().cpu().numpy() for k, v in inputs.items()})
        atomic_replace(temporary, path)
    with np.load(path, allow_pickle=False) as inputs:
        return {k: torch.from_numpy(inputs[k].copy()) for k in inputs.files}


def evaluate_cached(config: dict, root: Path, label: str, manifest: dict,
                    model_path: Path, model_sha: str, *, repeats: int = 1) -> dict:
    """Checkpoint complete clip batches; reject cache/settings/audio drift."""
    from optimum.intel.openvino import OVModelForSpeechSeq2Seq
    from transformers import AutoProcessor
    from .inference import evaluate
    from .graphs import serialized_bytes

    key = evaluation_key(config, manifest, model_sha, repeats)
    path = root / "results" / manifest["name"] / f"{label}.json"
    state = read_json(path) if path.exists() else {"key": key, "predictions": []}
    if state["key"] != key:
        raise RuntimeError(f"Evaluation cache changed: {path}")
    predictions = state["predictions"]
    clips = manifest["clips"]
    if len(predictions) > len(clips):
        raise RuntimeError("Cache has extra predictions")
    for predicted, clip in zip(predictions, clips):
        if predicted["id"] != clip["id"] or predicted["reference"] != clip["text"]:
            raise RuntimeError("Cache clip order or references changed")
    if len(predictions) < len(clips):
        model = OVModelForSpeechSeq2Seq.from_pretrained(model_path, compile=False, local_files_only=True)
        model.compile()
        processor = AutoProcessor.from_pretrained(config["model_id"],
                        revision=config["model_revision"], local_files_only=True)
        effective = {**config, "timing_repeats": repeats}
        for start in range(len(predictions), len(clips), 12):
            result = evaluate(model, processor, clips[start:start + 12], effective, "openvino",
                              features_for_clip=lambda clip: cached_features(config, root, processor, clip))
            predictions.extend(result["predictions"])
            save(path, {"key": key, "predictions": predictions, "complete": False})
        del model
    result = aggregate(predictions, serialized_bytes(model_path))
    save(path, {"key": key, "predictions": predictions, "complete": True,
                "result": result})
    return result


def specification(assignments: dict, group_size: int = 64,
                  mode: str = "INT4_ASYM", method: str | None = None) -> dict:
    return {"assignments": dict(sorted(assignments.items())), "group_size": group_size,
            "four_mode": mode, "method": method}


def candidate_id(spec: dict) -> str:
    return "q_" + checksum(spec)[:16]


def ablations(operations: dict) -> list[tuple[str, dict]]:
    groups = sorted({r["component"] for r in operations.values()})
    output = []
    for anchor, replacement, word in ((8, 4, "downgrade"), (4, 8, "rescue")):
        for group in groups:
            assignment = {n: replacement if r["component"] == group else anchor
                          for n, r in operations.items()}
            output.append((f"{word}_{group}", specification(assignment)))
    return output


def frontier(rows: dict) -> list[str]:
    ordered = sorted(rows, key=lambda n: (rows[n]["serialized_bytes"], rows[n]["wer"], n))
    best = math.inf
    result = []
    for label in ordered:
        if rows[label]["wer"] < best:
            result.append(label)
            best = rows[label]["wer"]
    return result


def shortlist(rows: dict, count: int, fp_wer: float) -> list[str]:
    """Take the frontier, then robust and lowest-WER alternatives."""
    chosen = frontier(rows)
    ranked = sorted(rows, key=lambda n: (max(0, rows[n]["wer"] / fp_wer - 1.02),
                                         rows[n]["serialized_bytes"], rows[n]["wer"], n))
    if len(chosen) > count:
        # Equally spaced frontier positions preserve the size/accuracy range.
        chosen = [chosen[round(i * (len(chosen) - 1) / (count - 1))] for i in range(count)]
    for label in ranked:
        if label not in chosen and len(chosen) < count:
            chosen.append(label)
    return chosen


def component_mask(spec: dict, operations: dict, groups: list[str]) -> tuple[int, ...] | None:
    mask = []
    for group in groups:
        values = {spec["assignments"][n] for n, r in operations.items() if r["component"] == group}
        if len(values) != 1 or not values.issubset({4, 8}):
            return None
        mask.append(int(next(iter(values)) == 4))
    return tuple(mask)


def beam_proposals(catalog: dict, rows: dict, operations: dict, known: set[str],
                   fp_wer: float, count: int = 6) -> list[tuple[str, dict]]:
    """Ridge surrogate with pair interactions proposes component flips and swaps.

    Only measured screen WER trains the surrogate. Predictions guide evaluation;
    they never qualify a model for selection or replace actual saved size.
    """
    import itertools
    import numpy as np

    groups = sorted({r["component"] for r in operations.values()})
    pairs = list(itertools.combinations(range(len(groups)), 2))
    feature = lambda v: np.array([1, *v, *(v[i] * v[j] for i, j in pairs)], dtype=float)
    eligible = {k: r for k, r in rows.items() if k in catalog and
                catalog[k]["spec"]["group_size"] == 64 and
                catalog[k]["spec"]["four_mode"] == "INT4_ASYM" and
                catalog[k]["spec"]["method"] is None and
                component_mask(catalog[k]["spec"], operations, groups) is not None}
    if not eligible:
        raise RuntimeError("No measured component anchors for beam search")
    masks = {k: component_mask(catalog[k]["spec"], operations, groups) for k in eligible}
    x = np.stack([feature(masks[k]) for k in eligible])
    y = np.array([eligible[k]["wer"] for k in eligible])
    penalty = np.eye(x.shape[1]) * .5
    penalty[0, 0] = .00001
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    size_coef = np.linalg.lstsq(x[:, :11], np.array([eligible[k]["serialized_bytes"] / 1e6
                                                    for k in eligible]), rcond=None)[0]
    anchors = shortlist(eligible, min(6, len(eligible)), fp_wer)
    masks_to_try = set()
    for label in anchors:
        mask = masks[label]
        for i in range(len(groups)):
            v = list(mask); v[i] = 1 - v[i]; masks_to_try.add(tuple(v))
        for i, j in pairs:
            if mask[i] != mask[j]:
                v = list(mask); v[i], v[j] = v[j], v[i]; masks_to_try.add(tuple(v))
    # Broaden if the local beam is exhausted; enumerate only 2^10 group choices.
    masks_to_try.update(itertools.product((0, 1), repeat=len(groups)))
    proposed = []
    for mask in sorted(masks_to_try):
        if not any(mask):
            continue  # The INT8 control is not a smaller candidate.
        spec = specification({n: 4 if mask[groups.index(r["component"])] else 8
                              for n, r in operations.items()})
        if candidate_id(spec) in known:
            continue
        predicted = float(feature(mask) @ coef)
        size = float(feature(mask)[:11] @ size_coef)
        proposed.append((mask, spec, predicted, size))
    chosen = []
    # Six targets retain a range of sizes; re-fit after each measured batch.
    for target in (57, 60, 63, 66, 69, 72)[:count]:
        if not proposed:
            break
        winner = min(proposed, key=lambda z: (
            max(0, z[2] / fp_wer - 1.04), abs(z[3] - target), z[2], z[0]))
        proposed.remove(winner)
        chosen.append((f"beam_target_{target}mb", winner[1]))
    return chosen


def choose_finalists(rows: dict, fp: dict) -> tuple[list[str], str]:
    for margin in (.02, .05):
        eligible = [k for k, r in rows.items() if passes(r, fp, margin)]
        if eligible:
            chosen = sorted(eligible, key=lambda k: (rows[k]["serialized_bytes"], rows[k]["wer"], k))[:2]
            rule = f"within_{margin:.0%}_dev"
            if len(chosen) == 1 and len(rows) > 1:
                runners, runner_rule = choose_finalists({k: r for k, r in rows.items() if k not in chosen}, fp)
                chosen.extend(runners[:1])
                rule += f"; runner_{runner_rule}"
            return chosen, rule
    return sorted(rows, key=lambda k: (rows[k]["wer"], rows[k]["serialized_bytes"], k))[:2], "no_dev_pass"
