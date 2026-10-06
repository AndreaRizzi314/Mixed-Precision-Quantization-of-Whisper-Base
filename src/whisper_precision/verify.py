"""Check the published evidence without downloading a model or running inference."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import zipfile

from whisper_precision.core import corpus_wer
from whisper_precision.io import checksum, read_json
from whisper_precision.search import assert_disjoint, choose_finalists, passes
from whisper_precision.statistics import paired_ci
from whisper_precision.paths import repository_root


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def check_scores(row, clips):
    predictions = row["predictions"]
    if [p["id"] for p in predictions] != [c["id"] for c in clips]:
        raise ValueError("Predictions do not match the manifest order")
    if any(p["reference"] != c["text"] for p, c in zip(predictions, clips, strict=True)):
        raise ValueError("A prediction reference differs from its manifest")
    score = corpus_wer([(p["reference"], p["hypothesis"]) for p in predictions])
    for field in ("errors", "reference_words", "wer", "clip_wer"):
        if row[field] != score[field]:
            raise ValueError(f"Stored {field} differs from recomputed word edits")


def validate_records(records, inputs, source_archive):
    config = records["config.json"]
    frozen = records["selection.json"]
    if frozen["config_sha256"] != checksum(config):
        raise ValueError("Frozen configuration changed")
    for name in ("reservations", "candidates", "screen_results", "development_results",
                 "search_complete", "preflight", "activation_scores", "audio_integrity"):
        if frozen[f"{name}_sha256"] != checksum(records[f"{name}.json"]):
            raise ValueError(f"Frozen {name} changed")
    with zipfile.ZipFile(source_archive) as archive:
        for name, expected in frozen["algorithm_sources_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Recorded numerical source changed: {name}")
    manifests = {name: records[f"{name}_manifest.json"] for name in
                 ("development", "screen", "test_a", "test_b", "other", "noise")}
    reservations = records["reservations.json"]
    for name in ("development", "screen", "test_a", "test_b"):
        if checksum(manifests[name]) != reservations[f"{name}_sha256"]:
            raise ValueError(f"Reserved manifest changed: {name}")
    assert_disjoint(*(manifests[n] for n in ("development", "test_a", "test_b")))
    previous = {c["id"] for path, m in inputs.items() if path.endswith("test_manifest.json")
                for c in m["clips"]}
    for name in ("test_a", "test_b"):
        clips = manifests[name]["clips"]
        if len(clips) != 150 or Counter(c["duration_band"] for c in clips) != config[f"{name}_targets"]:
            raise ValueError(f"Wrong fresh duration strata: {name}")
        if previous.intersection(c["id"] for c in clips):
            raise ValueError("Fresh clips overlap an earlier test")
    development = manifests["development"]["clips"]
    if len(development) != 180 or len(manifests["screen"]["clips"]) != 72:
        raise ValueError("Wrong development pool sizes")
    catalog = records["candidates.json"]
    screen = records["screen_results.json"]
    full = records["development_results.json"]
    if len(catalog) != 96 or Counter(r["family"] for r in catalog.values()) != config["family_limits"]:
        raise ValueError("Candidate count or search budgets changed")
    if set(screen) != set(catalog) | {"ov_fp32", "uniform_int8"}:
        raise ValueError("The common screen omits candidates or controls")
    if len(set(catalog) & set(full)) != 16:
        raise ValueError("Wrong full-development shortlist size")
    precision = records["precision_verification.json"]
    if precision["model_count"] != 96 or set(precision["models"]) != set(catalog):
        raise ValueError("Incomplete recorded packed-type verification")
    for label, candidate in catalog.items():
        metadata = candidate["metadata"]
        if candidate["status"] != "measured" or metadata["serialized_bytes"] >= config["maximum_bytes"]:
            raise ValueError(f"Unmeasured or oversized candidate: {label}")
        observed = precision["models"][label]["effective_bits"]
        if observed != metadata["effective_bits"]:
            raise ValueError(f"Packed-type records disagree: {label}")
        if any(observed[op] != bit for op, bit in candidate["spec"]["assignments"].items()):
            raise ValueError(f"Requested and recorded precision disagree: {label}")
    tests, diagnostics = records["test_results.json"], records["diagnostic_results.json"]
    for rows, clips in [(screen, manifests["screen"]["clips"]), (full, development),
                        *((rows, manifests[name]["clips"]) for name, rows in tests.items()),
                        *((rows, manifests[name]["clips"]) for name, rows in diagnostics.items())]:
        for row in rows.values():
            check_scores(row, clips)
    for label in set(catalog) & set(full):
        predictions = {p["id"]: p["hypothesis"] for p in full[label]["predictions"]}
        if any(predictions[p["id"]] != p["hypothesis"] for p in screen[label]["predictions"]):
            raise ValueError("Full development did not reproduce screen predictions")
    selected, rule = choose_finalists({k: full[k] for k in catalog if k in full}, full["ov_fp32"])
    if selected != frozen["ordered_labels"] or rule != frozen["rule"]:
        raise ValueError("The recorded selection does not follow the development rule")
    outcome = records["outcome.json"]
    if outcome["selection_sha256"] != checksum(frozen) or outcome["test_results_sha256"] != checksum(tests):
        raise ValueError("Outcome evidence changed")
    for label in selected:
        pass_a = passes(tests["test_a"][label], tests["test_a"]["ov_fp32"])
        if (label in tests["test_b"]) != pass_a:
            raise ValueError("Set B was evaluated contrary to the ordered gate")
        pass_b = pass_a and passes(tests["test_b"][label], tests["test_b"]["ov_fp32"])
        recorded = outcome["candidates"][label]
        if (recorded["passes_a"], recorded["passes_b"], recorded["meets_goal"]) != (pass_a, pass_b, pass_a and pass_b):
            raise ValueError("Candidate pass/fail decision is incorrect")
    if outcome["smallest_new_passing"] is not None:
        raise ValueError("This recorded search did not establish a smaller passing model")
    if not all(passes(rows["uniform_int8"], rows["ov_fp32"]) for rows in tests.values()):
        raise ValueError("INT8 did not pass both recorded confirmations")
    return config, frozen, manifests


def confirmation_rows(records, config, frozen, manifests):
    rows = []
    for split, scores in records["test_results.json"].items():
        baseline = scores["ov_fp32"]
        for label in ["ov_fp32", "uniform_int8", *frozen["ordered_labels"]]:
            if label not in scores:
                rows.append({"set": split, "model": label, "decision": "not evaluated: failed A"})
                continue
            row = scores[label]
            interval = paired_ci(baseline, row, manifests[split]["clips"], config["seed"] + (split == "test_b"))
            rows.append({"set": split, "model": label, "saved_bytes": row["serialized_bytes"],
                         "edits": row["errors"], "reference_words": row["reference_words"],
                         "wer": row["wer"], "relative_wer_change": row["wer"]/baseline["wer"] - 1,
                         "difference_ci_low_pp": 100*interval[0], "difference_ci_high_pp": 100*interval[1],
                         "decision": "pass" if passes(row, baseline) else "fail"})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=repository_root() / "results")
    parser.add_argument("--write-tables", action="store_true", help="Regenerate confirmation.csv and evidence checksums")
    args = parser.parse_args()
    root = args.results
    if not args.write_tables:
        for name, expected in read_json(root / "provenance" / "checksums.json")["files"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Evidence file changed: {name}")
    records = read_gzip(root / "raw" / "measurements.json.gz")["records"]
    inputs = read_gzip(root / "raw" / "inputs.json.gz")
    config, frozen, manifests = validate_records(records, inputs, root / "provenance" / "numerical_source.zip")
    if read_json(root / "provenance" / "selection.json") != frozen or read_json(root / "provenance" / "environment.json") != records["environment.json"]:
        raise ValueError("Readable records differ from the evidence bundle")
    with (root / "tables" / "candidates.csv").open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    if len(table) != 96 or {r["candidate"] for r in table} != set(records["candidates.json"]):
        raise ValueError("Candidate CSV omits recorded exports")
    for row in table:
        label = row["candidate"]
        screen = records["screen_results.json"][label]
        if int(row["saved_bytes"]) != screen["serialized_bytes"] or float(row["screen_wer"]) != screen["wer"]:
            raise ValueError("Candidate CSV disagrees with raw measurements")
        full = records["development_results.json"].get(label)
        if (row["development_wer"] != "") != (full is not None):
            raise ValueError("Candidate CSV misstates full-development evaluation")
        if full and float(row["development_wer"]) != full["wer"]:
            raise ValueError("Candidate CSV full-development score differs")
    rows = confirmation_rows(records, config, frozen, manifests)
    fields = ["set", "model", "saved_bytes", "edits", "reference_words", "wer", "relative_wer_change",
              "difference_ci_low_pp", "difference_ci_high_pp", "decision"]
    if args.write_tables:
        with (root / "tables" / "confirmation.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        manifest = {"format_version": 1, "recorded_run": "artifacts/phase5", "files": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "checksums.json"}}
        (root / "provenance" / "checksums.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    else:
        with (root / "tables" / "confirmation.csv").open(encoding="utf-8", newline="") as handle:
            actual = list(csv.DictReader(handle))
        expected = [{field: str(row.get(field, "")) for field in fields} for row in rows]
        if actual != expected:
            raise ValueError("Confirmation CSV differs from recomputed results and intervals")
    print("Verified 96 candidate records, 16 full-development repeats, frozen selection and source,")
    print("disjoint clip manifests, transcript-derived WER, speaker intervals, and ordered A/B decisions.")
    print("Smallest validated saved model: 76.43 MB INT8. Neither smaller finalist passed both sets.")
    print("This checks recorded evidence; model payloads and audio are excluded from the repository.")


if __name__ == "__main__":
    main()
