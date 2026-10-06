"""Restore development inputs and export the controls for a fresh search."""
import argparse
import gzip
import io
import json
import os
from pathlib import Path

from whisper_precision.io import read_json
from whisper_precision.search import immutable
from whisper_precision.paths import repository_root


def load_inputs():
    with gzip.open(repository_root() / "results" / "raw" / "inputs.json.gz", "rt", encoding="utf-8") as handle:
        return json.load(handle)


def workspace_path(value):
    """Historical manifests use Windows separators; keep paths inside this repo."""
    root = repository_root()
    target = (root / value.replace("\\", "/")).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Input path escapes the repository: {value}")
    return target


def materialize_audio(config, manifests):
    import soundfile as sf
    from datasets import load_dataset

    groups = {}
    for manifest in manifests:
        key = (manifest["dataset_config"], manifest["split"])
        for clip in manifest["clips"]:
            path = workspace_path(clip["audio"])
            if not path.exists():
                groups.setdefault(key, {})[clip["id"]] = clip
    for (dataset_config, split), wanted in groups.items():
        print(f"Downloading {len(wanted)} development clips from {dataset_config}/{split}", flush=True)
        stream = load_dataset(config["dataset_id"], dataset_config, split=split,
                              revision=config["dataset_revision"], streaming=True).decode(False)
        for row in stream:
            if row["id"] not in wanted:
                continue
            clip = wanted.pop(row["id"])
            if row["text"] != clip["text"]:
                raise RuntimeError(f"Pinned reference changed: {row['id']}")
            samples, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if rate != 16000 or samples.ndim != 1 or abs(len(samples)/rate - clip["seconds"]) > 1e-6:
                raise RuntimeError(f"Pinned audio format or duration changed: {row['id']}")
            path = workspace_path(clip["audio"])
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, samples, rate)
            if not wanted:
                break
        if wanted:
            raise RuntimeError(f"Development IDs not found: {sorted(wanted)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(repository_root() / "configs" / "search.json"))
    parser.add_argument("--check", action="store_true", help="Check input records without downloads or exports")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    os.chdir(repository_root())
    config = read_json(config_path)
    inputs = load_inputs()
    paths = config["development_manifests"] + config["clean_exclusion_manifests"] + config["other_exclusion_manifests"]
    if any(path not in inputs for path in paths):
        raise ValueError("The configuration refers to an input not included in the recorded study")
    manifests = [inputs[path] for path in config["development_manifests"]]
    if any(m["dataset_revision"] != config["dataset_revision"] for m in manifests):
        raise ValueError("The configured dataset revision differs from the recorded development inputs")
    clips = [clip for manifest in manifests for clip in manifest["clips"]]
    if len(clips) != 180 or len({c["id"] for c in clips}) != 180:
        raise ValueError("Development inputs must contain 180 distinct clips")
    for clip in clips:
        workspace_path(clip["audio"])
    workspace_path(config["source_root"])
    workspace_path(config["output_dir"])
    if args.check:
        print("Input records valid: 180 development clips and all exclusion manifests included.")
        return
    from whisper_precision.inference import environment
    from whisper_precision.graphs import export_fp32, compress_variant
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    # Restore only inputs. Held-out predictions are never read by the search.
    for path in set(paths):
        immutable(workspace_path(path), inputs[path])
    sensitivity = workspace_path("artifacts/layerwise/sensitivity.json")
    if sensitivity.exists():
        if read_json(sensitivity)["operations"] != inputs["artifacts/layerwise/sensitivity.json"]["operations"]:
            raise RuntimeError("The recorded operation metadata changed")
    else:
        immutable(sensitivity, inputs["artifacts/layerwise/sensitivity.json"])
    materialize_audio(config, manifests)
    AutoProcessor.from_pretrained(config["model_id"], revision=config["model_revision"])
    WhisperForConditionalGeneration.from_pretrained(config["model_id"], revision=config["model_revision"])
    source = workspace_path(config["source_root"])
    source.mkdir(parents=True, exist_ok=True)
    if not (source / "environment.json").exists():
        immutable(source / "environment.json", environment(config))
    export_fp32(config, source)
    compress_variant(source, "uniform_int8", set())
    print("Source models and development audio ready. Run: python -m whisper_precision.cli all")


if __name__ == "__main__":
    main()
