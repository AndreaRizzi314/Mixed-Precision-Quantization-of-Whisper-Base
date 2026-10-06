"""Pinned environment capture and deterministic speech generation."""
import importlib.metadata
import os
import platform
import subprocess
import statistics
import time
from .core import corpus_wer


def environment(config: dict) -> dict:
    import torch
    packages = ("torch", "transformers", "datasets", "soundfile", "huggingface-hub",
                "numpy", "openvino", "nncf", "optimum-intel")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    model_revision = config["model_revision"]
    try:
        nvidia = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                 "--format=csv,noheader"], capture_output=True, text=True,
                                timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        nvidia = None
    try:
        import openvino as ov
        openvino_devices = ov.Core().available_devices
    except Exception:
        openvino_devices = []
    return {"model_id": config["model_id"], "model_revision": model_revision,
            "python": platform.python_version(), "os": platform.platform(),
            "cpu": platform.processor(), "logical_cpu_count": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "nvidia_smi": nvidia, "openvino_devices": openvino_devices,
            "packages": versions, "seed": config["seed"]}

def device_for(config: dict) -> str:
    import torch
    selected = config["device"]
    if selected == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return selected

def _audio_features(processor, clip: dict):
    import soundfile as sf
    samples, rate = sf.read(clip["audio"], dtype="float32")
    return processor(samples, sampling_rate=rate, return_tensors="pt", return_attention_mask=True)

def evaluate(model, processor, clips: list[dict], config: dict, backend: str,
             features_for_clip=None) -> dict:
    import torch
    predictions = []
    timings = []
    device = device_for(config) if backend == "torch" else "cpu"
    for clip in clips:
        inputs = features_for_clip(clip) if features_for_clip is not None else _audio_features(processor, clip)
        if backend == "torch":
            inputs = inputs.to(device)
        clip_times = []
        decoded = None
        for _ in range(config["timing_repeats"]):
            if backend == "torch" and device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                tokens = model.generate(**inputs, language="en", task="transcribe",
                                        temperature=0.0, do_sample=False, num_beams=1,
                                        max_new_tokens=config["max_new_tokens"],
                                        condition_on_prev_tokens=False)
            if backend == "torch" and device == "cuda":
                torch.cuda.synchronize()
            clip_times.append(time.perf_counter() - start)
            text = processor.batch_decode(tokens, skip_special_tokens=True)[0]
            if decoded is not None and decoded != text:
                raise RuntimeError("Non-deterministic decoding across timing repeats")
            decoded = text
        predictions.append({"id": clip["id"], "reference": clip["text"],
                            "hypothesis": decoded, "seconds": clip["seconds"],
                            "latency_s": clip_times})
        timings.extend(clip_times)
    score = corpus_wer([(x["reference"], x["hypothesis"]) for x in predictions])
    return {"wer": score["wer"], "errors": score["errors"],
            "reference_words": score["reference_words"], "clip_wer": score["clip_wer"],
            "latency_median_s": statistics.median(timings), "latency_repeats_s": timings,
            "predictions": predictions, "backend": backend}
