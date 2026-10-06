"""Seeded duration-stratified sampling for clips up to thirty seconds."""
import io
import random


def duration_band(seconds: float) -> str | None:
    """Whisper's single 30-second window; exclude longer clips from this study."""
    if seconds < 10:
        return "short"
    if seconds < 20:
        return "medium"
    if seconds <= 30:
        return "long"
    return None

def choose_rows_by_duration(dataset, targets: dict[str, int], seed: int,
                            excluded_ids: set[str] | None = None):
    """Independent seeded reservoirs give predefined duration strata."""
    import soundfile as sf

    rng = random.Random(seed)
    chosen = {band: [] for band in targets}
    seen = {band: 0 for band in targets}
    scanned = 0
    excluded_ids = excluded_ids or set()
    for scanned, row in enumerate(dataset, 1):
        if str(row["id"]) in excluded_ids:
            continue
        raw = row["audio"]
        if not isinstance(raw, dict) or not raw.get("bytes"):
            raise RuntimeError("Streaming dataset did not supply encoded audio bytes")
        seconds = sf.info(io.BytesIO(raw["bytes"])).duration
        band = duration_band(seconds)
        if band not in targets:
            continue
        seen[band] += 1
        if len(chosen[band]) < targets[band]:
            chosen[band].append(row)
        else:
            index = rng.randrange(seen[band])
            if index < targets[band]:
                chosen[band][index] = row
    for band, target in targets.items():
        if len(chosen[band]) != target:
            raise RuntimeError(f"Duration band {band} has only {seen[band]} rows; requested {target}")
    rows = sorted((row for group in chosen.values() for row in group), key=lambda row: str(row["id"]))
    return rows, scanned, seen
