"""Paired speaker bootstrap of absolute WER differences."""
import random
from .core import word_errors


def paired_ci(base: dict, candidate: dict, clips: list[dict], seed: int,
              repeats: int = 2000) -> list[float]:
    speakers = {clip["id"]: str(clip["speaker_id"]) for clip in clips}
    groups: dict[str, list[tuple[int, int, int]]] = {}
    for left, right in zip(base["predictions"], candidate["predictions"], strict=True):
        if left["id"] != right["id"] or left["reference"] != right["reference"]:
            raise RuntimeError("Paired predictions do not align")
        a, count = word_errors(left["reference"], left["hypothesis"])
        b, other_count = word_errors(right["reference"], right["hypothesis"])
        if count != other_count:
            raise RuntimeError("Reference lengths differ")
        groups.setdefault(speakers[left["id"]], []).append((a, b, count))
    rng = random.Random(seed)
    keys = sorted(groups)
    draws = []
    for _ in range(repeats):
        totals = [0, 0, 0]
        for _ in keys:
            for row in groups[rng.choice(keys)]:
                for i in range(3):
                    totals[i] += row[i]
        draws.append((totals[1] - totals[0]) / totals[2])
    draws.sort()
    return [draws[int(repeats * .025)], draws[int(repeats * .975) - 1]]
