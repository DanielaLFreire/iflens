"""Helpers that make a run reproducible and auditable (seeds, hashes, per-image record)."""
import hashlib
import json
import random
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def derive_seed(master_seed: int, *parts: str) -> int:
    """Stable 63-bit seed from the master seed and labels (e.g. image name, stage).

    Independent of iteration order, so adding or removing images does not change
    the seed (hence the variation) of the others. Python's hash() is NOT used
    because it is salted per process.
    """
    text = "|".join([str(master_seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big") >> 1


def sample_names(names: list[str], k: int, master_seed: int) -> list[str]:
    """Reproducible sample without replacement; independent of the input order."""
    rng = random.Random(derive_seed(master_seed, "sample"))
    return rng.sample(sorted(names), k)


def write_record(path: str | Path, record: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)


ASPECT_TOLERANCE = 0.005  # relative; the editing model fixes height at 768 px but keeps the ratio


def aspect_deviation(size_a: tuple[int, int] | None, size_b: tuple[int, int] | None) -> float | None:
    """Relative difference between width/height of two images (None if unknown)."""
    if not size_a or not size_b:
        return None
    ra, rb = size_a[0] / size_a[1], size_b[0] / size_b[1]
    return abs(ra - rb) / ra
