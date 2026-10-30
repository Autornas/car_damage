"""
hashing.py

Perceptual-hash (dHash) computation and caching for every image under
data/raw/ -- data/image_hashes.json is the persistent cache, keyed by
filename, so a given image is only ever hashed once. Reused by
split_registry.py (leakage-safe split assignment) and
scripts/find_duplicates.py (near-duplicate reporting) so hashing/
Hamming-distance logic exists exactly once.

Uses a difference hash (adjacent-pixel-brightness comparison on an 8x9
grayscale thumbnail -> 64 bits) rather than a DCT-based pHash: simple,
needs nothing beyond cv2 + numpy (both already required), and robust
enough to recompression/re-encoding for this project's dedup needs. Not
robust to rotation or flipping -- an accepted limitation, not a bug.

Allowed imports: standard library, opencv-python, numpy.
"""
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HASHES_PATH = PROJECT_ROOT / "data" / "image_hashes.json"

HASH_SIZE = 8  # -> 64-bit hash
DEFAULT_DEDUP_THRESHOLD = 5  # max Hamming distance (out of 64 bits) to call two images near-duplicates


def compute_phash(image_path, hash_size=HASH_SIZE):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image for hashing: {image_path}")
    return _compute_phash(image, hash_size)


def compute_phash_from_bytes(image_bytes, hash_size=HASH_SIZE):
    """Computes the same difference hash as compute_phash for encoded image bytes."""
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("Failed to decode image bytes for hashing")
    return _compute_phash(image, hash_size)


def _compute_phash(image, hash_size):
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]

    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return format(value, f"0{hash_size * hash_size // 4}x")


def hamming_distance(hash_a, hash_b):
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


def load_hash_cache(path=None):
    path = Path(path) if path else DEFAULT_HASHES_PATH
    if not path.is_file():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_hash_cache(cache, path=None):
    path = Path(path) if path else DEFAULT_HASHES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def get_or_compute_phash(image_path, filename, source, cache=None, path=None, persist=True):
    """
    Returns filename's phash, computing + caching it only if it isn't
    already in the cache -- never recomputes a hash for a filename that's
    been seen before.

    Pass an explicit `cache` dict (e.g. from a prior load_hash_cache())
    to batch several lookups against the same in-memory cache; the caller
    then owns persisting it via save_hash_cache(). With no `cache` given,
    this loads the cache itself for a single one-off lookup and persists
    the result immediately (unless persist=False).
    """
    owns_cache = cache is None
    if cache is None:
        cache = load_hash_cache(path)

    entry = cache.get(filename)
    if entry is not None:
        return entry["phash"]

    phash = compute_phash(image_path)
    cache[filename] = {"phash": phash, "source": source}
    if owns_cache and persist:
        save_hash_cache(cache, path)
    return phash


def find_near_duplicate(phash, cache, threshold=DEFAULT_DEDUP_THRESHOLD, exclude_filename=None):
    """
    Scans `cache` (filename -> {"phash": ...}) for the closest entry
    within `threshold` Hamming distance of phash. Returns (filename,
    distance) for the closest match, or None if nothing qualifies.
    """
    best = None
    for fname, entry in cache.items():
        if fname == exclude_filename:
            continue
        distance = hamming_distance(phash, entry["phash"])
        if distance <= threshold and (best is None or distance < best[1]):
            best = (fname, distance)
    return best


def find_duplicate_groups(cache, threshold=DEFAULT_DEDUP_THRESHOLD):
    """
    Union-find over cache's filenames: two filenames land in the same
    group iff their phash Hamming distance is <= threshold. Returns a
    list of groups (each a sorted list of filenames); singletons (no
    duplicate found) are excluded.
    """
    filenames = list(cache.keys())
    parent = {f: f for f in filenames}

    def find(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, f1 in enumerate(filenames):
        for f2 in filenames[i + 1:]:
            if hamming_distance(cache[f1]["phash"], cache[f2]["phash"]) <= threshold:
                union(f1, f2)

    groups = {}
    for f in filenames:
        root = find(f)
        groups.setdefault(root, []).append(f)

    return [sorted(g) for g in groups.values() if len(g) > 1]
