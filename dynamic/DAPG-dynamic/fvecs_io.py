# ============================================================
# DAPG Dataset Conversion Utilities
# Author: Solmaz Seyed Monir
# Project: Distance-Aware Pruned Graph (DAPG)
# Description:
#   Utility functions for reading and writing fvecs/ivecs files
#   used in ANN benchmark preprocessing and evaluation.
# ============================================================

import os
import struct
from typing import Tuple

import numpy as np


def read_fvecs(path: str) -> Tuple[np.ndarray, int, int]:
    """Read standard fvecs: [int32 dim][float32 * dim] repeated."""
    a = np.fromfile(path, dtype=np.int32)
    if a.size == 0:
        raise ValueError(f"Empty file: {path}")
    dim = int(a[0])
    if dim <= 0:
        raise ValueError(f"Bad dim {dim} in {path}")
    if a.size % (dim + 1) != 0:
        raise ValueError(f"File size mismatch for fvecs: {path} (dim={dim})")
    n = a.size // (dim + 1)
    x = a.reshape(n, dim + 1)[:, 1:].view(np.float32).copy()
    return x, n, dim


def write_fvecs(path: str, x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    n, dim = x.shape
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        # Fast chunked writer for fvecs: per row [int32 dim][float32 * dim].
        # Writing per-vector in Python is extremely slow for large datasets (e.g., SIFT1M),
        # so we emit blocks using a structured dtype.
        dt = np.dtype([("d", "<i4"), ("x", "<f4", (dim,))])
        chunk = 65536  # ~64K vectors per chunk (tunable)
        for i0 in range(0, n, chunk):
            i1 = min(n, i0 + chunk)
            block = np.empty((i1 - i0,), dtype=dt)
            block["d"] = dim
            block["x"] = x[i0:i1]
            block.tofile(f)


def write_ivecs(path: str, ids: np.ndarray):
    """Write ivecs: for each row write int32 k then k int32 ids."""
    ids = np.asarray(ids, dtype=np.int32)
    if ids.ndim != 2:
        raise ValueError("ids must be 2D")
    k = int(ids.shape[1])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        for i in range(ids.shape[0]):
            f.write(struct.pack("<i", k))
            f.write(ids[i].tobytes(order="C"))


