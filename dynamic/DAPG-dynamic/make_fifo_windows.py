# ============================================================
# DAPG Dynamic Window and Ground-Truth Generator
# Author: Solmaz Seyed Monir
# Project: Distance-Aware Pruned Graph (DAPG)
# Description:
#   Generates dynamic ANN benchmark windows and per-window
#   ground-truth nearest-neighbor files for FIFO and random-new
#   insert/delete workloads.
# ============================================================

import argparse
import os
from typing import Literal

import numpy as np

from fvecs_io import read_fvecs, write_fvecs, write_ivecs


def compute_gt_faiss(
    xb: np.ndarray,
    xq: np.ndarray,
    k: int,
    metric: Literal["l2", "ip"] = "l2",
    normalize: bool = False,
    batch: int = 4096,
) -> np.ndarray:
    import faiss  # type: ignore

    xb = np.ascontiguousarray(xb, dtype=np.float32)
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    if normalize:
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)

    d = xb.shape[1]
    if metric == "l2":
        index = faiss.IndexFlatL2(d)
    elif metric == "ip":
        index = faiss.IndexFlatIP(d)
    else:
        raise ValueError(f"Unknown metric={metric}")
    index.add(xb)

    out = np.empty((xq.shape[0], k), dtype=np.int32)
    for i0 in range(0, xq.shape[0], batch):
        i1 = min(xq.shape[0], i0 + batch)
        _, I = index.search(xq[i0:i1], k)
        out[i0:i1] = I.astype(np.int32, copy=False)
    return out


def main():
    ap = argparse.ArgumentParser(description="Create dynamic base fvecs + per-window ground truth ivecs.")
    ap.add_argument("--base_fvecs", required=True)
    ap.add_argument("--query_fvecs", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window_size", type=int, required=True)
    ap.add_argument("--block_size", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True, help="Number of windows to generate")
    ap.add_argument("--gt_k", type=int, default=100)
    ap.add_argument("--metric", choices=["l2", "ip"], default="l2")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize vectors before gt (useful for cosine/IP)")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--max_base_n", type=int, default=-1, help="Debug: only use first max_base_n base points")
    ap.add_argument(
        "--workload",
        choices=["fifo", "random_new"],
        default="fifo",
        help="fifo: delete oldest/insert next; random_new: delete random active IDs and insert next unseen IDs",
    )
    ap.add_argument("--seed", type=int, default=100, help="Random seed for random_new workload")
    args = ap.parse_args()

    xb, n, dim = read_fvecs(args.base_fvecs)
    xq, nq, d2 = read_fvecs(args.query_fvecs)
    if d2 != dim:
        raise ValueError(f"Dim mismatch: base dim={dim}, query dim={d2}")

    if args.max_base_n > 0:
        xb = xb[: args.max_base_n]
        n = xb.shape[0]

    W = int(args.window_size)
    B = int(args.block_size)
    steps = int(args.steps)
    if W <= 0 or B <= 0 or steps <= 0:
        raise ValueError("window_size/block_size/steps must be > 0")
    if W > n:
        raise ValueError(f"window_size W={W} must be <= base n={n}")

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    active_ids = np.arange(W, dtype=np.int64)
    next_id = W

    for s in range(steps):
        if args.workload == "fifo":
            start = s * B
            end = start + W
            if end > n:
                print(f"Stop: step={s} would exceed base (end={end} > n={n})")
                break
            active_ids = np.arange(start, end, dtype=np.int64)
        elif s > 0:
            if next_id >= n:
                print(f"Stop: step={s} no unseen base vectors remain (next_id={next_id} >= n={n})")
                break
            replace_n = min(B, W, n - next_id)
            pos = rng.choice(W, size=replace_n, replace=False)
            active_ids[pos] = np.arange(next_id, next_id + replace_n, dtype=np.int64)
            next_id += replace_n

        step_dir = os.path.join(args.out_dir, f"step_{s:03d}")
        os.makedirs(step_dir, exist_ok=True)

        xb_w = xb[active_ids]
        base_out = os.path.join(step_dir, "base.fvecs")
        gt_out = os.path.join(step_dir, "gt.ivecs")
        meta_out = os.path.join(step_dir, "meta.txt")
        active_ids_out = os.path.join(step_dir, "active_ids.txt")

        write_fvecs(base_out, xb_w)
        np.savetxt(active_ids_out, active_ids, fmt="%d")

        gt = compute_gt_faiss(
            xb_w,
            xq,
            k=int(args.gt_k),
            metric=args.metric,
            normalize=bool(args.normalize),
            batch=int(args.batch),
        )
        write_ivecs(gt_out, gt)

        with open(meta_out, "w") as f:
            f.write(f"step={s}\n")
            if args.workload == "fifo":
                f.write(f"global_start={s * B}\n")
                f.write(f"global_end={s * B + W}\n")
            else:
                f.write(f"global_start=-1\n")
                f.write(f"global_end=-1\n")
            f.write(f"window_size={W}\n")
            f.write(f"block_size={B}\n")
            f.write(f"base_n_total={n}\n")
            f.write(f"dim={dim}\n")
            f.write(f"query_n={nq}\n")
            f.write(f"gt_k={int(args.gt_k)}\n")
            f.write(f"metric={args.metric}\n")
            f.write(f"normalize={bool(args.normalize)}\n")
            f.write(f"workload={args.workload}\n")
            f.write(f"seed={int(args.seed)}\n")
            f.write(f"active_ids=active_ids.txt\n")

        print(f"Wrote {args.workload} window step={s}: {step_dir}")


if __name__ == "__main__":
    main()


