# ============================================================
# DAPG Dynamic Baseline Experiment Runner
# Author: Solmaz Seyed Monir
# Project: Distance-Aware Pruned Graph (DAPG)
# Description:
#   Runs separated dynamic baseline experiments for NSG, GATE,
#   FAISS-HNSW rebuild, and optional hnswlib lazy deletion.
# ============================================================

import argparse
import os
import struct
import subprocess
from typing import List

import numpy as np


def run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def read_fbin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.read(8)
    if len(header) != 8:
        raise RuntimeError(f"Short fbin header: {path}")
    n, d = struct.unpack("<ii", header)
    if n <= 0 or d <= 0:
        raise RuntimeError(f"Bad fbin header in {path}: n={n} d={d}")
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(n, d))


def write_fvecs(path: str, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float32)
    n, dim = x.shape
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dt = np.dtype([("d", "<i4"), ("x", "<f4", (dim,))])
    with open(path, "wb") as f:
        for i0 in range(0, n, 65536):
            i1 = min(n, i0 + 65536)
            block = np.empty((i1 - i0,), dtype=dt)
            block["d"] = dim
            block["x"] = x[i0:i1]
            block.tofile(f)


def ensure_fvecs(path: str, out_path: str, max_n: int = -1) -> str:
    """Accept .fvecs directly; convert .fbin into a local .fvecs copy."""
    if os.path.exists(path) and path.lower().endswith(".fvecs"):
        return path
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if not path.lower().endswith(".fbin"):
        raise ValueError(f"Expected .fvecs or .fbin input, got: {path}")

    x = read_fbin(path)
    n = int(x.shape[0])
    if max_n > 0:
        n = min(n, int(max_n))
    if os.path.exists(out_path):
        print(f"Using existing converted fvecs: {out_path}", flush=True)
        return out_path
    print(f"Converting fbin -> fvecs: {path} -> {out_path} (n={n}, dim={x.shape[1]})", flush=True)
    write_fvecs(out_path, np.array(x[:n], dtype=np.float32, copy=True))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a separated dynamic-baseline experiment for NSG, GATE, FAISS-HNSW rebuild, "
            "and optionally hnswlib lazy deletion. This wrapper keeps outputs away from the "
            "older static/table experiments."
        )
    )
    parser.add_argument("--exp", required=True, help="Experiment tag, e.g. sift_fifo_k100")
    parser.add_argument("--base_fvecs", required=True, help="Base vectors in .fvecs, or .fbin which will be converted locally.")
    parser.add_argument("--query_fvecs", required=True, help="Query vectors in .fvecs, or .fbin which will be converted locally.")
    parser.add_argument("--gate_root", default="third_party/gate-v0.0.1")
    parser.add_argument("--out_root", default="dynamic_experiments")
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--Ls", nargs="+", type=int, default=[100, 120, 150, 200, 300, 400])
    parser.add_argument("--metric", choices=["l2", "ip"], default="l2")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--max_base_n", type=int, default=-1)
    parser.add_argument(
        "--workload",
        choices=["fifo", "random_new"],
        default="fifo",
        help="fifo or random_new (random delete active + insert unseen)",
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--mode", choices=["nsg", "gate", "both"], default="both")
    parser.add_argument("--run_hnsw_rebuild", action="store_true", help="Run FAISS-HNSW rebuild per FIFO window.")
    parser.add_argument("--run_hnsw_lazy", action="store_true", help="Run hnswlib lazy delete/insert FIFO baseline.")

    # NSG/GATE build parameters.
    parser.add_argument("--knng_k", type=int, default=200)
    parser.add_argument("--knng_M", type=int, default=32)
    parser.add_argument("--knng_efC", type=int, default=200)
    parser.add_argument("--knng_efS", type=int, default=200)
    parser.add_argument("--nsg_L", type=int, default=200)
    parser.add_argument("--nsg_R", type=int, default=64)
    parser.add_argument("--nsg_C", type=int, default=500)
    parser.add_argument("--iter1", type=int, default=8)
    parser.add_argument("--iter2", type=int, default=8)

    # HNSW parameters.
    parser.add_argument("--hnsw_M", type=int, default=32)
    parser.add_argument("--hnsw_efC", type=int, default=200)
    parser.add_argument("--hnsw_efs", nargs="+", type=int, default=[100, 120, 150, 200, 300, 400])
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    base_input = os.path.abspath(args.base_fvecs)
    query_input = os.path.abspath(args.query_fvecs)
    gate_root = os.path.abspath(args.gate_root)
    out_root = os.path.abspath(args.out_root)
    exp_dir = os.path.join(out_root, args.exp)
    inputs_dir = os.path.join(exp_dir, "inputs")
    windows_dir = os.path.join(exp_dir, "fifo_windows")
    baselines_dir = os.path.join(exp_dir, "baselines")
    os.makedirs(exp_dir, exist_ok=True)

    base_fvecs = ensure_fvecs(
        base_input,
        os.path.join(inputs_dir, "base.fvecs"),
        max_n=int(args.max_base_n),
    )
    query_fvecs = ensure_fvecs(
        query_input,
        os.path.join(inputs_dir, "query.fvecs"),
        max_n=-1,
    )

    readme_path = os.path.join(exp_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("Separated dynamic baseline experiment.\n")
        f.write("This directory is intentionally separate from older static APG/DAPG tables and curves.\n")
        f.write(f"exp={args.exp}\n")
        f.write(f"base_input={base_input}\n")
        f.write(f"query_input={query_input}\n")
        f.write(f"base_fvecs={base_fvecs}\n")
        f.write(f"query_fvecs={query_fvecs}\n")
        f.write(f"window_size={args.window_size}\n")
        f.write(f"block_size={args.block_size}\n")
        f.write(f"steps={args.steps}\n")
        f.write(f"K={args.K}\n")
        f.write(f"mode={args.mode}\n")
        f.write(f"workload={args.workload}\n")
        f.write(f"seed={int(args.seed)}\n")

    make_fifo = os.path.join(script_dir, "make_fifo_windows.py")
    run_gate_fifo = os.path.join(script_dir, "run_gate_fifo.py")
    run_hnsw_lazy = os.path.join(script_dir, "run_hnswlib_lazy_fifo.py")

    fifo_cmd = [
        "python3",
        make_fifo,
        "--base_fvecs",
        base_fvecs,
        "--query_fvecs",
        query_fvecs,
        "--out_dir",
        windows_dir,
        "--window_size",
        str(int(args.window_size)),
        "--block_size",
        str(int(args.block_size)),
        "--steps",
        str(int(args.steps)),
        "--gt_k",
        str(int(args.K)),
        "--metric",
        args.metric,
        "--workload",
        args.workload,
        "--seed",
        str(int(args.seed)),
    ]
    if args.normalize:
        fifo_cmd.append("--normalize")
    if int(args.max_base_n) > 0:
        fifo_cmd += ["--max_base_n", str(int(args.max_base_n))]
    run(fifo_cmd)

    gate_cmd = [
        "python3",
        run_gate_fifo,
        "--gate_root",
        gate_root,
        "--windows_dir",
        windows_dir,
        "--out_dir",
        baselines_dir,
        "--mode",
        args.mode,
        "--K",
        str(int(args.K)),
        "--Ls",
        *[str(int(x)) for x in args.Ls],
        "--query_fvecs",
        query_fvecs,
        "--knng_k",
        str(int(args.knng_k)),
        "--knng_M",
        str(int(args.knng_M)),
        "--knng_efC",
        str(int(args.knng_efC)),
        "--knng_efS",
        str(int(args.knng_efS)),
        "--nsg_L",
        str(int(args.nsg_L)),
        "--nsg_R",
        str(int(args.nsg_R)),
        "--nsg_C",
        str(int(args.nsg_C)),
        "--iter1",
        str(int(args.iter1)),
        "--iter2",
        str(int(args.iter2)),
        "--hnsw_M",
        str(int(args.hnsw_M)),
        "--hnsw_efC",
        str(int(args.hnsw_efC)),
        "--hnsw_efs",
        *[str(int(x)) for x in args.hnsw_efs],
    ]
    if args.run_hnsw_rebuild:
        gate_cmd.append("--run_hnsw")
    run(gate_cmd)

    if args.run_hnsw_lazy:
        lazy_csv = os.path.join(exp_dir, f"hnswlib_lazy_fifo_k{int(args.K)}.csv")
        lazy_cmd = [
            "python3",
            run_hnsw_lazy,
            "--windows_dir",
            windows_dir,
            "--query_fvecs",
            query_fvecs,
            "--out_csv",
            lazy_csv,
            "--k",
            str(int(args.K)),
            "--M",
            str(int(args.hnsw_M)),
            "--efC",
            str(int(args.hnsw_efC)),
            "--efs",
            *[str(int(x)) for x in args.hnsw_efs],
        ]
        run(lazy_cmd)

    print(f"\nDone. Separated dynamic baseline outputs are under: {exp_dir}")
    print(f"Summary CSV: {os.path.join(baselines_dir, 'fifo_summary_K' + str(int(args.K)) + '.csv')}")


if __name__ == "__main__":
    main()
