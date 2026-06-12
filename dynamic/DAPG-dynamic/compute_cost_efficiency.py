# ============================================================
# DAPG Cost-Efficiency Evaluation
# Author: Solmaz Seyed Monir
# Project: Distance-Aware Pruned Graph (DAPG)
# Description:
#   Computes recall-aware cost-efficiency for DAPG and baseline
#  dynamic update-aware vector search 
# ============================================================

import argparse
import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Curve:
    recall: np.ndarray
    qps: np.ndarray


def pareto_frontier_max_qps(recall: np.ndarray, qps: np.ndarray) -> Curve:
    a = np.array(list(zip(recall, qps)), dtype=float)
    a = a[~np.isnan(a).any(axis=1)]
    if a.size == 0:
        return Curve(np.array([], dtype=float), np.array([], dtype=float))
    a = a[a[:, 0].argsort()]  # recall asc
    keep: List[Tuple[float, float]] = []
    best = -1.0
    for r, q in a[::-1]:
        if q > best:
            keep.append((float(r), float(q)))
            best = float(q)
    keep = np.array(keep[::-1], dtype=float)
    return Curve(keep[:, 0], keep[:, 1])


def qps_at_recall(curve: Curve, target_recall: float) -> float:
    if curve.recall.size == 0:
        return float("nan")
    if target_recall < float(curve.recall.min()) or target_recall > float(curve.recall.max()):
        return float("nan")
    return float(np.interp(target_recall, curve.recall, curve.qps))


def pick_latest(patterns: List[str]) -> Optional[str]:
    cands: List[str] = []
    for p in patterns:
        cands += glob.glob(p)
    cands = sorted(set(cands), key=os.path.getmtime, reverse=True)
    return cands[0] if cands else None


def main():
    ap = argparse.ArgumentParser(description="Compute a concrete cost-efficiency metric from existing CSVs.")
    ap.add_argument("--sigmod_root", default=r"C:\Users\Soli1\SIGMOD")
    ap.add_argument("--dataset", choices=["audio", "mnist", "sift"], required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--Wtag", default="W600p000")
    ap.add_argument("--rec_target", type=float, default=0.995)
    ap.add_argument("--block_size", type=int, default=2000, help="FIFO block size B for incremental methods (ops per step).")
    args = ap.parse_args()

    sigmod = args.sigmod_root
    ds = args.dataset
    k = int(args.k)
    rec_t = float(args.rec_target)
    B = int(args.block_size)

    solmaz_indexes = os.path.join(sigmod, r"Solmaz\LSH-APG\cppCode\LSH-APG\indexes")
    fifo_out = os.path.join(sigmod, r"third_party\gate-addon-dynamic\runs", ds, "out")

    # --- inputs (auto-pick latest) ---
    solmaz_curve = pick_latest([
        os.path.join(solmaz_indexes, f"solmaz_curve_{ds}_{args.Wtag}_v*.csv"),
        os.path.join(solmaz_indexes, f"solmaz_curve_{ds}_{args.Wtag}.csv"),
    ])
    if not solmaz_curve or not os.path.exists(solmaz_curve):
        raise FileNotFoundError(f"Missing solmaz_curve for {ds} under {solmaz_indexes}")

   
    table_cands = sorted(glob.glob(os.path.join(solmaz_indexes, f"solmaz_table_{ds}_{args.Wtag}_v*.csv")), key=os.path.getmtime, reverse=True)
    solmaz_table = None
    for p in table_cands:
        try:
            t = pd.read_csv(p)
            if "UpdateCount" in t.columns and (t["UpdateCount"].fillna(0).astype(int).max() > 0):
                solmaz_table = p
                break
        except Exception:
            continue
    if not solmaz_table:
        # fallback to latest even if it has 0 updates
        solmaz_table = pick_latest([os.path.join(solmaz_indexes, f"solmaz_table_{ds}_{args.Wtag}_v*.csv")])
    if not solmaz_table or not os.path.exists(solmaz_table):
        raise FileNotFoundError(f"Missing solmaz_table for {ds} under {solmaz_indexes}")

    fifo_summary = os.path.join(fifo_out, f"fifo_summary_K{k}.csv")
    if not os.path.exists(fifo_summary):
        raise FileNotFoundError(f"Missing FIFO summary: {fifo_summary}")

    # --- load rebuild-per-window baselines ---
    fifo = pd.read_csv(fifo_summary)
    fifo["nsg_rebuild_s"] = fifo["build_knng_s"] + fifo["build_nsg_s"]
    fifo["gate_rebuild_s"] = fifo["build_knng_s"] + fifo["build_nsg_s"] + fifo["build_gate_routing_s"]
    nsg_rebuild_s = float(fifo["nsg_rebuild_s"].mean())
    gate_rebuild_s = float(fifo["gate_rebuild_s"].mean())

    def avg_fifo_qps(mode: str) -> float:
        files = sorted(glob.glob(os.path.join(fifo_out, "step_*", f"{mode}_curve_k{k}.csv")))
        vals = []
        for p in files:
            df = pd.read_csv(p)
            rcol = f"recall{k}"
            if rcol not in df.columns:
                continue
            curve = pareto_frontier_max_qps(df[rcol].astype(float).to_numpy(), df["qps"].astype(float).to_numpy())
            v = qps_at_recall(curve, rec_t)
            if np.isfinite(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    nsg_fifo_qps = avg_fifo_qps("nsg")
    gate_fifo_qps = avg_fifo_qps("gate")

   
    cdf = pd.read_csv(solmaz_curve)
    cdf = cdf[(cdf["SolmazDataset"] == ds) & (cdf["k"] == k)].copy()
    # pick any UpdateCount available (prefer max)
    if "UpdateCount" in cdf.columns and cdf["UpdateCount"].notna().any():
        uc = int(cdf["UpdateCount"].max())
        cdf = cdf[cdf["UpdateCount"] == uc].copy()
    else:
        uc = 0

    def solmaz_qps(method: str) -> float:
        s = cdf[cdf["SolmazMethod"] == method]
        if s.empty:
            return float("nan")
        curve = pareto_frontier_max_qps(s["Recall"].astype(float).to_numpy(), s["QPS"].astype(float).to_numpy())
        return qps_at_recall(curve, rec_t)

    # --- load solmaz table (maintenance ms/op) ---
    tdf = pd.read_csv(solmaz_table)
    tdf = tdf[(tdf["SolmazDataset"] == ds) & (tdf["k"] == k)].copy()
    # pick updatecount>0 if present
    if "UpdateCount" in tdf.columns and (tdf["UpdateCount"].fillna(0).astype(int).max() > 0):
        update_count = int(tdf["UpdateCount"].fillna(0).astype(int).max())
        tdf = tdf[tdf["UpdateCount"].astype(int) == update_count].copy()
    else:
        update_count = 0

    tdf["UpdateOp_ms"] = tdf["InsertAvg_ms"].astype(float) + tdf["DeleteAvg_ms"].astype(float)
    tdf["MaintStep_s"] = (tdf["UpdateOp_ms"] * B) / 1000.0


    rows = []
    for _, r in tdf.iterrows():
        method = str(r["SolmazMethod"])
        maint_s = float(r["MaintStep_s"]) if np.isfinite(r["MaintStep_s"]) else float("nan")
        qps = solmaz_qps(method)
        eff = (rec_t * qps / maint_s) if (maint_s > 0 and np.isfinite(qps)) else float("nan")
        rows.append((method, maint_s, qps, eff))

    # add FIFO rebuild baselines
    rows.append(("NSG+FIFO(rebuild)", nsg_rebuild_s, nsg_fifo_qps, (rec_t * nsg_fifo_qps / nsg_rebuild_s) if np.isfinite(nsg_fifo_qps) else float("nan")))
    rows.append(("GATE+FIFO(rebuild)", gate_rebuild_s, gate_fifo_qps, (rec_t * gate_fifo_qps / gate_rebuild_s) if np.isfinite(gate_fifo_qps) else float("nan")))

    out = pd.DataFrame(rows, columns=["Method", "MaintStep_s", f"QPS@R={rec_t}", "CostEfficiency"])
    out = out.sort_values("CostEfficiency", ascending=False)

    print("Using:")
    print(f"  dataset={ds} k={k} rec_target={rec_t} block_size(B)={B}")
    print(f"  solmaz_curve={solmaz_curve} (curve UpdateCount={uc})")
    print(f"  solmaz_table={solmaz_table} (table UpdateCount={update_count})")
    print(f"  fifo_summary={fifo_summary}")
    print()
    print(out.to_string(index=False, float_format=lambda x: f\"{x:.6f}\"))


if __name__ == "__main__":
    main()


