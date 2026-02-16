#!/usr/bin/env python
"""
Step 3/3: Merge per-block incoherent scores and apply threshold.

Loads all block_*.npz files from the scores directory, concatenates them,
applies the incoherent threshold, and saves the survivor list.

Usage:
    python scripts/merge_incoherent.py \
        --run-dir /data/matiasz/dot-pe/incoherent_1m/aligned_nrsur_into_xphm

Output:
    {run_dir}/
    ├── incoherent_all.npz       (full 1M scores: inds, lnlike_di, incoherent)
    ├── survivors.npz            (filtered: survivor_inds, survivor_lnlike_di, threshold)
    └── summary.json             (N_survivors, max_lnL, threshold, per-detector stats)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import json
import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge incoherent scores and apply threshold")
    parser.add_argument("--run-dir", required=True, help="Run directory")
    parser.add_argument("--max-drop", type=float, default=20.0,
                        help="Threshold: max(lnL) - max_drop (default: 20)")
    parser.add_argument("--save-full", action="store_true",
                        help="Also save the full 1M score array (large!)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    scores_dir = run_dir / "scores"
    setup_dir = run_dir / "setup"

    # --- Load run metadata ---
    with open(setup_dir / "run_meta.json") as f:
        run_meta = json.load(f)

    n_blocks = run_meta["n_blocks"]
    bank_size = run_meta["bank_size"]
    n_det = len(run_meta["detectors"])

    # --- Check completeness ---
    score_files = sorted(scores_dir.glob("block_*.npz"))
    found_blocks = set()
    for f in score_files:
        block_id = int(f.stem.split("_")[1])
        found_blocks.add(block_id)

    expected_blocks = set(range(n_blocks))
    missing = expected_blocks - found_blocks
    if missing:
        print(f"WARNING: {len(missing)} blocks missing: {sorted(missing)[:20]}...")
        print("Run merge anyway? Proceeding with available blocks.")

    print(f"Found {len(found_blocks)}/{n_blocks} block files")

    # --- Load and concatenate ---
    all_inds = []
    all_lnlike_di = []
    all_incoherent = []

    for block_id in sorted(found_blocks):
        data = np.load(scores_dir / f"block_{block_id}.npz")
        all_inds.append(data["inds"])
        all_lnlike_di.append(data["lnlike_di"])
        all_incoherent.append(data["incoherent"])

    inds = np.concatenate(all_inds)
    lnlike_di = np.concatenate(all_lnlike_di, axis=1)
    incoherent = np.concatenate(all_incoherent)

    print(f"Total samples: {len(inds):,}")
    print(f"Detector scores shape: {lnlike_di.shape}")

    # --- Apply threshold ---
    max_lnL = incoherent.max()
    threshold = max_lnL - args.max_drop
    selected = incoherent >= threshold
    n_survivors = selected.sum()

    survivor_inds = inds[selected]
    survivor_lnlike_di = lnlike_di[:, selected]
    survivor_incoherent = incoherent[selected]

    print(f"\nMax incoherent lnL: {max_lnL:.2f}")
    print(f"Threshold (drop={args.max_drop}): {threshold:.2f}")
    print(f"Survivors: {n_survivors:,} / {len(inds):,} ({100*n_survivors/len(inds):.3f}%)")

    # --- Per-detector stats ---
    det_stats = {}
    for d, det_name in enumerate(run_meta["detectors"]):
        det_max = lnlike_di[d].max()
        det_surv_max = survivor_lnlike_di[d].max() if n_survivors > 0 else float("-inf")
        det_stats[det_name] = {
            "max_lnL": float(det_max),
            "survivor_max_lnL": float(det_surv_max),
        }
        print(f"  {det_name}: max lnL = {det_max:.2f}")

    # --- Save results ---
    np.savez(run_dir / "survivors.npz",
             survivor_inds=survivor_inds,
             survivor_lnlike_di=survivor_lnlike_di,
             survivor_incoherent=survivor_incoherent,
             threshold=threshold,
             max_lnL=max_lnL)

    if args.save_full:
        np.savez(run_dir / "incoherent_all.npz",
                 inds=inds,
                 lnlike_di=lnlike_di,
                 incoherent=incoherent)
        print(f"  Saved full scores to {run_dir / 'incoherent_all.npz'}")

    summary = {
        "bank_dir": run_meta["bank_dir"],
        "bank_size": int(len(inds)),
        "n_blocks_found": len(found_blocks),
        "n_blocks_expected": n_blocks,
        "approximant": run_meta["approximant"],
        "n_modes": run_meta["n_modes"],
        "max_incoherent_lnL": float(max_lnL),
        "threshold": float(threshold),
        "max_drop": args.max_drop,
        "n_survivors": int(n_survivors),
        "survivor_fraction": float(n_survivors / len(inds)),
        "detectors": run_meta["detectors"],
        "per_detector": det_stats,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {run_dir}/")
    print(f"  survivors.npz: {n_survivors:,} survivors")
    print(f"  summary.json: run summary")


if __name__ == "__main__":
    main()
