#!/usr/bin/env python
"""
Step 2/3: Score one bank block's worth of intrinsic samples (SLURM worker).

Loads setup files from the shared run directory, processes one block of the bank,
and saves per-detector incoherent log-likelihood scores.

Usage (direct):
    python scripts/incoherent_block.py \
        --run-dir /data/matiasz/dot-pe/incoherent_1m/aligned_nrsur_into_xphm \
        --bank-dir /data/matiasz/dot-pe/bank_1m_xphm \
        --block-id 0

Usage (SLURM, called by slurm_incoherent_block.sh):
    Receives SLURM_ARRAY_TASK_ID as block index.

Output:
    {run_dir}/scores/block_{block_id}.npz
    Contains: inds (absolute bank indices), lnlike_di (n_det × n_samples)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import time
import pickle
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

from cogwheel.waveform import WaveformGenerator
from cogwheel.waveform_models import nrsurrogate  # noqa
from cogwheel.waveform_models import xode  # noqa — registers IMRPhenomXODE
from dot_pe.inference import run_for_single_detector, extract_single_detector_event_data
from dot_pe.likelihood_calculating import LinearFree


def main():
    parser = argparse.ArgumentParser(description="Incoherent scoring for one bank block")
    parser.add_argument("--run-dir", required=True, help="Run directory (from setup_incoherent.py)")
    parser.add_argument("--bank-dir", required=True, help="Path to waveform bank")
    parser.add_argument("--block-id", type=int, required=True, help="Block index to process")
    parser.add_argument("--n-phi", type=int, default=50, help="Phase grid size")
    parser.add_argument("--n-t", type=int, default=128, help="Time grid size")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    setup_dir = run_dir / "setup"
    bank_dir = Path(args.bank_dir)
    block_id = args.block_id

    output_path = run_dir / "scores" / f"block_{block_id}.npz"
    if output_path.exists():
        print(f"Block {block_id}: output already exists, skipping.")
        return

    t_start = time.time()

    # --- Load setup files ---
    with open(setup_dir / "event_data.pkl", "rb") as f:
        event_data = pickle.load(f)

    with open(setup_dir / "par_dic_0.json") as f:
        par_dic_0 = json.load(f)

    with open(setup_dir / "bank_config.json") as f:
        bank_config = json.load(f)

    fbin = np.array(bank_config["fbin"])
    m_arr = np.array(bank_config["m_arr"])
    blocksize = bank_config["blocksize"]
    approximant = bank_config["approximant"]

    # --- Compute indices for this block ---
    start_idx = block_id * blocksize
    end_idx = min(start_idx + blocksize, bank_config["bank_size"])
    inds = np.arange(start_idx, end_idx)
    n_samples = len(inds)

    print(f"Block {block_id}: samples [{start_idx}, {end_idx}) = {n_samples} samples")

    # --- Load pre-computed summary weights ---
    summaries = {}
    for det_name in event_data.detector_names:
        summary_path = setup_dir / f"summary_{det_name}.npz"
        data = np.load(summary_path)
        summaries[det_name] = (data["dh_weights"], data["hh_weights"])

    # --- Evaluate per-detector likelihoods ---
    lnlike_di = np.zeros((len(event_data.detector_names), n_samples))
    h_impb = None

    for d, det_name in enumerate(event_data.detector_names):
        t0 = time.time()
        temp = run_for_single_detector(
            event_data,
            det_name,
            par_dic_0,
            bank_dir,
            inds,
            fbin,
            h_impb,
            approximant,
            args.n_phi,
            n_samples,  # single_detector_blocksize = full block
            m_arr,
            args.n_t,
            size_limit=10**7,
            precomputed_summary=summaries[det_name],
        )

        if h_impb is None:
            lnlike_di[d] = temp[0]
            h_impb = temp[1]
        else:
            lnlike_di[d] = temp

        elapsed_det = time.time() - t0
        print(f"  {det_name}: max lnL = {lnlike_di[d].max():.2f}, "
              f"min = {lnlike_di[d].min():.2f} ({elapsed_det:.1f} s)")

    # --- Save output ---
    incoherent = np.sum(lnlike_di, axis=0)
    np.savez(output_path,
             inds=inds,
             lnlike_di=lnlike_di,
             incoherent=incoherent)

    elapsed_total = time.time() - t_start
    print(f"Block {block_id}: done in {elapsed_total:.1f} s "
          f"(max incoherent lnL = {incoherent.max():.2f})")


if __name__ == "__main__":
    main()
