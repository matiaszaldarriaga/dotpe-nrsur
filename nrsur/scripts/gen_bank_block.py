#!/usr/bin/env python
"""
Generate waveform bank blocks as a standalone process.

This is the fundamental unit of parallelism for bank generation.
Each invocation generates one or more waveform blocks, writing
amplitudes_block_{i}.npy and phase_block_{i}.npy to the bank's
waveform directory.

Thread env vars are set BEFORE any imports to prevent LAL/FFTW
internal thread spawning — the only reliable way to get clean
single-threaded execution per process.

Usage:
    python scripts/gen_bank_block.py \
        --bank-dir /data/matiasz/dot-pe/comparison_65k/bank_nrsur \
        --blocks 0 1 2 3 \
        --approximant NRSur7dq4 \
        [--blocksize 4096]
"""

import os

# ── Thread isolation: must happen before ANY library import ──
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
import json
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cogwheel import data, waveform
from cogwheel.waveform import WaveformGenerator
from dot_pe import config as dotpe_config
from dot_pe.waveform_banks import _gen_waveforms_from_index


def setup_logger(waveform_dir, block_idx):
    """Per-block log file to avoid contention between processes."""
    log_path = waveform_dir / f"block_{block_idx}.log"
    logger = logging.getLogger(f"block_{block_idx}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    # Also log to stderr for interactive monitoring
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        logging.Formatter("%(asctime)s block_%(name)s %(message)s")
    )
    logger.addHandler(stderr_handler)
    return logger


def block_is_complete(waveform_dir, block_idx, min_bytes=200):
    """Check if both .npy files exist and are non-trivially sized."""
    amp = waveform_dir / f"amplitudes_block_{block_idx}.npy"
    phase = waveform_dir / f"phase_block_{block_idx}.npy"
    return (
        amp.exists()
        and phase.exists()
        and amp.stat().st_size > min_bytes
        and phase.stat().st_size > min_bytes
    )


def clean_partial(waveform_dir, block_idx):
    """Remove partial files (one .npy but not the other) before regenerating."""
    amp = waveform_dir / f"amplitudes_block_{block_idx}.npy"
    phase = waveform_dir / f"phase_block_{block_idx}.npy"
    amp_exists = amp.exists()
    phase_exists = phase.exists()
    if amp_exists != phase_exists:
        # One exists but not the other → partial
        if amp_exists:
            amp.unlink()
        if phase_exists:
            phase.unlink()
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate waveform bank blocks (standalone process)"
    )
    parser.add_argument(
        "--bank-dir",
        type=Path,
        required=True,
        help="Bank directory containing bank_config.json and waveforms/",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        required=True,
        help="Block indices to generate (e.g. 0 1 2 3)",
    )
    parser.add_argument(
        "--approximant",
        type=str,
        required=True,
        help="Waveform approximant (e.g. NRSur7dq4, IMRPhenomXPHM)",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=4096,
        help="Samples per block (default: 4096)",
    )
    args = parser.parse_args()

    bank_dir = args.bank_dir
    waveform_dir = bank_dir / "waveforms"
    waveform_dir.mkdir(parents=True, exist_ok=True)

    # Register custom approximants if needed
    if "nrsur" in args.approximant.lower():
        from cogwheel.waveform_models import nrsurrogate  # noqa: F401
    if "xode" in args.approximant.lower():
        from cogwheel.waveform_models import xode  # noqa: F401

    # ── Read bank config ──
    config_path = bank_dir / "bank_config.json"
    with open(config_path, "r", encoding="utf-8") as fp:
        bank_config = json.load(fp)
    fbin = np.array(bank_config["fbin"])
    f_ref = bank_config["f_ref"]

    # ── Build override dict (replicates waveform_banks.py line 287) ──
    override_dic = dotpe_config.DEFAULT_PARAMS_DICT | {"f_ref": f_ref}

    # ── Load samples ──
    samples_path = bank_dir / "intrinsic_sample_bank.feather"
    intrinsic_samples = pd.read_feather(samples_path)

    # ── Create WaveformGenerator ──
    dummy_ed = data.EventData.gaussian_noise(
        "", **dotpe_config.EVENT_DATA_KWARGS
    )
    wfg = WaveformGenerator.from_event_data(dummy_ed, args.approximant)

    # Zero out precession for aligned-spin approximants
    if waveform.APPROXIMANTS[args.approximant].aligned_spins:
        intrinsic_samples[["s1x_n", "s1y_n", "s2x_n", "s2y_n"]] = 0.0

    t_start = time.time()
    n_done = 0
    n_skipped = 0

    for block_idx in args.blocks:
        logger = setup_logger(waveform_dir, block_idx)

        # Skip already-complete blocks
        if block_is_complete(waveform_dir, block_idx):
            logger.info("Already complete, skipping")
            n_skipped += 1
            continue

        # Clean partial files
        if clean_partial(waveform_dir, block_idx):
            logger.info("Cleaned partial files")

        logger.info(
            "Starting generation (approximant=%s, blocksize=%d)",
            args.approximant,
            args.blocksize,
        )
        t_block = time.time()

        _gen_waveforms_from_index(
            wfg=wfg,
            samples=intrinsic_samples,
            i=block_idx,
            waveform_dir=waveform_dir,
            blocksize=args.blocksize,
            fbin=fbin,
            override_dic=override_dic,
            logger=logger,
            t0=t_start,
        )

        elapsed = time.time() - t_block
        logger.info("Complete in %.1f s", elapsed)
        n_done += 1

    total_elapsed = time.time() - t_start
    print(
        f"Done: {n_done} generated, {n_skipped} skipped, "
        f"{total_elapsed:.1f} s total",
        flush=True,
    )


if __name__ == "__main__":
    main()
