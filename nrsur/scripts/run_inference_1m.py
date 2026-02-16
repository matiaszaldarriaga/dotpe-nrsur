#!/usr/bin/env python
"""Run coherent inference for one incoherent 1M combination.

Loads event data, survivor indices, and bank from the run's setup directory,
then calls inference.run() with standard parameters.

Usage:
    python scripts/run_inference_1m.py --run-dir /data/matiasz/dot-pe/incoherent_1m/aligned_nrsur_into_nrsur
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import pickle
import time
from functools import wraps
from pathlib import Path

from cogwheel.waveform_models import nrsurrogate  # noqa: F401
from dot_pe import inference


INFERENCE_KWARGS = dict(
    n_ext=1024,
    n_phi=50,
    n_t=128,
    blocksize=512,
    single_detector_blocksize=512,
    load_inds=True,
    max_incoherent_lnlike_drop=20,
    seed=42,
    draw_subset=True,
)


def get_phase_timers():
    """Create phase timing wrappers."""
    timings = {}

    def timed(phase_name, original_fn):
        @wraps(original_fn)
        def wrapper(*args, **kwargs):
            print(f"  >>> {phase_name} starting at {time.strftime('%H:%M:%S')}")
            t0 = time.time()
            result = original_fn(*args, **kwargs)
            dt = time.time() - t0
            timings[phase_name] = dt
            print(f"  <<< {phase_name} done in {dt:.2f}s ({dt/60:.2f} min)")
            return result
        return wrapper

    return timings, timed


def main():
    parser = argparse.ArgumentParser(description="Run inference for one 1M combination")
    parser.add_argument("--run-dir", required=True, help="Path to incoherent run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_name = run_dir.name

    # Load metadata
    meta = json.load(open(run_dir / "setup" / "run_meta.json"))
    bank_dir = meta["bank_dir"]
    mchirp_guess = meta.get("mchirp_guess", None)

    # Load event data
    with open(meta["event_pkl"], "rb") as f:
        event_data = pickle.load(f)

    # Indices file
    inds_path = run_dir / "inds_for_dotpe.npz"
    if not inds_path.exists():
        raise FileNotFoundError(f"inds_for_dotpe.npz not found in {run_dir}")

    # Output directory
    inference_dir = run_dir / "inference"
    inference_dir.mkdir(exist_ok=True)

    # Wrap phases with timing
    timings, timed = get_phase_timers()
    originals = {
        "prepare": inference.prepare_run_objects,
        "incoherent": inference.select_intrinsic_samples_per_bank_incoherently,
        "cross_bank": inference.select_intrinsic_samples_across_banks_by_incoherent_likelihood,
        "extrinsic": inference.draw_extrinsic_samples,
        "coherent": inference.run_coherent_inference_per_bank,
        "aggregate": inference.aggregate_and_save_results,
    }
    phase_map = {
        "prepare": "prepare_run_objects",
        "incoherent": "select_intrinsic_samples_per_bank_incoherently",
        "cross_bank": "select_intrinsic_samples_across_banks_by_incoherent_likelihood",
        "extrinsic": "draw_extrinsic_samples",
        "coherent": "run_coherent_inference_per_bank",
        "aggregate": "aggregate_and_save_results",
    }

    inference.prepare_run_objects = timed("1_preparation", originals["prepare"])
    inference.select_intrinsic_samples_per_bank_incoherently = timed(
        "2_incoherent_selection", originals["incoherent"])
    inference.select_intrinsic_samples_across_banks_by_incoherent_likelihood = timed(
        "3_cross_bank_threshold", originals["cross_bank"])
    inference.draw_extrinsic_samples = timed("4_extrinsic_sampling", originals["extrinsic"])
    inference.run_coherent_inference_per_bank = timed("5_coherent_inference", originals["coherent"])
    inference.aggregate_and_save_results = timed("6_aggregation", originals["aggregate"])

    print(f"{'=' * 70}")
    print(f"INFERENCE: {run_name}")
    print(f"Bank: {bank_dir}")
    print(f"Approximant: {meta['approximant']}")
    print(f"Event: {meta['event_pkl']}")
    print(f"{'=' * 70}\n", flush=True)

    t0 = time.time()
    inference.run(
        event=event_data,
        bank_folder=bank_dir,
        inds_path=str(inds_path),
        rundir=inference_dir,
        mchirp_guess=mchirp_guess,
        **INFERENCE_KWARGS,
    )
    t_total = time.time() - t0
    timings["total"] = t_total

    # Restore originals
    for key, fn in originals.items():
        setattr(inference, phase_map[key], fn)

    # Save timing data
    timing_file = inference_dir / "phase_timings.json"
    with open(timing_file, "w") as f:
        json.dump({"run_name": run_name, "phase_timings": timings}, f, indent=2)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"INFERENCE COMPLETE: {run_name}")
    print(f"Total: {t_total:.1f}s ({t_total / 60:.1f} min)")
    for phase, dt in sorted(timings.items()):
        if phase != "total":
            pct = 100 * dt / t_total
            print(f"  {phase:<30s} {dt:>8.1f}s ({pct:>5.1f}%)")
    print(f"{'=' * 70}")

    # Check for summary_results.json
    results_dir = inference_dir / "results" if (inference_dir / "results").exists() else inference_dir
    summary_path = results_dir / "summary_results.json"
    if summary_path.exists():
        summary = json.load(open(summary_path))
        print(f"\nResults:")
        print(f"  ln_evidence:        {summary['ln_evidence']:.2f}")
        print(f"  n_effective:        {summary['n_effective']:.2f}")
        print(f"  bestfit_lnlike_max: {summary['bestfit_lnlike_max']:.2f}")
        if "injection" in summary:
            print(f"  injection.bestfit:  {summary['injection']['bestfit_lnlike']:.2f}")
    else:
        print(f"\nWARNING: summary_results.json not found at {summary_path}")


if __name__ == "__main__":
    main()
