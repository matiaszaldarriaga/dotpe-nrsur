"""
Generate baseline timing and correctness data for the optimization loop.

Runs inference on the aligned_nrsur_into_nrsur test case (N=200 survivors, seed=42)
and saves results to optim/baseline.json + optim/baseline_results/.

Usage:
    python scripts/run_baseline.py [--force]

The --force flag re-runs even if baseline.json already exists.
"""
import sys
import os
import time
import json
import pickle
import shutil
import subprocess
import numpy as np
from pathlib import Path
from functools import wraps

# Register NRSur7dq4
from cogwheel.waveform_models import nrsurrogate  # noqa: F401
from dot_pe import inference

# ── Configuration ─────────────────────────────────────────────────

RUN_NAME = "aligned_nrsur_into_nrsur"
N_SURVIVORS = 200
BASE = Path("/data/matiasz/dot-pe/incoherent_1m") / RUN_NAME
OPTIM_DIR = Path(__file__).resolve().parent.parent / "optim"
BASELINE_JSON = OPTIM_DIR / "baseline.json"
BASELINE_RESULTS = OPTIM_DIR / "baseline_results"

INFERENCE_KWARGS = dict(
    n_ext=1024,
    n_phi=50,
    n_t=128,
    blocksize=512,
    load_inds=True,
    max_incoherent_lnlike_drop=20,
    seed=42,
)


def get_dotpe_commit():
    """Get current dot-PE commit hash."""
    dotpe_dir = Path(__file__).resolve().parent.parent / "packages" / "dot-PE"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=dotpe_dir, capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_phase_timers():
    """Create fresh phase timing wrappers."""
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
    force = "--force" in sys.argv

    if BASELINE_JSON.exists() and not force:
        print(f"Baseline already exists at {BASELINE_JSON}")
        print("Use --force to regenerate.")
        return

    # Load run metadata
    meta = json.load(open(BASE / "setup" / "run_meta.json"))

    # Load event
    with open(meta["event_pkl"], "rb") as f:
        event_data = pickle.load(f)

    # Load and slice survivors
    surv = np.load(BASE / "survivors.npz")
    inds = surv["survivor_inds"][:N_SURVIVORS]

    # Set up output directory
    outdir = OPTIM_DIR / "baseline_run"
    outdir.mkdir(parents=True, exist_ok=True)

    # Clean previous results
    results_dir = outdir / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)

    # Save survivor indices
    inds_file = outdir / "inds.npz"
    np.savez(
        inds_file,
        inds=inds,
        incoherent_lnlikes=surv["survivor_incoherent"][:N_SURVIVORS],
        lnlikes_di=surv["survivor_lnlike_di"][:, :N_SURVIVORS],
    )

    # Wrap inference phases with timing
    timings, timed = get_phase_timers()

    originals = {
        "prepare": inference.prepare_run_objects,
        "incoherent": inference.select_intrinsic_samples_per_bank_incoherently,
        "cross_bank": inference.select_intrinsic_samples_across_banks_by_incoherent_likelihood,
        "extrinsic": inference.draw_extrinsic_samples,
        "coherent": inference.run_coherent_inference_per_bank,
        "aggregate": inference.aggregate_and_save_results,
    }

    inference.prepare_run_objects = timed("1_preparation", originals["prepare"])
    inference.select_intrinsic_samples_per_bank_incoherently = timed(
        "2_incoherent_selection", originals["incoherent"]
    )
    inference.select_intrinsic_samples_across_banks_by_incoherent_likelihood = timed(
        "3_cross_bank_threshold", originals["cross_bank"]
    )
    inference.draw_extrinsic_samples = timed("4_extrinsic_sampling", originals["extrinsic"])
    inference.run_coherent_inference_per_bank = timed("5_coherent_inference", originals["coherent"])
    inference.aggregate_and_save_results = timed("6_aggregation", originals["aggregate"])

    run_kwargs = dict(
        event=event_data,
        bank_folder=meta["bank_dir"],
        inds_path=str(inds_file),
        rundir=results_dir,
        mchirp_guess=meta.get("mchirp_guess", None),
        **INFERENCE_KWARGS,
    )

    print(f"{'=' * 70}")
    print(f"BASELINE RUN: {RUN_NAME}, N={N_SURVIVORS}, seed=42")
    print(f"dot-PE commit: {get_dotpe_commit()}")
    print(f"{'=' * 70}\n")

    t0 = time.time()
    inference.run(**run_kwargs)
    t_total = time.time() - t0
    timings["total"] = t_total

    # Restore originals
    for key, fn in originals.items():
        phase_map = {
            "prepare": "prepare_run_objects",
            "incoherent": "select_intrinsic_samples_per_bank_incoherently",
            "cross_bank": "select_intrinsic_samples_across_banks_by_incoherent_likelihood",
            "extrinsic": "draw_extrinsic_samples",
            "coherent": "run_coherent_inference_per_bank",
            "aggregate": "aggregate_and_save_results",
        }
        setattr(inference, phase_map[key], fn)

    # Load summary results
    summary_path = results_dir / "summary_results.json"
    summary = json.load(open(summary_path))

    # Build baseline record
    baseline = {
        "run_name": RUN_NAME,
        "n_survivors": N_SURVIVORS,
        "seed": 42,
        "dotpe_commit": get_dotpe_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase_timings": timings,
        "correctness": {
            "bestfit_lnlike_max": summary["bestfit_lnlike_max"],
            "lnl_marginalized_max": summary["lnl_marginalized_max"],
            "ln_evidence": summary["ln_evidence"],
            "n_effective": summary["n_effective"],
            "injection_bestfit_lnlike": summary["injection"]["bestfit_lnlike"],
            "injection_lnl_marginalized": summary["injection"]["lnl_marginalized"],
            "n_distance_marginalizations": summary["n_distance_marginalizations"],
        },
        "inference_kwargs": {k: str(v) if isinstance(v, Path) else v
                            for k, v in INFERENCE_KWARGS.items()},
        "meta": {
            "bank_dir": meta["bank_dir"],
            "event_pkl": meta["event_pkl"],
            "approximant": meta["approximant"],
        },
    }

    # Save baseline JSON
    OPTIM_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_JSON, "w") as f:
        json.dump(baseline, f, indent=2)

    # Copy key result files to baseline_results/
    if BASELINE_RESULTS.exists():
        shutil.rmtree(BASELINE_RESULTS)
    shutil.copytree(results_dir, BASELINE_RESULTS)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"BASELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total: {t_total:.1f}s ({t_total/60:.1f} min)")
    for phase, dt in sorted(timings.items()):
        if phase != "total":
            print(f"  {phase:<30s} {dt:>8.1f}s ({100*dt/t_total:>5.1f}%)")
    print(f"\nCorrectness metrics:")
    for k, v in baseline["correctness"].items():
        print(f"  {k}: {v}")
    print(f"\nSaved to: {BASELINE_JSON}")
    print(f"Results copied to: {BASELINE_RESULTS}")


if __name__ == "__main__":
    main()
