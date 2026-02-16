"""
Optimization benchmark: run inference and compare against baseline.

Usage:
    python scripts/optim_benchmark.py [--n N] [--validate-full]

Options:
    --n N              Number of survivors (default: 200)
    --validate-full    Use N=1024 survivors for full-scale validation
"""
import sys
import os
import time
import json
import pickle
import shutil
import subprocess
import argparse
import numpy as np
from pathlib import Path
from functools import wraps

# Register NRSur7dq4
from cogwheel.waveform_models import nrsurrogate  # noqa: F401
from dot_pe import inference

# ── Configuration ─────────────────────────────────────────────────

RUN_NAME = "aligned_nrsur_into_nrsur"
BASE = Path("/data/matiasz/dot-pe/incoherent_1m") / RUN_NAME
OPTIM_DIR = Path(__file__).resolve().parent.parent / "optim"
BASELINE_JSON = OPTIM_DIR / "baseline.json"

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
    dotpe_dir = Path(__file__).resolve().parent.parent / "packages" / "dot-PE"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=dotpe_dir, capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_phase_timers():
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


def check_correctness(summary, baseline):
    """Compare summary results against baseline. Returns (passed, details)."""
    bl = baseline["correctness"]
    checks = []

    # bestfit_lnlike_max: abs diff < 1.0
    diff = abs(summary["bestfit_lnlike_max"] - bl["bestfit_lnlike_max"])
    checks.append({
        "metric": "bestfit_lnlike_max",
        "baseline": bl["bestfit_lnlike_max"],
        "current": summary["bestfit_lnlike_max"],
        "diff": diff,
        "tolerance": 1.0,
        "passed": diff < 1.0,
        "critical": True,
    })

    # lnl_marginalized_max: abs diff < 2.0
    diff = abs(summary["lnl_marginalized_max"] - bl["lnl_marginalized_max"])
    checks.append({
        "metric": "lnl_marginalized_max",
        "baseline": bl["lnl_marginalized_max"],
        "current": summary["lnl_marginalized_max"],
        "diff": diff,
        "tolerance": 2.0,
        "passed": diff < 2.0,
        "critical": False,
    })

    # ln_evidence: abs diff < 5.0
    diff = abs(summary["ln_evidence"] - bl["ln_evidence"])
    checks.append({
        "metric": "ln_evidence",
        "baseline": bl["ln_evidence"],
        "current": summary["ln_evidence"],
        "diff": diff,
        "tolerance": 5.0,
        "passed": diff < 5.0,
        "critical": False,
    })

    # n_effective: relative diff < 100%
    rel = abs(summary["n_effective"] - bl["n_effective"]) / bl["n_effective"]
    checks.append({
        "metric": "n_effective",
        "baseline": bl["n_effective"],
        "current": summary["n_effective"],
        "diff": rel,
        "tolerance": 1.0,
        "passed": rel < 1.0,
        "critical": False,
    })

    # injection bestfit_lnlike: abs diff < 0.5
    diff = abs(summary["injection"]["bestfit_lnlike"] - bl["injection_bestfit_lnlike"])
    checks.append({
        "metric": "injection.bestfit_lnlike",
        "baseline": bl["injection_bestfit_lnlike"],
        "current": summary["injection"]["bestfit_lnlike"],
        "diff": diff,
        "tolerance": 0.5,
        "passed": diff < 0.5,
        "critical": True,
    })

    # Overall: FAIL only if critical checks fail
    critical_pass = all(c["passed"] for c in checks if c["critical"])
    all_pass = all(c["passed"] for c in checks)

    return critical_pass, all_pass, checks


def print_comparison(timings, baseline, checks, critical_pass, all_pass):
    """Print formatted comparison table."""
    bl_timings = baseline["phase_timings"]

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK RESULTS")
    print(f"{'=' * 70}")

    # Timing table
    print(f"\n{'Phase':<25s} {'Baseline':>10s} {'Current':>10s} {'Speedup':>10s}")
    print("-" * 55)

    phase_order = [
        "1_preparation", "2_incoherent_selection", "3_cross_bank_threshold",
        "4_extrinsic_sampling", "5_coherent_inference", "6_aggregation",
    ]

    for phase in phase_order:
        bl_t = bl_timings.get(phase, 0)
        cur_t = timings.get(phase, 0)
        if bl_t > 0.1:  # Only show phases with meaningful time
            speedup = bl_t / cur_t if cur_t > 0 else float("inf")
            print(f"  {phase:<23s} {bl_t:>8.1f}s  {cur_t:>8.1f}s  {speedup:>8.2f}x")

    bl_total = bl_timings["total"]
    cur_total = timings["total"]
    speedup_total = bl_total / cur_total if cur_total > 0 else float("inf")
    print("-" * 55)
    print(f"  {'TOTAL':<23s} {bl_total:>8.1f}s  {cur_total:>8.1f}s  {speedup_total:>8.2f}x")

    # Correctness table
    print(f"\n{'Correctness Check':<30s} {'Baseline':>12s} {'Current':>12s} {'Diff':>10s} {'Tol':>8s} {'Result':>8s}")
    print("-" * 80)
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        crit = " *" if c["critical"] else ""
        if "relative" in c["metric"] or c["metric"] == "n_effective":
            diff_str = f"{c['diff']:.1%}"
            tol_str = f"{c['tolerance']:.0%}"
        else:
            diff_str = f"{c['diff']:.3f}"
            tol_str = f"{c['tolerance']:.1f}"
        print(f"  {c['metric']:<28s} {c['baseline']:>12.3f} {c['current']:>12.3f} {diff_str:>10s} {tol_str:>8s} {status:>6s}{crit}")

    print()
    if critical_pass and all_pass:
        print("  VERDICT: PASS (all checks)")
    elif critical_pass:
        print("  VERDICT: PASS (critical checks; some non-critical failures)")
    else:
        print("  VERDICT: *** FAIL *** (critical check failed)")

    return speedup_total


def main():
    parser = argparse.ArgumentParser(description="Optimization benchmark")
    parser.add_argument("--n", type=int, default=200, help="Number of survivors")
    parser.add_argument("--validate-full", action="store_true",
                        help="Use N=1024 for full validation")
    args = parser.parse_args()

    n_survivors = 1024 if args.validate_full else args.n

    # Load baseline
    if not BASELINE_JSON.exists():
        print(f"ERROR: No baseline found at {BASELINE_JSON}")
        print("Run: python scripts/run_baseline.py")
        sys.exit(1)

    baseline = json.load(open(BASELINE_JSON))

    # Load run metadata
    meta = json.load(open(BASE / "setup" / "run_meta.json"))

    # Load event
    with open(meta["event_pkl"], "rb") as f:
        event_data = pickle.load(f)

    # Load and slice survivors
    surv = np.load(BASE / "survivors.npz")
    max_available = len(surv["survivor_inds"])
    if n_survivors > max_available:
        print(f"WARNING: Requested {n_survivors} but only {max_available} available. Using {max_available}.")
        n_survivors = max_available

    inds = surv["survivor_inds"][:n_survivors]

    # Set up output directory
    outdir = OPTIM_DIR / "benchmark_run"
    outdir.mkdir(parents=True, exist_ok=True)

    results_dir = outdir / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)

    inds_file = outdir / "inds.npz"
    np.savez(
        inds_file,
        inds=inds,
        incoherent_lnlikes=surv["survivor_incoherent"][:n_survivors],
        lnlikes_di=surv["survivor_lnlike_di"][:, :n_survivors],
    )

    # Wrap inference phases
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
    print(f"BENCHMARK: {RUN_NAME}, N={n_survivors}, seed=42")
    print(f"dot-PE commit: {get_dotpe_commit()}")
    validate_tag = " [FULL VALIDATION]" if args.validate_full else ""
    print(f"Mode: benchmark{validate_tag}")
    print(f"{'=' * 70}\n")

    t0 = time.time()
    inference.run(**run_kwargs)
    t_total = time.time() - t0
    timings["total"] = t_total

    # Restore originals
    phase_map = {
        "prepare": "prepare_run_objects",
        "incoherent": "select_intrinsic_samples_per_bank_incoherently",
        "cross_bank": "select_intrinsic_samples_across_banks_by_incoherent_likelihood",
        "extrinsic": "draw_extrinsic_samples",
        "coherent": "run_coherent_inference_per_bank",
        "aggregate": "aggregate_and_save_results",
    }
    for key, fn in originals.items():
        setattr(inference, phase_map[key], fn)

    # Load results
    summary_path = results_dir / "summary_results.json"
    summary = json.load(open(summary_path))

    # Correctness check (only meaningful at same N as baseline)
    if n_survivors == baseline["n_survivors"]:
        critical_pass, all_pass, checks = check_correctness(summary, baseline)
        speedup = print_comparison(timings, baseline, checks, critical_pass, all_pass)
    else:
        # Different N — just show timings, no correctness comparison
        print(f"\n{'=' * 70}")
        print(f"BENCHMARK RESULTS (N={n_survivors}, no correctness comparison)")
        print(f"{'=' * 70}")
        print(f"Total: {t_total:.1f}s ({t_total/60:.1f} min)")
        for phase, dt in sorted(timings.items()):
            if phase != "total":
                print(f"  {phase:<30s} {dt:>8.1f}s ({100*dt/t_total:>5.1f}%)")
        critical_pass = True
        all_pass = True
        checks = []
        speedup = None

    # Save benchmark result
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_survivors": n_survivors,
        "dotpe_commit": get_dotpe_commit(),
        "phase_timings": timings,
        "correctness_pass": critical_pass,
        "correctness_all_pass": all_pass,
        "speedup_total": speedup,
        "summary": {
            "bestfit_lnlike_max": summary["bestfit_lnlike_max"],
            "lnl_marginalized_max": summary["lnl_marginalized_max"],
            "ln_evidence": summary["ln_evidence"],
            "n_effective": summary["n_effective"],
        },
    }

    latest_path = OPTIM_DIR / "benchmark_latest.json"
    with open(latest_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nBenchmark saved to: {latest_path}")

    if not critical_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
