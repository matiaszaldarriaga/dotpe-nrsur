"""
Scaling study: profile dot-PE inference at multiple survivor counts.

Usage:
    python scripts/scaling_study.py <run_name> <n1> <n2> <n3> ...

Example:
    python scripts/scaling_study.py aligned_nrsur_into_nrsur 4 50 200 1000 5000

Outputs per-count results to <base>/scaling_study/n_<N>/ and a combined
scaling_summary.json with all timing data.
"""
import sys
import os
import time
import json
import pickle
import cProfile
import pstats
import io
import resource
import traceback
import numpy as np
from pathlib import Path
from functools import wraps

# Register NRSur7dq4
from cogwheel.waveform_models import nrsurrogate  # noqa: F401
from dot_pe import inference

# ── Helpers ─────────────────────────────────────────────────────────

def get_phase_timers():
    """Create fresh phase timing wrappers. Must be called before each run
    because we need to re-wrap the original functions."""
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


def get_peak_memory_mb():
    """Peak RSS in MB (this process only, from OS)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024  # Linux reports in KB


def run_single(run_name, n_survivors, base, meta, surv, event_data):
    """Run inference for a single survivor count. Returns timing dict or error."""
    outdir = base / "scaling_study" / f"n_{n_survivors}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Clean previous results if any
    results_dir = outdir / "results"
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)

    # Slice survivors
    inds = surv["survivor_inds"][:n_survivors]
    inds_file = outdir / "inds.npz"
    np.savez(inds_file, inds=inds,
             incoherent_lnlikes=surv["survivor_incoherent"][:n_survivors],
             lnlikes_di=surv["survivor_lnlike_di"][:, :n_survivors])

    # Fresh phase timers
    timings, timed = get_phase_timers()

    # Save original functions (they may already be wrapped from a previous call)
    # We store originals on first call
    if not hasattr(run_single, '_originals'):
        run_single._originals = {
            'prepare': inference.prepare_run_objects,
            'incoherent': inference.select_intrinsic_samples_per_bank_incoherently,
            'cross_bank': inference.select_intrinsic_samples_across_banks_by_incoherent_likelihood,
            'extrinsic': inference.draw_extrinsic_samples,
            'coherent': inference.run_coherent_inference_per_bank,
            'aggregate': inference.aggregate_and_save_results,
        }

    orig = run_single._originals
    inference.prepare_run_objects = timed("1_preparation", orig['prepare'])
    inference.select_intrinsic_samples_per_bank_incoherently = timed(
        "2_incoherent_selection", orig['incoherent'])
    inference.select_intrinsic_samples_across_banks_by_incoherent_likelihood = timed(
        "3_cross_bank_threshold", orig['cross_bank'])
    inference.draw_extrinsic_samples = timed("4_extrinsic_sampling", orig['extrinsic'])
    inference.run_coherent_inference_per_bank = timed("5_coherent_inference", orig['coherent'])
    inference.aggregate_and_save_results = timed("6_aggregation", orig['aggregate'])

    run_kwargs = dict(
        event=event_data,
        bank_folder=meta["bank_dir"],
        n_ext=1024,
        n_phi=50,
        n_t=128,
        blocksize=512,
        load_inds=True,
        inds_path=str(inds_file),
        max_incoherent_lnlike_drop=20,
        seed=42,
        rundir=results_dir,
        mchirp_guess=meta.get("mchirp_guess", None),
    )

    mem_before = get_peak_memory_mb()
    t0 = time.time()

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        inference.run(**run_kwargs)
    finally:
        profiler.disable()

    t_total = time.time() - t0
    mem_after = get_peak_memory_mb()

    # Save cProfile
    profiler.dump_stats(str(outdir / "profile.prof"))

    # Save text reports
    for sort_key, fname in [(pstats.SortKey.CUMULATIVE, "profile_cumulative.txt"),
                            (pstats.SortKey.TIME, "profile_tottime.txt")]:
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs().sort_stats(sort_key).print_stats(60)
        with open(outdir / fname, "w") as f:
            f.write(s.getvalue())

    # Build result
    timings["total"] = t_total
    result = {
        "n_survivors": n_survivors,
        "phase_timings": timings,
        "peak_rss_mb": mem_after,
        "peak_rss_delta_mb": mem_after - mem_before,
    }

    with open(outdir / "phase_timings.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scaling_study.py <run_name> <n1> [n2] [n3] ...")
        sys.exit(1)

    run_name = sys.argv[1]
    counts = [int(x) for x in sys.argv[2:]]

    base = Path("/data/matiasz/dot-pe/incoherent_1m") / run_name
    meta = json.load(open(base / "setup" / "run_meta.json"))

    # Load survivors once
    surv = np.load(base / "survivors.npz")
    max_available = len(surv["survivor_inds"])

    # Validate counts
    for n in counts:
        if n > max_available:
            print(f"ERROR: requested {n} survivors but only {max_available} available")
            sys.exit(1)

    # Load event once
    with open(meta["event_pkl"], "rb") as f:
        event_data = pickle.load(f)

    print(f"{'=' * 70}")
    print(f"SCALING STUDY: {run_name}")
    print(f"Approximant: {meta['approximant']}")
    print(f"Bank: {meta['bank_dir']}")
    print(f"Survivor counts to test: {counts}")
    print(f"Available survivors: {max_available}")
    print(f"{'=' * 70}\n")

    all_results = []

    for i, n in enumerate(counts):
        print(f"\n{'=' * 70}")
        print(f"[{i+1}/{len(counts)}] Running with {n} survivors...")
        print(f"{'=' * 70}")

        try:
            result = run_single(run_name, n, base, meta, surv, event_data)
            all_results.append(result)

            # Print summary for this count
            t = result["phase_timings"]
            total = t["total"]
            print(f"\n  RESULT: {n} survivors → {total:.1f}s ({total/60:.1f} min)")
            print(f"  Peak RSS: {result['peak_rss_mb']:.0f} MB")
            for phase, dt in sorted(t.items()):
                if phase != "total":
                    print(f"    {phase:<30s} {dt:>8.1f}s ({100*dt/total:>5.1f}%)")

        except Exception as e:
            print(f"\n  ERROR at {n} survivors: {e}")
            traceback.print_exc()
            all_results.append({
                "n_survivors": n,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    # Save combined results
    summary_path = base / "scaling_study" / "scaling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print final summary table
    print(f"\n\n{'=' * 70}")
    print(f"SCALING STUDY COMPLETE — {run_name}")
    print(f"{'=' * 70}\n")

    header = f"{'N surv':>8}  {'Total':>8}  {'Prep':>8}  {'Extrin':>8}  {'Coher':>8}  {'Aggreg':>8}  {'RSS MB':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        if "error" in r:
            print(f"{r['n_survivors']:>8}  ERROR: {r['error'][:50]}")
            continue
        t = r["phase_timings"]
        print(f"{r['n_survivors']:>8}  {t['total']:>7.1f}s  {t.get('1_preparation',0):>7.1f}s  "
              f"{t.get('4_extrinsic_sampling',0):>7.1f}s  {t.get('5_coherent_inference',0):>7.1f}s  "
              f"{t.get('6_aggregation',0):>7.1f}s  {r['peak_rss_mb']:>7.0f}")

    print(f"\nResults saved to: {summary_path}")
