"""
Profile dot-PE inference: per-phase wall-clock + cProfile function-level detail.

Usage:
    python scripts/profile_inference.py [run_name] [--n-survivors N]

Default: precessing_xphm_into_nrsur (4 survivors from 1M campaign)
"""
import sys
import os
import time
import json
import pickle
import cProfile
import pstats
import io
import numpy as np
from pathlib import Path
from functools import wraps

# Register NRSur7dq4
from cogwheel.waveform_models import nrsurrogate  # noqa: F401

# ── Monkey-patch phase functions with wall-clock timing ──────────────

from dot_pe import inference

_phase_timings = {}

def _timed(phase_name, original_fn):
    """Wrap a function to record its wall-clock time."""
    @wraps(original_fn)
    def wrapper(*args, **kwargs):
        print(f"\n>>> PHASE: {phase_name} starting at {time.strftime('%H:%M:%S')}")
        t0 = time.time()
        result = original_fn(*args, **kwargs)
        dt = time.time() - t0
        _phase_timings[phase_name] = dt
        print(f"<<< PHASE: {phase_name} done in {dt:.2f}s ({dt/60:.2f} min)")
        return result
    return wrapper

# Patch each phase function
inference.prepare_run_objects = _timed("1_preparation", inference.prepare_run_objects)
inference.select_intrinsic_samples_per_bank_incoherently = _timed(
    "2_incoherent_selection", inference.select_intrinsic_samples_per_bank_incoherently)
inference.select_intrinsic_samples_across_banks_by_incoherent_likelihood = _timed(
    "3_cross_bank_threshold", inference.select_intrinsic_samples_across_banks_by_incoherent_likelihood)
inference.draw_extrinsic_samples = _timed("4_extrinsic_sampling", inference.draw_extrinsic_samples)
inference.run_coherent_inference_per_bank = _timed(
    "5_coherent_inference", inference.run_coherent_inference_per_bank)
inference.aggregate_and_save_results = _timed(
    "6_aggregation", inference.aggregate_and_save_results)

# ── Setup ────────────────────────────────────────────────────────────

run_name = sys.argv[1] if len(sys.argv) > 1 else "precessing_xphm_into_nrsur"
max_survivors = int(sys.argv[2]) if len(sys.argv) > 2 else None

base = Path("/data/matiasz/dot-pe/incoherent_1m") / run_name

meta = json.load(open(base / "setup" / "run_meta.json"))
event_pkl = meta["event_pkl"]
bank_dir = meta["bank_dir"]

# Load survivors
surv = np.load(base / "survivors.npz")
inds = surv["survivor_inds"]

if max_survivors is not None:
    inds = inds[:max_survivors]

print(f"=" * 70)
print(f"PROFILING: {run_name}")
print(f"Survivors: {len(inds)} (of {len(surv['survivor_inds'])} total)")
print(f"Bank: {bank_dir}")
print(f"Event: {event_pkl}")
print(f"=" * 70)

# Save survivors in dot-PE format
outdir = base / "profile_run"
outdir.mkdir(exist_ok=True)
inds_file = outdir / "inds.npz"
np.savez(inds_file, inds=inds,
         incoherent_lnlikes=surv["survivor_incoherent"][:len(inds)],
         lnlikes_di=surv["survivor_lnlike_di"][:, :len(inds)])

# Load event
with open(event_pkl, "rb") as f:
    event_data = pickle.load(f)

# ── Run with cProfile ────────────────────────────────────────────────

run_kwargs = dict(
    event=event_data,
    bank_folder=bank_dir,
    n_ext=1024,
    n_phi=50,
    n_t=128,
    blocksize=512,
    load_inds=True,
    inds_path=str(inds_file),
    max_incoherent_lnlike_drop=20,
    seed=42,
    rundir=outdir / "results",
    mchirp_guess=meta.get("mchirp_guess", None),
)

print(f"\nStarting profiled inference at {time.strftime('%H:%M:%S')}")
t_total_start = time.time()

profiler = cProfile.Profile()
profiler.enable()

rundir = inference.run(**run_kwargs)

profiler.disable()

t_total = time.time() - t_total_start

# ── Save and report results ──────────────────────────────────────────

# Save cProfile binary
prof_path = outdir / "profile.prof"
profiler.dump_stats(str(prof_path))

# Generate cProfile text report (top 60 by cumulative time)
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s)
ps.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(60)
profile_text_cumulative = s.getvalue()

# Also sort by tottime
s2 = io.StringIO()
ps2 = pstats.Stats(profiler, stream=s2)
ps2.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats(60)
profile_text_tottime = s2.getvalue()

# Save text reports
with open(outdir / "profile_cumulative.txt", "w") as f:
    f.write(profile_text_cumulative)
with open(outdir / "profile_tottime.txt", "w") as f:
    f.write(profile_text_tottime)

# Save phase timings as JSON
_phase_timings["total"] = t_total
with open(outdir / "phase_timings.json", "w") as f:
    json.dump(_phase_timings, f, indent=2)

# ── Print summary ────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print(f"PROFILING COMPLETE — {run_name} ({len(inds)} survivors)")
print(f"{'=' * 70}")
print(f"\nTotal wall time: {t_total:.1f}s ({t_total/60:.1f} min)\n")

print("Phase breakdown:")
print(f"  {'Phase':<30} {'Time (s)':>10} {'% of total':>12}")
print(f"  {'-'*30} {'-'*10} {'-'*12}")
for phase, dt in sorted(_phase_timings.items()):
    if phase != "total":
        pct = 100 * dt / t_total
        print(f"  {phase:<30} {dt:>10.1f} {pct:>11.1f}%")

print(f"\n\nTop 30 functions by cumulative time:")
print(f"{'=' * 70}")

# Print just the top 30 from the cumulative report
s3 = io.StringIO()
ps3 = pstats.Stats(profiler, stream=s3)
ps3.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(30)
print(s3.getvalue())

print(f"\n\nTop 30 functions by self time (tottime):")
print(f"{'=' * 70}")

s4 = io.StringIO()
ps4 = pstats.Stats(profiler, stream=s4)
ps4.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats(30)
print(s4.getvalue())

print(f"\nOutput saved to: {outdir}")
print(f"  phase_timings.json   — per-phase wall clock")
print(f"  profile.prof         — cProfile binary (load with pstats/snakeviz)")
print(f"  profile_cumulative.txt — top functions by cumulative time")
print(f"  profile_tottime.txt  — top functions by self time")
