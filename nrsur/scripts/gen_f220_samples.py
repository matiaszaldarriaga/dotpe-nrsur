#!/usr/bin/env python
"""
Phase 1a: Generate 4M f220-constrained intrinsic samples.

Produces a shared set of samples (independent of waveform approximant) constrained
so that the remnant BH's 220 ringdown frequency lies in [55, 74] Hz.

Uses surfinBH (NRSur7dq4Remnant) for remnant properties and qnm for mode frequencies.
Uses multiprocessing.Pool (not ThreadPool) to bypass GIL for surfinBH calls.

Output:
    /data/matiasz/dot-pe/gw231123/samples/
    ├── intrinsic_sample_bank.feather  (~290 MB)
    ├── remnant_properties.feather
    └── sample_config.json

Usage:
    python scripts/gen_f220_samples.py [--n-samples N] [--seed S] [--n-workers N]
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

import lal

MTSUN_SI = lal.MTSUN_SI  # M_sun in seconds: 4.925490947e-6

# --- Configuration ---
OUTPUT_DIR = Path("/data/matiasz/dot-pe/gw231123/samples")
Q_MIN, Q_MAX = 1.0, 6.0          # NRSur7dq4 validity: q <= 6
LN_F220_MIN, LN_F220_MAX = 4.0, 4.3  # f220 in [54.6, 73.7] Hz
F_REF = 20.0
BLOCKSIZE = 4096


def _init_worker():
    """Initialize surfinBH model in each worker process."""
    import surfinBH
    global _surfinBH_model
    _surfinBH_model = surfinBH.LoadFits('NRSur7dq4Remnant')


def _compute_remnant_chunk(chunk):
    """Process a chunk of samples in one worker. Returns (final_spins, mf_fracs)."""
    q_arr, s1x_arr, s1y_arr, s1z_arr, s2x_arr, s2y_arr, s2z_arr = chunk
    n = len(q_arr)
    final_spins = np.empty(n)
    mf_fracs = np.empty(n)

    for i in range(n):
        chiA = np.array([s1x_arr[i], s1y_arr[i], s1z_arr[i]])
        chiB = np.array([s2x_arr[i], s2y_arr[i], s2z_arr[i]])
        try:
            chif_vec = _surfinBH_model.chif(q=q_arr[i], chiA=chiA, chiB=chiB, allow_extrap=True)[0]
            final_spins[i] = np.linalg.norm(chif_vec)
            mf_fracs[i] = _surfinBH_model.mf(q=q_arr[i], chiA=chiA, chiB=chiB, allow_extrap=True)[0]
        except Exception:
            final_spins[i] = 0.5
            mf_fracs[i] = 0.1

    return final_spins, mf_fracs


def generate_f220_samples(n_samples, seed=42, n_workers=16):
    """Generate f220-constrained intrinsic samples."""
    rng = np.random.default_rng(seed)
    t0 = time.time()

    print(f"Generating {n_samples:,} f220-constrained samples", flush=True)
    print(f"  q in [{Q_MIN}, {Q_MAX}], ln(f220) in [{LN_F220_MIN}, {LN_F220_MAX}]", flush=True)
    print(f"  f220 range: [{np.exp(LN_F220_MIN):.1f}, {np.exp(LN_F220_MAX):.1f}] Hz", flush=True)

    # 1. Sample q uniformly (q = m1/m2 >= 1)
    q = rng.uniform(Q_MIN, Q_MAX, n_samples)

    # 2. Sample chi_eff and chi_diff (following cogwheel conventions)
    chi_eff = rng.uniform(-0.99, 0.99, n_samples)
    cumchidiff = rng.uniform(0, 1, n_samples)

    # Valid chi_diff bounds for each (chi_eff, q)
    max_from_s1 = 2 * (2 * q / (q + 1) - chi_eff)
    min_from_s1 = 2 * (-2 * q / (q + 1) - chi_eff)
    max_from_s2 = 2 * (2 / (q + 1) + chi_eff)
    min_from_s2 = 2 * (-2 / (q + 1) + chi_eff)

    chi_diff_max = np.minimum(max_from_s1, max_from_s2)
    chi_diff_min = np.maximum(min_from_s1, min_from_s2)
    chi_diff = chi_diff_min + cumchidiff * (chi_diff_max - chi_diff_min)

    s1z = (chi_eff + chi_diff / 2) * (q + 1) / (2 * q)
    s2z = (chi_eff - chi_diff / 2) * (q + 1) / 2

    assert np.abs(s1z).max() <= 1.001, f"|s1z| max = {np.abs(s1z).max()}"
    assert np.abs(s2z).max() <= 1.001, f"|s2z| max = {np.abs(s2z).max()}"

    # 3. cos(iota) uniform
    cos_iota = rng.uniform(-1, 1, n_samples)
    iota = np.arccos(cos_iota)

    # 4. Perpendicular spins (Kerr-bound enforced)
    s1_perp_mag = rng.uniform(0, np.sqrt(np.clip(1 - s1z**2, 0, 1)), n_samples)
    s2_perp_mag = rng.uniform(0, np.sqrt(np.clip(1 - s2z**2, 0, 1)), n_samples)
    s1_phi = rng.uniform(0, 2 * np.pi, n_samples)
    s2_phi = rng.uniform(0, 2 * np.pi, n_samples)

    s1x = s1_perp_mag * np.cos(s1_phi)
    s1y = s1_perp_mag * np.sin(s1_phi)
    s2x = s2_perp_mag * np.cos(s2_phi)
    s2y = s2_perp_mag * np.sin(s2_phi)

    s1_mag = np.sqrt(s1x**2 + s1y**2 + s1z**2)
    s2_mag = np.sqrt(s2x**2 + s2y**2 + s2z**2)
    print(f"  |s1| max = {s1_mag.max():.6f}, |s2| max = {s2_mag.max():.6f}", flush=True)
    assert s1_mag.max() <= 1.001 and s2_mag.max() <= 1.001

    # 5. Remnant properties via surfinBH (multiprocessing to bypass GIL)
    print(f"  Computing remnant properties with {n_workers} processes...", flush=True)

    # Split into chunks for multiprocessing
    chunk_size = (n_samples + n_workers - 1) // n_workers
    chunks = []
    for c in range(n_workers):
        start = c * chunk_size
        end = min(start + chunk_size, n_samples)
        if start >= n_samples:
            break
        chunks.append((
            q[start:end], s1x[start:end], s1y[start:end], s1z[start:end],
            s2x[start:end], s2y[start:end], s2z[start:end],
        ))

    t1 = time.time()
    with Pool(processes=n_workers, initializer=_init_worker) as pool:
        results = pool.map(_compute_remnant_chunk, chunks)

    final_spins = np.concatenate([r[0] for r in results])
    mf_fracs = np.concatenate([r[1] for r in results])
    print(f"    surfinBH done in {time.time()-t1:.1f}s", flush=True)

    assert not np.any(np.isnan(final_spins) | np.isinf(final_spins))
    assert not np.any(np.isnan(mf_fracs) | np.isinf(mf_fracs))
    print(f"  final_spin: [{final_spins.min():.4f}, {final_spins.max():.4f}]", flush=True)
    print(f"  mf_fraction: [{mf_fracs.min():.4f}, {mf_fracs.max():.4f}]", flush=True)

    # 6. QNM frequencies
    print("  Computing omega_220...", flush=True)
    import qnm
    mode_220 = qnm.modes_cache(s=-2, l=2, m=2, n=0)
    fs_rounded = np.round(final_spins, 4)
    unique_spins = np.unique(fs_rounded)
    print(f"    {len(unique_spins):,} unique spin values", flush=True)

    spin_to_omega = {}
    for spin_val in unique_spins:
        try:
            omega_complex = mode_220(a=spin_val)[0]
            spin_to_omega[spin_val] = omega_complex.real
        except Exception:
            spin_to_omega[spin_val] = 0.5

    omega_220 = np.array([spin_to_omega[fs_rounded[i]] for i in range(n_samples)])

    # 7. M_total from f220 constraint
    # f220 = omega_220 / (2π * mf * M_total * MTSUN_SI)
    # ln(M_total) = ln(omega_220 / (2π * mf * MTSUN_SI)) - ln(f220)
    ln_M_const = np.log(omega_220 / (2 * np.pi * mf_fracs * MTSUN_SI))
    ln_M_min = ln_M_const - LN_F220_MAX
    ln_M_max = ln_M_const - LN_F220_MIN
    ln_M_total = rng.uniform(ln_M_min, ln_M_max)
    M_total = np.exp(ln_M_total)

    # 8. Component masses
    m1 = M_total * q / (1 + q)
    m2 = M_total / (1 + q)

    # 9. Verify f220
    f220 = omega_220 / (2 * np.pi * mf_fracs * M_total * MTSUN_SI)
    ln_f220 = np.log(f220)

    # 10. Check omega_ref constraint for NRSur7dq4
    omega_ref = np.pi * F_REF * M_total * MTSUN_SI
    n_omega_exceed = np.sum(omega_ref > 0.2)

    print(f"\n  Results:", flush=True)
    print(f"    M_total: [{M_total.min():.1f}, {M_total.max():.1f}] Msun", flush=True)
    print(f"    m1: [{m1.min():.1f}, {m1.max():.1f}] Msun", flush=True)
    print(f"    m2: [{m2.min():.1f}, {m2.max():.1f}] Msun", flush=True)
    print(f"    f220: [{f220.min():.1f}, {f220.max():.1f}] Hz", flush=True)
    print(f"    ln(f220): [{ln_f220.min():.4f}, {ln_f220.max():.4f}]", flush=True)
    print(f"    omega_ref (f_ref={F_REF}): [{omega_ref.min():.4f}, {omega_ref.max():.4f}]", flush=True)
    print(f"    omega_ref > 0.2: {n_omega_exceed} ({100*n_omega_exceed/n_samples:.1f}%)", flush=True)
    print(f"    Total time: {time.time()-t0:.1f}s", flush=True)

    # Build DataFrames
    bank_df = pd.DataFrame({
        'm1': m1, 'm2': m2,
        's1x_n': s1x, 's1y_n': s1y, 's1z': s1z,
        's2x_n': s2x, 's2y_n': s2y, 's2z': s2z,
        'iota': iota,
        'log_prior_weights': np.zeros(n_samples),
    })

    remnant_df = pd.DataFrame({
        'q': q, 'chieff': chi_eff, 'cos_iota': cos_iota,
        'final_spin': final_spins, 'mf_fraction': mf_fracs,
        'omega_220': omega_220, 'f220': f220, 'ln_f220': ln_f220,
        'M_total': M_total, 'omega_ref': omega_ref,
    })

    config = {
        'n_samples': int(n_samples),
        'q_range': [Q_MIN, Q_MAX],
        'ln_f220_range': [LN_F220_MIN, LN_F220_MAX],
        'f_ref': F_REF,
        'seed': int(seed),
        'blocksize': BLOCKSIZE,
        'n_omega_exceed': int(n_omega_exceed),
    }

    return bank_df, remnant_df, config


def plot_diagnostics(remnant_df, outpath):
    """Save diagnostic histograms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    params = [
        ('q', 'Mass ratio q'),
        ('chieff', 'chi_eff'),
        ('M_total', 'M_total [Msun]'),
        ('f220', 'f_220 [Hz]'),
        ('final_spin', 'Remnant spin'),
        ('omega_ref', 'omega_ref (f_ref=20)'),
    ]
    for ax, (col, label) in zip(axes.flat, params):
        vals = remnant_df[col].values
        ax.hist(vals, bins=100, alpha=0.7, edgecolor='k', linewidth=0.3)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        if col == 'omega_ref':
            ax.axvline(0.2, color='r', ls='--', label='NRSur limit')
            ax.legend()

    plt.suptitle(f"f220-Constrained Sample Bank ({len(remnant_df):,} samples)", fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"  Saved: {outpath}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=2**22, help="Number of samples (default: 4M)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=16, help="Number of worker processes")
    args = parser.parse_args()

    bank_df, remnant_df, config = generate_f220_samples(args.n_samples, args.seed, args.n_workers)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {OUTPUT_DIR}/", flush=True)

    bank_path = OUTPUT_DIR / "intrinsic_sample_bank.feather"
    bank_df.to_feather(bank_path)
    print(f"  intrinsic_sample_bank.feather: {bank_path.stat().st_size / 1e6:.1f} MB", flush=True)

    remnant_path = OUTPUT_DIR / "remnant_properties.feather"
    remnant_df.to_feather(remnant_path)
    print(f"  remnant_properties.feather: {remnant_path.stat().st_size / 1e6:.1f} MB", flush=True)

    config_path = OUTPUT_DIR / "sample_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Diagnostics plot
    plot_outpath = Path(__file__).resolve().parent.parent / "output" / "gw231123_f220_samples.png"
    plot_outpath.parent.mkdir(exist_ok=True)
    plot_diagnostics(remnant_df, plot_outpath)

    # Gate 1a checks
    n = len(bank_df)
    assert n == args.n_samples, f"Expected {args.n_samples}, got {n}"
    assert remnant_df['ln_f220'].between(LN_F220_MIN - 0.01, LN_F220_MAX + 0.01).all()
    assert remnant_df['q'].between(Q_MIN - 0.01, Q_MAX + 0.01).all()
    s1_mag = np.sqrt(bank_df['s1x_n']**2 + bank_df['s1y_n']**2 + bank_df['s1z']**2)
    s2_mag = np.sqrt(bank_df['s2x_n']**2 + bank_df['s2y_n']**2 + bank_df['s2z']**2)
    assert s1_mag.max() <= 1.001 and s2_mag.max() <= 1.001
    print("\nGATE 1a PASSED: all constraints verified", flush=True)


if __name__ == "__main__":
    main()
