#!/usr/bin/env python
"""
Phase 1c: Prepare event data for GW231123 analysis.

Creates two pickled EventData objects:
1. Real GW231123 event (H+L, inpainted data)
2. GW231123-like injection in Gaussian noise (H+L, NRSur7dq4)

Usage:
    python scripts/setup_gw231123_event.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import pickle
import json
import numpy as np
from pathlib import Path

from cogwheel import data

OUTPUT_DIR = Path("/data/matiasz/dot-pe/gw231123/event_data")

# Real event data
REAL_EVENT_NPZ = Path("/home/matiasz/GW-2025/dot-PE/GW231123/data/GW231123_135430_inpainted.npz")

# GW231123 NRSur7dq4 max-likelihood parameters from LVK posterior
# for the injection signal
INJ_PAR_DIC = dict(
    m1=156.217,
    m2=142.091,
    d_luminosity=961.7,
    iota=1.095,
    phi_ref=2.346,       # phase from LVK NRSur7dq4 max-L
    f_ref=20.0,
    s1x_n=-0.902,
    s1y_n=-0.398,
    s1z=-0.054,
    s2x_n=0.197,
    s2y_n=0.860,
    s2z=-0.039,
    ra=3.237,
    dec=0.246,
    psi=2.237,
    t_geocenter=0.0,
    l1=0.0,
    l2=0.0,
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Real event ---
    print("Loading real GW231123 event data...")
    real_event = data.EventData.from_npz(filename=str(REAL_EVENT_NPZ))
    print(f"  Event: {real_event.eventname}")
    print(f"  Detectors: {real_event.detector_names}")
    print(f"  tgps: {real_event.tgps}")
    print(f"  Freq range: {real_event.frequencies[0]:.1f} - {real_event.frequencies[-1]:.1f} Hz")
    print(f"  df: {real_event.df:.6f} Hz")

    real_path = OUTPUT_DIR / "real_event.pkl"
    with open(real_path, "wb") as f:
        pickle.dump(real_event, f)
    print(f"  Saved: {real_path} ({real_path.stat().st_size / 1e6:.1f} MB)")

    # --- 2. Injection in Gaussian noise ---
    print("\nCreating GW231123-like injection...")
    # Match real event's frequency grid
    duration = 1.0 / real_event.df  # = 176 s
    fmax = real_event.frequencies[-1]  # 1600 Hz
    tgps = real_event.tgps

    print(f"  duration={duration:.1f}s, fmax={fmax:.0f}Hz, tgps={tgps:.1f}")

    # Need NRSur7dq4 for injection
    from cogwheel.waveform_models import nrsurrogate  # noqa

    inj_event = data.EventData.gaussian_noise(
        eventname="GW231123_injection",
        duration=duration,
        detector_names="HL",
        asd_funcs=["asd_H_O3", "asd_L_O3"],
        tgps=tgps,
        fmax=fmax,
        seed=12345,
    )

    # Two-pass injection: first compute optimal SNR, then scale distance to target SNR~23
    TARGET_SNR = 23.0
    print("  Pass 1: computing optimal SNR at d_L=961.7 Mpc...")
    inj_event.inject_signal(INJ_PAR_DIC, "NRSur7dq4")
    h_h_raw = inj_event.injection["h_h"]
    snr_raw = np.sqrt(np.sum(h_h_raw))
    print(f"  Raw network SNR: {snr_raw:.1f}")

    # Scale distance to get target SNR (SNR ∝ 1/d_L)
    d_L_scaled = INJ_PAR_DIC["d_luminosity"] * snr_raw / TARGET_SNR
    print(f"  Scaling d_L: {INJ_PAR_DIC['d_luminosity']:.1f} -> {d_L_scaled:.1f} Mpc")

    # Recreate with scaled distance (fresh noise)
    inj_event = data.EventData.gaussian_noise(
        eventname="GW231123_injection",
        duration=duration,
        detector_names="HL",
        asd_funcs=["asd_H_O3", "asd_L_O3"],
        tgps=tgps,
        fmax=fmax,
        seed=12345,
    )
    inj_par_scaled = INJ_PAR_DIC.copy()
    inj_par_scaled["d_luminosity"] = d_L_scaled

    print("  Pass 2: injecting with scaled distance...")
    inj_event.inject_signal(inj_par_scaled, "NRSur7dq4")

    inj = inj_event.injection
    h_h = inj["h_h"]
    snr_per_det = np.sqrt(h_h)
    snr_network = np.sqrt(np.sum(h_h))
    print(f"  Injected SNR per detector: {dict(zip(inj_event.detector_names, snr_per_det))}")
    print(f"  Network SNR: {snr_network:.1f}")

    inj_path = OUTPUT_DIR / "injection_event.pkl"
    with open(inj_path, "wb") as f:
        pickle.dump(inj_event, f)
    print(f"  Saved: {inj_path} ({inj_path.stat().st_size / 1e6:.1f} MB)")

    # Save injection metadata
    meta = {
        "injection_params": {k: float(v) for k, v in inj_par_scaled.items()},
        "injection_approximant": "NRSur7dq4",
        "snr_H": float(snr_per_det[0]),
        "snr_L": float(snr_per_det[1]),
        "snr_network": float(snr_network),
    }
    meta_path = OUTPUT_DIR / "injection_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # --- Plot event data spectra ---
    print("\nGenerating event data spectra plot...")
    plot_event_spectra(real_event, inj_event, snr_per_det)

    # --- Gate check ---
    print(f"\nGate 1c checks:")
    print(f"  Real event loads: OK")
    print(f"  Injection event loads: OK")
    print(f"  Injection SNR = {snr_network:.1f} (target ~23)")
    if 18 < snr_network < 28:
        print("  SNR within 20% of 23: PASS")
    else:
        print(f"  WARNING: SNR {snr_network:.1f} outside expected range [18, 28]")


def plot_event_spectra(real_event, inj_event, inj_snr_per_det):
    """Plot strain spectra for real and injection events."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (event, title) in zip(axes, [
        (real_event, "Real GW231123"),
        (inj_event, "GW231123-like Injection"),
    ]):
        f = event.frequencies
        mask = (f >= 20) & (f <= 500)
        for i, det in enumerate(event.detector_names):
            strain_amp = np.abs(event.strain[i])
            wht = event.wht_filter[i]
            ax.semilogy(f[mask], strain_amp[mask], alpha=0.5, label=f"{det} strain", linewidth=0.5)
            # Show ASD (1/wht)
            asd_mask = wht[mask] > 0
            asd = np.full_like(f[mask], np.nan, dtype=float)
            asd[asd_mask] = 1.0 / wht[mask][asd_mask]
            ax.semilogy(f[mask], asd, label=f"{det} ASD", linewidth=1)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Strain [1/Hz]")
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    outpath = Path(__file__).resolve().parent.parent / "output" / "gw231123_event_data.png"
    outpath.parent.mkdir(exist_ok=True)
    plt.savefig(outpath, dpi=150)
    print(f"  Saved: {outpath}")


if __name__ == "__main__":
    main()
