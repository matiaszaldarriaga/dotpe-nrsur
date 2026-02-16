#!/usr/bin/env python
"""
Validate relative binning accuracy for top-scoring templates.

For each approximant's top N templates (by incoherent lnL), computes the
full-frequency matched-filter lnL per detector and compares against the
relbin lnL from scoring. This catches relbin artifacts where the summary
weights fail for specific waveform shapes.

Usage:
    python scripts/validate_relbin_top.py \
        --run-dir /data/matiasz/dot-pe/gw231123/incoherent/real_into_nrsur \
        --bank-dir /data/matiasz/dot-pe/gw231123/banks/f220_nrsur7dq4_corrected \
        --n-top 10

Output:
    Prints table of (bank_idx, relbin_H, full_H, diff_H, relbin_L, full_L, diff_L).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Need cogwheel and dot-PE
sys.path.insert(0, '/home/matiasz/claude-projects/dot-pe/packages/cogwheel')
sys.path.insert(0, '/home/matiasz/claude-projects/dot-pe/packages/dot-PE')

import pickle
from cogwheel.waveform import WaveformGenerator
from cogwheel.waveform_models import nrsurrogate  # noqa - registers NRSur7dq4
from dot_pe.inference import extract_single_detector_event_data
from dot_pe.likelihood_calculating import LinearFree
from dot_pe.single_detector import SingleDetectorProcessor
from cogwheel import skyloc_angles
from cogwheel.gw_utils import DETECTORS, get_geocenter_delays
import pandas as pd


def full_freq_lnl(event_data_1d, wfg, par_dic_0, bank_params, n_t=320):
    """
    Compute the full-frequency matched-filter lnL for one template in one detector.

    Uses FFT-based matched filtering: <d|h>(t) = IFFT(d* h / Sn) gives
    the matched filter at all time shifts simultaneously. Then optimizes
    over (phi_ref, time_shift, distance/polarization).

    Returns the maximum lnL over (phi_ref, t_geocenter).
    """
    freqs = event_data_1d.frequencies  # positive freqs (rfft grid)
    wht_filter = event_data_1d.wht_filter[0]
    df = event_data_1d.df
    nfft = event_data_1d.nfft
    d_f = event_data_1d.strain[0]

    # Whiten data
    d_wht = d_f * wht_filter

    # Build time grid
    det_name = event_data_1d.detector_names[0]
    tgps = event_data_1d.tgps
    lat, lon = skyloc_angles.cart3d_to_latlon(
        skyloc_angles.normalize(DETECTORS[det_name].location))
    delay_val = get_geocenter_delays(det_name, lat, lon)[0]
    tcoarse = event_data_1d.tcoarse
    dt_sample = event_data_1d.times[1]
    t_grid = (np.arange(n_t) - n_t // 2) * dt_sample
    t_grid += par_dic_0["t_geocenter"] + tcoarse + delay_val

    # Convert time grid to sample indices in the IFFT output
    times_full = event_data_1d.times  # full time array
    t_indices = np.array([np.argmin(np.abs(times_full - t)) for t in t_grid])

    # Scan over phi_ref (cogwheel caches modes, so only 1 waveform generation)
    n_phi = 50
    phi_refs = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    best_lnl = -np.inf

    for phi_ref in phi_refs:
        wf_par_dic = {
            'm1': bank_params['m1'], 'm2': bank_params['m2'],
            's1x_n': bank_params['s1x_n'], 's1y_n': bank_params['s1y_n'],
            's1z': bank_params['s1z'],
            's2x_n': bank_params['s2x_n'], 's2y_n': bank_params['s2y_n'],
            's2z': bank_params['s2z'],
            'iota': bank_params['iota'], 'l1': 0.0, 'l2': 0.0,
            'f_ref': 20.0, 'd_luminosity': 1.0, 'phi_ref': phi_ref,
        }

        hpc = wfg.get_hplus_hcross(freqs, wf_par_dic, by_m=False)  # (2, nfreq)
        hp_wht = hpc[0] * wht_filter
        hx_wht = hpc[1] * wht_filter

        # Template inner products (time-independent)
        hpp = 4 * df * np.sum(np.abs(hp_wht)**2).real
        hpx = 4 * df * np.sum((hp_wht.conj() * hx_wht)).real
        hxx = 4 * df * np.sum(np.abs(hx_wht)**2).real
        hh_matrix = np.array([[hpp, hpx], [hpx, hxx]])

        det_hh = hpp * hxx - hpx**2
        if det_hh < 1e-30:
            continue
        hh_inv = np.array([[hxx, -hpx], [-hpx, hpp]]) / det_hh

        # FFT-based matched filter: <d|h>(t) for all t simultaneously
        # Inner product: <d|h>(t_n) = 4*df * Re[sum_k X[k] exp(-2pi i k n/N)]
        # irfft uses exp(+2pi i), so we conjugate X to get exp(-2pi i):
        #   <d|h>(t_n) = 2 * df * nfft * irfft(X.conj())[n]
        # Factor 2 (not 4) because irfft doubles non-DC bins (two-sided sum).
        norm = 2 * df * nfft
        X_plus = d_wht.conj() * hp_wht
        X_cross = d_wht.conj() * hx_wht
        dh_plus_t = np.fft.irfft(X_plus.conj()) * norm
        dh_cross_t = np.fft.irfft(X_cross.conj()) * norm

        # Extract values at our time grid
        dh_plus_grid = dh_plus_t[t_indices]
        dh_cross_grid = dh_cross_t[t_indices]

        # Vectorized 2×2 solve: lnL(t) = 0.5 * dh^T @ hh_inv @ dh
        # = 0.5 * (hh_inv[0,0]*dh+^2 + 2*hh_inv[0,1]*dh+*dhx + hh_inv[1,1]*dhx^2)
        lnl_t = 0.5 * (hh_inv[0, 0] * dh_plus_grid**2
                        + 2 * hh_inv[0, 1] * dh_plus_grid * dh_cross_grid
                        + hh_inv[1, 1] * dh_cross_grid**2)

        max_lnl_phi = lnl_t.max()
        if max_lnl_phi > best_lnl:
            best_lnl = max_lnl_phi

    return best_lnl


def main():
    parser = argparse.ArgumentParser(description="Validate relbin accuracy for top templates")
    parser.add_argument("--run-dir", required=True, help="Incoherent run directory")
    parser.add_argument("--bank-dir", required=True, help="Bank directory")
    parser.add_argument("--n-top", type=int, default=10, help="Number of top templates to check")
    parser.add_argument("--exclude-idx", type=int, nargs='*', default=[],
                        help="Bank indices to exclude (known artifacts)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    bank_dir = Path(args.bank_dir)
    setup_dir = run_dir / "setup"
    scores_dir = run_dir / "scores"

    # Load setup
    with open(setup_dir / "event_data.pkl", "rb") as f:
        event_data = pickle.load(f)
    with open(setup_dir / "par_dic_0.json") as f:
        par_dic_0 = json.load(f)
    with open(setup_dir / "bank_config.json") as f:
        bank_config = json.load(f)

    approximant = bank_config["approximant"]
    detectors = list(event_data.detector_names)
    intrinsic_bank = pd.read_feather(bank_dir / "intrinsic_sample_bank.feather")

    print(f"Run: {run_dir.name}")
    print(f"Approximant: {approximant}")
    print(f"Detectors: {detectors}")
    print(f"Checking top {args.n_top} templates")
    if args.exclude_idx:
        print(f"Excluding: {args.exclude_idx}")

    # Load survivors to find top templates
    surv = np.load(run_dir / "survivors.npz")
    surv_inds = surv["survivor_inds"]
    surv_incoh = surv["survivor_incoherent"]
    surv_lnlike_di = surv["survivor_lnlike_di"]

    # Exclude known artifacts
    if args.exclude_idx:
        mask = np.ones(len(surv_inds), dtype=bool)
        for idx in args.exclude_idx:
            mask &= surv_inds != idx
        surv_inds = surv_inds[mask]
        surv_incoh = surv_incoh[mask]
        surv_lnlike_di = surv_lnlike_di[:, mask]

    # Get top N by incoherent lnL
    top_order = np.argsort(surv_incoh)[::-1][:args.n_top]
    top_bidxs = surv_inds[top_order]
    top_incoh = surv_incoh[top_order]
    top_lnlike_di = surv_lnlike_di[:, top_order]

    print(f"\nTop {args.n_top} templates by incoherent lnL:")
    print(f"{'bank_idx':>10} {'incoh':>8} " + " ".join(f"{d}_relbin" for d in detectors))
    for i in range(len(top_bidxs)):
        det_str = " ".join(f"{top_lnlike_di[d, i]:>8.2f}" for d in range(len(detectors)))
        print(f"{top_bidxs[i]:>10} {top_incoh[i]:>8.2f} {det_str}")

    # Compute full-frequency lnL for each
    results = []
    for det_idx, det_name in enumerate(detectors):
        print(f"\n--- Detector {det_name} ---")
        event_data_1d = extract_single_detector_event_data(event_data, det_name)
        wfg = WaveformGenerator.from_event_data(event_data_1d, approximant)

        for i, bidx in enumerate(top_bidxs):
            relbin_lnl = top_lnlike_di[det_idx, i]
            bank_params = intrinsic_bank.iloc[int(bidx)].to_dict()

            print(f"  [{i+1}/{len(top_bidxs)}] bank_idx={bidx}, "
                  f"relbin {det_name}={relbin_lnl:.2f} ... ", end="", flush=True)

            full_lnl = full_freq_lnl(event_data_1d, wfg, par_dic_0, bank_params, n_t=320)
            diff = relbin_lnl - full_lnl
            print(f"full={full_lnl:.2f}, diff={diff:.2f}")

            results.append({
                'bank_idx': int(bidx),
                'det': det_name,
                'relbin_lnl': float(relbin_lnl),
                'full_lnl': float(full_lnl),
                'diff': float(diff),
            })

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    header = (f"{'bank_idx':>10} {'incoh':>8} "
              + " ".join(f"  {d}_rb  {d}_ff  {d}_err" for d in detectors))
    print(header)
    print("-" * len(header))

    max_err = 0
    for i, bidx in enumerate(top_bidxs):
        parts = [f"{bidx:>10}", f"{top_incoh[i]:>8.2f}"]
        for det_idx, det_name in enumerate(detectors):
            r = [x for x in results if x['bank_idx'] == bidx and x['det'] == det_name][0]
            parts.append(f"{r['relbin_lnl']:>7.2f}")
            parts.append(f"{r['full_lnl']:>7.2f}")
            err = abs(r['diff'])
            flag = " ***" if err > 2.0 else ""
            parts.append(f"{r['diff']:>7.2f}{flag}")
            max_err = max(max_err, err)
        print(" ".join(parts))

    print(f"\nMax |relbin - full_freq| = {max_err:.2f} lnL")
    if max_err < 1.0:
        print("RESULT: PASS — all relbin errors < 1 lnL")
    elif max_err < 5.0:
        print(f"RESULT: WARNING — max relbin error = {max_err:.2f} lnL (threshold: 5)")
    else:
        print(f"RESULT: FAIL — max relbin error = {max_err:.2f} lnL (>5, likely artifact)")


if __name__ == "__main__":
    main()
