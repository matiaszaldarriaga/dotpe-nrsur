#!/usr/bin/env python
"""
Step 1/3: Set up shared objects for distributed incoherent scoring.

Creates event data (with injection) and finds the reference waveform (par_dic_0),
then pre-computes per-detector summary weights. All outputs saved to a run directory
on the shared filesystem so SLURM workers can read them.

Usage:
    python scripts/setup_incoherent.py \
        --run-dir /data/matiasz/dot-pe/incoherent_1m/aligned_nrsur_into_xphm \
        --event-pkl /data/matiasz/dot-pe/comparison_65k/events/aligned_nrsur/event_data.pkl \
        --bank-dir /data/matiasz/dot-pe/bank_1m_xphm

Output structure:
    {run_dir}/
    ├── setup/
    │   ├── event_data.pkl
    │   ├── par_dic_0.json
    │   ├── bank_config.json       (copied from bank)
    │   ├── summary_H.npz          (dh_weights, hh_weights for detector H)
    │   ├── summary_L.npz
    │   └── summary_V.npz
    └── scores/                    (empty, workers write here)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
import time
import pickle
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

from cogwheel import gw_utils
from cogwheel.posterior import Posterior
from cogwheel.likelihood import RelativeBinningLikelihood
from cogwheel.waveform import WaveformGenerator
from cogwheel.waveform_models import nrsurrogate  # noqa — registers NRSur7dq4
from cogwheel.waveform_models import xode  # noqa — registers IMRPhenomXODE
from dot_pe.inference import extract_single_detector_event_data
from dot_pe.likelihood_calculating import LinearFree
from dot_pe.sample_processing import IntrinsicSampleProcessor


def main():
    parser = argparse.ArgumentParser(description="Setup for distributed incoherent scoring")
    parser.add_argument("--run-dir", required=True, help="Output directory for this run")
    parser.add_argument("--event-pkl", required=True, help="Path to pickled EventData")
    parser.add_argument("--bank-dir", required=True, help="Path to waveform bank")
    parser.add_argument("--mchirp-guess", type=float, default=None,
                        help="Chirp mass guess for reference finding (auto-detected if not given)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    setup_dir = run_dir / "setup"
    setup_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scores").mkdir(exist_ok=True)

    bank_dir = Path(args.bank_dir)

    # --- Load event data ---
    print(f"Loading event data from {args.event_pkl}...")
    with open(args.event_pkl, "rb") as f:
        event_data = pickle.load(f)

    # Copy event data to run dir
    with open(setup_dir / "event_data.pkl", "wb") as f:
        pickle.dump(event_data, f)

    # --- Load bank config ---
    with open(bank_dir / "bank_config.json") as f:
        bank_config = json.load(f)

    with open(setup_dir / "bank_config.json", "w") as f:
        json.dump(bank_config, f, indent=4)

    fbin = np.array(bank_config["fbin"])
    f_ref = bank_config["f_ref"]

    # --- Auto-detect mchirp guess from injection if not given ---
    mchirp_guess = args.mchirp_guess
    if mchirp_guess is None:
        meta_path = Path(args.event_pkl).parent / "injection_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            inj = meta["injection_params"]
            mchirp_guess = gw_utils.m1m2_to_mchirp(inj["m1"], inj["m2"])
            print(f"  Auto-detected mchirp_guess = {mchirp_guess:.2f} from injection metadata")
        else:
            mchirp_guess = (bank_config["mchirp_min"] + bank_config["mchirp_max"]) / 2
            print(f"  No injection metadata found, using midpoint mchirp_guess = {mchirp_guess:.1f}")

    # --- Find reference waveform (par_dic_0) ---
    print("Finding reference waveform (par_dic_0) with IMRPhenomXAS...")
    t0 = time.time()
    coherent_posterior = Posterior.from_event(
        event=event_data,
        mchirp_guess=mchirp_guess,
        likelihood_class=RelativeBinningLikelihood,
        approximant="IMRPhenomXAS",
        prior_class="CartesianIASPrior",
        likelihood_kwargs={"fbin": fbin, "pn_phase_tol": None},
        ref_wf_finder_kwargs={"time_range": (-5e-1, +5e-1), "f_ref": f_ref},
    )
    par_dic_0 = coherent_posterior.likelihood.par_dic_0.copy()
    elapsed = time.time() - t0
    print(f"  par_dic_0 found in {elapsed:.1f} s")

    # --- Refine par_dic_0 for the bank's approximant ---
    # XAS finder returns t_geocenter/phi_ref optimized for IMRPhenomXAS.
    # Different approximants (especially NRSur7dq4) can have merger times
    # shifted by ~50 ms, which exceeds the ±20 ms scoring time grid.
    # FFT refinement re-optimizes (t_geocenter, phi_ref) using the bank
    # approximant, keeping intrinsic parameters fixed. Cost: ~1-6 s.
    #
    # Cross-validation: At high mass (M_total > 200), NRSur7dq4 can find
    # spurious timing peaks 180+ ms from the signal. We cross-check against
    # XPHM and fall back if they disagree by more than 50 ms.
    bank_approx = bank_config["approximant"]
    if bank_approx != "IMRPhenomXAS":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'lib'))
        from waveform_tools.par_dic_0_refine import refine_par_dic_0_fft
        # Use XPHM cross-validation for NRSur to catch high-mass timing issues
        ref_approx = ("IMRPhenomXPHM" if bank_approx == "NRSur7dq4"
                       else None)
        print(f"Refining par_dic_0 for {bank_approx}"
              f"{f' (cross-validating with {ref_approx})' if ref_approx else ''}...")
        par_dic_0, refine_info = refine_par_dic_0_fft(
            event_data, par_dic_0, bank_approx, det_name='L',
            reference_approximant=ref_approx,
            max_dt_disagreement_ms=50.0)
        print(f"  dt = {refine_info['best_dt_ms']:+.3f} ms, "
              f"phi_ref = {par_dic_0['phi_ref']:.4f}")
        if refine_info.get('used_fallback'):
            print(f"  FALLBACK: {refine_info['fallback_reason']}")

    # Save par_dic_0
    par_dic_0_serializable = {k: float(v) for k, v in par_dic_0.items()}
    with open(setup_dir / "par_dic_0.json", "w") as f:
        json.dump(par_dic_0_serializable, f, indent=2)

    # --- Pre-compute summary weights per detector ---
    print("Pre-computing summary weights per detector...")
    waveform_dir = bank_dir / "waveforms"

    for det_name in event_data.detector_names:
        summary_path = setup_dir / f"summary_{det_name}.npz"
        if summary_path.exists():
            print(f"  {det_name}: already exists, skipping")
            continue

        t0 = time.time()
        event_data_1d = extract_single_detector_event_data(event_data, det_name)
        approx_for_ref = bank_config["approximant"]
        wfg = WaveformGenerator.from_event_data(event_data_1d, approx_for_ref)
        likelihood_linfree = LinearFree(event_data_1d, wfg, par_dic_0, fbin)
        isp = IntrinsicSampleProcessor(likelihood_linfree, waveform_dir)
        dh_weights, hh_weights = isp.get_summary()
        np.savez(summary_path, dh_weights=dh_weights, hh_weights=hh_weights)
        elapsed = time.time() - t0
        print(f"  {det_name}: summary computed in {elapsed:.1f} s "
              f"(dh shape={dh_weights.shape}, hh shape={hh_weights.shape})")

    # --- Save run metadata ---
    meta = {
        "event_pkl": str(args.event_pkl),
        "bank_dir": str(bank_dir),
        "bank_size": bank_config["bank_size"],
        "blocksize": bank_config["blocksize"],
        "n_blocks": bank_config["bank_size"] // bank_config["blocksize"],
        "approximant": bank_config["approximant"],
        "n_modes": len(bank_config["m_arr"]),
        "mchirp_guess": mchirp_guess,
        "detectors": list(event_data.detector_names),
    }
    with open(setup_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSetup complete: {setup_dir}")
    print(f"  Bank size: {meta['bank_size']:,}")
    print(f"  Blocks: {meta['n_blocks']}")
    print(f"  Detectors: {meta['detectors']}")
    print(f"\nNext: submit SLURM array with {meta['n_blocks']} tasks")


if __name__ == "__main__":
    main()
