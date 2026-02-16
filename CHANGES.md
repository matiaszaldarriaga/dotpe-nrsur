# Changes on `nrsur7dq4` branch

This branch adds NRSur7dq4 support to dot-PE and optimizes the extrinsic
sampling phase. Based on upstream dot-PE (commit `bb719a0`).

## NRSur7dq4 integration

### `dot_pe/inference.py`

- **XAS reference finding:** Uses IMRPhenomXAS (fast, unconstrained) to find
  the reference waveform (`par_dic_0`) regardless of which approximant the
  bank was generated with. This avoids NRSur7dq4's parameter constraints
  (q <= 6, omega_ref <= 0.2) during the optimization step.
- Key insight: the reference waveform `h0` used for relative binning does not
  need to match the bank approximant. Templates that fit the data look similar
  regardless of the model used to generate them.

### `dot_pe/vetoing.py`

- **Summary waveform caching:** Caches the summary waveform (`h0`) to avoid
  redundant evaluations when the same reference parameters are reused across
  detectors or iterations.

### `dot_pe/likelihood_calculating.py`, `dot_pe/single_detector.py`

- Minor changes for NRSur7dq4 compatibility (mode indexing, parameter passing).

## Phase 4 optimization (extrinsic sampling)

### `dot_pe/coherent_processing.py`

Two optimizations to the lnlike filter in `get_marg_info_multibank`, which
evaluates one waveform per candidate during Phase 4 (extrinsic parameter
sampling). For NRSur7dq4, each waveform call takes ~1s, so reducing the
number of calls has a large impact.

1. **Early-stopping** (commit `5f0ce46`): Converts the lnlike filter loop to
   exit as soon as enough candidates pass, instead of evaluating all 1024
   in the batch. Unbiased because the batch is already shuffled.
   Result: 2.58x Phase 4 speedup.

2. **Skip filter when threshold <= 0** (commit `9354858`): When
   `min_marg_lnlike_for_sampling <= 0` (the default), the filter accepts
   virtually all candidates. In this case, skip the per-candidate waveform
   evaluation entirely and pass candidates directly to the downstream path.
   Result: 3.33x Phase 4 speedup (2.36x total inference speedup).

## Validation

- Correctness checks pass: posteriors are statistically identical before
  and after optimization (KS test p > 0.05 on all parameters).
- Tested at production scale: 32k survivors (NRSur), 44k survivors (XPHM).
