# Optimization Log — dot-PE Inference

## Baseline (2026-02-13)
- **Test case**: aligned_nrsur_into_nrsur, 1M NRSur bank, N=200 survivors, seed=42
- **Total**: 320.8s | Phase 4: 259.0s (81%) | Phase 5: 30.5s (9%) | Phase 6: 26.1s (8%)
- **Correctness**: ln_evidence=161.864, n_effective=62.84, bestfit_lnlike_max=205.795
- **Commit**: `8c4a62c` (dot-PE nrsur7dq4 branch)

---

## Iterations

### Iteration 1: lnlike filter early-stopping
- **Date**: 2026-02-13
- **Hypothesis**: Evaluating all 1024 lnlike calls is wasteful when only 16 MarginalizationInfo objects are needed. The filter pass rate is high, so ~32 evaluations should suffice.
- **Change**: `get_marg_info_batch_multibank` now accepts `max_filter_valid` parameter. The lnlike filter loop exits early after collecting `max(2 * n_remaining, 32)` valid candidates. `get_marg_info_multibank` passes this down.
- **Result**:

| Phase | Baseline | Current | Speedup |
|-------|----------|---------|---------|
| 4_extrinsic | 267.7s | 104.0s | 2.58x |
| 5_coherent | 28.5s | 26.7s | 1.07x |
| 6_aggregation | 25.2s | 26.1s | 0.97x |
| TOTAL | 328.2s | 162.3s | **2.02x** |

- **Correctness**: PASS (all checks). bestfit_lnlike diff=0.142, injection diff=0.000
- **Surprise**: Phase 4 still 104s despite marginfo collection only taking 28s. The `draw_extrinsic_samples_from_indices` call (after marginalization) must account for the remaining ~75s.
- **Commit**: `5f0ce46` (dot-PE optimization branch)

### Iteration 2: Skip lnlike filter when threshold=0
- **Date**: 2026-02-13
- **Hypothesis**: With `min_marg_lnlike_for_sampling=0.0` (the default), the lnlike filter accepts all candidates — every NRSur call is wasted. Skip the filter entirely and pass candidates directly to the waveform-loading+MI path.
- **Change**: When `min_marg_lnlike_for_sampling <= 0`, take the first `n_to_process` candidates directly without calling `lnlike()`. Preserves filter for non-zero thresholds.
- **Result**:

| Phase | Baseline | Current | Speedup |
|-------|----------|---------|---------|
| 4_extrinsic | 267.7s | 80.5s | 3.33x |
| 5_coherent | 28.5s | 26.3s | 1.09x |
| 6_aggregation | 25.2s | 25.8s | 0.98x |
| TOTAL | 328.2s | 139.0s | **2.36x** |

- **Correctness**: PASS (all checks). bestfit_lnlike diff=0.022, injection diff=0.000
- **Surprise**: Phase 4 still 80.5s despite marginfo taking only 3.2s. The ~49s fixed overhead from `_set_d_h_weights` in cogwheel dominates. Plus ~27s from draw_extrinsic_samples_from_indices and I/O.
- **Commit**: `9354858` (dot-PE optimization branch)

### Iteration 3: Vectorize _set_d_h_weights — REVERTED
- **Date**: 2026-02-13
- **Hypothesis**: Batching 128 time bins into chunked sparse-dense matrix multiplication should be faster than 128 × 27 individual sparse-vector products.
- **Change**: Override `_set_d_h_weights` in dot-PE's `MarginalizationExtrinsicSamplerFreeLikelihood` with chunked batch sparse matmul (chunk_size=16).
- **Result**: REGRESSION! Phase 4 went from 80.5s → 108.8s (+35%). Memory usage spiked from 2.1 GB to 5.5 GB.
- **Analysis**: The large intermediate arrays from batching (16 × 9 × 3 × nrfft complex128) caused cache thrashing. The original per-vector sparse matmul is already cache-friendly. The overhead of creating/reshaping large arrays dominated any vectorization gains.
- **Correctness**: PASS (same results as iter 2).
- **Action**: REVERTED.

<!-- Template for each iteration:

### Iteration N: <title>
- **Date**: YYYY-MM-DD
- **Hypothesis**: ...
- **Change**: ...
- **Result**:

| Phase | Baseline | Current | Speedup |
|-------|----------|---------|---------|
| 4_extrinsic | 259.0s | Xs | X.Xx |
| 5_coherent | 30.5s | Xs | X.Xx |
| 6_aggregation | 26.1s | Xs | X.Xx |
| TOTAL | 320.8s | Xs | X.Xx |

- **Correctness**: PASS/FAIL
- **Surprise**: ...
- **Commit**: ...
-->
