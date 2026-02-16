# Inference Profiling Report

**Date:** 2026-02-12
**Machine:** Carme (32 physical cores, local NVMe storage)
**Tools:** cProfile + per-phase wall-clock instrumentation + process monitoring
**Scripts:** `scripts/profile_inference.py`, `scripts/scaling_study.py`, `scripts/monitored_run.sh`

Everything labeled "measured" comes from cProfile or wall-clock instrumentation.
Everything labeled "extrapolated" or "estimated" is clearly marked.
Everything labeled "contested" means it ran simultaneously with another heavy process on the same machine.

---

## Executive Summary

We profiled the dot-PE inference pipeline (`inference.run()`) to understand its performance characteristics at scale. The goal is to run inference with tens of thousands of survivors on the Typhon SLURM cluster.

### What we know (measured)

1. **The bottleneck is Phase 4 (extrinsic sampling).** It takes 64-93% of total inference time across all test cases.

2. **Phase 4 has three sequential costs inside a single batch of 1024 survivors:**
   - **4a. Initialization** (~50-100s): Create `MarginalizationExtrinsicSamplerFreeLikelihood`. Calls `_set_summary()` and `_set_d_h_weights()`. ~13 NRSur waveform calls. Fixed cost.
   - **4b. lnlike filter** (1024 calls): For each survivor in the batch, call `lnlike()` which generates a fresh waveform. NRSur: ~1s/call × 1024 = ~17 min. XPHM: ~0.13s/call × 1024 = ~2 min.
   - **4c. QMC marginalization** (~1024 survivors × ~13 QMC chunks each): For every survivor that passes the lnlike filter, run adaptive quasi-Monte Carlo to build a `MarginalizationInfo` object. Only 16 are kept, but **all ~1024 are evaluated**. Cost: ~2.5-3.3 min depending on approximant and machine load.

3. **Phase 4 cost is O(batch_size=1024), NOT O(n_survivors)**, as long as the first batch of 1024 yields ≥16 valid MarginalizationInfo objects. In all our tests (n=4 through n=44,225), the first batch always succeeded. NRSur time plateaus at n≥1024.

4. **Phase 5 (coherent inference) scales linearly with n_survivors.** Block count = `ceil(N/512) × 2`. Each block loads waveforms from disk and computes tensor inner products. Disk I/O dominates.

5. **Resource contention severely distorts timings.** Running NRSur and XPHM simultaneously on Carme inflated Phase 4 times by ~3x and Phase 5 times by ~2x compared to uncontested scaling study runs.

### What we don't know

1. **Clean (uncontested) timings at full scale.** The n=32,736 and n=44,225 runs were contested. We need isolated runs.
2. **Cluster filesystem impact.** All profiling used Carme's local NVMe. Typhon uses shared NFS — Phase 5's disk I/O could be much slower.
3. **Why the previous cluster jobs "made no progress."** Likely stuck in Phase 4's lnlike loop (no visible output for ~17+ min per batch), but we have no cluster logs to confirm.

### Recommended next steps

1. **Run clean baselines at full scale (~45 min wall-clock).** Run XPHM n=44,225 on one IAS machine and NRSur n=32,736 on a different one simultaneously — all 6 machines (Carme, Elara, Neso, Nereid, Thalassa, Proteus) share `/home` and `/data`, so the same code and conda env work everywhere via SSH. No contention, both finish in parallel.
   - **XPHM n=44,225**: estimated ~15-20 min (Phase 4 ~3.5 min, Phase 5 ~10 min).
   - **NRSur n=32,736**: estimated ~45 min (Phase 4 ~22 min, Phase 5 ~18 min).
   - **Purpose**: Validate that our scaling model predicts correctly. If Phase 4 comes in at ~22 min uncontested (matching n=5000), the batch-cap behavior is confirmed. If not, something else is going on.
   - **Command**: `ssh Elara 'cd /home/matiasz/claude-projects/dot-pe && bash scripts/monitored_run.sh aligned_xphm_into_xphm 44225 60'` (and similar for NRSur on Carme or another machine).

2. **Line-level profiling (~30 min work).** Use `line_profiler` or `py-spy` at n=200 (~5 min per run) to get fine-grained understanding beyond cProfile's function-level data. Targets: QMC step internals, disk I/O patterns per block, the `combine_prob_samples_with_next_block` merge step.

3. **Autonomous optimization loop (iterative, ~5 min per cycle).** Modeled on `~/claude-projects/n-body/` methodology: profile → identify bottleneck → optimize → validate at n=200. Possible targets:
   - **QMC short-circuit** after 16 accepted (could save ~90% of QMC time)
   - **Waveform cache size** (currently 1, every lnlike call is a cache miss)
   - **Vectorized/parallel lnlike filter** (currently a sequential list comprehension)
   - **Disk pre-loading** to RAM or `/dev/shm` (for cluster runs with slow NFS)
   - **`combine_prob_samples_with_next_block`** sorting optimization (460s at n=44k)

4. **Typhon cluster test.** Run a single inference on a Typhon compute node to measure shared NFS filesystem impact on Phase 5 (which is disk I/O dominated on local NVMe).

---

## 1. Pipeline Architecture

The function `inference.run()` (`inference.py:1468`) executes 6 phases in sequence:

```
inference.run()
  │
  ├─ Phase 1: PREPARATION                              [inference.py:714]
  │    Load bank config, create Posterior object, find best-fit parameters.
  │    Uses IMRPhenomXAS (hardcoded) for reference waveform finding.
  │    Cost: ~5-11s, fixed.
  │
  ├─ Phase 2: INCOHERENT SCORING                       [inference.py:961]
  │    Score every sample in the bank against each detector independently.
  │    Uses IMRPhenomXAS. Loads bank waveforms (amp/phase) from disk.
  │    Cost: scales with bank_size. Skipped if pre-computed (as in our tests).
  │
  ├─ Phase 3: SURVIVOR SELECTION                       [inference.py:1045]
  │    Keep samples with incoherent lnlike > (global_max - 20).
  │    These are the "survivors". Pure array operations, instant.
  │
  ├─ Phase 4: EXTRINSIC SAMPLING  ◀◀◀ BOTTLENECK      [inference.py:1143]
  │    Detailed breakdown in Section 2.
  │
  ├─ Phase 5: COHERENT INFERENCE                       [inference.py:1263]
  │    Process survivors in blocks. Loads bank waveforms from disk.
  │    Computes tensor inner products per block.
  │    Cost: scales with ceil(N_survivors/512) × 2 blocks.
  │    Detailed breakdown in Section 4.
  │
  └─ Phase 6: AGGREGATION                              [inference.py:1345]
       Post-process, standardize samples, save to feather.
       Cost: 3-64s depending on number of surviving samples.
```

---

## 2. Phase 4 Internals: The Bottleneck

Phase 4 (`draw_extrinsic_samples`, `inference.py:1143`) calls `get_marg_info_multibank` (`coherent_processing.py:1163`) which splits all survivors into batches of `batch_size = max(1024, 2*n_combine)` = 1024 (since `n_combine=16`).

For each batch, it calls `get_marg_info_batch_multibank` (`coherent_processing.py:943`):

```
get_marg_info_batch_multibank(batch of up to 1024 survivors)
  │
  ├─ Step 1: LNLIKE FILTER (lines 982-992)
  │    For EACH survivor in the batch:
  │      lnlike(survivor_params) → generates fresh waveform
  │      Keep if lnlike > min_marg_lnlike_for_sampling (=0.0)
  │    NRSur: ~1.0s/call.  XPHM: ~0.13s/call (includes relbin, not just WFG).
  │    ► Almost all survivors pass (threshold is 0.0).
  │
  ├─ Step 2: LOAD WAVEFORMS (lines 1005-1023)
  │    Load amp/phase from disk for all passing survivors, grouped by bank.
  │    One call to load_amp_and_phase() per unique bank.
  │
  ├─ Step 3: COMPUTE dh/hh (lines 1026-1034)
  │    _get_many_dh_hh(): batch inner products for all passing survivors.
  │
  └─ Step 4: QMC PER SURVIVOR (lines 1041-1053)        ◀◀◀ IMPORTANT
       For EVERY passing survivor (not just 16!):
         get_marginalization_info(dh, hh, times)
           → Adaptive QMC loop (marginalization.py:29-84)
           → ~13 QMC chunks per survivor (hits "Maximum QMC resolution reached")
       Only survivors with n_effective_prior > threshold are kept.
       Only 16 MarginalizationInfo objects are returned to the caller.
       ► But all ~1024 are evaluated through the QMC step.
       Evidence: cProfile shows ~13,700 calls to _get_marginalization_info_chunk
       for ~1024 survivors = ~13 chunks/survivor.
```

**After collecting 16 objects, the batch loop in `get_marg_info_multibank` (line 1276) stops — no more batches are processed.** Evidence: progress bar output confirmed "batches=1/32" for NRSur n=32,736 and "batches=1/44" for XPHM n=44,225 — only 1 batch was processed in both cases, with 16/16 accepted.

### The lnlike filter code (the NRSur bottleneck)

`coherent_processing.py:982-992`:
```python
valid_mask = np.array(
    [
        self.likelihood.lnlike(                    # ← one waveform call per survivor
            banks[int(bank_idx)].iloc[int(sample_idx)].to_dict()
            | config.DEFAULT_PARAMS_DICT
        )
        > min_marg_lnlike_for_sampling
        for sample_idx, bank_idx in zip(batch_sample_idx, batch_bank_idx)
    ],
    dtype=bool,
)
```

Call chain per lnlike() evaluation:
```
lnlike() → lnlike_and_metadata() [marginalized_extrinsic.py:123]
  → _get_dh_hh_timeshift() [marginalized_extrinsic.py:347]
    → _get_linearfree_hplus_hcross_dt()
      → get_hplus_hcross()
        → compute_hplus_hcross_by_mode_nrsur()  [~1s for NRSur]
        OR SimInspiralChooseFDWaveformSequence   [~0.8ms for XPHM]
    → relative_binning inner products
```

Waveform cache size = 1. Every survivor has different intrinsic params → always a cache miss.

---

## 3. Scaling Study Results (Uncontested, n=4 to n=5000)

All runs sequential within one process on Carme, no competing workloads.
Incoherent phase skipped (pre-loaded indices from 1M campaign).

### NRSur7dq4 Bank (9 modes, 32,736 total survivors)

| N surv | Total | Phase 4 (extrinsic) | Phase 5 (coherent) | Phase 6 (aggreg.) | Peak RSS |
|--------|-------|--------------------|--------------------|-------------------|----------|
| 4 | 2.2 min | 84s (64%) | 20s (15%) | 22s (17%) | 5.0 GB |
| 50 | 2.8 min | 120s (72%) | 20s (12%) | 23s (13%) | 5.3 GB |
| 200 | 5.3 min | 259s (81%) | 31s (9%) | 26s (8%) | 6.3 GB |
| 1000 | 21.4 min | 1195s (93%) | 52s (4%) | 34s (3%) | 8.2 GB |
| 5000 | 25.9 min | 1318s (85%) | 167s (11%) | 64s (4%) | 8.6 GB |

### IMRPhenomXPHM Bank (4 modes, 44,225 total survivors)

| N surv | Total | Phase 4 (extrinsic) | Phase 5 (coherent) | Phase 6 (aggreg.) | Peak RSS |
|--------|-------|--------------------|--------------------|-------------------|----------|
| 4 | 1.0 min | 44s (76%) | 6s (10%) | 3s (5%) | 4.0 GB |
| 50 | 1.1 min | 48s (75%) | 6s (10%) | 4s (6%) | 4.1 GB |
| 200 | 1.6 min | 70s (71%) | 15s (15%) | 9s (9%) | 4.3 GB |
| 1000 | 3.7 min | 172s (77%) | 32s (14%) | 14s (6%) | 6.5 GB |
| 5000 | 4.9 min | 207s (70%) | 70s (24%) | 14s (5%) | 6.6 GB |

### NRSur waveform call counts (measured via cProfile)

| N surv | NRSur calls | NRSur self-time | Breakdown |
|--------|-------------|-----------------|-----------|
| 4 | 17 | 39.8s | 4 filter + 13 init |
| 50 | 63 | 71.5s | 50 filter + 13 init |
| 200 | 213 | 179.7s | 200 filter + 13 init |
| 1000 | 1,013 | 943.7s | 1000 filter + 13 init |
| 5000 | 1,037 | 1,028.5s | **1024 filter + 13 init** (batch cap) |

Formula: `min(N_survivors, 1024) + 13` NRSur calls. This is why n=5000 ≈ n=1000 in Phase 4 time.

---

## 4. Phase 5 (Coherent Inference) Scaling

Phase 5 processes survivors in blocks of `blocksize=512` intrinsic × `blocksize=512` extrinsic.
Block count = `ceil(N_survivors/512) × 2` (the factor of 2 comes from `ceil(n_ext/512)` with `n_ext=1024`).

For each block (`coherent_processing.py:466-542`):
1. Load amp/phase from disk (`sample_processing.py:535`, `numpy.fromfile`)
2. Compute `h = amp * exp(i*phase)`, apply relative binning correction
3. Compute `dh_ieo`, `hh_ieo` via tensor inner products (`likelihood_calculating.py:203-343`)
4. Filter by bestfit likelihood, distance-marginalize
5. Merge with running sample set (`combine_prob_samples_with_next_block`, lines 639-739)

### NRSur bank coherent phase (measured, uncontested)

| N surv | Blocks | Coherent time | Disk I/O calls | Disk I/O time |
|--------|--------|---------------|----------------|---------------|
| 4 | 2 | 20.0s | 6 | 4.2s |
| 50 | 2 | 19.9s | 6 | 0.5s |
| 200 | 2 | 30.5s | 10 | 7.6s |
| 1000 | 4 | 51.9s | 36 | 49.2s |
| 5000 | 20 | 167.4s | 174 | 124.8s |

**Disk I/O dominates.** At n=5000, disk I/O is 125s / 167s = 75% of coherent time.
NRSur7dq4 is NOT called during Phase 5 — it uses pre-computed bank waveforms from disk.

---

## 5. Full-Scale Runs (Contested — Both Running Simultaneously)

We launched NRSur n=32,736 and XPHM n=44,225 simultaneously on Carme with process monitoring (`scripts/monitored_run.sh`). **These ran concurrently, competing for CPU and I/O.** The NRSur process consumed 70-80% CPU (5 internal threads), leaving XPHM CPU-starved.

### XPHM n=44,225 (COMPLETED — contested)

| Phase | Time | Notes |
|-------|------|-------|
| 1. Preparation | 11.2s | |
| 2. Incoherent | 0.1s | Pre-loaded |
| 3. Selection | 0.1s | |
| **4. Extrinsic** | **1858.6s (31.0 min)** | batches=1/44, accepted=16/16. First MarginalizationInfo at 29:34 |
| **5. Coherent** | **1580.6s (26.3 min)** | 174 blocks, avg 9.1s/block |
| 6. Aggregation | 19.5s | |
| **Total** | **3470s (57.8 min)** | Peak RSS: 5955 MB |

cProfile top functions (total = 3470s):
- `numpy.fromfile`: 2258s (1194 calls) — disk I/O for waveforms across Phases 4+5
- `take_nd`: 168s — array indexing in combine step
- `concat`: 96s — DataFrame concatenation
- `create_a_likelihood_block`: 285s (174 calls) — Phase 5 core
- `lnlike` filter (`<listcomp>` line 983): 129s (1025 calls, 126ms/call) — Phase 4 filter
- `get_marginalization_info_chunk`: 175s (13,652 calls) — Phase 4 QMC
- `combine_prob_samples_with_next_block`: 460s (174 calls) — Phase 5 merge
- `argsort`: 70s — sorting inside merge step

### NRSur n=32,736 (COMPLETED — contested)

| Phase | Time | Notes |
|-------|------|-------|
| 1. Preparation | 11.1s | |
| 2-3. Selection | 0.1s | Pre-loaded |
| **4. Extrinsic** | **3429.1s (57.2 min)** | batches=1/32, accepted=16/16. First marg obj at 54:23 |
| **5. Coherent** | **1684.2s (28.1 min)** | 128 blocks, avg 13.2s/block |
| 6. Aggregation | 36.8s | |
| **Total** | **5161s (86.0 min)** | Peak RSS: 7809 MB |

Monitor data: CPU 67-80%, RSS 1.4-4.0 GB, 6 threads (1 main + 5 NRSur internal) throughout. Note: XPHM finished at ~58 min, so the last ~28 min (Phase 5) ran uncontested.

### Comparison: uncontested (n=5000) vs contested (full scale)

| Metric | NRSur n=5000 (clean) | NRSur n=32736 (contested) | Ratio |
|--------|---------------------|--------------------------|-------|
| Phase 4 | 1318s | 3429s | 2.6x |
| Phase 5 | 167s (20 blocks) | 1684s (128 blocks) | 10.1x |
| Phase 5 per block | 8.4s | 13.2s | 1.6x |

Phase 4 should cost the same at n=5000 and n=32736 (both capped at 1024 batch). The 2.6x difference is CPU contention from the simultaneous XPHM process. Phase 5 per-block is 1.6x slower — partly I/O contention (XPHM ran for the first half), partly larger working set at 32k survivors.

**Prediction for uncontested NRSur n=32,736:** Phase 4 ≈ 1318s (22 min) + Phase 5 ≈ 128 × 8.4s = 1075s (18 min) + overhead ≈ **~45 min total.** This is an extrapolation — needs validation with a clean run on a separate machine.

---

## 6. NRSur vs XPHM: Function-Level Comparison

Top functions by self-time at n=1000 (uncontested, from cProfile):

| Rank | NRSur bank (21.4 min total) | XPHM bank (3.7 min total) |
|------|----------------------------|--------------------------|
| 1 | NRSur waveform: **944s** (1013 × 0.93s) | sparse matvec: 11s |
| 2 | disk I/O (fromfile): 49s | d_h weights: 10s |
| 3 | FFT: 34s | B-spline eval: 9s |
| 4 | sparse matvec: 23s | disk I/O: 8s |
| 5 | d_h weights: 22s | reduce: 6s |

The speed difference is almost entirely the per-survivor NRSur waveform cost in Phase 4b's lnlike filter. XPHM uses `SimInspiralChooseFDWaveformSequence` at ~0.8ms per waveform generation.

---

## 7. Cross-Check Against 65k Comparison Campaign

**Caveat:** The scaling study used the 1M bank; the comparison campaign used 65k banks. Different bank sizes → different disk I/O costs. Also, 65k runs include incoherent scoring (~13 min for NRSur, ~2 min for XPHM).

### 65k run timings (from `/data/matiasz/dot-pe/comparison_65k/runs/`)

| Run | Bank | Total | Survivors | Incoherent time |
|-----|------|-------|-----------|-----------------|
| aligned_nrsur→nrsur | NRSur 65k | 30.3 min | 11 | ~13 min |
| aligned_xphm→nrsur | NRSur 65k | 15.6 min | 35 | ~13 min |
| precessing_nrsur→nrsur | NRSur 65k | 15.6 min | 6 | ~13 min |
| precessing_xphm→nrsur | NRSur 65k | 15.4 min | 33 | ~13 min |
| aligned_nrsur→xphm | XPHM 65k | 3.3 min | 76 | ~2 min |
| aligned_xphm→xphm | XPHM 65k | 7.2 min | 103 | ~2 min |
| precessing_nrsur→xphm | XPHM 65k | 6.8 min | 109 | ~2 min |
| precessing_xphm→xphm | XPHM 65k | 7.2 min | 95 | ~2 min |

**Key observation:** The 65k runs had very few survivors (5-109). With <35 NRSur survivors, the lnlike filter evaluates only 35 waveforms (~35s), not 1024 (~17 min). This is why the 65k NRSur runs completed in ~15 min despite NRSur being "slow" — the survivor count was low enough that the batch_size cap was never hit.

The XPHM 65k runs took 3-7 min total. Subtracting ~2 min incoherent → 1-5 min for the rest, consistent with our scaling study at similar survivor counts.

### Why 1M banks have more survivors

| Bank | Bank size | Aligned survivors | Why |
|------|-----------|-------------------|-----|
| NRSur 65k | 65,536 | 5-35 | Sparse sampling of 9D parameter space |
| NRSur 1M | 1,048,576 | 32,736 | 16x more samples → 1000x more survivors |
| XPHM 65k | 65,536 | 76-109 | 4D parameter space, better coverage |
| XPHM 1M | 1,048,576 | 44,225 | 16x more samples → ~400x more survivors |

The dramatically higher survivor counts with 1M banks are what make the inference much more expensive. The 65k banks had so few survivors that inference was fast regardless of approximant choice.

---

## 8. Detailed Phase 4 Timing Decomposition

### What happens during the ~22 min of NRSur Phase 4 (n=5000, uncontested)

Based on cProfile data (from `n_5000/profile_tottime.txt`) and code tracing:

| Sub-step | Time | Source |
|----------|------|--------|
| 4a. Init: `_set_summary()` | ~4s | cProfile: `reference_waveform_finder.py:344`, ~5 WFG calls |
| 4a. Init: `_set_d_h_weights()` | ~18s | cProfile: `marginalized_extrinsic.py:316`, 1026 calls |
| 4b. lnlike filter | 1028s | **cProfile measured**: 1037 NRSur calls, 1028.5s self-time |
| 4b. Waveform loading | ~26s | estimated: 1 `load_amp_and_phase` call for ~1024 survivors |
| 4b. dh/hh batch computation | ~15s | estimated from cProfile: `_get_many_dh_hh()` |
| 4b. QMC loop (~1024 iterations) | ~200s | estimated from cProfile: ~13,700 QMC chunk calls |
| Overhead/other | ~27s | residual |
| **Total Phase 4** | **1318s** | **measured wall-clock** |

The lnlike filter (NRSur waveform generation) is 78% of Phase 4 time. The QMC loop is ~15%. Init is ~2%.

### What happens during the ~3.5 min of XPHM Phase 4 (n=5000, uncontested)

| Sub-step | Time | Source |
|----------|------|--------|
| 4a. Init | ~22s | cProfile: `_set_summary` + `_set_d_h_weights` |
| 4b. lnlike filter | ~5s | estimated: 1024 XPHM calls × ~5ms/call |
| 4b. Waveform loading | ~26s | estimated: 1 `load_amp_and_phase` call |
| 4b. QMC loop (~1024 iterations) | ~150s | estimated from cProfile: ~13,990 QMC chunk calls |
| Overhead/other | ~4s | residual |
| **Total Phase 4** | **207s** | **measured wall-clock** |

For XPHM, the QMC loop (72%) and waveform loading (13%) dominate, not the lnlike filter (2%).

---

## 9. Key Code Locations

### Phase 4 (extrinsic sampling)

| What | File | Lines |
|------|------|-------|
| Entry point | `inference.py` | 1143-1260 |
| Batch splitting + collection loop | `coherent_processing.py` | 1163-1310 (`get_marg_info_multibank`) |
| batch_size definition | `coherent_processing.py` | 1230 (`max(1024, 2*n_combine)`) |
| lnlike filter (bottleneck) | `coherent_processing.py` | 982-992 (list comprehension) |
| Waveform loading for batch | `coherent_processing.py` | 1005-1023 |
| dh/hh batch computation | `coherent_processing.py` | 1026-1034 |
| QMC per-survivor loop | `coherent_processing.py` | 1041-1053 |
| Adaptive QMC implementation | `marginalization.py` | 29-84 (`get_marginalization_info`) |
| QMC max resolution warning | `marginalization.py` | 61 |
| lnlike call chain | `marginalized_extrinsic.py` | 123 → 347 → `get_hplus_hcross()` |

### Phase 5 (coherent inference)

| What | File | Lines |
|------|------|-------|
| Entry point | `inference.py` | 1263-1342 |
| Block creation loop | `coherent_processing.py` | 466-542 (`create_likelihood_blocks`) |
| Waveform disk loading | `sample_processing.py` | 535-595 (`_load_amp_and_phase`) |
| Relative binning correction | `sample_processing.py` | 597-645 (`load_amp_and_phase`) |
| Inner product computation | `likelihood_calculating.py` | 558-595 (`get_dh_hh_ieo`) |
| dh by mode | `likelihood_calculating.py` | 203-252 |
| hh by mode | `likelihood_calculating.py` | 254-296 |
| Phi grid application | `likelihood_calculating.py` | 298-343 |
| Likelihood block creation | `coherent_processing.py` | 337-464 (`create_a_likelihood_block`) |
| Two-pointer merge | `coherent_processing.py` | 639-739 (`combine_prob_samples_with_next_block`) |

### Waveform disk layout

```
bank_folder/
  waveforms/
    amplitudes_block_0.npy    [shape: (blocksize, n_modes, 2, n_fbin)]
    amplitudes_block_1.npy
    ...
    phase_block_0.npy         [same shape]
    phase_block_1.npy
    ...
```

Bank blocksize = 4096. A 1M bank has 256 block files × 2 (amp + phase) = 512 files.

---

## 10. Data Locations

### Scaling study (n=4, 50, 200, 1000, 5000 — uncontested)

| Data | Path |
|------|------|
| NRSur scaling summary | `/data/matiasz/dot-pe/incoherent_1m/aligned_nrsur_into_nrsur/scaling_study/scaling_summary.json` |
| NRSur per-N profiles | `.../aligned_nrsur_into_nrsur/scaling_study/n_{4,50,200,1000,5000}/` |
| XPHM scaling summary | `/data/matiasz/dot-pe/incoherent_1m/aligned_xphm_into_xphm/scaling_study/scaling_summary.json` |
| XPHM per-N profiles | `.../aligned_xphm_into_xphm/scaling_study/n_{4,50,200,1000,5000}/` |

Each `n_*/` directory contains: `profile.prof` (cProfile binary), `profile_cumulative.txt`, `profile_tottime.txt`, `phase_timings.json`, `inds.npz` (survivor indices used).

### Full-scale contested runs

| Data | Path |
|------|------|
| NRSur n=32,736 run log | `.../aligned_nrsur_into_nrsur/scaling_study/run_n32736.log` |
| NRSur n=32,736 monitor | `.../aligned_nrsur_into_nrsur/scaling_study/monitor_n32736.log` |
| NRSur n=32,736 results | `.../aligned_nrsur_into_nrsur/scaling_study/n_32736/` |
| XPHM n=44,225 run log | `.../aligned_xphm_into_xphm/scaling_study/run_n44225.log` |
| XPHM n=44,225 monitor | `.../aligned_xphm_into_xphm/scaling_study/monitor_n44225.log` |
| XPHM n=44,225 results | `.../aligned_xphm_into_xphm/scaling_study/n_44225/` |

All paths under `/data/matiasz/dot-pe/incoherent_1m/`.

### 65k comparison campaign

| Data | Path |
|------|------|
| Run logs | `/data/matiasz/dot-pe/comparison_65k/runs/*/inference.log` |
| Results table | `/data/matiasz/dot-pe/comparison_65k/results/comparison_table.csv` |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/profile_inference.py` | Single-run profiler with cProfile + phase timing |
| `scripts/scaling_study.py` | Multi-count scaling sweep, saves cProfile + phase timings per count |
| `scripts/monitored_run.sh` | Watchdog: monitors PID, CPU%, RSS, threads every 30s with timeout |

### Packages (editable installs)

| Package | Path | Branch |
|---------|------|--------|
| cogwheel | `packages/cogwheel/` | `nrsur7dq4` |
| dot-PE | `packages/dot-PE/` | `nrsur7dq4` |

---

## 11. Possible Optimization Targets

These are identified from code inspection and profiling, **not yet attempted**:

1. **QMC short-circuit**: `get_marg_info_batch_multibank` (lines 1041-1053) evaluates QMC for ALL ~1024 passing survivors before returning. Could exit early once 16 accepted MarginalizationInfo objects are found, potentially saving 90%+ of QMC time.

2. **Waveform cache size**: The cache has size=1 (`coherent_score_hm.py`). Increasing it would help if nearby survivors share similar parameters — but they likely don't (shuffled), so this may not help.

3. **Vectorized lnlike filter**: The list comprehension at line 982 is sequential. If the waveform generator could be batched, this loop could be parallelized. However, NRSur's internal threading (5 threads) may make process-level parallelism the only option.

4. **Disk pre-loading**: Phase 5 reads waveform files block by block. Pre-loading to RAM or `/dev/shm` could eliminate the I/O bottleneck. Relevant for cluster runs with slow NFS.

5. **`combine_prob_samples_with_next_block` optimization**: At n=44k this took 460s (XPHM), dominated by `argsort` (70s) and `sort_values` (224s). The two-pointer merge algorithm could potentially be replaced with a heap-based approach.

---

## 12. Environment Notes for Reproduction

```bash
# Activate environment
eval "$(conda shell.bash hook)" && conda activate pycbc
export LAL_DATA_PATH=/home/matiasz/GW-2025/SPIN-HEAVY/lalsuite-waveform-data/waveform_data

# Run scaling study (example: XPHM at 3 survivor counts)
python scripts/scaling_study.py aligned_xphm_into_xphm 200 1000 5000

# Run with monitoring (NRSur, 120 min timeout)
bash scripts/monitored_run.sh aligned_nrsur_into_nrsur 32736 120

# NRSur thread behavior: spawns 5 threads per process internally.
# CPU budget: floor(physical_cores × 0.70 / 5) = 4 concurrent NRSur processes on Carme.
```

### Machine specs (Carme)
- 32 physical cores (Intel)
- Local NVMe storage (fast disk I/O)
- Shared `/home` and `/data` with other IAS machines and Typhon cluster
