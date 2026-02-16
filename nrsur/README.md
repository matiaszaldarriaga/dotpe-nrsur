# NRSur7dq4 with dot-PE: Full Workflow and Optimizations

This directory documents how to run dot-PE with the NRSur7dq4 numerical relativity
surrogate model at scale (4M+ template banks), including SLURM cluster workflows
and inference optimizations.

## Overview

dot-PE is a template-bank approach to gravitational wave parameter estimation.
The pipeline has three stages:

1. **Bank generation** — Pre-compute waveforms for millions of intrinsic parameter
   samples. Each sample produces frequency-domain amplitudes and phases for all
   (l,m) modes, stored as `.npy` files in blocks of 4096.

2. **Incoherent scoring** — For each template, compute the single-detector
   matched-filter log-likelihood using relative binning. Sum across detectors
   to get a combined "incoherent" score. Keep "survivors" within 20 lnL of the max.

3. **Coherent inference** — For the survivors, draw extrinsic parameter samples
   (sky location, distance, polarization, time) and compute coherent multi-detector
   posteriors.

### Why NRSur7dq4 is different

NRSur7dq4 is a time-domain surrogate with 9 modes (vs 4 for IMRPhenomXPHM).
This creates two challenges:

- **Bank generation is slow**: ~0.7 s/waveform (vs ~2 ms for XPHM). A 4M bank
  takes ~9 hours on a 64-node SLURM cluster vs minutes for XPHM.
- **Inference Phase 4 is slow**: The lnlike filter evaluates one waveform per
  candidate (~1 s/call for NRSur vs ~1 ms for XPHM). With batch_size=1024,
  this is ~17 minutes per batch.
- **More mode pairs**: The `<h|h>` inner product has quadratic mode scaling
  (45 pairs for 9 modes vs 10 for 4 modes).

NRSur7dq4 is **only used during bank generation**. All inference steps
(reference waveform finding, summary waveform, likelihood evaluation) use
IMRPhenomXAS/XPHM — a fast frequency-domain approximant. This works because
waveforms that fit the data all look similar regardless of approximant.

## Prerequisites

```bash
# Cogwheel fork with NRSur7dq4 support
git clone -b nrsur7dq4 git@github.com:matiaszaldarriaga/cogwheel-nrsur.git
pip install -e cogwheel-nrsur

# dot-PE fork with NRSur7dq4 integration + optimizations
git clone -b nrsur7dq4 git@github.com:matiaszaldarriaga/dotpe-nrsur.git
pip install -e dotpe-nrsur

# LAL data (NRSur7dq4 model files)
export LAL_DATA_PATH=/path/to/lalsuite-waveform-data/waveform_data
```

See `CHANGES.md` in each repo for what was modified from upstream.

## Stage 1: Bank Generation

### Intrinsic sample generation

First, generate the intrinsic parameter samples that define the bank. These are
shared across all approximants (same random seed → same parameters, different
waveforms).

```bash
python nrsur/scripts/gen_f220_samples.py \
    --output-dir /data/banks/my_campaign/samples \
    --n-samples 4194304 \
    --seed 42 \
    --n-workers 16
```

This produces `intrinsic_sample_bank.feather` (~300 MB for 4M samples) containing
`(m1, m2, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z, iota)` for each template.

### Waveform computation (per approximant)

Each approximant needs its own waveform bank. The bank is split into blocks of
4096 templates, each producing `amplitudes_block_N.npy` and `phase_block_N.npy`.

**Local (single machine):**
```bash
python nrsur/scripts/gen_bank_block.py \
    --bank-dir /data/banks/f220_nrsur7dq4 \
    --blocks 0-15 \
    --approximant NRSur7dq4
```

**SLURM cluster (recommended for NRSur):**
```bash
bash nrsur/scripts/launch_bank_slurm.sh \
    /data/banks/f220_nrsur7dq4 \
    NRSur7dq4
```

The launcher auto-detects missing blocks and submits a SLURM array job.
For 4M templates (1024 blocks), NRSur takes ~9 hours on a 64-node cluster.
XPHM takes minutes.

**NRSur threading caveat:** NRSur7dq4 spawns 5 threads per process internally
(via LAL/FFTW), ignoring `OMP_NUM_THREADS`. The SLURM scripts account for this
by requesting 5 CPUs per task. On a shared machine, limit concurrent NRSur
processes to `floor(physical_cores × 0.70 / 5)`.

### Bank validation

```bash
python nrsur/scripts/check_bank.py /data/banks/f220_nrsur7dq4
```

Checks all blocks are present and have the correct shapes.

## Stage 2: Incoherent Scoring

### Step 2a: Setup

Prepare the run directory with event data, reference waveform (`par_dic_0`),
and pre-computed summary weights for relative binning.

```bash
python nrsur/scripts/setup_incoherent.py \
    --run-dir /data/incoherent/real_into_nrsur \
    --event-pkl /data/event_data/real_event.pkl \
    --bank-dir /data/banks/f220_nrsur7dq4
```

This creates `setup/` with `event_data.pkl`, `par_dic_0.json`, `bank_config.json`,
and per-detector summary files. Takes ~1-5 minutes.

### Step 2b: Score all blocks

Each block of 4096 templates is scored independently (embarrassingly parallel).

**SLURM cluster:**
```bash
bash nrsur/scripts/launch_incoherent_slurm.sh \
    /data/incoherent/real_into_nrsur \
    /data/banks/f220_nrsur7dq4 \
    --n-t 320
```

`--n-t 320` sets the time search grid to ±50 ms (default 128 = ±20 ms). Use
the wider grid for NRSur as a safety margin.

Each block takes ~60 seconds regardless of approximant (the scoring uses
pre-computed bank waveforms, not live NRSur calls). A 1024-block bank
completes in ~20 minutes on a 64-node cluster.

### Step 2c: Merge and select survivors

```bash
python nrsur/scripts/merge_incoherent.py \
    /data/incoherent/real_into_nrsur \
    --max-drop 20
```

Concatenates all block scores, applies threshold `max_lnL - 20`, saves
`survivors.npz` and `summary.json`. Typical survivor counts: 20k-500k
depending on the event and approximant.

## Stage 3: Coherent Inference

```bash
python nrsur/scripts/run_inference_1m.py \
    --run-dir /data/incoherent/real_into_nrsur
```

Or via SLURM:
```bash
bash nrsur/scripts/launch_inference_slurm.sh
```

This runs `inference.run()` which executes 6 phases:
1. Preparation (find reference waveform using XPHM, ~10 s)
2. Incoherent scoring (skipped — pre-computed in Stage 2)
3. Survivor selection (instant)
4. **Extrinsic sampling** (the bottleneck — see Optimizations below)
5. Coherent block processing (disk I/O dominated, scales with N_survivors)
6. Aggregation (post-processing, save posteriors)

## Optimizations

### The bottleneck: Phase 4 (extrinsic sampling)

Phase 4 calls `get_marg_info_multibank` which processes survivors in batches
of 1024. For each batch:

1. **lnlike filter**: Evaluates `lnlike()` for each candidate, which generates
   a fresh waveform. NRSur: ~1 s/call × 1024 = ~17 min. XPHM: ~1 ms/call.
2. **Waveform loading**: Loads pre-computed amp/phase from disk.
3. **dh/hh computation**: Batch inner products.
4. **QMC marginalization**: Adaptive quasi-Monte Carlo for each passing candidate.

Only 16 `MarginalizationInfo` objects are needed, but the original code evaluates
all 1024 candidates through the filter and QMC steps.

### Optimization 1: lnlike filter early-stopping

**File:** `dot_pe/coherent_processing.py`, `get_marg_info_batch_multibank`

The batch is already shuffled, so truncating is unbiased. The filter loop now
exits early after collecting `max(2 * n_remaining, 32)` valid candidates instead
of evaluating all 1024.

**Result:** Phase 4 speedup 2.58x (267.7 s → 104.0 s at N=200 survivors).

### Optimization 2: Skip lnlike filter when threshold ≤ 0

**File:** `dot_pe/coherent_processing.py`, `get_marg_info_batch_multibank`

When `min_marg_lnlike_for_sampling ≤ 0` (the default), the filter accepts
virtually all candidates — every per-candidate NRSur waveform call is wasted.
The optimized code skips the filter entirely and passes candidates directly to
the waveform-loading → dh/hh → MI path.

**Result:** Phase 4 speedup 3.33x (267.7 s → 80.5 s). Total inference 2.36x.

### Optimization that failed: vectorized sparse matmul

Attempted to batch the `_set_d_h_weights` sparse operations (128 time bins ×
27 sparse-vector products). The batched sparse-dense matrix multiply caused
cache thrashing (memory 2.1 → 5.5 GB) and was 35% slower. **Reverted.**

The per-vector sparse matmul in cogwheel is already cache-optimal for this
problem size.

### Production-scale timings

Measured on a 32-core Intel machine (Carme), uncontested:

| Configuration | Phase 4 | Phase 5 | Total |
|---------------|---------|---------|-------|
| NRSur, N=200, **before** optimization | 259 s | 31 s | 321 s |
| NRSur, N=200, **after** optimization | 80 s | 27 s | 139 s |
| NRSur, N=32k (full scale, estimated) | ~22 min | ~18 min | ~45 min |
| XPHM, N=44k (full scale, measured) | ~31 min | ~26 min | ~58 min |

Phase 4 cost is O(batch_size=1024), NOT O(N_survivors) — it plateaus once
the first batch of 1024 yields ≥16 valid objects. Phase 5 scales linearly
with N_survivors (disk I/O dominated).

### NRSur vs XPHM cost breakdown

| | NRSur (9 modes) | XPHM (4 modes) | Ratio |
|---|---|---|---|
| Waveform generation | ~1 s | ~1 ms | ~1000x |
| dh terms (linear in modes) | 9 | 4 | 2.25x |
| hh pairs (quadratic in modes) | 45 | 10 | 4.5x |
| Total inner product terms | 54 | 14 | 3.86x |
| Observed wall-time ratio | | | ~6x |

The 6x ratio (higher than the 3.86x term count) includes I/O overhead from
loading 9 vs 4 mode arrays and memory bandwidth effects.

### Profiling tools

```bash
# Profile a single inference run with cProfile + phase timing
python nrsur/scripts/profile_inference.py aligned_nrsur_into_nrsur --n-survivors 200

# Scaling study across multiple survivor counts
python nrsur/scripts/scaling_study.py aligned_nrsur_into_nrsur 200 1000 5000

# Optimization benchmark against baseline
python nrsur/scripts/run_baseline.py          # Generate baseline (once)
python nrsur/scripts/optim_benchmark.py       # Benchmark current code
python nrsur/scripts/optim_correctness.py     # Verify correctness
```

### Remaining optimization targets (not yet attempted)

1. **QMC short-circuit**: The QMC loop evaluates all ~1024 passing candidates
   but only 16 are kept. Exiting after 16 accepted could save ~90% of QMC time.
2. **Disk pre-loading to RAM** for Phase 5 on slow NFS filesystems.
3. **`combine_prob_samples_with_next_block`** sorting (460 s at N=44k) — the
   two-pointer merge could potentially use a heap-based approach.

## Relative Binning Considerations

The relative binning approximation can occasionally fail for specific NRSur
templates whose waveform shape differs significantly from the reference waveform.
In our GW231123 campaign, 1 out of 4M templates had a catastrophic relbin error
(105.7 lnL in one detector).

**Recommendation:** Run full-frequency spot-checks on the top-scoring templates
after incoherent scoring. The script `validate_relbin_top.py` in the project
repo does this via FFT-based matched filtering.

The `relative_binning.py` modification in cogwheel (99% → 99.9% cumulative SNR
threshold) reduces relbin errors from ~4.4 to ~0.4 lnL for typical templates.

## File Layout

```
bank_dir/
├── bank_config.json              # Approximant, f_ref, fbin frequencies
├── intrinsic_sample_bank.feather # (N, 9) intrinsic parameters
└── waveforms/
    ├── amplitudes_block_0.npy    # (4096, n_modes, 2, n_fbin) float32
    ├── phase_block_0.npy         # same shape
    ├── amplitudes_block_1.npy
    ├── phase_block_1.npy
    └── ...

run_dir/                          # One per (event, approximant) combination
├── setup/
│   ├── event_data.pkl            # Detector strain data
│   ├── par_dic_0.json            # Reference waveform parameters
│   ├── bank_config.json          # Copy from bank
│   └── summary_*.npz             # Per-detector relative binning weights
├── scores/
│   ├── block_0.npz               # Per-block incoherent scores
│   └── ...
├── survivors.npz                 # Merged: survivor_inds, survivor_lnlike_di, ...
├── summary.json                  # Run statistics
└── results/                      # Posterior samples (after inference)
    └── samples.feather
```

## Scripts in this directory

See `scripts/` for the full set. The core workflow scripts are:

| Script | Stage | Purpose |
|--------|-------|---------|
| `gen_f220_samples.py` | 0 | Generate intrinsic parameter samples |
| `gen_bank_block.py` | 1 | Compute waveforms for one block |
| `launch_bank_slurm.sh` | 1 | Submit SLURM array for bank generation |
| `slurm_bank_block.sh` | 1 | SLURM per-task wrapper |
| `check_bank.py` | 1 | Validate completed bank |
| `setup_incoherent.py` | 2a | Prepare run directory |
| `incoherent_block.py` | 2b | Score one block |
| `launch_incoherent_slurm.sh` | 2b | Submit SLURM array for scoring |
| `slurm_incoherent_block.sh` | 2b | SLURM per-task wrapper |
| `merge_incoherent.py` | 2c | Merge scores, select survivors |
| `run_inference_1m.py` | 3 | Run coherent inference |
| `launch_inference_slurm.sh` | 3 | Submit SLURM inference jobs |
| `slurm_inference.sh` | 3 | SLURM per-task wrapper |
| `profile_inference.py` | — | Profile inference phases |
| `scaling_study.py` | — | Scaling study across survivor counts |
| `run_baseline.py` | — | Generate optimization baseline |
| `optim_benchmark.py` | — | Benchmark against baseline |
| `optim_correctness.py` | — | Correctness validation |
