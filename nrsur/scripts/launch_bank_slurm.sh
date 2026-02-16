#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# SLURM job array launcher for Typhon cluster
#
# Submits one SLURM array job where each task generates one
# waveform bank block. Only submits missing blocks.
#
# Usage:
#   ssh typhon-login1 'bash /home/matiasz/claude-projects/dot-pe/scripts/launch_bank_slurm.sh BANK_DIR APPROXIMANT [OPTIONS]'
#
# Examples:
#   bash scripts/launch_bank_slurm.sh /data/matiasz/dot-pe/comparison_65k/bank_nrsur NRSur7dq4
#   bash scripts/launch_bank_slurm.sh /data/matiasz/dot-pe/comparison_65k/bank_xphm IMRPhenomXPHM --time 01:00:00
#   bash scripts/launch_bank_slurm.sh /data/matiasz/dot-pe/comparison_65k/bank_nrsur NRSur7dq4 --dry-run
#
# Options:
#   --dry-run       Print sbatch command without submitting
#   --time HH:MM:SS Override time limit (auto: NRSur=06:00:00, XPHM=01:00:00)
#   --blocksize N   Override block size (default: 4096)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 BANK_DIR APPROXIMANT [--dry-run] [--time HH:MM:SS] [--blocksize N]"
    exit 1
fi

BANK_DIR="$1"
APPROXIMANT="$2"
shift 2

DRY_RUN=false
TIME_LIMIT=""
BLOCKSIZE=4096

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --blocksize) BLOCKSIZE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Validate ──
CONFIG_FILE="${BANK_DIR}/bank_config.json"
SAMPLES_FILE="${BANK_DIR}/intrinsic_sample_bank.feather"
# Absolute path to SLURM job script
SLURM_SCRIPT="/home/matiasz/claude-projects/dot-pe/scripts/slurm_bank_block.sh"

for f in "$CONFIG_FILE" "$SAMPLES_FILE" "$SLURM_SCRIPT"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done

WAVEFORM_DIR="${BANK_DIR}/waveforms"
mkdir -p "$WAVEFORM_DIR"

# ── Read bank size ──
BANK_SIZE=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['bank_size'])")
N_BLOCKS=$((BANK_SIZE / BLOCKSIZE))
echo "Bank: ${BANK_DIR}"
echo "Approximant: ${APPROXIMANT}"
echo "Total blocks: ${N_BLOCKS}"

# ── Detect missing blocks ──
MISSING=()
for ((i=0; i<N_BLOCKS; i++)); do
    AMP="${WAVEFORM_DIR}/amplitudes_block_${i}.npy"
    PHA="${WAVEFORM_DIR}/phase_block_${i}.npy"
    if [[ ! -f "$AMP" || ! -f "$PHA" ]]; then
        MISSING+=("$i")
    else
        AMP_SIZE=$(stat --printf="%s" "$AMP" 2>/dev/null || echo 0)
        PHA_SIZE=$(stat --printf="%s" "$PHA" 2>/dev/null || echo 0)
        if [[ "$AMP_SIZE" -lt 200 || "$PHA_SIZE" -lt 200 ]]; then
            MISSING+=("$i")
        fi
    fi
done

if [[ ${#MISSING[@]} -eq 0 ]]; then
    echo "All ${N_BLOCKS} blocks complete. Nothing to submit."
    exit 0
fi
echo "Missing blocks: ${#MISSING[@]}/${N_BLOCKS}"

# ── Build SLURM array specs (range syntax, max 1000 per batch) ──
# Try to compress consecutive IDs into ranges (e.g. "0-999,1001,1003-1023")
# Then split into chunks respecting MaxArraySize
compress_array_spec() {
    local -a ids=("$@")
    local -a ranges=()
    local start=${ids[0]}
    local end=${ids[0]}
    for ((i=1; i<${#ids[@]}; i++)); do
        if [[ ${ids[i]} -eq $((end + 1)) ]]; then
            end=${ids[i]}
        else
            if [[ $start -eq $end ]]; then
                ranges+=("$start")
            else
                ranges+=("${start}-${end}")
            fi
            start=${ids[i]}
            end=${ids[i]}
        fi
    done
    if [[ $start -eq $end ]]; then
        ranges+=("$start")
    else
        ranges+=("${start}-${end}")
    fi
    IFS=,; echo "${ranges[*]}"
}

MAX_ARRAY_IDX=1000  # Typhon MaxArraySize=1001, max index = 1000

N_MISSING=${#MISSING[@]}
echo "Missing blocks: ${N_MISSING}/${N_BLOCKS}"

# ── Auto time limit ──
if [[ -z "$TIME_LIMIT" ]]; then
    case "$APPROXIMANT" in
        NRSur7dq4)     TIME_LIMIT="06:00:00" ;;
        IMRPhenomXPHM) TIME_LIMIT="01:00:00" ;;
        *)             TIME_LIMIT="06:00:00" ;;
    esac
fi
echo "Time limit: ${TIME_LIMIT}"

# ── CPUs per task (NRSur spawns 5 threads internally via LAL/FFTW) ──
case "$APPROXIMANT" in
    NRSur7dq4)     CPUS_PER_TASK=5 ;;
    *)             CPUS_PER_TASK=1 ;;
esac
echo "CPUs per task: ${CPUS_PER_TASK}"

# ── SLURM log directory ──
SLURM_LOG_DIR="${BANK_DIR}/slurm_logs"
mkdir -p "$SLURM_LOG_DIR"

JOB_NAME="bank_$(basename "$BANK_DIR")"

# Split missing blocks into batches where all block IDs fit within [0, MAX_ARRAY_IDX].
# Blocks with ID > MAX_ARRAY_IDX use BLOCK_OFFSET to remap to low indices.
# Group: blocks <= MAX_ARRAY_IDX (batch 1, no offset), blocks > MAX_ARRAY_IDX (batch 2, with offset)
LOW_IDS=()
HIGH_IDS=()
for bid in "${MISSING[@]}"; do
    if [[ $bid -le $MAX_ARRAY_IDX ]]; then
        LOW_IDS+=("$bid")
    else
        HIGH_IDS+=("$bid")
    fi
done

BATCH_NUM=0
TOTAL_BATCHES=0
[[ ${#LOW_IDS[@]} -gt 0 ]] && TOTAL_BATCHES=$((TOTAL_BATCHES + 1))
[[ ${#HIGH_IDS[@]} -gt 0 ]] && TOTAL_BATCHES=$((TOTAL_BATCHES + 1))
echo "Batches: ${TOTAL_BATCHES}"

submit_batch() {
    local -a ids=("$@")
    local offset=$1; shift; ids=("$@")
    local n=${#ids[@]}
    BATCH_NUM=$((BATCH_NUM + 1))

    # Subtract offset from IDs to get array indices
    local -a array_ids=()
    for bid in "${ids[@]}"; do
        array_ids+=("$((bid - offset))")
    done
    local array_spec
    array_spec=$(compress_array_spec "${array_ids[@]}")

    echo ""
    echo "Batch ${BATCH_NUM}/${TOTAL_BATCHES}: ${n} tasks (offset=${offset}, array=${array_spec:0:60}...)"

    local export_vars="BANK_DIR=${BANK_DIR},APPROXIMANT=${APPROXIMANT},BLOCKSIZE=${BLOCKSIZE}"
    if [[ $offset -gt 0 ]]; then
        export_vars="${export_vars},BLOCK_OFFSET=${offset}"
    fi

    local SBATCH_CMD="sbatch \
        --job-name=${JOB_NAME} \
        --array=${array_spec} \
        --time=${TIME_LIMIT} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --output=${SLURM_LOG_DIR}/block_%a_%j.out \
        --error=${SLURM_LOG_DIR}/block_%a_%j.err \
        --export=${export_vars} \
        ${SLURM_SCRIPT}"

    if $DRY_RUN; then
        echo "=== DRY RUN ==="
        echo "$SBATCH_CMD"
    else
        eval "$SBATCH_CMD"
    fi
}

if [[ ${#LOW_IDS[@]} -gt 0 ]]; then
    submit_batch 0 "${LOW_IDS[@]}"
fi

if [[ ${#HIGH_IDS[@]} -gt 0 ]]; then
    # Use the minimum high ID as offset so array indices start near 0
    MIN_HIGH=${HIGH_IDS[0]}
    for bid in "${HIGH_IDS[@]}"; do
        [[ $bid -lt $MIN_HIGH ]] && MIN_HIGH=$bid
    done
    submit_batch "$MIN_HIGH" "${HIGH_IDS[@]}"
fi

if $DRY_RUN; then
    echo ""
    echo "=== Would submit ${N_MISSING} tasks in ${TOTAL_BATCHES} batch(es) ==="
    exit 0
fi

echo ""
echo "Monitor with: squeue -u \$USER -n ${JOB_NAME}"
echo "Cancel with:  scancel -n ${JOB_NAME}"
echo "Check progress: python3 /home/matiasz/claude-projects/dot-pe/scripts/check_bank.py ${BANK_DIR}"
