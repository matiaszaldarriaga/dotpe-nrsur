#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# SLURM launcher for distributed incoherent scoring
#
# Submits one array job where each task scores one bank block.
# Only submits blocks whose output files are missing.
#
# Usage:
#   bash scripts/launch_incoherent_slurm.sh RUN_DIR BANK_DIR [OPTIONS]
#
# Examples:
#   bash scripts/launch_incoherent_slurm.sh /data/matiasz/dot-pe/incoherent_1m/run1 /data/matiasz/dot-pe/bank_1m_xphm
#   bash scripts/launch_incoherent_slurm.sh /data/matiasz/dot-pe/incoherent_1m/run1 /data/matiasz/dot-pe/bank_1m_nrsur --dry-run
#
# Options:
#   --dry-run       Print sbatch command without submitting
#   --time HH:MM:SS Override time limit (auto: NRSur=02:00:00, XPHM=00:30:00)
#   --n-t N         Time grid size for scoring (default: 128, use 320 for ±50ms)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 RUN_DIR BANK_DIR [--dry-run] [--time HH:MM:SS]"
    exit 1
fi

RUN_DIR="$1"
BANK_DIR="$2"
shift 2

DRY_RUN=false
TIME_LIMIT=""
N_T=128

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --n-t) N_T="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Validate ──
SETUP_DIR="${RUN_DIR}/setup"
SCORES_DIR="${RUN_DIR}/scores"
CONFIG_FILE="${SETUP_DIR}/bank_config.json"
SLURM_SCRIPT="/home/matiasz/claude-projects/dot-pe/scripts/slurm_incoherent_block.sh"

for f in "$CONFIG_FILE" "${SETUP_DIR}/event_data.pkl" "${SETUP_DIR}/par_dic_0.json" "$SLURM_SCRIPT"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f"
        echo "Did you run setup_incoherent.py first?"
        exit 1
    fi
done

# ── Read config ──
BANK_SIZE=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['bank_size'])")
BLOCKSIZE=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['blocksize'])")
APPROXIMANT=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['approximant'])")
N_BLOCKS=$((BANK_SIZE / BLOCKSIZE))

echo "Run:         ${RUN_DIR}"
echo "Bank:        ${BANK_DIR}"
echo "Approximant: ${APPROXIMANT}"
echo "Bank size:   ${BANK_SIZE}, blocks: ${N_BLOCKS}"

# ── Detect missing blocks ──
MISSING=()
for ((i=0; i<N_BLOCKS; i++)); do
    SCORE_FILE="${SCORES_DIR}/block_${i}.npz"
    if [[ ! -f "$SCORE_FILE" ]]; then
        MISSING+=("$i")
    fi
done

if [[ ${#MISSING[@]} -eq 0 ]]; then
    echo "All ${N_BLOCKS} blocks scored. Nothing to submit."
    exit 0
fi
echo "Missing blocks: ${#MISSING[@]}/${N_BLOCKS}"

# ── Compress array spec into ranges ──
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

MAX_ARRAY_IDX=1000  # Typhon MaxArraySize=1001

# ── Auto settings by approximant ──
if [[ -z "$TIME_LIMIT" ]]; then
    case "$APPROXIMANT" in
        NRSur7dq4)     TIME_LIMIT="02:00:00" ;;
        IMRPhenomXPHM) TIME_LIMIT="00:30:00" ;;
        *)             TIME_LIMIT="02:00:00" ;;
    esac
fi

# NRSur spawns 5 internal threads; XPHM is single-threaded
case "$APPROXIMANT" in
    NRSur7dq4)     CPUS_PER_TASK=5 ;;
    *)             CPUS_PER_TASK=1 ;;
esac

echo "Time limit:    ${TIME_LIMIT}"
echo "CPUs/task:     ${CPUS_PER_TASK}"
echo "N_T:           ${N_T}"

# ── SLURM log directory ──
SLURM_LOG_DIR="${RUN_DIR}/slurm_logs"
mkdir -p "$SLURM_LOG_DIR"

JOB_NAME="incoh_$(basename "$RUN_DIR")"

# Split into blocks <= MAX_ARRAY_IDX and blocks > MAX_ARRAY_IDX
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

submit_batch() {
    local offset=$1; shift
    local -a ids=("$@")
    local n=${#ids[@]}
    BATCH_NUM=$((BATCH_NUM + 1))

    local -a array_ids=()
    for bid in "${ids[@]}"; do
        array_ids+=("$((bid - offset))")
    done
    local array_spec
    array_spec=$(compress_array_spec "${array_ids[@]}")

    echo ""
    echo "Batch ${BATCH_NUM}/${TOTAL_BATCHES}: ${n} tasks (offset=${offset}, array=${array_spec:0:60}...)"

    local export_vars="RUN_DIR=${RUN_DIR},BANK_DIR=${BANK_DIR},N_PHI=50,N_T=${N_T}"
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
    MIN_HIGH=${HIGH_IDS[0]}
    for bid in "${HIGH_IDS[@]}"; do
        [[ $bid -lt $MIN_HIGH ]] && MIN_HIGH=$bid
    done
    submit_batch "$MIN_HIGH" "${HIGH_IDS[@]}"
fi

if $DRY_RUN; then
    echo ""
    echo "=== Would submit ${#MISSING[@]} tasks in ${TOTAL_BATCHES} batch(es) ==="
    exit 0
fi

echo ""
echo "Monitor with: squeue -u \$USER -n ${JOB_NAME}"
echo "Cancel with:  scancel -n ${JOB_NAME}"
echo "After completion, run: python scripts/merge_incoherent.py --run-dir ${RUN_DIR}"
