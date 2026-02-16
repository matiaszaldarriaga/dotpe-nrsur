#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# SLURM launcher for coherent inference on all 8 incoherent 1M runs
#
# Submits one job per run. Skips runs that already have results.
#
# Usage:
#   bash scripts/launch_inference_slurm.sh [OPTIONS]
#
# Options:
#   --dry-run       Print sbatch commands without submitting
#   --only RUN_NAME Only submit this specific run
#   --time HH:MM:SS Override time limit (default: 03:00:00)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

OUTPUT_BASE="/data/matiasz/dot-pe/incoherent_1m"
SLURM_SCRIPT="/home/matiasz/claude-projects/dot-pe/scripts/slurm_inference.sh"

DRY_RUN=false
ONLY=""
TIME_LIMIT="03:00:00"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate SLURM script exists
if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "ERROR: SLURM script not found: $SLURM_SCRIPT"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Coherent inference: 1M bank runs on Typhon"
echo "═══════════════════════════════════════════════════════════"
echo "Time limit: ${TIME_LIMIT}"
echo ""

# All 8 runs
RUNS=(
    aligned_nrsur_into_nrsur
    aligned_nrsur_into_xphm
    aligned_xphm_into_nrsur
    aligned_xphm_into_xphm
    precessing_nrsur_into_nrsur
    precessing_nrsur_into_xphm
    precessing_xphm_into_nrsur
    precessing_xphm_into_xphm
)

SUBMITTED=0
SKIPPED=0

for run_name in "${RUNS[@]}"; do
    # Filter by --only if specified
    if [[ -n "$ONLY" && "$run_name" != "$ONLY" ]]; then
        continue
    fi

    run_dir="${OUTPUT_BASE}/${run_name}"

    # Validate prerequisites
    if [[ ! -f "${run_dir}/setup/run_meta.json" ]]; then
        echo "  SKIP ${run_name}: no setup found"
        ((SKIPPED++)) || true
        continue
    fi

    if [[ ! -f "${run_dir}/inds_for_dotpe.npz" ]]; then
        echo "  SKIP ${run_name}: no inds_for_dotpe.npz"
        ((SKIPPED++)) || true
        continue
    fi

    # Check if already done
    if find "${run_dir}/inference" -name "summary_results.json" 2>/dev/null | grep -q .; then
        echo "  DONE ${run_name}: summary_results.json exists"
        ((SKIPPED++)) || true
        continue
    fi

    # SLURM log directory
    slurm_log_dir="${run_dir}/slurm_logs"
    mkdir -p "$slurm_log_dir"

    JOB_NAME="inf_${run_name}"

    SBATCH_CMD="sbatch \
        --job-name=${JOB_NAME} \
        --time=${TIME_LIMIT} \
        --output=${slurm_log_dir}/inference_%j.out \
        --error=${slurm_log_dir}/inference_%j.err \
        --export=RUN_DIR=${run_dir} \
        ${SLURM_SCRIPT}"

    if $DRY_RUN; then
        echo "  [DRY RUN] ${run_name}:"
        echo "    $SBATCH_CMD"
    else
        echo -n "  Submitting ${run_name}... "
        output=$(eval "$SBATCH_CMD" 2>&1)
        job_id=$(echo "$output" | grep "Submitted batch job" | awk '{print $4}')
        if [[ -n "$job_id" ]]; then
            echo "Job ${job_id}"
        else
            echo "WARNING: could not parse job ID:"
            echo "    $output"
        fi
    fi
    ((SUBMITTED++)) || true
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Submitted: ${SUBMITTED}, Skipped: ${SKIPPED}"
echo "═══════════════════════════════════════════════════════════"

if [[ $SUBMITTED -gt 0 ]] && ! $DRY_RUN; then
    echo ""
    echo "Monitor:  squeue -u \$USER | grep inf_"
    echo "Cancel:   scancel -n inf_<run_name>"
fi
