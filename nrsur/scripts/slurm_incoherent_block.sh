#!/bin/bash
#SBATCH --job-name=incoh_block
#SBATCH --ntasks=1
#SBATCH --mem=12G
# NOTE: cpus-per-task set by launch_incoherent_slurm.sh (1 for XPHM, 5 for NRSur)

set -e

BLOCK_ID=$(( SLURM_ARRAY_TASK_ID + ${BLOCK_OFFSET:-0} ))

echo "=== SLURM Incoherent Block ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "Block ID:  $BLOCK_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Date:      $(date -Iseconds)"
echo "Run dir:   $RUN_DIR"
echo "Bank dir:  $BANK_DIR"

# Thread isolation
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Conda activation
source /home/matiasz/anaconda3/etc/profile.d/conda.sh
conda activate pycbc
export LAL_DATA_PATH=/home/matiasz/GW-2025/SPIN-HEAVY/lalsuite-waveform-data/waveform_data

SCRIPT="/home/matiasz/claude-projects/dot-pe/scripts/incoherent_block.py"

echo "Python: $(which python)"
echo "=== Starting scoring ==="

python -u "$SCRIPT" \
    --run-dir "$RUN_DIR" \
    --bank-dir "$BANK_DIR" \
    --block-id "$BLOCK_ID" \
    --n-phi "${N_PHI:-50}" \
    --n-t "${N_T:-128}"

EXIT_CODE=$?
echo "=== Done (exit code: $EXIT_CODE) ==="
exit $EXIT_CODE
