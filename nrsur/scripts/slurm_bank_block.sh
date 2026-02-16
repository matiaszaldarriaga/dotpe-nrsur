#!/bin/bash
#SBATCH --job-name=bank_block
#SBATCH --output=%x_%a_%j.out
#SBATCH --error=%x_%a_%j.err
#SBATCH --ntasks=1
#SBATCH --mem=8G
# NOTE: cpus-per-task is set dynamically by launch_bank_slurm.sh
# NRSur7dq4 spawns 5 threads internally (LAL/FFTW), needs --cpus-per-task=5
# XPHM is single-threaded, uses --cpus-per-task=1
# ═══════════════════════════════════════════════════════════════
# SLURM per-task script: generates one waveform bank block.
#
# SLURM_ARRAY_TASK_ID = block index (direct 1:1 mapping).
# Submitted by launch_bank_slurm.sh — do not run directly.
#
# Environment variables (set by launch_bank_slurm.sh):
#   BANK_DIR      - absolute path to bank directory
#   APPROXIMANT   - waveform approximant name
#   BLOCKSIZE     - samples per block
#   BLOCK_OFFSET  - (optional) offset added to SLURM_ARRAY_TASK_ID to get real block ID
# ═══════════════════════════════════════════════════════════════

set -e

BLOCK_ID=$(( SLURM_ARRAY_TASK_ID + ${BLOCK_OFFSET:-0} ))

echo "=== SLURM Bank Block ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "Block ID:  $BLOCK_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Date:      $(date -Iseconds)"
echo "Bank dir:  $BANK_DIR"
echo "Approx:    $APPROXIMANT"

# CRITICAL: Thread isolation (SLURM allocates CPUs but doesn't limit threads)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# CRITICAL: Absolute paths (SLURM copies scripts to /var/spool/slurmd/)
GEN_SCRIPT="/home/matiasz/claude-projects/dot-pe/scripts/gen_bank_block.py"

# Conda activation
source /home/matiasz/anaconda3/etc/profile.d/conda.sh
conda activate pycbc
export LAL_DATA_PATH=/home/matiasz/GW-2025/SPIN-HEAVY/lalsuite-waveform-data/waveform_data

echo "Python: $(which python)"
echo "Conda:  $CONDA_DEFAULT_ENV"
echo "=== Starting generation ==="

python -u "$GEN_SCRIPT" \
    --bank-dir "$BANK_DIR" \
    --blocks "$BLOCK_ID" \
    --approximant "$APPROXIMANT" \
    --blocksize "${BLOCKSIZE:-4096}"

EXIT_CODE=$?
echo "=== Done (exit code: $EXIT_CODE) ==="
exit $EXIT_CODE
