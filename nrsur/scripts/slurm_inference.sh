#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G

set -e

echo "=== SLURM Inference ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Date:      $(date -Iseconds)"
echo "Run dir:   $RUN_DIR"

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

SCRIPT="/home/matiasz/claude-projects/dot-pe/scripts/run_inference_1m.py"

echo "Python: $(which python)"
echo "=== Starting inference ==="

python -u "$SCRIPT" --run-dir "$RUN_DIR"

EXIT_CODE=$?
echo "=== Done (exit code: $EXIT_CODE) ==="
exit $EXIT_CODE
