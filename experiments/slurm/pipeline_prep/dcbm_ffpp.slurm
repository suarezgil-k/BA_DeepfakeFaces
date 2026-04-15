#!/bin/bash
#SBATCH --job-name=dcbm_ffpp
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/experiments/logs/dcbm_ffpp_%j.out

echo "Time: $(date)"

module load devel/python/3.11.7-gnu-14.2
source /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/dcbm_env_py311_cu128/bin/activate

cd /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/experiments || exit 1
export PYTHONPATH=/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox

echo "Python path:"
which python
python --version

python -c "import torch; print('torch works')"
python -c "from utils.dcbm_ba import *; print('dcbm_ba works')"

python dcbm_testing_ffpp.py
