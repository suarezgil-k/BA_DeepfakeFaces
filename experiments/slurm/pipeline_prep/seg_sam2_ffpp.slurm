#!/bin/bash
#SBATCH --job-name=sam2_ffpp
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/experiments/logs/sam2_ffpp_%j.out

echo "Time: $(date)"

module load devel/python/3.11.7-gnu-14.2
source /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/dcbm_env_py311_cu128/bin/activate

cd /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/Segments/scripts || exit 1
export PYTHONPATH=/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox

echo "Python path:"
which python
python --version

python create_segments_SAM2_ba.py --dataset FFpp_PipelineTest --device cuda:0 --check_existing False

