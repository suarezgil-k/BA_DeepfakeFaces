#!/bin/bash
#SBATCH --job-name=ffpp_imgemb_l14
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/experiments/logs/ffpp_imgemb_l14_%j.out

echo "Time: $(date)"

module load devel/python/3.11.7-gnu-14.2
source /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/dcbm_env_py311_cu128/bin/activate

cd /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/Datasets/scripts || exit 1
export PYTHONPATH=$PWD

echo "Python path:"
which python
python --version

python -c "import torch; print('torch works')"
python -c "import clip; print('clip works')"

python emb_CLIP_FFpp.py --dataset FFpp_PipelineTest --emb ViT-L/14 --device gpu
