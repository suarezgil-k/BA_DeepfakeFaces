#!/bin/bash
#SBATCH --job-name=cdf_imgemb_l14
# #SBATCH --partition=cpu
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/experiments/logs/cdf_l14_%j.out

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
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

python emb_CLIP_ba_test.py --dataset CelebDF --emb ViT-L/14 --device cuda:0
