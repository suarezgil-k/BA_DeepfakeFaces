#!/bin/bash
#SBATCH --job-name=dcbm_c100
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/experiments/logs/dcbm_cifar100_%j.out

echo "Time: $(date)"

#added module load because I changed from python 3.9 zu python 3.11
module load devel/python/3.11.7-gnu-14.2
source /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/dcbm_env_py311/bin/activate

echo "Python path:"
which python
python --version

#Import checks - checking previous errors
python -c "import clip; print('clip works')"
python -c "import pandas; print('pandas works')"
python -c "import mmcv; print('mmcv works')"

python /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/experiments/dcbm_testing.py --dataset cifar100 --model_name CLIP-ViT-L14
