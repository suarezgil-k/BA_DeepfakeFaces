#!/bin/bash
#SBATCH --job-name=dcbm_c100
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/dcbm_cifar100_%j.out

cd /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox

source /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/dcbm_env/bin/activate

#Last job failed due to clip error, trying to debug
echo "Python path:"
which python
python -c "import clip; print('clip works')"

cd experiments

python dcbm_testing.py --dataset cifar100 --model_name CLIP-ViT-L14
