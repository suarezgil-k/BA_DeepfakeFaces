#!/bin/bash
#SBATCH --job-name=dcbm_c100
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=dcbm_cifar100_%j.out

cd /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test
source dcbm_env/bin/activate
cd experiments

python dcbm_testing.py --dataset cifar10 --model_name CLIP-ViT-L14
