#!/bin/bash
#SBATCH --job-name=ffpp_c23_dl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/experiments/logs/ffpp_c23_downl_%j.out

echo "Time: $(date)"


module load devel/python/3.11.7-gnu-14.2

source /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/dcbm_env_py311/bin/activate
cd /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/FaceForensics/scripts || exit 1

printf '\n' | python faceforensics_download_v4.py \
    ../c23 \
    --dataset original \
    --compression c23 \
    --type videos \
    --server EU2

printf '\n' | python faceforensics_download_v4.py \
    ../c23 \
    --dataset Deepfakes \
    --compression c23 \
    --type videos \
    --server EU2
