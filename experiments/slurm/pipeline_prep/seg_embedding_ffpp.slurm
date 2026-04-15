#!/bin/bash
#SBATCH --job-name=embseg_ffpp
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/experiments/logs/embseg_ffpp_%j.out

echo "Time: $(date)"

module load devel/python/3.11.7-gnu-14.2
source /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/dcbm_env_py311_cu128/bin/activate

cd /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/Segments/scripts || exit 1
export PYTHONPATH=$PWD

echo "Python path:"
which python
python --version

python -c "import torch; print('torch works')"
python -c "import clip; print('clip works')"

python emb_segments_CLIP_ba.py --emb CLIP-ViT-L14 --dataset FFpp_PipelineTest --seg_model SAM2 --prompts None --device gpu
