#!/bin/bash
#SBATCH --job-name=subset_ffpp_c23
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/experiments/logs/subset_ffpp_c23_%j.out

echo "Time: $(date)"

module load devel/python/3.11.7-gnu-14.2
source /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/dcbm_env_py311_cu128/bin/activate

cd /pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/Embeddings/scripts || exit 1

echo "Creating subset directory..."
mkdir -p ../subsets/FFpp_c23

echo "Running subset creation..."
python create_subsets_ba.py --dataset FFpp_c23

echo "Done."
