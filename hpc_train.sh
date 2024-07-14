#!/bin/bash -l

#SBATCH --job-name=cpu_clustering_scioi21
#SBATCH --output=res_%j.txt     # output file
#SBATCH --error=res_%j.err      # error file
#SBATCH --ntasks=1
#SBATCH --time=0-34:00 
#SBATCH --mem-per-cpu=30000      # memory in MB per cpu allocated
#SBATCH --partition=ex_scioi_node # partition to submit to

# module load nvidia/cuda/10.0    # load required modules (depends upon your code which modules are needed)
# module load comp/gcc/7.2.0

source ~/micromamba/etc/profile.d/micromamba.sh

echo "deactivating base env"
micromamba deactivate
micromamba activate training_cpu_202407xx

echo "adding language encodings"
export LANG=UTF-8
export LC_ALL=en_US.UTF-8
export LD_LIBRARY_PATH=/home/users/b/beese/micromamba/envs/training_cpu_202407xx/lib/libtbb.so 

echo "JOB START"
wandb login 76c180bfb40d010967579bb430f98fcc8ef51042
python train.py --n_neighbors 15 --min_dist 0.1 --device -1 --data "/scratch/beese/proj_21/FE_tracks/FE_tracks_from2023" --threads_cpu -1

echo "JOB DONE"

echo "deactivating environments"
conda deactivate 

