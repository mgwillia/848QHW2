#!/bin/bash

#SBATCH --job-name=matt_job_bench_reranker_firstsent
#SBATCH --output=logs/matt_job_bench_reranker_firstsent.out.%j
#SBATCH --error=logs/matt_job_bench_reranker_firstsent.out.%j
#SBATCH --time=36:00:00
#SBATCH --account=abhinav
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load cuda/11.0.3

EPOCH_NUMS=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")

srun bash -c "hostname;"
for epoch_num in ${EPOCH_NUMS[@]}; do
    srun bash -c "python analyze_reranker.py --epoch_num $epoch_num --type first_sent;"
done
