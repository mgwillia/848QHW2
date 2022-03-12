#!/bin/bash

#SBATCH --job-name=matt_job_bench_reranker_default
#SBATCH --output=logs/matt_job_bench_reranker_default.out.%j
#SBATCH --error=logs/matt_job_bench_reranker_default.out.%j
#SBATCH --time=36:00:00
#SBATCH --account=abhinav
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load cuda/11.0.3

srun bash -c "python analyze_reranker.py --type default;"
