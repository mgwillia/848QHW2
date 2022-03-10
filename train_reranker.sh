#!/bin/bash

#SBATCH --job-name=matt_job_rerank
#SBATCH --output=logs/matt_job_rerank.out.%j
#SBATCH --error=logs/matt_job_rerank.out.%j
#SBATCH --time=72:00:00
#SBATCH --account=abhinav
#SBATCH --partition=dpart
#SBATCH --qos=medium
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load cuda/11.0.3

srun bash -c "hostname;"
srun bash -c "python finetune_reranker.py;"
