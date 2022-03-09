#!/bin/bash

#SBATCH --job-name=matt_job
#SBATCH --output=logs/matt_job.out.%j
#SBATCH --error=logs/matt_job.out.%j
#SBATCH --time=36:00:00
#SBATCH --account=abhinav
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load cuda/11.0.3

srun bash -c "hostname;"
srun bash -c "python models.py \
    --train_data data/qanta.train.2018.json \
    --dev_data data/qanta.dev.2018.json;"
