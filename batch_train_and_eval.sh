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
srun bash -c "tfidf_guesser.py \
    --guesstrain data/qanta.train.2018.json \
    --guessdev data/qanta.dev.2018.json \
    --model_path models/tfidf.pickle;"

srun bash -c "python run_e2e_eval.py;"
