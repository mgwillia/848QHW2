#!/bin/bash

#SBATCH --job-name=matt_job_tfidf
#SBATCH --output=logs/matt_job_tfidf.out.%j
#SBATCH --error=logs/matt_job_tfidf.out.%j
#SBATCH --time=36:00:00
#SBATCH --nodelist=brigid09
#SBATCH --account=abhinav
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load cuda/11.0.3

srun bash -c "hostname;"
srun bash -c "python tfidf_guesser.py;"
