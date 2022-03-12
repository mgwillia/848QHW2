#!/bin/bash

#SBATCH --job-name=matt_job_bench_guesser_9_splitter
#SBATCH --output=logs/matt_job_bench_guesser_9_splitter.out.%j
#SBATCH --error=logs/matt_job_bench_guesser_9_splitter.out.%j
#SBATCH --time=36:00:00
#SBATCH --account=abhinav
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load cuda/11.0.3

srun bash -c "hostname;"
srun bash -c "python analyze_guesser.py --epoch_num 9 --sentence_splitter;"
