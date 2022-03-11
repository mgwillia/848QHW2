#!/bin/bash

#SBATCH --job-name=matt_job_predict
#SBATCH --output=logs/matt_job_predict.out.%j
#SBATCH --error=logs/matt_job_predict.out.%j
#SBATCH --time=36:00:00
#SBATCH --account=abhinav
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load cuda/11.0.3

srun bash -c "hostname;"

#srun bash -c "python models.py --train_guesser;"
srun bash -c "python run_e2e_eval.py --mode predict --eval_dataset data/qanta.hw2.test.json;"
