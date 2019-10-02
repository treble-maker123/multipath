#!/bin/bash
#
#SBATCH --job-name=multipath-fb15k
#SBATCH -e outputs/errors/%j.txt
#SBATCH --output=outputs/logs/%j.txt
#SBATCH --partition=m40-long
#SBATCH --ntasks=8
#SBATCH --time=07-00:00
#SBATCH --mem=45000
#SBATCH --gres=gpu:1

# For debugging device-side assert errors
# export CUDA_LAUNCH_BLOCKING=1

# To make a boolean option False, simply prefix with "no-"
export cmd="python3 main.py \
--run-id=$SLURM_JOB_ID \
--log-level 20 \
--no-log-to-file \
--log-to-stdout \
--no-write-tensorboard \
--save-model \
--save-result \
--use-gpu \
--engine=multipath-link-predict \
--dataset-path=data/FB15K-237 \
--train-batch-size=1 \
--test-batch-size=1 \
--max-traversal-hops 3"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
