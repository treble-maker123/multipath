#!/bin/bash
#
#SBATCH --job-name=path-trans-lp-fb15k
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
--engine=path-transform-link-predict \
--dataset-path=data/FB15K-237 \
--data-size 10 \
--num-epochs=2 \
--num-workers=0 \
--train-batch-size=1 \
--validate-interval=2 \
--run-train-during-validate \
--test-batch-size=1 \
--max-traversal-hops 2"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
