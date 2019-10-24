#!/bin/bash
#
#SBATCH --job-name=path-trans-lp-fb15k
#SBATCH -e outputs/errors/%j.txt
#SBATCH --output=outputs/logs/%j.txt
#SBATCH --partition=m40-long
#SBATCH --ntasks=6
#SBATCH --time=07-00:00
#SBATCH --mem=45000
#SBATCH --gres=gpu:1

# For debugging device-side assert errors
 export CUDA_LAUNCH_BLOCKING=1

# To make a boolean option False, simply prefix with "no-"
export cmd="python3 main.py \
--run-id=$SLURM_JOB_ID \
--log-level 20 \
--no-log-to-file \
--log-to-stdout \
--write-tensorboard \
--save-model \
--save-result \
--use-gpu \
--engine=path-transform-link-predict \
--dataset-path=data/nell-995 \
--data-size=-1 \
--num-epochs=200 \
--num-workers=12 \
--train-batch-size=256 \
--test-batch-size=1 \
--hidden-dim=180 \
--num-transformer-layers=2 \
--num-attention-heads=6 \
--validate-interval=20 \
--max-paths=1000 \
--bucket-size=28000 "

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
