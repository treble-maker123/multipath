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
export cmd="python main.py \
--run-id=$SLURM_JOB_ID \
--no-interactive \
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
--negative-sample-factor=1 \
--num-epochs=100 \
--num-workers=8 \
--train-batch-size=256 \
--test-batch-size=1 \
--hidden-dim=300 \
--learn-rate=0.0001 \
--num-transformer-layers=5 \
--num-attention-heads=1 \
--validate-interval=2 \
--max-paths=50 \
--bucket-size=32000 \
--no-run-train-during-validate"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
