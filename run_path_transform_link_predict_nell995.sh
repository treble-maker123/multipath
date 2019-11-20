#!/bin/bash
#
#SBATCH --job-name=path-trans-nell-995
#SBATCH -e outputs/errors/%j.txt
#SBATCH --output=outputs/logs/%j.txt
#SBATCH --partition=titanx-long
#SBATCH --ntasks=6
#SBATCH --time=07-00:00
#SBATCH --mem=62GB
#SBATCH --gres=gpu:1

# For debugging device-side assert errors
# export CUDA_LAUNCH_BLOCKING=1

# To make a boolean option False, simply prefix with "no-"
export cmd="python main.py \
--run-id=$SLURM_JOB_ID \
--no-interactive \
--log-level 20 \
--no-log-to-file \
--log-to-stdout \
--no-write-tensorboard \
--no-save-model \
--no-save-result \
--use-gpu \
--engine=path-transform-link-predict \
--dataset-path=data/nell-995 \
--data-size=-1 \
--negative-sample-factor=1 \
--num-epochs=200 \
--num-workers=12 \
--train-batch-size=16 \
--test-batch-size=1 \
--hidden-dim=128 \
--learn-rate=0.0001 \
--lr-scheduler=multistep \
--lr-milestones=60,64,120 \
--lr-gamma=0.5 \
--weight-decay=0.01 \
--num-transformer-layers=2 \
--num-attention-heads=1 \
--validate-interval=10 \
--max-paths=100 \
--bucket-size=32000 \
--no-run-train-during-validate"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
