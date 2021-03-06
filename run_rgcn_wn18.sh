#!/bin/bash
#
#SBATCH --job-name=rgcn-wn18
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
--write-tensorboard \
--save-model \
--save-result \
--use-gpu \
--engine=rgcn \
--dataset-path=data/WN18 \
--data-size=-1 \
--num-epochs=6000 \
--num-rgcn-layers=1 \
--num-bases=2 \
--hidden-dim=140 \
--validate-interval=400 \
--test-batch-size=200 \
--learn-rate=0.01 \
--weight-decay=0.0 \
--embedding-decay=0.01 \
--rgcn-regularizer=basis"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
