#!/bin/bash
#
#SBATCH --job-name=rgcn-ctd-repodb
#SBATCH -e outputs/errors/%j.txt
#SBATCH --output=outputs/logs/%j.txt
#SBATCH --partition=m40-long
#SBATCH --ntasks=8
#SBATCH --time=07-00:00
#SBATCH --mem=45000
#SBATCH --gres=gpu:1

# For debugging device-side assert errors
#export CUDA_LAUNCH_BLOCKING=1

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
--dataset-path=data/CTD_RepoDB \
--graph-sample-size=20000 \
--data-size=-1 \
--num-epochs=3000 \
--num-rgcn-layers=2 \
--num-bases=100 \
--hidden-dim=200 \
--validate-interval=300 \
--test-batch-size=4000 \
--learn-rate=0.01 \
--weight-decay=0.0 \
--embedding-decay=0.0 \
--rgcn-regularizer=bdd \
--link-predict \
--no-run-train-during-validate"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
