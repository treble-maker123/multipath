#!/bin/bash

# Command to start everything:
# export CUDA_VISIBLE_DEVICES=""
# bash scripts/start_ray.sh && python3 scripts/generate_paths.py && ray stop && exit

export CUDA_VISIBLE_DEVICES=""

ray start --head \
  --node-ip-address=127.0.0.1 \
  --redis-port=8765 \
  --memory=4000000000 \
  --object-store-memory=32000000000 \
  --num-cpus=46
