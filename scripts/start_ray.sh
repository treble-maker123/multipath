#!/bin/bash

# Command to start everything:
# bash scripts/start_ray.sh && python scripts/generate_paths.py && ray stop && exit

ray start --head \
  --node-ip-address=127.0.0.1 \
  --redis-port=8765 \
  --memory=4000000000 \
  --object-store-memory=32000000000 \
  --num-cpus=8
