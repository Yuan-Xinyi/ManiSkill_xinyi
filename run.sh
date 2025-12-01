#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
python "$SCRIPT_DIR/examples/baselines/ppo/ppo.py" \
    --env_id PickCube-v1 \
    --num_envs 1024 \
    --update_epochs 8 \
    --num_minibatches 32 \
    --total_timesteps 500000000 \
    --track

