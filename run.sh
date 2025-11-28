#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
python "$SCRIPT_DIR/examples/baselines/ppo/ppo.py" \
    --env_id DrawStraightLine \
    --num_envs 16 \
    --update_epochs 8 \
    --num_minibatches 32 \
    --total_timesteps 500000000 \
    --track

