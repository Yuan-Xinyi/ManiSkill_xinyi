#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
python "$SCRIPT_DIR/examples/baselines/ppo/ppo_evaluate_single.py" \
    --env_id DrawStraightLine \
    --checkpoint runs/DrawStraightLine__ppo__1__1765772086/ckpt_676.pt \
    --capture-video