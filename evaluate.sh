#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
python "$SCRIPT_DIR/examples/baselines/ppo/ppo_evaluate.py" \
    --env_id DrawStraightLine \
    --run-folder runs/eg5_DrawStraightLine__ppo__1__1765429842 \
    --num-eval-envs 1024 \
    --output-file evaluation_results.jsonl