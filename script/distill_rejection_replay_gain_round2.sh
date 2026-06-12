#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python src/distillation/collect_teacher.py \
  --config config/distillation/collect_replay_gain_round2_specbench.yaml
python src/distillation/collect_teacher.py \
  --config config/distillation/collect_replay_gain_round2_general.yaml
python src/distillation/combine_jsonl.py \
  --input \
  data/distillation/acceptance_mixed/kd_train.jsonl \
  data/distillation/rejection_replay_gain_round2/specbench_train.jsonl \
  data/distillation/rejection_replay_gain_round2/general_train.jsonl \
  --output data/distillation/rejection_replay_gain_round2/kd_train.jsonl
python src/distillation/train_drafter.py \
  --config config/distillation/train_rejection_replay_gain_round2.yaml
