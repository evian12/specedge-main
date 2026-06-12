#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python src/distillation/train_drafter.py "$@"
