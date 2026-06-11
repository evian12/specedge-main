#!/usr/bin/env bash

cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate || exit 1
export PYTHONPATH="$PWD/src:${PYTHONPATH}"

python -O src/script/network_autoregressive_client.py
