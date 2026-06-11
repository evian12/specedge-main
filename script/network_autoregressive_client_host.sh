#!/usr/bin/env bash

usage() {
    echo "Usage: $(basename "$0") -f <config_file>" >&2
}

if [[ "$*" =~ --help ]] || [[ "$*" =~ -h ]]; then
    usage
    exit 0
fi

cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate || exit 1
export PYTHONPATH="$PWD/src:${PYTHONPATH}"

config_file="config.yaml"
while getopts "f:h" opt; do
    case "$opt" in
        f) config_file=$OPTARG ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done

python src/script/network_autoregressive_client_host.py \
    --config "$config_file"
