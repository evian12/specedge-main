#!/usr/bin/env bash
set -euo pipefail

WARMUP_SECONDS="${WARMUP_SECONDS:-180}"
CLIENT_NODE="${CLIENT_NODE:-jetson}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

CONFIGS=(
  "config/specedge_4090_jetson_multitenant_response_only.yaml"
  "config/specedge_4090_jetson_multitenant_specedge.yaml"
  "config/specedge_4090_jetson_multitenant_adaptive.yaml"
)

# Important: config/specedge_4090_jetson_multitenant_ar.yaml is a unary
# Validate-per-token ablation inside the SpecEdge verify protocol. It is not
# the realistic cloud AR baseline. The main comparison uses streaming AR
# backed by the same batch_server target model and unified FCFS scheduler.
STREAMING_AR_CONFIG="${STREAMING_AR_CONFIG:-config/specedge_4090_jetson_multitenant_streaming_ar.yaml}"

cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

cleanup_stale_servers() {
  pkill -f '[p]ython -O src/script/batch_server.py' 2>/dev/null || true
  pkill -f '[m]ultiprocessing.forkserver.*batch_server.py' 2>/dev/null || true
  pkill -f '[b]ash ./script/batch_server.sh' 2>/dev/null || true
  pkill -f '[p]ython -O src/script/network_autoregressive_server.py' 2>/dev/null || true
  pkill -f '[b]ash ./script/network_autoregressive_server.sh' 2>/dev/null || true
}

exp_name_from_config() {
  python - "$1" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as file:
    config = yaml.safe_load(file)
print(config["base"]["exp_name"])
PY
}

sync_client_logs() {
  local exp_name="$1"
  local dst="result/4090_jetson/${exp_name}"
  mkdir -p "${dst}"
  ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${CLIENT_NODE}" \
    "test -e ~/specedge/${dst}/client_0.jsonl" \
    && rsync -a \
      -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
      "${CLIENT_NODE}:~/specedge/${dst}/client_0."* \
      "${dst}/" || true
  ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${CLIENT_NODE}" \
    "test -e ~/specedge/${dst}/network_ar_client_0.jsonl" \
    && rsync -a \
      -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
      "${CLIENT_NODE}:~/specedge/${dst}/network_ar_client_0."* \
      "${dst}/" || true
}

clean_experiment_logs() {
  local exp_name="$1"
  local dst="result/4090_jetson/${exp_name}"
  rm -rf "${dst}"
  ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${CLIENT_NODE}" \
    "rm -rf ~/specedge/${dst}" || true
}

trap cleanup_server EXIT

for config in "${CONFIGS[@]}"; do
  exp_name="$(exp_name_from_config "${config}")"
  echo "==> Cleaning logs: ${exp_name}"
  cleanup_stale_servers
  clean_experiment_logs "${exp_name}"
  echo "==> Starting server: ${config}"
  setsid ./script/batch_server.sh -f "${config}" &
  SERVER_PID=$!
  echo "==> Waiting ${WARMUP_SECONDS}s for model/cache initialization"
  sleep "${WARMUP_SECONDS}"

  echo "==> Starting client: ${config}"
  ./script/client_host.sh -f "${config}"
  echo "==> Syncing client logs: ${exp_name}"
  sync_client_logs "${exp_name}"

  echo "==> Stopping server: ${config}"
  cleanup_server
  cleanup_stale_servers
  unset SERVER_PID
  sleep 5
done

streaming_ar_exp_name="$(exp_name_from_config "${STREAMING_AR_CONFIG}")"
echo "==> Cleaning logs: ${streaming_ar_exp_name}"
cleanup_stale_servers
clean_experiment_logs "${streaming_ar_exp_name}"
echo "==> Starting shared-model streaming AR server: ${STREAMING_AR_CONFIG}"
setsid ./script/batch_server.sh -f "${STREAMING_AR_CONFIG}" &
SERVER_PID=$!
echo "==> Waiting ${WARMUP_SECONDS}s for model initialization"
sleep "${WARMUP_SECONDS}"

echo "==> Starting streaming AR client: ${STREAMING_AR_CONFIG}"
./script/network_autoregressive_client_host.sh -f "${STREAMING_AR_CONFIG}"
echo "==> Syncing streaming AR client logs: ${streaming_ar_exp_name}"
sync_client_logs "${streaming_ar_exp_name}"

echo "==> Stopping shared-model streaming AR server"
cleanup_server
cleanup_stale_servers
unset SERVER_PID

python src/metric/runtime_modes.py \
  -d \
  result/4090_jetson/multitenant_response_only \
  result/4090_jetson/multitenant_specedge \
  result/4090_jetson/multitenant_response_only_adaptive \
  --labels response_only,specedge,response_only_adaptive \
  --csv results/multitenant_compare.csv

python src/metric/network_autoregressive.py \
  -d "result/4090_jetson/${streaming_ar_exp_name}"
