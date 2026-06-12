# SpecEdge \[NeurIPS 2025\]

> Scalable Edge-Assisted Serving Framework for Interactive LLMs

[paper](https://arxiv.org/abs/2505.17052)

SpecEdge is an edge-assisted LLM inference framework that leverages consumer-grade GPUs for cost-effective serving at scale. By splitting workloads between edge and server using speculative decoding, SpecEdge achieves 2.22x server throughput and 11.24% lower latency compared to server-only baselines. The framework features proactive edge drafting and pipeline-aware scheduling to maximize resource utilization and serving efficiency.

![](img/abstract_timeline.jpg)

![](img/performance.jpg)

## Experimental Setup

Our experiments were conducted with the following hardware:

### Server (A100 40GB)

- GCP a2-highgpu-1g
- Ubuntu 24.04 LTS NVIDIA version: 580

### Server (A100 80GB)

- GCP a2-ultragpu-1g
- Ubuntu 24.04 LTS NVIDIA version: 580

### Edge

- Ubuntu 20.04.6 LTS (Focal Fossa)
- Two Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz
- 208GiB Memory
- 10GbE network interface

## Setup

```bash
git clone https://github.com/kaist-ina/specedge
cd specedge
uv sync
```

## SpecEdge

### Usage

Before running, ensure that SSH communication is established between the node executing the `client_host.sh` script and other edge nodes. Additionally, the SpecEdge repository must be cloned to identical absolute paths across all edge nodes (e.g., all nodes should have the repository at `/home/user/specedge`).

```bash
# server
./script/batch_server.sh -f config/specedge_4090_jetson.yaml

# edge
./script/client_host.sh -f config/specedge_4090_jetson.yaml
```

### Configuration

The `specedge.example.yaml` configuration file contains the following settings:

**Base Settings:**

- `result_path`: Base directory where experiment results are saved
- `exp_name`: Experiment name for folder name and identification
- `dtype`: Model precision (fp16/fp32)
- `seed`: Random seed for reproducibility
- `ssh_key`: SSH key path for remote server access
- `max_len`: Maximum sequence length (max KV Cache length)

**Server Settings:**

- `process_name`: Server process identifier for logging
- `target_model`: HuggingFace model path (e.g., Qwen/Qwen3-14B)
- `device`: CUDA device identifier for the server target model (e.g., cuda:0)
- `temperature`: Sampling temperature for generation
- `max_batch_size`: Maximum batch size for concurrent requests
- `num_clients`: Expected number of concurrent edge clients, must match the number of edge clients
- `batch_type`: Batching strategy (static/dynamic)
- `cache_prefill`: Enable prefill KV cache preloading
  - `true`: Pre-compute and cache all dataset prompts at server startup for benchmark experiments
  - `false`: Perform prefill at runtime using client-provided prompts

**Client Settings:**

- `host`: Server endpoint (e.g., 127.0.0.1:8000)
- `process_name`: Client process identifier for logging
- `draft_model`: HuggingFace model path for draft generation (e.g., Qwen/Qwen3-1.7B)
- `dataset`: Benchmark dataset name (c4, mtbench, oasst, wikitext and specbench)
- `sample_req_cnt`: Sampling frequency of requests from dataset
- `reasoning`: Enable reasoning
- `req_offset`: Offset for request sampling
- `max_n_beams`, `max_beam_len`, `max_branch_width`, `max_budget`: Speculative decoding parameters
- `initial_draft`: Initial speculative-tree depth policy
  - `mode: fixed`: Original behavior using `max_beam_len`
  - `mode: linucb`: Online Disjoint LinUCB selection from `candidate_depths`
  - `warmup_per_depth`: Forced samples collected for each candidate depth
  - `exploration_weight`: LinUCB uncertainty bonus
  - `forced_exploration_interval`: Periodic least-sampled-depth exploration
- `proactive`: Proactive edge drafting configuration
  - `type`: Proactive drafting mode (excluded/included)
  - `mode`: Proactive execution policy
    - `baseline`: Original implementation; always completes the configured depth
    - `interruptible`: Checks for the server response between proactive layers
    - `adaptive`: Uses response deadlines, per-layer latency models, uncertainty,
      and alignment feedback to decide whether each proactive layer should start
  - `path_policy`: Proactive result selection policy
    - `single_best`: Original one-leaf, one-bonus behavior
    - `deepest_multi`: Selects multiple maximum-depth leaves and dynamically
      allocates bonus-token roots
  - `max_n_beams`, `max_beam_len`, `max_branch_width`, `max_budget`: Proactive drafting parameters
- `max_new_tokens`: Maximum tokens to generate per request
- `max_request_num`: Total requests to process (-1 for all)

**Node Settings:**

- `node-name`: Name of each edge node for SSH access (must match the SSH hostname configured in your SSH config or be a resolvable hostname)
  - `device`: CUDA device identifier for the edge process on this node (e.g., cuda:0, cuda:1)

### Metrics

Before running the metric script, you need to collect the JSONL files from both the server and edge into a single location.

```bash
. .venv/bin/activate
python src/metric/specedge.py -d result/demo/specedge --gpu "A100-40" # A100 40GB
python src/metric/specedge.py -d result/demo/specedge --gpu "A100-80" # A100 80GB
```

### Jetson + RTX 4090 Ablation

The original configuration remains the baseline. The additional configurations
change only the proactive policy and write to separate experiment directories:

```bash
# Original paper-code behavior
./script/client_host.sh -f config/specedge_4090_jetson.yaml

# No proactive drafting
./script/client_host.sh -f config/specedge_4090_jetson_no_proactive.yaml

# Stop after the current proactive layer when validation has returned
./script/client_host.sh -f config/specedge_4090_jetson_interruptible.yaml

# Interruptible execution plus online depth and alignment control
./script/client_host.sh -f config/specedge_4090_jetson_adaptive.yaml

# Adaptive multi-leaf, multi-bonus proactive drafting
./script/client_host.sh -f config/specedge_4090_jetson_deepest_multi.yaml

# LinUCB initial depth plus adaptive multi-leaf proactive drafting
./script/client_host.sh -f config/specedge_4090_jetson_bandit.yaml

# Replace TinyLlama with AMD-Llama-135M while keeping the original tree settings
./script/client_host.sh -f config/specedge_4090_jetson_amd135m.yaml
```

Client JSONL records keep the original `target.proactive` fields and add
`target.proactive_execution`, including planned/executed depth, proactive and
response latency, stop reason, per-layer wall/GPU time, deadline decisions, and
adaptive-controller EWMAs.

The adaptive controller records `response_received_ms` immediately after the
gRPC response bytes arrive, plus separate decode and observation timestamps.
Before setup and every proactive layer, it computes:

```text
remaining = conservative_response_deadline - request_elapsed
predicted_cost = layer_mean + uncertainty_scale * layer_error
```

The next stage starts only when `predicted_cost + safety_margin <= remaining`.
Each proactive depth has an independent latency and error EWMA.

The `deepest_multi` policy preserves `single_best` as the default baseline. It:

1. Finds all leaves at the maximum draft-tree depth and keeps the highest-score
   leaves up to `max_deepest_leaves`.
2. Gives every selected leaf at least `min_bonus_per_leaf` bonus candidate.
3. Adds more bonus candidates while
   `full_depth_acceptance * leaf_probability * bonus_probability` exceeds
   `min_root_probability`, subject to `max_roots`.
4. Expands all roots at the first proactive layer, then retains roots covering
   the configured cumulative probability at later layers.
5. Checks the adaptive response deadline before every batched proactive layer.
6. Keeps only the subtree whose `(leaf, bonus)` pair matches validation.

The complete-depth acceptance rate starts from `full_depth_prior` and is updated
online with an EWMA. Layer latency is tracked separately by depth and proactive
batch width.

The optional initial-draft LinUCB controller selects a depth before every
non-prefill speculative cycle. It uses only historical acceptance, draft cost,
validation-response, tree-size, context, and proactive-hit features available
before the decision. The reward is clipped per-cycle throughput:

```text
reward = 1000 * generated_tokens / cycle_latency_ms
```

Each candidate depth has an independent linear reward model. The original
fixed-depth behavior remains the default when `initial_draft` is omitted or
uses `mode: fixed`.

Compare completed runs with:

```bash
python src/metric/proactive.py -d \
  result/4090_jetson/specedge \
  result/4090_jetson/specedge_no_proactive \
  result/4090_jetson/specedge_interruptible \
  result/4090_jetson/specedge_adaptive \
  result/4090_jetson/specedge_deepest_multi \
  result/4090_jetson/specedge_bandit
```

## Drafter Distillation

The `src/distillation` subproject trains a draft model offline without changing
the SpecEdge inference path. It supports general instruction prompts, business
request history, and a bounded hard-example replay source. Generated datasets,
teacher logits, and checkpoints are ignored by Git.

完整中文复现教程（数据格式、SFT、KD、损失公式、拒绝回放、部署和实验矩阵）见
[`docs/drafter_training_zh.md`](docs/drafter_training_zh.md)。服务器墙钟成本与 GPU
活跃推理时间的区别见
[`docs/cost_accounting_zh.md`](docs/cost_accounting_zh.md)。

Canonical input JSONL accepts either:

```json
{"id":"1","prompt":"Explain speculative decoding."}
{"request_id":"2","messages":[{"role":"user","content":"Summarize this."}]}
```

Build prompt-level train, validation, and test splits:

```bash
./script/distill_build_dataset.sh \
  --config config/distillation/build_dataset.example.yaml
```

The split is assigned from a seeded hash of the complete prompt. Duplicate
prompts cannot cross splits. Do not include the final SpecBench or business
holdout test requests in any training source.

First collect teacher responses and run supervised fine-tuning:

```bash
./script/distill_collect.sh \
  --config config/distillation/collect_sft_amd135m.example.yaml
./script/distill_collect.sh \
  --config config/distillation/collect_sft_validation_amd135m.example.yaml
./script/distill_train.sh \
  --config config/distillation/train_sft_amd135m.example.yaml
```

Then generate on-policy continuations with the SFT student, score them with the
target model, and train with sparse top-k logit distillation:

```bash
./script/distill_collect.sh \
  --config config/distillation/collect_kd_amd135m.example.yaml
./script/distill_collect.sh \
  --config config/distillation/collect_kd_validation_amd135m.example.yaml
./script/distill_train.sh \
  --config config/distillation/train_kd_amd135m.example.yaml
```

The KD records store teacher top-k probabilities plus aggregate tail
probability instead of the complete vocabulary logits. Evaluate an untouched
validation or test set with:

```bash
./script/distill_evaluate.sh \
  --config config/distillation/evaluate_amd135m.example.yaml
```

The evaluator reports target/student top-1 agreement, target top-1 coverage in
the student top-k, student cross-entropy, and a greedy acceptance-depth proxy.
Actual SpecEdge throughput remains the final metric. To use a trained model in
the future, create a separate inference configuration whose `draft_model`
points at the selected checkpoint; no inference configuration is changed by
the distillation tools.

### Rejection Replay

The next acceptance-depth round can replay only the selected drafter's errors
on teacher-generated trajectories:

```bash
./script/distill_rejection_replay.sh
```

`loss_mask.mode: student_rejections` compares the current student's greedy
token with the teacher Top-1 token at every response position. The generated
KD record activates only confident disagreement positions; `window_size` can
also include a small number of valid teacher-trajectory states immediately
after each error. This differs from ordinary rejection weighting: easy tokens
do not enter the replay loss at all.

The replay training configuration mixes those hard records with the original
multi-domain KD data and adds a Top-1 margin loss:

```text
teacher logit >= strongest alternative logit + margin
```

Both features are opt-in. Existing SFT, KD, acceptance-KD, and SpecEdge
inference configurations preserve their previous behavior.

An optional second round weights each rejection by its estimated improvement
to capped mean acceptance depth:

```bash
./script/distill_rejection_replay_gain.sh
```

For each wrong token, the collector temporarily treats that token as corrected
and recomputes the sum of contiguous accepted depths, capped at the configured
verification depth. Errors that bridge two otherwise-correct runs receive
more weight than isolated errors. The logarithmic, capped weight avoids a few
long examples dominating training.

The selected replay checkpoint and a separate deployment configuration are
documented in
[`docs/acceptance_depth_experiment.md`](docs/acceptance_depth_experiment.md)
and
`config/specedge_4090_jetson_sequence_depth_replay.yaml`. The original
sequence-depth and baseline configurations are unchanged.

## Sequence-Depth Proactive Drafting

`sequence_depth` keeps the original proactive policies as baselines while
adding a single-sequence alternative:

```bash
./script/client_host.sh \
  -f config/specedge_4090_jetson_sequence_depth.yaml
```

Set `initial_draft.structure: sequence` to generate one greedy draft path. The
configured `sequence.acceptance_survival` contains offline
`P(accepted depth >= d)` values beginning with depth zero. The client converts
them to exact stopping probabilities and allocates `max_roots` bonus candidates
across every possible stopping depth. Candidates at the same proactive layer
are evaluated in one draft-model batch, while different proactive layers
remain sequential and are guarded by the adaptive validation deadline.

Every selected `(stop depth, bonus token)` pair starts one greedy proactive
continuation. When validation matches a pair, only that continuation is kept.
The following cycle generates:

```text
max(0, selected initial depth - actually reused proactive depth)
```

new draft layers. The actual completed depth is used because validation may
interrupt proactive generation before its configured maximum.

The acceptance-depth distillation experiment, anti-overfitting protocol, and
RTX 4090 results are documented in
[`docs/acceptance_depth_experiment.md`](docs/acceptance_depth_experiment.md).

## Auto Batch

### Usage

```bash
./script/auto_batch.sh -f config/auto_batch.example.yaml
```

### Configuration

The `auto_batch.example.yaml` configuration file contains the following settings:

**Base Settings:**

- `result_path`: Base directory where experiment results are saved
- `exp_name`: Experiment name for folder name and identification
- `seed`: Random seed for reproducibility
- `model`: HuggingFace model path (e.g., Qwen/Qwen3-14B)
- `device`: CUDA device identifier for the model (e.g., cuda:0)
- `dtype`: Model precision (fp16/fp32)
- `temperature`: Sampling temperature for generation
- `dataset`: Benchmark dataset name (c4, mtbench, oasst, wikitext and specbench)
- `batch_size`: Batch size for concurrent request processing
- `max_len`: Maximum sequence length (max KV Cache length)
- `max_new_tokens`: Maximum tokens to generate per request
- `max_request_num`: Total requests to process (-1 for all)
- `sample_req_cnt`: Number of sample requests from dataset

### Get metrics

```bash
. .venv/bin/activate
python src/metric/auto_batch.py -d result/demo/auto_batch --gpu "A100-40" # A100 40GB
python src/metric/auto_batch.py -d result/demo/auto_batch --gpu "A100-80" # A100 80GB
```

## Network Autoregressive

This baseline models a conventional cloud LLM API for the one-client,
one-server experiment. The edge sends each prompt once, the server performs
ordinary token-by-token autoregressive decoding, and generated tokens are
streamed back over the same gRPC request. Unlike `auto_batch`, its latency
therefore includes prompt upload, server prefill/decode, and token delivery.

Start the server on the RTX 4090:

```bash
./script/network_autoregressive_server.sh \
  -f config/network_autoregressive_4090_jetson.yaml
```

Then start the client host on the RTX 4090. It creates a reverse SSH tunnel and
runs the measurement client on the configured Jetson:

```bash
./script/network_autoregressive_client_host.sh \
  -f config/network_autoregressive_4090_jetson.yaml
```

The client records TTFT, TPOT, request end-to-end latency, server prefill/decode
time, and an estimated transport/scheduling overhead. Summarize the client log
with:

```bash
python src/metric/network_autoregressive.py -d \
  result/4090_jetson/network_autoregressive
```

Use the same dataset, request indices, generation length, target model, dtype,
temperature, and prefill policy when comparing this baseline with SpecEdge.
This implementation deliberately supports one client at a time so the first
comparison isolates cloud-edge network latency from server queueing effects.

## Server-Only

### Usage

```bash
./script/server_only.sh -f config/server_only.example.yaml
```

### Configuration

The `server_only.example.yaml` configuration file contains the following settings:

**Base Settings:**

- `result_path`: Base directory where experiment results are saved
- `exp_name`: Experiment name for folder name and identification
- `dtype`: Model precision (fp16/fp32)
- `seed`: Random seed for reproducibility
- `ssh_key`: SSH key path for remote server access (not required for server_only)
- `max_len`: Maximum sequence length (max KV Cache length)

**Server Settings:**

- `process_name`: Server process identifier for logging
- `target_model`: HuggingFace model path (e.g., Qwen/Qwen3-14B)
- `device`: CUDA device identifier for the server target model (e.g., cuda:0)
- `temperature`: Sampling temperature for generation
- `num_clients`: Expected number of concurrent clients

**Client Settings:**

- `host`: Server endpoint (not required for server_only)
- `process_name`: Client process identifier for logging
- `draft_model`: HuggingFace model path (e.g., Qwen/Qwen3-1.7B)
- `dataset`: Benchmark dataset name (c4, mtbench, oasst, wikitext and specbench)
- `max_n_beams`, `max_beam_len`, `max_branch_width`, `max_budget`: Speculative decoding parameters
- `max_batch_size`: Maximum batch size for requests
- `max_new_tokens`: Maximum tokens to generate per request
- `max_request_num`: Total requests to process (-1 for all)
- `sample_req_cnt`: Number of sample requests from dataset
- `device`: CUDA device identifier for the client draft model (e.g., cuda:0)

### Get metrics

```bash
. .venv/bin/activate
python src/metric/server_only.py -d result/demo/server_only --gpu "A100-40" # A100 40GB
python src/metric/server_only.py -d result/demo/server_only --gpu "A100-80" # A100 80GB
```

## Speculative Decoding Parameters

![](img/param.svg)

SpecEdge uses a tree-based speculative decoding approach to generate multiple candidate tokens efficiently. The following parameters control the structure and size of the speculation tree:

- `max_n_beams`: Maximum number of nodes that can be forwarded through the model in a single iteration. This parameter limits how many tree nodes are selected as candidates for the forward pass. Even if more nodes could potentially be forwarded based on their probabilities, only the top `max_n_beams` nodes are processed to control computational cost.
- `max_beam_len`: Maximum depth of the speculation tree, representing how many sequential token generation steps are performed during one draft phase. The tree grows iteratively for `max_beam_len` steps, with each step expanding the tree by forwarding selected candidate nodes.
- `max_branch_width`: Maximum number of child tokens generated from each parent node in the tree. When processing logits from a forward pass, up to `max_branch_width` tokens with the highest probabilities are selected as potential continuations from each node.
- `max_budget`: Maximum total number of nodes allowed in the speculation tree (excluding the prefix). This parameter controls the overall tree size by pruning nodes with lower probabilities. After tree construction, if the number of nodes exceeds `max_budget`, only the top `max_budget` nodes based on cumulative log probabilities are retained. The budget mechanism uses a priority-based filtering approach with a decay factor (0.9) applied to parent node scores.

This speculative decoding approach is based on the SpecExec algorithm. For more details, please refer to the [SpecExec paper](https://github.com/yandex-research/specexec).

## Citation

```
@inproceedings{park2025specedge,
  author = {Jinwoo Park and Seunggeun Cho and Dongsu Han},
  title = {SpecEdge: Scalable Edge-Assisted Serving Framework for Interactive LLMs},
  booktitle = {Annual Conference on Neural Information Processing Systems},
  year = {2025},
  eprint = {2505.17052},
  archivePrefix = {arXiv},
  primaryClass= {cs.CL}
}
```

原始 SpecEdge 假设 edge draft 足够快，在 Jetson 等资源受限设备上固定 proactive drafting 会产生严重计算超支。本方法根据设备状态、网络延迟和历史命中率动态控制预计算，实现更广泛的异构边缘部署。
