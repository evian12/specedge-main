# 最终实验配置索引

本文档记录当前保留的实验配置，避免后续把旧探索配置误当成最终方案。

## 主对比实验

主对比固定使用 `specbench`、`TinyLlama-1.1B` 草稿模型、`Llama-2-7b-chat-hf` 目标模型，并扫描 decode 延迟 `50/100/150/200ms`。

| 方案 | 配置文件 |
| --- | --- |
| 网络自回归 | `config/network_autoregressive_4090_jetson_decode_lat{50,100,150,200}.yaml` |
| 原始 SpecEdge | `config/specedge_4090_jetson_tree_original_decode_lat{50,100,150,200}.yaml` |
| 当前 response-only 概率 root 深度方案 | `config/specedge_4090_jetson_tree_prob_depth_response_only_decode_lat{50,100,150,200}.yaml` |

当前方案的关键策略：

- 初始推测：`initial_draft.mode = local_streak`，`controller = score`，树结构，深度范围 `[2, 4]`。
- 预计算路径：`proactive.path_policy = hybrid_sequence_multi_position`。
- root 深度：`multi.root_depth_mode = probability`，根据接受位置概率选择防守位置。
- root 数量：`multi.dynamic_roots.mode = online_marginal`，按在线收益动态决定。
- deadline：`adaptive.layer_deadline_mode = response_only`，取消逐层保守截止，主要按验证响应等待时间决定预计算预算。

## 后续消融实验

以下配置只用于拆解当前方案贡献，不作为主结果直接比较：

| 消融点 | 配置文件 | 目的 |
| --- | --- | --- |
| 单 root | `config/specedge_4090_jetson_tree_prob_depth_response_only_top1_decode_lat100.yaml` | 验证动态多 root 是否带来收益 |
| 更大初始/预计算深度 | `config/specedge_4090_jetson_tree_prob_depth_response_only_depth5_decode_lat200.yaml` | 验证高延迟下更深推测是否有效 |
| 命中后补生成 | `config/specedge_4090_jetson_tree_prob_depth_response_only_reuse_decode_lat200.yaml` | 验证复用后是否需要补齐初始推测长度 |
| 调参版本 | `config/specedge_4090_jetson_tree_prob_depth_response_only_tuned_decode_lat200.yaml` | 记录高延迟调参尝试 |
| v2 策略 | `config/specedge_4090_jetson_tree_prob_depth_response_only_v2_decode_lat{100,200}.yaml` | 对比 response-only 变体 |

## 指标口径

- 生成速度：`src/metric/network_autoregressive.py` 或 `src/metric/proactive.py` 输出的 `tok/s`。
- 预计算命中率：`src/metric/proactive.py` 输出的 `align %`。
- 全局复用深度：`reuse`，即所有 cycle 平均后的复用深度。
- 命中条件下复用深度：如需计算，使用 `match reuse / (align % / 100)`。

## 负载感知流程

第三部分负载感知现在包含两层实现：

1. 真实 SpecEdge batch server 中的背景请求生成与 step 状态日志。
2. 基于离线 profiling 的 AR / SpecEdge / adaptive 对比实验矩阵，用 EMA 估计服务器响应时间并输出 CSV。

真实 runtime 中，AR 与 SpecEdge 现在通过同一个 SpecEdge `Validate` RPC 共享目标模型实例和 target KV cache：

- SpecEdge 模式发送推测树验证请求。
- AR 模式发送单 token decode 请求，也进入同一个 server queue。
- 两者都由同一个 Llama2-7B batch server 处理，因此可以和背景租户请求一起 continuous batching。

KV cache 切换规则：

- `SpecEdge -> AR`：server 在 verify 后已经把 target KV cache reorder 到 accepted prefix，AR 可以直接继续发送单 token decode。
- `AR -> SpecEdge`：AR 期间 target KV cache 已经在 server 端持续更新；client 端 draft model 没有参与 AR token 生成，所以切回 SpecEdge 前会基于当前 accepted prefix 对 draft model 执行一次本地 prefill，重建 draft KV cache。
- 切换点仍限制在 accepted prefix 边界：SpecEdge 在 verify 完成后切换，AR 每 `decision_window` 个 token 后切换。

流程包括：

1. 离线 profiling：读取网络自回归、原始 SpecEdge、当前 response-only 方案在 `50/100/150/200ms` 下的性能曲线。
2. 背景负载生成：真实 server 可开启 `server.background`，用随机到达近似其他租户请求；每个背景请求有随机 prompt length 和 generation length。
3. 请求调度：背景请求和实验请求进入同一个 batch server，请求使用 FCFS 进入 batch，每个 decode/verify step 使用 continuous batching。
4. 状态记录：server 每个 step 记录

   `server_response_time, decode_latency, prefill_latency, batch_size, queue_length, pending_prefill_count`

   这些状态用于负载分析和后续建模。
5. EMA 响应时间估计：在线切换不直接使用状态公式，而是使用最近观测到的 server response time 更新 EMA：

   `estimated_server_time = (1 - alpha) * estimated_server_time + alpha * current_server_response_time`

   默认 `alpha = 0.2`；第一轮没有历史值时直接用当前观测值初始化。
6. 模式选择：每个安全切换点使用 `estimated_server_time`：

   - `estimated_server_time < switch_threshold_ms`：选择网络自回归 AR。
   - `estimated_server_time >= switch_threshold_ms`：选择 SpecEdge。
6. 安全切换点：
   - SpecEdge 在每轮验证结束后做下一轮模式选择。
   - 网络自回归每 `decision_window` 个 token 做一次模式选择，默认 `16`。
   - AR -> SpecEdge 时计入一次 draft KV cache 重建开销，默认 `48ms`。
   - SpecEdge -> AR 不额外计入 cache 重建开销。

默认策略：

- 当估计服务器响应/decode 延迟 `<= 60ms` 时，选择网络自回归。
- 当估计服务器响应/decode 延迟 `> 60ms` 时，选择当前 response-only 概率 root 深度方案。

运行方式：

```bash
python src/metric/load_aware.py \
  -r result/4090_jetson \
  --latencies 50,100,150,200 \
  --threshold-ms 60
```

也可以使用 YAML 配置：

```bash
python src/metric/load_aware.py \
  --config config/load_aware_4090_jetson.example.yaml
```

CSV 结果默认保存到：

```text
results/load_aware_results.csv
```

CSV 字段包括：

```text
mode, background_load, total_time, tokens_per_second,
system_tokens_per_second, average_latency_per_token,
average_server_response_time, average_acceptance_length,
number_of_cycles, mode_switch_count, specedge_ratio, ar_ratio
```

默认动态负载参数使用 `background_arrival_rate=0.2 req/s`、`decision_window=16`、`max_batch_size=8`。这个负载不会把队列直接打满，能同时覆盖低负载自回归有利区间和高负载推测解码有利区间。

如果需要模拟不同负载分布，可以指定延迟权重：

```bash
python src/metric/load_aware.py \
  -r result/4090_jetson \
  --latencies 50,100,150,200 \
  --threshold-ms 60 \
  --latency-weights 50=0.2,100=0.3,150=0.3,200=0.2
```

输出包含两张表：

- 每个延迟点下网络自回归、原始 SpecEdge、当前方案的 Jetson 侧生成速度，以及负载感知策略选择。
- 按给定延迟分布计算的整体系统吞吐量。吞吐量使用加权调和平均，表示相同 token 量在不同负载状态下完成的总体速度。
- 动态负载模拟表：在背景请求、FCFS 队列和 continuous batching 下，对比固定网络自回归、固定原始 SpecEdge、固定当前方案、负载感知安全切换四种策略的 Jetson 前台吞吐量和系统总吞吐量。

真实 server 背景负载配置示例：

```yaml
server:
  background:
    load: medium
    arrival_rate: 0.5
    max_active_requests: 8
    prompt_min_tokens: 16
    prompt_max_tokens: 128
    generation_min_tokens: 16
    generation_max_tokens: 64
    queue_poll_ms: 5.0
```

开启后，背景请求会占用同一个 Llama2-7B 目标模型实例和同一个 batch forward，但不会向真实 client 返回响应；它们只用于模拟其他租户的负载压力。
