# SpecEdge 成本与时间统计口径

## 1. 当前代码使用什么时间计算 4090 成本

`src/metric/specedge.py` 使用 `server.jsonl` 第一条和最后一条记录的时间戳差：

```text
server_wall_time = last_server_timestamp - first_server_timestamp
server_cost = server_wall_time × GPU每秒价格
```

因此当前“Server cost”表示整场实验期间云端 GPU 实例的占用成本，不是：

```text
整个项目时间 - Jetson草稿时间
```

## 2. 为什么不能用总时间减草稿时间

草稿、网络、4090 验证可能发生重叠：

```text
Jetson预计算  ─────────
网络请求          ───────
4090验证             ───────
```

这些时间不是互斥区间。直接相减会把重叠时间、网络等待和服务器排队错误归类。

## 3. 应同时报告两个服务器时间

### 3.1 云实例墙钟时间

含义：4090 实例从实验开始到结束被占用多久。

适用于：

- 云服务器租赁成本；
- dollars per 1M tokens；
- 实际部署预算。

即使 4090 在等待 Jetson 或网络，只要实例没有释放，通常仍然计费。

### 3.2 4090 活跃推理时间

当前 server log 中每个验证 batch 有：

```text
target.server_end_to_end_t
```

累计该字段得到目标模型实际执行验证/prefill 的近似活跃时间：

```text
server_active_time =
  Σ target.server_end_to_end_t
```

适用于：

- 判断 SpecEdge 是否减少大模型计算；
- 比较服务器利用率；
- 估算能耗或理论按需计算成本。

它不等于云厂商账单，除非平台真的只按 kernel 或请求活跃时间收费。

## 4. 推荐成本表

每个实验至少报告：

| 指标 | 计算方式 | 用途 |
|---|---|---|
| Server billed wall time | 服务器首尾日志时间戳 | 实际租赁成本 |
| Server active inference time | 累计 `server_end_to_end_t` | 大模型计算量 |
| Server active ratio | active / wall | 利用率 |
| Server billed cost | wall × 价格 | 主成本指标 |
| Server active-time cost | active × 价格 | 诊断指标 |
| Edge occupied time | Jetson 实验墙钟 | 边缘设备占用 |
| Energy | NVML/功率计积分 | 能耗比较 |
| Cost per 1M accepted tokens | 总账单/accepted tokens | 系统比较 |

## 5. 与云端自回归基线比较

主结论应使用相同资源占用边界的墙钟成本：

```text
云端自回归：
  4090墙钟时间 × 4090价格

SpecEdge：
  4090墙钟时间 × 4090价格
  + Jetson占用成本或折算成本
```

辅助结论再比较活跃推理时间，回答“SpecEdge 是否真正减少了目标模型计算”。

不要用 SpecEdge 的活跃时间成本去比较自回归的墙钟成本，这两个口径不对称。
