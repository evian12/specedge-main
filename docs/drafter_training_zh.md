# SpecEdge 草稿模型 SFT、蒸馏与拒绝回放完整指南

本文档描述当前项目中 AMD-Llama-135M 草稿模型的完整离线训练流程，包括：

1. 数据集构建与严格划分；
2. 教师响应采集；
3. 监督微调（SFT）；
4. 基于学生轨迹的知识蒸馏（KD）；
5. 面向推测解码接受率的损失函数；
6. 拒绝位置回放与接受深度增益加权；
7. 离线评测、反过拟合检查；
8. 从 RTX 4090 部署到 Jetson；
9. 如何将训练结果接入推测序列实验。

本文档对应的实现位于：

```text
src/distillation/
config/distillation/
script/distill_*.sh
```

训练过程不修改 SpecEdge 的推理语义。只有在推理配置中显式修改
`client.draft_model` 时，才会使用训练后的模型，因此原始基线始终保留。

## 1. 研究目标

推测解码中，草稿模型先生成候选 token，目标模型再验证这些 token。若草稿模型
与目标模型在当前位置选择相同 token，该 token 被接受；遇到第一个不一致位置后，
本轮后续草稿通常无法继续复用。

因此，草稿模型训练的主要目标不是获得最好的自然语言生成质量，而是：

- 提高目标模型与草稿模型的 Top-1 一致率；
- 延长连续一致 token 的长度；
- 提高平均接受深度；
- 在不同任务上保持泛化；
- 保持模型足够小，使 Jetson 草稿耗时低于节省的云端验证与网络等待时间。

仅优化语言模型交叉熵并不一定能最大化连续接受深度。当前子项目采用
“SFT 预热 + on-policy KD + 接受率损失 + 拒绝位置回放”的多阶段流程。

## 2. 参考文献与采用的思想

### 2.1 Speculative Decoding

Leviathan 等人的工作给出了推测解码的经典形式：小模型提出候选，大模型并行验证，
在保持目标模型分布正确性的前提下减少大模型串行解码次数。

- Fast Inference from Transformers via Speculative Decoding
  <https://arxiv.org/abs/2211.17192>

本项目主要使用贪心 Top-1 一致性分析接受深度。实际 SpecEdge 验证逻辑仍由原项目
实现，离线训练不会改变验证正确性。

### 2.2 DistillSpec

DistillSpec 强调：草稿模型应在与部署时相近的轨迹上学习，而且蒸馏距离应与实际
解码方式匹配。

- DistillSpec: Improving Speculative Decoding via Knowledge Distillation
  <https://arxiv.org/abs/2310.08461>

本项目采用的对应设计：

- 先用 SFT 模型自行生成 response，形成学生的 on-policy 轨迹；
- 再让目标模型对这些轨迹的每个位置打分；
- 保存教师 Top-K 分布和尾部总概率；
- 使用分布蒸馏、Top-1 监督和 TVD 联合训练。

### 2.3 AdaSPEC

AdaSPEC 的核心启发是：小模型容量有限，不应把同样的训练预算平均花在所有 token
上，应优先学习真正影响接受的 token。

- AdaSPEC: Selective Knowledge Distillation for Efficient Speculative
  Decoders
  <https://arxiv.org/abs/2510.19779>

本项目采用的对应设计：

- `min_teacher_confidence` 过滤教师置信度过低的位置；
- 对学生与教师 Top-1 不同的位置提高权重；
- 后续拒绝回放阶段只保留仍会导致验证失败的位置。

### 2.4 Draft-OPD

Draft-OPD 使用目标模型轨迹识别草稿模型的错误位置，并针对这些位置进行回放训练。

- Draft-OPD: On-Policy Distillation for Speculative Draft Models
  <https://arxiv.org/abs/2605.29343>

本项目采用的对应设计：

- 在教师正确轨迹上运行当前学生模型；
- 找到学生 Top-1 与教师 Top-1 不一致的位置；
- 将这些位置写入 `loss_mask`；
- 可按照“修复该错误后平均接受深度增加多少”设置 `loss_weights`。

### 2.5 泛化与反过拟合

任务定向微调可能提高训练域接受率，却降低其他任务或语言上的接受率。因此必须使用
跨域留出集和一次性 late holdout。

- Speculative Decoding Across Languages
  <https://arxiv.org/abs/2605.30580>

本项目采用的对应设计：

- SpecBench、OASST、C4、WikiText 混合训练；
- 独立跨域 holdout；
- 最终 141 条 late holdout 只评测一次；
- 不以训练 loss 或单个任务结果作为最终采用依据。

## 3. 模型与 tokenizer 约束

当前实验模型：

```text
教师/目标模型：Llama-2-7b-chat-hf
学生/草稿模型：AMD-Llama-135m
```

推测解码要求两个模型对同一文本产生兼容 token 序列。项目会在数据采集和评测前调用
`src/specedge/tokenizer.py` 检查 tokenizer：

- vocabulary 大小；
- token 到 id 的映射；
- BOS、EOS、PAD 等特殊 token；
- 若干探针文本的编码结果。

如果 tokenizer 不兼容，即使语义输出相似，也无法逐 token 验证，不能直接作为草稿
模型。不要通过忽略检查来强行训练。

## 4. 环境准备

RTX 4090 主机上的参考目录：

```text
/home/yypan/specedge
/home/yypan/models/Llama-2-7b-chat-hf
/home/yypan/models/AMD-Llama-135m
```

进入项目并激活环境：

```bash
cd ~/specedge
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

确认 GPU：

```bash
nvidia-smi
```

确认两个模型可加载，并执行 tokenizer 检查：

```bash
python - <<'PY'
from transformers import AutoTokenizer
from specedge.tokenizer import validate_tokenizer_compatibility

teacher = AutoTokenizer.from_pretrained(
    "/home/yypan/models/Llama-2-7b-chat-hf",
    legacy=False,
)
student = AutoTokenizer.from_pretrained(
    "/home/yypan/models/AMD-Llama-135m",
    legacy=False,
)
validate_tokenizer_compatibility(
    student,
    teacher,
    ["Tokenizer compatibility check.", "你好，推测解码。"],
)
print("tokenizer compatible")
PY
```

## 5. 数据格式

### 5.1 原始 prompt 数据

支持两类 JSONL。

单 prompt：

```json
{"id":"req-001","prompt":"用通俗语言解释什么是推测解码。"}
```

多轮消息：

```json
{
  "request_id":"req-002",
  "messages":[
    {"role":"system","content":"你是企业知识库助手。"},
    {"role":"user","content":"总结本季度退款率上升的主要原因。"}
  ]
}
```

`config/distillation/build_dataset.example.yaml` 中的每个数据源必须配置：

```yaml
sources:
  - name: business_history
    path: data/raw/business_history.jsonl
    messages_field: messages
    id_field: request_id
    max_samples: 10000
```

同一个 source 只能选择 `prompt_field` 或 `messages_field` 之一。

### 5.2 规范化后的 prompt

`build_dataset.py` 将不同来源统一为：

```json
{
  "source":"business_history",
  "source_id":"req-002",
  "prompt_id":"由完整prompt计算的SHA-256",
  "messages":[
    {"role":"system","content":"你是企业知识库助手。"},
    {"role":"user","content":"总结本季度退款率上升的主要原因。"}
  ],
  "metadata":{}
}
```

`prompt_id` 由完整 prompt 内容计算，而不是由行号计算。这样重复 prompt 会被识别，
并且同一 prompt 不会跨越训练、验证和测试集。

### 5.3 SFT 数据

教师生成 response 后，SFT JSONL 形如：

```json
{
  "prompt_id":"abc123...",
  "source":"specbench_first_turn",
  "prompt_text":"<s>[INST] 用通俗语言解释什么是推测解码。 [/INST]",
  "response":"推测解码先由较小模型生成候选，再由目标模型并行验证这些候选。",
  "generation_source":"/home/yypan/models/Llama-2-7b-chat-hf"
}
```

训练时：

```text
input_ids = prompt_ids + response_ids + EOS
labels    = -100...-100 + response_ids + EOS
```

prompt 部分 label 为 `-100`，不参与损失；模型只学习教师 response。

### 5.4 KD 数据

KD 数据在 SFT 字段之外保存稀疏教师分布。为便于阅读，下面只展示两个预测位置和
Top-3；实际配置通常使用 Top-32：

```json
{
  "prompt_id":"abc123...",
  "prompt_text":"<s>[INST] ... [/INST]",
  "response":"推测解码先由...",
  "generation_source":"checkpoints/AMD-Llama-135m-specbench-pilot-sft/best",
  "input_ids":[1,518,25580,29962,1234,5678,2],
  "loss_mask":[false,false,false,true,true,true],
  "teacher_topk_ids":[
    [29871,13,450],
    [1234,5678,910]
  ],
  "teacher_topk_logprobs":[
    [-0.20,-2.10,-3.00],
    [-0.35,-1.80,-2.70]
  ],
  "teacher_tail_logprobs":[-1.90,-1.75]
}
```

字段含义：

- `input_ids`：prompt 与 response 的完整 token 序列；
- `loss_mask`：哪些预测位置参与训练；
- `teacher_topk_ids`：教师在每个位置概率最高的 K 个 token；
- `teacher_topk_logprobs`：对应 log probability；
- `teacher_tail_logprobs`：Top-K 之外全部 token 的概率和再取 log。

不保存完整 32000 维 logits，可以显著降低数据体积。

### 5.5 拒绝回放数据

拒绝回放数据额外包含：

```json
{
  "loss_mask":[false,false,true,false,true,false],
  "loss_weights":[1.0,1.0,3.58,1.0,2.0,1.0],
  "replay_tokens":2
}
```

这里只有两个位置参与梯度更新。`loss_weights` 大于 1 表示修复该错误更可能连接前后
正确片段，从而提高连续接受深度。

## 6. 数据集划分

构建 SpecBench pilot：

```bash
./script/distill_build_dataset.sh \
  --config config/distillation/build_specbench_pilot.yaml
```

配置采用：

```yaml
splits:
  train: 0.8
  validation: 0.1
  test: 0.1
```

当前 480 条第一轮请求得到：

```text
train:      381
validation:  53
test:        46
```

划分由 `seed + prompt SHA-256` 决定，具有可复现性。重复记录会在划分前去重。

重要规则：

1. 不得把 test 或 late holdout 作为任何训练脚本的输入；
2. `collect_teacher.py` 默认拒绝文件名为 `test` 的训练输入；
3. 不能根据 test 结果反复调参后仍把它称为“未见测试集”；
4. 真实业务数据应按用户、会话或时间窗口隔离，避免相似请求泄漏。

## 7. 第一阶段：教师响应采集

训练集：

```bash
./script/distill_collect.sh \
  --config config/distillation/collect_sft_specbench_pilot.yaml
```

验证集：

```bash
./script/distill_collect.sh \
  --config config/distillation/collect_sft_validation_specbench_pilot.yaml
```

关键配置：

```yaml
mode: sft
teacher_model: /home/yypan/models/Llama-2-7b-chat-hf
max_samples: 240
generation:
  source: teacher
  max_prompt_tokens: 448
  max_new_tokens: 64
  temperature: 0.0
  top_p: 1.0
```

使用 `temperature: 0.0` 是为了得到确定性的教师轨迹，降低重复实验方差。若真实业务
采用采样解码，可以另建采样训练集，但评测时必须保持基线与新模型解码配置一致。

## 8. 第二阶段：监督微调（SFT）

运行：

```bash
./script/distill_train.sh \
  --config config/distillation/train_sft_specbench_pilot.yaml
```

参考超参数：

```yaml
model: /home/yypan/models/AMD-Llama-135m
batch_size: 8
gradient_accumulation_steps: 4
epochs: 3
learning_rate: 3.0e-5
weight_decay: 0.01
warmup_ratio: 0.05
scheduler: cosine
max_grad_norm: 1.0
gradient_checkpointing: true
```

有效 batch size 为：

```text
batch_size × gradient_accumulation_steps = 8 × 4 = 32
```

SFT 损失为 response token 的标准交叉熵：

```text
L_SFT = -平均值 log p_student(y_t | prompt, y_<t)
```

SFT 的作用：

- 让基础 135M 模型学会目标模型的回答风格；
- 缩小学生生成轨迹与教师轨迹的分布差异；
- 为后续 on-policy KD 提供可用的初始学生模型。

SFT 不是最终目标。它监督教师实际输出 token，却没有完整利用教师对其他 token 的
概率判断，也没有直接优化连续接受深度。

输出目录：

```text
checkpoints/AMD-Llama-135m-specbench-pilot-sft/
  epoch-1/
  epoch-2/
  epoch-3/
  best/
  final/
  training_metrics.jsonl
```

`best` 依据 validation loss 选择。最终是否采用仍以未见接受深度和 Jetson 吞吐为准。

## 9. 第三阶段：on-policy 知识蒸馏

### 9.1 为什么使用学生轨迹

如果只在教师自己生成的文本上训练，学生部署时一旦产生不同 token，就会进入训练中
很少见过的状态。DistillSpec 指出，使用草稿模型生成的 on-policy continuation 更贴近
部署分布。

采集训练 KD：

```bash
./script/distill_collect.sh \
  --config config/distillation/collect_kd_specbench_pilot.yaml
```

配置中的：

```yaml
generation:
  source: student
  temperature: 0.0
```

表示 response 由 SFT 学生贪心生成，随后教师模型对这个完整序列逐位置计算分布。

### 9.2 稀疏 Top-K 分布蒸馏

教师 Top-K 交叉熵与尾部桶损失可写为：

```text
L_KD =
  - Σ_(i∈TopK) p_T(i) log p_S(i)
  - p_T(tail) log p_S(tail)
```

其中：

```text
p_S(tail) = 1 - Σ_(i∈TopK) p_S(i)
```

代码位于 `sparse_topk_distillation_loss()`。

### 9.3 Total Variation Distance

推测解码接受概率与教师、学生分布重叠有关，因此加入近似 TVD：

```text
L_TVD = 1/2 × (
  Σ_(i∈TopK) |p_S(i) - p_T(i)|
  + |p_S(tail) - p_T(tail)|
)
```

代码位于 `sparse_topk_total_variation_loss()`。

### 9.4 教师 Top-1 硬目标

对于贪心验证，最直接的目标是让学生选择教师 Top-1：

```text
L_hard = CrossEntropy(student_logits, teacher_top1)
```

注意这与普通 SFT 不同。SFT target 是轨迹中的实际下一个 token；KD hard target 是
教师在当前轨迹状态下重新计算得到的 Top-1 token。

### 9.5 混合目标

基础 acceptance-KD 配置：

```yaml
hard_loss_weight: 0.35
kd_loss_weight: 0.25
tvd_loss_weight: 0.40
```

总损失：

```text
L = 0.35 L_hard + 0.25 L_KD + 0.40 L_TVD
```

## 10. 多域混合与反过拟合

最终混合训练包含：

```text
240 SpecBench
120 OASST
120 C4
120 WikiText
总计 600 条 on-policy 轨迹
```

准备通用域数据：

```bash
python src/distillation/prepare_builtin_mixture.py \
  --output-dir data/distillation/general_mixture \
  --train-per-source 120 \
  --validation-per-source 20 \
  --holdout-per-source 20 \
  --seed 2026
```

采集通用域 KD：

```bash
./script/distill_collect.sh \
  --config config/distillation/collect_kd_general_mixture.yaml
./script/distill_collect.sh \
  --config config/distillation/collect_kd_validation_general_mixture.yaml
```

组合数据：

```bash
python src/distillation/combine_jsonl.py \
  --input \
  data/distillation/specbench_pilot/kd_train.jsonl \
  data/distillation/general_mixture/kd_train.jsonl \
  --output data/distillation/acceptance_mixed/kd_train.jsonl
```

训练：

```bash
./script/distill_train.sh \
  --config config/distillation/train_acceptance_kd_mixed.yaml
```

多域数据不是为了让 135M 模型回答所有问题都很好，而是防止它只记住 SpecBench 的
模板、词频和回答风格。

## 11. 拒绝位置加权

在普通 KD batch 中，先计算：

```text
rejection_t =
  argmax(student_logits_t) != teacher_top1_t
```

若 `rejection_weight=3`、`window_size=4`，权重为：

```text
拒绝位置：3.0
后一个位置：2.5
后两个位置：2.0
后三个位置：1.5
其他位置：1.0
```

这是 soft emphasis：完整 response 仍参与训练，只是拒绝附近权重更大。

`min_teacher_confidence=0.20` 表示教师 Top-1 概率低于 0.20 的位置不参与该训练目标。
这些位置通常具有较高不确定性，让 135M 模型强行拟合可能浪费容量。

## 12. 拒绝回放

普通拒绝加权仍会在大量已经正确的 token 上计算损失。拒绝回放改为：

1. 选择当前学生 checkpoint；
2. 在教师生成的正确轨迹上运行学生；
3. 找到教师与学生 Top-1 不一致的位置；
4. 仅将这些位置设置为 `loss_mask=true`；
5. 与普通多域 KD 数据混合训练，防止只拟合错误点。

运行均匀拒绝回放：

```bash
./script/distill_rejection_replay.sh
```

关键配置：

```yaml
loss_mask:
  mode: student_rejections
  min_teacher_confidence: 0.20
  window_size: 1
  drop_no_rejections: true
```

`drop_no_rejections=true` 表示若一条 response 中没有符合条件的错误位置，就不写入回放
数据。

## 13. Top-1 Margin Loss

只有教师 token 成为学生 Top-1 才能改善贪心接受。因此加入排序 margin：

```text
L_margin =
  max(0, margin - z_student(teacher_top1) + z_student(best_alternative))
```

其中 `best_alternative` 是除教师 Top-1 外学生 logit 最高的 token。

当：

```text
z_teacher_token >= z_best_alternative + margin
```

该位置 margin loss 为零。

这比单纯降低交叉熵更直接地优化 Top-1 排序，但权重过高可能破坏完整分布，因此必须
与 KD、TVD 和普通数据混合。

## 14. 接受深度增益加权

并非所有错误对平均接受深度的影响相同。

例如学生一致性序列为：

```text
正确 正确 错误 正确 正确 错误
```

修复第三个 token 可以把前后两个正确片段连接起来，收益大于只修复最后一个错误。

实现对每个拒绝位置执行一次反事实计算：

1. 计算原始 capped contiguous acceptance depth 总和；
2. 暂时把一个拒绝位置改为正确；
3. 重新计算深度总和；
4. 两者差值记为 `gain`；
5. 使用：

```text
weight = min(max_weight, 1 + log2(gain))
```

运行第一轮增益回放：

```bash
./script/distill_rejection_replay_gain.sh
```

运行低学习率迭代回放：

```bash
./script/distill_rejection_replay_gain_round2.sh
```

第二轮必须用第一轮的新 checkpoint 重新寻找剩余错误，不能复用旧 mask。当前第二轮
将学习率降为 `5e-7`，并降低最大权重，减少对已经形成的连续正确片段的破坏。

## 15. 最终回放损失

当前低学习率回放配置：

```yaml
hard_loss_weight: 0.25
kd_loss_weight: 0.15
tvd_loss_weight: 0.40
margin_loss_weight: 0.20
top1_margin: 0.05
rejection_weight: 1.25
rejection_window_size: 2
```

总损失：

```text
L =
  0.25 L_hard
  + 0.15 L_KD
  + 0.40 L_TVD
  + 0.20 L_margin
```

每个 token 的最终权重为：

```text
token_weight =
  rejection_window_weight × acceptance_gain_weight
```

所有损失最后除以有效权重总和，避免样本长度和错误数量直接改变梯度尺度。

## 16. 离线评测

运行：

```bash
./script/distill_evaluate.sh \
  --config config/distillation/evaluate_mixed_specbench_pilot_test.yaml
```

评测指标：

### 16.1 Top-1 agreement

```text
教师 Top-1 == 学生 Top-1 的 token 比例
```

这是贪心推测验证的基础命中率。

### 16.2 Teacher Top-1 in student Top-K

教师 Top-1 是否出现在学生概率最高的 K 个 token 中。该指标可用于判断推测树或多分支
策略的潜力，但序列推测主要依赖 Top-1。

### 16.3 Student cross entropy

学生对轨迹实际 token 的标准交叉熵。它反映语言建模拟合，但不等价于接受深度。

### 16.4 Greedy acceptance depth

对每个 token 位置，从当前位置向后统计连续 Top-1 一致长度，并截断到
`max_accept_depth`：

```text
depth_t =
  min(max_depth, 1 + depth_(t+1))  教师与学生一致
  0                               不一致
```

`mean_greedy_accept_depth` 是所有有效位置的平均值。

### 16.5 Acceptance survival

```text
P(depth >= d)
```

例如：

```json
{
  "1":0.5720,
  "2":0.3488,
  "3":0.2205,
  "4":0.1529
}
```

它可直接用于推测序列预计算策略，估计验证可能在不同深度停止的概率。

## 17. 当前实验结果

最终选择：

```text
RTX 4090 checkpoint:
/home/yypan/specedge/checkpoints/
AMD-Llama-135m-rejection-replay-gain-round2/epoch-2

Jetson model:
/home/yypan/models/AMD-Llama-135m-replay-gain-round2
```

最终 `model.safetensors` 的 SHA-256：

```text
426bf56eafd491f6554cf2fa50c9a6b4cf916ede93f3cc29bc0b5414de84d6ab
```

最终一次性 141 条 late holdout：

| 模型 | Top-1 | Top-8 覆盖率 | 平均接受深度 | P(depth >= 2) |
|---|---:|---:|---:|---:|
| Mixed acceptance KD | 55.69% | 82.35% | 1.466 | 33.11% |
| Iterative gain replay | 57.88% | 84.98% | 1.535 | 34.90% |

平均接受深度相对提高 `4.72%`。这说明拒绝回放有效，但不能声称 135M 模型已经达到
平均深度 2。

## 18. 部署到 Jetson

推荐从 4090 复制整个 Hugging Face checkpoint 目录，至少包括：

```text
config.json
generation_config.json
model.safetensors
special_tokens_map.json
tokenizer.json
tokenizer.model
tokenizer_config.json
```

Jetson 当前目录：

```text
/home/yypan/models/AMD-Llama-135m-replay-gain-round2
```

校验模型：

```bash
cd ~/specedge
source .venv/bin/activate
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/home/yypan/models/AMD-Llama-135m-replay-gain-round2"
tokenizer = AutoTokenizer.from_pretrained(path, legacy=False)
model = AutoModelForCausalLM.from_pretrained(path)
print(type(model).__name__)
print(len(tokenizer))
print(sum(p.numel() for p in model.parameters()))
PY
```

预期参数量：

```text
134105856
```

## 19. 接入推测序列实验

使用独立配置，不修改原基线：

```bash
./script/client_host.sh \
  -f config/specedge_4090_jetson_sequence_depth_replay.yaml
```

关键配置：

```yaml
client:
  draft_model: /home/yypan/models/AMD-Llama-135m-replay-gain-round2
  max_n_beams: 1
  max_branch_width: 1
  initial_draft:
    mode: fixed
    structure: sequence
```

验证集重新标定后的 survival：

```yaml
acceptance_survival:
  [1.0, 0.5720, 0.3488, 0.2205, 0.1529, 0.1094, 0.0807, 0.0600]
```

不要使用 test 或 late holdout 的 survival 配置策略，否则会把测试信息泄漏到系统决策。

## 20. 推荐实验矩阵

固定以下条件：

```text
相同 60 个请求
相同 req_idx 顺序
相同 max_new_tokens
相同随机种子
相同目标模型
相同 4090/Jetson 功率模式
相同网络路径
```

至少比较：

| 实验 | 草稿模型 | 推测结构 | 预计算 |
|---|---|---|---|
| 云端自回归 | 无 | 无 | 无 |
| 原始 SpecEdge | TinyLlama 或旧 AMD | 树 | 原始 |
| AMD 基线 | 原始 AMD-135M | 序列 | 关闭 |
| 蒸馏模型 | Replay AMD-135M | 序列 | 关闭 |
| 完整方案 | Replay AMD-135M | 序列 | sequence-depth |

必须同时报告：

- 端到端 tokens/s；
- TTFT、TPOT 或 cycle latency；
- 平均接受深度；
- `P(depth >= 2/4)`；
- Jetson draft wall/GPU time；
- 4090 验证 wall/GPU time；
- 网络等待时间；
- 预计算命中率和浪费节点；
- 云实例墙钟成本；
- GPU 活跃推理时间。

## 21. 什么时候停止继续微调

满足任一条件应停止在当前数据上继续训练：

- late holdout 不再提升；
- 跨域 holdout 下降超过预设阈值；
- Top-1 上升但平均连续深度不升；
- Jetson 草稿耗时增长抵消接受深度收益；
- 已根据同一个测试集多次选择 epoch 或超参数。

若目标仍是平均深度 2，推荐下一步：

1. 增加 5k 到 20k 条不与测试集重合的真实业务请求；
2. 按时间划分训练和未来 holdout；
3. 重新执行一次 SFT、on-policy KD 和最多两轮拒绝回放；
4. 若仍低于 2，再测试 tokenizer 兼容的 300M 到 500M 草稿模型；
5. 最终以 Jetson 端到端吞吐和成本决定，而不是只看离线深度。

## 22. 常见问题

### 22.1 为什么不能直接在 test 上采集教师 logits 训练？

这样会泄漏最终评测答案。即使没有直接训练 response，只要根据 test 的拒绝位置更新
模型，也已经使用了测试信息。

### 22.2 为什么 SFT 后还要 KD？

SFT 只监督一个实际 token。KD 同时提供教师对多个候选 token 的相对概率，更适合
对齐完整分布和接受概率。

### 22.3 为什么不用完整教师 logits？

完整 logits 数据量很大。Top-K 加 tail bucket 保留主要概率质量，同时显著降低磁盘
和 I/O 开销。

### 22.4 为什么训练 loss 降低但接受深度可能不升？

平均分布误差下降不保证 Top-1 排序改变，也不保证错误位置形成更长连续片段。因此需要
Top-1 margin、拒绝回放和接受增益权重。

### 22.5 模型越大是否一定更快？

不是。更大模型通常提高接受率，但也增加 Jetson 每个草稿 token 的延迟。必须比较：

```text
节省的4090验证轮数和网络等待
是否大于
增加的Jetson草稿时间
```

### 22.6 模型与训练数据是否提交 Git？

不提交。以下目录被 `.gitignore` 排除：

```text
data/distillation/
checkpoints/
result/
```

Git 只保存代码、配置、脚本和文档。模型在 4090 与 Jetson 上分别保存，并通过
SHA-256 校验一致性。
