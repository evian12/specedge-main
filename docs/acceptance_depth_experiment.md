# Acceptance-Depth Distillation

This experiment improves the AMD-Llama-135M drafter without changing the
target model or speculative verification semantics.

## Motivation

The implementation follows three findings from recent speculative decoding
research:

1. [DistillSpec](https://arxiv.org/abs/2310.08461) shows that draft-generated
   on-policy prefixes and a decoding-appropriate divergence are important.
2. [AdaSPEC](https://arxiv.org/abs/2510.19779) shows that uniformly fitting all
   tokens wastes limited draft-model capacity; selective token filtering can
   improve acceptance.
3. [Speculative Decoding Across Languages](https://arxiv.org/abs/2605.30580)
   shows that task-specific fine-tuning can improve its training task while
   generalizing poorly to a held-out task. The experiment therefore requires
   both an untouched SpecBench test split and a cross-domain holdout.
4. [Draft-OPD](https://arxiv.org/abs/2605.29343) identifies drafter error
   positions on target-model trajectories and replays those positions instead
   of repeatedly training on already-correct tokens.

## Objective

The KD objective combines:

- teacher Top-1 cross entropy;
- sparse Top-K distribution distillation;
- sparse total variation distance, which directly increases target/draft
  distribution overlap;
- rejection-window weighting around positions where the current drafter and
  teacher choose different Top-1 tokens;
- teacher-confidence filtering to avoid spending 135M-model capacity on
  unstable teacher targets.

The next round adds two opt-in terms:

- teacher-trajectory rejection replay, where the loss mask contains only
  confident positions at which the selected student has the wrong Top-1;
- a Top-1 ranking margin that directly pushes the teacher token above the
  student's strongest alternative.
- an optional acceptance-gain weight that prioritizes a rejected token when
  correcting it would join long correct runs on both sides.

This is intentionally different from increasing `rejection_weight`. The old
weighting still normalizes a loss over the complete response. Replay creates a
separate hard-example record whose entire optimization signal comes from
states that currently terminate greedy acceptance.

Acceptance-gain weighting estimates the change in the exact offline metric:
it flips one rejection to an agreement, recomputes capped contiguous depths,
and uses `1 + log2(gain)` up to `max_weight`. It remains a training-only
heuristic; untouched holdouts and Jetson throughput decide whether it is kept.

The rejection window uses weights that decay after a rejection. With weight
`3` and window size `4`, the relative weights are:

```text
rejection position: 3.0
next position:      2.5
next position:      2.0
next position:      1.5
other positions:    1.0
```

## Data Isolation

The final model uses 600 on-policy trajectories:

- 240 SpecBench training prompts;
- 120 OASST prompts;
- 120 C4 prompts;
- 120 WikiText prompts.

The following are never used for gradient updates:

- 46-prompt SpecBench test split;
- 60-prompt cross-domain holdout containing 20 deterministic samples each
  from OASST, C4, and WikiText;
- separate mixed-domain validation records.

The cross-domain mixture builder explicitly removes the exact holdout prompts
before selecting training and validation samples.

## Results

All values are greedy acceptance-depth proxies capped at seven tokens.

| Evaluation | SFT baseline | Acceptance KD | Relative change |
|---|---:|---:|---:|
| SpecBench validation | 1.353 | 1.417 | +4.73% |
| SpecBench untouched test | 1.320 | 1.405 | +6.43% |
| Cross-domain holdout | 1.856 | 1.879 | +1.25% |

On the untouched SpecBench test split:

| Metric | SFT baseline | Acceptance KD |
|---|---:|---:|
| Top-1 agreement | 51.83% | 54.47% |
| Top-8 teacher coverage | 80.14% | 82.02% |
| `P(depth >= 2)` | 30.05% | 31.86% |
| `P(depth >= 4)` | 12.59% | 13.63% |

The single-domain acceptance-aware model was rejected: it improved SpecBench
but reduced cross-domain mean acceptance depth from `1.856` to `1.833`.
Adding multi-domain on-policy replay removed that regression.

## Rejection-Replay Results

Three controlled stages were run from the mixed acceptance-KD checkpoint:

| Stage | SpecBench diagnostic test | Cross-domain holdout |
|---|---:|---:|
| Mixed acceptance KD | 1.405 | 1.879 |
| Uniform rejection replay | 1.410 | 1.912 |
| Acceptance-gain replay | 1.472 | 1.911 |
| Low-LR iterative replay | 1.504 | 1.926 |

Because the 46-request diagnostic test was inspected after multiple stages, it
is no longer treated as the final untouched result. The original training
split contained 381 prompts, while every training and replay collector used
only its first 240. The remaining 141 prompts formed a one-time late holdout:

| Model | Top-1 | Top-8 coverage | Mean depth | `P(depth >= 2)` |
|---|---:|---:|---:|---:|
| Mixed acceptance KD | 55.69% | 82.35% | 1.466 | 33.11% |
| Iterative gain replay | 57.88% | 84.98% | 1.535 | 34.90% |

The final relative mean-depth gain on this late holdout is `4.72%`. This is a
real improvement, but it does not support a claim that a 135M drafter has
reached mean depth two.

## Selected Checkpoint

The selected checkpoint on the RTX 4090 host is:

```text
/home/yypan/specedge/checkpoints/AMD-Llama-135m-rejection-replay-gain-round2/epoch-2
```

Offline acceptance is only a proxy. The final deployment decision must compare
this checkpoint and the SFT baseline on Jetson using identical prompts,
initial draft depth, proactive policy, random seed, and token budget.

## Next-Round Gate

Run `script/distill_rejection_replay.sh` on the RTX 4090, then evaluate every
epoch on both untouched sets. Keep a checkpoint only when all conditions hold:

1. SpecBench test mean greedy acceptance depth increases;
2. cross-domain holdout does not regress by more than `0.02`;
3. the gain is visible in `P(depth >= 2)`, not only in rare long runs;
4. Jetson end-to-end throughput improves under an identical SpecEdge config.

The target `mean depth >= 2` is a research goal, not a guaranteed outcome for a
135M model. If rejection replay saturates below it, the next controlled change
is model capacity, for example a tokenizer-compatible 300M-class drafter,
rather than repeatedly fitting the same 600 trajectories.

One additional replay round is allowed for diagnosis. It must regenerate the
error mask with the newly selected checkpoint, halve the learning rate, and
reduce the maximum gain weight. Stop after that round if untouched evaluation
stagnates; repeated passes over a fixed prompt set are not evidence of useful
deployment improvement.
