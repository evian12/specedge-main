import asyncio
import time
from typing import Optional

import numpy as np
import torch

import log
import util
from config import SpecEdgeClientConfig as config
from specedge.client.initial_draft_policy import (
    InitialDraftDecision,
    LinUCBInitialDraftPolicy,
    initial_depth_after_proactive_reuse,
)
from specedge.client.proactive import (
    ProactiveDraftResult,
    SpecExecProactiveDraft,
)
from specedge.client.proactive_policy import AdaptiveProactivePolicy
from specedge.network.grpc import GrpcClientController
from specedge.tree import Tree


class SpecExecClient:
    _shared_adaptive_policy: Optional[AdaptiveProactivePolicy] = None
    _shared_full_depth_acceptance: Optional[float] = None
    _shared_initial_draft_policy: Optional[LinUCBInitialDraftPolicy] = None

    def __init__(
        self,
        engine,
        tokenizer,
        prompt: str,
        max_len: int,
    ) -> None:
        # logging
        self._logger = log.get_logger()
        self._result_logger = log.get_result_logger()

        self._logger.debug("Initializing SpecExecClient")

        self._optimization = config.optimization
        self._draft_forward_time_mode = (
            "no-sync" if self._optimization >= 2 else "event"
        )
        self._target_time_mode = "no-sync" if self._optimization >= 2 else "sync"

        self._device = config.device
        self._dtype = config.dtype

        self._max_n_beams = config.max_n_beams
        self._max_beam_len = config.max_beam_len
        self._max_branch_width = config.max_branch_width
        self._max_budget = config.max_budget
        self._initial_draft_mode = config.initial_draft_mode
        self._initial_draft_structure = config.initial_draft_structure

        self._proactive_type = config.proactive_type
        self._proactive_mode = config.proactive_mode
        self._proactive_path_policy = config.proactive_path_policy

        self._max_new_tokens = config.max_new_tokens
        self._client_idx = config.client_idx

        self._verify_configs()

        self._engine = engine
        self._tokenizer = tokenizer
        self._engine.reset()

        self._prompt = prompt
        self._prefix_tokens = self._tokenizer.encode(prompt, return_tensors="pt").to(
            self._device
        )[: config.max_len]
        self._num_original_tokens = self._prefix_tokens.numel()
        self._max_len = max_len

        self._tree = Tree(
            prefix_tokens=self._prefix_tokens,
            device=self._device,
            dtype=self._dtype,
            max_len=self._engine.max_len,
        )
        self._validator = GrpcClientController(host=config.host, device=self._device)

        self._proactive_client: Optional[SpecExecProactiveDraft] = None
        self._adaptive_policy: Optional[AdaptiveProactivePolicy] = None
        self._initial_draft_policy: Optional[LinUCBInitialDraftPolicy] = None
        self._previous_proactive_draft = False
        self._proactive_draft = False
        self._reused_proactive_depth = 0
        if self._proactive_type != "disabled":
            self._proactive_client = SpecExecProactiveDraft(
                tree=self._tree,
                engine=self._engine,
                max_len=self._max_len,
            )
        if (
            self._proactive_path_policy == "deepest_multi"
            and SpecExecClient._shared_full_depth_acceptance is None
        ):
            SpecExecClient._shared_full_depth_acceptance = (
                config.proactive_multi_full_depth_prior
            )

        if self._proactive_mode == "adaptive":
            if SpecExecClient._shared_adaptive_policy is None:
                SpecExecClient._shared_adaptive_policy = AdaptiveProactivePolicy(
                    max_depth=config.proactive_max_beam_len,
                    ewma_alpha=config.proactive_adaptive_ewma_alpha,
                    min_alignment_rate=(
                        config.proactive_adaptive_min_alignment_rate
                    ),
                    warmup_cycles=config.proactive_adaptive_warmup_cycles,
                    exploration_interval=(
                        config.proactive_adaptive_exploration_interval
                    ),
                    safety_margin_ms=(
                        config.proactive_adaptive_safety_margin_ms
                    ),
                    uncertainty_scale=(
                        config.proactive_adaptive_uncertainty_scale
                    ),
                )
            self._adaptive_policy = SpecExecClient._shared_adaptive_policy
        if self._initial_draft_mode == "linucb":
            if SpecExecClient._shared_initial_draft_policy is None:
                SpecExecClient._shared_initial_draft_policy = (
                    LinUCBInitialDraftPolicy(
                        candidate_depths=(
                            config.initial_draft_candidate_depths
                        ),
                        max_depth=self._max_beam_len,
                        exploration_weight=(
                            config.initial_draft_exploration_weight
                        ),
                        warmup_per_depth=(
                            config.initial_draft_warmup_per_depth
                        ),
                        forced_exploration_interval=(
                            config.initial_draft_forced_exploration_interval
                        ),
                        ridge_lambda=config.initial_draft_ridge_lambda,
                        reward_clip=config.initial_draft_reward_clip,
                        ewma_alpha=config.initial_draft_ewma_alpha,
                        seed=config.seed,
                    )
                )
            self._initial_draft_policy = (
                SpecExecClient._shared_initial_draft_policy
            )

    def _verify_configs(self):
        if self._initial_draft_mode not in ["fixed", "linucb"]:
            raise ValueError(
                f"Invalid initial_draft mode: {self._initial_draft_mode}"
            )
        if self._initial_draft_structure not in ["tree", "sequence"]:
            raise ValueError(
                "Invalid initial_draft structure: "
                f"{self._initial_draft_structure}"
            )
        if self._proactive_type not in ["included", "excluded", "disabled"]:
            raise ValueError(f"Invalid proactive_type: {self._proactive_type}")
        if self._proactive_mode not in [
            "baseline",
            "interruptible",
            "adaptive",
        ]:
            raise ValueError(f"Invalid proactive_mode: {self._proactive_mode}")
        if self._proactive_path_policy not in [
            "single_best",
            "deepest_multi",
            "sequence_depth",
        ]:
            raise ValueError(
                "Invalid proactive_path_policy: "
                f"{self._proactive_path_policy}"
            )
        if config.proactive_multi_max_deepest_leaves <= 0:
            raise ValueError("multi.max_deepest_leaves must be positive")
        if config.proactive_multi_min_bonus_per_leaf <= 0:
            raise ValueError("multi.min_bonus_per_leaf must be positive")
        if (
            config.proactive_multi_max_bonus_per_leaf
            < config.proactive_multi_min_bonus_per_leaf
        ):
            raise ValueError(
                "multi.max_bonus_per_leaf must be >= min_bonus_per_leaf"
            )
        if config.proactive_multi_max_roots <= 0:
            raise ValueError("multi.max_roots must be positive")
        if config.proactive_multi_max_roots > config.proactive_max_budget:
            raise ValueError(
                "multi.max_roots must not exceed proactive max_budget"
            )
        if config.proactive_multi_min_root_probability < 0.0:
            raise ValueError(
                "multi.min_root_probability must be non-negative"
            )
        if config.proactive_multi_leaf_temperature <= 0.0:
            raise ValueError("multi.leaf_temperature must be positive")
        if not 0.0 <= config.proactive_multi_full_depth_prior <= 1.0:
            raise ValueError("multi.full_depth_prior must be in [0, 1]")
        if not 0.0 < config.proactive_multi_acceptance_ewma_alpha <= 1.0:
            raise ValueError(
                "multi.acceptance_ewma_alpha must be in (0, 1]"
            )
        if any(
            not 0.0 <= coverage <= 1.0
            for coverage in config.proactive_multi_depth_probability_coverage
        ):
            raise ValueError(
                "multi.depth_probability_coverage values must be in [0, 1]"
            )
        if any(
            later > earlier
            for earlier, later in zip(
                config.proactive_multi_depth_probability_coverage,
                config.proactive_multi_depth_probability_coverage[1:],
            )
        ):
            raise ValueError(
                "multi.depth_probability_coverage must be non-increasing"
            )
        survival = config.proactive_sequence_acceptance_survival
        if not survival or abs(survival[0] - 1.0) > 1e-6:
            raise ValueError(
                "sequence.acceptance_survival must start with 1.0"
            )
        if any(not 0.0 <= probability <= 1.0 for probability in survival):
            raise ValueError(
                "sequence.acceptance_survival values must be in [0, 1]"
            )
        if any(
            later > earlier
            for earlier, later in zip(survival, survival[1:])
        ):
            raise ValueError(
                "sequence.acceptance_survival must be non-increasing"
            )
        if config.proactive_sequence_max_bonus_per_depth <= 0:
            raise ValueError(
                "sequence.max_bonus_per_depth must be positive"
            )
        if config.proactive_sequence_max_roots <= 0:
            raise ValueError("sequence.max_roots must be positive")
        if (
            config.proactive_sequence_max_roots
            > config.proactive_max_budget
        ):
            raise ValueError(
                "sequence.max_roots must not exceed proactive max_budget"
            )
        if config.proactive_sequence_min_root_probability < 0.0:
            raise ValueError(
                "sequence.min_root_probability must be non-negative"
            )
        sequence_coverage = (
            config.proactive_sequence_depth_probability_coverage
        )
        if any(
            not 0.0 <= coverage <= 1.0
            for coverage in sequence_coverage
        ):
            raise ValueError(
                "sequence.depth_probability_coverage values must be "
                "in [0, 1]"
            )
        if any(
            later > earlier
            for earlier, later in zip(
                sequence_coverage,
                sequence_coverage[1:],
            )
        ):
            raise ValueError(
                "sequence.depth_probability_coverage must be "
                "non-increasing"
            )

    async def _run_proactive_draft(
        self,
        target_result: asyncio.Task,
        request_start: float,
    ) -> ProactiveDraftResult:
        if self._proactive_client is None:
            return ProactiveDraftResult(
                skipped_reason="disabled",
                path_policy=self._proactive_path_policy,
            )

        if self._proactive_mode == "baseline":
            self._proactive_client.draft(
                SpecExecClient._shared_full_depth_acceptance
            )
            return self._proactive_client.last_result or ProactiveDraftResult(
                skipped_reason="no_candidate"
            )

        planned_depth = config.proactive_max_beam_len
        policy_reason = None
        setup_deadline_check = None
        if self._adaptive_policy is not None:
            planned_depth, policy_reason = self._adaptive_policy.begin_cycle()
            if planned_depth == 0:
                return ProactiveDraftResult(
                    planned_depth=0,
                    skipped_reason="adaptive_skip",
                    policy_reason=policy_reason,
                    path_policy=self._proactive_path_policy,
                )

        if target_result.done():
            return ProactiveDraftResult(
                planned_depth=planned_depth,
                stopped_by_response=True,
                skipped_reason="response_ready_before_proactive",
                policy_reason=policy_reason,
                path_policy=self._proactive_path_policy,
            )

        if self._adaptive_policy is not None:
            setup_decision = self._adaptive_policy.can_start_setup(
                (time.perf_counter() - request_start) * 1000
            )
            setup_deadline_check = {
                "stage": "setup",
                "allowed": setup_decision.allowed,
                "reason": setup_decision.reason,
                "remaining_ms": setup_decision.remaining_ms,
                "predicted_cost_ms": setup_decision.predicted_cost_ms,
            }
            if not setup_decision.allowed:
                return ProactiveDraftResult(
                    planned_depth=planned_depth,
                    skipped_reason="adaptive_deadline",
                    policy_reason=setup_decision.reason,
                    deadline_checks=[setup_deadline_check],
                    path_policy=self._proactive_path_policy,
                )

        setup_start = time.perf_counter()
        session = self._proactive_client.start_session(
            planned_depth,
            SpecExecClient._shared_full_depth_acceptance,
        )
        if setup_deadline_check is not None:
            session.result.deadline_checks.append(setup_deadline_check)
        if self._device.type == "cuda":
            await asyncio.to_thread(torch.cuda.synchronize, self._device)
        setup_ms = (time.perf_counter() - setup_start) * 1000
        session.result.setup_ms = setup_ms
        if self._adaptive_policy is not None:
            self._adaptive_policy.observe_setup(setup_ms)
        await asyncio.sleep(0)
        while session.can_step:
            if target_result.done():
                result = session.finish(stopped_by_response=True)
                result.policy_reason = policy_reason
                return result

            layer_index = session.result.executed_depth
            while True:
                batch_width = session.next_batch_width
                if batch_width == 0:
                    break
                if self._adaptive_policy is None:
                    break

                layer_decision = self._adaptive_policy.can_start_layer(
                    layer_index=layer_index,
                    request_elapsed_ms=(
                        time.perf_counter() - request_start
                    )
                    * 1000,
                    batch_width=batch_width,
                )
                pruned_root = False
                if not layer_decision.allowed:
                    pruned_root = session.prune_lowest_priority_root()
                session.result.deadline_checks.append(
                    {
                        "stage": f"layer_{layer_index}",
                        "allowed": layer_decision.allowed,
                        "reason": layer_decision.reason,
                        "remaining_ms": layer_decision.remaining_ms,
                        "predicted_cost_ms": (
                            layer_decision.predicted_cost_ms
                        ),
                        "batch_width": batch_width,
                        "pruned_root": pruned_root,
                    }
                )
                if layer_decision.allowed:
                    break
                if not pruned_root:
                    result = session.finish()
                    result.policy_reason = layer_decision.reason
                    return result

            batch_width = session.next_batch_width
            if batch_width == 0:
                break

            start_event = None
            end_event = None
            if self._device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            step_start = time.perf_counter()
            session.step()
            if self._device.type == "cuda":
                end_event.record()
                await asyncio.to_thread(torch.cuda.synchronize, self._device)
            wall_ms = (time.perf_counter() - step_start) * 1000
            gpu_ms = (
                start_event.elapsed_time(end_event)
                if start_event is not None and end_event is not None
                else None
            )
            session.result.layer_wall_ms.append(wall_ms)
            session.result.layer_gpu_ms.append(gpu_ms)
            if self._adaptive_policy is not None:
                self._adaptive_policy.observe_step(
                    layer_index=layer_index,
                    wall_ms=wall_ms,
                    gpu_ms=gpu_ms,
                    batch_width=batch_width,
                )
            await asyncio.sleep(0)

        result = session.finish()
        result.policy_reason = policy_reason
        return result

    async def generate(self, req_idx: int):
        """
        Generate a sequence using SpecExec up to max_new_tokens.
        """

        self._logger.info("Generating sequence req_idx=%d", req_idx)

        util.set_seed(config.seed)
        step_idx = 0

        # Prefill phase
        self._logger.debug("Prefill phase: req_idx=%d, step_idx=%d", req_idx, step_idx)
        warmup_tokens = await self._cycle(
            req_idx,
            step_idx,
            prefill=True,
            max_fresh_tokens=self._max_new_tokens,
        )
        self._prefix_tokens = torch.cat([self._prefix_tokens, warmup_tokens], dim=-1)

        step_idx = 1
        eos_flag = bool(
            (warmup_tokens == self._tokenizer.eos_token_id).any().item()
        )

        # speculative decoding phase
        while (
            self._prefix_tokens.numel()
            < self._max_new_tokens + self._num_original_tokens
            and not eos_flag
        ):
            self._logger.debug(
                "Speculative Decoding phase: req_idx=%d, step_idx=%d", req_idx, step_idx
            )
            remaining_tokens = (
                self._max_new_tokens
                - (self._prefix_tokens.numel() - self._num_original_tokens)
            )
            fresh_tokens = await self._cycle(
                req_idx,
                step_idx,
                max_fresh_tokens=remaining_tokens,
            )

            eos_positions = (fresh_tokens == self._tokenizer.eos_token_id).nonzero()
            if eos_positions.numel() > 0:
                eos_idx = eos_positions[0, 0].item()
                fresh_tokens = fresh_tokens[: eos_idx + 1]
                eos_flag = True

            self._prefix_tokens = torch.cat([self._prefix_tokens, fresh_tokens], dim=-1)
            step_idx += 1

        if eos_flag:
            self._logger.debug("EOS token found.")
        else:
            self._logger.debug("Max new tokens reached.")

        self._logger.info("Finished generating sequence req_idx=%d", req_idx)
        self._logger.info(
            "Generated sequence: \n%s",
            self._tokenizer.decode(self._prefix_tokens[0], skip_special_tokens=True),
        )

    async def _cycle(
        self,
        req_idx: int,
        step_idx: int,
        prefill=False,
        max_fresh_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        cycle_start = time.perf_counter()
        initial_draft_decision: Optional[InitialDraftDecision] = None
        selected_depth = self._max_beam_len
        if self._initial_draft_policy is not None and not prefill:
            initial_draft_decision = (
                self._initial_draft_policy.select_depth(
                    context_ratio=self._prefix_tokens.numel()
                    / max(1, self._max_len)
                )
            )
            selected_depth = initial_draft_decision.depth

        with util.Timing(device=self._device, mode="sync") as draft_t:
            draft_stats = self._grow_tree(
                prefill,
                selected_depth=selected_depth,
            )

        with util.Timing(device=self._device, mode="sync") as target_t:
            fresh_token_ids, target_stats = await self._validate_tree(req_idx, prefill)

        if max_fresh_tokens is not None:
            fresh_token_ids = fresh_token_ids[:max_fresh_tokens]

        eos_positions = (fresh_token_ids == self._tokenizer.eos_token_id).nonzero()
        if eos_positions.numel() > 0:
            fresh_token_ids = fresh_token_ids[: eos_positions[0, 0].item() + 1]

        target_stats["num_accepted_tokens"] = fresh_token_ids.numel()
        cycle_ms = (time.perf_counter() - cycle_start) * 1000
        initial_draft_log = {
            "mode": self._initial_draft_mode,
            "structure": self._initial_draft_structure,
            "selected_depth": selected_depth,
            "selection_reason": (
                initial_draft_decision.reason
                if initial_draft_decision is not None
                else ("prefill" if prefill else "fixed")
            ),
            "executed_depth": draft_stats["executed_depth"],
            "reused_proactive_depth": draft_stats[
                "reused_proactive_depth"
            ],
            "node_count": draft_stats["node_count"],
            "accepted_draft_depth": max(
                0, fresh_token_ids.numel() - 1
            ),
            "cycle_ms": cycle_ms,
            "reward": None,
            "features": (
                initial_draft_decision.features
                if initial_draft_decision is not None
                else None
            ),
            "scores": (
                {
                    str(depth): score
                    for depth, score in initial_draft_decision.scores.items()
                }
                if initial_draft_decision is not None
                else {}
            ),
        }
        if (
            self._initial_draft_policy is not None
            and initial_draft_decision is not None
        ):
            proactive_execution = target_stats["proactive_execution"]
            reward = self._initial_draft_policy.observe(
                decision=initial_draft_decision,
                accepted_tokens=fresh_token_ids.numel(),
                cycle_ms=cycle_ms,
                draft_ms=draft_t.elapsed,
                response_ms=proactive_execution.get(
                    "response_received_ms"
                ),
                node_count=draft_stats["node_count"],
                max_budget=self._max_budget,
                proactive_hit=target_stats["proactive"],
                proactive_depth=int(
                    proactive_execution.get("executed_depth", 0)
                ),
                proactive_max_depth=config.proactive_max_beam_len,
            )
            initial_draft_log["reward"] = reward
            initial_draft_log["controller"] = (
                self._initial_draft_policy.stats()
            )
            initial_draft_log["feature_names"] = (
                self._initial_draft_policy.feature_names
            )

        self._result_logger.log(
            {
                "client_idx": self._client_idx,
                "req_idx": req_idx,
                "step_idx": step_idx,
                "draft": {
                    "forward": draft_stats["forward_t"],
                    "end_to_end": draft_t.elapsed,
                    "initial_draft": initial_draft_log,
                },
                "target": {
                    "client_preprocess": target_stats["preprocess_t"],
                    "client_wait": target_stats["wait_t"],
                    "client_postprocess": target_stats["postprocess_t"],
                    "end_to_end": target_t.elapsed,
                    "prefill": target_stats["prefill"],
                    "proactive": target_stats["proactive"],
                    "prev_proactive": target_stats["previous_proactive"],
                    "proactive_execution": target_stats[
                        "proactive_execution"
                    ],
                },
                "num_accepted_tokens": target_stats["num_accepted_tokens"],
            }
        )

        return fresh_token_ids

    def _grow_tree(
        self,
        prefill: bool,
        selected_depth: Optional[int] = None,
    ):
        self._logger.debug("Growing tree")

        # draft forward times
        draft_forward_times = []

        max_beam_len = (
            self._max_beam_len
            if selected_depth is None
            else min(self._max_beam_len, max(0, selected_depth))
        )
        max_beam_len, reused_proactive_depth = (
            initial_depth_after_proactive_reuse(
                max_beam_len,
                proactive_hit=self._proactive_draft,
                reused_depth=self._reused_proactive_depth,
                proactive_type=self._proactive_type,
                path_policy=self._proactive_path_policy,
            )
        )

        if torch.where(self._tree.status == self._tree.CANDIDATE)[0].numel() == 0:
            max_beam_len = 0

        for cnt in range(max_beam_len):
            self._logger.debug("Growing tree: %d / %d", cnt, max_beam_len)

            logits, beam_indices, beam_positions, beam_scores, draft_forward_t = (
                self._process_candidates(prefill)
            )
            prefill = False

            draft_forward_times.append(draft_forward_t)

            if self._initial_draft_structure == "sequence":
                (
                    next_beam_ids,
                    next_beam_positions,
                    next_beam_indices,
                    beam_logprobs,
                ) = self._get_next_sequence(
                    logits=logits,
                    beam_indices=beam_indices,
                    beam_positions=beam_positions,
                    beam_scores=beam_scores,
                )
            else:
                (
                    next_beam_ids,
                    next_beam_positions,
                    next_beam_indices,
                    beam_logprobs,
                ) = self._get_next_beams(
                    logits=logits,
                    beam_indices=beam_indices,
                    beam_positions=beam_positions,
                    beam_scores=beam_scores,
                )

            if next_beam_ids.numel() == 0:
                self._logger.debug("No more beams to grow")
                break

            if (
                self._tree.end - self._tree.prefix_len >= self._max_budget
                and not self._check_new_token_in_budget(beam_logprobs)
            ):
                self._logger.debug("Max budget reached. early stopping")
                break

            self._tree.add(
                token_ids=next_beam_ids,
                token_positions=next_beam_positions,
                parent_indices=next_beam_indices,
                logprobs=beam_logprobs,
            )

        if self._tree.end - self._tree.prefix_len >= self._max_budget:
            self._logger.debug("Trimming tree")
            self._trim_by_budget()

        return {
            "forward_t": draft_forward_times,
            "executed_depth": len(draft_forward_times),
            "node_count": self._tree.end - self._tree.prefix_len,
            "reused_proactive_depth": reused_proactive_depth,
        }

    def _get_next_sequence(
        self,
        logits: torch.Tensor,
        beam_indices: torch.Tensor,
        beam_positions: torch.Tensor,
        beam_scores: torch.Tensor,
    ):
        """Choose one globally most likely continuation token."""
        logprobs = torch.log_softmax(logits, dim=-1)
        best_logprobs, best_token_ids = logprobs.max(dim=-1)
        scores = beam_scores + np.log(0.9) + best_logprobs
        best_beam_offset = scores.argmax()
        return (
            best_token_ids[best_beam_offset].view(1),
            (beam_positions[best_beam_offset] + 1).view(1),
            beam_indices[best_beam_offset].view(1),
            scores[best_beam_offset].view(1),
        )

    def _process_candidates(self, warmup: bool):
        self._logger.debug("Processing candidates")
        candidate_indices = torch.where(
            self._tree.status[: self._tree.end] == self._tree.CANDIDATE
        )[0]

        if candidate_indices.numel() > self._max_n_beams:
            self._logger.debug("Choosing top %d candidates", self._max_n_beams)
            cumulative_logprobs = self._tree.logprobs[candidate_indices]
            top_k_indices = cumulative_logprobs.topk(
                k=self._max_n_beams, sorted=False
            ).indices
            candidate_indices = candidate_indices[top_k_indices]
            candidate_indices, _ = candidate_indices.sort()

        if warmup:
            prefill_input_indices = torch.arange(
                candidate_indices.min().item(), device=self._device
            )
            prefill_input_ids = self._tree.tokens[prefill_input_indices].unsqueeze(0)
            prefill_position_ids = self._tree.positions[
                prefill_input_indices
            ].unsqueeze(0)
            prefill_cache_seq_indices = prefill_input_indices
            prefill_attention_mask = self._tree.amask[..., prefill_input_indices, :]

            self._engine.prefill(
                input_ids=prefill_input_ids,
                position_ids=prefill_position_ids,
                batch_idx=0,
                cache_seq_indices=prefill_cache_seq_indices,
                attention_mask=prefill_attention_mask,
            )

        input_indices = candidate_indices

        input_ids = self._tree.tokens[input_indices].unsqueeze(0)
        position_ids = self._tree.positions[input_indices].unsqueeze(0)
        cache_seq_indices = input_indices
        cache_batch_indices = torch.full_like(
            cache_seq_indices, 0, dtype=torch.long, device=self._device
        )
        attention_mask = self._tree.amask[..., input_indices, :]

        with util.Timing(device=self._device, mode=self._draft_forward_time_mode) as t:
            logits = self._engine.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
            )

        self._tree.status[candidate_indices] = self._tree.PROCESSED
        beam_scores = self._tree.logprobs[candidate_indices]
        beam_positions = self._tree.positions[candidate_indices]
        logits = logits[0, -candidate_indices.size(-1) :, :]

        return (logits, candidate_indices, beam_positions, beam_scores, t.elapsed)

    def _get_next_beams(
        self,
        logits: torch.Tensor,
        beam_indices: torch.Tensor,
        beam_positions: torch.Tensor,
        beam_scores: torch.Tensor,
    ):
        self._logger.debug("Getting next beams")
        DECAY_FACTOR = np.log(0.9)

        logprobs = torch.log_softmax(logits, dim=-1)  # shape: [n_beams, vocab_size]
        logprobs_k = logprobs.topk(
            k=self._max_branch_width, dim=-1, sorted=False
        )  # shape: [n_beams, max_branch_width]
        leaves_ids = logprobs_k.indices
        leaves_probs = logprobs_k.values

        flat_incoming_probs = (
            beam_scores.unsqueeze(-1) + DECAY_FACTOR + leaves_probs
        ).flatten()
        flat_incoming_ids = leaves_ids.flatten()

        joint_probs = torch.concat(
            [
                self._tree.logprobs[self._tree.prefix_len : self._tree.end],
                flat_incoming_probs,
            ]
        )

        if (
            joint_probs.size(-1) > self._max_budget
            or joint_probs.size(-1) + (self._tree.end - self._tree.prefix_len)
            > self._max_len
        ):
            min_joint_prob = joint_probs.topk(
                k=self._max_budget, sorted=False, dim=-1
            ).values.min()

            flat_best_mask = torch.where(flat_incoming_probs >= min_joint_prob)[0]
            flat_best_probs = flat_incoming_probs[flat_best_mask]
            flat_best_indices = flat_best_mask
            best_children_token_ids = flat_incoming_ids[flat_best_indices]

            if flat_best_indices.size(-1) + self._tree.end > self._max_len:
                raise NotImplementedError("Implement trim budget")

        else:
            flat_best_probs = flat_incoming_probs
            flat_best_indices = torch.arange(
                flat_incoming_probs.size(0), device=logits.device
            )
            best_children_token_ids = flat_incoming_ids

        best_hypo_ids = flat_best_indices // self._max_branch_width
        best_beam_indices = beam_indices[best_hypo_ids]
        best_children_positions = beam_positions[best_hypo_ids] + 1

        return (
            best_children_token_ids,
            best_children_positions,
            best_beam_indices,
            flat_best_probs,
        )

    def _check_new_token_in_budget(self, cumulative_beam_scores: torch.Tensor):
        lowest_tree_logprob = (
            self._tree.logprobs[self._tree.prefix_len : self._tree.end]
            .topk(k=self._max_budget, dim=-1, sorted=False)
            .values.min()
        )
        best_new_logprob = cumulative_beam_scores.max()

        return best_new_logprob >= lowest_tree_logprob

    def _trim_by_budget(self):
        src_indices = (
            self._tree.logprobs[self._tree.prefix_len : self._tree.end]
            .topk(k=self._max_budget, sorted=False)
            .indices
            + self._tree.prefix_len
        )
        dest_indices = torch.arange(
            self._tree.prefix_len,
            self._tree.prefix_len + src_indices.size(-1),
            device=self._device,
        )

        self._tree.gather(src_indices, dest_indices)
        self._engine.gather(src_indices, dest_indices)

    async def _validate_tree(self, req_idx: int, prefill=False):
        self._logger.debug("Validating tree")

        with util.Timing(
            device=self._device, mode=self._target_time_mode
        ) as preprocess_t:
            target_token_map_bool = (
                self._tree.status[: self._tree.end] >= self._tree.PROCESSED
            )
            target_token_map_bool[: self._tree.prefix_len] = False
            target_token_indices = torch.where(target_token_map_bool)[0]
            target_parent_indices = self._tree.parents[: self._tree.end][
                target_token_map_bool
            ]

            input_token_map_bool = target_token_map_bool.clone()
            input_token_map_bool[target_parent_indices] = True

            input_ids = self._tree.tokens[: self._tree.end][
                input_token_map_bool
            ].unsqueeze(0)
            position_ids = self._tree.positions[: self._tree.end][
                input_token_map_bool
            ].unsqueeze(0)
            cache_seq_indices = torch.where(input_token_map_bool)[0]
            attention_mask = self._tree.amask[..., cache_seq_indices, :]

        with util.Timing(device=self._device, mode=self._target_time_mode) as wait_t:
            prefix = self._prompt if prefill else None
            request_start = time.perf_counter()
            target_result = asyncio.create_task(
                self._validator.request(
                    client_idx=self._client_idx,
                    req_idx=req_idx,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cache_seq_indices=cache_seq_indices,
                    attention_mask=attention_mask,
                    parent_indices=target_parent_indices,
                    prefill=prefill,
                    prefix=prefix,
                )
            )
            await asyncio.sleep(0.00001)

            proactive_result = await self._run_proactive_draft(
                target_result,
                request_start,
            )

            validation_response = (
                target_result.result() if target_result.done() else await target_result
            )
            selection = validation_response.selection
            prefill_cnt = validation_response.prefill
            response_received_ms = (
                validation_response.received_at - request_start
            ) * 1000
            response_decoded_ms = (
                validation_response.decoded_at - request_start
            ) * 1000
            response_observed_ms = (
                time.perf_counter() - request_start
            ) * 1000

        with util.Timing(
            device=self._device, mode=self._target_time_mode
        ) as postprocess_t:
            interim_t = torch.ones_like(self._tree.tokens[: self._tree.end])
            interim_t[input_token_map_bool] = selection

            draft_token_choices = self._tree.tokens[: self._tree.end][
                target_token_map_bool
            ]
            target_token_choices = interim_t[target_parent_indices]

            accept_flags = draft_token_choices == target_token_choices

            accept_indices = target_token_indices[accept_flags]

            accept_mask = torch.zeros(self._tree.end, device=self._device)
            accept_mask[: self._tree.prefix_len] = 1
            accept_mask[accept_indices] = 1
            accepted_amask = attention_mask[0, 0, :, : self._tree.end] * accept_mask

            mask_row_sums = (
                attention_mask[0, 0, :, : self._tree.end].sum(dim=1).to(torch.long)
            )

            seq_lengths = accepted_amask.sum(dim=1).to(torch.long)
            best_seq_idx = (mask_row_sums * (mask_row_sums == seq_lengths)).argmax()
            best_seq_mask = attention_mask[0, 0, best_seq_idx, : self._tree.end].to(
                torch.bool
            )

            fresh_token_indices = (
                torch.where(best_seq_mask[self._tree.prefix_len :])[0]
                + self._tree.prefix_len
            )
            fresh_token_ids = self._tree.tokens[fresh_token_indices]

            last_accepted_token_idx = (
                fresh_token_indices[-1]
                if fresh_token_indices.numel() > 0
                else torch.tensor([self._tree.prefix_len - 1])
            ).to(self._device)

            # add one bonus token to num of accepted tokens
            self._logger.debug(
                "Num of accepted tokens: %d", fresh_token_indices.numel() + 1
            )

            extra_token_id = torch.tensor(
                [interim_t[last_accepted_token_idx]], device=self._device
            )
            last_accepted_idx_value = int(last_accepted_token_idx.item())
            extra_token_id_value = int(extra_token_id.item())

            if proactive_result.deepest_leaf_indices:
                observed_full_depth = (
                    last_accepted_idx_value
                    in proactive_result.deepest_leaf_indices
                )
                proactive_result.observed_full_depth = observed_full_depth
                if self._proactive_path_policy == "deepest_multi":
                    previous_rate = (
                        SpecExecClient._shared_full_depth_acceptance
                    )
                    if previous_rate is None:
                        previous_rate = (
                            config.proactive_multi_full_depth_prior
                        )
                    alpha = config.proactive_multi_acceptance_ewma_alpha
                    SpecExecClient._shared_full_depth_acceptance = (
                        alpha * float(observed_full_depth)
                        + (1.0 - alpha) * previous_rate
                    )

            if self._proactive_client is not None:
                self._previous_proactive_draft = self._proactive_draft

            if self._proactive_path_policy == "single_best":
                matched_root = (
                    proactive_result.roots[0]
                    if proactive_result.roots
                    and proactive_result.roots[0].leaf_idx
                    == last_accepted_idx_value
                    and proactive_result.roots[0].token_id
                    == extra_token_id_value
                    else None
                )
            else:
                matched_root = proactive_result.find_matching_root(
                    last_accepted_idx_value,
                    extra_token_id_value,
                )
            if (
                self._proactive_client is not None
                and matched_root is not None
                and proactive_result.tree_prefix_len is not None
                and proactive_result.tree_end is not None
            ):
                self._proactive_draft = True
                self._reused_proactive_depth = (
                    matched_root.executed_depth
                )
                if self._proactive_path_policy == "single_best":
                    self._reorder_by_sequence_proactive(
                        best_seq_mask,
                        proactive_result.tree_prefix_len,
                        proactive_result.tree_end,
                    )
                else:
                    self._reorder_by_sequence_proactive_multi(
                        best_seq_mask,
                        matched_root.node_indices,
                    )
            else:
                self._proactive_draft = False
                self._reused_proactive_depth = 0
                self._reorder_by_sequence(best_seq_mask)
                self._tree.add(
                    token_ids=extra_token_id,
                    token_positions=self._tree.positions[self._tree.end - 1] + 1,
                    parent_indices=torch.tensor(
                        [self._tree.end - 1], device=self._device
                    ),
                    logprobs=torch.tensor([0.0], device=self._device),
                )
                self._tree.prefix_len = self._tree.end
                self._tree.status[: self._tree.prefix_len - 1] = self._tree.PROMPT

            fresh_token_ids = torch.cat(
                [fresh_token_ids, extra_token_id], dim=-1
            ).unsqueeze(0)

        if self._adaptive_policy is not None:
            self._adaptive_policy.observe_cycle(
                response_ms=response_received_ms,
                aligned=self._proactive_draft,
                proactive_executed=(
                    len(proactive_result.roots) > 0
                ),
            )

        proactive_execution = {
            "mode": self._proactive_mode,
            "path_policy": proactive_result.path_policy,
            "planned_depth": proactive_result.planned_depth,
            "executed_depth": proactive_result.executed_depth,
            "elapsed_ms": proactive_result.elapsed_ms,
            "setup_ms": proactive_result.setup_ms,
            "layer_wall_ms": proactive_result.layer_wall_ms,
            "layer_gpu_ms": proactive_result.layer_gpu_ms,
            "response_received_ms": response_received_ms,
            "response_decoded_ms": response_decoded_ms,
            "response_observed_ms": response_observed_ms,
            "stopped_by_response": proactive_result.stopped_by_response,
            "skipped_reason": proactive_result.skipped_reason,
            "policy_reason": proactive_result.policy_reason,
            "deadline_checks": proactive_result.deadline_checks,
            "max_leaf_depth": proactive_result.max_leaf_depth,
            "deepest_leaf_count": proactive_result.deepest_leaf_count,
            "selected_leaf_count": proactive_result.selected_leaf_count,
            "sequence_path_depth": (
                proactive_result.sequence_path_depth
            ),
            "sequence_stop_probabilities": (
                proactive_result.sequence_stop_probabilities
            ),
            "root_count": len(proactive_result.roots),
            "roots": [
                root.as_dict() for root in proactive_result.roots
            ],
            "layer_batch_widths": proactive_result.layer_batch_widths,
            "layer_active_root_counts": (
                proactive_result.layer_active_root_counts
            ),
            "full_depth_acceptance": (
                proactive_result.full_depth_acceptance
            ),
            "observed_full_depth": proactive_result.observed_full_depth,
            "updated_full_depth_acceptance": (
                SpecExecClient._shared_full_depth_acceptance
                if self._proactive_path_policy == "deepest_multi"
                else None
            ),
            "matched_root_id": (
                matched_root.root_id if matched_root is not None else None
            ),
            "matched_stop_depth": (
                matched_root.stop_depth
                if matched_root is not None
                else None
            ),
            "matched_reused_depth": (
                matched_root.executed_depth
                if matched_root is not None
                else 0
            ),
            "proactive_node_count": sum(
                len(root.node_indices) for root in proactive_result.roots
            ),
            "matched_node_count": (
                len(matched_root.node_indices)
                if matched_root is not None
                else 0
            ),
            "wasted_node_count": (
                sum(
                    len(root.node_indices)
                    for root in proactive_result.roots
                )
                - (
                    len(matched_root.node_indices)
                    if matched_root is not None
                    else 0
                )
            ),
            "deadline_pruned_root_count": sum(
                bool(check.get("pruned_root"))
                for check in proactive_result.deadline_checks
            ),
        }
        if self._adaptive_policy is not None:
            proactive_execution["controller"] = (
                self._adaptive_policy.stats()
            )

        stats = {
            "preprocess_t": preprocess_t.elapsed,
            "wait_t": wait_t.elapsed,
            "postprocess_t": postprocess_t.elapsed,
            "num_accepted_tokens": fresh_token_ids.size(-1),
            "prefill": prefill_cnt,
            "previous_proactive": self._previous_proactive_draft
            if self._proactive_client
            else False,
            "proactive": self._proactive_draft if self._proactive_client else False,
            "proactive_execution": proactive_execution,
        }

        return fresh_token_ids, stats

    def _reorder_by_sequence(self, seq_mask: torch.Tensor):
        """
        Reorder the tree and engine's kv cache according to the validated sequence.

        Args:
            seq_mask: Sequence Mask
        """

        seq_indices = torch.where(seq_mask != 0)[0]

        self._engine.gather(
            seq_indices,
            torch.arange(seq_indices.size(-1), device=self._device),
        )

        self._tree.reorder_by_sequence(seq_mask, seq_indices)

    def _reorder_by_sequence_proactive(
        self,
        seq_mask: torch.Tensor,
        proactive_tree_prefix_len: int,
        proactive_tree_end: int,
    ):
        """Original single-root proactive tree reorder."""
        seq_indices = torch.where(seq_mask != 0)[0]
        max_src_idx = proactive_tree_end
        mapping_tensor = torch.full(
            (max_src_idx,), -1, dtype=torch.long, device=self._device
        )

        new_prefix_len = int(torch.sum(seq_mask).item())
        if torch.any(seq_mask[self._tree.prefix_len :]):
            src_indices = seq_indices[seq_indices >= self._tree.prefix_len]
            dest_indices = torch.arange(
                self._tree.prefix_len, new_prefix_len, device=self._device
            )
            mapping_tensor[src_indices] = dest_indices

            self._tree.tokens[dest_indices] = self._tree.tokens[src_indices]
            self._tree.positions[dest_indices] = dest_indices
            self._tree.parents[dest_indices] = dest_indices - 1
            self._tree.status[dest_indices] = self._tree.GENERATED

        src_indices = torch.arange(
            proactive_tree_prefix_len, proactive_tree_end, device=self._device
        )
        dest_indices = torch.arange(
            new_prefix_len,
            new_prefix_len + proactive_tree_end - proactive_tree_prefix_len,
            device=self._device,
        )
        mapping_tensor[src_indices] = dest_indices

        self._tree.tokens[dest_indices] = self._tree.tokens[src_indices]
        self._tree.positions[dest_indices] = self._tree.positions[src_indices]
        self._tree.parents[dest_indices] = mapping_tensor[
            self._tree.parents[src_indices]
        ]
        self._tree.status[dest_indices] = self._tree.status[src_indices]
        self._tree.logprobs[dest_indices] = self._tree.logprobs[src_indices]
        self._tree.amask[
            ...,
            dest_indices,
            new_prefix_len : new_prefix_len
            + proactive_tree_end
            - proactive_tree_prefix_len,
        ] = self._tree.amask[
            ..., src_indices, proactive_tree_prefix_len:proactive_tree_end
        ]

        self._tree.end = (
            new_prefix_len + proactive_tree_end - proactive_tree_prefix_len
        )
        self._tree.prefix_len = new_prefix_len + 1

        self._tree.status[: self._tree.prefix_len - 1] = self._tree.PROMPT
        self._tree.status[
            self._tree.prefix_len - 1 : self._tree.prefix_len + 1
        ] = self._tree.PROCESSED
        self._tree.status[
            self._tree.status == self._tree.POST_CANDIDATE
        ] = self._tree.CANDIDATE
        self._tree.status[
            self._tree.status == self._tree.POST_PROCESSED
        ] = self._tree.PROCESSED

        self._tree.logprobs[self._tree.end :].zero_()
        self._tree._data[:, self._tree.end :].zero_()

        causal_mask = torch.tril(
            torch.ones(
                self._tree.prefix_len,
                self._tree.prefix_len,
                dtype=self._dtype,
                device=self._device,
            )
        )
        self._tree.amask[
            ..., : self._tree.prefix_len, : self._tree.prefix_len
        ] = causal_mask
        self._tree.amask[
            ..., self._tree.prefix_len : self._tree.end, : self._tree.prefix_len
        ] = 1.0

        src_indices = seq_mask[: self._tree.prefix_len]
        src_indices = torch.where(src_indices)[0]
        dst_indices = torch.arange(src_indices.size(-1), device=self._device)
        self._engine.gather(src_indices, dst_indices)

    def _reorder_by_sequence_proactive_multi(
        self,
        seq_mask: torch.Tensor,
        proactive_node_indices: list[int],
    ):
        """Keep the validated sequence and one matching proactive subtree."""
        seq_indices = torch.where(seq_mask != 0)[0]
        proactive_src_indices = torch.tensor(
            sorted(proactive_node_indices),
            dtype=torch.long,
            device=self._device,
        )
        if proactive_src_indices.numel() == 0:
            self._reorder_by_sequence(seq_mask)
            return

        old_prefix_len = self._tree.prefix_len
        new_prefix_len = int(seq_indices.numel())
        proactive_dest_indices = torch.arange(
            new_prefix_len,
            new_prefix_len + proactive_src_indices.numel(),
            device=self._device,
        )
        max_src_idx = max(
            int(proactive_src_indices.max().item()) + 1,
            int(seq_indices.max().item()) + 1,
        )
        mapping_tensor = torch.full(
            (max_src_idx,), -1, dtype=torch.long, device=self._device
        )
        mapping_tensor[:old_prefix_len] = torch.arange(
            old_prefix_len, device=self._device
        )

        accepted_draft_indices = seq_indices[seq_indices >= old_prefix_len]
        if accepted_draft_indices.numel() > 0:
            dest_indices = torch.arange(
                old_prefix_len, new_prefix_len, device=self._device
            )
            mapping_tensor[accepted_draft_indices] = dest_indices
            accepted_tokens = self._tree.tokens[
                accepted_draft_indices
            ].clone()
            self._tree.tokens[dest_indices] = accepted_tokens
            self._tree.positions[dest_indices] = dest_indices
            self._tree.parents[dest_indices] = dest_indices - 1
            self._tree.status[dest_indices] = self._tree.GENERATED

        mapping_tensor[proactive_src_indices] = proactive_dest_indices
        proactive_tokens = self._tree.tokens[proactive_src_indices].clone()
        proactive_positions = self._tree.positions[
            proactive_src_indices
        ].clone()
        proactive_parents = self._tree.parents[
            proactive_src_indices
        ].clone()
        proactive_status = self._tree.status[proactive_src_indices].clone()
        proactive_logprobs = self._tree.logprobs[
            proactive_src_indices
        ].clone()

        processed_mask = proactive_status == self._tree.POST_PROCESSED
        processed_src_indices = proactive_src_indices[processed_mask]
        processed_dest_indices = proactive_dest_indices[processed_mask]
        cache_src_indices = torch.cat(
            [seq_indices, processed_src_indices]
        )
        cache_dest_indices = torch.cat(
            [
                torch.arange(new_prefix_len, device=self._device),
                processed_dest_indices,
            ]
        )
        self._engine.gather(cache_src_indices, cache_dest_indices)

        self._tree.tokens[proactive_dest_indices] = proactive_tokens
        self._tree.positions[proactive_dest_indices] = proactive_positions
        mapped_parents = mapping_tensor[proactive_parents]
        if torch.any(mapped_parents < 0):
            raise RuntimeError("Proactive subtree contains an unmapped parent")
        self._tree.parents[proactive_dest_indices] = mapped_parents
        self._tree.status[proactive_dest_indices] = proactive_status
        self._tree.logprobs[proactive_dest_indices] = proactive_logprobs

        self._tree.end = new_prefix_len + proactive_src_indices.numel()
        self._tree.prefix_len = new_prefix_len + 1
        self._tree.status[: self._tree.prefix_len] = self._tree.PROMPT
        self._tree.status[self._tree.prefix_len - 1] = self._tree.PROCESSED
        self._tree.status[
            self._tree.status == self._tree.POST_CANDIDATE
        ] = self._tree.CANDIDATE
        self._tree.status[
            self._tree.status == self._tree.POST_PROCESSED
        ] = self._tree.PROCESSED

        self._tree.amask.zero_()
        self._tree.amask[
            ..., : self._tree.prefix_len, : self._tree.prefix_len
        ] = torch.tril(
            torch.ones(
                self._tree.prefix_len,
                self._tree.prefix_len,
                dtype=self._dtype,
                device=self._device,
            )
        )
        for node_idx in range(self._tree.prefix_len, self._tree.end):
            parent_idx = int(self._tree.parents[node_idx].item())
            self._tree.amask[..., node_idx, : self._tree.end] = (
                self._tree.amask[..., parent_idx, : self._tree.end]
            )
            self._tree.amask[..., node_idx, node_idx] = 1.0

        self._tree.logprobs[self._tree.end :].zero_()
        self._tree._data[:, self._tree.end :].zero_()
