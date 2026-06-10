import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

import log
import util
from config import SpecEdgeClientConfig as config
from specedge.engine.graph import GraphEngine
from specedge.tree import Tree


@dataclass
class ProactiveDraftResult:
    root_leaf_idx: Optional[torch.Tensor] = None
    root_token_id: Optional[torch.Tensor] = None
    tree_prefix_len: Optional[int] = None
    tree_end: Optional[int] = None
    planned_depth: int = 0
    executed_depth: int = 0
    elapsed_ms: float = 0.0
    stopped_by_response: bool = False
    skipped_reason: Optional[str] = None
    policy_reason: Optional[str] = None
    setup_ms: Optional[float] = None
    layer_wall_ms: list[float] = field(default_factory=list)
    layer_gpu_ms: list[Optional[float]] = field(default_factory=list)
    deadline_checks: list[dict[str, Any]] = field(default_factory=list)


class ProactiveDraftSession:
    def __init__(
        self,
        client: "SpecExecProactiveDraft",
        planned_depth: int,
    ) -> None:
        self._client = client
        self._tree = client._tree
        self._planned_depth = planned_depth
        self._prev_tree_end = self._tree.end
        self._prev_tree_prefix_len = self._tree.prefix_len
        self._start_time = time.perf_counter()
        self._finished = False

        self.result = ProactiveDraftResult(planned_depth=planned_depth)
        best_token_idx, best_token_id = client._get_best_bonus_token_candidate()
        self.result.root_leaf_idx = best_token_idx
        self.result.root_token_id = best_token_id

        if best_token_idx is None or best_token_id is None:
            self.result.skipped_reason = "no_candidate"
            self._finished = True
            return

        self._tree.prefix_len = self._tree.end
        self._tree.add(
            token_ids=best_token_id,
            token_positions=self._tree.positions[best_token_idx] + 1,
            parent_indices=torch.tensor([best_token_idx], device=client._device),
            logprobs=torch.tensor(0.0, device=client._device),
            token_status=self._tree.POST_CANDIDATE,
        )

    @property
    def can_step(self) -> bool:
        return (
            not self._finished
            and self.result.executed_depth < self._planned_depth
        )

    @torch.inference_mode()
    def step(self) -> float:
        if not self.can_step:
            return 0.0

        step_start = time.perf_counter()
        idx = self.result.executed_depth
        self._client._logger.debug(
            "Growing tree proactively: %d / %d", idx + 1, self._planned_depth
        )
        logits, parent_indices, parent_scores, parent_positions = (
            self._client._process_candidates()
        )
        token_ids, token_positions, parent_indices, beam_scores = (
            self._client._get_next_beams(
                logits, parent_indices, parent_positions, parent_scores
            )
        )

        self.result.executed_depth += 1
        if token_ids.size(-1) == 0:
            self._finished = True
        else:
            self._tree.add(
                token_ids=token_ids,
                token_positions=token_positions,
                parent_indices=parent_indices,
                logprobs=beam_scores,
                token_status=self._tree.POST_CANDIDATE,
            )

        return (time.perf_counter() - step_start) * 1000

    def finish(self, stopped_by_response: bool = False) -> ProactiveDraftResult:
        self.result.stopped_by_response = stopped_by_response
        if self.result.root_leaf_idx is not None:
            self.result.tree_prefix_len = self._tree.prefix_len
            self.result.tree_end = self._tree.end

        self._tree.prefix_len = self._prev_tree_prefix_len
        self._tree.end = self._prev_tree_end
        self.result.elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        self._finished = True
        return self.result


class SpecExecProactiveDraft:
    def __init__(self, tree: Tree, engine: GraphEngine, max_len: int) -> None:
        self._logger = log.get_logger()
        self._engine = engine
        self._tree = tree

        # configuration
        self._device = config.device
        self._dtype = config.dtype

        self._max_n_beams = config.proactive_max_n_beams
        self._max_beam_len = config.proactive_max_beam_len
        self._max_branch_width = config.proactive_max_branch_width
        self._max_budget = config.proactive_max_budget

        # FIXME: remove hard-coded value
        self._max_len = max_len
        self._last_result: Optional[ProactiveDraftResult] = None

    @property
    def last_result(self) -> Optional[ProactiveDraftResult]:
        return self._last_result

    @torch.inference_mode()
    def draft(self):
        """
        Expand tree from the best bonus token candidate.
        """

        session = self.start_session(self._max_beam_len)
        while session.can_step:
            session.step()
        result = session.finish()
        self._last_result = result
        return (
            result.root_leaf_idx,
            result.root_token_id,
            result.tree_prefix_len,
            result.tree_end,
        )

    def start_session(self, max_depth: int) -> ProactiveDraftSession:
        return ProactiveDraftSession(
            client=self,
            planned_depth=min(self._max_beam_len, max(0, max_depth)),
        )

    def _get_best_bonus_token_candidate(self):
        """
        Get the best bonus token candidate to generate extra draft tree.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: best token index and token id
        """

        TOP_K_TOKENS = 1024

        #取所有leaf nodes
        self._logger.debug("Getting best bonus token candidate...")
        input_indices = self._get_leaves_nodes()

        if input_indices.numel() == 0:
            self._logger.debug("No candidate token found.")
            return None, None

        #构造整条path
        #取出 leaf 对应的 token
        input_ids = self._tree.tokens[input_indices].unsqueeze(0)
        #每个 token 在序列中的位置
        position_ids = self._tree.positions[input_indices].unsqueeze(0)
        attention_mask = self._tree.amask[..., input_indices, :]

        cache_seq_indices = input_indices
        cache_batch_indices = torch.full_like(
            cache_seq_indices, 0, dtype=torch.long, device=self._device
        )

        logits = self._engine.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_batch_indices=cache_batch_indices,
            cache_seq_indices=cache_seq_indices,
            attention_mask=attention_mask,
        )

        beam_scores = self._tree.logprobs[input_indices]

        logits = logits[0, -input_indices.size(-1) :, :]
        logprobs = torch.log_softmax(logits, dim=-1)
        logprobs = logprobs.topk(
            k=TOP_K_TOKENS, dim=-1, sorted=False
        )  # choose top 1024 tokens in the beam
        logprob_ids = logprobs.indices.flatten()
        #计算分数
        accumulate_logprobs = (
            beam_scores.unsqueeze(-1) + np.log(0.95) + logprobs.values
        ).flatten()

        _, best_beam_idx = accumulate_logprobs.max(dim=0)
        best_token_idx = input_indices[best_beam_idx // TOP_K_TOKENS]
        best_token_id = logprob_ids[best_beam_idx]

        return best_token_idx, best_token_id

    def _get_leaves_nodes(self):
        """
        Get leave nodes' indices from draft tree.

        Returns:
            torch.Tensor: leave nodes' indices
        """

        # get unique parent indices
        parent_indices = torch.unique(self._tree.parents)

        # mask for tokens that are not parents
        candidate_leaf_mask = ~torch.isin(
            torch.arange(self._tree.prefix_len, self._tree.end, device=self._device),
            parent_indices,
        )

        if candidate_leaf_mask.sum() > self._max_n_beams:
            candidate_leaf_indices = torch.where(candidate_leaf_mask)[0]
            topk_indices = (
                self._tree.logprobs[candidate_leaf_indices]
                .topk(k=self._max_n_beams, sorted=False)
                .indices
            )
            return candidate_leaf_indices[topk_indices]
        else:
            # mask for prefix tokens
            leaf_mask = torch.concat(
                [
                    torch.zeros(self._tree.prefix_len, device=self._device),
                    candidate_leaf_mask,
                ]
            ).to(torch.bool)

            return torch.nonzero(leaf_mask, as_tuple=True)[0]

    def _process_candidates(self):
        """
        Process candidates from continuous draft tree.
        """

        candidate_indices = torch.where(self._tree.status == self._tree.POST_CANDIDATE)[
            0
        ]

        if candidate_indices.numel() > self._max_n_beams:
            accumulated_logprobs = self._tree.logprobs[candidate_indices]
            in_budget_indices = accumulated_logprobs.topk(
                k=self._max_n_beams, sorted=False
            ).indices
            candidate_indices = candidate_indices[in_budget_indices]
            candidate_indices, _ = candidate_indices.sort()

        input_ids = self._tree.tokens[candidate_indices].unsqueeze(0)
        position_ids = self._tree.positions[candidate_indices].unsqueeze(0)
        attention_mask = self._tree.amask[..., candidate_indices, :]

        cache_seq_indices = candidate_indices
        cache_batch_indices = torch.full_like(
            cache_seq_indices, 0, dtype=torch.long, device=self._device
        )

        logits = self._engine.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_batch_indices=cache_batch_indices,
            cache_seq_indices=cache_seq_indices,
            attention_mask=util.invert_mask(attention_mask),
        )

        self._tree.status[candidate_indices] = self._tree.POST_PROCESSED
        beam_scores = self._tree.logprobs[candidate_indices]
        beam_positions = self._tree.positions[candidate_indices]

        return (logits, candidate_indices, beam_scores, beam_positions)

    def _get_next_beams(
        self,
        logits: torch.Tensor,
        beam_indices: torch.Tensor,
        beam_positions: torch.Tensor,
        beam_scores: torch.Tensor,
    ):
        DECAY_FACTOR = np.log(0.95)

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
        sorted_incoming_probs = flat_incoming_probs.sort(descending=True)
        flat_sorted_logprobs = sorted_incoming_probs.values
        flat_sorted_indices = sorted_incoming_probs.indices

        joint_probs = torch.concat(
            [
                self._tree.logprobs[self._tree.prefix_len : self._tree.end],
                flat_sorted_logprobs,
            ]
        )  # existing beams + new beams for finding threshold

        if (
            joint_probs.size(-1) > self._max_budget
            or joint_probs.size(-1) + (self._tree.end - self._tree.prefix_len)
            > self._max_len
        ):
            min_joint_prob = joint_probs.topk(
                k=self._max_budget, sorted=False, dim=-1
            ).values.min()

            flat_best_mask = torch.where(flat_sorted_logprobs >= min_joint_prob)[0]
            flat_best_probs = flat_sorted_logprobs[flat_best_mask]
            flat_best_indices = flat_sorted_indices[flat_best_mask]
            best_children_token_ids = flat_incoming_ids[flat_best_indices]

            if flat_best_indices.size(-1) + self._tree.end > self._max_len:
                raise NotImplementedError("Implement trim budget")

        else:
            flat_best_probs = flat_sorted_logprobs
            flat_best_indices = flat_sorted_indices
            best_children_token_ids = flat_incoming_ids[flat_best_indices]

        best_hypo_ids = flat_best_indices // self._max_branch_width
        best_beam_indices = beam_indices[best_hypo_ids]
        best_children_positions = beam_positions[best_hypo_ids] + 1

        return (
            best_children_token_ids,
            best_children_positions,
            best_beam_indices,
            flat_best_probs,
        )
