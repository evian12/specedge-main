import time
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Optional

import numpy as np
import torch

import log
import util
from config import SpecEdgeClientConfig as config
from specedge.client.proactive_selection import (
    ProactiveRootCandidate,
    acceptance_stop_probabilities,
    select_bonus_candidates,
    select_ids_by_probability,
    select_sequence_bonus_candidates,
    trace_main_sequence_nodes,
)
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
    path_policy: str = "single_best"
    max_leaf_depth: Optional[int] = None
    deepest_leaf_count: int = 0
    selected_leaf_count: int = 0
    full_depth_acceptance: Optional[float] = None
    observed_full_depth: Optional[bool] = None
    roots: list["ProactiveRootResult"] = field(default_factory=list)
    layer_batch_widths: list[int] = field(default_factory=list)
    layer_active_root_counts: list[int] = field(default_factory=list)
    deepest_leaf_indices: list[int] = field(default_factory=list)
    sequence_path_depth: Optional[int] = None
    sequence_stop_probabilities: list[float] = field(default_factory=list)

    def find_matching_root(
        self, leaf_idx: int, token_id: int
    ) -> Optional["ProactiveRootResult"]:
        for root in self.roots:
            if (
                root.leaf_idx == leaf_idx
                and root.token_id == token_id
                and root.executed_depth > 0
            ):
                return root
        return None


@dataclass
class ProactiveRootResult:
    root_id: int
    leaf_idx: int
    token_id: int
    leaf_probability: float
    bonus_probability: float
    joint_probability: float
    stop_depth: Optional[int] = None
    node_indices: list[int] = field(default_factory=list)
    executed_depth: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "leaf_idx": self.leaf_idx,
            "token_id": self.token_id,
            "leaf_probability": self.leaf_probability,
            "bonus_probability": self.bonus_probability,
            "joint_probability": self.joint_probability,
            "stop_depth": self.stop_depth,
            "node_count": len(self.node_indices),
            "executed_depth": self.executed_depth,
        }


def select_root_ids_by_coverage(
    roots: list[ProactiveRootResult],
    coverage: float,
) -> set[int]:
    return select_ids_by_probability(
        [
            (root.root_id, root.joint_probability)
            for root in roots
        ],
        coverage,
    )


class ProactiveDraftSession:
    def __init__(
        self,
        client: "SpecExecProactiveDraft",
        planned_depth: int,
        full_depth_acceptance: float,
    ) -> None:
        self._client = client
        self._tree = client._tree
        self._planned_depth = planned_depth
        self._prev_tree_end = self._tree.end
        self._prev_tree_prefix_len = self._tree.prefix_len
        self._start_time = time.perf_counter()
        self._finished = False
        self._disabled_root_ids: set[int] = set()

        self._node_root_ids: dict[int, int] = {}
        self.result = ProactiveDraftResult(
            planned_depth=planned_depth,
            path_policy=client._path_policy,
            full_depth_acceptance=full_depth_acceptance,
        )
        (
            candidates,
            deepest_leaf_indices,
            max_leaf_depth,
            selected_leaf_count,
        ) = client._get_proactive_root_candidates(full_depth_acceptance)
        self.result.deepest_leaf_indices = deepest_leaf_indices
        self.result.deepest_leaf_count = len(deepest_leaf_indices)
        self.result.max_leaf_depth = max_leaf_depth
        self.result.selected_leaf_count = selected_leaf_count
        if client._path_policy == "sequence_depth":
            self.result.sequence_path_depth = (
                client._last_sequence_path_depth
            )
            self.result.sequence_stop_probabilities = list(
                client._last_sequence_stop_probabilities
            )

        if not candidates:
            self.result.skipped_reason = "no_candidate"
            self._finished = True
            return

        self._tree.prefix_len = self._tree.end
        root_start = self._tree.end
        token_ids = torch.tensor(
            [candidate.token_id for candidate in candidates],
            dtype=torch.long,
            device=client._device,
        )
        parent_indices = torch.tensor(
            [candidate.leaf_idx for candidate in candidates],
            dtype=torch.long,
            device=client._device,
        )
        token_positions = self._tree.positions[parent_indices] + 1
        if client._path_policy == "single_best":
            root_logprobs = torch.zeros(
                len(candidates),
                dtype=torch.float32,
                device=client._device,
            )
        else:
            root_logprobs = torch.tensor(
                [
                    np.log(max(candidate.joint_probability, 1e-12))
                    for candidate in candidates
                ],
                dtype=torch.float32,
                device=client._device,
            )
        self._tree.add(
            token_ids=token_ids,
            token_positions=token_positions,
            parent_indices=parent_indices,
            logprobs=root_logprobs,
            token_status=self._tree.POST_CANDIDATE,
        )
        for root_id, candidate in enumerate(candidates):
            node_idx = root_start + root_id
            self._node_root_ids[node_idx] = root_id
            self.result.roots.append(
                ProactiveRootResult(
                    root_id=root_id,
                    leaf_idx=candidate.leaf_idx,
                    token_id=candidate.token_id,
                    leaf_probability=candidate.leaf_probability,
                    bonus_probability=candidate.bonus_probability,
                    joint_probability=candidate.joint_probability,
                    stop_depth=candidate.stop_depth,
                    node_indices=[node_idx],
                )
            )

        first_root = self.result.roots[0]
        self.result.root_leaf_idx = torch.tensor(
            first_root.leaf_idx, device=client._device
        )
        self.result.root_token_id = torch.tensor(
            first_root.token_id, device=client._device
        )

    @property
    def can_step(self) -> bool:
        return (
            not self._finished
            and self.result.executed_depth < self._planned_depth
        )

    def _coverage_for_layer(self, layer_index: int) -> float:
        coverages = self._client._depth_probability_coverage
        if not coverages:
            return 1.0
        return coverages[min(layer_index, len(coverages) - 1)]

    def _candidate_indices_for_step(self) -> tuple[torch.Tensor, set[int]]:
        candidate_indices = torch.where(
            self._tree.status[: self._tree.end] == self._tree.POST_CANDIDATE
        )[0]
        if self._client._path_policy == "single_best":
            return candidate_indices, {0} if candidate_indices.numel() else set()

        active_root_ids = select_root_ids_by_coverage(
            self.result.roots,
            self._coverage_for_layer(self.result.executed_depth),
        )
        active_root_ids -= self._disabled_root_ids
        filtered_indices = [
            int(index.item())
            for index in candidate_indices
            if self._node_root_ids.get(int(index.item())) in active_root_ids
        ]
        return (
            torch.tensor(
                filtered_indices,
                dtype=torch.long,
                device=self._client._device,
            ),
            active_root_ids,
        )

    @property
    def next_batch_width(self) -> int:
        if not self.can_step:
            return 0
        candidate_indices, _ = self._candidate_indices_for_step()
        return min(candidate_indices.numel(), self._client._max_n_beams)

    def prune_lowest_priority_root(self) -> bool:
        if self._client._path_policy == "single_best":
            return False
        candidate_indices, active_root_ids = self._candidate_indices_for_step()
        if candidate_indices.numel() == 0 or len(active_root_ids) <= 1:
            return False

        candidate_root_ids = {
            self._node_root_ids[int(index.item())]
            for index in candidate_indices
        }
        removable_roots = [
            root
            for root in self.result.roots
            if root.root_id in active_root_ids
            and root.root_id in candidate_root_ids
        ]
        if len(removable_roots) <= 1:
            return False
        lowest_priority_root = min(
            removable_roots,
            key=lambda root: root.joint_probability,
        )
        self._disabled_root_ids.add(lowest_priority_root.root_id)
        return True

    @torch.inference_mode()
    def step(self) -> float:
        if not self.can_step:
            return 0.0

        step_start = time.perf_counter()
        idx = self.result.executed_depth
        self._client._logger.debug(
            "Growing tree proactively: %d / %d", idx + 1, self._planned_depth
        )
        candidate_indices, active_root_ids = self._candidate_indices_for_step()
        if candidate_indices.numel() == 0:
            self._finished = True
            return (time.perf_counter() - step_start) * 1000

        logits, parent_indices, parent_scores, parent_positions = (
            self._client._process_candidates(candidate_indices)
        )
        batch_width = parent_indices.numel()
        self.result.layer_batch_widths.append(batch_width)
        self.result.layer_active_root_counts.append(len(active_root_ids))

        if self._client._path_policy == "single_best":
            token_ids, token_positions, parent_indices, beam_scores = (
                self._client._get_next_beams(
                    logits, parent_indices, parent_positions, parent_scores
                )
            )
        elif self._client._path_policy == "sequence_depth":
            proactive_node_count = self._tree.end - self._tree.prefix_len
            remaining_budget = max(
                0, self._client._max_budget - proactive_node_count
            )
            (
                token_ids,
                token_positions,
                parent_indices,
                beam_scores,
            ) = self._client._get_next_sequence_multi(
                logits,
                parent_indices,
                parent_positions,
                parent_scores,
                self._node_root_ids,
                remaining_budget,
            )
        else:
            proactive_node_count = self._tree.end - self._tree.prefix_len
            remaining_budget = max(
                0, self._client._max_budget - proactive_node_count
            )
            remaining_layers = self._planned_depth - self.result.executed_depth
            layer_quota = min(
                remaining_budget,
                max(
                    min(len(active_root_ids), remaining_budget),
                    ceil(remaining_budget / max(1, remaining_layers)),
                ),
            )
            (
                token_ids,
                token_positions,
                parent_indices,
                beam_scores,
            ) = self._client._get_next_beams_multi(
                logits,
                parent_indices,
                parent_positions,
                parent_scores,
                self._node_root_ids,
                layer_quota,
            )

        self.result.executed_depth += 1
        if token_ids.size(-1) == 0:
            self._finished = True
        else:
            child_start = self._tree.end
            self._tree.add(
                token_ids=token_ids,
                token_positions=token_positions,
                parent_indices=parent_indices,
                logprobs=beam_scores,
                token_status=self._tree.POST_CANDIDATE,
            )
            for child_offset, parent_idx in enumerate(parent_indices.tolist()):
                root_id = self._node_root_ids[parent_idx]
                child_idx = child_start + child_offset
                self._node_root_ids[child_idx] = root_id
                root = self.result.roots[root_id]
                root.node_indices.append(child_idx)
                root.executed_depth = max(
                    root.executed_depth,
                    self.result.executed_depth,
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
        self._path_policy = config.proactive_path_policy
        self._max_deepest_leaves = config.proactive_multi_max_deepest_leaves
        self._min_bonus_per_leaf = config.proactive_multi_min_bonus_per_leaf
        self._max_bonus_per_leaf = config.proactive_multi_max_bonus_per_leaf
        self._max_roots = config.proactive_multi_max_roots
        self._min_root_probability = (
            config.proactive_multi_min_root_probability
        )
        self._leaf_temperature = config.proactive_multi_leaf_temperature
        self._depth_probability_coverage = (
            config.proactive_multi_depth_probability_coverage
        )
        self._sequence_acceptance_survival = (
            config.proactive_sequence_acceptance_survival
        )
        self._sequence_max_bonus_per_depth = (
            config.proactive_sequence_max_bonus_per_depth
        )
        self._sequence_max_roots = config.proactive_sequence_max_roots
        self._sequence_min_root_probability = (
            config.proactive_sequence_min_root_probability
        )
        self._last_sequence_path_depth: Optional[int] = None
        self._last_sequence_stop_probabilities: list[float] = []
        if self._path_policy == "sequence_depth":
            self._depth_probability_coverage = (
                config.proactive_sequence_depth_probability_coverage
            )

        # FIXME: remove hard-coded value
        self._max_len = max_len
        self._last_result: Optional[ProactiveDraftResult] = None

    @property
    def last_result(self) -> Optional[ProactiveDraftResult]:
        return self._last_result

    @torch.inference_mode()
    def draft(self, full_depth_acceptance: Optional[float] = None):
        """
        Expand tree from the best bonus token candidate.
        """

        session = self.start_session(
            self._max_beam_len,
            full_depth_acceptance,
        )
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

    def start_session(
        self,
        max_depth: int,
        full_depth_acceptance: Optional[float] = None,
    ) -> ProactiveDraftSession:
        return ProactiveDraftSession(
            client=self,
            planned_depth=min(self._max_beam_len, max(0, max_depth)),
            full_depth_acceptance=(
                config.proactive_multi_full_depth_prior
                if full_depth_acceptance is None
                else full_depth_acceptance
            ),
        )

    def _get_proactive_root_candidates(
        self,
        full_depth_acceptance: float,
    ) -> tuple[list[ProactiveRootCandidate], list[int], Optional[int], int]:
        if self._path_policy == "single_best":
            best_token_idx, best_token_id = (
                self._get_best_bonus_token_candidate()
            )
            if best_token_idx is None or best_token_id is None:
                return [], [], None, 0
            candidate = ProactiveRootCandidate(
                leaf_idx=int(best_token_idx.item()),
                token_id=int(best_token_id.item()),
                leaf_probability=1.0,
                bonus_probability=1.0,
                joint_probability=full_depth_acceptance,
            )
            leaf_depth = int(
                (
                    self._tree.positions[best_token_idx]
                    - self._tree.positions[self._tree.prefix_len - 1]
                ).item()
            )
            return [candidate], [candidate.leaf_idx], leaf_depth, 1

        if self._path_policy == "sequence_depth":
            path_indices = self._get_main_sequence_nodes()
            if path_indices.numel() == 0:
                return [], [], None, 0

            input_ids = self._tree.tokens[path_indices].unsqueeze(0)
            position_ids = self._tree.positions[path_indices].unsqueeze(0)
            attention_mask = self._tree.amask[..., path_indices, :]
            cache_batch_indices = torch.zeros_like(
                path_indices,
                dtype=torch.long,
                device=self._device,
            )
            logits = self._engine.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=path_indices,
                attention_mask=attention_mask,
            )
            logits = logits[0, -path_indices.numel() :, :]
            bonus = torch.log_softmax(logits, dim=-1).topk(
                k=self._sequence_max_bonus_per_depth,
                dim=-1,
                sorted=True,
            )
            path_depth = path_indices.numel() - 1
            stop_probabilities = acceptance_stop_probabilities(
                self._sequence_acceptance_survival,
                path_depth,
            )
            self._last_sequence_path_depth = path_depth
            self._last_sequence_stop_probabilities = stop_probabilities
            candidates = select_sequence_bonus_candidates(
                path_indices,
                stop_probabilities,
                bonus.indices,
                bonus.values,
                max_bonus_per_depth=(
                    self._sequence_max_bonus_per_depth
                ),
                max_roots=self._sequence_max_roots,
                min_root_probability=(
                    self._sequence_min_root_probability
                ),
            )
            selected_depths = {
                candidate.stop_depth for candidate in candidates
            }
            self._logger.debug(
                "Sequence depth roots: path_depth=%d depths=%d roots=%d",
                path_depth,
                len(selected_depths),
                len(candidates),
            )
            return (
                candidates,
                [int(path_indices[-1].item())],
                path_depth,
                len(selected_depths),
            )

        if self._path_policy != "deepest_multi":
            raise ValueError(
                f"Invalid proactive path policy: {self._path_policy}"
            )

        leaf_indices = self._get_leaves_nodes()
        if leaf_indices.numel() == 0:
            return [], [], None, 0
        max_leaf_position = int(
            self._tree.positions[leaf_indices].max().item()
        )
        deepest_leaf_indices = leaf_indices[
            self._tree.positions[leaf_indices] == max_leaf_position
        ]
        max_leaf_depth = max_leaf_position - int(
            self._tree.positions[self._tree.prefix_len - 1].item()
        )

        input_ids = self._tree.tokens[deepest_leaf_indices].unsqueeze(0)
        position_ids = self._tree.positions[deepest_leaf_indices].unsqueeze(0)
        attention_mask = self._tree.amask[..., deepest_leaf_indices, :]
        cache_batch_indices = torch.zeros_like(
            deepest_leaf_indices,
            dtype=torch.long,
            device=self._device,
        )
        logits = self._engine.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_batch_indices=cache_batch_indices,
            cache_seq_indices=deepest_leaf_indices,
            attention_mask=attention_mask,
        )
        logits = logits[0, -deepest_leaf_indices.numel() :, :]
        bonus = torch.log_softmax(logits, dim=-1).topk(
            k=self._max_bonus_per_leaf,
            dim=-1,
            sorted=True,
        )
        candidates, selected_leaf_count = select_bonus_candidates(
            deepest_leaf_indices,
            self._tree.logprobs[deepest_leaf_indices],
            bonus.indices,
            bonus.values,
            full_depth_acceptance=full_depth_acceptance,
            max_deepest_leaves=self._max_deepest_leaves,
            min_bonus_per_leaf=self._min_bonus_per_leaf,
            max_bonus_per_leaf=self._max_bonus_per_leaf,
            max_roots=self._max_roots,
            min_root_probability=self._min_root_probability,
            leaf_temperature=self._leaf_temperature,
        )
        self._logger.debug(
            "Deepest multi roots: depth=%d leaves=%d selected_leaves=%d roots=%d",
            max_leaf_depth,
            deepest_leaf_indices.numel(),
            selected_leaf_count,
            len(candidates),
        )
        return (
            candidates,
            [int(index.item()) for index in deepest_leaf_indices],
            max_leaf_depth,
            selected_leaf_count,
        )

    def _get_main_sequence_nodes(self) -> torch.Tensor:
        """Return prefix tail plus the highest-probability deepest path."""
        leaf_indices = self._get_leaves_nodes()
        if leaf_indices.numel() == 0:
            return torch.empty(
                0, dtype=torch.long, device=self._device
            )

        prefix_tail = self._tree.prefix_len - 1
        path = trace_main_sequence_nodes(
            leaf_indices,
            self._tree.positions,
            self._tree.logprobs,
            self._tree.parents,
            prefix_tail=prefix_tail,
        )
        return torch.tensor(
            path,
            dtype=torch.long,
            device=self._device,
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
        parent_indices = torch.unique(self._tree.parents[: self._tree.end])

        # mask for tokens that are not parents
        candidate_leaf_mask = ~torch.isin(
            torch.arange(self._tree.prefix_len, self._tree.end, device=self._device),
            parent_indices,
        )

        if candidate_leaf_mask.sum() > self._max_n_beams:
            candidate_leaf_indices = (
                torch.where(candidate_leaf_mask)[0] + self._tree.prefix_len
            )
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

    def _process_candidates(
        self,
        candidate_indices: Optional[torch.Tensor] = None,
    ):
        """
        Process candidates from continuous draft tree.
        """

        if candidate_indices is None:
            candidate_indices = torch.where(
                self._tree.status[: self._tree.end]
                == self._tree.POST_CANDIDATE
            )[0]

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

    def _get_next_beams_multi(
        self,
        logits: torch.Tensor,
        beam_indices: torch.Tensor,
        beam_positions: torch.Tensor,
        beam_scores: torch.Tensor,
        node_root_ids: dict[int, int],
        max_new_nodes: int,
    ):
        if max_new_nodes <= 0 or beam_indices.numel() == 0:
            empty_long = torch.empty(
                0, dtype=torch.long, device=self._device
            )
            empty_float = torch.empty(
                0, dtype=torch.float32, device=self._device
            )
            return empty_long, empty_long, empty_long, empty_float

        logits = logits[0, -beam_indices.numel() :, :]
        logprobs = torch.log_softmax(logits, dim=-1)
        topk = logprobs.topk(
            k=self._max_branch_width,
            dim=-1,
            sorted=True,
        )
        candidate_scores = (
            beam_scores.unsqueeze(-1) + np.log(0.95) + topk.values
        )
        flat_scores = candidate_scores.flatten()
        flat_token_ids = topk.indices.flatten()
        flat_parent_offsets = torch.arange(
            beam_indices.numel(), device=self._device
        ).unsqueeze(1).expand_as(topk.indices).flatten()

        parent_root_ids = torch.tensor(
            [node_root_ids[int(index.item())] for index in beam_indices],
            dtype=torch.long,
            device=self._device,
        )
        flat_root_ids = parent_root_ids[flat_parent_offsets]

        selected_flat_indices: list[int] = []
        for root_id in torch.unique(parent_root_ids).tolist():
            root_flat_indices = torch.where(flat_root_ids == root_id)[0]
            best_offset = flat_scores[root_flat_indices].argmax()
            selected_flat_indices.append(
                int(root_flat_indices[best_offset].item())
            )

        selected_set = set(selected_flat_indices)
        for flat_index in flat_scores.argsort(descending=True).tolist():
            if len(selected_flat_indices) >= max_new_nodes:
                break
            if flat_index not in selected_set:
                selected_flat_indices.append(flat_index)
                selected_set.add(flat_index)

        selected = torch.tensor(
            selected_flat_indices[:max_new_nodes],
            dtype=torch.long,
            device=self._device,
        )
        parent_offsets = flat_parent_offsets[selected]
        selected_parent_indices = beam_indices[parent_offsets]
        return (
            flat_token_ids[selected],
            beam_positions[parent_offsets] + 1,
            selected_parent_indices,
            flat_scores[selected],
        )

    def _get_next_sequence_multi(
        self,
        logits: torch.Tensor,
        beam_indices: torch.Tensor,
        beam_positions: torch.Tensor,
        beam_scores: torch.Tensor,
        node_root_ids: dict[int, int],
        max_new_nodes: int,
    ):
        """Extend every active bonus root by at most one greedy token."""
        if max_new_nodes <= 0 or beam_indices.numel() == 0:
            empty_long = torch.empty(
                0, dtype=torch.long, device=self._device
            )
            empty_float = torch.empty(
                0, dtype=torch.float32, device=self._device
            )
            return empty_long, empty_long, empty_long, empty_float

        logits = logits[0, -beam_indices.numel() :, :]
        logprobs = torch.log_softmax(logits, dim=-1)
        best_logprobs, best_token_ids = logprobs.max(dim=-1)
        scores = beam_scores + np.log(0.95) + best_logprobs

        best_by_root = []
        root_ids = torch.tensor(
            [node_root_ids[int(index.item())] for index in beam_indices],
            dtype=torch.long,
            device=self._device,
        )
        for root_id in torch.unique(root_ids).tolist():
            offsets = torch.where(root_ids == root_id)[0]
            best_by_root.append(
                offsets[scores[offsets].argmax()]
            )
        selected = torch.stack(best_by_root)
        if selected.numel() > max_new_nodes:
            selected = selected[
                scores[selected].topk(max_new_nodes).indices
            ]
        return (
            best_token_ids[selected],
            beam_positions[selected] + 1,
            beam_indices[selected],
            scores[selected],
        )

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
