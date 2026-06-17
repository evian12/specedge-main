from dataclasses import dataclass
from math import floor

import torch


@dataclass(frozen=True)
class ProactiveRootCandidate:
    leaf_idx: int
    token_id: int
    leaf_probability: float
    bonus_probability: float
    joint_probability: float
    stop_depth: int | None = None


def acceptance_stop_probabilities(
    survival: list[float],
    max_depth: int,
) -> list[float]:
    """Convert P(accepted depth >= d) into P(accepted depth == d)."""
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if not survival:
        raise ValueError("acceptance survival probabilities are required")

    values = [
        survival[min(depth, len(survival) - 1)]
        for depth in range(max_depth + 1)
    ]
    stop_probabilities = [
        max(0.0, values[depth] - values[depth + 1])
        for depth in range(max_depth)
    ]
    stop_probabilities.append(max(0.0, values[max_depth]))
    return stop_probabilities


def allocate_sequence_bonus_counts(
    stop_probabilities: list[float],
    *,
    max_roots: int,
    max_bonus_per_depth: int,
) -> list[int]:
    """Allocate a bounded number of bonus roots proportional to stop mass."""
    if max_roots < 0:
        raise ValueError("max_roots must be non-negative")
    if max_bonus_per_depth <= 0:
        raise ValueError("max_bonus_per_depth must be positive")
    if not stop_probabilities or max_roots == 0:
        return [0] * len(stop_probabilities)

    weights = [max(0.0, value) for value in stop_probabilities]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return [0] * len(weights)

    positive_depths = [
        depth for depth, weight in enumerate(weights) if weight > 0.0
    ]
    slots = min(
        max_roots,
        len(positive_depths) * max_bonus_per_depth,
    )
    ideals = [slots * weight / total_weight for weight in weights]
    counts = [
        min(max_bonus_per_depth, floor(ideal))
        for ideal in ideals
    ]
    remaining = slots - sum(counts)
    order = sorted(
        positive_depths,
        key=lambda depth: (
            ideals[depth] - floor(ideals[depth]),
            weights[depth],
            -depth,
        ),
        reverse=True,
    )
    while remaining > 0:
        allocated = False
        for depth in order:
            if counts[depth] >= max_bonus_per_depth:
                continue
            counts[depth] += 1
            remaining -= 1
            allocated = True
            if remaining == 0:
                break
        if not allocated:
            break
    return counts


def select_sequence_bonus_candidates(
    path_node_indices: torch.Tensor,
    stop_probabilities: list[float],
    bonus_token_ids: torch.Tensor,
    bonus_logprobs: torch.Tensor,
    *,
    max_bonus_per_depth: int,
    max_roots: int,
    min_root_probability: float,
    min_bonus_probability: float = 0.0,
    selection_score: str = "joint",
    reuse_depth_bonus: float = 0.0,
) -> list[ProactiveRootCandidate]:
    """Select bonus roots across all possible stopping depths of one path."""
    depth_count = path_node_indices.numel()
    if depth_count == 0 or max_roots <= 0:
        return []
    if len(stop_probabilities) != depth_count:
        raise ValueError(
            "stop probabilities must match the sequence path length"
        )
    if bonus_token_ids.size(0) != depth_count:
        raise ValueError("bonus token rows must match the sequence path")
    if min_bonus_probability < 0.0:
        raise ValueError("min_bonus_probability must be non-negative")
    if reuse_depth_bonus < 0.0:
        raise ValueError("reuse_depth_bonus must be non-negative")
    if selection_score not in ["joint", "expected_reuse"]:
        raise ValueError("selection_score must be 'joint' or 'expected_reuse'")

    available_bonus = min(
        max_bonus_per_depth,
        bonus_token_ids.size(1),
        bonus_logprobs.size(1),
    )
    counts = allocate_sequence_bonus_counts(
        stop_probabilities,
        max_roots=max_roots,
        max_bonus_per_depth=available_bonus,
    )

    candidates: list[ProactiveRootCandidate] = []
    scored_candidates: list[tuple[float, ProactiveRootCandidate]] = []
    for depth, count in enumerate(counts):
        stop_probability = stop_probabilities[depth]
        for bonus_rank in range(count):
            bonus_probability = float(
                bonus_logprobs[depth, bonus_rank].exp().item()
            )
            if bonus_probability < min_bonus_probability:
                continue
            joint_probability = stop_probability * bonus_probability
            if joint_probability < min_root_probability:
                continue
            candidate = ProactiveRootCandidate(
                leaf_idx=int(path_node_indices[depth].item()),
                token_id=int(
                    bonus_token_ids[depth, bonus_rank].item()
                ),
                leaf_probability=stop_probability,
                bonus_probability=bonus_probability,
                joint_probability=joint_probability,
                stop_depth=depth,
            )
            if selection_score == "expected_reuse":
                reusable_tail = max(0, depth_count - depth - 1)
                score = joint_probability * (
                    1.0 + reuse_depth_bonus * reusable_tail
                )
            else:
                score = joint_probability
            scored_candidates.append((score, candidate))

    scored_candidates.sort(
        key=lambda item: (
            item[0],
            item[1].joint_probability,
            item[1].bonus_probability,
        ),
        reverse=True,
    )
    candidates = [candidate for _, candidate in scored_candidates]
    return candidates[:max_roots]


def trace_main_sequence_nodes(
    leaf_indices: torch.Tensor,
    positions: torch.Tensor,
    logprobs: torch.Tensor,
    parents: torch.Tensor,
    *,
    prefix_tail: int,
) -> list[int]:
    """Trace the highest-probability leaf at maximum depth to the prefix."""
    if leaf_indices.numel() == 0:
        return []
    deepest_position = positions[leaf_indices].max()
    deepest_leaves = leaf_indices[
        positions[leaf_indices] == deepest_position
    ]
    best_leaf = deepest_leaves[logprobs[deepest_leaves].argmax()]

    path = []
    node_idx = int(best_leaf.item())
    while node_idx > prefix_tail:
        path.append(node_idx)
        node_idx = int(parents[node_idx].item())
    if node_idx != prefix_tail:
        return []
    path.append(prefix_tail)
    path.reverse()
    return path


def select_bonus_candidates(
    leaf_indices: torch.Tensor,
    leaf_scores: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    bonus_logprobs: torch.Tensor,
    *,
    full_depth_acceptance: float,
    max_deepest_leaves: int,
    min_bonus_per_leaf: int,
    max_bonus_per_leaf: int,
    max_roots: int,
    min_root_probability: float,
    leaf_temperature: float,
) -> tuple[list[ProactiveRootCandidate], int]:
    """Allocate bonus candidates across the most likely deepest leaves."""
    if leaf_indices.numel() == 0 or max_roots <= 0:
        return [], 0
    if min_bonus_per_leaf <= 0:
        raise ValueError("min_bonus_per_leaf must be positive")
    if max_bonus_per_leaf < min_bonus_per_leaf:
        raise ValueError(
            "max_bonus_per_leaf must be >= min_bonus_per_leaf"
        )
    if leaf_temperature <= 0.0:
        raise ValueError("leaf_temperature must be positive")

    available_bonus = min(
        max_bonus_per_leaf,
        bonus_token_ids.size(1),
        bonus_logprobs.size(1),
    )
    if available_bonus < min_bonus_per_leaf:
        raise ValueError("not enough bonus candidates for the configured minimum")

    leaf_limit = min(
        leaf_indices.numel(),
        max_deepest_leaves,
        max_roots // min_bonus_per_leaf,
    )
    if leaf_limit <= 0:
        return [], 0

    selected_offsets = torch.topk(
        leaf_scores,
        k=leaf_limit,
        sorted=True,
    ).indices
    selected_leaf_indices = leaf_indices[selected_offsets]
    selected_leaf_scores = leaf_scores[selected_offsets]
    selected_bonus_ids = bonus_token_ids[selected_offsets, :available_bonus]
    selected_bonus_logprobs = bonus_logprobs[
        selected_offsets, :available_bonus
    ]

    leaf_probabilities = torch.softmax(
        selected_leaf_scores / leaf_temperature,
        dim=0,
    )
    bonus_probabilities = selected_bonus_logprobs.exp()

    candidates: list[ProactiveRootCandidate] = []
    optional: list[ProactiveRootCandidate] = []
    for leaf_offset in range(leaf_limit):
        for bonus_rank in range(available_bonus):
            leaf_probability = float(leaf_probabilities[leaf_offset].item())
            bonus_probability = float(
                bonus_probabilities[leaf_offset, bonus_rank].item()
            )
            candidate = ProactiveRootCandidate(
                leaf_idx=int(selected_leaf_indices[leaf_offset].item()),
                token_id=int(selected_bonus_ids[leaf_offset, bonus_rank].item()),
                leaf_probability=leaf_probability,
                bonus_probability=bonus_probability,
                joint_probability=(
                    full_depth_acceptance
                    * leaf_probability
                    * bonus_probability
                ),
            )
            if bonus_rank < min_bonus_per_leaf:
                candidates.append(candidate)
            elif candidate.joint_probability >= min_root_probability:
                optional.append(candidate)

    optional.sort(key=lambda candidate: candidate.joint_probability, reverse=True)
    candidates.extend(optional[: max(0, max_roots - len(candidates))])
    candidates.sort(key=lambda candidate: candidate.joint_probability, reverse=True)
    return candidates[:max_roots], leaf_limit


def select_ids_by_probability(
    probabilities: list[tuple[int, float]],
    coverage: float,
) -> set[int]:
    if not probabilities or coverage <= 0.0:
        return set()
    if coverage >= 1.0:
        return {item_id for item_id, _ in probabilities}

    ordered = sorted(
        probabilities,
        key=lambda item: item[1],
        reverse=True,
    )
    total_probability = sum(probability for _, probability in ordered)
    if total_probability <= 0.0:
        return {ordered[0][0]}

    selected: set[int] = set()
    cumulative_probability = 0.0
    for item_id, probability in ordered:
        selected.add(item_id)
        cumulative_probability += probability
        if cumulative_probability >= coverage * total_probability:
            break
    return selected
