from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProactiveRootCandidate:
    leaf_idx: int
    token_id: int
    leaf_probability: float
    bonus_probability: float
    joint_probability: float


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
