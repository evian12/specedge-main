import torch
import torch.nn.functional as F


def _weighted_masked_mean(
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not loss_mask.any():
        return values.sum() * 0.0
    selected = values[loss_mask]
    if token_weights is None:
        return selected.mean()
    weights = token_weights[loss_mask].to(selected.dtype)
    return (selected * weights).sum() / weights.sum().clamp_min(1e-8)


def masked_token_loss(
    shifted_logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not loss_mask.any():
        return shifted_logits.sum() * 0.0
    losses = F.cross_entropy(
        shifted_logits[loss_mask],
        targets[loss_mask],
        reduction="none",
    )
    if token_weights is None:
        return losses.mean()
    weights = token_weights[loss_mask].to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


def top1_margin_loss(
    shifted_logits: torch.Tensor,
    teacher_top1: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    margin: float,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Require the teacher Top-1 token to outrank every alternative."""
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if not loss_mask.any():
        return shifted_logits.sum() * 0.0

    teacher_logits = shifted_logits.gather(
        -1,
        teacher_top1.unsqueeze(-1),
    ).squeeze(-1)
    top_values, top_ids = shifted_logits.topk(2, dim=-1)
    strongest_alternative = torch.where(
        top_ids[..., 0] == teacher_top1,
        top_values[..., 1],
        top_values[..., 0],
    )
    losses = torch.relu(
        margin - teacher_logits + strongest_alternative
    )
    return _weighted_masked_mean(
        losses,
        loss_mask,
        token_weights,
    )


def hard_token_loss(
    student_logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    return masked_token_loss(
        student_logits[:, :-1, :],
        input_ids[:, 1:],
        loss_mask,
    )


def sparse_topk_distillation_loss(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_tail_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    temperature: float,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not loss_mask.any():
        return student_logits.sum() * 0.0

    student_logprobs = F.log_softmax(
        student_logits[:, :-1, :] / temperature,
        dim=-1,
    )
    selected_student_logprobs = student_logprobs.gather(
        -1,
        teacher_topk_ids,
    )
    teacher_topk_probs = teacher_topk_logprobs.exp()
    topk_cross_entropy = -(
        teacher_topk_probs * selected_student_logprobs
    ).sum(dim=-1)

    student_topk_mass = selected_student_logprobs.exp().sum(dim=-1)
    student_tail_logprob = torch.log(
        (1.0 - student_topk_mass).clamp_min(1e-8)
    )
    teacher_tail_prob = teacher_tail_logprobs.exp()
    tail_cross_entropy = -teacher_tail_prob * student_tail_logprob

    loss = topk_cross_entropy + tail_cross_entropy
    return (
        _weighted_masked_mean(loss, loss_mask, token_weights)
        * temperature**2
    )


def sparse_topk_total_variation_loss(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_tail_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    temperature: float,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Approximate total variation using teacher top-k plus one tail bucket."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not loss_mask.any():
        return student_logits.sum() * 0.0

    student_probs = F.softmax(
        student_logits[:, :-1, :] / temperature,
        dim=-1,
    )
    selected_student_probs = student_probs.gather(
        -1,
        teacher_topk_ids,
    )
    teacher_topk_probs = teacher_topk_logprobs.exp()
    student_tail_prob = (
        1.0 - selected_student_probs.sum(dim=-1)
    ).clamp_min(0.0)
    teacher_tail_prob = teacher_tail_logprobs.exp()
    total_variation = 0.5 * (
        (selected_student_probs - teacher_topk_probs)
        .abs()
        .sum(dim=-1)
        + (student_tail_prob - teacher_tail_prob).abs()
    )
    return _weighted_masked_mean(
        total_variation,
        loss_mask,
        token_weights,
    )


def rejection_window_weights(
    rejection_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    window_size: int,
    rejection_weight: float,
) -> torch.Tensor:
    """Emphasize verification-exposed errors and their following states."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if rejection_weight < 1.0:
        raise ValueError("rejection_weight must be at least 1")

    weights = torch.ones_like(loss_mask, dtype=torch.float32)
    for offset in range(window_size):
        shifted = torch.zeros_like(rejection_mask)
        if offset == 0:
            shifted = rejection_mask
        else:
            shifted[:, offset:] = rejection_mask[:, :-offset]
        factor = 1.0 + (
            rejection_weight - 1.0
        ) * (window_size - offset) / window_size
        weights = torch.maximum(
            weights,
            torch.where(
                shifted,
                torch.full_like(weights, factor),
                torch.ones_like(weights),
            ),
        )
    return weights * loss_mask
