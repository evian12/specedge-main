import torch


def copy_padded_1d(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    name: str,
) -> int:
    source_len = source.numel()
    target_len = target.numel()
    if source_len > target_len:
        raise ValueError(
            f"{name} has {source_len} entries, but server capacity is {target_len}. "
            "Increase client.max_budget or reduce the draft request size."
        )
    target.zero_()
    if source_len > 0:
        target[:source_len].copy_(source.reshape(-1))
    return source_len


def copy_padded_attention_mask(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    name: str,
) -> int:
    source_len = source.size(1)
    target_len = target.size(1)
    if source_len > target_len:
        raise ValueError(
            f"{name} has {source_len} rows, but server capacity is {target_len}. "
            "Increase client.max_budget or reduce the draft request size."
        )
    target.zero_()
    if source_len > 0:
        target[:, :source_len, :].copy_(source)
    return source_len


def validate_draft_request_shapes(
    *,
    input_len: int,
    position_len: int,
    cache_len: int,
    attention_len: int,
    parent_len: int,
) -> None:
    if not (input_len == position_len == cache_len == attention_len):
        raise ValueError(
            "Draft request tensors disagree on node count: "
            f"input_ids={input_len}, position_ids={position_len}, "
            f"cache_seq_indices={cache_len}, attention_mask={attention_len}."
        )
    expected_parent_len = max(0, input_len - 1)
    if parent_len != expected_parent_len:
        raise ValueError(
            "parent_indices must contain exactly one parent per draft node: "
            f"got {parent_len}, expected {expected_parent_len} for "
            f"{input_len} input nodes."
        )
