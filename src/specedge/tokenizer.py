def validate_tokenizer_compatibility(
    draft_tokenizer,
    target_tokenizer,
    prompts,
):
    if len(draft_tokenizer) != len(target_tokenizer):
        raise ValueError(
            "Draft and target tokenizers have different vocabulary sizes: "
            f"{len(draft_tokenizer)} != {len(target_tokenizer)}. "
            "Speculative decoding requires identical token IDs."
        )

    special_token_ids = (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
    )
    mismatched_special_tokens = [
        token_name
        for token_name in special_token_ids
        if getattr(draft_tokenizer, token_name, None)
        != getattr(target_tokenizer, token_name, None)
    ]
    if mismatched_special_tokens:
        raise ValueError(
            "Draft and target tokenizers use different special token IDs: "
            f"{', '.join(mismatched_special_tokens)}. "
            "Speculative decoding requires identical token IDs."
        )

    for prompt_idx, prompt in enumerate(prompts):
        draft_ids = draft_tokenizer.encode(prompt)
        target_ids = target_tokenizer.encode(prompt)
        if draft_ids != target_ids:
            raise ValueError(
                "Draft and target tokenizers produce different token IDs for "
                f"prompt {prompt_idx}. Speculative decoding requires compatible "
                "tokenizers; use models from the same tokenizer family."
            )
