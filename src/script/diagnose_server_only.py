import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _top_tokens(logits, tokenizer, count=16):
    values, indices = logits.topk(count)
    return [
        {
            "id": int(token_id),
            "logit": float(value),
            "text": tokenizer.decode([int(token_id)]),
        }
        for value, token_id in zip(values, indices, strict=True)
    ]


def _target_probability_mass(logits, token_ids, temperature, top_p=0.8):
    scores = logits.float() / temperature
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    cumulative_probs = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
    remove = cumulative_probs > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    sorted_scores[remove] = -torch.inf

    probs = torch.softmax(sorted_scores, dim=-1)
    original_probs = torch.zeros_like(probs).scatter(0, sorted_indices, probs)
    return float(original_probs[token_ids].sum())


def _parameter_stats(model):
    total = 0
    non_finite = 0
    for parameter in model.parameters():
        total += parameter.numel()
        non_finite += int((~torch.isfinite(parameter)).sum().item())
    return {
        "parameter_count": total,
        "non_finite_parameters": non_finite,
    }


def _reference_logits(
    model_name,
    token_ids,
    device,
    dtype,
    attn_implementation=None,
):
    from transformers import AutoModelForCausalLM

    load_options = {
        "dtype": dtype,
        "device_map": device,
    }
    if attn_implementation is not None:
        load_options["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_options).eval()
    parameter_stats = _parameter_stats(model)
    with torch.inference_mode():
        logits = model(input_ids=token_ids).logits[0, -1].float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return logits, parameter_stats


def _custom_logits(model_name, token_ids, device, dtype, max_len):
    import util
    from specedge.engine.graph import BatchGraphEngine

    model = util.load_graph_model(model_name, device=device, dtype=dtype)
    engine = BatchGraphEngine(
        model=model,
        max_len=max_len,
        max_batch_size=1,
        max_n_beams=1,
        use_cuda_graph=False,
    )

    prefill_ids = token_ids[..., :-1]
    prefill_len = prefill_ids.size(-1)
    prefill_positions = torch.arange(prefill_len, device=device).unsqueeze(0)
    prefill_cache_indices = torch.arange(prefill_len, device=device)
    prefill_mask = torch.ones(
        (1, 1, prefill_len, max_len), dtype=dtype, device=device
    ).tril_()
    engine.prefill(
        input_ids=prefill_ids,
        position_ids=prefill_positions,
        batch_idx=0,
        cache_seq_indices=prefill_cache_indices,
        attention_mask=prefill_mask,
    )

    root_position = token_ids.size(-1) - 1
    root_mask = torch.zeros((1, 1, 1, max_len), dtype=dtype, device=device)
    root_mask[..., : root_position + 1] = 1
    with torch.inference_mode():
        logits = engine.forward(
            input_ids=token_ids[..., -1:],
            position_ids=torch.tensor([[root_position]], device=device),
            cache_batch_indices=torch.tensor([0], device=device),
            cache_seq_indices=torch.tensor([root_position], device=device),
            attention_mask=root_mask,
        )[0, 0].float().cpu()

    del engine, model
    gc.collect()
    torch.cuda.empty_cache()
    return logits


def _custom_graph_logits(
    model_name,
    token_ids,
    device,
    dtype,
    max_len,
    max_n_beams,
):
    import util
    from specedge.engine.graph import BatchGraphEngine

    model = util.load_graph_model(model_name, device=device, dtype=dtype)
    engine = BatchGraphEngine(
        model=model,
        max_len=max_len,
        max_batch_size=1,
        max_n_beams=max_n_beams,
        use_cuda_graph=True,
    )

    prefill_ids = token_ids[..., :-1]
    prefill_len = prefill_ids.size(-1)
    engine.prefill(
        input_ids=prefill_ids,
        position_ids=torch.arange(prefill_len, device=device).unsqueeze(0),
        batch_idx=0,
        cache_seq_indices=torch.arange(prefill_len, device=device),
        attention_mask=torch.ones(
            (1, 1, prefill_len, max_len), dtype=dtype, device=device
        ).tril_(),
    )

    root_index = token_ids.size(-1) - 1
    candidate_scores = torch.full(
        (max_len,), torch.finfo(torch.float32).min, device=device
    )
    candidate_scores[root_index] = 0
    raw_candidate_indices = candidate_scores.topk(
        k=max_n_beams, sorted=False
    ).indices
    root_slot = int(torch.where(raw_candidate_indices == root_index)[0].item())

    input_ids = torch.zeros((1, max_n_beams), dtype=torch.long, device=device)
    position_ids = torch.full(
        (1, max_n_beams), max_len - 1, dtype=torch.long, device=device
    )
    cache_seq_indices = torch.full(
        (max_n_beams,), max_len - 1, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (1, 1, max_n_beams, max_len), dtype=dtype, device=device
    )
    input_ids[0, root_slot] = token_ids[0, -1]
    position_ids[0, root_slot] = root_index
    cache_seq_indices[root_slot] = root_index
    attention_mask[0, 0, root_slot, : root_index + 1] = 1

    with torch.inference_mode():
        logits = engine.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_batch_indices=torch.zeros(
                max_n_beams, dtype=torch.long, device=device
            ),
            cache_seq_indices=cache_seq_indices,
            attention_mask=attention_mask,
        )[0, root_slot].float().cpu()

    del engine, model
    gc.collect()
    torch.cuda.empty_cache()
    return logits


def _compare_model(
    model_name,
    prompt,
    device,
    dtype,
    max_len,
    graph_beams=None,
):
    import util

    tokenizer = util.load_tokenizer(model_name)
    token_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    reference, parameter_stats = _reference_logits(
        model_name,
        token_ids,
        device,
        dtype,
    )
    custom = _custom_logits(model_name, token_ids, device, dtype, max_len)

    result = {
        "prompt_tokens": token_ids.size(-1),
        **parameter_stats,
        "reference_non_finite_logits": int(
            (~torch.isfinite(reference)).sum().item()
        ),
        "custom_non_finite_logits": int((~torch.isfinite(custom)).sum().item()),
        "max_abs_logit_error": float((reference - custom).abs().max()),
        "mean_abs_logit_error": float((reference - custom).abs().mean()),
        "reference_top_tokens": _top_tokens(reference, tokenizer),
        "custom_top_tokens": _top_tokens(custom, tokenizer),
        "reference_logits": reference,
        "custom_logits": custom,
    }
    if graph_beams is not None:
        graph = _custom_graph_logits(
            model_name,
            token_ids,
            device,
            dtype,
            max_len,
            graph_beams,
        )
        result.update(
            {
                "graph_max_abs_logit_error": float(
                    (reference - graph).abs().max()
                ),
                "graph_mean_abs_logit_error": float(
                    (reference - graph).abs().mean()
                ),
                "graph_top_tokens": _top_tokens(graph, tokenizer),
                "graph_logits": graph,
            }
        )
    return result


def _diagnose_target_precision(model_name, prompt, device, max_len):
    import util

    tokenizer = util.load_tokenizer(model_name)
    token_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    results = {}
    for name, dtype, attn_implementation in [
        ("fp16_sdpa", torch.float16, None),
        ("fp16_eager", torch.float16, "eager"),
        ("bf16_sdpa", torch.bfloat16, None),
        ("bf16_eager", torch.bfloat16, "eager"),
    ]:
        logits, parameter_stats = _reference_logits(
            model_name,
            token_ids,
            device,
            dtype,
            attn_implementation=attn_implementation,
        )
        results[name] = {
            **parameter_stats,
            "non_finite_logits": int((~torch.isfinite(logits)).sum().item()),
            "finite_logit_min": (
                float(logits[torch.isfinite(logits)].min())
                if torch.isfinite(logits).any()
                else None
            ),
            "finite_logit_max": (
                float(logits[torch.isfinite(logits)].max())
                if torch.isfinite(logits).any()
                else None
            ),
        }

    custom_bf16 = _custom_logits(
        model_name,
        token_ids,
        device,
        torch.bfloat16,
        max_len,
    )
    results["bf16_custom"] = {
        "non_finite_logits": int(
            (~torch.isfinite(custom_bf16)).sum().item()
        )
    }
    return results


def main(config_path: Path, request_index: int, precision_check: bool):
    import util

    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)

    dtype = util.convert_dtype(config["base"]["dtype"])
    max_len = config["base"]["max_len"]
    draft_name = config["client"]["draft_model"]
    target_name = config["server"]["target_model"]
    draft_device = torch.device(config["client"]["device"])
    target_device = torch.device(config["server"]["device"])
    temperature = config["server"]["temperature"]

    dataset = util.load_dataset(
        config["client"]["dataset"],
        model_name=draft_name,
    )
    prompt = dataset[request_index]

    draft_tokenizer = util.load_tokenizer(draft_name)
    target_tokenizer = util.load_tokenizer(target_name)
    draft_prompt_ids = draft_tokenizer.encode(prompt)
    target_prompt_ids = target_tokenizer.encode(prompt)
    if draft_prompt_ids != target_prompt_ids:
        raise ValueError("Draft and target prompt token IDs differ.")

    draft = _compare_model(
        draft_name,
        prompt,
        draft_device,
        dtype,
        max_len,
        graph_beams=config["client"]["max_n_beams"],
    )
    target = _compare_model(target_name, prompt, target_device, dtype, max_len)

    draft_top_ids = torch.tensor(
        [entry["id"] for entry in draft["reference_top_tokens"]]
    )
    report = {
        "request_index": request_index,
        "draft_model": draft_name,
        "target_model": target_name,
        "draft": {
            key: value
            for key, value in draft.items()
            if not key.endswith("_logits")
        },
        "target": {
            key: value
            for key, value in target.items()
            if not key.endswith("_logits")
        },
        "reference_root_candidate_probability": _target_probability_mass(
            target["reference_logits"],
            draft_top_ids,
            temperature=temperature,
        ),
        "custom_root_candidate_probability": _target_probability_mass(
            target["custom_logits"],
            torch.tensor(
                [entry["id"] for entry in draft["custom_top_tokens"]]
            ),
            temperature=temperature,
        ),
        "graph_root_candidate_probability": _target_probability_mass(
            target["custom_logits"],
            torch.tensor(
                [entry["id"] for entry in draft["graph_top_tokens"]]
            ),
            temperature=temperature,
        ),
    }
    if precision_check:
        report["target_precision_check"] = _diagnose_target_precision(
            target_name,
            prompt,
            target_device,
            max_len,
        )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/server_only_4090.yaml"),
    )
    parser.add_argument("--request-index", type=int, default=0)
    parser.add_argument("--precision-check", action="store_true")
    args = parser.parse_args()
    main(args.config, args.request_index, args.precision_check)
