import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from distillation.data import format_prompt
from distillation.io import (
    iter_jsonl,
    load_yaml,
    model_load_kwargs,
    open_text,
    set_seed,
    torch_dtype,
)
from specedge.tokenizer import validate_tokenizer_compatibility


def _load_model(
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    model = AutoModelForCausalLM.from_pretrained(
        name,
        **model_load_kwargs(dtype),
    ).to(device)
    model.eval()
    return model


def _generate_response(
    model,
    tokenizer,
    prompt_text: str,
    *,
    device: torch.device,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    input_ids = tokenizer.encode(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )[:, -max_prompt_tokens:].to(device)
    attention_mask = torch.ones_like(input_ids)
    do_sample = temperature > 0
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_config.update(
            {
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
    response_ids = generated[0, input_ids.size(1) :]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def _response_for_record(
    record: dict[str, Any],
    prompt_text: str,
    *,
    source_model,
    source_tokenizer,
    generation: dict,
    device: torch.device,
) -> str:
    if generation.get("reuse_response", False) and record.get("response"):
        return str(record["response"])
    return _generate_response(
        source_model,
        source_tokenizer,
        prompt_text,
        device=device,
        max_prompt_tokens=int(generation["max_prompt_tokens"]),
        max_new_tokens=int(generation["max_new_tokens"]),
        temperature=float(generation.get("temperature", 0.0)),
        top_p=float(generation.get("top_p", 1.0)),
    )


def _encode_sequence(
    tokenizer,
    prompt_text: str,
    response: str,
    *,
    max_seq_len: int,
) -> tuple[list[int], int]:
    prompt_ids = tokenizer.encode(
        prompt_text,
        add_special_tokens=False,
    )
    response_ids = tokenizer.encode(
        response,
        add_special_tokens=False,
    )
    if tokenizer.eos_token_id is not None and (
        not response_ids or response_ids[-1] != tokenizer.eos_token_id
    ):
        response_ids.append(tokenizer.eos_token_id)

    max_prompt_len = max(1, max_seq_len - len(response_ids))
    prompt_ids = prompt_ids[-max_prompt_len:]
    response_ids = response_ids[: max_seq_len - len(prompt_ids)]
    return prompt_ids + response_ids, len(prompt_ids)


def _collect_sparse_logits(
    teacher,
    input_ids: list[int],
    *,
    prompt_len: int,
    device: torch.device,
    top_k: int,
    temperature: float,
) -> dict[str, Any]:
    tokens = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    attention_mask = torch.ones_like(tokens)
    with torch.inference_mode():
        logits = teacher(
            input_ids=tokens,
            attention_mask=attention_mask,
        ).logits[0, :-1].float()
    logits = logits / temperature
    log_normalizer = torch.logsumexp(logits, dim=-1)
    topk_logits, topk_ids = logits.topk(top_k, dim=-1)
    topk_logprobs = topk_logits - log_normalizer.unsqueeze(-1)
    topk_mass = topk_logprobs.exp().sum(dim=-1)
    tail_logprobs = torch.log((1.0 - topk_mass).clamp_min(1e-8))

    prediction_positions = torch.arange(
        len(input_ids) - 1,
        device=device,
    )
    loss_mask = prediction_positions + 1 >= prompt_len
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask.cpu().tolist(),
        "teacher_topk_ids": topk_ids.cpu().tolist(),
        "teacher_topk_logprobs": topk_logprobs.cpu().tolist(),
        "teacher_tail_logprobs": tail_logprobs.cpu().tolist(),
    }


def rejection_replay_mask(
    student_top1: torch.Tensor,
    teacher_top1: torch.Tensor,
    teacher_confidence: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    min_teacher_confidence: float,
    window_size: int,
) -> torch.Tensor:
    """Select errors made on the teacher trajectory and nearby states."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    rejection_mask = (
        (student_top1 != teacher_top1)
        & response_mask
        & (teacher_confidence >= min_teacher_confidence)
    )
    replay_mask = rejection_mask.clone()
    for offset in range(1, window_size):
        replay_mask[offset:] |= rejection_mask[:-offset]
    return replay_mask & response_mask


def acceptance_gain_weights(
    agreements: torch.Tensor,
    response_mask: torch.Tensor,
    rejection_mask: torch.Tensor,
    *,
    max_accept_depth: int,
    max_weight: float,
) -> torch.Tensor:
    """Weight errors by their gain in capped contiguous acceptance depth."""
    if max_accept_depth <= 0:
        raise ValueError("max_accept_depth must be positive")
    if max_weight < 1.0:
        raise ValueError("max_weight must be at least 1")

    def depths(values: torch.Tensor) -> list[int]:
        result = [0] * len(values)
        next_depth = 0
        for index in range(len(values) - 1, -1, -1):
            if bool(response_mask[index]) and bool(values[index]):
                next_depth = min(max_accept_depth, next_depth + 1)
            else:
                next_depth = 0
            result[index] = next_depth
        return result

    base_total = sum(depths(agreements))
    weights = torch.ones_like(agreements, dtype=torch.float32)
    for index in rejection_mask.nonzero(as_tuple=False).flatten().tolist():
        corrected = agreements.clone()
        corrected[index] = True
        gain = max(1, sum(depths(corrected)) - base_total)
        weights[index] = min(
            max_weight,
            1.0 + math.log2(gain),
        )
    return weights


def _student_rejection_replay_mask(
    student,
    input_ids: list[int],
    sparse_logits: dict[str, Any],
    *,
    device: torch.device,
    min_teacher_confidence: float,
    window_size: int,
    weighting: str,
    max_accept_depth: int,
    max_weight: float,
) -> tuple[list[bool], list[float], int]:
    tokens = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    with torch.inference_mode():
        student_top1 = student(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
        ).logits[0, :-1].argmax(dim=-1)
    teacher_top1 = torch.tensor(
        sparse_logits["teacher_topk_ids"],
        dtype=torch.long,
        device=device,
    )[:, 0]
    teacher_confidence = torch.tensor(
        sparse_logits["teacher_topk_logprobs"],
        dtype=torch.float32,
        device=device,
    )[:, 0].exp()
    response_mask = torch.tensor(
        sparse_logits["loss_mask"],
        dtype=torch.bool,
        device=device,
    )
    confident_rejections = (
        (student_top1 != teacher_top1)
        & response_mask
        & (teacher_confidence >= min_teacher_confidence)
    )
    replay_mask = rejection_replay_mask(
        student_top1,
        teacher_top1,
        teacher_confidence,
        response_mask,
        min_teacher_confidence=min_teacher_confidence,
        window_size=window_size,
    )
    if weighting == "uniform":
        loss_weights = torch.ones_like(
            teacher_confidence,
            dtype=torch.float32,
        )
    elif weighting == "acceptance_gain":
        loss_weights = acceptance_gain_weights(
            student_top1 == teacher_top1,
            response_mask,
            confident_rejections,
            max_accept_depth=max_accept_depth,
            max_weight=max_weight,
        )
    else:
        raise ValueError(
            "loss_mask.weighting must be uniform or acceptance_gain"
        )
    return (
        replay_mask.cpu().tolist(),
        loss_weights.cpu().tolist(),
        int(replay_mask.sum()),
    )


def collect(config: dict) -> int:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    mode = str(config["mode"])
    if mode not in {"sft", "kd"}:
        raise ValueError("mode must be sft or kd")

    input_path = Path(config["input_path"])
    if (
        input_path.stem == "test"
        and not config.get("allow_test_input", False)
    ):
        raise ValueError(
            "Refusing to collect training data from a test split. "
            "Set allow_test_input only for diagnostic evaluation."
        )

    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.get("device", "cuda:0"))
    dtype = torch_dtype(config.get("dtype", "fp16"))
    teacher_name = str(config["teacher_model"])
    student_name = config.get("student_model")
    generation = config["generation"]
    loss_mask_config = config.get("loss_mask", {})
    loss_mask_mode = str(loss_mask_config.get("mode", "response"))
    if loss_mask_mode not in {"response", "student_rejections"}:
        raise ValueError(
            "loss_mask.mode must be response or student_rejections"
        )

    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_name,
        legacy=False,
    )
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
    teacher = _load_model(
        teacher_name,
        device=device,
        dtype=dtype,
    )

    source_name = teacher_name
    source_model = teacher
    source_tokenizer = teacher_tokenizer
    student = None
    needs_student = (
        generation.get("source", "teacher") == "student"
        or loss_mask_mode == "student_rejections"
    )
    if needs_student:
        if not student_name:
            raise ValueError(
                "student_model is required for student generation "
                "or rejection replay"
            )
        student_tokenizer = AutoTokenizer.from_pretrained(
            student_name,
            legacy=False,
        )
        if student_tokenizer.pad_token_id is None:
            student_tokenizer.pad_token_id = (
                student_tokenizer.eos_token_id
            )
        validate_tokenizer_compatibility(
            student_tokenizer,
            teacher_tokenizer,
            ["Tokenizer compatibility check."],
        )
        student = _load_model(
            str(student_name),
            device=device,
            dtype=dtype,
        )
        if generation.get("source", "teacher") == "student":
            source_model = student
            source_tokenizer = student_tokenizer
            source_name = str(student_name)

    max_samples = int(config.get("max_samples", -1))
    apply_template = bool(config.get("apply_chat_template", True))
    count = 0
    with open_text(output_path, "wt") as output_file:
        for record in iter_jsonl(input_path):
            if max_samples >= 0 and count >= max_samples:
                break
            prompt_text = format_prompt(
                record,
                teacher_tokenizer,
                apply_chat_template=apply_template,
            )
            response = _response_for_record(
                record,
                prompt_text,
                source_model=source_model,
                source_tokenizer=source_tokenizer,
                generation=generation,
                device=device,
            )
            output = {
                "prompt_id": record.get("prompt_id"),
                "source": record.get("source"),
                "prompt_text": prompt_text,
                "response": response,
                "generation_source": source_name,
            }
            if mode == "kd":
                input_ids, prompt_len = _encode_sequence(
                    teacher_tokenizer,
                    prompt_text,
                    response,
                    max_seq_len=int(config["max_seq_len"]),
                )
                if len(input_ids) < 2 or prompt_len >= len(input_ids):
                    continue
                sparse_logits = _collect_sparse_logits(
                    teacher,
                    input_ids,
                    prompt_len=prompt_len,
                    device=device,
                    top_k=int(config.get("teacher_top_k", 64)),
                    temperature=float(
                        config.get(
                            "distillation_temperature",
                            1.0,
                        )
                    ),
                )
                if loss_mask_mode == "student_rejections":
                    replay_mask, loss_weights, replay_tokens = (
                        _student_rejection_replay_mask(
                            student,
                            input_ids,
                            sparse_logits,
                            device=device,
                            min_teacher_confidence=float(
                                loss_mask_config.get(
                                    "min_teacher_confidence",
                                    0.0,
                                )
                            ),
                            window_size=int(
                                loss_mask_config.get(
                                    "window_size",
                                    1,
                                )
                            ),
                            weighting=str(
                                loss_mask_config.get(
                                    "weighting",
                                    "uniform",
                                )
                            ),
                            max_accept_depth=int(
                                loss_mask_config.get(
                                    "max_accept_depth",
                                    7,
                                )
                            ),
                            max_weight=float(
                                loss_mask_config.get(
                                    "max_weight",
                                    6.0,
                                )
                            ),
                        )
                    )
                    if (
                        replay_tokens == 0
                        and loss_mask_config.get(
                            "drop_no_rejections",
                            True,
                        )
                    ):
                        continue
                    sparse_logits["loss_mask"] = replay_mask
                    sparse_logits["loss_weights"] = loss_weights
                    output["replay_tokens"] = replay_tokens
                output.update(sparse_logits)
            output_file.write(
                json.dumps(
                    output,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
            if count % 10 == 0:
                print(f"Collected {count} records", flush=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    count = collect(load_yaml(args.config))
    print(f"Finished collecting {count} records")


if __name__ == "__main__":
    main()
