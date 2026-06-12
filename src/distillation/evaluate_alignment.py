import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from distillation.collect_teacher import _encode_sequence
from distillation.io import (
    iter_jsonl,
    load_yaml,
    model_load_kwargs,
    set_seed,
    torch_dtype,
)
from specedge.tokenizer import validate_tokenizer_compatibility


def _load_model(name: str, device: torch.device, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        name,
        **model_load_kwargs(dtype),
    ).to(device)
    model.eval()
    return model


def _accept_depths(
    agreements: list[bool],
    *,
    max_depth: int,
) -> list[int]:
    depths = [0] * len(agreements)
    next_depth = 0
    for index in range(len(agreements) - 1, -1, -1):
        if agreements[index]:
            next_depth = min(max_depth, next_depth + 1)
        else:
            next_depth = 0
        depths[index] = next_depth
    return depths


def evaluate(config: dict) -> dict:
    set_seed(int(config.get("seed", 42)))
    device = torch.device(config.get("device", "cuda:0"))
    dtype = torch_dtype(config.get("dtype", "fp16"))
    teacher_name = str(config["teacher_model"])
    student_name = str(config["student_model"])
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_name,
        legacy=False,
    )
    student_tokenizer = AutoTokenizer.from_pretrained(
        student_name,
        legacy=False,
    )
    for tokenizer in (teacher_tokenizer, student_tokenizer):
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    validate_tokenizer_compatibility(
        student_tokenizer,
        teacher_tokenizer,
        ["Tokenizer compatibility check."],
    )

    teacher = _load_model(teacher_name, device, dtype)
    student = _load_model(student_name, device, dtype)
    max_samples = int(config.get("max_samples", -1))
    max_seq_len = int(config.get("max_seq_len", 2048))
    student_top_k = int(config.get("student_top_k", 8))
    max_accept_depth = int(config.get("max_accept_depth", 7))

    top1_agreements = []
    teacher_in_student_topk = []
    student_cross_entropies = []
    accepted_depths = []
    sample_count = 0
    token_count = 0
    for record in iter_jsonl(Path(config["input_path"])):
        if max_samples >= 0 and sample_count >= max_samples:
            break
        if "prompt_text" not in record or "response" not in record:
            raise ValueError(
                "Alignment evaluation expects SFT records with "
                "prompt_text and response"
            )
        input_ids, prompt_len = _encode_sequence(
            teacher_tokenizer,
            str(record["prompt_text"]),
            str(record["response"]),
            max_seq_len=max_seq_len,
        )
        if prompt_len >= len(input_ids):
            continue
        tokens = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        attention_mask = torch.ones_like(tokens)
        with torch.inference_mode():
            teacher_logits = teacher(
                input_ids=tokens,
                attention_mask=attention_mask,
            ).logits[0, :-1].float()
            student_logits = student(
                input_ids=tokens,
                attention_mask=attention_mask,
            ).logits[0, :-1].float()

        start = max(0, prompt_len - 1)
        teacher_logits = teacher_logits[start:]
        student_logits = student_logits[start:]
        targets = tokens[0, 1:][start:]
        teacher_top1 = teacher_logits.argmax(dim=-1)
        student_top1 = student_logits.argmax(dim=-1)
        agreements = teacher_top1 == student_top1
        top1_agreements.extend(agreements.cpu().tolist())
        student_topk = student_logits.topk(
            student_top_k,
            dim=-1,
        ).indices
        teacher_in_student_topk.extend(
            (student_topk == teacher_top1.unsqueeze(-1))
            .any(dim=-1)
            .cpu()
            .tolist()
        )
        student_cross_entropies.extend(
            F.cross_entropy(
                student_logits,
                targets,
                reduction="none",
            )
            .cpu()
            .tolist()
        )
        accepted_depths.extend(
            _accept_depths(
                agreements.cpu().tolist(),
                max_depth=max_accept_depth,
            )
        )
        token_count += len(agreements)
        sample_count += 1

    summary = {
        "samples": sample_count,
        "tokens": token_count,
        "top1_agreement": statistics.fmean(top1_agreements),
        "teacher_top1_in_student_topk": statistics.fmean(
            teacher_in_student_topk
        ),
        "student_cross_entropy": statistics.fmean(
            student_cross_entropies
        ),
        "mean_greedy_accept_depth": statistics.fmean(accepted_depths),
        "median_greedy_accept_depth": statistics.median(
            accepted_depths
        ),
        "acceptance_survival": {
            str(depth): statistics.fmean(
                accepted >= depth for accepted in accepted_depths
            )
            for depth in range(1, max_accept_depth + 1)
        },
    }
    output_path = config.get("output_path")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(load_yaml(args.config)), indent=2))


if __name__ == "__main__":
    main()
