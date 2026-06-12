import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)

from distillation.data import (
    JsonlOffsetDataset,
    KDCollator,
    SFTCollator,
)
from distillation.io import (
    load_yaml,
    model_load_kwargs,
    set_seed,
    torch_dtype,
)
from distillation.losses import (
    masked_token_loss,
    rejection_window_weights,
    sparse_topk_distillation_loss,
    sparse_topk_total_variation_loss,
    top1_margin_loss,
)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _batch_loss(
    model,
    batch: dict,
    *,
    mode: str,
    hard_loss_weight: float,
    kd_loss_weight: float,
    tvd_loss_weight: float,
    margin_loss_weight: float,
    top1_margin: float,
    distillation_temperature: float,
    rejection_weight: float,
    rejection_window_size: int,
    min_teacher_confidence: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if mode == "sft":
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return output.loss, {"hard_loss": float(output.loss.detach())}

    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    ).logits
    teacher_top1 = batch["teacher_topk_ids"][..., 0]
    teacher_confidence = (
        batch["teacher_topk_logprobs"][..., 0].exp()
    )
    selective_mask = (
        batch["loss_mask"]
        & (teacher_confidence >= min_teacher_confidence)
    )
    rejection_mask = (
        logits[:, :-1, :].detach().argmax(dim=-1)
        != teacher_top1
    ) & selective_mask
    token_weights = rejection_window_weights(
        rejection_mask,
        selective_mask,
        window_size=rejection_window_size,
        rejection_weight=rejection_weight,
    )
    token_weights = token_weights * batch.get(
        "loss_weights",
        torch.ones_like(token_weights),
    )
    hard_loss = masked_token_loss(
        logits[:, :-1, :],
        teacher_top1,
        selective_mask,
        token_weights,
    )
    kd_loss = sparse_topk_distillation_loss(
        logits,
        batch["teacher_topk_ids"],
        batch["teacher_topk_logprobs"],
        batch["teacher_tail_logprobs"],
        selective_mask,
        temperature=distillation_temperature,
        token_weights=token_weights,
    )
    tvd_loss = sparse_topk_total_variation_loss(
        logits,
        batch["teacher_topk_ids"],
        batch["teacher_topk_logprobs"],
        batch["teacher_tail_logprobs"],
        selective_mask,
        temperature=distillation_temperature,
        token_weights=token_weights,
    )
    margin_loss = top1_margin_loss(
        logits[:, :-1, :],
        teacher_top1,
        selective_mask,
        margin=top1_margin,
        token_weights=token_weights,
    )
    loss = (
        hard_loss_weight * hard_loss
        + kd_loss_weight * kd_loss
        + tvd_loss_weight * tvd_loss
        + margin_loss_weight * margin_loss
    )
    return loss, {
        "hard_loss": float(hard_loss.detach()),
        "kd_loss": float(kd_loss.detach()),
        "tvd_loss": float(tvd_loss.detach()),
        "margin_loss": float(margin_loss.detach()),
        "rejection_rate": float(
            rejection_mask.sum()
            / selective_mask.sum().clamp_min(1)
        ),
        "selected_token_rate": float(
            selective_mask.sum()
            / batch["loss_mask"].sum().clamp_min(1)
        ),
    }


def _evaluate(
    model,
    loader: Optional[DataLoader],
    *,
    device: torch.device,
    mode: str,
    hard_loss_weight: float,
    kd_loss_weight: float,
    tvd_loss_weight: float,
    margin_loss_weight: float,
    top1_margin: float,
    distillation_temperature: float,
    rejection_weight: float,
    rejection_window_size: int,
    min_teacher_confidence: float,
) -> Optional[float]:
    if loader is None:
        return None
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, device)
            loss, _ = _batch_loss(
                model,
                batch,
                mode=mode,
                hard_loss_weight=hard_loss_weight,
                kd_loss_weight=kd_loss_weight,
                tvd_loss_weight=tvd_loss_weight,
                margin_loss_weight=margin_loss_weight,
                top1_margin=top1_margin,
                distillation_temperature=distillation_temperature,
                rejection_weight=rejection_weight,
                rejection_window_size=rejection_window_size,
                min_teacher_confidence=min_teacher_confidence,
            )
            total_loss += float(loss)
            count += 1
    model.train()
    return total_loss / max(1, count)


def train(config: dict) -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    mode = str(config["mode"])
    if mode not in {"sft", "kd"}:
        raise ValueError("mode must be sft or kd")

    device = torch.device(config.get("device", "cuda:0"))
    dtype = torch_dtype(config.get("dtype", "fp16"))
    parameter_dtype_name = config.get("parameter_dtype")
    parameter_dtype = (
        torch_dtype(str(parameter_dtype_name))
        if parameter_dtype_name is not None
        else torch.float32 if dtype == torch.float16 else dtype
    )
    model_name = str(config["model"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_load_kwargs(parameter_dtype),
    ).to(device)
    model.config.use_cache = False
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    model.train()

    train_dataset = JsonlOffsetDataset(Path(config["train_path"]))
    validation_path = config.get("validation_path")
    validation_dataset = (
        JsonlOffsetDataset(Path(validation_path))
        if validation_path
        else None
    )
    if mode == "sft":
        collator = SFTCollator(
            tokenizer=tokenizer,
            max_seq_len=int(config["max_seq_len"]),
        )
    else:
        collator = KDCollator(
            pad_token_id=tokenizer.pad_token_id,
        )

    batch_size = int(config.get("batch_size", 1))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=int(config.get("num_workers", 0)),
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=int(config.get("num_workers", 0)),
        )
        if validation_dataset is not None
        else None
    )

    epochs = int(config.get("epochs", 1))
    accumulation_steps = int(
        config.get("gradient_accumulation_steps", 1)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / accumulation_steps
    )
    total_updates = max(1, epochs * updates_per_epoch)
    warmup_steps = int(
        total_updates * float(config.get("warmup_ratio", 0.03))
    )
    scheduler = get_scheduler(
        str(config.get("scheduler", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and dtype == torch.float16
    )
    hard_loss_weight = float(config.get("hard_loss_weight", 0.5))
    kd_loss_weight = float(config.get("kd_loss_weight", 0.5))
    tvd_loss_weight = float(config.get("tvd_loss_weight", 0.0))
    margin_loss_weight = float(
        config.get("margin_loss_weight", 0.0)
    )
    top1_margin = float(config.get("top1_margin", 0.0))
    distillation_temperature = float(
        config.get("distillation_temperature", 1.0)
    )
    rejection_weight = float(config.get("rejection_weight", 1.0))
    rejection_window_size = int(
        config.get("rejection_window_size", 1)
    )
    min_teacher_confidence = float(
        config.get("min_teacher_confidence", 0.0)
    )
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    log_every = int(config.get("log_every", 10))
    best_validation_loss = float("inf")
    global_update = 0

    metrics_path = output_dir / "training_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(epochs):
            for batch_index, batch in enumerate(train_loader):
                batch = _move_batch(batch, device)
                autocast_enabled = (
                    (
                        device.type == "cuda"
                        and dtype in {torch.float16, torch.bfloat16}
                    )
                    or (
                        device.type == "cpu"
                        and dtype == torch.bfloat16
                    )
                )
                autocast_context = (
                    torch.autocast(
                        device_type=device.type,
                        dtype=dtype,
                    )
                    if autocast_enabled
                    else nullcontext()
                )
                with autocast_context:
                    loss, details = _batch_loss(
                        model,
                        batch,
                        mode=mode,
                        hard_loss_weight=hard_loss_weight,
                        kd_loss_weight=kd_loss_weight,
                        tvd_loss_weight=tvd_loss_weight,
                        margin_loss_weight=margin_loss_weight,
                        top1_margin=top1_margin,
                        distillation_temperature=(
                            distillation_temperature
                        ),
                        rejection_weight=rejection_weight,
                        rejection_window_size=(
                            rejection_window_size
                        ),
                        min_teacher_confidence=(
                            min_teacher_confidence
                        ),
                    )
                    scaled_loss = loss / accumulation_steps
                scaler.scale(scaled_loss).backward()

                should_update = (
                    (batch_index + 1) % accumulation_steps == 0
                    or batch_index + 1 == len(train_loader)
                )
                if not should_update:
                    continue
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_update += 1

                record = {
                    "epoch": epoch + 1,
                    "update": global_update,
                    "loss": float(loss.detach()),
                    "learning_rate": scheduler.get_last_lr()[0],
                    **details,
                }
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                if global_update % log_every == 0:
                    print(record, flush=True)

            validation_loss = _evaluate(
                model,
                validation_loader,
                device=device,
                mode=mode,
                hard_loss_weight=hard_loss_weight,
                kd_loss_weight=kd_loss_weight,
                tvd_loss_weight=tvd_loss_weight,
                margin_loss_weight=margin_loss_weight,
                top1_margin=top1_margin,
                distillation_temperature=distillation_temperature,
                rejection_weight=rejection_weight,
                rejection_window_size=rejection_window_size,
                min_teacher_confidence=min_teacher_confidence,
            )
            epoch_dir = output_dir / f"epoch-{epoch + 1}"
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            if (
                validation_loss is not None
                and validation_loss < best_validation_loss
            ):
                best_validation_loss = validation_loss
                best_dir = output_dir / "best"
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
            print(
                {
                    "epoch": epoch + 1,
                    "validation_loss": validation_loss,
                },
                flush=True,
            )

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    train(load_yaml(args.config))


if __name__ == "__main__":
    main()
