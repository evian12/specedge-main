import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from torch.utils.data import Dataset

from distillation.io import iter_jsonl


def nested_value(record: dict[str, Any], field: str) -> Any:
    value: Any = record
    for key in field.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing field {field!r}")
        value = value[key]
    return value


def canonical_prompt(
    record: dict[str, Any],
    *,
    source_name: str,
    prompt_field: Optional[str] = None,
    messages_field: Optional[str] = None,
    record_id_field: Optional[str] = None,
) -> dict[str, Any]:
    if bool(prompt_field) == bool(messages_field):
        raise ValueError(
            "Each source must configure exactly one of prompt_field "
            "or messages_field"
        )

    result: dict[str, Any] = {
        "source": source_name,
        "metadata": {},
    }
    if prompt_field:
        prompt = nested_value(record, prompt_field)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
        result["prompt"] = prompt
    else:
        messages = nested_value(record, str(messages_field))
        if not isinstance(messages, list) or not messages:
            raise ValueError("Messages must be a non-empty list")
        for message in messages:
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError(
                    "Each message must contain string role and content"
                )
        result["messages"] = messages

    if record_id_field:
        result["source_id"] = str(
            nested_value(record, record_id_field)
        )
    return result


def prompt_key(record: dict[str, Any]) -> str:
    identity = record.get("messages", record.get("prompt"))
    serialized = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def assign_split(
    key: str,
    *,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def iter_source_records(
    source: dict[str, Any],
    *,
    seed: int,
) -> Iterator[dict[str, Any]]:
    path = Path(source["path"])
    records = iter_jsonl(path)
    max_samples = int(source.get("max_samples", -1))
    reservoir: list[dict[str, Any]] = []
    rng = random.Random(seed)

    for index, raw_record in enumerate(records):
        record = canonical_prompt(
            raw_record,
            source_name=str(source["name"]),
            prompt_field=source.get("prompt_field"),
            messages_field=source.get("messages_field"),
            record_id_field=source.get("id_field"),
        )
        if max_samples < 0:
            yield record
            continue
        if len(reservoir) < max_samples:
            reservoir.append(record)
            continue
        replacement = rng.randint(0, index)
        if replacement < max_samples:
            reservoir[replacement] = record

    if max_samples >= 0:
        rng.shuffle(reservoir)
        yield from reservoir


def format_prompt(
    record: dict[str, Any],
    tokenizer,
    *,
    apply_chat_template: bool,
    add_generation_prompt: bool = True,
) -> str:
    if "prompt_text" in record:
        return str(record["prompt_text"])
    if "messages" in record:
        return tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    prompt = str(record["prompt"])
    if not apply_chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


class JsonlOffsetDataset(Dataset):
    """Random-access JSONL dataset without loading all records into RAM."""

    def __init__(self, path: Path) -> None:
        if path.suffix == ".gz":
            raise ValueError(
                "Training datasets must be uncompressed JSONL files"
            )
        self.path = path
        self._offsets: list[int] = []
        with path.open("rb") as file:
            while True:
                offset = file.tell()
                line = file.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(offset)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as file:
            file.seek(self._offsets[index])
            return json.loads(file.readline())


@dataclass
class SFTCollator:
    tokenizer: Any
    max_seq_len: int

    def __call__(self, records: list[dict[str, Any]]) -> dict:
        encoded = []
        for record in records:
            prompt_ids = self.tokenizer.encode(
                record["prompt_text"],
                add_special_tokens=False,
            )
            response_ids = self.tokenizer.encode(
                record["response"],
                add_special_tokens=False,
            )
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None and (
                not response_ids or response_ids[-1] != eos_token_id
            ):
                response_ids.append(eos_token_id)

            max_prompt = max(1, self.max_seq_len - len(response_ids))
            prompt_ids = prompt_ids[-max_prompt:]
            response_ids = response_ids[
                : self.max_seq_len - len(prompt_ids)
            ]
            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids
            encoded.append((input_ids, labels))

        return _pad_sft_batch(
            encoded,
            pad_token_id=self.tokenizer.pad_token_id,
        )


def _pad_sft_batch(
    encoded: list[tuple[list[int], list[int]]],
    *,
    pad_token_id: int,
) -> dict:
    import torch

    max_len = max(len(input_ids) for input_ids, _ in encoded)
    input_ids = torch.full(
        (len(encoded), max_len),
        pad_token_id,
        dtype=torch.long,
    )
    labels = torch.full(
        (len(encoded), max_len),
        -100,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (len(encoded), max_len),
        dtype=torch.long,
    )
    for index, (tokens, token_labels) in enumerate(encoded):
        length = len(tokens)
        input_ids[index, :length] = torch.tensor(tokens)
        labels[index, :length] = torch.tensor(token_labels)
        attention_mask[index, :length] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


@dataclass
class KDCollator:
    pad_token_id: int

    def __call__(self, records: list[dict[str, Any]]) -> dict:
        import torch

        max_len = max(len(record["input_ids"]) for record in records)
        top_k = len(records[0]["teacher_topk_ids"][0])
        batch_size = len(records)
        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
        )
        loss_mask = torch.zeros(
            (batch_size, max_len - 1),
            dtype=torch.bool,
        )
        topk_ids = torch.zeros(
            (batch_size, max_len - 1, top_k),
            dtype=torch.long,
        )
        topk_logprobs = torch.full(
            (batch_size, max_len - 1, top_k),
            float("-inf"),
            dtype=torch.float32,
        )
        tail_logprobs = torch.full(
            (batch_size, max_len - 1),
            float("-inf"),
            dtype=torch.float32,
        )
        loss_weights = torch.ones(
            (batch_size, max_len - 1),
            dtype=torch.float32,
        )

        for index, record in enumerate(records):
            tokens = torch.tensor(record["input_ids"], dtype=torch.long)
            prediction_len = len(tokens) - 1
            input_ids[index, : len(tokens)] = tokens
            attention_mask[index, : len(tokens)] = 1
            loss_mask[index, :prediction_len] = torch.tensor(
                record["loss_mask"],
                dtype=torch.bool,
            )
            topk_ids[index, :prediction_len] = torch.tensor(
                record["teacher_topk_ids"],
                dtype=torch.long,
            )
            topk_logprobs[index, :prediction_len] = torch.tensor(
                record["teacher_topk_logprobs"],
                dtype=torch.float32,
            )
            tail_logprobs[index, :prediction_len] = torch.tensor(
                record["teacher_tail_logprobs"],
                dtype=torch.float32,
            )
            if "loss_weights" in record:
                loss_weights[index, :prediction_len] = torch.tensor(
                    record["loss_weights"],
                    dtype=torch.float32,
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "teacher_topk_ids": topk_ids,
            "teacher_topk_logprobs": topk_logprobs,
            "teacher_tail_logprobs": tail_logprobs,
            "loss_weights": loss_weights,
        }
