import gzip
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

import numpy as np
import torch
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML object in {path}")
    return config


def open_text(path: Path, mode: str = "rt") -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            yield value


def write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open_text(path, "wt") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )
            file.write("\n")
            count += 1
    return count


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str) -> torch.dtype:
    match name:
        case "fp16":
            return torch.float16
        case "bf16":
            return torch.bfloat16
        case "fp32":
            return torch.float32
        case _:
            raise ValueError(f"Unsupported dtype: {name}")


def model_load_kwargs(dtype: torch.dtype) -> dict[str, Any]:
    return {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": (
            importlib.util.find_spec("accelerate") is not None
        ),
    }
