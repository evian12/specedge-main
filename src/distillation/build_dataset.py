import argparse
import json
from pathlib import Path

from distillation.data import (
    assign_split,
    iter_source_records,
    prompt_key,
)
from distillation.io import load_yaml


def build_dataset(config: dict) -> dict[str, int]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    split_config = config.get("splits", {})
    train_ratio = float(split_config.get("train", 0.8))
    validation_ratio = float(
        split_config.get("validation", 0.1)
    )
    test_ratio = float(split_config.get("test", 0.1))
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Dataset split ratios must sum to 1")

    output_files = {
        split: (output_dir / f"{split}.jsonl").open(
            "w",
            encoding="utf-8",
        )
        for split in ("train", "validation", "test")
    }
    counts = {split: 0 for split in output_files}
    counts["duplicates"] = 0
    seen: set[str] = set()

    try:
        for source_index, source in enumerate(config["sources"]):
            for record in iter_source_records(
                source,
                seed=seed + source_index,
            ):
                key = prompt_key(record)
                if key in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(key)
                record["prompt_id"] = key
                split = assign_split(
                    key,
                    seed=seed,
                    train_ratio=train_ratio,
                    validation_ratio=validation_ratio,
                )
                output_files[split].write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                counts[split] += 1
    finally:
        for file in output_files.values():
            file.close()

    manifest = {
        "seed": seed,
        "splits": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "counts": counts,
        "sources": config["sources"],
    }
    with (output_dir / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    counts = build_dataset(load_yaml(args.config))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
