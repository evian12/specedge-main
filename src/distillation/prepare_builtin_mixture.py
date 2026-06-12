import argparse
import json
import random
from pathlib import Path

from distillation.prepare_builtin_holdout import _sample_records
from distillation.io import write_jsonl


def _load_source(path: Path, source: str) -> list[dict]:
    with path.open() as file:
        values = json.load(file)
    records = []
    for index, value in enumerate(values):
        prompt = value[1] if isinstance(value, list) else value
        records.append(
            {
                "prompt_id": f"{source}-mixture-{index}",
                "source": source,
                "prompt": str(prompt),
            }
        )
    return records


def prepare(
    output_dir: Path,
    *,
    train_per_source: int,
    validation_per_source: int,
    holdout_per_source: int,
    seed: int,
) -> dict[str, int]:
    train_records = []
    validation_records = []
    for offset, source in enumerate(("oasst", "c4", "wikitext")):
        path = Path(f"data/{source}_prompts.json")
        holdout = _sample_records(
            path,
            source=source,
            count=holdout_per_source,
            seed=seed + offset,
        )
        holdout_prompts = {record["prompt"] for record in holdout}
        candidates = [
            record
            for record in _load_source(path, source)
            if record["prompt"] not in holdout_prompts
        ]
        rng = random.Random(seed + 100 + offset)
        rng.shuffle(candidates)
        validation_records.extend(
            candidates[:validation_per_source]
        )
        train_records.extend(
            candidates[
                validation_per_source:
                validation_per_source + train_per_source
            ]
        )

    random.Random(seed).shuffle(train_records)
    random.Random(seed + 1).shuffle(validation_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "train": write_jsonl(
            output_dir / "train.jsonl",
            train_records,
        ),
        "validation": write_jsonl(
            output_dir / "validation.jsonl",
            validation_records,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/distillation/general_mixture"),
    )
    parser.add_argument("--train-per-source", type=int, default=120)
    parser.add_argument(
        "--validation-per-source",
        type=int,
        default=20,
    )
    parser.add_argument("--holdout-per-source", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    counts = prepare(
        args.output_dir,
        train_per_source=args.train_per_source,
        validation_per_source=args.validation_per_source,
        holdout_per_source=args.holdout_per_source,
        seed=args.seed,
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
