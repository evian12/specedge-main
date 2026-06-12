import argparse
import json
import random
from pathlib import Path

from distillation.io import write_jsonl


def _sample_records(
    path: Path,
    *,
    source: str,
    count: int,
    seed: int,
) -> list[dict]:
    with path.open() as file:
        values = json.load(file)
    rng = random.Random(seed)
    selected = rng.sample(values, min(count, len(values)))
    records = []
    for index, value in enumerate(selected):
        prompt = value[1] if isinstance(value, list) else value
        records.append(
            {
                "prompt_id": f"{source}-{index}",
                "source": source,
                "prompt": str(prompt),
            }
        )
    return records


def prepare(output_path: Path, samples_per_source: int, seed: int) -> int:
    records = []
    for offset, source in enumerate(("oasst", "c4", "wikitext")):
        records.extend(
            _sample_records(
                Path(f"data/{source}_prompts.json"),
                source=source,
                count=samples_per_source,
                seed=seed + offset,
            )
        )
    random.Random(seed).shuffle(records)
    return write_jsonl(output_path, records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/distillation/generalization_holdout/prompts.jsonl"
        ),
    )
    parser.add_argument("--samples-per-source", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    count = prepare(
        args.output,
        args.samples_per_source,
        args.seed,
    )
    print(f"Prepared {count} held-out prompts")


if __name__ == "__main__":
    main()
