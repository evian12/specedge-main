import argparse
from pathlib import Path

from distillation.io import iter_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    count = write_jsonl(
        args.output,
        (
            record
            for path in args.input
            for record in iter_jsonl(path)
        ),
    )
    print(f"Combined {count} records")


if __name__ == "__main__":
    main()
