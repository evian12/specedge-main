import json
import tempfile
import unittest
from pathlib import Path

from distillation.prepare_builtin_holdout import prepare
from distillation.prepare_builtin_mixture import (
    prepare as prepare_mixture,
)


class BuiltinHoldoutTest(unittest.TestCase):
    def test_samples_each_source_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            for source in ("oasst", "c4", "wikitext"):
                with (data_dir / f"{source}_prompts.json").open("w") as file:
                    json.dump(
                        [
                            [index, f"{source}-{index}"]
                            for index in range(5)
                        ],
                        file,
                    )
            output = root / "holdout.jsonl"
            previous = Path.cwd()
            try:
                import os

                os.chdir(root)
                count = prepare(output, samples_per_source=2, seed=7)
            finally:
                os.chdir(previous)

            records = [
                json.loads(line) for line in output.read_text().splitlines()
            ]
            self.assertEqual(count, 6)
            self.assertEqual(
                {record["source"] for record in records},
                {"oasst", "c4", "wikitext"},
            )

    def test_mixture_excludes_deterministic_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            for source in ("oasst", "c4", "wikitext"):
                with (data_dir / f"{source}_prompts.json").open("w") as file:
                    json.dump(
                        [
                            [index, f"{source}-{index}"]
                            for index in range(20)
                        ],
                        file,
                    )
            previous = Path.cwd()
            try:
                import os

                os.chdir(root)
                holdout_path = root / "holdout.jsonl"
                prepare(holdout_path, samples_per_source=3, seed=7)
                prepare_mixture(
                    root / "mixture",
                    train_per_source=5,
                    validation_per_source=2,
                    holdout_per_source=3,
                    seed=7,
                )
            finally:
                os.chdir(previous)

            holdout_prompts = {
                json.loads(line)["prompt"]
                for line in holdout_path.read_text().splitlines()
            }
            mixture_prompts = {
                json.loads(line)["prompt"]
                for name in ("train", "validation")
                for line in (
                    root / "mixture" / f"{name}.jsonl"
                ).read_text().splitlines()
            }
            self.assertTrue(holdout_prompts.isdisjoint(mixture_prompts))
