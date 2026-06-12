import json
import tempfile
import unittest
from pathlib import Path

from distillation.build_dataset import build_dataset
from distillation.data import (
    JsonlOffsetDataset,
    canonical_prompt,
    prompt_key,
)


class DistillationDataTest(unittest.TestCase):
    def test_canonical_messages_are_preserved(self):
        record = canonical_prompt(
            {
                "request_id": "abc",
                "payload": {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                    ]
                },
            },
            source_name="business",
            messages_field="payload.messages",
            record_id_field="request_id",
        )

        self.assertEqual(record["source"], "business")
        self.assertEqual(record["source_id"], "abc")
        self.assertEqual(
            record["messages"][0]["content"],
            "Hello",
        )

    def test_builder_deduplicates_and_keeps_splits_disjoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "source.jsonl"
            with source_path.open("w") as file:
                for index in range(20):
                    file.write(
                        json.dumps(
                            {
                                "id": index,
                                "prompt": f"prompt-{index}",
                            }
                        )
                        + "\n"
                    )
                file.write(
                    json.dumps({"id": 21, "prompt": "prompt-0"})
                    + "\n"
                )

            output_dir = root / "output"
            counts = build_dataset(
                {
                    "seed": 42,
                    "output_dir": str(output_dir),
                    "splits": {
                        "train": 0.7,
                        "validation": 0.2,
                        "test": 0.1,
                    },
                    "sources": [
                        {
                            "name": "general",
                            "path": str(source_path),
                            "prompt_field": "prompt",
                            "id_field": "id",
                        }
                    ],
                }
            )

            self.assertEqual(
                counts["train"]
                + counts["validation"]
                + counts["test"],
                20,
            )
            self.assertEqual(counts["duplicates"], 1)
            split_keys = []
            for split in ("train", "validation", "test"):
                dataset = JsonlOffsetDataset(
                    output_dir / f"{split}.jsonl"
                )
                split_keys.append(
                    {
                        prompt_key(dataset[index])
                        for index in range(len(dataset))
                    }
                )
            self.assertTrue(split_keys[0].isdisjoint(split_keys[1]))
            self.assertTrue(split_keys[0].isdisjoint(split_keys[2]))
            self.assertTrue(split_keys[1].isdisjoint(split_keys[2]))


if __name__ == "__main__":
    unittest.main()
