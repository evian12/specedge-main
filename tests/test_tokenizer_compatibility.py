import unittest

from specedge.tokenizer import validate_tokenizer_compatibility


class FakeTokenizer:
    def __init__(
        self,
        *,
        vocab_size=32_000,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
        unk_token_id=0,
        encoded=None,
    ):
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self._encoded = encoded or [1, 10, 2]

    def __len__(self):
        return self.vocab_size

    def encode(self, prompt):
        return self._encoded


class TokenizerCompatibilityTest(unittest.TestCase):
    def test_accepts_identical_tokenizers(self):
        validate_tokenizer_compatibility(
            FakeTokenizer(),
            FakeTokenizer(),
            ["prompt"],
        )

    def test_rejects_different_vocabulary_sizes(self):
        with self.assertRaisesRegex(
            ValueError,
            "different vocabulary sizes",
        ):
            validate_tokenizer_compatibility(
                FakeTokenizer(vocab_size=32_001),
                FakeTokenizer(),
                ["prompt"],
            )

    def test_rejects_different_special_token_ids(self):
        with self.assertRaisesRegex(
            ValueError,
            "different special token IDs",
        ):
            validate_tokenizer_compatibility(
                FakeTokenizer(bos_token_id=0),
                FakeTokenizer(),
                ["prompt"],
            )


if __name__ == "__main__":
    unittest.main()
