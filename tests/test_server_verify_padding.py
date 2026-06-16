import unittest

import torch

from strategy.server_verify.specexec.padding import (
    copy_padded_1d,
    copy_padded_attention_mask,
    validate_draft_request_shapes,
)


class ServerVerifyPaddingTest(unittest.TestCase):
    def test_pads_short_1d_tensor(self):
        target = torch.full((5,), 9, dtype=torch.long)
        source = torch.tensor([1, 2, 3], dtype=torch.long)

        copied = copy_padded_1d(target, source, name="input_ids")

        self.assertEqual(copied, 3)
        self.assertEqual(target.tolist(), [1, 2, 3, 0, 0])

    def test_rejects_oversized_1d_tensor(self):
        target = torch.zeros((2,), dtype=torch.long)
        source = torch.tensor([1, 2, 3], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "server capacity"):
            copy_padded_1d(target, source, name="input_ids")

    def test_pads_short_attention_mask(self):
        target = torch.full((1, 4, 6), 7.0)
        source = torch.ones((1, 2, 6))

        copied = copy_padded_attention_mask(target, source, name="attention_mask")

        self.assertEqual(copied, 2)
        self.assertTrue(torch.equal(target[:, :2, :], source))
        self.assertTrue(torch.equal(target[:, 2:, :], torch.zeros((1, 2, 6))))

    def test_validates_request_node_counts(self):
        validate_draft_request_shapes(
            input_len=5,
            position_len=5,
            cache_len=5,
            attention_len=5,
            parent_len=4,
        )

        with self.assertRaisesRegex(ValueError, "disagree on node count"):
            validate_draft_request_shapes(
                input_len=5,
                position_len=4,
                cache_len=5,
                attention_len=5,
                parent_len=4,
            )

        with self.assertRaisesRegex(ValueError, "one parent per draft node"):
            validate_draft_request_shapes(
                input_len=5,
                position_len=5,
                cache_len=5,
                attention_len=5,
                parent_len=5,
            )


if __name__ == "__main__":
    unittest.main()
