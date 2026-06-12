import unittest

import torch

from distillation.collect_teacher import (
    acceptance_gain_weights,
    rejection_replay_mask,
)


class DistillationReplayTest(unittest.TestCase):
    def test_replay_selects_confident_rejections_and_window(self):
        mask = rejection_replay_mask(
            torch.tensor([1, 4, 3, 2, 0]),
            torch.tensor([1, 2, 3, 1, 0]),
            torch.tensor([0.9, 0.8, 0.9, 0.1, 0.9]),
            torch.tensor([True, True, True, True, False]),
            min_teacher_confidence=0.2,
            window_size=2,
        )

        self.assertEqual(
            mask.tolist(),
            [False, True, True, False, False],
        )

    def test_replay_rejects_invalid_window(self):
        with self.assertRaises(ValueError):
            rejection_replay_mask(
                torch.tensor([1]),
                torch.tensor([0]),
                torch.tensor([1.0]),
                torch.tensor([True]),
                min_teacher_confidence=0.0,
                window_size=0,
            )

    def test_acceptance_gain_prioritizes_bridge_errors(self):
        weights = acceptance_gain_weights(
            torch.tensor([True, True, False, True, True, False]),
            torch.tensor([True, True, True, True, True, True]),
            torch.tensor([False, False, True, False, False, True]),
            max_accept_depth=7,
            max_weight=6.0,
        )

        self.assertGreater(float(weights[2]), float(weights[5]))
        self.assertEqual(float(weights[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
