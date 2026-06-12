import math
import unittest

import torch

from distillation.losses import (
    hard_token_loss,
    masked_token_loss,
    rejection_window_weights,
    sparse_topk_distillation_loss,
    sparse_topk_total_variation_loss,
    top1_margin_loss,
)


class DistillationLossTest(unittest.TestCase):
    def test_sparse_loss_prefers_teacher_matching_distribution(self):
        teacher_topk_ids = torch.tensor([[[0, 1]]])
        teacher_topk_logprobs = torch.tensor(
            [[[math.log(0.6), math.log(0.3)]]]
        )
        teacher_tail_logprobs = torch.tensor([[math.log(0.1)]])
        loss_mask = torch.tensor([[True]])
        matching_logits = torch.log(
            torch.tensor(
                [[[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]]]
            )
        )
        mismatching_logits = torch.log(
            torch.tensor(
                [[[0.1, 0.3, 0.6], [0.1, 0.3, 0.6]]]
            )
        )

        matching_loss = sparse_topk_distillation_loss(
            matching_logits,
            teacher_topk_ids,
            teacher_topk_logprobs,
            teacher_tail_logprobs,
            loss_mask,
            temperature=1.0,
        )
        mismatching_loss = sparse_topk_distillation_loss(
            mismatching_logits,
            teacher_topk_ids,
            teacher_topk_logprobs,
            teacher_tail_logprobs,
            loss_mask,
            temperature=1.0,
        )

        self.assertLess(matching_loss, mismatching_loss)

    def test_hard_loss_ignores_unselected_positions(self):
        logits = torch.tensor(
            [
                [
                    [0.0, 4.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0],
                ]
            ]
        )
        input_ids = torch.tensor([[0, 1, 2]])
        only_first_prediction = torch.tensor([[True, False]])
        both_predictions = torch.tensor([[True, True]])

        first_loss = hard_token_loss(
            logits,
            input_ids,
            only_first_prediction,
        )
        both_loss = hard_token_loss(
            logits,
            input_ids,
            both_predictions,
        )

        self.assertLess(first_loss, both_loss)

    def test_masked_loss_can_use_teacher_top1_targets(self):
        logits = torch.tensor(
            [[[0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]]
        )
        teacher_targets = torch.tensor([[1, 2]])
        loss = masked_token_loss(
            logits,
            teacher_targets,
            torch.tensor([[True, True]]),
        )

        self.assertLess(loss, 0.1)

    def test_total_variation_prefers_matching_distribution(self):
        teacher_ids = torch.tensor([[[0, 1]]])
        teacher_logprobs = torch.tensor(
            [[[math.log(0.7), math.log(0.2)]]]
        )
        teacher_tail = torch.tensor([[math.log(0.1)]])
        mask = torch.tensor([[True]])
        matching = torch.log(
            torch.tensor(
                [[[0.7, 0.2, 0.1], [0.7, 0.2, 0.1]]]
            )
        )
        mismatching = torch.log(
            torch.tensor(
                [[[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]]]
            )
        )

        matching_loss = sparse_topk_total_variation_loss(
            matching,
            teacher_ids,
            teacher_logprobs,
            teacher_tail,
            mask,
            temperature=1.0,
        )
        mismatching_loss = sparse_topk_total_variation_loss(
            mismatching,
            teacher_ids,
            teacher_logprobs,
            teacher_tail,
            mask,
            temperature=1.0,
        )

        self.assertLess(matching_loss, mismatching_loss)

    def test_rejection_window_emphasizes_following_states(self):
        weights = rejection_window_weights(
            torch.tensor([[False, True, False, False, False]]),
            torch.tensor([[True, True, True, True, True]]),
            window_size=3,
            rejection_weight=4.0,
        )

        self.assertEqual(
            weights.tolist(),
            [[1.0, 4.0, 3.0, 2.0, 1.0]],
        )

    def test_top1_margin_penalizes_wrong_ranking(self):
        teacher_top1 = torch.tensor([[1]])
        mask = torch.tensor([[True]])
        correct = torch.tensor([[[0.0, 2.0, 1.0]]])
        incorrect = torch.tensor([[[2.0, 1.0, 0.0]]])

        correct_loss = top1_margin_loss(
            correct,
            teacher_top1,
            mask,
            margin=0.1,
        )
        incorrect_loss = top1_margin_loss(
            incorrect,
            teacher_top1,
            mask,
            margin=0.1,
        )

        self.assertEqual(float(correct_loss), 0.0)
        self.assertGreater(float(incorrect_loss), 1.0)


if __name__ == "__main__":
    unittest.main()
