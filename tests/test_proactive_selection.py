import unittest

import torch

from specedge.client.proactive_selection import (
    select_bonus_candidates,
    select_ids_by_probability,
)


class ProactiveSelectionTest(unittest.TestCase):
    def setUp(self):
        self.leaf_indices = torch.tensor([10, 11])
        self.leaf_scores = torch.tensor([0.0, 0.0])
        self.bonus_token_ids = torch.tensor(
            [
                [100, 101, 102],
                [200, 201, 202],
            ]
        )
        self.bonus_logprobs = torch.tensor(
            [
                [0.6, 0.25, 0.1],
                [0.6, 0.25, 0.1],
            ]
        ).log()

    def _select(self, acceptance_rate: float):
        return select_bonus_candidates(
            self.leaf_indices,
            self.leaf_scores,
            self.bonus_token_ids,
            self.bonus_logprobs,
            full_depth_acceptance=acceptance_rate,
            max_deepest_leaves=8,
            min_bonus_per_leaf=1,
            max_bonus_per_leaf=3,
            max_roots=6,
            min_root_probability=0.02,
            leaf_temperature=1.0,
        )

    def test_each_selected_leaf_receives_top_one_bonus(self):
        candidates, selected_leaf_count = self._select(0.1)

        self.assertEqual(selected_leaf_count, 2)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            {(candidate.leaf_idx, candidate.token_id) for candidate in candidates},
            {(10, 100), (11, 200)},
        )

    def test_higher_acceptance_rate_allocates_more_bonus_tokens(self):
        low_acceptance, _ = self._select(0.1)
        high_acceptance, _ = self._select(0.8)

        self.assertEqual(len(low_acceptance), 2)
        self.assertEqual(len(high_acceptance), 6)

    def test_leaf_and_root_limits_are_respected(self):
        candidates, selected_leaf_count = select_bonus_candidates(
            torch.tensor([10, 11, 12, 13]),
            torch.tensor([0.0, 3.0, 2.0, 1.0]),
            torch.tensor(
                [
                    [100, 101],
                    [200, 201],
                    [300, 301],
                    [400, 401],
                ]
            ),
            torch.tensor(
                [
                    [0.8, 0.2],
                    [0.8, 0.2],
                    [0.8, 0.2],
                    [0.8, 0.2],
                ]
            ).log(),
            full_depth_acceptance=1.0,
            max_deepest_leaves=2,
            min_bonus_per_leaf=1,
            max_bonus_per_leaf=2,
            max_roots=3,
            min_root_probability=0.0,
            leaf_temperature=1.0,
        )

        self.assertEqual(selected_leaf_count, 2)
        self.assertEqual(len(candidates), 3)
        self.assertEqual({candidate.leaf_idx for candidate in candidates}, {11, 12})

    def test_probability_coverage_shrinks_active_roots(self):
        probabilities = [(0, 0.5), (1, 0.3), (2, 0.15), (3, 0.05)]

        self.assertEqual(
            select_ids_by_probability(probabilities, 1.0),
            {0, 1, 2, 3},
        )
        self.assertEqual(
            select_ids_by_probability(probabilities, 0.8),
            {0, 1},
        )
        self.assertEqual(
            select_ids_by_probability(probabilities, 0.5),
            {0},
        )


if __name__ == "__main__":
    unittest.main()
