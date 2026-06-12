import unittest

import torch

from specedge.client.proactive_selection import (
    acceptance_stop_probabilities,
    allocate_sequence_bonus_counts,
    select_bonus_candidates,
    select_ids_by_probability,
    select_sequence_bonus_candidates,
    trace_main_sequence_nodes,
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

    def test_acceptance_survival_becomes_stop_probability(self):
        stop = acceptance_stop_probabilities(
            [1.0, 0.51, 0.301, 0.193, 0.133],
            max_depth=4,
        )

        expected = [0.49, 0.209, 0.108, 0.06, 0.133]
        for actual, target in zip(stop, expected):
            self.assertAlmostEqual(actual, target)
        self.assertAlmostEqual(sum(stop), 1.0)

    def test_sequence_bonus_budget_follows_stop_probability(self):
        counts = allocate_sequence_bonus_counts(
            [0.49, 0.209, 0.108, 0.06, 0.133],
            max_roots=8,
            max_bonus_per_depth=4,
        )

        self.assertEqual(counts, [4, 2, 1, 0, 1])

    def test_sequence_candidates_are_attached_to_each_stop_depth(self):
        candidates = select_sequence_bonus_candidates(
            torch.tensor([10, 11, 12]),
            [0.6, 0.3, 0.1],
            torch.tensor(
                [
                    [100, 101],
                    [200, 201],
                    [300, 301],
                ]
            ),
            torch.tensor(
                [
                    [0.7, 0.3],
                    [0.8, 0.2],
                    [0.9, 0.1],
                ]
            ).log(),
            max_bonus_per_depth=2,
            max_roots=4,
            min_root_probability=0.0,
        )

        by_depth = {}
        for candidate in candidates:
            by_depth.setdefault(candidate.stop_depth, []).append(
                candidate.token_id
            )
        self.assertEqual(by_depth[0], [100, 101])
        self.assertEqual(by_depth[1], [200])
        self.assertEqual(by_depth[2], [300])

    def test_main_sequence_uses_best_deepest_leaf(self):
        path = trace_main_sequence_nodes(
            torch.tensor([4, 5, 6]),
            torch.tensor([0, 1, 2, 2, 3, 3, 2]),
            torch.tensor([0.0, 0.0, -0.2, -0.3, -1.0, -0.4, -0.1]),
            torch.tensor([0, 0, 1, 1, 2, 3, 1]),
            prefix_tail=1,
        )

        self.assertEqual(path, [1, 3, 5])


if __name__ == "__main__":
    unittest.main()
