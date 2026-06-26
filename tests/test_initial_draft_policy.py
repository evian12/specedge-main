import unittest

from specedge.client.initial_draft_policy import (
    InitialDraftDecision,
    LocalStreakInitialDraftPolicy,
    initial_depth_after_proactive_reuse,
)


class ProactiveReuseDepthTest(unittest.TestCase):
    def test_sequence_depth_subtracts_actual_reused_layers(self):
        remaining, reused = initial_depth_after_proactive_reuse(
            6,
            proactive_hit=True,
            reused_depth=2,
            proactive_type="excluded",
            path_policy="sequence_depth",
        )

        self.assertEqual(remaining, 4)
        self.assertEqual(reused, 2)

    def test_deepest_multi_subtracts_actual_reused_layers(self):
        remaining, reused = initial_depth_after_proactive_reuse(
            4,
            proactive_hit=True,
            reused_depth=2,
            proactive_type="excluded",
            path_policy="deepest_multi",
        )

        self.assertEqual(remaining, 2)
        self.assertEqual(reused, 2)

    def test_hybrid_sequence_subtracts_actual_reused_layers(self):
        remaining, reused = initial_depth_after_proactive_reuse(
            4,
            proactive_hit=True,
            reused_depth=3,
            proactive_type="excluded",
            path_policy="hybrid_sequence",
        )

        self.assertEqual(remaining, 1)
        self.assertEqual(reused, 3)

    def test_single_best_excluded_does_not_reuse_layers(self):
        remaining, reused = initial_depth_after_proactive_reuse(
            4,
            proactive_hit=True,
            reused_depth=2,
            proactive_type="excluded",
            path_policy="single_best",
        )

        self.assertEqual(remaining, 4)
        self.assertEqual(reused, 0)

    def test_reuse_is_capped_by_selected_depth(self):
        remaining, reused = initial_depth_after_proactive_reuse(
            3,
            proactive_hit=True,
            reused_depth=5,
            proactive_type="included",
            path_policy="deepest_multi",
        )

        self.assertEqual(remaining, 0)
        self.assertEqual(reused, 3)


class LocalStreakInitialDraftPolicyTest(unittest.TestCase):
    def create_local_policy(self) -> LocalStreakInitialDraftPolicy:
        return LocalStreakInitialDraftPolicy(
            initial_depth=3,
            min_depth=1,
            max_depth=4,
            state_window_size=5,
            very_slow_depth=1,
            slow_depth=2,
            mid_depth=3,
            fast_depth=4,
            very_slow_accept_threshold=1.2,
            very_slow_depth_threshold=0.2,
            very_slow_exit_accept_threshold=1.4,
            enter_very_slow_votes=2,
            fast_accept_threshold=2.4,
            fast_depth_threshold=2.0,
            slow_accept_threshold=1.6,
            slow_depth_threshold=0.6,
            enter_fast_votes=2,
            enter_slow_votes=2,
            fast_exit_accept_threshold=2.0,
            slow_exit_accept_threshold=1.8,
            reward_clip=20.0,
        )

    def observe_local(
        self,
        policy: LocalStreakInitialDraftPolicy,
        accepted_tokens: int,
    ) -> InitialDraftDecision:
        decision = policy.select_depth(context_ratio=0.0)
        policy.observe(
            decision=decision,
            accepted_tokens=accepted_tokens,
            cycle_ms=100.0,
            draft_ms=10.0,
            response_ms=50.0,
            node_count=4,
            max_budget=8,
            proactive_hit=False,
            proactive_depth=0,
            proactive_max_depth=0,
        )
        return decision

    def test_local_streak_starts_from_configured_depth(self):
        policy = self.create_local_policy()

        decision = policy.select_depth(context_ratio=0.25)

        self.assertEqual(decision.depth, 3)
        self.assertEqual(decision.reason, "local_state")

    def test_local_streak_enters_fast_after_strong_accepts(self):
        policy = self.create_local_policy()

        self.observe_local(policy, accepted_tokens=3)
        self.assertEqual(policy.current_depth, 3)
        self.observe_local(policy, accepted_tokens=3)

        self.assertEqual(policy.current_depth, 4)
        self.assertEqual(policy.stats()["state"], "fast")

    def test_local_streak_enters_slow_after_shallow_accepts(self):
        policy = self.create_local_policy()

        self.observe_local(policy, accepted_tokens=1)
        self.assertEqual(policy.current_depth, 3)
        self.observe_local(policy, accepted_tokens=2)

        self.assertEqual(policy.current_depth, 2)
        self.assertEqual(policy.stats()["state"], "slow")

    def test_local_streak_enters_and_exits_very_slow(self):
        policy = self.create_local_policy()

        self.observe_local(policy, accepted_tokens=1)
        self.assertEqual(policy.current_depth, 3)
        self.observe_local(policy, accepted_tokens=1)
        self.assertEqual(policy.current_depth, 1)
        self.assertEqual(policy.stats()["state"], "very_slow")

        self.observe_local(policy, accepted_tokens=2)
        self.assertEqual(policy.current_depth, 2)
        self.assertEqual(policy.stats()["state"], "slow")

    def test_local_streak_fast_exits_to_mid_on_persistent_weak_accepts(self):
        policy = self.create_local_policy()

        for _ in range(3):
            self.observe_local(policy, accepted_tokens=4)
        self.assertEqual(policy.current_depth, 4)

        for _ in range(5):
            self.observe_local(policy, accepted_tokens=1)

        self.assertLess(policy.current_depth, 4)

    def test_local_streak_respects_depth_bounds(self):
        policy = self.create_local_policy()

        for _ in range(6):
            self.observe_local(policy, accepted_tokens=5)
        self.assertEqual(policy.current_depth, 4)

        for _ in range(12):
            self.observe_local(policy, accepted_tokens=1)
        self.assertEqual(policy.current_depth, 1)


if __name__ == "__main__":
    unittest.main()
