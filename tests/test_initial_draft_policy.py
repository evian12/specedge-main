import unittest

from specedge.client.initial_draft_policy import (
    InitialDraftDecision,
    LinUCBInitialDraftPolicy,
    LocalStreakInitialDraftPolicy,
    initial_depth_after_proactive_reuse,
)


def create_policy(
    *,
    candidate_depths: list[int] = [1, 2, 3],
    warmup_per_depth: int = 0,
    exploration_weight: float = 0.0,
    forced_exploration_interval: int = 0,
) -> LinUCBInitialDraftPolicy:
    return LinUCBInitialDraftPolicy(
        candidate_depths=candidate_depths,
        max_depth=max(candidate_depths),
        exploration_weight=exploration_weight,
        warmup_per_depth=warmup_per_depth,
        forced_exploration_interval=forced_exploration_interval,
        ridge_lambda=1.0,
        reward_clip=20.0,
        ewma_alpha=0.2,
        seed=42,
    )


def observe(
    policy: LinUCBInitialDraftPolicy,
    decision: InitialDraftDecision,
    accepted_tokens: int,
    cycle_ms: float,
) -> float:
    return policy.observe(
        decision=decision,
        accepted_tokens=accepted_tokens,
        cycle_ms=cycle_ms,
        draft_ms=10.0,
        response_ms=50.0,
        node_count=16,
        max_budget=32,
        proactive_hit=False,
        proactive_depth=0,
        proactive_max_depth=3,
    )


class LinUCBInitialDraftPolicyTest(unittest.TestCase):
    def test_warmup_samples_every_candidate_depth(self):
        policy = create_policy(warmup_per_depth=1)
        selected = []

        for _ in policy.candidate_depths:
            decision = policy.select_depth(context_ratio=0.25)
            selected.append(decision.depth)
            self.assertEqual(decision.reason, "warmup")
            observe(policy, decision, accepted_tokens=2, cycle_ms=100.0)

        self.assertEqual(set(selected), set(policy.candidate_depths))

    def test_linucb_prefers_depth_with_higher_observed_reward(self):
        policy = create_policy()

        for depth, accepted_tokens in [(1, 1), (2, 1), (3, 4)] * 4:
            features = policy.build_features(context_ratio=0.5)
            decision = InitialDraftDecision(
                depth=depth,
                reason="test",
                features=features,
                scores={},
            )
            observe(
                policy,
                decision,
                accepted_tokens=accepted_tokens,
                cycle_ms=100.0,
            )

        decision = policy.select_depth(context_ratio=0.5)

        self.assertEqual(decision.reason, "linucb")
        self.assertEqual(decision.depth, 3)
        self.assertGreater(decision.scores[3], decision.scores[1])

    def test_forced_exploration_uses_least_sampled_depth(self):
        policy = create_policy(forced_exploration_interval=2)
        features = policy.build_features(context_ratio=0.0)
        observe(
            policy,
            InitialDraftDecision(1, "test", features, {}),
            accepted_tokens=2,
            cycle_ms=100.0,
        )

        first = policy.select_depth(context_ratio=0.0)
        second = policy.select_depth(context_ratio=0.0)

        self.assertEqual(first.reason, "linucb")
        self.assertEqual(second.reason, "forced_exploration")
        self.assertIn(second.depth, {2, 3})

    def test_reward_is_throughput_and_clipped(self):
        policy = create_policy()
        decision = policy.select_depth(context_ratio=0.0)

        reward = observe(
            policy,
            decision,
            accepted_tokens=10,
            cycle_ms=100.0,
        )

        self.assertEqual(reward, 20.0)
        self.assertEqual(policy.counts[decision.depth], 1)

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
            min_depth=2,
            max_depth=4,
            high_score=2.0,
            low_penalty=1.0,
            increase_score_threshold=3.0,
            decrease_score_threshold=3.0,
            protect_window=5,
            protect_avg_accepted_depth=2.0,
            neutral_score_decay=0.8,
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
        self.assertEqual(decision.reason, "local_streak")

    def test_local_streak_increases_after_strong_accepts(self):
        policy = self.create_local_policy()

        self.observe_local(policy, accepted_tokens=3)
        self.assertEqual(policy.current_depth, 3)
        self.observe_local(policy, accepted_tokens=3)

        self.assertEqual(policy.current_depth, 4)

    def test_local_streak_decreases_after_repeated_shallow_accepts(self):
        policy = self.create_local_policy()

        self.observe_local(policy, accepted_tokens=1)
        self.assertEqual(policy.current_depth, 3)
        self.observe_local(policy, accepted_tokens=2)
        self.assertEqual(policy.current_depth, 3)
        self.observe_local(policy, accepted_tokens=1)
        self.assertEqual(policy.current_depth, 2)

    def test_local_streak_protects_good_recent_history(self):
        policy = self.create_local_policy()

        for accepted_tokens in [4, 4, 4, 4, 4]:
            self.observe_local(policy, accepted_tokens=accepted_tokens)
        self.assertEqual(policy.current_depth, 4)

        for _ in range(3):
            self.observe_local(policy, accepted_tokens=1)

        self.assertEqual(policy.current_depth, 4)
        self.assertLess(policy.score, 0.0)

    def test_local_streak_respects_depth_bounds(self):
        policy = self.create_local_policy()

        for _ in range(6):
            self.observe_local(policy, accepted_tokens=5)
        self.assertEqual(policy.current_depth, 4)

        for _ in range(12):
            self.observe_local(policy, accepted_tokens=1)
        self.assertEqual(policy.current_depth, 2)


if __name__ == "__main__":
    unittest.main()
