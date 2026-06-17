import unittest

from specedge.client.proactive_policy import AdaptiveProactivePolicy


def create_policy(
    *,
    max_depth: int = 3,
    alpha: float = 1.0,
    min_alignment_rate: float = 0.1,
    low_alignment_depth: int = 0,
    warmup_cycles: int = 0,
    exploration_interval: int = 8,
    safety_margin_ms: float = 5.0,
    uncertainty_scale: float = 1.0,
) -> AdaptiveProactivePolicy:
    return AdaptiveProactivePolicy(
        max_depth=max_depth,
        ewma_alpha=alpha,
        min_alignment_rate=min_alignment_rate,
        low_alignment_depth=low_alignment_depth,
        warmup_cycles=warmup_cycles,
        exploration_interval=exploration_interval,
        safety_margin_ms=safety_margin_ms,
        uncertainty_scale=uncertainty_scale,
    )


class AdaptiveProactivePolicyTest(unittest.TestCase):
    def test_warmup_allows_maximum_depth(self):
        policy = create_policy(warmup_cycles=2)

        self.assertEqual(policy.begin_cycle(), (3, "warmup"))
        self.assertTrue(policy.can_start_setup(0.0).allowed)
        self.assertTrue(policy.can_start_layer(0, 0.0).allowed)
        self.assertEqual(policy.begin_cycle(), (3, "warmup"))

    def test_deadline_uses_response_and_layer_uncertainty(self):
        policy = create_policy()
        policy.observe_cycle(100.0, aligned=True, proactive_executed=True)
        policy.observe_cycle(80.0, aligned=True, proactive_executed=True)
        policy.observe_setup(10.0)
        policy.observe_setup(14.0)
        policy.observe_step(0, wall_ms=20.0, gpu_ms=15.0)
        policy.observe_step(0, wall_ms=30.0, gpu_ms=25.0)
        policy.begin_cycle()

        setup_decision = policy.can_start_setup(request_elapsed_ms=0.0)
        early_decision = policy.can_start_layer(
            layer_index=0,
            request_elapsed_ms=10.0,
        )
        late_decision = policy.can_start_layer(
            layer_index=0,
            request_elapsed_ms=20.0,
        )

        self.assertTrue(setup_decision.allowed)
        self.assertEqual(setup_decision.predicted_cost_ms, 18.0)
        self.assertTrue(early_decision.allowed)
        self.assertEqual(early_decision.remaining_ms, 50.0)
        self.assertEqual(early_decision.predicted_cost_ms, 40.0)
        self.assertFalse(late_decision.allowed)
        self.assertEqual(late_decision.reason, "layer_deadline")

    def test_each_layer_has_an_independent_latency_model(self):
        policy = create_policy(safety_margin_ms=0.0)
        policy.observe_cycle(100.0, aligned=True, proactive_executed=True)
        policy.observe_setup(10.0)
        policy.observe_step(0, wall_ms=20.0, gpu_ms=15.0)
        policy.observe_step(1, wall_ms=70.0, gpu_ms=60.0)
        policy.begin_cycle()

        first_layer = policy.can_start_layer(0, request_elapsed_ms=31.0)
        second_layer = policy.can_start_layer(1, request_elapsed_ms=31.0)

        self.assertTrue(first_layer.allowed)
        self.assertFalse(second_layer.allowed)
        self.assertEqual(second_layer.reason, "layer_deadline")

    def test_batch_width_has_an_independent_latency_model(self):
        policy = create_policy(safety_margin_ms=0.0)
        policy.observe_cycle(100.0, aligned=True, proactive_executed=True)
        policy.observe_step(
            0,
            wall_ms=20.0,
            gpu_ms=15.0,
            batch_width=2,
        )
        policy.observe_step(
            0,
            wall_ms=80.0,
            gpu_ms=70.0,
            batch_width=8,
        )
        policy.begin_cycle()

        narrow = policy.can_start_layer(
            0,
            request_elapsed_ms=30.0,
            batch_width=2,
        )
        wide = policy.can_start_layer(
            0,
            request_elapsed_ms=30.0,
            batch_width=8,
        )

        self.assertTrue(narrow.allowed)
        self.assertFalse(wide.allowed)

    def test_setup_is_skipped_when_response_may_arrive_early(self):
        policy = create_policy()
        policy.observe_cycle(100.0, aligned=True, proactive_executed=True)
        policy.observe_cycle(60.0, aligned=True, proactive_executed=True)
        policy.observe_setup(30.0)
        policy.begin_cycle()

        decision = policy.can_start_setup(request_elapsed_ms=0.0)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "setup_deadline")
        self.assertEqual(decision.remaining_ms, 20.0)

    def test_low_alignment_skips_until_exploration_cycle(self):
        policy = create_policy(
            min_alignment_rate=0.5,
            exploration_interval=2,
        )
        policy.observe_cycle(
            response_ms=100.0,
            aligned=False,
            proactive_executed=True,
        )

        self.assertEqual(policy.begin_cycle(), (0, "low_alignment_rate"))
        self.assertEqual(
            policy.begin_cycle(),
            (3, "alignment_exploration"),
        )

    def test_low_alignment_can_keep_shallow_exploration(self):
        policy = create_policy(
            min_alignment_rate=0.5,
            low_alignment_depth=1,
            exploration_interval=4,
        )
        policy.observe_cycle(
            response_ms=100.0,
            aligned=False,
            proactive_executed=True,
        )

        self.assertEqual(policy.begin_cycle(), (1, "low_alignment_limited"))


if __name__ == "__main__":
    unittest.main()
