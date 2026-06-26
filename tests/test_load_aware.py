import unittest

from metric.load_aware import (
    BackgroundScheduler,
    LoadProfilePoint,
    LoadState,
    ServerResponseTimeEmaEstimator,
    SimulationConfig,
    StrategyMetrics,
    choose_mode,
    predict_response_latency_ms,
    simulate_strategy,
    weighted_throughput,
)


class LoadAwarePolicyTest(unittest.TestCase):
    def test_threshold_selects_network_ar_at_or_below_threshold(self):
        self.assertEqual(choose_mode(50.0, 60.0), "network_ar")
        self.assertEqual(choose_mode(60.0, 60.0), "network_ar")
        self.assertEqual(choose_mode(100.0, 60.0), "response_only")

    def test_weighted_throughput_uses_harmonic_mean(self):
        points = [
            LoadProfilePoint(
                latency_ms=50.0,
                autoregressive=StrategyMetrics(20.0),
                original_specedge=StrategyMetrics(10.0),
                response_only=StrategyMetrics(15.0),
                selected_mode="network_ar",
                selected_tokens_per_second=20.0,
            ),
            LoadProfilePoint(
                latency_ms=100.0,
                autoregressive=StrategyMetrics(5.0),
                original_specedge=StrategyMetrics(8.0),
                response_only=StrategyMetrics(10.0),
                selected_mode="response_only",
                selected_tokens_per_second=10.0,
            ),
        ]

        throughput = weighted_throughput(
            points,
            weights={50.0: 1.0, 100.0: 1.0},
            strategy="load_aware",
        )

        self.assertAlmostEqual(throughput, 2 / (1 / 20 + 1 / 10))

    def test_zero_weight_latency_is_ignored(self):
        points = [
            LoadProfilePoint(
                latency_ms=50.0,
                autoregressive=StrategyMetrics(20.0),
                original_specedge=StrategyMetrics(10.0),
                response_only=StrategyMetrics(15.0),
                selected_mode="network_ar",
                selected_tokens_per_second=20.0,
            ),
            LoadProfilePoint(
                latency_ms=100.0,
                autoregressive=StrategyMetrics(5.0),
                original_specedge=StrategyMetrics(8.0),
                response_only=StrategyMetrics(10.0),
                selected_mode="response_only",
                selected_tokens_per_second=10.0,
            ),
        ]

        throughput = weighted_throughput(
            points,
            weights={50.0: 0.0, 100.0: 1.0},
            strategy="load_aware",
        )

        self.assertAlmostEqual(throughput, 10.0)

    def test_predict_response_latency_uses_scheduler_state(self):
        latency = predict_response_latency_ms(
            LoadState(
                batch_size=3,
                queue_length=4,
                prefill_count=2,
                base_decode_latency_ms=50.0,
            ),
            active_penalty_ms=10.0,
            queue_penalty_ms=5.0,
            prefill_penalty_ms=20.0,
        )

        self.assertEqual(latency, 50.0 + 2 * 10.0 + 4 * 5.0 + 2 * 20.0)

    def test_server_response_time_ema_estimator(self):
        estimator = ServerResponseTimeEmaEstimator(alpha=0.2)

        self.assertEqual(estimator.estimate_or(50.0), 50.0)
        self.assertEqual(estimator.observe(100.0), 100.0)
        self.assertAlmostEqual(estimator.observe(200.0), 120.0)

    def test_background_scheduler_completes_tokens(self):
        config = SimulationConfig(
            foreground_requests=1,
            foreground_tokens=8,
            decision_window=4,
            threshold_ms=60.0,
            base_decode_latency_ms=10.0,
            max_batch_size=2,
            background_arrival_rate=100.0,
            background_min_tokens=1,
            background_max_tokens=1,
            active_penalty_ms=0.0,
            queue_penalty_ms=0.0,
            prefill_penalty_ms=0.0,
            ar_to_specedge_prefill_ms=0.0,
            estimator_alpha=0.2,
            background_load_label="test",
            seed=1,
        )
        scheduler = BackgroundScheduler(config)

        scheduler.advance(1000.0)

        self.assertGreater(scheduler.completed_tokens, 0)
        self.assertGreaterEqual(scheduler.state().batch_size, 1)

    def test_dynamic_simulation_switches_when_load_crosses_threshold(self):
        points = [
            LoadProfilePoint(
                latency_ms=50.0,
                autoregressive=StrategyMetrics(20.0),
                original_specedge=StrategyMetrics(10.0, cycle_ms=200.0),
                response_only=StrategyMetrics(15.0, cycle_ms=200.0),
                selected_mode="network_ar",
                selected_tokens_per_second=20.0,
            ),
            LoadProfilePoint(
                latency_ms=100.0,
                autoregressive=StrategyMetrics(5.0),
                original_specedge=StrategyMetrics(8.0, cycle_ms=250.0),
                response_only=StrategyMetrics(12.0, cycle_ms=250.0),
                selected_mode="response_only",
                selected_tokens_per_second=12.0,
            ),
        ]
        config = SimulationConfig(
            foreground_requests=3,
            foreground_tokens=64,
            decision_window=4,
            threshold_ms=60.0,
            base_decode_latency_ms=50.0,
            max_batch_size=2,
            background_arrival_rate=2.0,
            background_min_tokens=8,
            background_max_tokens=8,
            active_penalty_ms=20.0,
            queue_penalty_ms=10.0,
            prefill_penalty_ms=20.0,
            ar_to_specedge_prefill_ms=0.0,
            estimator_alpha=0.2,
            background_load_label="test",
            seed=2,
        )

        result = simulate_strategy(points, config, "load_aware")

        self.assertEqual(result.foreground_tokens, 192)
        self.assertGreater(result.foreground_tokens_per_second, 0.0)
        self.assertGreater(result.mode_counts["network_ar"], 0)
        self.assertGreater(result.mode_counts["response_only"], 0)


if __name__ == "__main__":
    unittest.main()
