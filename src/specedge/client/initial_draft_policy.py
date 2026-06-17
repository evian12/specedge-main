from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from specedge.client.proactive_policy import EwmaEstimate


def initial_depth_after_proactive_reuse(
    selected_depth: int,
    *,
    proactive_hit: bool,
    reused_depth: int,
    proactive_type: str,
    path_policy: str,
) -> tuple[int, int]:
    """Return new draft layers and the number of reused proactive layers."""
    selected_depth = max(0, selected_depth)
    should_reuse = proactive_hit and (
        proactive_type == "included"
        or path_policy == "sequence_depth"
    )
    actual_reuse = (
        min(selected_depth, max(0, reused_depth))
        if should_reuse
        else 0
    )
    return selected_depth - actual_reuse, actual_reuse


@dataclass
class InitialDraftDecision:
    depth: int
    reason: str
    features: list[float]
    scores: dict[int, float]


class LocalStreakInitialDraftPolicy:
    """Per-request draft depth control with score and recent-history guard."""

    feature_names = [
        "current_depth",
        "score",
        "recent_accepted_depth_mean",
        "context_ratio",
    ]

    def __init__(
        self,
        *,
        initial_depth: int,
        min_depth: int,
        max_depth: int,
        high_score: float,
        low_penalty: float,
        increase_score_threshold: float,
        decrease_score_threshold: float,
        protect_window: int,
        protect_avg_accepted_depth: float,
        neutral_score_decay: float,
        reward_clip: float,
    ) -> None:
        if min_depth <= 0:
            raise ValueError("min_depth must be positive")
        if max_depth < min_depth:
            raise ValueError("max_depth must be >= min_depth")
        if not min_depth <= initial_depth <= max_depth:
            raise ValueError("initial_depth must be in [min_depth, max_depth]")
        if high_score <= 0.0:
            raise ValueError("high_score must be positive")
        if low_penalty <= 0.0:
            raise ValueError("low_penalty must be positive")
        if increase_score_threshold <= 0.0:
            raise ValueError("increase_score_threshold must be positive")
        if decrease_score_threshold <= 0.0:
            raise ValueError("decrease_score_threshold must be positive")
        if protect_window <= 0:
            raise ValueError("protect_window must be positive")
        if protect_avg_accepted_depth < 0.0:
            raise ValueError("protect_avg_accepted_depth must be non-negative")
        if not 0.0 <= neutral_score_decay <= 1.0:
            raise ValueError("neutral_score_decay must be in [0, 1]")
        if reward_clip <= 0.0:
            raise ValueError("reward_clip must be positive")

        self.initial_depth = initial_depth
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.high_score = high_score
        self.low_penalty = low_penalty
        self.increase_score_threshold = increase_score_threshold
        self.decrease_score_threshold = decrease_score_threshold
        self.protect_window = protect_window
        self.protect_avg_accepted_depth = protect_avg_accepted_depth
        self.neutral_score_decay = neutral_score_decay
        self.reward_clip = reward_clip

        self.cycles = 0
        self.current_depth = initial_depth
        self.score = 0.0
        self.recent_accepted_depths: deque[int] = deque(
            maxlen=protect_window
        )
        self.last_accepted_depth: Optional[int] = None
        self.last_reward: Optional[float] = None
        self.counts = {
            depth: 0 for depth in range(min_depth, max_depth + 1)
        }

    def _recent_accepted_depth_mean(self) -> Optional[float]:
        if not self.recent_accepted_depths:
            return None
        return sum(self.recent_accepted_depths) / len(
            self.recent_accepted_depths
        )

    def _features(self, context_ratio: float) -> list[float]:
        return [
            float(self.current_depth),
            float(self.score),
            float(self._recent_accepted_depth_mean() or 0.0),
            min(1.0, max(0.0, context_ratio)),
        ]

    def select_depth(self, context_ratio: float) -> InitialDraftDecision:
        self.cycles += 1
        return InitialDraftDecision(
            depth=self.current_depth,
            reason="local_streak",
            features=self._features(context_ratio),
            scores={},
        )

    def observe(
        self,
        decision: InitialDraftDecision,
        accepted_tokens: int,
        cycle_ms: float,
        draft_ms: float,
        response_ms: Optional[float],
        node_count: int,
        max_budget: int,
        proactive_hit: bool,
        proactive_depth: int,
        proactive_max_depth: int,
    ) -> float:
        if cycle_ms <= 0.0:
            raise ValueError("cycle_ms must be positive")

        reward = min(
            self.reward_clip,
            max(0.0, 1000.0 * accepted_tokens / cycle_ms),
        )
        depth = decision.depth
        self.counts[depth] = self.counts.get(depth, 0) + 1
        accepted_depth = max(0, accepted_tokens - 1)
        self.last_accepted_depth = accepted_depth
        self.last_reward = reward
        previous_recent_mean = self._recent_accepted_depth_mean() or 0.0

        if accepted_depth >= depth - 1:
            self.score += self.high_score
        elif accepted_depth <= 1:
            self.score -= self.low_penalty
        else:
            self.score *= self.neutral_score_decay

        if self.score >= self.increase_score_threshold:
            if self.current_depth < self.max_depth:
                self.current_depth += 1
            self.score = 0.0
        elif (
            self.score <= -self.decrease_score_threshold
            and self.current_depth > self.min_depth
        ):
            if previous_recent_mean >= self.protect_avg_accepted_depth:
                self.score = -self.low_penalty
            else:
                self.current_depth -= 1
                self.score = 0.0

        self.recent_accepted_depths.append(accepted_depth)
        return reward

    def stats(self) -> dict:
        recent_mean = self._recent_accepted_depth_mean()
        return {
            "cycles": self.cycles,
            "current_depth": self.current_depth,
            "initial_depth": self.initial_depth,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "score": self.score,
            "high_score": self.high_score,
            "low_penalty": self.low_penalty,
            "increase_score_threshold": self.increase_score_threshold,
            "decrease_score_threshold": self.decrease_score_threshold,
            "protect_window": self.protect_window,
            "protect_avg_accepted_depth": (
                self.protect_avg_accepted_depth
            ),
            "neutral_score_decay": self.neutral_score_decay,
            "recent_accepted_depth_mean": recent_mean,
            "recent_accepted_depths": list(self.recent_accepted_depths),
            "last_accepted_depth": self.last_accepted_depth,
            "last_reward": self.last_reward,
            "counts": {
                str(depth): count
                for depth, count in sorted(self.counts.items())
            },
        }


class LinUCBInitialDraftPolicy:
    """Online initial-draft depth selection with disjoint LinUCB models."""

    def __init__(
        self,
        candidate_depths: list[int],
        max_depth: int,
        exploration_weight: float,
        warmup_per_depth: int,
        forced_exploration_interval: int,
        ridge_lambda: float,
        reward_clip: float,
        ewma_alpha: float,
        seed: int,
    ) -> None:
        depths = sorted(set(candidate_depths))
        if not depths or depths[0] <= 0:
            raise ValueError("candidate_depths must contain positive depths")
        if depths[-1] > max_depth:
            raise ValueError("candidate_depths must not exceed max_beam_len")
        if exploration_weight < 0.0:
            raise ValueError("exploration_weight must be non-negative")
        if warmup_per_depth < 0:
            raise ValueError("warmup_per_depth must be non-negative")
        if forced_exploration_interval < 0:
            raise ValueError(
                "forced_exploration_interval must be non-negative"
            )
        if ridge_lambda <= 0.0:
            raise ValueError("ridge_lambda must be positive")
        if reward_clip <= 0.0:
            raise ValueError("reward_clip must be positive")
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")

        self.candidate_depths = depths
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight
        self.warmup_per_depth = warmup_per_depth
        self.forced_exploration_interval = forced_exploration_interval
        self.ridge_lambda = ridge_lambda
        self.reward_clip = reward_clip
        self.ewma_alpha = ewma_alpha
        self._rng = np.random.default_rng(seed)

        self.cycles = 0
        self.counts = {depth: 0 for depth in depths}
        self.reward_by_depth = {
            depth: EwmaEstimate(ewma_alpha) for depth in depths
        }
        self.draft_ms_by_depth = {
            depth: EwmaEstimate(ewma_alpha) for depth in depths
        }
        self.accepted_depth = EwmaEstimate(ewma_alpha)
        self.response_ms = EwmaEstimate(ewma_alpha)
        self.acceptance_survival = {
            depth: EwmaEstimate(ewma_alpha)
            for depth in range(1, max_depth + 1)
        }
        self.previous_node_ratio = 0.0
        self.previous_proactive_hit = 0.0
        self.previous_proactive_depth_ratio = 0.0

        self.feature_count = (
            1
            + max_depth
            + 2
            + len(depths)
            + 2
            + 4
        )
        self.feature_names = (
            ["bias"]
            + [
                f"accept_survival_{depth}"
                for depth in range(1, max_depth + 1)
            ]
            + ["accepted_depth_mean", "accepted_depth_error"]
            + [
                f"draft_ms_depth_{depth}"
                for depth in self.candidate_depths
            ]
            + [
                "response_ms_mean",
                "response_ms_error",
                "previous_node_ratio",
                "context_ratio",
                "previous_proactive_hit",
                "previous_proactive_depth_ratio",
            ]
        )
        self._a = {
            depth: np.eye(self.feature_count, dtype=np.float64)
            * ridge_lambda
            for depth in depths
        }
        self._b = {
            depth: np.zeros(self.feature_count, dtype=np.float64)
            for depth in depths
        }

    @staticmethod
    def _scaled(value: Optional[float], scale: float) -> float:
        return 0.0 if value is None else float(value) / scale

    @staticmethod
    def _estimate_stats(estimate: EwmaEstimate) -> dict[str, Optional[float]]:
        return {
            "mean": estimate.mean,
            "error": estimate.error,
        }

    def build_features(self, context_ratio: float) -> list[float]:
        features = [1.0]
        features.extend(
            float(self.acceptance_survival[depth].mean or 0.0)
            for depth in range(1, self.max_depth + 1)
        )
        features.extend(
            [
                self._scaled(self.accepted_depth.mean, self.max_depth),
                self._scaled(self.accepted_depth.error, self.max_depth),
            ]
        )
        features.extend(
            self._scaled(self.draft_ms_by_depth[depth].mean, 100.0)
            for depth in self.candidate_depths
        )
        features.extend(
            [
                self._scaled(self.response_ms.mean, 1000.0),
                self._scaled(self.response_ms.error, 1000.0),
                self.previous_node_ratio,
                min(1.0, max(0.0, context_ratio)),
                self.previous_proactive_hit,
                self.previous_proactive_depth_ratio,
            ]
        )
        return features

    def _least_sampled_depth(self, depths: list[int]) -> int:
        min_count = min(self.counts[depth] for depth in depths)
        choices = [
            depth for depth in depths if self.counts[depth] == min_count
        ]
        return int(self._rng.choice(choices))

    def select_depth(self, context_ratio: float) -> InitialDraftDecision:
        self.cycles += 1
        features = self.build_features(context_ratio)

        warmup_depths = [
            depth
            for depth in self.candidate_depths
            if self.counts[depth] < self.warmup_per_depth
        ]
        if warmup_depths:
            depth = self._least_sampled_depth(warmup_depths)
            return InitialDraftDecision(depth, "warmup", features, {})

        if (
            self.forced_exploration_interval > 0
            and self.cycles % self.forced_exploration_interval == 0
        ):
            depth = self._least_sampled_depth(self.candidate_depths)
            return InitialDraftDecision(
                depth,
                "forced_exploration",
                features,
                {},
            )

        x = np.asarray(features, dtype=np.float64)
        scores: dict[int, float] = {}
        for depth in self.candidate_depths:
            a_inv_x = np.linalg.solve(self._a[depth], x)
            theta = np.linalg.solve(self._a[depth], self._b[depth])
            predicted_reward = float(theta @ x)
            uncertainty = float(np.sqrt(max(0.0, x @ a_inv_x)))
            scores[depth] = (
                predicted_reward
                + self.exploration_weight * uncertainty
            )

        best_score = max(scores.values())
        best_depths = [
            depth
            for depth, score in scores.items()
            if np.isclose(score, best_score)
        ]
        depth = self._least_sampled_depth(best_depths)
        return InitialDraftDecision(depth, "linucb", features, scores)

    def observe(
        self,
        decision: InitialDraftDecision,
        accepted_tokens: int,
        cycle_ms: float,
        draft_ms: float,
        response_ms: Optional[float],
        node_count: int,
        max_budget: int,
        proactive_hit: bool,
        proactive_depth: int,
        proactive_max_depth: int,
    ) -> float:
        if cycle_ms <= 0.0:
            raise ValueError("cycle_ms must be positive")

        reward = min(
            self.reward_clip,
            max(0.0, 1000.0 * accepted_tokens / cycle_ms),
        )
        x = np.asarray(decision.features, dtype=np.float64)
        depth = decision.depth
        self._a[depth] += np.outer(x, x)
        self._b[depth] += reward * x
        self.counts[depth] += 1
        self.reward_by_depth[depth].observe(reward)
        self.draft_ms_by_depth[depth].observe(draft_ms)

        accepted_draft_depth = max(0, accepted_tokens - 1)
        self.accepted_depth.observe(float(accepted_draft_depth))
        for candidate_depth, estimate in self.acceptance_survival.items():
            estimate.observe(
                float(accepted_draft_depth >= candidate_depth)
            )
        if response_ms is not None:
            self.response_ms.observe(response_ms)

        self.previous_node_ratio = min(
            1.0,
            node_count / max(1, max_budget),
        )
        self.previous_proactive_hit = float(proactive_hit)
        self.previous_proactive_depth_ratio = min(
            1.0,
            proactive_depth / max(1, proactive_max_depth),
        )
        return reward

    def stats(self) -> dict:
        return {
            "cycles": self.cycles,
            "counts": {
                str(depth): self.counts[depth]
                for depth in self.candidate_depths
            },
            "reward_by_depth": {
                str(depth): self._estimate_stats(
                    self.reward_by_depth[depth]
                )
                for depth in self.candidate_depths
            },
            "draft_ms_by_depth": {
                str(depth): self._estimate_stats(
                    self.draft_ms_by_depth[depth]
                )
                for depth in self.candidate_depths
            },
            "accepted_depth": self._estimate_stats(self.accepted_depth),
            "response_ms": self._estimate_stats(self.response_ms),
            "acceptance_survival": {
                str(depth): self._estimate_stats(
                    self.acceptance_survival[depth]
                )
                for depth in range(1, self.max_depth + 1)
            },
            "feature_names": self.feature_names,
        }
