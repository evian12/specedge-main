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
    """Per-request draft depth control with a guarded throughput state."""

    feature_names = [
        "current_depth",
        "state_index",
        "recent_accept_mean",
        "recent_accepted_depth_mean",
        "fast_votes",
        "slow_votes",
        "very_slow_votes",
        "context_ratio",
    ]

    _STATE_TO_INDEX = {
        "very_slow": 0,
        "slow": 1,
        "mid": 2,
        "fast": 3,
    }

    def __init__(
        self,
        *,
        initial_depth: int,
        min_depth: int,
        max_depth: int,
        state_window_size: int,
        very_slow_depth: int,
        slow_depth: int,
        mid_depth: int,
        fast_depth: int,
        very_slow_accept_threshold: float,
        very_slow_depth_threshold: float,
        very_slow_exit_accept_threshold: float,
        enter_very_slow_votes: int,
        fast_accept_threshold: float,
        fast_depth_threshold: float,
        slow_accept_threshold: float,
        slow_depth_threshold: float,
        enter_fast_votes: int,
        enter_slow_votes: int,
        fast_exit_accept_threshold: float,
        slow_exit_accept_threshold: float,
        reward_clip: float,
    ) -> None:
        if min_depth <= 0:
            raise ValueError("min_depth must be positive")
        if max_depth < min_depth:
            raise ValueError("max_depth must be >= min_depth")
        if not min_depth <= initial_depth <= max_depth:
            raise ValueError("initial_depth must be in [min_depth, max_depth]")
        if state_window_size <= 0:
            raise ValueError("state_window_size must be positive")
        for name, depth in [
            ("very_slow_depth", very_slow_depth),
            ("slow_depth", slow_depth),
            ("mid_depth", mid_depth),
            ("fast_depth", fast_depth),
        ]:
            if not min_depth <= depth <= max_depth:
                raise ValueError(
                    f"{name} must be in [min_depth, max_depth]"
                )
        for name, threshold in [
            (
                "very_slow_accept_threshold",
                very_slow_accept_threshold,
            ),
            ("very_slow_depth_threshold", very_slow_depth_threshold),
            (
                "very_slow_exit_accept_threshold",
                very_slow_exit_accept_threshold,
            ),
            ("fast_accept_threshold", fast_accept_threshold),
            ("fast_depth_threshold", fast_depth_threshold),
            ("slow_accept_threshold", slow_accept_threshold),
            ("slow_depth_threshold", slow_depth_threshold),
            ("fast_exit_accept_threshold", fast_exit_accept_threshold),
            ("slow_exit_accept_threshold", slow_exit_accept_threshold),
        ]:
            if threshold < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name, votes in [
            ("enter_very_slow_votes", enter_very_slow_votes),
            ("enter_fast_votes", enter_fast_votes),
            ("enter_slow_votes", enter_slow_votes),
        ]:
            if votes <= 0:
                raise ValueError(f"{name} must be positive")
        if reward_clip <= 0.0:
            raise ValueError("reward_clip must be positive")

        self.initial_depth = initial_depth
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.state_window_size = state_window_size
        self.depth_by_state = {
            "very_slow": very_slow_depth,
            "slow": slow_depth,
            "mid": mid_depth,
            "fast": fast_depth,
        }
        self.very_slow_accept_threshold = very_slow_accept_threshold
        self.very_slow_depth_threshold = very_slow_depth_threshold
        self.very_slow_exit_accept_threshold = (
            very_slow_exit_accept_threshold
        )
        self.enter_very_slow_votes = enter_very_slow_votes
        self.fast_accept_threshold = fast_accept_threshold
        self.fast_depth_threshold = fast_depth_threshold
        self.slow_accept_threshold = slow_accept_threshold
        self.slow_depth_threshold = slow_depth_threshold
        self.enter_fast_votes = enter_fast_votes
        self.enter_slow_votes = enter_slow_votes
        self.fast_exit_accept_threshold = fast_exit_accept_threshold
        self.slow_exit_accept_threshold = slow_exit_accept_threshold
        self.reward_clip = reward_clip

        self.cycles = 0
        self.current_depth = initial_depth
        self.state = "mid"
        self.fast_votes = 0
        self.slow_votes = 0
        self.very_slow_votes = 0
        self.history: deque[tuple[int, int]] = deque(
            maxlen=state_window_size
        )
        self.last_accepted_depth: Optional[int] = None
        self.last_accepted_tokens: Optional[int] = None
        self.last_reward: Optional[float] = None
        self.counts = {
            depth: 0 for depth in range(min_depth, max_depth + 1)
        }

    def _recent_means(self) -> tuple[Optional[float], Optional[float]]:
        if not self.history:
            return None, None
        token_mean = sum(tokens for tokens, _depth in self.history) / len(
            self.history
        )
        depth_mean = sum(depth for _tokens, depth in self.history) / len(
            self.history
        )
        return token_mean, depth_mean

    def _has_recent_depth(self, minimum_depth: int) -> bool:
        return any(depth >= minimum_depth for _tokens, depth in self.history)

    def _bounded_depth_for_state(self, state: str) -> int:
        depth = self.depth_by_state[state]
        return min(self.max_depth, max(self.min_depth, depth))

    def _set_state(self, state: str) -> None:
        self.state = state
        self.current_depth = self._bounded_depth_for_state(state)

    def _decrement_vote(self, value: int) -> int:
        return max(0, value - 1)

    def _transition_state(self) -> None:
        recent_accept, recent_depth = self._recent_means()
        if recent_accept is None or recent_depth is None:
            return

        very_slow_condition = (
            recent_accept <= self.very_slow_accept_threshold
            and recent_depth <= self.very_slow_depth_threshold
        )
        fast_condition = (
            recent_accept >= self.fast_accept_threshold
            or recent_depth >= self.fast_depth_threshold
        )
        slow_condition = (
            recent_accept <= self.slow_accept_threshold
            or recent_depth <= self.slow_depth_threshold
        )

        if self.state == "very_slow":
            if (
                recent_accept > self.very_slow_exit_accept_threshold
                or self._has_recent_depth(1)
            ):
                self.very_slow_votes = 0
                self.slow_votes = 0
                self.fast_votes = 0
                self._set_state("slow")
            return

        if very_slow_condition:
            self.very_slow_votes += 1
        else:
            self.very_slow_votes = self._decrement_vote(
                self.very_slow_votes
            )

        if fast_condition:
            self.fast_votes += 1
        else:
            self.fast_votes = self._decrement_vote(self.fast_votes)

        if slow_condition:
            self.slow_votes += 1
        else:
            self.slow_votes = self._decrement_vote(self.slow_votes)

        if (
            self.state == "fast"
            and recent_accept < self.fast_exit_accept_threshold
        ):
            self.fast_votes = 0
            self._set_state("mid")
        elif (
            self.state == "slow"
            and recent_accept > self.slow_exit_accept_threshold
        ):
            self.slow_votes = 0
            self._set_state("mid")

        if self.very_slow_votes >= self.enter_very_slow_votes:
            self._set_state("very_slow")
        elif self.fast_votes >= self.enter_fast_votes:
            self._set_state("fast")
        elif self.slow_votes >= self.enter_slow_votes:
            self._set_state("slow")
        elif self.state not in ["fast", "slow"]:
            self._set_state("mid")

    def _features(self, context_ratio: float) -> list[float]:
        recent_accept, recent_depth = self._recent_means()
        return [
            float(self.current_depth),
            float(self._STATE_TO_INDEX[self.state]),
            float(recent_accept or 0.0),
            float(recent_depth or 0.0),
            float(self.fast_votes),
            float(self.slow_votes),
            float(self.very_slow_votes),
            min(1.0, max(0.0, context_ratio)),
        ]

    def select_depth(self, context_ratio: float) -> InitialDraftDecision:
        self.cycles += 1
        return InitialDraftDecision(
            depth=self.current_depth,
            reason="local_state",
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

        accepted_tokens = max(0, accepted_tokens)
        reward = min(
            self.reward_clip,
            max(0.0, 1000.0 * accepted_tokens / cycle_ms),
        )
        depth = decision.depth
        self.counts[depth] = self.counts.get(depth, 0) + 1
        accepted_depth = max(0, accepted_tokens - 1)
        self.last_accepted_tokens = accepted_tokens
        self.last_accepted_depth = accepted_depth
        self.last_reward = reward
        self.history.append((accepted_tokens, accepted_depth))
        self._transition_state()
        return reward

    def stats(self) -> dict:
        recent_accept, recent_depth = self._recent_means()
        return {
            "cycles": self.cycles,
            "current_depth": self.current_depth,
            "initial_depth": self.initial_depth,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "state": self.state,
            "state_index": self._STATE_TO_INDEX[self.state],
            "state_window_size": self.state_window_size,
            "depth_by_state": dict(self.depth_by_state),
            "recent_accept_mean": recent_accept,
            "recent_accepted_depth_mean": recent_depth,
            "recent_history": [
                {
                    "accepted_tokens": tokens,
                    "accepted_depth": depth,
                }
                for tokens, depth in self.history
            ],
            "fast_votes": self.fast_votes,
            "slow_votes": self.slow_votes,
            "very_slow_votes": self.very_slow_votes,
            "very_slow_accept_threshold": (
                self.very_slow_accept_threshold
            ),
            "very_slow_depth_threshold": self.very_slow_depth_threshold,
            "very_slow_exit_accept_threshold": (
                self.very_slow_exit_accept_threshold
            ),
            "enter_very_slow_votes": self.enter_very_slow_votes,
            "fast_accept_threshold": self.fast_accept_threshold,
            "fast_depth_threshold": self.fast_depth_threshold,
            "slow_accept_threshold": self.slow_accept_threshold,
            "slow_depth_threshold": self.slow_depth_threshold,
            "enter_fast_votes": self.enter_fast_votes,
            "enter_slow_votes": self.enter_slow_votes,
            "fast_exit_accept_threshold": self.fast_exit_accept_threshold,
            "slow_exit_accept_threshold": self.slow_exit_accept_threshold,
            "last_accepted_tokens": self.last_accepted_tokens,
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
