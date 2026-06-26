from collections import deque
from dataclasses import dataclass
from typing import Optional

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
    reusable_excluded_policies = {
        "deepest_multi",
        "hybrid_sequence",
        "hybrid_sequence_multi_position",
        "sequence_depth",
    }
    should_reuse = proactive_hit and (
        proactive_type == "included"
        or path_policy in reusable_excluded_policies
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
    """Per-request draft depth control.

    The controller can run the older score rule, the guarded state machine,
    or a probe-then-score hybrid. The hybrid spends a few early cycles
    measuring request-local acceptance before locking obviously easy or hard
    requests to a long or short depth.
    """

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
        controller: str = "state",
        initial_depth: int,
        min_depth: int,
        max_depth: int,
        increase_streak: int = 2,
        decrease_streak: int = 2,
        high_score: float = 2.0,
        low_penalty: float = 1.0,
        increase_score_threshold: float = 3.0,
        decrease_score_threshold: float = 3.0,
        protect_window: int = 5,
        protect_avg_accepted_depth: float = 2.0,
        neutral_score_decay: float = 0.8,
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
        if controller not in {"score", "state", "probe_score"}:
            raise ValueError(
                "controller must be one of: score, state, probe_score"
            )
        for name, value in [
            ("increase_streak", increase_streak),
            ("decrease_streak", decrease_streak),
            ("protect_window", protect_window),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in [
            ("high_score", high_score),
            ("low_penalty", low_penalty),
            ("increase_score_threshold", increase_score_threshold),
            ("decrease_score_threshold", decrease_score_threshold),
        ]:
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if protect_avg_accepted_depth < 0.0:
            raise ValueError("protect_avg_accepted_depth must be non-negative")
        if not 0.0 <= neutral_score_decay <= 1.0:
            raise ValueError("neutral_score_decay must be in [0, 1]")
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

        self.controller = controller
        self.initial_depth = initial_depth
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.increase_streak = increase_streak
        self.decrease_streak = decrease_streak
        self.high_score = high_score
        self.low_penalty = low_penalty
        self.increase_score_threshold = increase_score_threshold
        self.decrease_score_threshold = decrease_score_threshold
        self.protect_window = protect_window
        self.protect_avg_accepted_depth = protect_avg_accepted_depth
        self.neutral_score_decay = neutral_score_decay
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
        self.score = 0.0
        self.high_streak = 0
        self.low_streak = 0
        self.lock_state: Optional[str] = None
        self.probe_cycles = min(3, state_window_size)
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
        self.feature_names = [
            "current_depth",
            "state_index",
            "recent_accept_mean",
            "recent_accepted_depth_mean",
            "fast_votes",
            "slow_votes",
            "very_slow_votes",
            "score",
            "lock_index",
            "context_ratio",
        ]

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

    def _recent_depth_mean(self, window: int) -> Optional[float]:
        if not self.history:
            return None
        items = list(self.history)[-window:]
        return sum(depth for _tokens, depth in items) / len(items)

    def _update_score_controller(
        self,
        selected_depth: int,
        accepted_depth: int,
    ) -> None:
        if accepted_depth >= max(0, selected_depth - 1):
            self.score += self.high_score
            self.high_streak += 1
            self.low_streak = 0
        elif accepted_depth <= 1:
            self.score -= self.low_penalty
            self.low_streak += 1
            self.high_streak = 0
        else:
            self.score *= self.neutral_score_decay
            self.high_streak = 0
            self.low_streak = 0

        if (
            self.score >= self.increase_score_threshold
            or self.high_streak >= self.increase_streak
        ):
            self.current_depth = min(self.max_depth, self.current_depth + 1)
            self.score = 0.0
            self.high_streak = 0
            return

        if not (
            self.score <= -self.decrease_score_threshold
            or self.low_streak >= self.decrease_streak
        ):
            return

        recent_depth = self._recent_depth_mean(self.protect_window)
        if (
            recent_depth is not None
            and recent_depth >= self.protect_avg_accepted_depth
        ):
            self.score = -self.low_penalty
            self.low_streak = 0
            return

        self.current_depth = max(self.min_depth, self.current_depth - 1)
        self.score = 0.0
        self.low_streak = 0

    def _classify_probe(self) -> None:
        items = list(self.history)[-self.probe_cycles :]
        depths = [depth for _tokens, depth in items]
        high_votes = sum(
            depth >= self.fast_depth_threshold for depth in depths
        )
        low_votes = sum(
            depth <= self.very_slow_depth_threshold for depth in depths
        )

        if high_votes >= self.enter_fast_votes:
            self.lock_state = "fast"
            self._set_state("fast")
        elif low_votes >= self.enter_very_slow_votes:
            self.lock_state = "very_slow"
            self._set_state("very_slow")
        else:
            self.lock_state = None
            self._set_state("mid")
            self.score = 0.0

    def _update_probe_score_controller(
        self,
        decision: InitialDraftDecision,
        accepted_depth: int,
    ) -> None:
        if self.cycles <= self.probe_cycles:
            if self.cycles == self.probe_cycles:
                self._classify_probe()
            return

        recent_accept, _recent_depth = self._recent_means()
        if self.lock_state == "fast":
            if (
                recent_accept is not None
                and recent_accept < self.fast_exit_accept_threshold
            ):
                self.lock_state = None
                self._set_state("mid")
                self.score = 0.0
            return

        if self.lock_state == "very_slow":
            if (
                recent_accept is not None
                and recent_accept > self.very_slow_exit_accept_threshold
            ) or self._has_recent_depth(1):
                self.lock_state = None
                self._set_state("slow")
                self.score = 0.0
            return

        self._update_score_controller(decision.depth, accepted_depth)

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
            float(self.score),
            float(self._STATE_TO_INDEX.get(self.lock_state or "mid", 2)),
            min(1.0, max(0.0, context_ratio)),
        ]

    def select_depth(self, context_ratio: float) -> InitialDraftDecision:
        self.cycles += 1
        reason = f"local_{self.controller}"
        if (
            self.controller == "probe_score"
            and self.cycles <= self.probe_cycles
        ):
            reason = "local_probe"
        return InitialDraftDecision(
            depth=self.current_depth,
            reason=reason,
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
        if self.controller == "state":
            self._transition_state()
        elif self.controller == "score":
            self._update_score_controller(depth, accepted_depth)
        else:
            self._update_probe_score_controller(decision, accepted_depth)
        return reward

    def stats(self) -> dict:
        recent_accept, recent_depth = self._recent_means()
        return {
            "cycles": self.cycles,
            "current_depth": self.current_depth,
            "initial_depth": self.initial_depth,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "controller": self.controller,
            "state": self.state,
            "state_index": self._STATE_TO_INDEX[self.state],
            "score": self.score,
            "high_streak": self.high_streak,
            "low_streak": self.low_streak,
            "lock_state": self.lock_state,
            "probe_cycles": self.probe_cycles,
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
