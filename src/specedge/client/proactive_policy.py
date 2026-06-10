from dataclasses import dataclass
from typing import Optional

"""
    可用重叠时间 =
        预计验证响应时间
        - proactive 初始化时间
        - 安全余量

    proactive 深度 =
        floor(可用重叠时间 / 每层耗时)
"""
"""
    每一层前判断：
        预计下一层耗时 + 安全余量 <= 预计剩余响应时间
    保守剩余时间
        = 响应平均值
        - k × 响应误差
        - 当前已经消耗的时间

    保守执行成本
        = 层平均耗时
        + k × 层误差
        + safety margin

"""

@dataclass
class EwmaEstimate:
    alpha: float
    mean: Optional[float] = None
    error: Optional[float] = None

    def observe(self, value: float) -> None:
        if self.mean is None:
            self.mean = value
            self.error = 0.0
            return

        residual = abs(value - self.mean)
        self.mean = self.alpha * value + (1.0 - self.alpha) * self.mean
        self.error = self.alpha * residual + (1.0 - self.alpha) * (
            self.error or 0.0
        )

    def upper_bound(self, uncertainty_scale: float) -> Optional[float]:
        if self.mean is None:
            return None
        return self.mean + uncertainty_scale * (self.error or 0.0)

    def lower_bound(self, uncertainty_scale: float) -> Optional[float]:
        if self.mean is None:
            return None
        return max(
            0.0,
            self.mean - uncertainty_scale * (self.error or 0.0),
        )

    def stats(self) -> dict[str, Optional[float]]:
        return {
            "mean_ms": self.mean,
            "error_ms": self.error,
        }


@dataclass
class DeadlineDecision:
    allowed: bool
    reason: Optional[str]
    remaining_ms: Optional[float]
    predicted_cost_ms: Optional[float]


class AdaptiveProactivePolicy:
    def __init__(
        self,
        max_depth: int,
        ewma_alpha: float,
        min_alignment_rate: float,
        warmup_cycles: int,
        exploration_interval: int,
        safety_margin_ms: float,
        uncertainty_scale: float,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if not 0.0 <= min_alignment_rate <= 1.0:
            raise ValueError("min_alignment_rate must be in [0, 1]")
        if warmup_cycles < 0:
            raise ValueError("warmup_cycles must be non-negative")
        if safety_margin_ms < 0.0:
            raise ValueError("safety_margin_ms must be non-negative")
        if uncertainty_scale < 0.0:
            raise ValueError("uncertainty_scale must be non-negative")

        self._max_depth = max_depth
        self._alpha = ewma_alpha
        self._min_alignment_rate = min_alignment_rate
        self._warmup_cycles = warmup_cycles
        self._exploration_interval = exploration_interval
        self._safety_margin_ms = safety_margin_ms
        self._uncertainty_scale = uncertainty_scale

        self.cycles = 0
        self._current_warmup = False
        self.response = EwmaEstimate(ewma_alpha)
        self.setup = EwmaEstimate(ewma_alpha)
        self.layer_wall: dict[int, EwmaEstimate] = {}
        self.layer_gpu: dict[int, EwmaEstimate] = {}
        self.alignment_rate_ewma: Optional[float] = None

    def _update_ewma(self, previous: Optional[float], value: float) -> float:
        if previous is None:
            return value
        return self._alpha * value + (1.0 - self._alpha) * previous

    def begin_cycle(self) -> tuple[int, Optional[str]]:
        self.cycles += 1
        self._current_warmup = self.cycles <= self._warmup_cycles
        if self._current_warmup:
            return self._max_depth, "warmup"

        if (
            self.alignment_rate_ewma is not None
            and self.alignment_rate_ewma < self._min_alignment_rate
        ):
            is_exploration = (
                self._exploration_interval > 0
                and self.cycles % self._exploration_interval == 0
            )
            if not is_exploration:
                return 0, "low_alignment_rate"
            return self._max_depth, "alignment_exploration"

        return self._max_depth, None

    def _response_remaining_ms(
        self, request_elapsed_ms: float
    ) -> Optional[float]:
        response_budget_ms = self.response.lower_bound(
            self._uncertainty_scale
        )
        if response_budget_ms is None:
            return None
        return max(0.0, response_budget_ms - request_elapsed_ms)

    def can_start_setup(self, request_elapsed_ms: float) -> DeadlineDecision:
        if self._current_warmup:
            return DeadlineDecision(True, "warmup", None, None)

        remaining_ms = self._response_remaining_ms(request_elapsed_ms)
        setup_ms = self.setup.upper_bound(self._uncertainty_scale)
        if remaining_ms is None or setup_ms is None:
            return DeadlineDecision(
                True,
                "insufficient_timing_history",
                remaining_ms,
                setup_ms,
            )

        required_ms = setup_ms + self._safety_margin_ms
        return DeadlineDecision(
            allowed=required_ms <= remaining_ms,
            reason=None if required_ms <= remaining_ms else "setup_deadline",
            remaining_ms=remaining_ms,
            predicted_cost_ms=setup_ms,
        )

    def can_start_layer(
        self,
        layer_index: int,
        request_elapsed_ms: float,
    ) -> DeadlineDecision:
        if self._current_warmup:
            return DeadlineDecision(True, "warmup", None, None)

        remaining_ms = self._response_remaining_ms(request_elapsed_ms)
        estimate = self.layer_wall.get(layer_index)
        layer_ms = (
            estimate.upper_bound(self._uncertainty_scale)
            if estimate is not None
            else None
        )
        if remaining_ms is None or layer_ms is None:
            return DeadlineDecision(
                True,
                "unseen_layer_exploration",
                remaining_ms,
                layer_ms,
            )

        required_ms = layer_ms + self._safety_margin_ms
        return DeadlineDecision(
            allowed=required_ms <= remaining_ms,
            reason=None if required_ms <= remaining_ms else "layer_deadline",
            remaining_ms=remaining_ms,
            predicted_cost_ms=layer_ms,
        )

    def observe_setup(self, elapsed_ms: float) -> None:
        self.setup.observe(elapsed_ms)

    def observe_step(
        self,
        layer_index: int,
        wall_ms: float,
        gpu_ms: Optional[float],
    ) -> None:
        self.layer_wall.setdefault(
            layer_index, EwmaEstimate(self._alpha)
        ).observe(wall_ms)
        if gpu_ms is not None:
            self.layer_gpu.setdefault(
                layer_index, EwmaEstimate(self._alpha)
            ).observe(gpu_ms)

    def observe_cycle(
        self,
        response_ms: float,
        aligned: bool,
        proactive_executed: bool,
    ) -> None:
        self.response.observe(response_ms)
        if proactive_executed:
            self.alignment_rate_ewma = self._update_ewma(
                self.alignment_rate_ewma, float(aligned)
            )

    def stats(self) -> dict:
        return {
            "response": self.response.stats(),
            "setup": self.setup.stats(),
            "layer_wall": {
                str(index): estimate.stats()
                for index, estimate in sorted(self.layer_wall.items())
            },
            "layer_gpu": {
                str(index): estimate.stats()
                for index, estimate in sorted(self.layer_gpu.items())
            },
            "alignment_rate_ewma": self.alignment_rate_ewma,
            "uncertainty_scale": self._uncertainty_scale,
        }
