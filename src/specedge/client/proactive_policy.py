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
    分别维护：
        layer_0：均值 + 误差
        layer_1：均值 + 误差
        layer_2：均值 + 误差
"""
class AdaptiveProactivePolicy:
    def __init__(
        self,
        max_depth: int,
        ewma_alpha: float,
        min_alignment_rate: float,
        warmup_cycles: int,
        exploration_interval: int,
        safety_margin_ms: float,
    ) -> None:
        self._max_depth = max_depth
        self._alpha = ewma_alpha
        #命中率反馈阈值
        self._min_alignment_rate = min_alignment_rate
        self._warmup_cycles = warmup_cycles
        self._exploration_interval = exploration_interval
        self._safety_margin_ms = safety_margin_ms

        self.cycles = 0
        #4090验证响应时间
        self.response_ms_ewma: Optional[float] = None
        #proactive根节点选择和初始化时间
        self.setup_ms_ewma: Optional[float] = None
        #jetson每层proactive时间
        self.step_ms_ewma: Optional[float] = None
        #proactive结果命中率
        self.alignment_rate_ewma: Optional[float] = None

    def _update_ewma(self, previous: Optional[float], value: float) -> float:
        if previous is None:
            return value
        return self._alpha * value + (1.0 - self._alpha) * previous

    def choose_depth(self) -> tuple[int, Optional[str]]:
        self.cycles += 1

        if self.cycles <= self._warmup_cycles:
            return self._max_depth, "warmup"

        if (
            self.alignment_rate_ewma is not None
            and self.alignment_rate_ewma < self._min_alignment_rate
        ):
            if (
                self._exploration_interval <= 0
                or self.cycles % self._exploration_interval != 0
            ):
                return 0, "low_alignment_rate"

        if (
            self.response_ms_ewma is None
            or self.setup_ms_ewma is None
            or self.step_ms_ewma is None
        ):
            return self._max_depth, "insufficient_timing_history"

        overlap_budget_ms = max(
            0.0,
            self.response_ms_ewma
            - self._safety_margin_ms
            - self.setup_ms_ewma,
        )
        depth = int(overlap_budget_ms // max(self.step_ms_ewma, 1e-6))
        depth = min(self._max_depth, max(0, depth))

        if depth == 0 and (
            self._exploration_interval > 0
            and self.cycles % self._exploration_interval == 0
        ):
            return 1, "timing_exploration"

        return depth, None if depth > 0 else "insufficient_overlap_budget"

    def observe_step(self, elapsed_ms: float) -> None:
        self.step_ms_ewma = self._update_ewma(self.step_ms_ewma, elapsed_ms)

    def observe_setup(self, elapsed_ms: float) -> None:
        self.setup_ms_ewma = self._update_ewma(
            self.setup_ms_ewma, elapsed_ms
        )

    def observe_cycle(
        self, response_ms: float, aligned: bool, proactive_executed: bool
    ) -> None:
        self.response_ms_ewma = self._update_ewma(
            self.response_ms_ewma, response_ms
        )
        if proactive_executed:
            self.alignment_rate_ewma = self._update_ewma(
                self.alignment_rate_ewma, float(aligned)
            )

    def stats(self) -> dict[str, Optional[float]]:
        return {
            "response_ms_ewma": self.response_ms_ewma,
            "setup_ms_ewma": self.setup_ms_ewma,
            "step_ms_ewma": self.step_ms_ewma,
            "alignment_rate_ewma": self.alignment_rate_ewma,
        }
