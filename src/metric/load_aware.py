import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml

SPECEDGE_SRC = Path(__file__).resolve().parents[1]
if str(SPECEDGE_SRC) not in sys.path:
    sys.path.insert(0, str(SPECEDGE_SRC))

from metric import network_autoregressive, proactive


DEFAULT_LATENCIES = [50.0, 100.0, 150.0, 200.0]
DEFAULT_THRESHOLD_MS = 60.0


@dataclass(frozen=True)
class StrategyMetrics:
    tokens_per_second: float
    alignment_rate: Optional[float] = None
    reuse_depth: Optional[float] = None
    cycle_ms: Optional[float] = None


@dataclass(frozen=True)
class LoadProfilePoint:
    latency_ms: float
    autoregressive: StrategyMetrics
    original_specedge: StrategyMetrics
    response_only: StrategyMetrics
    selected_mode: str
    selected_tokens_per_second: float


@dataclass
class BackgroundRequest:
    remaining_tokens: int


@dataclass
class LoadState:
    batch_size: int
    queue_length: int
    prefill_count: int
    base_decode_latency_ms: float


@dataclass
class SimulationConfig:
    foreground_requests: int
    foreground_tokens: int
    decision_window: int
    threshold_ms: float
    base_decode_latency_ms: float
    max_batch_size: int
    background_arrival_rate: float
    background_min_tokens: int
    background_max_tokens: int
    active_penalty_ms: float
    queue_penalty_ms: float
    prefill_penalty_ms: float
    ar_to_specedge_prefill_ms: float
    estimator_alpha: float
    background_load_label: str
    seed: int


@dataclass
class StrategySimulationResult:
    strategy: str
    foreground_tokens: int
    foreground_ms: float
    background_completed_tokens: int
    system_tokens: int
    mode_counts: dict[str, int]
    mode_token_counts: dict[str, int]
    switch_count: int
    predicted_latency_mean_ms: float
    average_latency_per_token_ms: float
    average_acceptance_length: float
    number_of_cycles: int
    background_load: str

    @property
    def foreground_tokens_per_second(self) -> float:
        return self.foreground_tokens * 1000 / self.foreground_ms

    @property
    def system_tokens_per_second(self) -> float:
        return self.system_tokens * 1000 / self.foreground_ms

    @property
    def specedge_ratio(self) -> float:
        specedge_tokens = (
            self.mode_token_counts.get("original_specedge", 0)
            + self.mode_token_counts.get("response_only", 0)
        )
        return specedge_tokens / max(1, self.foreground_tokens)

    @property
    def ar_ratio(self) -> float:
        return (
            self.mode_token_counts.get("network_ar", 0)
            / max(1, self.foreground_tokens)
        )


class ServerResponseTimeEmaEstimator:
    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("estimator_alpha must be in (0, 1]")
        self._alpha = alpha
        self.estimated_server_time: Optional[float] = None

    def estimate_or(self, fallback_ms: float) -> float:
        return (
            fallback_ms
            if self.estimated_server_time is None
            else self.estimated_server_time
        )

    def observe(self, current_server_response_time: float) -> float:
        if self.estimated_server_time is None:
            self.estimated_server_time = current_server_response_time
        else:
            self.estimated_server_time = (
                (1.0 - self._alpha) * self.estimated_server_time
                + self._alpha * current_server_response_time
            )
        return self.estimated_server_time


def _format(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _format_percent(value: Optional[float], digits: int = 1) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}"


def _format_speedup(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}x"


def _latency_label(latency_ms: float) -> str:
    return str(int(latency_ms)) if latency_ms.is_integer() else str(latency_ms)


def _parse_latencies(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one latency is required")
    return sorted(values)


def _parse_weights(raw: Optional[str]) -> dict[float, float]:
    if raw is None:
        return {}
    weights = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        latency, weight = item.split("=", maxsplit=1)
        latency_ms = float(latency.strip())
        weight_value = float(weight.strip())
        if weight_value < 0.0:
            raise ValueError("Latency weights must be non-negative")
        weights[latency_ms] = weight_value
    return weights


def _interpolate(
    points: list[LoadProfilePoint],
    latency_ms: float,
    getter: Callable[[LoadProfilePoint], Optional[float]],
) -> Optional[float]:
    ordered = sorted(points, key=lambda point: point.latency_ms)
    if not ordered:
        raise ValueError("No profile points")
    if latency_ms <= ordered[0].latency_ms:
        return getter(ordered[0])
    if latency_ms >= ordered[-1].latency_ms:
        return getter(ordered[-1])
    for left, right in zip(ordered, ordered[1:]):
        if left.latency_ms <= latency_ms <= right.latency_ms:
            left_value = getter(left)
            right_value = getter(right)
            if left_value is None or right_value is None:
                return None
            ratio = (
                (latency_ms - left.latency_ms)
                / (right.latency_ms - left.latency_ms)
            )
            return left_value + ratio * (right_value - left_value)
    raise AssertionError("unreachable interpolation state")


def _proactive_metrics(data_dir: Path) -> StrategyMetrics:
    summary = proactive.summarize(data_dir)
    return StrategyMetrics(
        tokens_per_second=float(summary["tokens_per_second"]),
        alignment_rate=float(summary["alignment_rate"]),
        reuse_depth=(
            float(summary["reused_proactive_depth"])
            if summary["reused_proactive_depth"] is not None
            else None
        ),
        cycle_ms=(
            float(summary["cycle_ms"])
            if summary["cycle_ms"] is not None
            else None
        ),
    )


def _network_ar_metrics(data_dir: Path) -> StrategyMetrics:
    summary = network_autoregressive.summarize(data_dir)
    return StrategyMetrics(
        tokens_per_second=float(summary["tokens_per_second"]),
    )


def choose_mode(latency_ms: float, threshold_ms: float) -> str:
    return "network_ar" if latency_ms <= threshold_ms else "response_only"


def predict_response_latency_ms(
    state: LoadState,
    *,
    active_penalty_ms: float,
    queue_penalty_ms: float,
    prefill_penalty_ms: float,
) -> float:
    """Predict server response/decode latency from scheduler state.

    This is the first executable version of the paper notation
    `T_hat = f(B, Q, P, Ld)`: B is active batch size, Q is queue length,
    P is newly admitted prefill count, and Ld is the base decode latency.
    """

    return (
        state.base_decode_latency_ms
        + active_penalty_ms * max(0, state.batch_size - 1)
        + queue_penalty_ms * state.queue_length
        + prefill_penalty_ms * state.prefill_count
    )


class BackgroundScheduler:
    """Small FCFS continuous-batching load model for profiling replay.

    It does not run the target model. It produces the same online signals
    used by the design: active batch size, queue length, prefill admissions,
    and a changing server response estimate.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self._config = config
        self._rng = random.Random(config.seed)
        self._queue: list[BackgroundRequest] = []
        self._active: list[BackgroundRequest] = []
        self._arrival_credit = 0.0
        self._last_prefill_count = 0
        self.completed_tokens = 0

    def _new_request(self) -> BackgroundRequest:
        return BackgroundRequest(
            remaining_tokens=self._rng.randint(
                self._config.background_min_tokens,
                self._config.background_max_tokens,
            )
        )

    def _admit(self) -> None:
        admitted = 0
        while (
            self._queue
            and len(self._active) < self._config.max_batch_size
        ):
            self._active.append(self._queue.pop(0))
            admitted += 1
        self._last_prefill_count = admitted

    def advance(self, elapsed_ms: float) -> None:
        if elapsed_ms < 0.0:
            raise ValueError("elapsed_ms must be non-negative")

        expected_arrivals = (
            self._config.background_arrival_rate * elapsed_ms / 1000
        )
        self._arrival_credit += expected_arrivals
        while self._arrival_credit >= 1.0:
            self._queue.append(self._new_request())
            self._arrival_credit -= 1.0

        self._admit()
        remaining_ms = elapsed_ms
        while remaining_ms > 0.0 and self._active:
            step_ms = (
                self._config.base_decode_latency_ms
                + self._config.active_penalty_ms
                * max(0, len(self._active) - 1)
            )
            if step_ms <= 0.0:
                raise ValueError("background decode step must be positive")
            if remaining_ms < step_ms:
                break
            remaining_ms -= step_ms
            completed = []
            for request in self._active:
                request.remaining_tokens -= 1
                self.completed_tokens += 1
                if request.remaining_tokens <= 0:
                    completed.append(request)
            if completed:
                self._active = [
                    request
                    for request in self._active
                    if request.remaining_tokens > 0
                ]
                self._admit()

    def state(self) -> LoadState:
        return LoadState(
            batch_size=len(self._active) + 1,
            queue_length=len(self._queue),
            prefill_count=self._last_prefill_count,
            base_decode_latency_ms=self._config.base_decode_latency_ms,
        )


def _strategy_tokens_per_second(
    point: LoadProfilePoint,
    strategy: str,
) -> float:
    if strategy == "network_ar":
        return point.autoregressive.tokens_per_second
    if strategy == "original_specedge":
        return point.original_specedge.tokens_per_second
    if strategy == "response_only":
        return point.response_only.tokens_per_second
    if strategy == "load_aware":
        return point.selected_tokens_per_second
    raise ValueError(f"Unknown strategy: {strategy}")


def _interpolated_tokens_per_second(
    points: list[LoadProfilePoint],
    strategy: str,
    latency_ms: float,
) -> float:
    value = _interpolate(
        points,
        latency_ms,
        lambda point: _strategy_tokens_per_second(point, strategy),
    )
    if value is None:
        raise ValueError(f"Missing tokens_per_second for {strategy}")
    return value


def _interpolated_cycle_tokens(
    points: list[LoadProfilePoint],
    strategy: str,
    latency_ms: float,
    fallback_tokens: int,
) -> int:
    if strategy == "network_ar":
        return fallback_tokens
    cycle_ms = _interpolate(
        points,
        latency_ms,
        lambda point: (
            point.original_specedge.cycle_ms
            if strategy == "original_specedge"
            else point.response_only.cycle_ms
        ),
    )
    tokens_per_second = _interpolated_tokens_per_second(
        points,
        strategy,
        latency_ms,
    )
    if cycle_ms is None:
        return fallback_tokens
    return max(1, round(tokens_per_second * cycle_ms / 1000))


def _strategy_for_safe_point(
    strategy: str,
    predicted_latency_ms: float,
    threshold_ms: float,
) -> str:
    if strategy == "load_aware":
        return choose_mode(predicted_latency_ms, threshold_ms)
    return strategy


def simulate_strategy(
    points: list[LoadProfilePoint],
    config: SimulationConfig,
    strategy: str,
) -> StrategySimulationResult:
    if strategy not in {
        "network_ar",
        "original_specedge",
        "response_only",
        "load_aware",
    }:
        raise ValueError(f"Unknown strategy: {strategy}")

    scheduler = BackgroundScheduler(config)
    foreground_tokens = 0
    foreground_ms = 0.0
    mode_counts = {"network_ar": 0, "original_specedge": 0, "response_only": 0}
    mode_token_counts = {
        "network_ar": 0,
        "original_specedge": 0,
        "response_only": 0,
    }
    switch_count = 0
    latency_observations = []
    segment_lengths = []
    previous_mode: Optional[str] = None
    estimator = ServerResponseTimeEmaEstimator(config.estimator_alpha)

    # Warm the background state with one base decode interval so the first
    # foreground decision sees a non-empty scheduler if load is configured.
    scheduler.advance(config.base_decode_latency_ms)

    for _request_idx in range(config.foreground_requests):
        remaining_tokens = config.foreground_tokens
        while remaining_tokens > 0:
            state = scheduler.state()
            current_server_response_ms = predict_response_latency_ms(
                state,
                active_penalty_ms=config.active_penalty_ms,
                queue_penalty_ms=config.queue_penalty_ms,
                prefill_penalty_ms=config.prefill_penalty_ms,
            )
            estimated_latency_ms = estimator.estimate_or(
                current_server_response_ms
            )
            latency_observations.append(current_server_response_ms)
            mode = _strategy_for_safe_point(
                strategy,
                estimated_latency_ms,
                config.threshold_ms,
            )
            mode_counts[mode] += 1
            if previous_mode is not None and previous_mode != mode:
                switch_count += 1
            switch_penalty_ms = (
                config.ar_to_specedge_prefill_ms
                if previous_mode == "network_ar" and mode == "response_only"
                else 0.0
            )
            previous_mode = mode

            safe_point_tokens = _interpolated_cycle_tokens(
                points,
                mode,
                current_server_response_ms,
                config.decision_window,
            )
            segment_tokens = min(remaining_tokens, safe_point_tokens)
            tokens_per_second = _interpolated_tokens_per_second(
                points,
                mode,
                current_server_response_ms,
            )
            elapsed_ms = segment_tokens * 1000 / tokens_per_second
            elapsed_ms += switch_penalty_ms

            foreground_tokens += segment_tokens
            mode_token_counts[mode] += segment_tokens
            segment_lengths.append(segment_tokens)
            foreground_ms += elapsed_ms
            remaining_tokens -= segment_tokens
            estimator.observe(current_server_response_ms)
            scheduler.advance(elapsed_ms)

    background_tokens = scheduler.completed_tokens
    return StrategySimulationResult(
        strategy=strategy,
        foreground_tokens=foreground_tokens,
        foreground_ms=foreground_ms,
        background_completed_tokens=background_tokens,
        system_tokens=foreground_tokens + background_tokens,
        mode_counts=mode_counts,
        mode_token_counts=mode_token_counts,
        switch_count=switch_count,
        predicted_latency_mean_ms=(
            sum(latency_observations) / len(latency_observations)
            if latency_observations
            else 0.0
        ),
        average_latency_per_token_ms=foreground_ms / max(1, foreground_tokens),
        average_acceptance_length=(
            sum(segment_lengths) / len(segment_lengths)
            if segment_lengths
            else 0.0
        ),
        number_of_cycles=sum(mode_counts.values()),
        background_load=config.background_load_label,
    )


def weighted_throughput(
    points: list[LoadProfilePoint],
    weights: dict[float, float],
    strategy: str,
) -> float:
    if not points:
        raise ValueError("No profile points")
    effective_weights = {
        point.latency_ms: weights.get(point.latency_ms, 1.0)
        for point in points
    }
    total_weight = sum(effective_weights.values())
    if total_weight <= 0.0:
        raise ValueError("At least one latency weight must be positive")
    weighted_seconds_per_token = 0.0
    for point in points:
        weight = effective_weights[point.latency_ms]
        if weight == 0.0:
            continue
        tokens_per_second = _strategy_tokens_per_second(point, strategy)
        weighted_seconds_per_token += weight / tokens_per_second
    return total_weight / weighted_seconds_per_token


def load_profile(
    result_root: Path,
    latencies_ms: list[float],
    threshold_ms: float,
    network_dir_name: Callable[[float], str] = (
        lambda latency: (
            f"network_autoregressive_decode_lat{_latency_label(latency)}"
        )
    ),
    original_dir_name: Callable[[float], str] = (
        lambda latency: (
            f"specedge_tree_original_decode_lat{_latency_label(latency)}"
        )
    ),
    response_only_dir_name: Callable[[float], str] = (
        lambda latency: (
            "specedge_tree_prob_depth_response_only_decode_"
            f"lat{_latency_label(latency)}"
        )
    ),
) -> list[LoadProfilePoint]:
    points = []
    for latency_ms in latencies_ms:
        autoregressive = _network_ar_metrics(
            result_root / network_dir_name(latency_ms)
        )
        original_specedge = _proactive_metrics(
            result_root / original_dir_name(latency_ms)
        )
        response_only = _proactive_metrics(
            result_root / response_only_dir_name(latency_ms)
        )
        selected_mode = choose_mode(latency_ms, threshold_ms)
        selected_tokens_per_second = (
            autoregressive.tokens_per_second
            if selected_mode == "network_ar"
            else response_only.tokens_per_second
        )
        points.append(
            LoadProfilePoint(
                latency_ms=latency_ms,
                autoregressive=autoregressive,
                original_specedge=original_specedge,
                response_only=response_only,
                selected_mode=selected_mode,
                selected_tokens_per_second=selected_tokens_per_second,
            )
        )
    return points


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(
        "  ".join(
            header.ljust(widths[index])
            for index, header in enumerate(headers)
        )
    )
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  ".join(
                value.ljust(widths[index])
                for index, value in enumerate(row)
            )
        )


def print_latency_table(points: list[LoadProfilePoint]) -> None:
    headers = [
        "lat ms",
        "AR tok/s",
        "SpecEdge tok/s",
        "SpecEdge align %",
        "Current tok/s",
        "Current align %",
        "Current reuse",
        "selected",
        "selected tok/s",
        "sel/AR",
        "sel/SpecEdge",
    ]
    rows = []
    for point in points:
        rows.append(
            [
                _format(point.latency_ms, 0),
                _format(point.autoregressive.tokens_per_second),
                _format(point.original_specedge.tokens_per_second),
                _format_percent(point.original_specedge.alignment_rate),
                _format(point.response_only.tokens_per_second),
                _format_percent(point.response_only.alignment_rate),
                _format(point.response_only.reuse_depth),
                point.selected_mode,
                _format(point.selected_tokens_per_second),
                _format_speedup(
                    point.selected_tokens_per_second
                    / point.autoregressive.tokens_per_second
                ),
                _format_speedup(
                    point.selected_tokens_per_second
                    / point.original_specedge.tokens_per_second
                ),
            ]
        )
    _print_table(headers, rows)


def print_system_table(
    points: list[LoadProfilePoint],
    weights: dict[float, float],
) -> None:
    strategies = [
        ("network_ar", "Always network AR"),
        ("original_specedge", "Always original SpecEdge"),
        ("response_only", "Always current response-only"),
        ("load_aware", "Load-aware threshold"),
    ]
    load_aware_tps = weighted_throughput(points, weights, "load_aware")
    headers = [
        "strategy",
        "system tok/s",
        "vs AR",
        "vs SpecEdge",
        "vs current",
    ]
    baseline_ar = weighted_throughput(points, weights, "network_ar")
    baseline_specedge = weighted_throughput(
        points,
        weights,
        "original_specedge",
    )
    baseline_current = weighted_throughput(points, weights, "response_only")
    rows = []
    for strategy, label in strategies:
        tokens_per_second = weighted_throughput(points, weights, strategy)
        rows.append(
            [
                label,
                _format(tokens_per_second),
                _format_speedup(tokens_per_second / baseline_ar),
                _format_speedup(tokens_per_second / baseline_specedge),
                _format_speedup(tokens_per_second / baseline_current),
            ]
        )
    _print_table(headers, rows)
    print(
        "\nLoad-aware selected system throughput: "
        f"{load_aware_tps:.2f} tok/s"
    )


def print_dynamic_simulation_table(
    points: list[LoadProfilePoint],
    config: SimulationConfig,
) -> None:
    results = [
        simulate_strategy(points, config, "network_ar"),
        simulate_strategy(points, config, "original_specedge"),
        simulate_strategy(points, config, "response_only"),
        simulate_strategy(points, config, "load_aware"),
    ]
    baseline_ar = results[0]
    baseline_specedge = results[1]
    baseline_current = results[2]
    headers = [
        "strategy",
        "jetson tok/s",
        "system tok/s",
        "bg tokens",
        "mean T_hat",
        "AR safe pts",
        "SpecEdge pts",
        "Current pts",
        "switches",
        "sys/AR",
        "sys/SpecEdge",
        "sys/current",
    ]
    labels = {
        "network_ar": "Always network AR",
        "original_specedge": "Always original SpecEdge",
        "response_only": "Always current response-only",
        "load_aware": "Load-aware safe switch",
    }
    rows = []
    for result in results:
        rows.append(
            [
                labels[result.strategy],
                _format(result.foreground_tokens_per_second),
                _format(result.system_tokens_per_second),
                str(result.background_completed_tokens),
                _format(result.predicted_latency_mean_ms),
                str(result.mode_counts.get("network_ar", 0)),
                str(result.mode_counts.get("original_specedge", 0)),
                str(result.mode_counts.get("response_only", 0)),
                str(result.switch_count),
                _format_speedup(
                    result.system_tokens_per_second
                    / baseline_ar.system_tokens_per_second
                ),
                _format_speedup(
                    result.system_tokens_per_second
                    / baseline_specedge.system_tokens_per_second
                ),
                _format_speedup(
                    result.system_tokens_per_second
                    / baseline_current.system_tokens_per_second
                ),
            ]
        )
    _print_table(headers, rows)


def _background_load_to_rate(load: str) -> float:
    normalized = load.strip().lower()
    mapping = {
        "0": 0.0,
        "none": 0.0,
        "low": 0.1,
        "medium": 0.5,
        "high": 1.0,
    }
    if normalized in mapping:
        return mapping[normalized]
    return float(normalized)


def _parse_background_loads(raw: str) -> list[tuple[str, float]]:
    loads = []
    for item in raw.split(","):
        label = item.strip()
        if not label:
            continue
        loads.append((label, _background_load_to_rate(label)))
    if not loads:
        raise ValueError("At least one background load is required")
    return loads


def _csv_row(result: StrategySimulationResult) -> dict[str, str | float | int]:
    mode_label = {
        "network_ar": "ar",
        "original_specedge": "specedge",
        "load_aware": "adaptive",
        "response_only": "current_specedge",
    }.get(result.strategy, result.strategy)
    return {
        "mode": mode_label,
        "background_load": result.background_load,
        "total_time": result.foreground_ms,
        "tokens_per_second": result.foreground_tokens_per_second,
        "system_tokens_per_second": result.system_tokens_per_second,
        "average_latency_per_token": result.average_latency_per_token_ms,
        "average_server_response_time": result.predicted_latency_mean_ms,
        "average_acceptance_length": result.average_acceptance_length,
        "number_of_cycles": result.number_of_cycles,
        "mode_switch_count": result.switch_count,
        "specedge_ratio": result.specedge_ratio,
        "ar_ratio": result.ar_ratio,
        "foreground_tokens": result.foreground_tokens,
        "background_tokens": result.background_completed_tokens,
    }


def run_experiment_matrix(
    points: list[LoadProfilePoint],
    base_config: SimulationConfig,
    background_loads: list[tuple[str, float]],
) -> list[StrategySimulationResult]:
    results = []
    for label, arrival_rate in background_loads:
        config = SimulationConfig(
            foreground_requests=base_config.foreground_requests,
            foreground_tokens=base_config.foreground_tokens,
            decision_window=base_config.decision_window,
            threshold_ms=base_config.threshold_ms,
            base_decode_latency_ms=base_config.base_decode_latency_ms,
            max_batch_size=base_config.max_batch_size,
            background_arrival_rate=arrival_rate,
            background_min_tokens=base_config.background_min_tokens,
            background_max_tokens=base_config.background_max_tokens,
            active_penalty_ms=base_config.active_penalty_ms,
            queue_penalty_ms=base_config.queue_penalty_ms,
            prefill_penalty_ms=base_config.prefill_penalty_ms,
            ar_to_specedge_prefill_ms=(
                base_config.ar_to_specedge_prefill_ms
            ),
            estimator_alpha=base_config.estimator_alpha,
            background_load_label=label,
            seed=base_config.seed,
        )
        for strategy in ["network_ar", "original_specedge", "load_aware"]:
            results.append(simulate_strategy(points, config, strategy))
    return results


def write_results_csv(
    results: list[StrategySimulationResult],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "background_load",
        "total_time",
        "tokens_per_second",
        "system_tokens_per_second",
        "average_latency_per_token",
        "average_server_response_time",
        "average_acceptance_length",
        "number_of_cycles",
        "mode_switch_count",
        "specedge_ratio",
        "ar_ratio",
        "foreground_tokens",
        "background_tokens",
    ]
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(_csv_row(result))


def _apply_yaml_config(args: argparse.Namespace) -> None:
    if args.config is None:
        return
    with args.config.open() as file:
        config = yaml.safe_load(file) or {}

    base = config.get("base", {})
    simulation = config.get("simulation", {})

    def set_if_present(attr: str, source: dict, key: str, transform=None) -> None:
        if key not in source:
            return
        value = source[key]
        if transform is not None:
            value = transform(value)
        setattr(args, attr, value)

    set_if_present("result_root", base, "result_root", Path)
    set_if_present("output_csv", base, "output_csv", Path)
    if "latencies" in base:
        args.latencies = ",".join(str(value) for value in base["latencies"])

    if "switch_threshold_ms" in config:
        args.threshold_ms = float(config["switch_threshold_ms"])
    if "decision_window" in config:
        args.decision_window = int(config["decision_window"])
    if "estimator_alpha" in config:
        args.estimator_alpha = float(config["estimator_alpha"])

    set_if_present("foreground_requests", simulation, "foreground_requests", int)
    set_if_present("foreground_tokens", simulation, "foreground_tokens", int)
    set_if_present("max_batch_size", simulation, "max_batch_size", int)
    set_if_present(
        "base_decode_latency_ms",
        simulation,
        "base_decode_latency_ms",
        float,
    )
    set_if_present("active_penalty_ms", simulation, "active_penalty_ms", float)
    set_if_present("queue_penalty_ms", simulation, "queue_penalty_ms", float)
    set_if_present("prefill_penalty_ms", simulation, "prefill_penalty_ms", float)
    set_if_present(
        "ar_to_specedge_prefill_ms",
        simulation,
        "ar_to_specedge_prefill_ms",
        float,
    )
    if "background_loads" in simulation:
        args.background_loads = ",".join(
            str(value) for value in simulation["background_loads"]
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a load-aware switch between network autoregressive "
            "decoding and the response-only SpecEdge variant from offline "
            "latency profiling results."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config for load-aware experiment parameters.",
    )
    parser.add_argument(
        "-r",
        "--result-root",
        type=Path,
        default=Path("result/4090_jetson"),
        help="Directory containing latency-profile experiment results.",
    )
    parser.add_argument(
        "--latencies",
        default=",".join(str(int(value)) for value in DEFAULT_LATENCIES),
        help="Comma-separated latency points in ms.",
    )
    parser.add_argument(
        "--threshold-ms",
        type=float,
        default=DEFAULT_THRESHOLD_MS,
        help=(
            "Use network autoregressive decoding when the estimated server "
            "response/decode latency is at or below this threshold; otherwise "
            "use the current response-only SpecEdge variant."
        ),
    )
    parser.add_argument(
        "--latency-weights",
        default=None,
        help=(
            "Optional comma-separated latency distribution, for example "
            "'50=0.4,100=0.3,150=0.2,200=0.1'. Missing latencies default to 1."
        ),
    )
    parser.add_argument(
        "--skip-dynamic-simulation",
        action="store_true",
        help="Only print offline profiling tables.",
    )
    parser.add_argument(
        "--foreground-requests",
        type=int,
        default=60,
        help="Number of foreground Jetson requests in dynamic simulation.",
    )
    parser.add_argument(
        "--foreground-tokens",
        type=int,
        default=64,
        help="Generated tokens per foreground request in simulation.",
    )
    parser.add_argument(
        "--decision-window",
        type=int,
        default=16,
        help="Safe decision window for network autoregressive mode.",
    )
    parser.add_argument(
        "--base-decode-latency-ms",
        type=float,
        default=50.0,
        help="Base server decode/response latency Ld for load prediction.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum active background batch size in FCFS simulation.",
    )
    parser.add_argument(
        "--background-arrival-rate",
        type=float,
        default=0.2,
        help="Background request arrival rate, requests per second.",
    )
    parser.add_argument(
        "--background-min-tokens",
        type=int,
        default=16,
        help="Minimum background generated tokens per request.",
    )
    parser.add_argument(
        "--background-max-tokens",
        type=int,
        default=64,
        help="Maximum background generated tokens per request.",
    )
    parser.add_argument(
        "--active-penalty-ms",
        type=float,
        default=12.0,
        help="Latency penalty per additional active batch item.",
    )
    parser.add_argument(
        "--queue-penalty-ms",
        type=float,
        default=4.0,
        help="Latency penalty per queued request.",
    )
    parser.add_argument(
        "--prefill-penalty-ms",
        type=float,
        default=20.0,
        help="Latency penalty per newly admitted prefill request.",
    )
    parser.add_argument(
        "--ar-to-specedge-prefill-ms",
        type=float,
        default=48.0,
        help=(
            "Mode switch cost for rebuilding draft KV cache when switching "
            "from network AR to SpecEdge."
        ),
    )
    parser.add_argument(
        "--estimator-alpha",
        type=float,
        default=0.2,
        help="EMA alpha for online server response time estimation.",
    )
    parser.add_argument(
        "--background-loads",
        default="0,low,medium,high",
        help=(
            "Comma-separated load labels or arrival rates for CSV matrix, "
            "for example '0,low,medium,high' or '0,0.1,0.5,1.0,2.0'."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/load_aware_results.csv"),
        help="CSV path for AR / SpecEdge / adaptive load-aware results.",
    )
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help="Do not write the experiment matrix CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for background request generation.",
    )
    args = parser.parse_args()
    _apply_yaml_config(args)

    latencies_ms = _parse_latencies(args.latencies)
    weights = _parse_weights(args.latency_weights)
    points = load_profile(
        result_root=args.result_root,
        latencies_ms=latencies_ms,
        threshold_ms=args.threshold_ms,
    )

    print(
        f"Load-aware threshold: <= {args.threshold_ms:.2f} ms -> network_ar, "
        "> threshold -> response_only\n"
    )
    print_latency_table(points)
    print()
    print_system_table(points, weights)
    if not args.skip_dynamic_simulation:
        print("\nDynamic load-aware simulation")
        print(
            "Predictor: T_hat = Ld + active_penalty*(B-1) + "
            "queue_penalty*Q + prefill_penalty*P"
        )
        simulation_config = SimulationConfig(
            foreground_requests=args.foreground_requests,
            foreground_tokens=args.foreground_tokens,
            decision_window=args.decision_window,
            threshold_ms=args.threshold_ms,
            base_decode_latency_ms=args.base_decode_latency_ms,
            max_batch_size=args.max_batch_size,
            background_arrival_rate=args.background_arrival_rate,
            background_min_tokens=args.background_min_tokens,
            background_max_tokens=args.background_max_tokens,
            active_penalty_ms=args.active_penalty_ms,
            queue_penalty_ms=args.queue_penalty_ms,
            prefill_penalty_ms=args.prefill_penalty_ms,
            ar_to_specedge_prefill_ms=args.ar_to_specedge_prefill_ms,
            estimator_alpha=args.estimator_alpha,
            background_load_label=str(args.background_arrival_rate),
            seed=args.seed,
        )
        print_dynamic_simulation_table(points, simulation_config)
        if not args.skip_csv:
            matrix_results = run_experiment_matrix(
                points,
                simulation_config,
                _parse_background_loads(args.background_loads),
            )
            write_results_csv(matrix_results, args.output_csv)
            print(f"\nWrote CSV results to {args.output_csv}")


if __name__ == "__main__":
    main()
