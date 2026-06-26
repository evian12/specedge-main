import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Optional


def _mean(values: list[float]) -> Optional[float]:
    return fmean(values) if values else None


def _format(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _format_percent(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.1f}"


def _range(values: list[float]) -> Optional[str]:
    if not values:
        return None
    return f"{min(values):.2f}-{max(values):.2f}"


def _load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as file:
            for line_no, line in enumerate(file, start=1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        print(
                            f"warning: skipping malformed JSONL line "
                            f"{path}:{line_no}: {exc}",
                            file=sys.stderr,
                        )
    return records


def _client_records(data_dir: Path) -> list[dict[str, Any]]:
    return [
        record
        for record in _load_jsonl(sorted(data_dir.glob("client_[0-9]*.jsonl")))
        if "target" in record and "num_accepted_tokens" in record
    ]


def _server_records(data_dir: Path) -> list[dict[str, Any]]:
    return [
        record
        for record in _load_jsonl(sorted(data_dir.glob("server.jsonl")))
        if "server_step" in record
    ]


def _record_mode(record: dict[str, Any]) -> str:
    mode = record.get("mode")
    if mode in {"ar", "specedge"}:
        return mode
    target_mode = record["target"].get("proactive_execution", {}).get("mode")
    return "ar" if target_mode == "ar" else "specedge"


def summarize(data_dir: Path, label: Optional[str] = None) -> dict[str, Any]:
    clients = _client_records(data_dir)
    if not clients:
        raise ValueError(f"No runtime client records found in {data_dir}")
    servers = _server_records(data_dir)
    cycle_ms = [
        float(record["draft"]["end_to_end"])
        + float(record["target"]["end_to_end"])
        for record in clients
    ]
    tokens = [int(record["num_accepted_tokens"]) for record in clients]
    total_tokens = sum(tokens)
    total_time_ms = sum(cycle_ms)
    modes = [_record_mode(record) for record in clients]
    mode_switch_count = sum(
        int(prev != cur) for prev, cur in zip(modes, modes[1:])
    )
    logged_switches = [
        int(record.get("adaptive", {}).get("mode_switch_count", 0))
        for record in clients
        if record.get("adaptive", {}).get("mode_switch_count") is not None
    ]
    if logged_switches:
        mode_switch_count = max(mode_switch_count, max(logged_switches))

    ar_tokens = sum(
        token for token, mode in zip(tokens, modes) if mode == "ar"
    )
    specedge_tokens = total_tokens - ar_tokens
    response_times = [
        float(record["target"].get("proactive_execution", {}).get("server_response_ms"))
        for record in clients
        if record["target"].get("proactive_execution", {}).get("server_response_ms")
        is not None
    ]
    queue_wait_times = [
        float(record["target"].get("proactive_execution", {}).get("queue_wait_ms"))
        for record in clients
        if record["target"].get("proactive_execution", {}).get("queue_wait_ms")
        is not None
    ]
    server_compute_times = [
        float(record["target"].get("proactive_execution", {}).get("server_compute_ms"))
        for record in clients
        if record["target"].get("proactive_execution", {}).get("server_compute_ms")
        is not None
    ]
    estimated_server_times = [
        float(record["adaptive"]["estimated_server_time_ms"])
        for record in clients
        if record.get("adaptive", {}).get("estimated_server_time_ms")
        is not None
    ]
    switch_thresholds = [
        float(record["adaptive"]["switch_threshold_ms"])
        for record in clients
        if record.get("adaptive", {}).get("switch_threshold_ms")
        is not None
    ]
    switch_threshold_ms = (
        switch_thresholds[-1] if switch_thresholds else None
    )
    ema_specedge_decisions = [
        float(value >= switch_threshold_ms)
        for value in estimated_server_times
        if switch_threshold_ms is not None
    ]
    server_steps = [record["server_step"] for record in servers]
    background_tokens = (
        max(
            int(step.get("background_completed_tokens", 0))
            for step in server_steps
        )
        if server_steps
        else 0
    )
    background_loads = {
        str(step.get("background_load"))
        for step in server_steps
        if step.get("background_load") is not None
    }
    return {
        "experiment": str(data_dir),
        "label": label or data_dir.name,
        "requests": len({record["req_idx"] for record in clients}),
        "cycles": len(clients),
        "foreground_tokens": total_tokens,
        "background_tokens": background_tokens,
        "total_time_ms": total_time_ms,
        "tokens_per_second": total_tokens * 1000 / total_time_ms,
        "system_tokens_per_second": (
            (total_tokens + background_tokens) * 1000 / total_time_ms
        ),
        "average_latency_per_token": total_time_ms / total_tokens,
        "average_server_response_time": _mean(response_times),
        "average_queue_wait_time": _mean(queue_wait_times),
        "average_server_compute_time": _mean(server_compute_times),
        "estimated_server_time_mean": _mean(estimated_server_times),
        "estimated_server_time_range": _range(estimated_server_times),
        "switch_threshold_ms": switch_threshold_ms,
        "ema_specedge_decision_ratio": _mean(ema_specedge_decisions),
        "average_acceptance_length": _mean([float(token) for token in tokens]),
        "mode_switch_count": mode_switch_count,
        "specedge_ratio": specedge_tokens / max(1, total_tokens),
        "ar_ratio": ar_tokens / max(1, total_tokens),
        "background_load": ",".join(sorted(background_loads)) if background_loads else "-",
        "server_batch_size": _mean(
            [float(step.get("batch_size", 0)) for step in server_steps]
        ),
        "server_queue_length": _mean(
            [float(step.get("queue_length", 0)) for step in server_steps]
        ),
        "pending_prefill_count": _mean(
            [float(step.get("pending_prefill_count", 0)) for step in server_steps]
        ),
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "label",
        "requests",
        "cycles",
        "fg tok/s",
        "sys tok/s",
        "lat/tok",
        "server ms",
        "queue ms",
        "compute ms",
        "ema ms",
        "ema range",
        "thr",
        "ema spec %",
        "accept",
        "switch",
        "spec %",
        "ar %",
        "bg",
        "batch",
        "queue",
    ]
    values = []
    for row in rows:
        values.append(
            [
                row["label"],
                str(row["requests"]),
                str(row["cycles"]),
                _format(row["tokens_per_second"]),
                _format(row["system_tokens_per_second"]),
                _format(row["average_latency_per_token"]),
                _format(row["average_server_response_time"]),
                _format(row["average_queue_wait_time"]),
                _format(row["average_server_compute_time"]),
                _format(row["estimated_server_time_mean"]),
                row["estimated_server_time_range"] or "-",
                _format(row["switch_threshold_ms"]),
                _format_percent(row["ema_specedge_decision_ratio"]),
                _format(row["average_acceptance_length"]),
                str(row["mode_switch_count"]),
                _format_percent(row["specedge_ratio"]),
                _format_percent(row["ar_ratio"]),
                row["background_load"],
                _format(row["server_batch_size"]),
                _format(row["server_queue_length"]),
            ]
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in values:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--timeline", type=Path, default=None)
    args = parser.parse_args()

    labels = args.labels.split(",") if args.labels else [None] * len(args.data)
    if len(labels) != len(args.data):
        raise ValueError("--labels length must match --data length")
    rows = [summarize(path, label) for path, label in zip(args.data, labels)]
    _print_table(rows)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if args.timeline is not None:
        args.timeline.parent.mkdir(parents=True, exist_ok=True)
        with args.timeline.open("w") as file:
            for path, label in zip(args.data, labels):
                records = _client_records(path)
                for record in records:
                    target = record.get("target", {})
                    proactive = target.get("proactive_execution", {})
                    adaptive = record.get("adaptive", {})
                    row = {
                        "timestamp": record.get("timestamp"),
                        "experiment": str(path),
                        "label": label or path.name,
                        "client_idx": record.get("client_idx"),
                        "req_idx": record.get("req_idx"),
                        "step_idx": record.get("step_idx"),
                        "mode": _record_mode(record),
                        "estimated_server_time": adaptive.get(
                            "estimated_server_time_ms"
                        ),
                        "server_response_ms": proactive.get(
                            "server_response_ms"
                        ),
                        "queue_wait_ms": proactive.get("queue_wait_ms"),
                        "server_compute_ms": proactive.get(
                            "server_compute_ms"
                        ),
                        "client_cycle_ms": (
                            float(record.get("draft", {}).get("end_to_end", 0.0))
                            + float(target.get("end_to_end", 0.0))
                        ),
                        "batch_size": proactive.get("batch_size"),
                        "queue_length": proactive.get("queue_length"),
                        "background_arrival_rate": proactive.get(
                            "background_arrival_rate"
                        ),
                        "accepted_length": record.get("num_accepted_tokens"),
                    }
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
