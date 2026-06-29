import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Optional


def _mean(values: list[float]) -> Optional[float]:
    return fmean(values) if values else None


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open() as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def _parse_time(value: Any) -> Optional[float]:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _server_steps(data_dir: Path) -> list[dict[str, Any]]:
    return [
        record["server_step"]
        for record in _load_jsonl(data_dir / "server.jsonl")
        if "server_step" in record
    ]


def _background_tokens(steps: list[dict[str, Any]]) -> int:
    return max([int(step.get("background_completed_tokens", 0)) for step in steps] or [0])


def _server_mean(steps: list[dict[str, Any]], key: str) -> Optional[float]:
    return _mean([float(step[key]) for step in steps if step.get(key) is not None])


def _summarize_specedge(data_dir: Path, label: str) -> dict[str, Any]:
    records = [
        record
        for path in sorted(data_dir.glob("client_[0-9]*.jsonl"))
        for record in _load_jsonl(path)
        if "target" in record and "num_accepted_tokens" in record
    ]
    if not records:
        raise ValueError(f"No SpecEdge client records found in {data_dir}")
    steps = _server_steps(data_dir)
    tokens = sum(int(record["num_accepted_tokens"]) for record in records)
    total_time_ms = sum(
        float(record.get("draft", {}).get("end_to_end", 0.0))
        + float(record.get("target", {}).get("end_to_end", 0.0))
        for record in records
    )
    modes = [
        record.get("mode")
        if record.get("mode") in {"ar", "specedge"}
        else (
            "ar"
            if record["target"].get("proactive_execution", {}).get("mode")
            in {"ar", "ar_stream"}
            else "specedge"
        )
        for record in records
    ]
    ar_tokens = sum(
        int(record["num_accepted_tokens"])
        for record, mode in zip(records, modes)
        if mode == "ar"
    )
    switch_count = sum(
        1 for previous, current in zip(modes, modes[1:]) if previous != current
    )
    logged_switches = [
        int(record.get("adaptive", {}).get("mode_switch_count", 0))
        for record in records
        if record.get("adaptive", {}).get("mode_switch_count") is not None
    ]
    if logged_switches:
        switch_count = max(switch_count, max(logged_switches))
    proactive = [
        record["target"].get("proactive_execution", {})
        for record in records
    ]
    server_response = [
        float(item["server_response_ms"])
        for item in proactive
        if item.get("server_response_ms") is not None
    ]
    queue_wait = [
        float(item["queue_wait_ms"])
        for item in proactive
        if item.get("queue_wait_ms") is not None
    ]
    compute = [
        float(item["server_compute_ms"])
        for item in proactive
        if item.get("server_compute_ms") is not None
    ]
    ema = [
        float(record["adaptive"]["estimated_server_time_ms"])
        for record in records
        if record.get("adaptive", {}).get("estimated_server_time_ms") is not None
    ]
    ar_cost = [
        float(record["adaptive"]["ar_ms_per_token_ema"])
        for record in records
        if record.get("adaptive", {}).get("ar_ms_per_token_ema") is not None
    ]
    spec_cost = [
        float(record["adaptive"]["pred_specedge_ms_per_token"])
        for record in records
        if record.get("adaptive", {}).get("pred_specedge_ms_per_token") is not None
    ]
    bg_tokens = _background_tokens(steps)
    return {
        "label": label,
        "mode": "specedge_runtime",
        "requests": len({record.get("req_idx") for record in records}),
        "cycles": len(records),
        "foreground_tokens": tokens,
        "background_tokens": bg_tokens,
        "total_time_ms": total_time_ms,
        "tokens_per_second": tokens * 1000 / total_time_ms,
        "system_tokens_per_second": (tokens + bg_tokens) * 1000 / total_time_ms,
        "average_latency_per_token": total_time_ms / tokens,
        "average_server_response_time": _mean(server_response),
        "average_queue_wait_time": _mean(queue_wait),
        "average_server_compute_time": _mean(compute),
        "estimated_server_time_mean": _mean(ema),
        "ar_ms_per_token_ema_mean": _mean(ar_cost),
        "pred_specedge_ms_per_token_mean": _mean(spec_cost),
        "average_acceptance_length": _mean(
            [float(record["num_accepted_tokens"]) for record in records]
        ),
        "mode_switch_count": switch_count,
        "specedge_ratio": 1.0 - ar_tokens / max(1, tokens),
        "ar_ratio": ar_tokens / max(1, tokens),
        "server_batch_size": _server_mean(steps, "batch_size"),
        "server_queue_length": _server_mean(steps, "queue_length"),
        "background_load": ",".join(
            sorted(
                {
                    str(step.get("background_load"))
                    for step in steps
                    if step.get("background_load") is not None
                }
            )
        ),
    }


def _summarize_ar(data_dir: Path, label: str) -> dict[str, Any]:
    records = [
        record
        for path in sorted(data_dir.glob("network_ar_client_[0-9]*.jsonl"))
        for record in _load_jsonl(path)
        if "generated_tokens" in record
    ]
    if not records:
        raise ValueError(f"No streaming AR records found in {data_dir}")
    steps = _server_steps(data_dir)
    tokens = sum(int(record["generated_tokens"]) for record in records)
    total_time_ms = sum(float(record["end_to_end_ms"]) for record in records)
    server_response = [
        float(value)
        for record in records
        for value in record.get("server_response_ms", [])
    ]
    queue_wait = [
        float(value)
        for record in records
        for value in record.get("queue_wait_ms", [])
    ]
    compute = [
        float(value)
        for record in records
        for value in record.get("server_compute_ms", [])
    ]
    bg_tokens = _background_tokens(steps)
    return {
        "label": label,
        "mode": "streaming_ar",
        "requests": len(records),
        "cycles": tokens,
        "foreground_tokens": tokens,
        "background_tokens": bg_tokens,
        "total_time_ms": total_time_ms,
        "tokens_per_second": tokens * 1000 / total_time_ms,
        "system_tokens_per_second": (tokens + bg_tokens) * 1000 / total_time_ms,
        "average_latency_per_token": total_time_ms / tokens,
        "average_server_response_time": _mean(server_response),
        "average_queue_wait_time": _mean(queue_wait),
        "average_server_compute_time": _mean(compute),
        "estimated_server_time_mean": None,
        "ar_ms_per_token_ema_mean": None,
        "pred_specedge_ms_per_token_mean": None,
        "average_acceptance_length": 1.0,
        "mode_switch_count": 0,
        "specedge_ratio": 0.0,
        "ar_ratio": 1.0,
        "server_batch_size": _server_mean(steps, "batch_size"),
        "server_queue_length": _server_mean(steps, "queue_length"),
        "background_load": ",".join(
            sorted(
                {
                    str(step.get("background_load"))
                    for step in steps
                    if step.get("background_load") is not None
                }
            )
        ),
    }


def summarize(data_dir: Path, label: str) -> dict[str, Any]:
    if list(data_dir.glob("network_ar_client_[0-9]*.jsonl")):
        return _summarize_ar(data_dir, label)
    return _summarize_specedge(data_dir, label)


def write_timeline(data_dirs: list[Path], labels: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for data_dir, label in zip(data_dirs, labels):
            for client_path in sorted(data_dir.glob("client_[0-9]*.jsonl")):
                for record in _load_jsonl(client_path):
                    if "target" not in record or "num_accepted_tokens" not in record:
                        continue
                    proactive = record["target"].get("proactive_execution", {})
                    row = {
                        "timestamp": record.get("timestamp"),
                        "label": label,
                        "mode": record.get("mode"),
                        "estimated_server_time": record.get("adaptive", {}).get(
                            "estimated_server_time_ms"
                        ),
                        "queue_wait_ms_ema": record.get("adaptive", {}).get(
                            "queue_wait_ms_ema"
                        ),
                        "ar_ms_per_token_ema": record.get("adaptive", {}).get(
                            "ar_ms_per_token_ema"
                        ),
                        "specedge_cycle_ms_ema": record.get("adaptive", {}).get(
                            "specedge_cycle_ms_ema"
                        ),
                        "accepted_tokens_ema": record.get("adaptive", {}).get(
                            "accepted_tokens_ema"
                        ),
                        "pred_ar_ms_per_token": record.get("adaptive", {}).get(
                            "pred_ar_ms_per_token"
                        ),
                        "pred_specedge_ms_per_token": record.get("adaptive", {}).get(
                            "pred_specedge_ms_per_token"
                        ),
                        "selected_mode": record.get("adaptive", {}).get(
                            "selected_mode"
                        ),
                        "switch_reason": record.get("adaptive", {}).get(
                            "switch_reason"
                        ),
                        "tokens_since_last_switch": record.get("adaptive", {}).get(
                            "tokens_since_last_switch"
                        ),
                        "cycles_since_last_switch": record.get("adaptive", {}).get(
                            "cycles_since_last_switch"
                        ),
                        "server_response_ms": proactive.get("server_response_ms"),
                        "queue_wait_ms": proactive.get("queue_wait_ms"),
                        "server_compute_ms": proactive.get("server_compute_ms"),
                        "client_cycle_ms": (
                            float(record.get("draft", {}).get("end_to_end", 0.0))
                            + float(record.get("target", {}).get("end_to_end", 0.0))
                        ),
                        "batch_size": proactive.get("batch_size"),
                        "queue_length": proactive.get("queue_length"),
                        "background_arrival_rate": proactive.get(
                            "background_arrival_rate"
                        ),
                        "accepted_length": record.get("num_accepted_tokens"),
                    }
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
            for client_path in sorted(data_dir.glob("network_ar_client_[0-9]*.jsonl")):
                for record in _load_jsonl(client_path):
                    arrivals = record.get("arrival_ms", [])
                    for index, arrival_ms in enumerate(arrivals):
                        row = {
                            "timestamp": record.get("timestamp"),
                            "label": label,
                            "mode": "ar",
                            "estimated_server_time": None,
                            "queue_wait_ms_ema": None,
                            "ar_ms_per_token_ema": None,
                            "specedge_cycle_ms_ema": None,
                            "accepted_tokens_ema": None,
                            "pred_ar_ms_per_token": None,
                            "pred_specedge_ms_per_token": None,
                            "selected_mode": "ar",
                            "switch_reason": None,
                            "tokens_since_last_switch": None,
                            "cycles_since_last_switch": None,
                            "server_response_ms": (
                                record.get("server_response_ms", [None] * len(arrivals))[
                                    index
                                ]
                            ),
                            "queue_wait_ms": (
                                record.get("queue_wait_ms", [None] * len(arrivals))[
                                    index
                                ]
                            ),
                            "server_compute_ms": (
                                record.get("server_compute_ms", [None] * len(arrivals))[
                                    index
                                ]
                            ),
                            "client_cycle_ms": arrival_ms,
                            "batch_size": (
                                record.get("batch_size", [None] * len(arrivals))[index]
                            ),
                            "queue_length": (
                                record.get("queue_length", [None] * len(arrivals))[index]
                            ),
                            "background_arrival_rate": (
                                record.get(
                                    "background_arrival_rate",
                                    [None] * len(arrivals),
                                )[index]
                            ),
                            "accepted_length": 1,
                        }
                        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _scheme_suffix(label: str) -> tuple[str, str]:
    suffixes = [
        "adaptive_predictor",
        "adaptive_threshold",
        "adaptive",
        "optimized",
        "original",
        "ar",
    ]
    for suffix in suffixes:
        marker = f"_{suffix}"
        if label.endswith(marker):
            return label[: -len(marker)], suffix
    return label, "unknown"


def _dominant_mode(modes: list[str]) -> str:
    if not modes:
        return "-"
    counts = {mode: modes.count(mode) for mode in set(modes)}
    if len(counts) > 1:
        ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return f"mixed:{ordered[0][0]}"
    return modes[0]


def _foreground_events(data_dir: Path, label: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for client_path in sorted(data_dir.glob("client_[0-9]*.jsonl")):
        for record in _load_jsonl(client_path):
            if "target" not in record or "num_accepted_tokens" not in record:
                continue
            timestamp = _parse_time(record.get("timestamp"))
            if timestamp is None:
                continue
            proactive = record["target"].get("proactive_execution", {})
            mode = record.get("mode")
            if mode not in {"ar", "specedge"}:
                mode = (
                    "ar"
                    if proactive.get("mode") in {"ar", "ar_stream"}
                    else "specedge"
                )
            adaptive = record.get("adaptive", {})
            events.append(
                {
                    "timestamp": timestamp,
                    "label": label,
                    "mode": mode,
                    "tokens": int(record.get("num_accepted_tokens", 0)),
                    "queue_wait_ms": proactive.get("queue_wait_ms"),
                    "server_response_ms": proactive.get("server_response_ms"),
                    "batch_size": proactive.get("batch_size"),
                    "queue_length": proactive.get("queue_length"),
                    "pred_ar_ms_per_token": adaptive.get("pred_ar_ms_per_token"),
                    "pred_specedge_ms_per_token": adaptive.get(
                        "pred_specedge_ms_per_token"
                    ),
                }
            )
    for client_path in sorted(data_dir.glob("network_ar_client_[0-9]*.jsonl")):
        for record in _load_jsonl(client_path):
            end_ts = _parse_time(record.get("timestamp"))
            if end_ts is None:
                continue
            end_to_end_ms = float(record.get("end_to_end_ms", 0.0))
            start_ts = end_ts - end_to_end_ms / 1000
            arrivals = record.get("arrival_ms", [])
            generated = int(record.get("generated_tokens", len(arrivals)))
            for index in range(generated):
                arrival_ms = (
                    float(arrivals[index])
                    if index < len(arrivals)
                    else (index + 1) * end_to_end_ms / max(1, generated)
                )
                timestamp = start_ts + arrival_ms / 1000
                events.append(
                    {
                        "timestamp": timestamp,
                        "label": label,
                        "mode": "ar",
                        "tokens": 1,
                        "queue_wait_ms": (
                            record.get("queue_wait_ms", [None] * generated)[index]
                            if index < len(record.get("queue_wait_ms", []))
                            else None
                        ),
                        "server_response_ms": (
                            record.get("server_response_ms", [None] * generated)[index]
                            if index < len(record.get("server_response_ms", []))
                            else None
                        ),
                        "batch_size": (
                            record.get("batch_size", [None] * generated)[index]
                            if index < len(record.get("batch_size", []))
                            else None
                        ),
                        "queue_length": (
                            record.get("queue_length", [None] * generated)[index]
                            if index < len(record.get("queue_length", []))
                            else None
                        ),
                        "pred_ar_ms_per_token": None,
                        "pred_specedge_ms_per_token": None,
                    }
                )
    return events


def _server_window_stats(
    data_dir: Path,
    start_ts: float,
    window_size_s: float,
) -> dict[int, dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {}
    previous_background: Optional[int] = None
    for step in _server_steps(data_dir):
        timestamp = _parse_time(step.get("timestamp"))
        if timestamp is None:
            continue
        window = int((timestamp - start_ts) // window_size_s)
        if window < 0:
            continue
        entry = stats.setdefault(
            window,
            {
                "background_tokens": 0,
                "queue_wait_ms": [],
                "server_response_ms": [],
                "batch_size": [],
                "queue_length": [],
            },
        )
        current_background = int(step.get("background_completed_tokens", 0))
        if previous_background is not None:
            entry["background_tokens"] += max(0, current_background - previous_background)
        previous_background = current_background
        for key in ("queue_wait_ms", "server_response_ms", "batch_size", "queue_length"):
            if step.get(key) is not None:
                entry[key].append(float(step[key]))
    return stats


def write_window_summary(
    data_dirs: list[Path],
    labels: list[str],
    path: Path,
    window_size_s: float,
) -> None:
    rows: list[dict[str, Any]] = []
    for data_dir, label in zip(data_dirs, labels):
        events = _foreground_events(data_dir, label)
        timestamps = [event["timestamp"] for event in events]
        if not timestamps:
            continue
        start_ts = min(timestamps)
        end_ts = max(timestamps)
        total_windows = int((end_ts - start_ts) // window_size_s) + 1
        server_stats = _server_window_stats(data_dir, start_ts, window_size_s)
        for window in range(total_windows):
            window_start = window * window_size_s
            window_end = window_start + window_size_s
            window_events = [
                event
                for event in events
                if window_start <= event["timestamp"] - start_ts < window_end
            ]
            foreground_tokens = sum(int(event["tokens"]) for event in window_events)
            ar_tokens = sum(
                int(event["tokens"]) for event in window_events if event["mode"] == "ar"
            )
            modes = [str(event["mode"]) for event in window_events]
            stats = server_stats.get(window, {})
            background_tokens = int(stats.get("background_tokens", 0))
            queue_values = [
                float(event["queue_wait_ms"])
                for event in window_events
                if event.get("queue_wait_ms") is not None
            ] or stats.get("queue_wait_ms", [])
            response_values = [
                float(event["server_response_ms"])
                for event in window_events
                if event.get("server_response_ms") is not None
            ] or stats.get("server_response_ms", [])
            batch_values = [
                float(event["batch_size"])
                for event in window_events
                if event.get("batch_size") is not None
            ] or stats.get("batch_size", [])
            queue_len_values = [
                float(event["queue_length"])
                for event in window_events
                if event.get("queue_length") is not None
            ] or stats.get("queue_length", [])
            pred_ar_values = [
                float(event["pred_ar_ms_per_token"])
                for event in window_events
                if event.get("pred_ar_ms_per_token") is not None
            ]
            pred_spec_values = [
                float(event["pred_specedge_ms_per_token"])
                for event in window_events
                if event.get("pred_specedge_ms_per_token") is not None
            ]
            row = {
                "experiment": label,
                "scenario": _scheme_suffix(label)[0],
                "scheme": _scheme_suffix(label)[1],
                "window_start_s": window_start,
                "window_end_s": window_end,
                "mode": _dominant_mode(modes),
                "foreground_tokens": foreground_tokens,
                "background_tokens": background_tokens,
                "foreground_tok_s": foreground_tokens / window_size_s,
                "system_tok_s": (foreground_tokens + background_tokens) / window_size_s,
                "avg_queue_wait_ms": _mean(queue_values),
                "avg_server_response_ms": _mean(response_values),
                "avg_batch_size": _mean(batch_values),
                "avg_queue_length": _mean(queue_len_values),
                "specedge_ratio": 1.0 - ar_tokens / max(1, foreground_tokens),
                "ar_ratio": ar_tokens / max(1, foreground_tokens),
                "pred_ar_ms_per_token": _mean(pred_ar_values),
                "pred_specedge_ms_per_token": _mean(pred_spec_values),
                "oracle_best_mode": None,
                "oracle_best_tok_s": None,
                "adaptive_gap": None,
            }
            rows.append(row)

    by_window = {
        (row["scenario"], row["window_start_s"], row["scheme"]): row
        for row in rows
    }
    for row in rows:
        ar = by_window.get((row["scenario"], row["window_start_s"], "ar"))
        optimized = by_window.get(
            (row["scenario"], row["window_start_s"], "optimized")
        )
        if not ar or not optimized:
            continue
        ar_tps = float(ar["foreground_tok_s"])
        opt_tps = float(optimized["foreground_tok_s"])
        if ar_tps >= opt_tps:
            row["oracle_best_mode"] = "ar"
            row["oracle_best_tok_s"] = ar_tps
        else:
            row["oracle_best_mode"] = "optimized"
            row["oracle_best_tok_s"] = opt_tps
        row["adaptive_gap"] = row["oracle_best_tok_s"] - float(row["foreground_tok_s"])

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print(rows: list[dict[str, Any]]) -> None:
    headers = [
        "label",
        "mode",
        "requests",
        "fg tok/s",
        "sys tok/s",
        "lat/tok",
        "server ms",
        "queue ms",
        "compute ms",
        "switch",
        "spec %",
        "ar %",
        "batch",
        "queue",
    ]
    values = []
    for row in rows:
        values.append(
            [
                str(row["label"]),
                str(row["mode"]),
                str(row["requests"]),
                _fmt(row["tokens_per_second"]),
                _fmt(row["system_tokens_per_second"]),
                _fmt(row["average_latency_per_token"]),
                _fmt(row["average_server_response_time"]),
                _fmt(row["average_queue_wait_time"]),
                _fmt(row["average_server_compute_time"]),
                str(row["mode_switch_count"]),
                _fmt(row["specedge_ratio"] * 100),
                _fmt(row["ar_ratio"] * 100),
                _fmt(row["server_batch_size"]),
                _fmt(row["server_queue_length"]),
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
    parser.add_argument("--summary-csv", type=Path, default=Path("results/load_aware_summary.csv"))
    parser.add_argument("--timeline", type=Path, default=Path("results/load_aware_timeline.jsonl"))
    parser.add_argument(
        "--window-summary",
        type=Path,
        default=Path("results/load_aware_window_summary.csv"),
    )
    parser.add_argument("--window-size-s", type=float, default=10.0)
    args = parser.parse_args()

    labels = args.labels.split(",") if args.labels else [path.name for path in args.data]
    if len(labels) != len(args.data):
        raise ValueError("--labels length must match --data length")
    rows = [summarize(path, label) for path, label in zip(args.data, labels)]
    _print(rows)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    preferred = list(rows[0].keys())
    fieldnames = preferred + sorted(
        {key for row in rows for key in row.keys()} - set(preferred)
    )
    with args.summary_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_timeline(args.data, labels, args.timeline)
    write_window_summary(args.data, labels, args.window_summary, args.window_size_s)


if __name__ == "__main__":
    main()
