import argparse
import csv
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
    args = parser.parse_args()

    labels = args.labels.split(",") if args.labels else [path.name for path in args.data]
    if len(labels) != len(args.data):
        raise ValueError("--labels length must match --data length")
    rows = [summarize(path, label) for path, label in zip(args.data, labels)]
    _print(rows)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_timeline(args.data, labels, args.timeline)


if __name__ == "__main__":
    main()
