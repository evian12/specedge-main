import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Optional


def percentile(values: list[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def mean(values: list[float]) -> Optional[float]:
    return fmean(values) if values else None


def format_value(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def load_records(data_dir: Path) -> list[dict]:
    records = []
    for path in sorted(
        data_dir.glob("network_ar_client_[0-9]*.jsonl")
    ):
        with path.open() as file:
            records.extend(
                json.loads(line)
                for line in file
                if line.strip()
            )
    if not records:
        raise ValueError(
            f"No network autoregressive client records in {data_dir}"
        )
    return records


def summarize(data_dir: Path) -> dict:
    records = load_records(data_dir)
    ttft = [float(record["ttft_ms"]) for record in records]
    tpot = [float(record["tpot_ms"]) for record in records]
    end_to_end = [
        float(record["end_to_end_ms"]) for record in records
    ]
    decode = [
        float(value)
        for record in records
        for value in record["server_decode_ms"]
    ]
    model_decode = [
        float(value)
        for record in records
        for value in record.get(
            "server_model_decode_ms",
            record["server_decode_ms"],
        )
    ]
    simulated_decode_latency = [
        float(record.get("simulated_decode_latency_ms", 0.0))
        for record in records
    ]
    delivery = [
        float(value)
        for record in records
        for value in record["delivery_overhead_ms"]
    ]
    tokens = sum(int(record["generated_tokens"]) for record in records)
    return {
        "experiment": str(data_dir),
        "requests": len(records),
        "tokens": tokens,
        "tokens_per_second": tokens * 1000 / sum(end_to_end),
        "ttft_mean": mean(ttft),
        "ttft_p50": percentile(ttft, 0.50),
        "ttft_p95": percentile(ttft, 0.95),
        "tpot_mean": mean(tpot),
        "tpot_p50": percentile(tpot, 0.50),
        "tpot_p95": percentile(tpot, 0.95),
        "request_mean": mean(end_to_end),
        "request_p95": percentile(end_to_end, 0.95),
        "server_prefill_mean": mean(
            [
                float(record["server_prefill_ms"])
                for record in records
            ]
        ),
        "server_decode_mean": mean(decode),
        "server_model_decode_mean": mean(model_decode),
        "simulated_decode_latency_mean": mean(simulated_decode_latency),
        "delivery_overhead_mean": mean(delivery),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        nargs="+",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    headers = [
        "experiment",
        "requests",
        "tokens",
        "tok/s",
        "TTFT mean",
        "TTFT p50",
        "TTFT p95",
        "TPOT mean",
        "TPOT p50",
        "TPOT p95",
        "request mean",
        "request p95",
        "server prefill",
        "server decode",
        "model decode",
        "sim decode",
        "delivery",
    ]
    rows = []
    for data_dir in args.data:
        summary = summarize(data_dir)
        rows.append(
            [
                summary["experiment"],
                str(summary["requests"]),
                str(summary["tokens"]),
                format_value(summary["tokens_per_second"]),
                format_value(summary["ttft_mean"]),
                format_value(summary["ttft_p50"]),
                format_value(summary["ttft_p95"]),
                format_value(summary["tpot_mean"]),
                format_value(summary["tpot_p50"]),
                format_value(summary["tpot_p95"]),
                format_value(summary["request_mean"]),
                format_value(summary["request_p95"]),
                format_value(summary["server_prefill_mean"]),
                format_value(summary["server_decode_mean"]),
                format_value(summary["server_model_decode_mean"]),
                format_value(summary["simulated_decode_latency_mean"]),
                format_value(summary["delivery_overhead_mean"]),
            ]
        )

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


if __name__ == "__main__":
    main()
