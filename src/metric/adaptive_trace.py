import argparse
import json
from pathlib import Path
from typing import Any


def _load_client_records(data_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("client_[0-9]*.jsonl")):
        with path.open() as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "adaptive" in record and "target" in record:
                    records.append(record)
    return records


def _mode(record: dict[str, Any]) -> str:
    mode = record.get("mode")
    if mode in {"ar", "specedge"}:
        return mode
    target_mode = record["target"].get("proactive_execution", {}).get("mode")
    return "ar" if target_mode == "ar" else "specedge"


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=Path, required=True)
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Print every Nth runtime record.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=80,
        help="Maximum rows to print.",
    )
    args = parser.parse_args()

    records = _load_client_records(args.data)
    if not records:
        print(f"No adaptive client records found in {args.data}")
        return

    headers = [
        "idx",
        "req",
        "step",
        "mode",
        "accepted",
        "server ms",
        "queue ms",
        "compute ms",
        "client ms",
        "ema ms",
        "threshold",
        "next by ema",
        "switches",
    ]
    rows: list[list[str]] = []
    for index, record in enumerate(records):
        if index % max(1, args.every) != 0:
            continue
        adaptive = record.get("adaptive", {})
        target = record["target"].get("proactive_execution", {})
        ema = adaptive.get("estimated_server_time_ms")
        threshold = adaptive.get("switch_threshold_ms")
        next_mode = "-"
        if ema is not None and threshold is not None:
            next_mode = "specedge" if ema >= threshold else "ar"
        rows.append(
            [
                str(index),
                str(record.get("req_idx", "-")),
                str(record.get("step_idx", "-")),
                _mode(record),
                str(record.get("num_accepted_tokens", "-")),
                _fmt(target.get("server_response_ms")),
                _fmt(target.get("queue_wait_ms")),
                _fmt(target.get("server_compute_ms")),
                _fmt(target.get("response_received_ms")),
                _fmt(ema),
                _fmt(threshold),
                next_mode,
                str(adaptive.get("mode_switch_count", 0)),
            ]
        )
        if len(rows) >= args.limit:
            break

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


if __name__ == "__main__":
    main()
