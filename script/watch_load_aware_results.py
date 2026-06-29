#!/usr/bin/env python3
import argparse
import csv
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _is_running(pattern: str) -> bool:
    result = _run(["pgrep", "-af", pattern], check=False)
    lines = [
        line
        for line in result.stdout.splitlines()
        if pattern in line and "watch_load_aware_results.py" not in line
    ]
    return bool(lines)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _value(row: dict[str, str] | None, *keys: str, default: str = "0") -> str:
    if not row:
        return default
    for key in keys:
        if key in row and row[key] not in {None, ""}:
            return row[key]
    return default


def _tps(row: dict[str, str] | None) -> float:
    return _float(_value(row, "fg tok/s", "tokens_per_second"))


def _latency_per_token(row: dict[str, str] | None) -> float:
    return _float(_value(row, "lat/tok", "average_latency_per_token"))


def _server_ms(row: dict[str, str] | None) -> float:
    return _float(_value(row, "server ms", "average_server_response_time"))


def _queue_ms(row: dict[str, str] | None) -> float:
    return _float(_value(row, "queue ms", "average_queue_wait_time"))


def _spec_ratio_pct(row: dict[str, str] | None) -> float:
    value = _float(_value(row, "spec %", "specedge_ratio"))
    return value if value > 1.0 else value * 100.0


def _ar_ratio_pct(row: dict[str, str] | None) -> float:
    value = _float(_value(row, "ar %", "ar_ratio"))
    return value if value > 1.0 else value * 100.0


def _switches(row: dict[str, str] | None) -> int:
    return int(_float(_value(row, "switch", "mode_switch_count")))


def _parse_label(label: str) -> dict[str, object]:
    if label.startswith("loadaware_constant_"):
        rest = label.removeprefix("loadaware_constant_")
        load, scheme = _split_load_scheme(rest)
        return {
            "experiment": "constant",
            "load": float(load.replace("p", ".")),
            "scheme": scheme,
        }
    if label.startswith("loadaware_threshold_"):
        rest = label.removeprefix("loadaware_threshold_")
        rate_part, threshold_part = rest.split("_")
        return {
            "experiment": "threshold",
            "load": float(rate_part.removeprefix("r").replace("p", ".")),
            "threshold": int(threshold_part.removeprefix("t")),
            "scheme": "adaptive",
        }
    if label.startswith("loadaware_step_"):
        return {
            "experiment": "step",
            "scheme": label.removeprefix("loadaware_step_"),
        }
    if label.startswith("loadaware_bursty_"):
        return {
            "experiment": "bursty",
            "scheme": label.removeprefix("loadaware_bursty_"),
        }
    return {"experiment": "unknown", "scheme": label}


def _split_load_scheme(rest: str) -> tuple[str, str]:
    known_schemes = [
        "adaptive_predictor",
        "adaptive_threshold",
        "optimized",
        "original",
        "adaptive",
        "ar",
    ]
    for scheme in known_schemes:
        suffix = f"_{scheme}"
        if rest.endswith(suffix):
            return rest[: -len(suffix)], scheme
    return rest.rsplit("_", 1)


def _fmt(value: object, digits: int = 2) -> str:
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _generate_report(summary_csv: Path, timeline_jsonl: Path, report_md: Path) -> None:
    rows = _load_rows(summary_csv)
    parsed: list[dict[str, object]] = []
    for row in rows:
        info = _parse_label(row["label"])
        parsed.append({**row, **info})

    lines: list[str] = [
        "# Load-aware SpecEdge 实验自动报告",
        "",
        f"- Summary CSV: `{summary_csv}`",
        f"- Timeline JSONL: `{timeline_jsonl}`",
        f"- 实验数量: {len(rows)}",
        "",
    ]

    constant = [row for row in parsed if row.get("experiment") == "constant"]
    if constant:
        lines.extend(["## Constant Load 主实验", ""])
        by_load: dict[float, dict[str, dict[str, object]]] = {}
        for row in constant:
            by_load.setdefault(float(row["load"]), {})[str(row["scheme"])] = row
        table_rows: list[list[object]] = []
        for load in sorted(by_load):
            group = by_load[load]
            ar = group.get("ar")
            original = group.get("original")
            optimized = group.get("optimized")
            adaptive_threshold = group.get("adaptive_threshold") or group.get("adaptive")
            adaptive_predictor = group.get("adaptive_predictor")
            ar_tps = _tps(ar)
            original_tps = _tps(original)
            optimized_tps = _tps(optimized)
            threshold_tps = _tps(adaptive_threshold)
            predictor_tps = _tps(adaptive_predictor)
            oracle = max(ar_tps, optimized_tps)
            table_rows.append(
                [
                    _fmt(load, 1),
                    _fmt(ar_tps),
                    _fmt(original_tps),
                    _fmt(optimized_tps),
                    _fmt(threshold_tps),
                    _fmt(predictor_tps),
                    _fmt(oracle),
                    _fmt((oracle - predictor_tps) / oracle if oracle else 0.0, 3),
                    _fmt(_server_ms(adaptive_predictor)),
                    _fmt(_queue_ms(adaptive_predictor)),
                    _fmt(_spec_ratio_pct(adaptive_predictor)),
                    _fmt(_ar_ratio_pct(adaptive_predictor)),
                    str(_switches(adaptive_predictor)) if adaptive_predictor else "-",
                ]
            )
        lines.append(
            _markdown_table(
                [
                    "load",
                    "AR tok/s",
                    "Original tok/s",
                    "Optimized tok/s",
                    "Threshold tok/s",
                    "Predictor tok/s",
                    "Oracle tok/s",
                    "Predictor rel gap",
                    "Predictor server ms",
                    "Predictor queue ms",
                    "Predictor SpecEdge %",
                    "Predictor AR %",
                    "switches",
                ],
                table_rows,
            )
        )
        lines.append("")

    threshold = [row for row in parsed if row.get("experiment") == "threshold"]
    if threshold:
        lines.extend(["## Threshold Sensitivity", ""])
        table_rows = []
        for row in sorted(threshold, key=lambda item: (float(item["load"]), int(item["threshold"]))):
            table_rows.append(
                [
                    _fmt(row["load"], 1),
                    row["threshold"],
                    _fmt(_tps(row)),
                    _fmt(_latency_per_token(row)),
                    _fmt(_server_ms(row)),
                    _fmt(_queue_ms(row)),
                    _fmt(_spec_ratio_pct(row)),
                    _fmt(_ar_ratio_pct(row)),
                    str(_switches(row)),
                ]
            )
        lines.append(
            _markdown_table(
                [
                    "load",
                    "threshold",
                    "tok/s",
                    "lat/tok",
                    "server ms",
                    "queue ms",
                    "SpecEdge %",
                    "AR %",
                    "switches",
                ],
                table_rows,
            )
        )
        lines.append("")

    dynamic = [row for row in parsed if row.get("experiment") in {"step", "bursty"}]
    if dynamic:
        lines.extend(["## Dynamic Load", ""])
        table_rows = []
        for row in dynamic:
            table_rows.append(
                [
                    row["label"],
                    _fmt(_tps(row)),
                    _fmt(_latency_per_token(row)),
                    _fmt(_server_ms(row)),
                    _fmt(_queue_ms(row)),
                    _fmt(_spec_ratio_pct(row)),
                    _fmt(_ar_ratio_pct(row)),
                    str(_switches(row)),
                ]
            )
        lines.append(
            _markdown_table(
                [
                    "experiment",
                    "tok/s",
                    "lat/tok",
                    "server ms",
                    "queue ms",
                    "SpecEdge %",
                    "AR %",
                    "switches",
                ],
                table_rows,
            )
        )
        lines.append("")

    if rows:
        best = max(rows, key=_tps)
        worst_latency = max(rows, key=_latency_per_token)
        lines.extend(
            [
                "## Quick Read",
                "",
                f"- 最高前台吞吐: `{best['label']}`，`{_fmt(_tps(best))} tok/s`。",
                f"- 最高 token 延迟: `{worst_latency['label']}`，`{_fmt(_latency_per_token(worst_latency))} ms/token`。",
                "- 解释时优先看 `server ms = queue ms + compute ms`；Adaptive 的 EMA 只应该跟随这个 server-side 指标，而不是 client cycle。",
                "",
            ]
        )

    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines), encoding="utf-8")


def _summarize(
    result_root: Path,
    summary_csv: Path,
    timeline_jsonl: Path,
    report_md: Path,
    expected_count: int,
) -> None:
    data_dirs = sorted(path for path in result_root.glob("loadaware_*") if path.is_dir())
    if not data_dirs:
        raise RuntimeError(f"No loadaware result directories found under {result_root}")
    if expected_count > 0 and len(data_dirs) < expected_count:
        raise RuntimeError(
            f"Only found {len(data_dirs)} load-aware result directories under "
            f"{result_root}; expected at least {expected_count}. "
            "The matrix probably stopped early. Inspect result/load_aware_runs/*.log."
        )
    labels = [path.name for path in data_dirs]
    cmd = [
        "python",
        "src/metric/load_aware_runtime.py",
        "-d",
        *[str(path) for path in data_dirs],
        "--labels",
        ",".join(labels),
        "--summary-csv",
        str(summary_csv),
        "--timeline",
        str(timeline_jsonl),
    ]
    print("Running:", " ".join(cmd), flush=True)
    result = _run(cmd)
    print(result.stdout, flush=True)
    _generate_report(summary_csv, timeline_jsonl, report_md)
    print(f"Report written to {report_md}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-pattern", default="run_load_aware_matrix.py")
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--timeout-s", type=float, default=0.0)
    parser.add_argument("--result-root", default="result/4090_jetson")
    parser.add_argument("--summary-csv", default="results/load_aware_summary.csv")
    parser.add_argument("--timeline", default="results/load_aware_timeline.jsonl")
    parser.add_argument("--report-md", default="results/load_aware_report.md")
    parser.add_argument("--expected-count", type=int, default=44)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    start = time.monotonic()
    if not args.no_wait:
        while _is_running(args.runner_pattern):
            elapsed = time.monotonic() - start
            if args.timeout_s > 0 and elapsed >= args.timeout_s:
                raise TimeoutError(
                    f"Timed out after {elapsed:.1f}s waiting for {args.runner_pattern}"
                )
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Waiting for {args.runner_pattern}... elapsed={elapsed:.0f}s",
                flush=True,
            )
            time.sleep(args.poll_interval)

    _summarize(
        ROOT / args.result_root,
        ROOT / args.summary_csv,
        ROOT / args.timeline,
        ROOT / args.report_md,
        args.expected_count,
    )


if __name__ == "__main__":
    main()
