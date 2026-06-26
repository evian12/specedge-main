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


def _parse_label(label: str) -> dict[str, object]:
    if label.startswith("loadaware_constant_"):
        rest = label.removeprefix("loadaware_constant_")
        load, scheme = rest.rsplit("_", 1)
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
    if label == "loadaware_step_adaptive":
        return {"experiment": "step", "scheme": "adaptive"}
    if label == "loadaware_bursty_adaptive":
        return {"experiment": "bursty", "scheme": "adaptive"}
    return {"experiment": "unknown", "scheme": label}


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
            adaptive = group.get("adaptive")
            ar_tps = _float(ar["fg tok/s"]) if ar else 0.0
            original_tps = _float(original["fg tok/s"]) if original else 0.0
            optimized_tps = _float(optimized["fg tok/s"]) if optimized else 0.0
            adaptive_tps = _float(adaptive["fg tok/s"]) if adaptive else 0.0
            table_rows.append(
                [
                    _fmt(load, 1),
                    _fmt(ar_tps),
                    _fmt(original_tps),
                    _fmt(optimized_tps),
                    _fmt(adaptive_tps),
                    _fmt(adaptive_tps / ar_tps if ar_tps else 0.0, 3),
                    _fmt(adaptive_tps / original_tps if original_tps else 0.0, 3),
                    _fmt(_float(adaptive["server ms"]) if adaptive else 0.0),
                    _fmt(_float(adaptive["queue ms"]) if adaptive else 0.0),
                    _fmt(_float(adaptive["spec %"]) if adaptive else 0.0),
                    _fmt(_float(adaptive["ar %"]) if adaptive else 0.0),
                    str(int(_float(adaptive["switch"]))) if adaptive else "-",
                ]
            )
        lines.append(
            _markdown_table(
                [
                    "load",
                    "AR tok/s",
                    "Original tok/s",
                    "Optimized tok/s",
                    "Adaptive tok/s",
                    "Adaptive/AR",
                    "Adaptive/Original",
                    "Adaptive server ms",
                    "Adaptive queue ms",
                    "Adaptive SpecEdge %",
                    "Adaptive AR %",
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
                    _fmt(_float(row["fg tok/s"])),
                    _fmt(_float(row["lat/tok"])),
                    _fmt(_float(row["server ms"])),
                    _fmt(_float(row["queue ms"])),
                    _fmt(_float(row["spec %"])),
                    _fmt(_float(row["ar %"])),
                    str(int(_float(row["switch"]))),
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
                    _fmt(_float(row["fg tok/s"])),
                    _fmt(_float(row["lat/tok"])),
                    _fmt(_float(row["server ms"])),
                    _fmt(_float(row["queue ms"])),
                    _fmt(_float(row["spec %"])),
                    _fmt(_float(row["ar %"])),
                    str(int(_float(row["switch"]))),
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
        best = max(rows, key=lambda row: _float(row["fg tok/s"]))
        worst_latency = max(rows, key=lambda row: _float(row["lat/tok"]))
        lines.extend(
            [
                "## Quick Read",
                "",
                f"- 最高前台吞吐: `{best['label']}`，`{_fmt(_float(best['fg tok/s']))} tok/s`。",
                f"- 最高 token 延迟: `{worst_latency['label']}`，`{_fmt(_float(worst_latency['lat/tok']))} ms/token`。",
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
