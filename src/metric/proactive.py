import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Optional


def _mean(values: list[float]) -> Optional[float]:
    return fmean(values) if values else None


def _format(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _load_client_records(data_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(data_dir.glob("client_[0-9]*.jsonl")):
        with path.open() as file:
            for line in file:
                record = json.loads(line)
                if "req_idx" in record and "target" in record:
                    records.append(record)
    return records


def summarize(data_dir: Path) -> dict[str, Any]:
    records = _load_client_records(data_dir)
    if not records:
        raise ValueError(f"No client result records found in {data_dir}")

    details = [
        record["target"].get("proactive_execution", {})
        for record in records
    ]
    initial_drafts = [
        record["draft"].get("initial_draft", {})
        for record in records
    ]
    initial_decisions = [
        detail
        for detail in initial_drafts
        if detail.get("selection_reason") != "prefill"
    ]
    modes = {
        detail.get("mode")
        for detail in details
        if detail.get("mode") is not None
    }
    mode = ",".join(sorted(modes)) if modes else "legacy"
    path_policies = {
        detail.get("path_policy")
        for detail in details
        if detail.get("path_policy") is not None
    }
    path_policy = (
        ",".join(sorted(path_policies)) if path_policies else "single_best"
    )

    cycle_ms = [
        record["draft"]["end_to_end"] + record["target"]["end_to_end"]
        for record in records
    ]
    accepted_tokens = sum(record["num_accepted_tokens"] for record in records)
    proactive_attempts = [
        detail for detail in details if detail.get("planned_depth", 0) > 0
    ]
    layer_wall_ms = [
        float(value)
        for detail in details
        for value in detail.get("layer_wall_ms", [])
    ]
    layer_gpu_ms = [
        float(value)
        for detail in details
        for value in detail.get("layer_gpu_ms", [])
        if value is not None
    ]

    return {
        "experiment": str(data_dir),
        "mode": mode,
        "path_policy": path_policy,
        "initial_draft_mode": ",".join(
            sorted(
                {
                    detail.get("mode")
                    for detail in initial_drafts
                    if detail.get("mode") is not None
                }
            )
        )
        or "legacy",
        "cycles": len(records),
        "tokens_per_second": accepted_tokens * 1000 / sum(cycle_ms),
        "cycle_ms": _mean(cycle_ms),
        "wait_ms": _mean(
            [record["target"]["client_wait"] for record in records]
        ),
        "proactive_ms": _mean(
            [
                float(detail["elapsed_ms"])
                for detail in details
                if detail.get("elapsed_ms") is not None
            ]
        ),
        "executed_depth": _mean(
            [
                float(detail["executed_depth"])
                for detail in details
                if detail.get("executed_depth") is not None
            ]
        ),
        "deepest_leaf_count": _mean(
            [
                float(detail["deepest_leaf_count"])
                for detail in details
                if detail.get("deepest_leaf_count") is not None
            ]
        ),
        "selected_leaf_count": _mean(
            [
                float(detail["selected_leaf_count"])
                for detail in details
                if detail.get("selected_leaf_count") is not None
            ]
        ),
        "root_count": _mean(
            [
                float(detail["root_count"])
                for detail in details
                if detail.get("root_count") is not None
            ]
        ),
        "matched_stop_depth": _mean(
            [
                float(detail["matched_stop_depth"])
                for detail in details
                if detail.get("matched_stop_depth") is not None
            ]
        ),
        "matched_reused_depth": _mean(
            [
                float(detail["matched_reused_depth"])
                for detail in details
                if detail.get("matched_reused_depth") is not None
            ]
        ),
        "reused_proactive_depth": _mean(
            [
                float(detail["reused_proactive_depth"])
                for detail in initial_drafts
                if detail.get("reused_proactive_depth") is not None
            ]
        ),
        "proactive_node_count": _mean(
            [
                float(detail["proactive_node_count"])
                for detail in details
                if detail.get("proactive_node_count") is not None
            ]
        ),
        "wasted_node_count": _mean(
            [
                float(detail["wasted_node_count"])
                for detail in details
                if detail.get("wasted_node_count") is not None
            ]
        ),
        "layer_batch_width": _mean(
            [
                float(width)
                for detail in details
                for width in detail.get("layer_batch_widths", [])
            ]
        ),
        "full_depth_rate": _mean(
            [
                float(detail["observed_full_depth"])
                for detail in details
                if detail.get("observed_full_depth") is not None
            ]
        ),
        "response_received_ms": _mean(
            [
                float(detail["response_received_ms"])
                for detail in details
                if detail.get("response_received_ms") is not None
            ]
        ),
        "response_observation_lag_ms": _mean(
            [
                float(detail["response_observed_ms"])
                - float(detail["response_received_ms"])
                for detail in details
                if detail.get("response_observed_ms") is not None
                and detail.get("response_received_ms") is not None
            ]
        ),
        "layer_wall_ms": _mean(layer_wall_ms),
        "layer_gpu_ms": _mean(layer_gpu_ms),
        "alignment_rate": sum(
            bool(record["target"]["proactive"]) for record in records
        )
        / len(records),
        "interrupted_rate": (
            sum(
                bool(detail.get("stopped_by_response"))
                for detail in proactive_attempts
            )
            / len(proactive_attempts)
            if proactive_attempts
            else 0.0
        ),
        "skip_rate": sum(
            detail.get("skipped_reason") is not None for detail in details
        )
        / len(records),
        "deadline_stop_rate": sum(
            detail.get("policy_reason")
            in {"setup_deadline", "layer_deadline"}
            for detail in details
        )
        / len(records),
        "initial_selected_depth": _mean(
            [
                float(detail["selected_depth"])
                for detail in initial_decisions
                if detail.get("selected_depth") is not None
            ]
        ),
        "initial_accepted_depth": _mean(
            [
                float(detail["accepted_draft_depth"])
                for detail in initial_decisions
                if detail.get("accepted_draft_depth") is not None
            ]
        ),
        "initial_reward": _mean(
            [
                float(detail["reward"])
                for detail in initial_decisions
                if detail.get("reward") is not None
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        nargs="+",
        required=True,
        type=Path,
        help="One or more experiment result directories",
    )
    args = parser.parse_args()

    headers = [
        "experiment",
        "mode",
        "path",
        "initial",
        "cycles",
        "tok/s",
        "cycle ms",
        "wait ms",
        "proactive ms",
        "response ms",
        "response lag",
        "layer wall",
        "layer gpu",
        "depth",
        "leaves",
        "selected",
        "roots",
        "stop",
        "match reuse",
        "reuse",
        "nodes",
        "waste",
        "batch",
        "full %",
        "align %",
        "interrupt %",
        "skip %",
        "deadline %",
        "init depth",
        "accept depth",
        "reward",
    ]
    rows = []
    for data_dir in args.data:
        summary = summarize(data_dir)
        rows.append(
            [
                summary["experiment"],
                summary["mode"],
                summary["path_policy"],
                summary["initial_draft_mode"],
                str(summary["cycles"]),
                _format(summary["tokens_per_second"]),
                _format(summary["cycle_ms"]),
                _format(summary["wait_ms"]),
                _format(summary["proactive_ms"]),
                _format(summary["response_received_ms"]),
                _format(summary["response_observation_lag_ms"]),
                _format(summary["layer_wall_ms"]),
                _format(summary["layer_gpu_ms"]),
                _format(summary["executed_depth"]),
                _format(summary["deepest_leaf_count"]),
                _format(summary["selected_leaf_count"]),
                _format(summary["root_count"]),
                _format(summary["matched_stop_depth"]),
                _format(summary["matched_reused_depth"]),
                _format(summary["reused_proactive_depth"]),
                _format(summary["proactive_node_count"]),
                _format(summary["wasted_node_count"]),
                _format(summary["layer_batch_width"]),
                _format(
                    summary["full_depth_rate"] * 100
                    if summary["full_depth_rate"] is not None
                    else None,
                    1,
                ),
                _format(summary["alignment_rate"] * 100, 1),
                _format(summary["interrupted_rate"] * 100, 1),
                _format(summary["skip_rate"] * 100, 1),
                _format(summary["deadline_stop_rate"] * 100, 1),
                _format(summary["initial_selected_depth"]),
                _format(summary["initial_accepted_depth"]),
                _format(summary["initial_reward"]),
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(
        "  ".join(
            value.ljust(widths[index])
            for index, value in enumerate(headers)
        )
    )
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  ".join(
                value.ljust(widths[index]) for index, value in enumerate(row)
            )
        )


if __name__ == "__main__":
    main()
