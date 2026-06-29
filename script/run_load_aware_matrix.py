#!/usr/bin/env python3
import argparse
import copy
import subprocess
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "config" / "generated_load_aware"

TEMPLATES = {
    "ar": ROOT / "config/specedge_4090_jetson_multitenant_streaming_ar.yaml",
    "original": ROOT / "config/specedge_4090_jetson_multitenant_specedge.yaml",
    "optimized": ROOT / "config/specedge_4090_jetson_multitenant_response_only.yaml",
    "adaptive_threshold": ROOT / "config/specedge_4090_jetson_multitenant_adaptive.yaml",
    "adaptive_predictor": ROOT / "config/specedge_4090_jetson_multitenant_adaptive.yaml",
}


def _rate_label(rate: float) -> str:
    return str(rate).replace(".", "p")


def _load_yaml(path: Path) -> dict:
    with path.open() as file:
        return yaml.safe_load(file)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def _set_background(config: dict, *, profile: str, rate: float, load_label: str) -> None:
    background = config["server"].setdefault("background", {})
    background["profile"] = profile
    background["load"] = load_label
    background["arrival_rate"] = rate
    background.setdefault("max_active_requests", config["server"]["max_batch_size"])
    background.setdefault("prompt_min_tokens", 32)
    background.setdefault("prompt_max_tokens", 128)
    background.setdefault("generation_min_tokens", 16)
    background.setdefault("generation_max_tokens", 48)
    background.setdefault("start_on_first_foreground", False)
    config["server"]["simulated_latency_ms"] = 0.0
    config["server"]["simulated_decode_latency_ms"] = 0.0


def _apply_scheme(config: dict, scheme: str) -> None:
    decoding = config.setdefault("client", {}).setdefault("decoding", {})
    if scheme == "adaptive_threshold":
        decoding["controller"] = "threshold"
        decoding.setdefault("initial_mode", "specedge")
        decoding.setdefault("decision_window", 16)
    elif scheme == "adaptive_predictor":
        decoding["controller"] = "performance"
        decoding["initial_mode"] = "ar"
        decoding["decision_window"] = max(int(decoding.get("decision_window", 64)), 64)
        decoding.setdefault("ar_ms_per_token_prior", 45.0)
        decoding.setdefault("specedge_cycle_ms_prior", 90.0)
        decoding.setdefault("accepted_tokens_prior", 3.2)
        decoding.setdefault("switch_margin", 0.05)
        decoding.setdefault("min_mode_duration_tokens", 32)
        decoding.setdefault("min_mode_duration_cycles", 2)


def make_constant(loads: list[float], schemes: list[str]) -> list[Path]:
    configs: list[Path] = []
    for rate in loads:
        for scheme in schemes:
            template = TEMPLATES[scheme]
            cfg = _load_yaml(template)
            label = f"constant_{_rate_label(rate)}_{scheme}"
            cfg["base"]["exp_name"] = f"loadaware_{label}"
            _apply_scheme(cfg, scheme)
            _set_background(
                cfg,
                profile="constant",
                rate=rate,
                load_label=str(rate),
            )
            path = GENERATED / "constant" / f"{label}.yaml"
            _write_yaml(path, cfg)
            configs.append(path)
    return configs


def make_threshold(rates: list[float], thresholds: list[float]) -> list[Path]:
    configs: list[Path] = []
    template = TEMPLATES["adaptive_threshold"]
    for rate in rates:
        for threshold in thresholds:
            cfg = _load_yaml(template)
            label = f"threshold_r{_rate_label(rate)}_t{int(threshold)}"
            cfg["base"]["exp_name"] = f"loadaware_{label}"
            _apply_scheme(cfg, "adaptive_threshold")
            cfg["client"]["decoding"]["switch_threshold_ms"] = threshold
            _set_background(cfg, profile="constant", rate=rate, load_label=str(rate))
            path = GENERATED / "threshold" / f"{label}.yaml"
            _write_yaml(path, cfg)
            configs.append(path)
    return configs


def _strong_background(config: dict, *, min_tokens: int, max_tokens: int) -> None:
    background = config["server"].setdefault("background", {})
    background["generation_min_tokens"] = min_tokens
    background["generation_max_tokens"] = max_tokens


def make_step(schemes: list[str]) -> list[Path]:
    configs: list[Path] = []
    for scheme in schemes:
        cfg = _load_yaml(TEMPLATES[scheme])
        cfg["base"]["exp_name"] = f"loadaware_step_{scheme}"
        _apply_scheme(cfg, scheme)
        _set_background(cfg, profile="step", rate=0.0, load_label="step")
        _strong_background(cfg, min_tokens=128, max_tokens=256)
        cfg["server"]["background"]["step_schedule"] = [
            {"duration_s": 60, "arrival_rate": 0.0},
            {"duration_s": 60, "arrival_rate": 14.0},
            {"duration_s": 60, "arrival_rate": 0.0},
            {"duration_s": 60, "arrival_rate": 14.0},
            {"duration_s": 60, "arrival_rate": 0.0},
        ]
        path = GENERATED / "step" / f"{scheme}.yaml"
        _write_yaml(path, cfg)
        configs.append(path)
    return configs


def make_bursty(schemes: list[str]) -> list[Path]:
    configs: list[Path] = []
    for scheme in schemes:
        cfg = _load_yaml(TEMPLATES[scheme])
        cfg["base"]["exp_name"] = f"loadaware_bursty_{scheme}"
        _apply_scheme(cfg, scheme)
        _set_background(cfg, profile="step", rate=0.0, load_label="bursty")
        _strong_background(cfg, min_tokens=128, max_tokens=256)
        cfg["server"]["background"]["step_schedule"] = [
            {"duration_s": 60, "arrival_rate": 0.0},
            {"duration_s": 30, "arrival_rate": 14.0},
            {"duration_s": 60, "arrival_rate": 0.0},
            {"duration_s": 30, "arrival_rate": 14.0},
            {"duration_s": 60, "arrival_rate": 0.0},
            {"duration_s": 30, "arrival_rate": 14.0},
            {"duration_s": 60, "arrival_rate": 0.0},
        ]
        path = GENERATED / "bursty" / f"{scheme}.yaml"
        _write_yaml(path, cfg)
        configs.append(path)
    return configs


def exp_name(config: Path) -> str:
    return _load_yaml(config)["base"]["exp_name"]


def is_streaming_ar(config: Path) -> bool:
    return "streaming_ar" in config.name or config.stem.endswith("_ar")


def run_config(config: Path, warmup_seconds: int, client_node: str, ssh_key: str) -> None:
    name = exp_name(config)
    result = ROOT / "result/4090_jetson" / name
    subprocess.run(
        "pkill -f '[p]ython -O src/script/batch_server.py' 2>/dev/null || true; "
        "pkill -f '[m]ultiprocessing.forkserver.*batch_server.py' 2>/dev/null || true",
        shell=True,
        cwd=ROOT,
        check=False,
    )
    subprocess.run(["rm", "-rf", str(result)], cwd=ROOT, check=False)
    subprocess.run(
        [
            "ssh",
            "-i",
            ssh_key,
            "-o",
            "StrictHostKeyChecking=no",
            client_node,
            f"rm -rf ~/specedge/result/4090_jetson/{name}",
        ],
        cwd=ROOT,
        check=False,
    )
    server = subprocess.Popen(
        ["setsid", "./script/batch_server.sh", "-f", str(config)],
        cwd=ROOT,
    )
    try:
        time.sleep(warmup_seconds)
        client_script = (
            "./script/network_autoregressive_client_host.sh"
            if is_streaming_ar(config)
            else "./script/client_host.sh"
        )
        subprocess.run([client_script, "-f", str(config)], cwd=ROOT, check=True)
        result.mkdir(parents=True, exist_ok=True)
        prefix = "network_ar_client_0" if is_streaming_ar(config) else "client_0"
        subprocess.run(
            [
                "rsync",
                "-a",
                "-e",
                f"ssh -i {ssh_key} -o StrictHostKeyChecking=no",
                f"{client_node}:~/specedge/result/4090_jetson/{name}/{prefix}.*",
                f"result/4090_jetson/{name}/",
            ],
            cwd=ROOT,
            check=True,
        )
    finally:
        subprocess.run(f"kill -- -{server.pid} 2>/dev/null || true", shell=True)
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            subprocess.run(f"kill -9 -- -{server.pid} 2>/dev/null || true", shell=True)
            server.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["constant", "threshold", "step", "bursty", "all"],
        default="constant",
    )
    parser.add_argument("--loads", default="0,0.5,1.0,2.0,4.0,6.0,8.0,10.0,12.0,14.0")
    parser.add_argument(
        "--schemes",
        default="ar,original,optimized,adaptive_threshold,adaptive_predictor",
    )
    parser.add_argument("--threshold-rates", default="0.5,1.0,2.0")
    parser.add_argument("--thresholds", default="40,60,80,100,150,200")
    parser.add_argument("--warmup-seconds", type=int, default=180)
    parser.add_argument("--client-node", default="jetson")
    parser.add_argument("--ssh-key", default=str(Path.home() / ".ssh/id_ed25519"))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    schemes = [value.strip() for value in args.schemes.split(",") if value.strip()]
    unknown = sorted(set(schemes) - set(TEMPLATES))
    if unknown:
        raise ValueError(f"Unknown scheme(s): {unknown}")

    configs: list[Path] = []
    if args.experiment in {"constant", "all"}:
        configs.extend(
            make_constant(
                [float(value) for value in args.loads.split(",")],
                schemes,
            )
        )
    if args.experiment in {"threshold", "all"}:
        configs.extend(
            make_threshold(
                [float(value) for value in args.threshold_rates.split(",")],
                [float(value) for value in args.thresholds.split(",")],
            )
        )
    if args.experiment in {"step", "all"}:
        configs.extend(make_step(schemes))
    if args.experiment in {"bursty", "all"}:
        configs.extend(make_bursty(schemes))

    for config in configs:
        print(config)
        if args.run:
            run_config(config, args.warmup_seconds, args.client_node, args.ssh_key)

    if args.run:
        data_dirs = [f"result/4090_jetson/{exp_name(config)}" for config in configs]
        labels = [exp_name(config) for config in configs]
        subprocess.run(
            [
                "python",
                "src/metric/load_aware_runtime.py",
                "-d",
                *data_dirs,
                "--labels",
                ",".join(labels),
                "--summary-csv",
                "results/load_aware_summary.csv",
                "--timeline",
                "results/load_aware_timeline.jsonl",
                "--window-summary",
                "results/load_aware_window_summary.csv",
                "--window-size-s",
                "10",
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
