import argparse
import subprocess
import sys
from pathlib import Path

import yaml

import log

SPECEDGE_ROOT = Path(__file__).absolute().parents[2]


def main(config_file: str):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # base configuration
    result_path = config["base"]["result_path"]
    exp_name = config["base"]["exp_name"]
    dtype = config["base"]["dtype"]
    seed = config["base"]["seed"]
    ssh_key = config["base"]["ssh_key"]
    optimization = config["opt"]
    max_len = config["base"]["max_len"]

    log_config = log.get_default_log_config(Path(result_path) / exp_name, "client_host")
    log.configure_logging(log_config)
    log.log_unexpected_exception()

    logger = log.get_logger()

    logger.info("Starting client host")

    logger.debug("result_path: %s", result_path)
    logger.debug("exp_name: %s", exp_name)

    # client configuration
    host = config["client"]["host"]
    base_process_name = config["client"]["process_name"]
    draft_model = config["client"]["draft_model"]
    target_model = config["server"]["target_model"]
    dataset = config["client"]["dataset"]
    max_n_beams = config["client"]["max_n_beams"]
    max_beam_len = config["client"]["max_beam_len"]
    max_branch_width = config["client"]["max_branch_width"]
    max_budget = config["client"]["max_budget"]
    req_offset = config["client"]["req_offset"]
    sample_req_cnt = config["client"]["sample_req_cnt"]
    reasoning = config["client"]["reasoning"]

    logger.debug("draft_model: %s", draft_model)
    logger.debug("target_model: %s", target_model)
    logger.debug("dataset: %s", dataset)
    logger.debug("max_n_beams: %s", max_n_beams)
    logger.debug("max_beam_len: %s", max_beam_len)
    logger.debug("max_branch_width: %s", max_branch_width)
    logger.debug("max_budget: %s", max_budget)
    logger.debug("reasoning: %s", reasoning)

    # client praoctive draft configuration
    proactive_type = config["client"]["proactive"]["type"]
    proactive_max_n_beams = config["client"]["proactive"]["max_n_beams"]
    proactive_max_beam_len = config["client"]["proactive"]["max_beam_len"]
    proactive_max_branch_width = config["client"]["proactive"]["max_branch_width"]
    proactive_max_budget = config["client"]["proactive"]["max_budget"]
    proactive_mode = config["client"]["proactive"].get("mode", "baseline")
    proactive_adaptive = config["client"]["proactive"].get("adaptive", {})
    proactive_adaptive_ewma_alpha = proactive_adaptive.get("ewma_alpha", 0.2)
    proactive_adaptive_min_alignment_rate = proactive_adaptive.get(
        "min_alignment_rate", 0.1
    )
    proactive_adaptive_warmup_cycles = proactive_adaptive.get("warmup_cycles", 4)
    proactive_adaptive_exploration_interval = proactive_adaptive.get(
        "exploration_interval", 8
    )
    proactive_adaptive_safety_margin_ms = proactive_adaptive.get(
        "safety_margin_ms", 2.0
    )

    logger.debug("proactive_type: %s", proactive_type)
    logger.debug("proactive_mode: %s", proactive_mode)
    logger.debug("proactive_max_n_beams: %s", proactive_max_n_beams)
    logger.debug("proactive_max_beam_len: %s", proactive_max_beam_len)
    logger.debug("proactive_max_branch_width: %s", proactive_max_branch_width)
    logger.debug("proactive_max_budget: %s", proactive_max_budget)

    # experiment configuration
    max_new_tokens = config["client"]["max_new_tokens"]
    max_request_num = config["client"]["max_request_num"]

    # node configuration
    nodes = config["node"]
    logger.debug("nodes: %s", nodes)

    ssh_processes = {}
    client_idx = 0
    for node_name, node_info in nodes.items():
        for client_info in node_info:
            device = client_info["device"]

            logger.info("Starting a client_%s on %s, %s", client_idx, node_name, device)

            env_vars = {
                "SPECEDGE_OPTIMIZATION": optimization,
                "SPECEDGE_RESULT_PATH": result_path,
                "SPECEDGE_EXP_NAME": exp_name,
                "SPECEDGE_PROCESS_NAME": f"{base_process_name}_{client_idx}",
                "SPECEDGE_SEED": seed,
                "SPECEDGE_MAX_LEN": max_len,
                "SPECEDGE_DRAFT_MODEL": draft_model,
                "SPECEDGE_TARGET_MODEL": target_model,
                "SPECEDGE_DEVICE": device,
                "SPECEDGE_DTYPE": dtype,
                "SPECEDGE_DATASET": dataset,
                "SPECEDGE_MAX_N_BEAMS": max_n_beams,
                "SPECEDGE_MAX_BEAM_LEN": max_beam_len,
                "SPECEDGE_MAX_BRANCH_WIDTH": max_branch_width,
                "SPECEDGE_MAX_BUDGET": max_budget,
                "SPECEDGE_PROACTIVE_TYPE": proactive_type,
                "SPECEDGE_PROACTIVE_MODE": proactive_mode,
                "SPECEDGE_PROACTIVE_MAX_N_BEAMS": proactive_max_n_beams,
                "SPECEDGE_PROACTIVE_MAX_BEAM_LEN": proactive_max_beam_len,
                "SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH": proactive_max_branch_width,
                "SPECEDGE_PROACTIVE_MAX_BUDGET": proactive_max_budget,
                "SPECEDGE_PROACTIVE_ADAPTIVE_EWMA_ALPHA": (
                    proactive_adaptive_ewma_alpha
                ),
                "SPECEDGE_PROACTIVE_ADAPTIVE_MIN_ALIGNMENT_RATE": (
                    proactive_adaptive_min_alignment_rate
                ),
                "SPECEDGE_PROACTIVE_ADAPTIVE_WARMUP_CYCLES": (
                    proactive_adaptive_warmup_cycles
                ),
                "SPECEDGE_PROACTIVE_ADAPTIVE_EXPLORATION_INTERVAL": (
                    proactive_adaptive_exploration_interval
                ),
                "SPECEDGE_PROACTIVE_ADAPTIVE_SAFETY_MARGIN_MS": (
                    proactive_adaptive_safety_margin_ms
                ),
                "SPECEDGE_MAX_NEW_TOKENS": max_new_tokens,
                "SPECEDGE_MAX_REQUEST_NUM": max_request_num,
                "SPECEDGE_REQ_OFFSET": req_offset,
                "SPECEDGE_SAMPLE_REQ_CNT": sample_req_cnt,
                "SPECEDGE_HOST": host,
                "SPECEDGE_CLIENT_IDX": client_idx,
                "SPECEDGE_REASONING": reasoning,
            }

            cmd = f"cd {SPECEDGE_ROOT} && "

            for key, value in env_vars.items():
                cmd += f'export {key}="{value}" && '

            cmd += "bash ./script/client.sh"

            logger.debug("cmd: %s", cmd)
            process = subprocess.Popen(  # noqa: S603
                [
                    "ssh",
                    "-i",
                    ssh_key,
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-R",
                    "18001:127.0.0.1:8001",
                    node_name,
                    cmd,
                ],  # noqa: S607
                stdout=subprocess.PIPE,
                stderr=sys.stderr.buffer,
                text=True,
            )

            ssh_processes[client_idx] = process
            client_idx += 1

    for client_idx, process in ssh_processes.items():
        process.wait()
        logger.info("client_%d finished", client_idx)

    logger.info("All clients finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    main(args.config)
