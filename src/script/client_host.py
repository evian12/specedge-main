import argparse
import json
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
    initial_draft = config["client"].get("initial_draft", {})
    initial_draft_mode = initial_draft.get("mode", "fixed")
    initial_draft_structure = initial_draft.get("structure", "tree")
    initial_draft_candidate_depths = initial_draft.get(
        "candidate_depths", [max_beam_len]
    )
    initial_draft_warmup_per_depth = initial_draft.get(
        "warmup_per_depth", 4
    )
    initial_draft_exploration_weight = initial_draft.get(
        "exploration_weight", 0.3
    )
    initial_draft_forced_exploration_interval = initial_draft.get(
        "forced_exploration_interval", 32
    )
    initial_draft_ridge_lambda = initial_draft.get("ridge_lambda", 1.0)
    initial_draft_reward_clip = initial_draft.get("reward_clip", 20.0)
    initial_draft_ewma_alpha = initial_draft.get("ewma_alpha", 0.2)
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
    proactive_path_policy = config["client"]["proactive"].get(
        "path_policy", "single_best"
    )
    proactive_multi = config["client"]["proactive"].get("multi", {})
    proactive_multi_max_deepest_leaves = proactive_multi.get(
        "max_deepest_leaves", 8
    )
    proactive_multi_min_bonus_per_leaf = proactive_multi.get(
        "min_bonus_per_leaf", 1
    )
    proactive_multi_max_bonus_per_leaf = proactive_multi.get(
        "max_bonus_per_leaf", 4
    )
    proactive_multi_max_roots = proactive_multi.get("max_roots", 8)
    proactive_multi_min_root_probability = proactive_multi.get(
        "min_root_probability", 0.01
    )
    proactive_multi_leaf_temperature = proactive_multi.get(
        "leaf_temperature", 1.0
    )
    proactive_multi_full_depth_prior = proactive_multi.get(
        "full_depth_prior", 0.5
    )
    proactive_multi_acceptance_ewma_alpha = proactive_multi.get(
        "acceptance_ewma_alpha", 0.1
    )
    proactive_multi_depth_probability_coverage = proactive_multi.get(
        "depth_probability_coverage", [1.0, 0.8, 0.5]
    )
    proactive_sequence = config["client"]["proactive"].get(
        "sequence", {}
    )
    proactive_sequence_acceptance_survival = proactive_sequence.get(
        "acceptance_survival", [1.0, 0.5, 0.3, 0.2, 0.1]
    )
    proactive_sequence_max_bonus_per_depth = proactive_sequence.get(
        "max_bonus_per_depth", 4
    )
    proactive_sequence_max_roots = proactive_sequence.get(
        "max_roots", 8
    )
    proactive_sequence_min_root_probability = proactive_sequence.get(
        "min_root_probability", 0.0
    )
    proactive_sequence_min_bonus_probability = proactive_sequence.get(
        "min_bonus_probability", 0.0
    )
    proactive_sequence_selection_score = proactive_sequence.get(
        "selection_score", "joint"
    )
    proactive_sequence_reuse_depth_bonus = proactive_sequence.get(
        "reuse_depth_bonus", 0.0
    )
    proactive_sequence_depth_probability_coverage = (
        proactive_sequence.get(
            "depth_probability_coverage", [1.0, 0.8, 0.5]
        )
    )
    proactive_adaptive = config["client"]["proactive"].get("adaptive", {})
    proactive_adaptive_ewma_alpha = proactive_adaptive.get("ewma_alpha", 0.2)
    proactive_adaptive_min_alignment_rate = proactive_adaptive.get(
        "min_alignment_rate", 0.1
    )
    proactive_adaptive_low_alignment_depth = proactive_adaptive.get(
        "low_alignment_depth", 0
    )
    proactive_adaptive_warmup_cycles = proactive_adaptive.get("warmup_cycles", 4)
    proactive_adaptive_exploration_interval = proactive_adaptive.get(
        "exploration_interval", 8
    )
    proactive_adaptive_safety_margin_ms = proactive_adaptive.get(
        "safety_margin_ms", 2.0
    )
    proactive_adaptive_uncertainty_scale = proactive_adaptive.get(
        "uncertainty_scale", 1.0
    )

    logger.debug("proactive_type: %s", proactive_type)
    logger.debug("proactive_mode: %s", proactive_mode)
    logger.debug("proactive_path_policy: %s", proactive_path_policy)
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
                "SPECEDGE_INITIAL_DRAFT_MODE": initial_draft_mode,
                "SPECEDGE_INITIAL_DRAFT_STRUCTURE": (
                    initial_draft_structure
                ),
                "SPECEDGE_INITIAL_DRAFT_CANDIDATE_DEPTHS": json.dumps(
                    initial_draft_candidate_depths
                ),
                "SPECEDGE_INITIAL_DRAFT_WARMUP_PER_DEPTH": (
                    initial_draft_warmup_per_depth
                ),
                "SPECEDGE_INITIAL_DRAFT_EXPLORATION_WEIGHT": (
                    initial_draft_exploration_weight
                ),
                "SPECEDGE_INITIAL_DRAFT_FORCED_EXPLORATION_INTERVAL": (
                    initial_draft_forced_exploration_interval
                ),
                "SPECEDGE_INITIAL_DRAFT_RIDGE_LAMBDA": (
                    initial_draft_ridge_lambda
                ),
                "SPECEDGE_INITIAL_DRAFT_REWARD_CLIP": (
                    initial_draft_reward_clip
                ),
                "SPECEDGE_INITIAL_DRAFT_EWMA_ALPHA": (
                    initial_draft_ewma_alpha
                ),
                "SPECEDGE_PROACTIVE_TYPE": proactive_type,
                "SPECEDGE_PROACTIVE_MODE": proactive_mode,
                "SPECEDGE_PROACTIVE_PATH_POLICY": proactive_path_policy,
                "SPECEDGE_PROACTIVE_MULTI_MAX_DEEPEST_LEAVES": (
                    proactive_multi_max_deepest_leaves
                ),
                "SPECEDGE_PROACTIVE_MULTI_MIN_BONUS_PER_LEAF": (
                    proactive_multi_min_bonus_per_leaf
                ),
                "SPECEDGE_PROACTIVE_MULTI_MAX_BONUS_PER_LEAF": (
                    proactive_multi_max_bonus_per_leaf
                ),
                "SPECEDGE_PROACTIVE_MULTI_MAX_ROOTS": (
                    proactive_multi_max_roots
                ),
                "SPECEDGE_PROACTIVE_MULTI_MIN_ROOT_PROBABILITY": (
                    proactive_multi_min_root_probability
                ),
                "SPECEDGE_PROACTIVE_MULTI_LEAF_TEMPERATURE": (
                    proactive_multi_leaf_temperature
                ),
                "SPECEDGE_PROACTIVE_MULTI_FULL_DEPTH_PRIOR": (
                    proactive_multi_full_depth_prior
                ),
                "SPECEDGE_PROACTIVE_MULTI_ACCEPTANCE_EWMA_ALPHA": (
                    proactive_multi_acceptance_ewma_alpha
                ),
                "SPECEDGE_PROACTIVE_MULTI_DEPTH_PROBABILITY_COVERAGE": json.dumps(
                    proactive_multi_depth_probability_coverage
                ),
                "SPECEDGE_PROACTIVE_SEQUENCE_ACCEPTANCE_SURVIVAL": json.dumps(
                    proactive_sequence_acceptance_survival
                ),
                "SPECEDGE_PROACTIVE_SEQUENCE_MAX_BONUS_PER_DEPTH": (
                    proactive_sequence_max_bonus_per_depth
                ),
                "SPECEDGE_PROACTIVE_SEQUENCE_MAX_ROOTS": (
                    proactive_sequence_max_roots
                ),
                "SPECEDGE_PROACTIVE_SEQUENCE_MIN_ROOT_PROBABILITY": (
                    proactive_sequence_min_root_probability
                ),
                "SPECEDGE_PROACTIVE_SEQUENCE_MIN_BONUS_PROBABILITY": (
                    proactive_sequence_min_bonus_probability
                ),
                "SPECEDGE_PROACTIVE_SEQUENCE_SELECTION_SCORE": (
                    proactive_sequence_selection_score
                ),
                "SPECEDGE_PROACTIVE_SEQUENCE_REUSE_DEPTH_BONUS": (
                    proactive_sequence_reuse_depth_bonus
                ),
                "SPECEDGE_PROACTIVE_SEQUENCE_DEPTH_PROBABILITY_COVERAGE": json.dumps(
                    proactive_sequence_depth_probability_coverage
                ),
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
                "SPECEDGE_PROACTIVE_ADAPTIVE_LOW_ALIGNMENT_DEPTH": (
                    proactive_adaptive_low_alignment_depth
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
                "SPECEDGE_PROACTIVE_ADAPTIVE_UNCERTAINTY_SCALE": (
                    proactive_adaptive_uncertainty_scale
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
