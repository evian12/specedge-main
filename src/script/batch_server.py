import argparse
import asyncio
import os
import signal
from pathlib import Path

import grpc
import grpc.aio
import yaml

import log
import util
from config import SpecEdgeBatchServerConfig as config
from specedge.network.json_grpc import (
    SERVICE_NAME,
    deserialize_json,
    serialize_json,
)
from specedge_grpc import specedge_pb2_grpc
from strategy.server_verify.specexec.grpc import SpecExecBatchServer

shutdown_event = None


def signal_handler(signum, frame):
    """Handle shutdown signals (SIGINT, SIGTERM)"""
    if shutdown_event:
        shutdown_event.set()


async def serve():
    global shutdown_event

    shutdown_event = asyncio.Event()
    controller = SpecExecBatchServer(shutdown_event=shutdown_event)

    server = grpc.aio.server()
    specedge_pb2_grpc.add_SpecEdgeServiceServicer_to_server(controller, server)
    stream_handler = grpc.unary_stream_rpc_method_handler(
        controller.StreamGenerate,
        request_deserializer=deserialize_json,
        response_serializer=serialize_json,
    )
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                SERVICE_NAME,
                {"StreamGenerate": stream_handler},
            ),
        )
    )
    server.add_insecure_port("0.0.0.0:8001")

    try:
        await server.start()
        await shutdown_event.wait()

        await server.stop(grace=2.0)
        await controller.cleanup()

    except asyncio.CancelledError:
        await server.stop(0)

    except Exception as e:
        await server.stop(0)
        raise


def _load_config(config_file: Path):
    with open(config_file, "r") as f:
        config_yaml = yaml.safe_load(f)

    result_path = config_yaml["base"]["result_path"]
    exp_name = config_yaml["base"]["exp_name"]
    process_name = "server"
    seed = config_yaml["base"]["seed"]
    max_len = config_yaml["base"]["max_len"]
    batch_type = config_yaml["server"]["batch_type"]
    dataset = config_yaml["client"]["dataset"]
    sample_req_cnt = config_yaml["client"]["sample_req_cnt"]
    req_offset = config_yaml["client"]["req_offset"]

    target_model = config_yaml["server"]["target_model"]
    device = config_yaml["server"]["device"]
    dtype = config_yaml["base"]["dtype"]
    temperature = config_yaml["server"]["temperature"]

    max_batch_size = config_yaml["server"]["max_batch_size"]
    max_n_beams = config_yaml["client"]["max_n_beams"]
    max_budget = config_yaml["client"]["max_budget"]
    num_clients = config_yaml["server"]["num_clients"]
    cache_prefill = config_yaml["server"]["cache_prefill"]
    simulated_latency_ms = config_yaml["server"].get("simulated_latency_ms", 0.0)
    simulated_decode_latency_ms = config_yaml["server"].get(
        "simulated_decode_latency_ms",
        0.0,
    )
    validate_timeout_s = config_yaml["server"].get("validate_timeout_s", 120.0)
    scheduler_tick_ms = float(config_yaml["server"].get("scheduler_tick_ms", 5.0))
    background = config_yaml["server"].get("background", {})
    background_load = str(background.get("load", "0"))
    load_to_rate = {
        "0": 0.0,
        "none": 0.0,
        "low": 0.1,
        "medium": 0.5,
        "high": 1.0,
    }
    background_arrival_rate = float(
        background.get(
            "arrival_rate",
            load_to_rate.get(background_load, 0.0),
        )
    )
    background_profile = str(background.get("profile", "constant"))
    background_step_schedule = background.get("step_schedule", "")
    if isinstance(background_step_schedule, list):
        background_step_schedule = ",".join(
            f"{float(point.get('duration_s', point.get('duration', 0.0)))}:"
            f"{float(point.get('arrival_rate', point.get('rate', 0.0)))}"
            for point in background_step_schedule
        )
    background_bursty = background.get("bursty", {})
    background_bursty_base_rate = float(
        background_bursty.get("base_rate", background_arrival_rate)
    )
    background_bursty_burst_rate = float(
        background_bursty.get("burst_rate", max(background_arrival_rate, 1.0))
    )
    background_bursty_trigger_rate = float(
        background_bursty.get("trigger_rate", 0.05)
    )
    background_bursty_min_duration_s = float(
        background_bursty.get("min_duration_s", 5.0)
    )
    background_bursty_max_duration_s = float(
        background_bursty.get("max_duration_s", 15.0)
    )
    background_max_active_requests = int(
        background.get("max_active_requests", max_batch_size)
    )
    background_prompt_min_tokens = int(
        background.get("prompt_min_tokens", 16)
    )
    background_prompt_max_tokens = int(
        background.get("prompt_max_tokens", 128)
    )
    background_generation_min_tokens = int(
        background.get("generation_min_tokens", 16)
    )
    background_generation_max_tokens = int(
        background.get("generation_max_tokens", 64)
    )
    background_queue_poll_ms = float(background.get("queue_poll_ms", 5.0))
    background_start_delay_s = float(background.get("start_delay_s", 0.0))
    background_start_on_first_foreground = bool(
        background.get("start_on_first_foreground", False)
    )
    if simulated_latency_ms < 0.0:
        raise ValueError("server.simulated_latency_ms must be non-negative")
    if simulated_decode_latency_ms < 0.0:
        raise ValueError(
            "server.simulated_decode_latency_ms must be non-negative"
        )
    if validate_timeout_s <= 0.0:
        raise ValueError("server.validate_timeout_s must be positive")
    if scheduler_tick_ms < 0.0:
        raise ValueError("server.scheduler_tick_ms must be non-negative")
    if background_arrival_rate < 0.0:
        raise ValueError("server.background.arrival_rate must be non-negative")
    if background_profile not in {"constant", "step", "bursty"}:
        raise ValueError(
            "server.background.profile must be constant, step, or bursty"
        )
    if background_bursty_base_rate < 0.0 or background_bursty_burst_rate < 0.0:
        raise ValueError("server.background.bursty rates must be non-negative")
    if background_bursty_trigger_rate < 0.0:
        raise ValueError("server.background.bursty.trigger_rate must be non-negative")
    if background_bursty_min_duration_s <= 0.0:
        raise ValueError("server.background.bursty.min_duration_s must be positive")
    if background_bursty_max_duration_s < background_bursty_min_duration_s:
        raise ValueError("server.background.bursty duration range is invalid")
    if background_max_active_requests < 0:
        raise ValueError(
            "server.background.max_active_requests must be non-negative"
        )
    if background_prompt_min_tokens <= 0:
        raise ValueError(
            "server.background.prompt_min_tokens must be positive"
        )
    if background_prompt_max_tokens < background_prompt_min_tokens:
        raise ValueError(
            "server.background prompt token range is invalid"
        )
    if background_generation_min_tokens <= 0:
        raise ValueError(
            "server.background.generation_min_tokens must be positive"
        )
    if background_generation_max_tokens < background_generation_min_tokens:
        raise ValueError(
            "server.background generation token range is invalid"
        )
    if background_queue_poll_ms <= 0.0:
        raise ValueError("server.background.queue_poll_ms must be positive")
    if background_start_delay_s < 0.0:
        raise ValueError("server.background.start_delay_s must be non-negative")

    os.environ["SPECEDGE_RESULT_PATH"] = result_path
    os.environ["SPECEDGE_EXP_NAME"] = exp_name
    os.environ["SPECEDGE_PROCESS_NAME"] = process_name
    os.environ["SPECEDGE_SEED"] = str(seed)
    os.environ["SPECEDGE_MAX_LEN"] = str(max_len)
    os.environ["SPECEDGE_BATCH_TYPE"] = batch_type
    os.environ["SPECEDGE_DATASET"] = dataset
    os.environ["SPECEDGE_SAMPLE_REQ_CNT"] = str(sample_req_cnt)
    os.environ["SPECEDGE_REQ_OFFSET"] = str(req_offset)

    os.environ["SPECEDGE_TARGET_MODEL"] = target_model
    os.environ["SPECEDGE_SERVER_DEVICE"] = device
    os.environ["SPECEDGE_DTYPE"] = dtype
    os.environ["SPECEDGE_TEMPERATURE"] = str(temperature)

    os.environ["SPECEDGE_MAX_BATCH_SIZE"] = str(max_batch_size)
    os.environ["SPECEDGE_MAX_N_BEAMS"] = str(max_n_beams)
    os.environ["SPECEDGE_MAX_BUDGET"] = str(max_budget)

    os.environ["SPECEDGE_NUM_CLIENTS"] = str(num_clients)
    os.environ["SPECEDGE_CACHE_PREFILL"] = str(cache_prefill)
    os.environ["SPECEDGE_SIMULATED_LATENCY_MS"] = str(simulated_latency_ms)
    os.environ["SPECEDGE_SIMULATED_DECODE_LATENCY_MS"] = str(
        simulated_decode_latency_ms
    )
    os.environ["SPECEDGE_VALIDATE_TIMEOUT_S"] = str(validate_timeout_s)
    os.environ["SPECEDGE_SCHEDULER_TICK_MS"] = str(scheduler_tick_ms)
    os.environ["SPECEDGE_BACKGROUND_LOAD"] = background_load
    os.environ["SPECEDGE_BACKGROUND_ARRIVAL_RATE"] = str(
        background_arrival_rate
    )
    os.environ["SPECEDGE_BACKGROUND_PROFILE"] = background_profile
    os.environ["SPECEDGE_BACKGROUND_STEP_SCHEDULE"] = str(
        background_step_schedule
    )
    os.environ["SPECEDGE_BACKGROUND_BURSTY_BASE_RATE"] = str(
        background_bursty_base_rate
    )
    os.environ["SPECEDGE_BACKGROUND_BURSTY_BURST_RATE"] = str(
        background_bursty_burst_rate
    )
    os.environ["SPECEDGE_BACKGROUND_BURSTY_TRIGGER_RATE"] = str(
        background_bursty_trigger_rate
    )
    os.environ["SPECEDGE_BACKGROUND_BURSTY_MIN_DURATION_S"] = str(
        background_bursty_min_duration_s
    )
    os.environ["SPECEDGE_BACKGROUND_BURSTY_MAX_DURATION_S"] = str(
        background_bursty_max_duration_s
    )
    os.environ["SPECEDGE_BACKGROUND_MAX_ACTIVE_REQUESTS"] = str(
        background_max_active_requests
    )
    os.environ["SPECEDGE_BACKGROUND_PROMPT_MIN_TOKENS"] = str(
        background_prompt_min_tokens
    )
    os.environ["SPECEDGE_BACKGROUND_PROMPT_MAX_TOKENS"] = str(
        background_prompt_max_tokens
    )
    os.environ["SPECEDGE_BACKGROUND_GENERATION_MIN_TOKENS"] = str(
        background_generation_min_tokens
    )
    os.environ["SPECEDGE_BACKGROUND_GENERATION_MAX_TOKENS"] = str(
        background_generation_max_tokens
    )
    os.environ["SPECEDGE_BACKGROUND_QUEUE_POLL_MS"] = str(
        background_queue_poll_ms
    )
    os.environ["SPECEDGE_BACKGROUND_START_DELAY_S"] = str(
        background_start_delay_s
    )
    os.environ["SPECEDGE_BACKGROUND_START_ON_FIRST_FOREGROUND"] = str(
        background_start_on_first_foreground
    )

    log_config = log.get_default_log_config(
        Path(config.result_path) / config.exp_name, "server"
    )
    log.configure_logging(log_config)
    log.log_unexpected_exception()

    logger = log.get_logger()

    logger.debug("result_path: %s", result_path)
    logger.debug("exp_name: %s", exp_name)
    logger.debug("process_name: %s", process_name)
    logger.debug("seed: %s", seed)
    logger.debug("max_len: %s", max_len)
    logger.debug("target_model: %s", target_model)
    logger.debug("device: %s", device)
    logger.debug("dtype: %s", dtype)
    logger.debug("temperature: %s", temperature)
    logger.debug("max_batch_size: %s", max_batch_size)
    logger.debug("max_n_beams: %s", max_n_beams)
    logger.debug("max_budget: %s", max_budget)
    logger.debug("simulated_latency_ms: %s", simulated_latency_ms)
    logger.debug(
        "simulated_decode_latency_ms: %s",
        simulated_decode_latency_ms,
    )
    logger.debug("validate_timeout_s: %s", validate_timeout_s)
    logger.debug("scheduler_tick_ms: %s", scheduler_tick_ms)
    logger.debug("background_load: %s", background_load)
    logger.debug("background_arrival_rate: %s", background_arrival_rate)
    logger.debug("background_start_delay_s: %s", background_start_delay_s)
    logger.debug(
        "background_start_on_first_foreground: %s",
        background_start_on_first_foreground,
    )
    logger.debug(
        "background_max_active_requests: %s",
        background_max_active_requests,
    )
    logger.info("Config loaded successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    _load_config(Path(args.config))

    util.set_seed(config.seed)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger = log.get_logger()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        # Signal handler will take care of graceful shutdown
        pass
    except Exception as e:
        logger.exception("Fatal error: %s", e)
    finally:
        import logging

        logging.shutdown()
