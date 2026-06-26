import asyncio
import os
import random
import time
from pathlib import Path

import grpc.aio

import log
import util
from specedge.network.json_grpc import (
    STREAM_METHOD,
    deserialize_json,
    serialize_json,
)


GRPC_OPTIONS = [
    ("grpc.keepalive_time_ms", 20_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
]


async def main() -> None:
    logger = log.get_logger()
    result_logger = log.get_result_logger()
    host = os.environ["SPECEDGE_HOST"]
    target_model = os.environ["SPECEDGE_TARGET_MODEL"]
    dataset_name = os.environ["SPECEDGE_DATASET"]
    max_new_tokens = int(os.environ["SPECEDGE_MAX_NEW_TOKENS"])
    max_request_num = int(os.environ["SPECEDGE_MAX_REQUEST_NUM"])
    sample_req_cnt = int(os.environ["SPECEDGE_SAMPLE_REQ_CNT"])
    req_offset = int(os.environ["SPECEDGE_REQ_OFFSET"])
    client_idx = int(os.environ["SPECEDGE_CLIENT_IDX"])
    reasoning = os.environ.get("SPECEDGE_REASONING", "False") == "True"

    dataset = util.load_dataset(
        dataset_name,
        model_name=target_model,
        reasoning=reasoning,
    )
    request_limit = (
        len(dataset) if max_request_num == -1 else max_request_num
    )
    req_indices = list(range(len(dataset)))
    req_indices = req_indices[req_offset:request_limit:sample_req_cnt]
    random.seed(client_idx)
    random.shuffle(req_indices)

    async with grpc.aio.insecure_channel(
        host,
        options=GRPC_OPTIONS,
    ) as channel:
        stream_generate = channel.unary_stream(
            STREAM_METHOD,
            request_serializer=serialize_json,
            response_deserializer=deserialize_json,
        )

        for request_index, req_idx in enumerate(req_indices, start=1):
            logger.info(
                "Request %d/%d, req_idx: %d",
                request_index,
                len(req_indices),
                req_idx,
            )
            request_start = time.perf_counter()
            arrival_times = []
            server_elapsed_times = []
            decode_times = []
            model_decode_times = []
            server_response_times = []
            queue_wait_times = []
            server_compute_times = []
            batch_sizes = []
            queue_lengths = []
            background_arrival_rates = []
            token_ids = []
            prefill_ms = None
            prompt_tokens = None

            call = stream_generate(
                {
                    "client_idx": client_idx,
                    "req_idx": req_idx,
                    "prompt": dataset[req_idx],
                    "max_new_tokens": max_new_tokens,
                },
                timeout=600.0,
            )
            async for response in call:
                arrival_ms = (
                    time.perf_counter() - request_start
                ) * 1000
                arrival_times.append(arrival_ms)
                server_elapsed_times.append(
                    float(response["server_elapsed_ms"])
                )
                decode_times.append(float(response["decode_ms"]))
                model_decode_times.append(
                    float(
                        response.get(
                            "model_decode_ms",
                            response["decode_ms"],
                        )
                    )
                )
                server_response_times.append(
                    float(response.get("server_response_ms", response["decode_ms"]))
                )
                queue_wait_times.append(float(response.get("queue_wait_ms", 0.0)))
                server_compute_times.append(
                    float(
                        response.get(
                            "server_compute_ms",
                            response.get("decode_ms", 0.0),
                        )
                    )
                )
                batch_sizes.append(int(response.get("batch_size", 0)))
                queue_lengths.append(int(response.get("queue_length", 0)))
                background_arrival_rates.append(
                    float(response.get("background_arrival_rate", 0.0))
                )
                token_ids.append(int(response["token_id"]))
                prefill_ms = float(response["prefill_ms"])
                prompt_tokens = int(response["prompt_tokens"])

            if not arrival_times:
                raise RuntimeError(
                    f"No tokens returned for request {req_idx}"
                )

            ttft_ms = arrival_times[0]
            end_to_end_ms = arrival_times[-1]
            inter_token_ms = [
                later - earlier
                for earlier, later in zip(
                    arrival_times,
                    arrival_times[1:],
                )
            ]
            tpot_ms = (
                sum(inter_token_ms) / len(inter_token_ms)
                if inter_token_ms
                else 0.0
            )
            delivery_overhead_ms = [
                client_elapsed - server_elapsed
                for client_elapsed, server_elapsed in zip(
                    arrival_times,
                    server_elapsed_times,
                )
            ]

            result_logger.log(
                {
                    "client_idx": client_idx,
                    "req_idx": req_idx,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": len(token_ids),
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "end_to_end_ms": end_to_end_ms,
                    "client_tokens_per_second": (
                        len(token_ids) * 1000 / end_to_end_ms
                    ),
                    "arrival_ms": arrival_times,
                    "inter_token_ms": inter_token_ms,
                    "delivery_overhead_ms": delivery_overhead_ms,
                    "server_prefill_ms": prefill_ms,
                    "server_decode_ms": decode_times,
                    "server_model_decode_ms": model_decode_times,
                    "server_response_ms": server_response_times,
                    "queue_wait_ms": queue_wait_times,
                    "server_compute_ms": server_compute_times,
                    "batch_size": batch_sizes,
                    "queue_length": queue_lengths,
                    "background_arrival_rate": background_arrival_rates,
                    "simulated_decode_latency_ms": float(
                        response.get("simulated_decode_latency_ms", 0.0)
                    ),
                    "server_elapsed_ms": server_elapsed_times[-1],
                }
            )
            logger.info(
                "Finished req_idx=%d, tokens=%d, TTFT=%.2f ms, "
                "TPOT=%.2f ms, E2E=%.2f ms",
                req_idx,
                len(token_ids),
                ttft_ms,
                tpot_ms,
                end_to_end_ms,
            )


if __name__ == "__main__":
    log_dir = (
        Path(os.environ["SPECEDGE_RESULT_PATH"])
        / os.environ["SPECEDGE_EXP_NAME"]
    )
    process_name = os.environ["SPECEDGE_PROCESS_NAME"]
    log.configure_logging(
        log.get_default_log_config(log_dir, process_name)
    )
    log.log_unexpected_exception()
    asyncio.run(main())
