import argparse
import asyncio
import signal
import time
from pathlib import Path
from typing import AsyncIterator

import grpc
import grpc.aio
import torch
import yaml

import log
import util
from specedge.engine.graph import GraphEngine
from specedge.network.json_grpc import (
    SERVICE_NAME,
    deserialize_json,
    serialize_json,
)


GRPC_OPTIONS = [
    ("grpc.keepalive_time_ms", 20_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
]


class NetworkAutoregressiveService:
    def __init__(self, config: dict) -> None:
        self._logger = log.get_logger()
        self._result_logger = log.get_result_logger()
        self._device = torch.device(config["server"]["device"])
        self._dtype = util.convert_dtype(config["base"]["dtype"])
        self._model_name = config["server"]["target_model"]
        self._temperature = float(config["server"]["temperature"])
        self._max_len = int(config["base"]["max_len"])
        self._seed = int(config["base"]["seed"])
        self._request_lock = asyncio.Lock()
        configured_max_new_tokens = int(
            config["client"]["max_new_tokens"]
        )
        if not 0 < configured_max_new_tokens < self._max_len:
            raise ValueError(
                "client.max_new_tokens must be greater than zero and "
                "smaller than base.max_len"
            )

        self._logger.info(
            "Initializing remote autoregressive model %s on %s",
            self._model_name,
            self._device,
        )
        model = util.load_graph_model(
            name=self._model_name,
            device=self._device,
            dtype=self._dtype,
        )
        self._engine = GraphEngine(
            model=model,
            max_len=self._max_len,
            max_n_beams=1,
        )
        self._tokenizer = util.load_tokenizer(self._model_name)

    async def stream_generate(
        self,
        request: dict,
        context,
    ) -> AsyncIterator[dict]:
        async with self._request_lock:
            client_idx = int(request["client_idx"])
            req_idx = int(request["req_idx"])
            prompt = str(request["prompt"])
            max_new_tokens = int(request["max_new_tokens"])
            if not 0 < max_new_tokens < self._max_len:
                raise ValueError(
                    "max_new_tokens must be greater than zero and "
                    "smaller than max_len"
                )
            util.set_seed(self._seed)
            self._engine.reset()

            request_start = time.perf_counter()
            input_ids = self._tokenizer.encode(
                prompt,
                return_tensors="pt",
            ).to(self._device)
            max_prompt_len = max(1, self._max_len - max_new_tokens)
            input_ids = input_ids[:, -max_prompt_len:]
            prompt_len = input_ids.size(-1)

            position_ids = torch.arange(
                prompt_len,
                dtype=torch.long,
                device=self._device,
            ).unsqueeze(0)
            cache_seq_indices = torch.arange(
                prompt_len,
                dtype=torch.long,
                device=self._device,
            )
            prefill_attention_mask = torch.ones(
                (1, 1, prompt_len, self._max_len),
                dtype=self._dtype,
                device=self._device,
            ).tril_()

            with util.Timing(device=self._device, mode="sync") as prefill_t:
                self._engine.prefill(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    batch_idx=0,
                    cache_seq_indices=cache_seq_indices,
                    attention_mask=prefill_attention_mask,
                )

            current_token = input_ids[:, -1:]
            current_position = prompt_len - 1
            decode_times = []
            generated_tokens = []

            for token_index in range(max_new_tokens):
                attention_mask = torch.zeros(
                    (1, 1, 1, self._max_len),
                    dtype=self._dtype,
                    device=self._device,
                )
                attention_mask[..., : current_position + 1] = 1.0

                with util.Timing(device=self._device, mode="sync") as decode_t:
                    logits = self._engine.forward(
                        input_ids=current_token,
                        position_ids=torch.tensor(
                            [[current_position]],
                            dtype=torch.long,
                            device=self._device,
                        ),
                        cache_batch_indices=torch.tensor(
                            [0],
                            dtype=torch.long,
                            device=self._device,
                        ),
                        cache_seq_indices=torch.tensor(
                            [current_position],
                            dtype=torch.long,
                            device=self._device,
                        ),
                        attention_mask=attention_mask,
                    )
                    next_token = util.sampler_from_logits(
                        logits[:, -1:, :],
                        temperature=self._temperature,
                    ).reshape(1, 1)

                token_id = int(next_token.item())
                generated_tokens.append(token_id)
                decode_times.append(decode_t.elapsed)
                eos = token_id == self._tokenizer.eos_token_id
                server_elapsed_ms = (
                    time.perf_counter() - request_start
                ) * 1000

                yield {
                    "client_idx": client_idx,
                    "req_idx": req_idx,
                    "token_index": token_index,
                    "token_id": token_id,
                    "eos": eos,
                    "prompt_tokens": prompt_len,
                    "prefill_ms": prefill_t.elapsed,
                    "decode_ms": decode_t.elapsed,
                    "server_elapsed_ms": server_elapsed_ms,
                }

                if eos:
                    break
                current_token = next_token
                current_position += 1

            self._result_logger.log(
                {
                    "client_idx": client_idx,
                    "req_idx": req_idx,
                    "prompt_tokens": prompt_len,
                    "generated_tokens": len(generated_tokens),
                    "prefill_ms": prefill_t.elapsed,
                    "decode_ms": decode_times,
                    "server_end_to_end_ms": (
                        time.perf_counter() - request_start
                    )
                    * 1000,
                }
            )


def add_service(service: NetworkAutoregressiveService, server) -> None:
    handler = grpc.unary_stream_rpc_method_handler(
        service.stream_generate,
        request_deserializer=deserialize_json,
        response_serializer=serialize_json,
    )
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                SERVICE_NAME,
                {"StreamGenerate": handler},
            ),
        )
    )


async def serve(config: dict) -> None:
    service = NetworkAutoregressiveService(config)
    server = grpc.aio.server(options=GRPC_OPTIONS)
    add_service(service, server)
    port = int(config["server"].get("port", 8002))
    server.add_insecure_port(f"0.0.0.0:{port}")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await server.start()
    log.get_logger().info(
        "Network autoregressive server listening on port %d",
        port,
    )
    await stop_event.wait()
    await server.stop(grace=2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    with args.config.open() as file:
        config = yaml.safe_load(file)

    log_dir = (
        Path(config["base"]["result_path"])
        / config["base"]["exp_name"]
    )
    log.configure_logging(
        log.get_default_log_config(log_dir, "network_ar_server")
    )
    log.log_unexpected_exception()
    asyncio.run(serve(config))


if __name__ == "__main__":
    main()
