import asyncio
from dataclasses import dataclass, field
import heapq
import hashlib
import multiprocessing as mp
import os
import queue
import random
import threading
import time
from pathlib import Path
from typing import Optional

import torch
from rich.progress import track

import log
import util
from specedge.network.json_grpc import deserialize_json, serialize_json
from config import SpecEdgeBatchServerConfig as config
from specedge.engine.graph import BatchGraphEngine
from specedge_grpc import specedge_pb2, specedge_pb2_grpc
from strategy.server_verify.specexec.padding import (
    copy_padded_1d,
    copy_padded_attention_mask,
    validate_draft_request_shapes,
)


@dataclass
class _BackgroundRequestState:
    slot_idx: int
    req_idx: int
    prefix: str
    remaining_tokens: int
    current_token_id: int
    current_position: int
    ready_time: float
    prefilled: bool = False
    in_flight: bool = False


@dataclass(order=True)
class _QueuedRequest:
    arrival_time: float
    sequence_number: int
    request: specedge_pb2.ValidateRequest = field(compare=False)


class SpecExecBatchServer(specedge_pb2_grpc.SpecEdgeServiceServicer):
    def __init__(
        self,
        shutdown_event: asyncio.Event = None,
    ) -> None:
        self._logger = log.get_logger()

        self._loop = asyncio.get_event_loop()
        self._synced = 0
        self._num_clients = config.num_clients
        self._all_sync = asyncio.Condition()

        self._shutdown_event = shutdown_event
        self._resp_queue_task = None

        self._recv_queue = mp.Queue()
        self._resp_queue = mp.Queue()

        self._resp_futures = {}
        self._resp_lock = threading.Lock()
        self._stream_tokenizer = None

        self._resp_queue_task = self._loop.create_task(self._init_resp_queue_loop())
        self._init_inference_loop()

    async def _init_resp_queue_loop(self):
        self._logger.debug("Starting response queue loop")
        while True:
            try:
                if self._shutdown_event and self._shutdown_event.is_set():
                    self._logger.info("Response queue loop shutting down...")
                    break

                try:
                    raw_data, client_idx = await self._loop.run_in_executor(
                        None, self._resp_queue.get, True, 0.5  # block=True, timeout=0.5
                    )
                except queue.Empty:
                    continue

                if raw_data is None and client_idx == -1:
                    self._logger.info(
                        "Received shutdown sentinel, stopping response queue loop"
                    )
                    break

                self._logger.debug("Received response for client %d", client_idx)

                with self._resp_lock:
                    if client_idx in self._resp_futures:
                        self._resp_futures[client_idx].set_result(raw_data)
                    else:
                        self._logger.error("Client index not found in futures")
            except Exception as e:
                self._logger.error("Error processing response: %s", e)
                if self._shutdown_event and self._shutdown_event.is_set():
                    break

    async def Sync(self, request, context):
        async with self._all_sync:
            self._synced += 1

            if self._synced == self._num_clients:
                self._synced = 0
                self._all_sync.notify_all()
            else:
                await self._all_sync.wait()

        return specedge_pb2.SyncResponse()

    async def Validate(self, request, context):
        self._logger.info("Received request: %s", request.client_idx)
        selection, prefill_cnt, metadata = await self._submit_validate_request(request)
        return specedge_pb2.ValidateResponse(
            selection=selection,
            prefill=prefill_cnt,
            queue_wait_ms=float(metadata.get("queue_wait_ms", 0.0)),
            server_compute_ms=float(metadata.get("server_compute_ms", 0.0)),
            server_response_ms=float(metadata.get("server_response_ms", 0.0)),
            decode_ms=float(metadata.get("decode_ms", 0.0)),
            model_decode_ms=float(metadata.get("model_decode_ms", 0.0)),
            prefill_ms=float(metadata.get("prefill_ms", 0.0)),
            batch_size=int(metadata.get("batch_size", 0)),
            queue_length=int(metadata.get("queue_length", 0)),
            background_arrival_rate=float(
                metadata.get("background_arrival_rate", 0.0)
            ),
        )

    async def _submit_validate_request(self, request):
        fut = asyncio.Future()

        with self._resp_lock:
            self._resp_futures[request.client_idx] = fut

        self._recv_queue.put((time.perf_counter(), request.SerializeToString()))
        response = await asyncio.wait_for(
            fut, timeout=config.validate_timeout_s
        )
        if len(response) == 2:
            selection, prefill_cnt = response
            metadata = {}
        else:
            selection, prefill_cnt, metadata = response
        return selection, prefill_cnt, metadata

    def _get_stream_tokenizer(self):
        if self._stream_tokenizer is None:
            self._stream_tokenizer = util.load_tokenizer(config.target_model)
        return self._stream_tokenizer

    def _build_stream_ar_request(
        self,
        *,
        client_idx: int,
        req_idx: int,
        token_id: int,
        position: int,
        prefill: bool,
        prefix: Optional[str],
    ) -> specedge_pb2.ValidateRequest:
        input_ids = torch.tensor([[token_id]], dtype=torch.long)
        position_ids = torch.tensor([[position]], dtype=torch.long)
        cache_seq_indices = torch.tensor([position], dtype=torch.long)
        parent_indices = torch.empty((0,), dtype=torch.long)
        attention_mask = torch.zeros(
            (1, 1, 1, config.max_len),
            dtype=config.dtype,
        )
        attention_mask[..., : position + 1] = 1.0
        return specedge_pb2.ValidateRequest(
            client_idx=client_idx,
            req_idx=req_idx,
            input_ids=util.encode(input_ids),
            position_ids=util.encode(position_ids),
            cache_seq_indices=util.encode(cache_seq_indices),
            parent_indices=util.encode(parent_indices),
            attention_mask=util.encode(attention_mask),
            prefill=prefill,
            prefix=prefix if prefill else None,
        )

    async def StreamGenerate(self, request: dict, context):
        """Streaming AR path backed by the shared batch inference controller.

        The client sends one request and receives generated tokens over the
        same stream. Internally each decode step still enters the unified FCFS
        scheduler, so foreground streaming AR competes with background and
        SpecEdge verify work on the same target model instance.
        """

        tokenizer = self._get_stream_tokenizer()
        client_idx = int(request["client_idx"])
        req_idx = int(request["req_idx"])
        prompt = str(request["prompt"])
        max_new_tokens = int(request["max_new_tokens"])
        if not 0 < max_new_tokens < config.max_len:
            raise ValueError(
                "max_new_tokens must be greater than zero and smaller than max_len"
            )

        stream_start = time.perf_counter()
        should_prefill = bool(request.get("prefill", True))
        if should_prefill:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            max_prompt_len = max(1, config.max_len - max_new_tokens)
            input_ids = input_ids[:, -max_prompt_len:]
            prompt_len = int(input_ids.size(-1))
            current_token_id = int(input_ids[0, -1].item())
            current_position = prompt_len - 1
        else:
            if "current_token_id" not in request or "current_position" not in request:
                raise ValueError(
                    "current_token_id and current_position are required when "
                    "streaming AR starts without prefill"
                )
            current_token_id = int(request["current_token_id"])
            current_position = int(request["current_position"])
            prompt_len = int(request.get("prompt_tokens", current_position + 1))
        generated_tokens = 0
        decode_times: list[float] = []
        response_times: list[float] = []

        for token_index in range(max_new_tokens):
            step_request = self._build_stream_ar_request(
                client_idx=client_idx,
                req_idx=req_idx,
                token_id=current_token_id,
                position=current_position,
                prefill=should_prefill and token_index == 0,
                prefix=prompt if should_prefill and token_index == 0 else None,
            )
            step_start = time.perf_counter()
            selection, _, metadata = await self._submit_validate_request(
                step_request
            )
            response_ms = (time.perf_counter() - step_start) * 1000
            selected = util.decode(
                selection,
                device=torch.device("cpu"),
                dtype=torch.long,
                shape=(1,),
            )
            next_token_id = int(selected[0].item())
            current_token_id = next_token_id
            current_position += 1
            generated_tokens += 1
            decode_ms = float(metadata.get("decode_ms", response_ms))
            decode_times.append(decode_ms)
            response_times.append(response_ms)
            eos = next_token_id == tokenizer.eos_token_id
            server_elapsed_ms = (time.perf_counter() - stream_start) * 1000

            yield {
                "client_idx": client_idx,
                "req_idx": req_idx,
                "token_index": token_index,
                "token_id": next_token_id,
                "eos": eos,
                "prompt_tokens": prompt_len,
                "prefill_ms": float(metadata.get("prefill_ms", 0.0)),
                "decode_ms": decode_ms,
                "model_decode_ms": float(metadata.get("model_decode_ms", decode_ms)),
                "simulated_decode_latency_ms": config.simulated_decode_latency_ms,
                "queue_wait_ms": float(metadata.get("queue_wait_ms", 0.0)),
                "server_compute_ms": float(
                    metadata.get("server_compute_ms", response_ms)
                ),
                "server_response_ms": float(
                    metadata.get("server_response_ms", response_ms)
                ),
                "client_observed_response_ms": response_ms,
                "batch_size": int(metadata.get("batch_size", 0)),
                "queue_length": int(metadata.get("queue_length", 0)),
                "background_arrival_rate": float(
                    metadata.get("background_arrival_rate", 0.0)
                ),
                "server_elapsed_ms": server_elapsed_ms,
            }

            if eos:
                break

        self._logger.info(
            "Finished streaming AR req_idx=%d, tokens=%d, mean_response=%.2f ms",
            req_idx,
            generated_tokens,
            sum(response_times) / len(response_times) if response_times else 0.0,
        )

    def _init_inference_loop(self):
        self._inference_process = mp.Process(
            target=_init_inference,
            args=(
                self._num_clients,
                self._recv_queue,
                self._resp_queue,
            ),
            daemon=False,
        )
        self._inference_process.start()

    async def cleanup(self):
        """Clean up resources during shutdown"""
        self._logger.info("Starting cleanup...")

        # Send sentinel to inference process to trigger shutdown
        try:
            self._logger.info("Sending shutdown signal to inference process...")
            self._recv_queue.put(None)
        except Exception as e:
            self._logger.exception("Error sending shutdown signal %s", e)

        # Wait for inference process to finish (with timeout)
        if self._inference_process and self._inference_process.is_alive():
            self._logger.info("Waiting for inference process to terminate...")
            self._inference_process.join(timeout=10.0)

            if self._inference_process.is_alive():
                self._logger.warning("Inference process did not terminate, forcing...")
                self._inference_process.terminate()
                self._inference_process.join(timeout=2.0)

                if self._inference_process.is_alive():
                    self._logger.error("Inference process still alive, killing...")
                    self._inference_process.kill()

        # Send sentinel to response queue to stop the loop
        try:
            self._resp_queue.put((None, -1))
        except Exception as e:
            self._logger.error(f"Error sending sentinel to response queue: {e}")

        # Wait for response queue task to complete
        if self._resp_queue_task and not self._resp_queue_task.done():
            self._logger.info("Waiting for response queue task to complete...")
            try:
                await asyncio.wait_for(self._resp_queue_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._logger.warning("Response queue task did not complete in time")
                self._resp_queue_task.cancel()

        # Close queues
        try:
            self._recv_queue.close()
            self._resp_queue.close()
            self._recv_queue.join_thread()
            self._resp_queue.join_thread()
        except Exception as e:
            self._logger.exception("Error closing queue %s", e)

        self._logger.info("Cleanup complete")


class InferenceController:
    def __init__(
        self,
        num_clients: int,
        recv_queue: mp.Queue,
        resp_queue: mp.Queue,
    ) -> None:
        self._logger = log.get_logger()
        self._result_logger = log.get_result_logger()

        self._dtype = config.dtype
        self._device = config.device

        self._num_clients = num_clients
        self._foreground_num_clients = num_clients
        self._background_enabled = (
            (
                config.background_arrival_rate > 0.0
                or config.background_profile in {"step", "bursty"}
            )
            and config.background_max_active_requests > 0
        )
        self._background_max_active_requests = (
            config.background_max_active_requests
            if self._background_enabled
            else 0
        )
        self._num_cache_slots = (
            self._foreground_num_clients
            + self._background_max_active_requests
        )
        self._temperature = config.temperature
        self._batch_size = config.max_batch_size
        self._max_budget = config.max_budget
        self._max_n_beams = self._max_budget + 1
        self._max_len = config.max_len
        self._batch_type = config.batch_type
        self.dataset = util.load_dataset(config.dataset, config.target_model)

        self._unified_request_queue: list[_QueuedRequest] = []
        self._arrival_sequence = 0
        self._shutdown_requested = False
        self._recv_queue = recv_queue
        self._resp_queue = resp_queue
        self._rng = random.Random(config.seed + 1009)
        self._background_waiting: list[_BackgroundRequestState] = []
        self._background_active: dict[int, _BackgroundRequestState] = {}
        self._background_free_slots = list(
            range(
                self._foreground_num_clients,
                self._foreground_num_clients
                + self._background_max_active_requests,
            )
        )
        self._background_started = (
            not config.background_start_on_first_foreground
        )
        self._background_start_time = (
            time.perf_counter() + config.background_start_delay_s
            if self._background_started
            else float("inf")
        )
        self._background_epoch_time = self._background_start_time
        self._background_step_schedule = self._parse_step_schedule(
            config.background_step_schedule
        )
        self._background_in_burst = False
        self._background_burst_until = 0.0
        self._background_next_burst_check = self._background_start_time
        self._background_next_req_idx = 0
        self._background_next_arrival = self._next_background_arrival_time()
        self._background_completed_tokens = 0
        self._background_completed_requests = 0

        self._tokenizer = util.load_tokenizer(config.target_model)

        self._logger.info("Initializing inference controller")

        self._logger.debug("Loading model")
        self._model = util.load_graph_model(
            name=config.target_model,
            device=config.device,
            dtype=config.dtype,
        )

        self._engine = BatchGraphEngine(
            model=self._model,
            max_len=config.max_len,
            max_batch_size=config.max_batch_size,
            max_n_beams=self._max_n_beams,
        )

        self.k_cache = torch.zeros(
            (
                self._model.config.num_hidden_layers,
                self._num_cache_slots,
                self._model.config.num_key_value_heads,
                self._max_len,
                self._model.config.head_dim,
            ),
            dtype=self._dtype,
            device=self._device,
        )

        self.v_cache = torch.zeros_like(
            self.k_cache, dtype=self._dtype, device=self._device
        )

        self._client_indices = torch.zeros(
            (self._batch_size,),
            dtype=torch.long,
            device=self._device,
        )

        self._iter_idx = torch.zeros(
            (self._num_cache_slots,),
            dtype=torch.long,
            device=self._device,
        )

        self._input_ids = torch.zeros(
            (self._batch_size, self._max_n_beams),
            dtype=torch.long,
            device=self._device,
        )

        self._parent_indices = torch.zeros(
            (self._batch_size, self._max_budget), dtype=torch.long, device=self._device
        )

        self._position_ids = torch.zeros(
            (self._batch_size, self._max_n_beams),
            dtype=torch.long,
            device=self._device,
        )

        self._cache_batch_indices = torch.arange(
            self._batch_size, dtype=torch.long, device=self._device
        ).repeat_interleave(self._max_n_beams)

        self._cache_seq_indices = torch.zeros(
            (self._batch_size, self._max_n_beams),
            dtype=torch.long,
            device=self._device,
        )

        self._attention_mask = torch.zeros(
            (self._batch_size, 1, self._max_n_beams, self._max_len),
            dtype=self._dtype,
            device=self._device,
        )

        # Predefined tensors for prefill
        self._predefined_position_ids = torch.arange(
            self._max_len, dtype=torch.long, device=self._device
        ).unsqueeze(0)
        self._predefined_attention_mask = torch.ones(
            (1, 1, self._max_len, self._max_len), dtype=self._dtype, device=self._device
        ).tril_()

        self._kv_prefill_offloading = self._cache_prefill()

        self._logger.debug("Inference controller initialized")

    def _parse_step_schedule(self, raw: str) -> list[tuple[float, float]]:
        schedule: list[tuple[float, float]] = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                duration_raw, rate_raw = item.split(":", 1)
                duration_s = float(duration_raw)
                rate = float(rate_raw)
            else:
                duration_s = 30.0
                rate = float(item)
            if duration_s <= 0.0:
                raise ValueError("background step duration must be positive")
            if rate < 0.0:
                raise ValueError("background step rate must be non-negative")
            schedule.append((duration_s, rate))
        return schedule

    def _background_elapsed_s(self) -> float:
        if not self._background_started:
            return 0.0
        return max(0.0, time.perf_counter() - self._background_epoch_time)

    def _current_background_arrival_rate(self) -> float:
        profile = config.background_profile
        if profile == "constant":
            return config.background_arrival_rate
        if profile == "step":
            if not self._background_step_schedule:
                return config.background_arrival_rate
            elapsed = self._background_elapsed_s()
            cursor = 0.0
            for duration_s, rate in self._background_step_schedule:
                cursor += duration_s
                if elapsed < cursor:
                    return rate
            return self._background_step_schedule[-1][1]
        if profile == "bursty":
            now = time.perf_counter()
            if self._background_in_burst and now < self._background_burst_until:
                return config.background_bursty_burst_rate
            if self._background_in_burst and now >= self._background_burst_until:
                self._background_in_burst = False
                self._background_next_burst_check = now
            if now >= self._background_next_burst_check:
                self._background_next_burst_check = now + 1.0
                if (
                    config.background_bursty_trigger_rate > 0.0
                    and self._rng.random() < config.background_bursty_trigger_rate
                ):
                    duration = self._rng.uniform(
                        config.background_bursty_min_duration_s,
                        config.background_bursty_max_duration_s,
                    )
                    self._background_in_burst = True
                    self._background_burst_until = now + duration
                    return config.background_bursty_burst_rate
            return config.background_bursty_base_rate
        return config.background_arrival_rate

    def _next_background_arrival_time(self) -> float:
        if not self._background_enabled or not self._background_started:
            return float("inf")
        rate = self._current_background_arrival_rate()
        if rate <= 0.0:
            return float("inf")
        return max(
            time.perf_counter(),
            self._background_start_time,
        ) + self._rng.expovariate(rate)

    def _start_background_if_needed(self) -> None:
        if not self._background_enabled or self._background_started:
            return
        self._background_started = True
        self._background_start_time = (
            time.perf_counter() + config.background_start_delay_s
        )
        self._background_epoch_time = self._background_start_time
        self._background_next_burst_check = self._background_start_time
        self._background_next_arrival = self._next_background_arrival_time()

    def _safe_qsize(self) -> int:
        try:
            return self._recv_queue.qsize()
        except NotImplementedError:
            return 0

    def _background_has_work(self) -> bool:
        if not self._background_enabled:
            return False
        if not self._background_started:
            return False
        if time.perf_counter() < self._background_start_time:
            return False
        return (
            time.perf_counter() >= self._background_next_arrival
            or bool(self._background_waiting)
            or any(
                not state.in_flight
                for state in self._background_active.values()
            )
        )

    def _create_background_request(self) -> _BackgroundRequestState:
        dataset_idx = self._rng.randrange(len(self.dataset))
        token_ids = self._tokenizer.encode(
            self.dataset[dataset_idx],
            return_tensors="pt",
        )[0]
        if token_ids.numel() < 2:
            fallback = self._tokenizer.eos_token_id or 0
            token_ids = torch.tensor([fallback, fallback], dtype=torch.long)
        max_prompt_len = min(
            int(token_ids.numel()),
            max(2, config.background_prompt_max_tokens),
            self._max_len - 1,
        )
        min_prompt_len = min(
            max_prompt_len,
            max(2, config.background_prompt_min_tokens),
        )
        prompt_len = self._rng.randint(min_prompt_len, max_prompt_len)
        prompt_ids = token_ids[:prompt_len]
        generation_len = self._rng.randint(
            config.background_generation_min_tokens,
            config.background_generation_max_tokens,
        )
        generation_len = min(generation_len, self._max_len - prompt_len)
        state = _BackgroundRequestState(
            slot_idx=-1,
            req_idx=-(self._background_next_req_idx + 1),
            prefix=self._tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False,
            ),
            remaining_tokens=generation_len,
            current_token_id=int(prompt_ids[-1].item()),
            current_position=prompt_len - 1,
            ready_time=self._background_next_arrival,
        )
        self._background_next_req_idx += 1
        return state

    def _queue_request(
        self,
        request: specedge_pb2.ValidateRequest,
        arrival_time: float,
    ) -> None:
        heapq.heappush(
            self._unified_request_queue,
            _QueuedRequest(
                arrival_time=arrival_time,
                sequence_number=self._arrival_sequence,
                request=request,
            ),
        )
        self._arrival_sequence += 1

    def _handle_recv_item(self, item) -> bool:
        if item is None:
            self._shutdown_requested = True
            return False

        if isinstance(item, tuple):
            arrival_time, raw_data = item
        else:
            arrival_time = time.perf_counter()
            raw_data = item

        req = specedge_pb2.ValidateRequest()
        req.ParseFromString(raw_data)
        if req.client_idx < self._foreground_num_clients:
            self._start_background_if_needed()
        self._queue_request(req, float(arrival_time))
        return True

    def _drain_foreground_queue(self) -> None:
        while True:
            try:
                item = self._recv_queue.get(False)
            except queue.Empty:
                return
            self._handle_recv_item(item)
            if self._shutdown_requested:
                return

    def _collect_arrivals(self) -> None:
        deadline = time.perf_counter() + config.scheduler_tick_ms / 1000.0
        self._drain_foreground_queue()
        self._enqueue_background_work()

        while (
            not self._shutdown_requested
            and len(self._unified_request_queue) < self._batch_size
            and time.perf_counter() < deadline
        ):
            timeout = max(0.0, deadline - time.perf_counter())
            try:
                item = self._recv_queue.get(True, timeout)
                self._handle_recv_item(item)
            except queue.Empty:
                pass
            self._enqueue_background_work()

    def _build_background_validate_request(
        self,
        state: _BackgroundRequestState,
    ) -> specedge_pb2.ValidateRequest:
        input_ids = torch.tensor(
            [[state.current_token_id]],
            dtype=torch.long,
            device=self._device,
        )
        position_ids = torch.tensor(
            [[state.current_position]],
            dtype=torch.long,
            device=self._device,
        )
        cache_seq_indices = torch.tensor(
            [state.current_position],
            dtype=torch.long,
            device=self._device,
        )
        parent_indices = torch.empty(
            (0,),
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.zeros(
            (1, 1, 1, self._max_len),
            dtype=self._dtype,
            device=self._device,
        )
        attention_mask[..., : state.current_position + 1] = 1.0
        return specedge_pb2.ValidateRequest(
            client_idx=state.slot_idx,
            req_idx=state.req_idx,
            input_ids=util.encode(input_ids),
            position_ids=util.encode(position_ids),
            cache_seq_indices=util.encode(cache_seq_indices),
            parent_indices=util.encode(parent_indices),
            attention_mask=util.encode(attention_mask),
            prefill=not state.prefilled,
            prefix=state.prefix if not state.prefilled else None,
        )

    def _enqueue_background_work(self) -> None:
        if not self._background_enabled:
            return
        if not self._background_started:
            return
        if time.perf_counter() < self._background_start_time:
            return

        if (
            self._background_next_arrival == float("inf")
            and self._current_background_arrival_rate() > 0.0
        ):
            self._background_next_arrival = self._next_background_arrival_time()

        total_background = (
            len(self._background_waiting)
            + len(self._background_active)
        )
        while (
            time.perf_counter() >= self._background_next_arrival
            and total_background < self._background_max_active_requests
        ):
            self._background_waiting.append(
                self._create_background_request()
            )
            total_background += 1
            self._background_next_arrival = (
                self._next_background_arrival_time()
            )

        while self._background_waiting and self._background_free_slots:
            state = self._background_waiting.pop(0)
            state.slot_idx = self._background_free_slots.pop(0)
            self._background_active[state.slot_idx] = state

        for state in list(self._background_active.values()):
            if state.in_flight or state.remaining_tokens <= 0:
                continue
            if state.ready_time > time.perf_counter():
                continue
            self._queue_request(
                self._build_background_validate_request(state),
                state.ready_time,
            )
            state.in_flight = True

    def _server_step_log(
        self,
        *,
        batch_size: int,
        queue_wait_ms: float,
        server_compute_ms: float,
        server_response_ms: float,
        decode_ms: float,
        prefill_ms: float,
        prefill_count: int,
    ) -> dict:
        return {
            "queue_wait_ms": queue_wait_ms,
            "server_compute_ms": server_compute_ms,
            "server_response_ms": server_response_ms,
            "server_response_time": server_response_ms,
            "decode_latency": decode_ms
            + config.simulated_decode_latency_ms,
            "prefill_latency": prefill_ms,
            "batch_size": batch_size,
            "queue_length": self._safe_qsize() + len(self._unified_request_queue),
            "pending_prefill_count": prefill_count,
            "background_load": config.background_load,
            "background_profile": config.background_profile,
            "background_arrival_rate": self._current_background_arrival_rate(),
            "background_config_arrival_rate": config.background_arrival_rate,
            "background_start_delay_s": config.background_start_delay_s,
            "background_start_on_first_foreground": (
                config.background_start_on_first_foreground
            ),
            "background_started": self._background_started,
            "background_active": len(self._background_active),
            "background_waiting": len(self._background_waiting),
            "background_completed_tokens": self._background_completed_tokens,
            "background_completed_requests": self._background_completed_requests,
        }

    def _cache_prefill(self):
        # Skip prefill caching if disabled
        if not config.cache_prefill:
            self._logger.info("Prefill caching is disabled - will prefill at runtime")
            return {}

        dataset = util.load_dataset(config.dataset, config.target_model)
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")

        if xdg_cache_home is None:
            xdg_cache_home = os.path.join(os.path.expanduser("~"), ".cache")

        model_name = Path(config.target_model).name
        model_key = hashlib.sha256(config.target_model.encode()).hexdigest()[:12]
        cache_folder_name = f"{model_name}-{model_key}_{config.dataset}"
        cache_dir = Path(xdg_cache_home) / "specedge" / cache_folder_name

        kv_prefill_offloading: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        req_indices = list(range(len(dataset)))
        req_indices = req_indices[config.req_offset :][:: config.sample_req_cnt]

        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)

        for req_idx in track(req_indices, description="Prefilling cache"):
            k_cache_file_name = cache_dir / f"{req_idx}_key_cache.pt"
            v_cache_file_name = cache_dir / f"{req_idx}_value_cache.pt"

            if k_cache_file_name.exists() and v_cache_file_name.exists():
                self._logger.debug("Cache files already exist for req_idx=%d", req_idx)
                kv_prefill_offloading[req_idx] = (
                    torch.load(k_cache_file_name, map_location="cpu"),
                    torch.load(v_cache_file_name, map_location="cpu"),
                )
                continue

            prompt = dataset[req_idx]

            self._logger.debug("Creating cache files for req_idx=%d", req_idx)

            input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(
                self._device
            )[..., :-1]
            position_ids = self._predefined_position_ids[:, : input_ids.size(1)]
            cache_seq_indices = self._predefined_position_ids[:, : input_ids.size(1)]
            attention_mask = self._predefined_attention_mask[
                :, :, : input_ids.size(1), : self._max_len
            ]

            self._engine._past_key_values.clear()

            self._engine.prefill(
                input_ids=input_ids,
                position_ids=position_ids,
                batch_idx=0,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
            )

            k_cache = (
                self._engine._past_key_values.k_cache[
                    :, 0, :, : input_ids.size(-1), ...
                ]
                .squeeze(1)
                .clone()
                .detach()
                .cpu()
            )

            v_cache = (
                self._engine._past_key_values.v_cache[
                    :, 0, :, : input_ids.size(-1), ...
                ]
                .squeeze(1)
                .clone()
                .detach()
                .cpu()
            )

            kv_prefill_offloading[req_idx] = (k_cache, v_cache)

            torch.save(k_cache, k_cache_file_name)
            torch.save(v_cache, v_cache_file_name)

        return kv_prefill_offloading

    def loop(self):
        self._logger.debug("Starting inference loop")
        while True:
            self._collect_arrivals()
            if not self._unified_request_queue:
                if self._shutdown_requested:
                    self._logger.info("Inference loop shutting down gracefully")
                    return
                continue

            queued_batch = [
                heapq.heappop(self._unified_request_queue)
                for _ in range(min(self._batch_size, len(self._unified_request_queue)))
            ]
            compute_start = time.perf_counter()
            batch = [item.request for item in queued_batch]
            queue_waits_ms = [
                max(0.0, (compute_start - item.arrival_time) * 1000)
                for item in queued_batch
            ]

            self._logger.info("Batch size reached: %d", len(batch))
            self._client_indices.fill_(-1)

            with util.Timing(device=self._device, mode="sync") as inference_t:
                forward_t, prefill_indices, prefill_t = self._inference(
                    batch,
                    queue_waits_ms=queue_waits_ms,
                    batch_queue_length=self._safe_qsize()
                    + len(self._unified_request_queue),
                    batch_background_arrival_rate=(
                        self._current_background_arrival_rate()
                    ),
                )
            average_queue_wait_ms = (
                sum(queue_waits_ms) / len(queue_waits_ms)
                if queue_waits_ms
                else 0.0
            )
            server_compute_ms = (
                forward_t + prefill_t + config.simulated_decode_latency_ms
            )
            average_server_response_ms = (
                average_queue_wait_ms + server_compute_ms
            )

            self._result_logger.log(
                {
                    "target": {
                        "forward_t": forward_t,
                        "server_end_to_end_t": inference_t.elapsed,
                        "simulated_latency_ms": config.simulated_latency_ms,
                        "simulated_decode_latency_ms": (
                            config.simulated_decode_latency_ms
                        ),
                        "prefill": len(prefill_indices),
                    },
                    "server_step": self._server_step_log(
                        batch_size=len(batch),
                        queue_wait_ms=average_queue_wait_ms,
                        server_compute_ms=server_compute_ms,
                        server_response_ms=average_server_response_ms,
                        decode_ms=forward_t,
                        prefill_ms=prefill_t,
                        prefill_count=len(prefill_indices),
                    ),
                }
            )

    @torch.inference_mode()
    def _inference(
        self,
        batch: list[specedge_pb2.ValidateRequest],
        *,
        queue_waits_ms: list[float],
        batch_queue_length: int,
        batch_background_arrival_rate: float,
    ):
        prefill_indices: list[tuple[int, int]] = []
        request_token_counts: list[int] = []
        self._engine._past_key_values.clear()
        self._input_ids.zero_()
        self._position_ids.zero_()
        self._cache_seq_indices.zero_()
        self._parent_indices.zero_()
        self._attention_mask.zero_()

        for batch_idx, req in enumerate(batch):
            client_idx = req.client_idx
            self._client_indices[batch_idx] = client_idx

            if req.prefill:
                prefill_indices.append((batch_idx, req.req_idx))
                self._iter_idx[req.client_idx] = 0
            else:
                self._iter_idx[req.client_idx] += 1

            input_ids = util.decode(req.input_ids, self._device, torch.long, (-1,))
            position_ids = util.decode(
                req.position_ids, self._device, torch.long, (-1,)
            )
            cache_seq_indices = util.decode(
                req.cache_seq_indices, self._device, torch.long, (-1,)
            )
            parent_indices = (
                torch.empty((0,), dtype=torch.long, device=self._device)
                if len(req.parent_indices) == 0
                else util.decode(
                    req.parent_indices, self._device, torch.long, (-1,)
                )
            )
            attention_mask = util.decode(
                req.attention_mask,
                self._device,
                self._dtype,
                (1, -1, self._max_len),
            )

            validate_draft_request_shapes(
                input_len=input_ids.numel(),
                position_len=position_ids.numel(),
                cache_len=cache_seq_indices.numel(),
                attention_len=attention_mask.size(1),
                parent_len=parent_indices.numel(),
            )

            input_len = copy_padded_1d(
                self._input_ids[batch_idx], input_ids, name="input_ids"
            )
            request_token_counts.append(input_len)
            copy_padded_1d(
                self._position_ids[batch_idx], position_ids, name="position_ids"
            )
            copy_padded_1d(
                self._cache_seq_indices[batch_idx],
                cache_seq_indices,
                name="cache_seq_indices",
            )
            copy_padded_1d(
                self._parent_indices[batch_idx],
                parent_indices,
                name="parent_indices",
            )
            copy_padded_attention_mask(
                self._attention_mask[batch_idx],
                attention_mask,
                name="attention_mask",
            )

            if not req.prefill:
                self._engine._past_key_values.k_cache[:, batch_idx, ...].copy_(
                    self.k_cache[:, req.client_idx, ...]
                )
                self._engine._past_key_values.v_cache[:, batch_idx, ...].copy_(
                    self.v_cache[:, req.client_idx, ...]
                )

        while len(request_token_counts) < self._batch_size:
            request_token_counts.append(0)

        prefill_elapsed_ms = 0.0
        for batch_idx, req_idx in prefill_indices:
            req = batch[batch_idx]
            is_background = req.client_idx >= self._foreground_num_clients
            if config.cache_prefill and not is_background:
                # Load from cache
                k_cache, v_cache = self._kv_prefill_offloading[req_idx]

                self._engine._past_key_values.k_cache[
                    :, batch_idx, :, : k_cache.size(2), :
                ].copy_(k_cache)
                self._engine._past_key_values.v_cache[
                    :, batch_idx, :, : v_cache.size(2), :
                ].copy_(v_cache)
            else:
                # Perform runtime prefill
                if req.prefix is None or req.prefix == "":
                    raise ValueError(
                        f"Prefix is required for runtime prefill (req_idx={req_idx})"
                    )

                input_ids = self._tokenizer.encode(req.prefix, return_tensors="pt").to(
                    self._device
                )[..., :-1]
                position_ids = self._predefined_position_ids[:, : input_ids.size(1)]
                cache_seq_indices = self._predefined_position_ids[
                    :, : input_ids.size(1)
                ]
                attention_mask = self._predefined_attention_mask[
                    :, :, : input_ids.size(1), : self._max_len
                ]

                with util.Timing(device=self._device, mode="sync") as prefill_t:
                    self._engine.prefill(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        batch_idx=batch_idx,
                        cache_seq_indices=cache_seq_indices,
                        attention_mask=attention_mask,
                    )
                prefill_elapsed_ms += prefill_t.elapsed

        with util.Timing(device=self._device, mode="event") as forward_t:
            logits = self._engine.forward(
                input_ids=self._input_ids,
                position_ids=self._position_ids,
                cache_batch_indices=self._cache_batch_indices.flatten(),
                cache_seq_indices=self._cache_seq_indices.flatten(),
                attention_mask=self._attention_mask,
            )

        selection = util.sampler_from_logits(logits, temperature=self._temperature)
        if config.simulated_decode_latency_ms > 0:
            time.sleep(config.simulated_decode_latency_ms / 1000.0)
        if config.simulated_latency_ms > 0:
            time.sleep(config.simulated_latency_ms / 1000.0)
        prefill_batch_indices = {batch_idx for batch_idx, _ in prefill_indices}
        for batch_idx, client_idx in enumerate(self._client_indices):
            if client_idx == -1:
                continue
            request_token_count = request_token_counts[batch_idx]
            client_idx_value = int(client_idx.item())
            if client_idx_value < self._foreground_num_clients:
                queue_wait_ms = (
                    queue_waits_ms[batch_idx]
                    if batch_idx < len(queue_waits_ms)
                    else 0.0
                )
                server_compute_ms = (
                    forward_t.elapsed
                    + (
                        prefill_elapsed_ms
                        if batch_idx in prefill_batch_indices
                        else 0.0
                    )
                    + config.simulated_decode_latency_ms
                )
                metadata = {
                    "queue_wait_ms": queue_wait_ms,
                    "server_compute_ms": server_compute_ms,
                    "server_response_ms": queue_wait_ms + server_compute_ms,
                    "model_decode_ms": forward_t.elapsed,
                    "decode_ms": (
                        forward_t.elapsed
                        + config.simulated_decode_latency_ms
                    ),
                    "prefill_ms": (
                        prefill_elapsed_ms
                        if batch_idx in prefill_batch_indices
                        else 0.0
                    ),
                    "batch_size": len(batch),
                    "queue_length": batch_queue_length,
                    "background_arrival_rate": batch_background_arrival_rate,
                }
                self._resp_queue.put(
                    (
                        (
                            util.encode(selection[batch_idx, :request_token_count]),
                            len(prefill_indices),
                            metadata,
                        ),
                        client_idx_value,
                    )
                )
            else:
                state = self._background_active.get(client_idx_value)
                if state is not None:
                    state.current_token_id = int(selection[batch_idx, 0].item())
                    state.current_position += 1
                    state.remaining_tokens -= 1
                    self._background_completed_tokens += 1
                    state.prefilled = True
                    state.in_flight = False
                    if state.remaining_tokens <= 0:
                        self._background_completed_requests += 1
                        del self._background_active[client_idx_value]
                        self._background_free_slots.append(client_idx_value)
                    else:
                        state.ready_time = time.perf_counter()

        self._reorder_kv_cache(
            selection=selection,
            request_token_counts=request_token_counts,
        )
        return forward_t.elapsed, prefill_indices, prefill_elapsed_ms

    def _reorder_kv_cache(
        self,
        selection: torch.Tensor,
        request_token_counts: list[int],
    ):
        offset = self._cache_seq_indices[:, 0][None, :].T

        target_choices = torch.zeros_like(self._input_ids[..., 1:])
        valid_draft_mask = torch.zeros_like(target_choices, dtype=torch.bool)
        for batch_idx in range(self._batch_size):
            valid_draft_count = max(0, request_token_counts[batch_idx] - 1)
            if valid_draft_count == 0:
                continue

            offset_b = self._cache_seq_indices[batch_idx, 0]
            parent_indices_b = (
                self._parent_indices[batch_idx, :valid_draft_count] - offset_b
            )
            if parent_indices_b.numel() > 0 and (
                parent_indices_b.min() < 0
                or parent_indices_b.max() >= selection.size(1)
            ):
                raise ValueError(
                    "parent_indices are outside the current verification window: "
                    f"min={int(parent_indices_b.min().item())}, "
                    f"max={int(parent_indices_b.max().item())}, "
                    f"selection_size={selection.size(1)}."
                )

            target_choices[batch_idx, :valid_draft_count] = selection[
                batch_idx
            ].flatten()[parent_indices_b]
            valid_draft_mask[batch_idx, :valid_draft_count] = True

        logit_mask = (target_choices == self._input_ids[..., 1:]) & valid_draft_mask

        _batch_indices = self._cache_batch_indices.flatten()
        _seq_indices = self._cache_seq_indices.flatten()

        tree_mask = torch.zeros(
            (self._batch_size, self._max_budget, self._max_budget),
            dtype=self._dtype,
            device=self._device,
        )

        for batch_idx in range(self._batch_size):
            valid_draft_count = max(0, request_token_counts[batch_idx] - 1)
            if valid_draft_count == 0:
                continue
            b_offset = self._cache_seq_indices[batch_idx, 1]
            tree_mask[batch_idx, :valid_draft_count].copy_(
                self._attention_mask[
                    batch_idx,
                    0,
                    1 : 1 + valid_draft_count,
                    b_offset : b_offset + self._max_budget,
                ]
            )

        position = torch.where(
            valid_draft_mask,
            self._position_ids[:, 1:] - offset,
            torch.full_like(self._position_ids[:, 1:], -1),
        )

        accepted_mask = logit_mask[:, None, :] & tree_mask.to(torch.bool)

        last_accepted_val, last_accepted_indices = (
            position * (accepted_mask.sum(dim=-1) == position)
        ).max(dim=-1)

        last_accepted = torch.where(
            last_accepted_val == 0, 0, last_accepted_indices + 1
        )

        for batch_idx, client_idx in enumerate(self._client_indices):
            if client_idx == -1:
                continue

            src_mask = self._attention_mask[batch_idx, 0, last_accepted[batch_idx], :]
            b_src_indices = torch.where(src_mask)[0]
            b_dest_indices = torch.arange(
                b_src_indices.size(-1), dtype=torch.long, device=self._device
            )
            self._engine.gather(batch_idx, b_src_indices, b_dest_indices)
            self.k_cache[:, client_idx, ...].copy_(
                self._engine._past_key_values.k_cache[:, batch_idx, ...]
            )
            self.v_cache[:, client_idx, ...].copy_(
                self._engine._past_key_values.v_cache[:, batch_idx, ...]
            )


def _init_inference(
    num_clients: int,
    recv_queue: mp.Queue,
    resp_queue: mp.Queue,
):
    # Configure logging in child process
    from config import SpecEdgeBatchServerConfig as config

    log_config = log.get_default_log_config(
        Path(config.result_path) / config.exp_name, "server"
    )
    log.configure_logging(log_config)

    try:
        controller = InferenceController(num_clients, recv_queue, resp_queue)
        controller.loop()
    except KeyboardInterrupt:
        # Gracefully exit without printing traceback
        pass
