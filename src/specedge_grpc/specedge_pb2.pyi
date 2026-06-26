from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SyncRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class SyncResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class ValidateRequest(_message.Message):
    __slots__ = ["attention_mask", "cache_seq_indices", "client_idx", "input_ids", "parent_indices", "position_ids", "prefill", "prefix", "req_idx"]
    ATTENTION_MASK_FIELD_NUMBER: _ClassVar[int]
    CACHE_SEQ_INDICES_FIELD_NUMBER: _ClassVar[int]
    CLIENT_IDX_FIELD_NUMBER: _ClassVar[int]
    INPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    PARENT_INDICES_FIELD_NUMBER: _ClassVar[int]
    POSITION_IDS_FIELD_NUMBER: _ClassVar[int]
    PREFILL_FIELD_NUMBER: _ClassVar[int]
    PREFIX_FIELD_NUMBER: _ClassVar[int]
    REQ_IDX_FIELD_NUMBER: _ClassVar[int]
    attention_mask: bytes
    cache_seq_indices: bytes
    client_idx: int
    input_ids: bytes
    parent_indices: bytes
    position_ids: bytes
    prefill: bool
    prefix: str
    req_idx: int
    def __init__(self, client_idx: _Optional[int] = ..., req_idx: _Optional[int] = ..., input_ids: _Optional[bytes] = ..., position_ids: _Optional[bytes] = ..., cache_seq_indices: _Optional[bytes] = ..., parent_indices: _Optional[bytes] = ..., attention_mask: _Optional[bytes] = ..., prefill: bool = ..., prefix: _Optional[str] = ...) -> None: ...

class ValidateResponse(_message.Message):
    __slots__ = ["background_arrival_rate", "batch_size", "decode_ms", "model_decode_ms", "prefill", "prefill_ms", "queue_length", "queue_wait_ms", "selection", "server_compute_ms", "server_response_ms"]
    BACKGROUND_ARRIVAL_RATE_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    DECODE_MS_FIELD_NUMBER: _ClassVar[int]
    MODEL_DECODE_MS_FIELD_NUMBER: _ClassVar[int]
    PREFILL_FIELD_NUMBER: _ClassVar[int]
    PREFILL_MS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_LENGTH_FIELD_NUMBER: _ClassVar[int]
    QUEUE_WAIT_MS_FIELD_NUMBER: _ClassVar[int]
    SELECTION_FIELD_NUMBER: _ClassVar[int]
    SERVER_COMPUTE_MS_FIELD_NUMBER: _ClassVar[int]
    SERVER_RESPONSE_MS_FIELD_NUMBER: _ClassVar[int]
    background_arrival_rate: float
    batch_size: int
    decode_ms: float
    model_decode_ms: float
    prefill: int
    prefill_ms: float
    queue_length: int
    queue_wait_ms: float
    selection: bytes
    server_compute_ms: float
    server_response_ms: float
    def __init__(self, selection: _Optional[bytes] = ..., prefill: _Optional[int] = ..., queue_wait_ms: _Optional[float] = ..., server_compute_ms: _Optional[float] = ..., server_response_ms: _Optional[float] = ..., decode_ms: _Optional[float] = ..., model_decode_ms: _Optional[float] = ..., prefill_ms: _Optional[float] = ..., batch_size: _Optional[int] = ..., queue_length: _Optional[int] = ..., background_arrival_rate: _Optional[float] = ...) -> None: ...
