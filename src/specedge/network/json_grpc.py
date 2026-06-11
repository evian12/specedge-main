import json
from typing import Any


SERVICE_NAME = "specedge.NetworkAutoregressive"
STREAM_METHOD = f"/{SERVICE_NAME}/StreamGenerate"


def serialize_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def deserialize_json(value: bytes) -> dict[str, Any]:
    return json.loads(value.decode("utf-8"))
