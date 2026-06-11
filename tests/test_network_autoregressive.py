import json
import tempfile
import unittest
from pathlib import Path

from metric.network_autoregressive import summarize
from specedge.network.json_grpc import deserialize_json, serialize_json


class NetworkAutoregressiveTest(unittest.TestCase):
    def test_json_codec_round_trip(self):
        payload = {
            "req_idx": 7,
            "prompt": "hello",
            "max_new_tokens": 64,
        }

        self.assertEqual(deserialize_json(serialize_json(payload)), payload)

    def test_metric_summary_uses_client_end_to_end_time(self):
        records = [
            {
                "generated_tokens": 2,
                "ttft_ms": 100.0,
                "tpot_ms": 25.0,
                "end_to_end_ms": 125.0,
                "server_prefill_ms": 80.0,
                "server_decode_ms": [10.0, 10.0],
                "delivery_overhead_ms": [10.0, 15.0],
            },
            {
                "generated_tokens": 3,
                "ttft_ms": 200.0,
                "tpot_ms": 50.0,
                "end_to_end_ms": 300.0,
                "server_prefill_ms": 150.0,
                "server_decode_ms": [20.0, 20.0, 20.0],
                "delivery_overhead_ms": [20.0, 20.0, 20.0],
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir)
            log_path = result_path / "network_ar_client_0.jsonl"
            with log_path.open("w") as file:
                for record in records:
                    file.write(json.dumps(record) + "\n")

            summary = summarize(result_path)

        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["tokens"], 5)
        self.assertAlmostEqual(
            summary["tokens_per_second"],
            5 * 1000 / 425,
        )
        self.assertEqual(summary["ttft_mean"], 150.0)
        self.assertEqual(summary["server_decode_mean"], 16.0)


if __name__ == "__main__":
    unittest.main()
