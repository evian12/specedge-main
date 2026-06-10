import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def _model_files(model_path: Path):
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as index_file:
            index = json.load(index_file)
        return sorted(
            {model_path / name for name in index["weight_map"].values()}
        )

    single_file = model_path / "model.safetensors"
    if single_file.exists():
        return [single_file]

    files = sorted(model_path.glob("model-*.safetensors"))
    if files:
        return files

    raise FileNotFoundError(f"No safetensors weights found in {model_path}")


def main(model_path: Path):
    affected = []
    total_tensors = 0
    total_values = 0
    total_non_finite = 0

    for weight_file in _model_files(model_path):
        with safe_open(weight_file, framework="pt", device="cpu") as handle:
            for tensor_name in handle.keys():
                tensor = handle.get_tensor(tensor_name)
                total_tensors += 1
                total_values += tensor.numel()

                invalid_mask = ~torch.isfinite(tensor)
                invalid_count = int(invalid_mask.sum().item())
                if not invalid_count:
                    continue

                total_non_finite += invalid_count
                affected.append(
                    {
                        "file": weight_file.name,
                        "tensor": tensor_name,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "non_finite": invalid_count,
                        "nan": int(torch.isnan(tensor).sum().item()),
                        "positive_inf": int(torch.isposinf(tensor).sum().item()),
                        "negative_inf": int(torch.isneginf(tensor).sum().item()),
                    }
                )

    print(
        json.dumps(
            {
                "model_path": str(model_path),
                "total_tensors": total_tensors,
                "total_values": total_values,
                "total_non_finite": total_non_finite,
                "affected": affected,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()
    main(args.model_path)
