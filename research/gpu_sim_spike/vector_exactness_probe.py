"""Replay the immutable failed A100 state through both repaired feature paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from research.gpu_sim_spike import sparse_bench, sparse_units
from research.gpu_sim_spike.benchmark import source_commit, source_hashes
from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.vector61_featurizer import Vector61Params


def load_fixture(
    path: Path, device: str
) -> tuple[DeviceSim, Vector61Params, np.ndarray, np.ndarray]:
    """Load full ring capacity and ordered food without altering fixture bytes."""
    with np.load(path, allow_pickle=False) as saved:
        cfg = replace(BatchSimConfig(**json.loads(str(saved["config_json"]))), num_envs=1)
        params_dict = json.loads(str(saved["params_json"]))
        params_dict["actions"] = tuple(map(tuple, params_dict["actions"]))
        params = Vector61Params(**params_dict)
        sim = DeviceSim(cfg, device=device, cap=saved["bodies"].shape[1])
        for name in (
            "bodies",
            "head_ptr",
            "seg_count",
            "length",
            "direction",
            "alive",
            "food_cells",
            "food_present",
            "food_count",
        ):
            setattr(sim, name, torch.as_tensor(saved[name][None].copy(), device=device))
        return sim, params, saved["prev_dist"][None].copy(), saved["prev_id"][None].copy()


def probe(path: Path, device: str) -> dict:
    """Require all columns and pixel-distance carry to agree on the recorded input."""
    from types import SimpleNamespace

    from research.gpu_sim_spike.vector_geometry import IndexedVectorRuntime

    records = []
    for backend in ("dense", "static"):
        os.environ["GPU_SIM_VECTOR_FLOOD"] = backend
        for kind in ("direct", "indexed"):
            sim, params, prev_dist, prev_id = load_fixture(path, device)
            runtime = DeviceVectorRuntime(params)
            runtime.bind(sim)
            runtime.prev_dist[:] = torch.as_tensor(prev_dist, device=device)
            runtime.prev_id[:] = torch.as_tensor(prev_id, device=device)
            if kind == "indexed":
                os.environ["GPU_SIM_VECTOR_GEOMETRY"] = "native"
                runtime = IndexedVectorRuntime.from_runtime(runtime)
            result = sparse_bench._byte_oracle(sim, SimpleNamespace(vector_runtime=runtime))
            expected, memory = sparse_bench._features_cpu_oracle(sim, runtime, with_memory=True)
            state_before = {
                key: value.clone()
                for key, value in sim.__dict__.items()
                if isinstance(value, torch.Tensor)
            }
            runtime_before = {
                key: value.clone()
                for key, value in runtime.__dict__.items()
                if isinstance(value, torch.Tensor) and key not in {"prev_dist", "prev_id"}
            }
            actual = runtime.features(sim, update_memory=True).cpu().numpy()
            carry_match = (
                runtime.prev_dist.cpu().numpy().tobytes() == memory.prev_dist.tobytes()
                and runtime.prev_id.cpu().numpy().tobytes() == memory.prev_id.tobytes()
            )
            state_preserved = all(
                torch.equal(value, getattr(sim, key)) for key, value in state_before.items()
            )
            runtime_preserved = all(
                torch.equal(value, getattr(runtime, key)) for key, value in runtime_before.items()
            )
            result["unrelated_runtime_tensors_preserved"] = runtime_preserved
            result["carry_match"] = carry_match
            result["simulator_tensors_preserved"] = state_preserved
            result["match"] = (
                result["match"]
                and carry_match
                and runtime_preserved
                and state_preserved
                and actual.tobytes() == expected.tobytes()
            )
            records.append({"backend": backend, "runtime": kind, **result})
    return {
        "schema": "gpu-vector61-exactness-probe/v1",
        "status": "PASS" if all(row["match"] for row in records) else "INVALID",
        "device": device,
        "commit": source_commit(),
        "source_sha256": source_hashes(),
        "native_cuda_executed": device == "cuda",
        "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    value = probe(args.fixture, args.device)
    sparse_units.publish_json(args.output, value)
    if value["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
