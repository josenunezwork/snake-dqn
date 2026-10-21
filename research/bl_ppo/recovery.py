"""Complete explicit recovery of a boundary checkpoint; never retry a job."""

from __future__ import annotations

import copy
import dataclasses
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.bl_ppo.exports import verify_export

SCHEMA = "bl-ppo-full-snapshot/v1"


def _move(value: Any, device: str | torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if dataclasses.is_dataclass(value):
        return type(value)(
            **{f.name: _move(getattr(value, f.name), device) for f in dataclasses.fields(value)}
        )
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    if isinstance(value, deque):
        return deque((_move(v, device) for v in value), maxlen=value.maxlen)
    if isinstance(value, list):
        return [_move(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move(v, device) for v in value)
    return copy.deepcopy(value)


def capture_full_snapshot(
    policy: Any,
    learner: Any,
    collector: Any,
    adapter: Any,
    counters: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Capture only between complete rollouts, including the actual burn-in history."""
    if getattr(adapter, "records", []):
        raise ValueError("checkpoint requires a completed rollout boundary")
    adapter_state = {k: v for k, v in vars(adapter).items() if k not in {"policy", "device"}}
    return {
        "schema": SCHEMA,
        "identity": copy.deepcopy(identity),
        "policy": policy.checkpoint_dict(),
        "learner": _move(learner.learner_state_dict(), "cpu"),
        "collector": collector.snapshot_state(),
        "adapter": _move(adapter_state, "cpu"),
        "counters": copy.deepcopy(counters),
        "rng": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }


def read_snapshot(path: Path) -> dict[str, Any]:
    """Verify immutable sidecars before loading the trusted local checkpoint."""
    verify_export(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") != SCHEMA:
        raise ValueError("unrecognized complete checkpoint schema")
    required = {"identity", "policy", "learner", "collector", "adapter", "counters", "rng"}
    if not required.issubset(value):
        raise ValueError("checkpoint is missing complete resume state")
    return value


def restore_full_snapshot(
    path: Path,
    policy: Any,
    learner: Any,
    collector: Any,
    adapter: Any,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Restore all execution state after checking the exact run/input identity."""
    value = read_snapshot(path)
    if value["identity"] != identity:
        raise ValueError("resume identity differs from the frozen run")
    if value["adapter"].get("records"):
        raise ValueError("cannot resume an incomplete on-policy rollout")
    rng = value["rng"]
    if set(rng) != {"torch_cpu", "torch_cuda_all", "numpy", "python"}:
        raise ValueError("checkpoint RNG inventory incomplete")
    if len(rng["torch_cuda_all"]) != (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    ):
        raise ValueError("resume CUDA device inventory differs")
    for key in ("arm", "temperature", "b8_config"):
        if key in value["policy"] and getattr(policy, key, None) != value["policy"][key]:
            raise ValueError(f"resume policy metadata differs: {key}")
    policy.load_state_dict(value["policy"]["state_dict"])
    learner.load_learner_state_dict(value["learner"])
    collector.restore_state(value["collector"])
    for key in list(vars(adapter)):
        if key not in {"policy", "device"}:
            delattr(adapter, key)
    for key, item in _move(value["adapter"], adapter.device).items():
        setattr(adapter, key, item)
    # Restore RNG last: model construction / process startup must not perturb the next draw.
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if rng["torch_cuda_all"]:
        torch.cuda.set_rng_state_all(rng["torch_cuda_all"])
    return copy.deepcopy(value["counters"])
