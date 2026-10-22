"""Complete explicit recovery of a boundary checkpoint; never retry a job."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import random
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from research.bl_ppo.byte_bounds import component_limit
from research.bl_ppo.exports import export_bytes, export_torch, verify_export

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


_IMMUTABLE_REF = "__bl_ppo_immutable_state_ref__"


def _verify_immutable_within_bound(path: Path, checksum: str | None, cap: int) -> str:
    """Verify an existing immutable sidecar before reusing it in a snapshot."""
    if path.stat().st_size > cap:
        raise ValueError("immutable component exceeds its predeclared export byte bound")
    observed = verify_export(path)
    if checksum is not None and observed != checksum:
        raise ValueError("immutable component digest differs")
    return observed


def _adapter_snapshot_state(
    adapter: Any, immutable_dir: Path | None, byte_bounds: Mapping[str, Any] | None
) -> dict[str, Any]:
    state = {k: v for k, v in vars(adapter).items() if k not in {"policy", "device"}}
    if immutable_dir is None:
        return state
    if not isinstance(state.get("critic_refit_receipt"), dict):
        # Before qualification these banks are mutable. Each cadence capture
        # gets a distinct content-addressed immutable copy; an old copy is never
        # reused merely because its role/name is unchanged.
        for key, bound_name in (
            ("refit_bank", "refit_bank"),
            ("ordinary_bank", "ordinary_bank"),
            ("fixture_late_body_bank", "fixture_late_body_bank"),
        ):
            if key not in state:
                continue
            cap = component_limit(byte_bounds, bound_name) if byte_bounds is not None else None
            buffer = io.BytesIO()
            torch.save(_move(state[key], "cpu"), buffer)
            payload = buffer.getvalue()
            if cap is not None and len(payload) > cap:
                raise ValueError(f"pre-actor {key} exceeds frozen immutable component bound")
            checksum = hashlib.sha256(payload).hexdigest()
            target = Path(immutable_dir) / f"immutable-partial-{key}-{checksum[:16]}.pt"
            if target.exists():
                if cap is not None:
                    _verify_immutable_within_bound(target, checksum, cap)
                elif verify_export(target) != checksum:
                    raise ValueError("pre-actor immutable bank digest differs")
            else:
                export_bytes(target, payload)
            state[key] = {_IMMUTABLE_REF: {"name": target.name, "sha256": checksum}}
        return state
    from research.bl_ppo.bank import canonical_sha

    static: dict[str, tuple[str, str]] = {
        "refit_bank": (canonical_sha(state["critic_refit_receipt"]), "refit_bank")
    }
    ordinary = state.get("ordinary_bank")
    if getattr(ordinary, "frozen", False):
        static["ordinary_bank"] = (ordinary.sha256, "ordinary_bank")
    if "fixture_late_body_bank" in state:
        fixture_identity = hashlib.sha256()
        fixture_buffer = io.BytesIO()
        torch.save(_move(state["fixture_late_body_bank"], "cpu"), fixture_buffer)
        fixture_identity.update(fixture_buffer.getvalue())
        static["fixture_late_body_bank"] = (
            fixture_identity.hexdigest(),
            "fixture_late_body_bank",
        )
    for key, (identity, bound_name) in static.items():
        if key not in state:
            continue
        cap = component_limit(byte_bounds, bound_name) if byte_bounds is not None else None
        target = Path(immutable_dir) / f"immutable-{key}-{identity[:16]}.pt"
        if target.exists():
            checksum = (
                _verify_immutable_within_bound(target, None, cap)
                if cap is not None
                else verify_export(target)
            )
        else:
            exported = export_torch(target, _move(state[key], "cpu"), max_payload_bytes=cap)
            checksum = exported["sha256"]
        state[key] = {_IMMUTABLE_REF: {"name": target.name, "sha256": checksum}}
    return state


def _restore_adapter_value(value: Any, snapshot_path: Path) -> Any:
    if not isinstance(value, dict) or _IMMUTABLE_REF not in value:
        return value
    reference = value[_IMMUTABLE_REF]
    if set(value) != {_IMMUTABLE_REF} or not isinstance(reference, dict):
        raise ValueError("invalid immutable recovery reference")
    name = reference.get("name")
    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
        raise ValueError("unsafe immutable recovery reference")
    target = Path(snapshot_path).parent / name
    if verify_export(target) != reference.get("sha256"):
        raise ValueError("immutable recovery reference digest differs")
    return torch.load(target, map_location="cpu", weights_only=False)


def snapshot_files(path: Path) -> list[Path]:
    """Verify the complete flat snapshot closure, including its original Y3 origin."""
    paths: dict[str, Path] = {}
    visiting: set[Path] = set()

    def load(payload: Path, expected: str | None = None) -> None:
        checksum = verify_export(payload)
        if expected is not None and checksum != expected:
            raise ValueError("snapshot transitive reference digest differs")
        if payload.name in paths:
            if verify_export(paths[payload.name]) != checksum:
                raise ValueError("conflicting flattened snapshot basename")
            return
        if payload.resolve() in visiting:
            raise ValueError("cyclic snapshot reference")
        visiting.add(payload.resolve())
        paths[payload.name] = payload
        value = torch.load(payload, map_location="cpu", weights_only=False)

        def reference(row: Any) -> None:
            if not isinstance(row, dict):
                raise ValueError("invalid snapshot reference")
            name = row.get("name")
            if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
                raise ValueError("unsafe snapshot reference basename")
            load(payload.parent / name, row.get("sha256"))

        def visit(item: Any) -> None:
            if isinstance(item, dict):
                if set(item) == {_IMMUTABLE_REF}:
                    reference(item[_IMMUTABLE_REF])
                else:
                    for child in item.values():
                        visit(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    visit(child)

        visit(value)
        if isinstance(value, dict):
            origin = value.get("operational", {}).get("execution", {}).get("origin_snapshot")
            if origin is not None:
                reference(origin)
        visiting.remove(payload.resolve())

    load(Path(path))
    return list(paths.values())


def copy_snapshot_closure(path: Path, target: Path) -> Path:
    """Copy verified immutable exports to a fresh flat transport directory."""
    for source in snapshot_files(path):
        destination = Path(target) / source.name
        if destination.exists():
            if verify_export(destination) != verify_export(source):
                raise ValueError("conflicting flattened snapshot basename")
        else:
            export_bytes(destination, source.read_bytes())
    return Path(target) / path.name


def capture_full_snapshot(
    policy: Any,
    learner: Any,
    collector: Any,
    adapter: Any,
    counters: dict[str, Any],
    identity: dict[str, Any],
    *,
    immutable_dir: Path | None = None,
    boundary: dict[str, Any] | None = None,
    operational: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
    byte_bounds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture only between complete rollouts, including the actual burn-in history."""
    if getattr(adapter, "records", []):
        raise ValueError("checkpoint requires a completed rollout boundary")
    adapter_state = _adapter_snapshot_state(adapter, immutable_dir, byte_bounds)
    return {
        "schema": SCHEMA,
        **({"boundary": copy.deepcopy(boundary)} if boundary is not None else {}),
        **({"operational": copy.deepcopy(operational)} if operational is not None else {}),
        **({"execution": copy.deepcopy(execution)} if execution is not None else {}),
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
        setattr(adapter, key, _move(_restore_adapter_value(item, path), adapter.device))
    # Restore RNG last: model construction / process startup must not perturb the next draw.
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if rng["torch_cuda_all"]:
        torch.cuda.set_rng_state_all(rng["torch_cuda_all"])
    return copy.deepcopy(value["counters"])
