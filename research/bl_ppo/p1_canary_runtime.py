"""Operational P1 family-canary execution using the real runtime components.

This is transport evidence only.  It intentionally cannot qualify the critic,
the Y3 source run, or any scientific claim.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.bl_ppo.exports import export_json, export_torch, verify_export
from research.bl_ppo.recovery import capture_full_snapshot, restore_full_snapshot

SCHEMA = "bl-p1-family-canary/v1"
_REQUIRED = {
    "worlds": 2,
    "processes": 1,
    "rollout_steps": 16,
    "pause_at": 32,
}
_SEED = 2026102203


def load_config(raw: str, *, control_sha256: str, excluded_seeds: list[int]) -> dict[str, Any]:
    """Validate the explicit, tiny canary configuration.

    A P1 scientific control document is never a substitute for this isolated
    transport configuration.  Its four keys are its complete public surface;
    source binding and the seed inventory come from the already-validated P1
    control document.
    """
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("BL_P1_CANARY_CONFIG must contain JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("canary configuration must be a JSON object")
    if not isinstance(control_sha256, str) or len(control_sha256) != 64:
        raise ValueError("canary requires a verified source control digest")
    if set(value) != set(_REQUIRED):
        raise ValueError("canary configuration may contain only its fixed transport shape")
    for key, expected in _REQUIRED.items():
        if type(value.get(key)) is not int or value.get(key) != expected:
            raise ValueError(f"canary configuration must freeze {key}={expected!r}")
    if _SEED == 20261021 or _SEED in excluded_seeds:
        raise ValueError("canary seed must be distinct from P1 and excluded evaluation seeds")
    return value


def execute(
    *, control: dict[str, Any], control_path: Path, output: Path, checkpoint: Path
) -> dict[str, Any]:
    """Run one real 32-transition rollout and prove a complete paused-state restore."""
    if os.environ.get("SNAKE_COLAB_CANARY") != "1":
        raise ValueError("family canary is enabled only by SNAKE_COLAB_CANARY=1")
    if os.environ.get("SNAKE_COLAB_CANARY_PACKAGE") != "research/bl_ppo":
        raise ValueError("family canary requires SNAKE_COLAB_CANARY_PACKAGE=research/bl_ppo")
    if control.get("phase") != "y3-source" or control.get("scientific_admission") is not True:
        raise ValueError("family canary requires the validated y3-source control")
    raw = os.environ.get("BL_P1_CANARY_CONFIG")
    if not raw:
        raise ValueError("SNAKE_COLAB_CANARY=1 requires BL_P1_CANARY_CONFIG")
    config = load_config(
        raw,
        control_sha256=verify_export(control_path),
        excluded_seeds=list(control["evaluation_excluded_seeds"]),
    )
    canary_dir = output / "canary"
    canary_dir.mkdir(parents=True, exist_ok=False)
    from research.bl_ppo.collector import CollectorConfig, PersistentCollector
    from research.bl_ppo.learner import PPOLearner
    from research.bl_ppo.model import PPOPolicy
    from research.bl_ppo.run import PersistentPolicyAdapter

    device = os.environ.get("BL_PPO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("canary requested CUDA but it is unavailable")
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    policy = PPOPolicy("F", checkpoint, 1.0)
    adapter = PersistentPolicyAdapter(policy, device)
    learner = PPOLearner(policy)
    identity = {
        "mode": "p1-family-canary",
        "source_control_sha256": verify_export(control_path),
        "source_commit": control["source_commit"],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "seed": _SEED,
        "worlds": 2,
        "processes": 1,
        "rollout_steps": 16,
        "actor_enabled": False,
    }
    collector_config = CollectorConfig(
        worlds=2,
        processes=1,
        rollout_steps=16,
        seed=_SEED,
        roster="frozen-equal-mixes",
        native_v8=True,
        # The collector's explicit operational mode is the only supported
        # two-world shape.  This flag also prevents a tiny transport run from
        # being mistaken for an E2/P1 scientific collection.
        preflight=True,
        excluded_seeds=frozenset(control["evaluation_excluded_seeds"]),
    )
    counters: dict[str, Any] = {
        "actor_transitions": 0,
        "updates": 0,
        "actor_enabled": False,
        "collection_seconds": 0.0,
        "learner_seconds": 0.0,
    }
    started = time.monotonic()
    native_v8_actual = None
    with PersistentCollector(collector_config) as collector:
        raw_batch = collector.collect(adapter)
        counters["collection_seconds"] += raw_batch.wall_seconds
        learned = time.monotonic()
        # This is a real critic-only PPO update.  It is intentionally not the
        # P1 critic gate and never marks a critic as qualified.
        learner.update(adapter.take_rollout(raw_batch, collector.peek()), actor_enabled=False)
        adapter.refresh_hidden()
        counters["learner_seconds"] += time.monotonic() - learned
        counters["actor_transitions"] += int(raw_batch.active_transitions)
        counters["updates"] += 1
        if counters["actor_transitions"] != 32 or adapter.records:
            raise ValueError(
                "canary did not stop at the complete 32-transition boundary: "
                f"observed={counters['actor_transitions']}"
            )
        instrumented = collector.snapshot_state()
        native_v8_actual = instrumented[0]["frozen"].native_v8_actual
        if native_v8_actual != "s2-native-v8":
            raise ValueError("canary requires actual native_v8 roster instrumentation")
        snapshot = canary_dir / "pause-snapshot.pt"
        export_torch(
            snapshot,
            capture_full_snapshot(
                policy,
                learner,
                collector,
                adapter,
                counters,
                identity,
                boundary={"kind": "exact", "pause_at": 32, "actual_pause_at": 32},
            ),
            max_payload_bytes=control.get("max_snapshot_bytes"),
        )
        restored = restore_full_snapshot(snapshot, policy, learner, collector, adapter, identity)
    if restored != counters:
        raise ValueError("canary snapshot restore counters differ")
    pause = {
        "schema": "bl-p1-family-canary-pause/v1",
        "status": "PAUSED",
        "actor_transitions": 32,
        "snapshot": {"name": snapshot.name, "sha256": verify_export(snapshot)},
        "restore_verified": True,
        "transitive_components": ["policy", "optimizer", "collector", "adapter", "rng", "counters"],
    }
    export_json(canary_dir / "pause.json", pause)
    result = {
        "schema": SCHEMA,
        "status": "PAUSED",
        "purpose": "transport-family-canary",
        "scientific_result": False,
        "critic_qualified": False,
        "y3_source_admitted": False,
        "source_control_sha256": verify_export(control_path),
        "source_commit": control["source_commit"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "native_v8_actual": native_v8_actual,
        "config": {**{key: config[key] for key in sorted(_REQUIRED)}, "seed": _SEED},
        "pause": pause,
        "runtime_seconds": time.monotonic() - started,
        "export_scope": (
            "runtime payload and sidecars only; supervisor transport accounting is external"
        ),
    }
    export_json(output / "receipt.json", result)
    return result
