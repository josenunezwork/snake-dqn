"""Exact boundary-state restoration for the E16 runner."""

from __future__ import annotations

import copy
import random
from typing import Any, Mapping

import numpy as np
import torch


def restore_collectors(collectors: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    """Replace each live collector with its explicit snapshot state."""
    if set(collectors) != set(saved):
        raise ValueError("collector phase inventory differs from checkpoint")
    for name, collector in collectors.items():
        restored = type(collector).restore(collector.bank, saved[name])
        collector.__dict__.clear()
        collector.__dict__.update(restored.__dict__)


def restore_snapshot(
    policy: Any, trainer: Any, collectors: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Restore policy, both optimizers, live worlds, counters and every RNG stream."""
    required = {"policy", "trainer", "collector", "counters", "rng"}
    if not required.issubset(payload):
        raise ValueError("incomplete E16 snapshot")
    policy_state = payload["policy"].get("state_dict")
    if not isinstance(policy_state, Mapping):
        raise ValueError("snapshot lacks portable policy state")
    policy.load_state_dict(policy_state)
    state = payload["trainer"]
    trainer.actor_optimizer.load_state_dict(state["actor"])
    trainer.critic_optimizer.load_state_dict(state["critic"])
    trainer.refit_transitions = int(state["refit_transitions"])
    trainer.e12_passed = bool(state["e12_passed"])
    restore_collectors(collectors, payload["collector"])
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    cuda = rng.get("cuda", [])
    if cuda:
        if not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG inventory differs")
        torch.cuda.set_rng_state_all(cuda)
    return copy.deepcopy(payload["counters"])
