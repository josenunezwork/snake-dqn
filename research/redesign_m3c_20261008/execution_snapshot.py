"""Continuation state for AMP, compile fallback and frozen-policy wrappers."""

from __future__ import annotations

import copy
import pickle
from typing import Any


def capture_learner_execution(learner: Any) -> dict:
    """Called after flush: compiled callables are rebuilt, numerical choices survive."""
    scaler = getattr(learner, "scaler", None)
    return {
        "scaler": None if scaler is None else scaler.state_dict(),
        "amp_skipped": learner.amp_skipped,
        "which": getattr(learner, "_which", None),
        "compile": copy.deepcopy(getattr(learner, "_cstat", None)),
        "verified": copy.deepcopy(getattr(learner, "_verified", {})),
        "graph_count": (
            copy.deepcopy(vars(learner._gcount)) if hasattr(learner, "_gcount") else None
        ),
    }


def restore_learner_execution(learner: Any, state: dict) -> None:
    scaler = getattr(learner, "scaler", None)
    if (state.get("scaler") is None) != (scaler is None):
        raise ValueError("full-state AMP scaler configuration differs")
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    learner.amp_skipped = state.get("amp_skipped", 0)
    if state.get("which") is not None:
        learner._which = state["which"]
    saved = state.get("compile")
    if saved is None:
        if getattr(learner, "_cstat", None) is not None:
            raise ValueError("full-state learner compile configuration differs")
        return
    if getattr(learner, "_cstat", None) is None:
        raise ValueError("full-state learner compile configuration differs")
    learner._cstat.update(copy.deepcopy(saved))
    learner._verified = copy.deepcopy(state.get("verified", {}))
    if state.get("graph_count") is not None:
        vars(learner._gcount).update(copy.deepcopy(state["graph_count"]))
    if saved.get("fallback"):
        learner._loss_c = None
        learner._fwd_c = None


def capture_actor_execution(hero_forward: Any) -> dict | None:
    if hero_forward is None:
        return None
    return {
        "fallback": hero_forward.fallback,
        "calls": hero_forward.calls,
        "graph_count": copy.deepcopy(vars(hero_forward.count)),
        # Unused padded rows retain earlier observations and affect compiled probe/state.
        "stage": {name: array.copy() for name, array in hero_forward.stage.v.items()},
    }


def restore_actor_execution(hero_forward: Any, state: dict | None) -> None:
    if (hero_forward is None) != (state is None):
        raise ValueError("full-state actor compile configuration differs")
    if state is None:
        return
    hero_forward.calls = state["calls"]
    vars(hero_forward.count).update(copy.deepcopy(state["graph_count"]))
    for name, array in state["stage"].items():
        hero_forward.stage.v[name][...] = array
    if state["fallback"]:
        hero_forward.fallback = state["fallback"]
        hero_forward.fn = None


_POLICY_FIELDS = ("veto_counters", "boost_counters", "live_vetoes", "last_veto")


def capture_policy_execution(policies: dict) -> dict:
    """Wrappers contain mutable counters/hooks but immutable networks remain external."""
    return {
        key: {
            name: pickle.dumps(getattr(policy, name), protocol=pickle.HIGHEST_PROTOCOL)
            for name in _POLICY_FIELDS
            if hasattr(policy, name)
        }
        for key, policy in policies.items()
    }


def restore_policy_execution(policies: dict, state: dict) -> None:
    if set(policies) != set(state):
        raise ValueError("full-state opponent roster differs")
    for key, fields in state.items():
        for name, value in fields.items():
            setattr(policies[key], name, pickle.loads(value))
