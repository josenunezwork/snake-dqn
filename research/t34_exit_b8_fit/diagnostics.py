"""Held-out diagnostics and hard P4 fit tripwires.

The probe callback is intentionally external: it must run fresh held-out H5000
worlds and never receives gate worlds.  This module only validates its complete
trajectory records and makes selection deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


class FitTripwire(RuntimeError):
    """A registered fit tripwire fired; fitting must stop without retry."""


@dataclass(frozen=True)
class ProbeResult:
    survival: float
    trajectories: Sequence[Mapping[str, Any]]
    is_gate_data: bool = False


ProbeHook = Callable[[torch.nn.Module, int], ProbeResult]


def global_decile_edges(lengths: np.ndarray) -> np.ndarray:
    """Empirical global trajectory deciles, never fixed hand-written length bins."""
    if len(lengths) < 10:
        raise ValueError("full H5000 trajectory diagnostic needs at least ten decisions")
    return np.quantile(lengths.astype(np.float64), np.arange(1, 10) / 10.0)


def trajectory_diagnostics(
    *, previous_actions: np.ndarray, actions: np.ndarray, lengths: np.ndarray
) -> dict[str, Any]:
    """Compute all-length disagreement/boost stats over complete H5000 traces."""
    if not (len(previous_actions) == len(actions) == len(lengths)):
        raise ValueError("diagnostic action and length arrays must align")
    edges = global_decile_edges(lengths)
    actions = np.asarray(actions)
    previous_actions = np.asarray(previous_actions)
    if (
        not np.equal(actions, actions.astype(np.int64)).all()
        or not np.equal(previous_actions, previous_actions.astype(np.int64)).all()
    ):
        raise ValueError("trajectory actions must be integral")
    actions, previous_actions = actions.astype(np.int64), previous_actions.astype(np.int64)
    bins = np.searchsorted(edges, lengths, side="right")
    disagreement = actions != previous_actions
    per_decile = [
        float(disagreement[bins == d].mean()) if (bins == d).any() else 0.0 for d in range(10)
    ]
    return {
        "n": int(len(actions)),
        "decile_edges": edges.tolist(),
        "disagreement_by_global_length_decile": per_decile,
        "max_decile_disagreement": float(max(per_decile)),
        "boost_share": float((actions >= 3).mean()),
        "max_action_share": float(np.bincount(actions, minlength=6).max() / len(actions)),
        "previous_boost_share": float((previous_actions >= 3).mean()),
    }


def validate_probe(probe: ProbeResult, expected_worlds: int = 32) -> dict[str, Any]:
    """Require 32 complete, held-out H5000 trajectories; no vacuous pass."""
    if probe.is_gate_data:
        raise ValueError("selection probe must never use gate data")
    if len(probe.trajectories) != expected_worlds:
        raise ValueError(
            f"required {expected_worlds} held-out probe worlds, got {len(probe.trajectories)}"
        )
    if not np.isfinite(probe.survival) or not 0.0 <= probe.survival <= 1.0:
        raise ValueError("probe survival must be finite in [0, 1]")
    old, new, length, world_ids = [], [], [], set()
    for record in probe.trajectories:
        if int(record.get("horizon", 0)) != 5000 or not bool(record.get("complete", False)):
            raise ValueError("diagnostic probe requires complete H5000 trajectories")
        world_id = str(record.get("world_id", ""))
        if not world_id or world_id in world_ids:
            raise ValueError("probe needs unique held-out world identifiers")
        world_ids.add(world_id)
        prior, current, sizes = (
            record.get("previous_actions", ()),
            record.get("actions", ()),
            record.get("lengths", ()),
        )
        if len(prior) == 0 or len(prior) != len(current) or len(prior) != len(sizes):
            raise ValueError("each full trajectory needs nonempty aligned action/length traces")
        if (
            not np.isin(np.asarray(prior), np.arange(6)).all()
            or not np.isin(np.asarray(current), np.arange(6)).all()
        ):
            raise ValueError("trajectory actions must be integer actions 0..5")
        if not np.isfinite(np.asarray(sizes)).all():
            raise ValueError("trajectory lengths must be finite")
        old.extend(prior)
        new.extend(current)
        length.extend(sizes)
    return {
        "survival": float(probe.survival),
        **trajectory_diagnostics(
            previous_actions=np.asarray(old), actions=np.asarray(new), lengths=np.asarray(length)
        ),
    }


def enforce_tripwires(
    *, loss: float, heldout_decisive_agreement: float, probe: Mapping[str, Any]
) -> None:
    """Apply the exact P4 stop conditions, with no below-L600 exemption."""
    if not np.isfinite(loss):
        raise FitTripwire("non-finite fit loss")
    if heldout_decisive_agreement < 0.70:
        raise FitTripwire("held-out decisive agreement below 70%")
    if float(probe["max_action_share"]) > 0.90:
        raise FitTripwire("more than 90% one-action share on full H5000 trajectories")
    if float(probe["max_decile_disagreement"]) > 0.05:
        raise FitTripwire("disagreement exceeds 5% in a global length decile")
    old = float(probe["previous_boost_share"])
    if float(probe["boost_share"]) > 1.5 * old:
        raise FitTripwire("boost share exceeds 1.5x previous student")


def selection_key(metrics: Mapping[str, Any]) -> tuple[float, float, int]:
    """Fixed selection: decisive agreement, then fresh survival, then early epoch."""
    return (
        float(metrics["heldout_decisive_agreement"]),
        float(metrics["probe_survival"]),
        -int(metrics["epoch"]),
    )
