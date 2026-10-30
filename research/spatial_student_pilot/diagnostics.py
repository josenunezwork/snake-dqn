"""Amendment07 sealed diagnostics; never used for training or verdict selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from research.sandbox_a_resident.core import allowed_mask
from research.spatial_student_pilot.batching import stable_q
from research.spatial_student_pilot.observation import build_observation, compact_cat


def policy_record(logits: Sequence[float], mask: Sequence[bool]) -> dict[str, Any]:
    """Keep raw logits and the legal top-two gap (zero for a sole legal action)."""
    values, legal = np.asarray(logits, dtype=float), np.asarray(mask, dtype=bool)
    if values.shape != (6,) or legal.shape != (6,) or not legal.any():
        raise ValueError("malformed assessment logits/mask")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite assessment logits")
    order = np.sort(values[legal])
    return dict(
        student_logits=values.tolist(),
        legal_mask=legal.tolist(),
        student_action=int(np.where(legal, values, -np.inf).argmax()),
        student_top2_margin=float(order[-1] - order[-2]) if len(order) > 1 else 0.0,
    )


def label_confidence(row: Mapping[str, Any]) -> str:
    """Selection-only paired two-SE split, fixed before independent scoring."""
    mass = np.asarray(row["selection_mass"], dtype=float)
    legal = np.asarray(row["legal_mask"], dtype=bool)
    action = int(row["teacher_action"])
    if mass.shape != (6, 2) or legal.shape != (6,) or not 0 <= action < 6 or not legal[action]:
        raise ValueError("malformed selection confidence inputs")
    if not np.isfinite(mass[legal]).all():
        raise ValueError("nonfinite legal selection returns")
    for other in np.flatnonzero(legal):
        if other != action:
            paired = mass[action] - mass[other]
            se = float(paired.std(ddof=1) / np.sqrt(2))
            if not float(paired.mean()) > 2 * se:
                return "low"
    return "high"


@torch.inference_mode()
def heldout_bc_agreement(
    student: Any,
    roots: Sequence[Mapping[str, Any]],
    training_parents: set[int],
    *,
    device: str,
    microbatch: int,
) -> dict[str, Any]:
    """Predict once at the BC boundary; assessment rows never enter optimization."""
    if any(r["partition"] != "assessment" or r["world_id"] in training_parents for r in roots):
        raise ValueError("held-out BC diagnostic requires disjoint assessment parents")
    if len({r["uid"] for r in roots}) != len(roots):
        raise ValueError("duplicate held-out BC root")
    was_training = student.training
    student.eval()
    predictions = []
    try:
        for start in range(0, len(roots), microbatch):
            chunk = roots[start : start + microbatch]
            obs = build_observation(compact_cat([r["compact"] for r in chunk]), device=device)
            mask = torch.stack([torch.as_tensor(r["mask"]) for r in chunk]).to(device).bool()
            mask = allowed_mask(mask)
            logits = stable_q(student, obs, student(*obs), mask)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite held-out BC logits")
            actions = logits.masked_fill(~mask, -torch.inf).argmax(1).cpu().tolist()
            predictions.extend(
                dict(uid=r["uid"], world_id=int(r["world_id"]), action=int(a))
                for r, a in zip(chunk, actions)
            )
    finally:
        student.train(was_training)
    return bc_summary(roots, predictions)


def bc_summary(
    roots: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Reconstruct auditable counts from frozen assessment-root predictions."""
    expected = {r["uid"]: r for r in roots}
    if len(expected) != len(roots) or len(predictions) != len(expected):
        raise ValueError("held-out BC support differs")
    if {p["uid"] for p in predictions} != set(expected):
        raise ValueError("held-out BC UIDs differ")
    parents: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for prediction in predictions:
        root = expected[prediction["uid"]]
        action = prediction["action"]
        if (
            prediction["world_id"] != root["world_id"]
            or type(action) is not int
            or not 0 <= action < 6
            or not bool(root["mask"][action])
        ):
            raise ValueError("held-out BC prediction identity/action differs")
        parents[int(root["world_id"])][0] += int(action == root["b8_action"])
        parents[int(root["world_id"])][1] += 1
    return dict(
        schema="spatial-heldout-bc/v1",
        predictions=list(predictions),
        by_parent=[
            dict(world_id=p, agreed=parents[p][0], total=parents[p][1]) for p in sorted(parents)
        ],
        agreed=sum(v[0] for v in parents.values()),
        total=len(roots),
    )
