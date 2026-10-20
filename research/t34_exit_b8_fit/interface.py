"""Stable, deliberately small boundary between the ExIt engine and the fitter.

The production engine is built in the sibling worktree and is not imported at
module import time.  It must provide a factory returning :class:`ReplaySource`.
Each world is hash-verified by the engine *before* it is exposed to this
package.  A mismatching world is returned as ``verified=False`` and is dropped;
the loader reports its cumulative rate after the complete epoch.

The local fake below is solely a deterministic smoke fixture.  It has no game
semantics and cannot be used to make a research or gate claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch

EXPORT_SCHEMA = "exit-b8-export/v1"
REQUIRED_ROW_FIELDS = frozenset(
    {
        "local",
        "global",
        "scalars",
        "q_prev",
        "action_star",
        "forbidden_mask",
        "searched",
        "death_window",
        "decisive",
        "round_age",
        "length",
        "world_id",
    }
)


@dataclass(frozen=True)
class ReplayWorld:
    """One fully regenerated exported world and its hash verification result."""

    world_id: str
    rows: Sequence[Mapping[str, Any]]
    verified: bool
    hash_expected: str = ""
    hash_actual: str = ""
    reason: str = ""
    chunk_id: str = ""


class ReplaySource(Protocol):
    """Streaming producer; implementations must replay/regenerate per epoch.

    Row arrays stay immutable for their lifetime, including after the iterator advances;
    the bounded loader may retain a final partial batch while requesting the
    following world.  Engines that reuse backing buffers must yield owned rows.

    An optional ``training_row_counts(epoch) -> tuple[int, int]`` supplies the
    searched/unsearched counts of the verified retained stream, enabling exact
    bank-wide 1:1 sampling without another observation bank in memory.  Counts
    must exclude hash-rejected worlds.  Count drift invalidates the entire fit
    even when the ordinary world-drop rate is <=2%.  Without this metadata the
    loader requires each retained world to independently support a 1:1 sample.
    """

    is_fake: bool

    def replay_epoch(self, epoch: int) -> Iterable[ReplayWorld]:
        """Regenerate worlds for ``epoch``; never materialise the whole bank."""


class BatchModel(Protocol):
    """The tower and S1b models share this thin fitting-side call surface."""

    def forward_batch(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return raw Q values with shape ``(N, 6)``."""


def normalize_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Adapt explicit v2 replay labels; leave the v1 recipe byte-for-byte unchanged.

    The shadow measurement helper calls its *surviving* alternatives ``F``.
    Never feed that set to the forbidden-action hinge. For its LAD diagnostics,
    derive forbidden actions from tested per-action outcomes instead. A producer
    may supply an explicit forbidden_mask, whose semantics are unambiguous.
    This boundary consumes replayed observations, not unverified world exports.
    """
    kind = row.get("label_kind")
    if kind is None:
        return row
    if kind not in {"shadow_confirm", "lad", "transition"}:
        raise ValueError(f"unknown v2 label_kind: {kind}")
    result = dict(row)
    if kind == "shadow_confirm":
        result.update(
            searched=False,
            death_window=False,
            decisive=False,
            kl_anchor_only=True,
            action_star=int(np.asarray(row["q_prev"]).argmax()),
            forbidden_mask=np.zeros(6, dtype=bool),
        )
        return result
    if "t_star" not in row or "a_star" not in row or "F" not in row:
        raise ValueError("v2 LAD/transition labels require t_star, a_star, F")
    if type(row["t_star"]) is not int or row["t_star"] < 0:
        raise ValueError("v2 t_star must be a nonnegative decision index")
    if "decision" not in row or row["decision"] != row["t_star"]:
        raise ValueError("v2 label must be attached to its replayed t_star decision")
    action = row["a_star"]
    if type(action) is not int or not 0 <= action < 6:
        raise ValueError("v2 a_star must be an action in [0, 5]")
    # Amendment 2 exports diagnostic all-die rankings but assigns them no
    # training weight. Keep the historical LAD adapter unchanged for old banks.
    outcome_kind = row.get("outcome_kind")
    if outcome_kind is not None:
        if outcome_kind not in {"horizon_rescue", "all_die_delay", "bare_survives"}:
            raise ValueError("unknown v2 ranked outcome_kind")
        outcomes = row.get("per_action")
        if not isinstance(outcomes, Mapping) or str(action) not in outcomes:
            raise ValueError("ranked v2 rows require tested per_action outcomes")
        tested = {str(a): value for a, value in outcomes.items() if value.get("tested") is True}
        bare = int(row["student_action"])
        if str(action) not in tested or str(bare) not in tested:
            raise ValueError("ranked recommendation and bare control must be tested")

        def survives(value: Mapping[str, Any]) -> bool:
            return bool(value.get("survives", value.get("survive", False)))

        delay = outcome_kind == "all_die_delay"
        if delay and any(survives(value) for value in tested.values()):
            raise ValueError("all-die label contains a horizon survivor")
        if not delay and not survives(outcomes[str(action)]):
            raise ValueError("ranked recommendation must survive outside all-die rows")
        if outcome_kind == "horizon_rescue" and (
            str(bare) not in outcomes or survives(outcomes[str(bare)]) or action == bare
        ):
            raise ValueError("horizon rescue requires a dying bare action")
        if outcome_kind == "bare_survives" and action != bare:
            raise ValueError("surviving bare action must be retained")
        result.update(
            action_star=action,
            searched=True,
            death_window=True,
            decisive=outcome_kind == "horizon_rescue",
            kl_anchor_only=False,
            delay_only=delay,
            label_weight=0.0 if delay else 1.0,
            forbidden_mask=(
                np.asarray(
                    [str(a) in tested and not survives(tested[str(a)]) for a in range(6)],
                    dtype=bool,
                )
                if not delay
                else np.zeros(6, dtype=bool)
            ),
        )
        return result
    if "forbidden_mask" not in row:
        outcomes = row.get("per_action")
        if not isinstance(outcomes, Mapping) or str(action) not in outcomes:
            raise ValueError("v2 F is ambiguous: supply forbidden_mask or per_action outcomes")
        if not outcomes[str(action)].get("survives", outcomes[str(action)].get("survive")):
            raise ValueError("v2 a_star must survive when deriving forbidden actions")
        result["forbidden_mask"] = np.asarray(
            [
                str(a) in outcomes
                and not bool(outcomes[str(a)].get("survives", outcomes[str(a)].get("survive")))
                for a in range(6)
            ],
            dtype=bool,
        )
    result.update(
        action_star=action,
        searched=True,
        death_window=True,
        decisive=True,
        kl_anchor_only=False,
    )
    return result


def validate_row(row: Mapping[str, Any]) -> None:
    """Reject an incomplete export row before it can influence a fit."""
    missing = REQUIRED_ROW_FIELDS.difference(row)
    if missing:
        raise ValueError(f"exit-fit row missing required fields: {sorted(missing)}")
    q = np.asarray(row["q_prev"], dtype=np.float32)
    forbidden = np.asarray(row["forbidden_mask"], dtype=bool)
    if q.shape != (6,) or forbidden.shape != (6,):
        raise ValueError("q_prev and forbidden_mask must each contain six actions")
    if not np.isfinite(q).all() or not 0 <= int(row["action_star"]) < 6:
        raise ValueError("row has non-finite q_prev or invalid action_star")
    if forbidden[int(row["action_star"])]:
        raise ValueError("forbidden actions must exclude action_star")
    local, glob, scalars = (
        np.asarray(row["local"]),
        np.asarray(row["global"]),
        np.asarray(row["scalars"]),
    )
    if local.shape != (7, 31, 31) or glob.shape != (4, 37, 37) or scalars.shape != (60,):
        raise ValueError("P4 rows require ego2s-bv8 local/global/scalar shapes")
    if local.dtype != np.uint8 or glob.dtype != np.uint8 or not np.isfinite(scalars).all():
        raise ValueError("raster planes must be uint8 and scalars finite")
    if "full_sight" in row:
        full_sight = np.asarray(row["full_sight"])
        if full_sight.shape != (4, 88, 152) or full_sight.dtype != np.uint8:
            raise ValueError("full_sight must be uint8 (4, 88, 152)")
    if bool(row["decisive"]) and not bool(row["searched"]):
        raise ValueError("decisive labels must be searched decisions")
    if bool(row["death_window"]) and "death_window_id" not in row:
        raise ValueError("every death-window row requires its explicit window identifier")


def forward_batch(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Call either the P5 model API or the existing three-input S1b network."""
    method = getattr(model, "forward_batch", None)
    q = (
        method(batch)
        if callable(method)
        else model(batch["local"], batch["global"], batch["scalars"])
    )
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError(f"student model must return (N, 6) Q values, got {tuple(q.shape)}")
    return q


class FakeReplaySource:
    """Small deterministic fake for wiring/preflight tests only."""

    is_fake = True

    def __init__(self, worlds: int = 2, rows_per_world: int = 128, seed: int = 0) -> None:
        self.worlds, self.rows_per_world, self.seed = worlds, rows_per_world, seed

    def replay_epoch(self, epoch: int) -> Iterable[ReplayWorld]:
        # Regeneration must be stable across epochs; only loader ordering changes.
        rng = np.random.default_rng(self.seed)
        for w in range(self.worlds):
            rows = []
            for i in range(self.rows_per_world):
                q = rng.normal(size=6).astype(np.float32)
                action = int(q.argmax())
                forbidden = np.zeros(6, dtype=bool)
                if i % 5 == 0:
                    forbidden[(action + 1) % 6] = True
                rows.append(
                    {
                        "local": rng.integers(0, 255, (7, 31, 31), dtype=np.uint8),
                        "global": rng.integers(0, 255, (4, 37, 37), dtype=np.uint8),
                        "scalars": rng.normal(size=60).astype(np.float32),
                        "full_sight": rng.integers(0, 255, (4, 88, 152), dtype=np.uint8),
                        "q_prev": q,
                        "action_star": action,
                        "forbidden_mask": forbidden,
                        "searched": i % 2 == 0,
                        "death_window": i % 11 == 0,
                        "death_window_id": i // 11,
                        "decisive": i % 10 == 0,
                        "round_age": 0 if w == 0 else 1,
                        "length": int(rng.integers(5, 1300)),
                        "world_id": f"fake-{w}",
                    }
                )
            yield ReplayWorld(
                world_id=f"fake-{w}", rows=rows, verified=True, reason="FAKE_SMOKE_ONLY"
            )
