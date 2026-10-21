"""Opt-in native v8 opponent veto for T34 ExIt *branch* rows (speed program S2).

Scope (JUDGMENT 2026-10-20 §3 S2, §5 D3)
---------------------------------------
Frozen v8 opponents (``Vector61SimdPolicy`` with ``veto_variant="v8"``) decide through the
live :class:`~src.evaluation.safety_veto_v8.SpaceAndHeadVeto` hook objects, one Python
call per opponent row (~1 ms each on the Mac).  :func:`src.simd_env.v8_nb.v8_decide` is
the compiled, decision-identical form (``tests/test_v8_nb.py``).  The only state the live
hook carries is per-episode *bookkeeping* (probe counters and diagnostics); it never feeds
back into a decision.

Who consumes that bookkeeping in T34:

* **Root, shadow and doom-cache reconstruction rows** — yes: ``research.t34_shadow.shadow.
  full_digest`` hashes every opponent policy's ``live_vetoes`` hook state, and the shadow's
  confirm/invalidate control compares those digests.  These rows therefore keep the
  Python hook (unchanged; nothing here touches them).
* **Branch rows** (``ExitTeacher._rollout_merged``: verification/expansion continuations
  and the v2 bisection probes) — no: each call builds private per-branch policy clones
  (``engine._clone_merged_policy``), steps them, reads only hero survival/mass/row counts,
  and discards the clones.  Their hook counters are never read, digested or exported.

So :class:`NativeV8ExitTeacher` swaps the decision kernel for branch rows only, by giving
each *private merged clone* an instance-level ``_apply_veto`` that runs ``v8_decide``.
Every other step of the dispatch (advisory mask + normal-only fallback, the rowwise fp32
forward recipe, masked first-argmax base, vector61 trend-memory carry, compaction, the
CUDA slab path) is the unchanged stock code.  No ratified file is edited: a run is on this
path only when its caller constructs :class:`NativeV8ExitTeacher` (or calls
:func:`run_worlds_native`).

Fail-closed: the native base argmax must equal the stock torch argmax on every row, and
``audit=`` runs the stock Python hook beside the kernel on every row and raises
:class:`NativeV8Mismatch` on any difference in action, base, v2 counts or ``need``.
"""

from __future__ import annotations

import functools
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from research.t34_exit_b8.engine import ExitTeacher, ExitTeacherConfig

NATIVE_V8_RECIPE = "t34-native-v8-branch/v1"
_INSTALLED = "_t34_native_v8_installed"


class NativeV8Mismatch(RuntimeError):
    """The compiled v8 kernel disagreed with the stock path on a branch row."""


@dataclass
class NativeV8Audit:
    """Row-level differential counters (``audit`` mode) plus exercised-case coverage."""

    calls: int = 0
    rows: int = 0
    mismatches: int = 0
    overrides: int = 0
    v5_changes: int = 0
    v7_changes: int = 0
    head_changes: int = 0
    no_spacious: int = 0
    tie_rows: int = 0
    tie_overrides: int = 0
    boost_base_rows: int = 0
    long_rows: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def is_native_v8_candidate(policy: Any) -> bool:
    """A vector61 policy whose vetoed rows run the live v8 hook."""
    spec = getattr(policy, "veto_spec", None)
    return (
        bool(getattr(policy, "veto_slots", frozenset()))
        and getattr(policy, "veto_variant", None) == "v8"
        and spec is not None
        and getattr(spec, "lam", None) is not None
        and hasattr(policy, "runtime")
    )


def _tie_rows(q_rows: np.ndarray, mask_rows: np.ndarray) -> np.ndarray:
    """Rows where two masked-legal float32 Q values are exactly equal."""
    out = np.zeros(len(q_rows), dtype=bool)
    for i, (q, m) in enumerate(zip(q_rows, mask_rows)):
        legal = np.asarray(q, dtype=np.float32)[np.asarray(m, dtype=bool)]
        out[i] = len(legal) != len(np.unique(legal))
    return out


def native_apply_veto(
    policy: Any,
    sim: Any,
    slots: np.ndarray,
    masked_q: Any,
    mask: np.ndarray,
    actions: np.ndarray,
    *,
    audit: NativeV8Audit | None = None,
    membership: str = "painted",
) -> np.ndarray:
    """``Vector61SimdPolicy._apply_veto`` for ``veto_variant="v8"`` on the v8_nb kernel.

    Same inputs and return as the stock method.  ``masked_q`` is the stock
    ``where(mask, q, INVALID_Q_VALUE)`` tensor and ``actions`` its first argmax.
    """
    from src.simd_env.v8_nb import v8_decide
    from src.simd_env.vector61_policy import check_live_rule_config, spacious_from_counts

    slots = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
    pick = np.flatnonzero(np.isin(slots[:, 1], list(policy.veto_slots)))
    if not len(pick):
        return actions
    if getattr(policy, "_rule_config_checked", None) is not sim:
        check_live_rule_config(sim)  # the stock hook's once-per-sim guard, kept
        policy._rule_config_checked = sim
    rows = slots[pick]
    q_all = masked_q.cpu().numpy() if hasattr(masked_q, "cpu") else np.asarray(masked_q)
    q_rows = np.ascontiguousarray(q_all[pick], dtype=np.float32)
    mask_rows = np.ascontiguousarray(np.asarray(mask, dtype=bool)[pick])
    featurizer = policy.runtime.featurizer
    res = v8_decide(
        sim,
        rows,
        q_rows,
        mask_rows,
        float(policy.veto_spec.lam),
        grid_shape=(int(featurizer.gw), int(featurizer.gh)),
        membership=membership,
    )
    base = np.asarray(actions, dtype=np.int64)[pick]
    if not np.array_equal(res.base, base):
        raise NativeV8Mismatch(
            f"native base argmax differs from the stock argmax on rows {rows.tolist()}"
        )
    final = res.action.astype(np.int64)
    if audit is not None:
        _audit_rows(policy, sim, slots, masked_q, mask, actions, pick, rows, res, audit)
    policy.last_veto = {
        "rows": rows.copy(),
        "frame": sim.frame[rows[:, 0]].copy(),
        "counts": res.counts,
        "need": res.need,
        "spacious": spacious_from_counts(res.counts, res.need),
        "base": base.copy(),
        "final": final,
        "native": NATIVE_V8_RECIPE,
    }
    out = np.asarray(actions).copy()
    out[pick] = final
    return out


def _audit_rows(
    policy: Any,
    sim: Any,
    slots: np.ndarray,
    masked_q: Any,
    mask: np.ndarray,
    actions: np.ndarray,
    pick: np.ndarray,
    rows: np.ndarray,
    res: Any,
    audit: NativeV8Audit,
) -> None:
    """Run the stock Python v8 hook on the same rows; raise on any difference."""
    stock = type(policy)._apply_veto(policy, sim, slots, masked_q, mask, actions)
    reference = policy.last_veto
    expected = np.asarray(stock, dtype=np.int64)[pick]
    bad = res.action != expected
    for name in ("counts", "need", "base"):
        if not np.array_equal(np.asarray(getattr(res, name)), np.asarray(reference[name])):
            audit.mismatches += len(rows)
            raise NativeV8Mismatch(f"native {name} differs from the stock hook")
    audit.calls += 1
    audit.rows += len(rows)
    if bad.any():
        audit.mismatches += int(bad.sum())
        raise NativeV8Mismatch(
            f"native v8 action differs on rows {rows[bad].tolist()}: "
            f"native {res.action[bad].tolist()} stock {expected[bad].tolist()}"
        )
    q_all = masked_q.cpu().numpy() if hasattr(masked_q, "cpu") else np.asarray(masked_q)
    ties = _tie_rows(q_all[pick], np.asarray(mask, dtype=bool)[pick])
    over = res.action != res.base
    audit.overrides += int(over.sum())
    audit.v5_changes += int((res.v5_action != res.base).sum())
    audit.v7_changes += int((res.v7_action != res.v5_action).sum())
    audit.head_changes += int((res.action != res.v7_action).sum())
    audit.no_spacious += int((res.outcome == 2).sum())
    audit.tie_rows += int(ties.sum())
    audit.tie_overrides += int((ties & over).sum())
    audit.boost_base_rows += int((res.base >= 3).sum())
    audit.long_rows += int((sim.length[rows[:, 0], rows[:, 1]] >= 400).sum())


def install_native_v8(
    policy: Any, *, audit: NativeV8Audit | None = None, membership: str = "painted"
) -> bool:
    """Give one *private* policy object an instance-level native ``_apply_veto``.

    Only call this on disposable copies (the branch engine's merged clones): the
    instance attribute changes that object's veto bookkeeping, which the root and shadow
    digests read.  Returns whether it was installed.
    """
    if not is_native_v8_candidate(policy):
        return False
    if getattr(policy, _INSTALLED, False):
        return True
    policy._apply_veto = functools.partial(
        native_apply_veto, policy, audit=audit, membership=membership
    )
    setattr(policy, _INSTALLED, True)
    return True


class NativeV8ExitTeacher(ExitTeacher):
    """:class:`ExitTeacher` whose merged *branch* rows run v8 opponents natively.

    Root, shadow and reconstruction rows use ``ExitTeacher.actions`` (the stock
    dispatcher, untouched); only ``_dispatch_merged``, which is called solely from
    ``_rollout_merged`` on its private policy clones, is extended.
    """

    def __init__(
        self,
        config: ExitTeacherConfig = ExitTeacherConfig(),
        *,
        audit: bool = False,
        membership: str = "painted",
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        if membership not in ("painted", "owner_pad"):
            raise ValueError("membership must be 'painted' or 'owner_pad'")
        self.native_audit = NativeV8Audit() if audit else None
        self.native_membership = membership
        self.native_installs = 0

    def _dispatch_merged(self, merged: Any, grid: np.ndarray, active: np.ndarray) -> None:
        seen: set[int] = set()
        for row in merged.opponents:
            for policy in row:
                if id(policy) in seen:
                    continue
                seen.add(id(policy))
                if not getattr(policy, _INSTALLED, False) and install_native_v8(
                    policy, audit=self.native_audit, membership=self.native_membership
                ):
                    self.native_installs += 1
        super()._dispatch_merged(merged, grid, active)


def make_native_teacher(config: Mapping[str, Any], *, audit: bool = False) -> NativeV8ExitTeacher:
    """The teacher ``teacher_v2.run_worlds`` would build, with native branch v8."""
    return NativeV8ExitTeacher(
        ExitTeacherConfig(
            horizon=int(config.get("horizon", 300)), beta=float(config.get("beta", 1.0))
        ),
        device=str(config.get("device", "cpu")),
        policy=config.get("policy"),
        audit=audit,
        membership=str(config.get("native_v8_membership", "painted")),
    )


def run_worlds_native(config: Mapping[str, Any], *, audit: bool = False) -> dict[str, Any]:
    """``teacher_v2.run_worlds`` with the opt-in native branch teacher.

    The receipt gains ``native_v8`` (recipe, installs, audit counters); the world
    exports are written by the unchanged collector.
    """
    from research.t34_exit_b8.teacher_v2 import run_worlds

    if config.get("teacher") is not None:
        raise ValueError("run_worlds_native builds its own teacher")
    teacher = make_native_teacher(config, audit=audit)
    receipt = run_worlds({**dict(config), "teacher": teacher})
    receipt["native_v8"] = {
        "recipe": NATIVE_V8_RECIPE,
        "scope": "ExitTeacher._rollout_merged branch rows only",
        "membership": teacher.native_membership,
        "installs": teacher.native_installs,
        "audit": None if teacher.native_audit is None else teacher.native_audit.as_dict(),
    }
    return receipt
