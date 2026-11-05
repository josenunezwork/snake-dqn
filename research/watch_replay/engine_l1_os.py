"""Replay controller ``exact-stop-l1-os``: exact-stop-l1 + open-stop on unchecked ticks.

Behaviour (Bank-v2, CAMPAIGN_LOG "Bank-v2 death census PRE-REGISTERED"):

* check ticks (the exact-stop-l1 clock: L>=1, every 5th episode decision) run the
  unchanged exact search, with horizon ``exact_horizon_for(L)`` = 300 for L < 800 and
  min(max(300, L), 1200) for L >= 800;
* the 4 unchecked ticks run open-stop-h32-nb (K=1, ``research.watch_replay.open_stop``)
  and execute its choice (an override whenever it differs from bare B8).

Why a wrapper instead of a branch inside ``engine.run_arm``: ``research/watch_replay/
engine.py`` and ``run.py`` are hashed into the frozen science identity of the
watch6_l1_confirm study (``protocol.science_paths``; pinned in
tests/test_watch6_l1_confirm.py and re-checked by its final_look/admission), so they
must stay byte-identical.  This module therefore drives the *unchanged* audited
``engine.run_arm`` loop (live Watch <-> Grid mirror, delivered-action and death
checks) for controller ``exact-stop-l1`` and only swaps two of its module globals for
the duration of the call:

* ``engine.DenseTrigger`` -> ``_EveryTickClock``: runs the real exact-stop-l1 clock,
  remembers whether it fired, and reports every live decision as a search root;
* ``engine.dense_batch`` -> a dispatcher: real check tick -> exact search (the
  original ``engine.dense_batch`` object when H == 300, so exact-stop-l1 behaviour is
  byte-identical there; ``dense_batch_h`` when H > 300); unchecked tick ->
  ``open_stop_batch``.

``run_arm`` then splits the engine's ``batches`` into exact roots and
``open_stop_batches``.
"""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import patch

import numpy as np

from research.bl_k1.native import RowStats, configure
from research.bl_served_search import ceiling, exact_stop
from research.bl_served_search.ruler_operator import _clone_root, _root_bytes
from research.t3_rollout_labels_20261015.bank_engine import rollout_all
from research.t3_rollout_labels_20261015.trigger import DenseTrigger
from research.watch_replay import engine
from research.watch_replay.open_stop import open_stop_batch

L1_CONTROLLER = "exact-stop-l1"
L1_OS_CONTROLLER = "exact-stop-l1-os"
EXACT_BASE_HORIZON = 300
# Exact-search horizon scaling: H = min(max(300, L), 1200) when L >= 800, else 300.
# Cap justification (t34-sol/bank-v2-build/SUMMARY.md):
#  * storage: product.open_watch sizes body storage at 14,704 = 14,400 session frames
#    + an H300 continuation (length grows <= 1 segment/frame), so every extra horizon
#    frame comes out of the session budget: l1-os sessions are limited to
#    14,400 + 300 - 1,200 = 13,500 frames (served banks use 6,000);
#  * wall: the replay generator's faulthandler watchdog kills a unit after 900 s
#    without a recorded frame; an all-die root costs <= 6 x H branch frames, ~15 s at
#    the measured ~2.05 ms/branch frame (flat in L), far inside 900 s;
#  * coverage/cost: a self-enclosure resolves once the tail has travelled one body
#    length, so H=L is the intended horizon; every Bank-v2 census death had
#    L <= 1,167, so the cap keeps H=L exactly for all of them while bounding a
#    long-body root at 4x the served H300 cost.
L1_OS_SCALE_MIN_LENGTH = 800
L1_OS_MAX_HORIZON = 1200
L1_OS_MAX_SESSION_FRAMES = 14400 + EXACT_BASE_HORIZON - L1_OS_MAX_HORIZON


def exact_horizon_for(controller: str, length: int) -> int:
    """Exact-search branch horizon: 300 except the l1-os long-body scaling rule."""
    if controller == L1_OS_CONTROLLER and int(length) >= L1_OS_SCALE_MIN_LENGTH:
        return min(max(EXACT_BASE_HORIZON, int(length)), L1_OS_MAX_HORIZON)
    return EXACT_BASE_HORIZON


def dense_batch_h(
    world: Any, decision: int, q: np.ndarray, legal: np.ndarray, horizon: int
) -> dict:
    """``engine.dense_batch`` with an explicit branch horizon (H300 is engine's own)."""
    if not EXACT_BASE_HORIZON < int(horizon) <= L1_OS_MAX_HORIZON:
        raise ValueError("scaled exact horizon outside (300, L1_OS_MAX_HORIZON]")
    root = _clone_root(world)
    before = _root_bytes(root)
    stats = RowStats()
    with configure("kernel", stats):
        result = exact_stop.run(
            np.flatnonzero(legal).tolist(),
            int(np.where(legal, q.astype(np.float64), -np.inf).argmax()),
            q,
            lambda action: rollout_all(
                _clone_root(root), [action], int(horizon), decision, record_outcomes=True
            )[action],
            ceiling._choose,
        )
    if _root_bytes(root) != before:
        raise ValueError("branch mutated its parent RNG/carry")
    counters = stats.as_dict()
    if (
        counters["kernel_rows"] <= 0
        or counters["stock_rows"] != 0
        or counters["fallback_rows"] != 0
        or counters["noncandidate_rows"] != 0
    ):
        raise ValueError("missing native kernel per-decision coverage")
    return {**result, "native_v8": counters, "root_sha256": ceiling.digest_value(before.hex())}


class _Pending:
    """Hand-off of the real clock verdict from fires() to the very next search call."""

    value: tuple[bool, int] | None = None


def _every_tick_clock(pending: _Pending) -> type:
    class _EveryTickClock(DenseTrigger):
        def fires(self, decision: int, length: int, alive: bool = True) -> bool:
            real = super().fires(decision, length, alive)
            pending.value = (bool(real), int(length))
            return bool(alive and length >= self.min_length)

    return _EveryTickClock


def run_arm(
    seed: int,
    check: Callable[[], None],
    horizon: int = 6000,
    record: Callable[[dict], None] | None = None,
) -> dict:
    """Run one exact-stop-l1-os session through the unchanged audited engine loop."""
    if not 1 <= horizon <= L1_OS_MAX_SESSION_FRAMES:
        raise ValueError("exact-stop-l1-os sessions are limited by scaled-H body storage")
    pending = _Pending()
    exact_h300 = engine.dense_batch  # the very object exact-stop-l1 calls

    def dispatch(world: Any, decision: int, q: np.ndarray, legal: np.ndarray) -> dict:
        if pending.value is None:
            raise RuntimeError("search root without a clock verdict")
        real, length = pending.value
        pending.value = None
        if real:
            exact_h = exact_horizon_for(L1_OS_CONTROLLER, length)
            batch = (
                exact_h300(world, decision, q, legal)
                if exact_h == EXACT_BASE_HORIZON
                else dense_batch_h(world, decision, q, legal, exact_h)
            )
            return {**batch, "kind": "exact", "exact_horizon": exact_h}
        ostop = open_stop_batch(world, decision, q, legal)
        bare = int(np.where(legal, q.astype(np.float64), -np.inf).argmax())
        if int(ostop["bare"]) != bare:
            raise ValueError("open-stop bare/Q mismatch")
        return {**ostop, "kind": "open_stop"}

    with (
        patch.object(engine, "DenseTrigger", _every_tick_clock(pending)),
        patch.object(engine, "dense_batch", dispatch),
    ):
        raw = engine.run_arm(seed, L1_CONTROLLER, check, horizon=horizon, record=record)
    batches = raw["batches"]
    if any(b.get("kind") not in ("exact", "open_stop") for b in batches):
        raise RuntimeError("unclassified search root")
    raw["batches"] = [b for b in batches if b["kind"] == "exact"]
    raw["open_stop_batches"] = [b for b in batches if b["kind"] == "open_stop"]
    raw["controller"] = L1_OS_CONTROLLER
    return raw


def run_controller(
    seed: int,
    controller: str,
    check: Callable[[], None],
    horizon: int = 6000,
    record: Callable[[dict], None] | None = None,
) -> dict:
    """Controller dispatch for screens: l1-os here, everything else via engine.run_arm."""
    if controller == L1_OS_CONTROLLER:
        return run_arm(seed, check, horizon=horizon, record=record)
    return engine.run_arm(seed, controller, check, horizon=horizon, record=record)
