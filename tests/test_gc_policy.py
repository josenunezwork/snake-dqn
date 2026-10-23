"""Round-25 startup GC contract, isolated from pytest's process heap."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def run_code(code: str, policy: str = "default") -> None:
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "SNAKE_SERVE_GC_POLICY": policy},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("policy", ["default", "freeze_startup"])
def test_startup_receipt_once_thresholds_and_freeze_invariant(policy: str) -> None:
    run_code(
        """
import gc
from web.backend import gc_policy as p
from unittest.mock import patch
thresholds = gc.get_threshold()
frozen_before = gc.get_freeze_count()
with patch.object(p.release_monitoring, "emit") as emit:
    receipt = p.apply_startup_policy(stage="test")
    emit.assert_called_once_with("gc_policy", **receipt)
assert set(receipt) == {"policy", "interpreter", "build", "thresholds", "enabled",
    "tracked_before", "frozen_count", "tracked_after", "collect2_result", "stage", "monotonic_ns"}
assert receipt["policy"] == p._policy
assert gc.get_threshold() == thresholds
assert gc.isenabled()
if p._policy == "freeze_startup":
    assert receipt["frozen_count"] > 0
else:
    assert receipt["frozen_count"] == frozen_before
before = gc.get_freeze_count()
for i in range(10):
    graph = []; graph.append(graph); del graph
    gc.collect(2)
    assert gc.get_freeze_count() == before
try:
    p.apply_startup_policy(stage="again")
except RuntimeError:
    pass
else:
    raise AssertionError("second call accepted")
""",
        policy,
    )


def test_default_never_collects_or_freezes() -> None:
    run_code("""
from unittest.mock import patch
from web.backend import gc_policy as p
with patch.object(p.gc, "collect", side_effect=AssertionError("collect")), \\
     patch.object(p.gc, "freeze", side_effect=AssertionError("freeze")):
    r = p.apply_startup_policy(stage="test")
assert r["policy"] == "default" and r["collect2_result"] is None
""")


@pytest.mark.parametrize("policy", ["", "freeze", "DEFAULT", " freeze_startup"])
def test_invalid_environment_refused(policy: str) -> None:
    run_code(
        """
from web.backend import gc_policy as p
try:
    p.apply_startup_policy(stage="test")
except ValueError:
    pass
else:
    raise AssertionError("invalid policy accepted")
assert not p._applied
""",
        policy,
    )


def test_callback_allocates_no_tracked_containers_and_drain_resets() -> None:
    run_code("""
import gc
from web.backend.gc_policy import GcTickMeter
m = GcTickMeter()
start, stop = "start", "stop"
info = {"generation": 2}
callback = m.callback
# Preallocate the range iterator and stabilize bookkeeping before the baseline.
it = iter(range(1000))
before = gc.get_count()[0]
for i in it:
    callback(start, info)
    callback(stop, info)
after = gc.get_count()[0]
assert after == before, (before, after)
r = m.drain()
assert r["gc_gen2_count"] == 1000 and r["gc_gen2_pause_ms"] > 0
assert m.drain()["gc_pause_ms"] == 0
m.install(); m.install()
assert gc.callbacks.count(m.hook) == 1
m.close()
assert m.hook not in gc.callbacks
""")


def test_gc_behavior_is_owned_only_by_policy_module() -> None:
    import re

    root = Path(__file__).resolve().parents[1]
    matches = []
    for directory in ("src", "web", "research/bl_k1"):
        for path in (root / directory).rglob("*.py"):
            if re.search(r"gc\.(?:freeze|disable|unfreeze|set_threshold)\s*\(", path.read_text()):
                matches.append(path.relative_to(root).as_posix())
    assert matches == ["web/backend/gc_policy.py"]


def test_lifespan_freezes_between_store_and_first_session() -> None:
    import inspect

    from web.backend.app import lifespan

    source = inspect.getsource(lifespan)
    assert (
        source.index("ScoreStore(")
        < source.index("apply_startup_policy(")
        < source.index("asyncio.to_thread(GameSession)")
    )


def test_worker_import_keeps_default_gc_even_when_parent_requests_freeze() -> None:
    run_code(
        """
import gc
from web.backend import gc_policy as p
assert not p._applied
assert p.state()["gc_policy"] == "default"
assert p.meter.hook not in gc.callbacks
""",
        "freeze_startup",
    )
