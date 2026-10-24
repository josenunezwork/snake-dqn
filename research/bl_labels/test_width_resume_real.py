"""Operator-gated real three-parent proof for the non-production W1->W2 seam.

Run only after D256 through the pinned Mac dispatcher, which starts a detached
worktree and wraps pytest in the shared-lease guard command:
``tools/ops/dispatch.sh --mac f0-width-real-proof <candidate-pin> -- \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m \
  research.bl_labels.mac_driver guard-command --state-dir \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/width-real-proof \
  --seconds 7200 -- -- env BL_RUN_REAL_WIDTH_PROOF=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -p no:xdist \
  -m slow research/bl_labels/test_width_resume_real.py``.
It is deliberately skipped in ordinary development and never broadens the
production D1024/D4096 W6/W12 policy.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from research.bl_labels import mac_driver
from research.bl_labels.exports import aggregate_attempts, empty_ledger, publish_json
from research.bl_labels.speed_equiv import compare_payloads

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("BL_RUN_REAL_WIDTH_PROOF") != "1",
        reason="operator-only real W1->W2 resume proof",
    ),
]

B8 = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth")
FRP3 = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/frp3_m3_s12_u60000_20261005.pth")


def _allocation() -> list[dict[str, object]]:
    return [
        {
            "allocation_id": f"watch6-proof-{index}",
            "seed": 9100 + index,
            "stream": "watch6",
            "horizon": 8,
        }
        for index in range(3)
    ]


def _payloads(root: Path) -> list[dict[str, object]]:
    return [json.loads(path.read_text()) for path in sorted((root / "lives").glob("*.json"))]


def test_real_three_parent_w1_to_w2_resume_matches_uninterrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collect real labels with a parent-boundary preempt, then compare T-EQ payloads."""
    if not B8.is_file() or not FRP3.is_file():
        pytest.skip("pinned B8/FRP3 checkpoints are unavailable")
    monkeypatch.setattr(mac_driver, "_guard_snapshot", lambda _output: {"real_proof": True})
    monkeypatch.setattr(mac_driver, "_rss_many", lambda pids: {pid: 0 for pid in pids})
    monkeypatch.setattr(mac_driver, "_width_change_allowed", lambda *_args: True)
    control = {"namespace": "f0-width-real-proof-v1", "allocation": _allocation()}
    empty = tmp_path / "empty-ledger.json"
    publish_json(empty, empty_ledger(control["namespace"], control["allocation"]))
    armed = tmp_path / "proof-timing-armed"
    stop = threading.Event()

    def arm_after_first_session(output: Path) -> None:
        while not stop.is_set():
            if list((output / "lives").glob("*--session.json.complete")):
                armed.write_text("preempt at completed-parent boundary\n")
                return
            time.sleep(0.02)

    split = tmp_path / "split-w1"
    watcher = threading.Thread(target=arm_after_first_session, args=(split,), daemon=True)
    watcher.start()
    first = mac_driver.run(
        control,
        empty,
        B8,
        FRP3,
        split,
        workers=1,
        timing_armed=armed,
        deadline_seconds=3600,
        state_dir=tmp_path / "split-state",
    )
    stop.set()
    watcher.join(timeout=1)
    assert first["status"] == "PREEMPTED"
    resumed_ledger = tmp_path / "resumed-ledger.json"
    aggregate_attempts([split], resumed_ledger, control["namespace"], control["allocation"])
    armed.unlink()
    resumed = tmp_path / "resumed-w2"
    second = mac_driver.run(
        control,
        resumed_ledger,
        B8,
        FRP3,
        resumed,
        workers=2,
        timing_armed=armed,
        deadline_seconds=3600,
        state_dir=tmp_path / "split-state",
    )
    assert second["status"] == "COMPLETE"
    uninterrupted = tmp_path / "uninterrupted"
    baseline_ledger = tmp_path / "baseline-ledger.json"
    publish_json(baseline_ledger, empty_ledger(control["namespace"], control["allocation"]))
    baseline = mac_driver.run(
        control,
        baseline_ledger,
        B8,
        FRP3,
        uninterrupted,
        workers=1,
        timing_armed=armed,
        deadline_seconds=3600,
        state_dir=tmp_path / "baseline-state",
    )
    assert baseline["status"] == "COMPLETE"
    comparison = compare_payloads(
        [*_payloads(split), *_payloads(resumed)], _payloads(uninterrupted), compare_pairs=False
    )
    assert comparison["status"] == "PASS", comparison
    assert all(item["old"] == item["new"] for item in comparison["payload_hashes"].values())
