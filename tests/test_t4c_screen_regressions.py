"""Failure-path and provenance regressions without gameplay or model inference."""

import json
from argparse import Namespace

import numpy as np
import pytest

from research.t3_screen_20261015 import diagnostics_t4c as diagnostic
from research.t3_screen_20261015 import drift_t4c, watch_t4c


def gameplay(survival=0.1, mass=0, failed=False):
    return {
        "arms": {
            diagnostic.ARM: {
                "decision": "KILL" if failed else "GO",
                "failed_runs": ["failed"] if failed else [],
                "vs_ctrl": {
                    mix: {"mass": {"mean": mass}, "episode_survival": {"mean": survival}}
                    for mix in ("scripted", "frozen", "mixed")
                },
            }
        }
    }


def test_exact_declared_fork_and_uncovered_mass_failure():
    fell = {"within_ten_percentage_points": True}
    assert diagnostic.declared_fork(gameplay(), fell) == (
        "(c) falls to the drift baseline and round-65 survival moves without a mass fail: "
        "expert iteration with the tightened recipe is next, as planned."
    )
    assert diagnostic.declared_fork(gameplay(survival=0), fell) == (
        "(c) falls but survival does not move: return regression next, with outside-bank "
        "branches rolled out too, which is the principled form of item 2 and supplies "
        '"greedy is fine here" targets with magnitudes.'
    )
    assert diagnostic.declared_fork(gameplay(), {"within_ten_percentage_points": False}) == (
        "(c) does not fall even with outside-bank counters: the boundary is not expressible "
        "from bv8 at this capacity, and the campaign moves to inputs."
    )
    assert "No declared fork" in diagnostic.declared_fork(gameplay(mass=-31), fell)
    assert "nonzero training exit" in diagnostic.declared_fork(gameplay(failed=True), fell)


def test_fingerprint_bank_self_consistent_tamper_fails_pinned_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_t4c.drift, "POSTHOC", tmp_path / "trajectory")
    bank = tmp_path / "bank"
    bank.mkdir()
    path = bank / "TWO_SIDED_s10.npz"
    receipt = bank / "TWO_SIDED_s10.receipt.json"

    def write(value):
        np.savez(path, fingerprint=np.array([value]))
        receipt.write_text(json.dumps({"out": str(path), "out_sha256": drift_t4c.sha256(path)}))

    write("original")
    summary = {"evidence_sha256": {str(p): drift_t4c.sha256(p) for p in (path, receipt)}}
    assert drift_t4c._bank_fingerprints(summary).tolist() == ["original"]
    write("changed")
    with pytest.raises(ValueError, match="binding mismatch"):
        drift_t4c._bank_fingerprints(summary)


def test_finish_halted_run_writes_finite_kill_report_without_inference(tmp_path, monkeypatch):
    args = Namespace(
        out=tmp_path,
        train=tmp_path / "train",
        ctrl=tmp_path / "ctrl",
        results=tmp_path / "result.md",
    )
    monkeypatch.setattr(diagnostic, "load_drift_summary", lambda: {"schema": "synthetic"})
    monkeypatch.setattr(diagnostic, "read_telemetry", lambda roots: {})
    monkeypatch.setattr(
        diagnostic,
        "diagnose",
        lambda *args: {
            "drift": {"within_ten_percentage_points": None},
        },
    )

    def forbidden(*args, **kwargs):
        pytest.fail("halted run must not invoke drift inference")

    monkeypatch.setattr(drift_t4c, "capture", forbidden)
    monkeypatch.setattr(drift_t4c, "summarize", forbidden)
    assert watch_t4c._finish(args, [], {"TWO_SIDED_T_s10": 1}, {"pass": True}) == 0
    result = json.loads((tmp_path / "analysis.json").read_text())
    assert result["arms"][diagnostic.ARM]["decision"] == "KILL"
    assert "Decision: KILL" in args.results.read_text()
    assert "NaN" not in (tmp_path / "analysis.json").read_text()
