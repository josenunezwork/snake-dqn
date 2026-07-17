"""Exit-code contract for the kill-diagnosis orchestrator.

The script runs exactly one arm then exits, so its exit code IS its result: a
wrapper (or a human reading `$?`) has nothing else to go on. A gate that failed
must not exit 0 — that reports a broken arm as a clean one, and silently voids
the --smoke pipeline check.
"""

import json
from pathlib import Path
from typing import Dict, List

import pytest

from src.scripts import run_kill_diagnosis

# The exact per-candidate failure shape tournament_eval emits: no "per_mix" key,
# written to the JSON, and the process still exits 0 (see tournament_eval.main).
ERRORED_GATE = {
    "mode": "eval",
    "candidates": [
        {
            "candidate": "logs/diag/A/latest_apex.pth",
            "error": "size mismatch for backbone.0.weight",
            "decision": {"promote": False, "significant_mixes": [], "reasons": []},
        }
    ],
}

GOOD_GATE = {
    "mode": "eval",
    "candidates": [
        {
            "candidate": "logs/diag/A/latest_apex.pth",
            "per_mix": {
                "scripted": {
                    "summary": {"kills": 2.0, "probes": {"kill_opportunity_count": 7.0}},
                    "paired": {"mean_delta": 5.0},
                }
            },
            "decision": {"promote": True, "significant_mixes": ["scripted"], "reasons": []},
        }
    ],
}


@pytest.fixture
def stub_arm(monkeypatch: pytest.MonkeyPatch):
    """Replace both subprocess commands with no-op stubs writing the given gate."""

    def _install(payload: Dict[str, object]) -> None:
        def fake_train(
            arm: str, config: str, total_steps: int, num_actors: int, out_dir: Path
        ) -> List[str]:
            (out_dir / "latest_apex.pth").write_text("stub")
            return ["true"]

        def fake_eval(checkpoint: Path, frames: int, seeds: str, out_json: Path) -> List[str]:
            out_json.write_text(json.dumps(payload))
            return ["true"]

        monkeypatch.setattr(run_kill_diagnosis, "_train_command", fake_train)
        monkeypatch.setattr(run_kill_diagnosis, "_eval_command", fake_eval)

    return _install


class TestMainExitCode:
    """main() must propagate a failed gate as a non-zero exit code."""

    def test_failed_eval_exits_nonzero(
        self, tmp_path: Path, stub_arm, capsys: pytest.CaptureFixture
    ) -> None:
        stub_arm(ERRORED_GATE)

        rc = run_kill_diagnosis.main(["--arm", "C", "--smoke", "--out-dir", str(tmp_path)])

        # Exact code, not `!= 0`: a main() returning None satisfies `!= 0` and would
        # let the silent-exit-0 regression back in.
        assert rc == 1
        # The guard's whole point: the error is still recorded, not just raised away.
        verdict = json.loads((tmp_path / "verdict.json").read_text())
        assert "size mismatch" in str(verdict["error"])
        assert "EVAL FAILED" in capsys.readouterr().err

    def test_successful_eval_exits_zero(self, tmp_path: Path, stub_arm) -> None:
        stub_arm(GOOD_GATE)

        rc = run_kill_diagnosis.main(["--arm", "C", "--smoke", "--out-dir", str(tmp_path)])

        assert rc == 0
        verdict = json.loads((tmp_path / "verdict.json").read_text())
        assert "error" not in verdict
        assert verdict["kills_per_episode"] == 2.0

    def test_unreadable_gate_json_exits_nonzero(self, tmp_path: Path, stub_arm) -> None:
        stub_arm({})

        rc = run_kill_diagnosis.main(["--arm", "C", "--smoke", "--out-dir", str(tmp_path)])

        assert rc == 1
