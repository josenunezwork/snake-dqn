"""Robustness tests for the sweep orchestrator.

The sweep babysits multi-hour training jobs, so the two failure modes that matter
are (a) one bad config raising out of the loop and orphaning the rest, and (b) a
run that halted on a tripwire being ranked as if it were clean.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

from src.scripts import run_kill_diagnosis, sweep

# The exact per-candidate failure shape tournament_eval emits: no "per_mix" key,
# written to the JSON, and the process still exits 0 (see tournament_eval.main).
ERRORED_GATE = {
    "mode": "eval",
    "candidates": [
        {
            "candidate": "runs/sweep/kill_scale0.3/latest_pqn.pth",
            "error": "size mismatch for backbone.0.weight",
            "decision": {
                "promote": False,
                "significant_mixes": [],
                "reasons": ["size mismatch for backbone.0.weight"],
            },
        }
    ],
}


def _good_gate(mass_integral: float = 120.0) -> Dict[str, object]:
    """A gate JSON with a healthy per_mix block."""
    return {
        "mode": "eval",
        "candidates": [
            {
                "candidate": "runs/sweep/x/latest_pqn.pth",
                "per_mix": {
                    "scripted": {
                        "summary": {
                            "mass_integral": mass_integral,
                            "kills": 2.0,
                            "survival_fraction": 0.5,
                            "probes": {"boost_frame_fraction": 0.25},
                        },
                        "paired": {"mean_delta": 5.0},
                    }
                },
                "decision": {"promote": True, "significant_mixes": ["scripted"], "reasons": []},
            }
        ],
    }


def _args(tmp_path: Path, grid: List[str], parallel: int = 2) -> argparse.Namespace:
    """A Namespace with every field run_sweep reads."""
    return argparse.Namespace(
        grid=grid,
        out_dir=str(tmp_path),
        device="cpu",
        total_steps=1,
        envs=1,
        snakes=1,
        rollout_len=1,
        ckpt_every=1,
        parallel=parallel,
        threads=1,
        gate_frames=1,
        gate_seeds=1,
    )


@pytest.fixture(autouse=True)
def _no_poll_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Don't burn 5s per poll in tests."""
    monkeypatch.setattr(sweep, "POLL_SECONDS", 0)


def _stub_train(monkeypatch: pytest.MonkeyPatch, exit_code: int, write_ckpt: bool = True) -> None:
    """Replace the train command with a stub exiting exit_code (ckpt written first)."""

    def fake_cmd(combo: Dict[str, str], args: argparse.Namespace, out_dir: Path) -> List[str]:
        if write_ckpt:
            (out_dir / "latest_pqn.pth").write_text("stub")
        return [sys.executable, "-c", f"raise SystemExit({exit_code})"]

    monkeypatch.setattr(sweep, "build_train_cmd", fake_cmd)


def _stub_gate(monkeypatch: pytest.MonkeyPatch, payload: Dict[str, object]) -> None:
    """Replace the gate command with a stub that writes payload and exits 0."""

    def fake_cmd(ckpt: Path, args: argparse.Namespace, out_json: Path) -> List[str]:
        out_json.write_text(json.dumps(payload))
        return [sys.executable, "-c", "pass"]

    monkeypatch.setattr(sweep, "build_gate_cmd", fake_cmd)


class TestSummarizeGate:
    """summarize_gate must never raise on a gate JSON it is handed."""

    def test_errored_candidate_returns_error_not_raise(self, tmp_path: Path) -> None:
        gate_json = tmp_path / "gate.json"
        gate_json.write_text(json.dumps(ERRORED_GATE))

        out = sweep.summarize_gate(gate_json)

        assert "error" in out
        assert "size mismatch" in str(out["error"])
        assert "mass_integral" not in out

    def test_missing_candidates_returns_error(self, tmp_path: Path) -> None:
        gate_json = tmp_path / "gate.json"
        gate_json.write_text(json.dumps({"mode": "eval", "candidates": []}))

        assert "error" in sweep.summarize_gate(gate_json)

    def test_malformed_json_returns_error(self, tmp_path: Path) -> None:
        gate_json = tmp_path / "gate.json"
        gate_json.write_text("{not json")

        assert "error" in sweep.summarize_gate(gate_json)

    def test_healthy_gate_still_summarizes(self, tmp_path: Path) -> None:
        gate_json = tmp_path / "gate.json"
        gate_json.write_text(json.dumps(_good_gate(mass_integral=120.0)))

        out = sweep.summarize_gate(gate_json)

        assert out == {
            "mass_integral": 120.0,
            "kills": 2.0,
            "boost": 0.25,
            "survival": 0.5,
        }


class TestRunSweepSurvivesFailures:
    """A single bad config must not abort the sweep and orphan the other runs."""

    def test_errored_candidate_records_entry_and_finishes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_train(monkeypatch, exit_code=0)
        _stub_gate(monkeypatch, ERRORED_GATE)

        results = sweep.run_sweep(_args(tmp_path, ["kill_scale=0.3,0.6"]))

        assert len(results) == 2
        assert all("error" in r for r in results)

    def test_gate_exception_does_not_abort_the_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_train(monkeypatch, exit_code=0)
        _stub_gate(monkeypatch, _good_gate())

        def boom(gate_json: Path) -> Dict[str, object]:
            raise RuntimeError("kaboom")

        monkeypatch.setattr(sweep, "summarize_gate", boom)

        results = sweep.run_sweep(_args(tmp_path, ["kill_scale=0.3,0.6"]))

        assert len(results) == 2
        assert all("kaboom" in str(r["error"]) for r in results)

    def test_leaderboard_written_when_every_config_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        _stub_train(monkeypatch, exit_code=0)
        _stub_gate(monkeypatch, ERRORED_GATE)

        results = sweep.run_sweep(_args(tmp_path, ["kill_scale=0.3"]))
        sweep.write_leaderboard(results, tmp_path)

        assert json.loads((tmp_path / "leaderboard.json").read_text())
        assert "Winner:" not in capsys.readouterr().out


class TestTripwireHandling:
    """train_pqn exits 2 on a tripwire but still saves the checkpoint."""

    def test_tripped_run_is_flagged_on_the_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_train(monkeypatch, exit_code=sweep.TRIPWIRE_RC)
        _stub_gate(monkeypatch, _good_gate())

        results = sweep.run_sweep(_args(tmp_path, ["kill_scale=0.3"]))

        assert results[0]["train_rc"] == 2
        assert results[0]["tripped"] is True
        assert not sweep.is_clean(results[0])

    def test_tripwire_marker_in_progress_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        _stub_train(monkeypatch, exit_code=sweep.TRIPWIRE_RC)
        _stub_gate(monkeypatch, _good_gate())

        sweep.run_sweep(_args(tmp_path, ["kill_scale=0.3"]))

        assert "TRIPWIRE" in capsys.readouterr().out

    def test_tripped_run_is_never_declared_winner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        _stub_train(monkeypatch, exit_code=sweep.TRIPWIRE_RC)
        _stub_gate(monkeypatch, _good_gate(mass_integral=999.0))

        results = sweep.run_sweep(_args(tmp_path, ["kill_scale=0.3"]))
        sweep.write_leaderboard(results, tmp_path)
        out = capsys.readouterr().out

        assert "Winner:" not in out
        assert "No clean winner" in out
        assert "TRIPWIRE" in out
        entry = json.loads((tmp_path / "leaderboard.json").read_text())[0]
        assert entry["tripped"] is True

    def test_clean_run_still_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        _stub_train(monkeypatch, exit_code=0)
        _stub_gate(monkeypatch, _good_gate())

        results = sweep.run_sweep(_args(tmp_path, ["kill_scale=0.3"]))
        sweep.write_leaderboard(results, tmp_path)
        out = capsys.readouterr().out

        assert results[0]["train_rc"] == 0
        assert sweep.is_clean(results[0])
        assert "Winner: kill_scale0.3" in out

    def test_crashed_train_records_rc(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_train(monkeypatch, exit_code=1, write_ckpt=False)
        _stub_gate(monkeypatch, _good_gate())

        results = sweep.run_sweep(_args(tmp_path, ["kill_scale=0.3"]))

        assert results[0]["train_rc"] == 1
        assert results[0]["tripped"] is False
        assert "error" in results[0]


class TestRanking:
    """Clean rows outrank tripped rows outrank errored rows, regardless of score."""

    def test_high_scoring_tripped_run_ranks_below_clean_run(self, tmp_path: Path) -> None:
        results: List[Dict[str, object]] = [
            {"name": "tripped", "out_dir": "d", "train_rc": 2, "mass_integral": 999.0},
            {"name": "errored", "out_dir": "d", "train_rc": 0, "error": "eval failed"},
            {"name": "clean", "out_dir": "d", "train_rc": 0, "mass_integral": 1.0},
        ]

        ranked = sorted(results, key=sweep.rank_key, reverse=True)

        assert [r["name"] for r in ranked] == ["clean", "tripped", "errored"]

    def test_clean_runs_rank_by_mass_integral(self, tmp_path: Path) -> None:
        results: List[Dict[str, object]] = [
            {"name": "low", "out_dir": "d", "train_rc": 0, "mass_integral": 10.0},
            {"name": "high", "out_dir": "d", "train_rc": 0, "mass_integral": 50.0},
        ]

        ranked = sorted(results, key=sweep.rank_key, reverse=True)

        assert [r["name"] for r in ranked] == ["high", "low"]


class TestKillDiagnosisSummarize:
    """run_kill_diagnosis reads the same gate JSON and needs the same guard."""

    def test_errored_candidate_returns_error_not_raise(self, tmp_path: Path) -> None:
        eval_json = tmp_path / "gate.json"
        eval_json.write_text(json.dumps(ERRORED_GATE))

        out = run_kill_diagnosis._summarize_eval(eval_json)

        assert "error" in out
        assert "kills_per_episode" not in out

    def test_malformed_json_returns_error(self, tmp_path: Path) -> None:
        eval_json = tmp_path / "gate.json"
        eval_json.write_text("{not json")

        assert "error" in run_kill_diagnosis._summarize_eval(eval_json)

    def test_healthy_gate_still_summarizes(self, tmp_path: Path) -> None:
        eval_json = tmp_path / "gate.json"
        eval_json.write_text(json.dumps(_good_gate(mass_integral=120.0)))

        out = run_kill_diagnosis._summarize_eval(eval_json)

        assert out == {
            "kills_per_episode": 2.0,
            "kill_opportunities": 0.0,
            "mass_integral_delta": 5.0,
        }
