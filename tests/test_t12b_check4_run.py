"""Focused contract coverage for the isolated T12B Check-4 harness."""

from __future__ import annotations

import hashlib
import os

from research.t12b_check4_20261017 import run, seed_audit


def test_protocol_pin_matches_the_amendment_bytes():
    assert (
        run.PROTOCOL_SHA256
        == hashlib.sha256((run.HERE / "protocol_check4.md").read_bytes()).hexdigest()
    )


def test_fresh_latency_bank_is_fixed_and_disjoint():
    allocation = seed_audit.generate_banks()
    latency = allocation["banks"]["latency"]
    assert allocation["namespace"] == seed_audit.NAMESPACE
    assert latency["count"] == 4
    assert len(set(latency["seeds"])) == 4
    assert latency["seeds"] == [seed_audit.seed_for(index) for index in range(4)]


def test_quiet_snapshot_accepts_only_dead_tmux_panes_and_no_foreign_compute(monkeypatch):
    from research.t12_b8_serving_20261017 import measurements

    calls = iter(
        [
            {"returncode": 0, "stdout": "kept\t0.0\t12\tzsh\t1\n", "stderr": ""},
            {"returncode": 0, "stdout": f"{os.getpid()} 1 harness\n", "stderr": ""},
        ]
    )
    monkeypatch.setattr(run, "_command", lambda _command: next(calls))
    monkeypatch.setattr(
        measurements, "_mac_power_and_lid", lambda: {"ac_power": True, "lid_closed": False}
    )
    snapshot = run.quiet_snapshot()
    assert snapshot["quiet"] is True
    assert snapshot["tmux"]["present"] is True
    assert snapshot["tmux"]["all_panes_dead"] is True


def test_quiet_snapshot_blocks_foreign_python_compute(monkeypatch):
    from research.t12_b8_serving_20261017 import measurements

    calls = iter(
        [
            {"returncode": 1, "stdout": "", "stderr": "no server"},
            {
                "returncode": 0,
                "stdout": f"{os.getpid()} 1 harness\n777 1 python train.py\n",
                "stderr": "",
            },
        ]
    )
    monkeypatch.setattr(run, "_command", lambda _command: next(calls))
    monkeypatch.setattr(
        measurements, "_mac_power_and_lid", lambda: {"ac_power": True, "lid_closed": False}
    )
    snapshot = run.quiet_snapshot()
    assert snapshot["quiet"] is False
    assert snapshot["suspicious_processes"] == ["777 python train.py"]


def test_quiet_snapshot_allows_an_idle_foreign_tmux_shell(monkeypatch):
    from research.t12_b8_serving_20261017 import measurements

    calls = iter(
        [
            {
                "returncode": 0,
                "stdout": "git-sync\t0.0\t444\tzsh\t0\n",
                "stderr": "",
            },
            {"returncode": 0, "stdout": "444 1 zsh\n", "stderr": ""},
        ]
    )
    monkeypatch.setattr(run, "_command", lambda _command: next(calls))
    monkeypatch.setattr(
        measurements, "_mac_power_and_lid", lambda: {"ac_power": True, "lid_closed": False}
    )

    snapshot = run.quiet_snapshot()

    assert snapshot["quiet"] is True
    assert snapshot["tmux"]["workload_panes_exited"] is True
    assert snapshot["tmux"]["all_panes_dead"] is False


def test_ceiling_probe_contract_uses_set_speed_120_in_protocol():
    text = (run.HERE / "protocol_check4.md").read_text(encoding="utf-8")
    assert "session.set_speed(120)" in text
    assert "Merely requesting a lower speed does not validate" in text
    assert "Do not retry this default-speed probe" in text


def test_new_artifact_scan_detects_candidate_seed_in_a_nested_json(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    nested = tmp_path / "snake-dqn-artifacts" / "t12b-check4-20261017" / "nested"
    nested.mkdir(parents=True)
    (nested / "diagnostic.json").write_text('{"seed": 123456}', encoding="utf-8")

    result = seed_audit._unexcluded_new_artifact_hits(repo, {123456})

    assert result["hits"] == [123456]
