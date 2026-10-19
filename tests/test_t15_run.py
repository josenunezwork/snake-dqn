"""Focused contracts for the T15 create-only qualification harness."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.t15_mixed_roster_20261017 import run, seed_audit, ui_server


def test_seed_allocation_has_the_registered_non_smoke_count_and_separate_smoke_domain():
    allocation = seed_audit.generate_banks()
    assert allocation["namespace"] == seed_audit.NAMESPACE
    assert allocation["count"] == 33
    assert {name: row["count"] for name, row in allocation["banks"].items()} == {
        "watch": 25,
        "parity": 3,
        "latency": 3,
        "ui": 1,
        "controls": 1,
    }
    smoke = seed_audit.generate_banks(seed_audit.SMOKE_NAMESPACE)
    assert {value for row in allocation["banks"].values() for value in row["seeds"]}.isdisjoint(
        value for row in smoke["banks"].values() for value in row["seeds"]
    )


def test_seed_coordinates_are_bounded_and_purpose_labelled():
    assert seed_audit.seed_for("watch", 0) != seed_audit.seed_for("parity", 0)
    with pytest.raises(ValueError):
        seed_audit.seed_for("watch", 25)
    with pytest.raises(ValueError):
        seed_audit.seed_for("missing", 0)


def test_parity_plan_matches_protocol_horizons_and_roster_sizes():
    assert run.PARITY_PLAN == ((5000, 6), (5000, 8), (10000, 12))


_DIST = run.REPO / "web/frontend/dist"


@pytest.mark.skipif(
    not (_DIST.is_dir() and any(path.is_file() for path in _DIST.rglob("*"))),
    reason="needs a built frontend bundle (gitignored): run `npm run build` in web/frontend",
)
def test_source_closure_binds_html_css_and_the_actual_built_ui_bundle():
    bound = run.sources()
    assert "web/frontend/index.html" in bound
    assert any(name.endswith(".css") for name in bound)
    assert "web/frontend/dist/index.html" in bound
    assert any(name.startswith("web/frontend/dist/assets/") for name in bound)


def test_write_new_refuses_overwrite(tmp_path: Path):
    target = tmp_path / "receipt.json"
    run.write_new(target, {"one": 1})
    assert json.loads(target.read_text()) == {"one": 1}
    with pytest.raises(FileExistsError):
        run.write_new(target, {"two": 2})


def test_quiet_snapshot_exports_sanitized_process_rows_only(monkeypatch):
    own = os.getpid()

    def fake_command(command, **_kwargs):
        if command[0] == "ps":
            return SimpleNamespace(
                returncode=0,
                stdout=f"{own} 1 /usr/bin/python3 private-task.py --token secret\n"
                "4242 1 /usr/bin/python3 /private/other.py --secret value\n",
            )
        if command[0] == "pmset":
            return SimpleNamespace(
                returncode=0, stdout="Now drawing from 'AC Power'\n -InternalBattery\n"
            )
        return SimpleNamespace(returncode=0, stdout='"AppleClamshellState" = No\n')

    monkeypatch.setattr(run.subprocess, "run", fake_command)
    snapshot = run.quiet_snapshot()
    exported = json.dumps(snapshot)
    assert "secret" not in exported and "/private/other.py" not in exported
    assert snapshot["foreign_active_compute"] == [{"pid": 4242, "ppid": 1, "executable": "python3"}]
    assert snapshot["power"]["ac_power"] is True
    assert snapshot["power"]["lid_closed"] is False
    assert snapshot["quiet"] is False


def test_ui_backend_evidence_uses_the_actual_watch_opponent_checkpoint():
    class FakeSession:
        served_checkpoint = {"name": "b8-s3"}
        obs_spec = "ego2s-bv8"
        policy = object()
        watch_roster = "mixed"
        watch_roster_reason = None
        watch_opponent_checkpoint = {"name": "frp3-s12"}
        speed_ceiling = 12

        @staticmethod
        def safety_veto_state():
            return {"scope": "watch_opponents"}

    evidence = ui_server.backend_evidence(FakeSession(), "intent-hash", 7, "/tmp/scores.sqlite")
    assert evidence["opponent_checkpoint"] == {"name": "frp3-s12"}
    assert evidence["watch_roster"] == "mixed"
