"""Exercise the real handoff CLI without contacting an account or dispatcher."""

import argparse
import json
from pathlib import Path

import pytest

from research.s1_mature_20261018 import admission
from research.s1_mature_20261018 import prepare_parallel_handoff as handoff


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    args = argparse.Namespace(
        repo=tmp_path,
        plan=tmp_path / "plan.json",
        previous_plan=tmp_path / "old-plan.json",
        ledger=tmp_path / "ledger.json",
        colab=tmp_path / "never-call-colab",
        account_config=tmp_path / "account.json",
        zero_snapshot=tmp_path / "zero.json",
        out=tmp_path / "handoff.json",
        legacy_dispatcher_exited=True,
    )
    for path in (args.plan, args.previous_plan, args.ledger):
        path.write_text("{}")
    monkeypatch.setattr(handoff.run, "_verify_plan", lambda *_: {"run_rules": {"lanes": 3}})
    monkeypatch.setattr(handoff.run, "_parallel_amendment", lambda *_: tmp_path)
    outstanding = []

    class FakeLedger:
        def __init__(self, *_):
            pass

        def preview_parallel(self, *_, **__):
            return {"outstanding_jobs": outstanding}

        def _snapshot(self, path, *, require_zero):
            assert require_zero
            assert json.loads(path.read_text())["active_assignments"] == 0

    monkeypatch.setattr(handoff.ledger, "Ledger", FakeLedger)

    def snapshot(path, **kwargs):
        assert kwargs["include_active_records"] is True
        admission.write_once(path, {"active_assignments": 0, "sessions": {}, "active_records": []})
        return path

    monkeypatch.setattr(handoff.run, "_snapshot", snapshot)
    return args, outstanding


def test_handoff_cli_binds_all_identities_and_preserves_existing_ledger(prepared, capsys):
    args, _ = prepared
    original = args.ledger.read_bytes()
    command = ["--legacy-dispatcher-exited"]
    for name in (
        "repo",
        "plan",
        "previous_plan",
        "ledger",
        "colab",
        "account_config",
        "zero_snapshot",
        "out",
    ):
        command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    assert handoff.main(command) == 0
    assert str(args.out) in capsys.readouterr().out
    value = json.loads(args.out.read_text())
    assert value["legacy_dispatcher_exited"] is True
    for name in ("plan", "previous_plan", "ledger", "zero_snapshot"):
        assert value[name]["sha256"] == admission.sha256(getattr(args, name))
    assert args.ledger.read_bytes() == original
    with pytest.raises(admission.AdmissionRefused, match="already exists"):
        handoff.prepare(args)


def test_handoff_refuses_outstanding_without_account_calls(prepared):
    args, outstanding = prepared
    outstanding.append("legacy-job")
    with pytest.raises(admission.AdmissionRefused, match="reconciled"):
        handoff.prepare(args)
    assert not args.zero_snapshot.exists()
    assert not args.out.exists()


def test_handoff_requires_explicit_exit_observation(prepared):
    args, _ = prepared
    args.legacy_dispatcher_exited = False
    with pytest.raises(admission.AdmissionRefused, match="independently verify"):
        handoff.prepare(args)
    assert not args.zero_snapshot.exists()
