"""Operator renicer decisions: no signals, actual processes, or paid work."""

from __future__ import annotations

import pytest

from tools.ops import renicer


def test_admitted_root_never_hides_unknown_generic_descendant():
    rows = [
        dict(
            pid=10,
            nice=0,
            argv="python -m research.colab_cli.supervisor",
            control_identity={"role": "supervisor"},
        ),
        dict(pid=11, nice=0, argv="colab", control_identity={"role": "cli"}),
        dict(pid=12, nice=0, argv="/usr/bin/arbitrary", control_tree=True),
        dict(pid=13, nice=0, argv="codex exec task"),
        dict(pid=14, nice=15, argv="codex exec task"),
        dict(pid=15, nice=0, argv="tmux codex exec stale", verified_tmux_server=True),
    ]
    assert renicer.targets(rows) == [12, 13]


@pytest.mark.parametrize("change", ["birth", "executable"])
def test_pid_reuse_or_exec_during_renice_never_mutates(monkeypatch, change):
    row = dict(pid=10, ppid=1, nice=0, argv="codex exec task")
    monkeypatch.setattr(renicer, "process_rows", lambda *args: [row])
    calls = []
    identities = iter(
        [
            {"birth": "1", "executable": "/opt/bin/codex"},
            {
                "birth": "2" if change == "birth" else "1",
                "executable": "/bin/sh" if change == "executable" else "/opt/bin/codex",
            },
        ]
    )
    monkeypatch.setattr(renicer, "_native_identity", lambda _: next(identities))
    monkeypatch.setattr(renicer.os, "setpriority", lambda *args: calls.append(args))
    renicer.iteration({})
    assert not calls


def test_unreadable_identity_prevents_all_mutations(monkeypatch):
    rows = [dict(pid=pid, nice=0, argv="codex exec task") for pid in [10, 11]]
    monkeypatch.setattr(renicer, "process_rows", lambda *args: rows)
    calls = []

    def identity(pid):
        if pid == 11:
            raise PermissionError("unreadable")
        return {"birth": "1", "executable": "/opt/bin/codex"}

    monkeypatch.setattr(renicer, "_native_identity", identity)
    monkeypatch.setattr(renicer.os, "setpriority", lambda *args: calls.append(args))
    with pytest.raises(PermissionError):
        renicer.iteration({})
    assert not calls
