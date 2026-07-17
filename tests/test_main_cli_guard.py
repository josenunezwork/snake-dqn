"""CLI guard tests: checkpoint flags must not be silently dropped without a mode."""

from typing import Any, Dict, List

import pytest

from src import main as main_module


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: List[str]) -> None:
    """Invoke ``main()`` with a patched ``sys.argv``."""
    monkeypatch.setattr("sys.argv", ["src/main.py", *argv])
    main_module.main()


@pytest.mark.parametrize(
    "argv",
    [
        ["--load", "saved_snakes/DOES_NOT_EXIST.pth"],
        ["--eval"],
        ["--load-memory-db"],
        ["--load-memory-db", "snake_memories.db"],
        ["--human", "--load", "saved_snakes/DOES_NOT_EXIST.pth"],
        ["--load", "saved_snakes/DOES_NOT_EXIST.pth", "--eval"],
    ],
)
def test_checkpoint_flags_without_mode_exit_2(
    monkeypatch: pytest.MonkeyPatch, argv: List[str]
) -> None:
    """Flags only the training paths consume must not be accepted without a mode."""
    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, argv)

    assert excinfo.value.code == 2


def test_guard_message_names_the_flag_and_the_modes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The error must point at the flag that was dropped and at the usable modes."""
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, ["--load", "saved_snakes/DOES_NOT_EXIST.pth"])

    stderr = capsys.readouterr().err
    assert "--load" in stderr
    assert "--headless" in stderr
    assert "--health-smoke" in stderr


def test_health_smoke_with_load_still_dispatches(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--health-smoke --load must reach the smoke runner with the checkpoint path."""
    calls: Dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> Dict[str, Any]:
        calls.update(kwargs)
        return {"frames": 1}

    monkeypatch.setattr(main_module, "run_learning_health_smoke", fake_run)
    monkeypatch.setattr(main_module, "format_learning_health_smoke_report", lambda stats: "report")
    monkeypatch.setattr(main_module, "validate_learning_health_smoke", lambda stats: None)

    _run_main(monkeypatch, ["--health-smoke", "--load", "saved_snakes/best_apex.pth"])

    assert calls["checkpoint_path"] == "saved_snakes/best_apex.pth"
    assert calls["eval_mode"] is False
    assert "report" in capsys.readouterr().out


def test_headless_with_load_still_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """--headless --load must reach the trainer with the checkpoint path."""
    calls: Dict[str, Any] = {}

    def fake_train(*args: Any, **kwargs: Any) -> None:
        calls["args"] = args
        calls["kwargs"] = kwargs

    monkeypatch.setattr(main_module, "train_headless", fake_train)
    monkeypatch.setattr(main_module.mp, "set_start_method", lambda method, force=False: None)

    _run_main(
        monkeypatch, ["--headless", "--episodes", "1", "--load", "saved_snakes/best_apex.pth"]
    )

    assert "saved_snakes/best_apex.pth" in calls["args"]


def test_bare_invocation_still_exits_zero_with_banner(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No flags: still a clean exit that points at the web app."""
    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, [])

    assert excinfo.value.code == 0
    assert "web/serve.py" in capsys.readouterr().out


def test_human_without_checkpoint_flags_still_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--human alone keeps pointing at the web Play tab instead of erroring."""
    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, ["--human"])

    assert excinfo.value.code == 0
    assert "Play" in capsys.readouterr().out
