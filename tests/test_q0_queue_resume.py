"""Queue bindings for resumable Q0 mass development audits."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from research.q0q1_20261017 import run_queue_job as runner


def _mass_env(output: str, **overrides: str) -> dict[str, str]:
    env = {
        "Q0Q1_JOB_KIND": "q0_t23_mass",
        "Q0Q1_INPUT_ROOT": "/inputs",
        "Q0Q1_T23_ROOT": "/t23",
        "QUEUE_OUTPUT_DIR": output,
    }
    env.update(overrides)
    return env


def test_mass_command_binds_job_local_partials_workers_and_verified_resume(tmp_path) -> None:
    output = tmp_path / "fresh-output"
    output.mkdir()
    prior = tmp_path / "prior" / "manifest.json"
    command = runner.command(
        _mass_env(
            str(output),
            Q0Q1_MASS_WORKERS="2",
            Q0Q1_MASS_RESUME_PARTIALS=str(prior),
            Q0Q1_MASS_RESUME_PARTIALS_SHA256="a" * 64,
        ),
        tmp_path / "local",
    )
    assert command[command.index("--partials-dir") + 1] == str(output.resolve() / "mass-partials")
    assert command[command.index("--workers") + 1] == "2"
    assert command[command.index("--resume-partials") + 1] == str(prior.resolve())
    assert command[command.index("--resume-partials-sha256") + 1] == "a" * 64


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"Q0Q1_MASS_RESUME_PARTIALS": "/prior/manifest.json"}, "must be paired"),
        ({"Q0Q1_MASS_RESUME_PARTIALS_SHA256": "a" * 64}, "must be paired"),
        ({"Q0Q1_MASS_WORKERS": "0"}, "positive integer"),
    ],
)
def test_mass_command_refuses_incomplete_or_invalid_optional_bindings(
    tmp_path, overrides, error
) -> None:
    output = tmp_path / "fresh-output"
    output.mkdir()
    with pytest.raises(RuntimeError, match=error):
        runner.command(_mass_env(str(output), **overrides), tmp_path / "local")


def test_mass_command_refuses_resume_pointer_inside_current_output(tmp_path) -> None:
    output = tmp_path / "fresh-output"
    output.mkdir()
    with pytest.raises(RuntimeError, match="must not be inside"):
        runner.command(
            _mass_env(
                str(output),
                Q0Q1_MASS_RESUME_PARTIALS=str(output / "old" / "manifest.json"),
                Q0Q1_MASS_RESUME_PARTIALS_SHA256="a" * 64,
            ),
            tmp_path / "local",
        )


def test_run_uses_unbuffered_child_python(monkeypatch, tmp_path) -> None:
    output = tmp_path / "fresh-output"
    output.mkdir()
    captured = {}

    monkeypatch.setattr(runner, "require_owner_release", lambda env: None)
    monkeypatch.setattr(runner, "admission_checks", lambda env: None)
    monkeypatch.setattr(runner, "_scientific_done", lambda *args: None)

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.run(_mass_env(str(output))) == 0
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"
    assert captured["argv"][captured["argv"].index("--partials-dir") + 1] == str(
        output.resolve() / "mass-partials"
    )
