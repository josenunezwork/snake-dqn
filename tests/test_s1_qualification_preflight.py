"""Offline qualification rejects missing fixtures and failed retained receipts."""

import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from research.s1_mature_20261018 import admission, manifests
from research.s1_mature_20261018 import qualification_preflight as preflight
from tests.test_colab_cli_source import _manifest, _run
from tests.test_colab_cli_source import repo as _source_repo


def test_packaging_qualifies_once_before_admission(tmp_path: Path, monkeypatch) -> None:
    repo = _source_repo.__wrapped__(tmp_path)
    script = repo / "research/s1_mature_20261018/train.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/bash\nexit 0\n")
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "S1 fixture")
    calls = []

    def check(*args):
        calls.append(args[1]["job_id"])
        return {"status": "PASS_CPU_ONLY", "cuda_qualified": False}

    monkeypatch.setattr(preflight, "check", check)
    rows = [
        {
            **_manifest(repo),
            "script": str(script.relative_to(repo)),
            "job_id": f"job-{i}",
            "inputs": [],
        }
        for i in range(10)
    ]
    result = admission.packaging_preflight(repo, rows)
    assert calls == ["job-0"]
    assert all(v["qualification"]["cuda_qualified"] is False for v in result["jobs"].values())

    def missing(*args):
        raise ValueError("missing retained fixture")

    monkeypatch.setattr(preflight, "check", missing)
    with pytest.raises(admission.AdmissionRefused, match="missing retained fixture"):
        admission.packaging_preflight(repo, rows)


@pytest.mark.parametrize("failure", [None, "missing", "nonzero", "failed_receipt"])
def test_remote_layout_uses_staged_inputs_and_requires_every_receipt(
    tmp_path: Path, monkeypatch, failure: str | None
) -> None:
    source = tmp_path / "local-inputs"
    source.mkdir()
    inputs = {}
    for name in ("B8_s3_final.pth", "mac-parity-reference.json", "parity-bank.json"):
        inputs[name] = source / name
        inputs[name].write_text("fixture")
    inputs["league.tar.gz"] = source / "league.tar.gz"
    with tarfile.open(inputs["league.tar.gz"], "w:gz") as archive:
        archive.add(inputs["B8_s3_final.pth"], arcname="b8.pth")
    root = tmp_path / "remote"
    root.mkdir()
    (root / "repo").mkdir()
    commands = []

    def execute(command, *, cwd, env, **kwargs):
        commands.append(command)
        assert cwd == root / "repo"
        assert env["PYTHONPATH"] == str(cwd)
        assert env["OMP_NUM_THREADS"] == env["NUMBA_NUM_THREADS"] == "1"
        assert Path(env["Q0Q1_INPUT_ROOT"], "checkpoints").resolve() == root / "s1-work/league"
        assert Path(env["Q0Q1_INPUT_ROOT"], "init/B8_s3_final.pth").read_text() == "fixture"
        value = {"status": "CPU_PREFLIGHT_PASS"} if "--cpu-preflight" in command else {"pass": True}
        if failure == "failed_receipt":
            value = {"status": "INCOMPLETE", "error": "qualification failed"}
        if failure != "missing":
            Path(command[-1]).write_text(json.dumps(value))
        return subprocess.CompletedProcess(command, int(failure == "nonzero"), "", "")

    monkeypatch.setattr(preflight.subprocess, "run", execute)
    if failure:
        with pytest.raises(ValueError, match="remote-layout qualification failed"):
            preflight._run_layout(root, {"env": {}}, inputs)
        assert len(commands) == 1
    else:
        result = preflight._run_layout(root, {"env": {}}, inputs)
        assert len(commands) == 2
        assert [c[2] for c in commands][-1] == "research.q0q1_20261017.q1_parity"
        assert result["status"] == "PASS_CPU_ONLY"


def test_runtime_requirements_cover_imported_test_helpers() -> None:
    requirements = (admission.ROOT / manifests.RUNTIME_REQUIREMENTS).read_text()
    assert "-r ../../requirements.txt" in requirements
    assert "pytest>=9.0.0,<10.0.0" in requirements
    assert "tests/test_t22_random_safe.py" in manifests.RUNTIME_SOURCE_PATHS
    assert "research/q0q1_20261017/q0b_cuda_fixtures.py" in manifests.RUNTIME_SOURCE_PATHS
    assert "research/q0q1_20261017/results" not in manifests.RUNTIME_SOURCE_PATHS


def test_runtime_fallback_excludes_historical_result_payloads() -> None:
    from research.colab_cli import source

    for relative in (
        "research/q0q1_20261017/results/q0_mass_audit_receipt.json",
        "research/t23_r0_screen_20261017/WORLD_BANK.json",
        "research/t23_r0_screen_20261017/parity/mac.json.gz",
        "research/s1_mature_20261018/WORLD_BANK.json",
    ):
        assert not source._matches(relative, list(manifests.RUNTIME_SOURCE_PATHS)), relative
