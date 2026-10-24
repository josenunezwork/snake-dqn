"""Small refusal checks at the actual R32 worker and reducer entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.bl_ppo.exports import digest, verify_export
from research.r32_rehearsal import decisions, job, lifecycle, package


def test_worker_accepts_supervisor_precreated_layout(tmp_path: Path) -> None:
    root = tmp_path / "snake-cli" / "runtime-id"
    inputs = root / "inputs"
    package.prepare_inputs(inputs, tiny=True)
    (root / "repo").mkdir()
    output = root / "output"
    output.mkdir(mode=0o700)
    (root / "lease.json").write_text('{"generation": "test"}\n')
    (root / "state.json").write_text('{"status": "starting"}\n')
    (root / "job.log").touch()

    receipt = job.run_portable(inputs, output, mode="canary")

    assert receipt["done"] == receipt["expected"] == 3
    assert receipt["completed"] == [0, 1, 2]
    for name in ("allocation.json", "receipt.json", *(f"unit-{i:04d}.json" for i in range(3))):
        verify_export(output / name)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="not empty"):
        job.run_portable(inputs, output, mode="canary")
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


@pytest.mark.parametrize(
    "existing",
    [
        "allocation.json",
        "allocation.json.sha256",
        "allocation.json.complete",
        "unit-0000.json",
        "unit-0000.json.complete",
        "receipt.json",
        "unrelated-file",
    ],
)
def test_attempt_refuses_preexisting_output_files(tmp_path: Path, existing: str) -> None:
    inputs = tmp_path / "inputs"
    package.prepare_inputs(inputs, tiny=True)
    spec = json.loads((inputs / "canary-spec.json").read_text())
    output = tmp_path / "output"
    output.mkdir()
    (output / existing).write_bytes(b"immutable evidence")

    with pytest.raises(FileExistsError, match="not empty"):
        lifecycle.start_attempt(output, spec)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == {
        existing: b"immutable evidence"
    }


@pytest.mark.parametrize("kind", ["file", "symlink", "directory"])
def test_attempt_refuses_output_path_collisions(tmp_path: Path, kind: str) -> None:
    inputs = tmp_path / "inputs"
    package.prepare_inputs(inputs, tiny=True)
    spec = json.loads((inputs / "canary-spec.json").read_text())
    output = tmp_path / "output"
    if kind == "file":
        output.write_bytes(b"immutable evidence")
    elif kind == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    else:
        (output / "lifecycle-fixtures").mkdir(parents=True)

    with pytest.raises(FileExistsError):
        lifecycle.start_attempt(output, spec)
    if kind == "file":
        assert output.read_bytes() == b"immutable evidence"
    elif kind == "symlink":
        assert not list(target.iterdir())
    else:
        assert list(output.iterdir()) == [output / "lifecycle-fixtures"]


def test_worker_cannot_complete_a_failed_independent_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package.prepare_inputs(tmp_path / "inputs", tiny=True)
    monkeypatch.setattr(decisions, "run_case", lambda _: {"pass": False})
    with pytest.raises(ValueError, match="false-ready"):
        job.run_portable(tmp_path / "inputs", tmp_path / "output", mode="tiny")
    assert not (tmp_path / "output/receipt.json").exists()


@pytest.mark.parametrize("stop_verified", [False, True])
def test_reducer_refuses_unknown_stop_or_unbound_supervisor_exports(
    tmp_path: Path, stop_verified: bool
) -> None:
    inputs = tmp_path / "inputs"
    package.prepare_inputs(inputs, tiny=True)
    spec = json.loads((inputs / "spec.json").read_text())
    attempt = tmp_path / "attempt"
    exports = attempt / "exports"
    lifecycle.start_attempt(exports, spec)
    for index in range(3):
        lifecycle.publish_unit(exports, spec, index, {"synthetic": True})
    ledger = tmp_path / "ledger.json"
    lifecycle.build_ledger([exports], spec, ledger)
    manifest = {
        "commit": "a" * 40,
        "script": "research/r32_rehearsal/colab.sh",
        "env": {"R32_MODE": "main"},
        "inputs": [
            {
                "dest": "spec.json",
                "path": str(inputs / "spec.json"),
                "sha256": digest(inputs / "spec.json"),
            },
            {
                "dest": "verified-ledger.json",
                "path": str(inputs / "verified-ledger.json"),
                "sha256": digest(inputs / "verified-ledger.json"),
            },
        ],
    }
    (attempt / "manifest.json").write_text(json.dumps(manifest))
    receipt = attempt / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "stop_verified": stop_verified,
                "status": "PASS",
                "commit": "a" * 40,
                "manifest_sha256": digest(attempt / "manifest.json"),
                "elapsed_seconds": 1,
                "accounting": {"own_rate_units_per_hour": 0.08},
                "exports": [],
            }
        )
    )
    with pytest.raises(ValueError, match="STOP_UNVERIFIED|matching supervisor"):
        job.reduce_verified(inputs, ledger, tmp_path / "terminal.json", [receipt])
    assert not (inputs / "terminal-look.json").exists()
