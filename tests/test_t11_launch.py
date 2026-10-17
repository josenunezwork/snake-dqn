"""Focused checks for the unstarted T11 LOCAL30 MPS launcher."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/t11_local30_20261017/launch.py"


def _load():
    spec = importlib.util.spec_from_file_location("t11_launch", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_fixture(tmp_path: Path, module) -> tuple[Path, Path]:
    bank = tmp_path / "bank-local30"
    bank.mkdir()
    shard = bank / "bank-00000.npz"
    shard.write_bytes(b"local30 fixture")
    manifest = bank / "manifest.json"
    manifest.write_text(json.dumps({"shards": {shard.name: _sha(shard)}}))
    prereg = tmp_path / "prereg.md"
    prereg.write_text("immutable preregistration")
    binding = tmp_path / "launch_inputs.json"
    binding.write_text(
        json.dumps(
            {
                "schema": "t11-launch-inputs/v1",
                "files": {str(manifest): _sha(manifest), str(prereg): _sha(prereg)},
            }
        )
    )
    module.BANK = bank
    module.BINDING = binding
    module.RUNS = tmp_path / "runs"
    module.REPO = tmp_path / "repo"
    module.ARTIFACTS = tmp_path / "artifacts"
    return bank, binding


def test_command_is_exact_two_sided_mps_and_smoke_keeps_tripwire(tmp_path: Path) -> None:
    module = _load()
    module.REPO = tmp_path / "repo"
    module.ARTIFACTS = tmp_path / "artifacts"
    module.RUNS = tmp_path / "runs"
    module.BANK = tmp_path / "bank-local30"
    full = module.command(10)
    smoke = module.command(10, smoke=True)

    assert full[full.index("--out") + 1].endswith("LOCAL30_s10")
    assert smoke[smoke.index("--out") + 1].endswith("validation/smoke-mps")
    for flag, value in (
        ("--device", "mps"),
        ("--transitions", "8000000"),
        ("--label-frac", "0.5"),
        ("--label-bank-frac", "0.5"),
        ("--label-bank-counter-mode", "single-action"),
        ("--label-bank-counter-ratio", "1"),
        ("--label-bank-margin-scale", "4.75"),
        ("--td-huber-delta", "1.0"),
        ("--boost-halt-rate", "0.02"),
        ("--boost-halt-low-length-rate", "0.01"),
    ):
        assert full[full.index(flag) + 1] == value
    assert smoke[smoke.index("--transitions") + 1] == "200000"
    assert smoke[smoke.index("--boost-halt-rate") + 1] == "0.02"
    assert smoke[smoke.index("--boost-halt-low-length-rate") + 1] == "0.01"
    assert smoke[smoke.index("--arm-halt-file") + 1].endswith("validation/KILL.json")
    assert "--opp-forward" not in full
    assert module.command(14)[module.command(14).index("--init") + 1].endswith("B8_s4/final.pth")


def test_preflight_rejects_extra_or_tampered_local30_shards(tmp_path: Path) -> None:
    module = _load()
    bank, _ = _bound_fixture(tmp_path, module)
    module.verify_inputs()
    (bank / "extra.npz").write_bytes(b"unbound")
    with pytest.raises(ValueError, match="shard set"):
        module.verify_inputs()
    (bank / "extra.npz").unlink()
    (bank / "bank-00000.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash/path mismatch"):
        module.verify_inputs()


@pytest.mark.parametrize("lane,seeds", [(0, (10, 12, 14)), (1, (11, 13))])
def test_check_prints_only_bound_mps_commands_without_creating_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    lane: int,
    seeds: tuple[int, ...],
) -> None:
    module = _load()
    _bound_fixture(tmp_path, module)
    monkeypatch.setattr(sys, "argv", ["launch.py", "--lane", str(lane), "--check"])

    assert module.main() == 0
    output = capsys.readouterr().out
    assert output.count("--device mps") == len(seeds)
    assert output.count("--transitions 8000000") == len(seeds)
    assert output.count("--boost-halt-rate 0.02") == len(seeds)
    for seed in seeds:
        assert f"LOCAL30_s{seed}" in output
    assert not module.RUNS.exists()


def test_existing_kill_receipt_blocks_a_lane_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    _bound_fixture(tmp_path, module)
    module.RUNS.mkdir()
    (module.RUNS / "KILL.json").write_text('{"decision":"KILL"}\n')
    monkeypatch.setattr(sys, "argv", ["launch.py", "--lane", "0", "--check"])
    with pytest.raises(SystemExit, match="KILL"):
        module.main()


def test_launcher_exception_writes_shared_kill_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    _bound_fixture(tmp_path, module)
    monkeypatch.setattr(sys, "argv", ["launch.py", "--lane", "0"])

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic launch failure")

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="synthetic launch failure"):
        module.main()
    receipt = json.loads((module.RUNS / "KILL.json").read_text())
    assert receipt["decision"] == "KILL"
    assert receipt["reason"] == "launcher_exception:RuntimeError"
    assert receipt["seed"] == 10
