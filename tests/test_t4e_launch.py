"""Focused, side-effect-free tests for the prepared T4e MPS launch lanes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_DIR = ROOT / "research" / "t4e_return_20261016"


def _load_check_module():
    spec = importlib.util.spec_from_file_location(
        "t4e_launch_check", LAUNCH_DIR / "launch_check.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bank(root: Path, rows: int) -> None:
    shard = root / "world-000.npz"
    shard.write_bytes(b"test shard")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "collection_schema": "t4e-return-bank/v1",
                "publication": {"rows": rows},
                "shards": {shard.name: _digest(shard)},
            }
        )
    )


def test_launch_check_binds_both_published_banks_and_their_exact_shards(tmp_path: Path) -> None:
    module = _load_check_module()
    input_file = tmp_path / "input.txt"
    input_file.write_text("immutable")
    train = tmp_path / "train"
    holdout = tmp_path / "holdout"
    train.mkdir()
    holdout.mkdir()
    _write_bank(train, 7)
    _write_bank(holdout, 3)
    completion = tmp_path / "completion.json"
    completion.write_text(json.dumps({"training_decisive_states": 2}))
    binding = tmp_path / "binding.json"
    binding.write_text(
        json.dumps(
            {
                "schema": "t4e-return-launch-inputs/v1",
                "files": {str(input_file): _digest(input_file)},
                "training_return_bank": {
                    "path": str(train),
                    "completion_path": str(completion),
                    "completion_counts": {"training_decisive_states": 2},
                    "publication_rows": 7,
                },
                "holdout_return_bank": {"path": str(holdout), "publication_rows": 3},
            }
        )
    )

    assert module.verify(binding) == {
        "files": 1,
        "holdout_rows": 3,
        "holdout_shards": 1,
        "training_rows": 7,
        "training_shards": 1,
    }

    input_file.write_text("tampered")
    with pytest.raises(ValueError, match="registered input changed"):
        module.verify(binding)
    input_file.write_text("immutable")

    (train / "unexpected.npz").write_bytes(b"not declared")
    with pytest.raises(ValueError, match="shard set"):
        module.verify(binding)


@pytest.mark.parametrize(
    "lane,seeds", [("lane0.sh", ("10", "12", "14")), ("lane1.sh", ("11", "13"))]
)
def test_check_mode_prints_the_registered_mps_recipe_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str, seeds: tuple[str, ...]
) -> None:
    repo = tmp_path / "repo"
    artifact = tmp_path / "artifacts"
    (repo / "venv" / "bin").mkdir(parents=True)
    fake_python = repo / "venv" / "bin" / "python"
    command_log = tmp_path / "python-invocations.log"
    fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$PYTHON_LOG"\n')
    fake_python.chmod(0o755)
    monkeypatch.setenv("PYTHON_LOG", str(command_log))

    script = (LAUNCH_DIR / lane).read_text()
    script = script.replace("repo=/Users/josenunez/Projects/ml/snake-dqn-t4d", f'repo="{repo}"')
    script = script.replace(
        "art=/Users/josenunez/Projects/ml/snake-dqn-artifacts", f'art="{artifact}"'
    )
    test_lane = tmp_path / lane
    test_lane.write_text(script)
    test_lane.chmod(0o755)

    result = subprocess.run([str(test_lane), "--check"], check=True, capture_output=True, text=True)

    assert result.stderr == ""
    assert result.stdout.count("--device mps") == len(seeds)
    assert result.stdout.count("--transitions 8000000") == len(seeds)
    assert result.stdout.count("--return-frac 0.5") == len(seeds)
    assert result.stdout.count("--return-weight 1.0") == len(seeds)
    assert result.stdout.count("--td-huber-delta 1.0") == len(seeds)
    assert result.stdout.count("--return-huber-delta 100") == len(seeds)
    assert "--margin-bank" not in result.stdout
    assert "--label-bank" not in result.stdout
    assert result.stdout.count("--label-frac 1") == len(seeds)
    for seed in seeds:
        assert f"RETURN_s{seed}" in result.stdout
    assert command_log.read_text().splitlines() == [
        "research/t4e_return_20261016/launch_check.py "
        "research/t4e_return_20261016/launch_inputs.json"
    ]


def test_lane_refuses_preexisting_output_after_preflight(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    artifact = tmp_path / "artifacts"
    (repo / "venv" / "bin").mkdir(parents=True)
    fake_python = repo / "venv" / "bin" / "python"
    fake_python.write_text("#!/bin/sh\nexit 0\n")
    fake_python.chmod(0o755)
    runs = artifact / "redesign-t4e-mac-20261016"
    runs.mkdir(parents=True)
    (runs / "RETURN_s10.log").write_text("exists")

    script = (LAUNCH_DIR / "lane0.sh").read_text()
    script = script.replace("repo=/Users/josenunez/Projects/ml/snake-dqn-t4d", f'repo="{repo}"')
    script = script.replace(
        "art=/Users/josenunez/Projects/ml/snake-dqn-artifacts", f'art="{artifact}"'
    )
    test_lane = tmp_path / "lane0.sh"
    test_lane.write_text(script)
    test_lane.chmod(0o755)

    result = subprocess.run([str(test_lane), "--check"], capture_output=True, text=True)

    assert result.returncode == 2
    assert "refusing existing output or log" in result.stderr.lower()
