"""Prepared launch scripts cannot bypass their shared process-budget lock."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("lane", [0, 1])
def test_check_acquires_lock_before_python_and_never_trains(tmp_path, lane):
    root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    (repo / "venv/bin").mkdir(parents=True)
    marker = tmp_path / "python-called"
    python = repo / "venv/bin/python"
    python.write_text(f'#!/bin/sh\ncat >/dev/null\ntouch "{marker}"\n')
    python.chmod(0o755)
    artifacts = tmp_path / "artifacts"
    runs = artifacts / "redesign-t4c-mac-20261016"
    runs.mkdir(parents=True)
    script = tmp_path / f"lane{lane}.sh"
    source = (root / f"research/t4c_20261016/lane{lane}.sh").read_text()
    source = source.replace("repo=/Users/josenunez/Projects/ml/snake-dqn-t4c", f"repo={repo}")
    source = source.replace(
        "art=/Users/josenunez/Projects/ml/snake-dqn-artifacts", f"art={artifacts}"
    )
    script.write_text(source)
    lock = runs / ".t4c-lane-lock"
    lock.mkdir()
    rejected = subprocess.run(["bash", str(script), "--check"], capture_output=True, text=True)
    assert rejected.returncode == 2
    assert not marker.exists()
    assert lock.exists()  # another owner's lock is untouched
    lock.rmdir()
    checked = subprocess.run(["bash", str(script), "--check"], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    assert marker.exists()
    assert not lock.exists()
    assert not list(runs.glob("TWO_SIDED_T_s*"))
    assert "--device cpu" in checked.stdout
    assert "--label-bank-outside-margin-scale 1" in checked.stdout
    assert "--device mps" not in checked.stdout


@pytest.mark.parametrize("lane", [0, 1])
@pytest.mark.parametrize("change", ["extra", "missing"])
def test_preflight_rejects_changed_bank_shard_set(tmp_path, monkeypatch, lane, change):
    """Listed-file hashes alone must not admit unbound files loaded by a glob."""
    root = Path(__file__).resolve().parents[1]
    source = (root / f"research/t4c_20261016/lane{lane}.sh").read_text()
    preflight = source.split("<<'CHECK_INPUTS'\n", 1)[1].split("\nCHECK_INPUTS", 1)[0]
    bank = tmp_path / "bank-v3"
    bank.mkdir()
    shard = bank / "bank-00000.npz"
    shard.write_bytes(b"test shard")
    manifest = bank / "manifest.json"
    manifest.write_text(json.dumps({"shards": {shard.name: "fixture"}}))
    binding = tmp_path / "research/t4c_20261016/launch_inputs.json"
    binding.parent.mkdir(parents=True)
    binding.write_text(
        json.dumps({"files": {str(manifest): hashlib.sha256(manifest.read_bytes()).hexdigest()}})
    )
    monkeypatch.chdir(tmp_path)
    exec(compile(preflight, "lane-preflight", "exec"), {})
    if change == "extra":
        (bank / "extra.npz").write_bytes(b"unbound")
    else:
        shard.unlink()
    with pytest.raises(SystemExit, match="shard set differs"):
        exec(compile(preflight, "lane-preflight", "exec"), {})
