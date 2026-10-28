"""VM-fidelity regression: exact uploads, no Mac evidence or checkout reads."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research.watch_lead_p.uploaded import uploaded_checkpoints

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT.parent / "snake-dqn/saved_snakes"


def copy_inputs(target: Path) -> Path:
    target.mkdir()
    for dest, source in (
        ("b8-s3", "b8_s3_t7_20261016"),
        ("frp3-s12", "frp3_m3_s12_u60000_20261005"),
    ):
        shutil.copyfile(CHECKPOINTS / f"{source}.pth", target / f"{dest}.pth")
    return target


@pytest.mark.parametrize("damage", ["missing", "hash", "symlink", "no-env", "relative"])
@pytest.mark.parametrize("checkpoint", ["b8-s3", "frp3-s12"])
def test_uploaded_refuses_without_fallback(tmp_path, monkeypatch, damage, checkpoint):
    from web.backend import session

    original = session.resolve_served_checkpoint
    inputs = copy_inputs(tmp_path / "inputs")
    monkeypatch.setenv("COLAB_INPUT_DIR", str(inputs))
    if damage == "no-env":
        monkeypatch.delenv("COLAB_INPUT_DIR")
    elif damage == "relative":
        monkeypatch.setenv("COLAB_INPUT_DIR", "inputs")
    else:
        hero = inputs / f"{checkpoint}.pth"
        hero.unlink()
        if damage == "hash":
            hero.write_bytes(b"wrong checkpoint")
        elif damage == "symlink":
            hero.symlink_to(CHECKPOINTS / "b8_s3_t7_20261016.pth")
    with pytest.raises(ValueError, match="P .*upload"):
        with uploaded_checkpoints():
            pytest.fail("invalid inputs admitted")
    assert session.resolve_served_checkpoint is original


def test_uploaded_restores_serving_and_matches_live(tmp_path, monkeypatch):
    from research.watch_lead_p.live import LiveWorld
    from research.watch_lead_p.protocol import FIXTURE_SEED
    from web.backend import safety_veto_serving, session

    original = session.resolve_served_checkpoint
    original_opponent = session.resolve_mixed_watch_opponent
    original_veto = safety_veto_serving.v8_gated_checkpoint_sha256s
    baseline = LiveWorld(FIXTURE_SEED)
    expected = [baseline.step() for _ in range(12)]
    inputs = copy_inputs(tmp_path / "inputs")
    monkeypatch.setenv("SNAKE_MECHANICS_V2", "0")
    with uploaded_checkpoints(inputs):
        world = LiveWorld(FIXTURE_SEED)
        assert world.roster["hero_checkpoint_sha256"] == baseline.roster["hero_checkpoint_sha256"]
        assert [world.step() for _ in range(12)] == expected
    assert session.resolve_served_checkpoint is original
    assert session.resolve_mixed_watch_opponent is original_opponent
    assert safety_veto_serving.v8_gated_checkpoint_sha256s is original_veto
    assert os.environ["SNAKE_MECHANICS_V2"] == "0"


def test_real_entrypoint_without_mac_roots(tmp_path):
    """Copy the exact sparse inventory and uploads; block original roots in child."""
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    sources = json.loads((ROOT / "research/watch_lead_p/runtime-sources.json").read_text())
    for name in sources:
        target = sparse / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    inputs = copy_inputs(tmp_path / "inputs")
    home, cwd = tmp_path / "home", tmp_path / "elsewhere"
    home.mkdir()
    cwd.mkdir()
    output = tmp_path / "smoke.json"
    # The Python environment is deliberately allowed: it supplies installed
    # third-party dependencies, never project source, checkpoints or evidence.
    script = r"""
import atexit, json, os, runpy, sys
from pathlib import Path
sparse, output, original = map(Path, sys.argv[1:])
sys.path.insert(0, str(sparse))
blocked = [original, original.parent / "snake-dqn", original.parent / "snake-dqn-artifacts"]
venv = original.parent / "snake-dqn/venv"
violations = []
def check(path):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(path)).absolute()
    if path.is_relative_to(venv):
        return
    if any(path.is_relative_to(root) for root in blocked):
        violations.append(str(path))
        raise RuntimeError("VM forbids Mac path: " + str(path))
def audit(event, args):
    if event in ("open", "os.listdir", "os.scandir"):
        check(args[0])
sys.addaudithook(audit)
# Also block existence probes; these do not emit Python audit events.
stat = os.stat
access = os.access
def guarded_stat(path, *a, **kw):
    check(path)
    return stat(path, *a, **kw)
def guarded_access(path, *a, **kw):
    check(path)
    return access(path, *a, **kw)
os.stat, os.access = guarded_stat, guarded_access
# Prove that both existing roots really are unreachable without recording the
# intentional probes as runtime violations.
for root in blocked:
    try:
        open(root / "AGENTS.md")
    except RuntimeError:
        pass
    else:
        raise AssertionError("guard inactive")
violations.clear()
sys.argv = ["P", "smoke", "--output", str(output)]
runpy.run_module("research.watch_lead_p.run", run_name="__main__")
from research.watch_lead_p.uploaded import uploaded_checkpoints
from research.watch_lead_p.live import LiveWorld
from research.watch_lead_p.engine import submit
from research.watch_lead_p.protocol import FIXTURE_SEED
with uploaded_checkpoints():
    world = LiveWorld(FIXTURE_SEED)
    while world.hero.length < 600:
        world.step()
        assert world.frame < 6000
    for delay in (0, 48):
        certificate, detail = submit(world, "engineering", delay)
        assert certificate is not None and detail["result"]
    world.hero.die()
    for _ in range(200):
        world.step()
    assert world.hero.is_alive
assert not violations, violations
assert not (sparse / "saved_snakes").exists()
print("VM_FIDELITY_PASS sparse smoke + mature search P0/P48 + respawn; forbidden_accesses=0")
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SNAKE_", "COLAB_"))}
    env.update(
        HOME=str(home),
        PYTHONPATH="",
        PYTHONDONTWRITEBYTECODE="1",
        NUMBA_CACHE_DIR=str(tmp_path / "numba"),
        COLAB_INPUT_DIR=str(inputs),
        SNAKE_DQN_DEVICE="cpu",
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(sparse), str(output), str(ROOT)],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=150,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VM_FIDELITY_PASS" in result.stdout
    value = json.loads(output.read_text())
    assert value["engineering_only"]
    assert all(row["frames"] == 60 for row in value["arms"].values())
