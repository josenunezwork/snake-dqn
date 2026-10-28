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
            source = "b8_s3_t7_20261016" if checkpoint == "b8-s3" else "frp3_m3_s12_u60000_20261005"
            hero.symlink_to(CHECKPOINTS / f"{source}.pth")
    with pytest.raises(ValueError, match="P .*upload"):
        with uploaded_checkpoints():
            pytest.fail("invalid inputs admitted")
    assert session.resolve_served_checkpoint is original


def test_mac_gated_set_and_exact_choices(tmp_path, monkeypatch):
    from dataclasses import replace

    from research.watch_lead_p.live import LiveWorld
    from research.watch_lead_p.protocol import FIXTURE_SEED
    from web.backend import safety_veto_serving, session
    from web.backend.served_checkpoint import FRP3_S12_SHA256

    monkeypatch.delenv("SNAKE_MECHANICS_V2", raising=False)
    # This assertion calls the REAL Mac lookup before installing any binding.
    actual_gated = safety_veto_serving.v8_gated_checkpoint_sha256s()
    assert actual_gated == [FRP3_S12_SHA256]
    original = session.resolve_served_checkpoint
    original_opponent = session.resolve_mixed_watch_opponent
    original_veto = safety_veto_serving.v8_gated_checkpoint_sha256s
    hero = original(champion_path=session.DEFAULT_CHECKPOINT)
    opponent = original_opponent()
    inputs = copy_inputs(tmp_path / "inputs")
    monkeypatch.setenv("SNAKE_MECHANICS_V2", "0")
    with uploaded_checkpoints(inputs) as bindings:
        assert session.resolve_served_checkpoint(
            champion_path=session.DEFAULT_CHECKPOINT
        ) == replace(hero, path=str(inputs / "b8-s3.pth"))
        assert session.resolve_mixed_watch_opponent() == replace(
            opponent, path=str(inputs / "frp3-s12.pth")
        )
        assert safety_veto_serving.v8_gated_checkpoint_sha256s() == actual_gated
        for function in (
            session.resolve_served_checkpoint,
            session.resolve_mixed_watch_opponent,
            safety_veto_serving.v8_gated_checkpoint_sha256s,
        ):
            for args, kwargs in ((("other-role",), {}), ((), {"pins": {}})):
                with pytest.raises(ValueError, match="unexpected P"):
                    function(*args, **kwargs)
        for binding in bindings.values():
            binding.reset_mock()
        LiveWorld(FIXTURE_SEED)
        assert bindings["hero"].call_count == bindings["opponent"].call_count == 1
        assert bindings["v8"].call_count >= 1
        print(
            "MAC_GATED_SET_PASS",
            actual_gated,
            "real_constructor_calls=",
            {key: item.call_count for key, item in bindings.items()},
        )
    assert session.resolve_served_checkpoint is original
    assert session.resolve_mixed_watch_opponent is original_opponent
    assert safety_veto_serving.v8_gated_checkpoint_sha256s is original_veto
    assert os.environ["SNAKE_MECHANICS_V2"] == "0"


def test_adapter_full_smoke_and_mature_search_parity(tmp_path, monkeypatch):
    from research.watch_lead_p.engine import run_pair, submit
    from research.watch_lead_p.live import LiveWorld
    from research.watch_lead_p.protocol import FIXTURE_SEED

    inputs = copy_inputs(tmp_path / "inputs")
    original_init = LiveWorld.__init__

    def init_with_common_path_metadata(world, seed):
        original_init(world, seed)  # independent REAL construction on each side
        # Uploaded locations necessarily differ from registry locations. Map only
        # the two verified checkpoint path strings AFTER loading, before hashing
        # or pickling, so complete state/certificate/transport bytes can compare.
        hero_path = str(inputs / "b8-s3.pth")
        opponent_path = str(inputs / "frp3-s12.pth")
        session = world.session
        session.checkpoint_path = session.policy.checkpoint_path = hero_path
        for choice in (session.served_checkpoint, world.roster["served"]):
            choice["path"] = hero_path
        for choice in (session.watch_opponent_checkpoint, world.roster["opponent_checkpoint"]):
            choice["path"] = opponent_path

    monkeypatch.setattr(LiveWorld, "__init__", init_with_common_path_metadata)
    monkeypatch.delenv("SNAKE_MECHANICS_V2", raising=False)
    expected = run_pair(FIXTURE_SEED, frames=60)
    baseline = LiveWorld(FIXTURE_SEED)
    while baseline.hero.length < 600:
        baseline.step()
        assert baseline.frame < 6000
    expected_digest = baseline.digest()
    expected_search = [submit(baseline, "engineering", delay) for delay in (0, 48)]
    baseline.hero.die()
    expected_respawn = [baseline.step() for _ in range(200)]
    assert baseline.hero.is_alive
    expected_respawn_digest = baseline.digest()
    monkeypatch.delenv("SNAKE_MECHANICS_V2", raising=False)
    with uploaded_checkpoints(inputs):
        actual = run_pair(FIXTURE_SEED, frames=60)
        assert actual == expected  # every arm, roster, action trace and final state
        world = LiveWorld(FIXTURE_SEED)
        assert world.roster == baseline.roster
        while world.frame < expected_search[0][1]["source_frame"]:
            world.step()
        assert world.hero.length >= 600 and world.digest() == expected_digest
        actual_search = [submit(world, "engineering", delay) for delay in (0, 48)]
        from dataclasses import replace

        for (actual_cert, actual_detail), (expected_cert, expected_detail) in zip(
            actual_search, expected_search
        ):
            assert actual_cert is not None and expected_cert is not None
            # Pickle transport embeds tensor storage identities and diagnostic
            # duration counters. Each submit verifies restore digest itself;
            # compare ALL semantic certificate/detail fields, not pickle bytes.
            assert 0 < actual_cert.root_bytes <= 32 * 1024 * 1024
            assert 0 < expected_cert.root_bytes <= 32 * 1024 * 1024
            assert replace(actual_cert, root_bytes=0) == replace(expected_cert, root_bytes=0)
            transport_fields = {"root_bytes", "transport_sha256"}
            assert {k: v for k, v in actual_detail.items() if k not in transport_fields} == {
                k: v for k, v in expected_detail.items() if k not in transport_fields
            }
        print(
            "ADAPTER_PARITY_PASS full 60-tick arms+roster; L600 digest; P0/P48 semantic "
            "certificates+details; transport independently restored"
        )
        world.hero.die()
        assert [world.step() for _ in range(200)] == expected_respawn
        assert world.hero.is_alive and world.digest() == expected_respawn_digest


def test_real_entrypoint_without_mac_roots(tmp_path):
    run_vm_fixture(tmp_path)


def run_vm_fixture(
    tmp_path: Path,
    source_root: Path = ROOT,
    workers: int = 2,
    mature: bool = True,
    shared_inputs: Path | None = None,
    shared_sparse: Path | None = None,
    sessions: int = 1,
) -> None:
    """Run the guarded fixture from exact supplied source (including installed archive)."""
    sparse = shared_sparse if shared_sparse is not None else tmp_path / "sparse"
    sparse.mkdir(exist_ok=True)
    sources = json.loads((source_root / "research/watch_lead_p/runtime-sources.json").read_text())
    for name in sources:
        target = sparse / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, target)
    inputs = shared_inputs if shared_inputs is not None else copy_inputs(tmp_path / "inputs")
    home, cwd = tmp_path / "home", tmp_path / "elsewhere"
    home.mkdir()
    cwd.mkdir()
    output = tmp_path / "smoke.json"
    guard = tmp_path / "guard"
    guard.mkdir()
    shutil.copyfile(source_root / "research/watch_lead_p/vm_guard.py", guard / "sitecustomize.py")
    # The driver constrains only fixture seed/horizon/allocation size, in both
    # parent and spawn children. main(run), execute, worker, uploaded_checkpoints,
    # run_arm, publication, and ledger validation remain the real functions.
    script = tmp_path / "driver.py"
    script.write_text(r"""
import json, os, sys
from functools import partial
from pathlib import Path
from research.watch_lead_p import run
from research.watch_lead_p.protocol import FIXTURE_SEED
from research.watch_lead_p.resume import empty_ledger, read_ledger
run.SEEDS = (FIXTURE_SEED,) * int(os.environ["P_FIXTURE_SESSIONS"])
run.WORKERS = int(os.environ["P_FIXTURE_WORKERS"])
original_run_arm = run.run_arm
def fixture_arm(seed, arm, **kw):
    assert seed == FIXTURE_SEED
    row = original_run_arm(seed, arm, frames=60, **kw)
    base = Path(os.environ["P_VM_SCRATCH"])
    (base / ("worker-" + str(os.getpid()) + "-" + arm)).write_text("guarded real worker\n")
    assert os.getpriority(os.PRIO_PROCESS, 0) > os.getpriority(os.PRIO_PROCESS, os.getppid())
    return row
run.run_arm = fixture_arm
if __name__ == "__main__":
    base = Path(os.environ["P_VM_SCRATCH"])
    ledger = Path(os.environ["COLAB_INPUT_DIR"]) / "verified-ledger.json"
    ledger.write_text(json.dumps(empty_ledger()))
    sys.argv = ["P", "run", "--ledger", str(ledger), "--inputs", str(ledger.parent),
                "--output", str(base / "run-output")]
    run.main()
    assert (base / "run-output/inventory.json.complete").exists()
    unit = json.loads((base / "run-output/units/session-00.json").read_text())
    assert all(r["seed"] == FIXTURE_SEED and r["frames"] == 60 for r in unit["arms"].values())
    sys.argv = ["P", "smoke", "--output", str(base / "smoke.json")]
    run.main()
    if os.environ["P_FIXTURE_MATURE"] == "0":
        print("VM_FIDELITY_PASS real worker scheduling fixture")
        sys.exit(0)
    from research.watch_lead_p.uploaded import uploaded_checkpoints
    from research.watch_lead_p.live import LiveWorld
    from research.watch_lead_p.engine import submit
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
    assert not (Path(os.environ["P_VM_SPARSE"]) / "saved_snakes").exists()
    print("VM_FIDELITY_PASS real main(run)->execute->spawn worker + smoke + L600/P0/P48/respawn")
""")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SNAKE_", "COLAB_"))}
    env.update(
        HOME=str(home),
        PYTHONPATH=os.pathsep.join((str(guard), str(sparse))),
        P_VM_SCRATCH=str(tmp_path),
        P_VM_SPARSE=str(sparse),
        P_VM_ALLOWED=json.dumps([str(tmp_path), str(Path(sys.prefix).resolve())]),
        PYTHONDONTWRITEBYTECODE="1",
        NUMBA_CACHE_DIR=str(tmp_path / "numba"),
        COLAB_INPUT_DIR=str(inputs),
        SNAKE_DQN_DEVICE="cpu",
        SNAKE_MECHANICS_V2="0",
        P_FIXTURE_WORKERS=str(workers),
        P_FIXTURE_SESSIONS=str(sessions),
        P_FIXTURE_MATURE="1" if mature else "0",
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VM_FIDELITY_PASS" in result.stdout
    assert not list(tmp_path.glob("violation-*"))
    worker_pids = {p.name.split("-")[1] for p in tmp_path.glob("worker-*")}
    assert worker_pids
    assert all((tmp_path / f"guard-start-{pid}").exists() for pid in worker_pids)
    print(result.stdout.strip(), "workers=", sorted(worker_pids), "forbidden_accesses=0")
    value = json.loads(output.read_text())
    assert value["engineering_only"]
    assert all(row["frames"] == 60 for row in value["arms"].values())


def test_qualified_transport_publishes_regular_absolute_inputs(tmp_path):
    """Execute the qualified UPLOAD_ASSEMBLE RPC, not a stand-in file copy."""
    import hashlib

    from research.colab_cli import transfers

    # supervisor.py creates /content/snake-cli/<generation> with exist_ok=False,
    # supplies <root>/inputs as COLAB_INPUT_DIR, and mkdirs input parents.
    root = tmp_path.resolve() / "snake-cli" / "unique-generation"
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    inputs.mkdir()

    class LocalRPC:
        def recoverable_rpc(self, code, **kwargs):
            namespace = {}
            exec(code, namespace)
            return namespace["result"]

    for dest, source in (
        ("b8-s3", "b8_s3_t7_20261016"),
        ("frp3-s12", "frp3_m3_s12_u60000_20261005"),
    ):
        payload = (CHECKPOINTS / f"{source}.pth").read_bytes()
        parts = root / (dest + "-parts")
        parts.mkdir()
        (parts / "00000000.part").write_bytes(payload)
        checksum = hashlib.sha256(payload).hexdigest()
        target = inputs / f"{dest}.pth"
        spec = {
            "parts": str(parts),
            "temporary": str(inputs / (dest + ".tmp")),
            "target": str(target),
            "chunks": 1,
            "chunk_sha256": [checksum],
            "sha256": checksum,
            "size": len(payload),
        }
        assert transfers._rpc(LocalRPC(), "UPLOAD_ASSEMBLE", spec, 30) == {
            "sha256": checksum,
            "size": len(payload),
        }
        assert target.is_file() and not target.is_symlink()
    assert inputs.is_absolute() and inputs.is_dir() and not inputs.is_symlink()
    with uploaded_checkpoints(inputs):
        pass


def test_vm_guard_refuses_mac_paths_and_unguarded_children(tmp_path):
    guard = tmp_path / "guard"
    guard.mkdir()
    shutil.copyfile(ROOT / "research/watch_lead_p/vm_guard.py", guard / "sitecustomize.py")
    env = dict(
        os.environ,
        PYTHONPATH=str(guard),
        P_VM_SCRATCH=str(tmp_path),
        P_VM_ALLOWED=json.dumps([str(tmp_path), str(Path(sys.prefix).resolve())]),
    )
    script = r"""
import os, subprocess, sys
for root in sys.argv[1:]:
    path = root + "/forbidden"
    for action in (lambda: open(path), lambda: os.stat(path), lambda: os.lstat(path),
                   lambda: os.access(path, os.R_OK), lambda: os.chdir(root)):
        try:
            action()
        except RuntimeError:
            pass
        else:
            raise AssertionError("VM guard failed to refuse " + path)
for action in (lambda: subprocess.Popen(["/bin/cat", sys.argv[1]]),
               lambda: os.system("true"),
               lambda: subprocess.Popen([sys.executable, "-c", "pass"],
                                        env={**os.environ, "PYTHONPATH": ""})):
    try:
        action()
    except RuntimeError:
        pass
    else:
        raise AssertionError("VM guard allowed an unguarded child")
print("VM_GUARD_NEGATIVE_PROBES_PASS")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT),
            str(ROOT.parent / "snake-dqn"),
            str(ROOT.parent / "snake-dqn-artifacts"),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VM_GUARD_NEGATIVE_PROBES_PASS" in result.stdout
    assert (
        sum(len(path.read_text().splitlines()) for path in tmp_path.glob("violation-*.jsonl")) == 15
    )


def test_eleven_vs_twelve_workers_publish_identical_fixture_units(tmp_path):
    """Real spawned workers change scheduling only; W1/K1 is per session."""
    outputs = []
    shared_inputs = copy_inputs(tmp_path / "inputs")
    for count in (11, 12):
        root = tmp_path / str(count)
        root.mkdir()
        run_vm_fixture(
            root,
            workers=count,
            mature=False,
            shared_inputs=shared_inputs,
            shared_sparse=tmp_path / "shared-runtime",
            sessions=13,
        )
        units = sorted((root / "run-output/units").glob("session-??.json"))
        assert len(units) == 13
        records = []
        for unit in units:
            value = json.loads(unit.read_text())
            assert all(r["construction"]["mechanics_version"] == 2 for r in value["arms"].values())
            records.append(
                [
                    Path(str(unit) + suffix).read_bytes()
                    for suffix in (
                        "",
                        ".sha256",
                        ".complete",
                        ".metadata.json",
                        ".metadata.json.sha256",
                        ".metadata.json.complete",
                    )
                ]
            )
        outputs.append(records)
    assert outputs[0] == outputs[1]
    print(
        "SCHEDULING_PARITY_PASS workers=11,12 sessions=13 six components/unit byte-identical; lower priority"
    )
