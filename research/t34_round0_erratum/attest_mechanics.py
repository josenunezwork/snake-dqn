"""Attest Round 0 mechanics on exactly one excluded fixture, without study data.

Run from the repository root with nice -n 10 and the project venv:
    python -m research.t34_round0_erratum.attest_mechanics
The output directory is create-only; this is not a replay or analysis entrypoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-erratum-20261021")
FORBIDDEN = "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01"
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
CHAIN = (
    "research/t34_round0/runner.py",
    "research/t34_exit_b8/trajectory.py",
    "research/phase0_search_20261019/teacher.py",
    "research/t3_rollout_labels_20261015/bank_engine.py",
    "research/redesign_scope_20261007/grid_h5000_identity.py",
    "research/apex_safety_20260926/deployment.yaml",
    "research/apex_safety_20260926/dev_screen.py",
    "src/scripts/tournament_eval.py",
    "src/simd_env/eval_engine.py",
    "src/evaluation/protocol.py",
)


def sha256(path: Path) -> str:
    """Hash the exact file bytes recorded in the receipt."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deny_study_access(event: str, args: tuple[Any, ...]) -> None:
    """Reject accidental opens or directory scans of the embargoed study root."""
    if event in {"open", "os.listdir", "os.scandir"} and args:
        value = args[0]
        if isinstance(value, (str, bytes, os.PathLike)):
            path = os.path.realpath(os.fsdecode(value))
            if path == FORBIDDEN or path.startswith(FORBIDDEN + "/"):
                raise PermissionError("registered study artifacts are excluded from attestation")


def main() -> None:
    """Verify identities, construct one excluded world, and attest after three steps."""
    sys.addaudithook(deny_study_access)
    if platform.system() != "Darwin" or os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("attestation requires the Mac and nice value at least 10")
    for name in THREAD_ENV:
        os.environ[name] = "1"
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    sys.dont_write_bytecode = True

    package = ROOT / "research/t34_round0"
    bank_path = package / "BANK_ROUND0.json"
    registration_path = package / "REGISTRATION_ROUND0.json"
    bank = json.loads(bank_path.read_text())
    registration = json.loads(registration_path.read_text())
    if sha256(bank_path) != registration["bank_sha256"]:
        raise RuntimeError("registered bank digest differs")
    row = next(r for r in bank["excluded_preflight_rows"] if r["mix"] == "scripted")
    if not row["namespace"].startswith("t34-round0-preflight-20261020/v1/"):
        raise RuntimeError("fixture namespace differs")
    if any(r["world_seed"] == row["world_seed"] for r in bank["rows"]):
        raise RuntimeError("fixture seed overlaps the registered bank")
    hashes = {path: sha256(ROOT / path) for path in CHAIN}
    if any(registration["source_hashes"].get(path) != h for path, h in hashes.items()):
        raise RuntimeError("constructor/profile source differs from registration")
    checkpoint = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth")
    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8":
        raise RuntimeError("B8 s3 checkpoint digest differs")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    os.environ["NUMBA_CACHE_DIR"] = str(OUTPUT / "numba-cache")

    import numba
    import torch

    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t34_round0 import runner

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    numba.set_num_threads(1)
    context = gid._context(1)
    profile = context["profile"]
    world = runner.build_world(row, checkpoint)
    cfg = asdict(world.sim.cfg)
    if cfg["mechanics_version"] != 2:
        raise RuntimeError("Round 0 constructor did not select mechanics v2")
    teacher = runner.ExitTeacher(
        runner.ExitTeacherConfig(horizon=300, beta=1.0),
        device="cpu",
        policy=runner.B8Policy("cpu"),
    )
    actions = []
    for _ in range(3):
        runner.prepare_world(world)
        action = int(teacher.evaluate(world).greedy_action())
        actions.append(runner._step(world, teacher, action))
    if asdict(world.sim.cfg) != cfg:
        raise RuntimeError("simulator config changed during fixture steps")
    receipt = {
        "schema": "t34-round0-mechanics-attestation/v1",
        "result": "ATTESTED_MECHANICS_V2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "script_sha256": sha256(Path(__file__)),
        "registration_sha256": sha256(registration_path),
        "bank_sha256": sha256(bank_path),
        "constructor_source_hashes": hashes,
        "deployment_sha256": hashes["research/apex_safety_20260926/deployment.yaml"],
        "deployment_matches_registration": True,
        "profile_digest": profile.digest,
        "profile_descriptor": profile.descriptor(),
        "checkpoint_sha256": checkpoint_sha,
        "fixture": row,
        "fixture_seed_absent_from_registered_bank": True,
        "constructed_worlds": 1,
        "frames_executed": len(actions),
        "actions": actions,
        "sim_cfg": cfg,
        "mechanics_version_equals_2": world.sim.cfg.mechanics_version == 2,
        "platform": platform.platform(),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "thread_env": {name: os.environ[name] for name in THREAD_ENV},
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "numba_threads": numba.get_num_threads(),
        "study_artifact_access": "blocked by audit hook; no study outcomes read",
        "scope": "excluded preflight configuration attestation only; no scientific endpoint",
    }
    with (OUTPUT / "attestation.json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(receipt["result"], OUTPUT / "attestation.json", flush=True)


if __name__ == "__main__":
    main()
