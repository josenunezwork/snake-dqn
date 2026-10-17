"""Prepared, create-only T6 lanes. --check validates and prints without launching."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

from research.t6_restart_20261016.recipe import REPO, RUNS, STORE, command, environment

BINDING = REPO / "research/t6_restart_20261016/launch_inputs.json"
LANES = {0: (10, 12, 14), 1: (11, 13)}


def sha256(path: Path) -> str:
    """Hash an input without reading large artifacts into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs() -> None:
    """Verify pinned recipe, immutable data, init/control identities and smoke."""
    binding = json.loads(BINDING.read_text())
    if binding["schema"] != "t6-restart-launch-inputs/v1":
        raise ValueError("Unexpected launch binding")
    for raw, expected in binding["files"].items():
        path = Path(raw)
        if sha256(path) != expected:
            raise ValueError(f"Registered input changed: {path}")
    manifest = json.loads((STORE / "manifest.json").read_text())
    for entry in manifest["snapshots"]:
        name = entry["file"]
        if Path(name).name != name or sha256(STORE / name) != entry["sha256"]:
            raise ValueError(f"Snapshot hash/path mismatch: {name}")
    for raw, expected in manifest["checkpoints"].items():
        if sha256(Path(raw)) != expected:
            raise ValueError(f"Snapshot opponent changed: {raw}")
    smoke = json.loads((RUNS / "smoke_200k_s10/summary.json").read_text())
    if smoke["halt"] or smoke["actor_exitcodes"] != [0, 0] or smoke["transitions"] < 200000:
        raise ValueError("Smoke did not complete successfully")


def arm_failed() -> bool:
    """Observe only this arm's receipts, without inspecting or signalling processes."""
    if (RUNS / "KILL.json").exists():
        return True
    for seed in range(10, 15):
        path = RUNS / f"RESTART_s{seed}/summary.json"
        if path.exists():
            summary = json.loads(path.read_text())
            if summary["halt"] or any(code != 0 for code in summary["actor_exitcodes"]):
                return True
    return False


def main() -> int:
    """Print a lane or run its sequential seeds in create-only output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=int, choices=tuple(LANES), required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    verify_inputs()
    if arm_failed():
        raise SystemExit("RESTART is KILL; refusing further seeds")
    commands = [command(seed) for seed in LANES[args.lane]]
    env = environment()
    explicit_env = {
        key: env[key]
        for key in (
            "SNAKE_DQN_DEVICE",
            "PYTHONPATH",
            "PYTHONUNBUFFERED",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
    for cmd in commands:
        out = Path(cmd[cmd.index("--out") + 1])
        if out.exists() or out.with_suffix(".log").exists():
            raise SystemExit(f"Refusing existing output/log: {out}")
    if args.check:
        print(f"# cwd: {REPO}; inherited M3C_* overrides removed")
        for cmd in commands:
            print(shlex.join(["env", *[f"{k}={v}" for k, v in explicit_env.items()], *cmd]))
        return 0
    RUNS.mkdir(parents=True, exist_ok=True)
    # Exclusive lane lock persists after success/failure: never silently rerun a lane.
    with (RUNS / f"lane{args.lane}.started.json").open("x") as stream:
        json.dump({"commands": commands, "cwd": str(REPO), "environment": explicit_env}, stream)
    for cmd in commands:
        if arm_failed():
            raise SystemExit("RESTART is KILL; refusing further seeds")
        out = Path(cmd[cmd.index("--out") + 1])
        with out.with_suffix(".log").open("x") as stream:
            result = subprocess.run(cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        summary_path = out / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        success = (
            result.returncode == 0
            and summary.get("halt") is None
            and summary.get("actor_exitcodes") == [0, 0]
            and summary.get("transitions", 0) >= 8000000
        )
        with (RUNS / f"lane{args.lane}-status.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"out": str(out), "exit_code": result.returncode, "success": success})
                + "\n"
            )
        if not success:
            try:
                with (RUNS / "KILL.json").open("x") as stream:
                    json.dump(
                        {
                            "out": str(out),
                            "exit_code": result.returncode,
                            "halt": summary.get("halt"),
                            "decision": "KILL",
                        },
                        stream,
                    )
            except FileExistsError:
                pass
            return result.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
