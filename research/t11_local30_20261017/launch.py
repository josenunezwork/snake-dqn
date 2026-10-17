"""Prepared, create-only T11 LOCAL30 lanes; ``--check`` never starts training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO.parent / "snake-dqn-artifacts"
RUNS = ARTIFACTS / "redesign-t11-mac-20261017"
BANK = ARTIFACTS / "redesign-t11-mac-20261017/bank-local30"
BINDING = REPO / "research/t11_local30_20261017/launch_inputs.json"
LANES = {0: (10, 12, 14), 1: (11, 13)}
ARM = "LOCAL30"


def sha256(path: Path) -> str:
    """Return the streaming SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def output_path(seed: int, *, smoke: bool = False) -> Path:
    """Return the prescribed output directory for one T11 command."""
    return RUNS / "validation/smoke-mps" if smoke else RUNS / f"{ARM}_s{seed}"


def command(seed: int, *, smoke: bool = False) -> list[str]:
    """Build the registered TWO_SIDED command without executing it."""
    if seed not in range(10, 15):
        raise ValueError("T11 seeds are 10 through 14")
    if smoke and seed != 10:
        raise ValueError("The registered T11 smoke uses seed 10")
    result = [
        str(REPO / "venv/bin/python"),
        str(REPO / "research/redesign_m3c_20261008/train_m3c.py"),
        "--seed",
        str(seed),
        "--init",
        str(ARTIFACTS / f"redesign-m4b-mac-20261013/B8_s{seed - 10}/final.pth"),
        "--out",
        str(output_path(seed, smoke=smoke)),
        "--demo-data",
        str(ARTIFACTS / "redesign-m2b-20261008/data"),
        "--transitions",
        "200000" if smoke else "8000000",
        "--device",
        "mps",
        "--actors",
        "2",
        "--worlds",
        "128",
        "--segment-frames",
        "5000",
        "--reward",
        "mass",
        "--mass-kappa",
        "1.0",
        "--kill-scale",
        "0",
        "--gamma-start",
        "0.998",
        "--gamma-end",
        "0.998",
        "--gamma-hold",
        "0",
        "--gamma-anneal-end",
        "1",
        "--eps-end-at",
        "1",
        "--mixed-frac",
        "0.25",
        "--anchor-mode",
        "v8margin",
        "--anchor-hold",
        "0",
        "--anchor-floor-at",
        "1",
        "--anchor-floor",
        "0.1",
        "--onpolicy-labels",
        "--label-frac",
        "0.5",
        "--label-bank",
        str(BANK),
        "--label-bank-frac",
        "0.5",
        "--label-bank-counter-mode",
        "single-action",
        "--label-bank-counter-ratio",
        "1",
        "--label-bank-margin-scale",
        "4.75",
        "--td-huber-delta",
        "1.0",
        "--obs-version",
        "bv8",
        "--v61-backend",
        "numba",
        "--learner-fast",
        "--batch-scale",
        "2",
        "--obs-pack",
        "--tight-telemetry",
        "--checkpoint-every",
        "2000000",
    ]
    result.extend(
        [
            "--boost-halt-rate",
            "0.02",
            "--boost-halt-low-length-rate",
            "0.01",
            "--arm-halt-file",
            str(RUNS / "validation/KILL.json" if smoke else RUNS / "KILL.json"),
        ]
    )
    return result


def environment() -> dict[str, str]:
    """Pin the MPS launcher environment and discard inherited trainer overrides."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("M3C_")}
    env.update(PYTHONPATH=str(REPO), SNAKE_DQN_DEVICE="mps", PYTHONUNBUFFERED="1")
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"
    return env


def _verify_manifest() -> None:
    manifest_path = BANK / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shards = manifest.get("shards")
    if not isinstance(shards, dict) or not shards:
        raise ValueError("LOCAL30 manifest must declare a nonempty shard map")
    declared = set(shards)
    actual = {path.name for path in BANK.glob("*.npz")}
    if actual != declared:
        raise ValueError("LOCAL30 bank shard set differs from manifest")
    for name, expected in shards.items():
        path = BANK / name
        if Path(name).name != name or sha256(path) != expected:
            raise ValueError(f"LOCAL30 bank shard hash/path mismatch: {name}")


def verify_inputs() -> None:
    """Verify every pre-registered input plus the entire LOCAL30 shard set."""
    binding = json.loads(BINDING.read_text())
    if binding.get("schema") != "t11-launch-inputs/v1":
        raise ValueError("Unexpected T11 launch binding")
    files = binding.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("T11 launch binding has no registered files")
    for raw, expected in files.items():
        path = Path(raw)
        if sha256(path) != expected:
            raise ValueError(f"Registered input changed: {path}")
    _verify_manifest()


def _kill(reason: str, *, seed: int | None = None, exit_code: int | None = None) -> None:
    """Write the single shared, create-only arm KILL receipt."""
    payload: dict[str, object] = {"arm": ARM, "decision": "KILL", "reason": reason}
    if seed is not None:
        payload["seed"] = seed
    if exit_code is not None:
        payload["exit_code"] = exit_code
    try:
        with (RUNS / "KILL.json").open("x") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
    except FileExistsError:
        pass


def arm_failed() -> bool:
    """Read only this arm's receipts; never inspect or signal other processes."""
    if (RUNS / "KILL.json").exists():
        return True
    for seed in range(10, 15):
        summary_path = output_path(seed) / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary.get("halt") or any(code != 0 for code in summary.get("actor_exitcodes", [])):
                return True
    return False


def _existing_output(commands: list[list[str]]) -> Path | None:
    for cmd in commands:
        out = Path(cmd[cmd.index("--out") + 1])
        if out.exists() or out.with_suffix(".log").exists():
            return out
    return None


def main() -> int:
    """Print or sequentially run the chosen prepared lane."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=int, choices=tuple(LANES), required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    verify_inputs()
    commands = [command(seed) for seed in LANES[args.lane]]
    existing = _existing_output(commands)
    if existing is not None:
        raise SystemExit(f"Refusing existing output/log: {existing}")
    env = environment()
    if args.check:
        if arm_failed():
            raise SystemExit(f"{ARM} is KILL; refusing further seeds")
        explicit = {
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
            print(shlex.join(["env", *[f"{key}={value}" for key, value in explicit.items()], *cmd]))
        return 0

    RUNS.mkdir(parents=True, exist_ok=True)
    lock = RUNS / ".t11-local30-lane-lock"
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            time.sleep(10)
    try:
        with (RUNS / f"lane{args.lane}.started.json").open("x") as stream:
            json.dump({"commands": commands, "cwd": str(REPO)}, stream, sort_keys=True)
            stream.write("\n")
        verify_inputs()
        if arm_failed():
            _kill("prior_training_halt_or_nonzero_exit")
            raise SystemExit(f"{ARM} is KILL; refusing further seeds")
        for seed, cmd in zip(LANES[args.lane], commands, strict=True):
            verify_inputs()
            if arm_failed():
                raise SystemExit(f"{ARM} is KILL; refusing further seeds")
            out = Path(cmd[cmd.index("--out") + 1])
            try:
                with out.with_suffix(".log").open("x") as stream:
                    result = subprocess.run(
                        cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT
                    )
                summary_path = out / "summary.json"
                summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            except Exception as exc:
                _kill(f"launcher_exception:{type(exc).__name__}", seed=seed)
                raise
            halted = bool(summary.get("halt"))
            success = (
                result.returncode == 0
                and not halted
                and summary.get("actor_exitcodes") == [0, 0]
                and summary.get("transitions", 0) >= 8000000
            )
            with (RUNS / f"lane{args.lane}-status.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {"out": str(out), "exit_code": result.returncode, "success": success}
                    )
                    + "\n"
                )
            if not success:
                _kill("training_halt_or_nonzero_exit", seed=seed, exit_code=result.returncode)
                return result.returncode or 1
    finally:
        lock.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
