"""Explicitly requested, bounded CPU integration check using the real T4c inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def _run_bounded(
    command: list[str], root: Path, env: dict, timeout: float = 180
) -> subprocess.CompletedProcess:
    """Own a fresh process group, including actors, and clean it up on timeout."""
    process = subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        # Only this runner's newly created session is signaled. SIGINT lets the
        # trainer's finally block stop its actors and unlink its own rings.
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=40)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate(timeout=10)
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def main() -> int:
    """Run 512 transitions in an isolated output and bind the resulting receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"Refusing existing smoke output: {out}")
    root = Path(__file__).resolve().parents[2]
    artifacts = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
    binding_path = root / "research/t4c_20261016/launch_inputs.json"
    binding = json.loads(binding_path.read_text())
    bank = artifacts / "t4c-data-20261016/bank-v3"
    for directory in (bank, artifacts / "t4c-data-20261016/outside-bank"):
        manifest = json.loads((directory / "manifest.json").read_text())
        if {p.name for p in directory.glob("*.npz")} != set(manifest["shards"]):
            raise SystemExit(f"Bank shard set differs from its manifest: {directory}")
    init = artifacts / "redesign-m4b-mac-20261013/B8_s0/final.pth"
    demos = artifacts / "redesign-m2b-20261008/data"
    held = sorted((demos / "round-12").glob("chunk-*-part-*.npz"))[0]
    for path in (
        bank / "manifest.json",
        bank / "bank-00000.npz",
        artifacts / "t4c-data-20261016/outside-bank/manifest.json",
        artifacts / "t4c-data-20261016/outside-bank/bank-00000.npz",
        init,
        demos / "margin.json",
        held,
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != binding["files"][str(path)]:
            raise SystemExit(f"Registered smoke input changed: {path}")
    command = [
        sys.executable,
        "research/redesign_m3c_20261008/train_m3c.py",
        "--seed",
        "9",
        "--init",
        str(init),
        "--demo-data",
        str(demos),
        "--out",
        str(out),
        "--device",
        "cpu",
        "--actors",
        "1",
        "--worlds",
        "2",
        "--transitions",
        "512",
        "--warmup",
        "64",
        "--replay",
        "512",
        "--segment-frames",
        "500",
        "--frozen-frac",
        "0",
        "--mixed-frac",
        "0",
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
        "--obs-version",
        "bv8",
        "--anchor-mode",
        "v8margin",
        "--td-huber-delta",
        "1.0",
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
        str(bank),
        "--label-bank-frac",
        "0.5",
        "--label-bank-outside",
        str(artifacts / "t4c-data-20261016/outside-bank"),
        "--label-bank-outside-margin-scale",
        "1",
        "--label-bank-margin-scale",
        "4.75",
        "--label-bank-counter-mode",
        "single-action",
        "--label-bank-counter-ratio",
        "1",
        "--learner-fast",
        "--obs-pack",
        "--v61-backend",
        "numba",
        "--no-guard",
        "--ring-dir",
        str(out.parent),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "SNAKE_DQN_DEVICE": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    sources = (
        "research/redesign_m3c_20261008/train_m3c.py",
        "research/redesign_m3c_20261008/learner_m3c.py",
        "research/t3_rollout_labels_20261015/bank.py",
    )
    source_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
    }
    try:
        result = _run_bounded(command, root, env)
    except subprocess.TimeoutExpired:
        print(
            "CPU smoke exceeded 180 seconds; its owned process group was stopped.", file=sys.stderr
        )
        return 124
    if result.returncode:
        failure_path = out.parent / f"{out.name}-failure.json"
        with failure_path.open("x") as stream:
            json.dump(
                {
                    "schema": "t4c-real-bank-cpu-smoke-failure/v1",
                    "command": command,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "source_sha256": source_hashes,
                },
                stream,
                indent=2,
            )
        print(result.stdout[-4000:] + result.stderr[-4000:], file=sys.stderr)
        return result.returncode
    (out / "stdout.log").write_text(result.stdout)
    (out / "stderr.log").write_text(result.stderr)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["halt"] is None and summary["actor_exitcodes"] == [0], summary
    assert summary["transitions"] >= 512 and summary["updates"] > 0, summary
    assert summary["args"]["device"] == "cpu"
    assert summary["m3c"]["label_bank_counter_mode"] == "single-action"
    assert summary["m3c"]["label_bank_counter_ratio"] == 1.0
    # m3c_flags deliberately emits only non-default values; CLI args retain all values.
    assert float(summary["args"]["label_bank_outside_margin_scale"]) == 1.0
    assert summary["m3c"].get("label_bank_outside_margin_scale", 1.0) == 1.0
    assert summary["m3c"]["label_bank_outside"]
    assert (out / "final.pth").is_file()
    assert all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in source_hashes.items()
    ), "Smoke source changed during execution; rerun on the final source"
    (out / "stdout.log").write_text(result.stdout)
    (out / "stderr.log").write_text(result.stderr)
    receipt = {
        "schema": "t4c-real-bank-cpu-smoke/v1",
        "authority": "512-transition integration smoke only; no registered 8M training",
        "command": command,
        "returncode": result.returncode,
        "device": "cpu",
        "transitions": summary["transitions"],
        "updates": summary["updates"],
        "halt": summary["halt"],
        "actor_exitcodes": summary["actor_exitcodes"],
        "wall_seconds": summary["wall_seconds"],
        "m3c": summary["m3c"],
        "output": str(out),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "final_sha256": hashlib.sha256((out / "final.pth").read_bytes()).hexdigest(),
        "inputs_binding_sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "limits": "Short CLI smoke does not reach the 100k telemetry interval or 8M readout; "
        "side metrics and mixed anchors are checked by deterministic learner tests.",
    }
    (out / "smoke-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
