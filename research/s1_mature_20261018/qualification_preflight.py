"""Replay the CPU-capable S1 qualification in pinned remote layouts, without allocation.

These burned parity worlds and retained fixture streams are packaging checks,
never S1 training or efficacy evidence. CUDA admission is still mandatory on L4.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from research.s1_mature_20261018 import manifests


def _inputs(row: dict[str, Any]) -> dict[str, Path]:
    from research.s1_mature_20261018.review_jobs import verified_inputs

    local = {item.dest: item for item in verified_inputs()}
    declared = {item["dest"]: item["sha256"] for item in row["inputs"]}
    if any(declared.get(name) != item.as_manifest()["sha256"] for name, item in local.items()):
        raise ValueError("qualification inputs differ from the frozen S1 inventory")
    return {name: item.path for name, item in local.items()}


def _commands(root: Path) -> list[list[str]]:
    fixture = [sys.executable, "-m", "research.q0q1_20261017.q0b_cuda_fixtures"]
    return [
        fixture + ["--cpu-preflight", "--out", str(root / "cpu-fixtures.json")],
        [
            sys.executable,
            "-m",
            "research.q0q1_20261017.q1_parity",
            "--verify-reference",
            str(root / "inputs/mac-parity-reference.json"),
            "--bank",
            str(root / "inputs/parity-bank.json"),
            "--b8",
            str(root / "inputs/B8_s3_final.pth"),
            "--out",
            str(root / "runtime-parity.json"),
        ],
    ]


def _run_layout(root: Path, row: dict[str, Any], inputs: dict[str, Path]) -> dict[str, Any]:
    from research.s1_mature_20261018.train import _safe_extract

    staged = root / "inputs"
    staged.mkdir()
    for name, source in inputs.items():
        shutil.copyfile(source, staged / name)
    work = root / "s1-work"
    league = work / "league"
    league.mkdir(parents=True)
    _safe_extract(staged / "league.tar.gz", league)
    fixture = work / "fixture-inputs"
    (fixture / "init").mkdir(parents=True)
    (fixture / "checkpoints").symlink_to(league, target_is_directory=True)
    shutil.copyfile(staged / "B8_s3_final.pth", fixture / "init/B8_s3_final.pth")
    env = {
        **os.environ,
        **row["env"],
        "PYTHONPATH": str(root / "repo"),
        "SNAKE_DQN_DEVICE": "cpu",
        "SNAKE_CKPT_DIR": str(league),
        "Q0Q1_INPUT_ROOT": str(fixture),
        "COLAB_CLI_ROOT": str(root),
        "COLAB_INPUT_DIR": str(staged),
        "COLAB_OUTPUT_DIR": str(root / "output"),
        **{
            name: "1"
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    results = []
    for command in _commands(root):
        started = time.monotonic()
        completed = subprocess.run(
            command, cwd=root / "repo", env=env, capture_output=True, text=True, timeout=600
        )
        receipt = Path(command[-1])
        value = json.loads(receipt.read_text()) if receipt.is_file() else {}
        valid = (
            value.get("status") == "CPU_PREFLIGHT_PASS"
            if "--cpu-preflight" in command
            else value.get("pass") is True
        )
        if completed.returncode or not valid:
            raise ValueError(
                f"remote-layout qualification failed: {command[2:]}: "
                f"{value.get('error', value.get('comparison', {}))}; "
                f"{completed.stderr[-4000:]}"
            )
        results.append(
            {
                "module": command[2],
                "fixture": value.get("fixture", "runtime-parity"),
                "elapsed_seconds": time.monotonic() - started,
                "receipt_sha256": manifests.sha256_file(receipt),
                "status": "PASS_CPU_ONLY",
            }
        )
    return {"status": "PASS_CPU_ONLY", "checks": results}


def check(
    repo: Path, row: dict[str, Any], source_plan: dict[str, Any], staging: Path
) -> dict[str, Any]:
    """Check full GitHub-equivalent and exact sparse fallback inventories offline."""
    from research.colab_cli import source

    inputs = _inputs(row)
    results = {}
    for layout in ("github", "fallback"):
        root = staging / f"qualification-{layout}"
        root.mkdir()
        if layout == "fallback":
            shutil.copyfile(staging / source.ARCHIVE_NAME, root / source.ARCHIVE_NAME)
            source.install_fallback(root, source_plan)
        else:
            archive = root / "github.tar"
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "archive",
                    "--format=tar",
                    "-o",
                    str(archive),
                    row["commit"],
                ],
                check=True,
                timeout=60,
            )
            (root / "repo").mkdir()
            with tarfile.open(archive) as stream:
                stream.extractall(root / "repo", filter="data")
            archive.unlink()
        results[layout] = _run_layout(root, row, inputs)
    return {"status": "PASS_CPU_ONLY", "cuda_qualified": False, "layouts": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from research.colab_cli import source
    from research.s1_mature_20261018 import admission

    row = json.loads(args.manifest.read_text())
    with tempfile.TemporaryDirectory(prefix="s1-qualification-") as temporary:
        staging = Path(temporary)
        plan = source.prepare(args.repo, row, staging)
        result = check(args.repo, row, plan, staging)
    admission.write_once(args.out, {"commit": row["commit"], **result})
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
