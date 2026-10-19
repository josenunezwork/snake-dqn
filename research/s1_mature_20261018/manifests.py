"""Create the ten pinned, approval-gated S1 mature-training manifests.

This module only prepares JSON.  It never invokes the Colab CLI or allocates a
runtime.  A caller supplies the published commit and all external inputs after
the registration has been approved; intentionally, a DRAFT cannot be rendered
into a runnable production manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

APPROVAL = "APPROVED_S1_MATURE_20261018"
ARMS = ("n3", "n1")
TRAINING_DECISIONS = 8_000_000
L4_RATE_UNITS_PER_HOUR = 2.0
ESTIMATE_MINUTES = {"n3": 30, "n1": 68}
TIMEOUT_MINUTES = {"n3": 60, "n1": 136}
MAX_EXPORT_BYTES = 64 * 1024**2
RELEASE_BASE = "https://github.com/josenunezwork/snake-dqn/releases/download/s1-inputs-20261019"
RUNTIME_REQUIREMENTS = "research/s1_mature_20261018/requirements-runtime.txt"
# Explicitly retain qualification and trainer imports in the outage fallback.
RUNTIME_SOURCE_PATHS = (
    "src",
    "configs",
    "web/backend",
    "research/colab_cli",
    "research/s1_mature_20261018",
    "research/q0q1_20261017",
    "research/redesign_m3c_20261008",
    "research/redesign_m2_20261008/gen_data.py",
    "research/redesign_m2_20261008/ni_spec.py",
    "research/redesign_scope_20261007",
    "research/t23_r0_screen_20261017",
    "research/apex_safety_20260926/dev_screen.py",
    "research/apex_safety_20260926/deployment.yaml",
    "tests/__init__.py",
    "tests/test_t22_frozen_pool.py",
    "tests/test_t22_random_safe.py",
    "tests/test_t24_v8_bodymap.py",
    "tests/fixtures/t22",
    "tests/fixtures/t24",
    "requirements.txt",
    "pyproject.toml",
)


class ManifestRefused(ValueError):
    """The preregistration inputs are insufficient to make a runnable job."""


@dataclass(frozen=True)
class ExternalInput:
    """Immutable local evidence, optionally delivered from the approved release."""

    name: str
    path: Path
    dest: str
    url: str | None = None

    def as_manifest(self) -> dict[str, Any]:
        path = self.path.resolve(strict=True)
        if not path.is_file():
            raise ManifestRefused(f"input is not a regular file: {path}")
        row = {"sha256": sha256_file(path), "dest": self.dest}
        if self.url is not None:
            return {**row, "url": self.url, "size_bytes": path.stat().st_size}
        return {**row, "path": str(path)}

    def from_release(self) -> ExternalInput:
        """Retain the registered bytes while changing only their transport."""
        name = "SEED_BANK.json" if self.dest == "seed-bank.json" else self.dest
        return ExternalInput(self.name, self.path, self.dest, f"{RELEASE_BASE}/{name}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_seed_bank(path: Path) -> list[int]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestRefused(f"invalid seed bank: {path}") from exc
    seeds = value.get("seeds") if isinstance(value, Mapping) else None
    if (
        not isinstance(seeds, list)
        or len(seeds) != 5
        or len(set(seeds)) != 5
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ManifestRefused(
            "seed bank must contain exactly five distinct nonnegative integer seeds"
        )
    return list(seeds)


def _require_published_commit(commit: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ManifestRefused("published_commit must be a full 40-character Git SHA")


def namespace(arm: str, seed: int) -> str:
    if arm == "n3":
        return f"s1-mature-20261018/train/n3/seed-{seed}"
    if arm == "n1":
        # The trainer validates this legacy prefix for the task-aligned collector.
        return f"q1-task-aligned-n1-20261017/s1-mature-20261018/n1/seed-{seed}"
    raise ManifestRefused(f"unknown arm: {arm}")


def training_command(arm: str, seed: int) -> list[str]:
    """Return the runtime-relative real M3-C invocation for one S1 arm."""
    if arm not in ARMS:
        raise ManifestRefused(f"unknown arm: {arm}")
    seats, actors = (3, 6) if arm == "n3" else (1, 3)
    collection = "legacy" if arm == "n3" else "task-aligned-n1"
    return [
        "python",
        "-m",
        "research.redesign_m3c_20261008.train_m3c",
        "--seed",
        str(seed),
        "--init",
        "${S1_INIT_PATH}",
        "--demo-data",
        "${S1_DEMO_DIR}",
        "--out",
        "${S1_RUN_DIR}",
        "--transitions",
        str(TRAINING_DECISIONS),
        "--device",
        "cuda",
        "--actor-device",
        "cuda",
        "--worlds",
        "128",
        "--actors",
        str(actors),
        "--learner-seats",
        str(seats),
        "--checkpoint-every",
        "9000000",
        "--full-state-every",
        "0",
        "--final-target-weights",
        "--reward",
        "mass",
        "--mass-kappa",
        "1.0",
        "--kill-scale",
        "0",
        "--schedule-package",
        "mature",
        "--mixed-frac",
        "0.25",
        "--anchor-mode",
        "v8margin",
        "--onpolicy-labels",
        "--label-frac",
        "1.0",
        "--obs-version",
        "bv8",
        "--v61-backend",
        "numba",
        "--learner-fast",
        "--learner-precision",
        "fp32",
        "--batch-scale",
        "2",
        "--boost-halt-rate",
        "0.02",
        "--boost-halt-low-length-rate",
        "0.01",
        "--ring-dir",
        "/dev/shm",
        "--compile-learner",
        "reduce-overhead",
        "--league-b8-checkpoint",
        "${S1_INIT_PATH}",
        "--gather-threads",
        "3",
        "--actor-queue-frames",
        "2",
        "--collection-profile",
        collection,
        "--namespace",
        namespace(arm, seed),
    ]


def make_manifests(
    *,
    seed_bank: Path,
    published_commit: str,
    registration_status: str,
    owner_release: str,
    output_root: Path,
    inputs: Iterable[ExternalInput],
) -> list[dict[str, Any]]:
    """Build the fixed ten-job L4 plan; callers write each object create-only."""
    _require_published_commit(published_commit)
    seeds = _load_seed_bank(seed_bank)
    input_rows = [item.as_manifest() for item in inputs]
    names = {item["dest"] for item in input_rows}
    if len(names) != len(input_rows):
        raise ManifestRefused("uploaded input destinations must be unique")
    required = {
        "seed-bank.json",
        "B8_s3_final.pth",
        "demo.tar.gz",
        "league.tar.gz",
        "mac-parity-reference.json",
        "parity-bank.json",
        "q0b-l4.json",
        "cli-acceptance-result.json",
    }
    if not required.issubset(names):
        raise ManifestRefused(f"missing required uploaded inputs: {sorted(required - names)}")
    manifests: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            job_id = f"s1-mature-{arm}-{seed}"
            output = (output_root / job_id).resolve()
            manifests.append(
                {
                    "job_id": job_id,
                    "commit": published_commit,
                    "script": "research/s1_mature_20261018/train.sh",
                    "source_paths": list(RUNTIME_SOURCE_PATHS),
                    "env": {
                        "COLAB_CLI_REQUIREMENTS": RUNTIME_REQUIREMENTS,
                        "PYTHONUNBUFFERED": "1",
                        "S1_ARM": arm,
                        "S1_SEED": str(seed),
                        "S1_REGISTRATION_STATUS": registration_status,
                        "S1_OWNER_RELEASE": owner_release,
                        "S1_TRAINING_NAMESPACE": namespace(arm, seed),
                    },
                    "requirements": {
                        "gpu": "L4",
                        "allowed_gpus": ["L4"],
                        "min_vcpu": 12,
                        "est_minutes": ESTIMATE_MINUTES[arm],
                    },
                    "timeout_minutes": TIMEOUT_MINUTES[arm],
                    "output_dir": str(output),
                    "inputs": input_rows,
                    "budget": {
                        "unit_cap": TIMEOUT_MINUTES[arm] * L4_RATE_UNITS_PER_HOUR / 60,
                        "rate_units_per_hour": L4_RATE_UNITS_PER_HOUR,
                        "reserve_units": 0,
                    },
                    "lease_seconds": 180,
                    "max_export_bytes": MAX_EXPORT_BYTES,
                }
            )
    return manifests
