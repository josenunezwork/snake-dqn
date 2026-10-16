"""Exercise fail-closed collection/evaluation namespace checks."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from research.redesign_m2_20261008 import ni_spec

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("defect", [None, "overlap", "wrong_pair", "incomplete", "split_leak"])
def test_bank_seed_audit_rejects_collisions_and_incomplete_provenance(tmp_path, defect):
    staged = tmp_path / "staged"
    staged.mkdir()
    seeds = list(range(10_000, 11_200))
    if defect == "overlap":
        seeds[0] = ni_spec.distill_seeds(96, round_index=68)[0]
    names = [f"world-{i:04d}-scripted-s{seed}-rows.npz" for i, seed in enumerate(seeds)]
    for name in names:
        (staged / name).touch()
    if defect == "incomplete":
        (staged / names[-1]).unlink()
    for kind, index in (("train", 0), ("holdout", 1)):
        bank = tmp_path / "banks" / kind
        bank.mkdir(parents=True)
        if defect == "split_leak":
            index = 0
        (bank / "manifest.json").write_text(
            json.dumps({"publication": {"world_receipts": [names[index]]}})
        )
    plans = tmp_path / "plans"
    plans.mkdir()
    for cycle in range(6):
        jobs = [
            {"plan_index": i, "world_index": i % 40, "cycle": cycle, "seed": seeds[i]}
            for i in range(cycle * 200, cycle * 200 + 200)
        ]
        if defect == "wrong_pair" and cycle == 0:
            jobs[0]["seed"], jobs[1]["seed"] = jobs[1]["seed"], jobs[0]["seed"]
        (plans / f"cycle-{cycle:04d}.json").write_text(json.dumps({"jobs": jobs}))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "research/t4e_return_20261016/calibration_seed_audit.py"),
            "--artifacts",
            str(tmp_path),
            "--out",
            str(tmp_path / "audit.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == (0 if defect is None else 1), result.stderr
    receipt = json.loads((tmp_path / "audit.json").read_text())
    assert receipt["status"] == ("PASS" if defect is None else "FAIL")
