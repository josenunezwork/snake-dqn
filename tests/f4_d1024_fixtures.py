"""Synthetic F4 D256/D1024 worlds for no-fit package, census and R-A preparation tests.

Every life, row, ledger, control and checkpoint here is fabricated in a
temporary directory. Nothing reads factory payloads, fit outputs or the reserve.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_ppo.exports import digest, export_bytes, export_json
from research.bl_residual.data import allocate_lives, world_fold
from research.bl_residual.fit import FitRecipe
from research.bl_residual.protocol import (
    F4_REGISTRATION_SHA,
    NAMESPACE,
    ROUND_26_JUDGMENT_SHA,
    allocation_hash,
    canonical_sha,
    empty_ledger,
)

ROOT = Path(__file__).resolve().parents[1]
FAKE_B8 = b"synthetic B8 bytes; never a checkpoint"
FAKE_B8_SHA = __import__("hashlib").sha256(FAKE_B8).hexdigest()
MASK = np.array([True, False] * 3)


def _row_json(ordinary: bool, decision: int) -> str:
    local = json.dumps(np.zeros((7, 31, 31), dtype=int).tolist())
    global_ = json.dumps(np.zeros((4, 37, 37), dtype=int).tolist())
    return (
        '{"local": '
        + local
        + ', "global": '
        + global_
        + ', "scalars": '
        + json.dumps([0.0] * 60)
        + ', "q": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "legal": [true, true, true, true, true, true]'
        + ', "bare": 0, "died": [false, true, false, false, false, false]'
        + ', "status": ["EXACT", "EXACT", "UNKNOWN", "EXACT", "NOT_EVALUATED", "EXACT"]'
        + ', "mass": [1.0, 0.0, 0.0, 2.0, 0.0, 3.0]'
        + f', "sampling_kind": "{"ordinary" if ordinary else "doom"}"'
        + ', "sampling_prob": 0.25'
        + f', "decision": {decision}'
        + "}"
    )


def recipe_sha() -> str:
    return canonical_sha(dataclasses.asdict(FitRecipe()))


def judgment() -> dict[str, str]:
    return {
        "schema": "bl-residual-f4-judgment/v1",
        "decision": "APPROVE",
        "judgment_sha256": ROUND_26_JUDGMENT_SHA,
        "registration_sha256": F4_REGISTRATION_SHA,
        "recipe_sha256": recipe_sha(),
    }


def _parents(prefix: str, count: int, *, fold_zero_allowed: bool) -> list[str]:
    names = []
    index = 0
    while len(names) < count:
        name = f"{prefix}-{index:05d}"
        index += 1
        if fold_zero_allowed or world_fold(name) != 0:
            names.append(name)
    return names


def build_world(root: Path) -> dict[str, Any]:
    """Two tranches: D256 (all folds) and a fold 1-4 D1024 tranche, wrappers only."""
    lives_dir = root / "lives"
    lives_dir.mkdir(parents=True)
    ordinary_row = _row_json(True, 7)
    doom_row = _row_json(False, 3)
    tranches: dict[str, list[dict[str, Any]]] = {"d256": [], "d1024": []}
    specs = [
        ("d256", _parents("d256-p", 200, fold_zero_allowed=True), 0),
        ("d1024", _parents("d1024-p", 260, fold_zero_allowed=False), 1000),
    ]
    for tranche, parents, offset in specs:
        for number, parent in enumerate(parents):
            stream = "S-Watch6" if tranche == "d256" and number < 80 else "S-gate"
            for life in range(4 if tranche == "d1024" else 3):
                rescue = life < (3 if tranche == "d1024" else 2)
                life_id = f"{tranche}/{parent}/{life}"
                path = lives_dir / f"{tranche}-{parent}-{life}.json"
                rows = [doom_row] if rescue else [ordinary_row]
                path.write_text('{"rows": [' + ", ".join(rows) + "]}")
                tranches[tranche].append(
                    {
                        "life_id": life_id,
                        "parent_world": parent,
                        "stream": stream,
                        "rescue_bearing": rescue,
                        "allocation_index": (offset + number) * 1_000_000 + life,
                        "path": path,
                    }
                )
    catalog = [{k: v for k, v in life.items() if k != "path"} for life in tranches["d256"]]
    prior = allocate_lives(catalog)
    assert prior["doses"]["256"]["ready"]
    catalog_path = root / "PUBLIC_INPUTS.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema": "t34-exit-b8-public-inputs/v1",
                "status": "PUBLIC_RELEASE_VERIFIED_BY_OWNER_ADDENDUM",
                "assets": [
                    {
                        "name": "B8_s3_final.pth",
                        "url": "https://example.invalid/synthetic/B8_s3_final.pth",
                        "size_bytes": len(FAKE_B8),
                        "sha256": FAKE_B8_SHA,
                    }
                ],
            }
        )
    )
    return {
        "d256": tranches["d256"],
        "d1024": tranches["d1024"],
        "prior": prior,
        "public_inputs": catalog_path,
    }


def d256_control(inputs: Path) -> dict[str, Any]:
    """A sealed synthetic D256 fit control over the given input directory."""
    prior = json.loads((inputs / "life-allocation.json").read_text())
    control = {
        "schema": "bl-residual-control/v1",
        "synthetic": False,
        "namespace": NAMESPACE,
        "doses": [256],
        "recipe_sha256": recipe_sha(),
        "registration_sha256": F4_REGISTRATION_SHA,
        "input_hashes": {
            name: digest(inputs / name)
            for name in ("b8.pth", "F4_REGISTRATION.md", "judgment.json", "life-allocation.json")
        },
        "series_id": NAMESPACE,
        "reserve_allocation_sha256": canonical_sha(prior["doses"]["256"]["heldout_lives"]),
    }
    control["allocation"] = allocation_hash(control)
    export_json(inputs / "verified-ledger.json", empty_ledger(NAMESPACE, control["allocation"]))
    control["ledger_sha256"] = digest(inputs / "verified-ledger.json")
    return control


def build_d256_pack(root: Path, world: dict[str, Any]) -> Path:
    inputs = root / "d256-pack/inputs"
    inputs.mkdir(parents=True)
    export_bytes(inputs / "b8.pth", FAKE_B8)
    export_bytes(
        inputs / "F4_REGISTRATION.md",
        (ROOT / "research/bl_residual/F4_REGISTRATION.md").read_bytes(),
    )
    export_json(inputs / "judgment.json", judgment())
    export_json(inputs / "life-allocation.json", world["prior"])
    export_json(inputs / "control.json", d256_control(inputs))
    return inputs.parent


def build_staging(root: Path, d256_pack: Path) -> Path:
    """Mirror f4_d256_package.stage_d1024 over the synthetic D256 pack."""
    inputs = d256_pack / "inputs"
    prior = json.loads((inputs / "control.json").read_text())
    staging = root / "staging/inputs"
    staging.mkdir(parents=True)
    for name in ("b8.pth", "F4_REGISTRATION.md", "judgment.json", "life-allocation.json"):
        export_bytes(staging / name, (inputs / name).read_bytes())
    export_bytes(staging / "d256-control.json", (inputs / "control.json").read_bytes())
    export_json(
        staging / "d256-fit.json",
        {
            "schema": "bl-residual-stage-fit/v1",
            "dose": 256,
            "namespace": prior["namespace"],
            "allocation": prior["allocation"],
            "recipe_sha256": prior["recipe_sha256"],
            "source_control_sha256": digest(inputs / "control.json"),
            "fit_sha256": "f" * 64,
            "complete": True,
            "fit_fault_free": True,
            "trajectory_unconditional_status": "NOT_OBSERVED",
        },
    )
    control = {
        "schema": "bl-residual-control/v1",
        "synthetic": False,
        "namespace": prior["namespace"],
        "doses": [1024],
        "recipe_sha256": prior["recipe_sha256"],
        "series_id": prior["series_id"],
        "reserve_allocation_sha256": prior["reserve_allocation_sha256"],
        "stage_receipts": {"D256": "d256-fit.json", "D256_CONTROL": "d256-control.json"},
        "input_hashes": {
            name: digest(staging / name)
            for name in (
                "b8.pth",
                "F4_REGISTRATION.md",
                "judgment.json",
                "life-allocation.json",
                "d256-fit.json",
                "d256-control.json",
            )
        },
    }
    control["allocation"] = allocation_hash(control)
    export_json(staging / "verified-ledger.json", empty_ledger(NAMESPACE, control["allocation"]))
    control["ledger_sha256"] = digest(staging / "verified-ledger.json")
    export_json(staging / "control.json", control)
    return staging


def patch_package(monkeypatch: Any, world: dict[str, Any], calls: dict[str, Any]) -> None:
    """Replace only publication/source/transport boundaries; keep science code real."""
    from research.bl_residual import f4_d256_package as d256
    from research.bl_residual import f4_d1024_package as package
    from research.bl_residual import job

    monkeypatch.setattr(d256, "require_window_closed", lambda: None)
    monkeypatch.setattr(d256, "advisory_from_row", lambda _row: MASK.copy())
    monkeypatch.setattr(package, "B8_SHA", FAKE_B8_SHA)
    monkeypatch.setattr(package, "PUBLIC_INPUTS", world["public_inputs"])
    monkeypatch.setattr(job, "B8_SHA", FAKE_B8_SHA)
    monkeypatch.setattr(
        package,
        "require_publication",
        lambda local, relative, sha: calls.setdefault("published", []).append(relative),
    )
    monkeypatch.setattr(package, "_require_source", lambda commit: "c" * 40)
    monkeypatch.setattr(package, "_require_preflight", lambda *_: {"fixture": True})
    monkeypatch.setattr(
        package,
        "require_repaired_transport",
        lambda *_: {
            "tree_hash": package.QUALIFIED_TREE,
            "qualified_commit": "c" * 40,
            "acceptance_result": "/fixture/acceptance-result.json",
            "acceptance_result_sha256": "a" * 64,
            "d1024_synthetic_inventory_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(package, "require_d256_fit_publication", lambda *_: None)
    monkeypatch.setattr("tools.ops.qualification.check", lambda *_: package.QUALIFIED_TREE)
    monkeypatch.setattr("tools.ops.import_preflight.check", lambda *_: {"fixture": "imports"})
    monkeypatch.setattr(
        "research.colab_cli.supervisor.validate",
        lambda manifest, repo: calls.setdefault("validated", []).append(manifest["job_id"]),
    )

    def catalog(ledger: Path) -> list[dict[str, Any]]:
        return [dict(life) for life in world[ledger.name.split(".")[0]]]

    monkeypatch.setattr(d256, "_life_catalog", catalog)
