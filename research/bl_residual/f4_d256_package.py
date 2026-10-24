"""Prepare the registered F4 D256 fit package without reporting label outcomes.

This is a mechanical boundary: it consumes only a published F0 union ledger and
its immutable life exports, derives the registered ancestry allocation, and
writes the D256 training partition.  It never invokes ``fit_residual``, a
held-out/reserve evaluator, or an aggregate statistics routine.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_ppo.exports import digest, export_bytes, export_json, verify_export
from research.bl_residual.data import allocate_lives, validate_rows
from research.bl_residual.factory_masks import advisory_from_row
from research.bl_residual.fit import FitRecipe
from research.bl_residual.protocol import (
    B8_SHA,
    F4_REGISTRATION_SHA,
    NAMESPACE,
    QUALIFICATION_RECORD,
    QUALIFIED_TREE,
    allocation_hash,
    canonical_sha,
    empty_ledger,
    require_judgment,
)

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = "research/bl_residual/requirements.txt"
SOURCE_PATHS = [
    "src",
    "research/bl_residual",
    "research/a100_a2/__init__.py",
    "research/a100_a2/model.py",
    "research/bl_ppo/__init__.py",
    "research/bl_ppo/exports.py",
    "research/colab_cli",
    "research/metadata_archive.py",
]
FROZEN_REPACK_SOURCE = "c2c82ff240a28247dd9b18608831eecea6d94205"
REPACK_ALLOWED_SOURCE_PATHS = {
    "research/bl_residual/f4_d256_package.py",
    "research/bl_residual/f4_d256_colab.sh",
    "research/bl_residual/RUN_D256_NEXT_METADATA_20261024.sh",
    "research/metadata_archive.py",
    "tests/test_bl_residual_f4_d256_package.py",
    "tests/test_metadata_archive.py",
}
PUBLICATION_PATH = "research/bl_labels/ledger/D256_UNION_FINAL_20261023.json"
UNION_SHA = "30919209b62307f8d4392b4e5925a7c4b0c4a21a2abd55a5e647428453eeb809"
FIT_PUBLICATION_PATH = "research/bl_residual/ledger/F4_D256_FIT_PHASE_20261023.json"
TIMING_ARM = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/mac-timing-armed")


def require_window_closed() -> None:
    """Refuse to compete with the exclusive P1 window; never remove its arm."""
    if TIMING_ARM.exists():
        raise ValueError("HOLD: P1/timing window remains armed")


def require_publication(factory_ledger: Path) -> None:
    """Verify publication before opening any label payload or allocation flags."""
    if digest(factory_ledger) != UNION_SHA:
        raise ValueError("foreign final D256 union ledger")
    published = subprocess.check_output(
        ["git", "show", f"origin/main:{PUBLICATION_PATH}"], cwd=ROOT, timeout=30
    )
    if published != factory_ledger.read_bytes():
        raise ValueError("local D256 ledger differs from origin/main")
    if (ROOT / PUBLICATION_PATH).read_bytes() != published:
        raise ValueError("checkout D256 ledger differs from origin/main")


def _copy(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(target)
    shutil.copyfile(source, target)


def _life_catalog(factory_ledger: Path) -> list[dict[str, Any]]:
    """First streaming pass retains wrapper metadata only, never rows/outcomes."""
    # The F0 ledger reader is intentionally Mac-only.  The Colab runtime
    # imports this module for a canary but never opens a factory ledger.
    from research.bl_labels.exports import load_ledger

    value = json.loads(factory_ledger.read_text())
    namespace, allocation = value.get("namespace"), value.get("allocation")
    if not isinstance(namespace, str) or not isinstance(allocation, list):
        raise ValueError("published F0 ledger lacks its immutable contract")
    names = load_ledger(factory_ledger, namespace, allocation)
    by_parent = {row["allocation_id"]: row for row in allocation}
    lives: list[dict[str, Any]] = []
    for name in sorted(names):
        if "--life-" not in name:
            continue
        payload = factory_ledger.parent / "lives" / f"{name}.json"
        unit = json.loads(payload.read_text())
        if unit.get("schema") != "bl-labels-f0-life/v1" or unit.get("status") != "COMPLETE":
            raise ValueError("published ledger contains a non-complete life")
        required = ("life_id", "parent_world", "stream", "rescue_bearing", "allocation_index")
        if any(key not in unit for key in required):
            raise ValueError("published life lacks allocation metadata")
        parent = unit["parent_world"]
        expected = by_parent.get(parent)
        number = unit.get("life")
        if (
            expected is None
            or unit.get("namespace") != namespace
            or unit.get("allocation") != expected
            or unit.get("allocation_id") != parent
            or type(number) is not int
            or number < 0
            or unit["life_id"] != f"{namespace}/{parent}/{number}"
            or unit["allocation_index"] != expected["allocation_index"] * 1_000_000 + number
            or type(unit["rescue_bearing"]) is not bool
            or unit["stream"] != ("S-Watch6" if expected["stream"] == "watch6" else "S-gate")
        ):
            raise ValueError("foreign D256 life wrapper identity")
        lives.append({key: unit[key] for key in required} | {"path": payload})
    if not lives:
        raise ValueError("published ledger contains no completed lives")
    return lives


def _table(
    lives: list[dict[str, Any]], train_lives: set[str], train_parents: set[str]
) -> dict[str, np.ndarray]:
    """Convert selected lives to the strict pickle-free residual training schema."""
    values: list[tuple[str, str, dict[str, Any]]] = []
    for life in lives:
        life_id = str(life["life_id"])
        parent = str(life["parent_world"])
        selected = life_id in train_lives
        if not selected and (life["rescue_bearing"] or parent not in train_parents):
            continue
        unit = json.loads(Path(life["path"]).read_text())
        rows = unit.get("rows")
        if not isinstance(rows, list):
            raise ValueError("selected factory life lacks rows")
        for row in rows:
            # Only selected rescue lives contribute decisive rows.  The shared
            # parent contributes non-rescue ordinary rows as job.run permits.
            if selected or str(row.get("sampling_kind", "")) == "ordinary":
                values.append((life_id, parent, row))
    if not values:
        raise ValueError("D256 selected lives have no training rows")

    def array(key: str, dtype: Any) -> np.ndarray:
        return np.asarray([row[key] for _, _, row in values], dtype=dtype)

    status = np.asarray(
        [
            [str(item) if item is not None else "NOT_EVALUATED" for item in row["status"]]
            for _, _, row in values
        ]
    )
    died = np.asarray(
        [
            [bool(item) if item is not None else False for item in row["died"]]
            for _, _, row in values
        ]
    )
    mass = np.asarray(
        [
            [float(item) if item is not None else 0.0 for item in row["mass"]]
            for _, _, row in values
        ],
        dtype=np.float32,
    )
    ordinary = np.asarray(
        [str(row.get("sampling_kind", "")) == "ordinary" for _, _, row in values], dtype=bool
    )
    result = {
        "local": array("local", np.uint8),
        "global": array("global", np.uint8),
        "scalars": array("scalars", np.float32),
        "q": array("q", np.float32),
        "legal": array("legal", bool),
        # F4 R3 requires B8's advisory selection mask.  ``legal`` is only the
        # physical action set and must never be silently substituted here.
        "action_mask": np.asarray([advisory_from_row(row) for _, _, row in values], dtype=bool),
        "bare": array("bare", np.int64),
        "died": died,
        "status": status,
        "mass": mass,
        "ordinary": ordinary,
        "natural_stream": ordinary.copy(),
        "sampling_weight": np.asarray(
            [1.0 / float(row.get("sampling_prob", 1.0)) for _, _, row in values], dtype=np.float32
        ),
        "sampling_prob": np.asarray(
            [float(row.get("sampling_prob", 1.0)) for _, _, row in values], dtype=np.float32
        ),
        "parent_world": np.asarray([parent for _, parent, _ in values]),
        "life_id": np.asarray([life for life, _, _ in values]),
        "row_id": np.asarray([f"{life}/{int(row['decision'])}" for life, _, row in values]),
    }
    validate_rows(result)
    return result


def _metadata_archive_sha256(inventory: list[dict[str, Any]]) -> str:
    """Return the sole supervisor-bound metadata archive digest."""
    rows = [row for row in inventory if row.get("dest") == "metadata.tar"]
    if len(rows) != 1 or set(rows[0]) != {"path", "dest", "sha256"}:
        raise ValueError("F4 package requires exactly one metadata archive input")
    digest_value = rows[0]["sha256"]
    if not isinstance(digest_value, str) or len(digest_value) != 64:
        raise ValueError("F4 metadata archive digest is invalid")
    return digest_value


def manifests(
    output: Path, commit: str, inventory: list[dict], *, attempt: str = "01"
) -> tuple[dict, dict]:
    """Freeze bounded D256/canary reservations within the 20-unit fit series."""
    main_template = {
        "job_id": f"f4-d256-fit-{commit[:8]}-{attempt}",
        "commit": commit,
        "script": "research/bl_residual/f4_d256_colab.sh",
        "source_paths": SOURCE_PATHS,
        "device": "cuda:0",
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 1, "est_minutes": 30},
        "timeout_minutes": 120,
        "output_dir": str((output / "d256-output-l4-01").resolve()),
        "inputs": inventory,
        "required_exports": ["fit/allocation.json", "fit/fit-0000.pth", "fit/receipt.json"],
        "env": {
            "F4_MODE": "main",
            "F4_DEVICE": "cuda:0",
            "COLAB_CLI_REQUIREMENTS": REQUIREMENTS,
            "METADATA_ARCHIVE_SHA256": _metadata_archive_sha256(inventory),
        },
        "budget": {
            "unit_cap": 4,
            "rate_units_per_hour": 1.54,
            "reserve_units": 0.8,
            "other_assignments": {},
        },
        "resume": {
            "unit_kind": "complete-dose-fit",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "max_export_bytes": 128 * 1024**2,
        "f4": {
            "dose": 256,
            "fit_series_base_units": 10,
            "fit_series_effective_units": 20,
            "registered_updates": 2000,
            "qualified_tree": QUALIFIED_TREE,
            "base_units": 2,
            "effective_units": 4,
            "base_reserve_units": 0.4,
            "effective_reserve_units": 0.8,
            "base_timeout_minutes": 60,
            "effective_timeout_minutes": 120,
            "estimate_status": "CONSERVATIVE_ENVELOPE_NOT_MEASURED_FIT_RATE",
            "d256_total_reserved_units_including_canary": 6,
            "remaining_series_uncommitted_units": 14,
            "canary_required": True,
            "automatic_completion_retries": False,
        },
    }
    canary_manifest = {
        **main_template,
        "job_id": f"f4-d256-canary-{commit[:8]}-{attempt}",
        "output_dir": str((output / "canary-output-l4-01").resolve()),
        "timeout_minutes": 20,
        "budget": {
            "unit_cap": 1,
            "rate_units_per_hour": 1.54,
            "reserve_units": 0.2,
            "other_assignments": {},
        },
        "requirements": {**main_template["requirements"], "est_minutes": 10},
        "env": {
            **main_template["env"],
            "F4_MODE": "canary",
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": "research/bl_residual",
        },
        "canary": {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": "research/bl_residual/f4_d256_colab.sh",
        },
        "f4": {
            **main_template["f4"],
            "base_units": 0.5,
            "effective_units": 1,
            "base_reserve_units": 0.1,
            "effective_reserve_units": 0.2,
            "base_timeout_minutes": 10,
            "effective_timeout_minutes": 20,
        },
    }
    return main_template, canary_manifest


def _prepare(
    output: Path,
    *,
    factory_ledger: Path,
    b8: Path,
    judgment: Path,
    commit: str,
    preflight_receipt: Path,
) -> dict[str, Any]:
    """Write a fresh D256-only package; no reserve rows are uploaded."""
    require_window_closed()
    require_publication(factory_ledger)
    if output.exists():
        raise FileExistsError(output)
    commit = subprocess.check_output(
        ["git", "rev-parse", f"{commit}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != commit:
        raise ValueError("prepare from the exact pinned source checkout")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("prepare from a clean pinned source checkout")
    verify_export(preflight_receipt)
    preflight = json.loads(preflight_receipt.read_text())
    if (
        preflight.get("schema") != "bl-residual-f4-d256-nofit-preflight/v1"
        or preflight.get("status") != "PASS"
        or preflight.get("commit") != commit
        or preflight.get("fits_run") is not False
        or preflight.get("study_outcomes_opened") is not False
        or set(preflight.get("fixtures", {}))
        != {
            "tests/test_bl_residual_factory_masks.py",
            "tests/test_bl_residual_f4_d256_package.py",
            "research/bl_labels/test_factory_executable.py",
        }
        or any(digest(ROOT / name) != sha for name, sha in preflight["fixtures"].items())
    ):
        raise ValueError("same-source no-fit oracle/package preflight required before labels")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True
    )
    from tools.ops.import_preflight import check as imports
    from tools.ops.qualification import check as qualification

    if qualification(ROOT, Path(QUALIFICATION_RECORD), commit) != QUALIFIED_TREE:
        raise ValueError("F4 transport qualification differs")
    import_receipt = imports(
        ROOT,
        commit,
        SOURCE_PATHS,
        REQUIREMENTS,
        ["research.bl_residual.job", "research.bl_residual.f4_canary"],
    )
    if digest(b8) != B8_SHA:
        raise ValueError("B8 identity differs")
    registration = ROOT / "research/bl_residual/F4_REGISTRATION.md"
    recipe_sha = canonical_sha(dataclasses.asdict(FitRecipe()))
    require_judgment(judgment, registration, recipe_sha)
    lives = _life_catalog(factory_ledger)
    catalog = []
    for life in lives:
        catalog.append(
            {
                "life_id": str(life["life_id"]),
                "parent_world": str(life["parent_world"]),
                "stream": str(life["stream"]),
                "rescue_bearing": bool(life["rescue_bearing"]),
                "allocation_index": int(life["allocation_index"]),
            }
        )
    allocation = allocate_lives(catalog)
    if not allocation["doses"]["256"]["ready"]:
        raise ValueError("INSUFFICIENT: D256 selected support unavailable")
    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    _copy(b8, inputs / "b8.pth")
    _copy(registration, inputs / "F4_REGISTRATION.md")
    _copy(judgment, inputs / "judgment.json")
    _copy(factory_ledger, inputs / "factory-verified-ledger.json")
    export_json(inputs / "life-allocation.json", allocation)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        **_table(
            lives,
            set(allocation["doses"]["256"]["train_lives"]),
            {allocation["parents"][life] for life in allocation["doses"]["256"]["train_lives"]},
        ),
    )
    export_bytes(inputs / "train-256.npz", buffer.getvalue())
    input_hashes = {
        path.name: digest(path)
        for path in sorted(inputs.iterdir())
        if not path.name.endswith((".sha256", ".complete"))
    }
    control = {
        "schema": "bl-residual-control/v1",
        "synthetic": False,
        "namespace": NAMESPACE,
        "doses": [256],
        "recipe_sha256": recipe_sha,
        "registration_sha256": F4_REGISTRATION_SHA,
        "factory_union_ledger_sha256": digest(factory_ledger),
        "input_hashes": input_hashes,
        "series_id": NAMESPACE,
        "reserve_allocation_sha256": canonical_sha(allocation["doses"]["256"]["heldout_lives"]),
    }
    control["allocation"] = allocation_hash(control)
    export_json(inputs / "verified-ledger.json", empty_ledger(NAMESPACE, control["allocation"]))
    control["ledger_sha256"] = digest(inputs / "verified-ledger.json")
    export_json(inputs / "control.json", control)
    raw_inventory = [
        {"path": str(path.resolve()), "dest": path.name, "sha256": digest(path)}
        for path in sorted(inputs.iterdir())
    ]
    from research.metadata_archive import bundle_inputs

    inventory = bundle_inputs(raw_inventory, inputs / "metadata.tar")
    main_template, canary_manifest = manifests(output, commit, inventory)
    export_json(output / "main-template.json", main_template)
    export_json(output / "canary.json", canary_manifest)
    export_json(output / "clean-import-preflight.json", import_receipt)
    export_json(output / "numerical-preflight.json", preflight)
    from research.bl_residual.job import preflight_main_dispatch
    from research.colab_cli.supervisor import validate

    export_json(output / "main-input-preflight.json", preflight_main_dispatch(inputs))
    validate(canary_manifest, ROOT)
    export_json(
        output / "pack.json",
        {
            "schema": "bl-residual-f4-d256-pack/v1",
            "commit": commit,
            "factory_union_ledger_sha256": digest(factory_ledger),
            "main_template_sha256": digest(output / "main-template.json"),
            "canary_manifest_sha256": digest(output / "canary.json"),
            "outcome_reporting": "none",
        },
    )
    return {
        "pack": str(output),
        "canary_manifest": str(output / "canary.json"),
        "main_template": str(output / "main-template.json"),
        "factory_union_ledger_sha256": digest(factory_ledger),
    }


def repack_existing(existing: Path, output: Path, commit: str) -> dict[str, Any]:
    """Create a fresh archive transport package from frozen F4 input bytes.

    This package-only operation copies registered payloads and sidecars, checks
    their frozen manifest identities, and never opens the NPZ, model, factory
    rows, or fit outputs.
    """
    if output.exists():
        raise FileExistsError(output)
    existing = existing.resolve()
    for name in ("pack.json", "canary.json", "main-template.json"):
        verify_export(existing / name)
    prior = json.loads((existing / "pack.json").read_text())
    if (
        prior.get("schema") != "bl-residual-f4-d256-pack/v1"
        or prior.get("commit") != FROZEN_REPACK_SOURCE
    ):
        raise ValueError("repack requires the frozen D256 package source")
    prior_canary = json.loads((existing / "canary.json").read_text())
    prior_main = json.loads((existing / "main-template.json").read_text())
    if prior_canary.get("inputs") != prior_main.get("inputs"):
        raise ValueError("frozen canary/main input inventories differ")
    old_inventory = prior_canary.get("inputs")
    if not isinstance(old_inventory, list) or not old_inventory:
        raise ValueError("frozen package has no input inventory")
    commit = subprocess.check_output(
        ["git", "rev-parse", f"{commit}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != commit:
        raise ValueError("repack from the exact pinned source checkout")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("repack from a clean pinned source checkout")
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", FROZEN_REPACK_SOURCE, commit], cwd=ROOT, text=True
        ).splitlines()
    )
    if not changed <= REPACK_ALLOWED_SOURCE_PATHS:
        raise ValueError("repack source changes exceed the package-only closure")
    from tools.ops.qualification import check as qualification

    if qualification(ROOT, Path(QUALIFICATION_RECORD), commit) != QUALIFIED_TREE:
        raise ValueError("F4 transport qualification differs")
    if (
        prior.get("main_template_sha256") != digest(existing / "main-template.json")
        or prior.get("canary_manifest_sha256") != digest(existing / "canary.json")
        or prior_canary.get("commit") != FROZEN_REPACK_SOURCE
        or prior_main.get("commit") != FROZEN_REPACK_SOURCE
    ):
        raise ValueError("frozen package manifest bindings differ")
    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    copied: list[dict[str, str]] = []
    for row in old_inventory:
        if not isinstance(row, dict) or set(row) != {"path", "dest", "sha256"}:
            raise ValueError("frozen package input row differs")
        source = Path(row["path"])
        if source.is_symlink() or not source.is_file() or digest(source) != row["sha256"]:
            raise ValueError("frozen package input bytes differ")
        target = inputs / row["dest"]
        if target.name != row["dest"]:
            raise ValueError("frozen package input destination is unsafe")
        _copy(source, target)
        if digest(target) != row["sha256"]:
            raise ValueError("repacked input copy differs")
        copied.append({"path": str(target.resolve()), "dest": row["dest"], "sha256": row["sha256"]})
    if {path.name for path in inputs.iterdir()} != {row["dest"] for row in copied}:
        raise ValueError("repacked input closure differs")
    from research.metadata_archive import bundle_inputs

    inventory = bundle_inputs(copied, inputs / "metadata.tar")
    main_template, canary_manifest = manifests(output, commit, inventory, attempt="metadata-01")
    for old, new in ((prior_main, main_template), (prior_canary, canary_manifest)):
        for key in (
            "device",
            "requirements",
            "timeout_minutes",
            "required_exports",
            "budget",
            "resume",
            "max_export_bytes",
            "f4",
        ):
            if old.get(key) != new.get(key):
                raise ValueError("repack changed frozen F4 science or caps")
        if old.get("script") != new.get("script") or old.get("source_paths") + [
            "research/metadata_archive.py"
        ] != new.get("source_paths"):
            raise ValueError("repack source closure differs")
        for key, value in old.get("env", {}).items():
            if new.get("env", {}).get(key) != value:
                raise ValueError("repack changed frozen F4 environment")
        if old.get("canary") != new.get("canary"):
            raise ValueError("repack changed frozen canary contract")
    export_json(output / "main-template.json", main_template)
    export_json(output / "canary.json", canary_manifest)
    from research.colab_cli.supervisor import validate

    validate(canary_manifest, ROOT)
    export_json(
        output / "pack.json",
        {
            "schema": "bl-residual-f4-d256-metadata-repack/v1",
            "commit": commit,
            "frozen_source_pack": str(existing),
            "frozen_source_commit": FROZEN_REPACK_SOURCE,
            "frozen_input_inventory_sha256": canonical_sha(old_inventory),
            "unpacked_input_inventory_sha256": canonical_sha(copied),
            "metadata_archive_sha256": _metadata_archive_sha256(inventory),
            "staged_file_count_before": len(copied),
            "staged_file_count_after": len(inventory),
            "outcome_reporting": "none",
        },
    )
    return {
        "pack": str(output),
        "canary_manifest": str(output / "canary.json"),
        "main_template": str(output / "main-template.json"),
        "staged_file_count_before": len(copied),
        "staged_file_count_after": len(inventory),
    }


def prepare(output: Path, **kwargs: Any) -> dict[str, Any]:
    """Hold the Mac resource boundary during the post-P1 mechanical conversion."""
    require_window_closed()
    from research.mac_lease import held

    with held(
        "f4-d256-input-preparation", "mechanical partition; no fits or reserve look", seconds=1200
    ):
        return _prepare(output, **kwargs)


def seal_main(template: Path, canary_receipt: Path, output: Path) -> dict[str, Any]:
    """Bind a passed same-source canary receipt into a fresh main manifest."""
    if output.exists():
        raise FileExistsError(output)
    value = json.loads(template.read_text())
    verify_export(template)
    if value.get("env", {}).get("F4_MODE") != "main" or "canary" in value:
        raise ValueError("only the unbound D256 main template may be sealed")
    value["canary"] = {
        "receipt": {"path": str(canary_receipt.resolve()), "sha256": digest(canary_receipt)}
    }
    from research.colab_cli.supervisor import validate

    validate(value, ROOT)
    export_json(output, value)
    return {"manifest": str(output), "sha256": digest(output)}


def publish_fit(pack: Path, output: Path) -> dict[str, Any]:
    """Publish path/size/hash only, before any fit metadata or reserve look."""
    from research.bl_residual.protocol import build_ledger

    control = json.loads((pack / "inputs/control.json").read_text())
    run_output = pack / "d256-output-l4-01"
    receipt = json.loads((run_output / "receipt.json").read_text())
    if (
        receipt.get("status") != "PASS"
        or receipt.get("stop_verified") is not True
        or receipt.get("error")
        or receipt.get("cleanup_error")
    ):
        raise ValueError("HOLD: D256 supervisor completion and release not verified")
    value = build_ledger(
        [run_output / "exports/fit"], output, control["namespace"], control["allocation"]
    )
    if [row["index"] for row in value["units"]] != [0]:
        raise ValueError("HOLD: D256 complete immutable fit unit required; never refit a fault")
    return {"path": str(output), "sha256": digest(output)}


def stage_d1024(pack: Path, publication: Path, output: Path) -> dict[str, Any]:
    """Prepare collection admission from the single frozen, fault-free D256 fit."""
    require_window_closed()
    from research.bl_labels.mac_gates import _future_readiness
    from research.bl_residual.job import _validate_staging, write_stage_fit_receipt

    if (
        not publication.as_posix().endswith("/" + FIT_PUBLICATION_PATH)
        and publication.as_posix() != FIT_PUBLICATION_PATH
    ):
        raise ValueError("use the registered fit-phase publication path")
    rel = FIT_PUBLICATION_PATH
    published = subprocess.check_output(["git", "show", f"origin/main:{rel}"], cwd=ROOT, timeout=30)
    if published != publication.read_bytes():
        raise ValueError("fit-phase ledger must match origin/main before metadata consumption")
    inputs = pack / "inputs"
    verify_export(inputs / "control.json")
    prior = json.loads((inputs / "control.json").read_text())
    from research.bl_residual.protocol import read_ledger

    units = read_ledger(publication, prior["namespace"], prior["allocation"])
    checkpoint = pack / "d256-output-l4-01/exports/fit/fit-0000.pth"
    if set(units) != {0} or verify_export(checkpoint) != units[0]["sha256"]:
        raise ValueError("published D256 fit differs from verified checkpoint")
    names = ("b8.pth", "F4_REGISTRATION.md", "judgment.json", "life-allocation.json")
    payloads = {}
    for name in names:
        source = inputs / name
        if source.is_symlink() or digest(source) != prior.get("input_hashes", {}).get(name):
            raise ValueError(f"D256 staging source differs from sealed control: {name}")
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != prior["input_hashes"][name]:
            raise ValueError(f"D256 staging source changed while reading: {name}")
        payloads[name] = payload
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        export_bytes(output / name, payload)
    export_json(output / "d256-control.json", prior)
    write_stage_fit_receipt(inputs, checkpoint.parent, 256, output / "d256-fit.json")
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
            name: digest(output / name)
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
    export_json(
        output / "verified-ledger.json", empty_ledger(control["namespace"], control["allocation"])
    )
    control["ledger_sha256"] = digest(output / "verified-ledger.json")
    export_json(output / "control.json", control)
    _validate_staging(output, control, [1024], control["recipe_sha256"], set())
    _future_readiness("d1024", output / "control.json", output)
    return {"inputs": str(output.resolve()), "control_sha256": digest(output / "control.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repack-existing", type=Path, metavar="PACK")
    parser.add_argument("--factory-ledger", type=Path)
    parser.add_argument("--b8", type=Path)
    parser.add_argument("--judgment", type=Path)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--publish-fit", type=Path, metavar="PACK")
    parser.add_argument("--stage-d1024", type=Path, metavar="PACK")
    parser.add_argument("--fit-publication", type=Path)
    parser.add_argument("--seal-main-template", type=Path)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--main-manifest-out", type=Path)
    args = parser.parse_args()
    if args.repack_existing:
        if args.out is None:
            parser.error("--repack-existing requires --out")
        print(json.dumps(repack_existing(args.repack_existing, args.out, args.commit)))
        return
    if args.publish_fit:
        if args.out is None:
            parser.error("--publish-fit requires --out")
        print(json.dumps(publish_fit(args.publish_fit, args.out)))
        return
    if args.stage_d1024:
        if args.out is None or args.fit_publication is None:
            parser.error("--stage-d1024 requires --out and --fit-publication")
        print(json.dumps(stage_d1024(args.stage_d1024, args.fit_publication, args.out)))
        return
    if args.seal_main_template:
        if not args.canary_receipt or not args.main_manifest_out:
            parser.error("--seal-main-template requires --canary-receipt and --main-manifest-out")
        print(
            json.dumps(
                seal_main(args.seal_main_template, args.canary_receipt, args.main_manifest_out)
            )
        )
        return
    if not all((args.out, args.factory_ledger, args.b8, args.judgment, args.preflight_receipt)):
        parser.error("--out, --factory-ledger, --b8, --judgment and --preflight-receipt required")
    print(
        json.dumps(
            prepare(
                args.out,
                factory_ledger=args.factory_ledger,
                b8=args.b8,
                judgment=args.judgment,
                commit=args.commit,
                preflight_receipt=args.preflight_receipt,
            )
        )
    )


if __name__ == "__main__":
    main()
