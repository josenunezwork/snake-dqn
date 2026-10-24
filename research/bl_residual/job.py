"""Real F3 entrypoint: synthetic plumbing now, reviewed factory fits later.

Only training partitions are uploaded. Held-out outcome files remain on the Mac
until immutable final fit units have been verified. Remote outputs are sealed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.bl_ppo.exports import digest, export_json, export_torch, limit_exports, verify_export
from research.bl_residual.collection_receipts import (
    START_JUDGMENT_SHA,
    l4_accounting,
    sha256,
    validate_mac_projection,
)
from research.bl_residual.data import allocate_lives, tensor_cache, validate_rows, world_fold
from research.bl_residual.fit import FitFault, FitRecipe, fit_residual
from research.bl_residual.model import FrozenB8Residual, fuse_residual
from research.bl_residual.protocol import (
    B8_SHA,
    DOSES,
    F4_REGISTRATION_SHA,
    QUALIFIED_TREE,
    allocation_hash,
    canonical_sha,
    read_ledger,
    require_judgment,
)
from src.model.ego2s_network import ARCH, load_ego2s_checkpoint, spec_for

COLLECTION_DISPOSITION_SCHEMA = "bl-labels-d4096-collection-disposition/v1"
DISCARD_SCHEMA = "d4096-not-run-by-rule/v1"
DISCARD_FILENAME = "d4096-not-run-by-rule.json"


def _require_d4096_collection_state(input_dir: Path, control: dict, requested: list[int]) -> None:
    """Give B6 precedence, including an unlisted or partially published seal.

    I1 owns the canonical seal writer. Presence is a refusal, even if the
    candidate is malformed: a damaged discard can never grant fit admission.
    Embedded Mac paths are deliberately not followed on workers.
    """
    if control.get("synthetic") is True or 4096 not in requested:
        return
    stage = control.get("stage_receipts", {})
    if isinstance(stage, dict) and "D4096_DISCARD" in stage:
        raise ValueError("NOT_RUN_BY_RULE: D4096 discard binding present; terminal discard wins")
    if any(
        (input_dir / (DISCARD_FILENAME + suffix)).exists()
        or (input_dir / (DISCARD_FILENAME + suffix)).is_symlink()
        for suffix in ("", ".sha256", ".complete")
    ):
        raise ValueError("NOT_RUN_BY_RULE: D4096 terminal discard seal present; refuse fit")
    name = stage.get("D4096_COLLECTION_DISPOSITION") if isinstance(stage, dict) else None
    if not name:
        raise ValueError("HOLD: D4096 requires a released collection disposition")
    value = _sealed_stage_receipt(input_dir, control, name)
    if value.get("disposition") == "DISCARDED":
        raise ValueError("NOT_RUN_BY_RULE: D4096 collection disposition is discarded")
    if value.get("disposition") != "RELEASED":
        raise ValueError("HOLD: D4096 collection disposition is not released")


def _require_collection_release(
    input_dir: Path, control: dict, stage: dict, recipe_sha: str, funding_name: str
) -> None:
    """Require I1's sealed release of this union under this exact PASS funding.

    This schema is a consumer contract until I1 publishes its producer. Missing,
    pending, discarded or foreign dispositions never authorize opening labels.
    """
    name = stage.get("D4096_COLLECTION_DISPOSITION")
    if not name:
        raise ValueError("HOLD: D4096 requires a released collection disposition")
    value = _sealed_stage_receipt(input_dir, control, name)
    disposition = value.get("disposition")
    if disposition == "DISCARDED":
        raise ValueError("NOT_RUN_BY_RULE: D4096 collection disposition is discarded")
    if disposition != "RELEASED":
        raise ValueError("HOLD: D4096 collection disposition is not released")
    if (
        value.get("schema") != COLLECTION_DISPOSITION_SCHEMA
        or value.get("judgment_sha256") != START_JUDGMENT_SHA
        or value.get("series_id") != control.get("series_id")
        or value.get("registration_sha256") != F4_REGISTRATION_SHA
        or value.get("reserve_allocation_sha256") != control.get("reserve_allocation_sha256")
        or value.get("recipe_sha256") != recipe_sha
        or not sha256(control.get("factory_union_ledger_sha256"))
        or value.get("collection_ledger_sha256") != control["factory_union_ledger_sha256"]
        or value.get("funding_receipt_sha256") != digest(input_dir / funding_name)
    ):
        raise ValueError("HOLD: foreign D4096 collection release binding")


def synthetic_table(base: Any) -> dict[str, np.ndarray]:
    """Synthetic labels are isolated from every factory allocation and decision."""
    rng = np.random.default_rng(3403)
    n = 12
    raw = {
        "local": rng.integers(0, 256, (n, 7, 31, 31), dtype=np.uint8),
        "global": rng.integers(0, 256, (n, 4, 37, 37), dtype=np.uint8),
        "scalars": rng.normal(size=(n, 60)).astype(np.float32),
    }
    with torch.no_grad():
        raw["q"] = base(*[torch.as_tensor(raw[k]) for k in ("local", "global", "scalars")]).numpy()
    raw.update(
        legal=np.ones((n, 6), bool),
        action_mask=np.ones((n, 6), bool),
        bare=raw["q"].argmax(axis=1),
        died=rng.random((n, 6)) < 0.5,
        status=np.full((n, 6), "EXACT"),
        mass=rng.random((n, 6)).astype(np.float32) * 10,
        ordinary=np.arange(n) % 2 == 0,
        sampling_prob=np.ones(n, np.float32),
        sampling_weight=np.ones(n, np.float32),
        natural_stream=np.arange(n) % 2 == 0,
        parent_world=np.asarray([f"synthetic-{i // 3}" for i in range(n)]),
        life_id=np.asarray([f"synthetic-life-{i // 3}" for i in range(n)]),
        row_id=np.asarray([f"synthetic-row-{i}" for i in range(n)]),
    )
    return raw


def _sealed_stage_receipt(input_dir: Path, control: dict, name: str) -> dict:
    """Read one hash-bound, create-only staging receipt before labels are opened."""
    if Path(name).name != name or control.get("input_hashes", {}).get(name) is None:
        raise ValueError("unbound staging receipt")
    path = input_dir / name
    if digest(path) != control["input_hashes"][name]:
        raise ValueError("staging receipt hash mismatch")
    verify_export(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("malformed staging receipt")
    return value


def _fit_stage(
    input_dir: Path, control: dict, name: str, prior_control_name: str, dose: int, recipe_sha: str
) -> str:
    """Bind a passed predecessor fit to its own immutable control and allocation."""
    value = _sealed_stage_receipt(input_dir, control, name)
    prior = _sealed_stage_receipt(input_dir, control, prior_control_name)
    sha = value.get("fit_sha256")
    if (
        value.get("schema") != "bl-residual-stage-fit/v1"
        or value.get("dose") != dose
        or value.get("namespace") != control["namespace"]
        or prior.get("schema") != "bl-residual-control/v1"
        or prior.get("synthetic") is not False
        or allocation_hash(prior) != prior.get("allocation")
        or prior.get("namespace") != control["namespace"]
        or prior.get("recipe_sha256") != recipe_sha
        or prior.get("input_hashes", {}).get("b8.pth") != B8_SHA
        or value.get("allocation") != prior["allocation"]
        or value.get("recipe_sha256") != recipe_sha
        or value.get("source_control_sha256") != digest(input_dir / prior_control_name)
        or value.get("complete") is not True
        or value.get("fit_fault_free") is not True
        or not isinstance(sha, str)
        or len(sha) != 64
        or any(ch not in "0123456789abcdef" for ch in sha)
    ):
        raise ValueError("unpassed or foreign predecessor fit receipt")
    return sha


def _validate_staging(
    input_dir: Path, control: dict, requested: list[int], recipe_sha: str, skipped: set[int]
) -> None:
    """Enforce R6 stops before training partitions or output directories are touched."""
    if control.get("synthetic") is True:
        return
    _require_d4096_collection_state(input_dir, control, requested)
    stage = control.get("stage_receipts", {})
    if not isinstance(stage, dict) or any(not isinstance(v, str) for v in stage.values()):
        raise ValueError("bound stage_receipts mapping required")
    d256, d256_control = stage.get("D256"), stage.get("D256_CONTROL")
    d1024, d1024_control = stage.get("D1024"), stage.get("D1024_CONTROL")
    # A prior passed D256 is needed only when D1024 is a later invocation.  A single
    # D256/D1024 invocation remains allowed; an in-process unconditional fault raises
    # immediately and therefore cannot reach the D1024 partition.
    if 1024 in requested and (256 not in requested or DOSES.index(256) in skipped):
        if not d256 or not d256_control:
            raise ValueError("STOP: D1024 requires passed D256 stage receipt")
        _fit_stage(input_dir, control, d256, d256_control, 256, recipe_sha)
    if 4096 not in requested:
        return
    if not d256 or not d256_control or not d1024 or not d1024_control:
        raise ValueError("STOP: D4096 requires passed D256 and D1024 stage receipts")
    d256_sha = _fit_stage(input_dir, control, d256, d256_control, 256, recipe_sha)
    d1024_sha = _fit_stage(input_dir, control, d1024, d1024_control, 1024, recipe_sha)
    funding_name = stage.get("D4096_FUNDING")
    projection_name = stage.get("D4096_PROJECTION")
    if not funding_name or not projection_name:
        raise ValueError("STOP: D4096 requires funding and projection receipts")
    funding = _sealed_stage_receipt(input_dir, control, funding_name)
    series_id = control.get("series_id")
    reserve_sha = control.get("reserve_allocation_sha256")
    if not isinstance(series_id, str) or not series_id or not isinstance(reserve_sha, str):
        raise ValueError("D4096 requires a frozen series and reserve allocation identity")
    funding_ok = (
        funding.get("series_id") == series_id
        and funding.get("registration_sha256") == F4_REGISTRATION_SHA
        and funding.get("reserve_allocation_sha256") == reserve_sha
        and funding.get("recipe_sha256") == recipe_sha
        and funding.get("d256_fit_sha256") == d256_sha
        and funding.get("d1024_fit_sha256") == d1024_sha
        and funding.get("schema") == "bl-residual-funding-look/v1"
        and funding.get("verdict") == "PASS"
    )
    if not funding_ok:
        raise ValueError("STOP_FUNDING: D4096 funding receipt did not pass")
    _require_collection_release(input_dir, control, stage, recipe_sha, funding_name)
    projection = _sealed_stage_receipt(input_dir, control, projection_name)
    projected = projection.get("projected_units")
    collection_names = (stage.get("D256_COLLECTION"), stage.get("D1024_COLLECTION"))
    if any(not name for name in collection_names):
        raise ValueError("INSUFFICIENT: D4096 needs bound collection receipts")
    collections = [_sealed_stage_receipt(input_dir, control, name) for name in collection_names]
    measurements = projection.get("measurements")
    measured_units = 0.0
    selected_lives = 0
    measurements_valid = isinstance(measurements, list) and len(measurements) == 2
    if measurements_valid:
        for measurement, expected_dose, expected_sha in zip(
            measurements, (256, 1024), (d256_sha, d1024_sha)
        ):
            if not isinstance(measurement, dict):
                measurements_valid = False
                break
            units, lives = measurement.get("charged_units"), measurement.get(
                "selected_rescue_lives"
            )
            if (
                measurement.get("dose") != expected_dose
                or measurement.get("fit_sha256") != expected_sha
                or isinstance(units, bool)
                or isinstance(lives, bool)
                or not isinstance(units, (int, float))
                or type(lives) is not int
                or not np.isfinite(units)
                or units < 0
                or lives <= 0
            ):
                measurements_valid = False
                break
            measured_units += float(units)
            selected_lives += lives
    checked_measurements = measurements if isinstance(measurements, list) else []
    for collection, measurement, expected_dose, expected_sha in zip(
        collections, checked_measurements, (256, 1024), (d256_sha, d1024_sha)
    ):
        if (
            not isinstance(measurement, dict)
            or collection.get("schema") != "bl-residual-collection-receipt/v1"
            or collection.get("series_id") != series_id
            or collection.get("registration_sha256") != F4_REGISTRATION_SHA
            or collection.get("reserve_allocation_sha256") != reserve_sha
            or collection.get("dose") != expected_dose
            or collection.get("fit_sha256") != expected_sha
            or collection.get("qualified_tree") != QUALIFIED_TREE
            or type(collection.get("charged_units")) not in (int, float)
            or not np.isfinite(collection["charged_units"])
            or collection["charged_units"] < 0
            or collection.get("charged_units") != measurement.get("charged_units")
            or collection.get("selected_rescue_lives") != measurement.get("selected_rescue_lives")
        ):
            measurements_valid = False
    expected_projection = (
        measured_units / selected_lives * 3072 * 1.2 if selected_lives else float("inf")
    )
    zero_mac_ok = False
    if measurements_valid and measured_units == 0 and type(projected) in (int, float):
        mac_name = stage.get("D4096_MAC_PROJECTION")
        if mac_name and all(
            c.get("placement") == "mac"
            and type(c.get("charged_units")) in (int, float)
            and c["charged_units"] == 0
            for c in collections
        ):
            mac_projection = _sealed_stage_receipt(input_dir, control, mac_name)
            validate_mac_projection(mac_projection, collections)
            provenance_ok = True
            for collection, prior_name in zip(collections, (d256_control, d1024_control)):
                prior = _sealed_stage_receipt(input_dir, control, prior_name)
                if collection.get("collection_ledger_sha256") != prior.get(
                    "factory_union_ledger_sha256"
                ) or collection.get("source_control_sha256") != digest(input_dir / prior_name):
                    provenance_ok = False
            mac_sha = digest(input_dir / mac_name)
            zero_mac_ok = (
                projected == 0
                and provenance_ok
                and projection.get("mac_projection_receipt_sha256") == mac_sha
                and isinstance(projection.get("qualified_placement_receipts"), list)
                and mac_sha in projection.get("qualified_placement_receipts", [])
                and projection.get("collection_receipt_sha256")
                == [digest(input_dir / name) for name in collection_names]
                and selected_lives == 1024
                and [m.get("selected_rescue_lives") for m in checked_measurements] == [256, 768]
                and isinstance(projection.get("l4_tail_accounting"), dict)
            )
    tail = projection.get("l4_tail_accounting")
    if (
        tail is not None
        or projection.get("l4_tail_accounting_sha256") is not None
        or stage.get("D4096_L4_ACCOUNTING")
    ):
        # A separate sealed R7 ledger charges all L4 attempts. It never changes
        # the measured Mac projection, and the worker repeats its cap check.
        tail_name = stage.get("D4096_L4_ACCOUNTING")
        if tail_name:
            tail_value = _sealed_stage_receipt(input_dir, control, tail_name)
            if projection.get("l4_tail_accounting_sha256") != digest(input_dir / tail_name):
                raise ValueError("unbound D4096 L4 accounting")
        else:
            tail_value = None
            if projection.get("l4_tail_accounting_sha256") is not None:
                raise ValueError("missing D4096 L4 accounting binding")
        if tail != l4_accounting(tail_value, series_id):
            raise ValueError("D4096 L4 R7 accounting differs")
    projection_ok = (
        projection.get("schema") == "bl-residual-d4096-projection/v1"
        and projection.get("series_id") == series_id
        and projection.get("registration_sha256") == F4_REGISTRATION_SHA
        and projection.get("reserve_allocation_sha256") == reserve_sha
        and projection.get("recipe_sha256") == recipe_sha
        and projection.get("d256_fit_sha256") == d256_sha
        and projection.get("d1024_fit_sha256") == d1024_sha
        and projection.get("qualified_placement_tree") == QUALIFIED_TREE
        and isinstance(projection.get("qualified_placement_receipts"), list)
        and projection["qualified_placement_receipts"]
        and isinstance(projected, (int, float))
        and not isinstance(projected, bool)
        and np.isfinite(projected)
        and ((measured_units > 0 and float(projected) > 0) or zero_mac_ok)
        and measurements_valid
        and projected <= 480
        and np.isclose(float(projected), expected_projection, rtol=0.0, atol=1e-9)
        and projection.get("verdict") == "PASS"
    )
    if not projection_ok:
        raise ValueError("INSUFFICIENT: D4096 projection receipt did not pass")


def write_stage_fit_receipt(input_dir: Path, run_output: Path, dose: int, output: Path) -> dict:
    """Seal one passed main-fit identity for a later, separately frozen stage.

    This reads only control, fit metadata and run accounting.  It never opens a
    held-out table or any gameplay artifact.
    """
    if dose not in DOSES or output.exists():
        raise ValueError("fresh valid dose receipt required")
    control_path = input_dir / "control.json"
    control = json.loads(control_path.read_text())
    recipe_sha = canonical_sha(dataclasses.asdict(FitRecipe()))
    if (
        control.get("synthetic") is not False
        or allocation_hash(control) != control.get("allocation")
        or control.get("recipe_sha256") != recipe_sha
    ):
        raise ValueError("foreign or unregistered source control")
    require_judgment(input_dir / "judgment.json", input_dir / "F4_REGISTRATION.md", recipe_sha)
    index = DOSES.index(dose)
    checkpoint = run_output / f"fit-{index:04d}.pth"
    receipt_path = run_output / "receipt.json"
    fit_sha = verify_export(checkpoint)
    verify_export(receipt_path)
    values = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = values.get("meta", {})
    receipt = json.loads(receipt_path.read_text())
    completed = receipt.get("completed", [])
    if (
        meta.get("schema") != "bl-residual-fit/v1"
        or meta.get("synthetic") is not False
        or meta.get("dose") != dose
        or meta.get("index") != index
        or meta.get("namespace") != control.get("namespace")
        or meta.get("allocation") != control.get("allocation")
        or meta.get("recipe_sha256") != recipe_sha
        or meta.get("updates") != FitRecipe().updates
        or receipt.get("schema") != "bl-residual-job/v1"
        or receipt.get("mode") != "main"
        or receipt.get("synthetic") is not False
        or receipt.get("allocation") != control.get("allocation")
        or receipt.get("heldout_outcomes_opened") is not False
        or not any(row.get("index") == index and row.get("sha256") == fit_sha for row in completed)
    ):
        raise ValueError("unpassed or foreign main fit")
    value = {
        "schema": "bl-residual-stage-fit/v1",
        "dose": dose,
        "namespace": control["namespace"],
        "allocation": control["allocation"],
        "recipe_sha256": recipe_sha,
        "source_control_sha256": digest(control_path),
        "fit_sha256": fit_sha,
        "complete": True,
        # The fit worker observes numerical tripwires. Gameplay action/collapse
        # telemetry is a separate, later trace artifact and is not claimed here.
        "fit_fault_free": True,
        "trajectory_unconditional_status": "NOT_OBSERVED",
    }
    export_json(output, value)
    return value


def preflight_main_dispatch(input_dir: Path) -> dict:
    """Refuse an inadmissible main dispatch without opening any train partition.

    This is the pack/dispatch admission boundary.  It reads only bound controls,
    receipts and allocation metadata; workers repeat the checks in ``run``.
    """
    control = json.loads((input_dir / "control.json").read_text())
    if control.get("schema") != "bl-residual-control/v1" or control.get("synthetic") is not False:
        raise ValueError("main preflight requires a non-synthetic F4 control")
    _require_d4096_collection_state(input_dir, control, control.get("doses", []))
    for name, sha in control.get("input_hashes", {}).items():
        # Partition bytes are deliberately not opened at admission.  The runner
        # checks their hashes only after this guard has permitted dispatch.
        if name.startswith("train-") and name.endswith(".npz"):
            continue
        if Path(name).name != name or digest(input_dir / name) != sha:
            raise ValueError("input hash mismatch")
    if digest(input_dir / "b8.pth") != B8_SHA:
        raise ValueError("B8 identity mismatch")
    recipe_sha = canonical_sha(dataclasses.asdict(FitRecipe()))
    if control.get("recipe_sha256") != recipe_sha:
        raise ValueError("recipe binding mismatch")
    require_judgment(input_dir / "judgment.json", input_dir / "F4_REGISTRATION.md", recipe_sha)
    requested = control.get("doses")
    if (
        not isinstance(requested, list)
        or not requested
        or len(set(requested)) != len(requested)
        or any(type(dose) is not int or dose not in DOSES for dose in requested)
        or allocation_hash(control) != control.get("allocation")
    ):
        raise ValueError("frozen main allocation required")
    skip = read_ledger(
        input_dir / "verified-ledger.json", control["namespace"], control["allocation"]
    )
    if not set(skip) <= {DOSES.index(dose) for dose in requested}:
        raise ValueError("ledger outside requested dose allocation")
    if digest(input_dir / "verified-ledger.json") != control.get("ledger_sha256"):
        raise ValueError("ledger binding mismatch")
    allocation_path = input_dir / "life-allocation.json"
    if control.get("input_hashes", {}).get("life-allocation.json") != digest(allocation_path):
        raise ValueError("bound life allocation required")
    life_allocation = json.loads(allocation_path.read_text())
    if allocate_lives(life_allocation["catalog"]) != life_allocation:
        raise ValueError("life allocation differs from frozen allocator")
    if any(not life_allocation["doses"][str(dose)]["ready"] for dose in requested):
        raise ValueError("INSUFFICIENT: requested dose not ready")
    _validate_staging(input_dir, control, requested, recipe_sha, set(skip))
    return {
        "schema": "bl-residual-main-preflight/v1",
        "namespace": control["namespace"],
        "allocation": control["allocation"],
        "doses": requested,
        "recipe_sha256": recipe_sha,
    }


def run(input_dir: Path, output: Path, *, mode: str, device: str = "cpu") -> dict:
    """Fit complete deterministic units, skipping exactly Mac-verified units."""
    if mode not in ("smoke", "canary", "main"):
        raise ValueError("unrecognized mode")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    control = json.loads((input_dir / "control.json").read_text())
    if control.get("schema") != "bl-residual-control/v1":
        raise ValueError("foreign control")
    _require_d4096_collection_state(input_dir, control, control.get("doses", []))
    for name, sha in control["input_hashes"].items():
        if Path(name).name != name or digest(input_dir / name) != sha:
            raise ValueError("input hash mismatch")
    if digest(input_dir / "b8.pth") != B8_SHA:
        raise ValueError("B8 identity mismatch")
    recipe = FitRecipe()
    recipe_sha = canonical_sha(dataclasses.asdict(recipe))
    if control.get("recipe_sha256") != recipe_sha:
        raise ValueError("recipe binding mismatch")
    synthetic = mode != "main"
    if synthetic != bool(control.get("synthetic")):
        raise ValueError("synthetic/main namespace mismatch")
    if not synthetic:
        require_judgment(input_dir / "judgment.json", input_dir / "F4_REGISTRATION.md", recipe_sha)
    namespace = control["namespace"]
    requested = control.get("doses")
    if (
        not isinstance(requested, list)
        or not requested
        or len(set(requested)) != len(requested)
        or any(type(d) is not int or d not in DOSES for d in requested)
    ):
        raise ValueError("frozen requested doses required")
    requested_indices = {DOSES.index(d) for d in requested}
    allocation = allocation_hash(control)
    if control.get("allocation") != allocation:
        raise ValueError("allocation mismatch")
    skip = read_ledger(input_dir / "verified-ledger.json", namespace, allocation)
    if not set(skip) <= requested_indices:
        raise ValueError("ledger outside requested dose allocation")
    # Ledger is bound separately to avoid a self-referential allocation hash.
    if digest(input_dir / "verified-ledger.json") != control["ledger_sha256"]:
        raise ValueError("ledger binding mismatch")
    life_allocation = None
    if not synthetic:
        if "life-allocation.json" not in control["input_hashes"]:
            raise ValueError("bound life allocation required")
        life_allocation = json.loads((input_dir / "life-allocation.json").read_text())
        if allocate_lives(life_allocation["catalog"]) != life_allocation:
            raise ValueError("life allocation differs from frozen allocator")
        for dose in requested:
            if not life_allocation["doses"][str(dose)]["ready"]:
                raise ValueError("INSUFFICIENT: requested dose not ready")
        _validate_staging(input_dir, control, requested, recipe_sha, set(skip))
    if output.exists():
        raise FileExistsError("fresh output required")
    output.mkdir(parents=True)
    export_json(output / "allocation.json", {"namespace": namespace, "allocation": allocation})
    started = time.monotonic()
    stop = threading.Event()
    counts = {"done": len(skip)}

    def pulse() -> None:
        while not stop.wait(110):
            print(
                json.dumps(
                    {
                        "done": counts["done"],
                        "total": len(requested),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )

    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    completed = []
    try:
        with limit_exports(output, total_bytes=128 * 1024**2, file_bytes=32 * 1024**2):
            for index, dose in enumerate(DOSES):
                if dose not in requested or index in skip:
                    continue
                base = load_ego2s_checkpoint(str(input_dir / "b8.pth"))
                if synthetic:
                    table = synthetic_table(base)
                    actual_recipe = dataclasses.replace(
                        recipe, updates=3, batch_size=4, doom_batch_size=2, ordinary_batch_size=2
                    )
                else:
                    filename = f"train-{dose}.npz"
                    if filename not in control["input_hashes"]:
                        raise ValueError("training partition must be a bound input")
                    with np.load(input_dir / filename, allow_pickle=False) as archive:
                        table = {k: archive[k] for k in archive.files}
                    validate_rows(table)
                    if any(world_fold(str(p)) == 0 for p in table["parent_world"]):
                        raise ValueError("held-out parent in training partition")
                    expected_lives = set(life_allocation["doses"][str(dose)]["train_lives"])
                    catalog = {r["life_id"]: r for r in life_allocation["catalog"]}
                    actual = set(table["life_id"].tolist())
                    rescue_ids = {
                        life for life in actual if catalog.get(life, {}).get("rescue_bearing")
                    }
                    if rescue_ids != expected_lives:
                        raise ValueError("training life allocation mismatch")
                    parents = {catalog[life]["parent_world"] for life in expected_lives}
                    for life, parent, ordinary in zip(
                        table["life_id"], table["parent_world"], table["ordinary"]
                    ):
                        if (
                            life not in catalog
                            or catalog[life]["parent_world"] != parent
                            or parent not in parents
                            or (life not in expected_lives and not ordinary)
                        ):
                            raise ValueError(
                                "foreign parent or non-rescue row outside ordinary archive"
                            )
                    if len(expected_lives) != dose - 64:
                        raise ValueError("training dose would shrink N")
                    actual_recipe = recipe
                cache = tensor_cache(table, base)
                model = FrozenB8Residual(base).to(device)
                try:
                    state = fit_residual(model, cache, actual_recipe)
                except FitFault as exc:
                    # A scientific tripwire is terminal for this dose and the
                    # staging path.  Seal it before returning a non-zero run.
                    export_json(
                        output / f"fault-{index:04d}.json",
                        {
                            "schema": "bl-residual-fit-fault/v1",
                            "kind": "KILL_FIT",
                            "dose": dose,
                            "index": index,
                            "namespace": namespace,
                            "allocation": allocation,
                            "recipe_sha256": recipe_sha,
                            "reason": str(exc),
                            "heldout_outcomes_opened": False,
                        },
                    )
                    raise
                fused = fuse_residual(model)
                values = {
                    "obs_spec": spec_for(fused),
                    "arch": ARCH,
                    "config": fused.config,
                    "state_dict": fused.state_dict(),
                    "meta": {
                        "schema": "bl-residual-fit/v1",
                        "synthetic": synthetic,
                        "dose": dose,
                        "namespace": namespace,
                        "allocation": allocation,
                        "index": index,
                        "recipe": dataclasses.asdict(actual_recipe),
                        "recipe_sha256": recipe_sha,
                        "updates": state.updates_completed,
                    },
                    "fit_state": {
                        "residual": model.residual.state_dict(),
                        "optimizer": state.optimizer_state,
                        "rng": state.rng_state,
                        "updates": state.updates_completed,
                    },
                }
                result = export_torch(output / f"fit-{index:04d}.pth", values)
                completed.append({"index": index, "dose": dose, **result})
                counts["done"] += 1
            receipt = {
                "schema": "bl-residual-job/v1",
                "mode": mode,
                "synthetic": synthetic,
                "namespace": namespace,
                "allocation": allocation,
                "completed": completed,
                "skipped": sorted(skip),
                "expected_units": len(requested),
                "done": counts["done"],
                "elapsed_seconds": time.monotonic() - started,
                "scientific_result": False,
                "heldout_outcomes_opened": False,
            }
            export_json(output / "receipt.json", receipt)
            return receipt
    finally:
        stop.set()
        thread.join(timeout=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("smoke", "canary", "main"))
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cpu")
    parser.add_argument("--preflight-main", action="store_true")
    args = parser.parse_args()
    if args.preflight_main:
        if args.output is not None or args.mode is not None:
            parser.error("--preflight-main accepts only --input-dir")
        print(json.dumps(preflight_main_dispatch(args.input_dir)))
        return
    if args.output is None or args.mode is None:
        parser.error("--output and --mode are required unless --preflight-main is used")
    result = run(args.input_dir, args.output, mode=args.mode, device=args.device)
    print(json.dumps({"done": result["done"], "synthetic": result["synthetic"]}))


if __name__ == "__main__":
    main()
