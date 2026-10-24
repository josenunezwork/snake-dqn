"""Prepare I0 all-roster A100 packets; this module never dispatches a runtime."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from pathlib import Path
from typing import Any

from research.bl_upload import assert_upload_budget, convert_public_checkpoints
from research.gpu_sim_spike import a1_prof_pack, integration_units
from research.gpu_sim_spike.make_manifests import (
    ManifestRefused,
    checkpoint_inputs,
    git_head,
    sha256_file,
    write_once,
)
from research.metadata_archive import bundle_inputs

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/gpu_sim_spike"
SCRIPT = f"{PACKAGE}/integration_bench.sh"
TRANSPORT_PIN = "313758255ee5b9f1422a0202c14a80cc32554857"
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever")
RULINGS = (
    (
        ARTIFACTS / "round-36/JUDGMENT.md",
        "156c80b412064defd88281424e4c84e6a732d9cb8bc6e163561ef352ffb5fe3b",
    ),
    (
        ARTIFACTS / "exp/JUDGMENT_GPU_PARITY_STANDARD_20261023.md",
        "51b637243a57db2a64baa0d80174386994e3e9f71d545589c31628ad58b4948b",
    ),
    (
        ARTIFACTS / "exp/JUDGMENT_V8_ROUTE_20261023.md",
        "5358934afc504cbee94a74622dea0534c831de5e8d7d8f471d5ad4177871a108",
    ),
    (
        ARTIFACTS / "exp/JUDGMENT_V8_SPLIT_MAIN_20261024.md",
        "f8c1386dc6b82c3cd0f0b74a491b18433d78da48cd3fb28472f16c3a68db22b4",
    ),
)
PROTOCOL = ROOT / PACKAGE / "INTEGRATION_INTENT_20261024.md"
ADMISSION = ROOT / PACKAGE / "INTEGRATION_ADMISSION_20261024.json"
LEDGER_DEST = "inputs/verified-ledger.json"
CANARY_LEDGER_DEST = "canary-inputs/verified-ledger.json"


def _require_commit(commit: str) -> None:
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ManifestRefused("commit must be a full lowercase SHA")


def require_source_pin(commit: str) -> None:
    """Refuse unpublished or dirty sources before an immutable packet is made."""
    _require_commit(commit)
    if git_head() != commit:
        raise ManifestRefused("packet source is not the requested commit")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ManifestRefused("I0 source checkout must be clean")
    if "origin/" not in subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True
    ):
        raise ManifestRefused("I0 source pin must be published")
    for path, digest in RULINGS:
        if not path.is_file() or sha256_file(path) != digest:
            raise ManifestRefused("binding Astra ruling hash differs")
    if not PROTOCOL.is_file() or not ADMISSION.is_file():
        raise ManifestRefused("I0 protocol or component admission record is absent")
    from tools.ops.qualification import OPS, check

    check(ROOT, OPS / "colab-cli-qualified.json")
    check(ROOT, OPS / "colab-cli-qualified.json", ref=TRANSPORT_PIN)


def _verify(path: Path) -> str:
    return integration_units._verify(path)


def _ledger_inputs(ledger: Path, prefix: str) -> list[dict[str, str]]:
    body = json.loads(ledger.read_text())
    paths = [ledger]
    for row in body.get("units", []):
        paths.append(ledger.parent / row["path"])
    result = []
    for item in paths:
        for candidate in (item, Path(str(item) + ".sha256"), Path(str(item) + ".complete")):
            result.append(
                {
                    "path": str(candidate.resolve(strict=True)),
                    "sha256": sha256_file(candidate),
                    "dest": prefix + "/" + candidate.relative_to(ledger.parent).as_posix(),
                }
            )
    return result


def _bindings() -> list[dict[str, str]]:
    rows = []
    for path, digest in RULINGS:
        rows.append({"path": str(path), "sha256": digest, "dest": "inputs/rulings/" + path.name})
    for path, destination in (
        (PROTOCOL, "INTEGRATION_INTENT_20261024.md"),
        (ADMISSION, "INTEGRATION_ADMISSION_20261024.json"),
    ):
        rows.append(
            {"path": str(path), "sha256": sha256_file(path), "dest": "inputs/" + destination}
        )
    return rows


def preflight_binding(path: Path, commit: str) -> dict[str, Any]:
    _verify(path)
    body = json.loads(path.read_text())
    receipt = Path(str(body.get("receipt_path", "")))
    if not receipt.is_file() or _verify(receipt) != body.get("receipt_sha256"):
        raise ManifestRefused("I0 preflight receipt binding differs")
    expected = list(integration_units.allocation(smoke=True))
    if (
        body.get("status"),
        body.get("commit"),
        body.get("device"),
        body.get("completed_units"),
    ) != ("PREFLIGHT", commit, "cpu", expected):
        raise ManifestRefused("I0 needs current-pin CPU real-entrypoint preflight")
    from research.gpu_sim_spike.benchmark import source_hashes

    receipt_body = json.loads(receipt.read_text())
    if (
        not isinstance(body.get("source_sha256"), dict)
        or body.get("entrypoint_run_wall_seconds", 0) <= 0
        or receipt_body.get("commit") != commit
        or receipt_body.get("status") != "MEASURED_PENDING_MAC_REVIEW"
        or receipt_body.get("completed_units") != expected
        or receipt_body.get("source_sha256") != body["source_sha256"]
        or receipt_body.get("source_sha256") != source_hashes()
    ):
        raise ManifestRefused("I0 preflight is incomplete")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "receipt_path": str(receipt),
        "receipt_sha256": body["receipt_sha256"],
        "source_sha256": body["source_sha256"],
        "entrypoint_run_wall_seconds": body["entrypoint_run_wall_seconds"],
    }


def _admission(commit: str) -> dict[str, Any]:
    body = json.loads(ADMISSION.read_text())
    # This file is part of the source pin it admits, so it cannot contain a
    # self-referential final commit hash.  Its own immutable upload hash binds
    # the component evidence; the current pin is bound by preflight/manifest.
    if (
        body.get("schema") != "gpu-sim-spike-i0-admission/v1"
        or body.get("move_admitted") is not True
    ):
        raise ManifestRefused("I0 admission lacks admitted MOVE evidence")
    receipt_ref, manifest_ref = body.get("move_receipt"), body.get("move_manifest")
    inventory = body.get("move_export_inventory")
    if (
        not isinstance(receipt_ref, dict)
        or not isinstance(manifest_ref, dict)
        or not isinstance(inventory, list)
    ):
        raise ManifestRefused("I0 admission lacks hash-bound MOVE receipt, manifest, or inventory")
    receipt_path, manifest_path = Path(str(receipt_ref.get("path", ""))), Path(
        str(manifest_ref.get("path", ""))
    )
    if (
        not receipt_path.is_file()
        or not manifest_path.is_file()
        or sha256_file(receipt_path) != receipt_ref.get("sha256")
        or sha256_file(manifest_path) != manifest_ref.get("sha256")
    ):
        raise ManifestRefused("I0 admitted MOVE receipt or manifest bytes differ")
    receipt, manifest = json.loads(receipt_path.read_text()), json.loads(manifest_path.read_text())
    if (
        receipt.get("status"),
        receipt.get("commit"),
        receipt.get("stop_verified"),
        receipt.get("manifest_sha256"),
    ) != ("PASS", body.get("move_pin"), True, manifest_ref.get("sha256")):
        raise ManifestRefused("I0 admitted MOVE receipt is not a released benchmark result")
    if manifest.get("commit") != body.get("move_pin"):
        raise ManifestRefused("I0 admitted MOVE manifest pin differs")
    saved = receipt.get("exports")
    if not isinstance(saved, list) or len(saved) != len(inventory) or len(inventory) != 14:
        raise ManifestRefused("I0 admitted MOVE export inventory is incomplete")
    expected = {row.get("path"): row for row in inventory if isinstance(row, dict)}
    observed = {row.get("path"): row for row in saved if isinstance(row, dict)}
    if len(expected) != len(inventory) or expected != observed:
        raise ManifestRefused("I0 admitted MOVE receipt inventory differs")
    exports = receipt_path.parent / "exports"
    for relative, row in expected.items():
        path = exports / str(relative)
        if (
            not isinstance(relative, str)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row.get("size")
            or sha256_file(path) != row.get("sha256")
        ):
            raise ManifestRefused("I0 admitted MOVE export bytes differ")
    if (
        body.get("move_pin") != "fe404912436fe2f3f01db78f4a1778fa176ac056"
        or body.get("geometry_pin") != "85180521e87ffc3ddc79296bfb09dcd2c89fdc62"
        or body.get("v8_split_pin") != "669405e2bef629f7f69bc951cf49486d8b0a87bf"
        or body.get("indexed_v8_enabled") is not False
    ):
        raise ManifestRefused("I0 admitted component pins or indexed-v8 flag differ")
    source_inventory = body.get("component_source_sha256")
    if not isinstance(source_inventory, dict) or len(source_inventory) != 14:
        raise ManifestRefused("I0 frozen component source inventory is incomplete")
    for relative, digest in source_inventory.items():
        path = ROOT / PACKAGE / relative
        if Path(relative).name != relative or not path.is_file() or sha256_file(path) != digest:
            raise ManifestRefused("I0 admitted component source bytes differ")
    if body.get("flood_variant") != "static":
        raise ManifestRefused("I0 must retain static flood until sparse proof is admitted")
    if body.get("v8_selection") != "PENDING":
        raise ManifestRefused(
            "tracked I0 admission must retain the immutable pending V8 dependency"
        )
    return body


def _v8_selection_binding(path: Path) -> dict[str, str]:
    """Bind a completed V8 decision locally without mutating canary inputs."""
    _verify(path)
    body = json.loads(path.read_text())
    v8_commit = "669405e2bef629f7f69bc951cf49486d8b0a87bf"
    if body.get("commit") != v8_commit or body.get("verdict") not in {"SELECT_G", "SELECT_GV"}:
        raise ManifestRefused("I0 main requires V8 SELECT_G or SELECT_GV, never STOP_SPLIT")
    # The supplied result must live beside immutable producer exports. Run the
    # independent reducer again rather than trusting a copied verdict field.
    exports = path.parent
    receipt = exports.parent / "receipt.json"
    if not receipt.is_file():
        raise ManifestRefused("V8 selection needs its saved operational receipt beside exports")
    receipt_body = json.loads(receipt.read_text())
    saved_manifest = exports.parent / "manifest.json"
    if (
        receipt_body.get("status") != "PASS"
        or receipt_body.get("commit") != v8_commit
        or receipt_body.get("stop_verified") is not True
        or not saved_manifest.is_file()
        or sha256_file(saved_manifest) != receipt_body.get("manifest_sha256")
        or json.loads(saved_manifest.read_text()).get("commit") != v8_commit
    ):
        raise ManifestRefused("V8 selection operational receipt is not a released current-pin run")
    listed = receipt_body.get("exports")
    if not isinstance(listed, list):
        raise ManifestRefused("V8 selection operational receipt lacks export inventory")
    expected = {row.get("path"): row for row in listed if isinstance(row, dict)}
    selection_relative = path.relative_to(exports).as_posix()
    actual = {
        file.relative_to(exports).as_posix(): {
            "path": file.relative_to(exports).as_posix(),
            "sha256": sha256_file(file),
            "size": file.stat().st_size,
        }
        for file in exports.rglob("*")
        if file.is_file()
        and not file.name.endswith((".sha256", ".complete"))
        and file.relative_to(exports).as_posix() != selection_relative
    }
    if len(expected) != len(listed) or expected != actual:
        raise ManifestRefused("V8 selection operational export inventory differs")
    from research.gpu_sim_spike import v8_split_analyze

    reduced = v8_split_analyze.aggregate([exports], v8_commit)
    if reduced.get("verdict") != body["verdict"]:
        raise ManifestRefused("V8 selection analysis differs from independent export reducer")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "verdict": body["verdict"],
        "v8_split_pin": v8_commit,
        "operational_receipt": {"path": str(receipt.resolve()), "sha256": sha256_file(receipt)},
    }


def _canary_binding(receipt: Path, commit: str, preflight: dict[str, Any]) -> dict[str, str]:
    if not receipt.is_file():
        raise ManifestRefused("same-pin I0 canary receipt is absent")
    body = json.loads(receipt.read_text())
    manifest = receipt.parent / "manifest.json"
    if (body.get("status"), body.get("commit"), body.get("stop_verified")) != (
        "PASS",
        commit,
        True,
    ):
        raise ManifestRefused("same-pin native I0 PASS/release receipt required")
    if not manifest.is_file() or sha256_file(manifest) != body.get("manifest_sha256"):
        raise ManifestRefused("I0 canary manifest binding differs")
    saved = json.loads(manifest.read_text())
    if saved.get("script") != SCRIPT or saved.get("mac_preflight") != preflight:
        raise ManifestRefused("I0 canary is not from this exact packet")
    exports = receipt.parent / "exports"
    producer, native = exports / "integration-bench.json", exports / "native-parity.json"
    _verify(producer)
    _verify(native)
    listed = body.get("exports")
    if not isinstance(listed, list):
        raise ManifestRefused("I0 canary terminal export inventory is absent")
    expected = {row.get("path"): row for row in listed if isinstance(row, dict)}
    actual = {
        file.relative_to(exports).as_posix(): {
            "path": file.relative_to(exports).as_posix(),
            "sha256": sha256_file(file),
            "size": file.stat().st_size,
        }
        for file in exports.rglob("*")
        if file.is_file() and not file.name.endswith((".sha256", ".complete"))
    }
    if len(expected) != len(listed) or expected != actual:
        raise ManifestRefused("I0 canary terminal export inventory differs")
    native_body = json.loads(native.read_text())
    from research.gpu_sim_spike.benchmark import source_hashes

    move = native_body.get("move2_events")
    if (
        native_body.get("status") != "PASS"
        or native_body.get("commit") != commit
        or native_body.get("source_sha256") != source_hashes()
        or native_body.get("native_cuda_executed") is not True
        or not isinstance(move, dict)
        or move.get("status") != "PASS"
        or move.get("fallback") is not False
        or not isinstance(move.get("native_calls"), dict)
        or any(not isinstance(v, int) or v <= 0 for v in move["native_calls"].values())
    ):
        raise ManifestRefused("I0 canary native parity did not pass")
    report = json.loads(producer.read_text())
    cells = list(integration_units.allocation(canary=True))
    if (
        report.get("commit") != commit
        or report.get("source_sha256") != source_hashes()
        or report.get("completed_units") != cells
        or report.get("status") != "MEASURED_PENDING_MAC_REVIEW"
    ):
        raise ManifestRefused("I0 canary has incomplete roster coverage")
    from research.gpu_sim_spike import integration_analyze

    reduced = integration_analyze.aggregate([exports], commit, canary=True)
    if reduced.get("verdict") != "PASS_CANARY":
        raise ManifestRefused("independent I0 canary reducer did not pass")
    return {"path": str(receipt.resolve()), "sha256": sha256_file(receipt), "analysis": reduced}


def _main_forecast(canary_receipt: Path, commit: str) -> dict[str, Any]:
    """Scale measured E64 full calls, then charge the frozen continuity work.

    This is deliberately conservative: a canary unit includes both arms' plain
    and diagnostic ticks, and E scaling is linear.  The E1024 continuity ruler
    adds 96 progressing ticks per arm/roster on top of ordinary cells.
    """
    receipt = json.loads(canary_receipt.read_text())
    elapsed = receipt.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ManifestRefused("I0 canary lacks measured operational elapsed time")
    exports = canary_receipt.parent / "exports"
    rows = integration_units.load_ledger(
        exports / "verified-ledger.json",
        integration_units.namespace(commit, canary=True),
        integration_units.allocation(canary=True),
        exports=exports,
    )
    unit_seconds = []
    for row in rows.values():
        total = 0.0
        for arm in ("R", "I"):
            values = row.get("arms", {}).get(arm, {})
            total += sum(values.get("plain_seconds", []))
            total += sum(stage.get("wall_seconds", 0.0) for stage in values.get("stages", []))
        if not math.isfinite(total) or total <= 0:
            raise ManifestRefused("I0 canary timing rows are malformed")
        unit_seconds.append(total)
    if len(unit_seconds) != 3:
        raise ManifestRefused("I0 canary timing coverage is incomplete")
    e64_pair = max(unit_seconds)
    # 12 cells: three rosters at E6/E64/E256/E1024, each has three repeats.
    ordinary_pairs = 3 * 3 * sum(world / 64 for world in (6, 64, 256, 1024))
    # Each E1024 roster adds 3x32 progressing ticks for each of R/I, relative
    # to one E64 canary R/I pair. Reset is conservatively charged as another
    # pair per block.
    continuity_pairs = 3 * (3 * 32 + 3) * (1024 / 64)
    useful_seconds = e64_pair * (ordinary_pairs + continuity_pairs)
    setup_seconds = max(120.0, float(elapsed) - sum(unit_seconds))
    forecast_seconds = setup_seconds + useful_seconds
    minutes = math.ceil(forecast_seconds / 60)
    if minutes > 60 or forecast_seconds > 120 * 60:
        raise ManifestRefused("measured I0 canary forecast exceeds registered main ceiling")
    return {
        "method": "max-E64-full-pair linear scaling plus 32-tick continuity",
        "canary_elapsed_seconds": elapsed,
        "max_e64_pair_seconds": e64_pair,
        "setup_seconds": setup_seconds,
        "ordinary_pair_equivalents": ordinary_pairs,
        "continuity_pair_equivalents": continuity_pairs,
        "forecast_seconds": forecast_seconds,
        "forecast_minutes": max(3, minutes),
    }


def manifest(
    *,
    commit: str,
    output_dir: Path,
    ledger: Path,
    canary_ledger: Path,
    canary: bool,
    preflight: dict[str, Any],
    rate: float = 10.0,
    canary_receipt: Path | None = None,
    v8_selection_receipt: Path | None = None,
    template: bool = False,
) -> dict[str, Any]:
    if not math.isfinite(rate) or not 0 < rate <= 10:
        raise ManifestRefused("A100 rate must be positive and <=10 u/h")
    integration_units.load_ledger(
        ledger, integration_units.namespace(commit), integration_units.allocation()
    )
    integration_units.load_ledger(
        canary_ledger,
        integration_units.namespace(commit, canary=True),
        integration_units.allocation(canary=True),
    )
    if template and (canary or canary_receipt or v8_selection_receipt):
        raise ManifestRefused("only an unadmitted main may be a template")
    if not canary and not template and (canary_receipt is None or v8_selection_receipt is None):
        raise ManifestRefused("I0 main requires both native canary and local V8 selection evidence")
    admission = _admission(commit)
    # The 60-minute I0 forecast is split into a <=10-minute canary and a
    # <=60-minute main estimate.  Timeouts are exactly the qualified 2x floor.
    forecast = None
    canary_binding = None
    if canary:
        estimate, timeout, seconds, cap, reserve = 10, 24, 1200, 4.0, 2 / 3
    elif template:
        estimate, timeout, seconds, cap, reserve = 50, 120, 5400, 20.0, 10 / 3
    else:
        canary_binding = _canary_binding(canary_receipt, commit, preflight)
        forecast = _main_forecast(canary_receipt, commit)
        estimate, timeout, seconds, cap, reserve = (
            forecast["forecast_minutes"],
            120,
            5400,
            20.0,
            10 / 3,
        )
    result: dict[str, Any] = {
        "job_id": f"gpu-integration-i0-{'canary' if canary else 'main'}-{commit[:12]}-01",
        "commit": commit,
        "script": SCRIPT,
        "claim": "I0 engineering timing only; no hero, release, or serving claim.",
        "env": {
            "COLAB_CLI_REQUIREMENTS": f"{PACKAGE}/integration_requirements.txt",
            "PYTHONUNBUFFERED": "1",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_VECTOR_FLOOD": "static",
            "GPU_SIM_VECTOR_GEOMETRY": "native",
            "GPU_SIM_V8_INDEXED": "0",
            "INTEGRATION_BENCH_DEVICE": "cuda",
            "INTEGRATION_BENCH_SECONDS": str(seconds),
            "INTEGRATION_BENCH_LEDGER_PATH": CANARY_LEDGER_DEST if canary else LEDGER_DEST,
            "INTEGRATION_BENCH_SMOKE": "0",
            "INTEGRATION_BENCH_CANARY": "1" if canary else "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        },
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 12,
            "est_minutes": estimate,
        },
        "timeout_minutes": timeout,
        "output_dir": str(output_dir.resolve()),
        "inputs": checkpoint_inputs()
        + _ledger_inputs(ledger, "inputs")
        + _ledger_inputs(canary_ledger, "canary-inputs")
        + _bindings(),
        "budget": {
            "unit_cap": cap,
            "rate_units_per_hour": rate,
            "reserve_units": reserve,
            "other_assignments": {},
        },
        "lease_seconds": 600,
        "max_export_bytes": 64 * 1024**2,
        "required_exports": [
            "integration-bench.json",
            "native-parity.json",
            "verified-ledger.json",
            "discrepancy-register.json",
        ],
        "resume": {
            "unit_kind": "timing-cell",
            "ledger_path": CANARY_LEDGER_DEST if canary else LEDGER_DEST,
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "source_paths": a1_prof_pack._source_paths(),
        "transport_pin": TRANSPORT_PIN,
        "mac_preflight": preflight,
        "component_admission": admission,
        "integration_envelope": {
            "base_forecast_minutes": 60,
            "effective_wall_minutes": 144,
            "effective_units": 24,
            "canary": {
                "forecast_minutes": 10,
                "wall_minutes": 24,
                "unit_cap": 4,
                "reserve_units": 2 / 3,
            },
            "main": {
                "forecast_minutes": 50,
                "wall_minutes": 120,
                "unit_cap": 20,
                "reserve_units": 10 / 3,
            },
            "primary_nine_cells": list(integration_units.allocation()[:9]),
            "historical_e6_controls": list(integration_units.allocation()[9:]),
            "total_effective_units": 24,
        },
        "canary": (
            {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT}
            if canary
            else {"required": "same-pin native I0 canary receipt"}
        ),
    }
    if not canary and canary_receipt:
        result["canary"] = {"receipt": canary_binding}
        result["v8_selection"] = _v8_selection_binding(v8_selection_receipt)
        result["measured_main_forecast"] = forecast
    if canary:
        result["env"].update(SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE=PACKAGE)
    result["scientific_admission"] = not template
    if template:
        result["scientific_admission"] = False
        result["admission_reason"] = (
            "TEMPLATE_ONLY: same-pin canary and V8 SELECT_G/SELECT_GV are required"
        )
    result = convert_public_checkpoints(result, repo=ROOT)
    result["inputs"] = bundle_inputs(
        result["inputs"],
        output_dir.parent / f"metadata-{'canary' if canary else 'main'}.tar",
        reuse_verified=True,
    )
    result["env"]["METADATA_ARCHIVE_SHA256"] = next(
        x["sha256"] for x in result["inputs"] if x["dest"] == "metadata.tar"
    )
    assert_upload_budget(result, repo=ROOT)
    return result


def _write(path: Path, text: str) -> None:
    if path.exists() and path.read_text() != text:
        raise ManifestRefused(f"immutable packet differs: {path}")
    if not path.exists():
        write_once(path, text)


def prepare(
    artifact_dir: Path,
    commit: str,
    preflight_path: Path,
    rate: float,
    *,
    canary_receipt: Path | None = None,
    v8_selection_receipt: Path | None = None,
    validate: bool = False,
) -> dict[str, Any]:
    require_source_pin(commit)
    preflight = preflight_binding(preflight_path, commit)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ledgers = (
        (artifact_dir / "verified-ledger.json", False),
        (artifact_dir / "canary-ledger/verified-ledger.json", True),
    )
    for path, flag in ledgers:
        if not path.exists():
            integration_units.publish_json(
                path,
                integration_units.empty_ledger(
                    integration_units.namespace(commit, canary=flag),
                    integration_units.allocation(canary=flag),
                ),
            )
    main_ledger, canary_ledger = ledgers[0][0], ledgers[1][0]
    canary = manifest(
        commit=commit,
        output_dir=artifact_dir / "canary-output",
        ledger=main_ledger,
        canary_ledger=canary_ledger,
        canary=True,
        preflight=preflight,
        rate=rate,
    )
    main = manifest(
        commit=commit,
        output_dir=artifact_dir / "main-output",
        ledger=main_ledger,
        canary_ledger=canary_ledger,
        canary=False,
        preflight=preflight,
        rate=rate,
        canary_receipt=canary_receipt,
        v8_selection_receipt=v8_selection_receipt,
        template=canary_receipt is None,
    )
    if validate:
        from research.colab_cli.supervisor import validate as validate_manifest

        validate_manifest(main if canary_receipt else canary, repo=ROOT)
    _write(
        artifact_dir / "integration-canary.json",
        json.dumps(canary, indent=2, sort_keys=True) + "\n",
    )
    _write(
        artifact_dir
        / ("integration-main.json" if canary_receipt else "integration-main-template.json"),
        json.dumps(main, indent=2, sort_keys=True) + "\n",
    )
    dispatch = (
        shlex.join(
            [
                "nice",
                "-n",
                "15",
                str(ROOT / "tools/ops/dispatch.sh"),
                f"gpu-integration-i0-canary-{commit[:8]}",
                str(artifact_dir / "integration-canary.json"),
                "--broker",
                str(ROOT),
                "--cwd",
                str(ROOT),
                "--state-dir",
                str(artifact_dir / "state-canary"),
                "--dispatch-txt",
                str(artifact_dir / "DISPATCH.txt"),
            ]
        )
        + "\n"
    )
    _write(
        artifact_dir / "DISPATCH.txt",
        "# PREPARED ONLY. Validate/account before dispatch.\n"
        + dispatch
        + "# Main only after canary PASS and V8 SELECT_G/SELECT_GV admission.\n",
    )
    return {
        "prepared": True,
        "artifact_dir": str(artifact_dir),
        "commit": commit,
        "main_admitted": canary_receipt is not None and v8_selection_receipt is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--rate-units-per-hour", type=float, default=10.0)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--v8-selection-receipt", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.artifact_dir,
                args.commit,
                args.preflight,
                args.rate_units_per_hour,
                canary_receipt=args.canary_receipt,
                v8_selection_receipt=args.v8_selection_receipt,
                validate=args.validate,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
