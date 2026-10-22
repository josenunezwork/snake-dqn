"""Prepare Z-9 native ruler jobs offline, then bind real canary proof after release."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_served_search import fixture_bundle, native_ruler
from research.bl_served_search.exports import digest, export_bytes, export_json, verify_export
from research.bl_served_search.manifests import source_paths
from research.bl_served_search.transport_v2 import preflight
from research.colab_cli import canary
from research.colab_cli.schema import validate_manifest
from research.colab_cli.supervisor import Refused, validate

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "research/bl_served_search/run_native_l4.sh"
EXPORTS = ["ruler-receipt.json", "cuda-action-audit.json"]
RATE = 2.0


def runtime_source_paths(repo: Path = ROOT) -> list[str]:
    """Include files read for identity, beyond Python's static import closure."""
    return sorted(
        set(source_paths(repo))
        | set(native_ruler.source_identity())
        | {SCRIPT, "research/bl_upload"}
    )


def validate_pending(manifest: dict, *, repo: Path = ROOT) -> dict:
    """Exercise every broker check up to its last, necessarily live canary gate.

    No fabricated receipt or bypass flag is used. A production manifest cannot
    pass full admission before the requested, undispatched canary has run.
    """
    validate_manifest(manifest)
    try:
        validate(manifest, repo=repo)
    except Refused as exc:
        if str(exc) != "dispatch requires a SHA-bound family canary receipt":
            raise
    else:
        raise ValueError("pending manifest unexpectedly has dispatch authority")
    return {
        "status": "PASS_PRECANARY_STRUCTURE",
        "admission": "PENDING_REAL_CANARY_RELEASE",
        "gate": "unchanged supervisor.validate; exact expected final-gate refusal",
    }


def _input(path: Path, dest: str | None = None) -> dict[str, str]:
    return {"path": str(path.resolve()), "dest": dest or path.name, "sha256": digest(path)}


def _published(commit: str, repo: Path) -> None:
    refs = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=repo, text=True, timeout=10
    )
    if "origin/main" not in refs:
        raise ValueError("full source pin must already be published on origin/main")
    changed = subprocess.check_output(
        ["git", "diff", commit, "--name-only", "--", *runtime_source_paths(repo)],
        cwd=repo,
        text=True,
        timeout=20,
    )
    if changed.strip():
        raise ValueError("local runtime source differs from pin: " + changed)
    if canary.executable_identity(repo, commit) != canary.executable_identity(repo):
        raise ValueError("shared supervisor differs from pin")


def prepare(
    commit: str,
    out: Path,
    fixtures: Path,
    k1: Path,
    mac: Path,
    *,
    repo: Path = ROOT,
) -> dict[str, Any]:
    """Create fresh identical-upload canary/main manifests; never contact Colab."""
    _published(commit, repo)
    for path in (k1, mac):
        verify_export(path)
    mac_value = json.loads(mac.read_text())
    native_ruler.validate_paired_receipt(mac_value)
    if mac_value["platform"]["label"] != "mac":
        raise ValueError("passed Mac native receipt required")
    fixture_value = json.loads((fixtures / "ruler-fixtures.json").read_text())
    if fixture_value != mac_value["fixtures"]:
        raise ValueError("Mac receipt and uploaded fixtures differ")
    k1_value = json.loads(k1.read_text())
    if (
        k1_value.get("status") != "K1_PASS"
        or k1_value.get("speedup", 0) < 5
        or mac_value["k1_sha256"] != digest(k1)
    ):
        raise ValueError("same Mac-bound K1_PASS receipt required")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    bundle = fixture_bundle.pack(
        fixtures, out / fixture_bundle.BUNDLE_NAME, source_validator=native_ruler.validate_source
    )
    # Renaming an export requires a new name-bearing sidecar; payload is exact.
    portable_k1 = out / "inputs" / "k1-timing.json"
    export_bytes(portable_k1, k1.read_bytes())
    catalog = repo / "research/t34_exit_b8/PUBLIC_INPUTS.json"
    inputs = [_input(bundle), _input(catalog, "public-inputs.json")]
    inputs += [_input(Path(str(portable_k1) + suffix)) for suffix in ("", ".sha256", ".complete")]
    paths = runtime_source_paths(repo)
    env = {
        "BL_RULER_FIXTURES_SHA256": digest(bundle),
        "BL_NATIVE_CANARY": "0",
        "BL_NATIVE_MAX_SECONDS": "1800",
        "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
        "PYTHONUNBUFFERED": "1",
        "SNAKE_DQN_DEVICE": "cpu",
        **dict.fromkeys(native_ruler.THREAD_ENV, "1"),
    }
    token = out.name.lower().replace("_", "-")[-20:]
    main = {
        "job_id": "bl-native-l4-main-" + token,
        "commit": commit,
        "script": SCRIPT,
        "source_paths": paths,
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 12, "est_minutes": 36},
        "timeout_minutes": 72,
        "output_dir": str(out / "main-runtime"),
        "budget": {
            "unit_cap": 3,
            "rate_units_per_hour": RATE,
            "reserve_units": 0.6,
            "other_assignments": {},
        },
        "env": env,
        "inputs": inputs,
        "lease_seconds": 300,
        "required_exports": EXPORTS,
        "max_export_bytes": 64 * 1024**2,
        "bl_native_ruler": {
            "status": "PREPARED_NOT_DISPATCHED",
            "mac_receipt": _input(mac),
            "fixtures_sha256": digest(fixtures / "ruler-fixtures.json"),
            "cells": ["all", "exact_stop"],
            "repeats_per_fixture": 30,
            "workers": 12,
            "estimate_basis": (
                "prospective allocation envelope: <=30 min native entrypoint plus <=6 min "
                "central setup/export estimate; 2x whole-session margin; no L4 speed claim "
                "and no Mac timing transfer"
            ),
            "canary_plus_main_unit_cap": 4,
            "combined_reserve_units": 0.8,
            "combined_wall_liability_units": 3.0,
            "inventory": "refresh live account inventory before each admission",
        },
    }
    probe = copy.deepcopy(main)
    probe.update(
        job_id="bl-native-l4-canary-" + token,
        output_dir=str(out / "canary-runtime"),
        timeout_minutes=18,
    )
    probe["requirements"]["est_minutes"] = 9
    probe["bl_native_ruler"]["repeats_per_fixture"] = 1
    probe["budget"].update(unit_cap=1, reserve_units=0.2)
    probe["env"].update(
        BL_NATIVE_CANARY="1",
        BL_NATIVE_MAX_SECONDS="600",
        SNAKE_COLAB_CANARY="1",
        SNAKE_COLAB_CANARY_PACKAGE="research/bl_served_search",
    )
    # FAMILIES in the shared helper is a convenience registry, not admission
    # authority. The unchanged broker checks this new real script's full family,
    # commit, input inventory, scanner exports and CUDA evidence normally.
    probe["canary"] = {"schema": canary.SCHEMA, "mode": "probe", "family": canary.family(probe)}
    probe = validate(probe, repo=repo)
    pending = validate_pending(main, repo=repo)
    manifests = {}
    for name, value in (("canary", probe), ("main", main)):
        manifest_path = out / (name + ".json")
        export_json(manifest_path, value)
        transport = preflight(value, out / (name + "-transport.json"), repo=repo)
        manifests[name] = {
            "path": str(manifest_path),
            "sha256": digest(manifest_path),
            "validation": "PASS" if name == "canary" else pending["status"],
            "normal_upload_seconds": transport["normal_forecast_seconds"],
            "fallback_upload_seconds": transport["fallback_upload_forecast_seconds"],
            "actual_local_transport_seconds": transport["actual_transport_elapsed_seconds"],
            "local_input_count": transport["local_input_count"],
        }
    result = {
        "status": "PREPARED_NOT_DISPATCHED",
        "commit": commit,
        "manifests": manifests,
        "colab_calls": 0,
        "main_admission": pending,
        "combined_cap_units": 4,
        "combined_reserve_units": 0.8,
        "combined_wall_liability_units": 3,
        "scientific_worlds_executed": 0,
    }
    export_json(out / "preparation.json", result)
    return result


def bind(template: Path, receipt: Path, output: Path, *, repo: Path = ROOT) -> dict:
    """After real canary release, create the immutable broker-admitted main job."""
    verify_export(template)
    value = json.loads(template.read_text())
    if value.get("canary"):
        raise ValueError("unbound main template required")
    value["canary"] = {"receipt": {"path": str(receipt.resolve()), "sha256": digest(receipt)}}
    admitted = validate(value, repo=repo)  # independently re-derives real proof
    export_json(output, admitted)
    return admitted


def publish(output: Path, target: Path, mac: Path) -> None:
    """Publish exact recovered bytes only after clean broker release and pairing."""
    from research.bl_served_search.placement_stop import forecast

    receipt = json.loads((output / "receipt.json").read_text())
    saved = json.loads((output / "manifest.json").read_text())
    if (
        receipt.get("status") != "PASS"
        or receipt.get("stop_verified") is not True
        or receipt.get("error")
        or receipt.get("cleanup_error")
        or receipt.get("failure_class") is not None
        or receipt.get("manifest_sha256") != digest(output / "manifest.json")
        or receipt.get("commit") != saved["commit"]
        or receipt.get("job_id") != saved["job_id"]
    ):
        raise ValueError("clean source-bound broker PASS and verified release required")
    if saved.get("script") != SCRIPT or canary.is_canary(saved):
        raise ValueError("released main native ruler family required")
    canary.check(saved, ROOT)
    payload = output / "exports" / "ruler-receipt.json"
    verify_export(payload)
    observed = next((row for row in receipt["exports"] if row["path"] == payload.name), None)
    if observed != {
        "path": payload.name,
        "size": payload.stat().st_size,
        "sha256": digest(payload),
    }:
        raise ValueError("recovered native receipt differs from broker scanner inventory")
    verify_export(mac)
    forecast(json.loads(mac.read_text()), json.loads(payload.read_text()))
    export_bytes(target, payload.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--commit", required=True)
    for name in ("out", "fixtures", "k1", "mac"):
        prep.add_argument("--" + name, type=Path, required=True)
    binding = sub.add_parser("bind")
    for name in ("template", "receipt", "out"):
        binding.add_argument("--" + name, type=Path, required=True)
    publication = sub.add_parser("publish")
    for name in ("output", "target", "mac"):
        publication.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.commit, args.out, args.fixtures, args.k1, args.mac)))
    elif args.command == "bind":
        bind(args.template, args.receipt, args.out)
        print("BOUND_MAIN_VALIDATED_NOT_DISPATCHED")
    else:
        publish(args.output, args.target, args.mac)
        print("VERIFIED_NATIVE_RECEIPT_PUBLISHED")


if __name__ == "__main__":
    main()
