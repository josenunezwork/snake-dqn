"""Prepare amendment-06 source-repair packs without allocating Colab.

The writer intentionally emits *blocked* records.  It is only a deterministic
way to bind the imported origin and the judge's prices; Claude's live inventory
and terminal-receipt checks remain the only path to dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json
from research.bl_ppo.recovery import snapshot_files

QUALIFIED_CLI_TREE = "d0d92bdb572b770284a1fea6152417ffa5f06bea"
RULING_SHA256 = "fe0584595cca6f6fcb28071692e52e4fe8460f3ab90fc29429c027203ff9ebfc"
ORIGIN_COMMIT = "5d877123ce8734537ef1f6924d5543be50dacdf7"
PREFIX_ACTIONS = 49_152


def _tree(repo: Path, commit: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{commit}:research/colab_cli"], cwd=repo, text=True
    ).strip()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_row(path: Path, root: Path) -> dict[str, Any]:
    return {
        "name": str(path.relative_to(root)),
        "sha256": digest(path),
        "size_bytes": path.stat().st_size,
    }


def build_origin_import(
    *, origin_root: Path, failed_receipt: Path, source_control: Path, offline_audit: Path
) -> dict[str, Any]:
    """Build the hash-bound import block from the immutable origin closure."""
    origin_root, failed_receipt, source_control, offline_audit = (
        origin_root.resolve(),
        failed_receipt.resolve(),
        source_control.resolve(),
        offline_audit.resolve(),
    )
    snapshot = origin_root / "p1-y3-origin-pause-snapshot-00000.pt"
    status = origin_root / "training-status.json"
    if (
        not snapshot.is_file()
        or not status.is_file()
        or not failed_receipt.is_file()
        or not offline_audit.is_file()
    ):
        raise ValueError("amendment-06 requires the complete immutable origin set")
    control = json.loads(source_control.read_text())
    origin_status = json.loads(status.read_text())
    counters = origin_status.get("counters", {})
    if not isinstance(counters, dict) or not isinstance(origin_status.get("critic_refit"), dict):
        raise ValueError("origin status lacks frozen refit evidence")
    required = {
        "arm": "F",
        "seed": 20261021,
        "target": 100_000_000,
        "registration_sha256": control.get("registration_sha256"),
        "evaluation_bank_sha256": control.get("evaluation_bank_sha256"),
        "e8_critic_convergence_amendment_04_sha256": control.get(
            "e8_critic_convergence_amendment_04_sha256"
        ),
        "natural_collection_world_id_offset": 3_011_000,
    }
    if any(v is None for v in required.values()):
        raise ValueError("source control lacks amendment-06 identity fields")
    # This is the runner's definition of the complete transitive snapshot set;
    # do not accidentally import monitoring/status exports that are not state.
    catalog = [_file_row(path, origin_root) for path in snapshot_files(snapshot)]
    expected_counters = {
        name: value
        for name, value in counters.items()
        if name.startswith("critic_refit") or name.startswith("critic_fixed_holdout")
    }
    required_counters = {
        "critic_refit_transitions",
        "critic_refit_fit_transitions",
        "critic_fixed_holdout_transitions",
    }
    if not required_counters <= set(expected_counters) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in expected_counters.values()
    ):
        raise ValueError("origin status lacks exact refit counters")
    return {
        "schema": "bl-p1-y3-origin-import/v1",
        "origin_snapshot": {"path": snapshot.name, "sha256": digest(snapshot)},
        "origin_status": {"path": status.name, "sha256": digest(status)},
        "failed_receipt": {"path": failed_receipt.name, "sha256": digest(failed_receipt)},
        "offline_audit": {"path": offline_audit.name, "sha256": digest(offline_audit)},
        "source_commit": ORIGIN_COMMIT,
        "snapshot_catalog": catalog,
        "evidence_catalog": [
            _file_row(item, origin_root)
            for item in sorted(origin_root.glob("late-body-rows*"))
            if not item.name.endswith((".sha256", ".complete"))
        ],
        "expected_identity": required,
        "expected_counters": expected_counters,
        "critic_refit_sha256": _json_digest(origin_status["critic_refit"]),
        "prefix_witness": {
            "actor_actions": PREFIX_ACTIONS,
            "full_tape_sha256": "fb3c8c7b89bbe0f347ae89799fbb66cf9bf8373e987b1d8212fe286c7e95280c",
        },
        "allowed_diff_paths": [
            "research/bl_ppo/run.py",
            "research/bl_ppo/p1_runtime.py",
            "research/bl_ppo/p1_source_repair.py",
            "research/bl_ppo/test_p1_source_repair.py",
        ],
    }


# Runtime scientific modules are deliberately absent from this allowlist.
DIFF_PATHS = {
    "run.py",
    "p1_runtime.py",
    "p1_origin_import.py",
    "p1_source_repair.py",
    "p1_repair_admission.py",
    "p1_qualification.py",
    "p1_source_audit.py",
    "e8_04_source_admission.py",
    "p1_kernel_ambiguity_retry.py",
    "p1_next_stage.py",
}


def diff_scope(repo: Path, commit: str) -> dict[str, Any]:
    """Inspect the actual frozen diff on the Mac and bind both sides by hash."""
    selector = ["src", "research/bl_ppo/*.py"]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", f"{ORIGIN_COMMIT}..{commit}", "--", *selector],
        cwd=repo,
        text=True,
    ).splitlines()
    rows = []
    for name in changed:
        path = Path(name)
        if path.parent.as_posix() != "research/bl_ppo" or not (
            path.name in DIFF_PATHS
            or path.name.startswith("test_p1_")
            or path.name == "test_e8_04_source_admission.py"
        ):
            raise ValueError(f"out-of-scope origin-to-repair pin diff: {name}")
        body = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo)
        if (repo / name).read_bytes() != body:
            raise ValueError(f"working source differs from repair pin: {name}")
        old = subprocess.run(
            ["git", "show", f"{ORIGIN_COMMIT}:{name}"], cwd=repo, capture_output=True
        )
        rows.append(
            {
                "path": name,
                "repair_sha256": hashlib.sha256(body).hexdigest(),
                "origin_sha256": (
                    hashlib.sha256(old.stdout).hexdigest() if old.returncode == 0 else None
                ),
            }
        )
    stat = subprocess.check_output(
        ["git", "diff", "--stat", f"{ORIGIN_COMMIT}..{commit}", "--", *selector],
        cwd=repo,
        text=True,
    )
    return {
        "source_commit": ORIGIN_COMMIT,
        "repair_commit": commit,
        "paths": rows,
        "diff_stat": stat,
        "selector": selector,
    }


def _copy_exact(source: Path, target: Path) -> None:
    """Copy verified bytes with create-only sidecars, including raw broker JSON."""
    import shutil

    from research.bl_ppo.exports import export_bytes, verify_export

    target.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_size < 1_000_000:
        export_bytes(target, source.read_bytes())
    else:
        verify_export(source)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, 8 * 1024**2)
        for suffix in (".sha256", ".complete"):
            with Path(str(target) + suffix).open("xb") as dst:
                dst.write(Path(str(source) + suffix).read_bytes())
        verify_export(target)


def upload_preflight(manifest: dict[str, Any], repo: Path) -> dict[str, Any]:
    """Exercise real pinned source preparation; preserve any qualified-cap refusal."""
    import time

    from research.bl_upload import qualification as q

    started = time.monotonic()
    inputs = q._input_rows(manifest, repo)
    bootstrap = q._pinned_bootstrap_bytes(repo, manifest["commit"])
    total = sum(r["bytes"] for r in inputs + bootstrap)
    pinned = q._pinned_source_module(repo, manifest["commit"])
    plan, fallback_bytes, plan_sha = q._source_plan(repo, manifest, pinned)
    forecast = q._forecast(total + fallback_bytes, len(inputs) + 1)
    try:
        result = q.qualify(manifest, repo=repo)
    except ValueError as exc:
        result = {"status": "HOLD_QUALIFIED_UPLOAD_LIMIT", "reason": str(exc)}
    result.update(
        schema="bl-p1-repair-upload-preflight/v1",
        commit=manifest["commit"],
        actual_upload_bytes=total,
        fallback_upload_bytes=total + fallback_bytes,
        fallback_upload_forecast_seconds=forecast,
        source_plan_sha256=plan_sha,
        source_plan=plan,
        wall_seconds=time.monotonic() - started,
        provider_calls=0,
    )
    return result


def prepare_production(
    *,
    source_manifest: Path,
    acceptance: Path,
    root: Path,
    repo: Path,
    commit: str,
    origin_root: Path,
    failed_receipt: Path,
    offline_audit: Path,
) -> dict[str, Any]:
    """Freeze actual canary/repair inputs and ruled future-stage templates."""
    import copy

    from research.bl_ppo.export_budget import freeze_control_budget
    from research.bl_ppo.manifests import _bounded_source_paths, input_row
    from research.bl_ppo.p1_inputs import unpack_bundle
    from research.bl_ppo.p1_manifests import bundle_inputs
    from research.bl_ppo.p1_resume import empty_ledger
    from research.colab_cli.canary import prepare as canary_manifest
    from research.colab_cli.supervisor import validate

    root, repo = root.resolve(), repo.resolve()
    from research.colab_cli.admission import require_final_only

    require_final_only(json.loads(acceptance.read_text()), repo)
    if root.exists():
        raise ValueError("repair pack directory must be fresh")
    if _tree(repo, commit) != QUALIFIED_CLI_TREE:
        raise ValueError("repair pin does not contain the qualified CLI tree")
    scope = diff_scope(repo, commit)
    original = json.loads(source_manifest.read_text())
    if original.get("script") != "research/bl_ppo/p1_colab.sh":
        raise ValueError("requires the original source supervisor manifest")
    root.mkdir(parents=True)
    prior = root / "prior-inputs"
    bundle = next(r for r in original["inputs"] if r["dest"] == "p1-inputs.tar.gz")
    if digest(Path(bundle["path"])) != bundle["sha256"]:
        raise ValueError("original input bundle changed")
    unpack_bundle(Path(bundle["path"]), prior)
    block = build_origin_import(
        origin_root=origin_root,
        failed_receipt=failed_receipt,
        source_control=prior / "p1-control.json",
        offline_audit=offline_audit,
    )
    block["allowed_diff_paths"] = [r["path"] for r in scope["paths"]]
    block["diff_scope"] = scope
    for field, name in [
        ("origin_status", "origin-training-status.json"),
        ("failed_receipt", "failed-supervisor-receipt.json"),
        ("offline_audit", "offline-audit.json"),
    ]:
        block[field]["path"] = name
    control = json.loads((prior / "p1-control.json").read_text())
    control.update(
        source_commit=commit,
        scientific_admission=True,
        y3_origin_import=block,
        checkpoint_every_seconds=900,
    )
    attempt_tag = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    control["resume"].update(
        namespace=f"bl_p1_y3_pack04_repair06_{attempt_tag}",
        allocation="source_latebody_06_e8_04_pack04",
        max_outcome_blind_retries=1,
    )
    byte_budget = freeze_control_budget(
        control, timeout_minutes=200, max_export_bytes=original["max_export_bytes"]
    )
    small = root / "inputs"
    small.mkdir()
    replacements = {
        "p1-control.json",
        "canary-control.json",
        "verified-ledger.json",
        "canary-verified-ledger.json",
    }
    for item in prior.iterdir():
        if item.name.removesuffix(".sha256").removesuffix(".complete") not in replacements:
            with (small / item.name).open("xb") as out:
                out.write(item.read_bytes())
    export_json(small / "p1-control.json", control)
    export_json(
        small / "verified-ledger.json",
        empty_ledger(control["resume"]["namespace"], control["resume"]["allocation"]),
    )
    canary_control = json.loads((prior / "canary-control.json").read_text())
    canary_control.update(
        source_commit=commit,
        scientific_admission=True,
        y3_origin_import=block,
        checkpoint_every_seconds=900,
    )
    canary_control["resume"].update(
        namespace=f"bl_p1_canary_repair06_{attempt_tag}", max_outcome_blind_retries=1
    )
    freeze_control_budget(
        canary_control, timeout_minutes=30, max_export_bytes=original["max_export_bytes"]
    )
    export_json(small / "canary-control.json", canary_control)
    export_json(
        small / "canary-verified-ledger.json",
        empty_ledger(canary_control["resume"]["namespace"], canary_control["resume"]["allocation"]),
    )
    # Direct inputs retain the exact original snapshot bytes; never rebuild.
    payloads = root / "origin-inputs"
    for row in block["snapshot_catalog"] + block["evidence_catalog"]:
        _copy_exact(origin_root / row["name"], payloads / row["name"])
    for src, name in [
        (origin_root / "training-status.json", block["origin_status"]["path"]),
        (failed_receipt, block["failed_receipt"]["path"]),
        (offline_audit, block["offline_audit"]["path"]),
    ]:
        _copy_exact(src, payloads / name)
    rows = [input_row(p, p.name) for p in sorted(small.iterdir())]
    public = next(r for r in original["inputs"] if r["dest"] == "public-inputs.json")
    inputs = bundle_inputs(root, [public, *rows])
    inputs += [input_row(p, p.name) for p in sorted(payloads.iterdir())]
    # Validate-only is run against the same flat namespace the runtime receives.
    for item in payloads.iterdir():
        import os

        os.link(item, small / item.name)
    template = copy.deepcopy(original)
    template.pop("canary", None)
    template.update(
        commit=commit,
        job_id=f"bl-p1-source-repair-10-{attempt_tag}",
        output_dir=str(root / "repair-output"),
        inputs=inputs,
        timeout_minutes=200,
        scientific_admission=False,
        p1_finalisation_ruling_sha256=RULING_SHA256,
        max_total_export_bytes=byte_budget["max_total_export_bytes"],
        resume=copy.deepcopy(control["resume"]),
        device="cuda:0",
    )
    template["env"].update(BL_PPO_COMMIT=commit, BL_PPO_DEVICE="cuda:0")
    template["requirements"].update(gpu="L4", allowed_gpus=["L4"], est_minutes=50)
    template["budget"].update(
        unit_cap=6.16,
        reserve_units=1.03,
        rate_units_per_hour=1.54,
        rate_tolerance_units_per_hour=0.0,
        other_assignments={},
    )
    template.setdefault("transport", {})["dispatch_ready"] = False
    template["source_paths"] = sorted(
        set(template["source_paths"])
        | set(
            _bounded_source_paths(
                {"commit": commit, "source_paths": ["research/bl_ppo", "research/colab_cli"]}
            )
        )
    )
    template["bl_p1"]["control"] = control
    canary = canary_manifest(
        template,
        package="bl_ppo",
        output=root / "canary-output",
        job_id=f"bl-p1-source-repair-10-canary-{attempt_tag}",
        timeout_minutes=30,
        env={"BL_P1_CANARY_CONFIG": '{"pause_at":32,"processes":1,"rollout_steps":16,"worlds":2}'},
    )
    canary["budget"].update(unit_cap=2.0, reserve_units=0.4)
    canary["requirements"]["est_minutes"] = 15
    canary["required_exports"] = ["receipt.json", "canary/pause.json", "canary/pause-snapshot.pt"]
    canary["device"] = "cuda:0"
    canary["resume"] = copy.deepcopy(canary_control["resume"])
    canary["p1_finalisation_ruling_sha256"] = RULING_SHA256
    canary = validate(canary, repo)
    canary.update(scientific_admission=False)
    canary.setdefault("transport", {})["dispatch_ready"] = False
    export_json(root / "canary-manifest.json", canary)
    export_json(root / "repair-manifest.json", template)
    export_json(root / "cross-pin-diff.json", scope)
    upload = upload_preflight(canary, repo)
    export_json(root / "upload-preflight.json", upload)
    import statistics

    historical = json.loads(failed_receipt.read_text())
    downloads = [
        r["bytes_per_second"]
        for r in historical.get("transfers", [])
        if r.get("direction") == "download"
        and r.get("status") == "verified"
        and r.get("size_bytes", 0) >= 8 * 1024**2
        and r.get("bytes_per_second", 0) > 0
    ]
    if not downloads:
        raise ValueError("repair forecast requires measured historical large-file download rates")
    pause_size = next(
        r["size"]
        for r in historical["exports"]
        if r["path"].endswith("p1-y3-source-continuation-pause-snapshot-00000.pt")
    )
    expected_download_bytes = (
        sum(r["size_bytes"] for r in block["snapshot_catalog"] + block["evidence_catalog"])
        + 2 * pause_size
    )
    median_download_seconds = expected_download_bytes / statistics.median(downloads)
    worst_download_seconds = expected_download_bytes / min(downloads)
    forecast_minutes = (
        200 + upload["fallback_upload_forecast_seconds"] + 960 + median_download_seconds
    ) / 60
    forecast = {
        "schema": "bl-p1-repair-measured-forecast/v1",
        "estimate_minutes": 50,
        "measured_upload_preflight_wall_seconds": upload["wall_seconds"],
        "exact_upload_bytes": upload["actual_upload_bytes"],
        "upload_forecast_seconds": upload["fallback_upload_forecast_seconds"],
        "expected_download_bytes": expected_download_bytes,
        "download_rate_median_bytes_per_second": statistics.median(downloads),
        "download_rate_worst_bytes_per_second": min(downloads),
        "download_seconds_median": median_download_seconds,
        "download_seconds_worst": worst_download_seconds,
        "bootstrap_seconds": 200,
        "actor_cadence_plus_suffix_seconds": 960,
        "forecast_minutes": forecast_minutes,
        "twice_forecast_minutes": 2 * forecast_minutes,
        "frozen_timeout_minutes": 200,
        "frozen_cap_units": 6.16,
        "caps_raised": False,
        "no_live_transfer_claim": True,
        "status": (
            "HOLD_JUDGE_REQUIRED"
            if 2 * forecast_minutes > 200 or upload["status"] != "PASS_NOT_DISPATCHED"
            else "PASS_NOT_DISPATCHED"
        ),
    }
    export_json(root / "measured-forecast.json", forecast)
    # Preserve separately frozen stage prices, with unmet prerequisite fuses.
    futures = {}
    for stage, cap, reserve, estimate, cadence, retries in [
        ("replacement", 3.86, 0.78, 30, 900, 1),
        ("qualification", 4.0, 0.80, 40, 1800, 0),
    ]:
        future = copy.deepcopy(template)
        phase = "y3-replacement" if stage == "replacement" else "qual-1m"
        future["env"]["BL_P1_MODE"] = phase
        future["requirements"]["est_minutes"] = estimate
        future.update(
            job_id=f"bl-p1-repair10-{stage}-{attempt_tag}",
            timeout_minutes=120,
            output_dir=str(root / f"{stage}-output"),
        )
        future["budget"].update(unit_cap=cap, reserve_units=reserve)
        future["resume"]["max_outcome_blind_retries"] = retries
        future["held_reason"] = (
            "Fresh receipt-bound sibling required after source proof"
            if stage == "replacement"
            else "Full Y3 PASS, Claude and Codex-V required"
        )
        future["bl_p1"]["control"] = dict(
            control,
            phase=phase,
            checkpoint_every_seconds=cadence,
            pause_at=0 if stage == "replacement" else 1_000_064,
            requested_pause_at=0 if stage == "replacement" else 1_000_000,
        )
        export_json(root / f"{stage}-held-manifest.json", future)
        futures[stage] = {
            "cap": cap,
            "timeout_minutes": 120,
            "cadence_seconds": cadence,
            "max_mechanical_retries": retries,
            "admitted": False,
        }
    result = {
        "schema": "bl-p1-source-repair-preparation/v1",
        "status": "PREPARED_NOT_ADMITTED",
        "commit": commit,
        "ruling_sha256": RULING_SHA256,
        "qualified_colab_cli_tree": QUALIFIED_CLI_TREE,
        "manifest_sha256": {n: digest(root / f"{n}-manifest.json") for n in ("canary", "repair")},
        "future_stages": futures,
        "upload_preflight_status": upload["status"],
        "provider_calls": 0,
        "envelope_units": 34.50,
        "charged_legacy_units": 10.11,
        "worst_case_units": 34.15,
        "happy_path_units": 22.13,
        "warning": "WARNING_UNEXPLAINED_DEBIT",
        "acceptance": {"path": str(acceptance), "sha256": digest(acceptance)},
        "exact_origin_payload_bytes": sum(p.stat().st_size for p in payloads.iterdir()),
    }
    export_json(root / "preparation.json", result)
    return result


def main() -> None:
    """Create-only, zero-provider-call preparation at a frozen executable pin."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-manifest",
        "acceptance",
        "root",
        "repo",
        "origin-root",
        "failed-receipt",
        "offline-audit",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_production(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
