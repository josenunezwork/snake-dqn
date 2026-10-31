"""Prepare and seal A100 Colab queue manifests for DeviceSim mechanics-v2 parity."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from research.bl_upload import convert_public_checkpoints, qualify
from research.colab_cli import canary as colab_canary
from research.devicesim_v2_parity.units import (
    allocation_for_lane,
    empty_ledger,
    load_verified_ledger,
    namespace_for,
    sha256_file,
    verify_published_json,
    write_bytes_create_only,
)
from research.gpu_sim_spike.make_manifests import ManifestRefused, checkpoint_inputs
from tools.ops.import_preflight import import_closure, module_path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/devicesim_v2_parity"
SCRIPT = f"{PACKAGE}/job.sh"
REQUIREMENTS_REL = f"{PACKAGE}/requirements.txt"
QUALIFIED_TRANSPORT_TREE = "36285bc6cbb637b392b234f67b55cbaa03b37614"

MAIN_LEDGER_DEST = "inputs/verified-ledger.json"
CANARY_LEDGER_DEST = "canary-inputs/verified-ledger.json"
REQUIRED_EXPORTS = ["summary.json", "verified-ledger.json"]
IMPORT_ROOTS = ["research.devicesim_v2_parity.run", "research.bl_upload.inputs"]
WHOLE_SOURCE_DIRS = (
    "configs",
    "research/devicesim_v2_parity",
    "research/colab_cli",
    "research/bl_upload",
)
EXTRA_SOURCE_PATHS = (
    *WHOLE_SOURCE_DIRS,
    "web/backend/served_checkpoint_pins.json",
    "research/t34_exit_b8/PUBLIC_INPUTS.json",
    "research/apex_safety_20260926/deployment.yaml",
    "requirements.txt",
    "pyproject.toml",
    *(
        f"research/gpu_sim_spike/{stem}.{ext}"
        for stem in (
            "flood_cuda",
            "move_cuda",
            "event_cuda",
            "vector_geometry",
            "v8_indexed",
        )
        for ext in ("cpp", "cu")
    ),
)


def runtime_source_paths(repo: Path = ROOT) -> list[str]:
    """Compute sparse git-archive paths covering static import closure + CUDA/config assets."""
    repo = repo.resolve()
    closure, _ = import_closure(repo, list(IMPORT_ROOTS))
    py_paths: set[str] = set()
    for mod in closure:
        resolved = module_path(repo, mod.partition(":")[0])
        if resolved is None:
            continue
        rel = resolved.relative_to(repo).as_posix()
        if any(rel == d or rel.startswith(f"{d}/") for d in WHOLE_SOURCE_DIRS):
            continue
        py_paths.add(rel)
    return sorted(py_paths | set(EXTRA_SOURCE_PATHS))


# Owner 2x rule: record base & effective (2x caps, reserves, timeouts; est_minutes/rate unchanged).
CANARY_BASE = {
    "est_minutes": 10,
    "timeout_minutes": 12,
    "unit_cap": 2.0,
    "reserve_units": 1.0,
}
CANARY_EFFECTIVE = {
    "est_minutes": 10,
    "timeout_minutes": 24,
    "unit_cap": 4.0,
    "reserve_units": 2.0,
}
MAIN_BASE = {
    "est_minutes": 30,
    "timeout_minutes": 45,
    "unit_cap": 8.0,
    "reserve_units": 3.0,
}
MAIN_EFFECTIVE = {
    "est_minutes": 30,
    "timeout_minutes": 90,
    "unit_cap": 16.0,
    "reserve_units": 6.0,
}
A100_RATE_UNITS_PER_HOUR = 10.0


def _verify_commit_and_transport(repo: Path, commit: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ManifestRefused(f"commit must be 40 lowercase hex chars, got {commit!r}")
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    transport_tree = subprocess.check_output(
        ["git", "rev-parse", f"{commit}:research/colab_cli"],
        cwd=repo,
        text=True,
    ).strip()
    if transport_tree != QUALIFIED_TRANSPORT_TREE:
        raise ManifestRefused(
            f"commit {commit} has research/colab_cli tree {transport_tree}, "
            f"expected qualified tree {QUALIFIED_TRANSPORT_TREE}"
        )
    return transport_tree


def _verify_preflight_summary(preflight_summary: Path, *, commit: str) -> dict[str, Any]:
    _entry, summary = verify_published_json(preflight_summary)
    if summary.get("commit") != commit:
        raise ManifestRefused(
            f"preflight summary commit {summary.get('commit')!r} != target commit {commit!r}"
        )
    if summary.get("lane") != "preflight":
        raise ManifestRefused(f"expected preflight lane summary, got {summary.get('lane')!r}")
    if summary.get("status") != "PASS" or int(summary.get("unexplained_mismatches", -1)) != 0:
        raise ManifestRefused("local Mac preflight summary did not PASS with 0 mismatches")
    if not summary.get("complete_allocation_verified"):
        raise ManifestRefused("local Mac preflight did not complete full preflight allocation")
    return {
        "summary_path": str(preflight_summary.resolve()),
        "summary_sha256": sha256_file(preflight_summary),
        "commit": commit,
        "world_ticks": int(summary["world_ticks"]),
        "snake_ticks": int(summary["snake_ticks"]),
        "shard_elapsed_seconds_sum": float(summary["shard_elapsed_seconds_sum"]),
        "status": "PASS",
    }


def _ledger_input(path: Path, dest: str) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "dest": dest,
        "sha256": sha256_file(path),
    }


def resolve_canary_receipt_from_ops(ops_dir: Path, canary_queue_id: str) -> Path:
    """Resolve the terminal canary receipt across ready_queue mechanical retries."""
    from tools.ops import ready_queue

    queue_path = ops_dir / "ready-queue.json"
    if not queue_path.is_file():
        raise ManifestRefused(f"missing ready-queue state file: {queue_path}")
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    terminal_row = ready_queue.latest(queue, canary_queue_id)
    if terminal_row.get("status") != "PASS":
        raise ManifestRefused(
            f"canary queue row {terminal_row.get('id')!r} status is "
            f"{terminal_row.get('status')!r}, expected 'PASS'"
        )
    dispatch_name = terminal_row["argv"][1]
    observed = ready_queue.observations(ops_dir)
    _current_name, state = ready_queue.current_observation(observed, dispatch_name)
    if not state or not state.get("manifest"):
        raise ManifestRefused(f"no observed dispatch state for {dispatch_name!r}")
    executed_manifest = json.loads(Path(state["manifest"]).read_text(encoding="utf-8"))
    receipt_path = Path(executed_manifest["output_dir"]) / "receipt.json"
    if not receipt_path.is_file():
        raise ManifestRefused(f"missing canary receipt: {receipt_path}")
    return receipt_path


def _verify_canary_receipt(
    canary_receipt: Path,
    *,
    commit: str,
    preflight_record: dict[str, Any],
) -> dict[str, Any]:
    if not canary_receipt.is_file():
        raise ManifestRefused(f"missing canary receipt: {canary_receipt}")
    body = json.loads(canary_receipt.read_text(encoding="utf-8"))
    if (
        body.get("status") != "PASS"
        or body.get("stop_verified") is not True
        or body.get("error")
        or body.get("cleanup_error")
        or body.get("failure_class") is not None
    ):
        raise ManifestRefused(f"canary receipt did not PASS cleanly: {canary_receipt}")
    saved_manifest_path = canary_receipt.parent / "manifest.json"
    if not saved_manifest_path.is_file() or sha256_file(saved_manifest_path) != body.get(
        "manifest_sha256"
    ):
        raise ManifestRefused("canary receipt manifest_sha256 binding mismatch")
    saved_manifest = json.loads(saved_manifest_path.read_text(encoding="utf-8"))
    if saved_manifest.get("commit") != commit or saved_manifest.get("script") != SCRIPT:
        raise ManifestRefused("canary receipt commit/script differs from target main job")
    if saved_manifest.get("mac_preflight") != preflight_record:
        raise ManifestRefused("canary receipt mac_preflight differs from packet preflight")
    exports_dir = canary_receipt.parent / "exports"
    canary_summary_path = exports_dir / "summary.json"
    if not canary_summary_path.is_file():
        raise ManifestRefused(f"missing exported canary summary: {canary_summary_path}")
    canary_summary = json.loads(canary_summary_path.read_text(encoding="utf-8"))
    if (
        canary_summary.get("status") != "PASS"
        or int(canary_summary.get("unexplained_mismatches", -1)) != 0
        or canary_summary.get("completed_in_view_unit_ids") != list(allocation_for_lane("canary"))
    ):
        raise ManifestRefused("exported canary summary did not PASS with 0 unexplained mismatches")
    return {
        "path": str(canary_receipt.resolve()),
        "sha256": sha256_file(canary_receipt),
        "canary_world_ticks": int(canary_summary["world_ticks"]),
        "canary_snake_ticks": int(canary_summary["snake_ticks"]),
        "canary_tie_explained_decisions": int(canary_summary["tie_explained_decisions"]),
        "canary_elapsed_seconds": body.get("elapsed_seconds"),
    }


def build_manifest(
    *,
    repo: Path,
    commit: str,
    lane: str,
    output_dir: Path,
    main_ledger: Path,
    canary_ledger: Path,
    preflight_record: dict[str, Any],
    attempt_tag: str = "01",
    canary_receipt: Path | None = None,
    template: bool = False,
) -> dict[str, Any]:
    if lane not in ("canary", "main"):
        raise ManifestRefused(f"manifest lane must be canary or main, got {lane!r}")
    is_canary = lane == "canary"
    if template and (is_canary or canary_receipt is not None):
        raise ManifestRefused("only an unsealed main manifest may set template=True")
    if not is_canary and not template and canary_receipt is None:
        raise ManifestRefused("sealed main manifest requires a verified canary receipt")

    transport_tree = _verify_commit_and_transport(repo, commit)
    load_verified_ledger(
        main_ledger,
        expected_namespace=namespace_for(commit, "main"),
        expected_lane="main",
    )
    load_verified_ledger(
        canary_ledger,
        expected_namespace=namespace_for(commit, "canary"),
        expected_lane="canary",
    )

    eff = CANARY_EFFECTIVE if is_canary else MAIN_EFFECTIVE
    ledger_dest = CANARY_LEDGER_DEST if is_canary else MAIN_LEDGER_DEST
    binding = (
        None
        if is_canary or template
        else _verify_canary_receipt(
            canary_receipt,  # type: ignore[arg-type]
            commit=commit,
            preflight_record=preflight_record,
        )
    )

    manifest: dict[str, Any] = {
        "job_id": f"devicesim-v2-parity-{lane}-{commit[:12]}-{attempt_tag}",
        "commit": commit,
        "script": SCRIPT,
        "device": "cuda",
        "claim": (
            "Exploration engineering parity check only (DeviceSim + CTDeviceSim vs CPU "
            "GridBatchSim Watch6 mechanics v2); no release, serving, or held-out claim."
        ),
        "env": {
            "COLAB_CLI_REQUIREMENTS": REQUIREMENTS_REL,
            "PYTHONUNBUFFERED": "1",
            "DEVICESIM_V2_PARITY_COMMIT": commit,
            "DEVICESIM_V2_PARITY_LANE": lane,
            "DEVICESIM_V2_PARITY_DEVICE": "cuda",
            "DEVICESIM_V2_PARITY_LEDGER_PATH": ledger_dest,
            "GPU_SIM_VECTOR_FLOOD": "static",
            "GPU_SIM_VECTOR_GEOMETRY": "native",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        },
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 8,
            "est_minutes": eff["est_minutes"],
        },
        "timeout_minutes": eff["timeout_minutes"],
        "output_dir": str(output_dir.resolve()),
        "inputs": (
            checkpoint_inputs()
            + [
                _ledger_input(main_ledger, MAIN_LEDGER_DEST),
                _ledger_input(canary_ledger, CANARY_LEDGER_DEST),
            ]
        ),
        "budget": {
            "unit_cap": eff["unit_cap"],
            "rate_units_per_hour": A100_RATE_UNITS_PER_HOUR,
            "reserve_units": eff["reserve_units"],
            "other_assignments": {},
        },
        "lease_seconds": 600,
        "max_export_bytes": 32 * 1024**2,
        "required_exports": list(REQUIRED_EXPORTS),
        "resume": {
            "unit_kind": "shard",
            "ledger_path": ledger_dest,
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "source_paths": runtime_source_paths(repo),
        "transport_tree": transport_tree,
        "mac_preflight": preflight_record,
        "envelope_policy": {
            "owner_2x_rule": True,
            "canary_base": CANARY_BASE,
            "canary_effective": CANARY_EFFECTIVE,
            "main_base": MAIN_BASE,
            "main_effective": MAIN_EFFECTIVE,
            "allocation": list(allocation_for_lane(lane)),
        },
        "canary": (
            {
                "schema": "colab-family-canary/v1",
                "mode": "probe",
                "family": SCRIPT,
            }
            if is_canary
            else (
                {"receipt": binding}
                if binding is not None
                else {"required": "same-pin A100 family canary receipt"}
            )
        ),
        "scientific_admission": not template,
    }
    if is_canary:
        manifest["env"]["SNAKE_COLAB_CANARY"] = "1"
        manifest["env"]["SNAKE_COLAB_CANARY_PACKAGE"] = PACKAGE
    if template:
        manifest["admission_reason"] = (
            "TEMPLATE_ONLY: run prepare.py seal-main after canary PASS to bind canary receipt"
        )

    manifest = convert_public_checkpoints(manifest, repo=repo)
    qualify(manifest, repo=repo)
    if not template:
        colab_canary.check(manifest, repo=repo)
    return manifest


def prepare_packet(
    *,
    repo: Path,
    packet_dir: Path,
    commit: str,
    preflight_summary: Path,
    attempt_tag: str = "01",
) -> dict[str, str]:
    packet_dir = packet_dir.resolve()
    packet_dir.mkdir(parents=True, exist_ok=True)
    preflight_record = _verify_preflight_summary(preflight_summary, commit=commit)

    main_ledger_path = packet_dir / "inputs" / "verified-ledger.json"
    canary_ledger_path = packet_dir / "canary-inputs" / "verified-ledger.json"
    if not main_ledger_path.exists():
        main_empty = empty_ledger(
            namespace=namespace_for(commit, "main"),
            lane="main",
        )
        write_bytes_create_only(
            main_ledger_path,
            (json.dumps(main_empty, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    if not canary_ledger_path.exists():
        canary_empty = empty_ledger(
            namespace=namespace_for(commit, "canary"),
            lane="canary",
        )
        write_bytes_create_only(
            canary_ledger_path,
            (json.dumps(canary_empty, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )

    canary_output_dir = packet_dir / f"canary-run-{attempt_tag}"
    main_output_dir = packet_dir / f"main-run-{attempt_tag}"

    canary_manifest = build_manifest(
        repo=repo,
        commit=commit,
        lane="canary",
        output_dir=canary_output_dir,
        main_ledger=main_ledger_path,
        canary_ledger=canary_ledger_path,
        preflight_record=preflight_record,
        attempt_tag=attempt_tag,
    )
    main_template = build_manifest(
        repo=repo,
        commit=commit,
        lane="main",
        output_dir=main_output_dir,
        main_ledger=main_ledger_path,
        canary_ledger=canary_ledger_path,
        preflight_record=preflight_record,
        attempt_tag=attempt_tag,
        template=True,
    )

    canary_manifest_path = packet_dir / "canary.json"
    main_template_path = packet_dir / "main-template.json"
    write_bytes_create_only(
        canary_manifest_path,
        (json.dumps(canary_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    write_bytes_create_only(
        main_template_path,
        (json.dumps(main_template, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    qualify(
        canary_manifest,
        receipt_path=packet_dir / "canary-upload-qualification.json",
        repo=repo,
    )
    qualify(
        main_template,
        receipt_path=packet_dir / "main-template-upload-qualification.json",
        repo=repo,
    )
    return {
        "canary_manifest": str(canary_manifest_path),
        "main_template": str(main_template_path),
        "canary_output_dir": str(canary_output_dir),
        "main_output_dir": str(main_output_dir),
    }


def seal_main_manifest(
    *,
    repo: Path,
    packet_dir: Path,
    canary_receipt: Path | None = None,
    ops_dir: Path | None = None,
    canary_queue_id: str | None = None,
) -> Path:
    packet_dir = packet_dir.resolve()
    template_path = packet_dir / "main-template.json"
    if not template_path.is_file():
        raise ManifestRefused(f"missing main-template.json in {packet_dir}")
    template_doc = json.loads(template_path.read_text(encoding="utf-8"))

    if canary_receipt is None:
        if ops_dir is None or not canary_queue_id:
            raise ManifestRefused(
                "seal-main requires either --canary-receipt or (--ops-dir and --canary-queue-id)"
            )
        canary_receipt = resolve_canary_receipt_from_ops(ops_dir.resolve(), canary_queue_id)

    commit = str(template_doc["commit"])
    preflight_record = dict(template_doc["mac_preflight"])
    output_dir = Path(str(template_doc["output_dir"]))
    job_id = str(template_doc["job_id"])
    attempt_tag = job_id.rsplit("-", 1)[-1]
    main_ledger_path = packet_dir / "inputs" / "verified-ledger.json"
    canary_ledger_path = packet_dir / "canary-inputs" / "verified-ledger.json"

    sealed = build_manifest(
        repo=repo,
        commit=commit,
        lane="main",
        output_dir=output_dir,
        main_ledger=main_ledger_path,
        canary_ledger=canary_ledger_path,
        preflight_record=preflight_record,
        attempt_tag=attempt_tag,
        canary_receipt=canary_receipt.resolve(),
        template=False,
    )
    main_path = packet_dir / "main.json"
    if main_path.exists():
        existing = json.loads(main_path.read_text(encoding="utf-8"))
        if existing != sealed:
            raise ManifestRefused(f"existing {main_path} differs from newly sealed manifest")
        return main_path

    write_bytes_create_only(
        main_path,
        (json.dumps(sealed, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    qualify(
        sealed,
        receipt_path=packet_dir / "main-upload-qualification.json",
        repo=repo,
    )
    return main_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Create canary.json and main-template.json packet.")
    p_prep.add_argument("--repo", type=Path, default=ROOT)
    p_prep.add_argument("--commit", required=True)
    p_prep.add_argument("--preflight-summary", type=Path, required=True)
    p_prep.add_argument("--packet-dir", type=Path, required=True)
    p_prep.add_argument("--attempt-tag", default="01")

    p_seal = sub.add_parser("seal-main", help="Bind verified canary receipt and write main.json.")
    p_seal.add_argument("--repo", type=Path, default=ROOT)
    p_seal.add_argument("--packet-dir", type=Path, required=True)
    p_seal.add_argument("--canary-receipt", type=Path)
    p_seal.add_argument("--ops-dir", type=Path)
    p_seal.add_argument("--canary-queue-id")

    args = parser.parse_args()
    if args.cmd == "prepare":
        out = prepare_packet(
            repo=args.repo.resolve(),
            packet_dir=args.packet_dir,
            commit=args.commit,
            preflight_summary=args.preflight_summary,
            attempt_tag=args.attempt_tag,
        )
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    sealed_path = seal_main_manifest(
        repo=args.repo.resolve(),
        packet_dir=args.packet_dir,
        canary_receipt=args.canary_receipt,
        ops_dir=args.ops_dir,
        canary_queue_id=args.canary_queue_id,
    )
    print(json.dumps({"main_manifest": str(sealed_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
