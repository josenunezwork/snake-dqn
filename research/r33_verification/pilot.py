"""Prepare only: exact published pin, qualified transport, immutable input package."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from research.r33_verification.contract import (
    ALLOCATION,
    ATTEMPTS,
    INTERIM_RULING_SHA256,
    NAMESPACE,
    RULING_SHA256,
    SOURCE_PATHS,
    environment_identity,
)
from research.r33_verification.preflight import source_closure, source_inventory
from research.r33_verification.verify import completion_admission
from tools.ops.import_preflight import check as import_preflight
from tools.ops.package_spec import (
    Input,
    PackageRefused,
    build_cpu_manifest,
    digest,
    empty_ledger,
    publish_json,
    read_ledger,
    require_published_commit,
    stage_inputs,
    verify_immutable,
)
from tools.ops.proof_dependencies import dependency_inventory

ROOT = Path(__file__).resolve().parents[2]


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_evidence(
    preflight: Path, independent: Path, commit: str, identity: dict[str, str], closure: str
) -> None:
    for path in (preflight, independent):
        verify_immutable(path)
    first, second = json.loads(preflight.read_text()), json.loads(independent.read_text())
    if (
        first.get("schema") != "r33-mac-preflight/v1"
        or first.get("status") != "PASS"
        or first.get("identity") != identity
        or first.get("units") != 3
        or first.get("scrubbed") is not True
        or first.get("isolation") != "seatbelt-no-users-volumes-network"
        or first.get("admission_policy") != "ALLOW_NICE15_INTERIM_STDLIB_SERIAL"
        or first.get("interim_ruling_sha256") != INTERIM_RULING_SHA256
        or first.get("excluded") is not True
        or first.get("allocation_attempted") is not False
        or first.get("effective_nice", 0) < 15
        or first.get("platform") != "Darwin"
        or first.get("entrypoint") != ["/bin/bash", "research/r33_verification/colab.sh"]
    ):
        raise PackageRefused("missing or stale tiny real-entrypoint preflight")
    if (
        second.get("status") != "PASS_IMPLEMENTATION_LIGHT_VERIFICATION"
        or second.get("schema") != "r33-independent-light-verification/v1"
        or second.get("reviewer") != "Codex-V"
        or not isinstance(second.get("evidence"), list)
        or not second["evidence"]
        or any(
            not isinstance(row, dict)
            or not row.get("command")
            or row.get("status") != "PASS"
            or not row.get("evidence_sha256")
            for row in second["evidence"]
        )
    ):
        raise PackageRefused("independent Codex-V light acceptance required")
    for record in (first, second):
        if record.get("commit") != commit or record.get("source_closure_sha256") != closure:
            raise PackageRefused("foreign pin or stale proof closure")


def prepare(
    root: Path,
    commit: str,
    preflight: Path,
    independent_receipt: Path,
    *,
    previous_manifest: Path | None = None,
    verified_ledger: Path | None = None,
    prior_receipts: list[Path] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if root.exists():
        raise FileExistsError(root)
    commit = require_published_commit(ROOT, commit)
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *SOURCE_PATHS],
        cwd=ROOT,
        text=True,
    )
    delta = subprocess.run(["git", "diff", "--quiet", commit, "--", *SOURCE_PATHS], cwd=ROOT)
    if changed or delta.returncode:
        raise PackageRefused("package closure must exactly match the published pin")
    closure = source_closure(commit)
    identity = {
        "pin": commit,
        "environment_sha256": environment_identity(ROOT),
        "packet": digest(ROOT / "research/r33_verification/contract_inventory.json"),
        "source_closure_sha256": closure,
    }
    validate_evidence(preflight, independent_receipt, commit, identity, closure)
    attempt = 0
    if previous_manifest is not None:
        if verified_ledger is None or not prior_receipts:
            raise PackageRefused("completion requires verified ledger and prior stop receipts")
        previous = json.loads(previous_manifest.read_text())
        if previous.get("r33", {}).get("identity") != identity:
            raise PackageRefused("foreign completion identity")
        attempt = len(prior_receipts)
        seconds = sum(
            json.loads(path.read_text()).get("elapsed_seconds", float("nan"))
            for path in prior_receipts
        )
        completion_admission(
            prior_receipts,
            verified_ledger,
            attempts_used=attempt,
            series_minutes=seconds / 60,
            series_units=seconds * 0.08 / 3600,
            manifest=previous,
        )
    elif verified_ledger is not None or prior_receipts:
        raise PackageRefused("completion metadata requires original manifest")
    imports = import_preflight(
        ROOT,
        commit,
        ["research/r33_verification", "tools/ops/package_spec.py"],
        "research/r33_verification/requirements.txt",
        ["research.r33_verification.pilot_runtime"],
    )
    root.mkdir(parents=True)
    ledger = root / "inputs-original/verified-ledger.json"
    if verified_ledger is None:
        publish_json(ledger, empty_ledger(NAMESPACE, ALLOCATION, identity))
    else:
        read_ledger(verified_ledger, NAMESPACE, ALLOCATION, identity)
        # Immutable re-publication preserves the exact prior Mac ledger bytes.
        from tools.ops.package_spec import publish_immutable

        publish_immutable(ledger, verified_ledger.read_bytes())
    authority = root / "authority.json"
    publish_json(
        authority,
        {
            "schema": "r33-pilot-authority/v1",
            "ruling_sha256": RULING_SHA256,
            "excluded": True,
            "preflight_sha256": digest(preflight),
            "independent_receipt_sha256": digest(independent_receipt),
        },
    )
    publish_json(root / "clean-import-preflight.json", imports)
    binding = root / "source-inventory.json"
    publish_json(
        binding,
        {"schema": "r33-source-inventory/v1", "commit": commit, "files": source_inventory(commit)},
    )
    staged = stage_inputs(
        root / "staging",
        [
            Input(ledger, "verified-ledger.json"),
            Input(binding, "source-inventory.json"),
            Input(authority, "authority.json"),
            Input(preflight, "preflight.json"),
            Input(independent_receipt, "codex-v.json"),
        ],
    )
    manifest = build_cpu_manifest(
        repo=ROOT,
        commit=commit,
        output_dir=root / "remote-output",
        staged_inputs=staged,
        script="research/r33_verification/colab.sh",
        namespace=NAMESPACE,
        allocation=ALLOCATION,
        authority_receipts=[{"path": str(authority), "sha256": digest(authority)}],
        job_id=f"r33-cpu-pilot-{attempt + 1:02d}",
    )
    envelope = ATTEMPTS[attempt]
    manifest.update(timeout_minutes=envelope["effective_minutes"])
    manifest["requirements"]["est_minutes"] = envelope["estimate"]
    manifest["budget"].update(
        unit_cap=envelope["unit_cap"], reserve_units=envelope["reserve_units"]
    )
    manifest["env"].update(
        R33_PIN=commit,
        R33_ENVIRONMENT_SHA256=identity["environment_sha256"],
        R33_PACKET=identity["packet"],
        R33_SOURCE_CLOSURE=closure,
        SNAKE_DQN_DEVICE="cpu",
        **{
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "OMP_THREAD_LIMIT",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    )
    manifest["r33"].update(
        identity=identity,
        source_closure_sha256=closure,
        attempt=attempt,
        prior_receipt_sha256=[digest(path) for path in (prior_receipts or [])],
        attempt_envelopes=ATTEMPTS,
        no_cap_redistribution=True,
        mechanical_completion={
            "allowed_child_exit_codes": [-15, -9, 124, 137, 143],
            "max_completions": 2,
            "requires": [
                "verified-interruption",
                "verified-stop",
                "missing-shards",
                "same-pin",
                "remaining-frozen-liability",
            ],
            "rpc_or_cleanup_uncertainty": "HOLD",
            "test_failure": "CLOSE_PACKET",
        },
        native_pending=json.loads(
            (ROOT / "research/r33_verification/contract_inventory.json").read_text()
        )["native_pending"],
    )
    dependencies = [
        {"name": "source-closure", "kind": "source", "sha256": closure},
        {"name": "environment", "kind": "environment", "sha256": identity["environment_sha256"]},
        {"name": "platform", "kind": "platform", "sha256": _hash("Linux CPU development only")},
        {"name": "command", "kind": "command", "sha256": _hash(["bash", manifest["script"]])},
        {"name": "fixture-packet", "kind": "fixture", "sha256": identity["packet"]},
        {
            "name": "allocation",
            "kind": "allocation",
            "sha256": _hash([NAMESPACE, ALLOCATION, [0, 1, 2]]),
        },
        {"name": "contract", "kind": "contract", "sha256": _hash([RULING_SHA256, ATTEMPTS])},
        *[
            {"name": "input:" + row["dest"], "kind": "input", "sha256": row["sha256"]}
            for row in staged
        ],
    ]
    publish_json(
        root / "proof-dependencies.json", dependency_inventory(manifest["job_id"], dependencies)
    )
    publish_json(root / "manifest.json", manifest)
    (root / "DISPATCH.txt").write_text(
        "Claude/operator: refresh current qualification, account assignments, "
        "liabilities, spare CPU capacity and rate under the existing allocation lock "
        "before dispatch. "
        "A100/P1/F4 reservations retain priority. No automatic completion or allocation "
        "beyond existing exact accounting retry.\n"
    )
    receipt = {
        "schema": "r33-pilot-preparation/v1",
        "status": "PREPARED_NOT_DISPATCHED",
        "commit": commit,
        "manifest_sha256": digest(root / "manifest.json"),
        "identity": identity,
        "attempt": attempt,
        "forecast_minutes": 120 if attempt == 0 else 12,
        "series_effective_minutes": 288,
        "series_effective_units": 0.384,
    }
    publish_json(root / "preparation.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--independent-receipt", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path)
    parser.add_argument("--verified-ledger", type=Path)
    parser.add_argument("--prior-receipt", type=Path, action="append")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.root,
                args.commit,
                args.preflight,
                args.independent_receipt,
                previous_manifest=args.previous_manifest,
                verified_ledger=args.verified_ledger,
                prior_receipts=args.prior_receipt,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
