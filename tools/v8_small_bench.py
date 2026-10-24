"""V8 sibling entrypoint under ADOPT_SMALL_BENCH_FAST_PATH; no runtime changes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tools import small_bench_operator as operator
from tools import small_bench_register as register
from tools import v8_budget_main_amend as accepted

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
EXP = BASE / "snake-dqn-artifacts/big-lever/exp"
ART = accepted.ART
SPRINT = EXP / "small-bench-sprint-20261024"
CONFIG = SPRINT / "v8-config-release-tail-02.json"
RULING = EXP / "JUDGMENT_SMALL_BENCH_ADMISSION_20261024.md"
OPS = BASE / "snake-dqn-artifacts/ops"


def offline() -> dict[str, Any]:
    """Reuse exact offline acceptance, source/native canary and upload checks."""
    value = accepted.verify_preparation()
    helper = accepted.load(accepted.HOST / "tools/ops/repack_gpu_upload.py", "small_bench_upload")
    with tempfile.TemporaryDirectory(prefix="small-bench-upload-") as directory:
        upload = helper.check_upload(value, accepted.BROKER, Path(directory) / "upload")
    subprocess.run(
        [
            str(accepted.HOST / "tools/ops/dispatch.sh"),
            accepted.JOB,
            str(ART / "v8-split-main.json"),
            "--broker",
            str(accepted.BROKER),
            "--cwd",
            str(accepted.BROKER),
            "--state-dir",
            str(ART / "state-main"),
            "--dispatch-txt",
            str(ART / "DISPATCH.txt"),
            "--dry-run",
        ],
        cwd=accepted.HOST,
        env=dict(os.environ, PYTHONPATH=str(accepted.HOST)),
        check=True,
    )
    return {
        "status": "PASS_OFFLINE_REUSED_ACCEPTANCE",
        "allocations": 0,
        "source_files": upload["source_files"],
        "source_archive_bytes": upload["source_archive_bytes"],
    }


def configuration() -> dict[str, Any]:
    """Freeze a reusable local package descriptor; old accepted bytes stay intact."""
    paths = [
        RULING,
        ART / "v8-split-main.json",
        ART / "PREPARATION.json",
        ART / "CODEX_V_LIGHT.json",
        ART / "CODEX_V_ADMISSION_REVIEW.json",
    ]
    sources = [operator.binding(path) for path in paths]
    manifest = json.loads((ART / "v8-split-main.json").read_text())
    for row in manifest["inputs"]:
        sources.append(operator.binding(Path(row["path"])))
    command = [
        str(ROOT / "tools/ops/dispatch.sh"),
        accepted.JOB,
        str(ART / "v8-split-main.json"),
        "--broker",
        str(accepted.BROKER),
        "--cwd",
        str(accepted.BROKER),
        "--state-dir",
        str(ART / "state-main"),
        "--dispatch-txt",
        str(ART / "DISPATCH.txt"),
        "--small-bench-config",
        str(CONFIG),
    ]
    return {
        "schema": "small-bench-package/v1",
        "job_id": accepted.JOB,
        "packet_id": "V8_SPLIT_INITIAL_MAIN_20261024",
        "purpose": "timing",
        "ruling": operator.binding(RULING),
        "bindings": sources,
        "manifest": operator.binding(ART / "v8-split-main.json"),
        "offline_acceptance": operator.binding(ART / "CODEX_V_LIGHT.json"),
        "cap": {"units": 20, "wall_minutes": 120},
        "outside_tail": {"units": 5, "wall_minutes": 30},
        "outside_tail_basis": (
            "Known-endpoint cleanup: 1173s stop verification + 600s broker lock "
            "+ 2s poll - 45s main lead <=30min at10u/h. This alone forces packet HOLD; "
            "unknown allocation-identity recovery is not claimed universally bounded by this tail."
        ),
        "outside_tail_source": operator.binding(
            accepted.BROKER / "research/colab_cli/supervisor.py"
        ),
        "packet_ceiling": {"units": 25, "wall_minutes": 150},
        "scopes": ["packet", "v8", "q2", "q4", "global"],
        "deadline_utc": "2026-10-26T01:00:00+00:00",
        "allocation_wait_minutes": 10,
        "reservation_id": "small-bench-v8-split-initial-669405e2-budget02",
        "register": str(SPRINT / "register.jsonl"),
        "reservation_lock": str(OPS / "sprint-reservations.lock"),
        "adapter_review": str(SPRINT / "CODEX_V_ADAPTER_REVIEW.json"),
        "admissions_dir": str(ART / "small-bench-admissions"),
        "admission_pointer": str(ART / "OPERATOR_ADMISSION.current.json"),
        "campaign_log": str(ROOT / "docs/research/final_run_20261019/CAMPAIGN_LOG.md"),
        "account_root": "/Users/josenunez/.config/colab-cli/snake",
        "colab": "/Users/josenunez/.local/bin/colab",
        "previous_releases": [
            operator.binding(accepted.previous().SOURCE / "canary-output/receipt.json")
        ],
        "watch_globs": [
            {"root": "/Users/josenunez/.config/colab-cli/snake", "pattern": "active-*.json"},
            {"root": "/Users/josenunez/.config/colab-cli/snake", "pattern": "pending-*.json"},
            {"root": str(OPS / "manifests"), "pattern": "*.json"},
        ],
        "cwd": str(accepted.BROKER),
        "offline_check": [str(accepted.PY), "-m", "tools.v8_small_bench", "offline"],
        "dispatch_command": command,
        "dispatch_cwd": str(ROOT),
        "dispatch_pythonpath": str(ROOT),
    }


def bootstrap() -> dict[str, Any]:
    """Carry bounded operational records once, without a historical inventory rebuild."""
    v8_source = ART / "V8_CONSERVATIVE_CHARGES.json"
    v8 = json.loads(v8_source.read_text())["rows"]
    v8_jobs = {row["job_id"] for row in v8}
    charges, seen, manifests = [], set(), []
    for path in sorted((OPS / "manifests").glob("*.json")):
        if ".live." in path.name:
            continue
        value = json.loads(path.read_text())
        job = value.get("job_id", "")
        if "big-lever/exp/" not in value.get("output_dir", "") or not any(
            word in job for word in ("gpu-", "sparse-bench", "native-compile", "a1-prof")
        ):
            continue
        if job in seen:
            raise ValueError("duplicate historical manifest job identity")
        seen.add(job)
        manifests.append(operator.binding(path))
        if job in v8_jobs:
            continue
        receipt = Path(value["output_dir"]) / "receipt.json"
        sources = [operator.binding(path)]
        pair = {"units": value["budget"]["unit_cap"], "wall_minutes": value["timeout_minutes"]}
        basis = "FULL_ADMITTED_CAP_UNSETTLED"
        if receipt.exists():
            terminal = json.loads(receipt.read_text())
            sources.append(operator.binding(receipt))
            if terminal.get("job_id") != job:
                raise ValueError("historical manifest/receipt identity mismatch")
            if terminal.get("stop_verified") is True:
                minutes = terminal["elapsed_seconds"] / 60
                pair = {
                    "units": minutes * value["budget"]["rate_units_per_hour"] / 60,
                    "wall_minutes": minutes,
                }
                basis = "RELEASE_VERIFIED_ELAPSED_AT_CEILING"
        scopes = ["global"]
        if "a1-prof" not in job:
            # Conservative overlap attribution in both parents, global once.
            scopes += ["q2", "q4"]
        charges.append(
            {"id": job, "pair": pair, "scopes": scopes, "sources": sources, "basis": basis}
        )
    for row in v8:
        scopes = ["global", "q2", "q4", "v8"] + (["packet"] if row["split_packet"] else [])
        charges.append(
            {
                "id": row["job_id"],
                "pair": row["charged"],
                "scopes": scopes,
                "sources": [{"path": e["path"], "sha256": e["sha256"]} for e in row["evidence"]],
                "basis": "INHERITED_ASTRA_CONSERVATIVE_BOUND",
            }
        )
    ceilings = {
        "global": (1100, 7200),
        "q2": (144, 864),
        "q4": (192, 1152),
        "v8": (36, 216),
        "packet": (25, 150),
    }
    scopes = {}
    for name, (units, minutes) in ceilings.items():
        scopes[name] = {
            "ceiling": {"units": units, "wall_minutes": minutes},
            "spent": {
                key: sum(c["pair"][key] for c in charges if name in c["scopes"])
                for key in ("units", "wall_minutes")
            },
        }
    holds = []
    for name, units, minutes in (
        ("I0", 24, 144),
        ("Q3", 96, 576),
        ("Q5", 240, 1440),
        ("Q6", 384, 2304),
        ("O1", 48, 288),
    ):
        holds.append(
            {
                "id": name,
                "scopes": ["global"] + (["q4"] if name == "I0" else []),
                "pair": {"units": units, "wall_minutes": minutes},
                "account_units": units,
                "kind": "protected_required_work",
            }
        )
    active = []
    for path in Path("/Users/josenunez/.config/colab-cli/snake").glob("active-*.json"):
        record = json.loads(path.read_text())
        active.append(
            {
                "source": operator.binding(path),
                "endpoint": record["endpoint"],
                "full_cap": record["unit_cap"],
            }
        )
    return {
        "type": "init",
        "id": "small-bench-sprint-20261024-initial",
        "scopes": scopes,
        "holds": holds,
        "external_commitments_units": sum(row["full_cap"] for row in active),
        "external_assignment_caps": active,
        # Do not assume past provider debit has settled: hold all conservative spend once.
        "unsettled_debit_holds_units": scopes["global"]["spent"]["units"],
        "completeness_attestation": (
            "All admitted sprint operational manifests at initialization plus "
            "the six inherited V8 charges are carried by unique attempt; "
            "prepared unexecuted packages "
            "are not expenditure. Full required I0/Q3/Q5/Q6/O1 holds retained; "
            "conditional R/Q7 not triggered. All historical debit held pending batch settlement. "
            "New or unknown identities require a delta."
        ),
        "sources": [operator.binding(RULING), operator.binding(v8_source)],
        "historical_charges": charges,
        "covered_manifests": manifests,
        "operational_baseline": operator.fingerprint(configuration()),
        "t0_utc": "2026-10-24T01:00:00+00:00",
        "lower_authenticated_limit": None,
        "note": "1100 is inherited ceiling, not fresh balance; no original seal claimed recovered",
    }


def prepare() -> dict[str, Any]:
    offline()
    SPRINT.mkdir(exist_ok=True)
    with register.reservation_lock(OPS / "sprint-reservations.lock"):
        path = SPRINT / "register.jsonl"
        if path.exists():
            raise ValueError("standing register already initialized; append deltas only")
        row = register.append(path, bootstrap())
        operator.publish_json(CONFIG, configuration())
        operator.publish_json(
            SPRINT / "BATCH_AUDIT.json",
            {
                "status": "PENDING_FIRST_ADMISSION",
                "register": str(path),
                "initial_revision": row["revision"],
                "initial_row_hash": row["row_hash"],
                "reviewer": "Codex-V independent financial batch audit",
                "due": (
                    "24h from first unaudited admission; "
                    "before cumulative gross full reservations exceed 100u"
                ),
                "required": [
                    "source provenance",
                    "historical completeness",
                    "nested accounting",
                    "settlement",
                    "duplicates",
                    "explicit prefix revision and hash",
                ],
                "reset_policy": (
                    "Only completed independent ACCEPT of explicit prefix closes batch; "
                    "no day/package reset"
                ),
                "no_audit_yet": True,
            },
        )
    return {"status": "PREPARED_NOT_LIVE_ADMITTED", "config": str(CONFIG), "row": row}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "offline", "admit", "dispatch", "check"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.operation != "dispatch":
        parser.error("--dry-run is supported only for dispatch; no live action performed")
    if args.operation == "prepare":
        result = prepare()
    elif args.operation == "offline":
        result = offline()
    elif args.operation == "admit":
        result = operator.admit(CONFIG)
    elif args.operation == "check":
        result = operator.check_admission(CONFIG)
    else:
        result = operator.dispatch(CONFIG, args.dry_run)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
