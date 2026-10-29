"""Create-only timing-03 mechanical successor; no allocation interface.

Run prepare, review the concrete files, then seal and publish. The original
package and output-01 refusal remain immutable. Publication is a delta over the
already published timing-02 package, with every inherited object hash checked.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import export_bytes as create_export
from research.bl_ppo.exports import verify_export
from research.colab_cli.accounting import budget_warning
from tools.ops.f5_unit_guard import GUARD_DESCRIPTION, repair_guard, require_admissible

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent / "snake-dqn-artifacts/big-lever/exp/astra-20261024"
PACKAGE = BASE / "F5_W_TIMING_OPERATIONAL_20261026_ROUND08_02_REQUAL"
EVIDENCE = PACKAGE / "guard-successor-03"
BROKER = ROOT.parent / "snake-dqn-f5-timing02-requal"
OPS = ROOT.parent / "snake-dqn-artifacts/ops"
JOB = "bl-f5-w-timing-r08-sr75b14427-03"
DEST = ROOT / "research/bl_residual/ledger/F5_W_TIMING_GUARD_SUCCESSOR_20261026_03"
PYTHON = ROOT.parent / "snake-dqn/venv/bin/python"
REGISTRY = OPS / "colab-cli-qualified.json"


def export_bytes(path: Path, payload: bytes) -> dict:
    """Resume preparation only across exact, already complete immutable exports."""
    if path.exists():
        assert verify_export(path) == hashlib.sha256(payload).hexdigest(), path
        return binding(path)
    return create_export(path, payload)


def export_json(path: Path, value: Any) -> dict:
    return export_bytes(path, (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def binding(path: Path) -> dict:
    assert path.is_file() and not path.is_symlink(), path
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "size": path.stat().st_size}


def check(reference: dict) -> None:
    actual = binding(Path(reference["path"]))
    assert actual["sha256"] == reference["sha256"], reference["path"]
    if "size" in reference:
        assert actual["size"] == reference["size"], reference["path"]


def run(argv: list, *, cwd: Path = BROKER, timeout: int = 55) -> dict:
    started = now()
    args = [str(value) for value in argv]
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    row = {
        "argv": args,
        "started_utc": started,
        "finished_utc": now(),
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    assert result.returncode == 0, row
    return row


def fresh() -> None:
    assert run(["git", "rev-parse", "HEAD"])["stdout"].strip() == (
        "ac69b65c3a2eb40a2af30ab6a9e8e4bd450471ef"
    )
    assert not run(["git", "status", "--porcelain"])["stdout"]
    for path in (
        PACKAGE / "state-02",
        PACKAGE / "output-02",
        OPS / "dispatch" / JOB,
        OPS / "manifests" / f"{JOB}.json",
        OPS / "logs" / f"{JOB}.log",
    ):
        assert not path.exists(), path


def dispatch_command() -> list[str]:
    return [
        str(BROKER / "tools/ops/dispatch.sh"),
        JOB,
        str(PACKAGE / "manifest-03.json"),
        "--broker",
        str(BROKER),
        "--cwd",
        str(BROKER),
        "--state-dir",
        str(PACKAGE / "state-02"),
        "--dispatch-txt",
        str(PACKAGE / "DISPATCH-03.txt"),
    ]


def account_preflight() -> dict:
    calls = [
        run(["/Users/josenunez/.local/bin/colab", "--auth", "oauth2", operation])
        for operation in ("sessions", "usage", "sessions")
    ]
    for index in (0, 2):
        assert "No active sessions found on server." in calls[index]["stdout"]
    usage = {}
    for key, label in (
        ("balance", "Current balance"),
        ("rate", "Usage rate"),
        ("assignments", "Active assignments"),
    ):
        match = re.search(rf"{label}:\s*([0-9]+(?:\.[0-9]+)?)", calls[1]["stdout"])
        assert match, calls[1]
        usage[key] = float(match[1])
    assert usage["assignments"] == 0
    if usage["rate"] != 0:
        budget_warning("IDLE_ACCOUNT_RATE", rate=usage["rate"])
    account = Path("/Users/josenunez/.config/colab-cli/snake")
    assert not list(account.glob("active-*.json"))
    assert not list(account.glob("pending-*.json"))
    require_admissible(usage["balance"], 0)
    return {
        "schema": "bl-f5-live-account-preflight/v5",
        "status": "PASS",
        "observed_utc": now(),
        "calls": calls,
        "usage": usage,
        "sessions": {},
        "other_assignments": {},
        "committed_unit_caps": 0,
        "required_strict_balance_threshold": 0,
        "legacy_unit_threshold_warning_only": 100,
        "active_broker_records": [],
        "pending_broker_records": [],
        "account_assignment_cap": 3,
        "qualified_registry": binding(REGISTRY),
        "auth": "oauth2",
        "allocation_calls": 0,
    }


def differences(old: Any, new: Any, prefix: str = "") -> list[dict]:
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(old.keys() | new.keys()):
            result.extend(differences(old.get(key), new.get(key), f"{prefix}/{key}"))
        return result
    return [] if old == new else [{"path": prefix, "before": old, "after": new}]


def prepare() -> None:
    fresh()
    # Capture every pre-existing byte before adding anything; read payloads only as opaque hashes.
    baseline_path = EVIDENCE / "predecessor-inventory.json"
    if baseline_path.exists():
        verify_export(baseline_path)
        for reference in load(baseline_path).values():
            check(reference)
    else:
        assert not EVIDENCE.exists()
        baseline = {
            str(path.relative_to(PACKAGE)): binding(path)
            for path in sorted(PACKAGE.rglob("*"))
            if path.is_file()
        }
        export_json(baseline_path, baseline)
    export_bytes(EVIDENCE / "f5_timing_guard_successor-v2.py", Path(__file__).read_bytes())
    export_bytes(EVIDENCE / "f5_unit_guard.py", (ROOT / "tools/ops/f5_unit_guard.py").read_bytes())
    for name in ("ACCEPT_F5_W_TIMING02_REQUAL.md", "OWNER_RULING_COLAB_BUDGET_20261025.txt"):
        export_bytes(EVIDENCE / name, (BASE / name).read_bytes())
    old = load(PACKAGE / "manifest-02.json")
    receipt = load(PACKAGE / "output-01/receipt.json")
    assert receipt["endpoint"] is None and receipt["terminal"] is None
    assert receipt["error"] == "Refused('insufficient balance for cap plus study reserve')"
    assert receipt["job_id"] == old["job_id"] and receipt["exports"] == []
    assert [call["command"] for call in receipt["cli_calls"]] == ["version", "sessions", "usage"]
    refused_manifest = load(PACKAGE / "output-01/manifest.json")
    import sys

    # Use the exact accepted broker validator to reconcile normalized receipt bytes.
    sys.path.insert(0, str(BROKER))
    from research.colab_cli.supervisor import validate

    with tempfile.TemporaryDirectory(prefix="f5-refusal-normalization-") as temporary:
        normalized_input = dict(old, output_dir=str(Path(temporary) / "unused-output"))
        normalized = validate(normalized_input, BROKER)
        normalized["output_dir"] = old["output_dir"]
    assert normalized == refused_manifest
    export_json(
        EVIDENCE / "PREPARATION_RECOVERY.json",
        {
            "status": "LOCAL_PREPARATION_ONLY_NO_PROVIDER_CALLS_AT_FAILURE",
            "failure": (
                "Predecessor validator correctly refused already-used output-01. Reconcile "
                "normalization with a fresh temporary output identity, restore output-01 in "
                "memory and compare exact manifest meaning. No predecessor bytes changed."
            ),
            "original_builder": binding(EVIDENCE / Path(__file__).name),
            "successor_builder": binding(EVIDENCE / "f5_timing_guard_successor-v2.py"),
            "allocations": 0,
            "dispatches": 0,
        },
    )
    assert binding(PACKAGE / "output-01/manifest.json")["sha256"] == receipt["manifest_sha256"]
    refusal_files = {}
    for path in sorted((PACKAGE / "output-01").rglob("*")):
        if path.is_file():
            relative = path.relative_to(PACKAGE / "output-01")
            copy = EVIDENCE / "refusal-output-01" / relative
            export_bytes(copy, path.read_bytes())
            refusal_files[str(relative)] = {"source": binding(path), "snapshot": binding(copy)}
    export_json(
        EVIDENCE / "REFUSAL_EVIDENCE.json",
        {
            "status": "REFUSED_BEFORE_ALLOCATION",
            "files": refusal_files,
            "accepted_manifest": binding(PACKAGE / "manifest-02.json"),
            "normalized_manifest_matches_accepted_broker_validate": True,
            "elapsed_seconds": receipt["elapsed_seconds"],
            "endpoint": None,
            "allocation_calls": 0,
            "release_not_applicable_no_endpoint_allocated": True,
        },
    )
    export_json(EVIDENCE / "live-account-preflight-03.json", account_preflight())
    successor = repair_guard(old)
    successor.update(
        job_id=JOB,
        supervisor_state_dir=str(PACKAGE / "state-02"),
        output_dir=str(PACKAGE / "output-02"),
    )
    successor["operational_admission"]["live_account_preflight"] = binding(
        EVIDENCE / "live-account-preflight-03.json"
    )
    export_json(PACKAGE / "manifest-03.json", successor)
    delta = differences(old, successor)
    assert {row["path"] for row in delta} == {
        "/job_id",
        "/supervisor_state_dir",
        "/output_dir",
        "/budget/unit_cap",
        "/f5/technical_guards/base_unit_cap",
        "/f5/technical_guards/effective_unit_cap",
        "/f5/technical_budget",
        "/operational_admission/live_account_preflight/path",
        "/operational_admission/live_account_preflight/size",
        "/operational_admission/live_account_preflight/sha256",
    }
    export_json(EVIDENCE / "MECHANICAL_DIFF.json", {"science_change": False, "changes": delta})
    text = (PACKAGE / "DISPATCH-02.txt").read_text()
    for before, after in (
        (old["job_id"], JOB),
        ("manifest-02.json", "manifest-03.json"),
        ("DISPATCH-02.txt", "DISPATCH-03.txt"),
        ("state-01", "state-02"),
        ("output-01", "output-02"),
        (
            binding(PACKAGE / "manifest-02.json")["sha256"],
            binding(PACKAGE / "manifest-03.json")["sha256"],
        ),
        ("Schema unit placeholders base4999.5/effective9999, reserve0/0.", GUARD_DESCRIPTION),
        (
            str(PACKAGE / "live-account-preflight-02.json"),
            str(EVIDENCE / "live-account-preflight-03.json"),
        ),
        (
            str(PACKAGE / "PACKAGE_SEAL_OPERATIONAL.json"),
            str(PACKAGE / "PACKAGE_SEAL_OPERATIONAL-03.json"),
        ),
    ):
        text = text.replace(before, after)
    text += (
        "\nMechanical guard successor authorized by owner task 2026-10-26: timing-02 was "
        "refused before allocation. Original accepted files and refusal remain immutable. "
        "This is the operative manifest-03/DISPATCH-03 pair after successor publication. "
        "All implementation, inputs, exports, workload, placement, timeout and retry fences "
        "are unchanged. Predecessor seals cover their recorded inventories; this successor "
        "is sealed as an immutable delta. No dispatch occurred during preparation.\n"
    )
    export_bytes(PACKAGE / "DISPATCH-03.txt", text.encode())
    # Every historical F5 builder with this literal receives a new prospective revision.
    patched = []
    ledger = ROOT / "research/bl_residual/ledger"
    for path in sorted(ledger.glob("F5_*/*.py")):
        source = path.read_text()
        if "9999" not in source or not ("unit_cap" in source and "4999.5" in source):
            continue
        revised = source.replace("4999.5", "50.0").replace("9999.0", "100.0").replace("9999", "100")
        relative = path.relative_to(ledger)
        target = EVIDENCE / "prospective-builders" / relative
        export_bytes(target, revised.encode())
        compile(revised, str(target), "exec")
        patched.append(
            {"supersedes_immutable": binding(path), "prospective_revision": binding(target)}
        )
    export_json(
        EVIDENCE / "PROSPECTIVE_F5_BUILDERS.json",
        {
            "status": "PROSPECTIVE_REVISIONS_ONLY_NOT_EXECUTED",
            "revisions": patched,
            "policy": (
                "Use these guard-corrected revisions for future packages; choose fresh paths/job "
                "identities and reseal/review. Historical scripts remain immutable. Generic helper "
                "tools/ops/f5_unit_guard.py repairs only exact non-scientific placeholders."
            ),
            "active_generic_f5_package_builder_has_no_9999_placeholder": "9999"
            not in (ROOT / "research/bl_residual/f5_package.py").read_text(),
        },
    )
    commands = [
        run(
            [
                "python3",
                BROKER / "tools/ops/qualification.py",
                "check",
                "--repo",
                BROKER,
                "--record",
                REGISTRY,
                "--ref",
                ref,
            ]
        )
        for ref in ("HEAD", old["commit"])
    ]
    commands.append(
        run(
            [
                BROKER / "research/colab_cli/launch.sh",
                "--state-dir",
                PACKAGE / "state-02",
                "--colab",
                "/Users/josenunez/.local/bin/colab",
                "validate",
                PACKAGE / "manifest-03.json",
            ]
        )
    )
    export_json(EVIDENCE / "broker-validation-03.json", {"status": "PASS", "commands": commands})
    dry = run(dispatch_command() + ["--dry-run"])
    assert dry["stdout"].strip() == f"PASS_DRY_RUN {JOB} allocations=0 live_accounting=NOT_CHECKED"
    export_json(EVIDENCE / "dispatch-dry-run-03.json", {"status": "PASS", "command": dry})
    fresh()
    export_json(
        PACKAGE / "OPERATIVE_FILES-03.json",
        {
            "status": "MECHANICAL_GUARD_SUCCESSOR_PREPARED_NO_DISPATCH",
            "manifest": binding(PACKAGE / "manifest-03.json"),
            "dispatch": binding(PACKAGE / "DISPATCH-03.txt"),
            "predecessor_publication": binding(PACKAGE / "PACKAGE_PUBLICATION.json"),
            "refusal": binding(EVIDENCE / "REFUSAL_EVIDENCE.json"),
            "mechanical_diff": binding(EVIDENCE / "MECHANICAL_DIFF.json"),
            "account_preflight": binding(EVIDENCE / "live-account-preflight-03.json"),
            "dry_run": binding(EVIDENCE / "dispatch-dry-run-03.json"),
            "dispatch_command": shlex.join(dispatch_command()),
            "allocations": 0,
            "dispatches": 0,
        },
    )
    print(dry["stdout"].strip())


def seal() -> None:
    fresh()
    for reference in load(EVIDENCE / "predecessor-inventory.json").values():
        check(reference)
    old_seal = load(PACKAGE / "PACKAGE_SEAL_OPERATIONAL.json")
    for reference in old_seal["files"].values():
        check(reference)
    export_json(EVIDENCE / "account-recheck-before-seal-03.json", account_preflight())
    files = {}
    paths = list(EVIDENCE.rglob("*"))
    for name in ("manifest-03.json", "DISPATCH-03.txt", "OPERATIVE_FILES-03.json"):
        paths.extend(PACKAGE.glob(name + "*"))
    for path in sorted(paths):
        if path.is_file():
            if not path.name.endswith((".sha256", ".complete")):
                verify_export(path)
            files[str(path.relative_to(PACKAGE))] = binding(path)
    export_json(
        PACKAGE / "PACKAGE_SEAL_OPERATIONAL-03.json",
        {
            "schema": "bl-f5-mechanical-successor-delta-seal/v1",
            "status": "SEALED_MECHANICAL_GUARD_SUCCESSOR_NO_DISPATCH",
            "created_utc": now(),
            "job_id": JOB,
            "inherited_seal": binding(PACKAGE / "PACKAGE_SEAL_OPERATIONAL.json"),
            "inherited_publication": binding(PACKAGE / "PACKAGE_PUBLICATION.json"),
            "operative_files": load(PACKAGE / "OPERATIVE_FILES-03.json"),
            "files": files,
            "state_absent": True,
            "output_absent": True,
            "science_change": False,
            "allocations": 0,
            "dispatches": 0,
            "self_exclusions": [
                "PACKAGE_SEAL_OPERATIONAL-03.json and companions",
                "PACKAGE_PUBLICATION-03.json and companions",
            ],
        },
    )
    print("SEALED", binding(PACKAGE / "PACKAGE_SEAL_OPERATIONAL-03.json"))


def publish() -> None:
    fresh()
    assert not DEST.exists()
    seal_value = load(PACKAGE / "PACKAGE_SEAL_OPERATIONAL-03.json")
    rows = dict(seal_value["files"])
    for suffix in ("", ".sha256", ".complete"):
        path = PACKAGE / ("PACKAGE_SEAL_OPERATIONAL-03.json" + suffix)
        rows[str(path.relative_to(PACKAGE))] = binding(path)
    published = {}
    for relative, reference in rows.items():
        check(reference)
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(Path(reference["path"]).read_bytes())
        published[relative] = binding(target)
        assert published[relative]["sha256"] == reference["sha256"]
    publication = {
        "schema": "bl-f5-mechanical-successor-delta-publication/v1",
        "status": "PUBLISHED_MECHANICAL_SUCCESSOR_REPRESENTATION_NO_DISPATCH",
        "source_package": str(PACKAGE),
        "ledger_directory": str(DEST),
        "inherited_publication": binding(PACKAGE / "PACKAGE_PUBLICATION.json"),
        "seal": binding(PACKAGE / "PACKAGE_SEAL_OPERATIONAL-03.json"),
        "original_files": rows,
        "published_files": published,
        "dispatch_command": shlex.join(dispatch_command()),
        "dispatches": 0,
    }
    export_json(PACKAGE / "PACKAGE_PUBLICATION-03.json", publication)
    for suffix in ("", ".sha256", ".complete"):
        with (DEST / ("PACKAGE_PUBLICATION-03.json" + suffix)).open("xb") as stream:
            stream.write((PACKAGE / ("PACKAGE_PUBLICATION-03.json" + suffix)).read_bytes())
    print("PUBLICATION", binding(PACKAGE / "PACKAGE_PUBLICATION-03.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "seal", "publish"))
    args = parser.parse_args()
    {"prepare": prepare, "seal": seal, "publish": publish}[args.phase]()


if __name__ == "__main__":
    main()
