"""Publish F5 longrun refusal and seal a mechanical successor; never allocate."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import tempfile
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import export_bytes, export_json, verify_export
from tools.ops import f5_timing_guard_successor as common

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent / "snake-dqn-artifacts/big-lever/exp/f5-timing-recovery-offline-20261026"
OLD = BASE / "longrun-01-sealed"
NEW = BASE / "longrun-01-sealed-02"
REFUSAL = BASE / "longrun-01-refusal-evidence"
BROKER = ROOT.parent / "snake-dqn-f5-family-canary-911426db-04.noindex"
ASTRA = BASE.parent / "astra-20261024"
OPS = ROOT.parent / "snake-dqn-artifacts/ops"
LEDGER = ROOT / "research/bl_residual/ledger"
DEST = LEDGER / "F5_LONGRUN01_GUARD_SUCCESSOR_20261026_02"
REFUSAL_DEST = LEDGER / "F5_LONGRUN01_REFUSAL_20261026"
PIN = "911426dbace94ca2925d737d00d5d8ca8456cb6a"
JOB = "bl-f5-timing-recovery-911426db-02"
ACCOUNT = "accounts/live-account-preflight-02.json"
_MISSING = object()


def load(path: Path) -> Any:
    """Read JSON metadata only."""
    return json.loads(path.read_bytes())


def differences(old: Any, new: Any, pointer: str = "") -> list[dict]:
    """Compare every dictionary and list leaf, including added/removed leaves."""
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(old.keys() | new.keys()):
            path = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
            result.extend(differences(old.get(key, _MISSING), new.get(key, _MISSING), path))
        return result
    if old is _MISSING and isinstance(new, dict) and new:
        return differences({}, new, pointer)
    if new is _MISSING and isinstance(old, dict) and old:
        return differences(old, {}, pointer)
    if old is _MISSING:
        row = {"pointer": pointer, "before": None, "after": new}
        if new is None:
            row["before_present"] = False
        return [row]
    if new is _MISSING:
        row = {"pointer": pointer, "before": old, "after": None}
        if old is None:
            row["after_present"] = False
        return [row]
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [
            row
            for index, (before, after) in enumerate(zip(old, new))
            for row in differences(before, after, f"{pointer}/{index}")
        ]
    return [] if old == new else [{"pointer": pointer, "before": old, "after": new}]


def successor(old: dict, account: dict) -> dict:
    """Change only the owner-requested mechanical leaves, without source migration."""
    assert old["budget"]["unit_cap"] == 100000.0
    result = copy.deepcopy(old)
    result["budget"]["unit_cap"] = 100.0
    result["job_id"] = JOB
    result["output_dir"] = str(NEW / "output-02")
    result["supervisor_state_dir"] = str(NEW / "state-02")
    result["operational_admission"]["live_account_preflight"] = account
    allowed = {
        "/budget/unit_cap",
        "/job_id",
        "/output_dir",
        "/supervisor_state_dir",
        *{f"/operational_admission/live_account_preflight/{key}" for key in account},
    }
    assert set(account) == {"path", "sha256", "size"}
    assert {row["pointer"] for row in differences(old, result)} == allowed
    return result


def run(argv: list, timeout: int = 55) -> dict:
    """Bound commands and retain complete stdout/stderr for offline review."""
    return common.run(argv, cwd=BROKER, timeout=timeout)


def fresh() -> None:
    """Refuse used dispatch identities or a changed accepted broker."""
    assert run(["git", "rev-parse", "HEAD"])["stdout"].strip() == PIN
    assert not run(["git", "status", "--porcelain", "--untracked-files=all"])["stdout"]
    for path in (
        NEW / "output-02",
        NEW / "state-02",
        OPS / "dispatch" / JOB,
        OPS / "manifests" / f"{JOB}.json",
        OPS / "logs" / f"{JOB}.log",
        OPS / "state" / JOB,
    ):
        assert not path.exists(), path


def command() -> list[str]:
    """Return the sole prepared operator command, never executed here."""
    return [
        str(BROKER / "tools/ops/dispatch.sh"),
        JOB,
        str(NEW / "manifest.json"),
        "--broker",
        str(BROKER),
        "--cwd",
        str(BROKER),
        "--state-dir",
        str(NEW / "state-02"),
        "--dispatch-txt",
        str(NEW / "DISPATCH.txt"),
    ]


def copy_tree(source: Path, target: Path) -> dict:
    """Publish exact bytes once, retaining every path/size/hash mapping."""
    assert not target.exists(), target
    rows = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        reference = common.binding(path)
        relative = path.relative_to(source).as_posix()
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(path.read_bytes())
        assert common.binding(destination)["sha256"] == reference["sha256"]
        rows[relative] = {"source": reference, "published": common.binding(destination)}
    return rows


def refusal() -> None:
    """Publish all output-01 bytes before preparing the successor."""
    assert not REFUSAL.exists() and not REFUSAL_DEST.exists()
    old = load(OLD / "manifest.json")
    receipt = load(OLD / "output-01/receipt.json")
    assert receipt["endpoint"] is None and receipt["terminal"] is None
    assert receipt["job_id"] == old["job_id"]
    assert receipt["error"] == "Refused('insufficient balance for cap plus study reserve')"
    assert receipt["exports"] == receipt["transfers"] == receipt["contents_transfers"] == []
    assert [call["command"] for call in receipt["cli_calls"]] == ["version", "sessions", "usage"]
    assert common.binding(OLD / "output-01/manifest.json")["sha256"] == receipt["manifest_sha256"]
    # Validate normalization with the pinned broker, using a fresh throwaway output identity.
    code = (
        "import json,sys; from pathlib import Path; "
        "from research.colab_cli.supervisor import validate; "
        "m=json.loads(Path(sys.argv[1]).read_bytes()); original=m['output_dir']; "
        "m['output_dir']=sys.argv[2]; n=validate(m,Path.cwd()); n['output_dir']=original; "
        "assert n==json.loads(Path(sys.argv[3]).read_bytes()); print('PASS_NORMALIZATION')"
    )
    with tempfile.TemporaryDirectory(prefix="f5-longrun-refusal-") as temporary:
        normalization = run(
            [
                common.PYTHON,
                "-c",
                code,
                OLD / "manifest.json",
                Path(temporary) / "unused",
                OLD / "output-01/manifest.json",
            ]
        )
    files = copy_tree(OLD / "output-01", REFUSAL / "output-01")
    state = copy_tree(OLD / "state-01", REFUSAL / "state-01")
    export_json(
        REFUSAL / "REFUSAL_EVIDENCE.json",
        {
            "schema": "bl-f5-longrun01-refusal/v1",
            "status": "REFUSED_BEFORE_ALLOCATION",
            "accepted_manifest": common.binding(OLD / "manifest.json"),
            "files": files,
            "state_files": state,
            "receipt": common.binding(OLD / "output-01/receipt.json"),
            "normalization": normalization,
            "allocation_calls": 0,
            "endpoint": None,
            "elapsed_seconds": receipt["elapsed_seconds"],
            "release_not_applicable_no_endpoint_allocated": True,
            "stop_verified_false_is_not_an_unreleased_runtime": True,
        },
    )
    rows = copy_tree(REFUSAL, REFUSAL_DEST)
    export_json(REFUSAL_DEST / "PUBLICATION.json", {"files": rows, "dispatches": 0})
    print("REFUSAL_PUBLISHED", common.binding(REFUSAL / "REFUSAL_EVIDENCE.json"))


def inherited() -> dict:
    """Rehash every accepted static member and external dependency, without outcomes."""
    verify_export(OLD / "SHA256SUMS.json")
    seal = load(OLD / "SHA256SUMS.json")
    for relative, row in seal["files"].items():
        actual = common.binding(OLD / relative)
        assert actual["sha256"] == row["sha256"] and actual["size"] == row["size"], relative
    for name in ("DEPENDENCIES.json", "DEPENDENCIES-SUPPLEMENT01.json"):
        for row in load(OLD / name)["files"].values():
            common.check(row)
    return {
        "static_members": len(seal["files"]),
        "dependencies": len(load(OLD / "DEPENDENCIES.json")["files"]),
        "supplement": len(load(OLD / "DEPENDENCIES-SUPPLEMENT01.json")["files"]),
        "seal": common.binding(OLD / "SHA256SUMS.json"),
        "package": common.binding(OLD / "PACKAGE.json"),
        "publication": common.binding(
            LEDGER / "F5_ENVELOPE_LONGRUN01_SEALED_20261026/ARTIFACT_PACKET.json"
        ),
    }


def prepare() -> None:
    """Create the new package without changing any predecessor bytes."""
    fresh()
    assert not NEW.exists()
    verify_export(REFUSAL / "REFUSAL_EVIDENCE.json")
    inheritance = inherited()
    baseline = {
        str(p.relative_to(OLD)): common.binding(p) for p in sorted(OLD.rglob("*")) if p.is_file()
    }
    export_json(NEW / "PREDECESSOR_INVENTORY.json", baseline)
    export_json(NEW / "INHERITED_VERIFICATION.json", inheritance)
    export_bytes(NEW / Path(__file__).name, Path(__file__).read_bytes())
    for name in ("OWNER_RULING_COLAB_BUDGET_20261025.txt", "ACCEPT_F5_LONGRUN_01.md"):
        export_bytes(NEW / "authority" / name, (ASTRA / name).read_bytes())
    # This read-only snapshot is advisory; locked live admission remains mandatory at dispatch.
    export_json(NEW / ACCOUNT, common.account_preflight())
    old = load(OLD / "manifest.json")
    manifest = successor(old, common.binding(NEW / ACCOUNT))
    export_json(NEW / "manifest.json", manifest)
    export_json(
        NEW / "SEALED_TO_SEALED02_MANIFEST_DIFF.json",
        {
            "schema": "bl-f5-exhaustive-manifest-diff/v1",
            "science_change": False,
            "before": common.binding(OLD / "manifest.json"),
            "after": common.binding(NEW / "manifest.json"),
            "changes": differences(old, manifest),
        },
    )
    for name in ("SERIES_FREEZE.json",):
        export_bytes(NEW / name, (OLD / name).read_bytes())
    export_json(
        NEW / "MECHANICAL_SUCCESSOR.json",
        {
            "schema": "bl-f5-longrun01-mechanical-guard-successor/v1",
            "scope": "INITIAL_EMPTY_LEDGER_TIMING_ATTEMPT_ONLY",
            "refusal": common.binding(REFUSAL / "REFUSAL_EVIDENCE.json"),
            "owner_ruling": common.binding(ASTRA / "OWNER_RULING_COLAB_BUDGET_20261025.txt"),
            "guard": {
                "base_units": 50,
                "effective_units": 100,
                "reserve_units": 0,
                "observed_rate_units_per_hour": 1.54,
                "four_hour_forecast_units": 6.16,
                "priced_rate_unchanged": 2,
                "priced_four_hour_units": 8,
            },
            "guard_meaning": "Runaway guard only; no scientific budget or cap change.",
            "mapping": {
                "supersedes_initial_job": old["job_id"],
                "initial_job": JOB,
                "attempt_index": 0,
                "control_and_empty_ledger_bytes": "UNCHANGED",
            },
            "series_freeze": "Historical bytes unchanged. Its 100000 guard and initial dispatch "
            "paths are superseded only by this successor's manifest. The -02 job is the initial "
            "pre-allocation refusal successor, not a completion attempt. Any later completion "
            "requires separately admitted fresh identities, never reuse this -02 identity.",
            "unchanged": [
                "source",
                "inputs",
                "environment",
                "13 units",
                "120/240 minutes",
                "retry rules",
                "canary receipt",
                "allocation",
                "look discipline",
            ],
            "exact_successor_admission": "PENDING_ASTRA_REVIEW",
            "allocations": 0,
            "dispatches": 0,
        },
    )
    cmd = shlex.join(command())
    text = (
        "SEALED LONGRUN-01 MECHANICAL GUARD SUCCESSOR. Separate Astra exact-package admission "
        "required before operator execution. Prepared only; no dispatch.\n"
        "100 units is a runaway guard: base50/effective100, reserve0; 240 minutes uses about "
        "6.16 units at observed1.54 u/h (8 units at unchanged priced2 u/h). Owner ruling "
        "OWNER_RULING_COLAB_BUDGET_20261025.txt applies.\n"
        "Rehash SHA256SUMS, inherited seal/dependencies, refusal publication and manifest diff. "
        "Preserve frozen runtime source911426db, input bytes, empty ledger, 13 units, base120/"
        "effective240-minute wall envelope and no automatic retries. No fit, numerical look, "
        "serving or release authority. Normal locked live account admission is mandatory; "
        "saved account evidence and dry-run do not establish future availability.\n"
        "Exact dry-run:\n" + cmd + " --dry-run\n"
        "Exact operator dispatch after separate Astra exact-package admission:\n" + cmd + "\n"
        "This replaces only the refused initial dispatch; attempt_index remains0. Preserve "
        "historical SERIES_FREEZE bytes; see MECHANICAL_SUCCESSOR.json for guard/path mapping. "
        "Any future completion needs unused identities and separate admission.\n"
    )
    export_bytes(NEW / "DISPATCH.txt", text.encode())
    dry = run(command() + ["--dry-run"], timeout=120)
    assert dry["stdout"].strip() == f"PASS_DRY_RUN {JOB} allocations=0 live_accounting=NOT_CHECKED"
    export_json(NEW / "dry-run.json", dry)
    # Run pinned supervisor validation as well as the wrapper/canary dry-run.
    validation = run(
        [
            BROKER / "research/colab_cli/launch.sh",
            "--state-dir",
            NEW / "state-02",
            "--colab",
            "/Users/josenunez/.local/bin/colab",
            "validate",
            NEW / "manifest.json",
        ],
        timeout=120,
    )
    export_json(NEW / "broker-validation.json", validation)
    fresh()
    print(dry["stdout"].strip())


def seal() -> None:
    """Seal the exhaustive new package after independent review."""
    fresh()
    inherited()
    for row in load(NEW / "PREDECESSOR_INVENTORY.json").values():
        common.check(row)
    assert load(NEW / "INDEPENDENT_REVIEW.json")["status"] == "NO_BLOCKING_FINDINGS"
    assert differences(load(OLD / "manifest.json"), load(NEW / "manifest.json")) == (
        load(NEW / "SEALED_TO_SEALED02_MANIFEST_DIFF.json")["changes"]
    )
    export_json(NEW / "accounts/account-recheck-before-seal.json", common.account_preflight())
    export_json(
        NEW / "PACKAGE.json",
        {
            "schema": "bl-f5-longrun01-mechanical-guard-package/v1",
            "status": "SEALED_PENDING_ASTRA_EXACT_PACKAGE_ADMISSION",
            "manifest": common.binding(NEW / "manifest.json"),
            "manifest_diff": common.binding(NEW / "SEALED_TO_SEALED02_MANIFEST_DIFF.json"),
            "dispatch": common.binding(NEW / "DISPATCH.txt"),
            "dry_run": common.binding(NEW / "dry-run.json"),
            "inherited": inherited(),
            "runtime_source_commit": PIN,
            "refusal": common.binding(REFUSAL / "REFUSAL_EVIDENCE.json"),
            "allocations": 0,
            "dispatches": 0,
            "science_change": False,
        },
    )
    for marker in NEW.rglob("*.complete"):
        verify_export(Path(str(marker).removesuffix(".complete")))
    files = {
        str(p.relative_to(NEW)): common.binding(p) for p in sorted(NEW.rglob("*")) if p.is_file()
    }
    export_json(
        NEW / "SHA256SUMS.json",
        {
            "schema": "bl-f5-exhaustive-mechanical-package-seal/v1",
            "files": files,
            "excluded_self_components": [
                "SHA256SUMS.json",
                "SHA256SUMS.json.sha256",
                "SHA256SUMS.json.complete",
            ],
            "inherited": inherited(),
            "allocations": 0,
            "dispatches": 0,
        },
    )
    print("SEALED", common.binding(NEW / "SHA256SUMS.json"))


def publish() -> None:
    """Copy the complete successor to Git's publication boundary create-only."""
    fresh()
    verify_export(NEW / "SHA256SUMS.json")
    for row in load(NEW / "SHA256SUMS.json")["files"].values():
        common.check(row)
    expected = set(load(NEW / "SHA256SUMS.json")["files"]) | {
        "SHA256SUMS.json",
        "SHA256SUMS.json.sha256",
        "SHA256SUMS.json.complete",
    }
    assert {str(p.relative_to(NEW)) for p in NEW.rglob("*") if p.is_file()} == expected
    rows = copy_tree(NEW, DEST)
    export_json(
        DEST / "PUBLICATION.json",
        {
            "schema": "bl-f5-longrun01-guard-publication/v1",
            "files": rows,
            "seal": common.binding(NEW / "SHA256SUMS.json"),
            "dispatch_command": shlex.join(command()),
            "allocations": 0,
            "dispatches": 0,
        },
    )
    print("PUBLISHED", common.binding(DEST / "PUBLICATION.json"))


def main() -> None:
    """Expose preparation phases only, with no run or allocation operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("refusal", "prepare", "seal", "publish"))
    args = parser.parse_args()
    {"refusal": refusal, "prepare": prepare, "seal": seal, "publish": publish}[args.phase]()


if __name__ == "__main__":
    main()
