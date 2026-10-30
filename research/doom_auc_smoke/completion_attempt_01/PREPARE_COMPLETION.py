"""Create this one outcome-blind, zero-export transport completion; never dispatch."""

from __future__ import annotations

import copy
import json
import shlex
from pathlib import Path

from research.doom_auc_smoke.evidence import build_ledger, digest, json_once
from research.doom_auc_smoke.prelook import read_prelook
from research.doom_auc_smoke.resume import eligible_failure

ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-doom-auc-smoke")
ART = ROOT.parent / "snake-dqn-artifacts"
HERE = ART / "t34-sol/doom-fit-retry"
PRIOR = ART / "t34-sol/doom-admit-fix/admission-01"
OUTPUT = ART / "t34-sol/doom-auc-fix2/packet-01/fit-output-a100-fallback"
PACKET = HERE / "packet-01"
JOB = "t34-doom-fit-02-a100-completion-1"
OPS = ART / "ops"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def bound(path: Path) -> dict:
    return dict(path=str(path), sha256=digest(path))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    original = read(PRIOR / "fit.json")
    receipt = read(OUTPUT / "receipt.json")
    executed = read(OUTPUT / "manifest.json")
    admission = read(PRIOR / "ADMISSION.json")
    eligible_failure(receipt)
    require(original.get("completion_attempt", 0) == 0, "original only; attempt 1 required")
    require(receipt["manifest_sha256"] == digest(OUTPUT / "manifest.json"), "receipt binding")
    require(receipt["job_id"] == original["job_id"] == executed["job_id"], "job lineage")
    require(receipt["commit"] == original["commit"] == executed["commit"], "source lineage")
    require(receipt["status"] == "INCOMPLETE", "incomplete only")
    require(receipt["failure_class"] == "RPC_EXEC_POLL_TIMEOUT", "mechanical class")
    require(
        receipt["error"] == "Refused('kernel remains busy/unknown after telemetry hiccup')",
        "exact mechanical error required; class alone is insufficient",
    )
    require(receipt["cleanup_error"] is None, "cleanup error")
    require(receipt["exports"] == [] and receipt["terminal"] is None, "zero-output stop only")
    require(not (OUTPUT / "exports").exists(), "local export evidence needs generic resume")
    require(original["output_dir"] == str(OUTPUT), "original output lineage")
    # The supervisor inserts its default lease; dispatch refreshes account inventory.
    normalized = copy.deepcopy(executed)
    require(normalized.pop("lease_seconds") == 180, "unexpected lease")
    normalized["budget"]["other_assignments"] = original["budget"]["other_assignments"]
    require(normalized == original, "executed science/admission differs")
    require(original["requirements"]["allowed_gpus"] == ["A100"], "A100-only admission")
    require(original["resume"]["max_outcome_blind_retries"] == 2, "completion ceiling")
    require(
        original["technical_envelope"]["series_max_original_plus_completions"] == 3,
        "series ceiling",
    )
    for entry in original["inputs"]:
        require(digest(Path(entry["path"])) == entry["sha256"], "input digest differs")
    inputs = {entry["dest"]: Path(entry["path"]) for entry in original["inputs"]}
    config = read(inputs["config.json"])
    require(not Path(config["mac_look_file"]).exists(), "scientific look already reserved")
    ledger = build_ledger([OUTPUT], config["identity"])
    require(ledger == read(inputs["verified-ledger.json"]), "empty ledger differs")
    require(ledger["units"] == [], "completed units require generic resume")
    for name in ("pending_manifest", "h100_receipt", "a100_receipt", "prelook"):
        ref = admission[name]
        require(digest(Path(ref["path"])) == ref["sha256"], "admission evidence differs")
    require(admission["runtime_commit"] == config["commit"] == original["commit"], "pin")
    require(original["canary"]["receipt"] == admission["a100_receipt"], "canary admission")
    require(admission["mode"] == "A100_ONLY_UNAVAILABLE_H100", "admission mode")
    for name, sha in admission["mac_implementation"].items():
        require(digest(ROOT / "research/doom_auc_smoke" / name) == sha, "Mac admission drift")
    read_prelook(Path(admission["prelook"]["path"]), config)
    require(not PACKET.exists(), "completion packet must be fresh")
    for path in (
        OPS / "logs" / f"{JOB}.log",
        OPS / "manifests" / f"{JOB}.json",
        OPS / "dispatch" / JOB,
    ):
        require(not path.exists(), "job identity already consumed")
    # Preserve every prior packet/admission file and the exact operational log.
    files = set()
    for directory in (OUTPUT.parent, PRIOR):
        files.update(path for path in directory.rglob("*") if path.is_file())
    files.add(OPS / "logs/t34-doom-fit-02-a100-fallback.log")
    before = {
        str(path): dict(size=path.stat().st_size, sha256=digest(path)) for path in sorted(files)
    }
    json_once(HERE / "EVIDENCE_BEFORE.json", before)
    PACKET.mkdir()
    json_once(PACKET / "VERIFIED_ZERO_EXPORT_LEDGER.json", ledger)
    lineage = dict(
        schema="t34-doom-zero-export-completion/v1",
        completion_attempt=1,
        series_max_original_plus_completions=3,
        original_manifest=bound(PRIOR / "fit.json"),
        original_admission=bound(PRIOR / "ADMISSION.json"),
        executed_manifest=bound(OUTPUT / "manifest.json"),
        mechanical_receipt=bound(OUTPUT / "receipt.json"),
        original_output=str(OUTPUT),
        verified_units=0,
        missing_arms=["spatial", "b8"],
        previous_recovery_seconds=receipt["exec_poll_recovery_seconds"],
        recovery_base_seconds=600,
        recovery_effective_seconds=1200,
        recovery_scope="owned non-staging exec/poll; staging limits unchanged",
        inputs_unchanged=True,
        admission_unchanged=True,
        scoring_look_taken=False,
        preparation_script=bound(Path(__file__).resolve()),
    )
    json_once(PACKET / "LINEAGE.json", lineage)
    successor = copy.deepcopy(original)
    successor.update(
        job_id=JOB,
        output_dir=str(PACKET / "fit-output"),
        completion_attempt=1,
        exec_poll_recovery_seconds=1200,
        completion_lineage=bound(PACKET / "LINEAGE.json"),
    )
    json_once(PACKET / "fit.json", successor)
    argv = [
        str(ROOT / "tools/ops/dispatch.sh"),
        JOB,
        str(PACKET / "fit.json"),
        "--broker",
        str(ROOT),
        "--cwd",
        str(ROOT),
        "--state-dir",
        str(PACKET / (JOB + "-state")),
        "--dispatch-txt",
        str(PACKET / "DISPATCH.txt"),
    ]
    with (PACKET / "DISPATCH.txt").open("x") as stream:
        stream.write(shlex.join([argv[0], "--dry-run", *argv[1:]]) + "\n")
        stream.write(shlex.join(argv) + "\n")
    print(
        json.dumps(
            dict(
                status="PREPARED_NOT_DISPATCHED",
                job_id=JOB,
                manifest=bound(PACKET / "fit.json"),
                verified_units=0,
            )
        )
    )


if __name__ == "__main__":
    main()
