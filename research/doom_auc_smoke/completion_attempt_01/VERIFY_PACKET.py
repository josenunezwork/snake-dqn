"""Verify the sealed successor and prior evidence without reading outcome tensors."""

from __future__ import annotations

import json
from pathlib import Path

from research.doom_auc_smoke.evidence import build_ledger, digest

ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-doom-auc-smoke")
ART = ROOT.parent / "snake-dqn-artifacts"
HERE = ART / "t34-sol/doom-fit-retry"
PACKET = HERE / "packet-01"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify() -> dict:
    before = read(HERE / "EVIDENCE_BEFORE.json")
    for name, row in before.items():
        path = Path(name)
        check(
            path.stat().st_size == row["size"] and digest(path) == row["sha256"],
            "prior evidence changed: " + name,
        )
    lineage = read(PACKET / "LINEAGE.json")
    item = read(PACKET / "fit.json")
    for name in (
        "original_manifest",
        "original_admission",
        "executed_manifest",
        "mechanical_receipt",
        "preparation_script",
    ):
        row = lineage[name]
        check(digest(Path(row["path"])) == row["sha256"], "lineage differs: " + name)
    check(
        item["completion_lineage"]
        == dict(path=str(PACKET / "LINEAGE.json"), sha256=digest(PACKET / "LINEAGE.json")),
        "lineage binding",
    )
    original = read(Path(lineage["original_manifest"]["path"]))
    differences = sorted(k for k in original.keys() | item.keys() if original.get(k) != item.get(k))
    check(
        differences
        == [
            "completion_attempt",
            "completion_lineage",
            "exec_poll_recovery_seconds",
            "job_id",
            "output_dir",
        ],
        "unexpected manifest changes",
    )
    check(item["job_id"] == "t34-doom-fit-02-a100-completion-1", "job identity")
    check(item["completion_attempt"] == lineage["completion_attempt"] == 1, "attempt index")
    check(item["exec_poll_recovery_seconds"] == 1200, "recovery window")
    check(item["resume"]["max_outcome_blind_retries"] == 2, "retry ceiling")
    check(item["technical_envelope"]["series_max_original_plus_completions"] == 3, "series ceiling")
    config = read(Path(next(row["path"] for row in item["inputs"] if row["dest"] == "config.json")))
    check(not Path(config["mac_look_file"]).exists(), "scoring look exists")
    for row in item["inputs"]:
        check(digest(Path(row["path"])) == row["sha256"], "input changed")
    ledger = build_ledger([Path(lineage["original_output"])], config["identity"])
    check(ledger["units"] == [], "zero-export special case no longer applies")
    ledger_input = Path(
        next(row["path"] for row in item["inputs"] if row["dest"] == "verified-ledger.json")
    )
    check(
        ledger == read(ledger_input) == read(PACKET / "VERIFIED_ZERO_EXPORT_LEDGER.json"),
        "ledger changed",
    )
    check(not Path(item["output_dir"]).exists(), "output consumed")
    check(not (PACKET / (item["job_id"] + "-state")).exists(), "state consumed")
    for subdir, name in [
        ("dispatch", item["job_id"]),
        ("logs", item["job_id"] + ".log"),
        ("manifests", item["job_id"] + ".json"),
    ]:
        check(not (ART / "ops" / subdir / name).exists(), "dispatch identity consumed")
    return dict(
        status="PASS",
        prior_files_unchanged=len(before),
        changed_manifest_keys=differences,
        uploaded_inputs=len(item["inputs"]),
        runtime_source_paths=len(item["source_paths"]),
        config_identity=config["identity"],
        runtime_pin=item["commit"],
        verified_units=0,
        scoring_look_exists=False,
        dispatched=False,
        manifest_sha256=digest(PACKET / "fit.json"),
        lineage_sha256=digest(PACKET / "LINEAGE.json"),
        preparation_script_sha256=digest(HERE / "PREPARE_COMPLETION.py"),
    )


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
