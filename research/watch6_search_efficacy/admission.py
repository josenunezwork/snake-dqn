"""Fail-closed checks used by dispatch dry-run and the real runtime entrypoint."""

from __future__ import annotations

import json
import os
from pathlib import Path

from research.watch6_search_efficacy import protocol as p
from research.watch6_search_efficacy.run import completed
from tools.ops.package_spec import digest


def admit(manifest: dict, *, runtime: bool = False) -> None:
    """Check exact controller, allocation, identities and pre-data envelopes."""
    binding = manifest["watch6_efficacy"]
    if binding["science"] != p.identity() or binding["allocation"] != p.allocation():
        raise ValueError("foreign source/allocation")
    if manifest["script"] != p.SCRIPT or manifest["device"] != "cpu":
        raise ValueError("wrong entrypoint/device")
    if manifest["requirements"]["gpu"] != "L4" or manifest["requirements"]["allowed_gpus"] != [
        "L4"
    ]:
        raise ValueError("only registered L4 CPU placement")
    if manifest["timeout_minutes"] != 2 * binding["base_minutes"]:
        raise ValueError("requires pre-data 2x wall guard")
    if (
        binding["base_minutes"] > 720
        or manifest["requirements"]["est_minutes"] != binding["base_minutes"]
    ):
        raise ValueError("forecast exceeds safe provider session length")
    for key in ("unit_cap", "reserve_units"):
        if manifest["budget"][key] != 2 * binding["base_" + key]:
            raise ValueError("requires pre-data 2x pricing envelope")
    probe = manifest.get("canary", {}).get("mode") == "probe"
    env = manifest["env"]
    if env["WATCH6_SCIENCE"] != p.identity() or int(env["WATCH6_SHARD"]) not in range(6):
        raise ValueError("runtime science/shard mismatch")
    if env["WATCH6_FIXTURE"] != ("1" if probe else "0"):
        raise ValueError("canary/main fixture isolation mismatch")
    if manifest["resume"] != {
        "unit_kind": "paired-session",
        "ledger_path": "inputs/verified-ledger.json",
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }:
        raise ValueError("resume policy mismatch")
    if runtime:
        for key in ("WATCH6_SHARD", "WATCH6_FIXTURE", "WATCH6_SCIENCE"):
            if os.environ.get(key) != env[key]:
                raise ValueError("execute-time environment differs from frozen manifest")
        ledger = Path(os.environ["COLAB_INPUT_DIR"]) / "verified-ledger.json"
    else:
        item = next(i for i in manifest["inputs"] if i["dest"] == "inputs/verified-ledger.json")
        ledger = Path(item["path"])
    rows = completed(ledger)
    if probe and rows:
        raise ValueError("canary must use empty ledger")
    if not probe:
        # Deliberate non-bypassable missing prerequisite. Operator supplies the
        # independent judge's source-bound approval, not an invented env token.
        approval_path = (
            Path(os.environ["COLAB_INPUT_DIR"]) / "fable-approval.json"
            if runtime
            else Path(
                next(
                    i["path"]
                    for i in manifest["inputs"]
                    if i["dest"] == "inputs/fable-approval.json"
                )
            )
        )
        approval = json.loads(approval_path.read_text())
        if approval != {
            "status": "RATIFIED",
            "judge": "Fable",
            "science": p.identity(),
            "registration_sha256": digest(Path(__file__).with_name("REGISTRATION.md")),
        }:
            raise ValueError("independent Fable ratification absent or different")


def runtime_admit() -> None:
    """Enforce source/allocation/ledger and approval at actual process execution."""
    root = Path(os.environ["COLAB_INPUT_DIR"])
    contract = json.loads((root / "runtime-contract.json").read_text())
    if contract != {"science": p.identity(), "allocation": p.allocation()}:
        raise ValueError("runtime source/allocation mismatch")
    if os.environ.get("WATCH6_SCIENCE") != p.identity():
        raise ValueError("runtime science env mismatch")
    if os.environ.get("WATCH6_FIXTURE") not in ("0", "1") or int(
        os.environ["WATCH6_SHARD"]
    ) not in range(6):
        raise ValueError("runtime shard/fixture mismatch")
    rows = completed(root / "verified-ledger.json")
    if os.environ["WATCH6_FIXTURE"] == "1":
        if rows:
            raise ValueError("canary must use empty ledger")
    else:
        approval = json.loads((root / "fable-approval.json").read_text())
        if approval != {
            "status": "RATIFIED",
            "judge": "Fable",
            "science": p.identity(),
            "registration_sha256": digest(Path(__file__).with_name("REGISTRATION.md")),
        }:
            raise ValueError("independent Fable ratification absent or different")
