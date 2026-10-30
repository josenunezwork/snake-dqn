"""Fail-closed checks used by dispatch dry-run and the real runtime entrypoint."""

from __future__ import annotations

import json
import os
from pathlib import Path

from research.watch6_search_efficacy import protocol as p
from research.watch6_search_efficacy.run import completed
from tools.ops.package_spec import digest


def admit(manifest: dict) -> None:
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
    item = next(i for i in manifest["inputs"] if i["dest"] == "verified-ledger.json")
    root = Path(item["path"]).parent
    verify_runtime(root, env)


def verify_runtime(root: Path, env: dict) -> None:
    """Same execute-time guard, exercised by the ordinary host dry-run."""
    contract = json.loads((root / "runtime-contract.json").read_text())
    if contract != {"science": p.identity(), "allocation": p.allocation()}:
        raise ValueError("runtime source/allocation mismatch")
    if env.get("WATCH6_SCIENCE") != p.identity():
        raise ValueError("runtime science env mismatch")
    if env.get("WATCH6_FIXTURE") not in ("0", "1") or int(env["WATCH6_SHARD"]) not in range(6):
        raise ValueError("runtime shard/fixture mismatch")
    completed(root / "verified-ledger.json")
    if env["WATCH6_FIXTURE"] == "0":
        approval = json.loads((root / "fable-approval.json").read_text())
        if approval != {
            "status": "RATIFIED",
            "judge": "Fable",
            "science": p.identity(),
            "registration_sha256": digest(Path(__file__).with_name("REGISTRATION.md")),
        }:
            raise ValueError("independent Fable ratification absent or different")


def runtime_admit() -> None:
    """Enforce source/allocation/ledger and approval at actual process execution."""
    verify_runtime(Path(os.environ["COLAB_INPUT_DIR"]), dict(os.environ))
