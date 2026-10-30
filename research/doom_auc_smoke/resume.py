"""Offline outcome-blind completion packer; never launches or retries a job."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from research.compute.gpu_preference import variants
from research.doom_auc_smoke.evidence import ARMS, build_ledger, empty_ledger, json_once, latest
from research.doom_auc_smoke.prepare import binding, commands


def eligible_failure(receipt: dict) -> None:
    """Only verified mechanical interruption admits a completion, not numerical faults."""
    text = json.dumps(
        {k: receipt.get(k) for k in ("status", "error", "failure_class", "cleanup_error")}
    )
    if (
        receipt.get("stop_verified") is not True
        or receipt.get("status") == "PASS"
        or re.search(r"nan|nonfinite|overflow|integrity|invalid|numerical|assertion", text, re.I)
        or not re.search(
            r"timeout|deadline|lease|SIGUSR1|RPC_|connection|transport|interrupted", text, re.I
        )
    ):
        raise ValueError("verified mechanical failure and stop required; no scientific retry")


def completion(prior: Path, outputs: list[Path], packet: Path) -> dict:
    if not 1 <= len(outputs) <= 2 or len(set(outputs)) != len(outputs):
        raise ValueError("one original plus at most two completions")
    if packet.exists() or not packet.is_absolute():
        raise ValueError("fresh absolute completion packet required")
    previous = json.loads((prior / "fit.pending.json").read_text())
    attempt = previous.get("completion_attempt", 0) + 1
    if attempt > 2 or len(outputs) != attempt:
        raise ValueError(
            "complete original-through-current attempt history required; max two completions"
        )
    eligible_failure(json.loads((outputs[-1] / "receipt.json").read_text()))
    config = json.loads((prior / "config.json").read_text())
    union = build_ledger(outputs, config["identity"])
    if all(any(r["unit"] == arm + "-prediction" for r in union["units"]) for arm in ARMS):
        raise ValueError("no missing scientific work; completion forbidden")
    # A trajectory resumes only its latest verified eligible full-state snapshot.
    # Older snapshots remain in the immutable Mac union and cannot be recomputed.
    selected = []
    for arm in ARMS:
        pred = next((r for r in union["units"] if r["unit"] == arm + "-prediction"), None)
        row = pred or latest(union, arm)
        if row:
            selected.append(row)
    packet.mkdir(parents=True)
    for name in ("config.json", "REGISTRATION.md", "dataset.pt.gz", "assessment-MAC-ONLY.pt.gz"):
        shutil.copyfile(prior / name, packet / name)
    ledger = dict(empty_ledger(config["identity"]), units=selected)
    json_once(packet / "verified-ledger.json", ledger)
    json_once(packet / "prior-union-MAC-ONLY.json", union)
    resumed_inputs = []
    for row in selected:
        owners = [
            out / "exports/doom"
            for out in outputs
            if (out / "exports/doom" / row["path"]).is_file()
        ]
        if len(owners) != 1:
            raise ValueError("duplicate unit ownership")
        for suffix in ("", ".sha256", ".complete"):
            relative = row["path"] + suffix
            dest = packet / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(owners[0] / relative, dest)
            resumed_inputs.append(binding(dest, relative))
    item = previous
    item["completion_attempt"] = attempt
    item["inputs"] = [
        binding(packet / name, name)
        for name in ("config.json", "REGISTRATION.md", "dataset.pt.gz", "verified-ledger.json")
    ] + resumed_inputs
    item["job_id"] = f"t34-doom-fit-0{len(outputs)+1}"
    item["output_dir"] = str(packet / "fit-output")
    item["requirements"].pop("hardware_canaries", None)
    item.pop("canary", None)
    json_once(packet / "fit.pending.json", item)
    canary = json.loads((prior / "canary.json").read_text())
    canary["inputs"] = item["inputs"]
    canary["job_id"] = f"t34-doom-canary-0{len(outputs)+1}"
    canary["output_dir"] = str(packet / "canary-output")
    json_once(packet / "canary.json", canary)
    for variant in variants(canary):
        gpu = variant["requirements"]["gpu"].lower()
        variant["requirements"].pop("prefer", None)
        json_once(packet / f"canary-{gpu}.json", variant)
    (packet / "DISPATCH.txt").write_text(
        "Completion prepared; genuine new same-input canaries required.\n"
        + "\n".join(
            commands(packet, canary["job_id"], "canary.json")
            + commands(packet, item["job_id"], "fit.json")
        )
        + "\n"
    )
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-packet", type=Path, required=True)
    parser.add_argument("--outputs", nargs="+", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    result = completion(
        args.prior_packet.resolve(), [p.resolve() for p in args.outputs], args.packet.resolve()
    )
    print(json.dumps(dict(status="PREPARED_NOT_DISPATCHED", units=len(result["units"]))))


if __name__ == "__main__":
    main()
