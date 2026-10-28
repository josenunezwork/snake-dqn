"""Append a source-pinned 09e packet using the committed sealer; never dispatch."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from research.bl_served_search.exports import digest
from research.watch_lead_p import seal

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "research/watch_lead_p/packet-09/reseal-04"
ARTIFACTS = ROOT.parent / "snake-dqn-artifacts"
IDENTITY = "sandbox-p-watch-lead-20261028-09e"


def build(pin: str, broker: Path, runtime: Path) -> None:
    """Preserve prior accounting, add unused-09d evidence, and prepare completion 2."""
    prior_seal = json.loads((PRIOR / "seal.json").read_text())
    for name, checksum in prior_seal.items():
        if digest(PRIOR / name) != checksum:
            raise ValueError("prior sealed packet differs")
    accounting = json.loads((PRIOR / "retry-accounting.json").read_text())
    for record in accounting["evidence"]:
        path = Path(record["path"])
        if path.stat().st_size != record["size"] or digest(path) != record["sha256"]:
            raise ValueError("prior mechanical evidence differs")
    if json.loads((PRIOR / "verified-ledger.json").read_text())["units"]:
        raise ValueError("09e repair expects all 32 units missing")
    # No account calls: local create-only artifacts distinguish preparation from dispatch.
    for suffix in ("09d", "09e"):
        for path in (
            ARTIFACTS / f"t34-watch-lead-1028/attempt-{suffix}",
            ARTIFACTS / "ops/logs" / f"sandbox-p-watch-lead-20261028-{suffix}.log",
            ARTIFACTS / f"ops/dispatch/sandbox-p-watch-lead-20261028-{suffix}",
        ):
            if path.exists():
                raise ValueError(f"identity may have been used: {path}")
    if broker.exists() or runtime.exists():
        raise ValueError("09e requires fresh detached checkout paths")
    accounting.update(
        prepared_job_id=IDENTITY,
        supersedes_prepared_job_id=accounting["prepared_job_id"],
        authority=(
            "User fix9 task 2026-10-28; local preparation only, Opus reviews; no dispatch/push."
        ),
        repair=(
            "F1 accounting-derived sealer/benchmark; F2 identity-independent admission; "
            "F3 pre-arm reader and actual GameState construction checks; F4 distinct "
            "excluded engineering seeds and complete worker participation proof. "
            "Science, worker count/priority, input bytes, forecast and envelope unchanged."
        ),
    )
    accounting["preparation_09e"] = {
        "superseded_09d": "sandbox-p-watch-lead-20261028-09d",
        "09d_dispatched": False,
        "09d_consumed_manual_completions": 0,
        "09e_consumed_now": False,
        "prepared_completion_ordinal": 2,
        "maximum_manual_completions": 2,
        "remaining_before_09e_dispatch": 1,
        "remaining_after_09e_dispatch": 0,
        "prior_accounting_path": str(PRIOR / "retry-accounting.json"),
        "prior_accounting_sha256": digest(PRIOR / "retry-accounting.json"),
        "runtime_pin": pin,
        "historical_classification": (
            "retry_classification records prior 09d preparation verbatim"
        ),
        "basis": "09d real dry-run REFUSED before reservation/allocation; no 09d dispatch artifacts",
    }
    for name in ("SUMMARY.md", "dry-run.txt", "dry-run-diagnostic.txt"):
        path = ARTIFACTS / "t34-sol/astra-p-identity-09d" / name
        accounting["evidence"].append(
            {"path": str(path), "size": path.stat().st_size, "sha256": digest(path)}
        )
    # The recipe closure and construction-bound input bank must remain exact.
    prior_manifest = json.loads((PRIOR / "dispatch-manifest.json").read_text())
    from research.watch_lead_p.resume import empty_ledger

    if json.loads((PRIOR / "verified-ledger.json").read_text()) != empty_ledger(ROOT):
        raise ValueError("repair changed the frozen scientific recipe or construction bank")
    with tempfile.TemporaryDirectory(prefix="astra-p-fix9-accounting-") as scratch:
        path = Path(scratch) / "retry-accounting.json"
        seal.write(path, accounting)
        seal.seal(
            pin,
            broker,
            runtime,
            PRIOR / "forecast.json",
            attempt=9,
            revision=5,
            identity_suffix="e",
            retry_accounting=path,
        )
    packet = PRIOR.with_name("reseal-05")
    current = json.loads((packet / "dispatch-manifest.json").read_text())
    for key in ("requirements", "budget", "required_exports", "source_paths", "resume"):
        if current[key] != prior_manifest[key]:
            raise ValueError(f"repair unexpectedly changed {key}")
    for name in ("forecast.json", "exclusion-audit.json", "verified-ledger.json"):
        if (packet / name).read_bytes() != (PRIOR / name).read_bytes():
            raise ValueError(f"repair unexpectedly changed {name}")
    print(f"Created {packet}; completion 2 of 2; 09d consumed nothing; no dispatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--broker", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    build(args.pin, args.broker, args.runtime)
