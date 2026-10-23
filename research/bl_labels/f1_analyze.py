"""Mac-side F1 placement decision from verified timing exports only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .exports import _verify, publish_json
from .f1_ruler import bank_digest, load_bank

LABELS = ("one", "half", "all")


def _positive(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("F1 timing must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("F1 timing must be finite and positive")
    return float(value)


def transport_faults(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Exclude any observed failed/budget-exhausting CLI call, even if recovered.

    This conservative operational rule also excludes unresolved transport faults;
    it never interprets a slow scientific root as proof of an RPC stall.
    """
    calls = receipt.get("cli_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("F1 needs the supervisor CLI call census")
    faults = []
    for call in calls:
        elapsed = call.get("elapsed_seconds")
        timeout = _positive(call.get("timeout_seconds"))
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
            or call.get("status") not in {"completed", "failed"}
        ):
            raise ValueError("F1 CLI census is malformed")
        if call["status"] != "completed" or elapsed >= timeout:
            faults.append(call)
    return faults


def load_attempt(path: Path, bank_sha256: str) -> dict[str, Any]:
    """Bind an immutable runtime export to its terminal supervisor receipt."""
    receipt_path = path / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    faults = transport_faults(receipt)
    manifest_path = path / "manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != receipt.get("manifest_sha256"):
        raise ValueError("F1 canonical manifest differs from receipt")
    manifest = json.loads(manifest_path.read_text())
    platform = manifest.get("env", {}).get("F1_PLATFORM")
    if platform not in {"cpu", "l4"}:
        raise ValueError("F1 manifest has a foreign platform")
    result = {
        "platform": platform,
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "transport_faults": faults,
        "configs": [],
        "complete": False,
    }
    if receipt.get("stop_verified") is not True or receipt.get("cleanup_error"):
        raise ValueError("F1 attempt has no verified backend release")
    exported = path / "exports" / "runtime-result.json"
    if receipt.get("status") != "PASS":
        inventory = [
            r for r in receipt.get("exports", []) if r.get("path") == "runtime-result.json"
        ]
        if inventory:
            digest = _verify(exported)
            if len(inventory) != 1 or inventory[0].get("sha256") != digest:
                raise ValueError("F1 incomplete attempt runtime export differs from receipt")
            diagnostic = json.loads(exported.read_text())
            if (
                diagnostic.get("bank_sha256") != bank_sha256
                or diagnostic.get("platform") != platform
            ):
                raise ValueError("F1 incomplete attempt has foreign bank/platform")
            if diagnostic.get("status") == "INVALID":
                raise ValueError("INVALID F1 attempt cannot enter placement analysis")
            if diagnostic.get("worker_stall") is True:
                result["transport_faults"].append(
                    {"kind": "WORKER_STALL", "reason": diagnostic.get("reason")}
                )
        return result
    digest = _verify(exported)
    inventory = [r for r in receipt.get("exports", []) if r.get("path") == "runtime-result.json"]
    if len(inventory) != 1 or inventory[0].get("sha256") != digest:
        raise ValueError("F1 timing export is not bound to its supervisor receipt")
    if exported.stat().st_size != inventory[0].get("size"):
        raise ValueError("F1 timing export size differs from receipt")
    runtime = json.loads(exported.read_text())
    if runtime.get("platform") != platform or runtime.get("bank_sha256") != bank_sha256:
        raise ValueError("F1 timing export has foreign platform or bank")
    if runtime.get("source_commit") != manifest.get("commit"):
        raise ValueError("F1 runtime source differs from pinned manifest")
    result["source_commit"] = runtime["source_commit"]
    result["complete"] = runtime.get("status") == "COMPLETE"
    result["configs"] = runtime.get("configs", [runtime])
    result["runtime_sha256"] = digest
    return result


def decide(
    attempts: list[dict[str, Any]], root_ids: list[str], *, l4_not_run: bool = False
) -> dict[str, Any]:
    """Apply the frozen 25-percent rule to the fastest complete schedule per host."""
    if len(root_ids) != 64 or len(set(root_ids)) != 64:
        raise ValueError("F1 decision needs exactly 64 distinct frozen root IDs")
    source_commits = {a.get("source_commit") for a in attempts if a.get("complete")}
    if len(source_commits) > 1:
        raise ValueError("F1 cannot mix source implementations across attempts")
    by_platform: dict[str, dict[str, Any]] = {}
    for platform in ("cpu", "l4"):
        selected = [a for a in attempts if a.get("platform") == platform]
        faults = [f for a in selected for f in a.get("transport_faults", [])]
        if faults:
            by_platform[platform] = {"status": "EXCLUDED_THIS_WEEK", "faults": faults}
            continue
        configs = [c for a in selected if a.get("complete") for c in a.get("configs", [])]
        labels = [c.get("schedule_label") for c in configs]
        if len(labels) != len(set(labels)):
            raise ValueError("duplicate F1 timing schedule; do not select among repeat attempts")
        if set(labels) != set(LABELS):
            by_platform[platform] = {"status": "NOT_RUN" if not selected else "INCOMPLETE"}
            continue
        rates = []
        host_nproc = set()
        for config in configs:
            nproc = config.get("cpu", {}).get("nproc", config.get("nproc"))
            if type(nproc) is not int or nproc < 1:
                raise ValueError("F1 needs measured host nproc")
            host_nproc.add(nproc)
            expected = {"one": 1, "half": max(1, nproc // 2), "all": nproc}
            if config.get("source_commit") != next(iter(source_commits), None):
                raise ValueError("F1 schedule has a foreign source implementation")
            if config.get("workers") != expected[config["schedule_label"]]:
                raise ValueError("F1 workers differ from the frozen host schedule")
            if platform == "l4" and nproc != 12:
                raise ValueError("F1 L4 ruler requires the registered 12-vCPU host")
            records = config.get("records", [])
            ids = [row.get("root_id") for row in records]
            if config.get("root_count") != 64 or ids != root_ids:
                raise ValueError("F1 schedule has a partial, duplicate or foreign root bank")
            if config.get("integrity", {}).get("digest_checks") != 64:
                raise ValueError("F1 schedule lacks 64 successful root digest checks")
            elapsed = _positive(config.get("elapsed_seconds"))
            measured = 64 * 3600 / elapsed
            stated = _positive(config.get("roots_per_hour"))
            if not math.isclose(measured, stated, rel_tol=1e-12):
                raise ValueError("F1 roots/hour differs from its registered denominator")
            rates.append((measured, config["schedule_label"]))
        if len(host_nproc) != 1:
            raise ValueError("F1 schedules must use the same host capacity")
        rate, label = max(rates, key=lambda row: (row[0], -LABELS.index(row[1])))
        by_platform[platform] = {
            "status": "COMPLETE",
            "best_roots_per_hour": rate,
            "best_schedule_label": label,
            "nproc": next(iter(host_nproc)),
            "observed_transport_faults": 0,
        }
    cpu, l4 = by_platform["cpu"], by_platform["l4"]
    ratio = None
    if cpu["status"] == l4["status"] == "COMPLETE":
        ratio = cpu["best_roots_per_hour"] / l4["best_roots_per_hour"]
        verdict = "CPU_ALL_THREE_SLOTS" if ratio >= 0.25 else "L4_LABELS_CPU_OVERFLOW"
    elif cpu["status"] == "COMPLETE" and l4["status"] == "EXCLUDED_THIS_WEEK":
        verdict = "CPU_ALL_THREE_SLOTS_L4_EXCLUDED"
    elif l4["status"] == "COMPLETE" and cpu["status"] == "EXCLUDED_THIS_WEEK":
        verdict = "L4_ONLY_CPU_EXCLUDED"
    elif cpu["status"] == l4["status"] == "EXCLUDED_THIS_WEEK":
        verdict = "NO_PRODUCTION_PLATFORM"
    elif cpu["status"] == "COMPLETE" and l4["status"] == "NOT_RUN" and l4_not_run:
        verdict = "CPU_ONLY_L4_NOT_RUN"
    else:
        verdict = "INCOMPLETE"
    return {
        "schema": "bl-labels-f1-placement-decision/v1",
        "verdict": verdict,
        "platforms": by_platform,
        "cpu_to_l4_roots_per_hour": ratio,
        "cpu_threshold": 0.25,
        "rate_selection": "maximum complete schedule; ties one, half, all",
        "f0_admission": "SEPARATE_PENDING_COMPLETE_F0_AND_MEASURED_MAC_EQUIVALENCE",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, action="append", required=True)
    parser.add_argument("--l4-not-run", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bank_sha256 = bank_digest(args.bank)
    bank = load_bank(args.bank)
    attempts = [load_attempt(path, bank_sha256) for path in args.attempt]
    result = decide(attempts, [r["root_id"] for r in bank["roots"]], l4_not_run=args.l4_not_run)
    result["bank_sha256"] = bank_sha256
    result["attempts"] = attempts
    publish_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "attempts"}, sort_keys=True))


if __name__ == "__main__":
    main()
