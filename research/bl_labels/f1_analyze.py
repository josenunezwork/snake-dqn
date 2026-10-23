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

# This is deliberately a byte binding, rather than a policy selected by a
# caller-provided JSON file.  The narrow exception below exists only because
# this particular ruling preserved two immutable, pre-replay packaging exits.
REFUND_CENSUS_SCHEMA = "bl-labels-f1-refund-census/v1"
REFUND_RULING_SHA256 = "f2f35631e65e4234341a5404e3f512a740b43241c9060f4cf8a467673091b6bc"
REFUND_A1_RULING_SHA256 = "86e9ca69383d21a45ede31fd721d315ff39ae09736ad538496ce42b1abf1ac72"
REFUND_EXCLUSION = "PACKAGING_INVALID_EXCLUDED (JUDGMENT_F1_REFUND)"
REFUND_WITNESS_RECEIPT_SHA256 = "3386d25db932cff5835aca7feb4462c7d8af42700700e9ca59b2d8f7d6e0c29d"
REFUND_PACK08_NAMES = {"cpu-probe", "l4-probe"}
REFUND_PACK08_EVIDENCE = {
    "cpu-probe": {
        "receipt_sha256": "3a0ec4f8e4049fd73646057484296710f5539d0a233b781641bf163006e78707",
        "manifest_sha256": "85f35b6b52b27b7323ae679570277bffe89937afa6bffec167244cbdef25d570",
        "runtime_sha256": "a394fbc0e8dbeeb03090994eb28eebb1ee57a243e4845b0ec53a57abc89fccc9",
        "runtime_size": 197,
    },
    "l4-probe": {
        "receipt_sha256": "0361d20d462932408ebb85875dd9df66184007298a500c52781402d636a0b751",
        "manifest_sha256": "e0d48465b7ec945e6e026687802951a8949dd3f6e0b42a5e90099fa12d2fea97",
        "runtime_sha256": "37235e03b93e2f182c8029f5b8e7c035d0b8928e610583faa20de03718db9fb8",
        "runtime_size": 196,
    },
}
REFUND_PACK08_SOURCE_COMMIT = "70ead024c3552d104d4916ea38db667df87fc285"
REFUND_PACK08_BANK_SHA256 = "a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("F1 timing must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("F1 timing must be finite and positive")
    return float(value)


def _is_refund_upload_carveout(receipt: dict[str, Any], index: int, call: dict[str, Any]) -> bool:
    """Return whether this is the qualified recovered pre-delivery upload.

    The identity is intentionally much narrower than "a failed upload": it is
    the zero-based index-6 first-attempt upload in its registered 60-second
    slot, and it must have returned before that slot timed out.  This exact
    signature is present in the hash-bound qualified canary witness.
    """
    generation = receipt.get("generation")
    transfers = receipt.get("contents_transfers")
    if not isinstance(generation, str) or not generation or not isinstance(transfers, list):
        return False
    if len(transfers) < 2 or not all(isinstance(item, dict) for item in transfers[:2]):
        return False
    calls = receipt.get("cli_calls")
    expected_prefix = ("version", "sessions", "usage", "new", "exec", "upload", "upload", "exec")
    if (
        not isinstance(calls, list)
        or len(calls) < len(expected_prefix)
        or any(not isinstance(item, dict) for item in calls[: len(expected_prefix)])
        or tuple(item.get("command") for item in calls[: len(expected_prefix)]) != expected_prefix
    ):
        return False
    release, heartbeat = transfers[:2]
    if (
        release.get("direction") != "upload"
        or release.get("status") != "completed"
        or release.get("remote") != f"/content/snake-cli-{generation}-release.json"
        or heartbeat.get("direction") != "upload"
        or heartbeat.get("status") != "failed"
        or heartbeat.get("size_bytes") is not None
        or heartbeat.get("remote") != f"/content/snake-cli/{generation}/lease-heartbeat.json"
        or not isinstance(heartbeat.get("elapsed_seconds"), (int, float))
        or isinstance(heartbeat.get("elapsed_seconds"), bool)
    ):
        return False
    return (
        index == 6
        and call.get("attempt") == 1
        and call.get("command") == "upload"
        and call.get("status") == "failed"
        and isinstance(call.get("timeout_seconds"), (int, float))
        and not isinstance(call.get("timeout_seconds"), bool)
        and 59.0 <= float(call["timeout_seconds"]) <= 60.0
        and isinstance(call.get("elapsed_seconds"), (int, float))
        and not isinstance(call.get("elapsed_seconds"), bool)
        and 0.0 <= float(call["elapsed_seconds"]) < float(call["timeout_seconds"])
        and abs(float(call["elapsed_seconds"]) - float(heartbeat["elapsed_seconds"])) < 0.1
    )


def transport_faults(
    receipt: dict[str, Any], *, allow_refund_upload_carveout: bool = False
) -> list[dict[str, Any]]:
    """Exclude any observed failed/budget-exhausting CLI call, even if recovered.

    This conservative operational rule also excludes unresolved transport faults;
    it never interprets a slow scientific root as proof of an RPC stall.
    """
    calls = receipt.get("cli_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("F1 needs the supervisor CLI call census")
    faults = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError("F1 CLI census is malformed")
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
        if allow_refund_upload_carveout and _is_refund_upload_carveout(receipt, index, call):
            continue
        if call["status"] != "completed" or elapsed >= timeout:
            faults.append(call)
    return faults


def load_attempt(
    path: Path, bank_sha256: str, *, allow_refund_upload_carveout: bool = False
) -> dict[str, Any]:
    """Bind an immutable runtime export to its terminal supervisor receipt."""
    receipt_path = path / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    faults = transport_faults(receipt, allow_refund_upload_carveout=allow_refund_upload_carveout)
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
    result["transport_probe"] = runtime.get("transport_probe") is True
    # Probe receipts participate in the fault census, never in the 64-root
    # placement matrix. Their three one-root diagnostic schedules are excluded.
    result["complete"] = runtime.get("status") == "COMPLETE" and not result["transport_probe"]
    result["configs"] = [] if result["transport_probe"] else runtime.get("configs", [runtime])
    result["runtime_sha256"] = digest
    return result


def _load_refund_census(path: Path, attempt_paths: list[Path]) -> dict[str, Any]:
    """Validate the one ruled F1 exception before reading any attempt outcome."""
    census = json.loads(path.read_text())
    if not isinstance(census, dict) or set(census) != {
        "schema",
        "ruling",
        "qualification_witness",
        "excluded_attempts",
    }:
        raise ValueError("F1 refund census schema differs from the ruling")
    if census.get("schema") != REFUND_CENSUS_SCHEMA:
        raise ValueError("F1 refund census has a foreign schema")
    ruling = census.get("ruling")
    if not isinstance(ruling, dict) or set(ruling) != {"path", "sha256"}:
        raise ValueError("F1 refund census ruling binding is malformed")
    ruling_path = Path(ruling["path"])
    # A1 retains the pre-amendment authority for the census and requires its
    # bytes unchanged. Only that exact ratified amendment may replace the
    # on-disk ruling; this does not authorize arbitrary future amendments.
    if ruling.get("sha256") != REFUND_RULING_SHA256 or _sha(ruling_path) not in {
        REFUND_RULING_SHA256,
        REFUND_A1_RULING_SHA256,
    }:
        raise ValueError("F1 refund census is not bound to the ruled judgment")
    witness = census.get("qualification_witness")
    if not isinstance(witness, dict) or set(witness) != {"path", "receipt_sha256"}:
        raise ValueError("F1 refund census qualification witness is malformed")
    witness_path = Path(witness["path"])
    if (
        witness.get("receipt_sha256") != REFUND_WITNESS_RECEIPT_SHA256
        or _sha(witness_path) != REFUND_WITNESS_RECEIPT_SHA256
    ):
        raise ValueError("F1 refund upload carve-out lacks its qualified PASS witness")
    witness_receipt = json.loads(witness_path.read_text())
    if witness_receipt.get("status") != "PASS" or witness_receipt.get("stop_verified") is not True:
        raise ValueError("F1 refund upload witness is not a verified PASS")
    witness_calls = witness_receipt.get("cli_calls")
    if not isinstance(witness_calls, list) or len(witness_calls) <= 6:
        raise ValueError("F1 refund upload witness lacks the qualified call census")
    if not _is_refund_upload_carveout(witness_receipt, 6, witness_calls[6]):
        raise ValueError("F1 refund upload witness does not contain the ruled call identity")

    excluded = census.get("excluded_attempts")
    if not isinstance(excluded, list) or len(excluded) != 2:
        raise ValueError("F1 refund census must preserve exactly both pack-08 probes")
    expected_paths = {item.resolve() for item in attempt_paths}
    seen: set[Path] = set()
    for item in excluded:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "receipt_sha256",
            "manifest_sha256",
            "runtime_sha256",
            "classification",
        }:
            raise ValueError("F1 refund census excluded attempt is malformed")
        attempt_path = Path(item["path"]).resolve()
        if attempt_path not in expected_paths or attempt_path.name not in REFUND_PACK08_NAMES:
            raise ValueError("F1 refund census has a foreign or omitted pack-08 attempt")
        if attempt_path in seen:
            raise ValueError("F1 refund census duplicates a pack-08 attempt")
        seen.add(attempt_path)
        if item.get("classification") != REFUND_EXCLUSION:
            raise ValueError("F1 refund census altered the ruled classification")
        receipt_path = attempt_path / "receipt.json"
        manifest_path = attempt_path / "manifest.json"
        runtime_path = attempt_path / "exports" / "runtime-result.json"
        pinned = REFUND_PACK08_EVIDENCE[attempt_path.name]
        if (
            item.get("receipt_sha256") != _sha(receipt_path)
            or item.get("manifest_sha256") != _sha(manifest_path)
            or item.get("runtime_sha256") != _sha(runtime_path)
            or any(item.get(key) != value for key, value in pinned.items() if key != "runtime_size")
        ):
            raise ValueError("F1 refund census excluded evidence hash differs")
        receipt = json.loads(receipt_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        runtime = json.loads(runtime_path.read_text())
        inventory = [
            row for row in receipt.get("exports", []) if row.get("path") == "runtime-result.json"
        ]
        if (
            receipt.get("status") != "INCOMPLETE"
            or receipt.get("failure_class") != "CHILD_NONZERO_EXIT"
            or receipt.get("stop_verified") is not True
            or receipt.get("cleanup_error")
            or receipt.get("manifest_sha256") != _sha(manifest_path)
            or manifest.get("commit") != REFUND_PACK08_SOURCE_COMMIT
            or runtime.get("bank_sha256") != REFUND_PACK08_BANK_SHA256
            or runtime.get("status") != "INVALID"
            or runtime.get("reason") != "F0 export digest mismatch"
            or runtime.get("platform") != attempt_path.name.removesuffix("-probe")
            or _verify(runtime_path) != pinned["runtime_sha256"]
            or len(inventory) != 1
            or inventory[0].get("sha256") != pinned["runtime_sha256"]
            or inventory[0].get("size") != pinned["runtime_size"]
        ):
            raise ValueError("F1 refund census attempt is not the ruled packaging INVALID")
    if {path.name for path in seen} != REFUND_PACK08_NAMES:
        raise ValueError("F1 refund census omitted a ruled pack-08 probe")
    return census


def _load_ruled_excluded_attempt(
    path: Path, census_entry: dict[str, Any], bank_sha256: str
) -> dict[str, Any]:
    """Represent a preserved pack-08 INVALID without admitting it as timing data."""
    receipt_path = path / "receipt.json"
    manifest_path = path / "manifest.json"
    runtime_path = path / "exports" / "runtime-result.json"
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    if _verify(runtime_path) != census_entry["runtime_sha256"]:
        raise ValueError("F1 refund excluded runtime sidecar differs from census")
    platform = manifest.get("env", {}).get("F1_PLATFORM")
    if (
        platform not in {"cpu", "l4"}
        or runtime.get("platform") != platform
        or runtime.get("bank_sha256") != bank_sha256
        or receipt.get("manifest_sha256") != _sha(manifest_path)
    ):
        raise ValueError("F1 refund excluded attempt has foreign bank or platform")
    return {
        "platform": platform,
        "receipt_sha256": _sha(receipt_path),
        "runtime_sha256": _verify(runtime_path),
        "classification": REFUND_EXCLUSION,
        "complete": False,
        "configs": [],
        "transport_faults": [],
    }


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
    parser.add_argument("--refund-census", type=Path)
    parser.add_argument("--l4-not-run", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bank_sha256 = bank_digest(args.bank)
    bank = load_bank(args.bank)
    attempt_paths = [path.resolve() for path in args.attempt]
    if len(attempt_paths) != len(set(attempt_paths)):
        raise ValueError("duplicate F1 attempt path")
    excluded_by_path: dict[Path, dict[str, Any]] = {}
    if args.refund_census:
        census = _load_refund_census(args.refund_census, attempt_paths)
        excluded_by_path = {
            Path(item["path"]).resolve(): item for item in census["excluded_attempts"]
        }
        if len(attempt_paths) < 6:
            raise ValueError(
                "F1 refund analysis requires both pack-08 and all four pack-10 attempts"
            )
        initial_pack10 = {"cpu-probe", "cpu-ruler", "l4-probe", "l4-ruler"}
        normal_names = {path.name for path in attempt_paths if path not in excluded_by_path}
        if not initial_pack10 <= normal_names:
            raise ValueError("F1 refund analysis omitted an initial pack-10 probe or ruler")
    attempts = [
        (
            _load_ruled_excluded_attempt(path, excluded_by_path[path], bank_sha256)
            if path in excluded_by_path
            else load_attempt(
                path, bank_sha256, allow_refund_upload_carveout=bool(args.refund_census)
            )
        )
        for path in attempt_paths
    ]
    result = decide(attempts, [r["root_id"] for r in bank["roots"]], l4_not_run=args.l4_not_run)
    result["bank_sha256"] = bank_sha256
    result["attempts"] = attempts
    publish_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "attempts"}, sort_keys=True))


if __name__ == "__main__":
    main()
