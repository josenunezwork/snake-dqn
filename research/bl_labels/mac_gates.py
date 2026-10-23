"""Mac placement admission and the create-only, single registered Canary-B look.

Preparing or verifying a ledger never reads branch outcomes. Only ``look-canary``
invokes the registered aggregator, after publishing an immutable look claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import uuid
from pathlib import Path

from .exports import (
    _materialize_triplet,
    _verify,
    aggregate_attempts,
    analyze_aggregate,
    completed_session_parents,
    empty_ledger,
    load_ledger,
    publish_json,
)
from .f1_ruler import bank_digest, compare_mac_configs, load_bank
from .mac_package import CANARY_WALL_SECONDS, CANARY_WORKERS, F1_BANK_SHA256

ROOT = Path(__file__).resolve().parents[2]
GLOBAL_LOOK_ROOT = ROOT.parent / "snake-dqn-artifacts/bl-labels-f0/canary-b-single-look"
CODE_PATHS = tuple(
    "research/bl_labels/" + name
    for name in (
        "factory.py",
        "exports.py",
        "runtime.py",
        "f1_ruler.py",
        "f1_units.py",
        "mac_driver.py",
        "mac_package.py",
        "mac_gates.py",
        "replay_audit.py",
    )
)


def _read(path: Path) -> dict:
    _verify(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Mac gate expects a verified JSON object")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_pin(pack: Path) -> str:
    control = _read(pack / "control.json")
    pin = control.get("source_commit")
    if not isinstance(pin, str) or len(pin) != 40:
        raise ValueError("Mac package lacks frozen source commit")
    for name in CODE_PATHS:
        frozen = subprocess.check_output(["git", "show", f"{pin}:{name}"], cwd=ROOT)
        if frozen != (ROOT / name).read_bytes():
            raise ValueError(f"Mac executable source drift: {name}")
    return pin


def _f1(pack: Path, output: Path, pin: str) -> tuple[dict, dict]:
    bank_path = Path(_read(pack / "canary-b-manifest.json")["f1_bank_path"])
    if bank_digest(bank_path) != F1_BANK_SHA256:
        raise ValueError("foreign frozen F1 bank")
    bank = load_bank(bank_path)
    result = _read(output / "runtime-result.json")
    if (
        result.get("status") != "COMPLETE"
        or result.get("platform") != "mac"
        or result.get("source_commit") != pin
        or result.get("bank_sha256") != F1_BANK_SHA256
        or result.get("transport_probe") is not False
    ):
        raise ValueError("Mac F1 worker-invariance gate has not passed")
    configs = result.get("configs", [])
    for config in configs:
        if config.get("source_commit") != pin or config.get("bank_sha256") != F1_BANK_SHA256:
            raise ValueError("foreign Mac F1 cell source/bank")
    comparison = compare_mac_configs(configs, [r["root_id"] for r in bank["roots"]])
    if result.get("mac_equivalence") != comparison:
        raise ValueError("Mac F1 comparison receipt differs from verified cells")
    return result, comparison


def _approval(path: Path) -> None:
    from dataclasses import asdict

    from research.bl_residual.fit import FitRecipe
    from research.bl_residual.protocol import canonical_sha, require_judgment

    require_judgment(
        path, ROOT / "research/bl_residual/F4_REGISTRATION.md", canonical_sha(asdict(FitRecipe()))
    )


def _approved_control(pack: Path, stage: str, bindings: dict) -> dict:
    name = "control.json" if stage == "canary-b" else "d256-control.json"
    original = _read(pack / name)
    value = copy.deepcopy(original)
    if stage == "d256":
        look = _read(Path(bindings["f0"]["path"]))
        value["operational_canary_rate"] = (
            look["analysis"]["rescue_bearing_lives_per_mac_worker_hour"] * CANARY_WORKERS
        )
    value["mac_gate"] = {
        "stage": stage,
        "status": "APPROVED",
        "package_root": str(pack.resolve()),
        "original_control_sha256": _sha(pack / name),
        "bindings": bindings,
    }
    return value


def admit_canary(pack: Path, f1_output: Path, out: Path) -> dict:
    pin = _source_pin(pack)
    _f1(pack, f1_output, pin)
    path = f1_output / "runtime-result.json"
    value = _approved_control(
        pack, "canary-b", {"f1": {"path": str(path.resolve()), "sha256": _verify(path)}}
    )
    publish_json(out, value)
    return value


def require_admission(control: dict) -> None:
    """Recheck original control bytes and every admission dependency before launch."""
    gate = control.get("mac_gate")
    if gate is None:
        return  # Tiny development fixtures do not impersonate a registered package.
    if not isinstance(gate, dict) or gate.get("status") != "APPROVED":
        raise ValueError("Mac factory HOLD: required placement/F4 gates have not cleared")
    pack, stage = Path(gate["package_root"]), gate["stage"]
    if stage not in {"canary-b", "d256"}:
        raise ValueError("foreign Mac admission stage")
    pin = _source_pin(pack)
    bindings = gate["bindings"]
    f1 = bindings["f1"]
    f1_path = Path(f1["path"])
    if _verify(f1_path) != f1["sha256"]:
        raise ValueError("Mac F1 receipt changed after admission")
    _f1(pack, f1_path.parent, pin)
    if stage == "d256":
        look = bindings["f0"]
        look_path = Path(look["path"])
        if _verify(look_path) != look["sha256"]:
            raise ValueError("Mac F0 receipt changed after admission")
        _require_f0(pack, look_path, pin)
        approval = bindings["f4"]
        approval_path = Path(approval["path"])
        if _sha(approval_path) != approval["sha256"]:
            raise ValueError("F4 approval changed after admission")
        _approval(approval_path)
        audit = bindings["seed_audit"]
        if _verify(Path(audit["path"])) != audit["sha256"]:
            raise ValueError("D256 seed audit changed after admission")
        _seed_audit(pack)
    if control != _approved_control(pack, stage, bindings):
        raise ValueError("Mac admitted control differs from its frozen template")


def publish_ledger(pack: Path, attempts: list[Path], out: Path) -> dict:
    """Publish only verified wrappers/path/size/hash; do not calculate any outcome."""
    from .mac_driver import _verify_resume_metadata

    control = _read(pack / "control.json")
    namespace, allocation = control["namespace"], control["allocation"]
    value = empty_ledger(namespace, allocation)
    seen = set()
    for attempt in attempts:
        if list(attempt.glob("INVALID*.json")):
            raise ValueError("INVALID Mac attempt cannot contribute a ledger")
        if _read(attempt / "runtime-result.json").get("status") == "INVALID":
            raise ValueError("INVALID Mac attempt cannot contribute a ledger")
        for path in sorted((attempt / "lives").glob("*.json")):
            if not Path(str(path) + ".complete").is_file():
                continue
            name = path.stem
            if name in seen:
                raise ValueError("duplicate Mac completed unit across attempts")
            seen.add(name)
            digest = _verify(path)
            _materialize_triplet(path, out.parent / "lives" / path.name)
            value["units"].append(
                {
                    "unit_id": name,
                    "path": "lives/" + path.name,
                    "sha256": digest,
                    "size": path.stat().st_size,
                }
            )
    # Validate before the ledger itself becomes visible as completed evidence.
    temporary = out.parent / (".validation-" + uuid.uuid4().hex + ".json")
    publish_json(temporary, value)
    try:
        names = load_ledger(temporary, namespace, allocation)
        _verify_resume_metadata(temporary, namespace, allocation, names)
    finally:
        for suffix in ("", ".sha256", ".complete"):
            Path(str(temporary) + suffix).unlink(missing_ok=True)
    value["attempt_exports"] = [str(p.resolve()) for p in attempts]
    publish_json(out, value)
    return value


def _attempts(pack: Path, attempts: list[Path], pin: str) -> float:
    if not 1 <= len(attempts) <= 2 or len(set(p.resolve() for p in attempts)) != len(attempts):
        raise ValueError("Canary-B permits one original plus one mechanical completion")
    total = 0.0
    state_dirs: set[Path] = set()
    receipts = []
    for attempt in attempts:
        receipt = _read(attempt / "runtime-result.json")
        elapsed = receipt.get("elapsed_seconds")
        if (
            receipt.get("namespace") != _read(pack / "control.json")["namespace"]
            or receipt.get("workers") != CANARY_WORKERS
            or receipt.get("status") == "INVALID"
            or receipt.get("platform_attestation", {}).get("code_sha") != pin
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise ValueError("foreign, invalid or unmeasured Canary-B attempt")
        total += elapsed
        receipts.append(receipt)
        if not isinstance(receipt.get("state_dir"), str):
            raise ValueError("Canary-B lacks durable attempt accounting")
        state_dirs.add(Path(receipt["state_dir"]).resolve())
    if len(state_dirs) != 1:
        raise ValueError("Canary-B attempts must share one durable accounting state")
    state_dir = state_dirs.pop()
    starts = sorted(state_dir.glob("attempt-*.start.json"))
    terminal_names = set()
    outputs = set()
    state_total = 0.0
    for start_path in starts:
        start = _read(start_path)
        terminal_path = start_path.with_name(
            start_path.name.replace(".start.json", ".terminal.json")
        )
        terminal = _read(terminal_path)
        if (
            start.get("namespace") != _read(pack / "control.json")["namespace"]
            or start.get("workers") != CANARY_WORKERS
            or start.get("deadline_seconds") != CANARY_WALL_SECONDS
            or start.get("platform_attestation", {}).get("code_sha") != pin
        ):
            raise ValueError("foreign Canary-B durable start identity")
        output = Path(start["output"]).resolve()
        matching = [r for p, r in zip(attempts, receipts) if p.resolve() == output]
        if len(matching) != 1 or any(
            terminal.get(k) != matching[0].get(k)
            for k in (
                "namespace",
                "workers",
                "elapsed_seconds",
                "attempts_ledger",
            )
        ):
            raise ValueError("omitted or mismatched Canary-B attempt accounting")
        elapsed = terminal.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise ValueError("invalid Canary-B terminal wall charge")
        state_total += elapsed
        outputs.add(output)
        terminal_names.add(terminal_path.name)
    if outputs != {p.resolve() for p in attempts} or len(starts) != len(attempts):
        raise ValueError("omitted or foreign Canary-B attempts")
    if not any(set(r.get("attempts_ledger", [])) == terminal_names for r in receipts):
        raise ValueError("Canary-B does not bind its complete attempt chain")
    if abs(total - state_total) > 1e-6:
        raise ValueError("Canary-B wall charges do not reconcile")
    if total > CANARY_WALL_SECONDS:
        raise ValueError("Canary-B original three-hour wall cap exceeded")
    return CANARY_WORKERS * total / 3600


def look_canary(pack: Path, ledger: Path, attempts: list[Path], out: Path) -> dict:
    pin = _source_pin(pack)
    control = _read(pack / "control.json")
    value = _read(ledger)
    if value.get("attempt_exports") != [str(p.resolve()) for p in attempts]:
        raise ValueError("publish the ledger for these exact attempts before the F0 look")
    names = load_ledger(ledger, control["namespace"], control["allocation"])
    from .mac_driver import _verify_resume_metadata

    _verify_resume_metadata(ledger, control["namespace"], control["allocation"], names)
    if completed_session_parents(ledger, names) != {
        r["allocation_id"] for r in control["allocation"]
    }:
        raise ValueError("HOLD: Canary-B must be complete 36/36 before its single look")
    hours = _attempts(pack, attempts, pin)
    claim_path = pack / "f0-look-claim.json"
    global_claim = GLOBAL_LOOK_ROOT / (
        hashlib.sha256(control["namespace"].encode()).hexdigest() + ".json"
    )
    if claim_path.exists() or global_claim.exists():
        raise ValueError("F0 registered look already claimed; no re-look")
    claim = {
        "schema": "bl-labels-mac-f0-look-claim/v1",
        "nonce": uuid.uuid4().hex,
        "ledger": str(ledger.resolve()),
        "ledger_sha256": _verify(ledger),
        "out": str(out.resolve()),
        "source_commit": pin,
    }
    publish_json(global_claim, claim)
    publish_json(claim_path, claim)
    aggregate_path = out.parent / "registered-aggregate" / "verified-ledger.json"
    # The published ledger materializes exactly verified completed triplets.
    # Keep interrupted raw writes immutable; they cannot enter the registered
    # result set or consume the look through a second scan of partial exports.
    aggregate = aggregate_attempts(
        [ledger.parent], aggregate_path, control["namespace"], control["allocation"]
    )
    analysis = analyze_aggregate(aggregate, hours)
    result = {
        "schema": "bl-labels-mac-f0-look/v1",
        "source_commit": pin,
        "namespace": control["namespace"],
        "claim_sha256": _verify(claim_path),
        "ledger_sha256": _verify(ledger),
        "aggregate_sha256": _verify(aggregate_path),
        "worker_hours": hours,
        "analysis": analysis,
        "status": "PASS" if analysis["verdict"] == "COLLECTED" else analysis["verdict"],
    }
    publish_json(out, result)
    return result


def _require_f0(pack: Path, path: Path, pin: str) -> dict:
    value = _read(path)
    claim = _read(pack / "f0-look-claim.json")
    namespace = _read(pack / "control.json")["namespace"]
    global_claim = GLOBAL_LOOK_ROOT / (hashlib.sha256(namespace.encode()).hexdigest() + ".json")
    if _verify(global_claim) != _verify(pack / "f0-look-claim.json"):
        raise ValueError("F0 canonical global look claim differs")
    if (
        value.get("schema") != "bl-labels-mac-f0-look/v1"
        or value.get("source_commit") != pin
        or value.get("namespace") != _read(pack / "control.json")["namespace"]
        or value.get("status") != "PASS"
        or value.get("analysis", {}).get("verdict") != "COLLECTED"
        or value.get("claim_sha256") != _sha(pack / "f0-look-claim.json")
        or claim.get("out") != str(path.resolve())
    ):
        raise ValueError("D256 HOLD: registered complete Mac F0 look has not passed")
    if value.get("ledger_sha256") != _verify(Path(claim["ledger"])):
        raise ValueError("F0 published ledger changed after its look")
    if value.get("aggregate_sha256") != _verify(
        path.parent / "registered-aggregate/verified-ledger.json"
    ):
        raise ValueError("F0 registered aggregate changed after its look")
    return value


def admit_d256(pack: Path, f1_output: Path, f0_look: Path, approval: Path, out: Path) -> dict:
    pin = _source_pin(pack)
    _f1(pack, f1_output, pin)
    _require_f0(pack, f0_look, pin)
    _approval(approval)
    audit = _seed_audit(pack)
    value = _approved_control(
        pack,
        "d256",
        {
            "f1": {
                "path": str((f1_output / "runtime-result.json").resolve()),
                "sha256": _verify(f1_output / "runtime-result.json"),
            },
            "f0": {"path": str(f0_look.resolve()), "sha256": _verify(f0_look)},
            "f4": {"path": str(approval.resolve()), "sha256": _sha(approval)},
            "seed_audit": {"path": str(audit.resolve()), "sha256": _verify(audit)},
        },
    )
    publish_json(out, value)
    return value


def _seed_audit(pack: Path) -> Path:
    path = pack / "d256-seed-audit.json"
    audit = _read(path)
    allocation = _read(pack / "d256-control.json")["allocation"]
    if (
        audit.get("schema") != "bl-labels-f0-d256-seed-audit/v1"
        or audit.get("seeds") != [r["seed"] for r in allocation]
        or audit.get("outcome_blind") is not True
        or not audit.get("hits_by_root")
        or any(audit["hits_by_root"].values())
    ):
        raise ValueError("D256 HOLD: frozen allocation lacks a clean seed audit")
    return path


def replay_canary_a(pack: Path, out: Path) -> dict:
    """Audit the frozen 64 hash-selected x86 roots without opening label outcomes."""
    from .replay_audit import audit

    manifest = _read(pack / "canary-b-manifest.json")
    bank_path = Path(manifest["f1_bank_path"])
    if bank_digest(bank_path) != F1_BANK_SHA256:
        raise ValueError("foreign frozen Canary-A audit bank")
    bank = load_bank(bank_path)
    sources = {source["name"]: source for source in bank["source_sessions"]}
    records = [
        {**root, "tape": sources[root["source_session"]]["tape"][: root["tape_end"]]}
        for root in bank["roots"]
    ]
    checkpoint_dir = ROOT.parent / "snake-dqn/saved_snakes"
    try:
        result = audit(
            records,
            b8=checkpoint_dir / "b8_s3_t7_20261016.pth",
            frp3=checkpoint_dir / "frp3_m3_s12_u60000_20261005.pth",
        )
    except ValueError as error:
        result = {"schema": "bl-labels-f0-mac-replay/v1", "status": "INVALID", "reason": str(error)}
    result["mixing_status"] = (
        "X86_LABELS_MIXABLE" if result["status"] == "PASS" else "X86_LABELS_NOT_MIXABLE"
    )
    result["scope"] = "64 recorded-action roots; not a gate on Mac-only production"
    publish_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("admit-canary", "ledger", "look-canary", "admit-d256", "replay-canary-a"):
        p = sub.add_parser(command)
        p.add_argument("--pack", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if command in {"admit-canary", "admit-d256"}:
            p.add_argument("--f1-output", type=Path, required=True)
        if command in {"ledger", "look-canary"}:
            p.add_argument("--attempt", type=Path, action="append", required=True)
        if command == "look-canary":
            p.add_argument("--ledger", type=Path, required=True)
        if command == "admit-d256":
            p.add_argument("--f0-look", type=Path, required=True)
            p.add_argument("--f4-approval", type=Path, required=True)
    a = parser.parse_args()
    if a.command == "admit-canary":
        result = admit_canary(a.pack, a.f1_output, a.out)
    elif a.command == "ledger":
        result = publish_ledger(a.pack, a.attempt, a.out)
    elif a.command == "look-canary":
        result = look_canary(a.pack, a.ledger, a.attempt, a.out)
    elif a.command == "replay-canary-a":
        result = replay_canary_a(a.pack, a.out)
    else:
        result = admit_d256(a.pack, a.f1_output, a.f0_look, a.f4_approval, a.out)
    print(
        json.dumps(
            {"command": a.command, "out": str(a.out), "status": result.get("status", "PREPARED")}
        )
    )


if __name__ == "__main__":
    main()
