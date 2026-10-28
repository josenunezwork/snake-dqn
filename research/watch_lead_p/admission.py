"""Narrow P-only sandbox admission and metadata-only seed exclusion audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from research.watch_lead_p.protocol import FIXTURE_SEED, NAMESPACE, SEEDS, STUDY
from research.watch_lead_p.resume import empty_ledger
from tools.sandbox.registry import B8_SHA, DEFAULT_REGISTRY, digest, load_registry, world_seed

FRP3_SHA = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"


def exclusion_audit() -> dict[str, Any]:
    """Audit registry metadata and canonical TRAIN-01 monitor/nomination identities."""
    registry = load_registry()
    protected = set(registry["excluded_seeds"])
    monitor = []
    for start, count in ((400_000_000, 288), (500_000_000, 768)):
        for index in range(start, start + count):
            raw = hashlib.sha256(f"a100-dev/sandbox-a-train01/{index}".encode()).digest()[:8]
            # TRAIN-01's actual key mapping, including its existing low-63 mask.
            monitor.append(int.from_bytes(raw, "big") & 0x7FFF_FFFF_FFFF_FFFF)
    fixtures = [world_seed(STUDY, i) for i in range(32, 36)]
    assert FIXTURE_SEED == fixtures[0]
    collisions = sorted((set(SEEDS) | set(fixtures)) & (protected | set(monitor)))
    if collisions or len(set(SEEDS) | set(fixtures)) != 36:
        raise ValueError("P seed exclusion intersection")
    return {
        "namespace": NAMESPACE,
        "registry_sha256": digest(DEFAULT_REGISTRY),
        "registry_excluded_seed_count": len(protected),
        "train01_monitor_nomination_count": len(monitor),
        "train01_key_schema": "sha256-first8-big-endian-low63; offsets 400000000/500000000",
        "seeds": list(SEEDS),
        "excluded_engineering_indices": list(range(32, 36)),
        "excluded_engineering_seeds": fixtures,
        "collisions": collisions,
        "boundary": (
            "metadata-only; no protected payloads opened; " "no exhaustive historical-bank claim"
        ),
    }


def validate_inputs(payload: dict, registry: dict) -> None:
    """Bind the exact finite allocation, roster, resume ledger and terminal endpoint."""
    kinds = {
        kind: [r for r in payload["inputs"] if r["kind"] == kind]
        for kind in (
            "worlds",
            "frozen-checkpoint",
            "frozen-opponent",
            "registry",
            "verified-ledger",
            "verified-unit",
        )
    }
    if any(len(kinds[k]) != 1 for k in kinds if k != "verified-unit"):
        raise ValueError("P requires exactly one world allocation, B8, FRP3, registry and ledger")
    if set(r["kind"] for r in payload["inputs"]) - set(kinds):
        raise ValueError("P forbids data banks and foreign input kinds")
    worlds = kinds["worlds"][0]
    if worlds != {
        "kind": "worlds",
        "study": STUDY,
        "indices": list(range(32)),
        "seeds": list(SEEDS),
    }:
        raise ValueError("P allocation differs from frozen 32 worlds")
    if payload["salt"] != NAMESPACE or payload["endpoint"] != {"kind": "steps", "value": 576000}:
        raise ValueError("P namespace/endpoint mismatch")
    if (
        kinds["frozen-checkpoint"][0]["sha256"] != B8_SHA
        or kinds["frozen-opponent"][0]["sha256"] != FRP3_SHA
    ):
        raise ValueError("P requires exactly B8 s3 and FRP3-s12")
    if set(SEEDS) & set(registry["excluded_seeds"]):
        raise ValueError("P protected seed intersection")
    audit = exclusion_audit()
    packet_path = Path(payload["outputs"]["broker_manifest"]["path"])
    from tools.sandbox.preflight import verify_committed_manifest

    verify_committed_manifest(packet_path.parent / "seal.json")
    seal = json.loads((packet_path.parent / "seal.json").read_text())
    if set(seal) != {
        "DISPATCH.txt",
        "dispatch-manifest.json",
        "exclusion-audit.json",
        "forecast.json",
        "sandbox.json",
        "verified-ledger.json",
    }:
        raise ValueError("P packet seal is incomplete")
    for name, checksum in seal.items():
        path = packet_path.parent / name
        verify_committed_manifest(path)
        if digest(path) != checksum:
            raise ValueError("P packet seal differs")
    expected_audit = json.loads((packet_path.parent / "exclusion-audit.json").read_text())
    if expected_audit != audit:
        raise ValueError("P committed registry exclusion audit differs")
    packet = json.loads(packet_path.read_text())
    if (
        packet["device"] != "cpu"
        or packet["requirements"]["min_vcpu"] != 12
        or packet["technical_envelope"]["workers"] != 12
        or packet["timeout_minutes"] != 480
    ):
        raise ValueError("P worker/device/envelope differs from forecast")
    if packet["resume"] != {
        "unit_kind": "paired-Watch6-session",
        "ledger_path": "verified-ledger.json",
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }:
        raise ValueError("P resume declaration differs")
    ledger_path = Path(kinds["verified-ledger"][0]["path"])
    if ledger_path.resolve() != (packet_path.parent / "verified-ledger.json").resolve():
        raise ValueError("P ledger must belong to the sealed packet")
    # Admission checks first attempt without opening any outcomes. Completion
    # packets must preserve the same namespace and pass read_ledger offline.
    ledger = json.loads(ledger_path.read_text())
    if ledger != empty_ledger():
        # Materialize no outcomes: bind the shipped bytes and completion
        # sidecars to each path/size/hash-only ledger entry.
        from research.bl_served_search.exports import verify_export
        from research.watch_lead_p.resume import unit_id

        expected = empty_ledger()
        if {k: ledger.get(k) for k in expected if k != "units"} != {
            k: expected[k] for k in expected if k != "units"
        }:
            raise ValueError("foreign P completion allocation/recipe")
        if not 0 < len(ledger["units"]) < 32:
            raise ValueError("P completion requires 1..31 verified units")
        uploads = {}
        for item in packet["inputs"]:
            dest = item.get("dest")
            if not isinstance(dest, str) or not dest or dest in uploads:
                raise ValueError("P missing/duplicate upload destination")
            uploads[dest] = item
        admitted = {(r["path"], r["sha256"]) for r in kinds["verified-unit"]}
        used = set()
        indices = set()
        for record in ledger["units"]:
            index = record["allocation_index"]
            if index in indices or record["unit_id"] != unit_id(index):
                raise ValueError("duplicate/foreign P completion unit")
            indices.add(index)
            if record["path"] != f"units/session-{index:02d}.json":
                raise ValueError("foreign P completion path")
            upload = uploads.get(record["path"])
            if upload is None:
                raise ValueError("P missing completion upload destination")
            source = Path(upload["path"])
            if source.stat().st_size != record["size"] or verify_export(source) != record["sha256"]:
                raise ValueError("unverified P completion unit")
            for suffix in ("", ".sha256", ".complete"):
                item = uploads.get(record["path"] + suffix)
                if item is None:
                    raise ValueError("P missing completion sidecar destination")
                pair = (item["path"], item["sha256"])
                if pair not in admitted:
                    raise ValueError("unadmitted P completion payload/sidecar")
                used.add(pair)
        if used != admitted:
            raise ValueError("unreferenced P completion upload")
    if not ledger["units"] and kinds["verified-unit"]:
        raise ValueError("unreferenced P resume payloads")

    indices = {record["allocation_index"] for record in ledger["units"]}
    expected_exports = ["watch-lead/inventory.json"] + [
        f"watch-lead/units/session-{i:02d}.json" for i in range(32) if i not in indices
    ]
    if packet.get("required_exports") != expected_exports:
        raise ValueError("P required exports differ from missing units")
