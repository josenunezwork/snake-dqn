"""Narrow P-only sandbox admission and metadata-only seed exclusion audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from research.watch_lead_p.protocol import FIXTURE_SEED, NAMESPACE, SEEDS, STUDY
from research.watch_lead_p.resume import UNIT_SUFFIXES, empty_ledger, read_ledger, unit_id
from tools.sandbox.registry import B8_SHA, DEFAULT_REGISTRY, digest, load_registry, world_seed

ROOT = Path(__file__).resolve().parents[2]

FRP3_SHA = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"


def validate_reforecast_envelope(packet: dict, accounting: dict | None, payload: dict) -> None:
    """Admit only the operator's pre-data 09c envelope beyond the legacy 480 minutes."""
    if packet["timeout_minutes"] == 480:
        return
    envelope = packet["technical_envelope"]
    if (
        packet["job_id"] != "sandbox-p-watch-lead-20261028-09c"
        or packet["timeout_minutes"] != 570
        or packet["requirements"]["est_minutes"] != 285
        or packet["budget"]["unit_cap"] != 14.63
        or packet["budget"]["rate_units_per_hour"] != 1.54
        or packet["budget"]["reserve_units"] != 0
        or envelope["base_seconds"] != 17100
        or envelope["effective_seconds"] != 34200
        or envelope["base_unit_cap"] != 7.312484270908601
        or envelope["effective_unit_cap"] != 14.63
        or envelope["base_reserve_units"] != 0
        or envelope["effective_reserve_units"] != 0
        or payload["timeout"]
        != {"base_seconds": 17100, "seconds": 34200, "purpose": "runaway-guard"}
        or accounting is None
        or accounting["envelope"]["base_minutes"] != 285
        or accounting["envelope"]["effective_minutes"] != 570
        or accounting["envelope"]["base_units"] != envelope["base_unit_cap"]
        or accounting["envelope"]["effective_units"] != 14.63
    ):
        raise ValueError("P envelope differs from the operator-authorized 09c reforecast")


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
    expected_seal = {
        "DISPATCH.txt",
        "dispatch-manifest.json",
        "exclusion-audit.json",
        "forecast.json",
        "sandbox.json",
        "verified-ledger.json",
    }
    packet = json.loads(packet_path.read_text())
    accounting = None
    accounting_ref = packet.get("technical_envelope", {}).get("mechanical_retry_accounting")
    if accounting_ref is not None:
        expected_seal.add("retry-accounting.json")
        accounting_path = packet_path.parent / "retry-accounting.json"
        if accounting_ref != {"path": str(accounting_path), "sha256": digest(accounting_path)}:
            raise ValueError("P retry accounting binding differs")
        accounting = json.loads(accounting_path.read_text())
        if (
            accounting["prepared_job_id"] != packet["job_id"]
            or accounting["envelope"]["forecast_seconds"]
            != packet["technical_envelope"]["forecast_seconds"]
            or accounting["envelope"]["governing"] != "effective"
        ):
            raise ValueError("P retry accounting/forecast differs")
    if set(seal) != expected_seal:
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
    runtime_sources = json.loads((ROOT / "research/watch_lead_p/runtime-sources.json").read_text())
    science_sources = json.loads((ROOT / "research/watch_lead_p/science-sources.json").read_text())
    for paths in (runtime_sources, science_sources, packet["source_paths"]):
        if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
            raise ValueError("P source inventory must list paths")
        if len(paths) != len(set(paths)):
            raise ValueError("P duplicate source inventory path")
    if (
        "research/watch_lead_p/science-sources.json" not in runtime_sources
        or not set(science_sources).issubset(runtime_sources)
        or set(packet["source_paths"]) != set(runtime_sources)
    ):
        raise ValueError("P sparse runtime must contain the complete science inventory")
    if (
        packet["device"] != "cpu"
        or packet["requirements"]["min_vcpu"] != 12
        or packet["technical_envelope"]["workers"] != 11
    ):
        raise ValueError("P worker/device/envelope differs from forecast")
    validate_reforecast_envelope(packet, accounting, payload)
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
    # Destination binding applies to empty first attempts and completions alike.
    uploads = {}
    for item in packet["inputs"]:
        dest = item.get("dest")
        if (
            not isinstance(dest, str)
            or not dest
            or dest in uploads
            or "\\" in dest
            or dest == "."
            or dest.endswith("/")
            or PurePosixPath(dest).as_posix() != dest
            or PurePosixPath(dest).is_absolute()
            or ".." in PurePosixPath(dest).parts
        ):
            raise ValueError("P missing/duplicate/unsafe upload destination")
        uploads[dest] = item
    ledger_upload = uploads.get(packet["resume"]["ledger_path"])
    if (
        ledger_upload is None
        or Path(ledger_upload["path"]).resolve() != ledger_path.resolve()
        or ledger_upload.get("sha256") != kinds["verified-ledger"][0]["sha256"]
    ):
        raise ValueError("P ledger upload must bind the sealed ledger destination")
    ledger = json.loads(ledger_path.read_text())
    expected = empty_ledger(ROOT)
    if not isinstance(ledger, dict) or set(ledger) != set(expected):
        raise ValueError("P outcome-free ledger schema differs")
    if not isinstance(ledger["units"], list):
        raise ValueError("P ledger units must be a list")
    if {k: v for k, v in ledger.items() if k != "units"} != {
        k: v for k, v in expected.items() if k != "units"
    }:
        raise ValueError("foreign namespace/allocation ledger")
    if len(uploads) != 4 + len(UNIT_SUFFIXES) * len(ledger["units"]):
        raise ValueError("P upload count differs from the sealed ledger")
    completion_root = ledger_path.parent
    if ledger != expected:
        # Bind shipped payloads and their actual adjacent sidecars without
        # opening outcomes. The shared runtime validator is the final check.
        if not 0 < len(ledger["units"]) < 32:
            raise ValueError("P completion requires 1..31 verified units")
        admitted = {(r["path"], r["sha256"]) for r in kinds["verified-unit"]}
        used = set()
        indices = set()
        for record in ledger["units"]:
            if not isinstance(record, dict) or set(record) != {
                "unit_id",
                "allocation_index",
                "path",
                "size",
                "sha256",
            }:
                raise ValueError("P outcome-free unit schema differs")
            index = record["allocation_index"]
            if type(index) is not int or index in indices or record["unit_id"] != unit_id(index):
                raise ValueError("duplicate/foreign P completion unit")
            indices.add(index)
            if record["path"] != f"units/session-{index:02d}.json":
                raise ValueError("foreign P completion path")
            upload = uploads.get(record["path"])
            if upload is None:
                raise ValueError("P missing completion upload destination")
            source = Path(upload["path"]).resolve()
            if len(indices) == 1:
                completion_root = source.parent.parent
            if source != (completion_root / record["path"]).resolve():
                raise ValueError("P completion sources must share the ledger root")
            for suffix in UNIT_SUFFIXES:
                item = uploads.get(record["path"] + suffix)
                if item is None:
                    raise ValueError("P missing completion sidecar destination")
                if Path(item["path"]).resolve() != Path(f"{source}{suffix}").resolve():
                    raise ValueError("P completion sidecar must be adjacent to its payload")
                pair = (item["path"], item["sha256"])
                if pair not in admitted:
                    raise ValueError("unadmitted P completion payload/sidecar")
                used.add(pair)
        if used != admitted:
            raise ValueError("unreferenced P completion upload")
    if not ledger["units"] and kinds["verified-unit"]:
        raise ValueError("unreferenced P resume payloads")
    read_ledger(ledger_path, completion_root, recipe_root=ROOT)

    indices = {record["allocation_index"] for record in ledger["units"]}
    expected_exports = ["watch-lead/inventory.json"] + [
        f"watch-lead/units/session-{i:02d}.json{suffix}"
        for i in range(32)
        if i not in indices
        for suffix in ("", ".metadata.json")
    ]
    if packet.get("required_exports") != expected_exports:
        raise ValueError("P required exports differ from missing units")
