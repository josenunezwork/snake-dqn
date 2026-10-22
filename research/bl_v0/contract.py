"""Frozen V0 accounting, premises, source identity and admission checks."""

from __future__ import annotations

import datetime as dt
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from research.bl_k1.prepare import sources as k1_sources
from research.bl_mac_latency.ruler_run import distribution
from research.bl_served_search import ruler
from research.bl_served_search.exports import digest, export_json, verify_export

ROOT = Path(__file__).resolve().parents[2]
CELLS = ("C1", "C2", "C3", "C4")
FIXTURES = {
    f"{mix}-{length}" for mix in ("frozen", "mixed", "scripted") for length in (800, 1200, 1500)
}
DEFAULT_FIXTURES = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/"
    "ruler-fixtures-08-final/ruler-fixtures.json"
)
FIXTURE_SHA256 = "bbe6975642512091ba279d9ff7c870000d68faa18e30063807f47af1352e8574"
AUTHORITY = {
    "round13_judgment_sha256": "457a810f8ee69d4360a91113828b8871682d3692e8924e3043801c83422c011d",
    "round12_judgment_sha256": "cd5dc0932255ba897032018ff9608d7e1a89d63dda1a0d21f82d043f4c95d031",
}


def sources() -> dict[str, str]:
    """Bind the actual source closure without altering K-1's frozen package."""
    paths = [*Path(__file__).parent.glob("*.py"), Path(__file__).parent / "run_mac.sh"]
    paths += [Path(__file__).parent / "REGISTRATION.md"]
    paths += [ROOT / "research/bl_served_search/mac_lease.py"]
    return {**k1_sources(), **{str(p.relative_to(ROOT)): digest(p) for p in paths}}


def published() -> dict[str, str]:
    """Refuse unpublished source/ledger bytes before a look or canary."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True
    )
    paths = list(sources())
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True
    )
    if dirty:
        raise RuntimeError("V0 source closure contains uncommitted/untracked bytes")
    ledger = "research/bl_v0/REGISTRATION.md"
    remote = subprocess.check_output(["git", "show", f"origin/main:{ledger}"], cwd=ROOT)
    if __import__("hashlib").sha256(remote).hexdigest() != digest(ROOT / ledger):
        raise RuntimeError("local registration differs from pushed ledger")
    return {"commit": commit, "pushed_registration_sha256": digest(ROOT / ledger)}


def prepare(fixtures: Path, out: Path) -> Path:
    """Freeze existing Y1 roots and current published source without running them."""
    ledger = published()
    if verify_export(fixtures) != FIXTURE_SHA256:
        raise ValueError("V0 requires the original nine pinned Y1 fixture bytes")
    bank = json.loads(fixtures.read_text())
    rows = ruler.validate_fixtures(bank)
    if {row["fixture_id"] for row in rows} != FIXTURES:
        raise ValueError("V0 fixture coverage differs")
    out.mkdir(parents=True, exist_ok=False)
    path = out / "manifest.json"
    export_json(
        path,
        {
            "schema": "bl-v0-manifest/v1",
            "ledger": ledger,
            "authority": AUTHORITY,
            "sources": sources(),
            "fixture_export": {"path": str(fixtures.resolve()), "sha256": FIXTURE_SHA256},
            "fixtures": [{k: v for k, v in row.items() if k != "root_bytes_b64"} for row in rows],
            "cells": list(CELLS),
            "batches": 30,
            "max_seconds": 3600,
            "forecast_seconds": [2100, 3000],
            "forecast_basis": "round-13 §5 projection; NOT_READY(V0), not measured",
            "units": 0,
        },
    )
    return path


def verify_manifest(path: Path) -> dict[str, Any]:
    """Reject source, registration, protocol or fixture substitution."""
    verify_export(path)
    value = json.loads(path.read_text())
    if (
        value["schema"] != "bl-v0-manifest/v1"
        or value["sources"] != sources()
        or value.get("authority") != AUTHORITY
    ):
        raise ValueError("V0 source closure drift: prepare a fresh attempt")
    if value["cells"] != list(CELLS) or value["batches"] != 30 or value["max_seconds"] != 3600:
        raise ValueError("V0 frozen design changed")
    bank_path = Path(value["fixture_export"]["path"])
    if verify_export(bank_path) != FIXTURE_SHA256:
        raise ValueError("original fixture bytes changed")
    expected = ruler.validate_fixtures(json.loads(bank_path.read_text()))
    expected = [{k: v for k, v in row.items() if k != "root_bytes_b64"} for row in expected]
    if value["fixtures"] != expected:
        raise ValueError("manifest root identities differ from original Y1")
    published()
    return value


def window(now: dt.datetime | None = None, *, smoke: bool = False) -> float:
    """Return seconds to the allowed end, excluding K-1/X5 even for smoke."""
    now = now or dt.datetime.now(ZoneInfo("America/Los_Angeles"))
    now = now.astimezone(ZoneInfo("America/Los_Angeles"))
    date = now.date().isoformat()
    if (date == "2026-10-22" and 20 <= now.hour < 22) or (
        date == "2026-10-23" and 9 <= now.hour < 11
    ):
        raise RuntimeError("protected K-1/X5 window")
    if smoke:
        protected_starts = [
            dt.datetime(2026, 10, 22, 20, tzinfo=now.tzinfo),
            dt.datetime(2026, 10, 23, 9, tzinfo=now.tzinfo),
        ]
        return min(
            [180.0, *[(start - now).total_seconds() for start in protected_starts if start > now]]
        )
    end_hour = None
    if date == "2026-10-23" and now.hour == 22:
        end_hour = 23
    if date == "2026-10-24" and 8 <= now.hour < 10:
        end_hour = 10
    if end_hour is None:
        raise RuntimeError("V0 timing: Claude only, Fri 22-23 or Sat 08-10 PT")
    return (now.replace(hour=end_hour, minute=0, second=0, microsecond=0) - now).total_seconds()


def attestation(row: dict[str, Any]) -> list[str]:
    """Enforce executed branch counters literally, including zero-eligible fixtures."""
    stats = row["native_v8"]
    failures = []
    if stats.get("kernel_rows", 0) <= 0:
        failures.append("kernel_rows_not_positive")
    for key in ("stock_rows", "fallback_rows", "noncandidate_rows", "replayed_rows"):
        if stats.get(key, -1) != 0:
            failures.append(key)
    if stats.get("mode") != "kernel":
        failures.append("native_mode")
    if stats.get("rows") != stats.get("kernel_rows"):
        failures.append("row_reconciliation")
    return failures


def validate_execution(row: dict[str, Any], fixture: dict[str, Any]) -> None:
    """Reconcile observed branches and leaf work before publishing a batch claim."""
    semantic = row["semantic"]
    branches = semantic["branches"]
    legal = set(fixture["legal_actions"])
    if len(branches) != len(legal) or {b["action"] for b in branches} != legal:
        raise ValueError("branch inventory differs from frozen legal actions")
    evaluated = {b["action"] for b in branches if b["state"] == "EVALUATED"}
    skipped = {b["action"] for b in branches if b["state"] == "NOT_EVALUATED"}
    if evaluated & skipped or evaluated | skipped != legal or not evaluated:
        raise ValueError("branch evaluation state invalid")
    if (
        set(semantic["branches_evaluated"]) != evaluated
        or {b["action"] for b in semantic["branches_skipped"]} != skipped
    ):
        raise ValueError("branch counts do not reconcile")
    if row["cell"] in ("C1", "C3") and skipped:
        raise ValueError("all-branch cell skipped work")
    cap = 64 if row["cell"] in ("C3", "C4") else 300
    for branch in branches:
        frames = branch["frames_executed"]
        if (branch["state"] == "EVALUATED" and not 1 <= frames <= cap) or (
            branch["state"] == "NOT_EVALUATED" and frames != 0
        ):
            raise ValueError("executed branch frames outside registered horizon")
    if semantic["bare"] not in legal or semantic["chosen"] not in evaluated:
        raise ValueError("bare/chosen action outside executed legal branches")
    dummy = row["dummy_leaf"]
    if cap == 64:
        if (
            dummy["calls"] != 1
            or dummy["batch_size"] != len(legal)
            or dummy["parameters"] != 12806
            or dummy["dtype"] != "float32"
            or dummy["device"] != "cpu"
            or dummy["torch_intraop_threads"] != 1
            or dummy["torch_interop_threads"] != 1
            or row["leaf_call_wall_seconds"] <= 0
        ):
            raise ValueError("dummy leaf call/shape/CPU thread attestation invalid")
    elif dummy["calls"] != 0:
        raise ValueError("exact cell unexpectedly charged a leaf call")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain cold values and the complete warm29 wall/stage distributions."""
    result = {}
    for fixture in sorted(FIXTURES):
        for cell in CELLS:
            group = sorted(
                [r for r in rows if r["fixture_id"] == fixture and r["cell"] == cell],
                key=lambda r: r["repeat"],
            )
            if not group:
                continue
            metrics = {"operator_wall_seconds": [r["operator_wall_seconds"] for r in group]}
            metrics.update(
                {key: [r["stage_seconds"][key] for r in group] for key in group[0]["stage_seconds"]}
            )
            result[f"{fixture}/{cell}"] = {
                key: {
                    "cold": values[0],
                    "all30": distribution(values),
                    "warm29": {
                        **distribution(values[1:]),
                        "median": statistics.median(values[1:]) if len(values) > 1 else None,
                    },
                }
                for key, values in metrics.items()
            }
    return result


def decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Literal registered §5(c), never a post-data reinterpretation."""
    keys = [(r["fixture_id"], r["cell"], r["repeat"]) for r in rows]
    expected = {
        (fixture, cell, repeat) for fixture in FIXTURES for cell in CELLS for repeat in range(30)
    }
    if len(keys) != len(set(keys)) or any(
        attestation(r) or r.get("identity_ok") is not True for r in rows
    ):
        return {"status": "INVALID", "admits": "NONE"}
    if any(
        not math.isfinite(r["operator_wall_seconds"]) or r["operator_wall_seconds"] <= 0
        for r in rows
    ):
        return {"status": "INVALID", "admits": "NONE"}
    if set(keys) != expected:
        return {"status": "INCOMPLETE", "admits": "NONE"}
    metrics = summarize(rows)
    cap = {
        fixture: metrics[f"{fixture}/C3"]["operator_wall_seconds"]["warm29"]["p99"]
        for fixture in ("frozen-800", "frozen-1500")
    }
    c2 = metrics["frozen-800/C2"]["operator_wall_seconds"]["warm29"]["median"]
    c3 = metrics["frozen-800/C3"]["operator_wall_seconds"]["warm29"]["median"]
    status = (
        "LEARNED_CUTOFF_CLOSED"
        if max(cap.values()) > 1.458
        else "NO_CONSUMER" if c3 < 2 * c2 else "PASS"
    )
    return {
        "status": status,
        "admits": "V1_REGISTRATION_ONLY" if status == "PASS" else "NONE",
        "cap_p99": cap,
        "consumer_rule": "literal §5(c): C3_median < 2*C2_median",
        "consumer_values": {"C2_median": c2, "C3_median": c3, "C2_over_C3": c2 / c3},
        "burst_doom_p99_le_5": {
            cell: metrics[f"frozen-800/{cell}"]["operator_wall_seconds"]["warm29"]["p99"] <= 5
            for cell in ("C3", "C4")
        },
    }
