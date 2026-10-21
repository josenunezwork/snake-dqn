"""Shared, platform-labelled batch-rate ruler for the L2' amendment.

This module deliberately separates fixture identity from timing.  A fixture is
an immutable, burned root: it carries canonical bytes for bodies, food, RNG,
opponent carry and its roster/checkpoint identities.  The L4 receipt prices
only L4; the corresponding Mac receipt is evidence for L1' and is never used
as an L2' forecast.

The real root builder/runner is intentionally injected.  That keeps this file
from acquiring a second simulator or a shortcut around the unchanged T3
operator.  The production adapter must submit every legal action through the
existing T3 H300 operator and return its semantic output.  Synthetic adapters
are accepted only with ``--smoke`` and their receipt is marked PRECHECK_ONLY.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import importlib
import json
import math
import multiprocessing
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from research.bl_served_search.exports import digest, export_json, verify_export

SCHEMA = "bl-served-search-ruler-fixtures/v1"
RECEIPT_SCHEMA = "bl-served-search-ruler-receipt/v1"
FIXTURE_TARGETS = (800, 1200, 1500)
MIXES = ("scripted", "frozen", "mixed")
MIN_FRAME = 2500
HORIZON = 300
MIN_BATCHES = 30
L4_WORKERS = 12
L4_ADMISSION_RATE = 2.18
HEARTBEAT_SECONDS = 120


def canonical(value: Any) -> str:
    """Return finite, stable JSON used for all ruler identities."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    """Hash immutable fixture bytes."""
    return hashlib.sha256(value).hexdigest()


def source_identity() -> dict[str, str]:
    """Bind the ruler to the unchanged T3 operator source bytes."""
    root = Path(__file__).resolve().parents[2]
    names = (
        "research/bl_served_search/ruler.py",
        "research/bl_served_search/ruler_operator.py",
        "research/t3_rollout_labels_20261015/engine.py",
        "research/bl_served_search/ceiling.py",
        "research/bl_served_search/fixture_geometry.py",
        "research/bl_served_search/ruler_timing.py",
        "research/t3_rollout_labels_20261015/bank_engine.py",
        "research/t3_rollout_labels_20261015/trigger.py",
        "src/simd_env/grid_scenarios.py",
    )
    from research.bl_served_search.ceiling import source_manifest

    return {**source_manifest(), **{name: digest(root / name) for name in names}}


def platform_identity(label: str) -> dict[str, str]:
    """Return provenance which labels timing instead of transferring it."""
    return {
        "label": label,
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def hardware_provenance() -> dict[str, Any]:
    """Bind an L4 timing receipt to the qualified runtime hardware evidence."""
    root = os.environ.get("COLAB_CLI_ROOT")
    path = Path(root) / "ready.json" if root else None
    if path is not None and path.is_file():
        return {"path": str(path), "sha256": digest(path), "reported": json.loads(path.read_text())}
    return {"path": None, "sha256": None, "reported": None}


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def validate_fixtures(value: dict[str, Any], *, full: bool = True) -> list[dict[str, Any]]:
    """Fail closed unless all nine all-mix long-body frozen roots are present."""
    if value.get("schema") != SCHEMA:
        raise ValueError("unexpected ruler fixture schema")
    rows = value.get("fixtures")
    if not isinstance(rows, list):
        raise ValueError("ruler fixtures must be a list")
    expected = {(mix, length) for mix in MIXES for length in FIXTURE_TARGETS}
    observed: set[tuple[str, int]] = set()
    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed ruler fixture")
        fixture_id = row.get("fixture_id")
        if not isinstance(fixture_id, str) or not fixture_id or fixture_id in ids:
            raise ValueError("fixture ids must be unique and nonempty")
        ids.add(fixture_id)
        pair = (row.get("mix"), row.get("target_length"))
        if pair[0] not in MIXES or pair[1] not in FIXTURE_TARGETS or pair in observed:
            raise ValueError("requires one fixture per mix and target length")
        observed.add(pair)
        if int(row.get("root_frame", -1)) < MIN_FRAME:
            raise ValueError(f"{fixture_id}: root frame is below {MIN_FRAME}")
        if int(row.get("root_length", -1)) < int(pair[1]):
            raise ValueError(f"{fixture_id}: root does not meet length target")
        if int(row.get("horizon", -1)) != HORIZON:
            raise ValueError(f"{fixture_id}: H300 clip is required")
        if row.get("tie_rule") != "survival,bare,q_desc,action_asc":
            raise ValueError(f"{fixture_id}: wrong frozen tie rule")
        if row.get("s2_native_v8") is not True:
            raise ValueError(f"{fixture_id}: S2 native v8 must be on")
        if not isinstance(row.get("legal_actions"), list) or not row["legal_actions"]:
            raise ValueError(f"{fixture_id}: legal actions missing")
        for field in ("root_sha256", "roster_sha256", "checkpoint_sha256", "operator_sha256"):
            _sha(row.get(field), f"{fixture_id}.{field}")
        encoded = row.get("root_bytes_b64")
        if not isinstance(encoded, str):
            raise ValueError(f"{fixture_id}: root bytes missing")
        raw = base64.b64decode(encoded.encode(), validate=True)
        if sha256_bytes(raw) != row["root_sha256"]:
            raise ValueError(f"{fixture_id}: root bytes digest mismatch")
        root = json.loads(raw)
        for field in ("bodies", "food", "rng", "opponent_carry"):
            if field not in root:
                raise ValueError(f"{fixture_id}: root bytes omit {field}")
    if full and observed != expected:
        raise ValueError("requires exactly 3 mixes x L800/L1200/L1500 fixtures")
    return sorted(rows, key=lambda row: str(row["fixture_id"]))


class Operator(Protocol):
    """Boundary for the single unchanged T3 operator implementation."""

    def prepare(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def batch(self, fixture: dict[str, Any]) -> dict[str, Any]: ...


_WORKER_OPERATOR: Operator | None = None


def _worker_init(factory_spec: str) -> None:
    """Create one cached exact operator in each spawned worker process."""
    global _WORKER_OPERATOR
    _WORKER_OPERATOR = _load_factory(factory_spec)()


def _worker_batch(fixture: dict[str, Any]) -> dict[str, Any]:
    """Run exactly one batch; fixture-local root caches remain worker-local."""
    if _WORKER_OPERATOR is None:
        raise RuntimeError("ruler worker was not initialized")
    started = time.perf_counter()
    result = dict(_WORKER_OPERATOR.batch(fixture))
    result["worker_wall_seconds"] = time.perf_counter() - started
    return result


def _load_factory(spec: str) -> Callable[[], Operator]:
    module_name, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("operator must have module:factory form")
    factory = getattr(importlib.import_module(module_name), name)
    if not callable(factory):
        raise ValueError("operator factory is not callable")
    return factory


def build_fixtures(request: dict[str, Any], operator: Operator) -> dict[str, Any]:
    """Build immutable fixtures from real burned roots supplied by one operator.

    ``Operator.prepare`` must return the actual root bytes and must not source a
    selection-bank world.  This is deliberately checked here rather than by a
    later manifest consumer.
    """
    requested = request.get("fixtures")
    if not isinstance(requested, list):
        raise ValueError("fixture request must contain fixtures")
    rows = []
    for item in requested:
        if not isinstance(item, dict) or item.get("selection_bank") is True:
            raise ValueError("ruler fixture request must use burned non-selection roots")
        row = dict(operator.prepare(dict(item)))
        root_bytes = row.pop("root_bytes", None)
        if not isinstance(root_bytes, bytes):
            raise ValueError("operator did not return frozen root bytes")
        row["root_bytes_b64"] = base64.b64encode(root_bytes).decode("ascii")
        row["root_sha256"] = sha256_bytes(root_bytes)
        row.setdefault("horizon", HORIZON)
        row.setdefault("tie_rule", "survival,bare,q_desc,action_asc")
        row.setdefault("s2_native_v8", True)
        row.setdefault("operator_sha256", sha256_bytes(canonical(source_identity()).encode()))
        rows.append(row)
    value = {
        "schema": SCHEMA,
        "mode": "full",
        "source_identity": source_identity(),
        "fixtures": rows,
    }
    validate_fixtures(value)
    return value


def export_fixtures(out: Path, value: dict[str, Any]) -> None:
    """Publish flat portable root sidecars before the aggregate manifest.

    The aggregate's ``root_file`` entries are basenames only: a qualified
    transport may flatten this directory without accidentally retaining a Mac
    path.  The bytes remain embedded in the manifest for local validation, and
    each root is separately immutable for input hashing by the supervisor.
    """
    out = Path(out)
    rows = validate_fixtures(value)
    published = []
    for row in rows:
        name = f"fixture-{row['fixture_id']}.root.json"
        root = {
            "schema": "bl-served-search-ruler-root/v1",
            "fixture_id": row["fixture_id"],
            "root_sha256": row["root_sha256"],
            "root_bytes_b64": row["root_bytes_b64"],
        }
        export_json(out / name, root)
        published.append({**row, "root_file": name})
    export_json(out / "ruler-fixtures.json", {**value, "mode": "full", "fixtures": published})


def _semantic(record: dict[str, Any]) -> dict[str, Any]:
    """Keep only platform-independent operator results for cross-platform checks."""
    result = record.get("semantic")
    if not isinstance(result, dict):
        raise ValueError("operator batch lacks semantic output")
    return result


def run_ruler(
    fixtures: dict[str, Any],
    operator: Operator,
    *,
    platform_label: str,
    workers: int,
    min_batches: int = MIN_BATCHES,
    smoke: bool = False,
    operator_spec: str | None = None,
    heartbeat: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the ruler with exclusive stage timing and platform-labelled rate."""
    rows = validate_fixtures(fixtures, full=not smoke)
    if workers < 1:
        raise ValueError("workers must be positive")
    if platform_label == "l4" and not smoke and workers != L4_WORKERS:
        raise ValueError("full L4 ruler requires exactly 12 workers")
    if platform_label == "l4" and not smoke:
        hardware = hardware_provenance()
        if (
            hardware.get("reported", {}).get("hardware", {}).get("kind") != "L4"
            or (os.cpu_count() or 0) < 12
        ):
            raise ValueError("L4 ruler requires qualified ready.json L4 hardware and 12 vCPUs")
    if platform_label == "l4" and not smoke and platform.system() != "Linux":
        raise ValueError("an L4 ruler receipt must be produced on the qualified Linux runtime")
    if min_batches < 1 or (not smoke and min_batches < MIN_BATCHES):
        raise ValueError("full ruler requires at least 30 batches per fixture")
    started = time.monotonic()
    batches = 0
    records = {fixture["fixture_id"]: [] for fixture in rows}
    last_heartbeat = started
    jobs = [fixture for fixture in rows for _ in range(min_batches)]

    def validate_output(
        fixture: dict[str, Any], output: dict[str, Any], submitted: float, visible: float
    ) -> None:
        nonlocal batches, last_heartbeat
        stages = output.get("stage_seconds")
        if not isinstance(stages, dict) or set(stages) != {
            "sim",
            "observation",
            "hero_v8",
            "opponent_v8",
            "vector61_frp3",
            "b8_forward",
            "clone_digest",
            "submit_visible",
            "submit_consumed",
        }:
            raise ValueError("batch lacks the complete exclusive-stage timing set")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in stages.values()
        ):
            raise ValueError("batch has invalid stage timing")
        consumed = time.monotonic()
        stages = dict(stages)
        stages["submit_visible"] = max(0.0, visible - submitted)
        stages["submit_consumed"] = max(0.0, consumed - submitted)
        records[fixture["fixture_id"]].append(
            {
                "semantic": _semantic(output),
                "stage_seconds": stages,
                "worker_wall_seconds": float(output.get("worker_wall_seconds", 0.0)),
            }
        )
        batches += 1
        if heartbeat is not None and consumed - last_heartbeat >= HEARTBEAT_SECONDS:
            heartbeat({"completed_batches": batches, "elapsed_seconds": consumed - started})
            last_heartbeat = consumed

    if workers == 1 and operator_spec is None:
        for fixture in jobs:
            submitted = time.monotonic()
            output = dict(operator.batch(fixture))
            output["worker_wall_seconds"] = time.monotonic() - submitted
            validate_output(fixture, output, submitted, time.monotonic())
    else:
        if not operator_spec:
            raise ValueError("parallel ruler requires a module:factory operator specification")
        submitted: dict[concurrent.futures.Future[dict[str, Any]], float] = {}

        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_worker_init,
            initargs=(operator_spec,),
        ) as executor:
            future_fixture = {}
            for fixture in jobs:
                future = executor.submit(_worker_batch, fixture)
                submitted[future] = time.monotonic()
                future_fixture[future] = fixture
            pending = set(future_fixture)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=HEARTBEAT_SECONDS,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    if heartbeat is not None:
                        heartbeat(
                            {
                                "completed_batches": batches,
                                "elapsed_seconds": time.monotonic() - started,
                                "pending_batches": len(pending),
                            }
                        )
                    continue
                for future in done:
                    fixture = future_fixture[future]
                    output = future.result()
                    # Parent observation after a valid result is the submit→visible
                    # boundary.  It cannot race Future callback scheduling.
                    validate_output(fixture, output, submitted[future], time.monotonic())
    elapsed = time.monotonic() - started
    aggregate = batches / elapsed if elapsed else float("inf")
    status = (
        "PRECHECK_ONLY"
        if smoke
        else (
            "RULER_PASS"
            if platform_label != "l4" or aggregate >= L4_ADMISSION_RATE
            else "NOT_DISPATCHABLE"
        )
    )
    by_id = {row["fixture_id"]: row for row in rows}
    per_mix = {}
    for mix in MIXES:
        mix_records = [
            entry for fixture_id, entry in records.items() if by_id[fixture_id]["mix"] == mix
        ]
        worker_seconds = sum(
            float(batch["worker_wall_seconds"]) for entry in mix_records for batch in entry
        )
        completed = sum(len(entry) for entry in mix_records)
        per_mix[mix] = {
            "completed_batches": completed,
            "worker_seconds": worker_seconds,
            "batches_per_worker_second": completed / worker_seconds if worker_seconds else None,
        }
    return {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "platform": platform_identity(platform_label),
        "hardware": {**hardware_provenance(), "vcpu_count": os.cpu_count()},
        "fixture_sha256": sha256_bytes(canonical(fixtures).encode()),
        "source_identity": source_identity(),
        "workers": workers,
        "min_batches_per_fixture": min_batches,
        "completed_batches": batches,
        "elapsed_seconds": elapsed,
        "aggregate_batches_per_second": aggregate,
        "per_mix": per_mix,
        "fixture_identity": [
            {key: row[key] for key in ("fixture_id", "mix", "target_length", "root_sha256")}
            for row in rows
        ],
        "operator_spec": operator_spec,
        "records": [
            {"fixture_id": fixture_id, "batches": records[fixture_id]}
            for fixture_id in sorted(records)
        ],
    }


def verify_qualification(mac: dict[str, Any], l4: dict[str, Any]) -> dict[str, Any]:
    """Validate separate receipts and fail closed on a semantic mismatch."""
    for receipt, label in ((mac, "mac"), (l4, "l4")):
        if (
            receipt.get("schema") != RECEIPT_SCHEMA
            or receipt.get("platform", {}).get("label") != label
        ):
            raise ValueError(f"invalid {label} ruler receipt")
        if receipt.get("status") == "PRECHECK_ONLY":
            raise ValueError("smoke receipts cannot qualify the ruler")
        if receipt.get("status") not in {"RULER_PASS", "NOT_DISPATCHABLE"}:
            raise ValueError(f"{label} ruler has non-qualifying status")
        if receipt.get("source_identity") != source_identity():
            raise ValueError(f"{label} ruler source identity differs")
        if int(receipt.get("min_batches_per_fixture", 0)) < MIN_BATCHES:
            raise ValueError(f"{label} ruler did not complete 30 batches per fixture")
        _sha(receipt.get("fixture_sha256"), f"{label}.fixture_sha256")
        if (
            receipt.get("operator_spec")
            != "research.bl_served_search.ruler_operator:create_operator"
        ):
            raise ValueError("only the production T3 operator can qualify")
        identity = receipt.get("fixture_identity")
        expected_pairs = {(mix, length) for mix in MIXES for length in FIXTURE_TARGETS}
        if (
            not isinstance(identity, list)
            or len(identity) != 9
            or {(r["mix"], r["target_length"]) for r in identity} != expected_pairs
            or len({r["fixture_id"] for r in identity}) != 9
        ):
            raise ValueError("ruler fixture identity coverage is incomplete")
        for row in identity:
            _sha(row.get("root_sha256"), "root identity")
        rows = receipt.get("records")
        if not isinstance(rows, list) or len(rows) != 9:
            raise ValueError(f"{label} ruler did not cover the full nine-fixture bank")
        if any(
            not isinstance(row.get("batches"), list) or len(row["batches"]) < MIN_BATCHES
            for row in rows
        ):
            raise ValueError(f"{label} ruler has incomplete fixture batches")
        if len({row["fixture_id"] for row in rows}) != 9 or {row["fixture_id"] for row in rows} != {
            row["fixture_id"] for row in identity
        }:
            raise ValueError("ruler fixture records duplicate or disagree with identity")
        completed = sum(len(row["batches"]) for row in rows)
        if int(receipt.get("completed_batches", 0)) != completed:
            raise ValueError(f"{label} ruler completed-batch count is inconsistent")
        if completed < 270:
            raise ValueError(f"{label} ruler has fewer than 270 real batch records")
        elapsed, rate = receipt.get("elapsed_seconds"), receipt.get("aggregate_batches_per_second")
        if (
            type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
            or elapsed <= 0
            or type(rate) not in (int, float)
            or not math.isfinite(rate)
            or rate <= 0
            or not math.isclose(rate, completed / elapsed, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError("ruler rate does not reconcile to actual batches and elapsed time")
        for row in rows:
            for batch in row["batches"]:
                if batch["semantic"].get("fired") is not True:
                    raise ValueError("ruler batch did not fire dense search")
                stages = batch.get("stage_seconds", {})
                if any(
                    type(t) not in (int, float) or not math.isfinite(t) or t < 0
                    for t in stages.values()
                ):
                    raise ValueError("invalid ruler stage timings")
    if int(l4.get("workers", 0)) != L4_WORKERS:
        raise ValueError("L4 ruler did not use exactly 12 workers")
    hardware = l4.get("hardware")
    if not isinstance(hardware, dict) or not _sha(hardware.get("sha256"), "l4.hardware.sha256"):
        raise ValueError("L4 ruler lacks qualified hardware provenance")
    if (
        l4.get("platform", {}).get("system") != "Linux"
        or hardware.get("reported", {}).get("hardware", {}).get("kind") != "L4"
        or hardware.get("vcpu_count", 0) < 12
    ):
        raise ValueError("L4 hardware/platform provenance does not qualify")
    if mac.get("platform", {}).get("system") != "Darwin":
        raise ValueError("Mac ruler must be measured on macOS")
    if mac.get("fixture_identity") != l4.get("fixture_identity"):
        raise ValueError("platforms used different root identities")
    if mac.get("fixture_sha256") != l4.get("fixture_sha256"):
        raise ValueError("platforms used different fixtures")
    mac_rows = {row["fixture_id"]: row for row in mac.get("records", [])}
    l4_rows = {row["fixture_id"]: row for row in l4.get("records", [])}
    if set(mac_rows) != set(l4_rows):
        raise ValueError("platform fixture coverage differs")
    mismatches = []
    for key in sorted(mac_rows):
        left = [canonical(row["semantic"]) for row in mac_rows[key]["batches"]]
        right = [canonical(row["semantic"]) for row in l4_rows[key]["batches"]]
        if left != right:
            mismatches.append(key)
    if mismatches:
        return {"status": "INVALID", "semantic_mismatches": mismatches}
    l4_rate = float(l4["aggregate_batches_per_second"])
    return {
        "status": "RULER_PASS" if l4_rate >= L4_ADMISSION_RATE else "NOT_DISPATCHABLE",
        "semantic_mismatches": [],
        "l4_aggregate_batches_per_second": l4_rate,
        "l4_admission_rate": L4_ADMISSION_RATE,
        "timing_transfer": "FORBIDDEN",
    }


def validate_qualification(mac: dict[str, Any], l4: dict[str, Any]) -> dict[str, Any]:
    """Require two complete, semantically identical full-ruler receipts.

    This is the readiness import boundary.  It raises for an INVALID semantic
    comparison or for a non-passing L4 rate, so callers cannot accidentally
    treat an incomplete or Mac-only receipt as L2' admission evidence.
    """
    result = verify_qualification(mac, l4)
    if result["status"] == "INVALID":
        raise ValueError(f"ruler semantic mismatch: {result['semantic_mismatches']}")
    if result["status"] != "RULER_PASS":
        raise ValueError(
            "L4 ruler is NOT_DISPATCHABLE: "
            f"{result['l4_aggregate_batches_per_second']:.6f} < {L4_ADMISSION_RATE:.2f} batches/s"
        )
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--request", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument(
        "--operator", default="research.bl_served_search.ruler_operator:create_operator"
    )
    run = sub.add_parser("run")
    run.add_argument("--fixtures", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument(
        "--operator", default="research.bl_served_search.ruler_operator:create_operator"
    )
    run.add_argument("--platform", choices=("mac", "l4"), required=True)
    run.add_argument("--workers", type=int, required=True)
    run.add_argument("--min-batches", type=int, default=MIN_BATCHES)
    run.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    operator = _load_factory(args.operator)()
    if args.command == "prepare":
        value = build_fixtures(json.loads(args.request.read_text()), operator)
        export_fixtures(args.out, value)
        return
    fixtures = json.loads(args.fixtures.read_text())
    verify_export(args.fixtures)
    if fixtures.get("source_identity") != source_identity():
        parser.error("frozen fixture source differs from current operator")

    def heartbeat(value: dict[str, Any]) -> None:
        progress = args.out.parent / (args.out.name + ".progress.json")
        progress.parent.mkdir(parents=True, exist_ok=True)
        temporary = progress.with_suffix(progress.suffix + ".tmp")
        temporary.write_text(canonical(value) + "\n")
        os.replace(temporary, progress)

    heartbeat({"completed_batches": 0})
    receipt = run_ruler(
        fixtures,
        operator,
        platform_label=args.platform,
        workers=args.workers,
        min_batches=args.min_batches,
        smoke=args.smoke,
        operator_spec=args.operator,
        heartbeat=heartbeat,
    )
    export_json(args.out, receipt)
    heartbeat(
        {
            "completed_batches": receipt["completed_batches"],
            "status": receipt["status"],
            "elapsed_seconds": receipt["elapsed_seconds"],
        }
    )


if __name__ == "__main__":
    _main()
