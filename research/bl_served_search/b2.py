"""B2 offline checks, trajectory payload, verified ledger, and Mac-only single look.

Claude alone dispatches via tools/ops/dispatch.sh. This module never allocates,
retries, submits, mounts Drive, or changes serving. An incomplete frozen design
is a hard refusal, including during manifest preparation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_served_search import b2_engine as engine
from research.bl_served_search import b2_units as units
from research.bl_served_search.ceiling import canonical, digest_value
from research.bl_served_search.exports import digest, export_bytes, export_json, verify_export

SCHEMA = "bl-plan-v3-b2/v1"
PLAN_SHA256 = "d23690617519262bd1b1a3e6c5d8b414ef952003b5b92c86842c9f5fa4210201"
ROOT = Path(__file__).resolve().parents[2]
B8_FILE = "b8_s3_t7_20261016.pth"
FRP_FILE = "frp3_m3_s12_u60000_20261005.pth"


def read_export(path: Path) -> dict:
    verify_export(path)
    return json.loads(path.read_text())


def validate_spec(spec: dict) -> dict:
    """Refuse to silently import unratified advisor design values into B2."""
    if spec.get("schema") != SCHEMA + "/design" or spec.get("plan_sha256") != PLAN_SHA256:
        raise ValueError("foreign B2 authority")
    if spec.get("status") != "FROZEN_BEFORE_DATA":
        raise ValueError("NOT_READY: B2 horizon/grid/continuation ruling is not frozen")
    for key in ("trajectory_frames", "root_stride", "unit_timeout_seconds"):
        if type(spec.get(key)) is not int or spec[key] <= 0:
            raise ValueError(f"missing positive frozen B2 {key}")
    if type(spec.get("workers")) is not int or not 1 <= spec["workers"] <= 6:
        raise ValueError("B2 workers must be frozen from one through six")
    if spec.get("root_grid") != "absolute-trajectory-decision":
        raise ValueError("freeze B2 root grid as absolute-trajectory-decision")
    if spec.get("escape_total_frames") not in (1440, 1740):
        raise ValueError("freeze whether H1440 is total or additional to H300")
    if not isinstance(spec.get("namespace"), str) or not spec["namespace"].strip():
        raise ValueError("missing B2 namespace")
    seeds = spec.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 32
        or len(set(seeds)) != 32
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
    ):
        raise ValueError("B2 requires 32 fresh distinct uint32 seeds")
    if (
        type(spec.get("canary_seed")) is not int
        or not 0 <= spec["canary_seed"] < 2**32
        or spec["canary_seed"] in seeds
    ):
        raise ValueError("B2 requires a separate fresh excluded canary seed")
    audit = spec.get("freshness")
    if not isinstance(audit, dict) or audit.get("status") != "CLEAR":
        raise ValueError("missing CLEAR Mac freshness audit")
    if audit.get("seed_allocation_sha256") != digest_value([*seeds, spec["canary_seed"]]):
        raise ValueError("freshness audit binds another seed allocation")
    if spec.get("roster") != "bare-b8-s3-vs-5-frp3-s12-v8":
        raise ValueError("B2 requires the unchanged served mixed Watch6 roster")
    identity = {key: value for key, value in spec.items() if key != "allocation_digest"}
    if spec.get("allocation_digest") != digest_value(identity):
        raise ValueError("B2 frozen allocation digest differs")
    return spec


def validate_freshness(spec: dict, path: Path) -> dict:
    """Read the full immutable Mac audit, including collision and scan controls."""
    if verify_export(path) != spec.get("freshness_report_sha256"):
        raise ValueError("B2 freshness report digest differs")
    audit = json.loads(path.read_text())
    seeds = [*spec["seeds"], spec["canary_seed"]]
    if (
        audit.get("schema") != "bl-served-search-seed-audit/v1"
        or audit.get("status") != "CLEAR"
        or audit.get("namespace") != spec["namespace"]
        or audit.get("seeds") != seeds
        or audit.get("seed_allocation_sha256") != digest_value(seeds)
        or audit.get("candidate_count") != 33
        or audit.get("unique_count") != 33
        or audit.get("collisions") != []
        or audit.get("internal_collision") is not False
    ):
        raise ValueError("B2 full freshness report allocation/collision mismatch")
    explicit = audit.get("explicit_banks", {})
    exclusions = audit.get("training_exclusions", [])
    history = audit.get("history", {})
    required = {
        "research/t34_round0/BANK_ROUND0.json",
        "research/t34_round0d/BANK_ROUND0D.json",
        "research/t34_pathb/RETENTION_BANK_V2.json",
        "research/phase0_search_20261019/WORLD_BANK.json",
    }
    if not required <= set(explicit) or len(exclusions) < 7 or len(history.get("roots", [])) < 2:
        raise ValueError("B2 freshness report inventory coverage missing")
    entries = [*explicit.values(), *exclusions]
    if any(
        e.get("collisions") != [] or not e.get("sha256") or e.get("count", 0) <= 0 for e in entries
    ):
        raise ValueError("B2 freshness report collision/inventory evidence missing")
    hits = history.get("hits", {})
    observed = {seed for group in hits.values() for seed in group}
    controls = {entry["positive_control"] for entry in explicit.values()}
    if not controls <= observed or set(seeds) & observed:
        raise ValueError("B2 freshness report scan controls/collisions failed")
    return audit


def configured_drive_root() -> Path:
    """The owner's actual configured Drive-for-desktop root, Mac-side only."""
    cloud = Path.home() / "Library/CloudStorage"
    roots = [
        path / "My Drive" for path in cloud.glob("GoogleDrive-*") if (path / "My Drive").is_dir()
    ]
    if len(roots) != 1:
        raise ValueError("configured Drive-for-desktop root is unavailable or ambiguous")
    return roots[0].resolve()


def drive_backup_root(path: Path) -> Path:
    root, resolved = configured_drive_root(), path.resolve()
    if not path.is_absolute() or (resolved != root and root not in resolved.parents):
        raise ValueError("B2 backup is outside configured Drive-for-desktop root")
    return resolved


def check(spec_path: Path, ledger_path: Path, inputs: Path) -> dict:
    spec = validate_spec(read_export(spec_path))
    validate_freshness(spec, spec_path.parent / "freshness-audit.json")
    ledger = read_export(ledger_path)
    done = units.validate_ledger(ledger, spec["namespace"], spec["allocation_digest"])
    for name, checksum in ((B8_FILE, engine.ceiling.B8_SHA256), (FRP_FILE, engine.FRP3)):
        if digest(inputs / name) != checksum:
            raise ValueError("B2 uploaded checkpoint drift")
    return {"status": "CHECKED_NOT_DISPATCHED", "verified_units": len(done), "total": 32}


def run_unit(spec: dict, index: int, ledger: dict, inputs: Path, out: Path) -> None:
    """One trajectory per runtime, so a crash cannot lose multiple completed units."""
    spec = validate_spec(spec)
    unit = units.unit_id(spec["namespace"], index)
    done = units.validate_ledger(ledger, spec["namespace"], spec["allocation_digest"])
    probe = os.environ.get("BL_B2_CANARY") == "1"
    if probe:
        if os.environ.get("SNAKE_COLAB_CANARY") != "1":
            raise ValueError("B2 canary requires real family instrumentation")
        spec = {
            **spec,
            "namespace": spec["namespace"] + "/excluded-canary",
            "seeds": [spec["canary_seed"], *spec["seeds"][1:]],
        }
        unit = units.unit_id(spec["namespace"], index)
    if unit in done:
        # Do not re-export a second copy into the result set.
        export_json(out / "unit-status.json", {"unit_id": unit, "status": "SKIPPED_VERIFIED"})
        return
    if not os.environ.get("COLAB_CLI_ROOT"):
        raise ValueError("production B2 is Colab-only; Claude operates all runs")
    driver = engine.WatchDriver(spec["seeds"][index], inputs / B8_FILE, inputs / FRP_FILE)
    started = time.monotonic()

    def progress(value: dict) -> None:
        elapsed = time.monotonic() - started
        frames = value["done_frames"]
        print(
            canonical(
                {
                    "unit_id": unit,
                    "done": frames,
                    "total": value["total_frames"],
                    "elapsed_seconds": elapsed,
                    "trajectory_eta_seconds": (
                        elapsed * (value["total_frames"] - frames) / frames if frames else None
                    ),
                    "note": "operational only; alternative results sealed until Mac merge",
                }
            ),
            flush=True,
        )

    progress({"done_frames": 0, "total_frames": spec["trajectory_frames"]})
    result = engine.collect(spec, index, driver, progress=progress)
    if probe and not result["serial_parallel_root_checks"]:
        raise ValueError(
            "NOT_READY B2 canary did not exercise a natural root serial/parallel batch"
        )
    result["snapshot_sources"] = engine.snapshot_sources()
    exported = export_json(out / "trajectory.json", result)
    if probe:
        # Operational allowlist only. Admission never opens trajectory.json.
        branches = [branch for root in result["roots"] for branch in root["alternatives"]]
        export_json(
            out / "canary-admission.json",
            {
                "schema": SCHEMA + "/canary-admission",
                "allocation_digest": spec["allocation_digest"],
                "unit_id": unit,
                "seed": spec["canary_seed"],
                "trajectory_frames": spec["trajectory_frames"],
                "workers": spec["workers"],
                "serial_parallel_root_checks": result["serial_parallel_root_checks"],
                "serial_parallel_equal": True,  # collect raises on any unequal batch.
                "native_v8": [result["native_v8"], *[b["native_v8"] for b in branches]],
                "worker_threads": [b["worker_threads"] for b in branches],
                "phase_timings": result["phase_timings"],
            },
        )
    export_json(
        out / "unit-status.json",
        {"unit_id": unit, "status": "COMPLETED", "sha256": exported["sha256"]},
    )


def stage_unit(source: Path, exports: Path, spec: dict, index: int, receipt_path: Path) -> Path:
    """Outcome-blind stage only after broker release and rehashed Drive backup."""
    receipt = json.loads(receipt_path.read_text())
    manifest_path = receipt_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        receipt.get("status") not in {"PASS", "INCOMPLETE"}
        or (
            receipt.get("status") == "INCOMPLETE"
            and receipt.get("failure_class")
            not in {"ACCOUNTING", "RPC_EXEC_POLL_TIMEOUT", "SUPERVISOR", "CHILD_NONZERO_EXIT"}
        )
        or receipt.get("stop_verified") is not True
        or (receipt.get("status") == "PASS" and receipt.get("error"))
        or receipt.get("cleanup_error")
        or receipt.get("manifest_sha256") != digest(manifest_path)
        or receipt.get("job_id") != manifest.get("job_id")
        or manifest.get("env", {}).get("BL_B2_CANARY") != "0"
        or manifest.get("env", {}).get("BL_B2_INDEX") != str(index)
        or manifest.get("bl_b2", {}).get("allocation_digest") != spec["allocation_digest"]
    ):
        raise ValueError("B2 stage requires matching terminal/release/allocation evidence")
    expected_source = receipt_path.parent / "exports/trajectory.json"
    if source.resolve() != expected_source.resolve():
        raise ValueError("B2 stage source differs from broker export")
    checksum = verify_export(source)
    if not any(
        row.get("path") == "trajectory.json"
        and row.get("sha256") == checksum
        and row.get("size") == source.stat().st_size
        for row in receipt["exports"]
    ):
        raise ValueError("B2 broker receipt does not verify this payload")
    backup = json.loads((receipt_path.parent / "backup.json").read_text())
    base = drive_backup_root(Path(manifest["backup_dir"]))
    generation = receipt.get("generation")
    job = receipt["job_id"]
    if any(
        not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}
        for name in (generation, job)
    ):
        raise ValueError("B2 backup has invalid run identity")
    expected_destination = (base / job / generation).resolve()
    destination = Path(backup["destination"]).resolve()
    if (
        destination != expected_destination
        or destination == source.parent.resolve()
        or source.parent.resolve() in destination.parents
        or destination in source.parent.resolve().parents
    ):
        raise ValueError("B2 backup root/generation/source alias differs")
    drive_source = destination / "trajectory.json"
    if backup.get("hash_verified") is not True or verify_export(drive_source) != checksum:
        raise ValueError("B2 Drive backup is absent or differs")
    target = units.unit_payload_path(exports, units.unit_id(spec["namespace"], index))
    export_bytes(target, source.read_bytes())
    return target


def validate_branch(branch: dict, root: dict, endpoint: int) -> None:
    """Reject truncated/contradictory horizons before counting a rescue witness."""
    engine.attest(branch["native_v8"])
    h300 = branch["H300"]
    if type(h300.get("survives")) is not bool or type(h300.get("frames_executed")) is not int:
        raise ValueError("INVALID B2 H300 outcome types")
    if h300["survives"]:
        if h300["frames_executed"] != 300:
            raise ValueError("INVALID B2 truncated H300 survivor")
        final = branch["continuation"]
        if not isinstance(final, dict) or type(final.get("survives")) is not bool:
            raise ValueError("INCOMPLETE B2 escape continuation")
        frames = final.get("frames_executed")
        if type(frames) is not int or not 300 < frames <= endpoint:
            raise ValueError("INVALID B2 continuation horizon")
        if final["survives"] and frames != endpoint:
            raise ValueError("INCOMPLETE B2 escape continuation")
    else:
        if not 1 <= h300["frames_executed"] <= 300 or branch["continuation"] is not None:
            raise ValueError("INVALID B2 dead H300 branch")
        final = branch["death"]
        if not isinstance(final, dict) or final.get("survives") is not False:
            raise ValueError("INVALID B2 missing branch death")
        if final.get("frames_executed") != h300["frames_executed"]:
            raise ValueError("INVALID B2 death/H300 horizon mismatch")
        frames = h300["frames_executed"]
    if (
        final.get("alive") is not final["survives"]
        or final.get("life") != root["life"]
        or final.get("decision") != root["decision"] + frames - 1
    ):
        raise ValueError("INVALID B2 branch life/decision reconciliation")
    if final["survives"]:
        if branch["death"] is not None or final.get("death_cause") != "none":
            raise ValueError("INVALID B2 survivor death metadata")
    elif branch["death"] != final or final.get("death_cause") not in {
        "wall",
        "self",
        "head",
        "body",
    }:
        raise ValueError("INVALID B2 missing/contradictory death cause")


def merge(spec: dict, exports: Path, ledger: dict, out: Path) -> dict:
    """The only scientific look: require every verified allocation before parsing rows."""
    if platform.system() != "Darwin":
        raise ValueError("B2 merge/look must run on the Mac")
    validate_spec(spec)
    expected = {units.unit_id(spec["namespace"], i) for i in range(32)}
    done = units.verify_ledger(ledger, exports, spec["namespace"], spec["allocation_digest"])
    if done != expected:
        raise ValueError("INCOMPLETE B2: no partial look; require 32 verified trajectories")
    roots, productive, durable = [], set(), set()
    snapshot_sources = engine.snapshot_sources()
    for index in range(32):
        unit = units.unit_id(spec["namespace"], index)
        row = json.loads(units.unit_payload_path(exports, unit).read_text())
        if any(
            row.get(key) != value
            for key, value in {
                "schema": "bl-b2-trajectory/v1",
                "namespace": spec["namespace"],
                "allocation_digest": spec["allocation_digest"],
                "unit_id": unit,
                "trajectory_index": index,
                "seed": spec["seeds"][index],
                "trajectory_frames": spec["trajectory_frames"],
            }.items()
        ):
            raise ValueError("INVALID foreign B2 trajectory allocation")
        if row.get("snapshot_sources") != snapshot_sources:
            raise ValueError("INVALID B2 snapshot source closure differs")
        engine.attest(row["native_v8"])
        if len(row["roots"]) > 4:
            raise ValueError("INVALID B2 root count")
        if len(row["trajectory"]) != spec["trajectory_frames"]:
            raise ValueError("INCOMPLETE B2 trajectory trace")
        candidates = []
        for decision, event in enumerate(row["trajectory"]):
            selector = event["selector"]
            if (
                selector["decision"] != decision
                or selector["frame"] != decision + 1
                or type(selector["alive"]) is not bool
                or type(selector["life"]) is not int
                or selector["life"] < 0
                or type(selector["length"]) is not int
                or selector["length"] < 0
            ):
                raise ValueError("INVALID B2 selector trace identity")
            if (
                selector["alive"]
                and selector["length"] >= 800
                and decision % spec["root_stride"] == 0
            ):
                candidates.append(
                    {
                        "trajectory_index": index,
                        "life": selector["life"],
                        "frame": selector["frame"],
                        "length": selector["length"],
                        "selection_hash": units.root_hash(
                            spec["namespace"], index, selector["life"], selector["frame"]
                        ),
                    }
                )
        if row["candidate_roots"] != candidates:
            raise ValueError("INVALID B2 candidate inventory differs from trajectory")
        expected_hashes = sorted(c["selection_hash"] for c in candidates)[:4]
        if sorted(r["selection_hash"] for r in row["roots"]) != expected_hashes:
            raise ValueError("INVALID B2 roots are not bottom-four hash selected")
        seen = set()
        for root_index, root in enumerate(row["roots"]):
            key = (index, root["life"], root["frame"])
            if key in seen or root["length"] < 800 or root["mix"] != "mixed":
                raise ValueError("INVALID duplicate/foreign B2 root")
            seen.add(key)
            decision = root["decision"]
            if type(decision) is not int or not 0 <= decision < spec["trajectory_frames"]:
                raise ValueError("INVALID B2 root decision")
            selector = row["trajectory"][decision]["selector"]
            if (
                root["frame"] != decision + 1
                or root["life"] != selector["life"]
                or root["length"] != selector["length"]
                or decision % spec["root_stride"] != 0
                or root.get("trajectory_index") != index
            ):
                raise ValueError("INVALID B2 root/selector coordinates")
            if root["selection_hash"] != units.root_hash(spec["namespace"], *key):
                raise ValueError("INVALID B2 hash selection identity")
            snapshot = base64.b64decode(root["snapshot_b64"], validate=True)
            if hashlib.sha256(snapshot).hexdigest() != root["snapshot_sha256"]:
                raise ValueError("INVALID B2 restorable snapshot hash")
            branches = root["alternatives"]
            if len(branches) != len({b["action"] for b in branches}) or {
                b["action"] for b in branches
            } != set([root["bare"], *root["legal_actions"]]):
                raise ValueError("INVALID B2 missing/duplicate legal alternatives")
            if row["trajectory"][root["decision"]]["hero_action"] != root["bare"]:
                raise ValueError("INVALID B2 source bare action differs from natural trace")
            engine.validate_snapshot(snapshot, root)
            for branch in branches:
                validate_branch(branch, root, spec["escape_total_frames"])
            bare = next(b for b in branches if b["action"] == root["bare"])
            rescues = [b for b in branches if b["action"] != root["bare"] and b["H300"]["survives"]]
            if not bare["H300"]["survives"] and rescues:
                productive.add((index, root["life"]))
                if any(b["continuation"]["survives"] for b in rescues):
                    durable.add((index, root["life"]))
            roots.append(
                {
                    **{key: value for key, value in root.items() if key != "snapshot_b64"},
                    "fixture_id": (
                        f"natural-mixed-{index:02d}-life-{root['life']}" f"-frame-{root['frame']}"
                    ),
                    "snapshot_ref": {
                        "unit_path": f"units/{unit}.json",
                        "root_index": root_index,
                        "unit_sha256": next(
                            e["sha256"] for e in ledger["units"] if e["unit_id"] == unit
                        ),
                    },
                }
            )
    result = {
        "schema": SCHEMA + "/library",
        "allocation_digest": spec["allocation_digest"],
        "ledger_sha256": digest_value(ledger),
        "trajectories": 32,
        "unit_inventory": ledger["units"],
        "snapshot_sources": snapshot_sources,
        "rescue_positive_lives_H300": len(productive),
        "escape_endpoint_positive_lives": len(durable),
        "verdict": "INSUFFICIENT" if len(productive) < 8 else "LIBRARY_COMPLETE",
        "scope": "development fixtures only; C15 and L1 bars remain separate Mac gates",
        "roots": roots,
    }
    export_json(out, result)
    return result


def restore_library_root(library: Path, export_dir: Path, fixture_id: str) -> engine.WatchDriver:
    """Open one natural C15/L1 fixture from verified library/unit bytes."""
    value = read_export(library)
    if value["snapshot_sources"] != engine.snapshot_sources():
        raise ValueError("natural snapshot source closure differs")
    matches = [r for r in value["roots"] if r["fixture_id"] == fixture_id]
    if len(matches) != 1:
        raise ValueError("missing or duplicate natural fixture")
    selected = matches[0]
    ref = selected["snapshot_ref"]
    relative = Path(ref["unit_path"])
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != "units"
        or ".." in relative.parts
    ):
        raise ValueError("unsafe natural snapshot reference")
    path = export_dir / relative
    if verify_export(path) != ref["unit_sha256"]:
        raise ValueError("natural snapshot unit changed")
    payload = json.loads(path.read_text())["roots"][ref["root_index"]]
    snapshot = base64.b64decode(payload["snapshot_b64"], validate=True)
    if hashlib.sha256(snapshot).hexdigest() != selected["snapshot_sha256"]:
        raise ValueError("natural snapshot bytes changed")
    return engine.validate_snapshot(snapshot, selected)


def tiny_preflight(inputs: Path, out: Path) -> dict:
    """Real entrypoint, three tiny real units, snapshot restore and native rows.

    This burns no production seed and is never a B2 scientific look or estimate
    for a long natural trajectory. No constructed long-body roots are injected.
    """
    if os.environ.get("COLAB_CLI_ROOT"):
        raise ValueError("tiny Mac preflight is not the real transport family canary")
    from research.mac_lease import held

    started = time.monotonic()
    timings = []
    with held("b2-preflight", "tiny B2 entrypoint only", seconds=180):
        for i, seed in enumerate((993510401, 993510402, 993510403)):
            unit_started = time.monotonic()
            driver = engine.WatchDriver(seed, inputs / B8_FILE, inputs / FRP_FILE)
            driver.prepare()
            snapshot = driver.snapshot()
            restored = engine.WatchDriver.restore(snapshot)
            # Compare complete future snapshots (not merely one selected action).
            for _ in range(3):
                assert driver.step() == restored.step()
                driver.prepare()
                restored.prepare()
            left = driver.snapshot()
            right = restored.snapshot()
            # Pickle memo layout can differ after roundtrip; compare state arrays
            # and carried policy state by an independent recursive representation.
            from research.bl_served_search.ruler_operator import _jsonable

            def state(d: Any) -> str:
                sim = d.sim
                return canonical(
                    _jsonable(
                        {
                            "arrays": {
                                k: v for k, v in vars(sim).items() if isinstance(v, np.ndarray)
                            },
                            "rng": [x._rng.getstate() for x in sim._rngs],
                            "runtime": {
                                key: value
                                for key, value in vars(d.frozen.runtime).items()
                                if key == "memory"
                                or isinstance(value, (np.ndarray, str, int, float, bool))
                            },
                            "policies": [
                                {
                                    key: getattr(policy, key)
                                    for key in (
                                        "live_vetoes",
                                        "veto_counters",
                                        "boost_counters",
                                        "last_veto",
                                    )
                                    if hasattr(policy, key)
                                }
                                for policy in d.frozen.vector.values()
                            ],
                            "events": sim.get_step_events(),
                        }
                    )
                )

            if not left or not right or state(driver) != state(restored):
                raise ValueError("B2 snapshot next-state parity failed")
            engine.attest(driver.stats.as_dict())
            tiny = {
                "namespace": "b2-tiny-preflight/v1",
                "allocation_digest": "0" * 64,
                "seeds": [993510401, 993510402, 993510403],
                "trajectory_frames": 3,
                "root_stride": 1,
                "escape_total_frames": 1440,
                "unit_timeout_seconds": 120,
            }
            # The production collector, exercised on three real tiny units.
            payload = engine.collect(tiny, i, driver)
            export_json(out.parent / f"tiny-unit-{i}.json", payload)
            elapsed = time.monotonic() - unit_started
            timings.append(elapsed)
            print(canonical({"done": i + 1, "total": 3, "elapsed_seconds": elapsed}), flush=True)
    from research.bl_served_search.b2_package import source_identity

    receipt = {
        "sources": source_identity(),
        "schema": SCHEMA + "/tiny-preflight",
        "status": "PASS_TINY_ONLY",
        "real_units": 3,
        "collector_steps_each": 3,
        "restore_check_steps_each": 3,
        "unit_seconds": timings,
        "elapsed_seconds": time.monotonic() - started,
        "snapshot_restore": "next-state arrays, RNG, memory and event parity",
        "no_production_or_timing_cells": True,
        "production_forecast": "NOT_MEASURED; no L>=800 root batch in this smoke",
    }
    export_json(out, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for flag in ("check", "make-ledger", "run", "merge", "stage", "tiny-preflight"):
        action.add_argument("--" + flag, action="store_true")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--exports", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.tiny_preflight:
        if args.inputs is None or args.out is None:
            parser.error("--tiny-preflight requires --inputs and --out")
        tiny_preflight(args.inputs, args.out)
        return
    if args.spec is None:
        parser.error("--spec is required")
    # A draft check reports the missing design before looking for immutable inputs.
    if args.check:
        validate_spec(json.loads(args.spec.read_text()))
    spec = validate_spec(read_export(args.spec))
    if args.check:
        print(canonical(check(args.spec, args.ledger, args.inputs)))
    elif args.make_ledger:
        ids = [p.stem for p in sorted((args.exports / "units").glob("*.json"))]
        ledger = units.make_ledger(args.exports, spec["namespace"], spec["allocation_digest"], ids)
        units.export_ledger(args.out, ledger)
    elif args.stage:
        stage_unit(args.source, args.exports, spec, args.index, args.receipt)
    elif args.merge:
        merge(spec, args.exports, read_export(args.ledger), args.out)
    else:
        validate_freshness(spec, args.spec.parent / "freshness-audit.json")
        run_unit(spec, args.index, read_export(args.ledger), args.inputs, args.out)


if __name__ == "__main__":
    main()
