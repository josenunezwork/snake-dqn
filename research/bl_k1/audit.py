"""Independent stock/native trajectory audit; no timing qualification is produced."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

from research.bl_served_search.exports import export_json, verify_export
from research.bl_served_search.ruler_operator import ExactT3Operator
from research.mac_lease import DEFAULT_LEASE_PATH, held

DIAGNOSTIC_EXCLUSIONS = ["live_vetoes (probe/v5/v7/v8 bookkeeping only)"]


def stable(value: Any) -> bytes:
    """Exclude only wall-clock diagnostic fields; preserve exact array bytes."""
    if isinstance(value, np.ndarray):
        payload = stable(value.tolist()) if value.dtype.hasobject else value.tobytes()
        return repr((value.dtype.str, value.shape)).encode() + payload
    if isinstance(value, dict):
        return (
            b"{"
            + b"".join(
                str(key).encode() + b":" + stable(item)
                for key, item in sorted(value.items(), key=lambda x: str(x[0]))
                if not str(key).endswith(("_seconds", "_seconds_total", "_seconds_max"))
            )
            + b"}"
        )
    if isinstance(value, (set, frozenset)):
        return stable(sorted(value))
    if isinstance(value, (tuple, list)):
        return b"[" + b"".join(stable(item) + b"," for item in value) + b"]"
    if hasattr(value, "__dict__"):
        return stable(vars(value))
    return repr(value).encode()


def world_state(sim: Any) -> dict[str, Any]:
    """Physical arrays plus authoritative ordered food and selector flags."""
    return {
        **{key: value for key, value in vars(sim).items() if isinstance(value, np.ndarray)},
        "food_cells": sim.food_cells,
        "corpse_cells": sim.corpse_cells,
        "food_set": sim.food_set,
        "phase": (
            getattr(sim, "_policy_phase_prepared", False),
            getattr(sim, "_policy_step_in_progress", False),
        ),
        "constructor_food_pending": getattr(sim, "_constructor_food_draws_pending", None),
    }


class Tape:
    """Hash every Q/action/state event; the two trajectories own independent roots."""

    def __init__(self) -> None:
        self.hashes = {
            key: hashlib.sha256()
            for key in ("q", "actions", "world", "rng", "carry", "diagnostics")
        }
        self.counts = dict.fromkeys(self.hashes, 0)
        self.worlds: dict[int, Any] = {}

    def add(self, key: str, value: Any) -> None:
        payload = stable(value)
        self.hashes[key].update(len(payload).to_bytes(8, "little") + payload)
        self.counts[key] += 1

    def state(self, world: Any) -> None:
        sim = world.sim
        self.add(
            "world",
            world_state(sim),
        )
        self.add("rng", [getattr(rng, "_rng", rng).getstate() for rng in sim._rngs])
        self.add(
            "carry",
            {
                "runtime": {
                    key: value
                    for key, value in vars(world.runtime).items()
                    if isinstance(value, np.ndarray) or key == "memory"
                },
                "policies": [
                    {
                        key: getattr(policy, key)
                        for key in ("veto_counters", "boost_counters", "last_veto")
                        if hasattr(policy, key)
                    }
                    for policy in [world.hero, *world.opponents]
                ],
                "phase": (
                    getattr(sim, "_policy_phase_prepared", False),
                    getattr(sim, "_policy_step_in_progress", False),
                ),
            },
        )
        self.add(
            "diagnostics",
            [getattr(policy, "live_vetoes", {}) for policy in [world.hero, *world.opponents]],
        )

    def result(self) -> dict[str, Any]:
        return {
            key: {"sha256": value.hexdigest(), "events": self.counts[key]}
            for key, value in self.hashes.items()
        }

    @contextlib.contextmanager
    def recording(self) -> Iterator[None]:
        from research.t3_rollout_labels_20261015.bank_engine import CapturedWorld
        from src.simd_env import ego2s_policy
        from src.simd_env.grid_sim import GridBatchSim
        from src.simd_env.vector61_policy import Vector61Runtime, Vector61SimdPolicy

        clone = CapturedWorld.clone
        step = GridBatchSim.step
        observe = Vector61Runtime.observe_step
        vector_q = Vector61SimdPolicy.q_values
        hero_q = ego2s_policy.forward_q

        def cloned(world: Any) -> Any:
            result = clone(world)
            self.worlds[id(result.runtime)] = result
            return result

        def stepped(sim: Any, actions: np.ndarray, *args: Any, **kwargs: Any) -> Any:
            self.add("actions", actions)
            result = step(sim, actions, *args, **kwargs)
            self.add(
                "world",
                world_state(sim),
            )
            self.add("rng", [getattr(rng, "_rng", rng).getstate() for rng in sim._rngs])
            return result

        def observed(runtime: Any, sim: Any, *args: Any, **kwargs: Any) -> Any:
            result = observe(runtime, sim, *args, **kwargs)
            self.state(self.worlds[id(runtime)])
            return result

        def vector_forward(policy: Any, *args: Any, **kwargs: Any) -> Any:
            q = vector_q(policy, *args, **kwargs)
            self.add("q", q.detach().cpu().numpy())
            return q

        def hero_forward(*args: Any, **kwargs: Any) -> Any:
            q = hero_q(*args, **kwargs)
            self.add("q", q.detach().cpu().numpy())
            return q

        with (
            patch.object(CapturedWorld, "clone", cloned),
            patch.object(GridBatchSim, "step", stepped),
            patch.object(Vector61Runtime, "observe_step", observed),
            patch.object(Vector61SimdPolicy, "q_values", vector_forward),
            patch.object(ego2s_policy, "forward_q", hero_forward),
        ):
            yield


def compare(reference: dict[str, Any], candidate: dict[str, Any], mode: str) -> list[str]:
    """Diagnostic counters are required in replay mode and excluded in kernel mode."""
    keys = [key for key in reference if key != "diagnostics" or mode == "replay"]
    return [key for key in keys if reference[key] != candidate[key]]


def run_case(fixture: dict[str, Any], mode: str) -> dict[str, Any]:
    operator = ExactT3Operator(mode)
    # Force reconstruction before recording so model construction is not a Q event.
    operator._restore(fixture)
    tape = Tape()
    with tape.recording():
        output = operator.batch(fixture)
    return {
        "trace": tape.result(),
        "semantic": output["semantic"],
        "native_v8": output["native_v8"],
    }


def run(manifest: dict[str, Any], out: Path, *, smoke: bool = False) -> dict[str, Any]:
    """Audit every declared case in independent stock/replay/kernel executions."""
    rows = manifest["fixtures"] + manifest["fresh_roots"]
    repetitions = 1 if smoke else 30
    summary = []
    started = time.monotonic()
    for row in rows[:1] if smoke else rows:
        repeats = repetitions if row in manifest["fixtures"] else 1
        for repeat in range(repeats):
            if time.monotonic() - started >= 14400:
                raise RuntimeError("identity 4-hour cap exhausted; retain incomplete attempt")
            guard = None
            if __import__("platform").system() == "Darwin":
                from research.bl_mac_latency.host import check_guard, process_screen

                guard = {**check_guard(), "load": process_screen()}
            stock = run_case(row, "stock")
            cases = {}
            for mode in ("replay", "kernel"):
                candidate = run_case(row, mode)
                failures = compare(stock["trace"], candidate["trace"], mode)
                if stock["semantic"] != candidate["semantic"]:
                    failures.append("semantic")
                stats = candidate["native_v8"]
                if row["mix"] != "scripted" and stats["rows"] <= 0:
                    failures.append("zero_native_rows")
                if (
                    stats["rows"]
                    != stats["kernel_rows"] + stats["replayed_rows"] + stats["fallback_rows"]
                ):
                    failures.append("unattested_rows")
                cases[mode] = {**candidate, "mismatches": failures}
            record = {
                "fixture_id": row["fixture_id"],
                "repeat": repeat,
                "guard": guard,
                "stock": stock,
                "modes": cases,
            }
            case_path = out / "cases" / f"{row['fixture_id']}-{repeat:02d}.json"
            exported = export_json(case_path, record)
            summary.append(
                {
                    "fixture_id": row["fixture_id"],
                    "repeat": repeat,
                    "receipt": str(case_path.relative_to(out)),
                    "sha256": exported["sha256"],
                    "modes": {mode: value["native_v8"] for mode, value in cases.items()},
                }
            )
            print(json.dumps({"completed": len(summary), "case": row["fixture_id"]}), flush=True)
            if any(value["mismatches"] for value in cases.values()):
                return {"status": "INVALID", "failed_case": record, "completed": len(summary)}
    return {
        "status": "PRECHECK_ONLY" if smoke else "ACTION_IDENTITY_PASS",
        "schema": "bl-k1-identity/v1",
        "completed": len(summary),
        "cases": summary,
        "elapsed_seconds": time.monotonic() - started,
        "kernel_diagnostic_exclusions": DIAGNOSTIC_EXCLUSIONS,
        "compared": ["Q bytes", "actions", "world arrays", "RNG", "runtime memory", "last_veto"],
        "replay_diagnostics_compared": True,
        "timing_qualification": "NOT_RUN",
    }


def verify_identity(path: Path, manifest: dict[str, Any], manifest_hash: str) -> str:
    """Independently reconcile every immutable case before Mac timing admission."""
    checksum = verify_export(path)
    value = json.loads(path.read_text())
    expected = {
        (row["fixture_id"], repeat)
        for rows, repeats in ((manifest["fixtures"], 30), (manifest["fresh_roots"], 1))
        for row in rows
        for repeat in range(repeats)
    }
    if (
        value.get("status") != "ACTION_IDENTITY_PASS"
        or value.get("manifest_sha256") != manifest_hash
    ):
        raise ValueError("full same-manifest identity receipt required")
    if value.get("platform", {}) != {
        "system": platform.system(),
        "machine": platform.machine(),
        "node": platform.node(),
        "python": platform.python_version(),
    }:
        raise ValueError("identity must come from the same Mac/platform/Python")
    if (
        platform.system() != "Darwin"
        or any(item != "1" for item in value.get("threads", {}).values())
        or len(value.get("threads", {})) != 5
    ):
        raise ValueError("Mac one-thread identity required")
    cases = value.get("cases", [])
    if value.get("completed") != 294 or len(cases) != 294:
        raise ValueError("identity requires exactly 294 completed cases")
    seen = set()
    mixes = {
        row["fixture_id"]: row["mix"] for row in manifest["fixtures"] + manifest["fresh_roots"]
    }
    for case in cases:
        key = (case["fixture_id"], case["repeat"])
        if key in seen or key not in expected:
            raise ValueError("duplicate or unexpected identity case")
        seen.add(key)
        name = f"cases/{key[0]}-{key[1]:02d}.json"
        if case.get("receipt") != name:
            raise ValueError("identity case path mismatch")
        case_path = path.parent / name
        if verify_export(case_path) != case["sha256"]:
            raise ValueError("identity case hash mismatch")
        record = json.loads(case_path.read_text())
        if (record["fixture_id"], record["repeat"]) != key:
            raise ValueError("identity raw case differs")
        for mode in ("replay", "kernel"):
            candidate, stock = record["modes"][mode], record["stock"]
            if (
                candidate["mismatches"]
                or compare(stock["trace"], candidate["trace"], mode)
                or stock["semantic"] != candidate["semantic"]
            ):
                raise ValueError("identity trace/action mismatch")
            for field in ("q", "actions", "world", "rng", "carry"):
                if stock["trace"][field]["events"] <= 0:
                    raise ValueError("identity missing trace events")
            stats = candidate["native_v8"]
            if mixes[key[0]] != "scripted" and (
                stats["rows"] <= 0
                or stats["root_stock_rows"] <= 0
                or stats["rows"] != stock["native_v8"]["stock_rows"]
            ):
                raise ValueError("identity lacks native/stock/root row attestation")
            if (
                stats["rows"]
                != stats["kernel_rows"] + stats["replayed_rows"] + stats["fallback_rows"]
            ):
                raise ValueError("identity native row accounting mismatch")
            if case["modes"][mode] != stats:
                raise ValueError("identity summary/raw attestation mismatch")
            if mode == "kernel" and (
                stats["kernel_rows"] != stats["rows"]
                or stats["replayed_rows"]
                or stats["fallback_rows"]
            ):
                raise ValueError("kernel-only receipt used replay or fallback")
            if mode == "replay" and stats["kernel_rows"]:
                raise ValueError("replay receipt used kernel-only mode")
    if seen != expected:
        raise ValueError("identity missing cases")
    return checksum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--lease-owner", default="bl-k1-identity")
    parser.add_argument("--lease-path", type=Path, default=DEFAULT_LEASE_PATH)
    args = parser.parse_args()
    from research.bl_k1.prepare import verify_manifest

    manifest = verify_manifest(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("identity.json", "stopped.json", "cases"):
        if (args.out / name).exists():
            raise FileExistsError(f"fresh identity attempt required: {args.out / name}")
    (args.out / "cases").mkdir()
    with held(
        args.lease_owner, "bl-k1 identity audit", seconds=14400, path=args.lease_path
    ) as lease:
        try:
            receipt = run(manifest, args.out, smoke=args.smoke)
        except Exception as exc:
            export_json(
                args.out / "stopped.json",
                {"status": "STOPPED", "error": str(exc), "mac_lease": lease.receipt},
            )
            raise
        receipt["manifest_sha256"] = verify_export(args.manifest)
        receipt["platform"] = {
            "system": platform.system(),
            "machine": platform.machine(),
            "node": platform.node(),
            "python": platform.python_version(),
        }
        receipt["threads"] = {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        }
        receipt["diagnostic_only"] = True
        receipt["argv"] = sys.argv
        receipt["source_closure"] = manifest["sources"]
        receipt["mac_lease"] = lease.receipt
        export_json(args.out / "identity.json", receipt)
        if receipt["status"] == "INVALID":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
