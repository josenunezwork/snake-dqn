"""A5-D: dense-engine fidelity and throughput on one L4 lane (excluded worlds only).

Two parts in one supervised job (<= 60 min, <= 3 u), K=12 single-thread processes:

1. Fidelity. The first 8 work items are T3's dense census worlds (b8_s3, envs
   0..7 in bank order). Each must reproduce its historical per-world endpoints
   (survival, death decision, alive fraction, mass integral, decisions) and the
   exact set of batch decisions and changed actions (decision, greedy, chosen).
   Any difference is reported and HOLDS sizing until the operator explains it
   with the revalidated M2 queries; there is no automatic pass.
2. Throughput. The queue continues with the 30 excluded A5-D worlds (4:3:3).
   Completed dense worlds in the 1,200 s window after a 1,200 s cold fill,
   times three, is the lane rate r. KILL if r < 8 worlds/h/lane.

``prepare-fidelity`` runs on the Mac (it reads historical T3 receipts, not any
Round 0 output) and writes a portable, hash-bound fixture for upload.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from research.t34_round0.runner import _load_export, _metrics_from_trace, _write_new
from research.t34_round0d.common import B8_SHA, read_json, sha256, write_once
from research.t34_round0d.runner import dense_trace, public_trace

FIXTURE_SCHEMA = "t34-round0d-fidelity-fixture/v1"
BENCHMARK_SCHEMA = "t34-round0d-a5d-benchmark/v1"
T3_DENSE = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t3-rollout-labels-20261015/dense")
FIDELITY_WORLDS = 8
PROCESSES = 12
COLD_FILL_SECONDS = 1200
WINDOW_SECONDS = 1200
HORIZON = 5000


def _expected(receipt: Mapping[str, Any]) -> dict[str, Any]:
    rows = receipt["trigger_rows"]
    return {
        "episode_survival": bool(receipt["episode_survival"]),
        "death_t": receipt["death_t"],
        "alive_frame_fraction": float(receipt["alive_frame_fraction"]),
        "mass_integral": float(receipt["mass_integral"]),
        "decisions": int(receipt["decisions"]),
        "batch_decisions": [int(row["decision"]) for row in rows],
        "changes": [
            [int(row["decision"]), int(row["greedy"]), int(row["chosen"])]
            for row in rows
            if int(row["chosen"]) != int(row["greedy"])
        ],
    }


def prepare_fidelity(out: Path) -> dict[str, Any]:
    """Mac-only: freeze the first 8 b8_s3 T3 dense census worlds as a portable fixture."""
    from research.t34_pathb.precheck_v2 import _original_worlds

    seeds, rosters = _original_worlds()
    worlds = []
    for env in range(FIDELITY_WORLDS):
        path = T3_DENSE / "ceiling" / f"b8_s3_{env}.json"
        receipt = read_json(path)
        if (
            receipt.get("status") != "PASS"
            or receipt.get("trigger_mode") != "dense"
            or receipt.get("arm") != "b8_s3"
            or receipt.get("env") != env
            or receipt.get("checkpoint", {}).get("sha256") != B8_SHA
            or int(receipt.get("world_seed")) != int(seeds[env])
        ):
            raise ValueError(f"T3 dense receipt is not the registered b8_s3 census world: {path}")
        row = {
            "namespace": "t3-dense-ceiling/v1",
            "world_id": f"t3-dense-ceiling/v1/b8_s3_{env}",
            "world_seed": int(seeds[env]),
            "mix": "scripted",
            "opponents": [list(spec) for spec in rosters[seeds[env]]],
            "opponent_veto": "v8",
            "opponent_veto_lambda": 8.0,
            "horizon": HORIZON,
            "terminal_hero_slot": 0,
        }
        worlds.append(
            {
                "row": row,
                "t3_receipt": {"name": path.name, "sha256": sha256(path)},
                "expected": _expected(receipt),
            }
        )
    payload = {"schema": FIXTURE_SCHEMA, "checkpoint_sha256": B8_SHA, "worlds": worlds}
    write_once(out, payload)
    return payload


def compare(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> list[str]:
    """Return every endpoint/decision difference; empty means exact reproduction."""
    differences = []
    for key in ("episode_survival", "death_t", "decisions"):
        if expected[key] != observed[key]:
            differences.append(f"{key}: expected {expected[key]!r}, observed {observed[key]!r}")
    for key in ("alive_frame_fraction", "mass_integral"):
        if float(expected[key]) != float(observed[key]):
            differences.append(f"{key}: expected {expected[key]!r}, observed {observed[key]!r}")
    if list(expected["batch_decisions"]) != list(observed["batch_decisions"]):
        differences.append("batch decision schedule differs")
    if [list(x) for x in expected["changes"]] != [list(x) for x in observed["changes"]]:
        differences.append(
            f"changed actions differ: expected {expected['changes']}, "
            f"observed {observed['changes']}"
        )
    return differences


def _work(request: Mapping[str, Any]) -> dict[str, Any]:
    """One dense world in a single-threaded spawned process."""
    from research.t34_exit_b8.label import _cpu_ceiling

    _cpu_ceiling(1)
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    import torch

    torch.set_num_threads(1)
    row = dict(request["row"])
    checkpoint = Path(str(request["checkpoint"]))
    started = time.time()
    full = dense_trace(row, checkpoint, frames=int(row["horizon"]), device="cpu")
    finished = time.time()
    with tempfile.TemporaryDirectory(prefix="t34-round0d-a5d-") as temporary:
        public = Path(temporary) / "teacher.json"
        sealed = Path(temporary) / "teacher_labels.json"
        from research.t34_round0.runner import _write_export

        _write_export(public, public_trace(full))
        _write_export(sealed, full)
        metrics = _metrics_from_trace(_load_export(public), checkpoint)
        bytes_public, bytes_sealed = public.stat().st_size, sealed.stat().st_size
        if request["kind"] == "fidelity":
            target = Path(str(request["export_dir"])) / f"fidelity-{request['index']:02d}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_export(target, full)
    death = None if metrics["no_death"] else len(full["actions"]) - 1
    observed = {
        "episode_survival": bool(metrics["no_death"]),
        "death_t": death,
        "alive_frame_fraction": float(metrics["survival_fraction"]),
        "mass_integral": float(metrics["mass_integral"]),
        "decisions": int(metrics["decisions"]),
        "batch_decisions": [int(label["decision"]) for label in full["labels"]],
        "changes": [
            [int(x["decision"]), int(x["student_action"]), int(x["action"])]
            for x in full["labels"]
            if int(x["action"]) != int(x["student_action"])
        ],
    }
    return {
        "kind": request["kind"],
        "index": int(request["index"]),
        "world_id": row["world_id"],
        "mix": row["mix"],
        "started_at": started,
        "finished_at": finished,
        "seconds": finished - started,
        "observed": observed,
        "dense": {k: v for k, v in full["teacher_v2"].items() if isinstance(v, int)},
        "bytes_public": bytes_public,
        "bytes_sealed": bytes_sealed,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if platform.system() == "Darwin" and not args.local_smoke:
        raise RuntimeError("A5-D measures the L4 lane; use --local-smoke only for tests")
    checkpoint = Path(args.checkpoint)
    if sha256(checkpoint) != B8_SHA:
        raise RuntimeError("checkpoint differs from released B8 s3")
    fixture = read_json(Path(args.fidelity))
    bank = read_json(Path(args.bank))
    if fixture.get("schema") != FIXTURE_SCHEMA or len(fixture["worlds"]) != FIDELITY_WORLDS:
        raise RuntimeError("fidelity fixture differs from the registered 8 census worlds")
    from research.t34_round0d.bank import ordered_rows, validate_bank

    if not args.local_smoke:
        validate_bank(bank)
    a5d_rows = ordered_rows(bank["excluded_a5d_rows"])
    out = Path(args.output_dir)
    if out.exists():
        raise RuntimeError("output directory must be fresh")
    out.mkdir(parents=True)
    items = [
        {"kind": "fidelity", "index": i, "row": world["row"]}
        for i, world in enumerate(fixture["worlds"])
    ] + [{"kind": "throughput", "index": i, "row": row} for i, row in enumerate(a5d_rows)]
    if args.limit:
        items = items[: args.limit]
    started = time.time()
    deadline = started + args.seconds
    processes = args.processes
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=processes, mp_context=multiprocessing.get_context("spawn")
    )
    pending: dict[Any, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    cursor, heartbeat = 0, started
    try:
        while (cursor < len(items) or pending) and time.time() < deadline:
            while cursor < len(items) and len(pending) < processes:
                item = dict(items[cursor], checkpoint=str(checkpoint), export_dir=str(out))
                pending[pool.submit(_work, item)] = items[cursor]
                cursor += 1
            done, _ = concurrent.futures.wait(
                pending, timeout=5.0, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                item = pending.pop(future)
                try:
                    results.append(future.result())
                except Exception as exc:  # reported, never retried
                    errors.append({"world_id": item["row"]["world_id"], "error": repr(exc)})
            if time.time() - heartbeat >= 110:
                heartbeat = time.time()
                print(
                    json.dumps(
                        {
                            "event": "t34_round0d_a5d_progress",
                            "elapsed_seconds": heartbeat - started,
                            "completed": len(results),
                            "errors": len(errors),
                        }
                    ),
                    flush=True,
                )
    finally:
        for future in pending:
            future.cancel()
        for process in list(getattr(pool, "_processes", {}).values()):
            process.terminate()
        pool.shutdown(wait=False, cancel_futures=True)
    elapsed = time.time() - started
    window_start, window_end = (
        started + COLD_FILL_SECONDS,
        started + COLD_FILL_SECONDS + WINDOW_SECONDS,
    )
    in_window = [r for r in results if window_start <= r["finished_at"] < window_end]
    window_complete = elapsed >= COLD_FILL_SECONDS + WINDOW_SECONDS
    fidelity = []
    for i, world in enumerate(fixture["worlds"]):
        result = next((r for r in results if r["kind"] == "fidelity" and r["index"] == i), None)
        if result is None:
            fidelity.append({"world_id": world["row"]["world_id"], "status": "INCOMPLETE"})
            continue
        differences = compare(world["expected"], result["observed"])
        fidelity.append(
            {
                "world_id": world["row"]["world_id"],
                "status": "PASS" if not differences else "DIFFERENCE",
                "differences": differences,
                "seconds": result["seconds"],
            }
        )
    status = "PASS" if all(item["status"] == "PASS" for item in fidelity) else "HOLD"
    sizes = [2 * r["bytes_public"] + r["bytes_sealed"] + 8192 for r in results]
    rate = len(in_window) * 3600.0 / WINDOW_SECONDS if window_complete else None
    benchmark = {
        "schema": BENCHMARK_SCHEMA,
        "processes": processes,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "elapsed_seconds": elapsed,
        "window": {
            "cold_fill_seconds": COLD_FILL_SECONDS,
            "steady_window_seconds": WINDOW_SECONDS,
            "complete": window_complete,
            "completed_in_window": len(in_window),
            "completed_in_window_by_mix": {
                mix: sum(r["mix"] == mix for r in in_window)
                for mix in ("scripted", "frozen", "mixed")
            },
            "completed_worlds_per_hour": rate,
            "kill_below_worlds_per_hour": 8.0,
            "kill": rate is None or rate < 8.0,
        },
        "fidelity": {"status": status, "worlds": fidelity},
        "export_bytes_per_pair_upper": max(sizes) if sizes else None,
        "worlds": [{k: v for k, v in r.items() if k != "observed"} for r in results],
        "errors": errors,
        "local_smoke": bool(args.local_smoke),
    }
    _write_new(out / "benchmark.json", benchmark)
    return benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-fidelity")
    prep.add_argument("--out", type=Path, required=True)
    bench = sub.add_parser("run")
    bench.add_argument("--fidelity", required=True)
    bench.add_argument("--bank", required=True)
    bench.add_argument("--checkpoint", required=True)
    bench.add_argument("--output-dir", required=True)
    bench.add_argument("--seconds", type=float, default=3000.0)
    bench.add_argument("--processes", type=int, default=PROCESSES)
    bench.add_argument("--limit", type=int, default=0)
    bench.add_argument("--local-smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-fidelity":
        result = prepare_fidelity(args.out)
        print(json.dumps({"worlds": len(result["worlds"])}))
    else:
        if not args.local_smoke and args.processes != PROCESSES:
            parser.error("A5-D requires K=12 processes")
        print(json.dumps(run(args)["window"], sort_keys=True))


if __name__ == "__main__":
    main()
