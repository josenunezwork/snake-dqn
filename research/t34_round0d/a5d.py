"""A5-D: dense-engine fidelity and throughput on one L4 lane (excluded worlds only).

May run before the Round 0 verdict: it touches no Round 0 byte and no selection
world. One supervised job, K=12 single-thread processes, timeout 5 h, cap 10 u.

1. Fidelity. 8 T3 dense census worlds (b8_s3, envs 0..7) start at t=0 on 8 of the
   12 processes. Each must reproduce its historical endpoints (survival, death
   decision, alive fraction, mass integral, decisions), batch schedule and every
   changed action. Status per world: PASS; EXPLAINED_TIE (pre-registered: the first
   divergence is at a decision where this run batched and either the B8 top-2
   advisory Q gap or the Q gap among the tied-longest candidates is <= 1e-4, the
   known Mac<->x86 float32 BLAS-order flip); DIFFERENCE or INCOMPLETE -> HOLD.
2. Throughput. The other 4 processes start at t=0 on the 30 excluded A5-D worlds
   (4:3:3 order) and every freed process continues with them. Each throughput item
   runs the full paired worker recipe (bare + dense + metric replay), so bare and
   metric costs are measured on L4. The rate is computed by ``sizing.rate_estimates``
   from completed process-seconds and row-steps, never from a completion window.

``prepare-fidelity`` runs on the Mac (reads historical T3 receipts only).
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

import numpy as np

from research.t34_round0.runner import (
    _bare_trace,
    _load_export,
    _metrics_from_trace,
    _write_export,
    _write_new,
)
from research.t34_round0d.common import B8_SHA, read_json, sha256, write_once
from research.t34_round0d.runner import dense_trace, public_trace

FIXTURE_SCHEMA = "t34-round0d-fidelity-fixture/v1"
BENCHMARK_SCHEMA = "t34-round0d-a5d-benchmark/v2"
T3_DENSE = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t3-rollout-labels-20261015/dense")
FIDELITY_WORLDS = 8
PROCESSES = 12
HORIZON = 5000
TIE_EPSILON = 1e-4


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


def first_divergence(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> int | None:
    """Earliest decision at which the batch schedule or the changed actions differ."""
    candidates = []
    exp_b, obs_b = list(expected["batch_decisions"]), list(observed["batch_decisions"])
    for a, b in zip(exp_b, obs_b):
        if a != b:
            candidates.append(min(a, b))
            break
    else:
        if len(exp_b) != len(obs_b):
            longer = exp_b if len(exp_b) > len(obs_b) else obs_b
            candidates.append(longer[min(len(exp_b), len(obs_b))])
    exp_c = {int(c[0]): c for c in expected["changes"]}
    obs_c = {int(c[0]): c for c in observed["changes"]}
    candidates.extend(d for d in set(exp_c) | set(obs_c) if exp_c.get(d) != obs_c.get(d))
    return min(candidates) if candidates else None


def tie_explained(decision: int | None, labels: list[Mapping[str, Any]]) -> bool:
    """Pre-registered tie rule: the divergent decision batched here and was a float32 near-tie."""
    if decision is None:
        return False
    label = next((x for x in labels if int(x["decision"]) == decision), None)
    if label is None:
        return False
    q = np.asarray(label["q"], dtype=np.float64)
    legal = [a for a in range(6) if label["legal"][a]]
    top = sorted((q[a] for a in legal), reverse=True)
    if len(top) >= 2 and top[0] - top[1] <= TIE_EPSILON:
        return True
    steps = {int(a): int(v["alive_steps"]) for a, v in label["outcomes"].items()}
    best = max(steps.values())
    tied = sorted((q[a] for a, s in steps.items() if s == best), reverse=True)
    return len(tied) >= 2 and tied[0] - tied[1] <= TIE_EPSILON


def _observed(full: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_survival": bool(metrics["no_death"]),
        "death_t": None if metrics["no_death"] else len(full["actions"]) - 1,
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


def _work(request: Mapping[str, Any]) -> dict[str, Any]:
    """One A5-D item in a single-threaded spawned process."""
    from research.t34_exit_b8.label import _cpu_ceiling

    _cpu_ceiling(1)
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    import torch

    torch.set_num_threads(1)
    row = dict(request["row"])
    checkpoint = Path(str(request["checkpoint"]))
    result: dict[str, Any] = {
        "kind": request["kind"],
        "index": int(request["index"]),
        "world_id": row["world_id"],
        "mix": row["mix"],
        "started_at": time.time(),
    }
    with tempfile.TemporaryDirectory(prefix="t34-round0d-a5d-") as temporary:
        base = Path(temporary)
        bare_bytes = 0
        if request["kind"] == "throughput":
            started = time.monotonic()
            bare = _bare_trace(row, checkpoint, frames=int(row["horizon"]), device="cpu")
            result["bare_seconds"] = time.monotonic() - started
            _write_export(base / "bare.json", bare)
            bare_bytes = (base / "bare.json").stat().st_size
            started = time.monotonic()
            _metrics_from_trace(_load_export(base / "bare.json"), checkpoint)
            bare_metric = time.monotonic() - started
        started = time.monotonic()
        full = dense_trace(row, checkpoint, frames=int(row["horizon"]), device="cpu")
        result["teacher_seconds"] = time.monotonic() - started
        _write_export(base / "teacher.json", public_trace(full))
        _write_export(base / "teacher_labels.json", full)
        started = time.monotonic()
        metrics = _metrics_from_trace(_load_export(base / "teacher.json"), checkpoint)
        if request["kind"] == "throughput":
            result["metric_seconds"] = bare_metric + time.monotonic() - started
        result["bytes_per_pair"] = (
            bare_bytes
            + (base / "teacher.json").stat().st_size
            + (base / "teacher_labels.json").stat().st_size
            + 128 * 1024  # pair JSON incl. run identity, Round 0 measured order
        )
        if request["kind"] == "fidelity":
            target = Path(str(request["export_dir"])) / f"fidelity-{request['index']:02d}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_export(target, full)
    stats = full["teacher_v2"]
    result.update(
        finished_at=time.time(),
        rollout_row_steps=int(stats["rollout_row_steps"]),
        decisions=int(stats["decisions"]),
        batches=int(stats["batches"]),
        observed=_observed(full, metrics),
        labels_for_tie_rule=(
            [{k: x[k] for k in ("decision", "q", "legal", "outcomes")} for x in full["labels"]]
            if request["kind"] == "fidelity"
            else []
        ),
    )
    return result


def fidelity_status(fixture: Mapping[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    worlds = []
    for i, world in enumerate(fixture["worlds"]):
        result = next((r for r in results if r["kind"] == "fidelity" and r["index"] == i), None)
        if result is None:
            worlds.append({"world_id": world["row"]["world_id"], "status": "INCOMPLETE"})
            continue
        differences = compare(world["expected"], result["observed"])
        status = "PASS"
        divergence = None
        if differences:
            divergence = first_divergence(world["expected"], result["observed"])
            explained = tie_explained(divergence, result["labels_for_tie_rule"])
            status = "EXPLAINED_TIE" if explained else "DIFFERENCE"
        worlds.append(
            {
                "world_id": world["row"]["world_id"],
                "status": status,
                "differences": differences,
                "first_divergence": divergence,
                "teacher_seconds": result["teacher_seconds"],
            }
        )
    statuses = {w["status"] for w in worlds}
    if statuses <= {"PASS"}:
        overall = "PASS"
    elif statuses <= {"PASS", "EXPLAINED_TIE"}:
        overall = "EXPLAINED_TIE"
    else:
        overall = "HOLD"
    return {"status": overall, "worlds": worlds, "tie_epsilon": TIE_EPSILON}


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
    throughput_rows = ordered_rows(bank["excluded_a5d_rows"])
    out = Path(args.output_dir)
    if out.exists():
        raise RuntimeError("output directory must be fresh")
    out.mkdir(parents=True)
    fidelity_items = [
        {"kind": "fidelity", "index": i, "row": world["row"]}
        for i, world in enumerate(fixture["worlds"])
    ]
    throughput_items = [
        {"kind": "throughput", "index": i, "row": row} for i, row in enumerate(throughput_rows)
    ]
    if args.limit:
        fidelity_items = fidelity_items[: args.limit]
        throughput_items = throughput_items[: args.limit]
    # Fidelity occupies 8 processes from t=0; throughput starts at t=0 on the rest
    # and takes every freed process thereafter.
    items = fidelity_items + throughput_items
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
    from research.t34_round0d.sizing import SizingRefused, rate_estimates

    worlds = [
        {k: v for k, v in r.items() if k not in ("observed", "labels_for_tie_rule")}
        for r in results
    ]
    try:
        rates: dict[str, Any] = rate_estimates(worlds, processes)
    except SizingRefused as exc:
        rates = {"error": str(exc)}
    sizes = [r["bytes_per_pair"] for r in results if r["kind"] == "throughput"]
    benchmark = {
        "schema": BENCHMARK_SCHEMA,
        "processes": processes,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "elapsed_seconds": time.time() - started,
        "rates": rates,
        "kill_below_worlds_per_hour": 5.0,
        "fidelity": fidelity_status(fixture, results),
        "export_bytes_per_pair_upper": max(sizes) if sizes else None,
        "worlds": worlds,
        "in_progress_at_stop": [item["row"]["world_id"] for item in pending.values()],
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
    bench.add_argument("--seconds", type=float, default=16200.0)
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
        print(json.dumps(run(args)["rates"], sort_keys=True))


if __name__ == "__main__":
    main()
