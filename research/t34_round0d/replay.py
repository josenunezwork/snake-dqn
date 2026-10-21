"""A2-style cross-runtime replay of the frozen 10% sample on a second L4 runtime.

Deviation 1 from Round 0 (frozen pre-data): during Round 0-D the Mac is inside
the quiet-host Path B study, so the replay that Round 0 runs on the Mac runs
instead as one separate supervised L4 job (<= 60 min, <= 2 u). The sample rule is
Round 0's (``research.t34_round0.history.selected_rows``: ceil(10%) per chunk,
mix-stratified, bank order), selected from chunk slices only, before any outcome
is read. Both arms of every selected world are re-simulated from their exported
action traces; any hash or outcome divergence fails the replay (INCONCLUSIVE at
analysis). The replay runtime must differ from every producing runtime.

``mac`` subcommand (operator, frozen checkout): reserves the sole ``replay``
phase, builds and dispatches the job, verifies its receipt, and seals
``replay/replay.json``. ``job`` subcommand: runs inside the L4 runtime.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

from research.t34_round0.make_manifest import RATE_UNITS_PER_HOUR, RESERVE_FRACTION
from research.t34_round0.runner import _load, _metrics_from_trace, _write_new
from research.t34_round0d.common import BINDING_DOCS, PACKAGE, ROOT, read_json, sha256

REPORT_SCHEMA = "t34-round0d-replay-report/v1"
REPLAY_SCHEMA = "t34-round0d-replay/v1"
METRIC_KEYS = (
    "mass_integral",
    "survival_fraction",
    "no_death",
    "deaths",
    "decisions",
    "boost_decisions",
    "action_counts",
)


def _sidecars(path: Path) -> list[Path]:
    return [path, path.with_name(path.name + ".sha256"), path.with_name(path.name + ".complete")]


def plan(chunk_roots: list[Path], registration: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the sample from chunk slices and list exactly the files to upload."""
    from research.t34_round0d.history import selected_rows

    bank = read_json(PACKAGE / "BANK_ROUND0D.json")
    receipts = [read_json(root / "chunk.json") for root in chunk_roots]
    selected = selected_rows(bank["rows"], receipts)
    from research.t34_round0.runner import _safe_world_id

    by_world: dict[str, tuple[Path, dict[str, Any]]] = {}
    runtimes = set()
    tags = {_safe_world_id(row["world_id"]): row["world_id"] for row in selected}
    for root, receipt in zip(chunk_roots, receipts):
        runtimes.add(receipt["run_identity"]["runtime_id"])
        # Only sampled pairs are opened; the sample was fixed from slices above.
        for item in receipt["pairs"]:
            tag = Path(item["path"]).stem.split("-", 1)[-1]
            if tag not in tags:
                continue
            if sha256(root / item["path"]) != item["sha256"]:
                raise ValueError("chunk pair receipt hash differs")
            by_world[tags[tag]] = (root, item)
    if set(by_world) != set(tags.values()):
        raise ValueError("a sampled world has no completed pair (missing pair)")
    worlds, inputs = [], []
    for index, row in enumerate(selected):
        root, item = by_world[row["world_id"]]
        pair_path = root / item["path"]
        pair = read_json(pair_path)
        entry: dict[str, Any] = {
            "world_id": row["world_id"],
            "pair": {"dest": f"sample/{index:03d}/pair.json", "sha256": item["sha256"]},
            "traces": {},
        }
        inputs.append(
            {"path": str(pair_path), "sha256": item["sha256"], "dest": entry["pair"]["dest"]}
        )
        for arm in ("bare", "teacher"):
            trace = root / pair["arms"][arm]["trace"]["path"]
            dest = f"sample/{index:03d}/{arm}.json"
            entry["traces"][arm] = {"dest": dest, "sha256": pair["arms"][arm]["trace"]["sha256"]}
            for path, suffix in zip(_sidecars(trace), ("", ".sha256", ".complete")):
                inputs.append({"path": str(path), "sha256": sha256(path), "dest": dest + suffix})
        worlds.append(entry)
    return {
        "schema": "t34-round0d-replay-plan/v1",
        "selected_ids": [row["world_id"] for row in selected],
        "producer_runtime_ids": sorted(runtimes),
        "worlds": worlds,
        "inputs": inputs,
        "registration_commit": registration["commit"],
    }


def _check_world(request: Mapping[str, Any]) -> dict[str, Any]:
    from research.t34_exit_b8.exports import load_world
    from research.t34_exit_b8.label import _cpu_ceiling

    _cpu_ceiling(1)
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    base = Path(str(request["input_dir"]))
    checkpoint = Path(str(request["checkpoint"]))
    world = request["world"]
    pair_path = base / world["pair"]["dest"]
    divergences, traces = [], {}
    if sha256(pair_path) != world["pair"]["sha256"]:
        return {"world_id": world["world_id"], "divergences": ["pair bytes differ"]}
    pair = _load(pair_path)
    for arm in ("bare", "teacher"):
        meta = pair["arms"][arm]
        path = base / world["traces"][arm]["dest"]
        try:
            if sha256(path) != meta["trace"]["sha256"]:
                raise ValueError("trace bytes differ")
            exported = load_world(path)
            if exported.get("labels"):
                raise ValueError("public trace carries labels")
            measured = _metrics_from_trace(exported, checkpoint)
            for key in METRIC_KEYS:
                if measured[key] != meta.get(key):
                    raise ValueError(f"outcome mismatch: {key}")
            if (
                meta["trace"]["final_state_hash"]
                != exported["hashes"][str(len(exported["actions"]))]
            ):
                raise ValueError("final state hash differs")
            traces[arm] = sha256(path)
        except Exception as exc:  # every divergence is reported, never dropped
            divergences.append(f"{arm}: {exc}")
    return {
        "world_id": world["world_id"],
        "pair_sha256": world["pair"]["sha256"],
        "traces": traces,
        "divergences": divergences,
    }


def job(args: argparse.Namespace) -> dict[str, Any]:
    """Runtime side: re-simulate every sampled trace with K=12 single-thread processes."""
    if platform.system() == "Darwin":
        raise RuntimeError("Round 0-D replay runs on a second L4 runtime, not the Mac")
    base = Path(args.input_dir)
    replay_plan = _load(base / "REPLAY_PLAN.json")
    admission = _load(base / "REPLAY_ADMISSION.json")
    if admission.get("schema") != "t34-round0d-admission/v1" or not str(
        admission.get("phase_path", "")
    ).endswith("/replay"):
        raise RuntimeError("replay job lacks its create-only replay admission")
    requests = [
        {"input_dir": str(base), "checkpoint": args.checkpoint, "world": world}
        for world in replay_plan["worlds"]
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=12, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_check_world, requests))
    report = {
        "schema": REPORT_SCHEMA,
        "runtime_id": args.runtime_id,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "selected_ids": replay_plan["selected_ids"],
        "results": results,
    }
    _write_new(Path(args.output_dir) / "report.json", report)
    return report


def replay_manifest(
    registration: Mapping[str, Any], replay_plan_path: Path, admission: Path, out: Path
) -> dict[str, Any]:
    """The single replay job on a second L4 runtime, sized from A5-D at ratification.

    Timeout, estimate and cap come from ``registration["sizing"]["replay"]``
    (``sizing.replay_sizing``: timeout >= max(estimate + 25, 2 x estimate, 60)).
    """
    replay_plan = read_json(replay_plan_path)
    sized = registration["sizing"]["replay"]
    timeout = int(sized["timeout_minutes"])
    unit_cap = float(sized["unit_cap"])
    if unit_cap != timeout * RATE_UNITS_PER_HOUR / 60.0:
        raise ValueError("replay cap must price its registered timeout at 2.0 u/h")
    return {
        "job_id": "t34-round0d-replay-01",
        "commit": registration["commit"],
        "script": "research/t34_round0d/replay.sh",
        "claim": "ROUND0D_REPLAY_SOLE_PHASE",
        "env": {
            "PYTHONUNBUFFERED": "1",
            "T34_INPUT_MODE": "public",
            "T34_INPUT_CATALOG": "PUBLIC_INPUTS.json",
            "T34_ROUND0D_RUNTIME_ID": "t34-round0d-replay-01",
            "T34_ROUND0D_COMMIT": registration["commit"],
        },
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": sized["estimate_minutes"],
        },
        "timeout_minutes": timeout,
        "output_dir": str(out / "job"),
        "inputs": [
            {
                "path": str(ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json"),
                "sha256": sha256(ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json"),
                "dest": "PUBLIC_INPUTS.json",
            },
            {
                "path": str(replay_plan_path),
                "sha256": sha256(replay_plan_path),
                "dest": "REPLAY_PLAN.json",
            },
            {"path": str(admission), "sha256": sha256(admission), "dest": "REPLAY_ADMISSION.json"},
            *replay_plan["inputs"],
        ],
        "budget": {
            "unit_cap": unit_cap,
            "reserve_units": unit_cap * RESERVE_FRACTION,
            "rate_units_per_hour": RATE_UNITS_PER_HOUR,
            "other_assignments": {},
        },
        "lease_seconds": 600,
        "max_export_bytes": 64 * 1024**2,
        "source_paths": sorted({*registration["source_hashes"], *BINDING_DOCS}),
    }


def settle(report: Mapping[str, Any], replay_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Turn the runtime report into the sealed replay verdict consumed by analysis."""
    divergences = [
        {"world_id": r["world_id"], "reason": reason}
        for r in report["results"]
        for reason in r["divergences"]
    ]
    checked = [
        {"world_id": r["world_id"], "pair_sha256": r["pair_sha256"], "traces": r["traces"]}
        for r in report["results"]
        if not r["divergences"]
    ]
    cross_runtime = report["runtime_id"] not in replay_plan["producer_runtime_ids"]
    if not cross_runtime:
        divergences.append({"world_id": None, "reason": "replay runtime equals a producer"})
    passed = (
        not divergences
        and report["selected_ids"] == replay_plan["selected_ids"]
        and len(checked) == len(replay_plan["selected_ids"])
    )
    return {
        "schema": REPLAY_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "platform": report["platform"],
        "replay_runtime_id": report["runtime_id"],
        "producer_runtime_ids": replay_plan["producer_runtime_ids"],
        "selected_ids": replay_plan["selected_ids"],
        "sampling": "Round 0 rule: ceil(10%) per chunk, mix-stratified, bank order",
        "checked_pairs": checked,
        "divergences": divergences,
    }


def mac(args: argparse.Namespace) -> int:
    """Operator side, frozen checkout: one reserved replay phase, one supervised job."""
    from research.t34_round0d.history import (
        policy,
        require_active_phase,
        require_chunks,
        run_phase,
    )

    registration = read_json(args.ratification)
    root, _, _ = policy(registration)
    if Path(args.out) != root / "replay/replay.json":
        raise SystemExit("replay must use the fixed study replay/replay.json")
    chunks = [Path(path) for path in args.chunk_dirs]

    def action(out: Path) -> dict[str, Any]:
        from research.colab_cli.supervisor import validate

        require_active_phase(registration, "replay")
        require_chunks(chunks, registration)
        for chunk in chunks:
            if (chunk / "fault.json").is_file():
                return {
                    "schema": REPLAY_SCHEMA,
                    "status": "STOP_CONTROLLER_FAULT",
                    "fault_sha256": sha256(chunk / "fault.json"),
                }
        replay_plan = plan(chunks, registration)
        plan_path = out / "REPLAY_PLAN.json"
        _write_new(plan_path, replay_plan)
        manifest = replay_manifest(registration, plan_path, out / "admission.json", out)
        validate(manifest, repo=ROOT)
        manifest_path = out / "replay_manifest.json"
        _write_new(manifest_path, manifest)
        subprocess.run(
            [
                str(ROOT / "research/colab_cli/launch.sh"),
                "--state-dir",
                str(args.state_dir),
                "--colab",
                str(args.colab),
                "run",
                str(manifest_path),
            ],
            cwd=ROOT,
            check=True,
        )
        receipt = read_json(out / "job/receipt.json")
        report_path = out / "job/exports/replay/report.json"
        if (
            receipt.get("status") != "PASS"
            or receipt.get("stop_verified") is not True
            or not any(
                e.get("path") == "replay/report.json" and e.get("sha256") == sha256(report_path)
                for e in receipt.get("exports", [])
            )
        ):
            raise RuntimeError("replay job did not PASS with a verified, released runtime")
        return settle(read_json(report_path), replay_plan)

    result = run_phase(registration, "replay", action)
    print(json.dumps({"status": result["status"]}))
    return 0 if result["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runtime = sub.add_parser("job")
    runtime.add_argument("--input-dir", required=True)
    runtime.add_argument("--checkpoint", required=True)
    runtime.add_argument("--output-dir", required=True)
    runtime.add_argument("--runtime-id", required=True)
    operator = sub.add_parser("mac")
    operator.add_argument("--ratification", type=Path, required=True)
    operator.add_argument("--chunk-dirs", nargs="+", required=True)
    operator.add_argument("--state-dir", type=Path, required=True)
    operator.add_argument("--colab", type=Path, default=Path.home() / ".local/bin/colab")
    operator.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "job":
        job(args)
        return 0
    return mac(args)


if __name__ == "__main__":
    raise SystemExit(main())
