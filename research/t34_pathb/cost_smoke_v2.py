"""Bounded, cost-only T34 unpaced terminal-world smoke.

This tool deliberately records no gameplay outcome fields.  It exists only to
measure the operational cost of driving the real assisted adapter through a
full H5000 terminal world while the host is contended.  Its development seeds
are deterministic and rejected if they overlap the prospective efficacy bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

HORIZON = 5000
MIXES = ("scripted", "frozen", "mixed")
DEFAULT_MAX_ELAPSED_SECONDS = 30 * 60
PACED_HOURS = 20.370
TIMING_HOURS = 0.361
TERMINAL_WORLDS = 144
WATCH_SESSIONS = 32
SCHEMA = "t34-pathb-v2-cost-smoke/v1"


class CostSmokeError(RuntimeError):
    """The bounded cost-only smoke could not safely start or continue."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _development_seed(mix: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"t34-retention-fixes2/cost-01/{mix}".encode()).digest()[:4], "big"
    )


def development_rows(registration: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build one deterministic, explicitly non-registered terminal row per mix."""
    registered = {
        row.get("seed")
        for key in ("terminal_rows", "product_rows")
        for row in registration.get(key, [])
        if isinstance(row, Mapping)
    }
    timing = registration.get("timing_row", {})
    if isinstance(timing, Mapping):
        registered.add(timing.get("seed"))
    rows = []
    for mix in MIXES:
        seed = _development_seed(mix)
        if seed in registered:
            raise CostSmokeError(f"development cost seed overlaps registered seed: {seed}")
        rows.append(
            {
                "kind": "cost_only_development_terminal",
                "mix": mix,
                "seed": seed,
                "development_namespace": "t34-retention-fixes2/cost-01",
            }
        )
    return rows


def materialize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Use the exact terminal roster authority without touching the efficacy bank."""
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015 import bank_engine

    context = gid._context(1)
    plans = bank_engine.planned_rosters(
        context, [int(row["seed"]) for row in rows], [str(row["mix"]) for row in rows]
    )
    if len(plans) != len(rows):
        raise CostSmokeError("terminal roster authority returned an incomplete cost smoke")
    result = []
    for row, plan in zip(rows, plans):
        if plan.get("seed") != row["seed"] or plan.get("mix") != row["mix"]:
            raise CostSmokeError("terminal roster authority changed cost-smoke identity")
        result.append({**dict(row), "roster": plan})
    return result


def _preflight() -> dict[str, str]:
    batt = subprocess.check_output(["pmset", "-g", "batt"], text=True)
    if "AC Power" not in batt:
        raise CostSmokeError("refusing cost smoke: Mac is not on AC power")
    lid = subprocess.check_output(["ioreg", "-r", "-k", "AppleClamshellState"], text=True)
    if '"AppleClamshellState" = No' not in lid:
        raise CostSmokeError("refusing cost smoke: lid is closed")
    therm = subprocess.check_output(["pmset", "-g", "therm"], text=True)
    lowered = therm.lower()
    if (
        "warning" in lowered
        and "no thermal warning" not in lowered
        and "no performance warning" not in lowered
    ):
        raise CostSmokeError("refusing cost smoke: thermal or performance warning reported")
    return {"power": "AC", "lid": "open", "thermal": therm.strip()}


def _settle(session: Any, remaining: Callable[[], None]) -> None:
    settle = getattr(session, "settle_between_ticks", None)
    if not callable(settle):
        raise CostSmokeError(
            "cost smoke requires the unpaced adapter settle_between_ticks boundary"
        )
    settle(remaining)


def _run_world(
    row: Mapping[str, Any],
    opener: Callable[[Mapping[str, Any], int, str], Any],
    *,
    deadline: float,
    max_frames: int,
    now: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run exactly one cost-only unpaced world, retaining only operational cost."""
    started_wall, started_cpu = now(), time.process_time()
    session = None
    frames = 0
    timed_out = False
    completed = False
    next_progress = started_wall + 120.0

    def remaining() -> None:
        if now() >= deadline:
            raise TimeoutError("cost smoke elapsed bound reached")

    try:
        remaining()
        session = opener(row, 0, "unpaced")
        for _ in range(max_frames):
            _settle(session, remaining)
            session.step()
            frames += 1
            # Match the terminal adapter path: a dead terminal hero has no
            # further executable work. This control-flow observation is never
            # emitted as an outcome or metric.
            if not session.hero_alive:
                completed = True
                break
            if now() >= next_progress:
                progress(f"cost-only {row['mix']} frames={frames}/{HORIZON}")
                next_progress += 120.0
            remaining()
        else:
            completed = max_frames == HORIZON
    except TimeoutError:
        timed_out = True
    finally:
        if session is not None:
            session.close(wait=True)
    elapsed = now() - started_wall
    return {
        "operational_identity": {
            "kind": str(row["kind"]),
            "mix": str(row["mix"]),
            "seed": int(row["seed"]),
            "arm": "unpaced",
            "horizon": HORIZON,
            "max_frames": max_frames,
        },
        "frames_completed": frames,
        "world_completed_before_global_bound": completed,
        "elapsed_seconds_including_setup_and_close": elapsed,
        "process_cpu_seconds_including_setup_and_close": time.process_time() - started_cpu,
        "elapsed_bound_reached": timed_out,
    }


def projection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Forecast using terminal-only cost, explicitly retaining the Watch gap."""
    if not rows:
        raise CostSmokeError("no cost sample available for projection")
    estimated_costs = []
    censored = 0
    for row in rows:
        elapsed = float(row["elapsed_seconds_including_setup_and_close"])
        if row["world_completed_before_global_bound"]:
            estimated_costs.append(elapsed)
            continue
        frames = int(row["frames_completed"])
        if frames < 1:
            continue
        censored += 1
        estimated_costs.append(elapsed * HORIZON / frames)
    if not estimated_costs:
        return {
            "terminal_sample_worlds": len(rows),
            "censored_worlds_extrapolated_to_horizon": 0,
            "terminal_cost_seconds_per_world": None,
            "unpaced_proxy_hours_for_144_terminal_plus_32_watch": None,
            "paced_hours_fixed": PACED_HOURS,
            "timing_hours_fixed": TIMING_HOURS,
            "projected_total_hours": None,
            "cap_hours": 40.0,
            "over_cap": None,
            "forecast_kind": "unavailable_no_executable_frames",
            "limitation": (
                "The global bound fired before any frame executed; no cost projection is valid."
            ),
        }
    per_world = sum(estimated_costs) / len(estimated_costs)
    proxy_unpaced_hours = per_world * (TERMINAL_WORLDS + WATCH_SESSIONS) / 3600.0
    total_hours = proxy_unpaced_hours + PACED_HOURS + TIMING_HOURS
    return {
        "terminal_sample_worlds": len(rows),
        "censored_worlds_extrapolated_to_horizon": censored,
        "terminal_cost_seconds_per_world": per_world,
        "unpaced_proxy_hours_for_144_terminal_plus_32_watch": proxy_unpaced_hours,
        "paced_hours_fixed": PACED_HOURS,
        "timing_hours_fixed": TIMING_HOURS,
        "projected_total_hours": total_hours,
        "cap_hours": 40.0,
        "over_cap": total_hours > 40.0,
        "forecast_kind": "terminal_proxy_with_conservative_full_horizon_pacing",
        "limitation": (
            "Only terminal worlds were sampled. "
            "Watch uses a different live adapter. "
            "Applying terminal cost to its 32 sessions is an estimate. "
            "The fixed pacing allocation conservatively assumes full horizons. "
            "A global-bound-censored terminal row is extrapolated by its cost per executed frame."
        ),
    }


def run(
    *,
    registration: Mapping[str, Any],
    out: Path,
    max_elapsed_seconds: float = DEFAULT_MAX_ELAPSED_SECONDS,
    max_frames_per_world: int = 64,
    opener: Callable[[Mapping[str, Any], int, str], Any] | None = None,
    preflight: Callable[[], Mapping[str, str]] = _preflight,
    now: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Create one immutable cost-only receipt; no efficacy outcome is observed or written."""
    if max_elapsed_seconds <= 0 or max_elapsed_seconds > DEFAULT_MAX_ELAPSED_SECONDS:
        raise CostSmokeError("max elapsed must be in (0, 1800] seconds")
    if not 1 <= max_frames_per_world <= HORIZON:
        raise CostSmokeError("max frames per world must be in [1, 5000]")
    if out.exists():
        allowed = {"run-metadata.json", "run-source-binding.json"}
        names = {path.name for path in out.iterdir()}
        if names - allowed:
            raise CostSmokeError(f"refusing existing cost-smoke output: {out}")
    if opener is None:
        from research.t34_pathb.adapter_v2 import open_terminal

        opener = open_terminal
    guard = dict(preflight())
    started = now()
    deadline = started + max_elapsed_seconds
    rows = materialize_rows(development_rows(registration))
    records = []
    for index, row in enumerate(rows):
        if now() >= deadline:
            break
        remaining = len(rows) - index
        row_deadline = min(deadline, now() + max(0.0, deadline - now()) / remaining)
        record = _run_world(
            row,
            opener,
            deadline=row_deadline,
            max_frames=max_frames_per_world,
            now=now,
            progress=progress,
        )
        records.append(record)
        progress(
            f"cost-only completed mix={row['mix']} frames={record['frames_completed']}/{HORIZON}"
        )
    receipt = {
        "schema": SCHEMA,
        "label": "CONTENDED cost-only excluded from all efficacy",
        "development_only": True,
        "efficacy_rows_produced": False,
        "no_outcome_metrics_inspected_or_stored": True,
        "threads": {"total": 2, "main": 1, "controller": 1, "native_pools": 1},
        "preflight": guard,
        "max_elapsed_seconds": max_elapsed_seconds,
        "max_frames_per_world": max_frames_per_world,
        "smoke_elapsed_seconds": now() - started,
        "world_costs": records,
        "projection": projection(records),
    }
    _write_new(out / "cost-smoke.json", receipt)
    return receipt


def write_interrupted_attempt(out: Path, details: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve a superseded cost-only attempt without recording gameplay outcomes."""
    artifact = {
        "schema": "t34-pathb-v2-cost-smoke-interrupted/v1",
        "label": "CONTENDED cost-only interrupted and superseded before receipt",
        "development_only": True,
        "efficacy_rows_produced": False,
        "no_outcome_metrics_inspected_or_stored": True,
        **dict(details),
    }
    _write_new(out / "interrupted.json", artifact)
    return artifact


def write_corrected_projection(receipt_path: Path) -> dict[str, Any]:
    """Write a create-only interpretation for a receipt made before wording correction."""
    raw = json.loads(receipt_path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("world_costs"), list):
        raise CostSmokeError("cost receipt is malformed")
    artifact = {
        "schema": "t34-pathb-v2-cost-projection-correction/v1",
        "source_receipt": {"path": str(receipt_path), "sha256": _sha256(receipt_path)},
        "projection": projection(raw["world_costs"]),
        "interpretation": (
            "This is a terminal-cost proxy plus the conservative full-horizon pacing allocation. "
            "Watch was not sampled. A proxy above 40 hours is an estimate, not infeasibility proof."
        ),
    }
    _write_new(receipt_path.parent / "cost-projection.json", artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registration",
        type=Path,
        default=Path("research/t34_pathb/REGISTRATION_RETENTION_V2.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-elapsed-seconds", type=float, default=DEFAULT_MAX_ELAPSED_SECONDS)
    parser.add_argument("--max-frames-per-world", type=int, default=64)
    parser.add_argument(
        "--write-corrected-projection",
        action="store_true",
        help="write a create-only corrected projection next to an existing cost-smoke receipt",
    )
    args = parser.parse_args(argv)
    if args.write_corrected_projection:
        result = write_corrected_projection(args.out / "cost-smoke.json")
        print(
            json.dumps({"out": str(args.out), "projection": result["projection"]}, sort_keys=True)
        )
        return 0
    registration = json.loads(args.registration.read_text())
    if not isinstance(registration, dict):
        raise CostSmokeError("registration must be a JSON object")
    result = run(
        registration=registration,
        out=args.out,
        max_elapsed_seconds=args.max_elapsed_seconds,
        max_frames_per_world=args.max_frames_per_world,
    )
    print(json.dumps({"out": str(args.out), "projection": result["projection"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CostSmokeError as exc:
        print(f"cost smoke refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
