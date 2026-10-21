"""Build the immutable, collision-audited shared E6/E7 root bank.

The collector deliberately makes two deterministic passes over a source world:
the bounded first pass finds eligible terminal and ordinary worlds without
retaining simulator state, and the second pass captures only selected roots.
That keeps full GridBatchSim snapshots out of the source scan while preserving
the actual 128 decision history required by both experiments.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from research.bl_ppo.exports import export_json, verify_export
from research.bl_prefix import seeds
from research.bl_prefix.controllers import B8Controller
from research.bl_prefix.engine import (
    HORIZON,
    PREFIX_HISTORY,
    PrefixRecord,
    capture_root,
    initial_world,
    load_root,
    save_root,
    step,
    verify_root_parity,
)

MIXES = ("scripted", "frozen", "mixed")
LATE_QUOTAS = {"scripted": 44, "frozen": 42, "mixed": 42}
PREFLIGHT_POOL = 48
PREFLIGHT_SURVIVAL_FRAME = 528
DEFAULT_B8 = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth")


class BankIncomplete(RuntimeError):
    """Raised when the frozen bank cannot be built exactly."""


@dataclass(frozen=True)
class Source:
    seed: int
    mix: str
    plan: dict[str, Any]
    death_frame: int | None
    death_length: int | None
    survived: bool
    lengths: np.ndarray


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _b8_path(plan: Mapping[str, Any]) -> Path:
    configured = plan.get("b8_checkpoint")
    checkpoint_dir = os.environ.get("SNAKE_CKPT_DIR")
    value = Path(
        str(
            configured or (Path(checkpoint_dir) / DEFAULT_B8.name if checkpoint_dir else DEFAULT_B8)
        )
    )
    if not value.is_file():
        raise BankIncomplete(f"B8 checkpoint is unavailable: {value}")
    expected = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
    if _sha(value) != expected:
        raise BankIncomplete("B8 checkpoint hash does not match IDENTITY_ADDENDUM")
    return value


def _audit(plan: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(plan["seed_audit"]))
    verify_export(path)
    value = json.loads(path.read_text())
    allocation = value.get("allocation")
    if not isinstance(allocation, dict):
        raise BankIncomplete("seed audit has no allocation")
    try:
        seeds.verify_receipt(
            value,
            round_index=int(value.get("round_index", 0)),
            preflight_only=set(allocation) == {"preflight"},
        )
    except (TypeError, ValueError, seeds.SeedAuditFailed) as exc:
        raise BankIncomplete(f"seed audit cannot be reconstructed: {exc}") from exc
    required = {"preflight"} if plan.get("preflight") else set(seeds.FROZEN_COUNTS)
    if not required.issubset(allocation) or (
        not plan.get("preflight") and set(allocation) != required
    ):
        raise BankIncomplete("seed audit allocation does not cover the requested bank")
    return value


def _plans(source_seeds: Iterable[int]) -> list[dict[str, Any]]:
    """Materialize all mixed rosters once, retaining roster-pool order."""
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015 import bank_engine

    values = list(source_seeds)
    mixes = [MIXES[index % len(MIXES)] for index in range(len(values))]
    # The legacy strict artifact materializer validates uint32 seed labels,
    # although its member selection depends only on allocation position.
    # Ordinals construct the unchanged roster rotation; actual RNG identities
    # below remain the audited high-range values and are never narrowed.
    rows = bank_engine.planned_rosters(gid._context(1), list(range(len(values))), mixes)
    for row, seed in zip(rows, values):
        row["seed"] = int(seed)
    return rows


def _run_source(plan: Mapping[str, Any], b8_path: Path, *, stop_frame: int = HORIZON) -> Source:
    """Run a source through its required horizon, retaining length-at-frame facts."""
    if type(stop_frame) is not int or not PREFIX_HISTORY <= stop_frame <= HORIZON:
        raise ValueError("source stop frame must preserve prefix history and stay within H5000")
    world = initial_world(arm="s3", plan=plan, b8_path=b8_path)
    b8 = B8Controller(b8_path)
    death_frame = death_length = None
    lengths = np.zeros(HORIZON + 1, dtype=np.int16)
    lengths[0] = int(world.sim.length[0, 0])
    while world.sim.frame[0] < stop_frame and world.sim.alive[0, 0]:
        result = step(world, b8)
        lengths[result.frame] = result.length
        if result.died:
            death_frame, death_length = result.frame, result.pre_length
    return Source(
        int(plan["seed"]),
        str(plan["mix"]),
        dict(plan),
        death_frame,
        death_length,
        bool(world.sim.alive[0, 0]),
        lengths,
    )


def _source_task(task: tuple[dict[str, Any], str]) -> Source:
    """Picklable source-screen task; source ordering is restored by the parent."""
    plan, checkpoint = task
    return _run_source(plan, Path(checkpoint))


def _source_tasks(
    candidates: Iterable[dict[str, Any]], checkpoint: Path
) -> list[tuple[dict[str, Any], str]]:
    """Freeze the submitted source ordering before any workers are spawned."""
    return [(dict(candidate), str(checkpoint)) for candidate in candidates]


def _workers(plan: Mapping[str, Any], *, preflight: bool) -> int:
    value = plan.get("workers", 1 if preflight else 12)
    if type(value) is not int or not 1 <= value <= (2 if preflight else 12):
        raise BankIncomplete(
            "bank workers must be 1..2 for preflight or 1..12 for scientific collection"
        )
    if not preflight and value != 12:
        raise BankIncomplete("scientific source/capture collection requires exactly 12 workers")
    return value


def _deadline(plan: Mapping[str, Any], *, preflight: bool) -> float:
    value = plan.get("collection_deadline_seconds", 1800 if preflight else 5760)
    if type(value) not in {int, float} or not 1 <= float(value) <= (1800 if preflight else 5760):
        raise BankIncomplete("bank collection deadline is outside the frozen wall cap")
    return float(value)


def _ordered_parallel(
    function: Any,
    tasks: list[Any],
    *,
    workers: int,
    deadline: float,
    phase: str,
) -> list[Any]:
    """Run spawn-safe tasks concurrently while merging only in submitted order."""
    began, announced = time.monotonic(), time.monotonic()
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    ) as executor:
        futures = [executor.submit(function, task) for task in tasks]
        rows = []
        for index, future in enumerate(futures, start=1):
            remaining = deadline - (time.monotonic() - began)
            if remaining <= 0:
                raise BankIncomplete(
                    f"{phase} exceeded its collection deadline before task {index}"
                )
            try:
                rows.append(future.result(timeout=remaining))
            except concurrent.futures.TimeoutError as exc:
                raise BankIncomplete(
                    f"{phase} exceeded its collection deadline at task {index}"
                ) from exc
            if time.monotonic() - announced >= 120:
                print(
                    json.dumps({"phase": phase, "completed": index, "total": len(tasks)}),
                    file=sys.stderr,
                    flush=True,
                )
                announced = time.monotonic()
    return rows


def _capture(
    source: Source, b8_path: Path, targets: Mapping[str, int], *, stop_at_last: bool = False
) -> dict[str, Any]:
    """Replay a selected source and capture named unprepared decision boundaries."""
    world = initial_world(arm="s3", plan=source.plan, b8_path=b8_path)
    b8 = B8Controller(b8_path)
    history: collections.deque[PrefixRecord] = collections.deque(maxlen=PREFIX_HISTORY)
    captured: dict[str, Any] = {}
    wanted = {int(frame): name for name, frame in targets.items()}
    limit = max(targets.values()) + 1 if stop_at_last else HORIZON
    while world.sim.frame[0] < limit and world.sim.alive[0, 0]:
        frame = int(world.sim.frame[0])
        if frame in wanted:
            if len(history) != PREFIX_HISTORY:
                raise BankIncomplete(f"source {source.seed} target lacks 128 actual decisions")
            captured[wanted[frame]] = capture_root(world, tuple(history))
            if stop_at_last and len(captured) == len(targets):
                break
        result = step(world, b8)
        history.append(
            PrefixRecord(result.observation, result.action, result.legal, result.advisory)
        )
    if set(captured) != set(targets):
        raise BankIncomplete(
            f"source replay missed required roots: {sorted(set(targets) - set(captured))}"
        )
    if (
        not stop_at_last
        and source.death_frame is not None
        and int(world.sim.frame[0]) != source.death_frame
    ):
        raise BankIncomplete("source replay did not reproduce its recorded terminal frame")
    return captured


def _select(sources: list[Source]) -> tuple[list[Source], list[tuple[Source, Source]]]:
    late: list[Source] = []
    for mix in MIXES:
        rows = [
            s
            for s in sources
            if s.mix == mix
            and s.death_frame is not None
            and s.death_frame >= 2000
            and (s.death_length or 0) >= 700
        ]
        if len(rows) < LATE_QUOTAS[mix]:
            raise BankIncomplete(f"source exhaustion: {mix} has {len(rows)} eligible late deaths")
        late.extend(rows[: LATE_QUOTAS[mix]])
    used: set[int] = set()
    pairs: list[tuple[Source, Source]] = []
    allocation_order = {source.seed: index for index, source in enumerate(sources)}
    for death in late:
        candidates = [
            s for s in sources if s.survived and s.mix == death.mix and s.seed not in used
        ]
        root_frame = int(death.death_frame - 100)
        death_root_length = int(death.lengths[root_frame])
        matches: list[tuple[int, int, Source]] = []
        for ordinary in candidates:
            delta = abs(int(ordinary.lengths[root_frame]) - death_root_length)
            if delta <= 100:
                matches.append((delta, allocation_order[ordinary.seed], ordinary))
        if not matches:
            raise BankIncomplete(f"no ordinary same-mix length match for source {death.seed}")
        _, _, chosen = min(matches, key=lambda row: (row[0], row[1]))
        used.add(chosen.seed)
        pairs.append((death, chosen))
    return late, pairs


def _write_root(root: Any, path: Path) -> str:
    save_root(root, path)
    return str(path.name)


def _capture_pair_task(task: tuple[Source, Source, str, str]) -> dict[str, Any]:
    """Capture one late/ordinary pair into collision-free, pair-specific paths."""
    death, ordinary, checkpoint, output = task
    b8_path, out = Path(checkpoint), Path(output)
    frames = {
        "200": int(death.death_frame - 200),
        "150": int(death.death_frame - 150),
        "100": int(death.death_frame - 100),
    }
    captured = _capture(death, b8_path, frames)
    source_id = str(death.seed)
    ancestors = {
        ancestor: _write_root(root, out / f"{source_id}-a{ancestor}.npz")
        for ancestor, root in captured.items()
    }
    ordinary_root = _capture(ordinary, b8_path, {"ordinary": frames["100"]})["ordinary"]
    ordinary_path = _write_root(ordinary_root, out / f"{source_id}-ordinary.npz")
    return {
        "source_world": source_id,
        "mix": death.mix,
        "ordinary_world": ordinary.seed,
        "ancestors": ancestors,
        "ordinary_root": ordinary_path,
    }


def _capture_tasks(
    pairs: Iterable[tuple[Source, Source]], checkpoint: Path, out: Path
) -> list[tuple[Source, Source, str, str]]:
    """Freeze pair ordering and output locations before the capture fanout."""
    return [(death, ordinary, str(checkpoint), str(out)) for death, ordinary in pairs]


def _select_preflight_sources(
    source_plans: Iterable[dict[str, Any]], b8_path: Path
) -> tuple[list[Source], list[dict[str, Any]]]:
    """Select the first burned candidate surviving 528 frames in each fixed mix.

    This is fixture admission only.  It has no scientific eligibility label and
    it stops trying a mix as soon as its deterministic first usable source is
    found; any trialed early death remains auditable in the published bank.
    """
    plans = list(source_plans)
    if len(plans) != PREFLIGHT_POOL:
        raise BankIncomplete(
            f"preflight fixture pool must contain exactly {PREFLIGHT_POOL} candidates"
        )
    chosen: dict[str, Source] = {}
    rejected: list[dict[str, Any]] = []
    for source_plan in plans:
        mix = str(source_plan.get("mix"))
        if mix not in MIXES:
            raise BankIncomplete("preflight candidate has an invalid mix")
        if mix in chosen:
            continue
        source = _run_source(source_plan, b8_path, stop_frame=PREFLIGHT_SURVIVAL_FRAME)
        if source.survived:
            chosen[mix] = source
        else:
            rejected.append(
                {
                    "seed": source.seed,
                    "mix": source.mix,
                    "early_death_frame": source.death_frame,
                }
            )
        if len(chosen) == len(MIXES):
            break
    missing = [mix for mix in MIXES if mix not in chosen]
    if missing:
        raise BankIncomplete(f"preflight fixture pool exhausted without a survivor for: {missing}")
    return [chosen[mix] for mix in MIXES], rejected


def _build_preflight(
    plan: Mapping[str, Any], out: Path, audit: Mapping[str, Any], b8_path: Path
) -> dict[str, Any]:
    source_seeds = audit["allocation"]["preflight"]["seeds"][:PREFLIGHT_POOL]
    if len(source_seeds) != PREFLIGHT_POOL:
        raise BankIncomplete("preflight audit has fewer than the frozen 48 fixture candidates")
    selected, rejected = _select_preflight_sources(_plans(source_seeds), b8_path)
    entries, roots, early, parity = [], [], [], []
    for source in selected:
        captures = _capture(
            source, b8_path, {"200": 128, "150": 178, "100": 228}, stop_at_last=True
        )
        source_id = str(source.seed)
        ancestor_paths = {}
        for ancestor, root in captures.items():
            name = f"{source_id}-a{ancestor}.npz"
            ancestor_paths[ancestor] = _write_root(root, out / name)
        roots.append(
            {
                "stratum": "early",
                "mix": source.mix,
                "source_world": source_id,
                "root": ancestor_paths["100"],
            }
        )
        early.append({"source_world": source_id, "mix": source.mix, "ancestors": ancestor_paths})
        parity.append(
            {
                "source_world": source_id,
                "mix": source.mix,
                "root": ancestor_paths["100"],
                "receipt": verify_root_parity(load_root(out / ancestor_paths["100"]), frames=300),
            }
        )
        entries.append({"seed": source.seed, "mix": source.mix, "fixture": True})
    return {
        "sources": entries,
        "roots": roots,
        "early_roots": early,
        "parity": parity,
        "preflight_selection": {
            "candidate_pool_size": PREFLIGHT_POOL,
            "survival_frame": PREFLIGHT_SURVIVAL_FRAME,
            "selected": [{"seed": source.seed, "mix": source.mix} for source in selected],
            "rejected": rejected,
        },
    }


def build(plan: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """Build an immutable shared bank.  It never dispatches or starts Colab."""
    if not isinstance(plan, Mapping) or plan.get("experiment") not in {"bank", "e6e7-bank"}:
        raise BankIncomplete("bank plan must identify the shared E6/E7 bank")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"fresh output directory required: {out}")
    out.mkdir(parents=True, exist_ok=True)
    audit = _audit(plan)
    b8_path = _b8_path(plan)
    preflight = bool(plan.get("preflight"))
    workers = _workers(plan, preflight=preflight)
    deadline = _deadline(plan, preflight=preflight)
    collection_began = time.monotonic()
    if preflight:
        payload = _build_preflight(plan, out, audit, b8_path)
    else:
        source_seeds = audit["allocation"]["source"]["seeds"]
        candidates = _plans(source_seeds)
        sources = _ordered_parallel(
            _source_task,
            _source_tasks(candidates, b8_path),
            workers=workers,
            deadline=deadline,
            phase="source-screen",
        )
        late, pairs = _select(sources)
        remaining = deadline - (time.monotonic() - collection_began)
        if remaining <= 0:
            raise BankIncomplete("source-screen exhausted the collection deadline")
        captured_pairs = _ordered_parallel(
            _capture_pair_task,
            _capture_tasks(pairs, b8_path, out),
            workers=workers,
            deadline=remaining,
            phase="selected-root-capture",
        )
        roots, early = [], []
        for captured in captured_pairs:
            roots.append(
                {
                    "stratum": "early",
                    "mix": captured["mix"],
                    "source_world": captured["source_world"],
                    "root": captured["ancestors"]["100"],
                }
            )
            roots.append(
                {
                    "stratum": "ordinary",
                    "mix": captured["mix"],
                    "source_world": str(captured["ordinary_world"]),
                    "matched_death_world": captured["source_world"],
                    "ordinary_world": captured["ordinary_world"],
                    "root": captured["ordinary_root"],
                }
            )
            early.append(
                {
                    "source_world": captured["source_world"],
                    "mix": captured["mix"],
                    "ancestors": captured["ancestors"],
                }
            )
        payload = {
            "sources": [
                {
                    "seed": item.seed,
                    "mix": item.mix,
                    "death_frame": item.death_frame,
                    "death_length": item.death_length,
                }
                for item in late
            ],
            "roots": roots,
            "early_roots": early,
        }
    value = {
        "schema": "bl-prefix-bank/v1",
        "preflight": preflight,
        "scientific": not preflight,
        "workers": workers,
        "collection_deadline_seconds": deadline,
        "audit_sha256": _sha(Path(str(plan["seed_audit"]))),
        "b8": {"path": str(b8_path), "sha256": _sha(b8_path)},
        "plan_sha256": hashlib.sha256(json.dumps(dict(plan), sort_keys=True).encode()).hexdigest(),
        **payload,
    }
    export_json(out / "bank.json", value)
    return value


def run(plan: Mapping[str, Any], out: Path, *, preflight: bool = False) -> dict[str, Any]:
    """Supervisor-facing collection entry point with an explicit terminal status."""
    import time

    began = time.perf_counter()
    try:
        if bool(plan.get("preflight")) != bool(preflight):
            raise BankIncomplete("bank plan/preflight identity mismatch")
        value = build(plan, out)
        return {
            "status": "PREFLIGHT_ONLY" if preflight else "COMPLETE",
            "bank_path": str(Path(out) / "bank.json"),
            "preflight": preflight,
            "scientific": not preflight,
            "world_steps": None,
            "elapsed_seconds": time.perf_counter() - began,
            "bank": value,
        }
    except (BankIncomplete, KeyError, OSError, ValueError) as exc:
        return {
            "status": "INCOMPLETE",
            "preflight": preflight,
            "scientific": False,
            "reason": str(exc),
            "elapsed_seconds": time.perf_counter() - began,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if bool(args.preflight) != bool(plan.get("preflight")):
        raise SystemExit("--preflight must match frozen plan")
    print(json.dumps(run(plan, args.out, preflight=args.preflight), sort_keys=True))


if __name__ == "__main__":
    main()
