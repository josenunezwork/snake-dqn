#!/usr/bin/env python3
"""DEVELOPMENT-ONLY H5000 parity check: SIMD vector61 engine vs saved LIVE records.

Not a governance screen and not evidence for any gate. It replays, on the SIMD
engine (``run_simd_eval(vector61=True)``, rowwise bit-exact forward), the first
``WORLDS_PER_MIX`` worlds of each mix of the non-authoritative Apex safety
screen (``research/apex_safety_20260926/dev_screen.py``, run-v1) for:

  A  champion, no veto
  B  champion + v2 free-space veto (``hero_safety_veto=True``)

with the identical strict balanced rosters, profile and world identities, and
compares every SIMD record to the saved LIVE record of the same arm and world.

Equality is exact (canonical JSON, type-strict), on the whole record except
:data:`SIMD_ONLY_RECORD_KEYS`; the gate fields are also reported individually
(:data:`HEADLINE_FIELDS`). The live records are read, never written.

Usage (about 5 to 10 min):
  ./venv/bin/python research/simd_parity_h5000_20261001/aa_check.py
"""

from __future__ import annotations

import os

# Thread caps must be set before torch/numpy import anywhere in the process.
os.environ["SNAKE_DQN_DEVICE"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import argparse  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA = "simd-h5000-parity-check/v1"
AUTHORITY = "development-only-parity-check (not a screen, not gate evidence)"
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
LIVE_RUN = ARTIFACTS / "apex-safety-screen-20260926" / "run-v1"
DEFAULT_OUT = ARTIFACTS / "simd-h5000-parity-20261001" / "run-v1"
SLOT_LOCK = ARTIFACTS / "pqn-followup-20260909" / "cpu-slot-2.lock"
WORLDS_PER_MIX = 8
ARMS = ("A", "B")  # A: no veto, B: v2 veto
VECTOR61_FORWARD = "rowwise"  # bit-exact batch-1 forwards

# Record keys excluded from the comparison, and why. Nothing else is excluded.
SIMD_ONLY_RECORD_KEYS = {
    # SIMD binds a storage runtime spec (body storage capacity) for its arrays;
    # the live engine has no such allocation and never emits these keys.
    "world_runtime_spec": "SIMD-only runtime storage binding; live never emits it",
    "world_runtime_spec_digest": "digest of world_runtime_spec; SIMD-only",
    # Provenance added by vector61=True (engine, forward mode, bit-exactness, veto
    # flag). Checked separately against the expected value, not compared to live.
    "vector61_policy": "SIMD-only provenance; checked against vector61_provenance()",
}
# Live entry wrapper keys (outside ``record``) that are runtime or screen labels.
EXCLUDED_ENTRY_KEYS = {
    "wall_seconds": "runtime measurement, engine-dependent by design",
    "schema_version": "label of the live screen's file format",
    "authority": "label of the live screen",
}
# Wrapper keys compared (rebuilt for SIMD from the same roster row).
COMPARED_ENTRY_KEYS = (
    "arm",
    "mix",
    "world_index",
    "world_seed",
    "roster_member_sha256s",
    "hero_sha256",
    "safety_veto",
)
HEADLINE_FIELDS = (
    ("mass_integral",),
    ("survival_fraction",),
    ("deaths",),
    ("probes", "death_cause"),
    ("kills",),
    ("denominators",),
    ("probes", "safety_veto", "counters"),
    ("world_identity",),
)


# ---------------------------------------------------------------------------
# Comparison logic (pure; unit-tested without running any episode)
# ---------------------------------------------------------------------------
def normalize(value: Any) -> Any:
    """JSON round-trip (tuples -> lists, numpy scalars refused), as records are saved."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return type(value).__name__


def diff_paths(live: Any, simd: Any, path: str = "") -> Iterator[Tuple[str, Any, Any]]:
    """Yield ``(path, live, simd)`` for every type-strict difference, in sorted key order."""
    if isinstance(live, dict) and isinstance(simd, dict):
        for key in sorted(set(live) | set(simd)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in live:
                yield sub, "<missing>", simd[key]
            elif key not in simd:
                yield sub, live[key], "<missing>"
            else:
                yield from diff_paths(live[key], simd[key], sub)
        return
    if isinstance(live, list) and isinstance(simd, list):
        if len(live) != len(simd):
            yield f"{path}[len]", len(live), len(simd)
        for index, (a, b) in enumerate(zip(live, simd)):
            yield from diff_paths(a, b, f"{path}[{index}]")
        return
    if _kind(live) != _kind(simd) or live != simd:
        yield path, live, simd


def _get(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return "<missing>"
        value = value[key]
    return value


def comparison_view(record: Mapping[str, Any]) -> Dict[str, Any]:
    """The record minus :data:`SIMD_ONLY_RECORD_KEYS` (normalized)."""
    return normalize({k: v for k, v in record.items() if k not in SIMD_ONLY_RECORD_KEYS})


def compare_entry(
    live_entry: Mapping[str, Any],
    simd_entry: Mapping[str, Any],
    expected_provenance: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Compare one SIMD entry (wrapper + ``record``) to the saved live entry.

    ``identical`` is True only when every compared wrapper key, the whole
    record minus the SIMD-only keys (canonical JSON and type-strict diff), and
    the SIMD provenance all match.
    """
    live_record = live_entry["record"]
    simd_record = simd_entry["record"]
    problems: List[Dict[str, Any]] = []
    for key in COMPARED_ENTRY_KEYS:
        a, b = normalize(live_entry.get(key)), normalize(simd_entry.get(key))
        if canonical(a) != canonical(b):
            problems.append({"path": f"entry.{key}", "live": a, "simd": b})
    leaked = sorted(set(SIMD_ONLY_RECORD_KEYS) & set(live_record))
    if leaked:
        problems.append({"path": "record(live has SIMD-only keys)", "live": leaked, "simd": None})
    provenance = simd_record.get("vector61_policy", "<missing>")
    if expected_provenance is not None and normalize(provenance) != normalize(expected_provenance):
        problems.append(
            {
                "path": "record.vector61_policy",
                "live": dict(expected_provenance),
                "simd": provenance,
            }
        )
    live_view = comparison_view(live_record)
    simd_view = comparison_view(simd_record)
    record_diffs = [
        {"path": f"record.{p}", "live": a, "simd": b}
        for p, a, b in diff_paths(live_view, simd_view)
    ]
    canonical_equal = canonical(live_view) == canonical(simd_view)
    if canonical_equal != (not record_diffs):  # the two equality notions must agree
        problems.append({"path": "record(canonical vs diff disagree)", "live": None, "simd": None})
    problems.extend(record_diffs)
    headline = {
        ".".join(keys): {
            "live": _get(live_record, keys),
            "simd": _get(simd_record, keys),
            "equal": canonical(normalize(_get(live_record, keys)))
            == canonical(normalize(_get(simd_record, keys))),
        }
        for keys in HEADLINE_FIELDS
    }
    return {
        "arm": live_entry.get("arm"),
        "mix": live_entry.get("mix"),
        "world_seed": live_entry.get("world_seed"),
        "identical": not problems,
        "record_identical_minus_simd_only_keys": canonical_equal,
        "n_differences": len(problems),
        "first_difference": problems[0] if problems else None,
        "differences": problems[:50],
        "headline": headline,
    }


def tally(comparisons: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Counts of identical / differing episodes, per arm and mix."""
    by: Dict[str, Dict[str, int]] = {}
    for item in comparisons:
        cell = by.setdefault(f"{item['arm']}-{item['mix']}", {"identical": 0, "differing": 0})
        cell["identical" if item["identical"] else "differing"] += 1
    differing = [c for c in comparisons if not c["identical"]]
    return {
        "episodes": len(comparisons),
        "identical": len(comparisons) - len(differing),
        "differing": len(differing),
        "by_arm_mix": dict(sorted(by.items())),
        "first_differing": (
            None
            if not differing
            else {
                "arm": differing[0]["arm"],
                "mix": differing[0]["mix"],
                "world_seed": differing[0]["world_seed"],
                "first_difference": differing[0]["first_difference"],
            }
        ),
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def on_ac_power() -> bool:
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=True)
    return "AC Power" in out.stdout


def _git() -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty_paths": run("status", "--porcelain")}


def write_new_json(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--live-run", type=Path, default=LIVE_RUN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        print(f"refusing: {out} already exists (create-only)", file=sys.stderr)
        return 2
    if not on_ac_power():
        print("refusing: `pmset -g batt` does not show 'AC Power'", file=sys.stderr)
        return 2
    if not SLOT_LOCK.is_file():
        print(f"refusing: slot lock {SLOT_LOCK} missing (never created here)", file=sys.stderr)
        return 2
    lock = SLOT_LOCK.open("r")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print(f"refusing: CPU slot 2 busy ({SLOT_LOCK})", file=sys.stderr)
        return 2
    try:
        return _run(args, out, argv)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _run(args: argparse.Namespace, out: Path, argv: Sequence[str] | None) -> int:
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    from research.apex_safety_20260926 import dev_screen as ds
    from src.core.config_loader import load_and_initialize_config
    from src.evaluation.strict_promotion import _expected_world_identity
    from src.scripts.tournament_eval import evaluation_profile_for_name
    from src.simd_env.eval_engine import run_simd_eval
    from src.simd_env.vector61_policy import vector61_provenance

    live_run = args.live_run.resolve()
    intent = json.loads((live_run / "intent.json").read_text())
    # The live run's pins must equal the harness pins being reused.
    pool = [(row["name"], row["sha256"]) for row in intent["checkpoint_pool"]]
    if pool != [tuple(p) for p in ds.POOL] or intent["hero"]["sha256"] != ds.CHAMPION[1]:
        raise SystemExit("live intent checkpoint pool differs from dev_screen.POOL")
    for sha in dict(ds.POOL).values():
        recorded = live_run / "checkpoints" / f"{sha}.pth"
        if ds.sha256_file(recorded) != sha:
            raise SystemExit(f"live run checkpoint snapshot {recorded} does not hash to {sha}")
    if ds.sha256_file(ds.DEFAULT_CONFIG) != ds.CONFIG_SHA256:
        raise SystemExit("deployment config bytes differ from dev_screen.CONFIG_SHA256")
    if intent["config"]["sha256"] != ds.CONFIG_SHA256:
        raise SystemExit("live run config pin differs from dev_screen.CONFIG_SHA256")
    seeds = ds.screen_seeds(len(intent["worlds"]["seeds"]), ds.SCREEN_DOMAIN, ds.SCREEN_NAMESPACE)
    if seeds != intent["worlds"]["seeds"]:
        raise SystemExit("recomputed screen seeds differ from the live intent")

    load_and_initialize_config(str(ds.DEFAULT_CONFIG))
    profile = evaluation_profile_for_name(ds.PROFILE_NAME, ds.HORIZON)
    if profile.digest != ds.PROFILE_DIGEST or intent["profile"]["digest"] != ds.PROFILE_DIGEST:
        raise SystemExit("resolved profile differs from the live run's profile")

    rows_by_mix = {
        mix: [row for row in ds._design_rows(seeds) if row["mix"] == mix][:WORLDS_PER_MIX]
        for mix in ds.MIXES
    }
    world_index = {seed: index for index, seed in enumerate(seeds)}
    live_entries: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for mix, rows in rows_by_mix.items():
        for row in rows:
            for arm in ARMS:
                path = live_run / "records" / f"{arm}-{mix}-{row['world_seed']}.json"
                entry = json.loads(path.read_text())
                if entry["roster_member_sha256s"] != [s["member_sha256"] for s in row["slots"]]:
                    raise SystemExit(f"{path}: roster differs from the rebuilt roster")
                live_entries[(arm, mix, row["world_seed"])] = entry

    out.mkdir(parents=True)
    records_dir = out / "records"
    records_dir.mkdir()
    snapshots = ds.snapshot_checkpoints(ds.DEFAULT_CHECKPOINT_DIR, out)  # verifies pinned shas
    lookup = ds.agent_lookup(snapshots)
    hero = lookup[ds.CHAMPION[1]]
    write_new_json(
        out / "intent.json",
        {
            "schema_version": SCHEMA,
            "authority": AUTHORITY,
            "argv": list(sys.argv if argv is None else argv),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "git": _git(),
            "live_reference": {"run": str(live_run), "git": intent["git"]},
            "config": {"path": str(ds.DEFAULT_CONFIG), "sha256": ds.CONFIG_SHA256},
            "profile": {
                "name": ds.PROFILE_NAME,
                "digest": ds.PROFILE_DIGEST,
                "horizon": ds.HORIZON,
            },
            "checkpoint_snapshots": snapshots,
            "worlds": {mix: [r["world_seed"] for r in rows] for mix, rows in rows_by_mix.items()},
            "arms": {"A": "champion, no veto", "B": "champion + v2 free-space veto"},
            "engine": {"engine": "simd", "vector61": True, "vector61_forward": VECTOR61_FORWARD},
            "excluded_record_keys": SIMD_ONLY_RECORD_KEYS,
            "excluded_entry_keys": EXCLUDED_ENTRY_KEYS,
            "threads": {
                "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
                "torch_intraop": torch.get_num_threads(),
                "torch_interop": torch.get_num_interop_threads(),
            },
            "slot_lock": str(SLOT_LOCK),
            "ac_power_checked_at_start": True,
        },
    )

    comparisons: List[Dict[str, Any]] = []
    batches: List[Dict[str, Any]] = []
    started_total = time.monotonic()
    for mix, rows in rows_by_mix.items():
        mix_seeds = [row["world_seed"] for row in rows]
        rosters = {
            row["world_seed"]: [lookup[s["member_sha256"]] for s in row["slots"]] for row in rows
        }
        identities = {row["world_seed"]: _expected_world_identity(row) for row in rows}
        for arm in ARMS:
            veto = arm == "B"
            started = time.monotonic()
            records = run_simd_eval(
                hero,
                rosters[mix_seeds[0]],
                ds.HORIZON,
                mix_seeds,
                profile=profile,
                opponent_specs_by_world=rosters,
                world_identities=identities,
                mix_id=mix,
                vector61=True,
                hero_safety_veto=veto,
                vector61_forward=VECTOR61_FORWARD,
            )
            wall = time.monotonic() - started
            live_wall = [live_entries[(arm, mix, s)]["wall_seconds"] for s in mix_seeds]
            batch = {
                "arm": arm,
                "mix": mix,
                "episodes": len(mix_seeds),
                "simd_wall_seconds": wall,
                "simd_wall_seconds_per_episode_amortized": wall / len(mix_seeds),
                "live_wall_seconds_sum": sum(live_wall),
                "speedup_vs_live": sum(live_wall) / wall,
            }
            batches.append(batch)
            expected = vector61_provenance(VECTOR61_FORWARD, veto)
            for row, record in zip(rows, records):
                seed = row["world_seed"]
                entry = {
                    "schema_version": SCHEMA,
                    "authority": AUTHORITY,
                    "arm": arm,
                    "mix": mix,
                    "world_index": world_index[seed],
                    "world_seed": seed,
                    "roster_member_sha256s": [s["member_sha256"] for s in row["slots"]],
                    "hero_sha256": ds.CHAMPION[1],
                    "safety_veto": veto,
                    "wall_seconds_amortized": wall / len(mix_seeds),
                    "record": normalize(record),
                }
                write_new_json(records_dir / f"{arm}-{mix}-{seed}.json", entry)
                result = compare_entry(live_entries[(arm, mix, seed)], entry, expected)
                result["live_wall_seconds"] = live_entries[(arm, mix, seed)]["wall_seconds"]
                result["simd_wall_seconds_amortized"] = wall / len(mix_seeds)
                comparisons.append(result)
            done = [c for c in comparisons if c["arm"] == arm and c["mix"] == mix]
            print(
                json.dumps(
                    {
                        **{k: round(v, 3) if isinstance(v, float) else v for k, v in batch.items()},
                        "identical": sum(c["identical"] for c in done),
                    }
                ),
                flush=True,
            )
    total = time.monotonic() - started_total

    for sha, path in snapshots.items():
        if ds.sha256_file(Path(path)) != sha:
            raise RuntimeError(f"checkpoint snapshot {path} changed during the run")
    live_total = sum(b["live_wall_seconds_sum"] for b in batches)
    summary = {
        "schema_version": SCHEMA,
        "authority": AUTHORITY,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "tally": tally(comparisons),
        "timing": {
            "simd_total_wall_seconds": total,
            "simd_episode_wall_seconds_sum": sum(b["simd_wall_seconds"] for b in batches),
            "live_wall_seconds_sum_same_episodes": live_total,
            "speedup_vs_live_total": live_total / total,
            "batches": batches,
            "note": "SIMD runs each (mix, arm) as one batch of 8 envs; per-episode SIMD "
            "wall is the batch wall / 8 (amortized). Live wall_seconds are per episode, "
            "recorded by the live screen on 2026-09-27 (2 torch threads).",
        },
        "comparisons": comparisons,
    }
    write_new_json(out / "summary.json", summary)
    print(
        json.dumps(
            {
                "tally": summary["tally"],
                "timing": {k: v for k, v in summary["timing"].items() if k != "batches"},
            },
            indent=2,
        )
    )
    return 0 if summary["tally"]["differing"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
