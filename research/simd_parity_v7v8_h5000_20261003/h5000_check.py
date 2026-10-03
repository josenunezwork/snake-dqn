#!/usr/bin/env python3
"""DEVELOPMENT-ONLY H5000 parity check: SIMD vector61 + v7/v8 veto vs saved LIVE records.

Not a governance screen and not evidence for any gate. It mirrors the v5 check
(``snake-dqn-artifacts/simd-v5-parity-20261002/run-v1/h5000_check.py``, 12/12 identical)
for the v7 space-preference and v8 space-and-head vetoes: it replays, on the SIMD engine
(``run_simd_eval(vector61=True, hero_safety_veto="v7"|"v8", hero_safety_veto_lambda=...)``,
rowwise bit-exact forwards), the first ``--worlds-per-mix`` worlds (by ``world_index``) of
each mix of a saved LIVE arm, and compares every SIMD episode with the live entry of the
same arm and world. Live inputs are read, never written.

Per world, PASS needs all of:

* ``record``: the whole SIMD record minus :data:`SIMD_ONLY_RECORD_KEYS` equals the live
  entry's ``record`` (canonical JSON, type-strict; every differing path is listed);
* ``veto_diagnostics``: the SIMD record's ``veto_diagnostics`` equals the live entry's
  ``veto_diagnostics`` after removing wall-clock fields (any key containing ``seconds``,
  at any depth: :func:`strip_timing`);
* the probe ``method`` equals the live entry's ``safety_veto_method`` and the SIMD
  provenance equals ``vector61_provenance("rowwise", variant, lambda, reference_lambda)``.

Exit 0 iff every planned world was compared and passed; summary.json carries the tally.

Targets (``--targets``, default ``v7-screen-B v8-screen-B``): see :data:`TARGETS`.

Usage (after the strict gate closes; AC power; takes one free CPU slot, pool 3 by default):
  OMP_NUM_THREADS=2 ./venv/bin/python research/simd_parity_v7v8_h5000_20261003/h5000_check.py
  ... --dry-run   # validate the live inputs and print the plan (no torch, no episodes, no locks)
"""

from __future__ import annotations

import os

# Thread caps must be set before torch/numpy import anywhere in the process (as the screens).
os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA = "simd-v7v8-h5000-parity-check/v1"
AUTHORITY = "development-only-parity-check (not a screen, not gate evidence)"
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
DEFAULT_OUT = ARTIFACTS / "simd-v7v8-parity-20261003" / "run-v1"
DEFAULT_WORLDS_PER_MIX = 4
VECTOR61_FORWARD = "rowwise"  # bit-exact batch-1 forwards
# Record keys only the SIMD engine emits (the live screens store veto_diagnostics beside
# the record, in the entry; it is compared there, timing-stripped).
SIMD_ONLY_RECORD_KEYS = (
    "world_runtime_spec",
    "world_runtime_spec_digest",
    "vector61_policy",
    "veto_diagnostics",
)


@dataclass(frozen=True)
class Target:
    """One saved live arm to replay: where its entries live and which veto it ran."""

    runs: Tuple[str, ...]  # run dirs (shards) holding intent.json, records/, checkpoints/
    arm: str
    variant: str
    lam: float
    reference_lambda: Optional[float] = None


_V7_SCREEN = str(ARTIFACTS / "apex-veto-v7-screen-20261002" / "run-v1")
_V7_SWEEP = str(ARTIFACTS / "apex-veto-v7-lambda-sweep-20261002" / "run-v1")
_V8_SCREEN = tuple(
    str(ARTIFACTS / "apex-veto-v8-screen-20261002" / "run-v1" / f"shard-{k}") for k in range(3)
)
_V8_SWEEP = tuple(
    str(ARTIFACTS / "apex-veto-v8-lambda-sweep-20261002" / "run-v1" / f"shard-{k}")
    for k in range(3)
)
TARGETS: Dict[str, Target] = {
    # v7 screen arm B: v7 lambda 4 (40 worlds per mix).
    "v7-screen-B": Target((_V7_SCREEN,), "B", "v7", 4.0),
    # v8 screen arm B: v8 lambda 8, reference v7 at lambda 4 (60 worlds/mix over 3 shards).
    "v8-screen-B": Target(_V8_SCREEN, "B", "v8", 8.0, 4.0),
    # Optional extra coverage (other lambdas):
    "v8-screen-A": Target(_V8_SCREEN, "A", "v7", 4.0),
    "v7-sweep-L100": Target((_V7_SWEEP,), "L100", "v7", 1.0),
    "v7-sweep-L200": Target((_V7_SWEEP,), "L200", "v7", 2.0),
    "v8-sweep-H400": Target(_V8_SWEEP, "H400", "v8", 4.0, 4.0),
    "v8-sweep-H1600": Target(_V8_SWEEP, "H1600", "v8", 16.0, 4.0),
}
DEFAULT_TARGETS = ("v7-screen-B", "v8-screen-B")


# ---------------------------------------------------------------------------
# Pure logic (unit-tested without running any episode)
# ---------------------------------------------------------------------------
def normalize(value: Any) -> Any:
    """JSON round-trip, as records are saved."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def strip_timing(value: Any) -> Any:
    """``value`` without wall-clock fields: every key containing ``seconds``, recursively."""
    if isinstance(value, dict):
        return {k: strip_timing(v) for k, v in value.items() if "seconds" not in str(k)}
    if isinstance(value, list):
        return [strip_timing(v) for v in value]
    return value


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return type(value).__name__


def diff_paths(live: Any, simd: Any, path: str = "") -> List[Tuple[str, Any, Any]]:
    """Every type-strict difference as ``(path, live, simd)``, in sorted key order.

    The rule of ``research/simd_parity_h5000_20261001/aa_check.py::diff_paths`` (not
    imported: that module sets thread environment variables at import).
    """
    out: List[Tuple[str, Any, Any]] = []
    if isinstance(live, dict) and isinstance(simd, dict):
        for key in sorted(set(live) | set(simd)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in live:
                out.append((sub, "<missing>", simd[key]))
            elif key not in simd:
                out.append((sub, live[key], "<missing>"))
            else:
                out.extend(diff_paths(live[key], simd[key], sub))
        return out
    if isinstance(live, list) and isinstance(simd, list):
        if len(live) != len(simd):
            out.append((f"{path}[len]", len(live), len(simd)))
        for index, (a, b) in enumerate(zip(live, simd)):
            out.extend(diff_paths(a, b, f"{path}[{index}]"))
        return out
    if _kind(live) != _kind(simd) or live != simd:
        out.append((path, live, simd))
    return out


def target_spec(target: Target):
    """The :class:`SafetyVetoSpec` a target's arm ran (validated like run_simd_eval)."""
    from src.simd_env.vector61_policy import resolve_safety_veto_spec

    return resolve_safety_veto_spec(target.variant, target.lam, target.reference_lambda)


def check_intent(intent: Mapping[str, Any], target: Target) -> List[str]:
    """Problems with a live run's intent for ``target`` (empty = usable)."""
    from research.apex_safety_20260926 import dev_screen as ds

    problems: List[str] = []
    if intent.get("config", {}).get("sha256") != ds.CONFIG_SHA256:
        problems.append("config sha256 differs from dev_screen.CONFIG_SHA256")
    if intent.get("profile", {}).get("digest") != ds.PROFILE_DIGEST:
        problems.append("profile digest differs from dev_screen.PROFILE_DIGEST")
    if intent.get("profile", {}).get("horizon") != ds.HORIZON:
        problems.append("horizon differs from dev_screen.HORIZON")
    if intent.get("hero", {}).get("sha256") != ds.CHAMPION[1]:
        problems.append("hero is not the champion")
    arm = intent.get("arms", {}).get(target.arm)
    method = target_spec(target).method
    if isinstance(arm, Mapping):
        if float(arm.get("lambda", -1.0)) != target.lam or arm.get("method") != method:
            problems.append(f"arm {target.arm} is {arm}, expected lambda {target.lam} {method}")
    elif not (isinstance(arm, str) and method in arm):
        problems.append(f"arm {target.arm} description does not name {method}")
    return problems


def check_entry(entry: Mapping[str, Any], target: Target) -> List[str]:
    """Problems with one live entry for ``target`` (empty = usable)."""
    spec = target_spec(target)
    problems: List[str] = []
    if entry.get("arm") != target.arm:
        problems.append(f"arm {entry.get('arm')!r} != {target.arm!r}")
    if entry.get("safety_veto_method") != spec.method:
        problems.append(f"safety_veto_method {entry.get('safety_veto_method')!r} != {spec.method}")
    probe = entry.get("record", {}).get("probes", {}).get("safety_veto", {})
    if probe.get("method") != spec.method:
        problems.append(f"probe method {probe.get('method')!r} != {spec.method}")
    diagnostics = entry.get("veto_diagnostics")
    if not isinstance(diagnostics, Mapping):
        problems.append("entry has no veto_diagnostics")
    elif spec.variant == "v8" and diagnostics.get("reference_lambda") != spec.reference_lambda:
        problems.append(
            f"reference_lambda {diagnostics.get('reference_lambda')!r} != {spec.reference_lambda}"
        )
    return problems


def select_entries(
    entries: Sequence[Mapping[str, Any]], target: Target, mix: str, worlds: int
) -> List[Mapping[str, Any]]:
    """The first ``worlds`` entries of ``target.arm`` in ``mix`` by ``world_index``."""
    chosen = [e for e in entries if e.get("arm") == target.arm and e.get("mix") == mix]
    chosen.sort(key=lambda e: int(e["world_index"]))
    indices = [int(e["world_index"]) for e in chosen]
    if len(set(indices)) != len(indices):
        raise ValueError(f"{target.arm}/{mix}: duplicate world_index in the live entries")
    if len(chosen) < worlds:
        raise ValueError(f"{target.arm}/{mix}: only {len(chosen)} live entries, need {worlds}")
    return chosen[:worlds]


def compare_world(
    live_entry: Mapping[str, Any], simd_record: Mapping[str, Any], target: Target
) -> Dict[str, Any]:
    """Compare one SIMD record with the live entry of the same arm and world."""
    from src.simd_env.vector61_policy import vector61_provenance

    simd = normalize(simd_record)
    view = {k: v for k, v in simd.items() if k not in SIMD_ONLY_RECORD_KEYS}
    live_record = normalize(live_entry["record"])
    record_diffs = diff_paths(live_record, view)
    leaked = sorted(set(SIMD_ONLY_RECORD_KEYS) & set(live_record))
    live_diag = strip_timing(normalize(live_entry.get("veto_diagnostics")))
    simd_diag = strip_timing(simd.get("veto_diagnostics", "<missing>"))
    diag_diffs = diff_paths(live_diag, simd_diag)
    expected = normalize(
        vector61_provenance(VECTOR61_FORWARD, target.variant, target.lam, target.reference_lambda)
    )
    provenance_ok = simd.get("vector61_policy") == expected
    method = simd.get("probes", {}).get("safety_veto", {}).get("method")
    method_ok = method == live_entry.get("safety_veto_method")
    record_identical = not record_diffs and canonical(live_record) == canonical(view)
    diagnostics_identical = not diag_diffs
    return {
        "arm": live_entry.get("arm"),
        "mix": live_entry.get("mix"),
        "world_index": live_entry.get("world_index"),
        "world_seed": live_entry.get("world_seed"),
        "identical": record_identical
        and diagnostics_identical
        and provenance_ok
        and method_ok
        and not leaked,
        "record_identical": record_identical,
        "diagnostics_identical": diagnostics_identical,
        "provenance_ok": provenance_ok,
        "method_ok": method_ok,
        "live_has_simd_only_keys": leaked,
        "record_differences": [{"path": p, "live": a, "simd": b} for p, a, b in record_diffs[:25]],
        "diagnostics_differences": [
            {"path": p, "live": a, "simd": b} for p, a, b in diag_diffs[:25]
        ],
        "deaths": simd.get("deaths"),
        "mass_integral": simd.get("mass_integral"),
        "live_wall_seconds": live_entry.get("wall_seconds"),
    }


def tally(results: Sequence[Mapping[str, Any]], planned: int) -> Dict[str, Any]:
    identical = sum(bool(r["identical"]) for r in results)
    return {
        "planned": int(planned),
        "compared": len(results),
        "identical": identical,
        "different": len(results) - identical,
        "pass": len(results) == int(planned) and identical == int(planned),
    }


# ---------------------------------------------------------------------------
# Inputs and run
# ---------------------------------------------------------------------------
def load_target(name: str, worlds: int) -> Dict[str, Any]:
    """Read and validate a target's live intents, entries and checkpoint snapshots."""
    from research.apex_safety_20260926 import dev_screen as ds

    target = TARGETS[name]
    entries: List[Dict[str, Any]] = []
    snapshots: Dict[str, str] = {}
    for run in target.runs:
        run_dir = Path(run)
        intent = json.loads((run_dir / "intent.json").read_text())
        problems = check_intent(intent, target)
        if problems:
            raise SystemExit(f"{name}: {run_dir}/intent.json: {problems}")
        for path in sorted((run_dir / "records").glob(f"{target.arm}-*.json")):
            entries.append(json.loads(path.read_text()))
        for path in sorted((run_dir / "checkpoints").glob("*.pth")):
            snapshots.setdefault(path.stem, str(path))
    plan = {mix: select_entries(entries, target, mix, worlds) for mix in ds.MIXES}
    for mix, chosen in plan.items():
        for entry in chosen:
            problems = check_entry(entry, target)
            if problems:
                raise SystemExit(f"{name}/{mix}/{entry.get('world_seed')}: {problems}")
    # Verified here (also by --dry-run), before any output directory exists.
    needed = [s for c in plan.values() for e in c for s in e["roster_member_sha256s"]]
    verify_snapshots(snapshots, needed + [ds.CHAMPION[1]])
    return {"name": name, "target": target, "plan": plan, "snapshots": snapshots}


def verify_snapshots(snapshots: Mapping[str, str], needed: Sequence[str]) -> None:
    from research.apex_safety_20260926 import dev_screen as ds
    from src.evaluation.strict_promotion import scripted_agent

    scripted = {scripted_agent(n)["sha256"] for n in ("greedy_food", "random_safe")}
    for sha in sorted(set(needed) - scripted):
        if sha not in snapshots:
            raise SystemExit(f"no checkpoint snapshot for roster member {sha}")
        if ds.sha256_file(Path(snapshots[sha])) != sha:
            raise SystemExit(f"checkpoint snapshot {snapshots[sha]} does not hash to {sha}")


def thermal_problems() -> List[str]:
    from research.compute.thermal_guard import (
        parse_pmset_therm,
        read_pmset_therm,
        thermal_reasons,
    )

    reading = read_pmset_therm()
    if not reading.get("gated"):
        return []
    if not reading.get("available"):
        return [f"pmset unavailable: {reading.get('error')}"]
    return thermal_reasons(parse_pmset_therm(reading["text"]))


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


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", nargs="+", choices=sorted(TARGETS), default=DEFAULT_TARGETS)
    parser.add_argument("--worlds-per-mix", type=int, default=DEFAULT_WORLDS_PER_MIX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--slot-pool", type=int, choices=(2, 3), default=3)
    parser.add_argument("--dry-run", action="store_true", help="validate inputs, print the plan")
    args = parser.parse_args(argv)
    if args.worlds_per_mix < 1:
        parser.error("--worlds-per-mix must be >= 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    loaded = [load_target(name, args.worlds_per_mix) for name in args.targets]
    planned = sum(len(c) for item in loaded for c in item["plan"].values())
    for item in loaded:
        for mix, chosen in item["plan"].items():
            print(
                item["name"],
                mix,
                [(int(e["world_index"]), int(e["world_seed"])) for e in chosen],
                flush=True,
            )
    print(f"planned SIMD episodes: {planned}", flush=True)
    if args.dry_run:
        return 0

    from research.apex_safety_20260926 import dev_screen as ds

    out = args.out.resolve()
    if out.exists():
        print(f"refusing: {out} already exists (create-only)", file=sys.stderr)
        return 2
    if not ds.on_ac_power():
        print("refusing: not on AC power", file=sys.stderr)
        return 2
    problems = thermal_problems()
    if problems:
        print(f"refusing: thermal guard not ok: {problems}", file=sys.stderr)
        return 2
    try:
        slots = ds.acquire_cpu_slots(ds.DEFAULT_SLOT_LOCK_ROOT, 1, 0.0, pool=args.slot_pool)
    except (TimeoutError, FileNotFoundError) as exc:
        print(f"refusing: no free CPU slot ({exc})", file=sys.stderr)
        return 2
    try:
        return _run(args, out, loaded, planned, argv)
    finally:
        ds.release_cpu_slots(slots)


def _run(
    args: argparse.Namespace,
    out: Path,
    loaded: Sequence[Mapping[str, Any]],
    planned: int,
    argv: Optional[Sequence[str]],
) -> int:
    import torch

    torch.set_num_threads(2)  # the screens' _configure_torch
    torch.set_num_interop_threads(1)

    from research.apex_safety_20260926 import dev_screen as ds
    from src.core.config_loader import load_and_initialize_config
    from src.scripts.tournament_eval import evaluation_profile_for_name
    from src.simd_env.eval_engine import run_simd_eval

    if ds.sha256_file(ds.DEFAULT_CONFIG) != ds.CONFIG_SHA256:
        raise SystemExit("deployment config bytes differ from dev_screen.CONFIG_SHA256")
    load_and_initialize_config(str(ds.DEFAULT_CONFIG))
    profile = evaluation_profile_for_name(ds.PROFILE_NAME, ds.HORIZON)
    if profile.digest != ds.PROFILE_DIGEST:
        raise SystemExit("resolved profile differs from dev_screen.PROFILE_DIGEST")

    out.mkdir(parents=True)
    (out / "records").mkdir()
    write_new_json(
        out / "intent.json",
        {
            "schema_version": SCHEMA,
            "authority": AUTHORITY,
            "argv": list(sys.argv if argv is None else argv),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "git": _git(),
            "targets": {
                item["name"]: {
                    "runs": list(item["target"].runs),
                    "arm": item["target"].arm,
                    "variant": item["target"].variant,
                    "lambda": item["target"].lam,
                    "reference_lambda": item["target"].reference_lambda,
                    "worlds": {
                        mix: [int(e["world_seed"]) for e in chosen]
                        for mix, chosen in item["plan"].items()
                    },
                }
                for item in loaded
            },
            "planned_episodes": planned,
            "engine": {"engine": "simd", "vector61": True, "vector61_forward": VECTOR61_FORWARD},
            "excluded_record_keys": list(SIMD_ONLY_RECORD_KEYS),
            "diagnostics_rule": "veto_diagnostics minus every key containing 'seconds'",
            "threads": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "torch_intraop": torch.get_num_threads(),
                "torch_interop": torch.get_num_interop_threads(),
            },
        },
    )

    results: List[Dict[str, Any]] = []
    batches: List[Dict[str, Any]] = []
    stopped: Optional[str] = None
    started_total = time.monotonic()
    for item in loaded:
        target: Target = item["target"]
        lookup = ds.agent_lookup(item["snapshots"])  # snapshots verified by load_target
        hero = lookup[ds.CHAMPION[1]]
        for mix, chosen in item["plan"].items():
            problems = thermal_problems()
            if problems:
                stopped = f"thermal guard not ok before {item['name']}/{mix}: {problems}"
                break
            seeds = [int(e["world_seed"]) for e in chosen]
            rosters = {
                int(e["world_seed"]): [lookup[s] for s in e["roster_member_sha256s"]]
                for e in chosen
            }
            started = time.monotonic()
            records = run_simd_eval(
                hero,
                rosters[seeds[0]],
                ds.HORIZON,
                seeds,
                profile=profile,
                opponent_specs_by_world=rosters,
                mix_id=mix,
                vector61=True,
                hero_safety_veto=target.variant,
                vector61_forward=VECTOR61_FORWARD,
                hero_safety_veto_lambda=target.lam,
                hero_safety_veto_reference_lambda=target.reference_lambda,
            )
            wall = time.monotonic() - started
            rows = []
            for entry, record in zip(chosen, records):
                name = f"{item['name']}-{mix}-{entry['world_seed']}.json"
                write_new_json(out / "records" / name, normalize(record))
                row = compare_world(entry, record, target)
                row["target"] = item["name"]
                rows.append(row)
            results.extend(rows)
            live_wall = sum(float(r["live_wall_seconds"]) for r in rows)
            batch = {
                "target": item["name"],
                "mix": mix,
                "episodes": len(rows),
                "identical": sum(r["identical"] for r in rows),
                "simd_wall_seconds": wall,
                "live_wall_seconds_sum": live_wall,
                "speedup_vs_live": live_wall / wall if wall > 0 else None,
            }
            batches.append(batch)
            print(json.dumps(batch), flush=True)
            for row in rows:
                if not row["identical"]:
                    print("  DIFFERENT", json.dumps(row)[:2000], flush=True)
        if stopped:
            break

    summary = {
        "schema_version": SCHEMA,
        "authority": AUTHORITY,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "tally": tally(results, planned),
        "stopped": stopped,
        "timing": {
            "simd_total_wall_seconds": time.monotonic() - started_total,
            "batches": batches,
            "note": "SIMD runs each (target, mix) as one batch of worlds; live wall_seconds "
            "are the saved per-episode times of the live screens (2 torch threads).",
        },
        "results": results,
    }
    write_new_json(out / "summary.json", summary)
    (out / "h5000_check.py").write_text(Path(__file__).read_text())
    t = summary["tally"]
    print(
        f"TOTAL identical {t['identical']} different {t['different']} "
        f"compared {t['compared']}/{t['planned']} pass={t['pass']}",
        flush=True,
    )
    return 0 if t["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
