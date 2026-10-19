"""Descriptive search-budget ablation on audited calibration worlds only."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from research.phase0_search_20261019 import bank, teacher
from research.phase0_search_20261019.preflight import B8, fixture
from research.t33_batched_teacher import reference
from research.t33_batched_teacher.evidence import identity

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BANK = ROOT / "research/phase0_search_20261019/WORLD_BANK.json"
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"


def sha(path: Path) -> str:
    """Hash an input's bytes, never follow paths embedded in old receipts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_rows(path: Path = DEFAULT_BANK, *, smoke: bool = False) -> list[dict]:
    """Verify the existing bank/audit binding without accessing other worktrees."""
    value = json.loads(path.read_text())
    audit = json.loads(path.with_name("SEED_AUDIT.json").read_text())
    allocation = {"selection": value["selection"], "fixtures": value["fixtures"]}
    if (
        bank.digest(allocation) != value["allocation_semantic_sha256"]
        or audit.get("pass") is not True
        or audit["world_bank_semantic_sha256"] != value["allocation_semantic_sha256"]
        or audit["world_bank_file"]["sha256"] != sha(path)
    ):
        raise ValueError("invalid calibration bank or seed-audit binding")
    all_rows = value["selection"] + [r for rows in value["fixtures"].values() for r in rows]
    seeds = [int(row["world_seed"]) for row in all_rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("world allocation overlaps selection or another fixture pool")
    return value["fixtures"]["timing" if smoke else "calibration"]


def variants(horizons: list[int], rollouts: list[int], scale: float) -> list[dict]:
    """Return a fixed, named factorial; no winner is selected after looking."""
    return [
        {"id": f"h{h}-n{n}-{leaf}", "horizon": h, "rollouts": n, "leaf": leaf, "leaf_scale": scale}
        for h in horizons
        for n in rollouts
        for leaf in ("max", "value")
    ]


def compare(world: Any, settings: dict, choices: list[dict], gate: float | None) -> dict:
    """Score all variants on an immutable, already-prepared decision."""
    config = teacher.SearchConfig(**settings)
    decision = int(world.sim.frame[0]) - 1
    q, legal, select = teacher._q_and_legal(world)
    actions = list(map(int, np.flatnonzero(legal)))
    started = time.perf_counter()
    trace = reference.prefix_trace(
        world,
        actions,
        config,
        decision,
        common_random=True,
        horizons=[v["horizon"] for v in choices],
    )
    full = reference.trace_scores(trace, config.horizon, config.rollouts, "max", config.leaf_scale)
    legacy = teacher._reference_scores(world, actions, config, decision)
    winner = max(full, key=full.__getitem__)
    old_winner = max(legacy, key=legacy.__getitem__)
    result = []
    margin = teacher._margin(q, legal)
    for variant in choices:
        before = time.perf_counter()
        searched = gate is None or margin < gate
        if searched:
            scores = reference.trace_scores(
                trace,
                variant["horizon"],
                variant["rollouts"],
                variant["leaf"],
                variant["leaf_scale"],
            )
            action = max(scores, key=scores.__getitem__)
        else:
            scores = {}
            action = teacher._mask_argmax(q, select)
        result.append(
            {
                **variant,
                "action": action,
                "searched": searched,
                "scores": scores,
                "agreement": action == winner,
                "regret": full[winner] - full[action],
                "legacy_agreement": action == old_winner,
                "legacy_regret": legacy[old_winner] - legacy[action],
                "seconds": time.perf_counter() - before,
            }
        )
    return {
        "decision": decision,
        "length": int(world.sim.length[0, 0]),
        "margin": margin,
        "reference_action": winner,
        "reference_scores": full,
        "legacy_action": old_winner,
        "legacy_scores": legacy,
        "variants": result,
        "prefix_trace": {
            "branches": trace["branches"],
            "checkpoints": {
                str(h): {k: a.tolist() for k, a in c.items()}
                for h, c in trace["checkpoints"].items()
            },
            "horizon": trace["horizon"],
            "rollouts": trace["rollouts"],
        },
        "prefix_method": "single-full-CRN-run",
        "seconds": time.perf_counter() - started,
    }


def task(payload: tuple) -> list[dict]:
    """One world per single-thread worker; stream samples rather than clone a bank."""
    row, checkpoint, count, settings, choices, gate, smoke, *cache_arg = payload
    cache_root = Path(cache_arg[0]) if cache_arg and cache_arg[0] else None
    teacher.configure_cpu()
    from research.phase0_search_20261019.runner import operational_guard

    operational_guard()
    output = []
    if smoke:
        world = fixture(row, Path(checkpoint), 800)
        selected = {0}
    else:
        world = teacher.build_world(row, checkpoint)
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "row": row,
                    "checkpoint": sha(Path(checkpoint)),
                    "count": count,
                    "horizon": settings["horizon"],
                    "source": identity()["source_sha256"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cache = cache_root / (cache_key + ".json") if cache_root else None
        if cache is not None and cache.exists():
            cached = json.loads(cache.read_text())
            eligible, picked = cached.get("eligible"), cached.get("selected")
            if (
                cached.get("key") != cache_key
                or not isinstance(eligible, list)
                or any(type(v) is not int for v in eligible)
                or eligible != sorted(set(eligible))
                or any(v < 0 or v > 5000 - settings["horizon"] for v in eligible)
                or not isinstance(picked, list)
                or len(set(picked)) != count
                or len(picked) != count
                or not set(picked) <= set(eligible)
            ):
                raise ValueError("invalid identity-bound eligible inventory")
            selected = set(picked)
        else:
            # Uniform reservoir sampling keeps at most count prepared snapshots.
            # This visits the bare trajectory once, instead of enumerating all
            # eligible indices then replaying the entire trajectory a second time.
            eligible = []
            reservoir = []
            rng = np.random.default_rng(np.random.SeedSequence([20261019, row["world_seed"]]))

            class Reservoir:
                def choose(self, prepared_world: Any) -> tuple[int, dict]:
                    q, legal, select = teacher._q_and_legal(prepared_world)
                    decision = int(prepared_world.sim.frame[0]) - 1
                    length = int(prepared_world.sim.length[0, 0])
                    margin = teacher._margin(q, legal)
                    if length >= 600 or margin < 0:
                        eligible.append(decision)
                        index = len(eligible) - 1
                        replace_index = index if index < count else int(rng.integers(index + 1))
                        if replace_index < count:
                            snapshot = (decision, prepared_world.clone())
                            if replace_index == len(reservoir):
                                reservoir.append(snapshot)
                            else:
                                reservoir[replace_index] = snapshot
                    return teacher._mask_argmax(q, select), {"searched": False, "scores": {}}

            for _ in range(5000 - settings["horizon"] + 1):
                if not world.sim.alive[0, 0]:
                    break
                teacher.step_world(world, Reservoir())
            if len(reservoir) != count:
                raise ValueError(f"not enough searched decisions in {row['world_id']}")
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                bank.write_create(
                    cache,
                    {
                        "key": cache_key,
                        "eligible": eligible,
                        "selected": sorted(d for d, _ in reservoir),
                    },
                )
            for _, snapshot in sorted(reservoir, key=lambda item: item[0]):
                result = compare(snapshot, settings, choices, gate)
                output.append({"world_id": row["world_id"], "mix": row["mix"], **result})
                print(
                    json.dumps(
                        {
                            "event": "ablation_decision",
                            "world": row["world_id"],
                            "done": len(output),
                            "total": count,
                        }
                    ),
                    flush=True,
                )
                operational_guard()
            return output

    class Collector:
        def choose(self, prepared_world: Any) -> tuple[int, dict]:
            result = compare(prepared_world, settings, choices, gate)
            output.append({"world_id": row["world_id"], "mix": row["mix"], **result})
            q, legal, select = teacher._q_and_legal(prepared_world)
            return teacher._mask_argmax(q, select), {"searched": False, "scores": {}}

    for decision in range(max(selected) + 1):
        teacher.step_world(world, Collector() if decision in selected else None)
        if decision in selected:
            print(
                json.dumps(
                    {
                        "event": "ablation_decision",
                        "world": row["world_id"],
                        "done": len(output),
                        "total": len(selected),
                    }
                ),
                flush=True,
            )
            operational_guard()
    if len(output) != len(selected):
        raise RuntimeError("incomplete calibration replay")
    return output


def summarize(rows: list[dict]) -> dict:
    """Aggregate all samples in stable order, with no selection threshold."""
    result = {}
    for identifier in (v["id"] for v in rows[0]["variants"]):
        group = [v for row in rows for v in row["variants"] if v["id"] == identifier]
        result[identifier] = {
            "decisions": len(group),
            "searched": sum(v["searched"] for v in group),
            **{
                key: float(np.mean([v[key] for v in group]))
                for key in ("agreement", "regret", "legacy_agreement", "legacy_regret", "seconds")
            },
            "max_regret": max(v["regret"] for v in group),
        }
    return result


def calibrate(rows: list[dict]) -> dict:
    """Fit physical leaf units and report in-sample 0/1/fitted diagnostics.

    No intercept: a dead row has Q=0 and zero remaining mass. A nonnegative
    least-squares slope is the fixed prospective rule; no outcome selects it.
    """
    output = {}
    for variant in rows[0]["variants"]:
        h, n, leaf = variant["horizon"], variant["rollouts"], variant["leaf"]
        xs, ys = [], []
        traces = []
        for row in rows:
            raw = row["prefix_trace"]
            trace = {
                **raw,
                "checkpoints": {
                    int(k): {name: np.asarray(a) for name, a in c.items()}
                    for k, c in raw["checkpoints"].items()
                },
            }
            traces.append(trace)
            keep = np.array([r < n for _, r in trace["branches"]])
            prefix = trace["checkpoints"][h]
            xs.extend(prefix[leaf][keep])
            ys.extend((trace["checkpoints"][trace["horizon"]]["mass"] - prefix["mass"])[keep])
        x, y = np.asarray(xs), np.asarray(ys)
        denominator = float(x @ x)
        slope = max(0.0, float(x @ y) / denominator) if denominator else 0.0
        reports = {}
        for name, scale in (("zero", 0.0), ("one", 1.0), ("fitted", slope)):
            matches, regrets = [], []
            for row, trace in zip(rows, traces):
                scores = reference.trace_scores(trace, h, n, leaf, scale)
                action = max(scores, key=scores.__getitem__)
                full = row["reference_scores"]
                matches.append(action == row["reference_action"])
                regrets.append(full[row["reference_action"]] - full[action])
            reports[name] = {
                "scale": scale,
                "agreement": float(np.mean(matches)),
                "regret": float(np.mean(regrets)),
            }
        output[variant["id"]] = {
            "units": "segment-frames per raw network-Q unit",
            "fit": "nonnegative through-origin least squares; in-sample descriptive only",
            "samples": len(x),
            "nonzero_q_samples": int(np.count_nonzero(x)),
            "scale": slope,
            "rmse": float(np.sqrt(np.mean((y - slope * x) ** 2))),
            "reports": reports,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=B8)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--decisions", type=int, default=300)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--horizons", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--rollouts", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--reference-horizon", type=int, default=128)
    parser.add_argument("--reference-rollouts", type=int, default=8)
    parser.add_argument("--leaf-scale", type=float, default=1.0)
    parser.add_argument("--margin-gate", type=float)
    parser.add_argument(
        "--eligible-cache", type=Path, help="identity-bound reusable eligible decision inventories"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.decisions < 3 or args.decisions % 3 or (args.smoke and args.decisions != 3):
        parser.error("decisions must be a positive multiple of 3 (exactly 3 for smoke)")
    if sha(args.checkpoint) != B8_SHA:
        raise ValueError("checkpoint differs from released B8 identity")
    rows = calibration_rows(args.bank, smoke=args.smoke)
    settings = asdict(
        teacher.SearchConfig(
            horizon=args.reference_horizon, rollouts=args.reference_rollouts, length_gate=0
        )
    )
    choices = variants(args.horizons, args.rollouts, args.leaf_scale)
    for choice in choices:
        teacher.SearchConfig(**{k: v for k, v in choice.items() if k not in ("id", "leaf")})
    payloads = [
        (
            r,
            str(args.checkpoint),
            args.decisions // 3,
            settings,
            choices,
            args.margin_gate,
            args.smoke,
            str(args.eligible_cache) if args.eligible_cache else None,
        )
        for r in rows
    ]
    if any(
        v["horizon"] > args.reference_horizon or v["rollouts"] > args.reference_rollouts
        for v in choices
    ):
        parser.error("variant H/N must be prefixes of reference H/N")
    binding = identity()
    started = time.perf_counter()
    if args.workers == 1:
        batches = list(map(task, payloads))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers, 3), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            batches = list(pool.map(task, payloads))
    results = [r for batch in batches for r in batch]
    receipt = {
        "schema": "t33-ablation/v1",
        "scope": "synthetic-smoke" if args.smoke else "descriptive-calibration-only",
        "argv": sys.argv,
        "platform": platform.platform(),
        "seconds": time.perf_counter() - started,
        "workers": min(args.workers, 3),
        "checkpoint_sha256": sha(args.checkpoint),
        "bank_sha256": sha(args.bank),
        "audit_sha256": sha(args.bank.with_name("SEED_AUDIT.json")),
        "sources": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [Path(__file__), Path(reference.__file__), Path(teacher.__file__)]
        },
        "source": binding,
        "reference": settings,
        "margin_gate": args.margin_gate,
        "rows": results,
        "summary": summarize(results),
        "leaf_calibration": calibrate(results),
    }
    bank.write_create(args.out, receipt)
    print(json.dumps(receipt["summary"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
