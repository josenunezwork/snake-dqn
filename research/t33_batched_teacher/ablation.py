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
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from research.phase0_search_20261019 import bank, teacher
from research.phase0_search_20261019.preflight import B8, fixture
from research.t33_batched_teacher import reference

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
    full = reference.scores(world, actions, config, decision, common_random=True)
    legacy = teacher._reference_scores(world, actions, config, decision)
    winner = max(full, key=full.__getitem__)
    old_winner = max(legacy, key=legacy.__getitem__)
    result = []
    margin = teacher._margin(q, legal)
    for variant in choices:
        before = time.perf_counter()
        searched = gate is None or margin < gate
        if searched:
            cheap = replace(config, **{k: v for k, v in variant.items() if k not in ("id", "leaf")})
            scores = reference.scores(world, actions, cheap, decision, leaf=variant["leaf"])
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
        "seconds": time.perf_counter() - started,
    }


def task(payload: tuple) -> list[dict]:
    """One world per single-thread worker; stream samples rather than clone a bank."""
    row, checkpoint, count, settings, choices, gate, smoke = payload
    teacher.configure_cpu()
    from research.phase0_search_20261019.runner import operational_guard

    operational_guard()
    output = []
    if smoke:
        world = fixture(row, Path(checkpoint), 800)
        selected = {0}
    else:
        world = teacher.build_world(row, checkpoint)
        eligible = []
        for decision in range(5000 - settings["horizon"] + 1):
            if not world.sim.alive[0, 0]:
                break
            diag = teacher.step_world(world)
            if diag["length"] >= 600 or diag["margin"] < 0:
                eligible.append(decision)
        if len(eligible) < count:
            raise ValueError(f"not enough searched decisions in {row['world_id']}")
        rng = np.random.default_rng(np.random.SeedSequence([20261019, row["world_seed"]]))
        selected = set(map(int, rng.choice(eligible, count, replace=False)))
        world = teacher.build_world(row, checkpoint)

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
        )
        for r in rows
    ]
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
        "reference": settings,
        "margin_gate": args.margin_gate,
        "rows": results,
        "summary": summarize(results),
    }
    bank.write_create(args.out, receipt)
    print(json.dumps(receipt["summary"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
