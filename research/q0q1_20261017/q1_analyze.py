"""Fail-closed Q1 endpoint analysis and predeclared package contrasts."""

from __future__ import annotations
import argparse, hashlib, json, math, statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from src.scripts.eval_stats import student_t_isf
from research.q0q1_20261017 import q1_bank

ARMS = ("A", "B", "C", "D")
B8 = "B8"
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
MIX_COUNTS = q1_bank.MIX_COUNTS


def canonical(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def mean(x: Iterable[float]) -> float:
    x = list(x)
    return sum(x) / len(x)


def t90(x: list[float]) -> dict[str, float | int]:
    if len(x) != 5:
        raise ValueError("exactly five independent seed effects required")
    m = mean(x)
    se = statistics.stdev(x) / math.sqrt(5)
    c = student_t_isf(0.10, 4)
    return {
        "n": 5,
        "df": 4,
        "mean": m,
        "sample_sd": statistics.stdev(x),
        "standard_error": se,
        "critical_one_sided_90": c,
        "lower90": m - c * se,
        "upper90": m + c * se,
    }


def _bootstrap(
    effects: dict[str, dict[int, dict[str, list[float]]]], draws: int
) -> dict[str, list[float]]:
    rng = np.random.default_rng(20261017)
    out = []
    for _ in range(draws):
        seeds = rng.choice(q1_bank.TRAINING_SEEDS, 5, replace=True)
        vals = []
        for arm in ARMS:
            svals = []
            for seed in seeds:
                mix = [
                    mean(
                        rng.choice(
                            effects[arm][int(seed)][m],
                            len(effects[arm][int(seed)][m]),
                            replace=True,
                        )
                    )
                    for m in MIX_COUNTS
                ]
                svals.append(mean(mix))
            vals.append(mean(svals))
        out.append(vals)
    return {
        arm: [
            float(np.quantile([x[i] for x in out], 0.05)),
            float(np.quantile([x[i] for x in out], 0.95)),
        ]
        for i, arm in enumerate(ARMS)
    }


def _decision(
    g: dict[str, float | int],
    scripted: float,
    alive: dict[str, float],
    nodeath: dict[str, float],
    min_scripted_seed: float,
    tripwire: bool,
) -> tuple[str, list[str], bool]:
    guards = (
        [m for m in MIX_COUNTS if alive[m] < -0.03]
        + (["scripted per-seed alive"] if min_scripted_seed < -0.15 else [])
        + [m + " no-death" for m in MIX_COUNTS if nodeath[m] < 0.5]
    )
    if tripwire:
        return "KILL", ["training tripwire"], False
    if guards or float(g["upper90"]) <= 0:
        return "KILL", guards + (["upper90(G) <= 0"] if float(g["upper90"]) <= 0 else []), False
    go = float(g["lower90"]) > 0 and float(g["mean"]) >= 20 and scripted >= 0
    return (
        ("GO" if go else "INCONCLUSIVE"),
        [],
        not guards and float(g["lower90"]) > -10 and scripted >= -10,
    )


def analyze(manifest: dict[str, Any], bootstrap_draws: int = 2000) -> dict[str, Any]:
    errors = []
    rows = manifest.get("rows", [])
    bank = manifest.get("bank")
    if not isinstance(rows, list):
        return {"verdict": "INVALID", "validation_errors": ["rows must be a list"]}
    if not isinstance(bank, dict) or bank.get("schema") != q1_bank.SCHEMA:
        return {"verdict": "INVALID", "validation_errors": ["missing Q1 bank manifest"]}
    if bank.get("allocation_semantic_sha256") != q1_bank.canonical_sha(
        {"worlds": bank.get("worlds"), "parity_worlds": bank.get("parity_worlds")}
    ):
        return {"verdict": "INVALID", "validation_errors": ["Q1 bank semantic identity mismatch"]}
    bank_sha = hashlib.sha256(canonical(bank).encode()).hexdigest()
    expected = {
        (seed, mix, world["world_semantic_sha256"])
        for seed in q1_bank.TRAINING_SEEDS
        for mix in MIX_COUNTS
        for world in bank.get("worlds", {}).get(str(seed), [])
        if world.get("mix") == mix
    }
    by = defaultdict(dict)
    seen = set()
    for row in rows:
        key = (
            row.get("training_seed"),
            row.get("mix"),
            row.get("world_semantic_sha256"),
            row.get("arm"),
        )
        if (
            row.get("schema") != "q1-eval-row/v1"
            or key in seen
            or key[:3] not in expected
            or key[3] not in (*ARMS, B8)
        ):
            errors.append("invalid, duplicate, or unregistered row")
            continue
        seen.add(key)
        by[key[:3]][key[3]] = row
        if (
            row.get("horizon") != 5000
            or row.get("frames_stepped") != 5000
            or not row.get("clean_exit")
        ):
            errors.append("invalid H5000/clean receipt")
        if row.get("world_bank_sha256") != bank_sha:
            errors.append("row is not bound to supplied Q1 bank")
        if row.get("b8_checkpoint_sha256") not in (None, B8_SHA256):
            errors.append("row uses unpinned B8 comparator")
    for base in expected:
        names = by.get(base, {})
        if set(names) != {*ARMS, B8}:
            errors.append("missing identity coverage")
        elif (
            len({x.get("world_bank_sha256") for x in names.values()}) != 1
            or len({x.get("initial_world_sha256") for x in names.values()}) != 1
        ):
            errors.append("paired identity mismatch")
    if errors:
        return {
            "schema": "q1-analysis/v1",
            "verdict": "INVALID",
            "validation_errors": sorted(set(errors)),
        }
    effects = {a: {s: {m: [] for m in MIX_COUNTS} for s in q1_bank.TRAINING_SEEDS} for a in ARMS}
    alive = {a: {m: [] for m in MIX_COUNTS} for a in ARMS}
    death = {a: {m: [] for m in MIX_COUNTS} for a in ARMS}
    for (_, mix, _), identities in by.items():
        seed = int(next(iter(identities.values()))["training_seed"])
        base = identities[B8]
        for arm in ARMS:
            candidate = identities[arm]
            effects[arm][seed][mix].append(
                float(candidate["mass_integral"]) - float(base["mass_integral"])
            )
            alive[arm][mix].append(
                float(candidate["alive_frame_fraction"]) - float(base["alive_frame_fraction"])
            )
            death[arm][mix].append(float(bool(candidate["no_death_episode"])))
    arms = {}
    trip = manifest.get("tripwires", {})
    for arm in ARMS:
        mix_mass = {
            m: mean([x for s in q1_bank.TRAINING_SEEDS for x in effects[arm][s][m]])
            for m in MIX_COUNTS
        }
        seedg = {
            s: mean(mean(effects[arm][s][m]) for m in MIX_COUNTS) for s in q1_bank.TRAINING_SEEDS
        }
        g = t90(list(seedg.values()))
        av = {m: mean(alive[arm][m]) for m in MIX_COUNTS}
        nd = {m: mean(death[arm][m]) for m in MIX_COUNTS}
        verdict, reasons, preserve = _decision(
            g,
            mix_mass["scripted"],
            av,
            nd,
            min(mean(effects[arm][s]["scripted"]) for s in q1_bank.TRAINING_SEEDS),
            bool(trip.get(arm)),
        )
        arms[arm] = {
            "verdict": verdict,
            "kill_reasons": reasons,
            "G": g,
            "seed_effects": seedg,
            "mix_mass_differences": mix_mass,
            "alive_fraction_differences": av,
            "no_death_episode_fraction": nd,
            "preservation_certificate": preserve,
        }
    contrasts = {
        name: t90(
            [
                arms[right]["seed_effects"][s] - arms[left]["seed_effects"][s]
                for s in q1_bank.TRAINING_SEEDS
            ]
        )
        for name, left, right in (
            ("B-A", "A", "B"),
            ("D-C", "C", "D"),
            ("C-A", "A", "C"),
            ("D-B", "B", "D"),
        )
    }
    contrasts["interaction"] = {
        k: (
            contrasts["D-C"][k] - contrasts["B-A"][k]
            if isinstance(contrasts["D-C"][k], float)
            else contrasts["D-C"][k]
        )
        for k in contrasts["D-C"]
    }
    eligible = [a for a in ARMS if arms[a]["verdict"] == "GO"]
    purpose = (
        "GO_TO_MAC_PHASE_R"
        if eligible
        else (
            "Q2_PARENT" if any(arms[a]["preservation_certificate"] for a in ARMS) else "CLOSE_ROUTE"
        )
    )
    candidates = eligible or [a for a in ARMS if arms[a]["preservation_certificate"]]
    selected = (
        sorted(
            candidates,
            key=lambda a: (
                -float(arms[a]["G"]["lower90"]),
                -float(arms[a]["G"]["mean"]),
                "DCBA".index(a),
            ),
        )[0]
        if candidates
        else None
    )
    return {
        "schema": "q1-analysis/v1",
        "verdict": purpose,
        "selected_recipe": selected,
        "arms": arms,
        "contrasts": contrasts,
        "hierarchical_world_within_seed_bootstrap": _bootstrap(effects, bootstrap_draws),
        "validation_errors": [],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    a = p.parse_args()
    v = json.loads(a.input.read_text())
    r = analyze(v, a.bootstrap_draws)
    text = json.dumps(r, indent=2, sort_keys=True) + "\n"
    a.out.write_text(text) if a.out else print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
