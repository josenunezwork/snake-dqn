"""Fail-closed Q1 endpoint analysis and predeclared package contrasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from research.q0q1_20261017 import q1_bank, q1_evaluate, q1_parity
from src.scripts.eval_stats import student_t_isf

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


def _bootstrap(effects: dict[str, dict[int, dict[str, list[float]]]], draws: int) -> dict[str, Any]:
    """Resample whole seed blocks, then paired worlds identically across arms."""
    rng = np.random.default_rng(20261017)
    samples = {name: [] for name in (*ARMS, "B-A", "D-C", "C-A", "D-B", "interaction")}
    for _ in range(draws):
        seed_values = {arm: [] for arm in ARMS}
        for seed in rng.choice(q1_bank.TRAINING_SEEDS, 5, replace=True):
            indices = {
                mix: rng.integers(
                    0, len(effects["A"][int(seed)][mix]), size=len(effects["A"][int(seed)][mix])
                )
                for mix in MIX_COUNTS
            }
            for arm in ARMS:
                seed_values[arm].append(
                    mean(
                        mean([effects[arm][int(seed)][mix][i] for i in indices[mix]])
                        for mix in MIX_COUNTS
                    )
                )
        values = {arm: mean(seed_values[arm]) for arm in ARMS}
        values.update(
            {
                "B-A": values["B"] - values["A"],
                "D-C": values["D"] - values["C"],
                "C-A": values["C"] - values["A"],
                "D-B": values["D"] - values["B"],
            }
        )
        values["interaction"] = values["D-C"] - values["B-A"]
        for name, value in values.items():
            samples[name].append(value)
    return {
        "draws": draws,
        "rng_seed": 20261017,
        "method": "paired world-within-seed; common resampling across all four arms",
        "central90": {
            name: [float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))]
            for name, values in samples.items()
        },
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
    preserve = not guards and float(g["lower90"]) > -10 and scripted >= -10
    if tripwire:
        return "KILL", ["training tripwire"], False
    if guards or float(g["upper90"]) <= 0:
        return "KILL", guards + (["upper90(G) <= 0"] if float(g["upper90"]) <= 0 else []), preserve
    go = (
        float(g["lower90"]) > 0
        and float(g["mean"]) >= 20
        and scripted >= 0
        and sum(value > 0 for value in g.get("seed_effects", [])) >= 4
    )
    return (
        ("GO" if go else "INCONCLUSIVE"),
        [],
        preserve,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_input(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Require immutable complete shard receipts, never an unbound list of rows."""
    errors: list[str] = []
    if manifest.get("schema") != "q1-analysis-input/v1":
        return [], ["input must be a q1-analysis-input/v1 receipt collection"]
    bank, audit, shards = (manifest.get(k) for k in ("bank", "bank_audit", "shards"))
    if not isinstance(bank, dict) or bank.get("schema") != q1_bank.SCHEMA:
        return [], ["missing Q1 bank manifest"]
    if bank.get("allocation_semantic_sha256") != q1_bank.canonical_sha(
        {"worlds": bank.get("worlds"), "parity_worlds": bank.get("parity_worlds")}
    ):
        return [], ["Q1 bank semantic identity mismatch"]
    # A recomputed arbitrary bank is not the pre-registered allocation.
    registered = q1_bank.load_world_bank()
    if bank != registered:
        return [], ["bank differs from committed Q1 allocation"]
    if not isinstance(audit, dict) or audit != json.loads(q1_bank.AUDIT_OUTPUT.read_text()):
        return [], ["bank audit differs from committed collision audit"]
    if (
        audit.get("pass") is not True
        or audit.get("world_bank_semantic_sha256") != bank["allocation_semantic_sha256"]
    ):
        return [], ["bank collision audit failed"]
    if not isinstance(shards, list) or not shards:
        return [], ["missing immutable evaluation shards"]
    source = q1_evaluate.source_manifest()
    bank_sha = _digest(bank)
    by_identity: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    all_rows: list[dict[str, Any]] = []
    for shard in shards:
        try:
            if (
                not isinstance(shard, dict)
                or shard.get("schema") != q1_evaluate.SCHEMA
                or shard.get("status") != "completed"
                or shard.get("clean_exit") is not True
            ):
                raise ValueError("incomplete shard receipt")
            identity, rows = shard["identity"], shard["rows"]
            seed, arm = identity["training_seed"], identity["arm"]
            if seed not in q1_bank.TRAINING_SEEDS or arm not in (*ARMS, B8):
                raise ValueError("unregistered shard arm or seed")
            index, count = identity["shard"], identity["shards"]
            if type(index) is not int or type(count) is not int or not 0 <= index < count <= 128:
                raise ValueError("invalid shard partition")
            expected_worlds = q1_evaluate.shard_worlds(
                q1_bank.worlds_for_seed(bank, seed), index, count
            )
            if (
                identity.get("bank_sha256") != bank_sha
                or identity.get("b8_sha256") != B8_SHA256
                or identity.get("endpoint") != 8_000_000
            ):
                raise ValueError("shard bank/comparator/endpoint mismatch")
            if (
                identity.get("evaluator_source_manifest") != source
                or identity.get("evaluator_source_sha256")
                != source["research/q0q1_20261017/q1_evaluate.py"]
            ):
                raise ValueError("shard source differs from pinned evaluator")
            if identity.get("profile_digest") != bank["profile"]["base_profile_digest"]:
                raise ValueError("shard profile mismatch")
            if identity.get("world_hashes") != [
                w["world_semantic_sha256"] for w in expected_worlds
            ] or identity.get("world_count") != len(expected_worlds):
                raise ValueError("shard world allocation mismatch")
            parity = shard.get("runtime_parity")
            if not isinstance(parity, dict):
                raise ValueError("missing runtime parity receipt")
            q1_parity.validate_runtime_receipt(
                parity, bank_sha256=bank_sha, b8_sha256=B8_SHA256, source_manifest=source
            )
            if (
                identity.get("runtime_parity_receipt_sha256") != _digest(parity)
                or identity.get("environment_sha256") != parity["environment"]["sha256"]
            ):
                raise ValueError("shard runtime parity/environment binding mismatch")
            checkpoint_sha = identity.get("checkpoint_sha256")
            provenance = shard.get("training_provenance")
            if arm == B8:
                if checkpoint_sha != B8_SHA256 or provenance is not None:
                    raise ValueError("shared B8 identity mismatch")
            else:
                if not isinstance(provenance, dict):
                    raise ValueError("missing candidate training provenance")
                q1_evaluate.validate_provenance(provenance, arm, seed, checkpoint_sha)
                if identity.get("training_provenance_sha256") != _digest(provenance):
                    raise ValueError("training provenance digest mismatch")
            if (
                not isinstance(rows, list)
                or len(rows) != len(expected_worlds)
                or shard.get("rows_sha256") != _digest(rows)
            ):
                raise ValueError("shard row count/digest mismatch")
            for row, world in zip(rows, expected_worlds):
                required = {
                    "schema": q1_evaluate.ROW_SCHEMA,
                    "training_seed": seed,
                    "arm": arm,
                    "checkpoint_step": 8_000_000,
                    "checkpoint_sha256": checkpoint_sha,
                    "b8_checkpoint_sha256": B8_SHA256,
                    "world_bank_sha256": bank_sha,
                    "world_profile_sha256": identity["profile_digest"],
                    "initial_world_sha256": world["initial_world_sha256"],
                    "world_semantic_sha256": world["world_semantic_sha256"],
                    "roster_semantic_sha256": world["roster_semantic_sha256"],
                    "world_seed": world["world_seed"],
                    "mix": world["mix"],
                    "horizon": 5000,
                    "frames_stepped": 5000,
                    "clean_exit": True,
                    "evaluator_source_sha256": identity["evaluator_source_sha256"],
                }
                if not isinstance(row, dict) or any(row.get(k) != v for k, v in required.items()):
                    raise ValueError("row differs from checkpoint/world/source receipt")
                for name in ("mass_integral", "alive_frame_fraction"):
                    value = row.get(name)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                    ):
                        raise ValueError("nonfinite or invalid metric")
                if row["alive_frame_fraction"] > 1 or type(row.get("no_death_episode")) is not bool:
                    raise ValueError("invalid survival metric")
            by_identity[seed, arm].append(shard)
            all_rows.extend(rows)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            errors.append(str(exc))
    for seed in q1_bank.TRAINING_SEEDS:
        for arm in (*ARMS, B8):
            items = by_identity[seed, arm]
            if not items:
                errors.append(f"missing identity coverage: {arm}/{seed}")
                continue
            counts = {x["identity"]["shards"] for x in items}
            if len(counts) != 1 or sorted(x["identity"]["shard"] for x in items) != list(
                range(next(iter(counts)))
            ):
                errors.append(f"duplicate, missing, or mixed shard partitions: {arm}/{seed}")
            for field in ("checkpoint_sha256", "training_provenance_sha256"):
                if len({x["identity"].get(field) for x in items}) != 1:
                    errors.append(f"conflicting identity {field}: {arm}/{seed}")
    return all_rows, sorted(set(errors))


def analyze(manifest: dict[str, Any], bootstrap_draws: int = 2000) -> dict[str, Any]:
    if type(bootstrap_draws) is not int or bootstrap_draws <= 0:
        raise ValueError("bootstrap draws must be a positive integer")
    rows, errors = validate_input(manifest)
    if errors:
        return {"schema": "q1-analysis/v1", "verdict": "INVALID", "validation_errors": errors}
    by = defaultdict(dict)
    for row in sorted(
        rows, key=lambda r: (r["training_seed"], r["mix"], r["world_seed"], r["arm"])
    ):
        by[row["training_seed"], row["mix"], row["world_semantic_sha256"]][row["arm"]] = row
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
    trip = {}  # Every accepted candidate has a complete, tripwire-free training receipt.
    for arm in ARMS:
        mix_mass = {
            m: mean([x for s in q1_bank.TRAINING_SEEDS for x in effects[arm][s][m]])
            for m in MIX_COUNTS
        }
        seedg = {
            s: mean(mean(effects[arm][s][m]) for m in MIX_COUNTS) for s in q1_bank.TRAINING_SEEDS
        }
        g = t90(list(seedg.values()))
        g["seed_effects"] = list(seedg.values())
        av = {m: mean(alive[arm][m]) for m in MIX_COUNTS}
        nd = {m: mean(death[arm][m]) for m in MIX_COUNTS}
        verdict, reasons, preserve = _decision(
            g,
            mix_mass["scripted"],
            av,
            nd,
            min(
                mean(
                    [
                        value
                        for (s0, mix, _), identities in by.items()
                        if s0 == s and mix == "scripted"
                        for value in [
                            float(identities[arm]["alive_frame_fraction"])
                            - float(identities[B8]["alive_frame_fraction"])
                        ]
                    ]
                )
                for s in q1_bank.TRAINING_SEEDS
            ),
            bool(trip.get(arm)),
        )
        arms[arm] = {
            "verdict": verdict,
            "kill_reasons": reasons,
            "G": g,
            "seed_effects": seedg,
            "mix_mass_differences": mix_mass,
            "per_seed_mix_effects": {
                str(s): {m: mean(effects[arm][s][m]) for m in MIX_COUNTS}
                for s in q1_bank.TRAINING_SEEDS
            },
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
    contrasts["interaction"] = t90(
        [
            (arms["D"]["seed_effects"][s] - arms["C"]["seed_effects"][s])
            - (arms["B"]["seed_effects"][s] - arms["A"]["seed_effects"][s])
            for s in q1_bank.TRAINING_SEEDS
        ]
    )
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
    p.add_argument("input", type=Path, nargs="?")
    p.add_argument("--shard-root", type=Path, help="directory of extracted immutable shard JSONs")
    p.add_argument("--out", type=Path)
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    a = p.parse_args()
    if (a.input is None) == (a.shard_root is None):
        p.error("supply either input manifest or --shard-root")
    if a.shard_root is not None:
        v = {
            "schema": "q1-analysis-input/v1",
            "bank": q1_bank.load_world_bank(),
            "bank_audit": json.loads(q1_bank.AUDIT_OUTPUT.read_text()),
            "shards": [
                json.loads(path.read_text())
                for path in sorted(a.shard_root.rglob("shard-*-of-*.json"))
            ],
        }
    else:
        v = json.loads(a.input.read_text())
    r = analyze(v, a.bootstrap_draws)
    text = json.dumps(r, indent=2, sort_keys=True) + "\n"
    a.out.write_text(text) if a.out else print(text)
    return 2 if r["verdict"] == "INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
