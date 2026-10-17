"""Validate and summarize the fixed T13 run with paired seed bootstrap CIs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from research.t13_product_roster_20261017.run import ROSTERS

COUNTS = (
    "decisions",
    "deaths",
    "mass_sum",
    "boost_decisions",
    "food_eaten",
    "kills_made",
    "kills_taken",
)
CAUSES = ("head_on", "enemy_body", "wall", "self", "other", "starvation")


def metric_samples(
    rows: list[dict], weights: np.ndarray, horizon: int, snakes: int
) -> dict[str, np.ndarray]:
    """Recompute pooled metrics for each seed-multiplicity row (including quantiles)."""
    counts = {
        name: weights @ np.asarray([row[name] for row in rows], dtype=float) for name in COUNTS
    }
    decisions = counts["decisions"]
    if np.any(decisions <= 0):
        raise ValueError("empty decision sample")
    episodes = weights.sum(axis=1)
    result = {name: counts[name] for name in COUNTS if name != "mass_sum"}
    result.update(
        {
            "deaths_per_1000": 1000 * counts["deaths"] / decisions,
            "food_per_1000": 1000 * counts["food_eaten"] / decisions,
            "boost_rate": counts["boost_decisions"] / decisions,
            "mass_h5000": counts["mass_sum"] / (episodes * horizon),
            "mass_per_snake": counts["mass_sum"] / (episodes * horizon * snakes),
        }
    )
    for cause in CAUSES:
        values = np.asarray([row["death_causes"].get(cause, 0) for row in rows], dtype=float)
        result[f"death_cause_{cause}"] = weights @ values
    max_length = max(int(key) for row in rows for key in row["length_histogram"])
    hist = np.zeros((len(rows), max_length + 1))
    for index, row in enumerate(rows):
        for key, value in row["length_histogram"].items():
            hist[index, int(key)] = value
    outputs = {key: [] for key in ("length_median", "length_p95", "length_max")}
    for start in range(0, len(weights), 250):
        block = weights[start : start + 250] @ hist
        cumulative = np.cumsum(block, axis=1)
        total = cumulative[:, -1]
        for name, q in (("length_median", 0.5), ("length_p95", 0.95)):
            # Same nearest-order-statistic convention as T12 _quantiles.
            ranks = np.rint((total - 1) * q)
            outputs[name].append(np.argmax(cumulative > ranks[:, None], axis=1))
        outputs["length_max"].append(max_length - np.argmax((block > 0)[:, ::-1], axis=1))
    result.update({key: np.concatenate(value).astype(float) for key, value in outputs.items()})
    return result


def validate_episode(record: dict, roster: str, seed: int) -> None:
    if record["roster"] != roster or record["seed"] != seed:
        raise ValueError("episode identity mismatch")
    if record["horizon"] != 5000 or record["frames_stepped"] != 5000:
        raise ValueError("incomplete horizon")
    if len(record["per_snake"]) != 12:
        raise ValueError("incorrect roster size")
    for row in [record["hero"], record["arena"], *record["per_snake"].values()]:
        if sum(row["length_histogram"].values()) != row["decisions"]:
            raise ValueError("length histogram does not reconcile to decisions")
        if sum(row["death_causes"].values()) != row["deaths"]:
            raise ValueError("death causes do not reconcile")
        if set(row["death_causes"]) - set(CAUSES):
            raise ValueError("unrecognized death cause")
        if not 0 <= row["boost_decisions"] <= row["decisions"]:
            raise ValueError("invalid boost accounting")
        if row["kills_taken"] > row["deaths"]:
            raise ValueError("attributed deaths exceed deaths")
    if record["hero"] != record["per_snake"][str(record["hero_id"])]:
        raise ValueError("hero does not match slot counters")
    for key in COUNTS:
        if sum(row[key] for row in record["per_snake"].values()) != record["arena"][key]:
            raise ValueError(f"arena counter mismatch: {key}")
    for key in ("length_histogram", "death_causes"):
        pooled = Counter()
        for row in record["per_snake"].values():
            pooled.update(row[key])
        if dict(pooled) != record["arena"][key]:
            raise ValueError(f"arena histogram mismatch: {key}")
    if roster == "mixed":
        dispatch = record["mixed_b8_dispatch_ids"]
        if len(dispatch) != record["hero"]["decisions"] or any(
            sid != record["hero_id"] for sid in dispatch
        ):
            raise ValueError("mixed B8 dispatch accounting mismatch")
    if record["arena"]["kills_made"] != record["arena"]["kills_taken"]:
        raise ValueError("arena kills made/taken do not reconcile")


def analyze(root: Path, bootstrap: int = 10000) -> dict[str, Any]:
    intent = json.loads((root / "intent.json").read_text())
    completion = json.loads((root / "completion.json").read_text())
    if completion["status"] != "complete" or completion["episodes"] != 100:
        raise ValueError("100 complete episodes required")
    seeds = intent["world_seeds"]
    if len(seeds) != len(set(seeds)) or len(seeds) != 25:
        raise ValueError("25 unique paired seeds required")
    records = {}
    for roster in ROSTERS:
        records[roster] = []
        for seed in seeds:
            path = root / "episodes" / f"{roster}-{seed}.json"
            record = json.loads(path.read_text())
            validate_episode(record, roster, seed)
            receipt = next(
                r for r in completion["receipts"] if r["roster"] == roster and r["seed"] == seed
            )
            if (
                receipt["returncode"] != 0
                or hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]
            ):
                raise ValueError("raw episode digest/exit mismatch")
            records[roster].append(record)
    for index in range(25):
        worlds = [records[roster][index]["initial_world"] for roster in ROSTERS]
        if any(world != worlds[0] for world in worlds):
            raise ValueError(f"paired initial world/RNG differs for seed {seeds[index]}")
    for roster in ROSTERS:
        identities = [row["served"] for row in records[roster]]
        if any(identity != identities[0] for identity in identities):
            raise ValueError(f"served identity drift in {roster}")
    rng = np.random.default_rng(20261017)
    indices = rng.integers(0, 25, size=(bootstrap, 25))
    weights = np.asarray([np.bincount(row, minlength=25) for row in indices], dtype=float)
    weights = np.vstack((np.ones((1, 25)), weights))
    summary, draws = {}, {}
    for roster, episodes in records.items():
        summary[roster], draws[roster] = {}, {}
        for scope in ("hero", "arena"):
            values = metric_samples(
                [row[scope] for row in episodes], weights, 5000, 1 if scope == "hero" else 12
            )
            summary[roster][scope] = {key: float(value[0]) for key, value in values.items()}
            draws[roster][scope] = values
    pairs = {}
    for left, right in itertools.combinations(ROSTERS, 2):
        pair = {}
        for scope in ("hero", "arena"):
            pair[scope] = {}
            for metric, values in draws[left][scope].items():
                delta = values - draws[right][scope][metric]
                bounds = np.quantile(delta[1:], [0.025, 0.975])
                pair[scope][metric] = {"difference": float(delta[0]), "ci95": bounds.tolist()}
        pairs[f"{left} - {right}"] = pair
    return {
        "schema": "t13-summary/v1",
        "descriptive_not_gate": True,
        "measurement_git_sha": intent["git_sha"],
        "episodes_per_roster": 25,
        "horizon": 5000,
        "snake_count": 12,
        "world_seeds": seeds,
        "bootstrap": {
            "resamples": bootstrap,
            "seed": 20261017,
            "method": "paired seed percentile 95%, pooled metrics recomputed; no multiplicity adjustment",
        },
        "rosters": summary,
        "pairwise": pairs,
        "identities": {key: value[0]["served"] for key, value in records.items()},
        "mixed_policy_identities": records["mixed"][0]["mixed_policy_identities"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name in ("intent.json", "completion.json", "progress.jsonl"):
        (args.out / name).write_bytes((args.run / name).read_bytes())
    # Canonical compact raw episode archive, excluding bulky operational stderr.
    records = [json.loads(p.read_text()) for p in sorted((args.run / "episodes").glob("*.json"))]
    with (args.out / "episodes.json.gz").open("xb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as zipped:
            zipped.write(
                (json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )
    (args.out / "evidence_sha256.json").write_text(
        json.dumps(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(args.out.iterdir())
                if p.is_file()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(result["rosters"], indent=2))


if __name__ == "__main__":
    main()
