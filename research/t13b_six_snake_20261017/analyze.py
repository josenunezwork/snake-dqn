"""Validate and summarize T13b's six-snake pairs and all-B8 eight supplement."""

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

from research.t13_product_roster_20261017.analyze import CAUSES, COUNTS, metric_samples
from research.t13b_six_snake_20261017.run import SIX_ROSTERS, stem


def validate_episode(record: dict, roster: str, seed: int, snakes: int) -> None:
    """Fail closed on raw identity, count, horizon, and accounting drift."""
    if record.get("schema_version") != "t13b-six-snake-episode/v1":
        raise ValueError("incorrect T13b episode schema")
    if record.get("roster") != roster or record.get("seed") != seed:
        raise ValueError("episode identity mismatch")
    if record.get("horizon") != 5000 or record.get("frames_stepped") != 5000:
        raise ValueError("incomplete horizon")
    if record.get("requested_snake_count") != snakes or len(record.get("per_snake", {})) != snakes:
        raise ValueError("incorrect roster size")
    if record["initial_world"].get("effective_num_snakes") != snakes:
        raise ValueError("initial world count mismatch")
    for row in [record["hero"], record["arena"], *record["per_snake"].values()]:
        if sum(row["length_histogram"].values()) != row["decisions"]:
            raise ValueError("length histogram does not reconcile to decisions")
        if sum(row["death_causes"].values()) != row["deaths"]:
            raise ValueError("death causes do not reconcile")
        if (
            set(row["death_causes"]) - set(CAUSES)
            or not 0 <= row["boost_decisions"] <= row["decisions"]
        ):
            raise ValueError("invalid death/boost accounting")
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


def _paired(draws: dict[str, dict[str, dict[str, np.ndarray]]], left: str, right: str) -> dict:
    result = {}
    for scope in ("hero", "arena"):
        result[scope] = {}
        for metric, values in draws[left][scope].items():
            delta = values - draws[right][scope][metric]
            result[scope][metric] = {
                "difference": float(delta[0]),
                "ci95": np.quantile(delta[1:], [0.025, 0.975]).tolist(),
            }
    return result


def analyze(root: Path, bootstrap: int = 10000) -> dict[str, Any]:
    intent, completion = (
        json.loads((root / name).read_text()) for name in ("intent.json", "completion.json")
    )
    if completion["status"] != "complete" or completion["episodes"] != 125:
        raise ValueError("125 complete episodes required")
    seeds = intent["world_seeds"]
    if len(seeds) != len(set(seeds)) or len(seeds) != 25:
        raise ValueError("25 unique paired seeds required")
    specs = [(roster, 6) for roster in SIX_ROSTERS] + [("all_b8", 8)]
    records: dict[str, list[dict]] = {}
    receipt_index = {(r["roster"], r["snake_count"], r["seed"]): r for r in completion["receipts"]}
    for roster, snakes in specs:
        key = f"{roster}_n{snakes}"
        records[key] = []
        for seed in seeds:
            path = root / "episodes" / f"{stem(roster, snakes, seed)}.json"
            record = json.loads(path.read_text())
            validate_episode(record, roster, seed, snakes)
            receipt = receipt_index.get((roster, snakes, seed))
            if (
                receipt is None
                or receipt["returncode"] != 0
                or hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]
            ):
                raise ValueError("raw episode digest/exit mismatch")
            records[key].append(record)
    for index, seed in enumerate(seeds):
        worlds = [records[f"{r}_n6"][index]["initial_world"] for r in SIX_ROSTERS]
        if any(world != worlds[0] for world in worlds):
            raise ValueError(f"six-snake paired initial world/RNG differs for seed {seed}")
    for key, values in records.items():
        identities = [row["served"] for row in values]
        if any(identity != identities[0] for identity in identities):
            raise ValueError(f"served identity drift in {key}")
    rng = np.random.default_rng(20261017)
    indices = rng.integers(0, 25, size=(bootstrap, 25))
    weights = np.vstack(
        (
            np.ones((1, 25)),
            np.asarray([np.bincount(row, minlength=25) for row in indices], dtype=float),
        )
    )
    summary, draws = {}, {}
    for key, episodes in records.items():
        snakes = episodes[0]["requested_snake_count"]
        summary[key], draws[key] = {}, {}
        for scope in ("hero", "arena"):
            values = metric_samples(
                [row[scope] for row in episodes], weights, 5000, 1 if scope == "hero" else snakes
            )
            summary[key][scope] = {name: float(value[0]) for name, value in values.items()}
            draws[key][scope] = values
    pairs = {
        f"{left} - {right}": _paired(draws, f"{left}_n6", f"{right}_n6")
        for left, right in itertools.combinations(SIX_ROSTERS, 2)
    }
    supplemental = _paired(draws, "all_b8_n8", "all_b8_n6")
    return {
        "schema": "t13b-summary/v1",
        "descriptive_not_gate": True,
        "measurement_git_sha": intent["git_sha"],
        "episodes_per_row": 25,
        "horizon": 5000,
        "world_seeds": seeds,
        "bootstrap": {
            "resamples": bootstrap,
            "seed": 20261017,
            "method": "paired seed percentile 95%, pooled metrics recomputed; no multiplicity adjustment",
        },
        "rosters": summary,
        "six_snake_pairwise": pairs,
        "supplemental_all_b8_n8_minus_n6": supplemental,
        "identities": {key: values[0]["served"] for key, values in records.items()},
        "mixed_policy_identities": records["mixed_n6"][0]["mixed_policy_identities"],
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
    rows = [json.loads(path.read_text()) for path in sorted((args.run / "episodes").glob("*.json"))]
    with (args.out / "episodes.json.gz").open("xb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as zipped:
            zipped.write((json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode())
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
