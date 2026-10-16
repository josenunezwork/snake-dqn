"""Read-only T4a bank balance and fresh development-seed audits.

Side identity comes from original rollout outcomes, never the prefer-nonboost
label policy. The distribution thresholds are descriptive engineering criteria,
not a statistical equivalence test or evidence of training performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.t3_screen_20261015 import banks as prior_banks

QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def _distribution(decisive: np.ndarray, counter: np.ndarray) -> dict[str, Any]:
    """Compare finite one-dimensional values without assuming normality."""
    if not len(decisive) or not len(counter):
        raise ValueError("both bank sides must be nonempty")
    if not np.isfinite(decisive).all() or not np.isfinite(counter).all():
        raise ValueError("audit inputs must be finite")
    scale = float(np.sqrt((decisive.var() + counter.var()) / 2))
    delta = float(counter.mean() - decisive.mean())
    smd = delta / scale if scale else (0.0 if delta == 0 else None)
    iqr = float(np.quantile(decisive, 0.75) - np.quantile(decisive, 0.25))
    median_delta = float(np.median(counter) - np.median(decisive))
    relative = abs(median_delta) / iqr if iqr else (0.0 if median_delta == 0 else None)
    points = np.sort(np.concatenate((decisive, counter)))
    ks = float(
        np.max(
            np.abs(
                np.searchsorted(np.sort(decisive), points, side="right") / len(decisive)
                - np.searchsorted(np.sort(counter), points, side="right") / len(counter)
            )
        )
    )
    return {
        "decisive_quantiles": np.quantile(decisive, QUANTILES).tolist(),
        "counter_quantiles": np.quantile(counter, QUANTILES).tolist(),
        "decisive_mean": float(decisive.mean()),
        "counter_mean": float(counter.mean()),
        "standardized_mean_difference": smd,
        "absolute_median_difference_over_decisive_iqr": relative,
        "ks_distance": ks,
        "badly_mismatched": smd is None or relative is None or abs(smd) > 0.25 or relative > 0.25,
    }


def bank_audit(bank: Path) -> dict[str, Any]:
    """Verify bound source metadata and compare the two immutable bank sides."""
    bank = Path(bank).resolve()
    manifest_path = bank / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["source"]["path"])
    arrays: dict[str, list[np.ndarray]] = {}
    frames, provenance_decisive, bindings = [], [], {}
    for name, expected in sorted(manifest["shards"].items()):
        shard = bank / name
        if prior_banks._sha256(shard) != expected:
            raise ValueError(f"derived shard digest mismatch: {shard}")
        source_shard = source / name
        source_expected = manifest["source"]["files"][name]
        if prior_banks._sha256(source_shard) != source_expected:
            raise ValueError(f"original source shard digest mismatch: {source_shard}")
        provenance = source / Path(name).with_suffix(".provenance.jsonl")
        provenance_expected = manifest["source"]["files"][provenance.name]
        if prior_banks._sha256(provenance) != provenance_expected:
            raise ValueError(f"source provenance digest mismatch: {provenance}")
        with np.load(shard, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        records = [json.loads(line) for line in provenance.read_text().splitlines()]
        if len(records) != len(data["action"]) or any(
            record["row"] != index for index, record in enumerate(records)
        ):
            raise ValueError("provenance row alignment is invalid")
        if "surviving_original" not in data:
            raise ValueError("v2 bank must preserve original survival outcomes")
        for key in ("action", "surviving", "surviving_original", "length", "q"):
            arrays.setdefault(key, []).append(data[key])
        frames.extend(record["frame"] for record in records)
        provenance_decisive.extend(record["decisive"] for record in records)
        bindings[name] = {
            "sha256": expected,
            "source_sha256": source_expected,
            "provenance_sha256": provenance_expected,
        }
    values = {key: np.concatenate(parts) for key, parts in arrays.items()}
    rows = np.arange(len(values["action"]))
    counters = values["surviving_original"][rows, values["action"]]
    if not np.array_equal(~counters, provenance_decisive):
        raise ValueError("original survival side disagrees with collection provenance")
    derived_side = values["surviving"][rows, values["action"]]
    metrics = {
        "length": values["length"].astype(np.float64),
        "frame": np.asarray(frames, dtype=np.float64),
        "negative_max_q": -values["q"].max(axis=1).astype(np.float64),
    }
    distributions = {
        key: _distribution(value[~counters], value[counters]) for key, value in metrics.items()
    }
    return {
        "schema": "t4a-counter-balance-audit/v1",
        "bank": str(bank),
        "manifest_sha256": prior_banks._sha256(manifest_path),
        "source_manifest_sha256": manifest["source"]["files"]["manifest.json"],
        "shards": bindings,
        "rows": len(rows),
        "decisive_rows": int((~counters).sum()),
        "counter_rows": int(counters.sum()),
        "side_definition": "counter iff surviving_original[row, action_B8]; provenance checked",
        "prefer_nonboost_would_misclassify_counters": int((counters & ~derived_side).sum()),
        "frame_convention": "one-based simulator frame from original aligned provenance JSONL",
        "q_convention": "negative maximum of all six raw stored B8 Q values, matching prefilter",
        "quantile_probabilities": list(QUANTILES),
        "distributions": distributions,
        "criterion": (
            "badly mismatched iff any metric has absolute pooled-SD standardized mean "
            "difference >0.25 or absolute median difference >0.25 decisive IQR; "
            "a zero denominator with nonzero difference also fails"
        ),
        "badly_mismatched": any(item["badly_mismatched"] for item in distributions.values()),
        "matching_required": any(item["badly_mismatched"] for item in distributions.values()),
        "matching_recommendation": (
            "Use bank as stored when thresholds pass. If thresholds fail in a future bank, "
            "weight counter rows by decisive_count(bucket)/counter_count(bucket) jointly "
            "on length and frame; report unsupported decisive buckets and effective sample size."
        ),
        "limit": (
            "Descriptive threshold, not a formal equivalence test. Tail quantiles and KS "
            "remain visible; aggregate balance cannot establish conditional state equivalence."
        ),
    }


def seed_audit(repo: Path, artifacts: Path) -> dict[str, Any]:
    """Audit reserved rounds 65/66 against registered NI and observed JSON receipts."""
    from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b

    repo, artifacts = Path(repo).resolve(), Path(artifacts).resolve()
    registered = set(ni_spec.ni_seeds()) | set(ni_spec_m2b.ni_seeds())
    t2 = prior_banks._registered_t2(repo)
    for seeds in t2["registered_rounds"].values():
        registered.update(seeds)
    for item in t2["historical_distillation"].values():
        registered.update(item["seeds"])
    for round_index in (62, 63, 64):
        registered.update(ni_spec.distill_seeds(96, round_index=round_index))
    roots = [repo / "research"] + sorted(
        path
        for path in artifacts.iterdir()
        if path.is_dir()
        and any(token in path.name for token in ("probe", "eval", "strict", "gate"))
    )
    inventory = {str(root): prior_banks._json_inventory(root) for root in roots}
    observed = set().union(*(set(item["seeds"]) for item in inventory.values()))
    namespace_path = artifacts / "t3-rollout-labels-20261015/bank/namespace.json"
    label_seeds = json.loads(namespace_path.read_text())["seeds"]
    binding = prior_banks._manifest_binding(namespace_path.parent, namespace_path)
    observed.update(label_seeds)
    expected = {str(r): ni_spec.distill_seeds(96, round_index=r) for r in (65, 66)}
    checks = {}
    for name, seeds in expected.items():
        seed_set = set(seeds)
        other = set().union(*(set(v) for k, v in expected.items() if k != name))
        checks[name] = {
            "seeds": seeds,
            "count": len(seeds),
            "unique": len(seed_set) == len(seeds),
            "seed_digest": prior_banks._seed_digest(seeds),
            "registered_collisions": sorted(seed_set & registered),
            "observed_json_or_label_collisions": sorted(seed_set & observed),
            "other_reserved_round_collisions": sorted(seed_set & other),
        }
    return {
        "schema": "t4a-seed-bank-audit/v1",
        "scope": (
            "registered NI/distillation rounds, reserved rounds 62-64, repository research "
            "JSON, artifact directories containing probe/eval/strict/gate, immutable T3 namespace"
        ),
        "registered_t2_audit_pass": t2["pass"],
        "immutable_label_namespace_binding": binding,
        "inventories": {
            name: prior_banks._compact_inventory(item) for name, item in inventory.items()
        },
        "checks": checks,
        "pass": t2["pass"]
        and binding["ok"]
        and all(
            item["unique"]
            and item["count"] == 96
            and not item["registered_collisions"]
            and not item["observed_json_or_label_collisions"]
            and not item["other_reserved_round_collisions"]
            for item in checks.values()
        ),
        "limit": (
            "No claim about unrecorded experiments, arbitrary binary files or external systems."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "counter_balance": bank_audit(args.bank),
        "seed_namespace": seed_audit(args.repo, args.artifacts),
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "matching_required": result["counter_balance"]["matching_required"],
                "seed_audit_pass": result["seed_namespace"]["pass"],
            }
        )
    )


if __name__ == "__main__":
    main()
