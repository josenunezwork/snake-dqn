#!/usr/bin/env python3
"""Q0a descriptive T23 mass-loss audit; CPU-only and capped at 480 H5000 episodes.

By default this only audits saved T23 receipts. ``--run-development`` is an
explicit, bounded packet: it evaluates available 2M/4M checkpoints and B8 on
the T23 evaluator's *fresh development namespace*. Missing snapshots remain
missing and make the development packet incomplete rather than reconstructed.
This is diagnostic output, never a Q1 efficacy decision.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.q0q1_20261017 import audit_q0_bank  # noqa: E402
from research.q0q1_20261017.audit_common import (  # noqa: E402
    B8_SHA256,
    HORIZON,
    MIX_COUNTS,
    T23_SEEDS,
    iter_json,
    mean,
    sha256_file,
    stratum,
    write_json,
)
from research.q0q1_20261017.audit_episode_metrics import FrameObserver, PolicyProbe  # noqa: E402

SCHEMA = "q0-t23-mass-audit/v1"
MAX_EPISODES = 480


def _record(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("record", row)
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def summarize_rows(rows: Iterable[Mapping[str, Any]], *, grouped: bool = True) -> dict[str, Any]:
    """Extract every Q0 descriptive field available from versioned evaluator rows.

    Older T23 rows lack some probes; those fields are reported unavailable,
    rather than replaced with proxy arithmetic.
    """
    rows = list(rows)
    buckets: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    rows = [
        dict(
            row,
            arm="t23" if row.get("arm") == "candidate" else row.get("arm", "unknown"),
            checkpoint_label=row.get(
                "checkpoint_label", str(row.get("checkpoint_step", "unknown"))
            ),
        )
        for row in rows
    ]
    for row in rows:
        record = _record(row)
        probes = record.get("probes", {}) if isinstance(record.get("probes"), Mapping) else {}
        length = int(probes.get("peak_length", record.get("max_mass", 0)) or 0)
        buckets[stratum(length)].append(row)
    out: dict[str, Any] = {"episodes": 0, "strata": {}}
    for label, values in sorted(buckets.items()):
        records = [_record(v) for v in values]
        probes = [
            r.get("probes", {}) if isinstance(r.get("probes"), Mapping) else {} for r in records
        ]
        denoms = [
            r.get("denominators", {}) if isinstance(r.get("denominators"), Mapping) else {}
            for r in records
        ]
        mass = [_number(r.get("mass_integral")) for r in records]
        alive = [_number(r.get("survival_fraction")) for r in records]
        food = [_number(p.get("food_eaten")) for p in probes]
        alive_frames = [_number(d.get("alive_frames")) for d in denoms]
        boost = [_number(p.get("boost_frames_alive")) for p in probes]
        causes = collections.Counter(str(p.get("death_cause") or "survived") for p in probes)

        def present(xs):
            return [x for x in xs if x is not None]

        out["strata"][label] = {
            "episodes": len(values),
            "full_horizon_mass": mean(present(mass)),
            "alive_fraction": mean(present(alive)),
            "no_death_fraction": mean([float(not p.get("death_cause")) for p in probes]),
            "food_mass_per_alive_decision": (
                sum(present(food)) / sum(present(alive_frames))
                if present(food) and present(alive_frames) and sum(present(alive_frames)) > 0
                else None
            ),
            "food_mass_per_whole_horizon": (
                sum(present(food)) / (len(values) * HORIZON) if present(food) else None
            ),
            "boost_frames_per_alive_decision": (
                sum(present(boost)) / sum(present(alive_frames))
                if present(boost) and present(alive_frames) and sum(present(alive_frames)) > 0
                else None
            ),
            "death_causes": dict(sorted(causes.items())),
            "unavailable": [
                name
                for name, values_ in {
                    "executed_boost_eligible_denominator": [],
                    "greedy_boost_eligible_denominator": [],
                    "action_disagreement_with_b8": [],
                    "q_gap_and_scale": [],
                    "td_vs_anchor_loss": [],
                    "schedule_values": [],
                    "per-decision_length_distribution": [],
                }.items()
                if not values_
            ],
        }
        out["episodes"] += len(values)
    if grouped:
        identities = collections.defaultdict(list)
        for row in rows:
            key = "/".join(
                str(row.get(k, "unknown"))
                for k in ("arm", "checkpoint_label", "training_seed", "mix")
            )
            identities[key].append(row)
        out["by_identity_seed_mix"] = {
            key: summarize_rows(values, grouped=False) for key, values in sorted(identities.items())
        }
        out["paired_mass_decomposition"] = paired_decomposition(rows)
        # Pooled legacy summaries are retained only as explicitly descriptive
        # inventory; all scientific identity/mix comparisons use the groups.
        out["pooled_strata_are_descriptive_only"] = True
    metrics = [row["q0_metrics"] for row in rows if isinstance(row.get("q0_metrics"), Mapping)]
    if metrics:
        out["exact_event_metrics"] = aggregate_metrics(metrics)
    return out


def aggregate_metrics(metrics: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum event numerators/denominators before deriving rates; preserve strata."""
    additive = (
        "decision_frames",
        "alive_frames",
        "mass_sum",
        "food_mass_collected",
        "boost_eligible_decisions",
        "boost_executed_frames",
        "boost_length_burned",
        "boost_action_decisions",
        "boost_action_when_eligible",
        "reward_total",
        "death_events",
    )
    result = {key: sum(float(row.get(key, 0)) for row in metrics) for key in additive}
    for name in ("by_length", "by_time", "by_space"):
        keys = sorted({key for row in metrics for key in row.get(name, {})})
        result[name] = {
            key: aggregate_metrics([row[name][key] for row in metrics if key in row.get(name, {})])
            for key in keys
        }
    for name in (
        "policy_diagnostics",
        "reward_decomposition",
        "length_histogram",
        "death_causes",
        "cumulative_mass",
        "counterfactual_training_return",
    ):
        totals = collections.Counter()
        for row in metrics:
            totals.update(row.get(name, {}))
        result[name] = dict(sorted(totals.items()))
    decisions = result["decision_frames"]
    eligible = result["boost_eligible_decisions"]
    probe = result["policy_diagnostics"]
    count = probe.get("probe_decisions", 0)
    result.update(
        episodes=len(metrics),
        food_mass_per_alive_decision=(
            result["food_mass_collected"] / decisions if decisions else None
        ),
        food_mass_per_whole_horizon=result["food_mass_collected"] / (len(metrics) * HORIZON),
        executed_boost_per_eligible=(
            result["boost_action_when_eligible"] / eligible if eligible else None
        ),
        greedy_boost_per_eligible=(
            probe.get("greedy_boost_when_eligible", 0) / probe["policy_boost_eligible_decisions"]
            if probe.get("policy_boost_eligible_decisions")
            else None
        ),
        action_disagreement_with_b8=(
            probe.get("action_disagreements_b8", 0) / count if count else None
        ),
        mean_q_legal_gap=probe.get("q_legal_gap_sum", 0) / count if count else None,
        mean_q_abs_max=probe.get("q_abs_max_sum", 0) / count if count else None,
        mean_centered_q_drift=probe.get("q_centered_drift_l1_sum", 0) / count if count else None,
    )
    return result


def paired_decomposition(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Separate mass and survival loss on identical worlds, never pool policies."""
    bases = {
        (row.get("training_seed"), row.get("mix"), row.get("world_seed")): row
        for row in rows
        if row.get("arm") == "b8"
    }
    groups = collections.defaultdict(list)
    for row in rows:
        if row.get("arm") != "t23":
            continue
        base = bases.get((row.get("training_seed"), row.get("mix"), row.get("world_seed")))
        if base is None:
            continue
        groups[(row.get("training_seed"), row.get("checkpoint_label"), row.get("mix"))].append(
            (row, base)
        )
    result = []
    for (seed, label, mix), pairs in sorted(groups.items()):
        entry = {
            "training_seed": seed,
            "checkpoint_label": label,
            "mix": mix,
            "paired_worlds": len(pairs),
        }
        for key in ("mass_integral", "alive_frame_fraction", "no_death_episode_survival"):
            entry[key + "_delta"] = mean(
                [float(a[key]) - float(b[key]) for a, b in pairs if key in a and key in b]
            )
        # Counterfactual return rankings are descriptive and conditional on the
        # observed trajectory. They never replace full-horizon mass.
        for gamma in ("0.99", "0.998"):
            differences = [
                (
                    a.get("q0_metrics", {}).get("counterfactual_training_return", {}).get(gamma),
                    b.get("q0_metrics", {}).get("counterfactual_training_return", {}).get(gamma),
                )
                for a, b in pairs
            ]
            valid = [float(a) - float(b) for a, b in differences if a is not None and b is not None]
            delta = mean(valid)
            entry["counterfactual_return_delta_gamma_" + gamma] = delta
            mass = entry["mass_integral_delta"]
            entry["return_mass_ranking_agrees_gamma_" + gamma] = (
                (delta > 0) == (mass > 0) if delta is not None and mass is not None else None
            )
        result.append(entry)
    return result


def discover(artifact_root: Path) -> dict[str, Any]:
    """Inventory finals/snapshots and collect saved evaluation rows if present."""
    from research.q0q1_20261017.audit_common import checkpoint_inventory

    evidence = checkpoint_inventory(artifact_root)
    found = {
        seed: {
            label: [row["path"] for row in entries[label] if row["valid"]]
            for label in ("final", "2m", "4m")
        }
        for seed, entries in evidence.items()
    }
    rows: list[Mapping[str, Any]] = []
    for _, payload in iter_json(artifact_root):
        if isinstance(payload.get("rows"), list):
            rows.extend(x for x in payload["rows"] if isinstance(x, Mapping))
        elif "mass_integral" in _record(payload):
            rows.append(payload)
    telemetry = {}
    for seed in T23_SEEDS:
        path = artifact_root / f"seed-{seed}" / "log.jsonl"
        samples = []
        if path.is_file():
            for line in path.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and isinstance(row.get("transitions"), (int, float)):
                    samples.append(
                        {
                            key: row.get(key)
                            for key in (
                                "transitions",
                                "eps",
                                "gamma",
                                "lambda",
                                "td",
                                "anchor",
                                "qmax",
                                "boost_rate",
                                "greedy_boost_rate",
                                "greedy_boost_rate_by_length",
                                "greedy_boost_rate_lt_700",
                            )
                        }
                    )
        telemetry[str(seed)] = {"available": path.is_file(), "path": str(path), "samples": samples}
    return {
        "snapshots": found,
        "checkpoint_evidence": evidence,
        "saved_rows": rows,
        "telemetry": telemetry,
    }


def build_report(artifact_root: Path, b8: Path | None) -> dict[str, Any]:
    discovered = discover(artifact_root)
    snapshot_status = {
        seed: {label: {"available": bool(paths), "paths": paths} for label, paths in values.items()}
        for seed, values in discovered["snapshots"].items()
    }
    b8_info = {"available": bool(b8 and b8.is_file()), "path": str(b8) if b8 else None}
    if b8 and b8.is_file():
        b8_info["sha256"] = sha256_file(b8)
        b8_info["sha256_matches_release"] = b8_info["sha256"] == B8_SHA256
    return {
        "schema": SCHEMA,
        "authority": "Q0 descriptive audit only; no efficacy decision",
        "status": "INCOMPLETE",
        "caps": {
            "development_episodes_max": MAX_EPISODES,
            "horizon": HORIZON,
            "counts_per_seed": MIX_COUNTS,
        },
        "artifact_root": str(artifact_root),
        "b8": b8_info,
        "saved_artifact_availability": snapshot_status,
        "checkpoint_evidence": discovered["checkpoint_evidence"],
        "saved_evidence": summarize_rows(discovered["saved_rows"]),
        "saved_training_telemetry": discovered["telemetry"],
        "development_packet": {
            "status": "NOT_RUN",
            "reason": "pass --run-development only after checkpoints and a fresh Q0 bank are bound",
            "missing_snapshots_stay_missing": True,
        },
    }


def _single(paths: list[str], label: str) -> Path | None:
    """A Q0 packet never guesses among duplicate checkpoint candidates."""
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError(f"ambiguous {label} checkpoint paths: {paths}")
    return Path(paths[0])


def run_development(artifact_root: Path, b8: Path) -> dict[str, Any]:
    """Run at most 480 registered H5000 diagnostic episodes on a fresh Q0 bank."""
    if not b8.is_file() or sha256_file(b8) != B8_SHA256:
        raise ValueError("Q0 development requires the released B8 bytes")
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t23_r0_screen_20261017 import evaluate as t23_eval

    bank = audit_q0_bank.build_world_bank()
    audit = audit_q0_bank.audit_world_bank()
    if not audit["pass"]:
        raise ValueError(
            f"Q0 fresh-bank collision audit failed: {audit['registered_collision_seeds']}"
        )
    deadline = time.monotonic() + 8 * 3600
    context = gid._context(1)
    carrier = context["lookup"][gid.FRP3_S12[1]]
    discovered = discover(artifact_root)["snapshots"]
    if any(not discovered[str(seed)]["final"] for seed in T23_SEEDS):
        raise ValueError("Q0 development requires all five hash-pinned T23 finals")
    available: dict[str, dict[str, Path | None]] = {}
    for seed in T23_SEEDS:
        available[str(seed)] = {
            label: _single(discovered[str(seed)][label], f"seed-{seed}/{label}")
            for label in ("2m", "4m")
        }
    # B8 is intentionally evaluated once per fresh world and reused against both
    # snapshot identities: 5 * 32 * (B8 + 2M + 4M) == 480 maximum.
    rows: list[dict[str, Any]] = []
    from research.q0q1_20261017.audit_b8_init import CoverageReservoir, exact_tie_fixture

    reservoir = CoverageReservoir()

    def observed(world, checkpoint, arm, label, seed):
        if time.monotonic() >= deadline:
            raise TimeoutError("Q0 CPU runtime cap exhausted")
        with PolicyProbe(
            str(b8), reservoir=reservoir if arm == "b8" else None, mix=str(world["mix"])
        ) as probe:
            observer = FrameObserver(horizon=HORIZON, policy_probe=probe, deadline=deadline)
            row = t23_eval.evaluate_world(
                world, checkpoint, carrier, context, observer_factory=lambda _seeds: observer
            )
        metrics = observer.result()
        episode = metrics["episodes"].get(str(int(world["world_seed"])))
        if episode is None or not metrics["complete_horizon"]:
            raise RuntimeError("Q0 frame observer did not complete its H5000 world")
        return {
            "arm": arm,
            "checkpoint_label": label,
            "training_seed": seed,
            **row,
            "q0_metrics": episode,
        }

    for seed in T23_SEEDS:
        worlds = bank["worlds"][str(seed)]
        for world in worlds:
            rows.append(observed(world, b8, "b8", "b8", seed))
        for label, checkpoint in available[str(seed)].items():
            if checkpoint is None:
                continue
            for world in worlds:
                rows.append(observed(world, checkpoint, "t23", label, seed))
    if len(rows) > MAX_EPISODES:
        raise RuntimeError(f"Q0 episode cap exceeded: {len(rows)} > {MAX_EPISODES}")
    expected = MAX_EPISODES
    return {
        "status": "COMPLETE" if len(rows) == expected else "INCOMPLETE",
        "episodes_run": len(rows),
        "fidelity_corpus": {
            "rows": reservoir.rows + [exact_tie_fixture()],
            "candidate_observations_inspected": reservoir.seen,
            "sampling": "seeded bounded reservoir with protected natural coverage witnesses",
            "bank_audit_pass": audit["pass"],
            "additional_episodes": 0,
        },
        "episode_cap": MAX_EPISODES,
        "bank": {
            "schema": audit_q0_bank.SCHEMA,
            "digest": bank["allocation_semantic_sha256"],
            "audit": audit,
        },
        "missing_snapshots_stay_missing": True,
        "rows": rows,
        "summary": summarize_rows(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--b8", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-development", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(args.artifact_root, args.b8)
    if args.run_development:
        if args.b8 is None:
            raise ValueError("--run-development requires --b8")
        report["development_packet"] = run_development(args.artifact_root, args.b8)
        report["status"] = report["development_packet"]["status"]
    write_json(args.out, report)
    print(f"{report['schema']} {report['development_packet']['status']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
