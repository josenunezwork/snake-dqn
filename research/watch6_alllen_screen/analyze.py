"""Mac-only exploration analysis: paired CIs across B, L2' (L>=600), and L>=1 + decision rule."""

from __future__ import annotations

import argparse
import json
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t

from research.watch6_alllen_screen import protocol as p
from tools.ops import auto_retry, ready_queue
from tools.ops.package_spec import publish_json, verify_immutable
from web.backend.watch_replay import B8_SHA, FRP3_SHA

COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _paired_ci(diff: np.ndarray, confidence: float = 0.95) -> list[float]:
    n = int(diff.size)
    if n < 2:
        raise ValueError("paired t-interval requires n >= 2")
    alpha = 1.0 - confidence
    crit = float(t.ppf(1.0 - alpha / 2.0, n - 1))
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n))
    return [mean - crit * se, mean + crit * se]


def _fieller_ratio_ci95(
    candidate: np.ndarray, baseline: np.ndarray
) -> tuple[float | None, list[float] | None]:
    """Ratio of pooled means (candidate - baseline) / baseline with Fieller 95% CI."""
    n = int(candidate.size)
    crit = float(t.ppf(0.975, n - 1))
    d = candidate - baseline
    base_mean = float(baseline.mean())
    ratio = float(d.mean() / base_mean) if base_mean > 0 else None
    c = crit**2 / n
    a = float(base_mean**2 - c * baseline.var(ddof=1))
    b = float(-2.0 * d.mean() * base_mean + 2.0 * c * np.cov(d, baseline, ddof=1)[0, 1])
    q = float(d.mean() ** 2 - c * d.var(ddof=1))
    disc = b * b - 4.0 * a * q
    tol = 64.0 * np.finfo(float).eps * max(abs(b * b), abs(4.0 * a * q), 1.0)
    if -tol <= disc < 0:
        disc = 0.0
    bounds = (
        sorted([float((-b - np.sqrt(disc)) / (2.0 * a)), float((-b + np.sqrt(disc)) / (2.0 * a))])
        if a > 0 and disc >= 0
        else None
    )
    return ratio, bounds


def _compare_pair(
    rows: list[dict[str, Any]], candidate_arm: str, baseline_arm: str, horizon: int
) -> dict[str, Any]:
    cand_mass = np.array(
        [row["arms"][candidate_arm]["mass_sum"] / horizon for row in rows], dtype=np.float64
    )
    base_mass = np.array(
        [row["arms"][baseline_arm]["mass_sum"] / horizon for row in rows], dtype=np.float64
    )
    cand_deaths_1k = np.array(
        [row["arms"][candidate_arm]["deaths"] * 1000.0 / horizon for row in rows], dtype=np.float64
    )
    base_deaths_1k = np.array(
        [row["arms"][baseline_arm]["deaths"] * 1000.0 / horizon for row in rows], dtype=np.float64
    )
    mass_diff = cand_mass - base_mass
    death_diff = cand_deaths_1k - base_deaths_1k
    rel_mass, fieller_ci95 = _fieller_ratio_ci95(cand_mass, base_mass)
    total_base_deaths = sum(int(row["arms"][baseline_arm]["deaths"]) for row in rows)
    total_cand_deaths = sum(int(row["arms"][candidate_arm]["deaths"]) for row in rows)
    return {
        "candidate": candidate_arm,
        "baseline": baseline_arm,
        "n": len(rows),
        "df": len(rows) - 1,
        "mass_delta_mean": float(mass_diff.mean()),
        "mass_delta_sd": float(mass_diff.std(ddof=1)),
        "mass_delta_se": float(mass_diff.std(ddof=1) / np.sqrt(len(rows))),
        "mass_delta_ci95": _paired_ci(mass_diff, 0.95),
        "mass_delta_ci90": _paired_ci(mass_diff, 0.90),
        "mass_relative_difference": rel_mass,
        "mass_relative_ci95_fieller": fieller_ci95,
        "deaths_per_1k_delta_mean": float(death_diff.mean()),
        "deaths_per_1k_delta_sd": float(death_diff.std(ddof=1)),
        "deaths_per_1k_delta_se": float(death_diff.std(ddof=1) / np.sqrt(len(rows))),
        "deaths_per_1k_delta_ci95": _paired_ci(death_diff, 0.95),
        "deaths_per_1k_delta_ci90": _paired_ci(death_diff, 0.90),
        "death_reduction_fraction": (
            1.0 - total_cand_deaths / total_base_deaths if total_base_deaths > 0 else None
        ),
        "wins_mass": int(np.sum(mass_diff > 0)),
        "ties_mass": int(np.sum(mass_diff == 0)),
        "losses_mass": int(np.sum(mass_diff < 0)),
        "wins_fewer_deaths": int(np.sum(death_diff < 0)),
        "ties_deaths": int(np.sum(death_diff == 0)),
        "losses_more_deaths": int(np.sum(death_diff > 0)),
    }


def _arm_aggregate(rows: list[dict[str, Any]], arm: str, horizon: int) -> dict[str, Any]:
    mass = np.array([row["arms"][arm]["mass_sum"] / horizon for row in rows], dtype=np.float64)
    deaths_1k = np.array(
        [row["arms"][arm]["deaths"] * 1000.0 / horizon for row in rows], dtype=np.float64
    )
    roots = np.array([row["arms"][arm]["search_roots"] for row in rows], dtype=np.float64)
    roots_lt600 = np.array(
        [row["arms"][arm]["search_roots_lt600"] for row in rows], dtype=np.float64
    )
    roots_ge600 = np.array(
        [row["arms"][arm]["search_roots_ge600"] for row in rows], dtype=np.float64
    )
    interventions = np.array([row["arms"][arm]["interventions"] for row in rows], dtype=np.float64)
    interventions_lt600 = np.array(
        [row["arms"][arm]["interventions_lt600"] for row in rows], dtype=np.float64
    )
    interventions_ge600 = np.array(
        [row["arms"][arm]["interventions_ge600"] for row in rows], dtype=np.float64
    )
    branches = np.array([row["arms"][arm]["branches_evaluated"] for row in rows], dtype=np.float64)
    short_deaths = sum(int(row["arms"][arm]["short_body_deaths_lt600"]) for row in rows)
    long_deaths = sum(int(row["arms"][arm]["long_body_deaths_ge600"]) for row in rows)
    late_deaths = sum(int(row["arms"][arm]["late_large_body_deaths_ge800"]) for row in rows)
    return {
        "arm": arm,
        "label": p.ARM_SPECS[arm]["label"],
        "min_length": p.ARM_SPECS[arm]["min_length"],
        "mass_mean": float(mass.mean()),
        "mass_sd": float(mass.std(ddof=1)),
        "mass_ci95": _paired_ci(mass, 0.95),
        "deaths_per_1k_mean": float(deaths_1k.mean()),
        "deaths_per_1k_sd": float(deaths_1k.std(ddof=1)),
        "deaths_per_1k_ci95": _paired_ci(deaths_1k, 0.95),
        "total_deaths": sum(int(row["arms"][arm]["deaths"]) for row in rows),
        "short_body_deaths_lt600": short_deaths,
        "long_body_deaths_ge600": long_deaths,
        "late_large_body_deaths_ge800": late_deaths,
        "mean_max_length": float(np.mean([row["arms"][arm]["max_length"] for row in rows])),
        "mean_alive_fraction": float(
            np.mean([row["arms"][arm]["alive_frames"] / horizon for row in rows])
        ),
        "mean_search_roots": float(roots.mean()),
        "mean_search_roots_lt600": float(roots_lt600.mean()),
        "mean_search_roots_ge600": float(roots_ge600.mean()),
        "mean_interventions": float(interventions.mean()),
        "mean_interventions_lt600": float(interventions_lt600.mean()),
        "mean_interventions_ge600": float(interventions_ge600.mean()),
        "mean_branches_evaluated": float(branches.mean()),
        "override_rate": (
            float(interventions.sum() / roots.sum()) if float(roots.sum()) > 0 else 0.0
        ),
        "override_rate_lt600": (
            float(interventions_lt600.sum() / roots_lt600.sum())
            if float(roots_lt600.sum()) > 0
            else 0.0
        ),
        "override_rate_ge600": (
            float(interventions_ge600.sum() / roots_ge600.sum())
            if float(roots_ge600.sum()) > 0
            else 0.0
        ),
    }


def analyze(
    rows: list[dict[str, Any]],
    *,
    expected_namespace: str = p.NAMESPACE,
    expected_count: int = p.COUNT,
    expected_horizon: int = p.HORIZON,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Validate complete paired union and compute 3-arm comparisons + adoption verdict."""
    if len(rows) != expected_count or {r["allocation"]["index"] for r in rows} != set(
        range(expected_count)
    ):
        raise ValueError("incomplete or duplicate unit union; refusing readout")
    rows = sorted(rows, key=lambda r: r["allocation"]["index"])
    expected_alloc = p.allocation(
        expected_namespace, expected_count, expected_horizon, audit_live_mirror=False
    )
    source_commits: set[str] = set()
    contract_hashes: set[str] = set()

    for row, exp_row in zip(rows, expected_alloc["rows"]):
        commit = row.get("source_commit")
        if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
            raise ValueError("unit missing 40-char hex source_commit")
        expected_cid = p.contract_identity({**expected_alloc, "commit": commit})
        if (
            row.get("schema") != p.UNIT_SCHEMA
            or row.get("namespace") != expected_namespace
            or row["allocation"] != {**exp_row, "horizon": expected_horizon}
            or row.get("contract_identity") != expected_cid
            or row.get("checkpoint_shas") != {"hero": B8_SHA, "opponents": FRP3_SHA}
            or (not allow_fixture and row.get("fixture") is not False)
        ):
            raise ValueError("foreign namespace, fixture, or corrupted unit allocation")
        if set(row["arms"].keys()) != set(p.ARMS):
            raise ValueError("missing required arms in unit")
        for arm in p.ARMS:
            if row["arms"][arm]["frames"] != expected_horizon:
                raise ValueError(f"incomplete horizon in arm {arm}")
        source_commits.add(commit)
        contract_hashes.add(expected_cid["contract_sha256"])

    incremental = _compare_pair(rows, "exact_stop_alllen", "l2_prime_600", expected_horizon)
    l2_vs_bare = _compare_pair(rows, "l2_prime_600", "bare", expected_horizon)
    alllen_vs_bare = _compare_pair(rows, "exact_stop_alllen", "bare", expected_horizon)

    mass_bar_met = incremental["mass_delta_mean"] >= p.ADOPT_MASS_DELTA_MIN
    deaths_bar_met = incremental["deaths_per_1k_delta_mean"] <= p.ADOPT_DEATH_DELTA_PER_1K_MAX
    adopt_alllen = bool(mass_bar_met and deaths_bar_met)
    sorted_commits = sorted(source_commits)
    sorted_hashes = sorted(contract_hashes)

    return {
        "schema": "watch6-alllen-screen-readout/v1",
        "lane": p.LANE,
        "authority": (
            "exploration screen only (OPUS_META §3 row 2); informs Watch replay bank "
            "controller selection; not release or confirmation evidence"
        ),
        "namespace": expected_namespace,
        "source_commit": sorted_commits[-1],
        "source_commits": sorted_commits,
        "contract_sha256": sorted_hashes[-1],
        "contract_sha256s": sorted_hashes,
        "n": expected_count,
        "horizon": expected_horizon,
        "decision_rule": {
            "description": p.DECISION_RULE,
            "incremental_mass_delta_min": p.ADOPT_MASS_DELTA_MIN,
            "incremental_deaths_per_1k_delta_max": p.ADOPT_DEATH_DELTA_PER_1K_MAX,
            "observed_incremental_mass_delta": incremental["mass_delta_mean"],
            "observed_incremental_deaths_per_1k_delta": incremental["deaths_per_1k_delta_mean"],
            "mass_bar_met": mass_bar_met,
            "deaths_bar_met": deaths_bar_met,
            "adopt_l_ge_1_for_new_banks": adopt_alllen,
            "decision": (
                "ADOPT_EXACT_STOP_L_GE_1_FOR_NEW_BANKS"
                if adopt_alllen
                else "KEEP_SERVED_L2_PRIME_L_GE_600"
            ),
        },
        "arms": {arm: _arm_aggregate(rows, arm, expected_horizon) for arm in p.ARMS},
        "comparisons": {
            "exact_stop_alllen_vs_l2_prime_600": incremental,
            "l2_prime_600_vs_bare": l2_vs_bare,
            "exact_stop_alllen_vs_bare": alllen_vs_bare,
        },
        "per_session": [
            {
                "index": row["allocation"]["index"],
                "seed": row["allocation"]["seed"],
                "source_commit": row["source_commit"],
                "arm_execution_order": row["arm_execution_order"],
                "bare_mass": row["arms"]["bare"]["mass_integral"],
                "l2_prime_600_mass": row["arms"]["l2_prime_600"]["mass_integral"],
                "exact_stop_alllen_mass": row["arms"]["exact_stop_alllen"]["mass_integral"],
                "bare_deaths_per_1k": row["arms"]["bare"]["deaths_per_1k"],
                "l2_prime_600_deaths_per_1k": row["arms"]["l2_prime_600"]["deaths_per_1k"],
                "exact_stop_alllen_deaths_per_1k": row["arms"]["exact_stop_alllen"][
                    "deaths_per_1k"
                ],
            }
            for row in rows
        ],
    }


def resolve_queue_unit_dirs(ops: Path, queue_ids: list[str]) -> list[Path]:
    """Resolve queue IDs across auto-retry and -retry-1 chains to their exports/units dirs."""
    queue = auto_retry.read(ops / "ready-queue.json")
    dirs: list[Path] = []
    for qid in queue_ids:
        row = ready_queue.latest(queue, qid)
        name = row["argv"][1]
        internal = ops / "manifests" / f"{name}-retry-1.json"
        manifest_path = internal if internal.exists() else Path(row["manifest"])
        manifest = auto_retry.read(manifest_path)
        dirs.append(Path(manifest["output_dir"]) / "exports" / "units")
    return dirs


def collect_unit_files(roots_or_files: list[Path]) -> list[Path]:
    """Expand export directories (completed `unit-*.json` with `.complete`) or explicit files."""
    files: list[Path] = []
    for item in roots_or_files:
        if item.is_dir():
            for marker in sorted(item.glob("unit-*.json.complete")):
                files.append(item / marker.name[: -len(".complete")])
        elif item.exists():
            files.append(item)
    if not files:
        raise ValueError("no completed unit-*.json files found in supplied paths")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", nargs="*", type=Path, default=[])
    parser.add_argument("--ops", type=Path)
    parser.add_argument("--queue-ids", nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("analysis readout is Mac-only")
    if bool(args.queue_ids) and args.ops is None:
        parser.error("--queue-ids requires --ops")
    search_paths = [path.resolve() for path in args.units]
    if args.queue_ids and args.ops is not None:
        search_paths.extend(resolve_queue_unit_dirs(args.ops.resolve(), list(args.queue_ids)))
    if not search_paths:
        parser.error("supply --units and/or --ops with --queue-ids")
    unit_paths = collect_unit_files(search_paths)
    for path in unit_paths:
        verify_immutable(path)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in unit_paths]
    result = analyze(rows)
    publish_json(args.out.resolve(), result)
    print(
        json.dumps(
            {
                "decision": result["decision_rule"]["decision"],
                "n": result["n"],
                "source_commits": result["source_commits"],
                "incremental_L1_minus_L600": {
                    "mass_delta": result["comparisons"]["exact_stop_alllen_vs_l2_prime_600"][
                        "mass_delta_mean"
                    ],
                    "mass_ci95": result["comparisons"]["exact_stop_alllen_vs_l2_prime_600"][
                        "mass_delta_ci95"
                    ],
                    "deaths_per_1k_delta": result["comparisons"][
                        "exact_stop_alllen_vs_l2_prime_600"
                    ]["deaths_per_1k_delta_mean"],
                    "deaths_per_1k_ci95": result["comparisons"][
                        "exact_stop_alllen_vs_l2_prime_600"
                    ]["deaths_per_1k_delta_ci95"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
