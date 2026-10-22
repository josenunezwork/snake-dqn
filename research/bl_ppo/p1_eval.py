"""Fresh one-look P1 endpoint evaluation against bare B8 s3.

This module owns only the P1 evaluation boundary.  It makes a deterministic,
fresh H5000 bank (128 paired worlds in each mix), preserves every completed
world as a create-only export, and applies the fixed nomination rule.  It is
not a strict gate or a serving decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from statistics import mean, variance
from typing import Any, Mapping

from research.bl_ppo.exports import digest, export_json, verify_export
from research.published_ledger import bind_published_ledger_check, require_published_ledger

ROOT = Path(__file__).resolve().parents[2]
MIXES = ("scripted", "frozen", "mixed")
COUNT_PER_MIX = 128
HORIZON = 5000
NAMESPACE = "bl-ppo-p1-eval-20261021/v1"
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"


def canonical_sha(value: Any) -> str:
    """Hash finite JSON semantics in a portable, deterministic form."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _finite(
    value: Any, name: str, *, lower: float | None = None, upper: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    number = float(value)
    if (lower is not None and number < lower) or (upper is not None and number > upper):
        raise ValueError(f"{name} out of range")
    return number


def materialize(round_index: int = 0) -> dict[str, Any]:
    """Allocate P1's fresh paired endpoint worlds without evaluating gameplay."""
    if type(round_index) is not int or round_index < 0:
        raise ValueError("round_index must be a nonnegative integer")
    from research.t34_round0.bank import _rows

    namespace = f"{NAMESPACE}/round-{round_index}"
    value = {
        "schema": "bl-ppo-p1-eval-bank/v1",
        "namespace": namespace,
        "round_index": round_index,
        "counts": {mix: COUNT_PER_MIX for mix in MIXES},
        "profile": {
            "horizon": HORIZON,
            "hero_mode": "terminal",
            "hero_slot": 0,
            "hero_veto": "none",
            "frozen_veto": "v8",
            "mechanics_version": 1,
            "engine": "GridBatchSim",
            "comparator": "bare B8 s3",
            "one_look": True,
        },
        "rows": _rows(namespace + "/endpoint", {mix: COUNT_PER_MIX for mix in MIXES}, HORIZON),
    }
    seeds = [int(row["world_seed"]) for row in value["rows"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("whole P1 allocation has an internal seed collision")
    value["allocation_sha256"] = canonical_sha(value)
    return value


def validate_bank(bank: Mapping[str, Any]) -> None:
    """Refuse any row, namespace, roster, count, or horizon substitution."""
    if not isinstance(bank, dict) or bank != materialize(bank.get("round_index")):
        raise ValueError("bank differs from frozen deterministic P1 allocation")


def audit_bank(
    bank: Mapping[str, Any],
    repo: Path = ROOT,
    *,
    burned_inventories: tuple[Path, ...] = (),
    historical_audit: Any = None,
) -> dict[str, Any]:
    """Audit fresh identities with the established safe-pruned historical scanner."""
    validate_bank(bank)
    repo = Path(repo).resolve()
    candidates = {int(row["world_seed"]) for row in bank["rows"]}
    published, collisions = [], set()
    for path in sorted((repo / "research").rglob("*SEED_AUDIT*.json")):
        if path.is_symlink():
            # Existing same-directory aliases are covered by their regular
            # target.  Never resolve a link that could escape into artifacts.
            if (
                os.readlink(path) == "SEED_AUDIT.json"
                and path.with_name("SEED_AUDIT.json").is_file()
                and not path.with_name("SEED_AUDIT.json").is_symlink()
            ):
                continue
            raise ValueError("unsafe published seed inventory path")
        if "t34-round0-study-01" in path.parts:
            raise ValueError("unsafe published seed inventory path")
        value = json.loads(path.read_text(encoding="utf-8"))
        concrete = _inventory_seeds(value)
        collisions.update(candidates & concrete)
        published.append(
            {
                "path": str(path.relative_to(repo)),
                "sha256": digest(path),
                "concrete_seed_count": len(concrete),
            }
        )
    if not published:
        raise ValueError("no published seed inventory available for collision audit")
    burned = []
    for path in burned_inventories:
        path = Path(path).resolve(strict=True)
        if path.is_symlink() or not path.is_file():
            raise ValueError("unsafe burned seed inventory")
        concrete = _inventory_seeds(json.loads(path.read_text(encoding="utf-8")))
        collisions.update(candidates & concrete)
        if not concrete:
            raise ValueError("burned seed inventory contains no concrete seeds")
        burned.append({"path": str(path), "sha256": digest(path), "seed_count": len(concrete)})
    if not burned:
        raise ValueError("burned training/Y3/monitor seed inventories are required")
    if historical_audit is None:
        historical_audit = _historical_audit
    history = historical_audit(sorted(candidates), repo=repo)
    if history.get("collisions"):
        raise ValueError("P1 bank collides with safe-pruned historical inventory")
    if collisions:
        raise ValueError(f"P1 bank collides with published identities: {sorted(collisions)}")
    return {
        "schema": "bl-ppo-p1-eval-seed-audit/v1",
        "pass": True,
        "bank_semantic_sha256": bank["allocation_sha256"],
        "candidate_count": len(candidates),
        "collisions": [],
        "published_inventories": published,
        "burned_seed_inventories": burned,
        "historical_audit": history,
        "scope": "published research inventories plus safe-pruned historical text audit",
        "excluded": ["snake-dqn-artifacts", "t34-round0-study-01"],
        "limitations": [
            "No claim about unrecorded runs, opaque binaries, or unpublished artifact inventories.",
            "The sealed live Round-0 artifact directory was not opened or scanned.",
        ],
    }


def _historical_audit(candidates: list[int], *, repo: Path) -> dict[str, Any]:
    """Reuse the established safe-pruned scanner, including FRP exclusions."""
    from research.bl_ppo import bank as screen_bank
    from research.t34_round0d import bank as prior

    allocation = {
        "rows": [{"world_seed": seed} for seed in candidates],
        "excluded_preflight_rows": [],
        "excluded_a5d_rows": [],
    }
    record = prior.audit(
        allocation,
        scan=lambda seeds: screen_bank._scan_history(
            seeds,
            repo=repo,
            artifacts=repo.parent / "snake-dqn-artifacts",
            excluded_artifact_studies=frozenset({"bl-e2", "bl-p1"}),
        ),
    )
    return {
        "schema": record["schema"],
        "status": record["status"],
        "collisions": record["collisions"],
        "historical_frp_exclusions": record["historical_frp_exclusions"],
        "published_bank_count": len(record["explicit_banks"]),
        "excluded_live_study_roots": record["excluded_live_study_roots"],
    }


def _inventory_seeds(value: Any) -> set[int]:
    """Extract literal seed values from a published JSON inventory recursively."""
    found: set[int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"seed", "world_seed"} and type(item) is int:
                found.add(item)
            elif (
                key in {"seeds", "world_seeds", "episode_seeds"}
                and isinstance(item, list)
                and all(type(x) is int for x in item)
            ):
                found.update(item)
            else:
                found.update(_inventory_seeds(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_inventory_seeds(item))
    return found


def check_audit(bank: Mapping[str, Any], audit: Mapping[str, Any]) -> None:
    validate_bank(bank)
    if (
        audit.get("schema") != "bl-ppo-p1-eval-seed-audit/v1"
        or audit.get("pass") is not True
        or audit.get("bank_semantic_sha256") != bank["allocation_sha256"]
        or audit.get("candidate_count") != 3 * COUNT_PER_MIX
        or audit.get("collisions") != []
        or not audit.get("published_inventories")
        or not audit.get("burned_seed_inventories")
        or audit.get("historical_audit", {}).get("status") != "CLEAR"
        or audit.get("historical_audit", {}).get("collisions") != []
        or len(audit.get("historical_audit", {}).get("historical_frp_exclusions", [])) < 7
        or audit.get("excluded") != ["snake-dqn-artifacts", "t34-round0-study-01"]
    ):
        raise ValueError("missing, stale, or incomplete P1 collision audit")


def paired_equal_mix_bounds(
    groups: Mapping[str, list[float]], alpha: float = 0.10
) -> dict[str, Any]:
    """One-sided equal-mix Welch bound; the paired world is the only unit."""
    from src.scripts.eval_stats import student_t_isf

    if set(groups) != set(MIXES) or any(len(groups[mix]) != COUNT_PER_MIX for mix in MIXES):
        raise ValueError("P1 requires exactly 128 paired worlds per mix")
    if any(not math.isfinite(value) for values in groups.values() for value in values):
        raise ValueError("nonfinite paired delta")
    terms = [variance(groups[mix]) / COUNT_PER_MIX / 9 for mix in MIXES]
    var = sum(terms)
    denominator = sum(term * term / (COUNT_PER_MIX - 1) for term in terms)
    df = var * var / denominator if denominator else None
    estimate = mean(mean(groups[mix]) for mix in MIXES)
    se = math.sqrt(var)
    critical = student_t_isf(alpha, df) if df is not None else 0.0
    return {
        "estimate": estimate,
        "se": se,
        "df": df,
        "alpha_one_sided": alpha,
        "lower": estimate - critical * se,
        "upper": estimate + critical * se,
        "unit": "paired world",
        "mix_weight": 1 / 3,
        "n_by_mix": {mix: COUNT_PER_MIX for mix in MIXES},
    }


def _single_lower(values: list[float], alpha: float = 0.05) -> dict[str, float]:
    from src.scripts.eval_stats import student_t_isf

    if len(values) != COUNT_PER_MIX:
        raise ValueError("P1 survival guard requires exactly 128 worlds per mix")
    se = math.sqrt(variance(values) / len(values))
    return {
        "estimate": mean(values),
        "lower": mean(values) - student_t_isf(alpha, len(values) - 1) * se,
        "alpha_one_sided": alpha,
    }


def _validate_trace_metrics(metrics: Mapping[str, Any]) -> None:
    """Reconcile retained terminal trace bytes to compact endpoint metrics."""
    trace = metrics.get("trace")
    if not isinstance(trace, list) or not trace or len(trace) > HORIZON:
        raise ValueError("missing or oversized retained endpoint trace")
    if metrics.get("trace_sha256") != canonical_sha(trace):
        raise ValueError("endpoint trace hash does not bind retained trace bytes")
    if [item.get("frame") for item in trace] != list(range(len(trace))):
        raise ValueError("endpoint trace frames are not contiguous")
    alive = []
    masses = []
    for item in trace:
        if not isinstance(item, dict) or type(item.get("alive")) is not bool:
            raise ValueError("endpoint trace lacks alive state")
        mass = _finite(item.get("mass"), "trace mass", lower=0)
        if not item["alive"] and mass != 0:
            raise ValueError("dead trace frame has nonzero mass")
        alive.append(item["alive"])
        masses.append(mass)
    death = next((index for index, value in enumerate(alive) if not value), None)
    if death is not None and any(alive[death:]):
        raise ValueError("terminal trace resurrects after death")
    if len(trace) < HORIZON and alive[-1]:
        raise ValueError("short trace did not terminate in death")
    if not math.isclose(sum(masses) / HORIZON, _finite(metrics.get("mass_integral"), "mass")):
        raise ValueError("trace mass does not reconcile with mass integral")
    if not math.isclose(
        sum(alive) / HORIZON, _finite(metrics.get("alive_frame_fraction"), "alive")
    ):
        raise ValueError("trace alive frames do not reconcile")
    if metrics.get("no_death_episode") is not (len(trace) == HORIZON and all(alive)):
        raise ValueError("trace no-death indicator does not reconcile")
    actions = [item.get("action") for item in trace]
    if any(type(action) is not int or not 0 <= action <= 5 for action in actions):
        raise ValueError("endpoint trace action is outside the six-action space")
    if metrics.get("hero_decisions") != len(trace):
        raise ValueError("trace decision count does not reconcile")
    if metrics.get("boost_decisions") != sum(action >= 3 for action in actions):
        raise ValueError("trace boost count does not reconcile")


def _validate_endpoint_controls(
    bank: Mapping[str, Any], checkpoints: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    """Check immutable endpoint and parity controls before any gameplay begins."""
    endpoint = result.get("endpoint_receipt")
    endpoint_sha = result.get("endpoint_receipt_sha256")
    if not isinstance(endpoint, dict) or (
        endpoint.get("schema") != "bl-ppo-training-status/v1"
        or endpoint.get("arm") != "F"
        or endpoint.get("p1_runtime") is not True
        or endpoint.get("complete") is not True
        or endpoint.get("status") != "COMPLETE"
        or endpoint.get("stop_reason") != "endpoint"
        or endpoint.get("target_transitions") != 100_000_000
        or endpoint.get("pause_at") != 100_000_000
        or endpoint.get("critic_qualified") is not True
        or endpoint.get("checkpoint_sha256") != checkpoints["F"]
    ):
        raise ValueError("missing or unbound complete 100M F endpoint receipt")
    if not isinstance(endpoint_sha, str) or len(endpoint_sha) != 64:
        raise ValueError("100M endpoint receipt lacks immutable export digest")
    parity = result.get("e13_parity")
    parity_sha = result.get("e13_parity_sha256")
    if not isinstance(parity, dict) or (
        parity.get("schema") != "bl-p1-e13-parity/v1"
        or parity.get("pass") is not True
        or parity.get("complete") is not True
        or parity.get("checkpoint_sha256") != checkpoints
        or parity.get("bank_semantic_sha256") != bank["allocation_sha256"]
    ):
        raise ValueError("missing or unbound complete E13 parity evidence")
    if not isinstance(parity_sha, str) or len(parity_sha) != 64:
        raise ValueError("E13 parity lacks immutable export digest")


def _validate_results(
    bank: Mapping[str, Any], audit: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    check_audit(bank, audit)
    if result.get("schema") != "bl-ppo-p1-eval-results/v1":
        raise ValueError("unrecognized P1 evaluation result")
    if result.get("complete") is not True or result.get("cap_expired") is not False:
        raise ValueError("incomplete run or expired evaluation cap")
    if result.get("bank_semantic_sha256") != bank["allocation_sha256"]:
        raise ValueError("results do not bind the immutable fresh bank")
    checkpoints = result.get("checkpoint_sha256")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {"F", "B8"}:
        raise ValueError("result must bind exactly F and B8 endpoints")
    if checkpoints["B8"] != B8_SHA256:
        raise ValueError("wrong bare B8 s3 comparator")
    if not isinstance(checkpoints["F"], str) or len(checkpoints["F"]) != 64:
        raise ValueError("missing frozen F endpoint hash")
    _validate_endpoint_controls(bank, checkpoints, result)
    rows = result.get("rows")
    expected = {row["world_id"]: row for row in bank["rows"]}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("incomplete P1 paired world bank")
    if {row.get("world_id") for row in rows} != set(expected):
        raise ValueError("duplicate or substituted P1 world identities")
    for row in rows:
        world = expected[row["world_id"]]
        if any(row.get(key) != world[key] for key in ("world_seed", "mix", "horizon")):
            raise ValueError("world semantics differ from fresh bank")
        if row.get("roster_sha256") != canonical_sha(world["opponents"]):
            raise ValueError("opponent roster differs from fresh bank")
        controllers = row.get("controllers")
        if not isinstance(controllers, dict) or set(controllers) != {"F", "B8"}:
            raise ValueError("paired world must contain F and B8")
        for arm, metrics in controllers.items():
            if metrics.get("checkpoint_sha256") != checkpoints[arm]:
                raise ValueError("endpoint checkpoint substitution")
            if metrics.get("frames_stepped") != HORIZON:
                raise ValueError("incorrect H5000 terminal accounting")
            _finite(metrics.get("mass_integral"), "mass_integral", lower=0)
            _finite(metrics.get("alive_frame_fraction"), "alive_frame_fraction", lower=0, upper=1)
            if type(metrics.get("no_death_episode")) is not bool:
                raise ValueError("no_death_episode must be boolean")
            decisions = metrics.get("hero_decisions")
            boosts = metrics.get("boost_decisions")
            if (
                type(decisions) is not int
                or type(boosts) is not int
                or not 1 <= decisions <= HORIZON
            ):
                raise ValueError("invalid hero decision count")
            if not 0 <= boosts <= decisions:
                raise ValueError("invalid boost decision count")
            _validate_trace_metrics(metrics)


def analyze(
    bank: Mapping[str, Any], audit: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the registered P1 one-look endpoint rule with fail-closed precedence."""
    base = {
        "schema": "bl-ppo-p1-eval-decision/v1",
        "registered_verdict_eligible": False,
        "release_authority": False,
        "bank_semantic_sha256": bank.get("allocation_sha256"),
    }
    try:
        _validate_results(bank, audit, result)
    except (ValueError, KeyError, TypeError) as exc:
        return {**base, "status": "INCOMPLETE", "reasons": [str(exc)]}
    rows = result["rows"]
    groups = {
        mix: [
            float(row["controllers"]["F"]["mass_integral"])
            - float(row["controllers"]["B8"]["mass_integral"])
            for row in rows
            if row["mix"] == mix
        ]
        for mix in MIXES
    }
    mass = paired_equal_mix_bounds(groups)
    alive_f = {
        mix: [
            float(row["controllers"]["F"]["alive_frame_fraction"])
            for row in rows
            if row["mix"] == mix
        ]
        for mix in MIXES
    }
    alive_b8 = {
        mix: [
            float(row["controllers"]["B8"]["alive_frame_fraction"])
            for row in rows
            if row["mix"] == mix
        ]
        for mix in MIXES
    }
    survival_deltas = {mix: [a - b for a, b in zip(alive_f[mix], alive_b8[mix])] for mix in MIXES}
    survival = {
        "pooled": paired_equal_mix_bounds(survival_deltas, alpha=0.05),
        "by_mix": {mix: _single_lower(survival_deltas[mix]) for mix in MIXES},
        "candidate_alive_fraction": {mix: mean(alive_f[mix]) for mix in MIXES},
    }
    survival["pass"] = (
        survival["pooled"]["lower"] > -0.05
        and all(item["lower"] > -0.075 for item in survival["by_mix"].values())
        and all(value >= 0.30 for value in survival["candidate_alive_fraction"].values())
    )
    scripted_no_death = mean(
        int(row["controllers"]["F"]["no_death_episode"])
        - int(row["controllers"]["B8"]["no_death_episode"])
        for row in rows
        if row["mix"] == "scripted"
    )
    boost_by_mix = {
        mix: sum(row["controllers"]["F"]["boost_decisions"] for row in rows if row["mix"] == mix)
        / sum(row["controllers"]["F"]["hero_decisions"] for row in rows if row["mix"] == mix)
        for mix in MIXES
    }
    boost = mean(boost_by_mix.values())
    clauses = {
        "survival_band_v2": survival["pass"],
        "delivered_boost_in_005_10": 0.005 <= boost <= 0.10,
        "scripted_no_death_point_ge_zero": scripted_no_death >= 0,
        "equal_mix_pooled_point_ge_40": mass["estimate"] >= 40,
        "one_sided_LB90_strictly_positive": mass["lower"] > 0,
    }
    status = (
        "ARTIFACT_REJECTED_AT_BAR"
        if mass["upper"] < 40
        else "NOMINATE_MAC_STRICT" if all(clauses.values()) else "INCONCLUSIVE"
    )
    return {
        **base,
        "status": status,
        "mass_gain": mass,
        "survival_band_v2": survival,
        "scripted_no_death_point": scripted_no_death,
        "delivered_boost": {"equal_mix": boost, "by_mix": boost_by_mix, "range": [0.005, 0.10]},
        "clauses": clauses,
        "one_look": True,
    }


def evaluate_world(
    world: Mapping[str, Any], f_checkpoint: Path, b8_checkpoint: Path
) -> dict[str, Any]:
    """Evaluate one paired H5000 P1 world using the existing bare-policy driver."""
    from research.bl_ppo import screen

    candidate = screen.evaluate_world(dict(world), f_checkpoint, "F")
    baseline = screen.evaluate_world(dict(world), b8_checkpoint, "B8")
    return {
        **{key: world[key] for key in ("world_id", "world_seed", "mix", "horizon")},
        "roster_sha256": canonical_sha(world["opponents"]),
        "controllers": {"F": candidate, "B8": baseline},
    }


def _evaluate_pair(item: tuple[dict[str, Any], str, str]) -> dict[str, Any]:
    """Evaluate both paired arms in one worker, preserving shared world placement."""
    world, f_checkpoint, b8_checkpoint = item
    return evaluate_world(world, Path(f_checkpoint), Path(b8_checkpoint))


def evaluate_pairs(
    worlds: list[dict[str, Any]],
    f_checkpoint: Path,
    b8_checkpoint: Path,
    *,
    workers: int = 12,
    evaluator: Any = None,
) -> list[dict[str, Any]]:
    """Deterministically assemble paired worlds after one same-worker pair per root."""
    if type(workers) is not int or not 1 <= workers <= 12:
        raise ValueError("P1 evaluation workers must be in [1, 12]")
    items = [(world, str(f_checkpoint), str(b8_checkpoint)) for world in worlds]
    if evaluator is not None:
        # Tests inject a lightweight evaluator; threads preserve deterministic
        # input/output order without requiring a picklable test closure.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(evaluator, items))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_evaluate_pair, items))


def execute(
    bank: Mapping[str, Any],
    audit: Mapping[str, Any],
    f_checkpoint: Path,
    b8_checkpoint: Path,
    endpoint_receipt: Mapping[str, Any],
    endpoint_receipt_sha256: str,
    e13_parity: Mapping[str, Any],
    e13_parity_sha256: str,
    out: Path,
    *,
    workers: int = 12,
) -> dict[str, Any]:
    """Resume only immutable complete world exports, then seal a P1 endpoint result."""
    check_audit(bank, audit)
    if digest(b8_checkpoint) != B8_SHA256:
        raise ValueError("P1 evaluation requires the released bare B8 s3 checkpoint")
    out = Path(out)
    result = {
        "schema": "bl-ppo-p1-eval-results/v1",
        "complete": True,
        "cap_expired": False,
        "bank_semantic_sha256": bank["allocation_sha256"],
        "checkpoint_sha256": {"F": digest(f_checkpoint), "B8": digest(b8_checkpoint)},
        "endpoint_receipt": dict(endpoint_receipt),
        "endpoint_receipt_sha256": endpoint_receipt_sha256,
        "e13_parity": dict(e13_parity),
        "e13_parity_sha256": e13_parity_sha256,
        "rows": [],
    }
    # Refuse unbound endpoint/parity inputs before spending any evaluation time.
    _validate_endpoint_controls(bank, result["checkpoint_sha256"], result)
    loaded: dict[int, dict[str, Any]] = {}
    missing: list[tuple[int, dict[str, Any]]] = []
    for index, world in enumerate(bank["rows"]):
        path = out / "worlds" / f"world-{index:04d}.json"
        if path.exists():
            verify_export(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("bank_semantic_sha256") != bank["allocation_sha256"]:
                raise ValueError("stale P1 world export")
            loaded[index] = saved.get("row")
        else:
            missing.append((index, world))
    items = [(world, str(f_checkpoint), str(b8_checkpoint)) for _, world in missing]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Executor.map yields in input order.  Every available pair is sealed
        # immediately, so a later fault leaves a valid resumable prefix.
        for (index, _world), row in zip(missing, pool.map(_evaluate_pair, items)):
            path = out / "worlds" / f"world-{index:04d}.json"
            if path.exists():
                raise FileExistsError(f"concurrent P1 world export appeared: {path}")
            export_json(path, {"bank_semantic_sha256": bank["allocation_sha256"], "row": row})
            loaded[index] = row
    for index, world in enumerate(bank["rows"]):
        row = loaded[index]
        result["rows"].append(row)
    _validate_results(bank, audit, result)
    export_json(out / "results.json", result)
    return result


def _read_complete(path: Path) -> Any:
    verify_export(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_complete_bound(path: Path) -> tuple[Any, str]:
    """Read a complete JSON export and retain its immutable payload digest."""
    return _read_complete(path), verify_export(path)


def main(argv: list[str] | None = None) -> int:
    """Prepare, run, or analyze immutable P1 evaluation artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare-bank", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    actions.add_argument("--analyze", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--f-checkpoint", type=Path)
    parser.add_argument("--b8-checkpoint", type=Path)
    parser.add_argument("--endpoint-receipt", type=Path)
    parser.add_argument("--e13-parity", type=Path)
    parser.add_argument("--burned-inventory", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument(
        "--authoring-repo",
        type=Path,
        help="clean checkout that owns the published look ledger; required for --analyze",
    )
    parser.add_argument(
        "--ledger", help="repository-relative published look ledger; required for --analyze"
    )
    args = parser.parse_args(argv)
    if args.prepare_bank:
        bank = materialize(args.round)
        audit = audit_bank(bank, burned_inventories=tuple(args.burned_inventory))
        export_json(args.out / "bank.json", bank)
        export_json(args.out / "seed_audit.json", audit)
        return 0
    if not args.bank or not args.audit:
        parser.error("--evaluate and --analyze require --bank and --audit")
    if args.analyze:
        if not args.results:
            parser.error("--analyze requires --results")
        if args.authoring_repo is None or args.ledger is None:
            parser.error("--analyze is a claim and requires --authoring-repo and --ledger")
        # This happens before result bytes are opened: a Colab runtime can compute
        # pairs, but only the published authoring checkout may consume the look.
        proof = require_published_ledger(args.authoring_repo, args.ledger)
        bank, audit = _read_complete(args.bank), _read_complete(args.audit)
        decision = bind_published_ledger_check(
            analyze(bank, audit, _read_complete(args.results)), proof
        )
        export_json(args.out, decision)
        return 0
    bank, audit = _read_complete(args.bank), _read_complete(args.audit)
    if args.evaluate:
        if not all((args.f_checkpoint, args.b8_checkpoint, args.endpoint_receipt, args.e13_parity)):
            parser.error("--evaluate requires F/B8 checkpoints, endpoint receipt, and E13 parity")
        endpoint, endpoint_sha = _read_complete_bound(args.endpoint_receipt)
        parity, parity_sha = _read_complete_bound(args.e13_parity)
        execute(
            bank,
            audit,
            args.f_checkpoint,
            args.b8_checkpoint,
            endpoint,
            endpoint_sha,
            parity,
            parity_sha,
            args.out,
            workers=args.workers,
        )
        return 0
    raise AssertionError("unreachable action")


if __name__ == "__main__":
    raise SystemExit(main())
