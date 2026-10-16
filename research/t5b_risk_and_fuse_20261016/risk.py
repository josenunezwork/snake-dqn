"""Retrospective B8 survival-clause risk diagnostics; never invokes gameplay or gates.

Input contract: terminal H5000 records of the five exact B8 finals and v8 references.
Nullable alive-frame values remain unknown; episode survival never substitutes for them.
The canonical key is (policy identity, mix, world seed); H10000 inputs require H5000 prefix.
Repeated instrumentation enriches a canonical row only when known measurements agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr
from scipy.stats import binom

ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
MIXES = ("scripted", "frozen", "mixed")
REFERENCES = {
    "incumbent": "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723",
    "champion": "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93",
}
PRODUCER_SHA256 = {
    "research/b8_death_census_20261014/census.py": (
        "368d11a31da77424a1ec5a84b95cb1727161564b9e3d72e11f28b4422fad616a"
    ),
    "research/redesign_m2_20261008/dev_probe.py": (
        "b264bcbdff78844e853b0b4f0eca80d854348bfe90f36cc60ca1a093e6f90658"
    ),
    "research/redesign_m2_20261008/ni_spec.py": (
        "8fe85c9964774f1d42dac977a435f025dfcbb878d9d931e1efb787b4f649368b"
    ),
    "research/apex_safety_20260926/dev_screen.py": (
        "50374f4f236ecac208f250eaacfc94d5142cce99b96989dae231744dd30211fe"
    ),
}

AUTHORITY = "Retrospective diagnostic only; reused banks; never gate evidence or GO."


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Pool:
    """Merge nullable observations without ever counting replayed worlds twice."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.sources: dict[str, dict[str, Any]] = {}
        self.duplicates = 0

    def add(
        self,
        policy: str,
        mix: str,
        world: int,
        episode: float,
        alive: float | None,
        mass: float,
        source: Path,
        family: str,
    ) -> None:
        if mix not in MIXES or episode not in (0, 1):
            raise ValueError("invalid terminal episode/mix")
        if alive is not None and (not math.isfinite(alive) or not 0 <= alive <= 1):
            raise ValueError("invalid alive-frame fraction")
        key = (policy, mix, int(world))
        data = dict(
            policy=policy,
            mix=mix,
            world=int(world),
            episode=episode,
            alive=alive,
            mass=mass,
            family=family,
        )
        if key in self.rows:
            old = self.rows[key]
            for field in ("episode", "alive", "mass"):
                if old[field] is None:
                    old[field] = data[field]
                elif data[field] is not None and not math.isclose(
                    old[field], data[field], rel_tol=0, abs_tol=1e-10
                ):
                    raise ValueError(
                        f"conflicting duplicate {key} {field}: {old[field]} != {data[field]}"
                    )
            self.duplicates += 1
        else:
            self.rows[key] = data
        name = str(source)
        if name not in self.sources:
            self.sources[name] = dict(path=name, sha256=sha(source), observations=0)
        self.sources[name]["observations"] += 1


def validate_reference(policy: str, checkpoint: dict[str, Any], engine: dict[str, Any]) -> None:
    """References with payload identities must bind the exact incumbent/champion and v8."""
    if checkpoint.get("sha256") != REFERENCES[policy] or engine.get("veto") != "v8":
        raise ValueError("wrong reference identity/veto")
    if engine.get("sim_engine") != "grid" or engine.get("vector61_forward") != "rowwise":
        raise ValueError("wrong reference engine")


def validate_coverage(pool: Pool) -> None:
    """Freeze the 2026-10-16 source inventory and reject missing reference/paired banks."""
    for seed in range(5):
        policy = f"B8_s{seed}"
        for mix in MIXES:
            m4 = {
                w
                for (p, m, w), r in pool.rows.items()
                if p == policy and m == mix and r["family"] == "M4"
            }
            if len(m4) != 32:
                raise ValueError(f"incomplete M4 candidate bank: {policy}/{mix}")
            for w in m4:
                ref = pool.rows.get(("incumbent", mix, w))
                if ref is None or ref["alive"] is None or ref["family"] != "M4":
                    raise ValueError("missing M4 served reference")
                if seed == 0:
                    ref = pool.rows.get(("champion", mix, w))
                    if ref is None or ref["alive"] is None or ref["family"] != "M4":
                        raise ValueError("missing M4 champion reference")
            r52 = {
                w
                for (p, m, w), r in pool.rows.items()
                if p == policy and m == mix and r["family"] == "round52"
            }
            if len(r52) != (96 if mix == "scripted" else 32):
                raise ValueError(f"incomplete round52 candidate bank: {policy}/{mix}")
            paired = observations(pool, mix, "alive", "incumbent", seed=seed)
            if len(paired) != (64 if mix == "scripted" else 32):
                raise ValueError(f"incomplete paired alive bank: {policy}/{mix}")
            champ = observations(pool, mix, "episode", "champion", seed=seed)
            if len(champ) != (
                64
                if seed == 0 and mix == "scripted"
                else 32 if seed == 0 or mix == "scripted" else 0
            ):
                raise ValueError(f"incomplete champion episode bank: {policy}/{mix}")


def load_pool(root: Path = ART) -> tuple[Pool, dict[str, str]]:
    """Read approved historical producers; validate exact final identities and no veto."""
    pool = Pool()
    repo = Path(__file__).resolve().parents[2]
    for relative, expected in PRODUCER_SHA256.items():
        producer = repo / relative
        if sha(producer) != expected:
            raise ValueError(f"changed identity-bearing historical producer: {relative}")
        pool.sources[str(producer)] = dict(path=str(producer), sha256=expected, observations=0)
    shas = {
        f"B8_s{i}": sha(root / f"redesign-m4b-mac-20261013/B8_s{i}/final.pth") for i in range(5)
    }
    by_sha = {v: k for k, v in shas.items()}

    def read_source(path: Path) -> str:
        raw = path.read_bytes()
        pool.sources[str(path)] = dict(
            path=str(path), sha256=hashlib.sha256(raw).hexdigest(), observations=0
        )
        return raw.decode("utf-8")

    def add_episodes(p: Path, d: dict[str, Any]) -> None:
        if d.get("frames") != 5000:
            raise ValueError(f"non-H5000 census {p}")
        ckpt = d.get("checkpoint")
        policy = by_sha[ckpt["sha256"]] if ckpt else d["arm"]
        if d.get("round_index") != 52:
            raise ValueError(f"wrong census bank {p}")
        if ckpt:
            if (
                d["engine"]["veto"] is not None
                or d["engine"]["ego2s_forward"] != "rowwise"
                or d["engine"]["ego2s_mask"] != "advisory"
            ):
                raise ValueError(f"wrong census candidate selection {p}")
        elif policy not in REFERENCES or d["engine"]["veto"] != "v8(lambda=8.0)":
            raise ValueError(f"wrong census reference {p}")
        # Legacy census lacks a checkpoint field for references; the four pinned producer
        # hashes above bind arm_call and its referenced identity constants prospectively.
        if len(d["episodes"]) != (32 if "b8-death-census" in str(p) else len(d["seeds"])):
            raise ValueError(f"incomplete census episodes {p}")
        for e in d["episodes"]:
            pool.add(
                policy,
                d["mix"],
                e["world_seed"],
                float(e["deaths"] == 0),
                e["survival_fraction"],
                e["mass_integral"],
                p,
                "round52",
            )

    for p in sorted((root / "b8-death-census-20261014/runs").glob("*.json")):
        if p.stem in ("incumbent", "champion") or p.stem in [f"b8_s{i}" for i in range(5)]:
            add_episodes(p, json.loads(read_source(p)))
    # The final look comprises incremental look0/1/2 records. H10000 is never pooled.
    for p in sorted(
        (root / "redesign-m4-phase-r-nv-20261013/shards").glob("look-*/records/*.json")
    ):
        d = json.loads(read_source(p))
        if d["control"] or d["hero"] not in ("M4-selected-noveto", "incumbent", "champion"):
            continue
        policy = (
            by_sha[d["hero_checkpoint"]["sha256"]]
            if d["hero"] == "M4-selected-noveto"
            else d["hero"]
        )
        if policy in REFERENCES:
            validate_reference(policy, d["hero_checkpoint"], d["engine"])
        if policy in shas and (
            d["engine"]["veto"] is not None
            or d["engine"]["ego2s_forward"] != "rowwise"
            or d["engine"]["ego2s_mask"] != "advisory"
        ):
            raise ValueError(f"wrong candidate selection {p}")
        profile = d["record"]["evaluation_profile"]
        if (
            profile["name"] != "promotion-v2-watch-rect"
            or profile["learn"]
            or profile["runtime"]["training"]
            or not profile["runtime"]["hero_terminal"]
        ):
            raise ValueError(f"wrong evaluation profile {p}")
        e = d["prefix_h5000"] if d["horizon"] == 10000 else d["record"]
        if e["denominators"]["scored_frames"] != 5000:
            raise ValueError(f"invalid prefix horizon {p}")
        if not math.isclose(e["survival_fraction"], e["denominators"]["alive_frames"] / 5000):
            raise ValueError(f"invalid explicit denominator {p}")
        pool.add(
            policy,
            d["mix"],
            d["world_seed"],
            float(e["deaths"] == 0),
            e["survival_fraction"],
            e["mass_integral"],
            p,
            "M4",
        )
    paths = list((root / "redesign-m4-mac-probes-20261013").glob("B8_s?__final__none.jsonl"))
    paths += [root / "redesign-m3c-probes-20261008/incumbent__dev__v8.jsonl"]
    paths += list((root / "research-tracks-20261014/ensemble").glob("*B8_s?_K1.jsonl"))
    paths += list((root / "m7-diagnostics-20261014/governor").glob("rows_B8_twin_lane*.jsonl"))
    paths += list((root / "m7-diagnostics-20261014/governor_verify").glob("*/rows_verify.jsonl"))
    for p in sorted(set(paths)):
        for line in read_source(p).splitlines():
            d = json.loads(line)
            if d.get("veto") != "none" and d.get("arm") != "baseline":
                continue
            if "cap" in d and d["cap"] != "off":
                continue
            if "ensemble" in d:
                members = d["ensemble"]["member_sha256"]
                if len(members) != 1 or members[0] not in by_sha:
                    continue
                policy = by_sha[members[0]]
            elif d.get("student_sha256") in by_sha:
                policy = by_sha[d["student_sha256"]]
            elif d.get("arm") == "baseline" and d.get("veto") == "v8":
                policy = "incumbent"
            else:
                continue
            if d.get("round_index") != 52 or d.get("frames", 5000) != 5000:
                raise ValueError(f"unexpected dev contract {p}")
            for w, dead in d["deaths"].items():
                alive = None
                if "diag" in d:
                    per = d["diag"]["per_world"][w]
                    if per["last_frame"] != per["decisions"]:
                        raise ValueError("non-contiguous terminal decision history")
                    alive = (per["decisions"] - dead) / 5000
                elif "per_world" in d:
                    alive = (d["per_world"][w]["decisions"] - dead) / 5000
                pool.add(
                    policy, d["mix"], int(w), float(dead == 0), alive, d["mass"][w], p, "round52"
                )
    for pattern in (
        "m7-diagnostics-20261014/feature_auc/runs/B8_s?.json",
        "m7-diagnostics-20261014/feature_auc_verify/*/runs/B8_s?.json",
    ):
        for p in sorted(root.glob(pattern)):
            add_episodes(p, json.loads(read_source(p)))
    # Content identities are checked after extraction to reject concurrently changed inputs.
    for s in pool.sources.values():
        if sha(Path(s["path"])) != s["sha256"]:
            raise ValueError(f'input changed: {s["path"]}')
    for policy, expected in shas.items():
        if sha(root / f"redesign-m4b-mac-20261013/{policy}/final.pth") != expected:
            raise ValueError(f"checkpoint changed: {policy}")
    validate_coverage(pool)
    return pool, shas


def observations(
    pool: Pool,
    mix: str,
    metric: str,
    reference: str | None = None,
    seed: int | None = None,
    family: str | None = None,
) -> list[tuple[int, int, float]]:
    out = []
    for (policy, m, world), row in sorted(pool.rows.items()):
        if not policy.startswith("B8_s") or m != mix or row[metric] is None:
            continue
        s = int(policy[-1])
        if (seed is not None and s != seed) or (family and row["family"] != family):
            continue
        value = row[metric]
        if reference:
            ref = pool.rows.get((reference, mix, world))
            if ref is None or ref[metric] is None:
                continue
            value -= ref[metric]
        out.append((s, world, value))
    return out


def estimate(obs: list[tuple[int, int, float]], future_per_seed: int) -> dict[str, Any]:
    """Equal-final estimate and source-world clustered sandwich SE; future iid bank SE."""
    seeds = sorted({s for s, _, _ in obs})
    values = {s: np.array([v for t, _, v in obs if t == s]) for s in seeds}
    if not seeds or any(len(v) < 2 for v in values.values()):
        raise ValueError("insufficient seed observations")
    means = {s: float(v.mean()) for s, v in values.items()}
    influence: dict[int, float] = defaultdict(float)
    for s, w, value in obs:
        influence[w] += (value - means[s]) / (len(seeds) * len(values[s]))
    g = len(influence)
    source_se = math.sqrt(g / (g - 1) * sum(v * v for v in influence.values()))
    future_se = math.sqrt(
        sum(float(v.var(ddof=1)) for v in values.values()) / (len(seeds) ** 2 * future_per_seed)
    )
    return dict(
        mean=float(np.mean(list(means.values()))),
        observed_cluster_se=source_se,
        observed_rows=len(obs),
        distinct_worlds=g,
        rows_per_final={str(s): len(v) for s, v in values.items()},
        per_final_means={str(s): v for s, v in means.items()},
        future_worlds=len(seeds) * future_per_seed,
        future_se=future_se,
    )


def fail_probability(mean: float, se: float, threshold: float) -> float:
    if se == 0:
        return float(mean < threshold)
    return float(ndtr((threshold - mean) / se))


def summarize(pool: Pool, draws: int = 200000) -> dict[str, Any]:
    clauses = {}
    for i, mix in enumerate(MIXES):
        n = 96 if mix == "scripted" else 64
        obs = observations(pool, mix, "alive", "incumbent")
        est = estimate(obs, n)
        est.update(
            threshold=-0.03,
            metric="alive-frame candidate minus served",
            failure_probability=fail_probability(est["mean"], est["future_se"], -0.03),
        )
        clauses[f"c2_{mix}"] = est
        obs = observations(pool, mix, "episode")
        est = estimate(obs, n)
        est.update(
            threshold=0.5,
            metric="candidate terminal episode survival",
            failure_probability=fail_probability(est["mean"], est["future_se"], 0.5),
        )
        est["future_se_at_480"] = estimate(obs, 96)["future_se"]
        est["failure_probability_at_480"] = fail_probability(
            est["mean"], est["future_se_at_480"], 0.5
        )
        est["binomial_failure_probability_at_480"] = float(binom.cdf(239, 480, est["mean"]))
        clauses[f"absolute_{mix}"] = est
    for s in range(5):
        est = estimate(observations(pool, "scripted", "alive", "incumbent", seed=s), 96)
        est.update(
            threshold=-0.15,
            metric="scripted alive-frame candidate minus served",
            failure_probability=fail_probability(est["mean"], est["future_se"], -0.15),
        )
        clauses[f"seed_{s}_floor"] = est
    # Joint normal approximation: each final's fresh bank is independent of other finals.
    # All six mix/metric correlations come ONLY from its complete 32-world M4 rows.
    # Scripted uses96; frozen/mixed share first64. Cov(means)=cov(worlds)*min(nj,nk)/(nj*nk).
    rng = np.random.default_rng(20261016)
    future = np.zeros((draws, 5, 6))
    correlations = {}
    future_480 = np.zeros_like(future)
    for s in range(5):
        columns = [
            observations(pool, m, metric, ref, seed=s)
            for metric, ref in (("alive", "incumbent"), ("episode", None))
            for m in MIXES
        ]
        historical = [dict((w, v) for _, w, v in c) for c in columns]
        m4_columns = [
            observations(pool, m, metric, ref, seed=s, family="M4")
            for metric, ref in (("alive", "incumbent"), ("episode", None))
            for m in MIXES
        ]
        complete = [dict((w, v) for _, w, v in c) for c in m4_columns]
        worlds = sorted(set.intersection(*(set(c) for c in complete)))
        if len(worlds) != 32:
            raise ValueError("incomplete M4 cross-mix covariance bank")
        matrix = np.array([[c[w] for c in complete] for w in worlds])
        corr = np.corrcoef(matrix, rowvar=False)
        corr = np.nan_to_num(corr)
        np.fill_diagonal(corr, 1)
        sds = np.array([np.std(list(c.values()), ddof=1) for c in historical])
        ns = np.array([96, 64, 64, 96, 64, 64])
        covariance = corr * np.outer(sds, sds) * np.minimum.outer(ns, ns) / np.outer(ns, ns)
        means = np.array([np.mean(list(c.values())) for c in historical])
        future[:, s] = rng.multivariate_normal(means, covariance, size=draws, check_valid="raise")
        ns480 = np.array([96, 64, 64, 96, 96, 96])
        covariance480 = (
            corr * np.outer(sds, sds) * np.minimum.outer(ns480, ns480) / np.outer(ns480, ns480)
        )
        future_480[:, s] = rng.multivariate_normal(
            means, covariance480, size=draws, check_valid="raise"
        )
        correlations[str(s)] = dict(
            source="M4 only", complete_worlds=len(worlds), matrix=corr.tolist()
        )
    pooled = future.mean(axis=1)
    fails = np.column_stack(
        [pooled[:, i] < -0.03 for i in range(3)]
        + [pooled[:, i + 3] < 0.5 for i in range(3)]
        + [future[:, s, 0] < -0.15 for s in range(5)]
    )
    pooled480 = future_480.mean(axis=1)
    fails480 = np.column_stack(
        [pooled480[:, i] < -0.03 for i in range(3)]
        + [pooled480[:, i + 3] < 0.5 for i in range(3)]
        + [future_480[:, s, 0] < -0.15 for s in range(5)]
    )
    p = float(np.any(fails, axis=1).mean())
    ps = [c["failure_probability"] for c in clauses.values()]
    combined = dict(
        joint_plugin_probability=p,
        joint_probability_absolute_480_all_mixes=float(np.any(fails480, axis=1).mean()),
        monte_carlo_se=math.sqrt(p * (1 - p) / draws),
        draws=draws,
        rng_seed=20261016,
        marginal_independence_product=1 - math.prod(1 - v for v in ps),
        dependence_free_bounds=[max(ps), min(1.0, sum(ps))],
        modeled_scope="11 requested survival clauses only; unchanged efficacy clauses unmodeled",
        covariance_correlations=correlations,
    )
    tier2 = estimate(observations(pool, "scripted", "episode", "champion"), 96)
    tier2.update(
        threshold=-0.05,
        retrospective_status="PASS" if tier2["mean"] >= -0.05 else "FAIL",
        metric="episode survival candidate minus champion_a5 + v8",
        failure_probability=fail_probability(tier2["mean"], tier2["future_se"], -0.05),
    )
    sensitivity = {}
    for family in ("round52", "M4"):
        for ref in ("incumbent", "champion"):
            for metric in ("alive", "episode"):
                obs = observations(pool, "scripted", metric, ref, family=family)
                if obs:
                    sensitivity[f"{family}_{ref}_{metric}"] = estimate(obs, 96)
    return dict(
        authority=AUTHORITY,
        clauses=clauses,
        combined=combined,
        tier2=tier2,
        sensitivity=sensitivity,
        methodology=dict(
            source_se=(
                "equal-final means; cluster sandwich by unique world seed, "
                "including shared round52 reference repeats"
            ),
            forecast=(
                "plug-in normal future-bank sampling noise conditional on observed "
                "means/variances; no parameter-uncertainty integration"
            ),
            covariance=(
                "within-final cross-metric/mix correlations from32 complete M4 worlds; "
                "independent future banks across finals; scripted96, "
                "frozen/mixed64 nested worlds"
            ),
            absolute_480=(
                "480-world requested sensitivity also shown for frozen/mixed; "
                "amendment uses320 there"
            ),
            missing=(
                "later64 round52 scripted B8 worlds lack served/champion outcomes; "
                "dev alive-frame references unavailable for frozen/mixed"
            ),
            exclusions=(
                "T3/T4e source trajectories have outcome-dependent stopping/quota "
                "selection and are not complete conventional probes; governor-on, "
                "ensemble K>1, non-final checkpoints, controls and H10000 excluded"
            ),
            limitation=(
                "seen-bank selection and world-distribution uncertainty not quantified; "
                "normal plug-in tails are approximate"
            ),
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=ART)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=200000)
    args = ap.parse_args()
    pool, shas = load_pool(args.artifacts)
    result = summarize(pool, args.draws)
    result["checkpoints"] = shas
    result["inventory"] = dict(
        canonical_rows=len(pool.rows),
        duplicate_observations=pool.duplicates,
        sources=sorted(pool.sources.values(), key=lambda s: s["path"]),
        counts_by_policy_mix={
            f"{p}/{m}": n for (p, m), n in sorted(Counter((p, m) for p, m, _ in pool.rows).items())
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out.with_name("risk-pool.json").write_text(
        json.dumps(list(pool.rows.values()), indent=1, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k in ("authority", "clauses", "combined", "tier2")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
