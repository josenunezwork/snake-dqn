"""Operating characteristics of a sequential Phase R plan, by simulation (numpy, one thread).

Data model (per replicate): every cell (primary, control, guard) draws, per seed, ``N`` world
blocks with replacement from the variance pool (resampled real per-world paired deltas, so
skew and heavy tails are kept), then adds the scenario's shifts:

* H5000 mass: ``effect + u_s`` with seed effects ``u_s ~ N(0, tau^2)``;
* MI10: ``mi10_ratio * effect + u10_s``, ``u10`` with correlation 0.7 to ``u`` and SD
  ``mi10_ratio * tau``;
* H5000 survival: ``survival_per_mass * effect`` (+ an optional per-mix harm shift);
* the guard cell (hero - champion on the reference seed) has effect ``effect + guard_gap``.

The vectorised evaluator below implements the same clause semantics as
``src.evaluation.sequential_phase_r`` (a test replays replicates through the pure decision
function and requires identical statuses and stop looks). The fixed design is the same rule
with one look at nominal 0.10 for GO and KILL, no margin and no NO_GO -- the FRP rule as
pre-registered.

``python -m research.sequential_phase_r.simulate --reps 20000 --out <json>`` writes every
scenario x design row with Monte Carlo SEs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from research.sequential_phase_r import example_rules
from src.evaluation.sequential_phase_r import SequentialPhaseRPlan, make_plan
from src.scripts.eval_stats import student_t_isf

HERE = Path(__file__).resolve().parent
POOL = HERE / "variance_pool_20261007.json"
METRIC_INDEX = {"mi5": 0, "surv5": 1, "mi10": 2}
CHUNK = 2000
GUARD_RHO = 0.5
_DF_GRID = np.exp(np.linspace(0.0, math.log(5000.0), 4000))


# --------------------------------------------------------------------------- pool
def load_pool(path: Path = POOL) -> np.ndarray:
    data = json.loads(Path(path).read_text())
    return np.asarray(data["blocks"], dtype=float)  # (B, mixes, metrics)


# --------------------------------------------------------------------------- designs
def fixed_plan(rule: Mapping[str, Any], n_worlds: int) -> Dict[str, Any]:
    """The pre-registered fixed design as a one-look plan (nominal 0.10 GO / KILL)."""
    rule = dict(rule)
    rule["no_go"] = None
    return {
        "name": "fixed",
        "rule": rule,
        "n_worlds": n_worlds,
        "look_sizes": (n_worlds,),
        "go_nominal_p": (0.10,),
        "kill_nominal_p": (0.10,),
        "margin_z": 0.0,
        "protective_margin_z": 0.0,
    }


def sequential_design(plan: SequentialPhaseRPlan, name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "rule": plan.rule,
        "n_worlds": plan.n_worlds,
        "look_sizes": plan.look_sizes,
        "go_nominal_p": plan.go_nominal_p,
        "kill_nominal_p": plan.kill_nominal_p,
        "margin_z": plan.margin_z,
        "protective_margin_z": plan.protective_margin_z,
    }


# --------------------------------------------------------------------------- data
def draw(
    pool: np.ndarray,
    rule: Mapping[str, Any],
    scenario: Mapping[str, Any],
    n_worlds: int,
    reps: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """``cell -> (reps, seeds, N, mixes, 3 metrics)`` per-world paired deltas."""
    tau = math.sqrt(float(scenario.get("tau2", 0.0)))
    ratio = float(scenario.get("mi10_ratio", 1.5))
    per_mass = float(scenario.get("survival_per_mass", 0.00035))
    harm = scenario.get("survival_harm") or {}
    mixes = rule["mixes"]
    effects = {
        "primary": float(scenario["effect"]),
        "control": float(scenario.get("control_effect", 0.0)),
    }
    gap = float(scenario.get("guard_gap", 75.0))
    rho = float(scenario.get("guard_rho", GUARD_RHO))
    out: Dict[str, np.ndarray] = {}
    means: Dict[str, np.ndarray] = {}
    for cell in ("primary", "control"):
        if cell not in rule["cells"]:
            continue
        k = len(rule["cells"][cell]["seeds"])
        x = pool[rng.integers(0, pool.shape[0], size=(reps, k, n_worlds))].copy()
        d = effects[cell]
        u = rng.normal(0.0, tau, size=(reps, k)) if tau > 0 else np.zeros((reps, k))
        z = rng.normal(0.0, tau, size=(reps, k)) if tau > 0 else np.zeros((reps, k))
        u10 = ratio * (0.7 * u + math.sqrt(1.0 - 0.49) * z)
        mean = np.zeros((reps, k, 1, len(mixes), 3))
        mean[..., 0] = (d + u)[:, :, None, None]
        mean[..., 2] = (ratio * d + u10)[:, :, None, None]
        mean[..., 1] = per_mass * d
        if cell == "primary":
            for mix, shift in harm.items():
                mean[..., mixes.index(mix), 1] += float(shift)
        if (
            cell == "control"
            and rule["cells"]["control"]["seeds"] == rule["cells"]["primary"]["seeds"]
        ):
            # both arms are paired with the same incumbent episodes on the same worlds
            x = rho * (out["primary"] - means["primary"]) + math.sqrt(1.0 - rho**2) * x
        out[cell] = x + mean
        means[cell] = mean
    if "guard" in rule["cells"]:
        # hero - champion on the reference seed shares the hero's episodes with the primary
        # cell (hero - incumbent on the same worlds): correlation ``rho`` (FRP-v4: 0.54 for
        # H5000 mass, 0.51 for survival), the hero's own effects and harm, plus the gap.
        ref = rule["cells"]["primary"]["seeds"].index(rule["cells"]["guard"]["seeds"][0])
        primary = out["primary"][:, ref : ref + 1]
        mean = means["primary"][:, ref : ref + 1]
        fresh = pool[rng.integers(0, pool.shape[0], size=(reps, 1, n_worlds))]
        shift = np.zeros((len(mixes), 3))
        shift[:, :] = [gap, per_mass * gap, ratio * gap]
        for mix, extra in (scenario.get("guard_harm") or {}).items():
            shift[mixes.index(mix), 1] += float(extra)
        out["guard"] = (
            mean
            + rho * (primary - mean)
            + math.sqrt(1.0 - rho**2) * fresh
            + shift[None, None, None]
        )
    return out


# --------------------------------------------------------------------------- statistics
class _TTable:
    """Vectorised Student-t upper quantiles at a fixed level (interpolated in log df)."""

    def __init__(self) -> None:
        self._cache: Dict[float, np.ndarray] = {}

    def __call__(self, p: float, df: Any) -> Any:
        if p not in self._cache:
            self._cache[p] = np.array([student_t_isf(p, float(v)) for v in _DF_GRID])
        df = np.asarray(df, dtype=float)
        return np.interp(np.log(np.clip(df, 1.0, _DF_GRID[-1])), np.log(_DF_GRID), self._cache[p])


T = _TTable()


def _seed_effects(x: np.ndarray, metric: str) -> tuple:
    m = METRIC_INDEX[metric]
    n = x.shape[2]
    means = x[..., m].mean(axis=2)  # R, k, M
    var = x[..., m].var(axis=2, ddof=1) / n
    mixes = x.shape[3]
    return means.mean(axis=2), var.sum(axis=2) / mixes**2


def _hk(y: np.ndarray, v: np.ndarray) -> tuple:
    k = y.shape[1]
    w = 1.0 / v
    sw = w.sum(1)
    mfe = (w * y).sum(1) / sw
    q = (w * (y - mfe[:, None]) ** 2).sum(1)
    c = sw - (w**2).sum(1) / sw
    tau2 = np.where(c > 0, np.maximum(0.0, (q - (k - 1)) / np.where(c > 0, c, 1.0)), 0.0)
    ws = 1.0 / (v + tau2[:, None])
    sws = ws.sum(1)
    mu = (ws * y).sum(1) / sws
    qh = np.maximum((ws * (y - mu[:, None]) ** 2).sum(1) / (k - 1), 1.0)
    return mu, np.sqrt(qh / sws), np.full(mu.shape, float(k - 1))


def _stat(stat: Mapping[str, Any], data: Mapping[str, np.ndarray], n: int, mixes: list) -> tuple:
    kind, metric = stat["kind"], stat["metric"]
    if kind == "hk":
        y, v = _seed_effects(data[stat["cell"]][:, :, :n], metric)
        return _hk(y, v)
    if kind == "welch":
        a = _seed_effects(data[stat["cells"][0]][:, :, :n], metric)[0]
        b = _seed_effects(data[stat["cells"][1]][:, :, :n], metric)[0]
        va, vb = a.var(1, ddof=1) / a.shape[1], b.var(1, ddof=1) / b.shape[1]
        se = np.sqrt(va + vb)
        denom = va**2 / (a.shape[1] - 1) + vb**2 / (b.shape[1] - 1)
        df = np.where(denom > 0, (va + vb) ** 2 / np.where(denom > 0, denom, 1.0), np.inf)
        return a.mean(1) - b.mean(1), se, df
    x = data[stat["cell"]][:, :, :n, :, METRIC_INDEX[metric]]
    reps, k = x.shape[0], x.shape[1]
    flat = x.reshape(reps, k * n, x.shape[3])
    if kind == "mix_mean":
        col = flat[:, :, mixes.index(stat["mix"])]
        return col.mean(1), col.std(1, ddof=1) / math.sqrt(k * n), np.full(reps, k * n - 1.0)
    terms = flat.var(1, ddof=1) / (k * n)  # R, M
    m = len(mixes)
    var = terms.sum(1) / m**2
    denom = ((terms / m**2) ** 2 / (k * n - 1)).sum(1)
    df = np.where(denom > 0, var**2 / np.where(denom > 0, denom, 1.0), np.inf)
    return flat.mean(1).mean(1), np.sqrt(var), df


def _compare(value: np.ndarray, op: str, threshold: float, tol: float) -> np.ndarray:
    if op == ">=":
        return value >= threshold - tol
    if op == ">":
        return value > threshold
    if op == "<=":
        return value <= threshold + tol
    return value < threshold


def _point(spec, data, n, mixes, shrink, margin_z, with_point=False):
    est, se, _ = _stat(spec["stat"], data, n, mixes)
    margin = margin_z * se * shrink
    adjusted = est - margin if spec["op"] in (">=", ">") else est + margin
    ok = _compare(adjusted, spec["op"], spec["threshold"], spec["tolerance"])
    if with_point:
        return ok, _compare(est, spec["op"], spec["threshold"], spec["tolerance"])
    return ok


def _bound(est, se, df, p, upper):
    half = np.where(se > 0, T(p, df) * se, 0.0)
    return est + half if upper else est - half


def _feasibility_ratio(clause: Mapping[str, Any]) -> float:
    """``c7`` passes iff ``SD*/MDE* <= R`` (``n_fixed`` is monotone in ``SD*/MDE*``)."""
    cap, infl = clause["cap"], clause["inflation"]
    m = int(math.floor(cap / infl + 1e-9)) + 1
    while m >= 2 and int(math.ceil(infl * m - 1e-9)) > cap:
        m -= 1
    ta = student_t_isf(clause["per_mix_alpha"], m - 1)
    tb = student_t_isf(1.0 - clause["power"], m - 1)
    return math.sqrt(m) / (ta + tb)


def evaluate(
    design: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    flags: Optional[Mapping[str, bool]] = None,
) -> Dict[str, np.ndarray]:
    """Status (string array), stop look and diagnostics for every replicate."""
    rule = design["rule"]
    mixes = list(rule["mixes"])
    flags = dict(flags or {})
    sizes = list(design["look_sizes"])
    n_total = design["n_worlds"]
    reps = next(iter(data.values())).shape[0]
    status = np.full(reps, "", dtype=object)
    stop = np.full(reps, -1)
    efficacy_at_stop = np.zeros(reps, bool)
    efficacy_ever = np.zeros(reps, bool)
    efficacy_nonbinding = np.zeros(reps, bool)  # any look, every stop ignored
    open_ = np.ones(reps, bool)
    statuses = rule["statuses"]
    for look, n in enumerate(sizes):
        final = look == len(sizes) - 1
        shrink = 1.0 - math.sqrt(n / n_total)
        shrink_protective = math.sqrt(1.0 - n / n_total)
        eff_spec = rule["efficacy"]["stat"]
        mu, se, df = _stat(eff_spec, data, n, mixes)
        lb = _bound(mu, se, df, design["go_nominal_p"][look], upper=False)
        eff = lb > rule["efficacy"]["lower_bound_above"]
        clause_pass: Dict[str, np.ndarray] = {}
        point_pass: Dict[str, np.ndarray] = {}
        for clause in rule["go_clauses"]:
            kind = clause["kind"]
            if kind == "point":
                if clause.get("protective"):
                    ok, pt = _point(
                        clause,
                        data,
                        n,
                        mixes,
                        shrink_protective,
                        design["protective_margin_z"],
                        with_point=True,
                    )
                else:
                    ok, pt = _point(
                        clause, data, n, mixes, shrink, design["margin_z"], with_point=True
                    )
                point_pass[clause["name"]] = pt
            elif kind == "positive_seeds":
                y = _seed_effects(data[clause["cell"]][:, :, :n], clause["metric"])[0]
                ok = (y > 0).sum(1) >= clause["min"]
            elif kind == "flag":
                ok = np.full(reps, bool(flags.get(clause["flag"], True)))
            else:
                x = data[eff_spec["cell"]][:, :, :n, :, METRIC_INDEX[eff_spec["metric"]]]
                flat = x.reshape(reps, -1, x.shape[3])
                sd_star = flat.std(1, ddof=1).max(1)
                mde = np.maximum(lb, clause["shrink"] * mu)
                ratio = sd_star / np.where(mde > 0, mde, 1.0)
                ok = (mde > 0) & (ratio <= _feasibility_ratio(clause))
            clause_pass[clause["name"]] = ok
            point_pass.setdefault(clause["name"], ok)
        go = eff.copy()
        for ok in clause_pass.values():
            go &= ok

        def uppers(parts):
            out = np.ones(reps, bool)
            for part in parts:
                est, s, d = _stat(part["stat"], data, n, mixes)
                out &= (
                    _bound(est, s, d, design["kill_nominal_p"][look], upper=True)
                    < part["upper_below"]
                )
            return out

        kill = uppers(rule["kill"]["all_of"])
        no_go = (
            uppers(rule["no_go"]["all_of"])
            if (rule.get("no_go") and not final)
            else np.zeros(reps, bool)
        )
        holds = []
        for outcome in rule["final_outcomes"]:
            ok = np.ones(reps, bool)
            if outcome.get("requires_efficacy"):
                ok &= eff
            for name in outcome["all_of"]:
                ok &= point_pass[name]
            for name in outcome["none_of"]:
                ok &= ~point_pass[name]
            for spec in outcome["extra"]:
                ok &= _point(spec, data, n, mixes, 0.0, 0.0)
            holds.append((outcome["status"], ok))
        blocked = np.zeros(reps, bool)
        for _, ok in holds:
            blocked |= ok
        here = np.full(reps, "", dtype=object)
        if final:
            here[:] = statuses["partial"]
            here[kill] = statuses["kill"]
            for name, ok in reversed(holds):
                here[ok] = name
            here[go] = statuses["go"]
        else:
            here[no_go & ~blocked] = rule["no_go"]["status"] if rule.get("no_go") else ""
            here[kill & ~blocked] = statuses["kill"]
            here[go] = statuses["go"]
        efficacy_ever |= open_ & eff
        efficacy_nonbinding |= eff
        sel = open_ & (here != "")
        status[sel] = here[sel]
        stop[sel] = look
        efficacy_at_stop[sel] = eff[sel]
        open_ &= ~sel
    return {
        "status": status,
        "stop": stop,
        "fraction": np.asarray(sizes)[stop] / n_total,
        "efficacy_at_stop": efficacy_at_stop,
        "efficacy_ever": efficacy_ever,
        "efficacy_nonbinding": efficacy_nonbinding,
    }


# --------------------------------------------------------------------------- scenarios
def scenarios(family: str) -> List[Dict[str, Any]]:
    """Effects -20..+65 (H5000 mass, vs the incumbent), the KILL threshold, and survival-harm
    candidates (+65 with a survival shift in one mix), each at tau^2 = 0 and 500."""
    control = {"control_effect": 7.0} if family == "v5" else {}
    threshold = 20.0 if family == "v4" else 32.0  # v5: H - C contrast exactly +25
    rows = []
    for tau2 in (0.0, 500.0):
        for effect in (-20.0, 0.0, 20.0, 30.0, 40.0, 50.0, 65.0):
            rows.append({"family": family, "effect": effect, "tau2": tau2, **control})
        if family == "v5":
            rows.append(
                {
                    "family": family,
                    "effect": threshold,
                    "tau2": tau2,
                    "label": "kill_threshold",
                    **control,
                }
            )
        for mix, harm in (
            ("scripted", -0.05),
            ("scripted", -0.07),
            ("scripted", -0.10),
            ("frozen", -0.07),
        ):
            rows.append(
                {
                    "family": family,
                    "effect": 65.0,
                    "tau2": tau2,
                    "label": f"harm_{mix}{harm:+.2f}",
                    "survival_harm": {mix: harm},
                    **control,
                }
            )
        rows.append(
            {
                "family": family,
                "effect": 65.0,
                "tau2": tau2,
                "label": "harm_guard_scripted-0.10",
                "guard_harm": {"scripted": -0.10},
                **control,
            }
        )
    return rows


def summarise(result: Mapping[str, np.ndarray], labels: Sequence[str]) -> Dict[str, Any]:
    reps = len(result["status"])

    def rate(mask: np.ndarray) -> Dict[str, float]:
        p = float(np.mean(mask))
        return {"p": p, "mc_se": math.sqrt(max(p * (1 - p), 1e-12) / reps)}

    out = {label: rate(result["status"] == label) for label in labels}
    frac = result["fraction"]
    out["expected_fraction"] = {
        "mean": float(frac.mean()),
        "mc_se": float(frac.std(ddof=1) / math.sqrt(reps)),
    }
    out["saving"] = 1.0 - float(frac.mean())
    out["stop_by_look"] = [
        float(np.mean(result["stop"] == k)) for k in range(int(result["stop"].max()) + 1)
    ]
    out["efficacy_at_stop"] = rate(result["efficacy_at_stop"])
    out["efficacy_ever"] = rate(result["efficacy_ever"])
    out["efficacy_any_look_nonbinding"] = rate(result["efficacy_nonbinding"])
    out["reps"] = reps
    return out


def run(reps: int, seed: int, n_worlds: int = 32, no_go: float = 50.0) -> Dict[str, Any]:
    pool = load_pool()
    rows = []
    for family in ("v4", "v5"):
        maker = example_rules.frp_v4_like if family == "v4" else example_rules.frp_v5_like
        base = make_plan(n_worlds, maker())
        with_no_go = make_plan(n_worlds, maker(no_go=no_go))
        designs = [
            fixed_plan(base.rule, n_worlds),
            sequential_design(base, "sequential"),
            sequential_design(with_no_go, f"sequential+NO_GO({no_go:g})"),
        ]
        labels = [
            base.rule["statuses"]["go"],
            *[o["status"] for o in base.rule["final_outcomes"]],
            base.rule["statuses"]["kill"],
            "NO_GO_EARLY",
            base.rule["statuses"]["partial"],
        ]
        for index, scenario in enumerate(scenarios(family)):
            started = time.time()
            rng = np.random.default_rng([seed, index, 0 if family == "v4" else 1])
            parts: Dict[str, List[Dict[str, np.ndarray]]] = {d["name"]: [] for d in designs}
            for start in range(0, reps, CHUNK):
                data = draw(pool, base.rule, scenario, n_worlds, min(CHUNK, reps - start), rng)
                for design in designs:  # common random numbers across designs
                    parts[design["name"]].append(evaluate(design, data))
            merged = {
                design["name"]: {
                    key: np.concatenate([part[key] for part in parts[design["name"]]])
                    for key in parts[design["name"]][0]
                }
                for design in designs
            }
            fixed = merged["fixed"]
            for design in designs:
                result = merged[design["name"]]
                paired = {}
                for label in (base.rule["statuses"]["go"], base.rule["statuses"]["kill"]):
                    diff = (result["status"] == label).astype(float) - (
                        fixed["status"] == label
                    ).astype(float)
                    paired[label] = {
                        "diff": float(diff.mean()),
                        "paired_se": float(diff.std(ddof=1) / math.sqrt(len(diff))),
                    }
                rows.append(
                    {
                        "scenario": dict(scenario),
                        "design": design["name"],
                        **summarise(result, labels),
                        "vs_fixed": paired,
                    }
                )
            print(
                f"{family} {scenario.get('label', '')} effect {scenario['effect']:+g} tau2 "
                f"{scenario['tau2']:g}: {time.time() - started:.1f}s",
                flush=True,
            )
        rows.append(
            {
                "family": family,
                "plan": {k: v for k, v in base.as_dict().items() if k not in ("rule",)},
                "plan_sha256": base.sha256(),
                "plan_no_go_sha256": with_no_go.sha256(),
            }
        )
    return {
        "schema": "sequential-phase-r-oc/v1",
        "reps": reps,
        "seed": seed,
        "n_worlds": n_worlds,
        "no_go_threshold": no_go,
        "pool": str(POOL.name),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists (create-only)")
    result = run(args.reps, args.seed)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
