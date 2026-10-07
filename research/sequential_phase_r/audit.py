"""Independent audit of a sequential Phase R root (standard library only; imports nothing from
the repository).

``python -I audit.py --root <seq root> [--record-dirs <shard dir> ...] [--out <json>]``
(exit 0 PASS, 1 FAIL or UNCLOSED). It recomputes, in its own code:

* the plan: the plan sha256, look sizes, information fractions and the GO / KILL nominal levels
  (its own Simpson-grid recursion of the spending functions, within ``BOUNDARY_TOLERANCE``);
* every look receipt: the chain (previous receipt sha256, plan sha256, look order, nothing
  after a STOP / HALT), every record file's sha256, the record key set (exactly the planned
  heroes x seeds x mixes x the bank's first ``n_k`` worlds), the per-world deltas from the raw
  records, and the decision (HK / Welch / mix means / stratified means, interim margins,
  positive seeds, flags, feasibility, KILL / NO_GO bounds, final outcomes, precedence) with its
  own Student-t; status and action must agree with the receipt, numbers within
  ``REL_TOLERANCE``;
* prefix integrity: no decision record in ``--record-dirs`` beyond the stopping look;
* closure: a root whose last receipt says CONTINUE is ``UNCLOSED`` (never PASS).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

AUDIT_SCHEMA = "sequential-phase-r-audit/v1"
PLAN_SCHEMA = "sequential-phase-r-plan/v1"
RECEIPT_SCHEMA = "sequential-phase-r-look/v1"
METHOD = "sequential-phase-r-obf-hk-v1"
BOUNDARY_TOLERANCE = 2e-4
REL_TOLERANCE = 1e-6
GRID_POINTS = 401
GRID_SD = 10.0
_N = NormalDist()


class AuditError(Exception):
    pass


# ----------------------------------------------------------------------------- helpers
def canonical_sha(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc


def close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=REL_TOLERANCE, abs_tol=1e-9)


# ----------------------------------------------------------------------------- numerics
def spend(t: float, alpha: float, kind: str) -> float:
    if t <= 0.0:
        return 0.0
    if kind == "obf":
        return 2.0 * _N.cdf(-_N.inv_cdf(1.0 - alpha / 2.0) / math.sqrt(t))
    if kind == "pocock":
        return alpha * math.log(1.0 + (math.e - 1.0) * t)
    if kind.startswith("hsd:"):
        g = float(kind[4:])
        return alpha * (1.0 - math.exp(-g * t)) / (1.0 - math.exp(-g))
    raise AuditError(f"unknown spending {kind!r}")


def _simpson(h: float, m: int) -> List[float]:
    w = [2.0 if i % 2 == 0 else 4.0 for i in range(m)]
    w[0] = w[-1] = 1.0
    return [x * h / 3.0 for x in w]


@lru_cache(maxsize=16)
def boundaries(fractions: Tuple[float, ...], alpha: float, kind: str) -> Tuple[float, ...]:
    """z boundaries whose null first-crossing probabilities equal the spending increments."""
    spent = [spend(t, alpha, kind) for t in fractions]
    increments = [b - a for a, b in zip([0.0] + spent[:-1], spent)]
    out: List[float] = []
    grid: List[float] = []
    mass: List[float] = []
    previous = 0.0
    for k, t in enumerate(fractions):
        root_t, root_step = math.sqrt(t), math.sqrt(t - previous)

        def crossing(c: float) -> float:
            if k == 0:
                return _N.cdf(-c)
            upper = c * root_t
            return sum(m * _N.cdf(-(upper - s) / root_step) for s, m in zip(grid, mass))

        low, high = -10.0, 40.0
        for _ in range(200):
            mid = 0.5 * (low + high)
            if crossing(mid) > increments[k]:
                low = mid
            else:
                high = mid
        c = 0.5 * (low + high)
        out.append(c)
        top = c * root_t
        bottom = min(-GRID_SD * root_t, top - 1e-9)
        h = (top - bottom) / (GRID_POINTS - 1)
        new_grid = [bottom + i * h for i in range(GRID_POINTS)]
        weights = _simpson(h, GRID_POINTS)
        if k == 0:
            density = [
                math.exp(-0.5 * (s / root_t) ** 2) / (root_t * math.sqrt(2 * math.pi))
                for s in new_grid
            ]
        else:
            norm = 1.0 / (root_step * math.sqrt(2 * math.pi))
            density = [
                norm
                * sum(m * math.exp(-0.5 * ((s - g) / root_step) ** 2) for g, m in zip(grid, mass))
                for s in new_grid
            ]
        grid, mass = new_grid, [w * d for w, d in zip(weights, density)]
        previous = t
    return tuple(out)


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 2000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(front) * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


def t_isf(p: float, df: Optional[float]) -> float:
    if df is None or math.isinf(df):
        return -_N.inv_cdf(p)
    low, high = 0.0, 1.0
    while t_sf(high, df) > p:
        high *= 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if t_sf(mid, df) > p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def mean(v: Sequence[float]) -> float:
    return math.fsum(v) / len(v)


def var(v: Sequence[float]) -> float:
    m = mean(v)
    return math.fsum((x - m) ** 2 for x in v) / (len(v) - 1)


# ----------------------------------------------------------------------------- statistics
def seed_effects(deltas, cell: str, metric: str, rule) -> Tuple[List[float], List[float]]:
    mixes = rule["mixes"]
    e, v = [], []
    for seed in rule["cells"][cell]["seeds"]:
        d = deltas[cell][metric][seed]
        e.append(mean([mean(d[m]) for m in mixes]))
        v.append(math.fsum(var(d[m]) / len(d[m]) for m in mixes) / len(mixes) ** 2)
    return e, v


def hk(y: Sequence[float], v: Sequence[float]) -> Dict[str, float]:
    if len(y) < 2 or any(x <= 0 for x in v):
        raise AuditError("HK needs >= 2 seeds with positive variances")
    k = len(y)
    w = [1.0 / x for x in v]
    sw = math.fsum(w)
    mfe = math.fsum(a * b for a, b in zip(w, y)) / sw
    q = math.fsum(a * (b - mfe) ** 2 for a, b in zip(w, y))
    c = sw - math.fsum(a * a for a in w) / sw
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    ws = [1.0 / (x + tau2) for x in v]
    sws = math.fsum(ws)
    mu = math.fsum(a * b for a, b in zip(ws, y)) / sws
    qh = max(math.fsum(a * (b - mu) ** 2 for a, b in zip(ws, y)) / (k - 1), 1.0)
    return {"estimate": mu, "se": math.sqrt(qh / sws), "df": float(k - 1)}


def stat(spec, deltas, rule) -> Dict[str, Any]:
    kind, metric = spec["kind"], spec["metric"]
    mixes = rule["mixes"]
    if kind == "hk":
        return hk(*seed_effects(deltas, spec["cell"], metric, rule))
    if kind == "welch":
        a = seed_effects(deltas, spec["cells"][0], metric, rule)[0]
        b = seed_effects(deltas, spec["cells"][1], metric, rule)[0]
        va, vb = var(a) / len(a), var(b) / len(b)
        denom = (va**2 / (len(a) - 1) if va > 0 else 0.0) + (
            vb**2 / (len(b) - 1) if vb > 0 else 0.0
        )
        return {
            "estimate": mean(a) - mean(b),
            "se": math.sqrt(va + vb),
            "df": (va + vb) ** 2 / denom if denom > 0 else None,
        }
    seeds = rule["cells"][spec["cell"]]["seeds"]
    if kind == "mix_mean":
        vals = [x for s in seeds for x in deltas[spec["cell"]][metric][s][spec["mix"]]]
        return {
            "estimate": mean(vals),
            "se": math.sqrt(var(vals) / len(vals)),
            "df": len(vals) - 1.0,
        }
    by = {m: [x for s in seeds for x in deltas[spec["cell"]][metric][s][m]] for m in mixes}
    terms = [var(by[m]) / len(by[m]) for m in mixes]
    m_ = len(mixes)
    variance = math.fsum(terms) / m_**2
    denom = math.fsum((t / m_**2) ** 2 / (len(by[m]) - 1) for t, m in zip(terms, mixes))
    return {
        "estimate": mean([mean(by[m]) for m in mixes]),
        "se": math.sqrt(variance),
        "df": variance**2 / denom if denom > 0 else None,
    }


def bound(value, p: float, upper: bool) -> float:
    if value["se"] == 0.0:
        return value["estimate"]
    half = t_isf(p, value["df"]) * value["se"]
    return value["estimate"] + half if upper else value["estimate"] - half


def compare(x: float, op: str, thr: float, tol: float) -> bool:
    return {
        ">=": x >= thr - tol,
        ">": x > thr,
        "<=": x <= thr + tol,
        "<": x < thr,
    }[op]


def point(spec, deltas, rule, factor: float, z: float) -> Dict[str, Any]:
    v = stat(spec["stat"], deltas, rule)
    margin = z * v["se"] * factor
    adjusted = v["estimate"] - margin if spec["op"] in (">=", ">") else v["estimate"] + margin
    return {
        "estimate": v["estimate"],
        "adjusted": adjusted,
        "pass": compare(adjusted, spec["op"], spec["threshold"], spec["tolerance"]),
    }


def n_need(n: int, sd: float, mde: float, a: float, power: float) -> float:
    return ((t_isf(a, n - 1) + t_isf(1.0 - power, n - 1)) * sd / mde) ** 2


def n_fixed(sd: float, mde: float, a: float, power: float) -> Optional[int]:
    if mde <= 0 or not math.isfinite(mde):
        return None
    if sd == 0.0 or 2 >= n_need(2, sd, mde, a, power):
        return 2
    lo, hi = 2, 4
    while hi < n_need(hi, sd, mde, a, power):
        lo, hi = hi, hi * 2
        if hi > 1_000_000:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid >= n_need(mid, sd, mde, a, power):
            hi = mid
        else:
            lo = mid
    return hi


def decide(plan: Mapping[str, Any], look: int, deltas, flags) -> Dict[str, Any]:
    """The audit's own reading of the rule at one look (prefixes already cut)."""
    rule = plan["rule"]
    final = look == len(plan["look_sizes"]) - 1
    t = plan["look_sizes"][look] / plan["n_worlds"]
    shrink, shrink_p = 1.0 - math.sqrt(t), math.sqrt(1.0 - t)
    go_p, kill_p = plan["go_nominal_p"][look], plan["kill_nominal_p"][look]
    eff_spec = rule["efficacy"]["stat"]
    eff = stat(eff_spec, deltas, rule)
    lb = bound(eff, go_p, upper=False)
    eff_pass = lb > rule["efficacy"]["lower_bound_above"]
    clauses: Dict[str, bool] = {}
    points: Dict[str, bool] = {}
    for c in rule["go_clauses"]:
        if c["kind"] == "point":
            z = plan["protective_margin_z"] if c.get("protective") else plan["margin_z"]
            factor = shrink_p if c.get("protective") else shrink
            clauses[c["name"]] = point(c, deltas, rule, factor, z)["pass"]
            points[c["name"]] = point(c, deltas, rule, 0.0, 0.0)["pass"]
        elif c["kind"] == "positive_seeds":
            e = seed_effects(deltas, c["cell"], c["metric"], rule)[0]
            clauses[c["name"]] = sum(1 for x in e if x > 0) >= c["min"]
        elif c["kind"] == "flag":
            if not isinstance(flags.get(c["flag"]), bool):
                raise AuditError(f"flag {c['flag']} missing from the receipt")
            clauses[c["name"]] = flags[c["flag"]]
        else:
            seeds = rule["cells"][eff_spec["cell"]]["seeds"]
            sd_star = max(
                math.sqrt(
                    var(
                        [
                            x
                            for s in seeds
                            for x in deltas[eff_spec["cell"]][eff_spec["metric"]][s][m]
                        ]
                    )
                )
                for m in rule["mixes"]
            )
            mde = max(lb, c["shrink"] * eff["estimate"])
            nf = n_fixed(sd_star, mde, c["per_mix_alpha"], c["power"])
            clauses[c["name"]] = (
                nf is not None and math.ceil(c["inflation"] * nf - 1e-9) <= c["cap"]
            )

    for name, ok in clauses.items():
        points.setdefault(name, ok)

    def uppers(parts) -> Tuple[bool, List[float]]:
        ubs = [bound(stat(p["stat"], deltas, rule), kill_p, upper=True) for p in parts]
        return all(u < p["upper_below"] for u, p in zip(ubs, parts)), ubs

    kill, kill_ubs = uppers(rule["kill"]["all_of"])
    no_go = False
    if rule.get("no_go") and not final:
        no_go, _ = uppers(rule["no_go"]["all_of"])
    holding = []
    for o in rule["final_outcomes"]:
        ok = (eff_pass or not o.get("requires_efficacy")) and all(points[n] for n in o["all_of"])
        ok = ok and not any(points[n] for n in o["none_of"])
        ok = ok and all(point(e, deltas, rule, 0.0, 0.0)["pass"] for e in o["extra"])
        if ok:
            holding.append(o["status"])
    go = eff_pass and all(clauses.values())
    st = rule["statuses"]
    if go:
        status, action = st["go"], "STOP"
    elif final:
        status = holding[0] if holding else (st["kill"] if kill else st["partial"])
        action = "STOP"
    elif kill and not holding:
        status, action = st["kill"], "STOP"
    elif no_go and not holding:
        status, action = rule["no_go"]["status"], "STOP"
    else:
        status, action = "CONTINUE", "CONTINUE"
    return {
        "status": status,
        "action": action,
        "efficacy_estimate": eff["estimate"],
        "efficacy_lower_bound": lb,
        "kill_upper_bounds": kill_ubs,
        "clauses": clauses,
    }


# ----------------------------------------------------------------------------- the audit
def metric_value(record: Mapping[str, Any], metric: str) -> float:
    if record.get("prefix_h5000") is None:
        raise AuditError("a decision record without a prefix block")
    src = record["prefix_h5000"] if metric in ("mi5", "surv5") else record["record"]
    return float(src["mass_integral" if metric.startswith("mi") else "survival_fraction"])


def required(rule) -> Dict[str, List[str]]:
    stats = [rule["efficacy"]["stat"]]
    stats += [c["stat"] for c in rule["go_clauses"] if c["kind"] == "point"]
    stats += [p["stat"] for p in rule["kill"]["all_of"]]
    if rule.get("no_go"):
        stats += [p["stat"] for p in rule["no_go"]["all_of"]]
    for o in rule["final_outcomes"]:
        stats += [e["stat"] for e in o["extra"]]
    need: Dict[str, set] = {}
    for s in stats:
        for cell in s.get("cells") or [s["cell"]]:
            need.setdefault(cell, set()).add(s["metric"])
    for c in rule["go_clauses"]:
        if c["kind"] == "positive_seeds":
            need.setdefault(c["cell"], set()).add(c["metric"])
    if rule.get("candidate"):
        need.setdefault(rule["candidate"]["cell"], set()).add(rule["candidate"]["metric"])
    return {c: sorted(m) for c, m in need.items()}


def expected_keys(plan, banks, look: int) -> set:
    rule = plan["rule"]
    size = plan["look_sizes"][look]
    keys = set()
    for cell in required(rule):
        spec = rule["cells"][cell]
        for seed in spec["seeds"]:
            for mix in rule["mixes"]:
                for world in banks[str(seed)][:size]:
                    keys.add((spec["hero"], int(seed), mix, int(world)))
                    keys.add((spec["baseline"], int(seed), mix, int(world)))
    return keys


def audit(root: Path, record_dirs: Sequence[Path] = ()) -> Dict[str, Any]:
    problems: List[str] = []
    plan_file = load(Path(root) / "plan.json")
    if plan_file.get("schema") != PLAN_SCHEMA:
        problems.append("plan schema")
    plan = plan_file["plan"]
    if canonical_sha(plan) != plan_file.get("plan_sha256"):
        problems.append("plan sha256 does not match its content")
    if plan.get("method") != METHOD:
        problems.append(f"method {plan.get('method')!r}")
    rule = plan["rule"]
    if canonical_sha(rule) != plan.get("rule_sha256"):
        problems.append("rule sha256 does not match")
    n = int(plan["n_worlds"])
    sizes = [int(math.ceil(f * n - 1e-9)) for f in plan["fractions"][:-1]] + [n]
    if sizes != list(plan["look_sizes"]):
        problems.append(f"look sizes {plan['look_sizes']} != recomputed {sizes}")
    fractions = tuple(s / n for s in plan["look_sizes"])
    for side in ("go", "kill"):
        z = boundaries(fractions, float(plan[f"{side}_alpha"]), plan[f"{side}_spending"])
        for k, (mine, theirs) in enumerate(zip(z, plan[f"{side}_boundaries"])):
            if abs(mine - theirs) > BOUNDARY_TOLERANCE:
                problems.append(f"{side} boundary {k}: {theirs} vs audit {mine}")
            if not math.isclose(_N.cdf(-theirs), plan[f"{side}_nominal_p"][k], rel_tol=1e-9):
                problems.append(f"{side} nominal level {k} does not match its boundary")
    if plan["go_spending"] != "obf":
        problems.append("GO spending is not O'Brien-Fleming")
    banks = plan_file["banks"]
    for seed in rule["seeds"]:
        bank = banks.get(str(seed))
        if bank is None or len(bank) != n or len(set(bank)) != n:
            problems.append(f"bank of seed {seed}")
    plan_file_sha = sha256_file(Path(root) / "plan.json")
    looks: List[Dict[str, Any]] = []
    previous_sha: Optional[str] = None
    ended = False
    k = 0
    while (Path(root) / "looks" / f"look-{k}.json").exists():
        path = Path(root) / "looks" / f"look-{k}.json"
        receipt = load(path)
        row: Dict[str, Any] = {"look": k, "problems": []}
        bad = row["problems"]
        if ended:
            bad.append("a receipt after the study ended")
        if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("look") != k:
            bad.append("receipt schema / look index")
        if receipt.get("plan_sha256") != plan_file.get("plan_sha256"):
            bad.append("plan sha256")
        if receipt.get("plan_file_sha256") != plan_file_sha:
            bad.append("plan file sha256")
        if receipt.get("previous_receipt_sha256") != previous_sha:
            bad.append("previous receipt sha256 (chain)")
        if k >= len(plan["look_sizes"]):
            bad.append("more receipts than planned looks")
            looks.append(row)
            break
        records = {}
        for rec in receipt.get("records", []):
            p = Path(rec["path"])
            if not p.exists():
                bad.append(f"record missing {p}")
                continue
            if sha256_file(p) != rec["sha256"]:
                bad.append(f"record sha256 mismatch {p}")
                continue
            data = load(p)
            key = (str(data["hero"]), int(data["seed"]), str(data["mix"]), int(data["world_seed"]))
            if list(key) != list(rec["key"]):
                bad.append(f"record key mismatch {p}")
            if data.get("control"):
                bad.append(f"control record used as decision data {p}")
            records[key] = data
        if canonical_sha(receipt.get("records", [])) != receipt.get("records_sha256"):
            bad.append("records_sha256")
        want = expected_keys(plan, banks, k)
        if set(records) != want:
            bad.append(
                f"record set != planned look set ({len(set(records) - want)} extra, "
                f"{len(want - set(records))} missing)"
            )
        decision = receipt.get("decision", {})
        if not bad:
            mine = None
            try:
                deltas: Dict[str, Any] = {}
                for cell, metrics in required(rule).items():
                    spec = rule["cells"][cell]
                    deltas[cell] = {m: {} for m in metrics}
                    for seed in spec["seeds"]:
                        worlds = banks[str(seed)][: plan["look_sizes"][k]]
                        for metric in metrics:
                            deltas[cell][metric][seed] = {
                                mix: [
                                    metric_value(records[(spec["hero"], seed, mix, w)], metric)
                                    - metric_value(
                                        records[(spec["baseline"], seed, mix, w)], metric
                                    )
                                    for w in worlds
                                ]
                                for mix in rule["mixes"]
                            }
                plain = {
                    c: {m: {str(s): v for s, v in by.items()} for m, by in bm.items()}
                    for c, bm in deltas.items()
                }
                if canonical_sha(plain) != receipt.get("deltas_sha256"):
                    bad.append("deltas digest")
                # replay every earlier look from prefixes (no earlier stop may exist)
                for j in range(k + 1):
                    cut = {
                        c: {
                            m: {
                                s: {x: v[: plan["look_sizes"][j]] for x, v in bs.items()}
                                for s, bs in bm.items()
                            }
                            for m, bm in cm.items()
                        }
                        for c, cm in deltas.items()
                    }
                    mine = decide(plan, j, cut, receipt.get("flags", {}))
                    if j < k and mine["action"] != "CONTINUE":
                        bad.append(f"look {j} replays to {mine['status']}: look {k} must not exist")
            except (AuditError, KeyError, ValueError, ZeroDivisionError) as exc:
                if decision.get("status") != "INVALID_ANALYSIS":
                    bad.append(f"audit could not decide ({exc}) but the receipt is not INVALID")
                mine = {"status": "INVALID_ANALYSIS", "action": "HALT"}
            if mine is not None:
                row["audit_status"] = mine["status"]
                if mine["status"] != decision.get("status"):
                    bad.append(f"status {decision.get('status')} != audit {mine['status']}")
                expected_action = mine["action"] if decision.get("valid") else "HALT"
                if receipt.get("action") != expected_action:
                    bad.append(f"action {receipt.get('action')} != audit {expected_action}")
                if "efficacy_estimate" in mine and "efficacy" in decision:
                    for a, b, what in (
                        (
                            decision["efficacy"]["estimate"],
                            mine["efficacy_estimate"],
                            "efficacy point",
                        ),
                        (
                            decision["efficacy"]["lower_bound"],
                            mine["efficacy_lower_bound"],
                            "efficacy bound",
                        ),
                    ):
                        if not close(a, b):
                            bad.append(f"{what} {a} vs audit {b}")
                    for part, ub in zip(decision["kill"]["parts"], mine["kill_upper_bounds"]):
                        if not close(part["upper_bound"], ub):
                            bad.append(
                                f"kill bound {part['name']} {part['upper_bound']} vs audit {ub}"
                            )
                    for name, ok in mine["clauses"].items():
                        if decision["clauses"][name]["pass"] != ok:
                            bad.append(
                                f"clause {name} {decision['clauses'][name]['pass']} vs audit {ok}"
                            )
        row["status"] = decision.get("status")
        row["action"] = receipt.get("action")
        looks.append(row)
        problems += [f"look {k}: {p}" for p in bad]
        ended = receipt.get("action") != "CONTINUE"
        previous_sha = sha256_file(path)
        k += 1
    # prefix integrity against the shard directories
    if looks and record_dirs:
        stop = min(looks[-1]["look"], len(plan["look_sizes"]) - 1)
        size = plan["look_sizes"][stop]
        position = {str(s): {int(w): i for i, w in enumerate(b)} for s, b in banks.items()}
        heroes = {c["hero"] for c in rule["cells"].values()} | {
            c["baseline"] for c in rule["cells"].values()
        }
        for d in record_dirs:
            for p in sorted(Path(d, "records").glob("*.json")):
                r = load(p)
                if r.get("control") or r.get("hero") not in heroes:
                    continue
                idx = position.get(str(r.get("seed")), {}).get(int(r.get("world_seed", -1)), -1)
                if idx >= size:
                    problems.append(f"record beyond the stopping look ({size} worlds): {p}")
    if not looks:
        verdict = "UNCLOSED"
    elif problems:
        verdict = "FAIL"
    elif looks[-1]["action"] == "CONTINUE":
        verdict = "UNCLOSED"
    else:
        verdict = "PASS"
    return {
        "schema": AUDIT_SCHEMA,
        "root": str(root),
        "verdict": verdict,
        "final_status": looks[-1]["status"] if looks else None,
        "looks": looks,
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--record-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.root, args.record_dirs)
    except AuditError as exc:
        result = {"schema": AUDIT_SCHEMA, "verdict": "FAIL", "problems": [str(exc)]}
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        if args.out.exists():
            print(f"{args.out} exists (create-only)", file=sys.stderr)
            return 1
        args.out.write_text(text + "\n")
    print(text)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
