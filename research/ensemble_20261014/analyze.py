#!/usr/bin/env python3
"""Summarize the ensemble dev probes against their members and the incumbent (dev info only).

  ./venv/bin/python research/ensemble_20261014/analyze.py --out RESULTS.json

Reads the ensemble rows written by ``ensemble_probe.py run`` and the members' existing
dev-probe rows (same worlds: round 52, 32 per mix, H5000). Per arm: mean mass (all 96 worlds),
per-mix mass / survival, paired mass delta vs the incumbent (95% t CI over worlds), deaths
and deaths at length >= 800 (ensembles: length at the hero's last decision; B8 members:
scripted mix from the B8 death census), the GO check (ensembles with K > 1 only),
paired ensemble-minus-member-average mass / survival, and the Q-calibration summary (per mix
and decision-weighted over the mixes: ``calibration_pooled``).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
OUT_DIR = ART / "research-tracks-20261014/ensemble"
INCUMBENT = ART / "redesign-m3c-probes-20261008-recovery/incumbent__dev__v8.jsonl"
B8_MEMBERS = {
    f"B8_s{s}": ART / f"redesign-m4-mac-probes-20261013/B8_s{s}__final__none.jsonl"
    for s in range(5)
}
CTRL_PROBES = ART / "redesign-m6-mac-probes-20261014"
CENSUS = ART / "b8-death-census-20261014/runs"
ENSEMBLES = {
    "mean_B8x5": OUT_DIR / "ens_mean_B8x5.jsonl",
    "vote_B8x5": OUT_DIR / "ens_vote_B8x5.jsonl",
    "mean_CTRL": OUT_DIR / "ens_mean_CTRL.jsonl",
    "vote_CTRL": OUT_DIR / "ens_vote_CTRL.jsonl",
    "identity_B8_s0_K1": OUT_DIR / "identity_B8_s0_K1.jsonl",
}
MIXES = ("frozen", "scripted", "mixed")
LONG = 800
GO_SCRIPTED_SURV = 0.64
GO_MASS_MARGIN = 60.0
T95 = {95: 1.985, 31: 2.040}


def load(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["mix"]] = r
    return rows


def ci(xs: List[float]) -> Dict[str, Any]:
    n = len(xs)
    if n < 2:
        return {"n": n}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    t = T95.get(n - 1, 1.96 if n > 100 else 2.0)
    return {"n": n, "mean": m, "lo95": m - t * sd / math.sqrt(n), "hi95": m + t * sd / math.sqrt(n)}


def arm_summary(rows: Dict[str, dict], ref: Dict[str, dict]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"per_mix": {}}
    allm: List[float] = []
    deltas: List[float] = []
    for mix in MIXES:
        if mix not in rows:
            continue
        r = rows[mix]
        mass = {str(k): float(v) for k, v in r["mass"].items()}
        deaths = {str(k): int(v) for k, v in r["deaths"].items()}
        allm += list(mass.values())
        deltas += [mass[s] - float(ref[mix]["mass"][s]) for s in mass if s in ref[mix]["mass"]]
        pm: Dict[str, Any] = {
            "n": len(mass),
            "mass": sum(mass.values()) / len(mass),
            "survival": sum(1 for d in deaths.values() if d == 0) / len(deaths),
            "deaths": sum(deaths.values()),
        }
        diag = r.get("diag")
        if diag:
            per = diag["per_world"]
            dead_len = [per[s]["last_len"] for s, d in deaths.items() if d and s in per]
            pm["deaths_ge800"] = sum(1 for x in dead_len if x is not None and x >= LONG)
            pm["death_len_median"] = sorted(dead_len)[len(dead_len) // 2] if dead_len else None
            dec_long = sum(per[s]["decisions_ge_long"] for s in per)
            pm["decisions_ge800"] = dec_long
            pm["hazard_ge800_per_1k"] = 1000.0 * pm["deaths_ge800"] / dec_long if dec_long else None
            pm["decisions"] = sum(per[s]["decisions"] for s in per)
            pm["long_decision_frac"] = dec_long / pm["decisions"] if pm["decisions"] else None
            peaks = [float(v) for v in r.get("peak", {}).values()]
            if peaks:
                pm["peak_median"] = sorted(peaks)[len(peaks) // 2]
        out["per_mix"][mix] = pm
    out["mass"] = sum(allm) / len(allm) if allm else None
    out["vs_incumbent"] = ci(deltas)
    sc = out["per_mix"].get("scripted")
    out["go"] = bool(
        sc
        and out["mass"] is not None
        and len(out["per_mix"]) == 3
        and sc["survival"] >= GO_SCRIPTED_SURV
        and out["mass"] >= ref_mass(ref) + GO_MASS_MARGIN
    )
    return out


def ref_mass(ref: Dict[str, dict]) -> float:
    m = [float(v) for mix in MIXES for v in ref[mix]["mass"].values()]
    return sum(m) / len(m)


def census_scripted(member: str) -> Optional[Dict[str, Any]]:
    """B8 member scripted-mix deaths at >= 800 and decisions at >= 800 (census)."""
    import numpy as np

    s = member.split("_s")[-1]
    j = CENSUS / f"b8_s{s}.json"
    z = CENSUS / f"b8_s{s}_series.npz"
    if not j.exists():
        return None
    d = json.loads(j.read_text())
    lens = [x["length"] for x in d["deaths"]]
    dec = int((np.load(z)["length"] >= LONG).sum())
    n = sum(1 for x in lens if x >= LONG)
    return {
        "deaths_ge800": n,
        "decisions_ge800": dec,
        "hazard_ge800_per_1k": 1000.0 * n / dec if dec else None,
        "death_len_median": sorted(lens)[len(lens) // 2] if lens else None,
    }


def paired_vs_members(ens: Dict[str, dict], members: Dict[str, Dict[str, dict]]) -> Dict:
    """Per world: ensemble minus the member-average (mass, survived)."""
    out: Dict[str, Any] = {}
    dm_all: List[float] = []
    for mix in MIXES:
        if mix not in ens or not all(mix in m for m in members.values()):
            continue
        dm: List[float] = []
        ds: List[float] = []
        for s, v in ens[mix]["mass"].items():
            mm = [float(m[mix]["mass"][s]) for m in members.values()]
            ms = [1.0 if int(m[mix]["deaths"][s]) == 0 else 0.0 for m in members.values()]
            dm.append(float(v) - sum(mm) / len(mm))
            ds.append((1.0 if int(ens[mix]["deaths"][s]) == 0 else 0.0) - sum(ms) / len(ms))
        out[mix] = {"mass": ci(dm), "survival": ci(ds)}
        dm_all += dm
    out["all_mass"] = ci(dm_all)
    return out


def member_pool(members: Dict[str, Dict[str, dict]], ref: Dict[str, dict]) -> Dict[str, Any]:
    sums = {k: arm_summary(v, ref) for k, v in members.items()}
    keys = list(sums)
    if not keys:
        return {}
    pool: Dict[str, Any] = {
        "mass_mean": sum(sums[k]["mass"] for k in keys) / len(keys),
        "mass_best": max(sums[k]["mass"] for k in keys),
        "per_mix_survival_mean": {
            mix: sum(sums[k]["per_mix"][mix]["survival"] for k in keys) / len(keys) for mix in MIXES
        },
        "scripted_survival_best": max(sums[k]["per_mix"]["scripted"]["survival"] for k in keys),
    }
    return {"members": sums, "pool": pool}


def _scripted_world_rows(first: Dict[str, dict], extra_path: Path) -> Optional[Dict[str, Any]]:
    """Per-world scripted outcomes: the first 32 worlds (``first``) + worlds 32..95 (extra)."""
    extra = load(extra_path).get("scripted")
    if "scripted" not in first or extra is None:
        return None
    out: Dict[str, Any] = {"mass": {}, "deaths": {}, "last_len": {}, "dec800": {}}
    for row in (first["scripted"], extra):
        out["mass"].update({str(k): float(v) for k, v in row["mass"].items()})
        out["deaths"].update({str(k): int(v) for k, v in row["deaths"].items()})
        if row.get("diag"):
            for s, w in row["diag"]["per_world"].items():
                out["last_len"][str(s)] = w["last_len"]
                out["dec800"][str(s)] = w["decisions_ge_long"]
    return out


def _arm96(w: Dict[str, Any], census: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    n = len(w["mass"])
    dead = [s for s, d in w["deaths"].items() if d]
    res: Dict[str, Any] = {
        "n": n,
        "mass": sum(w["mass"].values()) / n,
        "survival": 1 - len(dead) / n,
        "deaths": len(dead),
    }
    known = [s for s in dead if s in w["last_len"]]
    if len(known) == len(dead) and len(w["dec800"]) == n:
        d800 = sum(1 for s in dead if w["last_len"][s] >= LONG)
        dec = sum(w["dec800"].values())
    elif census is not None and len(w["dec800"]) == n - 32:
        # first 32 scripted worlds from the census, the rest from the extra run's diag
        d800 = census["deaths_ge800"] + sum(1 for s in known if w["last_len"][s] >= LONG)
        dec = census["decisions_ge800"] + sum(w["dec800"].values())
    else:
        return res
    res.update(
        {
            "deaths_ge800": d800,
            "decisions_ge800": dec,
            "hazard_ge800_per_1k": round(1000.0 * d800 / dec, 4) if dec else None,
        }
    )
    return res


def scripted96(b8: Dict[str, Dict[str, dict]]) -> Dict[str, Any]:
    """Scripted mix on 96 dev-probe worlds (first 32 + extra 64) for members and ensembles."""
    members = {}
    for k, rows in b8.items():
        w = _scripted_world_rows(rows, OUT_DIR / f"extra_scripted_{k}_K1.jsonl")
        if w is not None:
            members[k] = w
    out: Dict[str, Any] = {"arms": {}, "vs_member_average": {}}
    for k, w in members.items():
        out["arms"][k] = _arm96(w, census_scripted(k))
    if len(members) == len(b8):
        pooled = {
            "n": sum(a["n"] for a in out["arms"].values()),
            "deaths": sum(a["deaths"] for a in out["arms"].values()),
        }
        pooled["survival"] = 1 - pooled["deaths"] / pooled["n"]
        pooled["mass"] = sum(a["mass"] for a in out["arms"].values()) / len(members)
        if all("deaths_ge800" in a for a in out["arms"].values()):
            pooled["deaths_ge800"] = sum(a["deaths_ge800"] for a in out["arms"].values())
            pooled["decisions_ge800"] = sum(a["decisions_ge800"] for a in out["arms"].values())
            pooled["hazard_ge800_per_1k"] = round(
                1000.0 * pooled["deaths_ge800"] / pooled["decisions_ge800"], 4
            )
        out["arms"]["B8_pooled"] = pooled
    for name in ("vote_B8x5", "mean_B8x5"):
        w = _scripted_world_rows(load(ENSEMBLES[name]), OUT_DIR / f"extra_scripted_{name}.jsonl")
        if w is None:
            continue
        out["arms"][name] = _arm96(w)
        if len(members) == len(b8):
            dm, ds = [], []
            for s in w["mass"]:
                mm = [m["mass"][s] for m in members.values()]
                ms = [1.0 if m["deaths"][s] == 0 else 0.0 for m in members.values()]
                dm.append(w["mass"][s] - sum(mm) / len(mm))
                ds.append((1.0 if w["deaths"][s] == 0 else 0.0) - sum(ms) / len(ms))
            out["vs_member_average"][name] = {"mass": ci(dm), "survival": ci(ds)}
    return out


def calibration_pooled(cal: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Decision-weighted pool of the per-mix calibration stats, one entry per length bucket.

    Is averaging Q across independently trained seeds meaningful? Compare the members'
    value level (masked max Q) and advantage scale (sd of centered Q), their correlation, and
    how often the mean-Q rule disagrees with the scale-free z-mean rule.
    """
    out: Dict[str, Any] = {}
    for bucket in ("lt800", "ge800"):
        parts = [c[bucket] for c in cal.values() if c.get(bucket, {}).get("decisions")]
        if not parts:
            continue
        n = sum(p["decisions"] for p in parts)
        k = len(parts[0]["vmax_mean"])

        def wavg(key: str) -> Any:
            first = parts[0][key]
            if isinstance(first, list):
                return [
                    sum(p[key][i] * p["decisions"] for p in parts) / n for i in range(len(first))
                ]
            return sum(p[key] * p["decisions"] for p in parts) / n

        def offdiag(key: str) -> List[float]:
            return [
                sum(p[key][i][j] * p["decisions"] for p in parts) / n
                for i in range(k)
                for j in range(i + 1, k)
            ]

        vmax = wavg("vmax_mean")
        adv = wavg("adv_sd")
        agree = wavg("member_agree_chosen")
        pair = offdiag("pair_agree")
        vcorr = offdiag("vmax_corr")
        acorr = offdiag("adv_corr")
        out[bucket] = {
            "decisions": n,
            "vmax_mean_per_member": vmax,
            "vmax_mean_range": max(vmax) - min(vmax),
            "adv_sd_per_member": adv,
            "adv_sd_max_over_min": max(adv) / max(min(adv), 1e-12),
            "vmax_corr_min_mean": [min(vcorr), sum(vcorr) / len(vcorr)] if vcorr else None,
            "adv_corr_min_mean": [min(acorr), sum(acorr) / len(acorr)] if acorr else None,
            "top2_gap_mean": sum(wavg("gap_mean")) / k,
            "vmax_member_spread": wavg("vmax_member_spread"),
            "unanimous_frac": wavg("unanimous_frac"),
            "pair_agree_mean": sum(pair) / len(pair) if pair else 1.0,
            "chosen_ne_member_frac": [1.0 - a for a in agree],
            "mean_ne_vote_frac": wavg("mean_ne_vote_frac"),
            "mean_ne_zmean_frac": wavg("mean_ne_zmean_frac"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR / "results.json")
    args = ap.parse_args()
    ref = load(INCUMBENT)
    b8 = {k: load(p) for k, p in B8_MEMBERS.items()}
    ctrl = {
        p.name.split("__")[0]: load(p)
        for p in sorted(CTRL_PROBES.glob("CTRL_s*__final__none.jsonl"))
    }
    res: Dict[str, Any] = {
        "authority": "development information only (dev probe, round 52); not a screen",
        "incumbent": {"file": str(INCUMBENT), **arm_summary(ref, ref)},
        "go_rule": f"scripted survival >= {GO_SCRIPTED_SURV} and mass >= incumbent + "
        f"{GO_MASS_MARGIN}",
        "b8_members": member_pool(b8, ref),
        "ctrl_members": member_pool(ctrl, ref),
        "b8_census_scripted": {k: census_scripted(k) for k in b8},
        "ensembles": {},
    }
    for name, path in ENSEMBLES.items():
        rows = load(path)
        if not rows:
            continue
        first = next(iter(rows.values()))
        ens_members = first.get("ensemble", {}).get("members", [])
        e: Dict[str, Any] = {
            "file": str(path),
            "rule": first.get("ensemble", {}).get("rule"),
            "members": ens_members,
            "wall_per_mix": {m: rows[m]["wall"] for m in rows},
            **arm_summary(rows, ref),
        }
        if "CTRL" in name:
            used = {k: v for k, v in ctrl.items() if any(f"/{k}/" in m for m in ens_members)}
            e["vs_member_average"] = paired_vs_members(rows, used)
            e["member_probes_used"] = sorted(used)
        elif "B8x5" in name:
            e["vs_member_average"] = paired_vs_members(rows, b8)
        e["calibration"] = {m: rows[m]["diag"]["calibration"] for m in rows if rows[m].get("diag")}
        e["calibration_pooled"] = calibration_pooled(e["calibration"])
        # GO is a property of a real ensemble; the K=1 identity row is only a replay check
        e["go"] = bool(e["go"] and len(ens_members) > 1)
        res["ensembles"][name] = e
    res["scripted96"] = scripted96(b8)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res["scripted96"].get("arms", {}).items():
        print(
            f"scripted96 {k:12s} n={v['n']} surv {v['survival']:.3f} mass {v['mass']:6.1f} "
            f"d>=800 {v.get('deaths_ge800')}/{v['deaths']} haz800 {v.get('hazard_ge800_per_1k')}"
        )
    for k, v in res["scripted96"].get("vs_member_average", {}).items():
        print(f"scripted96 {k} - member avg: {json.dumps(v)}")

    def line(name: str, s: Dict[str, Any]) -> str:
        pm = s["per_mix"]
        surv = " ".join(f"{m[:4]} {pm[m]['survival']:.3f}" for m in MIXES if m in pm)
        g800 = " ".join(
            f"{m[:4]} {pm[m].get('deaths_ge800')}/{pm[m]['deaths']}" for m in MIXES if m in pm
        )
        v = s["vs_incumbent"]
        dv = f"{v['mean']:+.1f} [{v['lo95']:+.1f},{v['hi95']:+.1f}]" if "mean" in v else "-"
        return f"{name:22s} mass {s['mass']:6.1f} dInc {dv:24s} surv {surv} | d>=800 {g800}"

    print(line("incumbent", res["incumbent"]))
    for k, s in res["b8_members"].get("members", {}).items():
        print(line(k, s))
    for k, s in res["ctrl_members"].get("members", {}).items():
        print(line(k, s))
    for k, s in res["ensembles"].items():
        print(line(k, s), "GO" if s["go"] else "")
    for k, s in res["ensembles"].items():
        for b, c in s.get("calibration_pooled", {}).items():
            if len(c["vmax_mean_per_member"]) < 2:
                continue
            print(
                f"calib {k:12s} {b}: n={c['decisions']} vmax range {c['vmax_mean_range']:.2f} "
                f"adv_sd max/min {c['adv_sd_max_over_min']:.2f} "
                f"adv_corr min {c['adv_corr_min_mean'][0]:.3f} top2 gap {c['top2_gap_mean']:.3f} "
                f"vmax spread {c['vmax_member_spread']:.2f} unanimous {c['unanimous_frac']:.3f} "
                f"mean!=z {c['mean_ne_zmean_frac']:.4f} mean!=vote {c['mean_ne_vote_frac']:.4f} "
                f"chosen!=member {max(c['chosen_ne_member_frac']):.3f} max"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
