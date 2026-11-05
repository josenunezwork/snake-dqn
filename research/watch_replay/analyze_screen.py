"""Pre-registered Bank-v2 screen analysis (CAMPAIGN_LOG 2026-11-05 14:54 PT row).

Rule (fixed before data): ADOPT exact-stop-l1-os only if
  (1) the world-paired mean hero-mass delta (l1-os - l1, hero_mass_integral) >= +15, and
  (2) pooled deaths per 1k ticks of l1-os <= that of l1.
The decision uses the point estimates; world-paired 95% percentile bootstrap CIs
(resampling worlds with replacement, fixed seed) are reported for both quantities.
Every one of the 48 worlds must have both arms verified, else the analysis refuses
(no partial looks).  Dev screen: exploration lane, not release evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.watch_replay.screen_l1_os import ARMS, allocation, identity
from tools.ops.package_spec import verify_immutable

MASS_DELTA_MIN = 15.0
BOOTSTRAP_SEED = 20261105
BOOTSTRAP_DRAWS = 10000
CI = 0.95


def load_units(out: Path, contract: dict) -> dict[tuple[int, str], dict]:
    if {k: v for k, v in contract.items() if k != "commit"} != allocation(
        contract["namespace"], contract["count"], contract["horizon"]
    ):
        raise ValueError("contract differs from the frozen screen allocation")
    rows: dict[tuple[int, str], dict] = {}
    for unit in contract["units"]:
        path = out / "units" / f"{unit['index']:06d}.json"
        verify_immutable(path)
        value = json.loads(path.read_text())
        if (
            value.get("allocation") != identity(contract)
            or value.get("index") != unit["index"]
            or value.get("arm") != unit["arm"]
            or value.get("seed") != unit["seed"]
            or value.get("horizon") != contract["horizon"]
        ):
            raise ValueError(f"foreign or mismatched unit result: {path}")
        rows[(unit["world"], unit["arm"])] = value
    return rows


def analyze(rows: dict[tuple[int, str], dict], count: int, horizon: int) -> dict[str, Any]:
    """Pure decision function on per-(world, arm) records."""
    base, cand = ARMS
    worlds = list(range(count))
    if any((w, a) not in rows for w in worlds for a in ARMS):
        raise ValueError("incomplete pairs: every world needs both arms")
    mass = np.array([[rows[(w, a)]["hero_mass_integral"] for a in ARMS] for w in worlds], float)
    deaths = np.array([[rows[(w, a)]["hero_deaths"] for a in ARMS] for w in worlds], float)
    delta = mass[:, 1] - mass[:, 0]
    ticks = horizon * count
    d1k = 1000.0 * deaths.sum(axis=0) / ticks

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, count, size=(BOOTSTRAP_DRAWS, count))
    boot_mass = delta[draws].mean(axis=1)
    boot_d1k = 1000.0 * (deaths[draws, 1].sum(axis=1) - deaths[draws, 0].sum(axis=1)) / ticks
    lo, hi = 100 * (1 - CI) / 2, 100 * (1 + CI) / 2

    mass_delta = float(delta.mean())
    mass_ok = bool(mass_delta >= MASS_DELTA_MIN)
    deaths_ok = bool(d1k[1] <= d1k[0])
    return {
        "schema": "t34-bank-v2-screen-readout/v1",
        "rule": (
            f"ADOPT iff paired mean mass delta >= {MASS_DELTA_MIN:g} and deaths/1k(l1-os) "
            "<= deaths/1k(l1) (point estimates; world-paired bootstrap CIs reported)"
        ),
        "worlds": count,
        "horizon": horizon,
        "baseline": base,
        "candidate": cand,
        "mass_mean": {base: float(mass[:, 0].mean()), cand: float(mass[:, 1].mean())},
        "paired_mass_delta": mass_delta,
        "paired_mass_delta_ci95": [
            float(np.percentile(boot_mass, lo)),
            float(np.percentile(boot_mass, hi)),
        ],
        "worlds_mass_improved": int((delta > 0).sum()),
        "deaths_total": {base: int(deaths[:, 0].sum()), cand: int(deaths[:, 1].sum())},
        "deaths_per_1k": {base: float(d1k[0]), cand: float(d1k[1])},
        "deaths_per_1k_delta": float(d1k[1] - d1k[0]),
        "deaths_per_1k_delta_ci95": [
            float(np.percentile(boot_d1k, lo)),
            float(np.percentile(boot_d1k, hi)),
        ],
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "unit": "world"},
        "mass_bar_met": mass_ok,
        "deaths_bar_met": deaths_ok,
        "verdict": "ADOPT" if mass_ok and deaths_ok else "DO_NOT_ADOPT",
        "open_stop_fires_total": int(sum(rows[(w, cand)]["open_stop_fires"] for w in worlds)),
        "scaled_horizon_roots_total": int(
            sum(rows[(w, cand)]["scaled_horizon_roots"] for w in worlds)
        ),
        "compute_hours": {
            a: float(sum(rows[(w, a)]["compute_seconds"] for w in worlds) / 3600) for a in ARMS
        },
        "scope": "dev screen (exploration lane); not release or strict-gate evidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="screen output dir (units/)")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    report = analyze(load_units(args.out, contract), contract["count"], contract["horizon"])
    report["allocation"] = identity(contract)
    args.readout.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("verdict", "paired_mass_delta", "paired_mass_delta_ci95", "deaths_per_1k")
            }
        )
    )


if __name__ == "__main__":
    main()
