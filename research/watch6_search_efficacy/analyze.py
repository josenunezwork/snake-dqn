"""Mac-only one-look reduction of the complete verified union of 24 sessions."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
from scipy.stats import t

from research.watch6_search_efficacy import protocol as p
from tools.ops.package_spec import publish_json, verify_immutable


def decide(rows: list[dict]) -> dict:
    """Paired t limits, session cluster; co-primary intersection-union test."""
    if len(rows) != p.COUNT or {r["allocation"]["index"] for r in rows} != set(range(p.COUNT)):
        raise ValueError("INCOMPLETE or duplicate union; no scientific look")
    rows = sorted(rows, key=lambda r: r["allocation"]["index"])
    for row, expected in zip(rows, p.allocation()):
        if (
            row["namespace"] != p.NAMESPACE
            or row["science"] != p.identity()
            or row["allocation"] != expected
            or row["fixture"] is not False
        ):
            raise ValueError("foreign allocation, fixture or science identity")
        for arm in row["arms"].values():
            if arm["frames"] != p.HORIZON or arm["mirror_identity_frames"] != p.HORIZON:
                raise ValueError("incomplete product session")
    bare = np.array([r["arms"]["bare"]["mass_sum"] / p.HORIZON for r in rows])
    dense = np.array([r["arms"]["exact_stop"]["mass_sum"] / p.HORIZON for r in rows])
    delta = np.array(
        [
            (r["arms"]["exact_stop"]["deaths"] - r["arms"]["bare"]["deaths"]) * 1000 / p.HORIZON
            for r in rows
        ]
    )
    critical = t.ppf(0.975, p.COUNT - 1)

    def interval(values: np.ndarray) -> list:
        mean = float(values.mean())
        width = float(critical * values.std(ddof=1) / np.sqrt(p.COUNT))
        return [mean - width, mean + width]

    # Ratio of pooled full-horizon means. Fieller inversion, not averaging
    # per-session percentages (which overweight tiny-mass baseline sessions).
    d = dense - bare
    ratio = float(d.mean() / bare.mean()) if bare.mean() > 0 else None
    c = critical**2 / p.COUNT
    a = float(bare.mean() ** 2 - c * bare.var(ddof=1))
    b = float(-2 * d.mean() * bare.mean() + 2 * c * np.cov(d, bare, ddof=1)[0, 1])
    q = float(d.mean() ** 2 - c * d.var(ddof=1))
    disc = b * b - 4 * a * q
    bounds = (
        sorted([(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)])
        if a > 0 and disc >= 0
        else None
    )
    deaths_ci = interval(delta)
    death_pass = deaths_ci[1] < 0
    mass_pass = bounds is not None and bounds[0] >= -0.05
    # Evidence against benefit or preservation closes this design on this bank;
    # all other failures to demonstrate both bars are honestly inconclusive.
    verdict = (
        "GO_EFFICACY_ONLY"
        if death_pass and mass_pass
        else (
            "FAIL_EFFICACY"
            if (deaths_ci[0] >= 0 and deaths_ci[1] > 0) or (bounds and bounds[1] < -0.05)
            else "INCONCLUSIVE"
        )
    )
    baseline_deaths = sum(r["arms"]["bare"]["deaths"] for r in rows)
    candidate_deaths = sum(r["arms"]["exact_stop"]["deaths"] for r in rows)
    return {
        "verdict": verdict,
        "n": p.COUNT,
        "death_delta_per_1k": float(delta.mean()),
        "death_delta_ci95": deaths_ci,
        "mass_relative_difference": ratio,
        "mass_relative_ci95_fieller": bounds,
        "death_pass": death_pass,
        "mass_pass": mass_pass,
        "death_reduction_fraction": (
            1 - candidate_deaths / baseline_deaths if baseline_deaths else None
        ),
        "authority": "development efficacy only; no timing, strict, serving or release authority",
        "session_diagnostics": [
            {"index": r["allocation"]["index"], "arms": r["arms"]} for r in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("registered look is Mac-only")
    for path in args.units:
        verify_immutable(path)
    publish_json(args.out, decide([json.loads(path.read_text()) for path in args.units]))


if __name__ == "__main__":
    main()
