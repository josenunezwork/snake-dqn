"""Prospective death-count sensitivity; historical mass is not a power prior."""

from __future__ import annotations

import json

from scipy.stats import norm, t

from research.watch6_search_efficacy import protocol as p


def sensitivity() -> list[dict]:
    """Normal planning approximation, t critical95, unknown session variance shown."""
    result = []
    for baseline in (1.8, 0.296 * p.HORIZON / 1000):
        for reduction in (0.2, 0.3, 5 / 11):
            for dispersion in (1, 2):
                # Independent-arm Poisson counts, conservatively no paired gain.
                variance = dispersion * baseline * (2 - reduction)
                signal = baseline * reduction * (p.COUNT / variance) ** 0.5
                critical = t.ppf(0.975, p.COUNT - 1)
                result.append(
                    {
                        "n": p.COUNT,
                        "baseline_deaths_per_session": baseline,
                        "expected_baseline_deaths": p.COUNT * baseline,
                        "death_reduction_fraction": reduction,
                        "dispersion": dispersion,
                        "paired_correlation_assumed": 0,
                        "signal_se": signal,
                        "approximate_death_bar_power": float(
                            norm.cdf(signal - critical) + norm.cdf(-signal - critical)
                        ),
                    }
                )
    return result


if __name__ == "__main__":
    print(json.dumps(sensitivity(), indent=2))
