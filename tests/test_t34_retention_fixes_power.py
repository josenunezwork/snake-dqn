"""Development-only power checks for the T34 retention amendment."""

from __future__ import annotations

import numpy as np

from research.t34_pathb import retention_oc_v2 as oc


def _pool() -> dict:
    shape = []
    for i in range(32):
        base_alive = 1.0 if i % 2 else 0.6
        candidate_alive = 1.0 if i % 3 else 0.7
        shape.append(
            [
                float(i * 4),
                candidate_alive - base_alive,
                base_alive,
                int(candidate_alive == 1) - int(base_alive == 1),
                int(base_alive == 1),
            ]
        )
    t15 = [
        {"seed": index, "deaths": float(1 + index % 3), "decisions": 5000.0} for index in range(25)
    ]
    t13b = [
        {"seed": 100 + index, "deaths": float(1 + (2 * index) % 4), "decisions": 5000.0}
        for index in range(25)
    ]
    return {
        "t3_pools": {"b8_s3": shape},
        "watch_baselines": [{**row, "decision_mean_length": 100.0} for row in t15],
        "watch6_product_baselines": {"t15_mixed_watch6": t15, "t13b_mixed_watch6": t13b},
        "source_files": {"development.json": "a" * 64},
    }


def test_pathb_power_grid_is_fixed_and_complete() -> None:
    expected = {
        f"scripted_survival_plus_{survival:g}|mass_plus_{mass:g}"
        for survival in (0.10, 0.25)
        for mass in (40.0, 60.0, 92.0)
    }
    assert set(oc.pathb_power_scenarios()) == expected
    assert oc.N == 48
    assert oc.WATCH_PRODUCT_SESSIONS == 32
    assert oc.SEQUENTIAL_EARLY_LOOK == 24


def test_pathb_and_product_power_are_deterministic_and_development_only() -> None:
    pool = _pool()
    first = oc.simulate_batch(pool, seed=21, offset=0, count=250)
    second = oc.simulate_batch(pool, seed=21, offset=0, count=250)
    assert first == second
    pathb = {key: value for key, value in first.items() if "|pathb_joint_go|" in key}
    assert len(pathb) == len(oc._models(pool)) * len(oc.pathb_power_scenarios()) * 2 * 2
    assert all(0 <= value <= 250 for value in pathb.values())
    product = {key: value for key, value in first.items() if key.startswith("product_power|")}
    assert len(product) == 2 * len(oc.WATCH_REDUCTIONS)
    assert all(
        "paired_ci95_excludes_zero" in key and 0 <= value <= 250 for key, value in product.items()
    )


def test_product_power_uses_two_sided_paired_95_percent_upper_endpoint() -> None:
    samples = np.array([[-1.0, -1.0, -1.0, -1.0], [-2.0, 0.0, -2.0, 0.0]])
    upper = oc._paired_upper95(samples)
    assert upper[0] == -1.0
    assert upper[1] > 0.0
