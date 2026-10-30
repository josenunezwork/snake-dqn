"""Counterexamples to optimistic, cold-body or wrong-batch ruler admission."""

from unittest.mock import patch

from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.forecast import (
    export_projection,
    historical_support_projection,
    late_collection_rate,
    support_projection,
    tick_forecast,
)


def test_whole_slab_tick_prices_charge_full_horizons_and_closed_loop_rate() -> None:
    costs = tick_forecast(
        collection_tick=0.02,
        b8_branch_tick=0.03,
        student_branch_tick=0.09,
        gameplay_ticks={"b8": 0.04, "student": 0.12},
        fit_update=0.2,
    )
    assert costs["gameplay"] == 600  # one pairedE384 loop; no early-death discount
    assert costs["search"] == 108
    assert costs["assessment"] == 117
    assert costs["collection"] == 100
    assert costs["fit"] == 1600


def test_collection_uses_mature_tail_and_support_counts_distinct_parents() -> None:
    cfg = PilotConfig()
    ids = list(range(128))
    parts = cfg.partitions()
    reached = [
        i
        for p, n in (("train", 33), ("assessment", 8))
        for i in [i for i in ids if parts[i] == p][:n]
    ]
    batch = dict(
        world_ids=ids,
        reached_l600_world_ids=reached * 3,
        tick_samples=[
            dict(tick=i, seconds=1 if i < 60 else 4, mature_worlds=1 if i >= 60 else 0)
            for i in range(80)
        ],
    )
    assert late_collection_rate(batch)["seconds_per_tick"] == 4
    assert late_collection_rate(batch)["qualified"]
    assert support_projection(cfg, batch)["qualified"]
    batch["reached_l600_world_ids"] = reached[:31]
    assert not support_projection(cfg, batch)["qualified"]
    batch["tick_samples"][-1]["mature_worlds"] = 0
    assert not late_collection_rate(batch)["qualified"]


def test_size_projection_checks_decoded_gzip_and_total_caps() -> None:
    import torch

    root = dict(hero_length=700, compact={"body_cells": torch.zeros((1, 6, 700, 2))})
    batch = dict(roots=[root])
    for sizes, qualified in (
        ((100, 100), True),
        ((200000, 100), False),
        ((100, 200000), False),
        ((100, 20000), False),
    ):
        with patch("research.spatial_student_pilot.forecast._sizes", return_value=sizes):
            report = export_projection(
                batch,
                [root],
                [{"optimizer": 1}],
                1000,
                assessment_rows=[{"measured": True}],
                gameplay_rows=[{"measured": True}],
            )
        assert report["qualified"] is qualified
    assert not export_projection(batch, [], [], 1000)["qualified"]


def test_gameplay_floor_charges_mature_tick_and_extra_student_forward() -> None:
    costs = tick_forecast(
        collection_tick=2,
        b8_branch_tick=1,
        student_branch_tick=1,
        gameplay_ticks={"b8": 0.01, "student": 0.02},
        fit_update=0.1,
        student_forward_tick=0.5,
    )
    assert costs["gameplay"] == 12500


def test_historical_support_never_uses_restored_reach_counts() -> None:
    batch = dict(
        reached_l600_world_ids=list(range(128)),
        tick_samples=[dict(seconds=0.5, mean_hero_length=1 + i / 5) for i in range(80)],
    )
    report = historical_support_projection(batch)
    assert report["qualified"]
    assert not report["fresh_support_proven"]
    assert report["train"]["projected_worlds"] == 768 * 554 / 820
    batch["tick_samples"] = [dict(seconds=0.5, mean_hero_length=1)] * 80
    assert not historical_support_projection(batch)["qualified"]


def test_assessment_and_gameplay_sizes_are_measured_not_fixed_allowances() -> None:
    import torch

    root = dict(hero_length=700, compact={"body_cells": torch.zeros((1, 6, 700, 2))})
    small = export_projection(
        dict(roots=[root]),
        [root],
        [{"optimizer": 1}],
        1000,
        assessment_rows=[dict(a=1)],
        gameplay_rows=[dict(a=1)],
    )
    large = export_projection(
        dict(roots=[root]),
        [root],
        [{"optimizer": 1}],
        1000,
        assessment_rows=[dict(a=b"x" * 1000000)],
        gameplay_rows=[dict(a=b"x" * 1000000)],
    )
    for kind in ("assess", "gameplay"):
        assert large["units"][kind]["decoded"] > small["units"][kind]["decoded"]
    assert not export_projection(dict(roots=[root]), [root], [1], 1000)["qualified"]


def test_checkpoint_pricing_uses_real_batches_and_refuses_missing_components() -> None:
    import pytest

    from research.spatial_student_pilot.forecast import checkpoint_bound, checkpoint_components

    payload = dict(
        records={
            "w0/t0/demo": dict(compact=b"a" * 1000),
            "w0/t0/ordinary": dict(snapshot=b"b" * 1000),
        },
        snapshots=[dict(body=b"c" * 1000)],
        tick_samples=[dict(tick=0)],
        demos={0: [(0, 0, "w0/t0/demo")]},
        roots={(0, "ordinary"): [(0, 0, "w0/t0/ordinary")]},
    )
    measured = checkpoint_components(payload)
    assert measured["demo_records"]["count"] == 1
    assert measured["snapshots"]["decoded"] > measured["snapshots"]["gzip"] > 0
    assert checkpoint_bound([measured], 1.25)["gzip"] > 0
    payload["records"].pop("w0/t0/demo")
    empty = checkpoint_components(payload)
    with pytest.raises(ValueError, match="component samples"):
        checkpoint_bound([empty], 1.25)
    with pytest.raises(ValueError, match="component samples"):
        checkpoint_bound([], 1.25)
