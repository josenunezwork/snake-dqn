"""One-run prefix identity and leaf-unit calibration, independent short searches."""

import numpy as np
import pytest

from research.phase0_search_20261019 import teacher
from research.t33_batched_teacher import ablation, reference
from research.t33_batched_teacher.benchmark import prepared


def test_prefixes_equal_separate_searches_with_same_rng() -> None:
    if not ablation.B8.exists():
        pytest.skip("local B8 inventory absent")
    teacher.configure_cpu()
    world = prepared(ablation.fixture(ablation.calibration_rows(smoke=True)[2], ablation.B8, 800))
    config = teacher.SearchConfig(horizon=8, rollouts=3, epsilon=0.5, length_gate=0)
    actions = list(map(int, np.flatnonzero(world.sim.get_legal_action_mask()[0, 0])))
    decision = int(world.sim.frame[0]) - 1
    trace = reference.prefix_trace(world, actions, config, decision, horizons=[1, 3, 8])
    for h, n in ((1, 1), (3, 2), (8, 3)):
        for leaf in ("max", "value"):
            short = teacher.SearchConfig(horizon=h, rollouts=n, epsilon=0.5, length_gate=0)
            assert reference.trace_scores(trace, h, n, leaf, 1.0) == reference.scores(
                world, actions, short, decision, leaf=leaf
            )


def test_leaf_fit_recovers_segment_frames_per_q_and_reports_zero_one() -> None:
    raw = {
        "branches": [(0, 0), (1, 0)],
        "horizon": 4,
        "rollouts": 1,
        "checkpoints": {
            "2": {"mass": [10, 20], "max": [2, 3], "value": [1, 2]},
            "4": {"mass": [30, 50], "max": [0, 0], "value": [0, 0]},
        },
    }
    row = {
        "prefix_trace": raw,
        "reference_action": 1,
        "reference_scores": {0: 30, 1: 50},
        "variants": [{"id": "h2-n1-max", "horizon": 2, "rollouts": 1, "leaf": "max"}],
    }
    fitted = ablation.calibrate([row])["h2-n1-max"]
    assert fitted["scale"] == 10.0
    assert fitted["rmse"] == 0.0
    assert fitted["samples"] == 2
    assert fitted["reports"]["zero"]["scale"] == 0.0
    assert fitted["reports"]["one"]["scale"] == 1.0
    assert "segment-frames" in fitted["units"]
