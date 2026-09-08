"""Independent tapes for full-horizon event accounting."""

from src.evaluation.metrics import EvaluationMetricsAccumulator, PostStepState, StepEvents


def test_mass_is_post_step_and_death_frames_are_zero_but_events_remain() -> None:
    metrics = EvaluationMetricsAccumulator(scored_horizon=4)
    metrics.observe(True, PostStepState(True, 2), StepEvents())
    # Food is exact even though a pending growth update leaves mass unchanged.
    metrics.observe(True, PostStepState(True, 2), StepEvents(food_eaten=1, boost_executed=True))
    # Eat, kill, boost and die together: score is zero but all events count.
    metrics.observe(
        True,
        PostStepState(False, 3),
        StepEvents(food_eaten=1, boost_executed=True, kills=1, death=True, death_cause="wall"),
    )
    metrics.observe(False, PostStepState(False, 3), StepEvents())

    result = metrics.result()
    assert result["mass_integral"] == 1.0
    assert result["survival_fraction"] == 0.5
    assert result["kills"] == 1
    assert result["deaths"] == 1
    assert result["probes"]["food_eaten"] == 2
    assert result["probes"]["boost_frame_fraction"] == 0.5
    assert result["denominators"] == {
        "scored_frames": 4,
        "decision_frames": 4,
        "alive_frames": 2,
        "food_event_frames": 2,
        "boost_executed_frames": 2,
    }
    assert result["probes"]["entrapment_event"] is None
    assert result["probes"]["kill_opportunity_count"] is None


def test_incomplete_horizon_cannot_be_reported_as_a_score() -> None:
    metrics = EvaluationMetricsAccumulator(scored_horizon=2)
    metrics.observe(True, PostStepState(True, 2), StepEvents())
    try:
        metrics.result()
    except ValueError as error:
        assert "incomplete evaluation" in str(error)
    else:
        raise AssertionError("partial scores must fail closed")
