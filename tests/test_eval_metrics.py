"""Independent tapes for full-horizon event accounting."""

from src.evaluation.metrics import (
    EvaluationMetricsAccumulator,
    PostStepState,
    StepEvents,
)


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
    assert result["probes"]["boost_frame_fraction"] == 2 / 3
    assert result["denominators"] == {
        "scored_frames": 4,
        "decision_frames": 3,
        "alive_frames": 2,
        "food_event_frames": 2,
        "boost_executed_frames": 2,
    }
    assert result["probes"]["entrapment_event"] is None
    assert result["probes"]["kill_opportunity_count"] is None
    assert "temporal probe" in result["probe_unavailable_reasons"]["entrapment_event"]


def test_incomplete_horizon_cannot_be_reported_as_a_score() -> None:
    metrics = EvaluationMetricsAccumulator(scored_horizon=2)
    metrics.observe(True, PostStepState(True, 2), StepEvents())
    try:
        metrics.result()
    except ValueError as error:
        assert "incomplete evaluation" in str(error)
    else:
        raise AssertionError("partial scores must fail closed")


def test_explicit_action_marker_uses_decisions_not_scored_frames() -> None:
    invalid = EvaluationMetricsAccumulator(scored_horizon=1)
    try:
        invalid.observe(True, PostStepState(True, 2), StepEvents(boost_executed=True), acted=False)
    except ValueError as error:
        assert "boost event" in str(error)
    else:
        raise AssertionError("boost padding without a decision must fail")

    metrics = EvaluationMetricsAccumulator(scored_horizon=4)
    metrics.observe(True, PostStepState(True, 2), StepEvents(), acted=True)
    metrics.observe(True, PostStepState(True, 2), StepEvents(boost_executed=True), acted=True)
    metrics.observe(
        True, PostStepState(False, 2), StepEvents(death=True, boost_executed=True), acted=True
    )
    metrics.observe(False, PostStepState(False, 2), StepEvents(), acted=False)
    assert metrics.result()["probes"]["boost_frame_fraction"] == 2 / 3


def test_no_decisions_is_null_not_a_plausible_zero_probe() -> None:
    metrics = EvaluationMetricsAccumulator(scored_horizon=2)
    metrics.observe(False, PostStepState(False, 0), StepEvents(), acted=False)
    metrics.observe(False, PostStepState(False, 0), StepEvents(), acted=False)
    result = metrics.result()
    assert result["probes"]["boost_frame_fraction"] is None
    assert result["probe_unavailable_reasons"]["boost_frame_fraction"]


def test_dead_padding_cannot_be_counted_as_a_decision() -> None:
    metrics = EvaluationMetricsAccumulator(scored_horizon=1)
    try:
        metrics.observe(False, PostStepState(False, 0), StepEvents(), acted=True)
    except ValueError as error:
        assert "dead hero" in str(error)
    else:
        raise AssertionError("dead padding cannot be a decision frame")


def test_death_event_cannot_claim_a_live_post_step_hero() -> None:
    metrics = EvaluationMetricsAccumulator(scored_horizon=1)
    try:
        metrics.observe(True, PostStepState(True, 2), StepEvents(death=True), acted=True)
    except ValueError as error:
        assert "dead post-step" in str(error)
    else:
        raise AssertionError("death event requires a terminal post-step state")
