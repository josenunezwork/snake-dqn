"""Unit contracts for the Amendment-2 Path B shadow search surface."""

from __future__ import annotations

from research.t34_pathb.engine_v2 import (
    PathBSearchEngine,
    PreparedDecision,
    SearchOutcome,
    SearchRequest,
    SearchResult,
)


def _request(*, bare: int = 0, q: tuple[float, ...] = (9, 8, 7, 6, 5, 4)) -> SearchRequest:
    return SearchRequest(
        shadow_id=7,
        source_digest="source",
        target_frame=99,
        decision=PreparedDecision(99, object(), bare, q, (True, True, True, False, False, False)),
        horizon=500,
    )


def test_search_ranks_earliest_horizon_rescue_by_q_then_index(monkeypatch) -> None:
    request = _request(q=(10, 3, 3, 0, 0, 0))

    def outcomes(*_args, **_kwargs):
        return {
            0: SearchOutcome(0, False, 3, 9, 500),
            1: SearchOutcome(1, True, 500, 100, 500),
            2: SearchOutcome(2, True, 500, 99, 500),
        }

    monkeypatch.setattr("research.t34_pathb.engine_v2.evaluate_prepared", outcomes)
    result = PathBSearchEngine.run_search(request)
    assert result.status == "IMPROVING"
    assert result.action == 1
    assert result.target_frame == 99


def test_search_keeps_bare_that_reaches_horizon(monkeypatch) -> None:
    request = _request()

    def outcomes(*_args, **_kwargs):
        return {
            0: SearchOutcome(0, True, 500, 20, 500),
            1: SearchOutcome(1, True, 500, 999, 500),
            2: SearchOutcome(2, False, 4, 20, 500),
        }

    monkeypatch.setattr("research.t34_pathb.engine_v2.evaluate_prepared", outcomes)
    result = PathBSearchEngine.run_search(request)
    assert result.status == "NO_IMPROVEMENT"
    assert result.action is None


def test_search_all_die_returns_only_strictly_longer_delay(monkeypatch) -> None:
    request = _request()

    def outcomes(*_args, **_kwargs):
        return {
            0: SearchOutcome(0, False, 10, 40, 500),
            1: SearchOutcome(1, False, 11, 44, 500),
            2: SearchOutcome(2, False, 12, 48, 500),
        }

    monkeypatch.setattr("research.t34_pathb.engine_v2.evaluate_prepared", outcomes)
    result = PathBSearchEngine.run_search(request)
    assert result.status == "IMPROVING"
    assert result.action == 2
    assert not result.outcomes[result.action].survives


def test_result_is_stale_after_invalidation_and_late_after_target() -> None:
    engine = PathBSearchEngine()
    response = engine.result_current(
        SearchResult("IMPROVING", 7, 99, 1, 0, {}, "source"),
        current_frame=98,
    )
    assert response.status == "STALE"
