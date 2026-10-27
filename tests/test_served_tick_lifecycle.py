"""Actual Watch oracle checks across lifecycle and long-body transitions."""

from contextlib import nullcontext

from research.bl_served_search.served_tick_probe import (
    dto_digest,
    new_probe_watch,
    session_state_digest,
)
from research.bl_served_search.tick_trace import TraceRecorder
from research.t1_commitment_20261015.replay import configure_cpu
from tools.sandbox.registry import world_seed


def _history(instrumented: bool) -> tuple[list, list]:
    configure_cpu()
    session, _ = new_probe_watch(world_seed("served-tick-20261026", 3))
    recorder = TraceRecorder(5)
    history, contexts = [], []
    try:
        with recorder.attach(session) if instrumented else nullcontext():
            for tick in range(5):
                if tick == 1:
                    for snake in session.game.snakes[:2]:
                        snake.die()
                        snake.respawn_timer = 0
                if tick == 2:
                    session.reset_game()
                if tick == 3:
                    hero = session.game.snakes[0]
                    size = hero.segment_size
                    width = session.game._game_width // size
                    # A deterministic serpentine geometry, including a long tail.
                    hero.segments = [
                        ((column if row % 2 == 0 else width - column - 1) * size, row * size)
                        for row in range((600 + width - 1) // width)
                        for column in range(width)
                    ][:600]
                    hero.length = len(hero.segments)
                    hero.direction = (1, 0)
                    hero.is_boosting = True
                    hero.invalidate_selection_cache()
                    session.policy._invalidate_cache()
                if instrumented:
                    import time

                    now = time.perf_counter_ns()
                    recorder.begin_tick(tick, now, now)
                session.step()
                dto = session.snapshot()
                if instrumented:
                    recorder.capture_session_context(session)
                    recorder.end_tick()
                # Hashing stays outside every measured span, including in this oracle.
                history.append(
                    (
                        session_state_digest(session),
                        dto_digest(dto),
                        [snake.last_action for snake in session.game.snakes],
                    )
                )
                contexts.append((session.game.frame, session.game.snakes[0].length))
        return history, contexts
    finally:
        session.close(wait=True)


def test_instrumentation_preserves_forced_respawn_reset_and_long_body_state() -> None:
    reference, reference_context = _history(False)
    candidate, candidate_context = _history(True)
    assert candidate == reference
    assert candidate_context == reference_context
    assert reference_context[2][0] == 1  # The reset rewinds the actual game clock.
