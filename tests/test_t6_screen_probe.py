"""Detached T6 H5000 terminal-mass observer contracts."""

from research.t6_screen_20261017.probe import TerminalLengthObserver


def test_terminal_mass_is_post_frame_and_dead_hero_is_zero() -> None:
    observer = TerminalLengthObserver([11, 12])
    observer(
        {
            "seeds": [11, 12],
            "frame": 4999,
            "pre": {"alive": [True, True], "logical_mass": [711, 901]},
            "post": {"alive": [True, False], "logical_mass": [712, 901]},
        }
    )

    assert observer.length_at_5000 == {11: 712, 12: 0}
    assert observer.death_frame == {11: None, 12: 5000}
    assert observer.length_at_death == {11: None, 12: 901}
