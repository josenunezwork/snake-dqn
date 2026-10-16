"""Detached terminal-length observer preserves T2 death accounting."""

from research.t3_screen_20261015.probe_t4c import TerminalLengthObserver


def test_terminal_length_is_post_frame_and_dead_is_zero():
    observer = TerminalLengthObserver([11, 12])
    snapshot = {
        "seeds": [11, 12],
        "frame": 4998,
        "pre": {"alive": [True, True], "logical_mass": [710, 900]},
        "post": {"alive": [True, True], "logical_mass": [711, 901]},
    }
    observer(snapshot)
    assert observer.length_at_5000 == {}
    snapshot.update(
        frame=4999,
        pre={"alive": [True, True], "logical_mass": [711, 901]},
        post={"alive": [True, False], "logical_mass": [712, 901]},
    )
    observer(snapshot)
    assert observer.length_at_5000 == {11: 712, 12: 0}
    assert observer.death_frame == {11: None, 12: 5000}
    assert observer.length_at_death == {11: None, 12: 901}
    assert observer.alive_frames == {11: 2, 12: 1}
    assert snapshot["post"]["logical_mass"] == [712, 901]
