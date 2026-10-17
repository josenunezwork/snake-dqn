"""Detached T11 frame-specific shared-survivor observation contracts."""

from research.t11_screen_20261017.probe import LENGTH_FRAMES, LengthCurveObserver


def test_length_curve_uses_post_frame_mass_and_zeroes_deaths() -> None:
    observer = LengthCurveObserver([11, 12])
    observer(
        {
            "seeds": [11, 12],
            "frame": 999,
            "pre": {"alive": [True, True], "logical_mass": [711, 901]},
            "post": {"alive": [True, False], "logical_mass": [712, 901]},
        }
    )
    observer(
        {
            "seeds": [11, 12],
            "frame": 4999,
            "pre": {"alive": [True, False], "logical_mass": [811, 0]},
            "post": {"alive": [True, False], "logical_mass": [812, 0]},
        }
    )

    assert observer.length_at_frames[1000] == {11: 712, 12: 0}
    assert observer.length_at_frames[5000] == {11: 812, 12: 0}
    assert observer.death_frame == {11: None, 12: 1000}
    assert observer.length_at_death == {11: None, 12: 901}


def test_registered_length_curve_has_every_thousand_frame() -> None:
    assert LENGTH_FRAMES == (1000, 2000, 3000, 4000, 5000)
