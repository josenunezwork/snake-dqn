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


def test_terminal_observer_preserves_cpu_h5000_gameplay():
    """Additional observation cannot change historical-world gameplay metrics."""
    import torch

    from research.t2_screen_20261015 import probe_t2
    from src.core.device_manager import DeviceManager

    torch.set_num_threads(1)
    DeviceManager.override_device(torch.device("cpu"))
    args = probe_t2.build_parser().parse_args(
        [
            "--student",
            "/Users/josenunez/Projects/ml/snake-dqn-artifacts/"
            "redesign-m6-mac-20261014/CTRL_s10/final.pth",
            "--arm",
            "student",
            "--mix",
            "scripted",
            "--worlds",
            "2",
            "--round-index",
            "60",
            "--out",
            "/tmp/t4c-unused.json",
            "--name",
            "CTRL_s10",
        ]
    )
    observers = []

    def factory(seeds):
        observer = TerminalLengthObserver(seeds)
        observers.append(observer)
        return observer

    original = probe_t2.probe(args)
    extended = probe_t2.probe(args, observer_factory=factory)
    for field in ("mass", "deaths", "causes", "alive_frames", "death_frame", "length_at_death"):
        assert extended[field] == original[field]
    assert set(observers[0].length_at_5000) == set(original["mass"])
    for seed, length in observers[0].length_at_5000.items():
        assert (length > 0) == (original["deaths"][seed] == 0)
    DeviceManager.reset_for_testing()
