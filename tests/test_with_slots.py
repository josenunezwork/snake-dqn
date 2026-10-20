"""Focused safety tests for the shared CPU-slot wrapper."""

from __future__ import annotations

from research.perf_sim_20261007 import with_slots


def test_rechecks_power_after_waiting_for_slots_and_releases(monkeypatch) -> None:
    """A power change while waiting prevents child execution and frees the slots."""

    handles = [object()]
    events: list[object] = []
    power_states = iter((True, False))

    def on_ac() -> bool:
        value = next(power_states)
        events.append(f"power-{value}")
        return value

    monkeypatch.setattr(with_slots.sys, "argv", ["with_slots.py", "--slots", "1", "--", "child"])
    monkeypatch.setattr(with_slots, "on_ac", on_ac)
    monkeypatch.setattr(with_slots, "lid_open", lambda: events.append("lid") or True)
    monkeypatch.setattr(
        with_slots.dev_screen,
        "acquire_cpu_slots",
        lambda *args, **kwargs: events.append("acquire") or handles,
    )
    monkeypatch.setattr(
        with_slots.dev_screen,
        "release_cpu_slots",
        lambda released: events.append(("release", released)),
    )
    monkeypatch.setattr(
        with_slots.subprocess,
        "call",
        lambda *args, **kwargs: events.append("child") or 0,
    )

    assert with_slots.main() == 2
    assert events == ["power-True", "lid", "acquire", "power-False", ("release", handles)]
