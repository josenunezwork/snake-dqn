"""Functional protocol/parity checks, outside every scored timing panel."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path

import pytest

from research.bl_served_search import dto_split as split
from research.bl_served_search import served_tick_probe as probe


def manifest() -> dict:
    return json.loads(Path(probe.__file__).with_name("served_dto_split_manifest.json").read_text())


def test_new_manifest_is_opt_in_and_frozen() -> None:
    value = manifest()
    assert probe.preflight(value)["cells"] == 12
    assert value["workers"] == value["max_pending"] == 0
    assert value["arms"] == ["timer", "inline", "isolated", split.ARM]
    for key, replacement in (
        ("seeds", [1, 2, 3]),
        ("timeout_seconds", 10),
        ("transport", {}),
        ("order", []),
    ):
        changed = copy.deepcopy(value)
        changed[key] = replacement
        with pytest.raises(ValueError, match="committed fixed panel"):
            probe.preflight(changed)


def test_original_manifest_bytes_preserved() -> None:
    path = Path(probe.__file__).with_name("served_tick_manifest.json")
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "e9cf9bdab21f031a292728a6eb7ee9d2cbcf73a2bfd4471f8093effda9562133"
    )
    assert probe.preflight(json.loads(path.read_text()))["cells"] == 12


def _history(remote: bool) -> list:
    from research.t1_commitment_20261015.replay import configure_cpu

    configure_cpu()
    session, _ = probe.new_probe_watch(manifest()["observer_seed"])
    renderer = None
    commands = split.Commands()
    history = []
    try:
        if remote:
            renderer = split.Renderer(split.bootstrap(session))
        for tick in range(7):
            if tick == 1:
                for snake in session.game.snakes[:2]:
                    snake.die()
                    snake.respawn_timer = 0
            if tick == 2:
                commands.apply(session, 0, "playing", False)
                commands.apply(session, 1, "food", 120)
                commands.apply(session, 2, "reset")
                commands.apply(session, 3, "playing", True)
            if tick == 3:
                # Long body, reset clock and stock cache invalidation coverage.
                hero = session.game.snakes[0]
                size = hero.segment_size
                width = session.game._game_width // size
                hero.segments = [
                    ((column if row % 2 == 0 else width - column - 1) * size, row * size)
                    for row in range((600 + width - 1) // width)
                    for column in range(width)
                ][:600]
                hero.length = 600
                hero.direction = (1, 0)
                hero.is_boosting = True
                hero.invalidate_selection_cache()
                session.policy._invalidate_cache()
            if tick == 4:
                commands.apply(session, 4, "hero", 1)  # FRP vector inspector
            if tick == 5:
                commands.apply(session, 5, "hero", 0)
                commands.apply(session, 6, "reset")
            if tick == 6:
                session.game.snakes[0].die()  # dead hero => no inspector
                commands.apply(session, 7, "playing", False)
            session.step()
            if remote:
                packed = split.pack_export(session, tick, commands.generation)
                assert isinstance(packed, bytes)
                assert renderer.publish(packed, tick, commands.generation)
                service = renderer.drain(commands.generation)
                assert service["sequence"] == tick and not service["stale"]
                dto = json.loads(renderer.last_wire)
            else:
                dto = session.snapshot()
            history.append(
                (
                    probe.session_state_digest(session),
                    probe.dto_digest(dto),
                    [snake.last_action for snake in session.game.snakes],
                )
            )
        return history
    finally:
        if renderer is not None:
            renderer.close()
            assert not renderer.process.is_alive()
            renderer.close()  # idempotent
        session.close(wait=True)


CHECKPOINTS = all(
    Path("saved_snakes", name).is_file()
    for name in ("b8_s3_t7_20261016.pth", "frp3_m3_s12_u60000_20261005.pth")
)


@pytest.mark.skipif(not CHECKPOINTS, reason="requires local frozen released artifacts")
def test_stock_state_dto_and_actions_match_through_lifecycle() -> None:
    assert _history(True) == _history(False)


@pytest.fixture
def session():
    if not CHECKPOINTS:
        pytest.skip("requires local frozen released artifacts")
    from research.t1_commitment_20261015.replay import configure_cpu

    configure_cpu()
    value, _ = probe.new_probe_watch(manifest()["observer_seed"])
    value.step()
    try:
        yield value
    finally:
        value.close(wait=True)


def test_immutable_ownership_and_geometry_overflow(session) -> None:
    packed = split.pack_export(session, 0, 0)
    before = pickle.loads(packed)
    session.game.snakes[0].segments[0] = (9999, 9999)
    session.game.food.clear()
    assert pickle.loads(packed)["food"] == before["food"]
    assert pickle.loads(packed)["snakes"][0]["segments"] == before["snakes"][0]["segments"]
    session.game.snakes[0].segments = [(0, 0)] * (split.MAX_SEGMENTS + 1)
    with pytest.raises(ValueError, match="body overflow"):
        split.pack_export(session, 1, 0)


def test_backpressure_bound_stale_reset_and_shutdown(session) -> None:
    renderer = split.Renderer(split.bootstrap(session))
    try:
        packed = split.pack_export(session, 0, 0)
        # Even after the worker completes, no second publication until ack/poll.
        assert renderer.publish(packed, 0, 0)
        for seq in range(1, 10):
            assert not renderer.publish(packed, seq, 0)
        assert renderer.busy == 9
        assert renderer.pending == (0, 0)
        reply = renderer.drain(1)  # old generation must be classified stale
        assert reply["stale"] == renderer.stale == 1
        assert renderer.last_wire == b""  # never expose a pre-reset DTO
        with pytest.raises(ValueError, match="obsolete"):
            renderer.publish(packed, 10, 0)
        with pytest.raises(ValueError, match="increase"):
            renderer.publish(packed, 0, 1)
        session.reset_game()
        assert renderer.publish(split.pack_export(session, 10, 1), 10, 1)
        assert renderer.drain(1)["generation"] == 1
        # Try-lock has a bounded immediate fallback when the slot is held.
        with renderer.lock:
            assert not renderer.publish(packed, 11, 1)
            assert renderer.poll(1) is None
        assert renderer.publish(split.pack_export(session, 11, 1), 11, 1)
        renderer.close()  # shutdown with an outstanding export
    finally:
        renderer.close()
    assert not renderer.process.is_alive()
    assert renderer.poll(1) is None
    assert not renderer.publish(packed, 12, 1)


def test_failure_retains_attempted_tick_trace(session, monkeypatch) -> None:
    class BrokenRenderer:
        calls = 0

        def __init__(self, frozen):
            pass

        def poll(self, generation):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("injected renderer crash")

        def publish(self, *args):
            return True

        def close(self):
            pass

    monkeypatch.setattr(split, "Renderer", BrokenRenderer)
    monkeypatch.setattr(probe, "new_probe_watch", lambda seed: (session, {}))
    result = split.run_game(dict(seed=1, warmup_ticks=0, measured_ticks=3))
    assert result["status"] == "INVALID" and "injected renderer crash" in result["reason"]
    assert len(result["trace"]) == 2
    assert all(row["end_ns"] >= row["start_ns"] for row in result["trace"])
    assert result["partial_transport"]["ticks"][0]["published"]


def test_worker_render_error_and_abrupt_crash_fail_closed(session) -> None:
    renderer = split.Renderer(split.bootstrap(session))
    try:
        assert renderer.publish(b"malformed-pickle", 0, 0)
        with pytest.raises(RuntimeError, match="failed|crashed"):
            renderer.drain(0)
    finally:
        renderer.close()
    renderer = split.Renderer(split.bootstrap(session))
    try:
        renderer.process.terminate()  # Only the test's own child.
        renderer.process.join(timeout=3)
        assert not renderer.publish(split.pack_export(session, 0, 0), 0, 0)
        with pytest.raises(RuntimeError, match="crashed"):
            renderer.poll(0)
    finally:
        renderer.close()


def test_ordered_commands_reject_skips_duplicates_and_foreign_operations() -> None:
    from types import SimpleNamespace

    observed = []
    session = SimpleNamespace(
        set_playing=lambda value: observed.append(("playing", value)),
        reset_game=lambda: observed.append(("reset",)),
    )
    commands = split.Commands()
    for seq, name in ((1, "reset"), (0, "load_checkpoint")):
        with pytest.raises(ValueError, match="unordered or unsupported"):
            commands.apply(session, seq, name)
    commands.apply(session, 0, "playing", False)
    commands.apply(session, 1, "reset")
    with pytest.raises(ValueError):
        commands.apply(session, 1, "reset")
    assert observed == [("playing", False), ("reset",)]
    assert commands.generation == 1 and commands.next_sequence == 2


def test_tiny_spawned_arm_matches_inline(session) -> None:
    from research.bl_served_search.isolated_game import run_isolated_arm

    spec = dict(arm="inline", seed=manifest()["observer_seed"], warmup_ticks=1, measured_ticks=2)
    inline = probe.run_inline_arm(spec)
    candidate = run_isolated_arm({**spec, "arm": split.ARM}, timeout_seconds=60)
    assert probe._verify_pair(inline, candidate)["pass"]
    assert len(candidate["trace"]) == len(candidate["dto_transport"]["ticks"]) == 2
    # Smoke is functional evidence, even if a shared Mac causes a busy fallback.
    for row, cost in zip(candidate["trace"], candidate["dto_transport"]["ticks"]):
        assert row["start_ns"] <= cost["copy_start_ns"] <= cost["publish_end_ns"] <= row["end_ns"]


def test_invalid_bootstrap_startup_is_bounded() -> None:
    with pytest.raises(RuntimeError, match="startup failed"):
        split.Renderer(b"invalid", startup_seconds=10)


def transport_fixture() -> dict:
    ticks, services, trace = [], [], []
    for index in range(2):
        start = 1000 + index * 1000
        ticks.append(
            dict(
                sequence=index + 1,
                published=True,
                export_bytes=100,
                copy_start_ns=start + 10,
                copy_end_ns=start + 20,
                copy_wall_ns=10,
                copy_cpu_ns=8,
                publish_start_ns=start + 20,
                publish_end_ns=start + 30,
                publish_wall_ns=10,
                publish_cpu_ns=8,
                poll_start_ns=start,
                poll_end_ns=start + 5,
                poll_wall_ns=5,
                poll_cpu_ns=4,
            )
        )
        trace.append(dict(start_ns=start, end_ns=start + 100))
    for sequence in range(3):
        services.append(
            dict(
                sequence=sequence,
                generation=0,
                stale=0,
                publish_ns=100 + sequence * 100,
                receive_ns=110 + sequence * 100,
                end_ns=120 + sequence * 100,
                poll_ns=150 + sequence * 100,
                service_wall_ns=10,
                service_cpu_ns=8,
                service_latency_ns=20,
                wire_bytes=100,
                rss_bytes=1000,
            )
        )
    return dict(
        measured_ticks=2,
        warmup_ticks=1,
        trace=trace,
        dto_transport=dict(
            ticks=ticks,
            services=services,
            capacity=1,
            max_export_bytes=split.MAX_EXPORT_BYTES,
            max_wire_bytes=split.MAX_WIRE_BYTES,
            busy=0,
            crashes=0,
            stale=0,
            final_diagnostic_publication=False,
            completed=3,
            startup_ns=1,
            game_peak_rss_bytes=2000,
            endpoint="fixture",
        ),
    )


def test_transport_reducer_reports_cold_warm_and_charged_costs() -> None:
    from research.bl_served_search.dto_split_report import validate_transport

    report = validate_transport(transport_fixture())
    assert report["warm_service_wall_ns"]["count"] == 2
    assert report["cold_first_service"]["sequence"] == 0
    assert report["costs"]["copy_wall_ns"]["p95"] == 10


@pytest.mark.parametrize("damage", ["lost", "busy", "stale", "unpaid", "identity", "size", "clock"])
def test_transport_reducer_refuses_invalid_exposure_before_timing(damage) -> None:
    from research.bl_served_search.dto_split_report import validate_transport

    result = transport_fixture()
    data = result["dto_transport"]
    if damage == "lost":
        data["services"].pop()
    elif damage == "busy":
        data["busy"] = 1
    elif damage == "stale":
        data["services"][0]["stale"] = 1
    elif damage == "unpaid":
        data["ticks"][0]["publish_end_ns"] = result["trace"][0]["end_ns"] + 1
    elif damage == "identity":
        data["services"][1]["sequence"] = 0
    elif damage == "size":
        data["ticks"][0]["export_bytes"] = split.MAX_EXPORT_BYTES + 1
    else:
        data["services"][0]["publish_ns"] = 10000
    with pytest.raises(ValueError):
        validate_transport(result)


def test_split_panel_reducer_preserves_invalid_fail_pass_order(monkeypatch) -> None:
    from research.bl_served_search import dto_split_report, tick_trace

    value = manifest()
    evidence = dict(
        actions=[[1]],
        state_digest={"initial": "a", "final": "b"},
        dto_digest={"final": "c"},
        processing_wall_ns=[10] * 120,
        processing_thread_cpu_ns=[5] * 120,
    )
    binding = dict(commit="a" * 40, tracked_dirty="", source_sha256={"split": "b" * 64})
    raw = dict(
        manifest=value,
        source_before=binding,
        source_after=binding,
        host={"status": "COMPLETE"},
        cells=[
            dict(
                cell=cell,
                result=dict(
                    **copy.deepcopy(evidence),
                    arm=cell["arm"],
                    seed=value["seeds"][cell["seed_index"]],
                    fps=24,
                    warmup_ticks=200,
                    measured_ticks=2400,
                    trace=[],
                ),
            )
            for cell in value["order"]
        ],
        observer_control=dict(
            ticks=120,
            arms=[
                dict(name=name, result=copy.deepcopy(evidence))
                for name in value["instrumented_observer_order"]
            ],
        ),
    )
    monkeypatch.setattr(dto_split_report, "validate_transport", lambda result: {})
    monkeypatch.setattr(tick_trace, "reduce_trace", lambda *a, **kw: {"verdict": "PASS"})
    assert probe.reduce_trace(raw)["verdict"] == "PASS"
    monkeypatch.setattr(tick_trace, "reduce_trace", lambda *a, **kw: {"verdict": "FAIL"})
    assert probe.reduce_trace(raw)["verdict"] == "FAIL"
    raw["cells"][3]["result"]["actions"] = [[2]]
    assert probe.reduce_trace(raw)["verdict"] == "INVALID"
    raw["cells"].pop()
    assert probe.reduce_trace(raw)["verdict"] == "INCOMPLETE"


def _abandon_renderer(frozen: bytes, connection, locked: bool) -> None:
    import os

    renderer = split.Renderer(frozen)
    if locked:
        renderer.lock.acquire()  # Simulate the game dying midway through publication.
    connection.send(renderer.process.pid)
    connection.close()
    os._exit(0)  # Deliberately bypass finally, as a crashed authoritative owner does.


@pytest.mark.parametrize("locked", [False, True])
def test_renderer_exits_when_authoritative_owner_crashes(session, locked) -> None:
    import multiprocessing
    import os
    import time

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    owner = context.Process(
        target=_abandon_renderer, args=(split.bootstrap(session), child, locked)
    )
    owner.start()
    child.close()
    try:
        assert parent.poll(10)
        renderer_pid = parent.recv()
        owner.join(timeout=5)
        assert owner.exitcode == 0
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(renderer_pid, 0)  # Existence check for this test's own child only.
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            pytest.fail("orphan DTO renderer survived authoritative owner crash")
    finally:
        parent.close()
        if owner.is_alive():
            owner.terminate()
            owner.join(timeout=3)
