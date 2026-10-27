"""Opt-in W0/K0 authoritative game / inspector DTO process experiment.

One fixed shared slot, protected by a process lock, has explicit ownership:
FREE -> REQUEST -> REPLY -> FREE. The game only tries the lock once per tick;
it never waits for the renderer. Immutable pickle bytes contain value projections,
never live game/policy objects. JSON WebSocket payload construction stays remote.
This module is research-only; it is not imported by production serving code.
"""

from __future__ import annotations

import ctypes
import json
import multiprocessing as mp
import os
import pickle
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

ARM = "dto_split_w0_k0"
STUDY = "served-dto-split-20261027"
MAX_EXPORT_BYTES = 4 * 1024 * 1024
MAX_WIRE_BYTES = 8 * 1024 * 1024
MAX_SEGMENTS = 10400
MAX_FOOD = 20000
SESSION_FIELDS = (
    "playing speed speed_ceiling mode hero_id checkpoint_path config_path obs_spec last_error "
    "last_error_overridable reward_override_active serving_contract watch_roster "
    "watch_roster_reason viewer_count"
).split()
GAME_FIELDS = (
    "frame alive_snakes episode_food_eaten episode_deaths episode_kills episode_best_length "
    "_game_width _game_height"
).split()
SNAKE_FIELDS = (
    "id color color_name is_alive is_boosting length segments head current_loss last_action"
).split()
# Phase, input size, output size, sequence, generation, publish, worker receive,
# worker finish, service wall, service CPU, output JSON bytes, worker RSS bytes.
PHASE, SIZE, OUT_SIZE, SEQ, GEN, PUB, RECV, DONE, WALL, CPU, WIRE, RSS = range(12)


def _values(obj: Any, names: list[str]) -> dict[str, Any]:
    return {name: getattr(obj, name) for name in names if hasattr(obj, name)}


def pack_export(session: Any, sequence: int, generation: int) -> bytes:
    """Copy bounded geometry and exact inspector inputs, charged to the game tick."""
    import numpy as np

    from web.backend.serialize import _hero_state_tensor

    game = session.game
    if len(game.snakes) != 6 or len(game.food) > MAX_FOOD:
        raise ValueError("export geometry overflow")
    if any(len(s.segments) > MAX_SEGMENTS for s in game.snakes):
        raise ValueError("export body overflow")
    if session.mode != "watch" or getattr(session, "pathb", None) is not None:
        raise ValueError("DTO split only supports unassisted Watch6")
    hero = next((s for s in game.snakes if s.id == session.hero_id), None)
    observation = state = None
    if hero is not None and hero.is_alive:
        if hero.policy is session.policy:
            # snapshot's hero_activations also calls this. It may refresh the
            # live observation/dispatch cache; omitting it changes future state.
            session.policy._ensure_frame()
            observation = session.policy._obs
        else:
            state = _hero_state_tensor(hero, [s for s in game.snakes if s is not hero], game.food)
            if state is not None:
                state = np.asarray(state.detach().cpu().numpy())
    payload = {
        "sequence": sequence,
        "generation": generation,
        "session": _values(session, SESSION_FIELDS),
        "game": _values(game, GAME_FIELDS),
        "snakes": [_values(s, SNAKE_FIELDS) for s in game.snakes],
        "food": game.food,
        "max_food": game.food_manager.max_food,
        "veto": session.safety_veto_state(),
        "observation": observation,
        "id_to_row": session.policy._id_to_row,
        "hero_state": state,
    }
    packed = pickle.dumps(payload, protocol=5)
    if len(packed) > MAX_EXPORT_BYTES:
        raise ValueError("immutable export byte overflow")
    return packed


def bootstrap(session: Any) -> bytes:
    """Copy frozen inspector models once, outside pacing; no live game crosses."""
    from src.core.game_config import get_config

    return pickle.dumps(
        (session.policy.net, session.watch_opponent_policy.dqn, get_config()), protocol=5
    )


class _InspectorPolicy(SimpleNamespace):
    def _ensure_frame(self) -> None:
        """The exact observation was already frozen by the authoritative game."""

    def hero_activations(self, snake_id: int) -> Any:
        from web.backend.ego2s_policy import Ego2sServingPolicy

        return Ego2sServingPolicy.hero_activations(self, snake_id)


class _View(SimpleNamespace):
    def control_state(self) -> dict[str, Any]:
        from web.backend.session import GameSession

        return GameSession.control_state(self)

    def play_state(self) -> None:
        return None  # This arm refuses every mode except Watch.


class _Veto(SimpleNamespace):
    def to_dict(self) -> dict[str, Any]:
        return self.value


def render_export(packed: bytes, models: tuple[Any, Any, Any]) -> tuple[dict[str, Any], bytes]:
    """Use stock inspector and DTO code against a detached value-only facade."""
    import torch

    from web.backend.serialize import build_frame

    row = pickle.loads(packed)
    ego, vector, _config = models
    policy = _InspectorPolicy(
        net=ego,
        device=torch.device("cpu"),
        epsilon=0.0,
        training=False,
        _obs=row["observation"],
        _id_to_row=row["id_to_row"],
    )
    other = SimpleNamespace(dqn=vector, device=torch.device("cpu"), obs_spec="vector61")
    snakes = [SimpleNamespace(**fields) for fields in row["snakes"]]
    for index, snake in enumerate(snakes):
        snake.policy = policy if index == 0 else other
        if snake.id == row["session"]["hero_id"] and row["hero_state"] is not None:
            snake.last_state = torch.as_tensor(row["hero_state"])
    game = SimpleNamespace(
        **row["game"],
        snakes=snakes,
        food=row["food"],
        food_manager=SimpleNamespace(max_food=row["max_food"], food=row["food"]),
    )
    session = _View(
        **row["session"],
        game=game,
        policy=policy,
        _lock=threading.RLock(),
        safety_veto=_Veto(value=row["veto"]) if row["veto"] is not None else None,
    )
    dto = build_frame(session)
    wire = json.dumps(dto, separators=(",", ":"), allow_nan=False).encode()
    if len(wire) > MAX_WIRE_BYTES:
        raise ValueError("WebSocket JSON byte overflow")
    return dto, wire


@contextmanager
def _owner_lock(lock: Any, owner_pid: int) -> Iterator[bool]:
    """A dead game may abandon the lock; the renderer must still exit."""
    while os.getppid() == owner_pid:
        if lock.acquire(timeout=0.05):
            try:
                yield True
            finally:
                lock.release()
            return
    yield False


def _worker(
    slot: Any,
    output: Any,
    meta: Any,
    lock: Any,
    stop: Any,
    ready: Any,
    frozen: bytes,
    owner_pid: int,
) -> None:
    """Own inspector execution and JSON construction; never step a game."""
    import resource

    from research.t1_commitment_20261015.replay import configure_cpu
    from src.core.game_config import initialize_config

    try:
        configure_cpu()
        models = pickle.loads(frozen)
        initialize_config(models[2])
        models[0].eval()
        models[1].eval()
        ready.send({"ready": True})
        while not stop.value and os.getppid() == owner_pid:
            with _owner_lock(lock, owner_pid) as owned:
                if not owned:
                    return
                if meta[PHASE] == 1:
                    received, cpu = time.perf_counter_ns(), time.thread_time_ns()
                    request = bytes(slot[: meta[SIZE]])
                    sequence, generation = meta[SEQ], meta[GEN]
                else:
                    request = None
            if request is None:
                time.sleep(0.001)
                continue
            identity = pickle.loads(request)
            if (identity["sequence"], identity["generation"]) != (sequence, generation):
                raise ValueError("export/header identity mismatch")
            _dto, wire = render_export(request, models)
            with _owner_lock(lock, owner_pid) as owned:
                if not owned:
                    return
                output[: len(wire)] = wire
                meta[OUT_SIZE] = len(wire)
                finished = time.perf_counter_ns()
                cpu = time.thread_time_ns() - cpu
                meta[RECV], meta[DONE] = received, finished
                meta[WALL], meta[CPU], meta[WIRE] = finished - received, cpu, len(wire)
                # macOS ru_maxrss is bytes; Linux is KiB. No shell on the tick.
                import sys

                meta[RSS] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                    1 if sys.platform == "darwin" else 1024
                )
                meta[PHASE] = 2  # completion is published last under the lock
    except BaseException as exc:
        with _owner_lock(lock, owner_pid) as owned:
            if owned:
                meta[PHASE] = -1
        try:
            ready.send({"error": f"{type(exc).__name__}: {exc}"})
        except (OSError, EOFError):
            pass
    finally:
        ready.close()


class Renderer:
    """One in-flight export; try-lock publication/poll has no blocking tick path."""

    def __init__(self, frozen: bytes, startup_seconds: float = 60) -> None:
        context = mp.get_context("spawn")
        self.slot = context.RawArray(ctypes.c_ubyte, MAX_EXPORT_BYTES)
        self.output = context.RawArray(ctypes.c_ubyte, MAX_WIRE_BYTES)
        self.meta = context.RawArray(ctypes.c_int64, 12)
        self.lock = context.Lock()
        # A killed worker must not leave an Event's internal condition locked.
        self.stop = context.RawValue(ctypes.c_bool, False)
        self.parent, child = context.Pipe(duplex=False)
        self.process = context.Process(
            target=_worker,
            args=(
                self.slot,
                self.output,
                self.meta,
                self.lock,
                self.stop,
                child,
                frozen,
                os.getpid(),
            ),
        )
        self.pending: tuple[int, int] | None = None
        self.completed = 0
        self.busy = self.crashes = self.stale = 0
        self.last_sequence = -1
        self.generation = 0
        self.last_wire = b""
        self.closed = False
        started = time.perf_counter_ns()
        self.process.start()
        child.close()
        try:
            if not self.parent.poll(startup_seconds):
                raise TimeoutError("DTO worker startup timeout")
            message = self.parent.recv()
            if not message.get("ready"):
                raise RuntimeError(f"DTO worker startup failed: {message}")
            self.startup_ns = time.perf_counter_ns() - started
        except BaseException:
            self.close()
            raise

    def publish(self, packed: bytes, sequence: int, generation: int) -> bool:
        if len(packed) > MAX_EXPORT_BYTES:
            raise ValueError("export byte overflow")
        if self.closed or not self.process.is_alive():
            self.crashes += 1
            return False
        if generation < self.generation:
            raise ValueError("cannot publish an obsolete reset generation")
        if not self.lock.acquire(False):
            self.busy += 1
            return False
        try:
            if self.meta[PHASE] == -1:
                self.crashes += 1
                return False
            if self.pending is not None or self.meta[PHASE] != 0:
                self.busy += 1
                return False
            if sequence <= self.last_sequence:
                raise ValueError("publication sequence must increase across resets")
            if generation != self.generation:
                self.generation = generation
                self.last_wire = b""
            self.meta[PUB] = time.perf_counter_ns()
            self.slot[: len(packed)] = packed
            self.meta[SIZE], self.meta[SEQ], self.meta[GEN] = len(packed), sequence, generation
            self.pending = sequence, generation
            self.last_sequence = sequence
            self.meta[PHASE] = 1  # ownership transfers after all bytes/metadata
            return True
        finally:
            self.lock.release()

    def poll(self, generation: int) -> dict[str, int] | None:
        if self.closed:
            return None
        if generation < self.generation:
            raise ValueError("cannot poll an obsolete reset generation")
        if generation != self.generation:
            self.generation = generation
            self.last_wire = b""
        if not self.process.is_alive():
            self.crashes += 1
            raise RuntimeError("DTO worker crashed")
        if not self.lock.acquire(False):
            return None
        try:
            if self.meta[PHASE] == -1:
                self.crashes += 1
                raise RuntimeError("DTO worker failed")
            if self.meta[PHASE] != 2:
                return None
            seq, gen = self.meta[SEQ], self.meta[GEN]
            if self.pending != (seq, gen):
                raise RuntimeError("DTO reply identity mismatch")
            self.pending = None
            self.meta[PHASE] = 0
            self.completed += 1
            if gen != generation:
                self.stale += 1
            return dict(
                sequence=seq,
                generation=gen,
                publish_ns=self.meta[PUB],
                receive_ns=self.meta[RECV],
                end_ns=self.meta[DONE],
                service_wall_ns=self.meta[WALL],
                service_cpu_ns=self.meta[CPU],
                service_latency_ns=self.meta[DONE] - self.meta[PUB],
                poll_ns=time.perf_counter_ns(),
                wire_bytes=self.meta[WIRE],
                rss_bytes=self.meta[RSS],
                stale=int(gen != generation),
            )
        finally:
            self.lock.release()

    def drain(self, generation: int, seconds: float = 10) -> dict[str, int] | None:
        """Outside timing only: bounded wait and final wire copy."""
        deadline = time.monotonic() + seconds
        while self.pending is not None:
            reply = self.poll(generation)
            if reply is not None:
                if not reply["stale"]:
                    with self.lock:
                        self.last_wire = bytes(self.output[: self.meta[OUT_SIZE]])
                return reply
            if time.monotonic() >= deadline:
                raise TimeoutError("DTO renderer drain timeout")
            time.sleep(0.001)
        return None

    def close(self) -> None:
        """Only stop the child this instance created, with bounded escalation."""
        if self.closed:
            return
        self.closed = True
        self.stop.value = True
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)
        self.parent.close()
        if self.process.is_alive():
            raise RuntimeError("DTO worker failed shutdown")


class Commands:
    """Single authoritative owner applies ordered controls before step/export."""

    def __init__(self) -> None:
        self.next_sequence = 0
        self.generation = 0

    def apply(self, session: Any, sequence: int, name: str, value: Any = None) -> None:
        methods = {
            "playing": "set_playing",
            "hero": "set_hero",
            "food": "set_food",
            "speed": "set_speed",
            "reset": "reset_game",
        }
        if sequence != self.next_sequence or name not in methods:
            raise ValueError("unordered or unsupported authoritative command")
        method = getattr(session, methods[name])
        method() if name == "reset" else method(value)
        self.next_sequence += 1
        if name == "reset":
            self.generation += 1


def run_game(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Child-owned game loop: step + copy + try-publish + poll all on tick clock."""
    import resource
    import sys

    from research.bl_served_search import served_tick_probe as probe
    from research.bl_served_search.tick_trace import TraceRecorder
    from research.t1_commitment_20261015.replay import configure_cpu

    configure_cpu()
    session, roster = probe.new_probe_watch(int(spec["seed"]))
    renderer = None
    measured, warmup = int(spec["measured_ticks"]), int(spec["warmup_ticks"])
    recorder = TraceRecorder(measured + 8)
    initial = None
    actions = [None] * measured
    processing, processing_cpu = [0] * measured, [0] * measured
    transport = [None] * measured
    services = []
    try:
        renderer = Renderer(bootstrap(session))
        sequence = 0
        for _ in range(warmup):
            session.step()
            packed = pack_export(session, sequence, 0)
            if not renderer.publish(packed, sequence, 0):
                raise RuntimeError("warmup publication refused")
            services.append(renderer.drain(0))
            sequence += 1
        initial = probe.session_state_digest(session)
        with recorder.attach(session):
            epoch = time.perf_counter_ns()
            for index in range(measured):
                target = epoch + index * 1_000_000_000 // probe.FPS
                wake = probe._sleep_until(target)
                started, cpu = time.perf_counter_ns(), time.thread_time_ns()
                recorder.begin_tick(index, target, wake)
                try:
                    poll_start, poll_cpu = time.perf_counter_ns(), time.thread_time_ns()
                    reply = renderer.poll(0)
                    poll_end, poll_cpu_end = time.perf_counter_ns(), time.thread_time_ns()
                    if reply is not None:
                        services.append(reply)
                    session.step()
                    pack_start, pack_cpu = time.perf_counter_ns(), time.thread_time_ns()
                    with recorder.span("snapshot"):
                        packed = pack_export(session, sequence, 0)
                    pack_end, pack_cpu_end = time.perf_counter_ns(), time.thread_time_ns()
                    publish_start, publish_cpu = time.perf_counter_ns(), time.thread_time_ns()
                    published = renderer.publish(packed, sequence, 0)
                    publish_end, publish_cpu_end = time.perf_counter_ns(), time.thread_time_ns()
                    actions[index] = [s.last_action for s in session.game.snakes]
                    transport[index] = dict(
                        sequence=sequence,
                        published=published,
                        export_bytes=len(packed),
                        copy_start_ns=pack_start,
                        copy_end_ns=pack_end,
                        publish_start_ns=publish_start,
                        publish_end_ns=publish_end,
                        poll_start_ns=poll_start,
                        poll_end_ns=poll_end,
                        copy_wall_ns=pack_end - pack_start,
                        copy_cpu_ns=pack_cpu_end - pack_cpu,
                        publish_wall_ns=publish_end - publish_start,
                        publish_cpu_ns=publish_cpu_end - publish_cpu,
                        poll_wall_ns=poll_end - poll_start,
                        poll_cpu_ns=poll_cpu_end - poll_cpu,
                    )
                    recorder.capture_session_context(session)
                finally:
                    recorder.end_tick(time.perf_counter_ns())
                    processing[index] = time.perf_counter_ns() - started
                    processing_cpu[index] = time.thread_time_ns() - cpu
                sequence += 1
        # Hashes, drains and any final diagnostic publication are outside scoring.
        final = probe.session_state_digest(session)
        reply = renderer.drain(0)
        if reply is not None:
            services.append(reply)
        final_diagnostic = not transport[-1]["published"]
        if final_diagnostic:
            if not renderer.publish(packed, sequence, 0):
                raise RuntimeError("final diagnostic publication refused")
            services.append(renderer.drain(0))
        wire = json.loads(renderer.last_wire)
        return dict(
            schema=probe.SCHEMA,
            arm=ARM,
            seed=int(spec["seed"]),
            fresh_process=True,
            effective_seed_uint32=int(spec["seed"]) & 0xFFFFFFFF,
            fps=probe.FPS,
            warmup_ticks=warmup,
            measured_ticks=measured,
            identity=probe._identity(session, roster),
            trace=recorder.rows(),
            state_digest={"initial": initial, "final": final},
            dto_digest={"final": probe.dto_digest(wire)},
            actions=actions,
            processing_wall_ns=processing,
            processing_thread_cpu_ns=processing_cpu,
            dto_transport=dict(
                capacity=1,
                max_export_bytes=MAX_EXPORT_BYTES,
                max_wire_bytes=MAX_WIRE_BYTES,
                startup_ns=renderer.startup_ns,
                ticks=transport,
                services=services,
                game_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                completed=renderer.completed,
                busy=renderer.busy,
                crashes=renderer.crashes,
                stale=renderer.stale,
                final_diagnostic_publication=final_diagnostic,
                endpoint="stock inspector/DTO + WebSocket JSON construction; no socket client",
            ),
            client_receive="NOT_MEASURED_NO_WEBSOCKET_CLIENT",
            frame_send="NOT_MEASURED_NO_WEBSOCKET_CLIENT",
        )
    except Exception as exc:
        # Keep the attempted failing cell and every completed/failed tick.
        # The coordinator seals this payload exactly like a successful cell;
        # reduction checks INVALID status before considering any timing bar.
        return dict(
            schema=probe.SCHEMA,
            arm=ARM,
            seed=int(spec["seed"]),
            fps=probe.FPS,
            warmup_ticks=warmup,
            measured_ticks=measured,
            status="INVALID",
            reason=f"{type(exc).__name__}: {exc}",
            identity=probe._identity(session, roster),
            trace=recorder.rows(),
            actions=actions,
            state_digest={"initial": initial, "final": None},
            dto_digest={"final": None},
            processing_wall_ns=processing,
            processing_thread_cpu_ns=processing_cpu,
            partial_transport={"ticks": transport, "services": services},
        )
    finally:
        if renderer is not None:
            renderer.close()
        session.close(wait=True)
