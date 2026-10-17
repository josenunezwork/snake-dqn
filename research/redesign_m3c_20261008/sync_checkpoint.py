"""Opt-in full-state checkpoint controller for the deterministic synchronous trainer."""

from __future__ import annotations

import json
import time
from collections import deque
from types import SimpleNamespace
from typing import Any

from research.redesign_m3c_20261008.actor_snapshot import (
    capture_actor_state,
    restore_actor_state,
)
from research.redesign_m3c_20261008.async_checkpoint import (
    AsyncCheckpointer,
    capture_state,
    load_latest_snapshot,
)
from research.redesign_m3c_20261008.demo_snapshot import capture_demo_state, restore_demo_state
from research.redesign_m3c_20261008.execution_snapshot import (
    capture_policy_execution,
    restore_policy_execution,
)


class SyncCheckpointSession:
    """All methods run only when full-state save or resume was explicitly requested."""

    def __init__(
        self,
        args,
        torch,
        online,
        target,
        optimizer,
        rng,
        counters,
        schedules,
        replay,
        ring,
        demos,
        log_path,
    ) -> None:
        from research.redesign_m3c_20261008.train_m3c import restore_full_state

        self.args, self.torch = args, torch
        self.online, self.target, self.optimizer = online, target, optimizer
        self.rng, self.counters, self.schedules = rng, counters, schedules
        self.replay, self.ring, self.demos = replay, ring, demos
        self.log_path = log_path
        self.events: list[dict] = []
        self.controls: dict = {}
        self.pending_actor = None
        self.resumed = False
        self.resumed_with_replay = False
        self.complete: bool | None = None
        self.last_submitted = None
        if args.resume_full_state is not None:
            saved, manifest = load_latest_snapshot(args.resume_full_state)
            if saved.get("schedules") != schedules.__dict__:
                raise ValueError("full-state schedule recipe differs")
            extra = saved.get("extra", {})
            if extra.get("trainer") != "sync":
                raise ValueError("synchronous resume requires a synchronous full-state snapshot")
            signature = self._signature()
            if extra.get("recipe") != signature:
                raise ValueError("full-state synchronous recipe differs")
            self.resumed_with_replay = restore_full_state(
                torch, saved, online, target, optimizer, rng, replay, None, [ring]
            )
            restore_demo_state(demos, extra["demos"])
            counters.clear()
            counters.update(saved["counters"])
            self.controls = extra["controls"]
            self.pending_actor = extra["actor"]
            self.resumed = True
            if not self.resumed_with_replay:
                counters["since_update"] = 0
                # IDs in pending n-step windows refer to omitted observations.
                self.pending_actor = dict(self.pending_actor)
            self._record(
                {
                    "event": "resume-full-state",
                    "source": str(args.resume_full_state),
                    "snapshot": manifest["snapshot"],
                    "transitions": counters["transitions"],
                    "mode": (
                        "exact-full-state" if self.resumed_with_replay else "resumed-without-replay"
                    ),
                    "warmup_rows": 0 if self.resumed_with_replay else args.warmup,
                }
            )
        self.next_snapshot = (
            ((counters["transitions"] // args.full_state_every) + 1) * args.full_state_every
            if args.full_state_every
            else None
        )
        self.writer = (
            AsyncCheckpointer(args.ckpt_local_dir, args.ckpt_durable_dir, keep=2)
            if args.full_state_every
            else None
        )

    def _signature(self) -> dict:
        # Paths/output/cadence/device can differ after VM migration. Training math cannot.
        ignored = {
            "out",
            "init",
            "demo_data",
            "device",
            "ckpt_local_dir",
            "ckpt_durable_dir",
            "resume_full_state",
            "full_state_every",
            "full_state_replay",
            "transitions",
            "checkpoint_every",
            "no_guard",
            "ring_dir",
            "demo_mmap",
            "max_wall_seconds",
        }
        return {key: str(value) for key, value in vars(self.args).items() if key not in ignored}

    def _record(self, event: dict) -> None:
        self.events.append(event)
        with self.log_path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def restore_segment(self, sim, runtime, pend, last_obs, policies=None):
        """Restore after build_segment; returns n-step queues and last observation IDs."""
        if self.pending_actor is None:
            return pend, last_obs
        state = SimpleNamespace(pend=pend, last_obs=last_obs)
        restored = restore_actor_state(
            self.pending_actor, sim, runtime, state, self.rng, self.online
        )
        restore_policy_execution(policies or {}, restored["extra"].get("policies", {}))
        self.pending_actor = None
        if not self.resumed_with_replay:
            state.pend = [deque() for _ in range(sim.E)]
            state.last_obs.fill(-1)
        return state.pend, state.last_obs

    def capture(
        self,
        sim,
        runtime,
        pend,
        last_obs,
        segment: int,
        next_frame: int,
        gamma_seg: float,
        next_log: int,
        next_ckpt: int,
        *,
        force: bool = False,
        policies=None,
    ) -> bool:
        from research.redesign_m3c_20261008.train_m3c import full_state_payload

        if self.writer is None:
            return False
        for event in self.writer.poll_events():
            self._record(event)
        transitions = self.counters["transitions"]
        if not force and transitions < self.next_snapshot:
            return False
        if self.writer.busy:
            if not force or not self.writer.wait_local(30):
                return False
        thread = self.demos._thread if self.demos is not None else None
        if not force and thread is not None and thread.is_alive():
            return False
        begin = time.perf_counter()
        actor = capture_actor_state(
            sim,
            runtime,
            SimpleNamespace(pend=pend, last_obs=last_obs),
            self.rng,
            self.online,
            segment,
            next_frame,
            0,
            extra={"policies": capture_policy_execution(policies or {})},
        )
        payload = full_state_payload(
            self.torch,
            self.online,
            self.target,
            self.optimizer,
            self.rng,
            self.counters,
            self.schedules,
            self.replay,
            None,
            [self.ring],
            self.args.full_state_replay,
            extra={
                "trainer": "sync",
                "recipe": self._signature(),
                "actor": actor,
                "demos": capture_demo_state(self.demos),
                "controls": {
                    "segment": segment,
                    "next_frame": next_frame,
                    "gamma_seg": gamma_seg,
                    "next_log": next_log,
                    "next_ckpt": next_ckpt,
                },
            },
        )
        detached = capture_state(payload)
        stall = (time.perf_counter() - begin) * 1000
        accepted = self.writer.submit(
            detached,
            transitions,
            {
                "replay": self.args.full_state_replay,
                "capture_stall_ms": stall,
                "trainer": "sync",
            },
        )
        if accepted:
            self.last_submitted = (transitions, self.counters["updates"])
            self.next_snapshot = (
                (transitions // self.args.full_state_every) + 1
            ) * self.args.full_state_every
        self._record(
            {
                "event": "full-state-snapshot",
                "transitions": transitions,
                "capture_stall_ms": stall,
                "accepted": accepted,
            }
        )
        return accepted

    def close(self) -> bool | None:
        if self.writer is None:
            return None
        self.complete = self.writer.close(30)
        while self.writer.workers_alive:
            for event in self.writer.poll_events():
                self._record(event)
            self.complete = self.writer.close(30)
        for event in self.writer.poll_events():
            self._record(event)
        self.complete = bool(self.complete) and self.last_submitted == (
            self.counters["transitions"],
            self.counters["updates"],
        )
        return self.complete

    def summary(self) -> dict[str, Any]:
        return {
            "full_state_events": self.events[-200:],
            "full_state_complete": self.complete,
            "resumed_with_full_replay": self.resumed_with_replay if self.resumed else None,
            "resumed_without_replay": self.resumed and not self.resumed_with_replay,
        }
