"""Default-OFF, Watch6-only staged assistance; no speculative work in the tick."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Callable

from research.t34_pathb.live import evaluate, prepared_fingerprint, snapshot
from web.backend.served_checkpoint import B8_S3_SHA256

_LOG = logging.getLogger(__name__)


def eligible(session: Any) -> bool:
    """Only the exact released mixed Watch6 hero can opt into the product hook."""
    choice = session.served_checkpoint or {}
    return bool(
        sys.platform == "darwin"
        and session.mode == "watch"
        and len(session.game.snakes) == 6
        and session.watch_roster == "mixed"
        and choice.get("name") == "b8-s3"
        and choice.get("sha256") == B8_S3_SHA256
        and not choice.get("reason")
        and not choice.get("pin_problems")
        and session.game.snakes[0].policy is session.policy
        and session.game.snakes[0].safety_veto is None
    )


class ServingAssistance:
    """Stage next-frame work and consume it only at the actual action boundary."""

    def __init__(self, session: Any, mode: str = "async", smoke: bool = False) -> None:
        from web.backend.pathb_controller import DeadlineController

        if mode not in {"async", "sync"}:
            raise ValueError("Path B mode must be async or sync")
        self.session = session
        self.mode = mode
        self.depth, self.min_length, self.interval = (2, 0, 1) if smoke else (40, 600, 5)
        self.controller = DeadlineController(evaluate)
        self.generation = uuid.uuid4().hex
        self.life = 0
        self.active = True
        self.log_sink: Callable[[dict[str, Any]], None] | None = None
        self._sync: tuple[tuple, dict[str, Any]] | None = None
        self._scheduled_reason = "no_request"
        self._sync_requests = 0
        self.cap_deadline: float | None = None
        self._alive = bool(session.game.snakes[0].is_alive)
        self.attach()

    def attach(self) -> None:
        self.session.game.snakes[0].action_transform = self.apply

    def invalidate(self) -> None:
        """Fence old results without joining or spawning another worker."""
        self.generation = uuid.uuid4().hex
        self._sync = None

    def identity(self, frame: int) -> tuple:
        return (self.generation, int(frame), self.life, B8_S3_SHA256)

    def apply(self, snake: Any, others: Any, q: Any, mask: Any, bare: int) -> int:
        if not self.active:
            return int(bare)
        frame = int(self.session.game.frame)
        valid = mask.detach().cpu().numpy().astype(bool).tolist()
        expected = self.identity(frame)

        def validate(result: dict[str, Any]) -> bool:
            return result.get("prepared_fingerprint") == prepared_fingerprint(self.session.game)

        if self.mode == "sync":
            pending, self._sync = self._sync, None
            proposed = None if pending is None else pending[1].get("proposed_action")
            accepted = bool(
                pending is not None
                and pending[0] == expected
                and pending[1].get("bare_action") == bare
                and pending[1].get("mask") == valid
                and validate(pending[1])
                and type(proposed) is int
                and 0 <= proposed < 6
                and valid[proposed]
            )
            action = int(proposed) if accepted else int(bare)
            log = {
                "requested": pending is not None,
                "applied": accepted and action != bare,
                "accepted": accepted,
                "stale_discarded": pending is not None and not accepted,
                "deadline_miss": False,
                "fallback_reason": None if accepted else "no_request_or_world_mismatch",
                "proposed_action": proposed,
                "executed_action": action,
            }
            # Use the same bounded logging/counters for the descriptive arm.
            self.controller.record(log)
        else:
            action, log = self.controller.poll(
                expected,
                int(bare),
                valid,
                validate=validate,
                no_request_reason=self._scheduled_reason,
            )
        log.update(hero_id=int(snake.id), frame=frame, life=self.life, bare_action=int(bare))
        _LOG.info("pathb-decision: %s", json.dumps(log, sort_keys=True))
        if self.log_sink is not None:
            self.log_sink(dict(log))
        return int(action)

    def after_step(self) -> None:
        if not self.active:
            return
        hero = self.session.game.snakes[0]
        alive = bool(hero.is_alive)
        if alive != self._alive:
            self.life += 1
            self._alive = alive
        target = int(self.session.game.frame) + 1
        if not alive or int(hero.length) < self.min_length or target % self.interval:
            self._scheduled_reason = "not_triggered"
            return
        # Submission starts before serialization, so transfer/copy cost counts.
        deadline = time.monotonic() + 2.0 / max(1.0, float(self.session.speed))
        if self.mode == "async" and self.controller.busy:
            self._scheduled_reason = "busy"
            return
        try:
            cutoff = deadline if self.mode == "async" else None
            if self.cap_deadline is not None:
                cutoff = self.cap_deadline if cutoff is None else min(cutoff, self.cap_deadline)
            payload = snapshot(self.session.game, self.depth, cutoff)
            if self.mode == "sync":
                self._sync = self.identity(target), evaluate(payload)
                self._sync_requests += 1
            else:
                self.controller.submit(payload, self.identity(target), deadline)
            self._scheduled_reason = "submitted"
        except Exception as exc:
            self._scheduled_reason = f"snapshot_or_worker_error:{type(exc).__name__}"
            # Failure is a counted bare fallback, never a lost serving tick.
            self.controller.record(
                {
                    "requested": False,
                    "applied": False,
                    "stale_discarded": False,
                    "deadline_miss": False,
                    "fallback_reason": f"snapshot_or_worker_error:{type(exc).__name__}",
                    "proposed_action": None,
                    "executed_action": None,
                }
            )

    def state(self) -> dict[str, Any]:
        state = {**self.controller.state(), "active": self.active, "hero_id": 0}
        if self.mode == "sync":
            state["counts"]["requested"] = self._sync_requests
        return state

    def close(self, wait: bool = False) -> None:
        self.active = False
        self.invalidate()
        hero = self.session.game.snakes[0]
        if getattr(hero, "action_transform", None) == self.apply:
            del hero.action_transform
        self.controller.close(wait=wait)


def install(session: Any, mode: str = "async", smoke: bool = False) -> ServingAssistance:
    """Explicit harness seam; product selection is separately scope-gated."""
    if sys.platform != "darwin":
        raise RuntimeError("T34 Path B is Mac CPU only")
    import torch

    if str(session.game.snakes[0].policy.device) != "cpu":
        raise RuntimeError("Path B requires CPU policy inference")
    torch.set_num_threads(1)
    previous = getattr(session, "pathb", None)
    if previous is not None:
        raise RuntimeError("one Path B controller per session; close before replacing")
    session.pathb = ServingAssistance(session, mode=mode, smoke=smoke)
    return session.pathb


def configure(session: Any) -> None:
    """Called after a successful build; the absent flag never creates a worker."""
    requested = os.environ.get("SNAKE_SERVE_PATHB", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    current = getattr(session, "pathb", None)
    if current is not None:
        current.invalidate()
        current.active = requested and eligible(session)
        if current.active:
            current.attach()
        elif getattr(session.game.snakes[0], "action_transform", None) == current.apply:
            del session.game.snakes[0].action_transform
    elif requested and eligible(session):
        install(session)
