"""Fixed-delay W1/K1 scheduler on full, continuously respawning Watch sessions."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from typing import Any, Callable

from research.bl_served_search.ceiling import B8_SHA256
from research.watch_lead_p.live import LiveWorld, search
from research.watch_lead_p.protocol import (
    ARMS,
    FRAMES,
    MAX_ROOT_BYTES,
    NAMESPACE,
    Certificate,
    Identity,
    accept,
)


def identity(world: LiveWorld, session: str, proposal: dict[str, Any]) -> Identity:
    return Identity(
        session,
        world.frame,
        int(proposal.get("life", world.life)),
        B8_SHA256,
        world.digest(),
        tuple(proposal.get("mask", [False] * 6)),
    )


def submit(world: LiveWorld, session: str, delay: int) -> tuple[Certificate | None, dict]:
    """Transport a source snapshot, forecast bare, and search at the target boundary."""
    source = world.digest()
    payload = world.snapshot()
    detail = {"source_frame": world.frame, "target_frame": world.frame + delay}
    detail["root_bytes"] = len(payload)
    if len(payload) > MAX_ROOT_BYTES:
        return None, {**detail, "drop": "overflow"}
    shadow = LiveWorld.restore(payload)
    if shadow.digest() != source:
        raise RuntimeError("source snapshot/transport identity mismatch")
    source_life = world.life
    for _ in range(delay):
        shadow.step()  # Frozen bare fallback, exclusively copied RNG and carry.
    proposal = shadow.proposal()
    target = identity(shadow, session, proposal)
    if not proposal.get("q") or target.life != source_life or shadow.waiting:
        return None, {**detail, "drop": "stale"}
    result = search(shadow, proposal)
    if world.digest() != source:
        raise RuntimeError("search mutated the actual source session")
    detail.update(
        result=result,
        target=target.record(),
        source_sha256=source,
        transport_sha256=hashlib.sha256(payload).hexdigest(),
    )
    return (
        Certificate(world.frame, target, int(result["chosen"]), target.frame, len(payload)),
        detail,
    )


def run_arm(
    seed: int,
    arm: str,
    *,
    frames: int = FRAMES,
    pulse: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Complete every tick, including quiet/death-free sessions and dead-frame mass zero."""
    if arm not in ARMS:
        raise ValueError("foreign P arm")
    world = LiveWorld(seed)
    session = f"{NAMESPACE}{seed}"
    initial = world.digest()
    counts: Counter = Counter()
    mass = deaths = 0
    pending: Certificate | None = None
    pending_drop: str | None = None
    target_frame = -1
    last_submit = -5
    trace = hashlib.sha256()
    roots = []
    started = time.monotonic()
    last_log = started
    for _ in range(frames):
        frame = world.frame
        action = None
        full = pending is not None or pending_drop is not None
        if full and frame >= target_frame:
            if pending is not None:
                proposal = world.proposal()
                actual = identity(world, session, proposal)
                status = accept(pending, actual)
                if status == "accepted":
                    action = pending.action
                    counts["accepted"] += 1
                    counts["changed"] += int(action != proposal["bare"])
                else:
                    counts[f"drop_{status}"] += 1
            else:
                counts[f"drop_{pending_drop}"] += 1
            pending, pending_drop = None, None
            # A delivery tick does not also submit another root.
        elif arm != "bare" and world.hero.is_alive and world.hero.length >= 600:
            if frame - last_submit >= 5:
                if full:
                    counts["skip_full"] += 1
                else:
                    last_submit = frame
                    counts["submitted"] += 1
                    delay = 0 if arm == "P0" else 48
                    pending, detail = submit(world, session, delay)
                    target_frame = frame + delay
                    roots.append(detail)
                    pending_drop = detail.get("drop")
                    if delay == 0:
                        if pending is not None:
                            proposal = world.proposal()
                            status = accept(pending, identity(world, session, proposal))
                            if status == "accepted":
                                action = pending.action
                                counts["accepted"] += 1
                                counts["changed"] += int(action != proposal["bare"])
                            else:
                                counts[f"drop_{status}"] += 1
                        else:
                            counts[f"drop_{pending_drop}"] += 1
                        pending, pending_drop = None, None
        if action is None:
            counts["bare_on_miss"] += 1
        event = world.step(action)
        mass += event["mass"]
        deaths += int(event["death"])
        trace.update(json.dumps(event, sort_keys=True, separators=(",", ":")).encode())
        if pulse is not None:
            pulse({"arm": arm, "frame": world.frame, "seed": seed})
        now = time.monotonic()
        if now - last_log >= 60:
            print(
                f"P_PROGRESS arm={arm} frame={world.frame}/{frames} seed={seed} "
                f"elapsed={now-started:.1f}s roots={counts['submitted']}",
                flush=True,
            )
            last_log = now
    if pending is not None or pending_drop is not None:
        counts["drop_late"] += 1  # Never extend a session to consume trailing work.
    return {
        "arm": arm,
        "seed": seed,
        "frames": frames,
        "initial_state": initial,
        "final_state": world.digest(),
        "trace_sha256": trace.hexdigest(),
        "deaths": deaths,
        "mass": mass / frames,
        "roster": world.roster,
        "counts": dict(counts),
        "roots": roots,
    }


def run_pair(seed: int, *, frames: int = FRAMES, pulse: Callable | None = None) -> dict:
    arms = {arm: run_arm(seed, arm, frames=frames, pulse=pulse) for arm in ARMS}
    if len({r["initial_state"] for r in arms.values()}) != 1:
        raise RuntimeError("paired initial world mismatch")
    return {"namespace": NAMESPACE, "seed": seed, "arms": arms}
