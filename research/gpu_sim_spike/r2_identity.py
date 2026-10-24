"""Engine-level R2 byte-identity recorder (old engine versus R2 candidate).

One record runs the unchanged I0 resident tick (``a100_ruler._full_tick``) for
several consecutive progressing ticks on a long-body fixture and captures, per
tick, SHA-256 digests of every observable output (advisory masks, all 15 v8
result tensors, ego observations, vector61 features, hero Q/legal/action, the
full action grid) and the complete persistent state inventory
(``integration_engine.snapshot``: every simulator field including ring padding,
vector runtime carry, policy fields and collector buffers).

The recorder imports only names that exist at the pinned runtime ``968f0276``,
so ``--root`` may point at the read-only qualified pinned checkout: the old
engine is then literally the pinned code, not a disabled flag.  ``mode`` is
``pin`` (never touch R2), ``off`` (R2 code present, simulator not enabled),
``r2`` or ``r2-debug`` (R2 enabled; debug recomputes every cache hit with the
pinned full-capacity code and raises on a difference).

``wrap`` physically rotates every ring (sim and reset template alike) so each
live body crosses storage index zero; the logical state is unchanged, but the
padding/ring layout differs, exercising wrapped gathers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

MODES = ("pin", "off", "r2", "r2-debug")
MIXES = ("frozen", "mixed", "scripted")


def _bytes(value: Any) -> bytes:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        head = f"{tensor.dtype}|{list(tensor.shape)}|".encode()
        return head + tensor.numpy().tobytes()
    if dataclass_fields(value):
        return b"".join(_bytes(getattr(value, name)) for name in dataclass_fields(value))
    if isinstance(value, (tuple, list)):
        return b"(" + b",".join(_bytes(item) for item in value) + b")"
    return repr(value).encode()


def dataclass_fields(value: Any) -> tuple[str, ...]:
    import dataclasses

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return tuple(field.name for field in dataclasses.fields(value))
    return ()


def _sha(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def wrap_rings(sim: Any, salt: int = 0) -> None:
    """Rotate physical ring storage so every head lands at a small index."""
    import torch

    e = torch.arange(sim.E, device=sim.device)[:, None]
    s = torch.arange(sim.S, device=sim.device)[None, :]
    target = (3 + 5 * s + e + salt) % sim.cap
    shift = (target - sim.head_ptr) % sim.cap
    index = (torch.arange(sim.cap, device=sim.device)[None, None, :] - shift[..., None]) % sim.cap
    sim.bodies = sim.bodies.gather(2, index[..., None].expand(-1, -1, -1, 2)).contiguous()
    sim.head_ptr = (sim.head_ptr + shift) % sim.cap


def exception_fixture(sim: Any) -> None:
    """Deterministic exception-heavy perturbation of a long-body world batch.

    World ``e % 3 == 0``: every snake one frame from a boost length burn.
    World ``e % 3 == 1``: slot 3 dead for many frames (stale dead ring padding)
    and slots 4/5 dead and ready, so the next ``prepare`` respawns them.
    World ``e % 3 == 2``: slot 2's head is moved onto an interior cell of slot
    1's serpentine body, so every move is a body collision: death, ordered
    corpse food and kill credit happen on the first tick.  Slot 1's interior
    cell is offset ``2 * 141 + 70`` (three full serpentine rows deep).
    """
    import torch

    e = torch.arange(sim.E, device=sim.device)
    burn = (e % 3 == 0)[:, None]
    cost = int(sim.cfg.boost_length_cost_frames)
    sim.boost_frames = torch.where(
        burn, torch.full_like(sim.boost_frames, cost - 1), sim.boost_frames
    )
    dead = e % 3 == 1
    for slot, timer in ((3, 1000), (4, 1), (5, 1)):
        sim.alive[:, slot] = sim.alive[:, slot] & ~dead
        sim.respawn_timer[:, slot] = torch.where(
            dead, torch.full_like(sim.respawn_timer[:, slot], timer), sim.respawn_timer[:, slot]
        )
    trap = e % 3 == 2
    inner = (sim.head_ptr[:, 1] - (2 * 141 + 70)) % sim.cap
    cell = sim.bodies[e, 1, inner]
    head = sim.bodies[e, 2, sim.head_ptr[:, 2]]
    sim.bodies[e, 2, sim.head_ptr[:, 2]] = torch.where(trap[:, None], cell, head)


def build(
    checkpoint: Path, mix: str, worlds: int, mode: str, wrap: bool, exceptions: bool = False
) -> tuple:
    from research.gpu_sim_spike import integration_engine as engine

    case = engine._case(checkpoint, mix, worlds, "cpu", "I")
    sim, template = case[0], case[2]
    if exceptions:
        for target in (sim, template):
            exception_fixture(target)
    if wrap:
        for target in (sim, template):
            wrap_rings(target)
        if not bool((sim.head_ptr < sim.seg_count - 1).all()):
            raise AssertionError("wrap fixture did not wrap every live ring")
    if mode in ("r2", "r2-debug"):
        from research.gpu_sim_spike import r2_cache

        for target in (sim, template):
            r2_cache.enable(target, debug=mode == "r2-debug")
    return case


def record(case: tuple, ticks: int) -> list[dict[str, Any]]:
    """Run ``ticks`` progressing full ticks from a reset; digest outputs and state."""
    import torch

    from research.gpu_sim_spike import a100_ruler as ruler
    from research.gpu_sim_spike import device_policy
    from research.gpu_sim_spike import integration_engine as engine
    from research.gpu_sim_spike.device_sim import DeviceSim
    from research.gpu_sim_spike.vector_geometry import IndexedVectorRuntime

    sim, policy, template, policy_template, collection, _name = case
    calls: list[tuple[str, str]] = []

    def observed(label: str, original: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            value = original(*args, **kwargs)
            calls.append((label, _sha(value)))
            return value

        return call

    rows = []
    with ExitStack() as stack:
        for owner, name, label in (
            (device_policy, "v8_decide", "v8_decide"),
            (device_policy, "build_observation", "observation"),
            (device_policy, "choose_anchor", "anchor"),
            (DeviceSim, "advisory_mask", "advisory_mask"),
            (IndexedVectorRuntime, "features", "vector61_features"),
            (device_policy.DevicePolicy, "evaluate", "hero_q_legal_action"),
            (device_policy.DevicePolicy, "actions", "action_grid"),
        ):
            stack.enter_context(patch.object(owner, name, observed(label, getattr(owner, name))))
        stack.enter_context(engine._arm("I"))
        stack.enter_context(torch.inference_mode())
        ruler._reset(sim, policy, template, policy_template)
        for index in range(ticks):
            del calls[:]
            ruler._full_tick(sim, policy, collection, 0)
            rows.append(
                {
                    "tick": index,
                    "calls": list(calls),
                    "state_sha256": engine.digest(case),
                    "alive": int(sim.alive.sum()),
                    "deaths": int(sim._last_done.sum()),
                    "corpse_food": int(sim.corpse_food.sum()),
                    "boosted": int(sim._boosted_this_step.sum()),
                    "max_seg_count": int(sim.seg_count.max()),
                }
            )
    return rows


def run(
    checkpoint: Path,
    mix: str,
    worlds: int,
    mode: str,
    wrap: bool,
    ticks: int,
    exceptions: bool = False,
) -> dict:
    case = build(checkpoint, mix, worlds, mode, wrap, exceptions)
    result = {
        "mix": mix,
        "worlds": worlds,
        "mode": mode,
        "wrap": wrap,
        "exceptions": exceptions,
        "ticks": record(case, ticks),
    }
    if mode in ("r2", "r2-debug"):
        from research.gpu_sim_spike import r2_cache

        result["r2_stats"] = r2_cache.stats(case[0])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="source tree to import")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mix", choices=MIXES, required=True)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--wrap", action="store_true")
    parser.add_argument("--exceptions", action="store_true")
    parser.add_argument("--ticks", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve()))
    value = run(
        args.checkpoint, args.mix, args.worlds, args.mode, args.wrap, args.ticks, args.exceptions
    )
    args.output.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
