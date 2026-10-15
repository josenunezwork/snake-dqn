"""Mandatory clone/recorded-series precondition and measured timing receipt."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.b8_death_census_20261014 import census
from research.t1_commitment_20261015.replay import (
    ARTIFACTS,
    advance_recorded,
    assert_same_world,
    build_replay,
    configure_cpu,
    load_inputs,
    state_fields,
)


def write_json(path: Path, value: Any) -> None:
    """Publish a complete receipt atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(census._jsonable(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def compare_world(env: int, warmup: int = 2500, frames: int = 1000, context: Any = None) -> dict:
    """Check every recorded length and every full-state field after cloning."""
    _, series = load_inputs("b8_s0")
    live = build_replay("b8_s0", [env], context)
    start = time.perf_counter()
    checked = advance_recorded(live, series, [env], warmup)
    clone = live.clone()
    assert not np.shares_memory(live.sim.bodies, clone.sim.bodies)
    for grid in ("owner", "slot", "food"):
        assert np.shares_memory(getattr(clone.sim, f"_{grid}"), getattr(clone.sim, f"_{grid}_pad"))
        assert not np.shares_memory(getattr(live.sim, f"_{grid}"), getattr(clone.sim, f"_{grid}"))
    assert_same_world(live, clone)
    for t in range(warmup, warmup + frames):
        checked += advance_recorded(live, series, [env], t + 1)
        checked += advance_recorded(clone, series, [env], t + 1)
        assert_same_world(live, clone)
    return {
        "source_env": env,
        "world_seed": live.seeds[0],
        "warmup_decisions": warmup,
        "cloned_decisions": frames,
        "length_values_compared": checked,
        "full_digest_comparisons": frames + 1,
        "digest_fields": sorted(state_fields(live.sim, live.opponents)),
        "view_and_storage_independence": True,
        "wall_seconds": time.perf_counter() - start,
        "identical": True,
    }


def timings(envs: Sequence[int], context: Any) -> dict:
    """Measure E=1 and E=32 using the exact same census executor."""
    _, series = load_inputs("b8_s0")
    replay = build_replay("b8_s0", envs, context)
    advance_recorded(replay, series, envs, 100)
    start = time.perf_counter()
    for _ in range(10):
        replay.clone()
    clone_s = (time.perf_counter() - start) / 10
    forced = series["action"][list(envs), 100]
    no_forward = replay.clone()
    start = time.perf_counter()
    for _ in range(20):
        no_forward.step(forced)
    no_forward_s = (time.perf_counter() - start) / 20
    forward = replay.clone()
    # Warm compiled observation/inference paths before timing.
    forward.step()
    start = time.perf_counter()
    for _ in range(20):
        forward.step()
    with_forward_s = (time.perf_counter() - start) / 20
    rollout = replay.clone()
    start = time.perf_counter()
    for _ in range(300):
        rollout.step()
    return {
        "world_count": len(envs),
        "clone_seconds": clone_s,
        "step_without_hero_forward_seconds": no_forward_s,
        "step_with_hero_forward_seconds": with_forward_s,
        "rollout_300_seconds": time.perf_counter() - start,
        "rollout_env_frames": rollout.env_frames,
        "timing_state_decision": 100,
    }


def run_precondition(out: Path = ARTIFACTS) -> dict:
    """Stop with a failure receipt if any precondition fails."""
    configure_cpu()
    started = time.perf_counter()
    receipt = {
        "authority": "development diagnostic, not gate evidence",
        "status": "RUNNING",
        "clone_implementation": "unmodified census.clone_sim (pad views restored)",
        "worlds": [],
        "timing": [],
    }
    write_json(out / "parity.json", receipt)
    try:
        context = census.load_context()
        data, _ = load_inputs("b8_s0")
        envs = [d["env"] for d in data["deaths"] if d["fatal_t"] > 3500][:2]
        assert len(envs) == 2
        for env in envs:
            receipt["worlds"].append(compare_world(env, context=context))
            print(json.dumps({"parity_world": receipt["worlds"][-1]}), flush=True)
            write_json(out / "parity.json", receipt)
        for selected in ([envs[0]], list(range(32))):
            receipt["timing"].append(timings(selected, context))
            print(json.dumps({"timing": receipt["timing"][-1]}), flush=True)
        receipt["status"] = "PASS"
    except BaseException:
        receipt["status"] = "FAIL_STOP_BEFORE_STEP_1"
        receipt["failure"] = traceback.format_exc()
        raise
    finally:
        receipt["wall_seconds"] = time.perf_counter() - started
        write_json(out / "parity.json", receipt)
        write_json(out / "timing.json", receipt)
    return receipt


if __name__ == "__main__":
    run_precondition()
