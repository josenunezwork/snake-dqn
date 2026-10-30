"""Bounded mapped mature-world proof on the allocated ruler device, never science."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from research.gpu_sim_spike import r2_cache
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim, CTOpponentRuntime
from research.sandbox_a_resident.ct_snapshots import (
    _SMALL_ENV_TENSOR_FIELDS,
    CompactWorldSnapshot,
    restore_world_snapshots_into,
)
from research.sandbox_a_resident.training_rosters import training_rosters
from research.spatial_student_pilot.batching import BatchedOpponents, observe_hero_bv8


def mapped_mature_proof(
    cfg: Any, fixtures: list[dict[str, Any]], *, b8: Any, networks: Any, device: str
) -> dict[str, Any]:
    """Three full tick/action/carry comparisons, E128 legacy vs E1024 candidate."""
    if cfg.mode != "ruler" or len(fixtures) < 128:
        raise ValueError("mapped mature proof is ruler-only with128 fixtures")
    table = training_rosters(cfg.worlds)
    sims = []
    actions = []
    tapes = []
    for count, runtime in ((128, CTOpponentRuntime), (1024, BatchedOpponents)):
        sim = CTDeviceSim(
            cfg.sim_config(count),
            device=device,
            train_mode=False,
            allow_respawn=True,
            move_backend="indexed" if device.startswith("cuda") else "reference",
        )
        r2_cache.enable(sim)
        rows = [fixtures[i % 128] for i in range(count)]
        opponents = runtime(
            sim,
            [table[r["roster_index"]] for r in rows],
            b8_network=b8,
            checkpoint_networks=networks,
        )
        restore_world_snapshots_into(
            [CompactWorldSnapshot.from_state_dict(r["snapshot"]) for r in rows], sim, opponents
        )
        stream = hashlib.sha256()
        tape = []
        for _ in range(3):
            sim.prepare()
            action = opponents.actions(sim, observe_hero_bv8(sim, b8)[-1])
            # Compare canonical mapped action bytes without treating duplicate
            # fixture rows as independent scientific observations.
            tape.append(action.cpu())
            stream.update(tape[-1].numpy().tobytes())
            sim.step(action)
            opponents.after_step(sim)
        sims.append((sim, opponents))
        actions.append(stream.hexdigest())
        tapes.append(tape)
    digests = []
    for sim, opponents in sims:
        tensors = {
            key: getattr(sim, key).cpu()
            for key in (
                *_SMALL_ENV_TENSOR_FIELDS,
                "bodies",
                "food_cells",
                "food_present",
                "corpse_food",
            )
        }
        rows = []
        states = (
            opponents.snapshot_rows(list(range(sim.E)))
            if hasattr(opponents, "snapshot_rows")
            else [opponents.snapshot_row(i) for i in range(sim.E)]
        )
        for i, state in enumerate(states):
            sha = hashlib.sha256()
            for key, tensor in sorted(tensors.items()):
                sha.update(key.encode())
                sha.update(tensor[i].numpy().tobytes())
            for key, tensor in sorted(state["carry"].items()):
                sha.update(key.encode())
                sha.update(tensor.numpy().tobytes())
            sha.update(str((state["rng_key"], state["rng_step"])).encode())
            rows.append(sha.hexdigest())
        digests.append(rows)
    exact_actions = all(torch.equal(b, a.repeat(8, 1)) for a, b in zip(*tapes))
    exact = digests[1] == digests[0] * 8 and exact_actions
    result = dict(
        exact=exact,
        exact_actions=exact_actions,
        ticks=3,
        reference_worlds=128,
        candidate_worlds=1024,
        distinct_historical_fixtures=128,
        fixture_replicas=8,
        reference_digests=digests[0],
        candidate_digests=digests[1],
        action_stream_sha256=actions,
        scope="finite mature mapped fixture proof; not universal arithmetic equivalence",
    )
    return result
