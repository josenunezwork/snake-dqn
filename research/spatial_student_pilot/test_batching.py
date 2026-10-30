"""Exact mapped-world, restore and state-extraction proofs for mechanical batching."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from research.gpu_sim_spike import r2_cache
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim, CTOpponentRuntime
from research.sandbox_a_resident.ct_snapshots import (
    _SMALL_ENV_TENSOR_FIELDS,
    capture_world_snapshot,
    restore_world_snapshots_into,
)
from research.sandbox_a_resident.training_rosters import (
    CHAMPION_SHA256,
    FRP3_S12_SHA256,
    training_rosters,
)
from research.spatial_student_pilot.batching import (
    BatchedOpponents,
    capture_rows,
    observe_hero_bv8,
    stable_q,
)
from research.spatial_student_pilot.collect import collect_batch
from research.spatial_student_pilot.collection_resume import (
    publish_collection,
    restore_collection,
)
from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.observation import compact_from_sim
from research.spatial_student_pilot.resume import UnitStore, empty_ledger


class FixedPolicy(nn.Module):
    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        # Deliberately exact integer-valued margins; no accidental CPU GEMM
        # assumptions in the simulator scheduling proof.
        return torch.arange(6, device=args[0].device).float().expand(len(args[0]), -1)


def networks() -> dict[str, nn.Module]:
    return {FRP3_S12_SHA256: FixedPolicy(), CHAMPION_SHA256: FixedPolicy()}


def exact(value: Any, expected: Any) -> None:
    if isinstance(value, torch.Tensor):
        assert value.dtype == expected.dtype and torch.equal(value, expected)
    elif isinstance(value, np.ndarray):
        assert value.dtype == expected.dtype and np.array_equal(value, expected)
    elif isinstance(value, dict):
        assert value.keys() == expected.keys()
        for key in value:
            exact(value[key], expected[key])
    elif isinstance(value, (tuple, list)):
        assert len(value) == len(expected)
        for a, b in zip(value, expected):
            exact(a, b)
    else:
        assert value == expected


def simulator(cfg: PilotConfig, ids: list[int], merged: bool = True) -> tuple[Any, Any]:
    sim = CTDeviceSim(cfg.sim_config(len(ids)), device="cpu", train_mode=False, allow_respawn=True)
    r2_cache.enable(sim, debug=True)
    sim.rng_key.copy_(torch.tensor([cfg.world_seed(i) for i in ids]))
    sim.rng_counter.zero_()
    sim.alive.fill_(False)
    sim.terminal_slots.fill_(False)
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True
    table = training_rosters(cfg.worlds)
    runtime = BatchedOpponents if merged else CTOpponentRuntime
    opp = runtime(
        sim,
        [table[i] for i in ids],
        b8_network=FixedPolicy(),
        checkpoint_networks=networks(),
        opp_rng_keys=[cfg.opp_seed(i) for i in ids],
    )
    return sim, opp


def digest_rows(sim: Any, opp: Any) -> list[str]:
    small = {
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
    for i in range(sim.E):
        h = hashlib.sha256()
        for key in sorted(small):
            h.update(key.encode())
            h.update(small[key][i].numpy().tobytes())
        state = opp.snapshot_row(i)
        for key, value in sorted(state["carry"].items()):
            h.update(key.encode())
            h.update(value.numpy().tobytes())
        h.update(str((state["rng_key"], state["rng_step"])).encode())
        rows.append(h.hexdigest())
    return rows


@pytest.mark.parametrize("merged", [False, True])
def test_batched_snapshot_and_compact_exact_scalar_bytes(merged: bool) -> None:
    cfg = PilotConfig().development("preflight")
    sim, opp = simulator(cfg, list(range(6)), merged)
    # Wraps, unequal spans, duplicate cells, dead row, food holes and corpse.
    for row in range(6):
        count = row + 1
        sim.head_ptr[row, 0] = row % 3
        sim.seg_count[row, 0] = count
        sim.length[row, 0] = count + 2
        ring = (sim.head_ptr[row, 0] - torch.arange(count)) % sim.cap
        sim.bodies[row, 0, ring] = torch.tensor([[4, 4]] * count)
    sim.alive[2, 0] = False
    sim.food_present[:, 2] = False
    sim.corpse_food[:, 5] = True
    sim.prepare()
    opp.actions(sim, observe_hero_bv8(sim, FixedPolicy())[-1])
    selected = [4, 1, 4, 0, 2]
    compact, snaps = capture_rows(
        sim, opp, selected, world_ids=selected, tick=7, max_body_capacity=cfg.body_capacity
    )
    for row, observation, snap in zip(selected, compact, snaps):
        exact(observation, compact_from_sim(sim, [row]))
        expected = capture_world_snapshot(
            sim, opp, row, world_id=row, life_id=0, tick=7, max_body_capacity=cfg.body_capacity
        )
        exact(snap.to_state_dict(), expected.to_state_dict())


def test_e128_e1024_mapped_exact_digests_and_legacy_opponents(tmp_path: Any) -> None:
    cfg = replace(PilotConfig().development("preflight"), worlds=1024)
    wide, wide_opp = simulator(cfg, list(range(1024)))
    narrow = [simulator(cfg, list(range(i, i + 128))) for i in range(0, 1024, 128)]
    # Old five-roster path is an independently retained implementation.
    old, old_opp = simulator(cfg, list(range(128)), merged=False)
    proof = []
    for _ in range(3):
        for sim, opp in [(wide, wide_opp), *narrow, (old, old_opp)]:
            sim.prepare()
            actions = observe_hero_bv8(sim, FixedPolicy())[-1]
            sim.step(opp.actions(sim, actions))
            opp.after_step(sim)
        expected = [h for sim, opp in narrow for h in digest_rows(sim, opp)]
        candidate = digest_rows(wide, wide_opp)
        legacy = digest_rows(old, old_opp)
        assert candidate == expected
        assert legacy == expected[:128]
        proof.append(dict(e128=expected, e1024=candidate, legacy_e128=legacy))
    import json

    (tmp_path / "mapped-parity.json").write_text(json.dumps(dict(exact=True, ticks=proof)))


def test_near_tie_uses_same_single_row_fp32_forward() -> None:
    class TiePolicy(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((len(x), 6))
            out[:, 0] = 1
            out[:, 1] = 1 + (2e-6 if len(x) > 128 else -2e-6)
            return out

    net = TiePolicy()
    for size in (1, 128, 1024):
        x = torch.zeros(size, 1)
        q = stable_q(net, (x,), net(x))
        assert q.argmax(1).tolist() == [0] * size
        assert torch.equal(q, net(x[:1]).expand(size, -1))


def test_whole_sim_checkpoint_resume_same_records_and_next_decisions(tmp_path: Any) -> None:
    cfg = PilotConfig().development("preflight")
    store = UnitStore(tmp_path / "first", cfg, empty_ledger(cfg), tmp_path)
    kwargs = dict(
        b8=FixedPolicy(),
        networks=networks(),
        device="cpu",
        stop=lambda: False,
        progress=lambda _: None,
    )
    full = collect_batch(
        cfg, range(cfg.worlds), **kwargs, checkpoint=lambda s: publish_collection(store, cfg, s)
    )
    state = restore_collection(store, cfg)
    assert state["next_tick"] == 4
    resumed = collect_batch(cfg, range(cfg.worlds), **kwargs, resume_state=state)
    for key in ("roots", "demos", "decisions", "decisions_by_world", "reached_l600_world_ids"):
        exact(resumed[key], full[key])
    # Completion records contain references plus new rows, never duplicate row IDs.
    all_ids = [key for name in store.list_ids("cstate-") for key in store.get(name)["records"]]
    assert len(all_ids) == len(set(all_ids))


def frozen_module(name: str) -> Any:
    """Independent pre-change oracle from the owner's exact starting commit."""
    import subprocess
    import types

    module = types.ModuleType("spatial_frozen_" + name)
    source = subprocess.check_output(
        [
            "git",
            "show",
            "e86fe37b573f6983fa44886e847ff092ca659f45:" f"research/spatial_student_pilot/{name}.py",
        ]
    )
    exec(compile(source, f"<frozen-e86fe37b-{name}>", "exec"), module.__dict__)
    return module


def test_collector_record_identity_frozen_e3_vs_resident_e6() -> None:
    cfg = PilotConfig().development("preflight")
    kwargs = dict(
        b8=FixedPolicy(),
        networks=networks(),
        device="cpu",
        stop=lambda: False,
        progress=lambda _: None,
    )
    old = frozen_module("collect")
    expected = [
        old.collect_batch(replace(cfg, world_batch_size=3), range(i, i + 3), **kwargs)
        for i in (0, 3)
    ]
    actual = collect_batch(cfg, range(6), **kwargs)
    for key in ("roots", "demos"):

        def sort(rows: list[Any]) -> list[Any]:
            return sorted(rows, key=lambda r: (r["world_id"], r["tick"], r.get("kind", "")))

        assert all(torch.equal(r["q_b8"], torch.arange(6).float()) for r in actual[key])
        # The only added field is diagnostic Q; registered record bytes are exact.
        records = [{k: v for k, v in r.items() if k != "q_b8"} for r in actual[key]]
        exact(sort(records), sort([r for b in expected for r in b[key]]))
    assert actual["decisions"] == sum(b["decisions"] for b in expected)


def test_branch_e128_e1024_exact_returns_and_actions(tmp_path: Any) -> None:
    from research.spatial_student_pilot.search import _Job, _rollouts
    from research.spatial_student_pilot.test_search_readout import fixture_root

    root, cfg, b8 = fixture_root()
    legal = np.flatnonzero(root["mask"])
    jobs = [
        _Job(0, int(legal[i % len(legal)]), (0, 1, 4, 5)[i % 4], i % 5 == 0) for i in range(1024)
    ]
    kwargs = dict(b8=b8, networks=networks(), device="cpu", student=FixedPolicy())
    narrow = _rollouts(
        [root], jobs, cfg=replace(cfg, branch_batch_size=128, branch_horizon=4), **kwargs
    )
    wide = _rollouts(
        [root], jobs, cfg=replace(cfg, branch_batch_size=1024, branch_horizon=4), **kwargs
    )
    for a, b in zip(wide, narrow):
        exact(a, b)
    import json

    hashes = [
        hashlib.sha256(b"".join(np.asarray(v).tobytes() for v in result)).hexdigest()
        for result in (narrow, wide)
    ]
    (tmp_path / "branch-parity.json").write_text(
        json.dumps(
            dict(
                exact=True,
                jobs=1024,
                ticks=4,
                suffixes=[0, 1, 4, 5],
                closed_loop=True,
                e128_sha256=hashes[0],
                e1024_sha256=hashes[1],
            )
        )
    )


def test_cached_fit_exact_final_model_optimizer_rng(tmp_path: Any) -> None:
    from research.spatial_student_pilot.fit import fit_student
    from research.spatial_student_pilot.observation import build_observation, compact_cat

    cfg = PilotConfig().development("preflight")
    kwargs = dict(
        b8=FixedPolicy(),
        networks=networks(),
        device="cpu",
        stop=lambda: False,
        progress=lambda _: None,
    )
    data = collect_batch(cfg, range(6), **kwargs)
    demos = data["demos"]
    roots = [
        dict(r, target_action=r["b8_action"]) for r in data["roots"] if r["partition"] == "train"
    ]
    from research.spatial_student_pilot.fit import cache_search_observations

    cached = cache_search_observations(roots, "cpu", cfg.microbatch)
    for root in roots:
        exact(cached[id(root)], build_observation(compact_cat([root["compact"]])))
    stores = [UnitStore(tmp_path / key, cfg, empty_ledger(cfg), tmp_path) for key in ("old", "new")]
    old = frozen_module("fit")
    for function, store in zip((old.fit_student, fit_student), stores):
        function(
            cfg,
            demos,
            roots,
            device="cpu",
            store=store,
            stop=lambda: False,
            progress=lambda _: None,
        )
    exact(stores[1].get("fit-0000002"), stores[0].get("fit-0000002"))


def test_real_network_argmax_e128_e1024() -> None:
    from research.sandbox_a_resident.ct_run import _load_vector61
    from research.spatial_student_pilot.prepare import CHECKPOINT_NAMES, ROOT
    from src.model.ego2s_network import load_ego2s_checkpoint

    base = ROOT.parent / "snake-dqn" / "saved_snakes"
    b8 = load_ego2s_checkpoint(str(base / CHECKPOINT_NAMES["b8-s3.pth"]), device="cpu")
    nets = {
        key: _load_vector61(base / CHECKPOINT_NAMES[name], torch.device("cpu"))
        for name, key in (("frp3-s12.pth", FRP3_S12_SHA256), ("champion.pth", CHAMPION_SHA256))
    }
    cfg = replace(PilotConfig().development("preflight"), worlds=1024)
    # Full-network arithmetic is tested on128 distinct states, mapped eight times
    # into the larger slab, independently of the simulator scheduling proof.
    sim, old = simulator(cfg, list(range(128)), merged=False)
    sim.prepare()
    snap = [
        capture_world_snapshot(
            sim, old, i, world_id=i, life_id=0, tick=0, max_body_capacity=cfg.body_capacity
        )
        for i in range(128)
    ]
    actions = []
    for count, runtime in (
        (128, CTOpponentRuntime),
        (128, BatchedOpponents),
        (1024, BatchedOpponents),
    ):
        target = CTDeviceSim(
            cfg.sim_config(count), device="cpu", train_mode=False, allow_respawn=True
        )
        r2_cache.enable(target)
        table = training_rosters(cfg.worlds)
        opponents = runtime(
            target, [table[i % 128] for i in range(count)], b8_network=b8, checkpoint_networks=nets
        )
        opponents.verify_q_bytes = True
        restore_world_snapshots_into([snap[i % 128] for i in range(count)], target, opponents)
        hero = observe_hero_bv8(target, b8)[-1]
        actions.append(opponents.actions(target, hero))
    assert torch.equal(actions[1], actions[0])
    assert torch.equal(actions[2], actions[0].repeat(8, 1))
