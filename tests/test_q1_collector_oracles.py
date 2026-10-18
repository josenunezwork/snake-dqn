"""CPU unit oracles for Q1 collection; these do not qualify a CUDA runtime."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.apex_safety_20260926 import dev_screen as DS
from research.redesign_m3c_20261008 import task_collection as C
from research.redesign_m3c_20261008 import train_m3c as T
from research.redesign_scope_20261007 import grid_h5000_identity as GID
from src.core.config_loader import load_and_initialize_config
from src.core.world_runtime import WorldRuntimeSpec
from src.scripts import tournament_eval as TE
from src.simd_env import eval_engine as EE
from src.simd_env import vector61_policy as VP
from src.simd_env.batch_sim import BatchSim
from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy


@pytest.fixture
def context(monkeypatch):
    """Real deployed environment/profile; tiny fixed networks replace checkpoint I/O only."""
    load_and_initialize_config(str(DS.DEFAULT_CONFIG))
    profile = TE.evaluation_profile_for_name(DS.PROFILE_NAME, DS.HORIZON)
    lookup = DS.agent_lookup({sha: name for name, sha in DS.POOL})

    def load(path):
        network = torch.nn.Linear(61, 6)
        with torch.no_grad():
            network.weight.zero_()
            network.bias.copy_(torch.tensor([1.0, 4.0, 2.0, -1.0, -2.0, -3.0]))
        return SimpleNamespace(dqn=network, input_size=61, training=True, epsilon=0.5)

    monkeypatch.setattr(TE, "build_policy_from_checkpoint", load)
    monkeypatch.setattr(VP, "checkpoint_is_vector61", lambda path: True)
    return {"profile": profile, "lookup": lookup}


def args():
    return T.build_parser().parse_args(
        [
            "--seed",
            "7391",
            "--init",
            "unused",
            "--demo-data",
            ".",
            "--out",
            "unused",
            "--actors",
            "1",
            "--worlds",
            "3",
            "--learner-seats",
            "1",
            "--collection-profile",
            C.PROFILE,
            "--namespace",
            C.fresh_namespace("C", 7391),
            "--v61-backend",
            "python",
        ]
    )


def roster_oracle(seeds):
    """Invoke the older strict-gate constructor independently of dev_screen._design_rows."""
    from src.evaluation.strict_promotion import materialize_rosters, scripted_agent

    pool = [{"kind": "checkpoint", "sha256": sha} for _, sha in DS.POOL]
    anchors = [scripted_agent("greedy_food"), scripted_agent("random_safe")]
    rules = [
        {
            "slot": s,
            "eligible_member_sha256s": (
                [p["sha256"] for p in pool] if s % 2 else [anchors[1]["sha256"]]
            ),
        }
        for s in range(1, 6)
    ]
    return {
        (int(r["world_seed"]), r["mix"]): r
        for r in materialize_rosters(
            final_world_seeds=list(seeds),
            roster_width=5,
            checkpoint_pool=pool,
            scripted_anchors=anchors,
            mixed_slot_rules=rules,
        )
    }


def assert_rosters(ctx, sim, runtime, policies, groups, mixes):
    expected = roster_oracle(sim._seeds)
    actual = {(int(e), int(s)): spec for spec, cells in groups.items() for e, s in cells}
    assert len(actual) == 5 * sim.E
    assert not any(slot == 0 for _, slot in actual)
    for e, seed in enumerate(sim._seeds):
        row = expected[(int(seed), str(mixes[e]))]
        for s, member in enumerate(row["slots"], 1):
            spec = ctx["lookup"][member["member_sha256"]]
            assert actual[e, s] == spec
            policy = policies[spec]
            if spec[0] == "scripted":
                assert isinstance(policy, FastProfileAnchorSimdPolicy)
                assert policy._world_seeds == tuple(sim._seeds)
                assert not runtime.controlled[e, s]
            else:
                assert isinstance(policy, VP.Vector61SimdPolicy)
                assert policy.forward == "rowwise"
                assert policy.checkpoint_path == spec[1]
                assert not policy.veto_slots  # Historical opponents are unwrapped.
                assert runtime.controlled[e, s]
    assert not runtime.controlled[:, 0].any()
    assert T.learner_seat_mask(7391, 0, sim.E, 1).tolist() == [[True] + [False] * 5] * sim.E


def test_actual_build_and_reset_rosters_match_independent_gate(context, monkeypatch):
    mixes = np.array(["scripted", "frozen", "mixed"], dtype=object)
    monkeypatch.setattr(C, "equal_mix_assignments", lambda seed, indexes: mixes.copy())
    a = args()
    sim, runtime, policies, groups = T.build_segment(a, context, 0, 0.998)
    assert_rosters(context, sim, runtime, policies, groups, mixes)
    replacements = [811, 812, 813]
    T.reset_task_worlds(sim, np.ones(3, dtype=bool), replacements)
    mixes = np.roll(mixes, 1)
    T.refresh_task_aligned_rosters(a, context, sim, runtime, policies, groups, mixes)
    assert_rosters(context, sim, runtime, policies, groups, mixes)


def _long_fixture(sim):
    # A genuine 800-cell ordered body, not merely a large target length.
    body = [
        (x, y) for y in range(4, 12) for x in (range(4, 104) if y % 2 == 0 else range(103, 3, -1))
    ]
    for e in range(sim.E):
        ring = (int(sim.head_ptr[e, 0]) - np.arange(len(body))) % sim.cap
        sim.bodies[e, 0, ring] = np.array(body)
        sim.seg_count[e, 0] = sim.length[e, 0] = len(body)
        sim._reward_prev_length[e, 0] = len(body)
        sim.direction[e, 0] = 3
    sim._rebuild_traversed_from_heads(np.ones(sim.E, dtype=bool))
    sim._refresh_action_masks(np.ones(sim.E, dtype=bool))
    if hasattr(sim, "_rebuild_env_grids"):
        for e in range(sim.E):
            sim._rebuild_env_grids(e)


@pytest.mark.parametrize("long_body", [False, True], ids=["short", "800-cell-body"])
def test_actual_collector_simulator_matches_old_gate_mechanics(context, long_body):
    a = args()
    sim, _, _, _ = T.build_segment(a, context, 0, 0.998)
    old_config = EE._config_from_game_config(6, 0.998, context["profile"])
    old_config = replace(
        old_config,
        num_envs=3,
        body_storage_capacity=WorldRuntimeSpec.source_exact(
            context["profile"]
        ).body_storage_capacity,
    )
    oracle = BatchSim(old_config, seeds=sim._seeds, train_mode=False, allow_respawn=True)
    if long_body:
        _long_fixture(sim)
        _long_fixture(oracle)
        assert (sim.seg_count[:, 0] == 800).all()
    for frame in range(6):
        # Fixed exogenous normal actions exercise collisions, body movement and food/RNG.
        actions = np.full((3, 6), frame % 3, dtype=np.int64)
        sim.step(actions)
        oracle.step(actions)
        assert [GID.env_digest(sim, e) for e in range(3)] == [
            GID.env_digest(oracle, e) for e in range(3)
        ]
    T.reset_task_worlds(sim, np.ones(3, dtype=bool), [914, 915, 916])
    fresh = BatchSim(old_config, seeds=[914, 915, 916], train_mode=False, allow_respawn=True)
    assert [GID.env_digest(sim, e) for e in range(3)] == [
        GID.env_digest(fresh, e) for e in range(3)
    ]


def test_actual_frozen_policies_match_scalar_actions_and_never_update(context, monkeypatch):
    monkeypatch.setattr(
        C,
        "equal_mix_assignments",
        lambda seed, indexes: np.array(["scripted", "frozen", "mixed"], dtype=object),
    )
    sim, runtime, policies, groups = T.build_segment(args(), context, 0, 0.998)
    before = {
        spec: {k: v.clone() for k, v in p.network.state_dict().items()}
        for spec, p in policies.items()
        if isinstance(p, VP.Vector61SimdPolicy)
    }
    decisions = []

    def choose(prepared):
        runtime.prepare(prepared)
        actions = np.ones((sim.E, sim.S), dtype=np.int64)
        alive = sim.get_alive()
        for spec, cells in groups.items():
            cells = cells[alive[cells[:, 0], cells[:, 1]]]
            if not len(cells):
                continue
            p = policies[spec]
            masks = sim.get_resolved_action_mask()[cells[:, 0], cells[:, 1]]
            got = p.actions(masks, sim, cells)
            if isinstance(p, VP.Vector61SimdPolicy):
                states, advisory = runtime.selection(cells)
                expected = []
                for state, allowed in zip(states, advisory):
                    allowed = allowed.copy()
                    if not allowed.any():
                        allowed[:3] = True
                    with torch.no_grad():
                        q = p.network(torch.tensor(state).unsqueeze(0))[0].numpy()
                    expected.append(max(range(6), key=lambda k: q[k] if allowed[k] else -1e9))
                assert got.tolist() == expected
                assert not p.q_values(states).requires_grad
                assert not p.network.training
                assert not hasattr(p, "optimizer")
                decisions.extend(got.tolist())
            actions[cells[:, 0], cells[:, 1]] = got
        return actions

    for _ in range(5):
        sim.step_with_policy(choose)
        runtime.observe_step(sim)
    assert len(decisions) >= 5
    for spec, frozen in before.items():
        p = policies[spec]
        assert all(param.grad is None for param in p.network.parameters())
        assert all(torch.equal(v, p.network.state_dict()[k]) for k, v in frozen.items())


@pytest.mark.parametrize("gamma", [0.99, 0.998])
def test_nstep_death_reset_and_horizon_match_scalar_old_path(gamma):
    fast, old = T.NStepBuilder(2), T.NStepBuilderRef(2)
    emitted = []
    for frame in range(9):
        worlds = np.array([0, 1])
        ids = 10 * frame + worlds
        rewards = np.array([2.0 + frame, 100.0 + frame])
        done = np.array([frame in (2, 7), frame == 4])
        for builder in (fast, old):
            builder.step(worlds, ids, np.array([1, 2]), rewards, done, gamma)
            if frame == 8:
                builder.end_worlds(worlds, gamma)
        got, expected = fast.take(), old.take()
        for left, right in zip(got, expected):
            assert np.array_equal(left, right)
        emitted.extend(zip(*got))
    # The first life cannot bootstrap into observation 30 from the replacement world.
    first_life = [row for row in emitted if row[0] in (0, 10, 20)]
    assert len(first_life) == 3
    assert all(row[3] == 0 and row[4] == -1 for row in first_life)
    assert first_life[0][2] == pytest.approx(2 + gamma * 3 + gamma**2 * 4)


def test_actual_actor_counts_only_hero_and_flushes_before_terminal_reset(
    context, monkeypatch, tmp_path
):
    """Bounded collection only: inject a death signal after a real simulator step.

    Injection makes the terminal branch deterministic without waiting for stochastic
    deaths; all production actor dispatch, ring, n-step, reset and message code runs.
    No learner update, scientific episode, CUDA or checkpoint qualification occurs.
    """
    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint
    from tests.test_perf_actor import _run_actor

    monkeypatch.setattr(GID, "_context", lambda threads: context)
    original_build = T.build_segment
    original_reset = T.reset_task_worlds
    resets = []
    builds = []

    def build(*pos, **kw):
        sim, runtime, policies, groups = original_build(*pos, **kw)
        builds.append(sim)
        original_step = sim.step_with_policy
        steps = 0

        def step(*step_args, **step_kw):
            nonlocal steps
            result = original_step(*step_args, **step_kw)
            if steps in (0, 3):
                sim._last_done[:, 0] = True
                sim.alive[:, 0] = False
            steps += 1
            return result

        sim.step_with_policy = step
        return sim, runtime, policies, groups

    def reset(sim, mask, seeds):
        resets.append((mask.copy(), list(seeds)))
        original_reset(sim, mask, seeds)
        assert sim.alive[mask, 0].all()
        assert (sim.frame[mask] == 0).all()

    monkeypatch.setattr(T, "build_segment", build)
    monkeypatch.setattr(T, "reset_task_worlds", reset)
    init = tmp_path / "unit-init.pth"
    with torch.random.fork_rng():
        torch.manual_seed(1901)
        save_ego2s_checkpoint(init, Ego2sNet(7, 30))
    messages, _ = _run_actor(
        tmp_path,
        init,
        "vectorized",
        "python",
        7,
        extra=(
            "--worlds",
            "3",
            "--learner-seats",
            "1",
            "--collection-profile",
            C.PROFILE,
            "--namespace",
            C.fresh_namespace("D", 7391),
            "--segment-frames",
            "5000",
            "--reward",
            "mass",
            "--kill-scale",
            "0",
            "--gamma-start",
            ".998",
        ),
    )
    assert len(builds) == 1 and len(messages) == 7
    assert len(resets) >= 2 and resets[0][0].all()
    assert len(set(resets[0][1])) == 3
    assert all(message["decisions"] == 3 for message in messages)
    assert all(message["live_rows_by_policy"]["learner"] == 3 for message in messages)
    first = messages[0]
    assert len(first["s"]) == 3
    assert (first["disc"] == 0).all() and (first["nxt"] == -1).all()
    assert sum(first["task_collection"]["death_resets"].values()) == 3
    assert sum(message["replay_rows"] for message in messages) >= 12
    # End-of-life rows emitted at frame 3 terminate; no row may bootstrap across reset.
    assert (messages[3]["disc"] == 0).all()
    assert (messages[3]["nxt"] == -1).all()
