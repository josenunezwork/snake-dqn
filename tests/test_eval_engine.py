"""Tests for src.simd_env.eval_engine (the --engine simd promotion gate).

Focus: the terminal-hero contract. The live gate marks the hero
``auto_respawn = False`` (``tournament_eval.rollout``), so its death is final and
its mass integral integrates 0 over every dead frame. ``--engine simd`` is a
drop-in substitute under the same METRIC_KEYS, so it must agree.
"""

import numpy as np
import pytest


@pytest.fixture
def tiny_v2_config(tmp_path):
    """Tiny mechanics-v2 arena the BatchSim eval engine can run."""
    cfg = tmp_path / "tiny_v2_eval_engine.yaml"
    cfg.write_text(
        "game:\n"
        "  width: 400\n"
        "  height: 300\n"
        "  num_snakes: 3\n"
        "  initial_food: 30\n"
        "  max_food: 30\n"
        "  mechanics_version: 2\n"
    )
    return cfg


@pytest.fixture
def tiny_v2(setup_config, tiny_v2_config):
    """Activate the tiny mechanics-v2 arena as the global game config."""
    from src.core.config_loader import apply_config_to_game_config, load_config
    from src.core.game_config import initialize_config

    cfg = load_config(str(tiny_v2_config))
    initialize_config(cfg)
    apply_config_to_game_config(cfg)
    return cfg


def _straight_policy_factory():
    """A hero policy that always goes straight — it walks into a wall and dies."""
    from src.simd_env.eval_engine import SimdPolicy

    class _AlwaysStraight(SimdPolicy):
        def actions(self, masks, sim, slots):
            return np.ones(masks.shape[0], dtype=np.int64)

    return _AlwaysStraight


def _build_sim(num_snakes, seeds):
    from src.simd_env.batch_sim import BatchSimConfig
    from src.simd_env.eval_engine import _config_from_game_config, _TerminalHeroBatchSim

    cfg = _config_from_game_config(num_snakes, 0.99)
    cfg = BatchSimConfig(**{**cfg.__dict__, "num_envs": len(seeds)})
    return _TerminalHeroBatchSim(cfg, seeds=list(seeds), train_mode=False)


def _legacy_actions(sim, masks, hero_policies, opp_policies):
    """Pre-cache nested-loop dispatch, retained here as a parity oracle."""
    actions = np.ones((sim.E, sim.S), dtype=np.int64)
    for env, policy in enumerate(hero_policies):
        if sim.get_alive()[env, 0]:
            actions[env, 0] = int(
                policy.actions(masks[env : env + 1, 0, :], sim, np.array([[env, 0]]))[0]
            )
    for slot in range(1, sim.S):
        for env, row in enumerate(opp_policies):
            if sim.get_alive()[env, slot]:
                actions[env, slot] = int(
                    row[slot - 1].actions(
                        masks[env : env + 1, slot, :], sim, np.array([[env, slot]])
                    )[0]
                )
    return actions


class TestTerminalHeroSim:
    """The eval sim exempts arena slot 0 from the non-train respawn sweep."""

    def test_dead_hero_is_not_respawned_but_dead_opponents_are(self, tiny_v2):
        sim = _build_sim(num_snakes=3, seeds=[0])
        # Kill the hero and one opponent outright, with respawn timers elapsed so
        # the sweep would resurrect anything it is willing to resurrect.
        sim.alive[0, 0] = False
        sim.alive[0, 1] = False
        sim.respawn_timer[0, 0] = 0
        sim.respawn_timer[0, 1] = 0

        sim._respawn_dead()

        assert not sim.get_alive()[0, 0], "hero (slot 0) must stay dead — death is terminal"
        assert sim.get_alive()[0, 1], "opponents must still respawn to keep arena pressure"

    def test_hero_stays_dead_for_the_rest_of_the_horizon(self, tiny_v2):
        sim = _build_sim(num_snakes=3, seeds=[0])
        trace = []
        for _ in range(120):
            sim.step(np.ones((1, 3), dtype=np.int64))  # hero drives straight into a wall
            trace.append(bool(sim.get_alive()[0, 0]))

        assert not all(trace), "an always-straight hero must die inside the horizon"
        # Once dead, never alive again: the trace is alive* then dead*.
        first_death = trace.index(False)
        assert not any(trace[first_death:]), f"hero was resurrected: {trace}"

    def test_a_terminal_hero_never_burns_its_respawn_timer(self, tiny_v2):
        sim = _build_sim(num_snakes=2, seeds=[0])
        sim.alive[0, 0] = False
        sim.respawn_timer[0, 0] = 7

        sim._respawn_dead()

        assert int(sim.respawn_timer[0, 0]) == 7
        assert not sim.get_alive()[0, 0]


class TestSimdEvalTerminalHeroMetrics:
    """run_simd_eval's gate metrics must charge the hero for its dead frames."""

    def test_suicidal_hero_dies_once_and_scores_near_zero(self, tiny_v2, monkeypatch):
        import src.simd_env.eval_engine as ee

        straight = _straight_policy_factory()
        real_build = ee.build_simd_policy
        monkeypatch.setattr(
            ee,
            "build_simd_policy",
            lambda spec, seed: (
                straight() if spec == ("scripted", "_straight") else real_build(spec, seed)
            ),
        )

        frames = 300
        recs = ee.run_simd_eval(
            ("scripted", "_straight"),
            [("scripted", "random_safe"), ("scripted", "random_safe")],
            frames=frames,
            seeds=[0, 1, 2],
        )

        for r in recs:
            # A terminal hero can die at most once — the live rollout's contract.
            assert r["deaths"] <= 1.0, r
            assert r["deaths"] == 1.0, f"an always-straight hero must die: {r}"
            # Dead frames integrate 0, so a hero that dies early cannot look immortal.
            assert 0.0 < r["survival_fraction"] < 0.2, r
            # Mass integral is over TOTAL frames with dead frames at 0, so it is
            # bounded by the mass it ever reached times the frames it was alive.
            assert r["mass_integral"] <= r["survival_fraction"] * r["max_mass"] + 1e-9, r
            assert r["probes"]["death_cause"] is not None, r

    def test_survival_fraction_and_mass_integral_match_the_alive_frames(self, tiny_v2, monkeypatch):
        """Exact accounting, checked against a hand-run sim (solo arena == deterministic)."""
        import src.simd_env.eval_engine as ee

        straight = _straight_policy_factory()
        monkeypatch.setattr(ee, "build_simd_policy", lambda spec, seed: straight())

        frames, seeds = 200, [0, 1]
        recs = ee.run_simd_eval(("scripted", "_straight"), [], frames=frames, seeds=seeds)

        # Same world, same always-straight actions, no opponents => reproducible.
        sim = _build_sim(num_snakes=1, seeds=seeds)
        alive_frames = np.zeros(len(seeds), dtype=np.int64)
        mass = np.zeros(len(seeds), dtype=np.float64)
        for _ in range(frames):
            sim.step(np.ones((len(seeds), 1), dtype=np.int64))
            alive = sim.get_alive()[:, 0]
            alive_frames += alive.astype(np.int64)
            mass += np.where(alive, sim.get_lengths()[:, 0], 0)

        for e, r in enumerate(recs):
            assert r["survival_fraction"] == pytest.approx(alive_frames[e] / frames)
            assert r["mass_integral"] == pytest.approx(mass[e] / frames)
            # The denominator is the FULL horizon: dead frames drag the score down.
            assert alive_frames[e] < frames, "hero should have died in this arena"
            assert r["survival_fraction"] < 1.0
            assert r["deaths"] == 1.0, f"a terminal hero dies exactly once: {r}"


class TestCheckpointDispatchCaching:
    """Checkpoint batching preserves the old action order and world outcomes."""

    def test_grouped_checkpoint_rows_match_legacy_actions_and_world(self, tiny_v2):
        from src.simd_env.eval_engine import NetworkSimdPolicy, _dispatch_actions

        class DeterministicNetwork(NetworkSimdPolicy):
            def __init__(self):
                self.calls = []

            def actions(self, masks, sim, slots):
                self.calls.append((masks.copy(), slots.copy()))
                out = np.ones(masks.shape[0], dtype=np.int64)
                for i, row in enumerate(masks):
                    safe = np.nonzero(row)[0]
                    out[i] = int(safe[0]) if safe.size else 1
                return out

        seeds = [5, 9]
        grouped_sim = _build_sim(num_snakes=3, seeds=seeds)
        legacy_sim = _build_sim(num_snakes=3, seeds=seeds)
        grouped_network = DeterministicNetwork()
        legacy_network = DeterministicNetwork()
        grouped_hero = [grouped_network, grouped_network]
        legacy_hero = [legacy_network, legacy_network]
        grouped_opps = [[grouped_network, grouped_network] for _ in seeds]
        legacy_opps = [[legacy_network, legacy_network] for _ in seeds]

        for _ in range(12):
            grouped_masks = grouped_sim.get_action_mask()
            grouped_actions = np.ones((grouped_sim.E, grouped_sim.S), dtype=np.int64)
            _dispatch_actions(
                grouped_sim, grouped_masks, grouped_actions, grouped_hero, grouped_opps
            )
            legacy_actions = _legacy_actions(
                legacy_sim, legacy_sim.get_action_mask(), legacy_hero, legacy_opps
            )
            np.testing.assert_array_equal(grouped_actions, legacy_actions)
            grouped_sim.step(grouped_actions)
            legacy_sim.step(legacy_actions)
            np.testing.assert_array_equal(grouped_sim.get_alive(), legacy_sim.get_alive())
            np.testing.assert_array_equal(grouped_sim.get_lengths(), legacy_sim.get_lengths())

        # One network call per frame; each call receives every live controlled row.
        assert len(grouped_network.calls) == 12
        assert all(
            call_slots.shape[0] == call_masks.shape[0]
            for call_masks, call_slots in grouped_network.calls
        )

    def test_identical_checkpoint_loads_once_and_dispatches_once_per_frame(
        self, tiny_v2, monkeypatch
    ):
        import src.simd_env.eval_engine as ee

        class SpyNetwork(ee.NetworkSimdPolicy):
            def __init__(self):
                self.calls = []

            def actions(self, masks, sim, slots):
                self.calls.append((masks.copy(), slots.copy()))
                return np.ones(masks.shape[0], dtype=np.int64)

        path = "same-raster-checkpoint.pth"
        created = []

        def build(spec, seed):
            if spec == ("checkpoint", path):
                policy = SpyNetwork()
                created.append(policy)
                return policy
            return ee.GreedyFoodSimdPolicy()

        monkeypatch.setattr(ee, "build_simd_policy", build)
        ee.run_simd_eval(
            ("checkpoint", path),
            [("checkpoint", path), ("checkpoint", path)],
            frames=1,
            seeds=[2, 4, 6],
        )

        assert len(created) == 1
        assert len(created[0].calls) == 1
        masks, slots = created[0].calls[0]
        assert slots.tolist() == [
            [0, 0],
            [1, 0],
            [2, 0],
            [0, 1],
            [1, 1],
            [2, 1],
            [0, 2],
            [1, 2],
            [2, 2],
        ]
        assert masks.shape == (9, 6)
