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
