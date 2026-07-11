"""Tests for the repaired promotion gate (src/scripts/tournament_eval.py).

Covers the pure helpers (agent-spec parsing, opponent-mix construction, the
§5.2 promotion rule) and a 2-seed miniature end-to-end run that uses scripted
stand-ins for hero/baseline/opponents, so no checkpoint is needed.
"""

import argparse
import json
import os

import pytest

from src.scripts.tournament_eval import (
    MIX_NAMES,
    build_mix_specs,
    main,
    parse_agent_spec,
    parse_mix_list,
    promotion_decision,
)


class TestBackwardCompat:
    def test_ci95_alias_survives_for_ensemble_eval(self):
        # ensemble_eval.py imports ci95 from tournament_eval; keep it exported.
        from src.scripts.eval_stats import ci95_halfwidth
        from src.scripts.tournament_eval import ci95

        assert ci95 is ci95_halfwidth


class TestParseAgentSpec:
    def test_checkpoint_path_passthrough(self):
        assert parse_agent_spec("saved_snakes/foo.pth") == ("checkpoint", "saved_snakes/foo.pth")

    def test_scripted_kinds(self):
        assert parse_agent_spec("scripted:greedy_food") == ("scripted", "greedy_food")
        assert parse_agent_spec("scripted:random_safe") == ("scripted", "random_safe")

    def test_unknown_scripted_kind_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_agent_spec("scripted:chess_engine")


class TestParseMixList:
    def test_valid_mixes(self):
        assert parse_mix_list("frozen,scripted") == ["frozen", "scripted"]

    def test_unknown_mix_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_mix_list("frozen,quantum")

    def test_duplicates_are_deduped_order_preserving(self):
        assert parse_mix_list("scripted,frozen,scripted,frozen") == ["scripted", "frozen"]


class TestBuildMixSpecs:
    POOL = [("checkpoint", "a.pth"), ("checkpoint", "b.pth")]

    def test_frozen_round_robins_the_pool(self):
        specs = build_mix_specs("frozen", 5, self.POOL)
        assert specs == [self.POOL[0], self.POOL[1], self.POOL[0], self.POOL[1], self.POOL[0]]

    def test_scripted_is_all_greedy_anchors(self):
        specs = build_mix_specs("scripted", 4, self.POOL)
        assert specs == [("scripted", "greedy_food")] * 4

    def test_mixed_alternates_pool_and_random_safe(self):
        specs = build_mix_specs("mixed", 5, self.POOL)
        assert specs == [
            self.POOL[0],
            ("scripted", "random_safe"),
            self.POOL[1],
            ("scripted", "random_safe"),
            self.POOL[0],
        ]

    def test_frozen_requires_a_pool(self):
        with pytest.raises(ValueError):
            build_mix_specs("frozen", 3, [])

    def test_unknown_mix_rejected(self):
        with pytest.raises(ValueError):
            build_mix_specs("quantum", 3, self.POOL)


def _paired(significant: bool, regression: bool = False) -> dict:
    return {"significant": significant, "regression": regression}


class TestPromotionDecision:
    def test_promotes_on_two_significant_mixes_and_anchor_ok(self):
        decision = promotion_decision(
            {"frozen": _paired(True), "scripted": _paired(True), "mixed": _paired(False)},
            MIX_NAMES,
        )
        assert decision["promote"] is True
        assert decision["significant_mixes"] == ["frozen", "scripted"]

    def test_rejects_with_one_significant_mix(self):
        decision = promotion_decision(
            {"frozen": _paired(True), "scripted": _paired(False), "mixed": _paired(False)},
            MIX_NAMES,
        )
        assert decision["promote"] is False
        assert any(">= 2 required" in r for r in decision["reasons"])

    def test_rejects_on_scripted_anchor_regression(self):
        decision = promotion_decision(
            {
                "frozen": _paired(True),
                "mixed": _paired(True),
                "scripted": _paired(False, regression=True),
            },
            MIX_NAMES,
        )
        assert decision["promote"] is False
        assert any("scripted anchor" in r for r in decision["reasons"])

    def test_rejects_when_anchor_mix_missing(self):
        decision = promotion_decision(
            {"frozen": _paired(True), "mixed": _paired(True)},
            ["frozen", "mixed"],
        )
        assert decision["promote"] is False
        assert any("anchor" in r for r in decision["reasons"])

    def test_rejects_on_single_mix(self):
        decision = promotion_decision({"scripted": _paired(True)}, ["scripted"])
        assert decision["promote"] is False

    def test_duplicate_mix_entries_count_as_one_mix(self):
        """A repeated mix cannot satisfy the >= 2 distinct-mixes clause."""
        decision = promotion_decision({"scripted": _paired(True)}, ["scripted", "scripted"])
        assert decision["promote"] is False
        assert decision["significant_mixes"] == ["scripted"]
        assert any(">= 2 required" in r for r in decision["reasons"])


class TestEndToEndMiniature:
    """2-seed, tiny-frames gate run with scripted stand-ins (no checkpoints)."""

    @pytest.fixture
    def tiny_config(self, tmp_path):
        cfg = tmp_path / "tiny_eval.yaml"
        cfg.write_text(
            "game:\n"
            "  width: 400\n"
            "  height: 300\n"
            "  num_snakes: 3\n"
            "  initial_food: 30\n"
            "  max_food: 30\n"
        )
        return cfg

    def test_gate_loop_runs_and_writes_json(self, setup_config, tiny_config, tmp_path):
        out = tmp_path / "gate.json"
        rc = main(
            [
                "scripted:random_safe",
                "--baseline",
                "scripted:greedy_food",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted,mixed",
                "--frames",
                "120",
                "--seeds",
                "0,1",
                "--config",
                str(tiny_config),
                "--json-output",
                str(out),
            ]
        )
        assert rc == 0  # without --gate the exit code is 0 regardless of verdict

        data = json.loads(out.read_text())
        assert data["baseline"] == "scripted:greedy_food"
        assert data["mixes"] == ["scripted", "mixed"]

        (candidate,) = data["candidates"]
        assert "error" not in candidate
        assert set(candidate["per_mix"]) == {"scripted", "mixed"}
        for mix in ("scripted", "mixed"):
            runs = candidate["per_mix"][mix]["runs"]
            assert [r["seed"] for r in runs] == [0, 1]
            for run in runs:
                assert 0.0 <= run["survival_fraction"] <= 1.0
                assert run["mass_integral"] >= 0.0
                # Integral over the FULL horizon is never above the alive-mean.
                assert run["mass_integral"] <= run["mean_mass_alive"] + 1e-9
                assert "boost_frame_fraction" in run["probes"]
            paired = candidate["per_mix"][mix]["paired"]
            assert paired["n"] == 2
            assert len(paired["deltas"]) == 2
        combined = candidate["combined_paired"]
        assert combined["n"] == 4
        assert "promote" in candidate["decision"]

    def test_gate_exit_code_reflects_decision(self, setup_config, tiny_config):
        # random_safe vs itself as baseline: deltas ~0, never significant on
        # 2 seeds -> the gate must REJECT (exit 1).
        rc = main(
            [
                "scripted:random_safe",
                "--baseline",
                "scripted:random_safe",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted,mixed",
                "--frames",
                "60",
                "--seeds",
                "0,1",
                "--config",
                str(tiny_config),
                "--gate",
            ]
        )
        assert rc == 1

    def test_pilot_mode_recommends_seed_count(self, setup_config, tiny_config, tmp_path):
        out = tmp_path / "pilot.json"
        rc = main(
            [
                "--pilot",
                "--baseline",
                "scripted:greedy_food",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted",
                "--frames",
                "60",
                "--seeds",
                "0,1",
                "--config",
                str(tiny_config),
                "--json-output",
                str(out),
            ]
        )
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["mode"] == "pilot"
        mix = data["pilot"]["per_mix"]["scripted"]
        assert mix["mass_integral_mean"] > 0.0
        assert mix["recommended_seeds"] is None or mix["recommended_seeds"] >= 2


# ---------------------------------------------------------------------------
# --engine simd: batched eval engine (blueprint §5.4)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_v2_config(tmp_path):
    """Tiny mechanics-v2 arena the BatchSim eval engine can run."""
    cfg = tmp_path / "tiny_v2_eval.yaml"
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


class TestSimdEvalEngine:
    """Unit tests for src.simd_env.eval_engine (used by --engine simd)."""

    def test_metrics_have_live_engine_keys_and_valid_ranges(self, setup_config, tiny_v2_config):
        from src.core.config_loader import apply_config_to_game_config, load_config
        from src.core.game_config import initialize_config
        from src.simd_env.eval_engine import run_simd_eval

        cfg = load_config(str(tiny_v2_config))
        initialize_config(cfg)
        apply_config_to_game_config(cfg)

        seeds = [0, 1, 2, 3]
        recs = run_simd_eval(
            ("scripted", "greedy_food"),
            [("scripted", "random_safe"), ("scripted", "random_safe")],
            frames=150,
            seeds=seeds,
            gamma=0.99,
        )
        assert [r["seed"] for r in recs] == seeds  # seed order == env order
        for r in recs:
            # Same headline keys the live rollout emits.
            for key in ("mass_integral", "max_mass", "kills", "deaths", "survival_fraction"):
                assert key in r
            assert 0.0 <= r["survival_fraction"] <= 1.0
            assert r["mass_integral"] >= 0.0
            # Integral over the full horizon never exceeds the alive-mean.
            assert r["mass_integral"] <= r["mean_mass_alive"] + 1e-9
            probes = r["probes"]
            assert 0.0 <= probes["boost_frame_fraction"] <= 1.0
            assert probes["food_eaten"] >= 0
            assert probes["peak_length"] >= 1

    def test_raster_checkpoint_runs_through_simd_engine(
        self, setup_config, tiny_v2_config, tmp_path
    ):
        # A raster (raster31v2) checkpoint builds a NetworkSimdPolicy and runs
        # through the fast gate — the P4 promotion path. Hermetic: an untrained
        # tiny raster net, saved with obs_spec, gated vs a scripted anchor.
        import torch

        from src.core.config_loader import apply_config_to_game_config, load_config
        from src.core.game_config import initialize_config
        from src.model.obs_spec import OBS_SPEC_KEY, RASTER31V2
        from src.model.raster_network import RasterDuelingNetwork
        from src.simd_env.eval_engine import NetworkSimdPolicy, build_simd_policy, run_simd_eval

        cfg = load_config(str(tiny_v2_config))
        initialize_config(cfg)
        apply_config_to_game_config(cfg)

        net = RasterDuelingNetwork()
        ckpt = tmp_path / "raster.pth"
        torch.save({"dqn_state_dict": net.state_dict(), OBS_SPEC_KEY: RASTER31V2}, ckpt)

        assert isinstance(build_simd_policy(("checkpoint", str(ckpt)), seed=0), NetworkSimdPolicy)

        recs = run_simd_eval(
            ("checkpoint", str(ckpt)),
            [("scripted", "random_safe")],
            frames=60,
            seeds=[0, 1],
            gamma=0.997,
        )
        assert [r["seed"] for r in recs] == [0, 1]
        for r in recs:
            assert r["mass_integral"] >= 0.0
            assert 0.0 <= r["survival_fraction"] <= 1.0

    def test_greedy_food_hero_grows_by_eating(self, setup_config, tiny_v2_config):
        # The greedy-food hero should eat and survive on a food-rich tiny arena.
        from src.core.config_loader import apply_config_to_game_config, load_config
        from src.core.game_config import initialize_config
        from src.simd_env.eval_engine import run_simd_eval

        cfg = load_config(str(tiny_v2_config))
        initialize_config(cfg)
        apply_config_to_game_config(cfg)

        recs = run_simd_eval(
            ("scripted", "greedy_food"),
            [("scripted", "random_safe"), ("scripted", "random_safe")],
            frames=120,
            seeds=[0, 1],
        )
        assert all(r["probes"]["food_eaten"] > 0 for r in recs)
        assert all(r["survival_fraction"] == 1.0 for r in recs)

    def test_boost_frame_fraction_detects_a_boosting_hero(
        self, setup_config, tiny_v2_config, monkeypatch
    ):
        """A boost-emitting hero yields boost_frame_fraction > 0.

        The scripted anchors never boost (they emit actions 0-2), so the probe
        is coincidentally 0.0 for them and could not detect a broken boost
        signal. This injects a hero that grows toward food, then engages boost
        whenever the boost-straight mask bit is safe, and asserts the probe
        reports the boosting — the promotion gate's boost-abuse detector. This
        would report 0.0 if boosting were read from a field reset by step().
        """
        import numpy as np

        from src.core.config_loader import apply_config_to_game_config, load_config
        from src.core.game_config import initialize_config
        from src.simd_env import eval_engine
        from src.simd_env.eval_engine import (
            GreedyFoodSimdPolicy,
            SimdPolicy,
            run_simd_eval,
        )

        cfg = load_config(str(tiny_v2_config))
        initialize_config(cfg)
        apply_config_to_game_config(cfg)

        class BoostWhenAblePolicy(SimdPolicy):
            """Boost straight when that bit is safe; else greedily seek food."""

            def __init__(self, seed):
                self._greedy = GreedyFoodSimdPolicy()
                self._seed = seed

            def actions(self, masks, sim, slots):
                out = self._greedy.actions(masks, sim, slots)
                # Prefer boost-straight (action 4) whenever it is a safe move.
                boost_straight_safe = masks[:, 4]
                out = np.where(boost_straight_safe, 4, out)
                return out.astype(np.int64)

        real_build = eval_engine.build_simd_policy
        hero_spec = ("scripted", "boost_when_able")

        def fake_build(spec, seed):
            if spec == hero_spec:
                return BoostWhenAblePolicy(seed)
            return real_build(spec, seed)

        monkeypatch.setattr(eval_engine, "build_simd_policy", fake_build)

        recs = run_simd_eval(
            hero_spec,
            [("scripted", "random_safe"), ("scripted", "random_safe")],
            frames=200,
            seeds=[0, 1, 2, 3],
        )
        # At least one seed's hero must have grown to boost-eligible length and
        # actually boosted, so the probe registers it.
        assert any(r["probes"]["boost_frame_fraction"] > 0.0 for r in recs)
        assert all(0.0 <= r["probes"]["boost_frame_fraction"] <= 1.0 for r in recs)

    def test_vector61_checkpoint_agent_rejected(self):
        # A 61-D vector champion is not featurized by the batch sim, so the simd
        # engine rejects it (raster31v2 checkpoints are supported — see
        # NetworkSimdPolicy). champion_a5 is a vector61 model.
        from src.simd_env.eval_engine import build_simd_policy

        ckpt = "saved_snakes/champion_a5_freespace_20260621.pth"
        if not os.path.exists(ckpt):
            pytest.skip("vector champion checkpoint not present")
        with pytest.raises(ValueError, match="raster31v2"):
            build_simd_policy(("checkpoint", ckpt), seed=0)

    def test_unknown_scripted_kind_raises(self):
        from src.simd_env.eval_engine import build_simd_policy

        with pytest.raises(ValueError):
            build_simd_policy(("scripted", "chess_engine"), seed=0)

    def test_empty_seeds_and_bad_frames_raise(self):
        from src.simd_env.eval_engine import run_simd_eval

        with pytest.raises(ValueError):
            run_simd_eval(("scripted", "random_safe"), [("scripted", "random_safe")], 100, [])
        with pytest.raises(ValueError):
            run_simd_eval(("scripted", "random_safe"), [("scripted", "random_safe")], 0, [0])


class TestSimdEngineEndToEnd:
    """main(--engine simd) drives the gate through the batched engine."""

    def test_gate_calibration_greedy_beats_random(self, setup_config, tiny_v2_config, tmp_path):
        # The calibration ordering the blueprint requires (greedy_food anchor
        # beats random_safe with paired significance) must hold via --engine simd.
        out = tmp_path / "simd_gate.json"
        rc = main(
            [
                "scripted:greedy_food",
                "--baseline",
                "scripted:random_safe",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted,mixed",
                "--frames",
                "150",
                "--seeds",
                "0,1,2,3",
                "--config",
                str(tiny_v2_config),
                "--engine",
                "simd",
                "--json-output",
                str(out),
            ]
        )
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["engine"] == "simd"
        (candidate,) = data["candidates"]
        assert "error" not in candidate
        assert set(candidate["per_mix"]) == {"scripted", "mixed"}
        for mix in ("scripted", "mixed"):
            runs = candidate["per_mix"][mix]["runs"]
            assert [r["seed"] for r in runs] == [0, 1, 2, 3]
        # greedy_food should decisively beat random_safe -> promote.
        assert candidate["decision"]["promote"] is True

    def test_checkpoint_baseline_rejected_up_front(self, setup_config, tiny_v2_config):
        # A checkpoint baseline under --engine simd fails fast (argparse error).
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "scripted:greedy_food",
                    "--baseline",
                    "saved_snakes/champion_a5_freespace_20260621.pth",
                    "--opponents",
                    "scripted:random_safe",
                    "--mixes",
                    "scripted,mixed",
                    "--frames",
                    "40",
                    "--seeds",
                    "0,1",
                    "--config",
                    str(tiny_v2_config),
                    "--engine",
                    "simd",
                ]
            )
        assert exc.value.code == 2  # argparse p.error exit code

    def test_checkpoint_candidate_recorded_as_error_not_crash(
        self, setup_config, tiny_v2_config, tmp_path
    ):
        # A checkpoint CANDIDATE (not baseline/opponent) is recorded as a
        # per-candidate error under --engine simd; the run still completes.
        out = tmp_path / "simd_ckpt_cand.json"
        rc = main(
            [
                "saved_snakes/champion_a5_freespace_20260621.pth",
                "--baseline",
                "scripted:random_safe",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted,mixed",
                "--frames",
                "40",
                "--seeds",
                "0,1",
                "--config",
                str(tiny_v2_config),
                "--engine",
                "simd",
                "--json-output",
                str(out),
            ]
        )
        assert rc == 0
        data = json.loads(out.read_text())
        (candidate,) = data["candidates"]
        assert "error" in candidate
        assert "raster" in candidate["error"]
        assert candidate["decision"]["promote"] is False

    def test_live_engine_default_unchanged(self, setup_config, tiny_v2_config, tmp_path):
        # --engine live (default) still works and records engine=live.
        out = tmp_path / "live_gate.json"
        rc = main(
            [
                "scripted:random_safe",
                "--baseline",
                "scripted:greedy_food",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted,mixed",
                "--frames",
                "60",
                "--seeds",
                "0,1",
                "--config",
                str(tiny_v2_config),
                "--json-output",
                str(out),
            ]
        )
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["engine"] == "live"
