"""Tests for the opt-in serving-time free-space veto (src/evaluation/safety_veto.py)."""

from __future__ import annotations

import json

import pytest
import torch

from src.core.game_config import GameConfig
from src.evaluation.safety_veto import (
    OUTCOME_KEPT,
    OUTCOME_NO_SPACIOUS,
    OUTCOME_VETOED,
    FreeSpaceVeto,
    SafetyVetoCounters,
    free_space_threshold,
    install_free_space_veto,
    spacious_directions,
    veto_choice,
)
from src.game.ai_snake import AISnake
from src.game.scripted_snake import ScriptedSnake
from src.game.snake_state import FREE_SPACE_BFS_CAP, FREE_SPACE_MIN_CAP

ALL = [True] * 6
NORMAL_ONLY = [True, True, True, False, False, False]


class TestPureVetoChoice:
    def test_all_spacious_is_identical_to_base(self):
        q = [0.1, 0.5, 0.2, 0.9, 0.3, 0.4]
        for base in range(6):
            assert veto_choice(q, ALL, [True, True, True], base) == (base, OUTCOME_KEPT)

    def test_veto_changes_choice_only_when_base_direction_is_cramped(self):
        q = [5.0, 1.0, 2.0, 4.0, 0.5, 3.0]
        # Base 0 (left) is cramped; best same-mode (non-boost) eligible is 2
        # (right, Q 2.0), not the higher-Q boost 5.
        assert veto_choice(q, ALL, [False, True, True], 0) == (2, OUTCOME_VETOED)
        # Base 2 (right) is spacious: kept even though left has higher Q.
        assert veto_choice(q, ALL, [False, True, True], 2) == (2, OUTCOME_KEPT)

    def test_non_boost_base_is_never_converted_to_boost_when_avoidable(self):
        # Boost Q-values dominate (Apex over-boosts); the veto still keeps the
        # non-boost speed mode whenever a spacious non-boost move is legal.
        q = [1.0, 0.0, 0.5, 50.0, 40.0, 30.0]
        for bits in range(1, 8):
            spacious = [bool(bits & (1 << d)) for d in range(3)]
            for base in range(3):
                action, outcome = veto_choice(q, ALL, spacious, base)
                if outcome == OUTCOME_VETOED:
                    assert action < 3 and spacious[action]
        assert veto_choice(q, ALL, [False, True, True], 0) == (2, OUTCOME_VETOED)

    def test_boost_base_stays_boost_when_possible(self):
        q = [9.0, 8.0, 7.0, 0.0, 1.0, 2.0]
        # Base 3 (left+boost) cramped: best same-mode eligible is 5 (right+boost),
        # even though non-boost moves have higher Q.
        assert veto_choice(q, ALL, [False, True, True], 3) == (5, OUTCOME_VETOED)

    def test_falls_back_to_other_speed_mode_only_when_needed(self):
        q = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        normal_only = [True, True, True, False, False, False]
        boost_right_only = [True, True, False, False, False, True]
        # Boost base, no boost legal: falls back to the best non-boost move.
        assert veto_choice(q, normal_only, [False, True, True], 3) == (2, OUTCOME_VETOED)
        # Non-boost base, the only spacious legal move is right+boost.
        assert veto_choice(q, boost_right_only, [False, False, True], 0) == (5, OUTCOME_VETOED)
        counters = SafetyVetoCounters()
        counters.record(0, 5, OUTCOME_VETOED)
        counters.record(3, 2, OUTCOME_VETOED)
        counters.record(0, 2, OUTCOME_VETOED)
        assert counters.vetoes_speed_switched == 2 and counters.vetoes_applied == 3

    def test_boost_shares_its_direction_flag(self):
        q = [0.0, 1.0, 0.0, 0.0, 9.0, 0.0]
        # Straight is cramped: both 1 and 4 are ineligible.
        action, outcome = veto_choice(q, ALL, [True, False, True], 4)
        assert outcome == OUTCOME_VETOED
        assert action in (0, 2, 3, 5) and action % 3 != 1

    def test_never_selects_a_masked_action(self):
        q = [0.0, 1.0, 2.0, 100.0, 100.0, 100.0]
        mask = [False, True, False, True, False, False]
        # Spacious: left and right only; the only masked-legal spacious action is 3.
        assert veto_choice(q, mask, [True, False, True], 1) == (3, OUTCOME_VETOED)
        # Exhaustive: whatever the flags, the result is masked-legal whenever the
        # veto replaces the base.
        for bits in range(8):
            spacious = [bool(bits & (1 << d)) for d in range(3)]
            for base in (1, 3):
                action, outcome = veto_choice(q, mask, spacious, base)
                if outcome == OUTCOME_VETOED:
                    assert mask[action] and spacious[action % 3]

    def test_no_masked_legal_spacious_action_leaves_choice_unchanged(self):
        q = [3.0, 2.0, 1.0, 0.0, 0.0, 0.0]
        # Right is spacious but only its (masked) boost variant... right normal is
        # masked out too, so nothing is eligible.
        mask = [True, True, False, False, False, False]
        assert veto_choice(q, mask, [False, False, True], 0) == (0, OUTCOME_NO_SPACIOUS)
        assert veto_choice(q, ALL, [False, False, False], 1) == (1, OUTCOME_NO_SPACIOUS)

    def test_ties_resolve_to_lowest_index(self):
        q = [9.0, 1.0, 1.0, 9.0, 1.0, 1.0]
        assert veto_choice(q, ALL, [False, True, True], 0) == (1, OUTCOME_VETOED)

    def test_masked_q_sentinels_are_never_preferred(self):
        invalid = -1.0e9
        q = [7.0, invalid, 2.0, invalid, invalid, invalid]
        mask = [True, False, True, False, False, False]
        assert veto_choice(q, mask, [False, True, True], 0) == (2, OUTCOME_VETOED)

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            veto_choice([0.0] * 5, ALL, [True] * 3, 0)
        with pytest.raises(ValueError):
            spacious_directions([1.0, 1.0], 32, 3)


class TestThreshold:
    @pytest.mark.parametrize("length", [1, 3, 10, 16, 17, 50, 80, 81, 150])
    def test_matches_scripted_cap_formula(self, setup_config, length):
        scripted = ScriptedSnake(0, (255, 0, 0), (200, 200), 10, 800, 600, kind="greedy_food")
        scripted.length = length
        cap, need = free_space_threshold(scripted.length, scripted._logical_length())
        assert cap == scripted._free_space_cap()
        assert need == min(scripted._logical_length(), scripted._free_space_cap())
        assert FREE_SPACE_MIN_CAP <= cap <= FREE_SPACE_BFS_CAP

    def test_spacious_uses_scripted_rounding(self):
        cap, need = 32, 20
        assert spacious_directions([20 / 32, 19 / 32, 1.0], cap, need) == [True, False, True]


class TestCounters:
    def test_counters_partition_decisions(self):
        counters = SafetyVetoCounters()
        counters.record(0, 0, OUTCOME_KEPT)
        counters.record(4, 2, OUTCOME_VETOED)
        counters.record(1, 5, OUTCOME_VETOED)
        counters.record(2, 2, OUTCOME_NO_SPACIOUS)
        data = counters.to_dict()
        assert data == {
            "decisions": 4,
            "kept_base": 1,
            "vetoes_applied": 2,
            "fallback_no_spacious": 1,
            "vetoes_to_boost": 1,
            "vetoed_base_boost": 1,
            "vetoes_speed_switched": 2,
        }
        assert data["decisions"] == (
            data["kept_base"] + data["vetoes_applied"] + data["fallback_no_spacious"]
        )
        with pytest.raises(ValueError):
            counters.record(0, 0, "bogus")


class _StubSnake:
    """Minimal snake surface the veto hook reads."""

    def __init__(self, features, length=20):
        self.features = list(features)
        self.length = length
        self.calls = []

    def _logical_length(self):
        return max(1, int(self.length))

    def _get_free_space_features(self, other_snakes):
        self.calls.append(list(other_snakes))
        return list(self.features)


class TestHook:
    def test_apply_counts_each_outcome(self):
        veto = FreeSpaceVeto()
        cap, _ = free_space_threshold(20, 20)  # cap 40, need 20
        cramped, roomy = 5 / cap, 1.0
        q = torch.tensor([5.0, 1.0, 2.0, 4.0, 0.5, 3.0])
        mask = torch.tensor([True] * 6)
        assert veto.apply(_StubSnake([roomy] * 3), [], q, mask, 0) == 0
        assert veto.apply(_StubSnake([cramped, roomy, roomy]), [], q, mask, 0) == 2
        assert veto.apply(_StubSnake([cramped] * 3), [], q, mask, 0) == 0
        assert veto.counters.to_dict() == {
            "decisions": 3,
            "kept_base": 1,
            "vetoes_applied": 1,
            "fallback_no_spacious": 1,
            "vetoes_to_boost": 0,
            "vetoed_base_boost": 0,
            "vetoes_speed_switched": 0,
        }
        record = veto.record()
        assert record["method"] == "free-space-veto/v2-speed-preserving"
        assert record["replacement_rule"] == "highest-q-eligible-same-speed-mode-then-other"
        assert record["counters"]["decisions"] == 3
        veto.reset()
        assert veto.counters.decisions == 0

    def test_install_requires_an_ai_snake(self):
        with pytest.raises(TypeError):
            install_free_space_veto(object())


def _make_ai_snake(q_row):
    class DqnPolicyStub:
        epsilon = 0.0

        def dqn(self, state):
            return torch.tensor([q_row])

        def select_action(self, state):
            raise AssertionError("masked DQN branch should handle action selection")

    snake = AISnake(
        id=0,
        color=(255, 0, 0),
        start_pos=(400, 300),
        segment_size=10,
        game_width=800,
        game_height=600,
        policy=DqnPolicyStub(),
    )
    snake.direction = (1, 0)
    return snake


class TestAISnakeIntegration:
    Q = [9.0, 1.0, 2.0, 0.0, 0.0, 0.0]  # prefers left

    def test_default_off_keeps_greedy_choice(self, setup_config, monkeypatch):
        snake = _make_ai_snake(self.Q)
        assert snake.safety_veto is None
        # Even if left were a pocket, no veto runs by default.
        monkeypatch.setattr(snake, "_get_free_space_features", lambda others: [0.0, 1.0, 1.0])
        snake.update([snake], [(100, 100)])
        assert snake._pre_collision_action == 0

    def test_installed_veto_steers_out_of_a_pocket(self, setup_config, monkeypatch):
        snake = _make_ai_snake(self.Q)
        veto = install_free_space_veto(snake)
        monkeypatch.setattr(snake, "_get_free_space_features", lambda others: [0.0, 1.0, 1.0])
        snake.update([snake], [(100, 100)])
        # Best eligible: right (Q 2.0) over straight (Q 1.0); boosts are masked
        # (length below MIN_BOOST_LENGTH).
        assert snake.length < GameConfig.MIN_BOOST_LENGTH
        assert snake._pre_collision_action == 2
        assert veto.counters.vetoes_applied == 1
        assert veto.counters.decisions == 1

    def test_installed_veto_is_a_no_op_in_open_space(self, setup_config):
        snake = _make_ai_snake(self.Q)
        veto = install_free_space_veto(snake)
        snake.update([snake], [(100, 100)])  # real flood fill: all open
        assert snake._pre_collision_action == 0
        assert veto.counters.to_dict()["kept_base"] == 1


@pytest.fixture
def tiny_live(setup_config, tmp_path):
    """Tiny live world plus a random 58-D vector checkpoint hero."""
    from src.core.config_loader import load_and_initialize_config
    from src.model.apex_network import ApexNetwork

    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(
        "game:\n"
        "  width: 300\n"
        "  height: 200\n"
        "  num_snakes: 3\n"
        "  initial_food: 12\n"
        "  max_food: 12\n"
    )
    load_and_initialize_config(str(cfg))
    torch.manual_seed(0)
    net = ApexNetwork(input_size=58, hidden_size=16, output_size=6)
    path = tmp_path / "vector.pth"
    torch.save({"dqn_state_dict": net.state_dict(), "input_size": 58, "hidden_size": 16}, path)
    return ("checkpoint", str(path))


class TestRolloutHook:
    OPPONENTS = [("scripted", "random_safe"), ("scripted", "greedy_food")]

    def test_default_off_record_is_byte_identical(self, tiny_live):
        from src.scripts.tournament_eval import rollout

        base = rollout(tiny_live, self.OPPONENTS, 80, 3)
        explicit_off = rollout(tiny_live, self.OPPONENTS, 80, 3, hero_safety_veto=False)
        assert json.dumps(base, sort_keys=True) == json.dumps(explicit_off, sort_keys=True)
        assert "safety_veto" not in base["probes"]

    def test_veto_on_records_counters(self, tiny_live):
        from src.scripts.tournament_eval import rollout

        record = rollout(tiny_live, self.OPPONENTS, 80, 3, hero_safety_veto=True)
        veto = record["probes"]["safety_veto"]
        counters = veto["counters"]
        assert veto["method"] == "free-space-veto/v2-speed-preserving"
        assert counters["decisions"] >= 1
        assert counters["decisions"] == (
            counters["kept_base"] + counters["vetoes_applied"] + counters["fallback_no_spacious"]
        )
        # One decision per frame the hero started alive.
        assert counters["decisions"] <= 80

    def test_veto_rejects_scripted_hero(self, tiny_live):
        from src.scripts.tournament_eval import rollout

        with pytest.raises(ValueError, match="vector61 (or ego2s )?checkpoint hero"):
            rollout(("scripted", "greedy_food"), self.OPPONENTS, 5, 0, hero_safety_veto=True)


def _entry(arm, mix, seed, mass, survival=0.5, cause="self", veto=None):
    probes = {"death_cause": cause}
    if veto is not None:
        probes["safety_veto"] = {"counters": veto}
    return {
        "arm": arm,
        "mix": mix,
        "world_seed": seed,
        "wall_seconds": 1.0,
        "record": {
            "mass_integral": mass,
            "survival_fraction": survival,
            "probes": probes,
            "denominators": {"decision_frames": veto["decisions"] if veto else 10},
        },
    }


class TestDevScreen:
    VETO = {
        "decisions": 10,
        "kept_base": 8,
        "vetoes_applied": 2,
        "fallback_no_spacious": 0,
        "vetoes_to_boost": 0,
        "vetoed_base_boost": 1,
        "vetoes_speed_switched": 1,
    }

    def _entries(self, gains, seeds, control=2, corrupt_control=False):
        entries = []
        for mix, gain in gains.items():
            for index, seed in enumerate(seeds):
                base = 30.0 + index
                entries.append(_entry("A", mix, seed, base))
                bump = gain + (0.5 if index % 2 else -0.5)
                entries.append(
                    _entry("B", mix, seed, base + bump, 0.6, "survived", dict(self.VETO))
                )
            for seed in seeds[:control]:
                mass = 30.0 + seeds.index(seed) + (1.0 if corrupt_control else 0.0)
                entries.append(_entry("C", mix, seed, mass))
        return entries

    def test_screen_seeds_are_stable_and_disjoint_from_challenger_namespaces(self):
        from research.apex_safety_20260926 import dev_screen

        seeds = dev_screen.screen_seeds(40)
        assert seeds == dev_screen.screen_seeds(40) and len(set(seeds)) == 40
        report = dev_screen.disjointness_report(seeds, None)
        assert report["disjoint"] is True
        assert all(not row["overlap"] for row in report["challenger_namespaces"].values())

    def test_decision_rule(self):
        from research.apex_safety_20260926.dev_screen import summarize

        seeds = list(range(1000, 1010))
        design = (10, 2)  # this test's own small design; the screen's is (40, 8)
        good = summarize(
            self._entries({"frozen": 5, "scripted": 5, "mixed": 0}, seeds),
            seeds,
            2,
            preregistered=design,
        )
        assert good["decision"] == "RECOMMEND_STRICT_GATE"
        assert good["preregistered_design"]["matches"] is True
        assert good["holm_primary"]["successful_mixes"] == ["frozen", "scripted"]
        assert good["determinism_control"]["passes"] is True
        frozen = good["per_mix"]["frozen"]
        assert frozen["death_causes"] == {"A": {"self": 10}, "B": {"survived": 10}}
        veto = frozen["safety_veto_B"]
        assert veto["totals"]["vetoes_applied"] == 20 and veto["veto_rate_per_decision"] == 0.2
        assert veto["episodes_decisions_match_decision_frames"] == 10

        weak = summarize(
            self._entries({"frozen": 5, "scripted": 0, "mixed": 0}, seeds),
            seeds,
            2,
            preregistered=design,
        )
        assert weak["decision"] == "NOT_ADVANCED"

        entries = self._entries(
            {"frozen": 5, "scripted": 5, "mixed": 5}, seeds, corrupt_control=True
        )
        assert (
            summarize(entries, seeds, 2, preregistered=design)["decision"]
            == "INVALID_NONDETERMINISTIC"
        )

        partial = self._entries({"frozen": 5, "scripted": 5, "mixed": 5}, seeds)[2:]
        assert summarize(partial, seeds, 2, preregistered=design)["decision"] == "INCOMPLETE"

    def test_non_preregistered_sizes_cannot_reach_a_decision(self):
        from research.apex_safety_20260926.dev_screen import (
            PREREGISTERED_DESIGN,
            preregistered_design_report,
            summarize,
        )

        assert PREREGISTERED_DESIGN == (40, 8)
        seeds = list(range(1000, 1010))
        strong = self._entries({"frozen": 5, "scripted": 5, "mixed": 5}, seeds)
        result = summarize(strong, seeds, 2)  # default = the pre-registered (40, 8)
        assert result["complete"] is True
        assert result["holm_primary"]["passes"] is True
        assert result["decision"] == "NON_PREREGISTERED_DESIGN"
        assert result["preregistered_design"] == {
            "preregistered_worlds_per_mix": 40,
            "preregistered_determinism_worlds": 8,
            "run_worlds_per_mix": 10,
            "run_determinism_worlds": 2,
            "matches": False,
        }
        assert preregistered_design_report(40, 8)["matches"] is True
        assert preregistered_design_report(40, 7)["matches"] is False
        # The full pre-registered design reaches the scientific decision rule.
        seeds40 = list(range(2000, 2040))
        full = self._entries({"frozen": 5, "scripted": 5, "mixed": 5}, seeds40, control=8)
        assert summarize(full, seeds40, 8)["decision"] == "RECOMMEND_STRICT_GATE"
        # Smoke runs stay SMOKE_NO_DECISION whatever their size.
        assert summarize(strong, seeds, 2, smoke=True)["decision"] == "SMOKE_NO_DECISION"

    def test_all_zero_deltas_summary_serializes(self, tmp_path):
        from research.apex_safety_20260926.dev_screen import summarize, write_new_json

        seeds = list(range(1000, 1010))
        entries = []
        for mix in ("frozen", "scripted", "mixed"):
            for index, seed in enumerate(seeds):
                entries.append(_entry("A", mix, seed, 30.0 + index))
                entries.append(_entry("B", mix, seed, 30.0 + index, veto=dict(self.VETO)))
            entries.extend(_entry("C", mix, seed, 30.0 + i) for i, seed in enumerate(seeds[:2]))
        summary = summarize(entries, seeds, 2, preregistered=(10, 2))
        assert summary["decision"] == "NOT_ADVANCED"
        json.dumps(summary, allow_nan=False)  # must not raise
        write_new_json(tmp_path / "summary.json", summary)
        loaded = json.loads((tmp_path / "summary.json").read_text())
        pooled = loaded["pooled_informational"]
        for name in ("stratified_mean_of_means", "crossed_mixes_by_worlds"):
            block = pooled[name]
            assert block["df"] is None and block["df_nonfinite"] == "positive_infinity"
        assert pooled["non_finite_fields"] == [
            "stratified_mean_of_means.df",
            "crossed_mixes_by_worlds.df",
        ]

    def test_preflight_requires_pilot_checks_outside_smoke(self, tmp_path):
        from research.apex_safety_20260926.dev_screen import (
            disjointness_report,
            preflight_failures,
            roster_parity_report,
            screen_seeds,
        )

        seeds = screen_seeds(4)
        missing = tmp_path / "nonexistent"
        disjoint = disjointness_report(seeds, missing)
        parity = roster_parity_report(missing)
        assert disjoint["disjoint"] is True and parity["checked"] is False
        assert preflight_failures(disjoint, parity, smoke=True) == []
        failures = preflight_failures(disjoint, parity, smoke=False)
        assert any("recipe reproduction not checked" in f for f in failures)
        assert any("roster parity not checked" in f for f in failures)
        # An existing but empty pilot dir fails recipe reproduction and parity count.
        empty = tmp_path / "empty"
        empty.mkdir()
        failures = preflight_failures(
            disjointness_report(seeds, empty), roster_parity_report(empty), smoke=False
        )
        assert any("not disjoint" in f for f in failures)
        assert any("reproduce the strict pilot" in f for f in failures)
        assert any("mismatches" in f for f in failures)
        ok_disjoint = {
            "disjoint": True,
            "strict_pilot_observed": {"development": {}, "pilot": {}},
            "recipe_reproduces_pilot": True,
        }
        full = {"checked": True, "compared": 48, "mismatches": []}
        assert preflight_failures(ok_disjoint, full, smoke=False) == []
        short = {"checked": True, "compared": 47, "mismatches": []}
        assert preflight_failures(ok_disjoint, short, smoke=False) == [
            "roster parity compared 47 worlds, expected 48"
        ]

    def test_pooled_informational_summaries(self):
        from research.apex_safety_20260926.dev_screen import summarize

        seeds = list(range(1000, 1010))
        entries = self._entries({"frozen": 6, "scripted": 3, "mixed": 0}, seeds)
        pooled = summarize(entries, seeds, 2)["pooled_informational"]
        strat = pooled["stratified_mean_of_means"]
        assert strat["mean_of_means"] == pytest.approx(3.0)
        assert strat["confidence"] == 0.90 and strat["strata"] == ["frozen", "scripted", "mixed"]
        assert strat["upper_below_zero"] is False
        random = pooled["random_effects_across_mixes"]
        assert random["k"] == 3 and random["strata"] == ["frozen", "scripted", "mixed"]
        crossed = pooled["crossed_mixes_by_worlds"]
        assert crossed["n_seeds"] == 3 and crossed["n_worlds"] == 10
        assert crossed["mean"] == pytest.approx(3.0)

        harmful = self._entries({"frozen": -6, "scripted": -6, "mixed": -6}, seeds)
        strat = summarize(harmful, seeds, 2)["pooled_informational"]["stratified_mean_of_means"]
        assert strat["upper_below_zero"] is True
        assert strat["strata_upper_below_zero"] == ["frozen", "scripted", "mixed"]

        few = summarize(self._entries({"frozen": 1}, seeds[:1], control=0), seeds[:1], 0)
        assert few["pooled_informational"]["stratified_mean_of_means"]["available"] is False

    def test_smoke_guard_limits_episodes(self, tmp_path):
        from research.apex_safety_20260926.dev_screen import parse_args

        base = ["--out", str(tmp_path / "x"), "--deadline-utc", "2099-01-01T00:00:00+00:00"]
        ok = parse_args(
            base + ["--worlds-per-mix", "1", "--determinism-worlds", "0", "--smoke-frames", "500"]
        )
        assert ok.mixes == ("scripted",)
        with pytest.raises(SystemExit):
            parse_args(base + ["--worlds-per-mix", "2", "--smoke-frames", "500"])
        with pytest.raises(SystemExit):
            parse_args(
                base
                + ["--worlds-per-mix", "1", "--determinism-worlds", "0", "--smoke-frames", "501"]
            )
        assert parse_args(base).mixes == ("frozen", "scripted", "mixed")
