"""Tests for the pure event-based reward v2 module (blueprint §3.2)."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from src.core.reward_events import (
    DEATH_REWARD,
    KILL_REWARD_PER_VICTIM_LENGTH,
    PHI_LENGTH_DIVISOR,
    REWARD_V2_TERM_KEYS,
    RewardEvents,
    compute_reward_v2,
    phi,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "core" / "reward_events.py"

GAMMA = 0.997


class TestHandComputedCases:
    """Blueprint §3.2 semantics, verified against hand-computed values."""

    def test_constants_match_blueprint(self):
        assert PHI_LENGTH_DIVISOR == 10.0
        assert KILL_REWARD_PER_VICTIM_LENGTH == 0.3
        assert DEATH_REWARD == -3.0

    def test_eat_one_food_at_gamma_0997(self):
        # Eating +1 length at L=10: gamma*Phi(11) - Phi(10).
        total, breakdown = compute_reward_v2(
            RewardEvents(prev_length=10, new_length=11, died=False, gamma=GAMMA)
        )
        assert total == GAMMA * (11 / 10) - (10 / 10)
        assert breakdown["potential"] == total
        assert breakdown["death"] == 0.0
        assert breakdown["kill"] == 0.0

    def test_death_at_length_40_forfeits_potential(self):
        # Phi(death) = 0: potential term = -Phi(40) = -4.0; death event -3.0.
        total, breakdown = compute_reward_v2(
            RewardEvents(prev_length=40, new_length=40, died=True, gamma=GAMMA)
        )
        assert breakdown["potential"] == -4.0
        assert breakdown["death"] == -3.0
        assert total == -7.0

    def test_new_length_is_irrelevant_on_death(self):
        dead_a, _ = compute_reward_v2(
            RewardEvents(prev_length=40, new_length=0, died=True, gamma=GAMMA)
        )
        dead_b, _ = compute_reward_v2(
            RewardEvents(prev_length=40, new_length=123, died=True, gamma=GAMMA)
        )
        assert dead_a == dead_b == -7.0

    def test_kill_of_victim_length_20_pays_6(self):
        total, breakdown = compute_reward_v2(
            RewardEvents(prev_length=10, new_length=10, died=False, gamma=1.0, kills=(20,))
        )
        assert breakdown["kill"] == 6.0
        assert total == pytest.approx(6.0)

    def test_same_frame_killer_death_still_pays_kill(self):
        # Killer at length 30 dies while killing a length-20 victim:
        # potential -3.0, death -3.0, kill +6.0 => 0.0 total.
        total, breakdown = compute_reward_v2(
            RewardEvents(prev_length=30, new_length=30, died=True, gamma=GAMMA, kills=(20,))
        )
        assert breakdown["potential"] == -3.0
        assert breakdown["death"] == -3.0
        assert breakdown["kill"] == 6.0
        assert total == 0.0

    def test_kill_reward_is_unclamped(self):
        # 0.3 x 100 = 30.0, far above the v1 reward_max=5.0 clamp.
        total, breakdown = compute_reward_v2(
            RewardEvents(prev_length=10, new_length=10, died=False, gamma=1.0, kills=(100,))
        )
        assert breakdown["kill"] == 30.0
        assert total > 5.0

    def test_death_while_huge_is_unclamped(self):
        # Potential forfeiture -20.0 plus death -3.0, below the v1 reward_min=-12.
        total, _ = compute_reward_v2(
            RewardEvents(prev_length=200, new_length=200, died=True, gamma=GAMMA)
        )
        assert total == -23.0

    def test_multiple_kills_sum(self):
        _, breakdown = compute_reward_v2(
            RewardEvents(prev_length=10, new_length=10, died=False, gamma=1.0, kills=(10, 20, 5))
        )
        assert breakdown["kill"] == pytest.approx(0.3 * 35)

    def test_mapping_input_matches_dataclass_input(self):
        events = {
            "prev_length": 30,
            "new_length": 29,
            "died": True,
            "kills": [20, 7],
            "gamma": GAMMA,
        }
        from_mapping = compute_reward_v2(events)
        from_dataclass = compute_reward_v2(
            RewardEvents(prev_length=30, new_length=29, died=True, gamma=GAMMA, kills=(20, 7))
        )
        assert from_mapping == from_dataclass

    def test_breakdown_schema_and_exact_sum(self):
        total, breakdown = compute_reward_v2(
            RewardEvents(prev_length=17, new_length=16, died=True, gamma=GAMMA, kills=(13,))
        )
        assert tuple(breakdown.keys()) == REWARD_V2_TERM_KEYS
        running = 0.0
        for value in breakdown.values():
            running += value
        assert running == total  # bit-exact in insertion order


class TestTelescoping:
    """PBRS telescoping: potential terms over an alive trajectory sum to a Phi diff."""

    LENGTHS = [1, 2, 2, 5, 4, 10, 9, 25]

    def _potential_sum(self, gamma: float) -> float:
        total = 0.0
        for prev_length, new_length in zip(self.LENGTHS, self.LENGTHS[1:]):
            step_total, breakdown = compute_reward_v2(
                RewardEvents(
                    prev_length=prev_length, new_length=new_length, died=False, gamma=gamma
                )
            )
            assert step_total == breakdown["potential"]
            total += step_total
        return total

    def test_gamma_one_telescopes_exactly_to_length_difference(self):
        total = self._potential_sum(gamma=1.0)
        assert total == pytest.approx(
            phi(self.LENGTHS[-1]) - phi(self.LENGTHS[0]), rel=1e-12, abs=1e-12
        )

    def test_gamma_near_one_approximates_length_difference(self):
        gamma = GAMMA
        total = self._potential_sum(gamma=gamma)
        ideal = phi(self.LENGTHS[-1]) - phi(self.LENGTHS[0])
        # Each step's deviation from pure telescoping is (gamma - 1) * Phi(new).
        bound = (1.0 - gamma) * sum(phi(length) for length in self.LENGTHS[1:])
        assert abs(total - ideal) <= bound + 1e-12


class TestPurity:
    """reward_events must stay importable by the vectorized sim with no game deps."""

    ALLOWED_IMPORT_ROOTS = {"__future__", "dataclasses", "typing"}

    def test_module_imports_are_stdlib_only(self):
        tree = ast.parse(MODULE_PATH.read_text())
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module is not None, "relative imports are forbidden"
                roots.add(node.module.split(".")[0])
        assert roots <= self.ALLOWED_IMPORT_ROOTS, f"non-stdlib imports found: {roots}"

    def test_import_pulls_in_no_game_or_torch_modules(self):
        code = (
            "import sys; import src.core.reward_events; "
            "bad = [m for m in sys.modules if m.startswith('src.game') or m == 'torch']; "
            "assert not bad, bad"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            cwd=str(REPO_ROOT),
        )
