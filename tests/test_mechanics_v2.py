"""Tests for game mechanics v2 (blueprint §1) and version plumbing.

Every v2 mechanic is opt-in via ``game.mechanics_version: 2``; each test class
carries a matching v1-unchanged guard so legacy behavior stays bit-identical.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.config_loader import load_config
from src.core.game_config import AppConfig, GameConfig, initialize_config
from src.core.mechanics_constants import (
    CORPSE_DROP_FRACTION_V2,
    HEADON_SIZE_RATIO,
    POPULATION_FLOOR_V2,
)
from src.game.food_manager import FoodManager
from src.game.game_state import GameState
from src.game.snake import Snake
from tests.conftest import make_test_snake

pytestmark = pytest.mark.usefixtures("setup_config")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _init_mechanics(version: int) -> None:
    """Initialize the global config with the given mechanics version."""
    config = AppConfig.from_defaults()
    config = replace(config, game=replace(config.game, mechanics_version=version))
    initialize_config(config)


def _bare_game_state(food_manager=None) -> GameState:
    """Minimal GameState for collision/food unit tests (no policies)."""
    game_state = GameState.__new__(GameState)
    game_state.snakes = []
    game_state.food_manager = food_manager if food_manager is not None else SimpleNamespace(food=[])
    game_state.episode_deaths = 0
    game_state.episode_kills = 0
    game_state.episode_collision_counts = {"wall": 0, "self": 0, "head": 0, "body": 0}
    game_state._game_width = 800
    game_state._game_height = 600
    return game_state


def _make_food_manager(max_food: int = 10, initial_food: int = 0) -> FoodManager:
    return FoodManager(
        game_width=800,
        game_height=600,
        max_food=max_food,
        initial_food=initial_food,
        segment_size=10,
        wall_thickness=10,
    )


class TestVersionPlumbing:
    """GameConfig.MECHANICS_VERSION / REWARD_VERSION and YAML mapping."""

    def test_defaults_are_version_1(self):
        assert GameConfig.MECHANICS_VERSION == 1
        assert GameConfig.REWARD_VERSION == 1

    def test_initialize_config_flips_mechanics_version(self):
        _init_mechanics(2)
        assert GameConfig.MECHANICS_VERSION == 2
        assert GameConfig.REWARD_VERSION == 1  # independent switch

    def test_mechanics_v2_yaml_loads_both_versions(self):
        config = load_config(str(REPO_ROOT / "configs" / "mechanics_v2.yaml"))
        assert config.game.mechanics_version == 2
        assert config.rewards.version == 2
        # The contract-bearing champion_a5 fields are preserved from
        # free_space_v2.yaml.
        assert config.network.input_size == 61
        assert config.network.use_free_space is True
        assert config.rewards.death == -11.0
        assert config.apex.n_step == 3

    def test_free_space_v2_yaml_stays_version_1(self):
        config = load_config(str(REPO_ROOT / "configs" / "free_space_v2.yaml"))
        assert config.game.mechanics_version == 1
        assert config.rewards.version == 1

    @pytest.mark.parametrize("bad_version", [0, 3])
    def test_out_of_range_mechanics_version_rejected(self, bad_version, tmp_path):
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(f"game:\n  mechanics_version: {bad_version}\n")
        with pytest.raises(ValueError):
            load_config(str(config_path))

    @pytest.mark.parametrize("bad_version", [0, 3])
    def test_out_of_range_reward_version_rejected(self, bad_version, tmp_path):
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(f"rewards:\n  version: {bad_version}\n")
        with pytest.raises(ValueError):
            load_config(str(config_path))


class TestBoostTrailPellet:
    """§1.2: each boost segment burn drops a pellet at the vacated tail cell."""

    def _make_burning_snake(self) -> Snake:
        snake = make_test_snake(0, (100, 100))
        snake.length = 6
        snake.segments = [(100 - i * 10, 100) for i in range(6)]
        snake.is_boosting = True
        # One frame away from paying the length cost.
        snake.boost_frames = GameConfig.BOOST_LENGTH_COST_FRAMES - 1
        return snake

    def test_v2_burn_records_pellet_at_vacated_tail_cell(self):
        _init_mechanics(2)
        snake = self._make_burning_snake()

        snake.move()

        assert snake.head == (120, 100)
        assert snake.length == 5
        # Movement pops (50,100) then (60,100); the BURN pops (70,100).
        assert snake.pending_trail_pellets == [(70, 100)]

    def test_v2_no_pellet_on_non_burn_boost_frame(self):
        _init_mechanics(2)
        snake = self._make_burning_snake()
        snake.boost_frames = 0  # burn not due this frame

        snake.move()

        assert snake.pending_trail_pellets == []
        assert snake.length == 6

    def test_v1_burn_drops_no_pellet(self):
        _init_mechanics(1)
        snake = self._make_burning_snake()

        snake.move()

        # Identical movement/length cost to v2, but the mass vanishes (legacy).
        assert snake.head == (120, 100)
        assert snake.length == 5
        assert snake.pending_trail_pellets == []

    def test_update_converts_pellets_to_corpse_food(self):
        """GameState.update() turns pending pellets into cap-exempt food."""
        _init_mechanics(2)

        class TrailSnake(Snake):
            total_reward = 0.0

            def update(self, other_snakes, food):
                self.move()

        snake = TrailSnake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake.length = 6
        snake.segments = [(100 - i * 10, 100) for i in range(6)]
        snake.is_boosting = True
        snake.boost_frames = GameConfig.BOOST_LENGTH_COST_FRAMES - 1

        game_state = _bare_game_state(food_manager=_make_food_manager(max_food=0))
        game_state.snakes = [snake]
        game_state.frame = 0
        game_state.alive_snakes = 1
        game_state.headless = True
        game_state.frame_collisions = {}
        game_state.frame_kills = {}
        game_state.frame_death_causes = {}
        game_state.frame_ate_food = {}
        game_state.episode_food_eaten = 0
        game_state.episode_best_length = 6
        game_state.episode_current_reward = 0.0
        game_state.episode_best_reward = 0.0
        game_state._shared_policy = None

        game_state.update(train_mode=True, learn=False)

        assert (70, 100) in game_state.food_manager.food
        assert game_state.food_manager.corpse_count == 1
        assert game_state.food_manager.ambient_count == 0
        assert snake.pending_trail_pellets == []

    def test_update_skips_out_of_arena_pellets(self):
        """Pellets outside the arena are dropped, not added as food."""
        _init_mechanics(2)

        class StubSnake:
            id = 0
            is_alive = True
            segment_size = 10
            game_width = 800
            game_height = 600
            respawn_timer = 0
            total_reward = 0.0

            def __init__(self):
                self.segments = [(50, 50)]
                self.length = 1
                self.last_move_positions = []
                self.pending_trail_pellets = []

            @property
            def head(self):
                return self.segments[0]

            def update(self, other_snakes, food):
                self.pending_trail_pellets = [(-20, 50), (150, 150)]

        snake = StubSnake()
        game_state = _bare_game_state(food_manager=_make_food_manager(max_food=0))
        game_state.snakes = [snake]
        game_state.frame = 0
        game_state.alive_snakes = 1
        game_state.headless = True
        game_state.frame_collisions = {}
        game_state.frame_kills = {}
        game_state.frame_death_causes = {}
        game_state.frame_ate_food = {}
        game_state.episode_food_eaten = 0
        game_state.episode_best_length = 1
        game_state.episode_current_reward = 0.0
        game_state.episode_best_reward = 0.0
        game_state._shared_policy = None

        game_state.update(train_mode=True, learn=False)

        assert game_state.food_manager.food == [(150, 150)]
        assert game_state.food_manager.corpse_count == 1


class TestCorpseFoodAccounting:
    """§1.3: corpse-class food is exempt from the ambient maintain_count cap."""

    def test_corpse_food_does_not_suppress_ambient_topup(self):
        _init_mechanics(2)
        fm = _make_food_manager(max_food=5)
        fm.food = [(20, 20), (40, 40), (60, 60)]  # 3 ambient pellets
        for i in range(10):  # big corpse, well above the cap
            assert fm.add_food((100 + i * 20, 100), corpse=True) is True

        assert fm.corpse_count == 10
        assert fm.ambient_count == 3

        spawned = fm.maintain_count([])

        assert spawned == 2  # tops ambient up to max_food, ignoring the corpse
        assert fm.ambient_count == 5
        assert fm.corpse_count == 10

    def test_consuming_corpse_food_updates_corpse_count(self):
        _init_mechanics(2)
        fm = _make_food_manager()
        fm.add_food((200, 200), corpse=True)

        assert fm.consume_at((205, 203), 10) is True  # same cell
        assert fm.corpse_count == 0
        assert fm.food == []

    def test_reset_clears_corpse_tracking(self):
        _init_mechanics(2)
        fm = _make_food_manager()
        fm.add_food((200, 200), corpse=True)
        fm.reset(0)
        assert fm.corpse_count == 0
        assert fm.food == []

    def test_v2_death_drops_full_corpse_as_cap_exempt_food(self):
        _init_mechanics(2)
        assert CORPSE_DROP_FRACTION_V2 == 1.0
        fm = _make_food_manager(max_food=2)
        game_state = _bare_game_state(food_manager=fm)
        snake = make_test_snake(
            0, (100, 100), segments=[(100, 100), (90, 100), (80, 100), (70, 100)]
        )

        game_state._drop_food_from_snake(snake)

        assert sorted(fm.food) == [(70, 100), (80, 100), (90, 100), (100, 100)]
        assert fm.corpse_count == 4
        assert fm.ambient_count == 0
        # Ambient top-up is unaffected by the corpse sitting above the cap.
        assert fm.maintain_count([]) == 2

    def test_v1_death_drops_half_corpse_as_ambient_food(self):
        _init_mechanics(1)
        fm = _make_food_manager(max_food=2)
        game_state = _bare_game_state(food_manager=fm)
        snake = make_test_snake(
            0, (100, 100), segments=[(100, 100), (90, 100), (80, 100), (70, 100)]
        )

        game_state._drop_food_from_snake(snake)

        # Legacy: every other segment, ambient pool (counts against the cap).
        assert sorted(fm.food) == [(80, 100), (100, 100)]
        assert fm.corpse_count == 0
        assert fm.ambient_count == 2
        assert fm.maintain_count([]) == 0  # corpse drops suppress the top-up (legacy)


class TestSizeResolvedHeadOns:
    """§1.4: >= 1.15x logical length survives a head-on and gets kill credit."""

    def _head_on_pair(self, big_length: int, small_length: int):
        big = make_test_snake(0, (200, 200))
        big.length = big_length
        small = make_test_snake(1, (205, 200))  # same cell as big's head
        small.length = small_length
        return big, small

    def test_v2_larger_snake_survives_and_gets_credit(self):
        _init_mechanics(2)
        big, small = self._head_on_pair(12, 10)  # 12 >= 1.15 * 10
        game_state = _bare_game_state()
        game_state.snakes = [big, small]

        frame_collisions = game_state.handle_collisions()

        assert big.is_alive is True
        assert small.is_alive is False
        assert frame_collisions == {1: "head"}
        assert game_state.frame_kills == {0: [1]}
        assert game_state.frame_death_causes == {1: "head_on"}
        assert game_state.episode_kills == 1
        assert game_state.episode_deaths == 1

    def test_v2_ratio_boundary_is_inclusive(self):
        _init_mechanics(2)
        big, small = self._head_on_pair(23, 20)  # 23 == 1.15 * 20 exactly
        assert big.length >= HEADON_SIZE_RATIO * small.length
        game_state = _bare_game_state()
        game_state.snakes = [big, small]

        game_state.handle_collisions()

        assert big.is_alive is True
        assert small.is_alive is False
        assert game_state.frame_kills == {0: [1]}

    def test_v2_near_equal_stays_mutual_death(self):
        _init_mechanics(2)
        big, small = self._head_on_pair(11, 10)  # 11 < 11.5: near-equal
        game_state = _bare_game_state()
        game_state.snakes = [big, small]

        frame_collisions = game_state.handle_collisions()

        assert big.is_alive is False
        assert small.is_alive is False
        assert frame_collisions == {0: "head", 1: "head"}
        assert game_state.frame_kills == {}
        assert game_state.episode_kills == 0

    def test_v1_head_on_stays_size_blind_mutual_death(self):
        _init_mechanics(1)
        big, small = self._head_on_pair(12, 10)
        game_state = _bare_game_state()
        game_state.snakes = [big, small]

        frame_collisions = game_state.handle_collisions()

        assert big.is_alive is False
        assert small.is_alive is False
        assert frame_collisions == {0: "head", 1: "head"}
        assert game_state.frame_kills == {}

    def test_v2_same_frame_dead_killer_keeps_head_on_credit(self):
        """A head-on winner that dies the same frame from a body collision is
        still credited in frame_kills (the reward side reads frame_kills)."""
        _init_mechanics(2)
        big, small = self._head_on_pair(12, 10)
        body_owner = make_test_snake(2, (300, 200), segments=[(300, 200), (200, 200), (190, 200)])
        game_state = _bare_game_state()
        game_state.snakes = [big, small, body_owner]

        frame_collisions = game_state.handle_collisions()

        # big wins the head-on against small...
        assert small.is_alive is False
        assert game_state.frame_kills.get(0) == [1]
        # ...but dies itself from body_owner's body the same frame.
        assert big.is_alive is False
        assert frame_collisions[0] == "body"
        # And body_owner is credited for big.
        assert game_state.frame_kills.get(2) == [0]
        assert game_state.episode_kills == 2


class TestPopulationFloor:
    """§1.7: training episodes end when alive snakes drop below the floor."""

    def _floor_state(self, num_snakes: int, alive: int, train_mode: bool) -> GameState:
        game_state = GameState.__new__(GameState)
        game_state.num_snakes = num_snakes
        game_state.alive_snakes = alive
        game_state._train_mode = train_mode
        return game_state

    def test_v2_train_mode_floor_reached(self):
        _init_mechanics(2)
        assert POPULATION_FLOOR_V2 == 3
        game_state = self._floor_state(6, POPULATION_FLOOR_V2 - 1, train_mode=True)
        assert game_state.population_floor_reached is True

    def test_v2_train_mode_at_floor_not_reached(self):
        _init_mechanics(2)
        game_state = self._floor_state(6, POPULATION_FLOOR_V2, train_mode=True)
        assert game_state.population_floor_reached is False

    def test_v2_non_train_mode_never_reaches_floor(self):
        _init_mechanics(2)
        game_state = self._floor_state(6, 1, train_mode=False)
        assert game_state.population_floor_reached is False

    def test_v2_small_initial_population_exempt(self):
        """Envs that START below the floor (e.g. 1-2 snakes) are exempt."""
        _init_mechanics(2)
        game_state = self._floor_state(2, 1, train_mode=True)
        assert game_state.population_floor_reached is False

    def test_v1_never_reaches_floor(self):
        _init_mechanics(1)
        game_state = self._floor_state(6, 1, train_mode=True)
        assert game_state.population_floor_reached is False

    def test_update_records_train_mode(self):
        """update() records train_mode so the property reflects the loop mode."""
        _init_mechanics(2)

        class StubSnake:
            id = 0
            is_alive = True
            segment_size = 10
            game_width = 800
            game_height = 600
            respawn_timer = 0
            total_reward = 0.0

            def __init__(self):
                self.segments = [(50, 50)]
                self.length = 1
                self.last_move_positions = []

            @property
            def head(self):
                return self.segments[0]

            def update(self, other_snakes, food):
                pass

        game_state = _bare_game_state(food_manager=_make_food_manager(max_food=0))
        game_state.snakes = [StubSnake()]
        game_state.frame = 0
        game_state.alive_snakes = 1
        game_state.headless = True
        game_state.frame_collisions = {}
        game_state.frame_kills = {}
        game_state.frame_death_causes = {}
        game_state.frame_ate_food = {}
        game_state.episode_food_eaten = 0
        game_state.episode_best_length = 1
        game_state.episode_current_reward = 0.0
        game_state.episode_best_reward = 0.0
        game_state._shared_policy = None
        game_state.num_snakes = 6

        game_state.update(train_mode=True, learn=False)
        assert game_state._train_mode is True
        assert game_state.population_floor_reached is True  # 1 alive < 3

        game_state.update(train_mode=False, learn=False)
        assert game_state._train_mode is False
        assert game_state.population_floor_reached is False
