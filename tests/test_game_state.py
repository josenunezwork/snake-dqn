"""Tests for GameState lifecycle, scaled-board setup, spawning, and food handling."""

from src.core.game_config import (
    AppConfig,
    GameConfig,
    GameSettings,
    get_config,
    initialize_config,
)
from src.game.game_logic import GameLogic
from src.game.game_state import GameState


class FakePolicy:
    """Small stand-in for GameState construction tests."""

    epsilon = 0.0


class TestGameStateBoardScale:
    """Tests for curriculum board-scale wiring."""

    def test_board_scale_is_applied_to_snake_dimensions(self):
        original_config = get_config()
        config = AppConfig(
            game=GameSettings(
                width=800,
                height=600,
                num_snakes=1,
                initial_food=1,
                max_food=1,
            )
        )

        try:
            initialize_config(config)
            game_state = GameState(
                headless=True,
                num_snakes=1,
                shared_policy=FakePolicy(),
                board_scale=0.5,
            )

            snake = game_state.snakes[0]

            assert game_state.food_manager.game_width == 400
            assert game_state.food_manager.game_height == 300
            assert snake.game_width == 400
            assert snake.game_height == 300
            assert 0 <= snake.head[0] < 400
            assert 0 <= snake.head[1] < 300
        finally:
            initialize_config(original_config)

    def test_food_multiplier_is_applied_to_snake_density_capacity(self):
        original_config = get_config()
        config = AppConfig(
            game=GameSettings(
                width=800,
                height=600,
                num_snakes=1,
                initial_food=4,
                max_food=100,
            )
        )

        try:
            initialize_config(config)
            game_state = GameState(
                headless=True,
                num_snakes=1,
                shared_policy=FakePolicy(),
                food_multiplier=0.25,
            )

            snake = game_state.snakes[0]

            assert game_state.food_manager.max_food == 25
            assert snake.food_capacity == 25
        finally:
            initialize_config(original_config)

    def test_scaled_snake_dimensions_drive_wall_collision(self):
        original_config = get_config()
        config = AppConfig(
            game=GameSettings(
                width=800,
                height=600,
                num_snakes=1,
                initial_food=1,
                max_food=1,
            )
        )

        try:
            initialize_config(config)
            game_state = GameState(
                headless=True,
                num_snakes=1,
                shared_policy=FakePolicy(),
                board_scale=0.5,
            )

            snake = game_state.snakes[0]
            snake.segments = [(400, 100)]

            assert GameLogic.check_wall_collision(snake)
        finally:
            initialize_config(original_config)


class TestGameStateSpawnPositions:
    """Tests for non-overlapping snake placement at episode boundaries."""

    def _make_two_snake_config(self):
        return AppConfig(
            game=GameSettings(
                width=200,
                height=200,
                num_snakes=2,
                initial_food=0,
                max_food=0,
                frame_rate=1,
            )
        )

    def test_initial_snake_creation_avoids_previous_snake_position(self, monkeypatch):
        original_config = get_config()
        positions = iter([(50, 50), (50, 50)])

        def repeated_position(game_state):
            return next(positions, (50, 50))

        def find_empty_position(width, height, snakes):
            assert len(snakes) == 1
            return (90, 50)

        try:
            initialize_config(self._make_two_snake_config())
            monkeypatch.setattr(GameState, "get_random_position", repeated_position)
            monkeypatch.setattr(
                GameLogic,
                "find_empty_position",
                staticmethod(find_empty_position),
            )

            game_state = GameState(
                headless=True,
                num_snakes=2,
                shared_policy=FakePolicy(),
            )

            assert [snake.head for snake in game_state.snakes] == [(50, 50), (90, 50)]
        finally:
            initialize_config(original_config)

    def test_episode_reset_avoids_already_repositioned_snakes(self, monkeypatch):
        original_config = get_config()

        def find_empty_position(width, height, snakes):
            assert len(snakes) == 1
            return (90, 50)

        try:
            initialize_config(self._make_two_snake_config())
            game_state = GameState(
                headless=True,
                num_snakes=2,
                shared_policy=FakePolicy(),
            )
            monkeypatch.setattr(game_state, "get_random_position", lambda: (50, 50))
            monkeypatch.setattr(
                GameLogic,
                "find_empty_position",
                staticmethod(find_empty_position),
            )

            game_state.reset()

            assert [snake.head for snake in game_state.snakes] == [(50, 50), (90, 50)]
        finally:
            initialize_config(original_config)


class TestGameStateRespawnMode:
    """Tests for terminal training episodes versus continuous-play respawns."""

    def test_training_update_keeps_dead_snake_terminal_until_reset(self):
        original_config = get_config()
        config = AppConfig(
            game=GameSettings(
                width=200,
                height=200,
                num_snakes=1,
                initial_food=0,
                max_food=0,
                frame_rate=1,
            )
        )

        try:
            initialize_config(config)
            game_state = GameState(
                headless=True,
                num_snakes=1,
                shared_policy=FakePolicy(),
            )
            snake = game_state.snakes[0]
            snake.update = lambda *args, **kwargs: None
            snake.die()
            snake.respawn_timer = 0
            game_state.alive_snakes = 0

            game_state.update(train_mode=True, learn=False)

            assert not snake.is_alive
            assert game_state.alive_snakes == 0

            game_state.reset()

            assert snake.is_alive
            assert game_state.alive_snakes == 1
        finally:
            initialize_config(original_config)

    def test_default_update_still_respawns_dead_snakes_for_continuous_play(self):
        original_config = get_config()
        config = AppConfig(
            game=GameSettings(
                width=200,
                height=200,
                num_snakes=1,
                initial_food=0,
                max_food=0,
                frame_rate=1,
            )
        )

        try:
            initialize_config(config)
            game_state = GameState(
                headless=True,
                num_snakes=1,
                shared_policy=FakePolicy(),
            )
            snake = game_state.snakes[0]
            snake.update = lambda *args, **kwargs: None
            snake.die()
            snake.respawn_timer = 0
            game_state.alive_snakes = 0

            game_state.update(train_mode=False, learn=False)

            assert snake.is_alive
            assert game_state.alive_snakes == 1
        finally:
            initialize_config(original_config)


class TestGameStateEpisodeRewardTracking:
    """Tests for current versus best episode reward metrics."""

    def test_current_reward_preserves_negative_episode_totals(self):
        """Training diagnostics should not hide bad episodes behind initial zero."""
        original_config = get_config()
        config = AppConfig(
            game=GameSettings(
                width=200,
                height=200,
                num_snakes=1,
                initial_food=0,
                max_food=0,
                frame_rate=1,
            )
        )

        try:
            initialize_config(config)
            game_state = GameState(
                headless=True,
                num_snakes=1,
                shared_policy=FakePolicy(),
            )
            snake = game_state.snakes[0]
            snake.update = lambda *args, **kwargs: None
            snake._total_reward = -2.5

            game_state.update(train_mode=True, learn=False)

            assert game_state.episode_current_reward == -2.5
            assert game_state.episode_best_reward == 0.0
        finally:
            initialize_config(original_config)


class TestGameStateMovePathLifecycle:
    """Tests for frame-local movement path bookkeeping."""

    def test_update_clears_stale_move_paths_before_collision_detection(self):
        original_config = get_config()
        config = AppConfig(
            game=GameSettings(
                width=200,
                height=200,
                num_snakes=2,
                initial_food=0,
                max_food=0,
                frame_rate=1,
            )
        )

        try:
            initialize_config(config)
            game_state = GameState(
                headless=True,
                num_snakes=2,
                shared_policy=FakePolicy(),
            )
            snake1, snake2 = game_state.snakes
            snake1.update = lambda *args, **kwargs: None
            snake2.update = lambda *args, **kwargs: None
            snake1.segments = [(50, 50)]
            snake1.length = 1
            snake2.segments = [(100, 100)]
            snake2.length = 1
            snake1.last_move_positions = [snake2.head]

            game_state.update(train_mode=True, learn=False)

            assert snake1.is_alive
            assert snake2.is_alive
            assert game_state.frame_collisions == {}
            assert snake1.last_move_positions == []
        finally:
            initialize_config(original_config)


class TestGameStateActionSelectionOrder:
    """Tests for unbiased action-observation ordering."""

    def test_later_snakes_observe_pre_frame_positions(self):
        """Every snake should choose from the same pre-frame world snapshot."""
        original_config = get_config()
        config = AppConfig(
            game=GameSettings(
                width=200,
                height=200,
                num_snakes=2,
                initial_food=0,
                max_food=0,
                frame_rate=1,
            )
        )

        try:
            initialize_config(config)
            game_state = GameState(
                headless=True,
                num_snakes=2,
                shared_policy=FakePolicy(),
            )
            snake1, snake2 = game_state.snakes
            snake1.segments = [(50, 50)]
            snake1.length = 1
            snake2.segments = [(100, 50)]
            snake2.length = 1
            observed_heads = {}

            def update_first(other_snakes, food):
                observed_heads[snake1.id] = [snake.head for snake in other_snakes]
                snake1.segments = [(60, 50)]
                snake1.last_move_positions = [(60, 50)]

            def update_second(other_snakes, food):
                observed_heads[snake2.id] = [snake.head for snake in other_snakes]
                snake2.segments = [(110, 50)]
                snake2.last_move_positions = [(110, 50)]

            snake1.update = update_first
            snake2.update = update_second

            game_state.update(train_mode=True, learn=False)

            assert observed_heads[snake1.id] == [(50, 50), (100, 50)]
            assert observed_heads[snake2.id] == [(50, 50), (100, 50)]
        finally:
            initialize_config(original_config)

    def test_headless_updates_disable_display_q_recording(self):
        """Headless training should tell AI snakes to skip display-only Q capture."""
        original_config = get_config()
        config = AppConfig(
            game=GameSettings(
                width=200,
                height=200,
                num_snakes=1,
                initial_food=0,
                max_food=0,
                frame_rate=1,
            )
        )

        try:
            initialize_config(config)
            game_state = GameState(
                headless=True,
                num_snakes=1,
                shared_policy=FakePolicy(),
            )
            snake = game_state.snakes[0]
            snake.segments = [(50, 50)]
            snake.length = 1
            observed_record_flags = []

            def update_spy(other_snakes, food):
                observed_record_flags.append(snake.record_q_values)
                snake.segments = [(60, 50)]
                snake.last_move_positions = [(60, 50)]

            snake.update = update_spy

            game_state.update(train_mode=True, learn=False)

            assert observed_record_flags == [False]
        finally:
            initialize_config(original_config)


class TestGameStateFoodConsumption:
    """Tests for food checks owned by GameState."""

    def _make_single_snake_game(self):
        config = AppConfig(
            game=GameSettings(
                width=200,
                height=200,
                num_snakes=1,
                initial_food=0,
                max_food=0,
                frame_rate=1,
            )
        )
        initialize_config(config)
        return GameState(
            headless=True,
            num_snakes=1,
            shared_policy=FakePolicy(),
        )

    def _place_long_snake(self, game_state):
        snake = game_state.snakes[0]
        snake.direction = (1, 0)
        snake.length = GameConfig.MIN_BOOST_LENGTH
        snake.segments = [(100 - i * 10, 100) for i in range(snake.length)]
        return snake

    def test_normal_move_consumes_food_at_head(self):
        """Normal movement should still eat food at the final head position."""
        original_config = get_config()
        try:
            game_state = self._make_single_snake_game()
            snake = self._place_long_snake(game_state)
            game_state.food_manager.food = [(110, 100)]
            initial_length = snake.length

            snake.move()
            ate_food = game_state.check_food_consumption(snake)

            assert ate_food is True
            assert snake.head == (110, 100)
            assert snake.length == initial_length + 1
            assert game_state.food_manager.food == []
        finally:
            initialize_config(original_config)

    def test_training_replaces_eaten_food_before_reward_capture(self):
        """Replay next_state capture should see the food supply for the next action."""
        game_state = GameState.__new__(GameState)
        events = []
        captured = {}

        class FoodManagerStub:
            def __init__(self):
                self.food = [(60, 50)]
                self.maintain_calls = 0

            def maintain_count(self, snakes):
                self.maintain_calls += 1
                if self.maintain_calls == 1:
                    events.append("maintain_start")
                    return 0
                events.append("maintain_after_eat")
                assert self.food == []
                self.food.append((120, 50))
                return 1

            def consume_at(self, position, radius):
                if position == (60, 50):
                    self.food = []
                    return True
                return False

        class SnakeStub:
            id = 0
            is_alive = True
            segment_size = 10
            record_q_values = True

            def __init__(self):
                self.segments = [(50, 50)]
                self.length = 1
                self.last_move_positions = []
                self._total_reward = 0.0

            @property
            def total_reward(self):
                return self._total_reward

            def update(self, other_snakes, food):
                events.append("update")
                self.segments = [(60, 50)]
                self.last_move_positions = [(60, 50)]

            def grow(self, amount=1):
                self.length += amount

            def compute_reward_and_train(
                self,
                other_snakes,
                food,
                ate_food=False,
                collided=False,
                frame_kills=None,
            ):
                events.append("reward")
                captured["ate_food"] = ate_food
                captured["food"] = list(food)

        snake = SnakeStub()
        game_state.snakes = [snake]
        game_state.frame = 0
        game_state.alive_snakes = 1
        game_state.headless = True
        game_state.food_manager = FoodManagerStub()
        game_state.frame_collisions = {}
        game_state.frame_kills = {}
        game_state.episode_food_eaten = 0
        game_state.episode_best_length = 1
        game_state.episode_current_reward = 0.0
        game_state.episode_best_reward = 0.0
        game_state._shared_policy = None
        game_state.handle_collisions = lambda: {}

        game_state.update(train_mode=True, learn=False)

        assert events == ["maintain_start", "update", "maintain_after_eat", "reward"]
        assert captured == {"ate_food": True, "food": [(120, 50)]}
        assert game_state.episode_food_eaten == 1

    def test_boost_consumes_food_crossed_between_start_and_final_head(self):
        """Boosted movement should not skip food on the intermediate cell."""
        original_config = get_config()
        try:
            game_state = self._make_single_snake_game()
            snake = self._place_long_snake(game_state)
            snake.is_boosting = True
            game_state.food_manager.food = [(110, 100)]
            initial_length = snake.length

            snake.move()
            ate_food = game_state.check_food_consumption(snake)

            assert snake.head == (120, 100)
            assert snake.last_move_positions == [(110, 100), (120, 100)]
            assert ate_food is True
            assert snake.length == initial_length + 1
            assert game_state.food_manager.food == []
        finally:
            initialize_config(original_config)

    def test_death_drop_food_skips_existing_food(self):
        """Body-to-food drops should not recreate duplicate food targets."""
        original_config = get_config()
        try:
            game_state = self._make_single_snake_game()
            snake = game_state.snakes[0]
            snake.segments = [(100, 100), (90, 100), (80, 100)]
            snake.length = 3
            game_state.food_manager.food = [(100, 100)]

            game_state._drop_food_from_snake(snake)

            assert game_state.food_manager.food == [(100, 100), (80, 100)]
        finally:
            initialize_config(original_config)
