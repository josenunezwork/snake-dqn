"""Snake factory for creating snake instances.

This module provides the SnakeFactory class which handles creation of
different snake types (AI snakes, human-controlled snakes) with proper
dependency injection. Uses Apex policy for AI snakes.
"""

from typing import TYPE_CHECKING, Callable, Optional, Tuple

from src.core.game_config import GameConfig

if TYPE_CHECKING:
    from src.game.ai_snake import AISnake
    from src.game.human_snake import HumanSnake
    from src.training.apex_policy import ApexPolicy


class SnakeFactory:
    """Factory for creating snake instances.

    This factory centralizes snake creation logic, ensuring consistent
    initialization across the codebase. It supports both AI-controlled
    (using Apex policy) and human-controlled snakes.

    Usage:
        # Create an AI snake with Apex policy
        snake = SnakeFactory.create_ai_snake(
            snake_id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            get_frame=lambda: game_state.frame,
            set_frame=lambda f: setattr(game_state, 'frame', f)
        )

        # Create a human-controlled snake
        snake = SnakeFactory.create_human_snake(
            snake_id=0,
            color=(255, 0, 0),
            start_pos=(100, 100)
        )
    """

    @staticmethod
    def create_ai_snake(
        snake_id: int,
        color: Tuple[int, int, int],
        start_pos: Tuple[int, int],
        game_width: Optional[int] = None,
        game_height: Optional[int] = None,
        segment_size: Optional[int] = None,
        food_capacity: Optional[int] = None,
        get_frame: Optional[Callable[[], int]] = None,
        set_frame: Optional[Callable[[int], None]] = None,
        policy: Optional["ApexPolicy"] = None,
        actor_id: int = 0,
        num_actors: int = 1,
    ) -> "AISnake":
        """Create an AI-controlled snake using Apex policy.

        Args:
            snake_id: Unique identifier for the snake
            color: RGB color tuple for the snake
            start_pos: (x, y) starting position
            game_width: Game area width (defaults to GameConfig.WIDTH)
            game_height: Game area height (defaults to GameConfig.HEIGHT)
            segment_size: Size of snake segments (defaults to GameConfig.SEGMENT_SIZE)
            food_capacity: Effective max food count for state density features
            get_frame: Callback to get current game frame
            set_frame: Callback to set game frame (for checkpoint loading)
            policy: Optional pre-created policy instance (if None, creates Apex policy)
            actor_id: Actor index for Apex epsilon differentiation (0-indexed)
            num_actors: Total number of actors sharing this policy

        Returns:
            Configured AISnake instance with Apex policy
        """
        # Use defaults from GameConfig if not specified
        game_width = game_width or GameConfig.WIDTH
        game_height = game_height or GameConfig.HEIGHT
        segment_size = segment_size or GameConfig.SEGMENT_SIZE
        food_capacity = GameConfig.MAX_FOOD if food_capacity is None else food_capacity

        # Create Apex policy if not provided
        if policy is None:
            from src.training.apex_policy import ApexPolicy

            policy = ApexPolicy(
                GameConfig.INPUT_SIZE,
                GameConfig.HIDDEN_SIZE,
                GameConfig.OUTPUT_SIZE,
                use_gru=GameConfig.USE_GRU,
            )

        # Import here to avoid circular imports
        from src.game.ai_snake import AISnake

        return AISnake(
            id=snake_id,
            color=color,
            start_pos=start_pos,
            segment_size=segment_size,
            game_width=game_width,
            game_height=game_height,
            policy=policy,
            policy_type="apex",
            get_frame=get_frame,
            set_frame=set_frame,
            actor_id=actor_id,
            num_actors=num_actors,
            food_capacity=food_capacity,
        )

    @staticmethod
    def create_ai_snake_with_policy(
        snake_id: int,
        color: Tuple[int, int, int],
        start_pos: Tuple[int, int],
        policy: "ApexPolicy",
        game_width: Optional[int] = None,
        game_height: Optional[int] = None,
        segment_size: Optional[int] = None,
        food_capacity: Optional[int] = None,
        get_frame: Optional[Callable[[], int]] = None,
        set_frame: Optional[Callable[[int], None]] = None,
        actor_id: int = 0,
        num_actors: int = 1,
    ) -> "AISnake":
        """Create an AI-controlled snake with an existing Apex policy instance.

        This is useful for testing or when policies need custom configuration.

        Args:
            snake_id: Unique identifier for the snake
            color: RGB color tuple for the snake
            start_pos: (x, y) starting position
            policy: Pre-created Apex policy instance
            game_width: Game area width (defaults to GameConfig.WIDTH)
            game_height: Game area height (defaults to GameConfig.HEIGHT)
            segment_size: Size of snake segments (defaults to GameConfig.SEGMENT_SIZE)
            food_capacity: Effective max food count for state density features
            get_frame: Callback to get current game frame
            set_frame: Callback to set game frame
            actor_id: Actor index for Apex epsilon differentiation (0-indexed)
            num_actors: Total number of actors sharing this policy

        Returns:
            Configured AISnake instance with Apex policy
        """
        return SnakeFactory.create_ai_snake(
            snake_id=snake_id,
            color=color,
            start_pos=start_pos,
            game_width=game_width,
            game_height=game_height,
            segment_size=segment_size,
            food_capacity=food_capacity,
            get_frame=get_frame,
            set_frame=set_frame,
            policy=policy,
            actor_id=actor_id,
            num_actors=num_actors,
        )

    @staticmethod
    def create_human_snake(
        snake_id: int,
        color: Tuple[int, int, int],
        start_pos: Tuple[int, int],
        game_width: Optional[int] = None,
        game_height: Optional[int] = None,
        segment_size: Optional[int] = None,
        food_capacity: Optional[int] = None,
    ) -> "HumanSnake":
        """Create a human-controlled snake.

        Args:
            snake_id: Unique identifier for the snake
            color: RGB color tuple for the snake
            start_pos: (x, y) starting position
            game_width: Game area width (defaults to GameConfig.WIDTH)
            game_height: Game area height (defaults to GameConfig.HEIGHT)
            segment_size: Size of snake segments (defaults to GameConfig.SEGMENT_SIZE)
            food_capacity: Effective max food count for state density features

        Returns:
            Configured HumanSnake instance
        """
        # Use defaults from GameConfig if not specified
        game_width = game_width or GameConfig.WIDTH
        game_height = game_height or GameConfig.HEIGHT
        segment_size = segment_size or GameConfig.SEGMENT_SIZE
        food_capacity = GameConfig.MAX_FOOD if food_capacity is None else food_capacity

        # Import here to avoid circular imports
        from src.game.human_snake import HumanSnake

        return HumanSnake(
            id=snake_id,
            color=color,
            start_pos=start_pos,
            segment_size=segment_size,
            game_width=game_width,
            game_height=game_height,
            food_capacity=food_capacity,
        )

    @staticmethod
    def create_snakes_for_game(
        num_snakes: int,
        position_generator: Callable[[], Tuple[int, int]],
        human_mode: bool = False,
        get_frame: Optional[Callable[[], int]] = None,
        set_frame: Optional[Callable[[int], None]] = None,
        shared_policy: Optional["ApexPolicy"] = None,
        game_width: Optional[int] = None,
        game_height: Optional[int] = None,
        segment_size: Optional[int] = None,
        food_capacity: Optional[int] = None,
    ) -> list:
        """Create all snakes for a game with a shared Apex policy.

        All AI snakes share a single policy instance (network, optimizer,
        replay buffer). Each snake gets a different actor_id for Apex-style
        epsilon differentiation, enabling diverse exploration strategies.

        Args:
            num_snakes: Number of snakes to create
            position_generator: Function to generate starting positions
            human_mode: If True, first snake is human-controlled
            get_frame: Callback to get current game frame
            set_frame: Callback to set game frame
            shared_policy: Optional pre-existing policy to reuse across
                curriculum promotions. If None, a new policy is created.
            game_width: Effective game width for snake state and collisions.
            game_height: Effective game height for snake state and collisions.
            segment_size: Snake segment size.
            food_capacity: Effective max food count for state density features.

        Returns:
            List of snake instances (AI snakes share one Apex policy)
        """
        snakes = []
        game_width = game_width or GameConfig.WIDTH
        game_height = game_height or GameConfig.HEIGHT
        segment_size = segment_size or GameConfig.SEGMENT_SIZE
        food_capacity = GameConfig.MAX_FOOD if food_capacity is None else food_capacity

        from src.game.game_logic import GameLogic

        def get_non_overlapping_start_pos() -> Tuple[int, int]:
            candidate = position_generator()
            if not snakes or not GameLogic.position_overlaps_snakes(candidate, snakes):
                return candidate

            empty_position = GameLogic.find_empty_position(game_width, game_height, snakes)
            return empty_position if empty_position is not None else candidate

        # Count how many AI snakes we'll create (for epsilon differentiation)
        num_ai_snakes = num_snakes - (1 if human_mode else 0)

        # Reuse provided policy or create ONE shared policy for all AI snakes
        if shared_policy is None and num_ai_snakes > 0:
            from src.training.apex_policy import ApexPolicy

            shared_policy = ApexPolicy(
                GameConfig.INPUT_SIZE,
                GameConfig.HIDDEN_SIZE,
                GameConfig.OUTPUT_SIZE,
                use_gru=GameConfig.USE_GRU,
            )

        ai_index = 0
        for i in range(num_snakes):
            color = GameConfig.SNAKE_COLORS[i % len(GameConfig.SNAKE_COLORS)]
            start_pos = get_non_overlapping_start_pos()

            if human_mode and i == 0:
                snake = SnakeFactory.create_human_snake(
                    snake_id=i,
                    color=color,
                    start_pos=start_pos,
                    game_width=game_width,
                    game_height=game_height,
                    segment_size=segment_size,
                    food_capacity=food_capacity,
                )
            else:
                snake = SnakeFactory.create_ai_snake(
                    snake_id=i,
                    color=color,
                    start_pos=start_pos,
                    game_width=game_width,
                    game_height=game_height,
                    segment_size=segment_size,
                    food_capacity=food_capacity,
                    get_frame=get_frame,
                    set_frame=set_frame,
                    policy=shared_policy,
                    actor_id=ai_index,
                    num_actors=num_ai_snakes,
                )
                ai_index += 1

            snakes.append(snake)

        return snakes
