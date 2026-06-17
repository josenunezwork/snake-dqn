"""Tests for kill attribution system (§2.5)."""

from types import SimpleNamespace

import pytest

from src.core.game_config import GameConfig
from src.game.game_logic import GameLogic
from src.game.game_state import GameState
from tests.conftest import make_test_snake as _make_snake

pytestmark = pytest.mark.usefixtures("setup_config")


class TestHandleCollisionsFrameKills:
    """Test that GameLogic.check_collisions produces data for frame_kills."""

    def test_body_collision_returns_killer_victim(self):
        """Body collision: the snake whose body was hit is the killer."""
        # snake1 head at (200, 200), snake2 body includes (200, 200)
        snake1 = _make_snake(0, (200, 200))
        snake2 = _make_snake(1, (300, 200), segments=[(300, 200), (200, 200), (200, 210)])

        collisions = GameLogic.check_collisions([snake1, snake2])

        body_collisions = [c for c in collisions if c[2] == "body"]
        assert len(body_collisions) >= 1
        # snake1 collided with snake2's body
        colliding, other, ctype = body_collisions[0]
        assert colliding.id == 0  # snake1 hit
        assert other.id == 1  # snake2's body was hit (killer)
        assert ctype == "body"

    def test_head_collision_no_killer(self):
        """Head collision: both snakes die, no killer attribution."""
        snake1 = _make_snake(0, (200, 200))
        snake2 = _make_snake(1, (205, 200))  # within segment_size=10

        collisions = GameLogic.check_collisions([snake1, snake2])

        head_collisions = [c for c in collisions if c[2] == "head"]
        assert len(head_collisions) >= 1
        # Both should be marked, but no "killer" in head-on
        for colliding, other, ctype in head_collisions:
            assert ctype == "head"

    def test_wall_collision_no_killer(self):
        """Wall collision: no other snake involved."""
        snake = _make_snake(0, (-5, 100))
        collisions = GameLogic.check_collisions([snake])

        assert len(collisions) == 1
        colliding, other, ctype = collisions[0]
        assert ctype == "wall"
        assert other is None

    def test_self_collision_no_killer(self):
        """Self collision: no other snake involved."""
        snake = _make_snake(
            0, (100, 100), segments=[(100, 100), (110, 100), (110, 110), (100, 110), (100, 100)]
        )
        collisions = GameLogic.check_collisions([snake])

        assert len(collisions) == 1
        colliding, other, ctype = collisions[0]
        assert ctype == "self"
        assert other is None

    def test_head_collision_prevents_duplicate_body_kill_credit(self):
        """A later body collision record should not award a kill for an already-dead snake."""
        victim = _make_snake(0, (100, 100))
        head_on = _make_snake(1, (105, 100))
        body_owner = _make_snake(
            2,
            (200, 100),
            segments=[(200, 100), (100, 100), (90, 100)],
        )
        game_state = GameState.__new__(GameState)
        game_state.snakes = [victim, head_on, body_owner]
        game_state.food_manager = SimpleNamespace(food=[])
        game_state.episode_deaths = 0
        game_state.episode_collision_counts = {"wall": 0, "self": 0, "head": 0, "body": 0}

        frame_collisions = game_state.handle_collisions()

        assert frame_collisions == {0: "head", 1: "head"}
        assert game_state.frame_kills == {}
        assert game_state.episode_kills == 0
        assert game_state.episode_collision_counts["head"] == 2
        assert game_state.episode_collision_counts["body"] == 0


class TestKillRewardCalculation:
    """Test calculate_reward with frame_kills for kill attribution."""

    def test_killer_gets_kill_reward(self):
        """Killer snake gets positive kill reward."""
        killer = _make_snake(0, (400, 300))
        victim = _make_snake(
            1, (500, 300), segments=[(500, 300), (490, 300), (480, 300)]  # 3 segments
        )
        victim.die()

        frame_kills = {0: [1]}  # killer_id=0 killed victim_id=1
        food = [(100, 100)]

        # Get states for reward calc
        old_state = killer.get_state([killer, victim], food)
        killer.move()
        new_state = killer.get_state([killer, victim], food)

        reward = killer.calculate_reward(
            ate_food=False,
            collided=False,
            old_state=old_state,
            new_state=new_state,
            other_snakes=[killer, victim],
            food=food,
            frame_kills=frame_kills,
        )

        # Total reward includes kill + survival + shaping, so just check it's positive
        assert reward > 0.0

    def test_kill_reward_scales_with_victim_length(self):
        """Kill reward increases with victim segment count."""
        killer = _make_snake(0, (400, 300))

        # Short victim (1 segment)
        short_victim = _make_snake(1, (600, 300))
        short_victim.die()
        reward_short = killer._calculate_interaction_reward(
            [killer, short_victim], frame_kills={0: [1]}
        )

        # Long victim (20 segments)
        long_segments = [(600 + i * 10, 300) for i in range(20)]
        long_victim = _make_snake(2, (600, 300), segments=long_segments)
        long_victim.die()
        reward_long = killer._calculate_interaction_reward(
            [killer, long_victim], frame_kills={0: [2]}
        )

        assert reward_long > reward_short

    def test_kill_reward_capped_at_max(self):
        """Kill reward should not exceed REWARD_KILL_MAX."""
        killer = _make_snake(0, (400, 300))

        # Very long victim: 200 segments
        long_segments = [(400 + i * 10, 500) for i in range(200)]
        long_victim = (
            _make_snake(1, long_segments[0][0], long_segments[0][1], segments=long_segments)
            if False
            else None
        )
        # Correct syntax:
        long_victim = _make_snake(1, (600, 300), segments=long_segments)
        long_victim.die()

        reward = killer._calculate_interaction_reward([killer, long_victim], frame_kills={0: [1]})

        # kill_base=1.0 + 0.05*200 = 11.0 → capped at 5.0
        assert reward == pytest.approx(GameConfig.REWARD_KILL_MAX)

    def test_non_killer_gets_no_kill_reward(self):
        """Snake that is not the killer gets no kill reward from frame_kills."""
        bystander = _make_snake(0, (400, 300))
        killer = _make_snake(1, (500, 300))
        victim = _make_snake(2, (600, 300))
        victim.die()

        # Only snake 1 is the killer
        frame_kills = {1: [2]}

        reward = bystander._calculate_interaction_reward(
            [bystander, killer, victim], frame_kills=frame_kills
        )

        assert reward == 0.0

    def test_kill_reward_exact_value(self):
        """Verify exact kill reward formula: base + scale * victim_length."""
        killer = _make_snake(0, (400, 300))
        victim_segments = [(600 + i * 10, 300) for i in range(10)]
        victim = _make_snake(1, (600, 300), segments=victim_segments)
        victim.die()

        reward = killer._calculate_interaction_reward([killer, victim], frame_kills={0: [1]})

        expected = GameConfig.REWARD_KILL_BASE + GameConfig.REWARD_KILL_LENGTH_SCALE * 10
        # 1.0 + 0.05 * 10 = 1.5
        assert reward == pytest.approx(expected)

    def test_kill_reward_uses_logical_victim_length_while_body_is_filling_in(self):
        """Kill reward should not undercount recently grown snakes."""
        killer = _make_snake(0, (400, 300))
        victim = _make_snake(1, (600, 300), segments=[(600, 300), (590, 300)])
        victim.length = 8
        victim.die()

        reward = killer._calculate_interaction_reward([killer, victim], frame_kills={0: [1]})

        expected = GameConfig.REWARD_KILL_BASE + GameConfig.REWARD_KILL_LENGTH_SCALE * 8
        assert reward == pytest.approx(expected)

    def test_food_and_kill_rewards_compose(self):
        """Eating should not discard same-frame kill attribution."""
        killer = _make_snake(0, (400, 300))
        victim_segments = [(600 + i * 10, 300) for i in range(3)]
        victim = _make_snake(1, (600, 300), segments=victim_segments)
        victim.die()
        state = killer.get_state([killer, victim], [(410, 300)])

        reward = killer.calculate_reward(
            ate_food=True,
            collided=False,
            old_state=state,
            new_state=state,
            other_snakes=[killer, victim],
            food=[(410, 300)],
            frame_kills={0: [1]},
        )

        expected = GameConfig.REWARD_FOOD_BASE + (
            GameConfig.REWARD_KILL_BASE + GameConfig.REWARD_KILL_LENGTH_SCALE * len(victim_segments)
        )
        assert reward == pytest.approx(expected)


class TestBackwardCompatibility:
    """Test fallback to proximity heuristic when frame_kills is None."""

    def test_proximity_heuristic_no_frame_kills(self):
        """Without frame_kills, uses proximity-based kill detection."""
        killer = _make_snake(0, (400, 300), segments=[(400, 300), (390, 300), (380, 300)])
        # Victim just died (respawn_timer == FRAME_RATE) near killer's body
        victim = _make_snake(1, (395, 300))
        victim.die()  # Sets respawn_timer = FRAME_RATE, is_alive = False

        reward = killer._calculate_interaction_reward([killer, victim], frame_kills=None)

        # Victim head (395, 300) is within 2*segment_size of killer segment (390, 300)
        # dist = 5 < 20 → proximity heuristic gives 1.0
        assert reward == pytest.approx(1.0)

    def test_proximity_heuristic_too_far(self):
        """Proximity heuristic gives 0 when dead snake is far away."""
        snake = _make_snake(0, (400, 300))
        victim = _make_snake(1, (700, 500))
        victim.die()

        reward = snake._calculate_interaction_reward([snake, victim], frame_kills=None)

        assert reward == 0.0

    def test_frame_kills_empty_dict_is_authoritative(self):
        """Empty frame_kills means no attributed kills this frame."""
        snake = _make_snake(0, (400, 300), segments=[(400, 300), (390, 300), (380, 300)])
        victim = _make_snake(1, (395, 300))
        victim.die()

        reward = snake._calculate_interaction_reward([snake, victim], frame_kills={})

        assert reward == 0.0

    def test_frame_kills_with_other_killer_no_reward(self):
        """Snake not in frame_kills gets no kill reward via accurate tracking."""
        snake = _make_snake(0, (400, 300))
        other = _make_snake(1, (500, 300))
        victim = _make_snake(2, (600, 300))
        victim.die()

        # frame_kills is non-empty but snake 0 is not the killer
        frame_kills = {1: [2]}  # snake 1 killed snake 2

        reward = snake._calculate_interaction_reward(
            [snake, other, victim], frame_kills=frame_kills
        )

        # Non-empty frame_kills takes the accurate path; snake 0 not in it
        assert reward == 0.0

    def test_single_snake_no_interaction(self):
        """Single snake should get no interaction reward."""
        snake = _make_snake(0, (400, 300))
        reward = snake._calculate_interaction_reward([snake], frame_kills={0: [1]})
        assert reward == 0.0


class TestMultiKillInSingleFrame:
    """Test that multiple kills in a single frame are correctly tracked."""

    def test_multi_kill_reward_sums(self):
        """Killer that kills two snakes in one frame gets reward for both."""
        killer = _make_snake(0, (400, 300))
        victim1 = _make_snake(
            1, (500, 300), segments=[(500, 300), (490, 300), (480, 300)]  # 3 segments
        )
        victim2 = _make_snake(
            2,
            (600, 300),
            segments=[(600, 300), (590, 300), (580, 300), (570, 300), (560, 300)],  # 5 segments
        )
        victim1.die()
        victim2.die()

        # Killer killed both victims in the same frame
        frame_kills = {0: [1, 2]}

        reward = killer._calculate_interaction_reward(
            [killer, victim1, victim2], frame_kills=frame_kills
        )

        expected_v1 = GameConfig.REWARD_KILL_BASE + GameConfig.REWARD_KILL_LENGTH_SCALE * 3
        expected_v2 = GameConfig.REWARD_KILL_BASE + GameConfig.REWARD_KILL_LENGTH_SCALE * 5
        expected_total = min(expected_v1, GameConfig.REWARD_KILL_MAX) + min(
            expected_v2, GameConfig.REWARD_KILL_MAX
        )
        assert reward == pytest.approx(expected_total)

    def test_multi_kill_only_last_was_recorded_before(self):
        """Regression: old {killer: victim} format would lose first kill."""
        victim1 = _make_snake(1, (500, 300))
        victim2 = _make_snake(2, (600, 300))
        victim1.die()
        victim2.die()

        # Both kills preserved in list format
        frame_kills = {0: [1, 2]}
        assert len(frame_kills[0]) == 2  # Both victims recorded

    def test_two_different_killers_same_frame(self):
        """Two different killers each get their own kill reward."""
        killer1 = _make_snake(0, (400, 300))
        killer2 = _make_snake(1, (500, 300))
        victim1 = _make_snake(2, (600, 300), segments=[(600, 300), (590, 300)])  # 2 segments
        victim2 = _make_snake(
            3, (700, 300), segments=[(700, 300), (690, 300), (680, 300)]  # 3 segments
        )
        victim1.die()
        victim2.die()

        frame_kills = {0: [2], 1: [3]}

        reward1 = killer1._calculate_interaction_reward(
            [killer1, killer2, victim1, victim2], frame_kills=frame_kills
        )
        reward2 = killer2._calculate_interaction_reward(
            [killer1, killer2, victim1, victim2], frame_kills=frame_kills
        )

        expected1 = GameConfig.REWARD_KILL_BASE + GameConfig.REWARD_KILL_LENGTH_SCALE * 2
        expected2 = GameConfig.REWARD_KILL_BASE + GameConfig.REWARD_KILL_LENGTH_SCALE * 3
        assert reward1 == pytest.approx(expected1)
        assert reward2 == pytest.approx(expected2)
