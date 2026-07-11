"""Pull-based behavioral probes for episode telemetry (blueprint §5.3 / §7).

Collects per-episode, per-snake behavioral metrics from a live ``GameState``
without requiring any changes to the loop that drives it: call
``probes.observe(game_state)`` once per frame (after ``game_state.update()``)
and ``probes.finalize_episode()`` when the episode ends. Any loop — headless
training in main.py, tournament_eval, or an Apex actor — can adopt it.

Metrics per snake per episode: kills, death cause, survival frames, final and
peak length, boost-frame fraction (actions >= 3), food eaten, kill-opportunity
count, per-term reward totals (from ``Snake.last_reward_breakdown``), and an
encircle-detector v0 that flags entrapment events on death.

Food eaten is read from ``GameState.frame_ate_food`` (per-frame consumption
map) when the driving GameState exposes it, so it is measured uniformly for
learning AND non-learning snakes (scripted anchors, human play); the AISnake
reward-breakdown food term is only a fallback. ``reward_terms`` and
``kill_opportunity_count`` remain 0 for non-learning snakes — they have no
reward breakdown or observed state vector.

Design notes:
    - Kill-opportunity counting reuses the state's kill-opportunity feature
      (``StateIndices.KILL_OPPORTUNITY``, index 53 of the 58/61-D vector): a
      frame counts as an opportunity when the feature is >= the threshold
      (default 0.5). This was chosen over a head-distance heuristic because it
      is exactly the signal the policy observes, so probe counts line up with
      what the network could have acted on.
    - Encircle detector v0: on each death, take a final-frame occupancy
      snapshot (all snake bodies mapped onto a segment-resolution grid, the
      same discretization as ``SnakeStateMixin._get_free_space_features``) and
      run a bounded BFS from the victim's final head cell to measure reachable
      free space. An ``entrapment_event`` is recorded when reachable free
      space < 2x the victim's length (in cells) AND a single enemy owns >= 60%
      of the occupied blocking cells within radius 8 cells (Chebyshev) of the
      victim's head. The victim's last K=60 head positions are recorded on the
      event for downstream analysis.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.core.game_config import GameConfig, StateIndices
from src.game.game_state import DEATH_CAUSE_BY_COLLISION
from src.game.snake_reward import REWARD_TERM_KEYS

# First boost action index: actions 0-2 are normal speed, 3-5 are boosted.
BOOST_ACTION_START = 3


@dataclass
class _SnakeTrack:
    """Per-snake accumulator for a single episode."""

    frames_seen: int = 0
    acted_frames: int = 0
    boost_frames: int = 0
    kills: int = 0
    food_eaten: int = 0
    kill_opportunity_count: int = 0
    final_length: int = 0
    peak_length: int = 0
    death_cause: Optional[str] = None
    death_frame: Optional[int] = None
    entrapment_event: bool = False
    entrapment: Optional[Dict[str, Any]] = None
    reward_terms: Dict[str, float] = field(
        default_factory=lambda: {key: 0.0 for key in REWARD_TERM_KEYS}
    )
    head_history: Deque[Tuple[float, float]] = field(default_factory=deque)


class BehaviorProbes:
    """Per-episode behavioral probe harness for a GameState loop.

    Usage::

        probes = BehaviorProbes()
        for _ in range(max_frames):
            game_state.update(...)
            probes.observe(game_state)
        records = probes.finalize_episode()  # list[dict], one per snake
        summary = probes.summarize()         # aggregate over all episodes

    Attributes:
        episodes: Flat list of per-snake episode records across all finalized
            episodes (each dict carries its ``episode`` index).
    """

    def __init__(
        self,
        history_frames: int = 60,
        blocker_radius_cells: int = 8,
        kill_opportunity_threshold: float = 0.5,
        entrapment_owner_fraction: float = 0.6,
    ) -> None:
        """Initialize the probe harness.

        Args:
            history_frames: K — how many recent head positions to keep per
                snake for the entrapment record.
            blocker_radius_cells: Radius (in grid cells, Chebyshev) around the
                victim's head within which blocker ownership is measured.
            kill_opportunity_threshold: Minimum kill-opportunity feature value
                for a frame to count as a kill opportunity.
            entrapment_owner_fraction: Minimum fraction of nearby blockers a
                single enemy must own for a death to count as entrapment.
        """
        self.history_frames = history_frames
        self.blocker_radius_cells = blocker_radius_cells
        self.kill_opportunity_threshold = kill_opportunity_threshold
        self.entrapment_owner_fraction = entrapment_owner_fraction
        self.episodes: List[Dict[str, Any]] = []
        self._tracks: Dict[int, _SnakeTrack] = {}
        self._episode_index = 0
        self._frames_observed = 0

    # =========================================================================
    # Per-frame observation
    # =========================================================================

    def observe(self, game_state: Any) -> None:
        """Record one frame of telemetry. Call once per frame after update().

        Args:
            game_state: The live GameState (duck-typed: needs ``frame``,
                ``snakes``, ``frame_collisions``, ``frame_kills``; uses
                ``frame_death_causes`` when present).
        """
        self._frames_observed += 1
        frame = int(getattr(game_state, "frame", self._frames_observed))
        frame_collisions: Dict[int, str] = getattr(game_state, "frame_collisions", {}) or {}
        frame_kills: Dict[int, List[int]] = getattr(game_state, "frame_kills", {}) or {}
        death_causes: Dict[int, str] = getattr(game_state, "frame_death_causes", {}) or {}
        frame_ate_food: Optional[Dict[int, bool]] = getattr(game_state, "frame_ate_food", None)

        for killer_id, victim_ids in frame_kills.items():
            self._track_for(killer_id).kills += len(victim_ids)

        for snake in game_state.snakes:
            track = self._track_for(snake.id)
            died_this_frame = snake.id in frame_collisions
            if not (snake.is_alive or died_this_frame):
                continue

            track.frames_seen += 1
            track.final_length = int(snake.length)
            track.peak_length = max(track.peak_length, int(snake.length))
            track.head_history.append((float(snake.head[0]), float(snake.head[1])))
            while len(track.head_history) > self.history_frames:
                track.head_history.popleft()

            # Food: prefer GameState's per-frame consumption map when exposed
            # — it exists for EVERY snake type, including scripted/human
            # snakes that never set the AISnake reward breakdown. The
            # breakdown food term below is the fallback for driving loops
            # that do not expose the map.
            ate_this_frame: Optional[bool] = None
            if frame_ate_food is not None:
                ate_this_frame = bool(frame_ate_food.get(snake.id, False))
                if ate_this_frame:
                    track.food_eaten += 1

            # AI snakes stamp last_transition_frame when they computed a
            # reward this frame; gate action/reward accounting on it so stale
            # values from earlier frames are never double counted.
            acted = getattr(snake, "last_transition_frame", None) == frame
            if acted:
                track.acted_frames += 1
                action = getattr(snake, "last_action", None)
                if action is not None and int(action) >= BOOST_ACTION_START:
                    track.boost_frames += 1

                breakdown = getattr(snake, "last_reward_breakdown", None)
                if breakdown:
                    for key, value in breakdown.items():
                        track.reward_terms[key] = track.reward_terms.get(key, 0.0) + float(value)
                    if ate_this_frame is None and breakdown.get("food", 0.0) > 0.0:
                        track.food_eaten += 1

                state = getattr(snake, "last_state", None)
                if state is not None and len(state) > StateIndices.KILL_OPPORTUNITY:
                    kill_opp = float(state[StateIndices.KILL_OPPORTUNITY])
                    if kill_opp >= self.kill_opportunity_threshold:
                        track.kill_opportunity_count += 1
            elif not hasattr(snake, "last_transition_frame"):
                # Scripted/human snakes never stamp last_transition_frame but
                # do act on every frame they are observed alive: count the
                # frame (boost_frame_fraction denominator) and their boost
                # flag so those probe columns are real measurements instead
                # of silent zeros. reward_terms and kill_opportunity_count
                # stay 0 for them — they have no per-frame reward breakdown
                # or observed state vector to read.
                track.acted_frames += 1
                if getattr(snake, "is_boosting", False):
                    track.boost_frames += 1

            if died_this_frame and track.death_frame is None:
                cause = death_causes.get(snake.id) or getattr(snake, "last_death_cause", None)
                if cause is None:
                    cause = DEATH_CAUSE_BY_COLLISION.get(frame_collisions[snake.id], "other")
                track.death_cause = cause
                track.death_frame = frame
                self._analyze_entrapment(game_state, snake, track)

    # =========================================================================
    # Episode lifecycle
    # =========================================================================

    def finalize_episode(self) -> List[Dict[str, Any]]:
        """Close the current episode and return its per-snake records.

        Records are also appended to ``self.episodes``. Internal per-episode
        state is reset so the same instance can observe the next episode.

        Returns:
            One dict per tracked snake with the episode's probe metrics.
        """
        records: List[Dict[str, Any]] = []
        for snake_id in sorted(self._tracks):
            track = self._tracks[snake_id]
            boost_fraction = (
                track.boost_frames / track.acted_frames if track.acted_frames > 0 else 0.0
            )
            records.append(
                {
                    "episode": self._episode_index,
                    "snake_id": snake_id,
                    "kills": track.kills,
                    "death_cause": track.death_cause,
                    "death_frame": track.death_frame,
                    "survival_frames": track.frames_seen,
                    "final_length": track.final_length,
                    "peak_length": track.peak_length,
                    "boost_frame_fraction": boost_fraction,
                    "food_eaten": track.food_eaten,
                    "kill_opportunity_count": track.kill_opportunity_count,
                    "reward_terms": dict(track.reward_terms),
                    "entrapment_event": track.entrapment_event,
                    "entrapment": track.entrapment,
                }
            )
        self.episodes.extend(records)
        self._tracks = {}
        self._frames_observed = 0
        self._episode_index += 1
        return records

    def summarize(self) -> Dict[str, Any]:
        """Aggregate all finalized episode records.

        Returns:
            Aggregate metrics: episode/record counts, kill and food totals and
            means, deaths by cause, mean survival/boost-fraction, entrapment
            event count, and per-term reward totals and per-record means.
        """
        records = self.episodes
        n = len(records)
        deaths_by_cause: Dict[str, int] = {}
        reward_terms_total: Dict[str, float] = {key: 0.0 for key in REWARD_TERM_KEYS}
        for record in records:
            cause = record["death_cause"]
            if cause is not None:
                deaths_by_cause[cause] = deaths_by_cause.get(cause, 0) + 1
            for key, value in record["reward_terms"].items():
                reward_terms_total[key] = reward_terms_total.get(key, 0.0) + value

        def _mean(key: str) -> float:
            return sum(float(record[key]) for record in records) / n if n else 0.0

        return {
            "episodes": self._episode_index,
            "snake_episodes": n,
            "total_kills": sum(record["kills"] for record in records),
            "mean_kills": _mean("kills"),
            "deaths_by_cause": deaths_by_cause,
            "mean_survival_frames": _mean("survival_frames"),
            "mean_boost_frame_fraction": _mean("boost_frame_fraction"),
            "total_food_eaten": sum(record["food_eaten"] for record in records),
            "mean_food_eaten": _mean("food_eaten"),
            "mean_kill_opportunity_count": _mean("kill_opportunity_count"),
            "entrapment_events": sum(1 for record in records if record["entrapment_event"]),
            "reward_terms_total": reward_terms_total,
            "reward_terms_mean": {
                key: (value / n if n else 0.0) for key, value in reward_terms_total.items()
            },
        }

    # =========================================================================
    # Internals
    # =========================================================================

    def _track_for(self, snake_id: int) -> _SnakeTrack:
        """Return (creating if needed) the accumulator for a snake id."""
        track = self._tracks.get(snake_id)
        if track is None:
            track = _SnakeTrack()
            self._tracks[snake_id] = track
        return track

    def _analyze_entrapment(self, game_state: Any, victim: Any, track: _SnakeTrack) -> None:
        """Run the encircle-detector v0 for a snake that died this frame.

        Builds a final-frame occupancy snapshot on a segment-resolution grid,
        BFS-floods the victim's reachable free space (bounded, pattern from
        ``SnakeStateMixin._get_free_space_features``), and measures what
        fraction of nearby blocking cells each enemy owns.

        Args:
            game_state: The live GameState at the death frame.
            victim: The snake that died.
            track: The victim's episode accumulator (mutated in place).
        """
        ss = max(int(getattr(victim, "segment_size", 1)), 1)
        width = int(getattr(game_state, "_game_width", 0) or getattr(victim, "game_width", 0))
        height = int(getattr(game_state, "_game_height", 0) or getattr(victim, "game_height", 0))
        if width <= 0 or height <= 0:
            return

        grid_w = max(int(math.ceil(width / ss)), 1)
        grid_h = max(int(math.ceil(height / ss)), 1)

        def to_cell(x: float, y: float) -> Tuple[int, int]:
            return (int(x // ss), int(y // ss))

        circular = GameConfig.ARENA_TYPE == "circular"
        cx = cy = radius_sq = 0.0
        if circular:
            from src.game.game_logic import GameLogic

            cx, cy, radius = GameLogic.get_circular_arena(width, height)
            radius_sq = float(radius) ** 2

        def in_bounds(cell: Tuple[int, int]) -> bool:
            gx, gy = cell
            if gx < 0 or gx >= grid_w or gy < 0 or gy >= grid_h:
                return False
            if circular:
                px, py = gx * ss, gy * ss
                if (px - cx) ** 2 + (py - cy) ** 2 > radius_sq:
                    return False
            return True

        # Occupancy snapshot: cell → owning snake id. Include snakes that died
        # this frame (their bodies are still on the board). The victim's own
        # body is mapped first so shared cells attribute to the enemy on it.
        died_ids = set(getattr(game_state, "frame_collisions", {}) or {})
        owners: Dict[Tuple[int, int], int] = {}
        for sx, sy in victim.segments:
            owners[to_cell(sx, sy)] = victim.id
        for snake in game_state.snakes:
            if snake.id == victim.id or not (snake.is_alive or snake.id in died_ids):
                continue
            for sx, sy in snake.segments:
                owners[to_cell(sx, sy)] = snake.id

        victim_length = max(int(victim.length), 1)
        free_space_threshold = 2 * victim_length

        # Bounded BFS from the cells adjacent to the victim's final head cell
        # (the head cell itself is occupied by the victim's own body). Stop as
        # soon as the free-space threshold is provably met.
        head_cell = to_cell(victim.head[0], victim.head[1])
        cap = free_space_threshold + 1
        seen = set()
        stack = [
            cell
            for cell in (
                (head_cell[0] + 1, head_cell[1]),
                (head_cell[0] - 1, head_cell[1]),
                (head_cell[0], head_cell[1] + 1),
                (head_cell[0], head_cell[1] - 1),
            )
            if cell not in owners and in_bounds(cell)
        ]
        seen.update(stack)
        free_space = 0
        while stack and free_space < cap:
            gx, gy = stack.pop()
            free_space += 1
            for nb in ((gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)):
                if nb not in seen and nb not in owners and in_bounds(nb):
                    seen.add(nb)
                    stack.append(nb)

        # Blocker ownership within the analysis radius of the victim's head.
        radius = self.blocker_radius_cells
        blocker_counts: Dict[int, int] = {}
        blocker_total = 0
        for (gx, gy), owner_id in owners.items():
            if max(abs(gx - head_cell[0]), abs(gy - head_cell[1])) <= radius:
                blocker_total += 1
                blocker_counts[owner_id] = blocker_counts.get(owner_id, 0) + 1

        ownership = (
            {owner_id: count / blocker_total for owner_id, count in blocker_counts.items()}
            if blocker_total > 0
            else {}
        )
        enemy_ownership = {
            owner_id: fraction for owner_id, fraction in ownership.items() if owner_id != victim.id
        }
        dominant_enemy_id: Optional[int] = None
        dominant_fraction = 0.0
        if enemy_ownership:
            dominant_enemy_id = max(enemy_ownership, key=lambda sid: enemy_ownership[sid])
            dominant_fraction = enemy_ownership[dominant_enemy_id]

        entrapped = (
            free_space < free_space_threshold
            and dominant_fraction >= self.entrapment_owner_fraction
        )
        track.entrapment_event = entrapped
        track.entrapment = {
            "free_space_cells": free_space,
            "free_space_threshold": free_space_threshold,
            "victim_length": victim_length,
            "blocker_total": blocker_total,
            "blocker_ownership": ownership,
            "dominant_enemy_id": dominant_enemy_id,
            "dominant_fraction": dominant_fraction,
            "head_history": list(track.head_history),
        }
