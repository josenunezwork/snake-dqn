#!/usr/bin/env python3
"""OpenSkill (Plackett-Luce) pool ratings from round-robin FFA episodes.

Blueprint §5.3: rate a pool of checkpoints (plus optional scripted anchors) by
running seeded free-for-all episodes and converting each episode's death order
into one Plackett-Luce rating update over all participants. Ratings persist in
a JSON file between runs so the tool can run nightly and stay comparable over
time (keep 1-2 permanent frozen anchors in the pool for cross-month anchoring).

Ranking rule per episode: survivors rank first (later death = better), ties on
death frame (including the "both survived to the cap" case) break on final
mass, and exact ties share a rank.

Checkpoints are driven by the forward-only :class:`InferenceAgent` (no
optimizer / replay / target network). Scripted anchors are resolved through
``SnakeFactory.create_scripted_snake`` and named on the command line as e.g.
``anchor:greedy_food`` / ``anchor:random_safe``.

Usage:
  SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/rate_pool.py saved_snakes \
    --anchor anchor:greedy_food --anchor anchor:random_safe \
    --episodes 40 --per-episode 6 --frames 3000 --seed 0 \
    --ratings logs/ratings.json --markdown logs/ratings.md
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

ANCHOR_PREFIX = "anchor:"
DEFAULT_MU = 25.0
DEFAULT_SIGMA = 25.0 / 3.0


# =============================================================================
# Pure rating math (unit-testable without the game or checkpoints)
# =============================================================================
def ranks_from_episode(
    death_frames: Sequence[Optional[int]], final_masses: Sequence[float]
) -> List[int]:
    """Convert an FFA episode outcome into OpenSkill ranks (lower = better).

    Survivors (``death_frame is None``) rank ahead of every dead snake; among
    the dead, a later death frame is better. Ties on death frame — including
    multiple survivors at the frame cap — break on final mass (bigger is
    better). Exact ties on both keys share the same rank (competition style).

    Args:
        death_frames: Per-participant death frame, ``None`` for survivors.
        final_masses: Per-participant mass at death (or at episode end).

    Returns:
        A rank per participant, aligned with the input order, 0 = winner.
    """
    if len(death_frames) != len(final_masses):
        raise ValueError("death_frames and final_masses must have equal length")
    keys: List[Tuple[float, float]] = []
    for frame, mass in zip(death_frames, final_masses):
        frame_key = math.inf if frame is None else float(frame)
        keys.append((-frame_key, -float(mass)))  # smaller tuple = better placement
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    ranks = [0] * len(keys)
    rank = 0
    for pos, idx in enumerate(order):
        if pos > 0 and keys[idx] != keys[order[pos - 1]]:
            rank = pos
        ranks[idx] = rank
    return ranks


def update_ratings(
    ratings: Dict[str, Dict[str, Any]],
    participant_ids: Sequence[str],
    ranks: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    """Apply one episode's ranks as a Plackett-Luce update, in place.

    Unknown participants start at the OpenSkill default (mu=25, sigma=25/3).

    Args:
        ratings: Mutable ratings store: id -> {mu, sigma, ordinal, games, wins}.
        participant_ids: Episode participants, aligned with ``ranks``.
        ranks: Episode ranks from :func:`ranks_from_episode` (lower = better).

    Returns:
        The same ``ratings`` dict, updated.
    """
    from openskill.models import PlackettLuce

    model = PlackettLuce()
    teams = []
    for pid in participant_ids:
        entry = ratings.get(pid)
        mu = float(entry["mu"]) if entry else DEFAULT_MU
        sigma = float(entry["sigma"]) if entry else DEFAULT_SIGMA
        teams.append([model.rating(mu=mu, sigma=sigma, name=pid)])
    rated = model.rate(teams, ranks=list(ranks))

    best_rank = min(ranks)
    for pid, team, rank in zip(participant_ids, rated, ranks):
        rating = team[0]
        prev = ratings.get(pid, {})
        ratings[pid] = {
            "mu": float(rating.mu),
            "sigma": float(rating.sigma),
            "ordinal": float(rating.mu - 3.0 * rating.sigma),
            "games": int(prev.get("games", 0)) + 1,
            "wins": int(prev.get("wins", 0)) + (1 if rank == best_rank else 0),
        }
    return ratings


# =============================================================================
# Persistence
# =============================================================================
def load_ratings(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load a ratings store written by :func:`save_ratings` (empty if absent)."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    ratings = data.get("ratings", data)
    if not isinstance(ratings, dict):
        raise ValueError(f"Unrecognized ratings file format: {path}")
    return {str(k): dict(v) for k, v in ratings.items()}


def save_ratings(
    path: str | Path,
    ratings: Dict[str, Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Write the ratings store as JSON (model + timestamp + per-id entries)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "PlackettLuce",
        "updated": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
        "ratings": ratings,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def ratings_markdown(ratings: Dict[str, Dict[str, Any]]) -> str:
    """Render the ratings store as a markdown table sorted by ordinal (mu - 3*sigma)."""
    rows = sorted(ratings.items(), key=lambda kv: kv[1].get("ordinal", -math.inf), reverse=True)
    lines = [
        "| rank | participant | ordinal | mu | sigma | games | wins |",
        "|-----:|:------------|--------:|-----:|------:|------:|-----:|",
    ]
    for i, (pid, entry) in enumerate(rows, 1):
        lines.append(
            f"| {i} | {pid} | {entry.get('ordinal', 0.0):.2f} | {entry.get('mu', 0.0):.2f} "
            f"| {entry.get('sigma', 0.0):.2f} | {entry.get('games', 0)} "
            f"| {entry.get('wins', 0)} |"
        )
    return "\n".join(lines)


# =============================================================================
# Participants
# =============================================================================
@dataclass(frozen=True)
class Participant:
    """One rated pool member: a checkpoint file or a scripted anchor."""

    participant_id: str
    kind: str  # "checkpoint" | "anchor"
    source: str  # checkpoint path, or anchor kind (e.g. "greedy_food")


def discover_participants(pool_dir: str, anchors: Sequence[str]) -> List[Participant]:
    """Build the participant pool from a checkpoint directory + anchor ids.

    Args:
        pool_dir: Directory scanned (non-recursively) for ``*.pth`` checkpoints.
        anchors: Anchor ids of the form ``anchor:<kind>``.

    Returns:
        Checkpoint participants (id = file name) followed by anchor participants.
    """
    participants: List[Participant] = []
    pool = Path(pool_dir)
    if not pool.is_dir():
        raise FileNotFoundError(f"Checkpoint pool directory not found: {pool_dir}")
    for path in sorted(pool.glob("*.pth")):
        participants.append(Participant(path.name, "checkpoint", str(path)))
    for anchor in anchors:
        if not anchor.startswith(ANCHOR_PREFIX):
            raise ValueError(f"Anchor id must start with '{ANCHOR_PREFIX}': {anchor}")
        participants.append(Participant(anchor, "anchor", anchor[len(ANCHOR_PREFIX) :]))
    return participants


class _InferencePolicyAdapter:
    """Duck-typed read-only policy over an :class:`InferenceAgent`.

    ``AISnake.update`` drives whatever exposes ``.dqn`` (masked greedy argmax);
    ``training=False`` short-circuits every replay/learning path, and
    ``select_action`` covers the fallback branch.
    """

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.dqn = agent.network
        self.device = agent.device
        self.epsilon = 0.0
        self.training = False

    def select_action(self, state: Any, action_mask: Any = None) -> int:
        return int(self.agent.act(state, action_mask=action_mask))


def _set_seed(seed: int) -> None:
    """Seed random / numpy / torch for a reproducible rollout."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# Episode runner — the ONLY place that touches the game loop. Adapt here when
# eval-loop details (engine, respawn rules, metrics) change.
# =============================================================================
def run_ffa_episode(
    entries: Sequence[Tuple[Participant, Optional[Any]]],
    frames: int,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Run one seeded free-for-all episode and return per-participant outcomes.

    Deaths are terminal (no respawn); the episode ends early once at most one
    snake is alive. Final mass is the last mass observed while alive, so
    simultaneous deaths tie-break on size at death.

    Args:
        entries: ``(participant, inference_agent_or_None)`` per seat; anchors
            carry ``None`` and are built via ``SnakeFactory.create_scripted_snake``.
        frames: Frame cap for the episode.
        seed: RNG seed (also seeds scripted anchors, offset by seat index).

    Returns:
        Mapping participant_id -> {death_frame (None if survived), final_mass, kills}.
    """
    _set_seed(seed)

    from src.game.game_state import GameState
    from src.game.game_state_factory import configure_eval_game_state

    num = len(entries)
    game_state = GameState(headless=True, snake_policies=["apex"] * num, num_snakes=num)

    for i, (participant, agent) in enumerate(entries):
        seat = game_state.snakes[i]
        if participant.kind == "anchor":
            # Lazy import: the scripted factory may land after this module.
            from src.game.snake_factory import SnakeFactory

            game_state.snakes[i] = SnakeFactory.create_scripted_snake(
                snake_id=seat.id,
                color=seat.color,
                start_pos=tuple(seat.segments[0]),
                kind=participant.source,
                game_width=game_state._game_width,
                game_height=game_state._game_height,
                seed=seed + i,
            )
        else:
            seat.policy = _InferencePolicyAdapter(agent)
    game_state._shared_policy = None  # never train
    configure_eval_game_state(game_state)

    snakes = game_state.snakes
    index_by_snake_id = {snakes[i].id: i for i in range(num)}
    death_frame: List[Optional[int]] = [None] * num
    final_mass = [float(len(s.segments)) for s in snakes]
    kills = [0] * num
    prev_alive = [bool(s.is_alive) for s in snakes]

    for frame in range(1, frames + 1):
        game_state.update(train_mode=True, learn=False)
        for i, snake in enumerate(snakes):
            if snake.is_alive:
                final_mass[i] = float(len(snake.segments))
            elif prev_alive[i]:
                death_frame[i] = frame
            prev_alive[i] = bool(snake.is_alive)
        for killer_id, victims in game_state.frame_kills.items():
            if killer_id in index_by_snake_id:
                kills[index_by_snake_id[killer_id]] += len(victims)
        if sum(prev_alive) <= 1:
            break

    game_state.full_cleanup()
    return {
        entries[i][0].participant_id: {
            "death_frame": death_frame[i],
            "final_mass": final_mass[i],
            "kills": kills[i],
        }
        for i in range(num)
    }


# =============================================================================
# CLI
# =============================================================================
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pool_dir", help="Directory of *.pth checkpoints to rate")
    parser.add_argument(
        "--anchor",
        action="append",
        default=[],
        dest="anchors",
        help="Scripted anchor id, e.g. anchor:greedy_food (repeatable)",
    )
    parser.add_argument("--config", default="configs/eval_free_space.yaml")
    parser.add_argument("--episodes", type=int, default=20, help="FFA episodes to run")
    parser.add_argument(
        "--per-episode", type=int, default=6, help="Participants sampled per episode"
    )
    parser.add_argument("--frames", type=int, default=3000, help="Frame cap per episode")
    parser.add_argument("--seed", type=int, default=0, help="Base seed (episode e uses seed+e)")
    parser.add_argument("--ratings", default="logs/ratings.json", help="Persistent ratings JSON")
    parser.add_argument("--markdown", default=None, help="Optional markdown table output path")
    args = parser.parse_args(argv)

    # Lazy imports so `import rate_pool` never drags in torch/the game (and so
    # the module loads even while SnakeFactory.create_scripted_snake is landing).
    from src.core.config_loader import load_and_initialize_config
    from src.core.game_config import GameConfig
    from src.model.inference_agent import InferenceAgent

    load_and_initialize_config(args.config)

    participants = discover_participants(args.pool_dir, args.anchors)
    if len(participants) < 2:
        print("Need at least 2 participants (checkpoints + anchors).", file=sys.stderr)
        return 2
    if args.anchors:
        from src.game.snake_factory import SnakeFactory

        if not hasattr(SnakeFactory, "create_scripted_snake"):
            print(
                "SnakeFactory.create_scripted_snake is not available yet; "
                "drop --anchor or update the factory.",
                file=sys.stderr,
            )
            return 2

    agents: Dict[str, Any] = {}
    for participant in participants:
        if participant.kind != "checkpoint":
            continue
        agent = InferenceAgent.from_checkpoint(participant.source)
        if agent.input_size != GameConfig.INPUT_SIZE:
            print(
                f"Checkpoint {participant.participant_id} has input_size={agent.input_size} "
                f"but the active config builds {GameConfig.INPUT_SIZE}-D states "
                f"({args.config}); pick a matching --config.",
                file=sys.stderr,
            )
            return 2
        agents[participant.participant_id] = agent

    ratings = load_ratings(args.ratings)
    per_episode = min(args.per_episode, len(participants))
    rng = random.Random(args.seed)

    print(
        f"Pool: {len(participants)} participants | episodes={args.episodes} "
        f"per_episode={per_episode} frames={args.frames} seed={args.seed}\n"
    )
    for episode in range(args.episodes):
        sampled = rng.sample(participants, per_episode)
        entries = [(p, agents.get(p.participant_id)) for p in sampled]
        outcomes = run_ffa_episode(entries, frames=args.frames, seed=args.seed + episode)

        ids = [p.participant_id for p in sampled]
        ranks = ranks_from_episode(
            [outcomes[pid]["death_frame"] for pid in ids],
            [outcomes[pid]["final_mass"] for pid in ids],
        )
        update_ratings(ratings, ids, ranks)
        winner = ids[ranks.index(min(ranks))]
        print(f"  episode {episode + 1:>3}/{args.episodes}: winner={winner}")

    save_ratings(
        args.ratings,
        ratings,
        meta={
            "config": args.config,
            "pool_dir": args.pool_dir,
            "anchors": list(args.anchors),
            "episodes": args.episodes,
            "frames": args.frames,
            "seed": args.seed,
        },
    )
    table = ratings_markdown(ratings)
    print("\n" + table)
    print(f"\nWrote {args.ratings}")
    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(table + "\n")
        print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
