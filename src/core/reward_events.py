"""Event-based reward v2 (blueprint §3.2), shared with the future vectorized sim.

PURE module: computes the v2 reward from a small per-step event summary with no
game, torch, numpy, or config dependencies (stdlib/typing only). Both the live
simulator (``SnakeRewardMixin``) and the future vectorized simulator import the
same function, so the two can never drift on reward semantics.

Semantics (all constants are load-bearing; change only with a rewards.version bump):

- Potential-based shaping: ``r_pot = gamma * Phi(s') - Phi(s)`` with
  ``Phi = length / 10`` and ``Phi(death) = 0`` — dying forfeits all accumulated
  potential, so dying rich is intrinsically penalized. (Acknowledged deviation
  from strict PBRS: the forfeiture is deliberate shaping equivalent to a
  mass-at-death penalty.)
- Sparse kill event: ``+0.3 * victim_length`` per victim, UNCLAMPED, paid on the
  kill frame even if the killer dies on that same frame.
- Sparse death event: ``-3.0``.
- Nothing else, and no clamping anywhere: food value flows through the length
  potential, boost economics are mechanical (trail pellets, mechanics v2), and
  hunger is an observed scalar rather than a reward term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple, Union

# Phi(length) = length / PHI_LENGTH_DIVISOR.
PHI_LENGTH_DIVISOR: float = 10.0

# Sparse kill reward per unit of victim (logical) length. Unclamped by design.
KILL_REWARD_PER_VICTIM_LENGTH: float = 0.3

# Sparse death event reward.
DEATH_REWARD: float = -3.0

# Breakdown keys in accumulation order: summing a breakdown's values in this
# (insertion) order reproduces the returned total bit-exactly.
REWARD_V2_TERM_KEYS: Tuple[str, str, str] = ("potential", "death", "kill")


@dataclass(frozen=True)
class RewardEvents:
    """Per-step event summary consumed by :func:`compute_reward_v2`.

    Attributes:
        prev_length: Snake length before this step (the Phi(s) baseline).
        new_length: Snake length after this step (ignored when ``died`` is True,
            because Phi(death) = 0).
        died: Whether the snake died this step.
        gamma: Discount factor used by the learner's TD targets (PBRS requires
            the same gamma the value function is trained with).
        kills: Victim logical lengths for every kill credited to this snake this
            step (empty tuple when there were none).
    """

    prev_length: float
    new_length: float
    died: bool
    gamma: float
    kills: Tuple[float, ...] = ()


def phi(length: float) -> float:
    """Return the length potential ``Phi(length) = length / 10``.

    Args:
        length: Snake length (any real-valued mass measure).

    Returns:
        The potential value of that length.
    """
    return float(length) / PHI_LENGTH_DIVISOR


def compute_reward_v2(
    events: Union[RewardEvents, Mapping[str, Any]],
) -> Tuple[float, Dict[str, float]]:
    """Compute the v2 reward for one step from its event summary.

    The reward is ``gamma * Phi(s') - Phi(s)`` (with ``Phi(death) = 0``), plus
    ``+0.3 * victim_length`` per kill (unclamped, paid even if the killer died
    this same step), plus ``-3.0`` on death. There are no other terms and no
    clamping.

    Args:
        events: A :class:`RewardEvents` instance, or a mapping with keys
            ``prev_length``, ``new_length``, ``died``, ``gamma`` and optionally
            ``kills`` (iterable of victim lengths).

    Returns:
        Tuple of (total reward, per-term breakdown). The breakdown has exactly
        the keys :data:`REWARD_V2_TERM_KEYS` and, summed in insertion order,
        reproduces the total bit-exactly.
    """
    if isinstance(events, Mapping):
        kills_raw = events.get("kills", ()) or ()
        events = RewardEvents(
            prev_length=float(events["prev_length"]),
            new_length=float(events["new_length"]),
            died=bool(events["died"]),
            gamma=float(events["gamma"]),
            kills=tuple(float(victim_length) for victim_length in kills_raw),
        )

    new_potential = 0.0 if events.died else phi(events.new_length)
    potential_term = float(events.gamma) * new_potential - phi(events.prev_length)
    death_term = DEATH_REWARD if events.died else 0.0
    kill_term = 0.0
    for victim_length in events.kills:
        kill_term += KILL_REWARD_PER_VICTIM_LENGTH * float(victim_length)

    # Accumulation order mirrors REWARD_V2_TERM_KEYS (and the position of these
    # keys inside snake_reward.REWARD_TERM_KEYS) so per-term telemetry sums
    # reproduce the total bit-exactly.
    total = potential_term + death_term + kill_term
    breakdown: Dict[str, float] = {
        "potential": potential_term,
        "death": death_term,
        "kill": kill_term,
    }
    return total, breakdown
