"""Opt-in serving-time free-space veto for the Apex vector policy.

The Apex/vector61 champion observes per-action flood-fill free space (the last
three state features, :meth:`SnakeStateMixin._get_free_space_features`) but
nothing forces it to act on them, and at the H5000 deployment profile it dies
of self-collision in most episodes. Scripted snakes already guard against this
with :meth:`ScriptedSnake._apply_free_space_veto`. This module applies the SAME
rule to the Apex policy's greedy choice at serving time, with no training:

1. The policy computes Q-values and the existing hard action mask as usual,
   and picks ``base = argmax(masked Q)``.
2. For each relative direction ``d`` (0=left, 1=straight, 2=right) the bounded
   flood fill gives a normalized reachable-space feature ``f[d]``; with the
   scripted formulas ``cap = min(FREE_SPACE_BFS_CAP, max(FREE_SPACE_MIN_CAP,
   2 * length))`` and ``need = min(logical_length, cap)``, direction ``d`` is
   *spacious* when ``round(f[d] * cap) >= need``.
3. An action ``a`` in 0..5 (``a % 3`` is its direction, ``a >= 3`` boosts) is
   *eligible* when it is allowed by the action mask AND its direction is
   spacious. If no action is eligible the choice is left unchanged; if the
   base action is eligible it is kept; otherwise the veto picks the
   highest-Q eligible action (ties resolve to the lowest action index, the
   same convention as ``torch.argmax``).

One-step approximation for boost moves: a boosted action moves the head two
cells, but the free-space feature is measured from the ONE-step next-head cell
of its direction (exactly what the policy observes and what the scripted veto
uses). The mask already guarantees the two-step boosted path is not an
immediate collision; the veto does not re-flood-fill from the second cell, so
a boost whose second step enters a pocket the first cell does not reveal can
still pass. Boost also burns length, which the length-scaled cap ignores.

Default behavior is untouched: :class:`~src.game.ai_snake.AISnake` only calls
into this module when ``snake.safety_veto`` is set (it is ``None`` unless a
caller installs one with :func:`install_free_space_veto`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from src.game.snake_state import FREE_SPACE_BFS_CAP, FREE_SPACE_MIN_CAP

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.game.snake import Snake

VETO_METHOD = "free-space-veto/v1"
NUM_DIRECTIONS = 3
NUM_ACTIONS = 6

OUTCOME_KEPT = "kept"
OUTCOME_VETOED = "vetoed"
OUTCOME_NO_SPACIOUS = "no_spacious"


def free_space_threshold(length: int, logical_length: int) -> Tuple[int, int]:
    """Return ``(cap, need)`` exactly as the scripted veto computes them.

    ``cap`` mirrors ``ScriptedSnake._free_space_cap`` (itself the BFS cap in
    ``SnakeStateMixin._get_free_space_features``) and ``need`` mirrors
    ``min(self._logical_length(), cap)`` in ``_apply_free_space_veto``.
    """
    cap = min(FREE_SPACE_BFS_CAP, max(FREE_SPACE_MIN_CAP, int(length) * 2))
    need = min(int(logical_length), cap)
    return cap, need


def spacious_directions(features: Sequence[float], cap: int, need: int) -> List[bool]:
    """Per relative direction, whether its reachable-cell count meets ``need``.

    Recovers the raw count from the normalized feature with the scripted
    veto's ``round(feature * cap)``.
    """
    if len(features) != NUM_DIRECTIONS:
        raise ValueError(f"expected {NUM_DIRECTIONS} free-space features, got {len(features)}")
    return [round(float(value) * cap) >= need for value in features]


def veto_choice(
    q_values: Sequence[float],
    action_mask: Sequence[bool],
    spacious: Sequence[bool],
    base_action: int,
) -> Tuple[int, str]:
    """Apply the veto to one greedy decision (pure function).

    Args:
        q_values: Six Q-values (masked or raw; only eligible entries are read).
        action_mask: Six booleans, the policy's existing hard action mask.
        spacious: Three booleans from :func:`spacious_directions`.
        base_action: The policy's own masked-argmax choice.

    Returns:
        ``(action, outcome)`` where ``outcome`` is ``"kept"`` (base already
        eligible), ``"vetoed"`` (replaced by the best eligible action) or
        ``"no_spacious"`` (no masked-legal spacious action; base unchanged).
    """
    if len(q_values) != NUM_ACTIONS or len(action_mask) != NUM_ACTIONS:
        raise ValueError(f"expected {NUM_ACTIONS} Q-values and mask entries")
    if len(spacious) != NUM_DIRECTIONS:
        raise ValueError(f"expected {NUM_DIRECTIONS} spacious flags")
    eligible = [
        bool(action_mask[a]) and bool(spacious[a % NUM_DIRECTIONS]) for a in range(NUM_ACTIONS)
    ]
    if not any(eligible):
        return int(base_action), OUTCOME_NO_SPACIOUS
    if 0 <= int(base_action) < NUM_ACTIONS and eligible[int(base_action)]:
        return int(base_action), OUTCOME_KEPT
    best: Optional[int] = None
    for action in range(NUM_ACTIONS):
        if eligible[action] and (best is None or float(q_values[action]) > float(q_values[best])):
            best = action
    assert best is not None  # any(eligible) above
    return best, OUTCOME_VETOED


@dataclass
class SafetyVetoCounters:
    """Per-episode veto bookkeeping (one hero, one episode).

    ``decisions == kept_base + vetoes_applied + fallback_no_spacious`` always
    holds. ``vetoes_to_boost`` counts vetoes whose replacement action boosts;
    ``vetoed_base_boost`` counts vetoes whose overridden base action boosted.
    """

    decisions: int = 0
    kept_base: int = 0
    vetoes_applied: int = 0
    fallback_no_spacious: int = 0
    vetoes_to_boost: int = 0
    vetoed_base_boost: int = 0

    def record(self, base_action: int, action: int, outcome: str) -> None:
        """Count one decision outcome."""
        self.decisions += 1
        if outcome == OUTCOME_KEPT:
            self.kept_base += 1
        elif outcome == OUTCOME_VETOED:
            self.vetoes_applied += 1
            if action >= NUM_DIRECTIONS:
                self.vetoes_to_boost += 1
            if base_action >= NUM_DIRECTIONS:
                self.vetoed_base_boost += 1
        elif outcome == OUTCOME_NO_SPACIOUS:
            self.fallback_no_spacious += 1
        else:
            raise ValueError(f"unknown veto outcome {outcome!r}")

    def to_dict(self) -> Dict[str, int]:
        """JSON-safe counter mapping."""
        return {key: int(value) for key, value in asdict(self).items()}


class FreeSpaceVeto:
    """Stateful per-snake veto hook called from ``AISnake.update``.

    Free space is recomputed with the snake's own
    ``_get_free_space_features(other_snakes)`` at decision time (the exact call
    the state builder and the scripted veto make), rather than read back from
    the possibly carried-forward state tensor.
    """

    method = VETO_METHOD

    def __init__(self) -> None:
        self.counters = SafetyVetoCounters()

    def reset(self) -> None:
        """Zero the counters (call at episode start)."""
        self.counters = SafetyVetoCounters()

    def apply(
        self,
        snake: "Snake",
        other_snakes: Sequence["Snake"],
        q_values: Any,
        action_mask: Any,
        base_action: int,
    ) -> int:
        """Return the (possibly vetoed) action for one greedy decision.

        Args:
            snake: The deciding snake (supplies length and the flood fill).
            other_snakes: The roster passed to ``update`` (may include ``snake``).
            q_values: Six (masked) Q-values; a tensor or a sequence.
            action_mask: Six booleans; a tensor or a sequence.
            base_action: The policy's own masked-argmax choice.
        """
        features = snake._get_free_space_features(list(other_snakes))
        cap, need = free_space_threshold(snake.length, snake._logical_length())
        spacious = spacious_directions(features, cap, need)
        action, outcome = veto_choice(
            _as_list(q_values), [bool(x) for x in _as_list(action_mask)], spacious, base_action
        )
        self.counters.record(int(base_action), action, outcome)
        return action

    def descriptor(self) -> Dict[str, Any]:
        """Static identity of this veto (no counters)."""
        return {
            "method": VETO_METHOD,
            "free_space_bfs_cap": FREE_SPACE_BFS_CAP,
            "free_space_min_cap": FREE_SPACE_MIN_CAP,
            "boost_approximation": "one-step-direction-feature",
        }

    def record(self) -> Dict[str, Any]:
        """Descriptor plus the current counters, for rollout probes."""
        return {**self.descriptor(), "counters": self.counters.to_dict()}


def _as_list(values: Any) -> List[Any]:
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        values = tolist()
    return list(values)


def install_free_space_veto(snake: Any) -> FreeSpaceVeto:
    """Attach a fresh :class:`FreeSpaceVeto` to an AISnake and return it."""
    if not hasattr(snake, "safety_veto"):
        raise TypeError("safety veto can only be installed on an AISnake")
    veto = FreeSpaceVeto()
    snake.safety_veto = veto
    return veto
