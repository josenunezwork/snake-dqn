"""Frozen O2 controller selection over the retained Path B v2 search tree.

The O2 arm is deliberately a composition, not a second planner.  ``D`` is the
unmodified decision emitted by :class:`PathBSearchEngine`; ``D+O`` first takes
that rescue decision and considers a harvest action only when the exact same
completed search tree says that the bare continuation survives.  A caller must
therefore provide harvest ledgers for actions already evaluated by the Path B
tree.  This prevents O from spending a second branch budget or using a
post-hoc O1 root search.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

HORIZON = 500
"""The frozen continuation horizon used by both retained D and O scoring."""

DELTA_MASS_MIN = 2_500
"""Integer form of the registered ``delta_A500 >= 5`` harvest threshold."""


class ControllerError(ValueError):
    """Raised when a candidate would violate the shared-tree contract."""


@dataclass(frozen=True)
class HarvestScore:
    """O1-compatible, integer accounting for one already-evaluated action."""

    action: int
    delta_mass_sum: int
    death_delta_mass_sum: int
    survives: bool

    def eligible(self) -> bool:
        """Return the frozen O1 predicate without a floating-point boundary."""
        return (
            self.survives
            and self.delta_mass_sum >= DELTA_MASS_MIN
            and self.death_delta_mass_sum >= self.delta_mass_sum
        )


@dataclass(frozen=True)
class ControllerChoice:
    """One arm's action and audit-safe explanation at an online opportunity."""

    action: int
    kind: str
    rescue_action: int
    bare_action: int
    bare_survives: bool
    eligible_harvest_actions: tuple[int, ...]
    rejected_harvest_actions: tuple[int, ...]
    evaluated_actions: tuple[int, ...]

    def record(self) -> dict[str, Any]:
        """Return a JSON-safe canonical telemetry object."""
        return asdict(self)


@dataclass(frozen=True)
class OpportunitySchedule:
    """Prospective online search schedule, independent of future outcomes.

    The runner calls :meth:`due` before stepping a live decision.  Nearby-death
    events are supplied by the current simulator transition only; they never
    inspect a future or replayed trajectory.  ``max_opportunities`` bounds the
    number of retained Path B trees per H5000 world.
    """

    time_roots: tuple[int, ...] = (1000, 2000, 3000, 4000)
    event_dedupe: int = 64
    max_event_roots: int = 4

    def __post_init__(self) -> None:
        if (
            self.time_roots != tuple(sorted(set(self.time_roots)))
            or len(self.time_roots) not in (2, 4)
            or any(root < 1 or root >= 5000 for root in self.time_roots)
            or self.event_dedupe < 1
            or self.max_event_roots < 0
        ):
            raise ControllerError(
                "O2 schedule must have four increasing in-horizon roots and valid event bounds"
            )

    def due(
        self,
        decision: int,
        event_roots_used: int,
        *,
        nearby_death: bool = False,
        last_event: int = -64,
    ) -> bool:
        """Return whether an online tree may be evaluated at this decision."""
        if decision < 0 or event_roots_used < 0:
            raise ControllerError("decision and event_roots_used must be nonnegative")
        if decision in self.time_roots:
            return True
        return (
            nearby_death
            and event_roots_used < self.max_event_roots
            and decision - last_event >= self.event_dedupe
        )


def _search_fields(search: Any) -> tuple[str, int, int, Mapping[int, Any]]:
    """Read only the stable retained SearchResult surface, fail closed otherwise."""
    status = str(getattr(search, "status", ""))
    bare_action = getattr(search, "bare_action", None)
    action = getattr(search, "action", None)
    outcomes = getattr(search, "outcomes", None)
    if not isinstance(bare_action, int) or not 0 <= bare_action < 6:
        raise ControllerError("Path B search result has no valid bare action")
    if action is not None and (not isinstance(action, int) or not 0 <= action < 6):
        raise ControllerError("Path B search result has an invalid rescue action")
    if not isinstance(outcomes, Mapping) or bare_action not in outcomes:
        raise ControllerError("Path B search result lacks the bare outcome")
    if any(not isinstance(key, int) or not 0 <= key < 6 for key in outcomes):
        raise ControllerError("Path B search result has invalid evaluated actions")
    return status, bare_action, bare_action if action is None else action, outcomes


def _scores_by_action(
    scores: Sequence[HarvestScore], evaluated: Mapping[int, Any]
) -> dict[int, HarvestScore]:
    result: dict[int, HarvestScore] = {}
    for score in scores:
        if not isinstance(score, HarvestScore):
            raise ControllerError("harvest scores must be HarvestScore objects")
        if score.action not in evaluated:
            raise ControllerError(
                "harvest score action was not evaluated by the retained Path B tree"
            )
        if score.action in result:
            raise ControllerError("duplicate harvest score action")
        result[score.action] = score
    return result


def choose_d(search: Any) -> ControllerChoice:
    """Use Path B v2's rescue-only result without changing its ranking.

    ``IMPROVING`` is the only retained status that may change the bare action;
    all other complete outcomes keep bare B8.  The engine itself fixes the
    chronology, legality, survivor and longest-life ranking.
    """
    status, bare, rescue, outcomes = _search_fields(search)
    applied = status == "IMPROVING" and rescue != bare
    return ControllerChoice(
        action=rescue if applied else bare,
        kind="rescue" if applied else "bare",
        rescue_action=rescue,
        bare_action=bare,
        bare_survives=bool(getattr(outcomes[bare], "survives", False)),
        eligible_harvest_actions=(),
        rejected_harvest_actions=(),
        evaluated_actions=tuple(sorted(outcomes)),
    )


def choose_d_plus_o(search: Any, scores: Sequence[HarvestScore]) -> ControllerChoice:
    """Apply rescue priority, then one qualifying harvest action from that tree.

    The O1 predicate is ``death_delta_mass_sum >= delta_mass_sum >= 2500``.
    Harvest can only be selected where the Path B tree's bare **and alternative**
    H500 continuations survive.  It ranks every surviving action by realised
    mass delta with exact ties to bare and then action index.  The top action
    must itself meet the O1 attribution predicate; a lower qualifying action
    cannot be cherry-picked after a higher-ranked failure.
    """
    d = choose_d(search)
    _status, bare, _rescue, outcomes = _search_fields(search)
    indexed = _scores_by_action(scores, outcomes)
    rejected = tuple(sorted(action for action, value in indexed.items() if not value.eligible()))
    eligible = tuple(sorted(action for action, value in indexed.items() if value.eligible()))
    if d.kind == "rescue":
        return ControllerChoice(
            **{
                **d.record(),
                "eligible_harvest_actions": eligible,
                "rejected_harvest_actions": rejected,
            }
        )
    if not bool(getattr(outcomes[bare], "survives", False)):
        return ControllerChoice(
            **{
                **d.record(),
                "eligible_harvest_actions": eligible,
                "rejected_harvest_actions": rejected,
            }
        )
    surviving = [action for action, value in indexed.items() if value.survives]
    # Bare is the explicit tie winner at delta zero.  Alternatives are only
    # compared within the exact Path B candidate tree.
    winner = min(
        [bare, *surviving],
        key=lambda action: (
            0 if action == bare else -indexed[action].delta_mass_sum,
            action != bare,
            action,
        ),
    )
    if winner == bare or not indexed[winner].eligible():
        return ControllerChoice(
            **{
                **d.record(),
                "eligible_harvest_actions": eligible,
                "rejected_harvest_actions": rejected,
            }
        )
    return ControllerChoice(
        action=winner,
        kind="harvest",
        rescue_action=d.rescue_action,
        bare_action=bare,
        bare_survives=True,
        eligible_harvest_actions=eligible,
        rejected_harvest_actions=rejected,
        evaluated_actions=d.evaluated_actions,
    )


def shared_tree_budget(search: Any, scores: Sequence[HarvestScore]) -> dict[str, int]:
    """Expose an auditable equal-budget counter for D and D+O.

    The only evaluated branches are ``SearchResult.outcomes``.  O scoring is
    metadata over that exact set, hence it always reports zero added branches.
    """
    _status, _bare, _rescue, outcomes = _search_fields(search)
    _scores_by_action(scores, outcomes)
    return {
        "pathb_evaluated_actions": len(outcomes),
        "harvest_scored_actions": len(scores),
        "harvest_added_branches": 0,
        "total_evaluated_actions": len(outcomes),
    }


def choose_shared_tree(
    outcomes: Mapping[str, Mapping[str, Any]],
    *,
    q: Sequence[float],
    bare_action: int,
    scores: Sequence[HarvestScore],
) -> tuple[ControllerChoice, ControllerChoice]:
    """Return ``(D, D+O)`` from one O1-style completed online tree.

    O2's unpaced evaluator calls this after one ``evaluate_root``-equivalent
    all-advisory-legal H500 continuation with provenance attached.  The D arm
    imports Path B's retained full-action ranking verbatim.  The D+O arm uses
    the same rows and only changes a non-rescue bare-survives choice.
    """
    from research.t34_exit_b8.teacher_v2 import _rank_outcomes

    if len(q) != 6 or not 0 <= bare_action < 6:
        raise ControllerError("shared tree needs six Q values and a valid bare action")
    if str(bare_action) not in outcomes:
        raise ControllerError("shared tree lacks its bare outcome")
    ranked = _rank_outcomes(outcomes, q=[float(value) for value in q], bare_action=bare_action)
    tested = {int(action): value for action, value in outcomes.items() if value.get("tested", True)}
    synthetic = type(
        "SharedSearch",
        (),
        {
            "status": "IMPROVING" if int(ranked["action"]) != bare_action else "NO_IMPROVEMENT",
            "bare_action": bare_action,
            "action": int(ranked["action"]),
            "outcomes": {
                action: type("Outcome", (), {"survives": bool(value["survives"])})()
                for action, value in tested.items()
            },
        },
    )()
    d = choose_d(synthetic)
    # Harvest ranking is realised mass delta, exact ties to bare, then action.
    # The O1 ledger already holds integer deltas and provenance attribution.
    do = choose_d_plus_o(synthetic, scores)
    return d, do
