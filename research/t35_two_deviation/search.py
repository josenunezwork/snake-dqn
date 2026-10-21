"""Engine-agnostic two-deviation rescue search (T35).

Semantics (fixed by PREREGISTRATION.md, including Amendment 1):

* A *deviation* at decision ``t`` replaces B8's greedy action by one other
  advisory-legal action; every later hero action is B8 greedy (eps 0, no veto).
* Layer 1 expands every decision of the event window: every non-greedy
  alternative is rolled ``horizon`` steps (capped at the H5000 episode end).
  A layer-1 alternative alive at its horizon is a depth-1 survivor.
* Otherwise a stratified beam of ``beam`` first deviations is chosen
  (:func:`select_beam`): first deviations already proven all-die elsewhere
  (``exclude``) are removed; the window is split into ``beam`` equal index bins
  and the best branch of each bin (by :func:`beam_key`: latest death decision,
  then earlier deviation, higher B8 Q, lower action index) is kept; empty bins
  are filled by the best remaining branches at a not-yet-used ``t1``.  Each beam
  branch is continued greedily to its own death ``d1``; layer 2 expands every
  decision of ``[max(t1 + 1, d1 - window2 + 1), d1]`` on that branch, scanning
  backward from ``d1``.  A layer-2 alternative alive at its horizon is a
  *candidate*; it is a *rescue* only if the same greedy continuation is still
  alive ``confirm_extra`` steps later (cap H5000), which excludes delays.
* A wall-cap expiry (``TimeoutError``) during layer 2 keeps every completed
  branch: an already confirmed rescue stays ``RESCUED_DEPTH2`` (constructive
  proof, ``cap_expired`` set); otherwise the event is ``INCOMPLETE_TIMEOUT``.

The backend owns the simulator.  It must provide::

    expand(state, decision, doom) -> {"greedy": int, "q": [6 floats],
        "outcomes": {str(action): {"survives", "death_decision",
                     "frames_executed", "horizon", ...}},
        "batch": {"actual_env_frames": int}}
    branch(state, decision, action, stop) -> (death_decision | None,
        {decision: prepared_state}, row_steps)
    confirm(state, decision, action, depth) -> {"survives": bool,
        "death_decision": int | None, "env_frames": int}

``expand`` rolls only non-greedy alternatives and reports the greedy action as
the known doom (a free control), exactly like the T34 shadow M2 expansion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Collection, Mapping, Optional, Sequence

EPISODE_END = 5_000


class ControlMismatch(AssertionError):
    """A re-simulation disagreed with the outcome it must reproduce."""


@dataclass(frozen=True)
class SearchParams:
    """Registered search shape (defaults are the registered values)."""

    beam: int = 8
    window2: int = 300
    horizon: int = 300
    confirm_extra: int = 300

    def __post_init__(self) -> None:
        if self.beam < 1 or self.window2 < 1 or self.horizon < 1 or self.confirm_extra < 0:
            raise ValueError(f"invalid search parameters: {self}")


def alternatives_in_order(expansion: Mapping[str, Any]) -> list[int]:
    """Non-greedy expanded actions in descending B8 Q order (lowest index breaks ties)."""
    greedy = int(expansion["greedy"])
    q = expansion["q"]
    actions = [int(a) for a in expansion["outcomes"] if int(a) != greedy]
    return sorted(actions, key=lambda a: (-float(q[a]), a))


def beam_key(candidate: Mapping[str, Any]) -> tuple:
    """Sort key: longest surviving first (latest death), then earlier deviation, Q, index."""
    return (
        -int(candidate["death_decision"]),
        int(candidate["decision"]),
        -float(candidate["q"]),
        int(candidate["action"]),
    )


def select_beam(
    candidates: Sequence[Mapping[str, Any]],
    window1: Sequence[int],
    k: int,
    exclude: Collection[tuple[int, int]] = (),
) -> list[Mapping[str, Any]]:
    """Registered stratified beam: best branch per t1-bin, distinct t1, exclusions removed.

    Bin of decision ``t`` = ``floor(k * (t - window1[0]) / len(window1))``.  Returned in
    ``beam_key`` order (rank 1 = latest death).
    """
    excluded = {(int(t), int(a)) for t, a in exclude}
    eligible = sorted(
        (c for c in candidates if (int(c["decision"]), int(c["action"])) not in excluded),
        key=beam_key,
    )
    first, n = int(window1[0]), len(window1)
    chosen: list[Mapping[str, Any]] = []
    seen_bins: set[int] = set()
    for candidate in eligible:  # best-first, so the first hit per bin is that bin's best
        index = (k * (int(candidate["decision"]) - first)) // n
        if index not in seen_bins:
            seen_bins.add(index)
            chosen.append(candidate)
    chosen = chosen[:k]
    used_t = {int(c["decision"]) for c in chosen}
    for candidate in eligible:  # fill empty bins: distinct t1 first
        if len(chosen) >= k:
            break
        if candidate not in chosen and int(candidate["decision"]) not in used_t:
            chosen.append(candidate)
            used_t.add(int(candidate["decision"]))
    for candidate in eligible:  # then any remaining branch
        if len(chosen) >= k:
            break
        if candidate not in chosen:
            chosen.append(candidate)
    return sorted(chosen, key=beam_key)


def _row_steps(expansion: Mapping[str, Any]) -> int:
    return int(expansion["batch"]["actual_env_frames"])


def _confirmed(
    backend: Any,
    state: Any,
    decision: int,
    action: int,
    params: SearchParams,
    expected_horizon: int,
) -> dict[str, Any]:
    outcome = backend.confirm(state, decision, action, params.horizon + params.confirm_extra)
    death = outcome.get("death_decision")
    if death is not None and int(death) < decision + int(expected_horizon):
        # The confirmation's first `horizon` steps must reproduce the expansion's survival.
        raise ControlMismatch(
            ("confirm died before the expansion horizon", decision, action, death)
        )
    return {
        "survives": bool(outcome["survives"]),
        "death_decision": death,
        "env_frames": int(outcome["env_frames"]),
    }


def _scan_window(
    backend: Any,
    states: Mapping[int, Any],
    decisions: Sequence[int],
    doom: int,
    params: SearchParams,
    stop_on_rescue: bool,
    progress: Optional[Callable[[dict[str, Any]], None]],
    layer: int,
) -> dict[str, Any]:
    """Expand decisions in the given order; confirm horizon survivors in Q order."""
    expansions: dict[int, dict[str, Any]] = {}
    unconfirmed: list[dict[str, Any]] = []
    rescue: Optional[dict[str, Any]] = None
    row_steps = 0
    for decision in decisions:
        if decision not in states:
            raise ControlMismatch(("missing prepared state", layer, decision, doom))
        expansion = backend.expand(states[decision], decision, doom)
        expansions[decision] = expansion
        row_steps += _row_steps(expansion)
        for action in alternatives_in_order(expansion):
            outcome = expansion["outcomes"][str(action)]
            if rescue is not None or not outcome["survives"]:
                continue
            check = _confirmed(
                backend, states[decision], decision, action, params, outcome["horizon"]
            )
            row_steps += check["env_frames"]
            hit = {"decision": decision, "action": action, "confirm": check}
            if check["survives"]:
                rescue = hit
                break
            unconfirmed.append(hit)
        if progress is not None:
            progress({"layer": layer, "decision": decision, "row_steps": row_steps})
        if rescue is not None and stop_on_rescue:
            break
    return {
        "expansions": expansions,
        "rescue": rescue,
        "unconfirmed_survivors": unconfirmed,
        "row_steps": row_steps,
    }


def two_deviation_search(
    backend: Any,
    root_states: Mapping[int, Any],
    window1: Sequence[int],
    doom: int,
    params: SearchParams = SearchParams(),
    progress: Optional[Callable[[dict[str, Any]], None]] = None,
    exclude: Collection[tuple[int, int]] = (),
    after_layer1: Optional[Callable[[dict[str, Any]], list]] = None,
) -> dict[str, Any]:
    """Run layer 1 over ``window1`` then, if nothing survives, the layer-2 beam.

    ``root_states`` maps every window-1 decision to its prepared pre-action state
    on the greedy (doomed) trajectory; ``doom`` is that trajectory's fatal decision.
    ``after_layer1(result)`` may return identity mismatches; any stops the event
    as ``INCOMPLETE_IDENTITY`` before layer 2.
    """
    if not window1 or sorted(set(window1)) != list(window1) or window1[-1] > doom:
        raise ValueError("window1 must be strictly increasing decisions ending at or before doom")
    # Layer 1: the whole window is always expanded (identity control + beam ranking).
    layer1 = _scan_window(
        backend, root_states, list(window1), doom, params, False, progress, layer=1
    )
    survivors1 = [
        {"decision": t, "action": a}
        for t, expansion in sorted(layer1["expansions"].items())
        for a in alternatives_in_order(expansion)
        if expansion["outcomes"][str(a)]["survives"]
    ]
    result: dict[str, Any] = {
        "params": asdict(params),
        "doom": int(doom),
        "window1": [int(window1[0]), int(window1[-1]), len(window1)],
        "excluded_first_deviations": sorted([int(t), int(a)] for t, a in exclude),
        "layer1_alternatives": sum(
            len(alternatives_in_order(e)) for e in layer1["expansions"].values()
        ),
        "layer1_row_steps": layer1["row_steps"],
        "layer1_survivors": survivors1,
        "layer1_outcomes": {
            str(t): {
                "greedy": int(e["greedy"]),
                "outcomes": {
                    a: {
                        k: o.get(k)
                        for k in ("survives", "death_decision", "frames_executed", "horizon")
                    }
                    for a, o in e["outcomes"].items()
                },
            }
            for t, e in sorted(layer1["expansions"].items())
        },
        "cap_expired": False,
    }
    mismatches = after_layer1(result) if after_layer1 is not None else []
    if mismatches:
        result.update(
            status="INCOMPLETE_IDENTITY",
            identity_mismatches=list(mismatches)[:50],
            depth=None,
            rescue=None,
            beam=[],
            row_steps=layer1["row_steps"],
        )
        return result
    if survivors1:
        rescue = layer1["rescue"]
        result.update(
            status="RESCUED_DEPTH1" if rescue is not None else "SURVIVOR_DEPTH1_UNCONFIRMED",
            depth=1 if rescue is not None else None,
            rescue=rescue,
            unconfirmed_survivors=layer1["unconfirmed_survivors"],
            beam=[],
            row_steps=layer1["row_steps"],
        )
        return result
    candidates = [
        {
            "decision": t,
            "action": a,
            "q": float(expansion["q"][a]),
            "death_decision": int(expansion["outcomes"][str(a)]["death_decision"]),
        }
        for t, expansion in layer1["expansions"].items()
        for a in alternatives_in_order(expansion)
    ]
    beam = select_beam(candidates, list(window1), params.beam, exclude)
    total = layer1["row_steps"]
    branches = []
    try:
        for rank, candidate in enumerate(beam, start=1):
            t1, a1 = candidate["decision"], candidate["action"]
            expected = candidate["death_decision"]
            stop = min(EPISODE_END, t1 + params.horizon)
            d1, states, branch_steps = backend.branch(root_states[t1], t1, a1, stop)
            if d1 != expected:
                raise ControlMismatch(("branch death differs from layer 1", t1, a1, d1))
            total += int(branch_steps)
            first = max(t1 + 1, d1 - params.window2 + 1)
            decisions = list(range(d1, first - 1, -1))  # backward from the branch's death
            scan = (
                _scan_window(backend, states, decisions, d1, params, True, progress, layer=2)
                if decisions
                else {"expansions": {}, "rescue": None, "unconfirmed_survivors": [], "row_steps": 0}
            )
            total += scan["row_steps"]
            branches.append(
                {
                    "rank": rank,
                    **candidate,
                    "window2": [first, d1, len(decisions)],
                    "expanded_decisions": len(scan["expansions"]),
                    "latest_layer2_death": max(
                        (
                            int(o["death_decision"])
                            for e in scan["expansions"].values()
                            for a, o in e["outcomes"].items()
                            if int(a) != int(e["greedy"]) and o.get("death_decision") is not None
                        ),
                        default=None,
                    ),
                    "rescue": scan["rescue"],
                    "unconfirmed_survivors": scan["unconfirmed_survivors"],
                    "branch_row_steps": int(branch_steps),
                    "layer2_row_steps": scan["row_steps"],
                }
            )
    except TimeoutError:
        result["cap_expired"] = True
    rescued = [b for b in branches if b["rescue"] is not None]
    candidates_300 = [
        {"rank": b["rank"], **hit}
        for b in branches
        for hit in ([b["rescue"]] if b["rescue"] else []) + b["unconfirmed_survivors"]
    ]
    if rescued:
        status = "RESCUED_DEPTH2"
    elif result["cap_expired"]:
        status = "INCOMPLETE_TIMEOUT"
    else:
        status = "NOT_RESCUED"
    result.update(
        status=status,
        depth=2 if rescued else None,
        rescue=(
            None
            if not rescued
            else {
                "rank": rescued[0]["rank"],
                "t1": rescued[0]["decision"],
                "a1": rescued[0]["action"],
                **rescued[0]["rescue"],
            }
        ),
        rescued_ranks=[b["rank"] for b in rescued],
        completed_branches=len(branches),
        planned_beam=[
            {k: c[k] for k in ("decision", "action", "death_decision", "q")} for c in beam
        ],
        layer2_candidates_300=candidates_300,
        unconfirmed_survivors=[s for b in branches for s in b["unconfirmed_survivors"]],
        beam=branches,
        row_steps=total,
    )
    return result
