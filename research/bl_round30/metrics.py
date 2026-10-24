"""Pure, conservative reducers for the Round-30 conversion audit.

This module intentionally consumes exported original label rows.  It does not
read F4 data, simulate branches, or alter the registered residual objective.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

HORIZON = 300
EXACT = "EXACT"
UNKNOWN = "UNKNOWN"


def _six(row: Mapping[str, Any], name: str) -> Sequence[Any]:
    value = row.get(name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 6:
        raise ValueError(f"{name} must be a six-action sequence")
    return value


def _parent(row: Mapping[str, Any]) -> Any:
    if "parent_world" in row:
        return row["parent_world"]
    if "parent_id" in row:
        return row["parent_id"]
    raise ValueError("row needs parent_world or parent_id")


def _life(row: Mapping[str, Any]) -> Any:
    if "life_id" not in row:
        raise ValueError("row needs life_id")
    return row["life_id"]


def _exact(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.upper() == EXACT)


def _action(row: Mapping[str, Any], key: str) -> int:
    action = row.get(key)
    if not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < 6:
        raise ValueError(f"{key} must be an action in [0, 5]")
    return action


def selectable_actions(row: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the served advisory mask, including normal-only fallback."""
    advisory = tuple(bool(v) for v in _six(row, "action_mask"))
    if not any(advisory):
        advisory = (True, True, True, False, False, False)
    return tuple(i for i, enabled in enumerate(advisory) if enabled)


def served_selection(row: Mapping[str, Any], q_key: str = "q") -> int:
    """Apply B8's mask/fallback and lowest-index equal-Q tie rule."""
    q = _six(row, q_key)
    values = [float(v) for v in q]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{q_key} must be finite")
    selectable = selectable_actions(row)
    return max(selectable, key=lambda action: (values[action], -action))


def complete_h300(row: Mapping[str, Any]) -> bool:
    """Whether every physical, served-selectable branch is exact and finite."""
    if int(row.get("horizon", HORIZON)) != HORIZON:
        return False
    legal, status, mass = (_six(row, name) for name in ("legal", "status", "mass"))
    for action in selectable_actions(row):
        if bool(legal[action]) and (
            not _exact(status[action])
            or not math.isfinite(float(mass[action]))
            or float(mass[action]) < 0
        ):
            return False
    return True


def row_outcomes(row: Mapping[str, Any]) -> dict[str, Any]:
    """Expose exact selectable survival alternatives without treating unknown as death."""
    legal, status, died, mass = (_six(row, name) for name in ("legal", "status", "died", "mass"))
    exact = tuple(
        action
        for action in selectable_actions(row)
        if (
            bool(legal[action])
            and _exact(status[action])
            and math.isfinite(float(mass[action]))
            and float(mass[action]) >= 0
        )
    )
    survivors = tuple(action for action in exact if not bool(died[action]))
    nonsurvivors = tuple(action for action in exact if bool(died[action]))
    return {"exact": exact, "survivors": survivors, "nonsurvivors": nonsurvivors}


def survivor_reference_regret(row: Mapping[str, Any], selected: int) -> float | None:
    """Return survivor-only regret; unknown/non-survivor selections are not scored."""
    outcomes = row_outcomes(row)
    if not complete_h300(row) or not outcomes["survivors"]:
        return None
    status, died, mass = (_six(row, name) for name in ("status", "died", "mass"))
    if (
        selected not in outcomes["survivors"]
        or not _exact(status[selected])
        or bool(died[selected])
    ):
        return None
    best = max(float(mass[action]) for action in outcomes["survivors"])
    return max(0.0, best - float(mass[selected]))


def summarize_training_rows(rows: Iterable[Mapping[str, Any]], q_key: str = "q") -> dict[str, Any]:
    """Report action choices and exact survivor/non-survivor margins by life/parent."""
    grouped: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in rows:
        selected = served_selection(row, q_key)
        outcomes = row_outcomes(row)
        q = [float(value) for value in _six(row, q_key)]
        competitors = tuple(
            action for action in selectable_actions(row) if action not in outcomes["survivors"]
        )
        margin = (
            max(q[action] for action in outcomes["survivors"])
            - max(q[action] for action in competitors)
            if outcomes["survivors"] and competitors
            else None
        )
        status = _six(row, "status")
        key = (_parent(row), _life(row))
        value = grouped.setdefault(key, {"parent_world": key[0], "life_id": key[1], "rows": []})
        value["rows"].append(
            {
                "selected": selected,
                "selected_status": status[selected],
                "complete_h300": complete_h300(row),
                "margin": margin,
                "survivors": list(outcomes["survivors"]),
                "diers": list(outcomes["nonsurvivors"]),
                "unknown": [
                    action for action in selectable_actions(row) if not _exact(status[action])
                ],
            }
        )
    return {"by_parent_life": list(grouped.values())}


def _parent_life_mean(values: list[tuple[Any, Any, float]]) -> float | None:
    by_parent: dict[Any, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    for parent, life, value in values:
        by_parent[parent][life].append(value)
    if not by_parent:
        return None
    parents = []
    for lives in by_parent.values():
        parents.append(sum(sum(v) / len(v) for v in lives.values()) / len(lives))
    return sum(parents) / len(parents)


def _life_macro_mean(values: list[tuple[Any, Any, float]]) -> float | None:
    """Average row means equally across lives (the R30 retention estimand)."""
    by_life: dict[Any, list[float]] = defaultdict(list)
    for _, life, value in values:
        by_life[life].append(value)
    if not by_life:
        return None
    return sum(sum(group) / len(group) for group in by_life.values()) / len(by_life)


def validate_input_rows(
    rows: Iterable[Mapping[str, Any]], *, expected_namespace: str | None = None
) -> list[str]:
    """Return identity/lineage errors that must take INVALID precedence.

    A caller may omit a namespace only for a synthetic fixture.  Production
    callers bind one expected namespace; duplicate immutable row IDs and a
    rejected lineage are never repaired by dropping those rows.
    """
    errors: list[str] = []
    seen: set[Any] = set()
    for index, row in enumerate(rows):
        try:
            _parent(row)
            _life(row)
            for name in ("action_mask", "legal", "status", "died", "mass", "q"):
                _six(row, name)
            served_selection(row)
            legal, status, mass = (_six(row, name) for name in ("legal", "status", "mass"))
            for action in selectable_actions(row):
                if (
                    bool(legal[action])
                    and _exact(status[action])
                    and (not math.isfinite(float(mass[action])) or float(mass[action]) < 0)
                ):
                    errors.append(f"invalid exact mass at row {index}, action {action}")
            if expected_namespace is not None and row.get("namespace") != expected_namespace:
                errors.append(f"foreign namespace at row {index}")
            if row.get("lineage_valid") is False or row.get("admitted") is False:
                errors.append(f"invalid lineage at row {index}")
            if "row_id" in row:
                row_id = row["row_id"]
                if row_id in seen:
                    errors.append(f"duplicate row_id {row_id!r}")
                seen.add(row_id)
            if "sampling_prob" in row and (
                not math.isfinite(float(row["sampling_prob"]))
                or not 0 < float(row["sampling_prob"]) <= 1
            ):
                errors.append(f"invalid sampling_prob at row {index}")
            if not complete_h300(row):
                # Incomplete values are scientific support failures, not invalid identity.
                continue
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(f"malformed row {index}: {exc}")
            continue
    return errors


def conversion_build_trigger(
    rows: Iterable[Mapping[str, Any]],
    control_key: str = "control_action",
    base_key: str = "base_action",
    expected_namespace: str | None = None,
) -> dict[str, Any]:
    """Apply the immutable R30-2 training trigger with parent/life aggregation."""
    row_items = list(rows)
    invalid_reasons = validate_input_rows(row_items, expected_namespace=expected_namespace)
    choice_loss: list[tuple[Any, Any, float]] = []
    advantage: list[tuple[Any, Any, float]] = []
    nonretention: list[tuple[Any, Any, float]] = []
    parents: set[Any] = set()
    for row in row_items:
        if bool(row.get("ordinary", False)) or not complete_h300(row):
            continue
        out = row_outcomes(row)
        base = _action(row, base_key)
        if len(out["survivors"]) < 2 or base not in out["nonsurvivors"]:
            continue
        parent, life = _parent(row), _life(row)
        parents.add(parent)
        control = _action(row, control_key)
        mass = _six(row, "mass")
        best = max(float(mass[a]) for a in out["survivors"])
        retained = control in out["survivors"]
        loss = max(0.0, best - float(mass[control])) if retained else 0.0
        choice_loss.append((parent, life, loss))
        advantage.append((parent, life, max(0.0, best - float(mass[base]))))
        nonretention.append((parent, life, float(not retained)))
    numerator, denominator = _parent_life_mean(choice_loss), _parent_life_mean(advantage)
    insufficient = len(parents) < 32 or numerator is None or denominator is None or denominator <= 0
    ratio = None if insufficient else numerator / denominator
    return {
        "eligible_parents": len(parents),
        "choice_loss": numerator,
        "available_advantage": denominator,
        "ratio": ratio,
        "control_nonretention": _parent_life_mean(nonretention),
        "invalid_reasons": invalid_reasons,
        "label": (
            "INVALID"
            if invalid_reasons
            else ("INSUFFICIENT" if insufficient else ("GO" if ratio >= 0.10 else "HOLD"))
        ),
    }


def archive_harm_metrics(
    rows: Iterable[Mapping[str, Any]], selected_key: str, base_key: str = "base_action"
) -> dict[str, Any]:
    """Report all R30 archive sensitivities; absent coverage is explicitly incomplete."""
    items = list(rows)
    has_declared_nonrescue = any(bool(row.get("declared_nonrescue", False)) for row in items)
    natural_ordinary_ok = all(
        "natural_ordinary" not in row or bool(row["natural_ordinary"]) for row in items
    )
    if not items or not has_declared_nonrescue or not natural_ordinary_ok:
        return {
            "coverage": False,
            "harm_count": None,
            "unclipped_design_rate": None,
            "clipped_design_rate": None,
            "ess": None,
            "override_frequency": None,
        }
    harms: list[float] = []
    weights: list[float] = []
    changed: list[float] = []
    for row in items:
        selected, base = _action(row, selected_key), _action(row, base_key)
        status, died = _six(row, "status"), _six(row, "died")
        unknown = not _exact(status[selected]) or not _exact(status[base])
        harm = selected != base and (unknown or (not bool(died[base]) and bool(died[selected])))
        if (
            "sampling_prob" not in row
            or not math.isfinite(float(row["sampling_prob"]))
            or not 0 < float(row["sampling_prob"]) <= 1
        ):
            raise ValueError("natural archive rows require sampling_prob in (0, 1]")
        harms.append(float(harm))
        weights.append(1.0 / float(row["sampling_prob"]))
        changed.append(float(selected != base))
    total = sum(weights)
    unclipped = sum(h * w for h, w in zip(harms, weights)) / total
    clipped_weights = [min(5.0, weight) for weight in weights]
    clipped = sum(h * w for h, w in zip(harms, clipped_weights)) / sum(clipped_weights)
    return {
        "coverage": True,
        "harm_count": int(sum(harms)),
        "unclipped_design_rate": unclipped,
        "clipped_design_rate": clipped,
        "ess": total * total / sum(w * w for w in weights),
        "override_frequency": sum(changed) / len(changed),
        "row_count": len(items),
        "harm_row_ids": [
            row.get("row_id", (_parent(row), _life(row))) for row, harm in zip(items, harms) if harm
        ],
    }


def heldout_reducer(
    rows: Iterable[Mapping[str, Any]],
    successor_key: str = "successor_action",
    control_key: str = "control_action",
    base_key: str = "base_action",
    natural_rows: Iterable[Mapping[str, Any]] | None = None,
    expected_namespace: str | None = None,
) -> dict[str, Any]:
    """Single-look R30-2 reducer retaining failures and unknown selections."""
    held_rows = list(rows)
    archive_items = list(natural_rows or [])
    invalid_reasons = validate_input_rows(
        [*held_rows, *archive_items], expected_namespace=expected_namespace
    )
    regrets: dict[str, list[tuple[Any, Any, float]]] = {"successor": [], "control": []}
    retention: dict[str, list[tuple[Any, Any, float]]] = {"successor": [], "control": []}
    failed = {"successor": 0, "control": 0}
    unknown = {"successor": 0, "control": 0}
    complete = decisive = 0
    for row in held_rows:
        if bool(row.get("ordinary", False)) or not complete_h300(row):
            continue
        out = row_outcomes(row)
        base = _action(row, base_key)
        if base not in out["nonsurvivors"] or not out["survivors"]:
            continue
        complete += 1
        parent, life = _parent(row), _life(row)
        mass = _six(row, "mass")
        best = max(float(mass[action]) for action in out["survivors"])
        for name, key in (("successor", successor_key), ("control", control_key)):
            action = _action(row, key)
            status = _six(row, "status")[action]
            legal = _six(row, "legal")
            if (
                action not in selectable_actions(row)
                or not bool(legal[action])
                or not _exact(status)
            ):
                unknown[name] += 1
                retention[name].append((parent, life, 0.0))
                continue
            # Complete rows make every physical selectable branch exact and finite.
            regrets[name].append((parent, life, max(0.0, best - float(mass[action]))))
            if action not in out["survivors"]:
                failed[name] += 1
            retention[name].append((parent, life, float(action in out["survivors"])))
        decisive += 1
    control_regret = _parent_life_mean(regrets["control"])
    successor_regret = _parent_life_mean(regrets["successor"])
    control_retention = _life_macro_mean(retention["control"])
    successor_retention = _life_macro_mean(retention["successor"])
    reduction = (
        None
        if not control_regret
        else (control_regret - (successor_regret or 0.0)) / control_regret
    )
    try:
        archive = archive_harm_metrics(archive_items, successor_key, base_key)
        control_archive = archive_harm_metrics(archive_items, control_key, base_key)
    except ValueError as exc:
        invalid_reasons.append(f"invalid archive: {exc}")
        archive = archive_harm_metrics([], successor_key, base_key)
        control_archive = archive_harm_metrics([], control_key, base_key)
    successor_harm_ids = set(archive.get("harm_row_ids", []))
    control_harm_ids = set(control_archive.get("harm_row_ids", []))
    incomplete = bool(unknown["successor"] or unknown["control"])
    insufficient = (
        complete == 0
        or control_regret is None
        or successor_regret is None
        or control_retention is None
        or successor_retention is None
    )
    numeric_pass = (
        not insufficient
        and reduction is not None
        and reduction >= 0.10
        and successor_retention >= 0.70
        and successor_retention >= control_retention
        and archive["harm_count"] == 0
    )
    return {
        "complete_decisive_rows": complete,
        "base_non_survivor_rows": decisive,
        "control_regret": control_regret,
        "successor_regret": successor_regret,
        "regret_reduction": reduction,
        "control_retention": control_retention,
        "successor_retention": successor_retention,
        "failed_selection": failed,
        "unknown_selection": unknown,
        "archive": archive,
        "control_archive": control_archive,
        "new_harm_row_ids_vs_control": sorted(successor_harm_ids - control_harm_ids, key=repr),
        "invalid_reasons": invalid_reasons,
        "label": decision_label(
            invalid=bool(invalid_reasons),
            incomplete=incomplete,
            insufficient=insufficient,
            hold=not archive["coverage"],
            numeric_pass=bool(numeric_pass),
        ),
    }


def decision_label(
    *,
    invalid: bool = False,
    incomplete: bool = False,
    insufficient: bool = False,
    hold: bool = False,
    numeric_pass: bool = False,
    f4_closed: bool = False,
) -> str:
    """Apply the ruling's required terminal precedence and F4 closure fence."""
    if invalid:
        return "INVALID"
    if incomplete:
        return "INCOMPLETE"
    if insufficient:
        return "INSUFFICIENT"
    if hold or f4_closed:
        return "HOLD"
    return "GO_OFFLINE" if numeric_pass else "KILL_SUCCESSOR"
