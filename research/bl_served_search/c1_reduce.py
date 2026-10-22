"""Pure, fail-closed reduction for PLAN_v3 C1's single registered look.

C1 execution and immutable exports deliberately live in the coordinator.  This
module accepts only the already-verified raw session objects and never opens a
file or starts a game.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from scipy.stats import t as student_t

HORIZON = 14_400
SESSION_IDS = frozenset(f"session-{index:02d}" for index in range(24))
ONE_SIDED_90_DF23 = float(student_t.ppf(0.9, 23))


def _fail(message: str) -> None:
    raise ValueError(f"invalid C1 raw evidence: {message}")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{label} must be finite numeric")
    return float(value)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    _number(value, label)
    if int(value) != value or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return int(value)


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{label} must be bool")
    return value


def _quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p90": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)

    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * percent
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _outcome(outcomes: dict[Any, Any], action: int, label: str) -> dict[str, Any]:
    raw = outcomes.get(str(action), outcomes.get(action))
    if not isinstance(raw, dict):
        _fail(f"{label} outcome missing for action {action}")
    survives = _bool(raw.get("survives"), f"{label} outcome survives")
    frames = _integer(raw.get("frames_executed"), f"{label} outcome frames_executed")
    return {"survives": survives, "frames_executed": frames}


def summarize_arm(frames: list[dict], decisions: list[dict], batches: list[dict]) -> dict:
    """Validate and summarize one raw C1 arm without inspecting an outcome rule.

    A session's actual decision count is deliberately its own denominator.  The
    returned witness and death identifiers make all endpoint counts auditable.
    """
    if not isinstance(frames, list) or len(frames) != HORIZON:
        _fail(f"frames must contain exactly {HORIZON} rows")
    if not isinstance(decisions, list) or not isinstance(batches, list):
        _fail("decisions and batches must be lists")

    normalized_frames: dict[int, dict[str, Any]] = {}
    masses: list[int] = []
    death_frames: list[int] = []
    death_rows: list[dict[str, int]] = []
    death_by_life: dict[int, list[int]] = {}
    for expected, row in enumerate(frames, start=1):
        if not isinstance(row, dict):
            _fail("frame row must be an object")
        frame = _integer(row.get("frame"), "frame", 1)
        if frame != expected:
            _fail("frames must be ordered, unique, and cover 1..14400")
        mass = _integer(row.get("mass"), "frame mass")
        alive = _bool(row.get("alive"), "frame alive")
        death = _bool(row.get("death"), "frame death")
        life = _integer(row.get("life"), "frame life", 1)
        if death and alive:
            _fail("a death frame cannot be alive")
        if not alive and mass != 0:
            _fail("dead frames must have zero mass")
        normalized = {"frame": frame, "mass": mass, "alive": alive, "death": death, "life": life}
        normalized_frames[frame] = normalized
        masses.append(mass)
        if death:
            death_frames.append(frame)
            death_rows.append({"frame": frame, "life": life})
            death_by_life.setdefault(life, []).append(frame)

    decision_by_frame: dict[int, dict[str, int]] = {}
    decision_lengths: list[int] = []
    boosts = 0
    for row in decisions:
        if not isinstance(row, dict):
            _fail("decision row must be an object")
        frame = _integer(row.get("frame"), "decision frame", 1)
        life = _integer(row.get("life"), "decision life", 1)
        length = _integer(row.get("length"), "decision length")
        bare = _integer(row.get("bare"), "decision bare action")
        chosen = _integer(row.get("chosen"), "decision chosen action")
        if bare > 5 or chosen > 5:
            _fail("C1 action is outside the six legal actions")
        if frame in decision_by_frame:
            _fail("duplicate decision frame")
        source = normalized_frames.get(frame)
        if (
            source is None
            or source["life"] != life
            or (not source["alive"] and not source["death"])
        ):
            _fail("decision must reference an alive or death frame in the same life")
        decision_by_frame[frame] = {
            "frame": frame,
            "life": life,
            "length": length,
            "bare": bare,
            "chosen": chosen,
        }
        decision_lengths.append(length)
        boosts += int(chosen >= 3)

    witnesses: dict[int, dict[str, int]] = {}
    negative_witnesses: list[dict[str, int]] = []
    intervention_rows: list[dict[str, Any]] = []
    batch_frames: set[int] = set()
    for row in batches:
        if not isinstance(row, dict):
            _fail("batch row must be an object")
        frame = _integer(row.get("frame"), "batch frame", 1)
        life = _integer(row.get("life"), "batch life", 1)
        bare = _integer(row.get("bare"), "batch bare action")
        chosen = _integer(row.get("chosen"), "batch chosen action")
        if frame in batch_frames:
            _fail("duplicate batch frame")
        batch_frames.add(frame)
        decision = decision_by_frame.get(frame)
        if decision is None or (life, bare, chosen) != (
            decision["life"],
            decision["bare"],
            decision["chosen"],
        ):
            _fail("batch does not match its raw decision")
        outcomes = row.get("outcomes")
        if not isinstance(outcomes, dict):
            _fail("batch outcomes must be an object")
        bare_outcome = _outcome(outcomes, bare, "bare")
        chosen_outcome = _outcome(outcomes, chosen, "chosen")
        intervention_rows.append(
            {
                "frame": frame,
                "life": life,
                "bare": bare,
                "chosen": chosen,
                "outcomes": outcomes,
                "bare_outcome": bare_outcome,
                "chosen_outcome": chosen_outcome,
            }
        )
        if chosen_outcome["survives"] and not bare_outcome["survives"]:
            existing = witnesses.get(life)
            if existing is None or frame < existing["frame"]:
                witnesses[life] = {"life": life, "frame": frame, "bare": bare, "chosen": chosen}
        if not chosen_outcome["survives"] and bare_outcome["survives"]:
            negative_witnesses.append(
                {"life": life, "frame": frame, "bare": bare, "chosen": chosen}
            )

    witness_rows = [witnesses[life] for life in sorted(witnesses)]
    re_deaths_300: list[dict[str, int]] = []
    re_deaths_1000: list[dict[str, int]] = []
    for witness in witness_rows:
        redeath = next(
            (death for death in death_by_life.get(witness["life"], []) if death > witness["frame"]),
            None,
        )
        if redeath is not None:
            row = {**witness, "death_frame": redeath, "elapsed_frames": redeath - witness["frame"]}
            if row["elapsed_frames"] <= 300:
                re_deaths_300.append(row)
            if row["elapsed_frames"] <= 1000:
                re_deaths_1000.append(row)

    actual_decisions = len(decisions)
    if actual_decisions == 0:
        _fail("session arm contains no actual hero decisions")
    if any(frame not in decision_by_frame for frame in death_frames):
        _fail("each post-step death frame must retain its actual decision")
    if any(len(life_deaths) > 1 for life_deaths in death_by_life.values()):
        _fail("a continuous-respawn life cannot contain more than one death")
    return {
        "frames": HORIZON,
        "actual_decisions": actual_decisions,
        "deaths": len(death_frames),
        "death_frames": death_frames,
        "death_rows": death_rows,
        "deaths_per_1000_decisions": len(death_frames) * 1000.0 / actual_decisions,
        "mass_sum": sum(masses),
        "mass_per_frame": sum(masses) / HORIZON,
        "boost_actions": boosts,
        "boost_share": boosts / actual_decisions,
        "length_quantiles": _quantiles(decision_lengths),
        "distinct_rescue_witness_lives": len(witness_rows),
        "rescue_witnesses": witness_rows,
        "newly_caused_death_witnesses_h300": negative_witnesses,
        "newly_caused_death_witness_count_h300": len(negative_witnesses),
        "interventions": intervention_rows,
        "same_life_redeaths_within_300": len(re_deaths_300),
        "same_life_redeaths_within_1000": len(re_deaths_1000),
        "same_life_redeath_300_rows": re_deaths_300,
        "same_life_redeath_1000_rows": re_deaths_1000,
    }


def _bound(values: Iterable[float]) -> dict[str, float | int]:
    data = list(values)
    if len(data) != 24 or not all(math.isfinite(value) for value in data):
        _fail("registered look requires 24 finite paired values")
    origin = data[0]
    mean = origin + math.fsum(value - origin for value in data) / len(data)
    variance = math.fsum((value - mean) ** 2 for value in data) / (len(data) - 1)
    standard_error = math.sqrt(variance / len(data))
    margin = ONE_SIDED_90_DF23 * standard_error
    return {
        "n": 24,
        "df": 23,
        "mean": mean,
        "se": standard_error,
        "lb90": mean - margin,
        "ub90": mean + margin,
    }


def reduce_sessions(units: list[dict]) -> dict:
    """Perform the sole frozen C1 look after validating all raw paired sessions."""
    if not isinstance(units, list) or len(units) != 24:
        _fail("registered look requires exactly 24 units")
    session_ids: set[str] = set()
    namespaces: set[str] = set()
    seeds: set[int] = set()
    session_rows: list[dict[str, Any]] = []
    for unit in units:
        if not isinstance(unit, dict):
            _fail("unit must be an object")
        unit_id = unit.get("unit_id")
        namespace = unit.get("namespace")
        if not isinstance(unit_id, str) or not isinstance(namespace, str) or not namespace:
            _fail("unit identifier and namespace are required")
        seed = _integer(unit.get("seed"), "unit seed")
        if unit.get("horizon") != HORIZON:
            _fail("unit horizon must be 14400")
        arms = unit.get("arms")
        if not isinstance(arms, dict) or set(arms) != {"bare", "dense"}:
            _fail("unit must contain exactly bare and dense arms")
        if unit_id in session_ids or seed in seeds:
            _fail("duplicate session identifier or seed")
        session_ids.add(unit_id)
        namespaces.add(namespace)
        seeds.add(seed)
        summaries = {}
        for name in ("bare", "dense"):
            arm = arms[name]
            if not isinstance(arm, dict) or not {"frames", "decisions", "batches"} <= set(arm):
                _fail(f"{name} arm raw shape is invalid")
            summaries[name] = summarize_arm(arm["frames"], arm["decisions"], arm["batches"])
        if summaries["bare"]["mass_sum"] <= 0:
            _fail("bare mass must be positive for relative mass endpoint")
        dense_deaths = summaries["dense"]["deaths_per_1000_decisions"]
        bare_deaths = summaries["bare"]["deaths_per_1000_decisions"]
        dense_mass = summaries["dense"]["mass_sum"]
        bare_mass = summaries["bare"]["mass_sum"]
        newly_caused = [
            frame
            for frame in summaries["dense"]["death_frames"]
            if arms["bare"]["frames"][frame - 1]["alive"]
        ]
        session_rows.append(
            {
                "unit_id": unit_id,
                "seed": seed,
                "bare": summaries["bare"],
                "dense": summaries["dense"],
                "death_rate_delta_per_1000": dense_deaths - bare_deaths,
                "relative_mass_difference": (dense_mass - bare_mass) / bare_mass,
                "dense_death_while_bare_alive_frames": newly_caused,
                "dense_death_while_bare_alive_count": len(newly_caused),
                "paired_frame_death_disclosure": (
                    "descriptive paired-frame endpoint; causal attribution is not established"
                ),
            }
        )
    if session_ids != SESSION_IDS:
        _fail("session ids must be exactly session-00 through session-23")
    if len(namespaces) != 1:
        _fail("all units must share one namespace")
    session_rows.sort(key=lambda row: row["unit_id"])
    death = _bound(row["death_rate_delta_per_1000"] for row in session_rows)
    mass = _bound(row["relative_mass_difference"] for row in session_rows)
    baseline_deaths = sum(row["bare"]["deaths"] for row in session_rows)
    witness_sessions = sum(
        row["dense"]["distinct_rescue_witness_lives"] > 0 for row in session_rows
    )
    mass_stop = mass["ub90"] < -0.05
    if mass_stop:
        verdict = "STOP"
    elif baseline_deaths < 8:
        verdict = "INSUFFICIENT"
    elif death["ub90"] < 0 and mass["lb90"] >= -0.05 and witness_sessions >= 4:
        verdict = "GO"
    elif death["lb90"] >= 0:
        verdict = "STOP"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "schema": "bl-plan-v3-c1-reducer/v1",
        "namespace": next(iter(namespaces)),
        "session_count": 24,
        "session_ids": sorted(session_ids),
        "sessions": session_rows,
        "baseline_deaths": baseline_deaths,
        "witness_sessions": witness_sessions,
        "death_rate_delta_per_1000": death,
        "relative_mass_difference": mass,
        "verdict": verdict,
        "rules": {
            "go": (
                "death UB90 < 0; mass LB90 >= -0.05; witnesses in >=4 sessions; "
                ">=8 baseline deaths"
            ),
            "stop": "death LB90 >= 0 with >=8 baseline deaths, or mass UB90 < -0.05",
            "insufficient_precedence": (
                "after mass STOP, fewer than 8 baseline deaths is INSUFFICIENT"
            ),
        },
    }
