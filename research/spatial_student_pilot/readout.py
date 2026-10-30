"""Single final-checkpoint development readout, run only on the Mac.

Root bootstrap resamples whole parent worlds with equal root weights using 10,000
prospectively fixed draws. No outcome is used to select checkpoints or roots.
"""

from __future__ import annotations

import platform
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

MIXES = ("scripted", "frozen", "mixed")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 3432


def clustered_lb90(
    rows: Sequence[Mapping[str, Any]], values: Sequence[float], *, stratified: bool = False
) -> float:
    """One-sided LB90: resample whole parents, preserving within-parent roots.

    Roots use equal root weight without mix stratification. Gameplay optionally
    resamples paired worlds independently within each mix then weights mixes equally.
    """
    if len(rows) != len(values) or not rows:
        raise ValueError("nonempty aligned rows and values required")
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row, value in zip(rows, values):
        if row["mix"] not in MIXES or not np.isfinite(value):
            raise ValueError("unknown mix or nonfinite bootstrap input")
        group = str(row["mix"]) if stratified else "all"
        grouped[group][int(row["world_id"])].append(float(value))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rep = np.zeros(BOOTSTRAP_DRAWS, dtype=np.float64)
    groups = MIXES if stratified else ("all",)
    for group in groups:
        parents = grouped[group]
        if not parents:
            raise ValueError("bootstrap requires registered mix support")
        ordered = sorted(parents)
        sums = np.asarray([sum(parents[p]) for p in ordered])
        counts = np.asarray([len(parents[p]) for p in ordered])
        indices = rng.integers(0, len(ordered), (BOOTSTRAP_DRAWS, len(ordered)))
        rep += (sums[indices].sum(1) / counts[indices].sum(1)) / len(groups)
    return float(np.quantile(rep, 0.10))


def _equal_mix_mean(rows: Sequence[Mapping[str, Any]], values: Sequence[float]) -> float:
    means = [np.mean([v for r, v in zip(rows, values) if r["mix"] == mix]) for mix in MIXES]
    if not np.isfinite(means).all():
        raise ValueError("missing mix support")
    return float(np.mean(means))


def _dead(row: Mapping[str, Any], arm: str) -> bool:
    return all(float(x) == 0.0 for x in row[arm + "_survival"])


def evaluate_readout(
    root_rows: Sequence[Mapping[str, Any]],
    gameplay_rows: Sequence[Mapping[str, Any]],
    *,
    support: Mapping[str, Any],
    latency_p95_ms: float | None,
    complete: bool,
    integrity: bool = True,
) -> dict[str, Any]:
    """Apply frozen INVALID/INCOMPLETE precedence, then every mandatory GO clause.

    ``support`` is the Mac verification boundary: collision_audit and resume_verified
    must both be true. ``allow_non_mac`` is intentionally unsupported. Tiny units
    are mechanism fixtures and cannot obtain a pilot GO.
    """
    result: dict[str, Any] = {
        "schema": "spatial-student-readout/v1",
        "verdict": "INVALID",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "clauses": {},
    }
    invalid = []
    namespace = "t34-spatial-student-20261030/v1"
    if support.get("namespace") != namespace or support.get("tiny") or support.get("preflight"):
        invalid.append("foreign scientific namespace or development fixture")
    if any(not str(row.get("uid", "")).startswith(namespace + "/a0/") for row in root_rows):
        invalid.append("foreign root namespace/allocation prefix")
    if platform.system() != "Darwin":
        invalid.append("registered readout is Mac only")
    if (
        not integrity
        or support.get("collision_audit") is not True
        or support.get("resume_verified") is not True
    ):
        invalid.append("unverified integrity, collision audit, or resume ledger")
    if len({str(r["uid"]) for r in root_rows}) != len(root_rows):
        invalid.append("duplicate root UID")
    if len({int(r["world_id"]) for r in gameplay_rows}) != len(gameplay_rows):
        invalid.append("duplicate gameplay world")
    late = [r for r in root_rows if r["kind"] == "late"]
    ordinary = [r for r in root_rows if r["kind"] == "ordinary"]
    parents = Counter(int(r["world_id"]) for r in late)
    if len(late) != 128 or len(ordinary) != 256 or len(root_rows) != 384:
        invalid.append("missing frozen assessment support (128 late,256 ordinary)")
    if len(parents) < 64 or (parents and max(parents.values()) > 2):
        invalid.append("late support requires >=64 parents and <=2 roots per parent")
    if (
        any(sum(r["mix"] == mix for r in gameplay_rows) != 64 for mix in MIXES)
        or len(gameplay_rows) != 192
    ):
        invalid.append("gameplay requires 192 paired worlds,64 per mix")
    nonfinite = False
    try:
        for row in root_rows:
            if row.get("partition") != "assessment":
                invalid.append("foreign assessment partition")
            if list(row["scoring_suffixes"]) != [4, 5] or not row["selection_freeze_sha256"]:
                invalid.append("unverified independent scoring suffixes or selection freeze")
            for arm in ("b8", "teacher", "student", "closed_loop"):
                for key in ("mass", "survival"):
                    values = np.asarray(row[arm + "_" + key], dtype=float)
                    if values.shape != (2,) or not np.isfinite(values).all():
                        nonfinite = True
                if any(float(x) not in (0.0, 1.0) for x in row[arm + "_survival"]):
                    invalid.append("invalid suffix survival values")
        for row in gameplay_rows:
            left, right = row.get("b8_initial_state_sha256"), row.get(
                "student_initial_state_sha256"
            )
            if not isinstance(left, str) or len(left) != 64 or left != right:
                invalid.append("missing or unequal paired initial state digests")
            counts = np.asarray(row.get("student_action_counts", [0] * 6), dtype=float)
            if (
                counts.shape != (6,)
                or not np.isfinite(counts).all()
                or (counts < 0).any()
                or not np.equal(counts, np.floor(counts)).all()
            ):
                invalid.append("malformed student action counts")
            if row["mix"] not in MIXES:
                invalid.append("unknown gameplay mix")
            if not all(
                np.isfinite(row[a + k]) for a in ("b8", "student") for k in ("_mass", "_alive")
            ):
                nonfinite = True
    except (KeyError, ValueError, TypeError):
        invalid.append("malformed or missing endpoint records")
    if latency_p95_ms is not None and not np.isfinite(latency_p95_ms):
        invalid.append("nonfinite latency result")
    if nonfinite:
        invalid.append("nonfinite/malformed endpoint outputs")
    if invalid:
        result["reasons"] = sorted(set(invalid))
        return result
    if not complete or latency_p95_ms is None:
        result.update(
            verdict="INCOMPLETE", reasons=["unfinished endpoint or Mac latency measurement"]
        )
        return result
    baseline_deaths = sum(_dead(r, "b8") for r in late)
    try:
        teacher_delta = [float(np.mean(r["teacher_mass"]) - np.mean(r["b8_mass"])) for r in late]
        student_delta = [float(np.mean(r["student_mass"]) - np.mean(r["b8_mass"])) for r in late]
        teacher_gain = float(np.mean(teacher_delta))
        student_gain = float(np.mean(student_delta))
        teacher_lb = clustered_lb90(late, teacher_delta)
        student_lb = clustered_lb90(late, student_delta)
    except ValueError as error:
        result["reasons"] = [str(error)]
        return result
    student_deaths = sum(_dead(r, "student") for r in late)
    flips = sum(
        _dead(r, "student") and all(float(x) == 1.0 for x in r["b8_survival"]) for r in ordinary
    )
    mass_student = _equal_mix_mean(gameplay_rows, [float(r["student_mass"]) for r in gameplay_rows])
    mass_b8 = _equal_mix_mean(gameplay_rows, [float(r["b8_mass"]) for r in gameplay_rows])
    alive_deltas = {
        mix: float(
            np.mean([r["student_alive"] - r["b8_alive"] for r in gameplay_rows if r["mix"] == mix])
        )
        for mix in MIXES
    }
    counts = np.zeros(6, dtype=np.int64)
    for row in gameplay_rows:
        counts += np.asarray(row.get("student_action_counts", [0] * 6), dtype=np.int64)
    if counts.sum() == 0 or (counts < 0).any():
        result["reasons"] = ["missing actual alive student action counts"]
        return result
    collapse = float(counts.max() / counts.sum())
    illegal = any(bool(r.get("illegal_output", False)) for r in [*root_rows, *gameplay_rows])
    clauses = {
        "baseline_late_deaths_at_least16": baseline_deaths >= 16,
        "teacher_mass_lb90_positive": teacher_lb > 0.0,
        "student_mass_lb90_positive": student_lb > 0.0,
        "student_retains_half_teacher": student_gain >= 0.5 * teacher_gain,
        "late_death_reduction20pct": student_deaths <= 0.8 * baseline_deaths,
        "ordinary_lethal_flips_at_most1": flips <= 1,
        "equal_mix_gameplay_mass_at_least_b8": mass_student >= mass_b8,
        "each_mix_alive_loss_at_most03": all(v >= -0.03 for v in alive_deltas.values()),
        "mac_observation_actor_p95_at_most4ms": latency_p95_ms is not None
        and np.isfinite(latency_p95_ms)
        and latency_p95_ms <= 4.0,
        "no_action_collapse": collapse <= 0.90,
        "legal_outputs": not illegal,
    }
    result.update(
        verdict="GO" if all(clauses.values()) else "KILL",
        clauses=clauses,
        metrics={
            "teacher_gain": teacher_gain,
            "teacher_lb90": teacher_lb,
            "student_gain": student_gain,
            "student_lb90": student_lb,
            "baseline_late_deaths": baseline_deaths,
            "student_late_deaths": student_deaths,
            "ordinary_lethal_flips": flips,
            "gameplay_mass_b8": mass_b8,
            "gameplay_mass_student": mass_student,
            "gameplay_mass_delta_lb90_diagnostic": clustered_lb90(
                gameplay_rows,
                [float(r["student_mass"] - r["b8_mass"]) for r in gameplay_rows],
                stratified=True,
            ),
            "gameplay_alive_deltas": alive_deltas,
            "action_counts": counts.tolist(),
            "action_collapse_fraction": collapse,
            "latency_p95_ms": latency_p95_ms,
        },
    )
    return result
