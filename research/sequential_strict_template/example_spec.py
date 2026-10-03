"""Instantiation template for :mod:`sequential_runner` (copy into the study's package).

Nothing here plays a game: ``episode_runner`` raises until a study replaces it.  A study
(for example the v8 veto challenge) copies this file into its own package, fills in the
four marked parts and points ``prepare --spec <its.module>:SPEC`` at it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from research.sequential_strict_template.sequential_runner import (
    DEFAULT_CLOSURE_ROOTS,
    StudySpec,
    default_row,
    survival_bands,
)

MIXES = ("frozen", "scripted", "mixed")


def arm_identities() -> Dict[str, Dict[str, Any]]:
    """(1) Each arm's identity: checkpoint sha256, wrapper method/descriptor, and the sha256
    of every module the arm executes (computed from the files, never typed in)."""
    return {
        "incumbent": {"method": "REPLACE-ME/incumbent", "source_sha256s": {}},
        "candidate": {"method": "REPLACE-ME/candidate", "source_sha256s": {}},
    }


def episode_runner(
    episode: Mapping[str, Any], row: Mapping[str, Any], context: Any
) -> Mapping[str, Any]:
    """(2) Play one hero episode of ``episode['arm']`` on ``row`` and return its record.

    A real study installs the arm's wrapper and calls ``tournament_eval.rollout`` here (see
    ``research/apex_veto_v7_strict_20261002/strict_run.py::run_unit_episode``).
    """
    raise NotImplementedError("example_spec: instantiate episode_runner for a real study")


def build_row(phase: str, mix: str, world_index: int, world_seed: int) -> Dict[str, Any]:
    """(3) The world's roster (opponents, world identity); keep the four base keys."""
    return default_row(phase, mix, world_index, world_seed)


def validate_record(entry: Mapping[str, Any], row: Mapping[str, Any]) -> List[str]:
    """(4) Record-shape rules (e.g. ``strict_promotion.validate_strict_world_record``)."""
    return []


SPEC = StudySpec(
    study_id="example-sequential-strict-REPLACE-ME",
    schema="example-sequential-strict/v1",
    namespaces={
        "calibration": "example-sequential-strict-dev-v1",
        "final": "example-sequential-strict-final-v1",
    },
    arm_identities=arm_identities,
    episode_runner=episode_runner,
    mixes=MIXES,
    scripted_mix="scripted",
    primary_metric="mass_integral",
    ni_fraction=0.03,
    bands=survival_bands(MIXES),
    build_row=build_row,
    validate_record=validate_record,
    closure_roots=DEFAULT_CLOSURE_ROOTS,
    protocol_path=None,
)
