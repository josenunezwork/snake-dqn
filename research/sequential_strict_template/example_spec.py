"""Instantiation template for :mod:`sequential_runner` (copy into the study's package).

Nothing here plays a game: ``episode_runner`` raises until a study replaces it.  A study
(for example the v8 veto challenge) copies this file into its own package, fills in the
six marked parts (a seventh for a gate on RunPod, template v3) and points
``prepare --spec <its.module>:SPEC`` at it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

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


def excluded_seeds() -> Mapping[str, Sequence[int]]:
    """(5) Every earlier namespace (screens, sweeps, strict banks, serving lanes, smokes)
    whose seeds the new banks must avoid, e.g. the first 1000 seeds of each domain."""
    raise NotImplementedError("example_spec: list every earlier namespace before prepare")


def remote_worker_setup(intent: Mapping[str, Any], ckpt_dir: Path) -> Any:
    """(7, template v3 only) The per-worker ``context`` on a RunPod worker: the same as
    ``worker_setup`` but every checkpoint is ``ckpt_dir / f"{sha256}.pth"`` (verify each
    sha256 before use). Leave as is for a gate that stays on the Mac."""
    raise NotImplementedError("example_spec: instantiate remote_worker_setup for a remote gate")


def remote_checkpoints() -> Sequence[str]:
    """(7, template v3 only) sha256 of every checkpoint a remote episode loads; each must be
    on research/runpod_fanout/checkpoint_allowlist.json (the owner approves uploads)."""
    raise NotImplementedError("example_spec: list the remote checkpoints for a remote gate")


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
    excluded_seeds=excluded_seeds,
    closure_roots=DEFAULT_CLOSURE_ROOTS,
    # (7) Template v3 only (prepare --remote-config, gate on RunPod serverless):
    remote_worker_setup=remote_worker_setup,
    remote_checkpoints=remote_checkpoints,
    # (6) Pre-registration documents, relative to the repo, frozen by sha256 in the intent:
    protocol_path="research/REPLACE-ME/protocol.md",
    oc_report_path="research/REPLACE-ME/operating_characteristics.json",
    band_cost_report_path="research/REPLACE-ME/look1_band_cost.json",
)
