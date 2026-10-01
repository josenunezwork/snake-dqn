"""Shared record shapes for the Apex veto web-serving run (standard library only).

Imported by the producer (``serving_run.py``) and loaded by file path by the
stdlib audit (``serving_audit.py``), so field names cannot drift between them
(governance_tiers_2026-09-26.md "Audit scope", rule 3). It holds names and
layout only: no decision logic, no pinned identities, no seed recipe.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "apex-veto-web-serving/v1"
STUDY_ID = "apex-veto-serving-20261001"
AUTHORITY = "serving-qualification (non-promoting)"

KIND_WATCH = "watch_hero"
KIND_PLAY = "play_ai"
KINDS = (KIND_WATCH, KIND_PLAY)

STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"

ENDED_HORIZON = "horizon"
ENDED_HUMAN_DEATH = "human_death"

# Exactly the keys of SafetyVetoCounters.to_dict() (src/evaluation/safety_veto.py).
COUNTER_KEYS = (
    "decisions",
    "kept_base",
    "vetoes_applied",
    "fallback_no_spacious",
    "vetoes_to_boost",
    "vetoed_base_boost",
    "vetoes_speed_switched",
)

# Output layout under --out.
INTENT = "intent.json"
DEFAULT_CHECK = "default_check.json"
RECEIPT = "receipt.json"
RECORDS_DIR = "records"
PARITY_DIR = "parity"
AUDIT_DIR = "audit"

EPISODE_KEYS = (
    "schema_version",
    "study_id",
    "authority",
    "intent_sha256",
    "episode_index",
    "episode_id",
    "kind",
    "world_seed",
    "status",
    "error",
    "horizon_frames",
    "frames_stepped",
    "ended_by",
    "wall_seconds",
    "checkpoint",
    "wrapper",
    "served",
    "veto",
    "decision_frames",
    "episode",
)

PARITY_KEYS = (
    "schema_version",
    "study_id",
    "intent_sha256",
    "probe_index",
    "probe_id",
    "world_seed",
    "status",
    "error",
    "horizon_frames",
    "config",
    "session",
    "rollout",
    "frames_compared",
    "first_divergence_frame",
    "trace_sha256_equal",
    "veto_counters_equal",
    "wall_seconds",
)


def canonical_json(value: Any) -> str:
    """Canonical JSON used for hashing (sorted keys, no NaN)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_id(kind: str, index: int) -> str:
    return f"{kind}-{index:03d}"


def parity_id(index: int) -> str:
    return f"parity-{index:03d}"
