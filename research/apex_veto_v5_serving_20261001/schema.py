"""Shared record shapes for the Apex v5-veto web-serving run (standard library only).

Imported by the producer (``serving_run.py``) and loaded by file path by the stdlib
audit (``serving_audit.py``), so field names cannot drift between them. Names and
layout only: no decision logic, no pinned identities, no seed recipe. The layout
(intent/default_check/records/parity/receipt) is the v2 serving run's
(``research/apex_veto_serving_20261001/schema.py``); the schema id, study id, the Play
kind and the v5 diagnostics keys are new.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "apex-veto-v5-web-serving/v1"
STUDY_ID = "apex-veto-v5-serving-20261001"
AUTHORITY = "serving-qualification (non-promoting)"

KIND_WATCH = "watch_hero"
KIND_PLAY = "play"  # served unwrapped: Play AI wrapping is not part of the v5 release
KINDS = (KIND_WATCH, KIND_PLAY)

STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"

ENDED_HORIZON = "horizon"
ENDED_HUMAN_DEATH = "human_death"

# Exactly the keys of SafetyVetoCounters.to_dict() (src/evaluation/safety_veto.py), which
# BoostAwareFreeSpaceVeto.counters reuses unchanged.
COUNTER_KEYS = (
    "decisions",
    "kept_base",
    "vetoes_applied",
    "fallback_no_spacious",
    "vetoes_to_boost",
    "vetoed_base_boost",
    "vetoes_speed_switched",
)

# The integer keys of BoostAwareCounters.to_dict() (src/evaluation/safety_veto_v5.py).
# Its wall-clock keys (apply_seconds_total/max, mean_apply_seconds) are reported only.
DIAGNOSTIC_KEYS = (
    "decisions",
    "kept",
    "vetoes_applied",
    "no_spacious",
    "boost_base_decisions",
    "base_landing_failed",
    "boost_landing_vetoes",
    "boost_to_normal_same_direction",
    "boost_landing_v2_rule",
    "boost_landing_no_eligible",
    "landing_checks",
    "landing_failures",
    "action_differs_from_v2",
)
DIAGNOSTIC_TIMING_KEYS = ("apply_seconds_total", "apply_seconds_max", "mean_apply_seconds")

# The three serving env vars, as the harness set them for a build (None = unset).
ENV_KEYS = (
    "SNAKE_SERVE_VETO_WATCH_HERO",
    "SNAKE_SERVE_VETO_PLAY_AI",
    "SNAKE_SERVE_VETO_VARIANT",
)

# Output layout under --out (same names as the v2 serving run).
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
    "env",
    "checkpoint",
    "wrapper",
    "served",
    "veto",
    "snakes_with_veto",
    "decision_frames",
    "episode",
)

# Keys of an episode's ``veto`` block.
VETO_KEYS = (
    "active",
    "variant",
    "variant_requested",
    "scope",
    "reason",
    "flags",
    "strict_receipt_checkpoint_match",
    "strict_receipt_wrapper_match",
    "wrapped_snake_ids",
    "per_snake",
    "total",
    "diagnostics_per_snake",
    "diagnostics_total",
    "diagnostics_timing",
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
    "env",
    "session",
    "rollout",
    "frames_compared",
    "first_divergence_frame",
    "trace_sha256_equal",
    "veto_counters_equal",
    "diagnostics_equal",
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
