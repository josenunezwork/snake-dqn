"""Shared record shapes for the Apex v7-veto web-serving run (standard library only).

Imported by the producer (``serving_run.py``) and loaded by file path by the stdlib
audit (``serving_audit.py``), so field names cannot drift between them. Names and
layout only: no decision logic, no pinned identities, no seed recipe. The layout
(intent/default_check/records/parity/receipt) is the v5 serving run's
(``research/apex_veto_v5_serving_20261001/schema.py``); the schema id, study id and the
v7 diagnostics keys (with the v5 keys nested under ``v5``) are new.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "apex-veto-v7-web-serving/v1"
STUDY_ID = "apex-veto-v7-serving-20261002"
AUTHORITY = "serving-qualification (non-promoting)"

KIND_WATCH = "watch_hero"
KIND_PLAY = "play"  # served unwrapped: Play AI wrapping is not part of a v7 release
KINDS = (KIND_WATCH, KIND_PLAY)

STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"

ENDED_HORIZON = "horizon"
ENDED_HUMAN_DEATH = "human_death"

# Exactly the keys of SafetyVetoCounters.to_dict() (src/evaluation/safety_veto.py), which
# SpacePreferenceVeto.counters reuses unchanged.
COUNTER_KEYS = (
    "decisions",
    "kept_base",
    "vetoes_applied",
    "fallback_no_spacious",
    "vetoes_to_boost",
    "vetoed_base_boost",
    "vetoes_speed_switched",
)

# The integer keys of SpacePreferenceCounters.to_dict() (src/evaluation/safety_veto_v7.py),
# excluding its nested ``v5`` dict. Its wall-clock keys are reported only.
DIAGNOSTIC_KEYS = (
    "decisions",
    "kept",
    "vetoes_applied",
    "no_spacious",
    "rerank_decisions",
    "rerank_pruned",
    "rerank_scored",
    "rerank_changes",
    "rerank_changes_of_v5_kept",
    "rerank_changes_boost_mode",
    "rerank_changes_tail_release_driven",
    "area_evaluations",
    "area_cap_hits",
    "area_cap_max",
    "static_area_evaluations",
    "extra_landing_checks",
)
# Aggregated across snakes by max, not sum (a per-snake peak, not a count).
DIAGNOSTIC_MAX_KEYS = ("area_cap_max",)
DIAGNOSTIC_TIMING_KEYS = (
    "apply_seconds_total",
    "apply_seconds_max",
    "mean_apply_seconds",
    "area_eval_seconds_total",
    "area_eval_seconds_max",
    "mean_area_eval_seconds_per_evaluation",
)
# The nested v5 counters: the integer keys of BoostAwareCounters.to_dict().
V5_DIAGNOSTIC_KEYS = (
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
V5_NESTED_KEY = "v5"

# The three serving env vars, as the harness set them for a build (None = unset).
ENV_KEYS = (
    "SNAKE_SERVE_VETO_WATCH_HERO",
    "SNAKE_SERVE_VETO_PLAY_AI",
    "SNAKE_SERVE_VETO_VARIANT",
)

# Output layout under --out (same names as the v2/v5 serving runs).
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
