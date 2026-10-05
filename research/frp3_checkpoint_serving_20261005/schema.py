"""Shared record shapes for the FRP-v3 s12 checkpoint-swap web-serving run (stdlib only).

Imported by the producer (``serving_run.py``) and loaded by file path by the stdlib audit
(``serving_audit.py``), so field names cannot drift between them. Names and layout only: no
decision logic, no pinned identities, no seed recipe. The layout
(intent/default_check/records/parity/receipt) and the v8 counter/diagnostics key sets are the
v8 serving lane's (``research/apex_veto_v8_serving_20261003/schema.py``, loaded here by file
path and re-exported unchanged); the schema id, study id, the fourth env key
(``SNAKE_SERVE_CHECKPOINT``), the ``served_checkpoint`` block and the SIMD parity keys are new.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_V8_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "apex_veto_v8_serving_20261003" / "schema.py"
)


def _load_v8_schema() -> Any:
    name = "frp3_serving_v8_schema"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _V8_SCHEMA_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v8schema = _load_v8_schema()

SCHEMA = "frp3-checkpoint-web-serving/v1"
STUDY_ID = "frp3-checkpoint-serving-20261005"
AUTHORITY = "serving-qualification (non-promoting)"

KIND_WATCH = v8schema.KIND_WATCH
KIND_PLAY = v8schema.KIND_PLAY  # served unwrapped (Play AI wrapping is not released)
KINDS = (KIND_WATCH, KIND_PLAY)
STATUS_COMPLETE = v8schema.STATUS_COMPLETE
STATUS_ERROR = v8schema.STATUS_ERROR
ENDED_HORIZON = v8schema.ENDED_HORIZON
ENDED_HUMAN_DEATH = v8schema.ENDED_HUMAN_DEATH

# The v8 counter / diagnostics key sets, unchanged (the served veto is the released v8).
COUNTER_KEYS = v8schema.COUNTER_KEYS
DIAGNOSTIC_KEYS = v8schema.DIAGNOSTIC_KEYS
DIAGNOSTIC_TIMING_KEYS = v8schema.DIAGNOSTIC_TIMING_KEYS
HEAD_AVOIDANCE_KEY = v8schema.HEAD_AVOIDANCE_KEY
REFERENCE_LAMBDA_KEY = v8schema.REFERENCE_LAMBDA_KEY
V7_NESTED_KEY = v8schema.V7_NESTED_KEY
V7_PROBE_COUNTERS_KEY = v8schema.V7_PROBE_COUNTERS_KEY
V7_DIAGNOSTIC_KEYS = v8schema.V7_DIAGNOSTIC_KEYS
V7_DIAGNOSTIC_MAX_KEYS = v8schema.V7_DIAGNOSTIC_MAX_KEYS
V5_DIAGNOSTIC_KEYS = v8schema.V5_DIAGNOSTIC_KEYS
V5_NESTED_KEY = v8schema.V5_NESTED_KEY

# The four serving env vars, as the harness set them for a build (None = unset).
ENV_KEYS = (
    "SNAKE_SERVE_CHECKPOINT",
    "SNAKE_SERVE_VETO_WATCH_HERO",
    "SNAKE_SERVE_VETO_PLAY_AI",
    "SNAKE_SERVE_VETO_VARIANT",
)

INTENT = v8schema.INTENT
DEFAULT_CHECK = v8schema.DEFAULT_CHECK
RECEIPT = v8schema.RECEIPT
RECORDS_DIR = v8schema.RECORDS_DIR
PARITY_DIR = v8schema.PARITY_DIR
AUDIT_DIR = v8schema.AUDIT_DIR

# v8's episode keys plus the served-checkpoint resolution the session made.
EPISODE_KEYS = tuple(v8schema.EPISODE_KEYS) + ("served_checkpoint",)
VETO_KEYS = v8schema.VETO_KEYS
# Keys of ``served_checkpoint`` (web.backend.served_checkpoint.ServedCheckpointChoice.to_dict).
SERVED_CHECKPOINT_KEYS = ("requested", "name", "path", "sha256", "reason", "pin_problems")

# Session side vs SIMD side (run_simd_eval, vector61 rowwise, hero v8) on the same seed.
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
    "simd",
    "frames_compared",
    "first_divergence_frame",
    "divergence",
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
