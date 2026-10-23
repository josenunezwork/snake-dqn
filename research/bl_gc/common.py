"""Shared production-path setup and immutable exports for round-25 GC checks.

The helpers deliberately construct only the parts of the app lifespan that exist
before the live ``GameSession``.  In particular, a qualification must never make
a disposable game before ``freeze_startup`` is applied.
"""

from __future__ import annotations

import gc
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from research.bl_served_search.exports import export_json

JUDGMENT_SHA256 = "df3c55403da147195dd0508c0f5185e955f20d939ae4f7853e1d2ba2f14d20e7"
_STORE: Any | None = None


def _source_hashes() -> dict[str, str]:
    """Bind each qualification receipt to the exact candidate source closure."""
    root = Path(__file__).resolve().parents[2]
    paths = (
        "web/backend/gc_policy.py",
        "web/backend/app.py",
        "web/backend/session.py",
        "web/backend/release_monitoring.py",
        "research/bl_gc/common.py",
        "research/bl_gc/identity.py",
        "research/bl_gc/lifecycle.py",
        "research/bl_gc/probe.py",
        "research/bl_gc/sunday_regression.py",
        "research/bl_gc/sunday_server.py",
        "research/bl_gc/run_timing_waiter.sh",
        "research/bl_k1/m1.py",
        "research/t15_mixed_roster_20261017/episodes.py",
        "research/t15_mixed_roster_20261017/measurements.py",
    )
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def _git_commit() -> str:
    """Return the exact checked-out commit without shell interpolation."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _idle_census() -> dict[str, Any]:
    """Capture the established idle schema when this Mac can provide it."""
    try:
        from research.bl_mac_latency.host import snapshot

        return snapshot()
    except Exception as exc:
        # Untimed identity/lifecycle subprocesses are still auditable on hosts
        # where the Mac-only snapshot cannot run. Timed A1 admission rejects it.
        return {"schema": "bl-mac-host/v1", "status": "UNAVAILABLE", "error": str(exc)}


def initialize(policy: str, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Initialize the exact pre-session serving boundary once per process.

    The production app import pins torch's serving thread count.  ScoreStore is
    then constructed exactly where lifespan constructs it, before the GC policy
    and before any game/session exists.
    """
    if policy not in {"default", "freeze_startup"}:
        raise ValueError(f"unknown GC policy {policy!r}")
    os.environ["SNAKE_SERVE_GC_POLICY"] = policy
    os.environ["SNAKE_SERVE_PATHB"] = "0"
    # This is still the lifespan's ScoreStore construction, but a qualification
    # process must not mutate the operator's persistent leaderboard database.
    os.environ.setdefault(
        "SNAKE_SCORES_DB", str(Path(tempfile.gettempdir()) / f"bl-gc-scores-{os.getpid()}.db")
    )
    # The real Watch constructor is part of the qualification process-lifetime
    # import closure, so load it before (never after) a startup freeze.
    import research.t15_mixed_roster_20261017.episodes  # noqa: F401
    from web.backend import app
    from web.backend.gc_policy import apply_startup_policy

    global _STORE
    if _STORE is None:
        _STORE = app.ScoreStore(os.environ.get("SNAKE_SCORES_DB", app.DEFAULT_SCORES_DB))
    receipt = dict(apply_startup_policy(stage=stage))
    attestation = {
        "judgment_sha256": JUDGMENT_SHA256,
        "interpreter": sys.executable,
        "version": sys.version,
        "git_commit": _git_commit(),
        "source_hashes": _source_hashes(),
        "gc_policy": receipt,
        "torch_num_threads": int(torch.get_num_threads()),
        "pathb": os.environ.get("SNAKE_SERVE_PATHB"),
        "thresholds": list(gc.get_threshold()),
        "enabled": bool(gc.isenabled()),
        "idle": _idle_census(),
    }
    if attestation["torch_num_threads"] != 1:
        raise RuntimeError("GC qualification requires torch.get_num_threads() == 1")
    return receipt, attestation


def new_session(seed: int, count: int = 6) -> tuple[Any, dict[str, Any]]:
    """Build a seeded real Watch session only after :func:`initialize`."""
    from research.t15_mixed_roster_20261017.episodes import new_session as build
    from research.t15_mixed_roster_20261017.episodes import roster_identity

    session = build(int(seed), int(count))
    return session, roster_identity(session)


def attest(session: Any, receipt: dict[str, Any]) -> dict[str, Any]:
    """Return the d.0 process and live-roster facts for one harness receipt."""
    from research.t15_mixed_roster_20261017.episodes import roster_identity

    return {
        "judgment_sha256": JUDGMENT_SHA256,
        "interpreter": sys.executable,
        "version": sys.version,
        "git_commit": _git_commit(),
        "source_hashes": _source_hashes(),
        "gc_policy": dict(receipt),
        "torch_num_threads": int(torch.get_num_threads()),
        "pathb": os.environ.get("SNAKE_SERVE_PATHB"),
        "roster": roster_identity(session),
        "idle": _idle_census(),
    }


def write_result(path: Path | str, payload: dict[str, Any]) -> dict[str, Any]:
    """Publish canonical JSON with SHA-256 and completion marker, create-only."""
    return export_json(Path(path), payload)
