"""Opt-in serving hooks for the Apex free-space safety veto (default OFF).

``src/evaluation/safety_veto.py`` (``free-space-veto/v2-speed-preserving``) is
the serving-time wrapper the strict gate measured (one wrapped hero against
unwrapped opponents). This module installs that exact wrapper, unchanged, on
snakes the web app serves. Nothing here runs unless an operator turns it on:

* ``SNAKE_SERVE_VETO_WATCH_HERO=1`` wraps the served Watch hero (the slot-0
  snake every Watch build starts with); the other Watch snakes stay unwrapped,
  matching the measured configuration.
* ``SNAKE_SERVE_VETO_PLAY_AI=1`` wraps every AI snake in Play mode (never the
  human). Wrapping more than one hero was NOT measured by the strict gate.

RELEASED 2026-10-01 (user-approved, strict receipt 18b65519…, serving run
SERVING_PASS): when ``SNAKE_SERVE_VETO_WATCH_HERO`` is unset the Watch hero flag
is ON. Rollback: set it to ``0``/``false``/``no``/``off``. ``SNAKE_SERVE_VETO_PLAY_AI``
stays opt-in: only ``1``/``true``/``yes``/``on`` enable it. The flags are
read on every session build, like ``SNAKE_MECHANICS_V2``; a
:class:`GameSession` may instead be given explicit :class:`ServingVetoFlags`.

The wrapper is installed only on a vector61 Apex policy in watch or play mode,
and only when the build is bound to the strict-gate evidence (fail closed): the
served checkpoint bytes must hash to :data:`STRICT_RECEIPT_CHECKPOINT_SHA256`
and the wrapper source to :data:`STRICT_RECEIPT_WRAPPER_SOURCE_SHA256`. Untrained
weights (no checkpoint) and any other checkpoint, including one a client loads
with the ``load_checkpoint`` control, are never wrapped. Train mode, raster
checkpoints and HumanSnakes are never wrapped either. A requested flag that
cannot take effect is reported in :attr:`ServingVetoState.reason` rather than
raised, so a misconfigured flag cannot take the server down.

Whenever a flag is requested, each build logs one INFO line (prefix
:data:`LOG_PREFIX`) with the scope, wrapped ids, checkpoint match and wrapper
source sha256, or the reason nothing was installed, so an operator can confirm a
release or rollback on the running server. The wire payload is unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from src.evaluation import safety_veto as _veto_module
from src.evaluation.safety_veto import (
    VETO_METHOD,
    FreeSpaceVeto,
    install_free_space_veto,
)
from src.model.obs_spec import VECTOR61

ENV_WATCH_HERO = "SNAKE_SERVE_VETO_WATCH_HERO"
ENV_PLAY_AI = "SNAKE_SERVE_VETO_PLAY_AI"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})
# Released default for the Watch hero when its env var is unset (rollback: set it to "0").
WATCH_HERO_RELEASED_DEFAULT = True

SCOPE_WATCH_HERO = "watch_hero"
SCOPE_PLAY_AI = "play_ai"

# The strict-gate receipt this wrapper's evidence is bound to (run-v3 STRICT_PASS).
STRICT_RECEIPT_CHECKPOINT_SHA256 = (
    "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93"
)
STRICT_RECEIPT_WRAPPER_SOURCE_SHA256 = (
    "1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428"
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

LOG_PREFIX = "safety-veto-serving:"
logger = logging.getLogger(__name__)


def _truthy(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() in _TRUTHY


def _released_flag(value: Optional[str], default: bool) -> bool:
    """Unset or unrecognized -> ``default``; an explicit truthy/falsy word wins."""
    if value is None:
        return default
    word = value.strip().lower()
    if word in _TRUTHY:
        return True
    if word in _FALSY:
        return False
    return default


@dataclass(frozen=True)
class ServingVetoFlags:
    """Which served snakes get the wrapper. Both default to off."""

    watch_hero: bool = False
    play_ai: bool = False

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "ServingVetoFlags":
        """Read the two flags from ``environ`` (``os.environ`` when None)."""
        env = os.environ if environ is None else environ
        return cls(
            watch_hero=_released_flag(env.get(ENV_WATCH_HERO), WATCH_HERO_RELEASED_DEFAULT),
            play_ai=_truthy(env.get(ENV_PLAY_AI)),
        )

    def to_dict(self) -> Dict[str, bool]:
        return {"watch_hero": bool(self.watch_hero), "play_ai": bool(self.play_ai)}


def wrapper_identity() -> Dict[str, Any]:
    """Method, static descriptor and source sha256 of the installed wrapper.

    Same shape as ``strict_run.wrapper_identity`` (the sha256 is of the source
    file bytes), except ``source_path`` is repo-relative.
    """
    path = os.path.abspath(_veto_module.__file__)
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return {
        "method": VETO_METHOD,
        "descriptor": FreeSpaceVeto().descriptor(),
        "source_path": os.path.relpath(path, REPO_ROOT),
        "source_sha256": digest,
    }


@dataclass
class ServingVetoState:
    """What the last session build installed (or why it installed nothing)."""

    flags: ServingVetoFlags
    mode: str
    obs_spec: str
    checkpoint_sha256: Optional[str]
    scope: Optional[str] = None
    reason: Optional[str] = None
    wrapper: Optional[Dict[str, Any]] = None
    vetoes: Dict[int, FreeSpaceVeto] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return bool(self.vetoes)

    @property
    def wrapped_snake_ids(self) -> List[int]:
        return sorted(self.vetoes)

    def counters(self) -> Dict[int, Dict[str, int]]:
        """Current cumulative counters per wrapped snake id."""
        return {sid: veto.counters.to_dict() for sid, veto in sorted(self.vetoes.items())}

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe report (snake ids become strings)."""
        return {
            "active": self.active,
            "flags": self.flags.to_dict(),
            "mode": self.mode,
            "obs_spec": self.obs_spec,
            "scope": self.scope,
            "reason": self.reason,
            "checkpoint_sha256": self.checkpoint_sha256,
            "strict_receipt_checkpoint_match": (
                self.checkpoint_sha256 == STRICT_RECEIPT_CHECKPOINT_SHA256
            ),
            "wrapper": self.wrapper,
            "wrapped_snake_ids": self.wrapped_snake_ids,
            "counters": {str(k): v for k, v in self.counters().items()},
        }


def install_serving_vetoes(
    game: Any,
    policy: Any,
    mode: str,
    obs_spec: str,
    checkpoint_sha256: Optional[str],
    flags: ServingVetoFlags,
) -> ServingVetoState:
    """Install the wrapper on the snakes ``flags`` select for this build.

    Args:
        game: The freshly built GameState.
        policy: The session's shared policy.
        mode: The session mode the build resolved to (watch/train/play).
        obs_spec: The served checkpoint's observation spec.
        checkpoint_sha256: sha256 of the served checkpoint bytes (None = untrained).
        flags: The operator's flags for this build.

    Returns:
        The build's :class:`ServingVetoState`; inactive (no snake touched) unless a
        flag for ``mode`` is on, the served policy is a vector61 Apex policy, and
        the checkpoint and wrapper source match the strict receipt (fail closed).
    """
    state = _install(game, policy, mode, obs_spec, checkpoint_sha256, flags)
    log_build(state)
    return state


def _install(
    game: Any,
    policy: Any,
    mode: str,
    obs_spec: str,
    checkpoint_sha256: Optional[str],
    flags: ServingVetoFlags,
) -> ServingVetoState:
    state = ServingVetoState(
        flags=flags, mode=str(mode), obs_spec=str(obs_spec), checkpoint_sha256=checkpoint_sha256
    )
    if mode == "watch" and flags.watch_hero:
        state.scope = SCOPE_WATCH_HERO
    elif mode == "play" and flags.play_ai:
        state.scope = SCOPE_PLAY_AI
    else:
        if flags.watch_hero or flags.play_ai:
            state.reason = f"no serving veto flag applies to {mode} mode"
        return state
    if obs_spec != VECTOR61 or not hasattr(policy, "dqn"):
        state.reason = "the safety veto applies only to a vector61 Apex policy"
        return state
    if checkpoint_sha256 != STRICT_RECEIPT_CHECKPOINT_SHA256:
        state.reason = (
            "no strict-gate evidence for this checkpoint "
            f"(sha256 {checkpoint_sha256 or 'none: untrained weights'})"
        )
        return state
    identity = wrapper_identity()
    if identity["source_sha256"] != STRICT_RECEIPT_WRAPPER_SOURCE_SHA256:
        state.reason = f"wrapper source sha256 {identity['source_sha256']} is not the gated one"
        return state

    from src.game.ai_snake import AISnake
    from src.game.human_snake import HumanSnake

    snakes = list(getattr(game, "snakes", []))
    if state.scope == SCOPE_WATCH_HERO:
        targets = snakes[:1]
    else:
        targets = [s for s in snakes if isinstance(s, AISnake) and not isinstance(s, HumanSnake)]
    targets = [s for s in targets if isinstance(s, AISnake) and s.policy is policy]
    if not targets:
        state.reason = "no served AI snake to wrap"
        return state
    state.wrapper = identity
    for snake in targets:
        state.vetoes[int(snake.id)] = install_free_space_veto(snake)
    return state


def _ensure_visible() -> None:
    """Make the INFO line reach stderr when nothing else configured logging.

    ``web/serve.py`` runs uvicorn at ``log_level="warning"`` and configures no
    root handler, so an INFO record would otherwise be dropped. A handler is
    attached to this module's logger only when no handler exists on its path.
    """
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)


def log_build(state: ServingVetoState) -> None:
    """One INFO line per build when a flag is requested (nothing when both are off)."""
    if not (state.flags.watch_hero or state.flags.play_ai):
        return
    _ensure_visible()
    wrapper = state.wrapper or {}
    logger.info(
        "%s active=%s scope=%s mode=%s wrapped_ids=%s flags=%s checkpoint_sha256=%s "
        "strict_checkpoint_match=%s wrapper_source_sha256=%s reason=%s",
        LOG_PREFIX,
        state.active,
        state.scope,
        state.mode,
        state.wrapped_snake_ids,
        state.flags.to_dict(),
        state.checkpoint_sha256,
        state.checkpoint_sha256 == STRICT_RECEIPT_CHECKPOINT_SHA256,
        wrapper.get("source_sha256"),
        state.reason,
    )
