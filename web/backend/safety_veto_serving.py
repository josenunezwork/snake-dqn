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

Variant (``SNAKE_SERVE_VETO_VARIANT``, Watch hero only). ``SNAKE_SERVE_VETO_WATCH_HERO``
stays the master switch; the variant chooses which wrapper the Watch hero gets:

* ``v2`` (the default while :data:`VARIANT_RELEASED_DEFAULT` is ``"v2"``): the released
  ``free-space-veto/v2-speed-preserving`` wrapper above, installed exactly as before.
* ``v5``: ``free-space-veto/v5-boost-aware``
  (:func:`src.evaluation.safety_veto_v5.install_boost_aware_veto`), bound fail closed to
  the v5 STRICT_PASS receipt (``apex-veto-v5-strict-20261001/run-v1``, receipt sha256
  :data:`V5_STRICT_RECEIPT_SHA256`): the served checkpoint must hash to
  :data:`V5_STRICT_RECEIPT_CHECKPOINT_SHA256` and ``safety_veto_v5.py``,
  ``safety_veto.py`` and ``safety_veto_v3.py`` to :data:`V5_STRICT_RECEIPT_SOURCE_SHA256S`.

Unset or blank selects the default; ``v2``/``v5`` (case-insensitive) select a variant.
Any other value falls back to :data:`VARIANT_RELEASED_DEFAULT`, like an unrecognized
master-switch word falls back to the released default, so a typo cannot silently remove
the released Watch-hero veto; the build's ``reason`` and log line name the bad value. The
variant never affects Play: ``SNAKE_SERVE_VETO_PLAY_AI`` keeps its v2 behavior.

Whenever a flag is requested, each build logs one INFO line (prefix
:data:`LOG_PREFIX`) with the scope, wrapped ids, checkpoint and source match flags, the
wrapper source sha256, the variant and the reason (if any), so an operator can confirm a
release or rollback on the running server. The line starts with the released v2 fields in
their released order (``active=... scope=... mode=... wrapped_ids=... flags=...
checkpoint_sha256=... strict_checkpoint_match=... wrapper_source_sha256=...``); the variant
fields follow, and ``reason=`` stays last. The wire payload is unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from src.evaluation import safety_veto as _veto_module
from src.evaluation import safety_veto_v3 as _veto_v3_module
from src.evaluation import safety_veto_v5 as _veto_v5_module
from src.evaluation.safety_veto import (
    VETO_METHOD,
    FreeSpaceVeto,
    install_free_space_veto,
)
from src.evaluation.safety_veto_v5 import (
    VETO_METHOD_V5,
    BoostAwareFreeSpaceVeto,
    install_boost_aware_veto,
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

# Variant selector for the Watch hero's wrapper (SNAKE_SERVE_VETO_WATCH_HERO stays the switch).
ENV_VARIANT = "SNAKE_SERVE_VETO_VARIANT"
VARIANT_V2 = "v2"
VARIANT_V5 = "v5"
VARIANTS = (VARIANT_V2, VARIANT_V5)
# Released default variant when ENV_VARIANT is unset or blank (the v5 release flips this).
# Released 2026-10-02 (v5 STRICT_PASS + SERVING_PASS); rollback: SNAKE_SERVE_VETO_VARIANT=v2.
VARIANT_RELEASED_DEFAULT = VARIANT_V5

# The v5 STRICT_PASS receipt (apex-veto-v5-strict-20261001/run-v1) and the candidate
# identity its intent.json binds: champion bytes and the three veto source files.
V5_STRICT_RECEIPT_SHA256 = "cd843edfcfaf07158fb5135ce247735d3f63c10a24568cf0f47a24b76f4e3ceb"
V5_STRICT_RECEIPT_CHECKPOINT_SHA256 = (
    "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93"
)
V5_STRICT_RECEIPT_SOURCE_SHA256S = {
    "src/evaluation/safety_veto.py": (
        "1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428"
    ),
    "src/evaluation/safety_veto_v3.py": (
        "ed3a6d860b09afd982bc6c87ea0a86566455dfcf9562133d1772e595b4bb5be1"
    ),
    "src/evaluation/safety_veto_v5.py": (
        "d86d084e7778fc514c4932b27f3750f5f11543e3c7afa44571869407870ec86c"
    ),
}
V5_SOURCE_PATH = "src/evaluation/safety_veto_v5.py"

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


def _variant(value: Optional[str]) -> str:
    """Unset or blank -> the released default; otherwise the normalized word (may be unknown;
    :func:`_install` then falls back to the released default and says so)."""
    if value is None or not value.strip():
        return VARIANT_RELEASED_DEFAULT
    return value.strip().lower()


@dataclass(frozen=True)
class ServingVetoFlags:
    """Which served snakes get the wrapper (both default to off), and the Watch variant.

    ``variant`` only chooses the Watch hero's wrapper; :meth:`to_dict` keeps the two
    flags' original shape, and the variant is reported beside it.
    """

    watch_hero: bool = False
    play_ai: bool = False
    variant: str = VARIANT_RELEASED_DEFAULT

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "ServingVetoFlags":
        """Read the two flags and the variant from ``environ`` (``os.environ`` when None)."""
        env = os.environ if environ is None else environ
        return cls(
            watch_hero=_released_flag(env.get(ENV_WATCH_HERO), WATCH_HERO_RELEASED_DEFAULT),
            play_ai=_truthy(env.get(ENV_PLAY_AI)),
            variant=_variant(env.get(ENV_VARIANT)),
        )

    def to_dict(self) -> Dict[str, bool]:
        return {"watch_hero": bool(self.watch_hero), "play_ai": bool(self.play_ai)}


def _sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def v5_source_sha256s() -> Dict[str, str]:
    """sha256 of each source file the v5 receipt binds, keyed by repo-relative path."""
    out = {}
    for module in (_veto_module, _veto_v3_module, _veto_v5_module):
        path = os.path.abspath(module.__file__)
        out[os.path.relpath(path, REPO_ROOT)] = _sha256_file(path)
    return dict(sorted(out.items()))


def wrapper_identity_v5() -> Dict[str, Any]:
    """Identity of the v5 wrapper, in the v5 strict intent's ``wrapper_identity`` shape.

    ``source_path`` is repo-relative (the intent recorded an absolute path).
    """
    sources = v5_source_sha256s()
    return {
        "method": VETO_METHOD_V5,
        "descriptor": BoostAwareFreeSpaceVeto().descriptor(),
        "source_path": V5_SOURCE_PATH,
        "source_sha256": sources.get(V5_SOURCE_PATH),
        "source_sha256s": sources,
    }


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
    vetoes: Dict[int, Any] = field(default_factory=dict)
    # The wrapper variant this build's scope uses (Watch: flags.variant; Play: always v2).
    variant: Optional[str] = None
    # Whether every bound wrapper source hashed to its receipt sha (None = not checked).
    wrapper_sources_match: Optional[bool] = None

    @property
    def checkpoint_match(self) -> bool:
        """The served checkpoint is the one the variant's strict receipt binds."""
        pinned = (
            V5_STRICT_RECEIPT_CHECKPOINT_SHA256
            if self.variant == VARIANT_V5
            else STRICT_RECEIPT_CHECKPOINT_SHA256
        )
        return self.checkpoint_sha256 == pinned

    @property
    def active(self) -> bool:
        return bool(self.vetoes)

    @property
    def wrapped_snake_ids(self) -> List[int]:
        return sorted(self.vetoes)

    def counters(self) -> Dict[int, Dict[str, int]]:
        """Current cumulative counters per wrapped snake id."""
        return {sid: veto.counters.to_dict() for sid, veto in sorted(self.vetoes.items())}

    def diagnostics(self) -> Dict[int, Dict[str, Any]]:
        """v5 per-snake diagnostics (``diagnostics_record``); empty for the v2 wrapper."""
        return {
            sid: veto.diagnostics_record()
            for sid, veto in sorted(self.vetoes.items())
            if hasattr(veto, "diagnostics_record")
        }

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe report (snake ids become strings)."""
        return {
            "active": self.active,
            "flags": self.flags.to_dict(),
            "variant_requested": self.flags.variant,
            "variant": self.variant,
            "mode": self.mode,
            "obs_spec": self.obs_spec,
            "scope": self.scope,
            "reason": self.reason,
            "checkpoint_sha256": self.checkpoint_sha256,
            "strict_receipt_checkpoint_match": self.checkpoint_match,
            "strict_receipt_wrapper_match": self.wrapper_sources_match,
            "wrapper": self.wrapper,
            "wrapped_snake_ids": self.wrapped_snake_ids,
            "counters": {str(k): v for k, v in self.counters().items()},
            "diagnostics": {str(k): v for k, v in self.diagnostics().items()},
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
    notice = None
    if mode == "watch" and flags.watch_hero:
        state.scope, state.variant = SCOPE_WATCH_HERO, flags.variant
        if state.variant not in VARIANTS:
            # Like an unrecognized master-switch word: fall back to the released default.
            notice = (
                f"unknown serving veto variant {flags.variant!r} "
                f"({ENV_VARIANT} must be one of {', '.join(VARIANTS)}); "
                f"fell back to the released default {VARIANT_RELEASED_DEFAULT}"
            )
            state.variant = VARIANT_RELEASED_DEFAULT
    elif mode == "play" and flags.play_ai:
        state.scope, state.variant = SCOPE_PLAY_AI, VARIANT_V2  # Play never uses the variant
    else:
        if flags.watch_hero or flags.play_ai:
            state.reason = f"no serving veto flag applies to {mode} mode"
        return state
    _install_scoped(state, game, policy, obs_spec, checkpoint_sha256)
    if notice is not None:
        state.reason = notice if state.reason is None else f"{notice}; {state.reason}"
    return state


def _install_scoped(
    state: ServingVetoState,
    game: Any,
    policy: Any,
    obs_spec: str,
    checkpoint_sha256: Optional[str],
) -> None:
    """Install ``state.variant``'s wrapper on ``state.scope``'s snakes (fail closed)."""
    if obs_spec != VECTOR61 or not hasattr(policy, "dqn"):
        state.reason = "the safety veto applies only to a vector61 Apex policy"
        return
    if not state.checkpoint_match:
        under = " under variant v5" if state.variant == VARIANT_V5 else ""  # v2 text unchanged
        state.reason = (
            f"no strict-gate evidence for this checkpoint{under} "
            f"(sha256 {checkpoint_sha256 or 'none: untrained weights'})"
        )
        return
    if state.variant == VARIANT_V5:
        identity = wrapper_identity_v5()
        changed = sorted(
            path
            for path, sha in V5_STRICT_RECEIPT_SOURCE_SHA256S.items()
            if identity["source_sha256s"].get(path) != sha
        )
        state.wrapper_sources_match = not changed
        if changed:
            state.reason = f"v5 wrapper source sha256 differs from the gated one: {changed}"
            return
        install = install_boost_aware_veto
    else:
        identity = wrapper_identity()
        state.wrapper_sources_match = (
            identity["source_sha256"] == STRICT_RECEIPT_WRAPPER_SOURCE_SHA256
        )
        if not state.wrapper_sources_match:
            state.reason = f"wrapper source sha256 {identity['source_sha256']} is not the gated one"
            return
        install = install_free_space_veto

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
        return
    state.wrapper = identity
    for snake in targets:
        state.vetoes[int(snake.id)] = install(snake)


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
        # The released v2 fields keep their released order (operators grep for
        # "active=True scope=watch_hero"); the variant fields follow; reason stays last.
        "%s active=%s scope=%s mode=%s wrapped_ids=%s flags=%s checkpoint_sha256=%s "
        "strict_checkpoint_match=%s wrapper_source_sha256=%s variant=%s variant_requested=%s "
        "wrapper_sources_match=%s wrapper_method=%s reason=%s",
        LOG_PREFIX,
        state.active,
        state.scope,
        state.mode,
        state.wrapped_snake_ids,
        state.flags.to_dict(),
        state.checkpoint_sha256,
        state.checkpoint_match,
        wrapper.get("source_sha256"),
        state.variant,
        state.flags.variant,
        state.wrapper_sources_match,
        wrapper.get("method"),
        state.reason,
    )
