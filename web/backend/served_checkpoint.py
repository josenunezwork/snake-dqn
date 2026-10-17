"""Pinned registry for the checkpoint the web app serves by default (pin-gated swap, fail closed).

``GameSession()`` built without an explicit checkpoint (what ``web/backend/app.py`` does)
asks :func:`resolve_served_checkpoint` which checkpoint to serve. The operator chooses by
name with ``SNAKE_SERVE_CHECKPOINT``; only names in :data:`REGISTRY` exist:

* ``champion`` (the released default until 2026-10-06):
  ``saved_snakes/champion_a5_freespace_20260621.pth``, served exactly as before this module
  existed (same path, same no-checkpoint fallback to untrained weights). Rollback target:
  ``SNAKE_SERVE_CHECKPOINT=champion``.
* ``frp3-s12`` (the previous released default, 2026-10-06 through 2026-10-17;
  with no checkpoint named, ``SNAKE_SERVE_VETO_VARIANT=v7/v5/v2`` serves the champion
  instead, so those veto rollbacks keep their pre-swap meaning): FRP-v3 arm M3 seed 12 at
  60000 learner updates
  (``apex_mark_u60000.pth``, sha256 :data:`FRP3_S12_SHA256`), served with the released v8
  veto (lambda 8) unchanged. It is served only when ALL of these hold, otherwise the build
  falls back to the released default and says why (fail closed, never raised):

  - its strict-gate receipt pin in :data:`PINS_PATH` is filled (the shipped file holds a
    ``null`` placeholder, so the name is refused until the owner records the strict
    receipt after the gate passes, with ``research/frp3_checkpoint_serving_20261005/
    fill_strict_pin.py``);
  - the pin binds this checkpoint's sha256 and the v8 method the swap was gated with;
  - a file exists at ``saved_snakes/<filename>`` or, failing that, at the pinned artifact
    path, and its bytes hash to the pinned sha256.

* ``b8-s3`` is the T7-selected raster checkpoint and released default since 2026-10-17
  (:data:`CHECKPOINT_RELEASED_DEFAULT`). Its independent T7 decision/audit/review pin
  uses ``veto_variant: none``;
  raster observations cannot use the vector61 serving veto. If its pin or bytes fail, serving
  tries verified ``frp3-s12`` and then the champion.

Unset or blank selects :data:`CHECKPOINT_RELEASED_DEFAULT`. An unknown name falls back to
the released default with a ``reason`` (a typo cannot take the server down or silently
serve something unpinned). The env var is read on every ``GameSession()`` construction;
mode switches and resets rebuild from the session's resolved path, so a session keeps one
checkpoint for its lifetime. An explicit ``GameSession(checkpoint=...)`` and the client's
``load_checkpoint`` control bypass this module entirely.

The strict pin also extends the v8 serving veto's checkpoint binding
(``safety_veto_serving.ServingVetoState.checkpoint_match``): under variant ``v8`` the Watch
hero is wrapped for the champion (the v8 receipt's checkpoint, unchanged) or for any
registry checkpoint whose filled pin names v8 (:func:`v8_gated_checkpoint_sha256s`). Every
other variant keeps its own receipt's champion-only binding, so serving ``frp3-s12`` under
a v7/v5/v2 rollback leaves the hero unwrapped with a ``reason``.

Each resolution logs one INFO line (prefix :data:`LOG_PREFIX`) when the env var is set, a
request was refused, or a non-champion checkpoint is served, so an operator can confirm a
swap or rollback on stderr.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVED_DIR = os.path.join(REPO_ROOT, "saved_snakes")
PINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "served_checkpoint_pins.json")
PINS_SCHEMA = "served-checkpoint-strict-pins/v1"

ENV_CHECKPOINT = "SNAKE_SERVE_CHECKPOINT"
ENV_WATCH_ROSTER = "SNAKE_SERVE_WATCH_ROSTER"
NAME_CHAMPION = "champion"
NAME_FRP3_S12 = "frp3-s12"
NAME_B8_S3 = "b8-s3"
# Released default when ENV_CHECKPOINT is unset or blank.
# b8-s3 released 2026-10-17 after T7 STRICT_PASS + Tier 2 and T12/T12B serving checks.
# Rollback (restart, no code change): SNAKE_SERVE_CHECKPOINT=frp3-s12 or =champion.
CHECKPOINT_RELEASED_DEFAULT = NAME_B8_S3

# Watch roster selection is deliberately separate from checkpoint selection.  The
# released Watch default remains the homogeneous B8 roster until its own serving
# qualification is complete.  "mixed" is an opt-in qualification posture.
WATCH_ROSTER_ALL_B8 = "all_b8"
WATCH_ROSTER_MIXED = "mixed"
WATCH_ROSTERS = (WATCH_ROSTER_ALL_B8, WATCH_ROSTER_MIXED)
# Kept separate from the candidate constant so a release erratum can change one
# reviewed line without touching checkpoint selection or the mixed implementation.
WATCH_ROSTER_RELEASED_DEFAULT = WATCH_ROSTER_ALL_B8

# The veto-variant env key (read here only to keep the pre-swap veto rollbacks meaningful;
# web/backend/safety_veto_serving.py owns the variant itself). The previous frp3-s12 default
# was gated and qualified with v8 alone; preserve that rollback when requesting an older variant
# (SNAKE_SERVE_VETO_VARIANT=v7/v5/v2) with no checkpoint named, the released default yields
# to the champion: "v7" keeps meaning the pre-swap champion + v7 configuration.
ENV_VETO_VARIANT = "SNAKE_SERVE_VETO_VARIANT"
PRE_SWAP_VETO_VARIANTS = ("v7", "v5", "v2")

CHAMPION_FILENAME = "champion_a5_freespace_20260621.pth"
CHAMPION_SHA256 = "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93"
FRP3_S12_FILENAME = "frp3_m3_s12_u60000_20261005.pth"
FRP3_S12_SHA256 = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
FRP3_S12_ARTIFACT_PATH = (
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/frp-v3-20261005/train/arm-M3/seed-12/"
    "checkpoints/apex_mark_u60000.pth"
)
B8_S3_FILENAME = "b8_s3_t7_20261016.pth"
B8_S3_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
B8_S3_ARTIFACT_PATH = (
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m4b-mac-20261013/" "B8_s3/final.pth"
)
# The veto the swap is gated and served with: the released v8 (lambda 8), unchanged.
V8_VARIANT = "v8"
V8_METHOD = "free-space-veto/v8-space-and-head(lambda=8.0)"

LOG_PREFIX = "served-checkpoint:"
logger = logging.getLogger(__name__)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
# The three strict artifacts a filled pin binds (each {"path": <rel to root>, "sha256": hex}).
STRICT_FILES = ("receipt", "intent", "audit_report")
T7_PIN_FILES = ("decision", "audit_report", "independent_review")


@dataclass(frozen=True)
class PinnedCheckpoint:
    """One servable checkpoint: where it lives, its bytes, and whether it needs a strict pin."""

    name: str
    filename: str
    sha256: str
    artifact_path: Optional[str] = None
    # True: served only with a filled strict-receipt pin in PINS_PATH (fail closed).
    requires_strict_pin: bool = True
    veto_variant: Optional[str] = V8_VARIANT
    veto_method: Optional[str] = V8_METHOD
    strict_files: tuple = STRICT_FILES


REGISTRY: Dict[str, PinnedCheckpoint] = {
    NAME_CHAMPION: PinnedCheckpoint(
        NAME_CHAMPION, CHAMPION_FILENAME, CHAMPION_SHA256, requires_strict_pin=False
    ),
    NAME_FRP3_S12: PinnedCheckpoint(
        NAME_FRP3_S12, FRP3_S12_FILENAME, FRP3_S12_SHA256, artifact_path=FRP3_S12_ARTIFACT_PATH
    ),
    NAME_B8_S3: PinnedCheckpoint(
        NAME_B8_S3,
        B8_S3_FILENAME,
        B8_S3_SHA256,
        artifact_path=B8_S3_ARTIFACT_PATH,
        veto_variant="none",
        veto_method="none",
        strict_files=T7_PIN_FILES,
    ),
}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pins(path: Optional[str] = None) -> Dict[str, Any]:
    """The pins document (``{}`` entries when missing or unreadable; never raises)."""
    try:
        with open(path or PINS_PATH, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError):
        return {"schema": None, "checkpoints": {}}
    if not isinstance(doc, dict) or not isinstance(doc.get("checkpoints"), dict):
        return {"schema": None, "checkpoints": {}}
    return doc


def strict_pin_problems(name: str, pins: Optional[Mapping[str, Any]] = None) -> List[str]:
    """Why ``name``'s strict-receipt pin does not authorize serving it ([] = filled and valid).

    A valid pin: the pins document has the expected schema; the entry names the registry
    checkpoint's sha256 and that checkpoint's registered veto binding; ``strict_receipt`` is
    a mapping with a ``verdict`` of ``STRICT_PASS``, an absolute ``root`` and, for the entry's
    evidence files, a relative ``path`` and a 64-hex ``sha256``. ``null`` (the shipped
    placeholder) is refused.
    """
    entry = REGISTRY.get(name)
    if entry is None:
        return [f"unknown checkpoint name {name!r}"]
    doc = load_pins() if pins is None else pins
    if doc.get("schema") != PINS_SCHEMA:
        return [f"pins file schema is not {PINS_SCHEMA!r}"]
    pin = (doc.get("checkpoints") or {}).get(name)
    if not isinstance(pin, Mapping):
        return [f"no pin entry for {name!r}"]
    out = []
    if pin.get("checkpoint_sha256") != entry.sha256:
        out.append("pin checkpoint_sha256 is not the registry sha256")
    if pin.get("veto_variant") != entry.veto_variant or pin.get("veto_method") != entry.veto_method:
        out.append(f"pin veto is not {entry.veto_variant} {entry.veto_method!r}")
    receipt = pin.get("strict_receipt")
    if receipt is None:
        out.append("strict receipt pin is the unfilled placeholder (null)")
        return out
    if not isinstance(receipt, Mapping):
        return out + ["strict_receipt is not a mapping"]
    if receipt.get("verdict") != "STRICT_PASS":
        out.append("strict_receipt.verdict is not STRICT_PASS")
    root = receipt.get("root")
    if not isinstance(root, str) or not os.path.isabs(root):
        out.append("strict_receipt.root is not an absolute path")
    for key in entry.strict_files:
        item = receipt.get(key)
        if not (
            isinstance(item, Mapping)
            and isinstance(item.get("path"), str)
            and item["path"]
            and not os.path.isabs(item["path"])
            and isinstance(item.get("sha256"), str)
            and _HEX64.match(item["sha256"])
        ):
            out.append(f"strict_receipt.{key} needs a relative path and a 64-hex sha256")
    return out


def v8_gated_checkpoint_sha256s(pins: Optional[Mapping[str, Any]] = None) -> List[str]:
    """sha256s of registry checkpoints (other than the champion) whose filled strict pin was
    gated with the v8 veto; the v8 serving veto may wrap these besides the champion."""
    doc = load_pins() if pins is None else pins
    return sorted(
        entry.sha256
        for name, entry in REGISTRY.items()
        if entry.requires_strict_pin
        and entry.veto_variant == V8_VARIANT
        and not strict_pin_problems(name, doc)
    )


@dataclass
class ServedCheckpointChoice:
    """What :func:`resolve_served_checkpoint` decided for one session construction."""

    requested: Optional[str]
    name: Optional[str] = None
    path: Optional[str] = None
    sha256: Optional[str] = None
    reason: Optional[str] = None
    pin_problems: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "reason": self.reason,
            "pin_problems": list(self.pin_problems),
        }


def _candidate_paths(entry: PinnedCheckpoint, saved_dir: str) -> List[str]:
    out = [os.path.join(saved_dir, entry.filename)]
    if entry.artifact_path:
        out.append(entry.artifact_path)
    return out


def _resolve_pinned(
    entry: PinnedCheckpoint, saved_dir: str, pins: Mapping[str, Any]
) -> ServedCheckpointChoice:
    """Serve ``entry`` (strict pin + bytes verified) or say why not (never raises)."""
    choice = ServedCheckpointChoice(requested=entry.name)
    choice.pin_problems = strict_pin_problems(entry.name, pins)
    if choice.pin_problems:
        choice.reason = f"{entry.name} has no valid strict-receipt pin: {choice.pin_problems}"
        return choice
    for path in _candidate_paths(entry, saved_dir):
        if not os.path.isfile(path):
            continue
        try:
            digest = _sha256_file(path)
        except OSError as exc:  # never raise into GameSession(): fail closed
            choice.reason = f"{entry.name} at {path} is unreadable: {exc}"
            return choice
        if digest != entry.sha256:
            choice.reason = f"{entry.name} at {path} has sha256 {digest}, not the pinned one"
            return choice
        choice.name, choice.path, choice.sha256 = entry.name, path, digest
        return choice
    choice.reason = f"{entry.name} file not found at {_candidate_paths(entry, saved_dir)}"
    return choice


def resolve_mixed_watch_opponent(
    saved_dir: Optional[str] = None, pins: Optional[Mapping[str, Any]] = None
) -> ServedCheckpointChoice:
    """Resolve the FRP3 opponent required by the opt-in mixed Watch roster.

    This deliberately uses the same strict-pin and byte verification as a named
    FRP3 serving selection.  A missing or mismatched pin is a roster failure,
    never a reason to construct a partially mixed arena.
    """
    saved = SAVED_DIR if saved_dir is None else saved_dir
    doc = load_pins() if pins is None else pins
    return _resolve_pinned(REGISTRY[NAME_FRP3_S12], saved, doc)


def _champion_choice(path: str) -> ServedCheckpointChoice:
    """The pre-registry behavior: the champion path if the file exists, else None."""
    if os.path.exists(path):
        return ServedCheckpointChoice(requested=None, name=NAME_CHAMPION, path=path)
    return ServedCheckpointChoice(requested=None)


def _requested_pre_swap_variant(env: Mapping[str, str]) -> Optional[str]:
    """The normalized SNAKE_SERVE_VETO_VARIANT when it names a pre-swap variant, else None."""
    raw = env.get(ENV_VETO_VARIANT)
    word = raw.strip().lower() if raw is not None else ""
    return word if word in PRE_SWAP_VETO_VARIANTS else None


def _released_default(
    saved_dir: str,
    champion_path: str,
    pins: Mapping[str, Any],
    env: Optional[Mapping[str, str]] = None,
) -> ServedCheckpointChoice:
    if CHECKPOINT_RELEASED_DEFAULT == NAME_CHAMPION:
        return _champion_choice(champion_path)
    variant = _requested_pre_swap_variant({} if env is None else env)
    if variant is not None:
        # An older variant is a veto rollback inherited from the frp3 release,
        # which keeps its pre-swap meaning (the champion that variant's receipts bind).
        fallback = _champion_choice(champion_path)
        fallback.reason = (
            f"{ENV_VETO_VARIANT}={variant} is a pre-swap veto rollback; the released default "
            f"{CHECKPOINT_RELEASED_DEFAULT} yields to the champion for that variant"
        )
        return fallback
    # After a release flip: the flipped default is itself pin-gated; fail closed to the
    # qualified fallback chain when its pin or bytes do not verify.
    choice = _resolve_pinned(REGISTRY[CHECKPOINT_RELEASED_DEFAULT], saved_dir, pins)
    if choice.path is None:
        if CHECKPOINT_RELEASED_DEFAULT == NAME_B8_S3:
            fallback = _fallback_after_refusal(NAME_B8_S3, saved_dir, champion_path, pins)
            fallback.reason = (
                f"released default {choice.reason}; {fallback.reason or 'served frp3-s12'}"
            )
            fallback.pin_problems = choice.pin_problems
            return fallback
        fallback = _champion_choice(champion_path)
        fallback.reason = f"released default {choice.reason}; serving the champion"
        fallback.pin_problems = choice.pin_problems
        return fallback
    return choice


def _fallback_after_refusal(
    requested: str, saved_dir: str, champion_path: str, pins: Mapping[str, Any]
) -> ServedCheckpointChoice:
    """Fail closed in registry order, without changing the released default.

    B8 is the released raster checkpoint. Its failure must first use the previously
    released, independently pinned FRP3 checkpoint; only an FRP3 verification failure reaches
    the champion. This deliberately ignores veto rollback env state: checkpoint resolution is
    deterministic and the veto layer subsequently refuses raster observations on its own guard.
    """
    if requested == NAME_B8_S3:
        frp = _resolve_pinned(REGISTRY[NAME_FRP3_S12], saved_dir, pins)
        if frp.path is not None:
            return frp
        champion = _champion_choice(champion_path)
        champion.reason = f"b8-s3 fallback frp3-s12 failed: {frp.reason}; serving the champion"
        champion.pin_problems = frp.pin_problems
        return champion
    return _released_default(saved_dir, champion_path, pins)


def resolve_served_checkpoint(
    environ: Optional[Mapping[str, str]] = None,
    saved_dir: Optional[str] = None,
    pins: Optional[Mapping[str, Any]] = None,
    champion_path: Optional[str] = None,
) -> ServedCheckpointChoice:
    """Which checkpoint a default-constructed ``GameSession`` serves (never raises).

    Args:
        environ: Environment to read :data:`ENV_CHECKPOINT` from (``os.environ`` when None).
        saved_dir: The saved-snakes directory (the repo's when None).
        pins: The pins document (read from :data:`PINS_PATH` when None).
        champion_path: The champion file (``saved_dir``/:data:`CHAMPION_FILENAME` when
            None; ``GameSession`` passes ``session.DEFAULT_CHECKPOINT``).

    Returns:
        The choice. ``path`` None means no checkpoint (untrained weights), exactly as
        before this module when the champion file is absent. ``requested`` is the
        normalized env word (None when unset or blank) and ``reason`` says why a request
        was not honored.
    """
    env = os.environ if environ is None else environ
    saved = SAVED_DIR if saved_dir is None else saved_dir
    doc = load_pins() if pins is None else pins
    champion = os.path.join(saved, CHAMPION_FILENAME) if champion_path is None else champion_path
    raw = env.get(ENV_CHECKPOINT)
    requested = raw.strip().lower() if raw is not None and raw.strip() else None
    if requested is None:
        choice = _released_default(saved, champion, doc, env)
    elif requested not in REGISTRY:
        choice = _released_default(saved, champion, doc, env)
        notice = (
            f"unknown {ENV_CHECKPOINT} {raw!r} (must be one of {', '.join(sorted(REGISTRY))}); "
            f"fell back to the released default {CHECKPOINT_RELEASED_DEFAULT}"
        )
        choice.reason = notice if choice.reason is None else f"{notice}; {choice.reason}"
    elif not REGISTRY[requested].requires_strict_pin:  # the champion: served as before
        choice = _champion_choice(champion)
    else:
        choice = _resolve_pinned(REGISTRY[requested], saved, doc)
        if choice.path is None:
            fallback = _fallback_after_refusal(requested, saved, champion, doc)
            reason = choice.reason
            if fallback.reason:
                reason = f"{reason}; {fallback.reason}"
            fallback.reason = f"{reason}; served the released default instead"
            fallback.pin_problems = choice.pin_problems
            choice = fallback
    choice.requested = requested
    log_choice(choice)
    return choice


def _ensure_visible() -> None:
    """Make the INFO line reach stderr when nothing else configured logging (as the veto
    serving module does: ``web/serve.py`` runs uvicorn at warning level)."""
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)


def log_choice(choice: ServedCheckpointChoice) -> None:
    """One INFO line when the env var was set, a request was refused, or a non-champion
    checkpoint is served (so a released swap is visible on stderr with the env unset)."""
    if choice.requested is None and choice.reason is None and choice.name in (NAME_CHAMPION, None):
        return
    _ensure_visible()
    logger.info(
        "%s name=%s requested=%s sha256=%s path=%s reason=%s",
        LOG_PREFIX,
        choice.name,
        choice.requested,
        choice.sha256,
        choice.path,
        choice.reason,
    )
