"""Sealed ordinary-desktop identity admission for ``classified-desktop-v1``.

This module deliberately does not inspect processes or execute signature tools
from ``classify``.  Expensive code-identity checks and PID/birth bindings are
made by the operator before a scored observation and cached in this object.
New PIDs require off-tick admission. An exec, UID transition, or revoked
identity is permanently ineligible for the rest of this registry attempt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Iterable

from research.bl_k1.m1_background import SYSTEM_EXECUTABLE_ROOTS
from research.bl_paced_identity.events import canonical_bytes

POLICY = "classified-desktop-v1"
DEFAULT_REGISTRY = Path(__file__).with_name("DESKTOP_APP_REGISTRY_20261027_02.json")
_WORKLOAD_NAMES = frozenset(
    {"python", "python3", "node", "pytest", "uv", "codex", "claude", "codex-exec", "agy"}
)
_WORKLOAD_TEXT = re.compile(r"(?:^|/)(?:Projects|research)(?:/|$)")
_ELIGIBLE_ROLES = frozenset(
    {
        "ordinary_mail_ui",
        "ordinary_browser_ui",
        "ordinary_browser_renderer",
        "remote_chat_ui",
        "remote_chat_renderer",
    }
)
_UNSET = object()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=True)


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _valid_verified_identity(command: str, identity: Any, reg_dict: Mapping[str, Any]) -> bool:
    """Validate that an exported verified identity matches the sealed registry."""
    if (
        not isinstance(identity, Mapping)
        or not isinstance(reg_dict, Mapping)
        or not isinstance(identity.get("file_identity"), Mapping)
        or not isinstance(identity.get("sha256"), str)
        or len(identity["sha256"]) != 64
    ):
        return False
    matching_role = None
    for app in reg_dict.get("applications", ()):
        if not isinstance(app, Mapping) or app.get("id") != identity.get("entry"):
            continue
        app_bundle = str(app.get("bundle_path", ""))
        for role in app.get("executables", ()):
            if not isinstance(role, Mapping) or role.get("role") != identity.get("role"):
                continue
            rel = role.get("relative_executable_path")
            if isinstance(rel, str) and str(Path(app_bundle) / rel) == command:
                matching_role = (app, role)
                break
    if matching_role is None:
        return False
    app, role = matching_role
    relative_raw = role.get("relative_executable_path")
    if not isinstance(relative_raw, str) or not relative_raw:
        return False
    relative = Path(relative_raw)
    if relative.is_absolute() or ".." in relative.parts:
        return False
    bundle_path = str(app.get("bundle_path", ""))
    if not bundle_path or ".." in Path(bundle_path).parts:
        return False
    expected_executable = str(Path(bundle_path) / relative)
    if (
        command != expected_executable
        or not _under(Path(command), Path(bundle_path))
        or ".." in Path(command).parts
    ):
        return False
    role_sha256 = role.get("sha256")
    if not isinstance(role_sha256, str) or len(role_sha256) != 64:
        return False
    signed = reg_dict.get("signed_identities", {})
    sealed = signed.get(role_sha256, role) if isinstance(signed, Mapping) else role
    if not isinstance(sealed, Mapping):
        return False
    return not (
        identity.get("application") != app.get("display_name")
        or identity.get("role") != role.get("role")
        or identity.get("sha256") != role_sha256
        or identity.get("designated_requirement") != sealed.get("designated_requirement")
        or identity.get("cdhash") != sealed.get("cdhash")
        or identity.get("signing_authorities") != sealed.get("signing_authorities")
        or identity.get("executable") != expected_executable
        or identity.get("bundle_path") != bundle_path
        or identity.get("bundle_id") != app.get("bundle_id")
        or identity.get("version") != app.get("version")
        or identity.get("build") != app.get("build")
        or identity.get("team_identifier") != app.get("team_identifier")
        or role.get("role") not in _ELIGIBLE_ROLES
    )


def _is_workload_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in _WORKLOAD_NAMES or bool(re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", lowered))


def _extract_argv_evidence(row: Mapping[str, Any]) -> tuple[list[str], str, str | None]:
    """Normalize exact argv and capture status from a process row."""
    obs_error = row.get("identity_observation_error")
    if obs_error:
        return [], "unavailable", str(obs_error)
    if "argv" in row:
        raw_argv = row.get("argv")
        if (
            isinstance(raw_argv, (list, tuple))
            and len(raw_argv) > 0
            and all(isinstance(arg, str) and bool(arg) for arg in raw_argv)
        ):
            return [str(arg) for arg in raw_argv], "ok", None
        return [], "unavailable", "argv_identity_unavailable"
    command = row.get("command")
    if isinstance(command, str) and command:
        return [command], "ok", None
    return [], "unavailable", "argv_identity_unavailable"


def _extract_display_argv(
    row: Mapping[str, Any],
    argv_list: list[str] | None = None,
    argv_status: str = "ok",
) -> str:
    """Normalize displayed argument evidence (ps args) from a process row."""
    raw_display = row.get("argv_display") or row.get("args") or row.get("cmdline")
    if isinstance(raw_display, (list, tuple)):
        text = " ".join(str(x).strip() for x in raw_display if x is not None and str(x).strip())
        if text:
            return text
    elif isinstance(raw_display, str) and raw_display.strip():
        return " ".join(raw_display.split())
    if argv_list is None:
        argv_list, argv_status, _ = _extract_argv_evidence(row)
    if argv_status == "ok" and argv_list:
        return " ".join(str(x).strip() for x in argv_list if str(x).strip())
    command = row.get("command")
    if isinstance(command, str) and command.strip():
        return command.strip()
    return ""


def _intrinsic_workload_reason(
    row: Mapping[str, Any], reg_dict: Mapping[str, Any], verified: Mapping[str, Any]
) -> str | None:
    """Pure workload-precedence check before any desktop or system classification."""
    if row.get("known_workload") or row.get("workload"):
        return "known_workload"
    command = row.get("command") if isinstance(row.get("command"), str) else ""
    raw_argv = row.get("argv", ())
    if isinstance(raw_argv, (list, tuple)):
        argv_tokens = [str(x) for x in raw_argv if isinstance(x, str) and x]
        argv_text = " ".join(argv_tokens)
    elif isinstance(raw_argv, str):
        argv_tokens = raw_argv.split()
        argv_text = raw_argv
    else:
        argv_tokens = []
        argv_text = ""

    raw_display = row.get("argv_display") or row.get("args") or row.get("cmdline") or ""
    if isinstance(raw_display, (list, tuple)):
        display_text = " ".join(str(x) for x in raw_display if x is not None)
    elif isinstance(raw_display, str):
        display_text = raw_display
    else:
        display_text = ""

    raw_cwd = row.get("cwd")
    cwd_text = str(raw_cwd) if isinstance(raw_cwd, (str, Path)) and str(raw_cwd) else ""

    ancestor_commands: list[str] = []
    ancestor_texts: list[str] = []
    for anc_key in ("ancestry", "ancestors", "parent_command", "parent_argv"):
        raw_anc = row.get(anc_key)
        if raw_anc is None:
            continue
        items = raw_anc if isinstance(raw_anc, (list, tuple)) else [raw_anc]
        for item in items:
            if isinstance(item, Mapping):
                if item.get("known_workload") or item.get("workload"):
                    return "known_workload"
                anc_cmd = item.get("command")
                if isinstance(anc_cmd, str) and anc_cmd:
                    ancestor_commands.append(anc_cmd)
                    ancestor_texts.append(anc_cmd)
                for sub_key in ("argv", "argv_display", "args", "cmdline", "cwd"):
                    sub_val = item.get(sub_key)
                    if isinstance(sub_val, (list, tuple)):
                        ancestor_texts.extend(str(x) for x in sub_val if x is not None)
                    elif isinstance(sub_val, (str, Path)) and str(sub_val):
                        ancestor_texts.append(str(sub_val))
            elif isinstance(item, (str, Path)) and str(item):
                text_item = str(item)
                ancestor_commands.append(text_item)
                ancestor_texts.append(text_item)

    exclusions = {
        excl.get("exact_executable")
        for excl in reg_dict.get("workload_exclusions", ())
        if isinstance(excl, Mapping) and isinstance(excl.get("exact_executable"), str)
    }
    if command and command in exclusions:
        return "workload_exclusion"
    if any(tok in exclusions for tok in (*argv_tokens, *display_text.split(), *ancestor_commands)):
        return "workload_exclusion"

    if command:
        path = Path(command)
        if ".." not in path.parts:
            verified_gui_main = (
                isinstance(verified, Mapping)
                and command in verified
                and path.name in {"Claude", "Codex"}
            )
            if (_is_workload_name(path.name) and not verified_gui_main) or _WORKLOAD_TEXT.search(
                command
            ):
                return "workload_exclusion"

    for text in (argv_text, display_text, cwd_text, *ancestor_texts):
        if text and _WORKLOAD_TEXT.search(text):
            return "workload_exclusion"

    for anc_cmd in ancestor_commands:
        for tok in anc_cmd.split():
            anc_path = Path(tok)
            if ".." in anc_path.parts:
                continue
            anc_verified = (
                isinstance(verified, Mapping)
                and tok in verified
                and anc_path.name in {"Claude", "Codex"}
            )
            if _is_workload_name(anc_path.name) and not anc_verified:
                return "workload_exclusion"

    enrolled_command = isinstance(verified, Mapping) and bool(command) and command in verified
    extra_tokens: list[str] = list(argv_tokens[1:] if enrolled_command else argv_tokens)
    if display_text:
        rem_display = (
            display_text[len(command) :].strip()
            if command and display_text.startswith(command)
            else display_text
        )
        extra_tokens.extend(rem_display.split())
    for tok in extra_tokens:
        if tok.startswith("-"):
            continue
        tok_path = Path(tok)
        if ".." in tok_path.parts:
            continue
        tok_verified = (
            isinstance(verified, Mapping)
            and tok in verified
            and tok_path.name in {"Claude", "Codex"}
        )
        if _is_workload_name(tok_path.name) and not tok_verified:
            return "workload_exclusion"

    return None


def _make_history(
    *,
    pid: int | None,
    birth: str | None,
    uid: int | None,
    command: str | None,
    argv: list[str] | None = None,
    argv_status: str = "ok",
    argv_display: str | None = None,
    status: str,
    kind: str,
    reason: str,
    revoked: bool,
    revoked_reason: str | None,
    identity_binding_changed: bool,
    binding: dict[str, Any] | None,
    executable_status: str,
    prior_history: Mapping[str, Any] | None = None,
    sequence: int | None = None,
) -> dict[str, Any]:
    resolved_argv = (
        list(argv)
        if isinstance(argv, (list, tuple))
        else ([command] if isinstance(command, str) and command and argv_status == "ok" else [])
    )
    if isinstance(argv_display, str) and argv_display.strip():
        resolved_display = " ".join(argv_display.split())
    elif argv_status == "ok" and resolved_argv:
        resolved_display = " ".join(str(x).strip() for x in resolved_argv if str(x).strip())
    elif isinstance(command, str) and command.strip():
        resolved_display = command.strip()
    else:
        resolved_display = ""
    state: dict[str, Any] = {
        "pid": pid,
        "birth": birth,
        "uid": uid,
        "command": command,
        "argv": resolved_argv,
        "argv_status": str(argv_status),
        "argv_display": resolved_display,
        "status": status,
        "kind": kind,
        "reason": reason,
        "revoked": bool(revoked),
        "revoked_reason": revoked_reason,
        "identity_binding_changed": bool(identity_binding_changed),
        "binding": dict(binding) if isinstance(binding, Mapping) else None,
        "executable_status": executable_status,
    }
    if type(sequence) is int and sequence >= 0:
        seq = sequence
    elif isinstance(prior_history, Mapping):
        prior_seq = (
            prior_history["sequence"]
            if type(prior_history.get("sequence")) is int and prior_history["sequence"] >= 0
            else 0
        )
        prior_comparable = {k: v for k, v in prior_history.items() if k != "sequence"}
        if "argv_display" not in prior_comparable:
            prior_comparable["argv_display"] = _extract_display_argv(
                prior_comparable,
                (
                    [str(x) for x in prior_comparable["argv"]]
                    if isinstance(prior_comparable.get("argv"), (list, tuple))
                    else []
                ),
                str(prior_comparable.get("argv_status", "ok")),
            )
        seq = prior_seq if state == prior_comparable else prior_seq + 1
    else:
        seq = 0
    state["sequence"] = seq
    return state


def decision_from_history(kind: str, history: Mapping[str, Any], reason: str) -> dict[str, Any]:
    """Project a `(kind, new_history, reason)` tuple to the exported decision dict."""
    binding = history.get("binding")
    if (
        kind in {"desktop_app", "desktop_renderer"}
        and history.get("status") == "bound"
        and isinstance(binding, Mapping)
    ):
        return {
            "kind": kind,
            "reason": reason,
            "entry": binding["entry"],
            "application": binding["application"],
            "role": binding["role"],
            "identity": binding["identity"],
        }
    result: dict[str, Any] = {
        "kind": kind,
        "reason": reason,
        "identity_binding_changed": bool(history.get("identity_binding_changed")),
    }
    if reason == "known_workload" or history.get("revoked_reason") == "known_workload":
        result["known_workload"] = True
    return result


def _extract_registry_context(
    registry: DesktopRegistry | Mapping[str, Any],
) -> tuple[str | None, Mapping[str, Any], Mapping[str, Any]]:
    if isinstance(registry, DesktopRegistry):
        reg_policy = registry.registry.get("policy", POLICY)
        return (
            reg_policy if isinstance(reg_policy, str) else None,
            registry.registry,
            registry._verified,
        )
    if not isinstance(registry, Mapping):
        return None, {}, {}
    if isinstance(registry.get("registry"), Mapping):
        reg_dict = registry["registry"]
        policy = registry.get("policy", reg_dict.get("policy"))
        verified = registry.get("verified", {})
        return (
            policy if isinstance(policy, str) else None,
            reg_dict,
            verified if isinstance(verified, Mapping) else {},
        )
    policy = registry.get("policy", POLICY)
    verified = registry.get("verified", {})
    return (
        policy if isinstance(policy, str) else None,
        registry,
        verified if isinstance(verified, Mapping) else {},
    )


def _resolve_executable_evidence(
    row: Mapping[str, Any],
    key: tuple[int, str, int, str],
    reg_dict: Mapping[str, Any],
    verified: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any] | None]:
    _, _, _, command = key
    if ".." in Path(command).parts:
        return "symlink_or_noncanonical_executable", command, None
    identity = verified.get(command) if isinstance(verified, Mapping) else None
    valid_verified = _valid_verified_identity(command, identity, reg_dict)
    classification = row.get("classification")
    recorded = row.get("executable_status")
    if not isinstance(recorded, str) and isinstance(classification, Mapping):
        recorded = classification.get("executable_status")
    if isinstance(recorded, str) and recorded:
        if recorded == "ok":
            if not valid_verified or not isinstance(identity, Mapping):
                return "unenrolled_or_unverified_executable", command, None
            return "ok", command, dict(identity)
        return (
            recorded,
            command,
            dict(identity) if isinstance(identity, Mapping) else None,
        )
    cls_reason = classification.get("reason") if isinstance(classification, Mapping) else None
    if cls_reason in {
        "executable_replaced",
        "symlink_or_noncanonical_executable",
        "missing_or_noncanonical_executable",
        "unenrolled_or_unverified_executable",
    }:
        return (
            str(cls_reason),
            command,
            dict(identity) if isinstance(identity, Mapping) else None,
        )
    if valid_verified and isinstance(identity, Mapping):
        return "ok", command, dict(identity)
    return "unenrolled_or_unverified_executable", command, None


def classify_row(
    history: Mapping[str, Any] | None,
    row: Mapping[str, Any],
    registry: DesktopRegistry | Mapping[str, Any],
) -> tuple[str, dict[str, Any], str]:
    """Pure per-(pid, birth) classifier shared by the live collector and reducer.

    Takes the prior ``history`` for ``(pid, birth)`` as of the moment ``row`` is
    evaluated (or ``None`` if unseen), the process ``row``, and ``registry``
    (either a ``DesktopRegistry`` or an exported registry mapping), and returns
    ``(kind, new_history, reason)`` without hidden mutable or filesystem state.
    """
    if not isinstance(row, Mapping):
        raise ValueError("process row must be a mapping")
    policy, reg_dict, verified = _extract_registry_context(registry)
    raw_pid = row.get("pid") if type(row.get("pid")) is int else None
    raw_birth = row.get("birth") if isinstance(row.get("birth"), str) else None
    raw_uid = row.get("uid") if type(row.get("uid")) is int else None
    raw_command = row.get("command") if isinstance(row.get("command"), str) else None
    argv_list, argv_status, obs_error = _extract_argv_evidence(row)
    argv_display = _extract_display_argv(row, argv_list, argv_status)
    prior_map = history if isinstance(history, Mapping) else None
    if policy != POLICY:
        reason = "desktop_registry_policy_mismatch"
        return (
            "unknown",
            _make_history(
                pid=raw_pid,
                birth=raw_birth,
                uid=raw_uid,
                command=raw_command,
                argv=argv_list,
                argv_status=argv_status,
                argv_display=argv_display,
                status="revoked",
                kind="unknown",
                reason=reason,
                revoked=True,
                revoked_reason=reason,
                identity_binding_changed=bool(
                    prior_map is not None and prior_map.get("identity_binding_changed")
                ),
                binding=None,
                executable_status="policy_mismatch",
                prior_history=prior_map,
            ),
            reason,
        )
    classification = row.get("classification")
    if "admit" in row:
        admit = bool(row["admit"])
    elif isinstance(classification, Mapping) and "admit" in classification:
        admit = bool(classification["admit"])
    elif (
        isinstance(classification, Mapping)
        and classification.get("reason") == "process_identity_not_revalidated"
    ):
        admit = False
    else:
        admit = True

    workload_reason = _intrinsic_workload_reason(row, reg_dict, verified)
    try:
        key = DesktopRegistry._key(row)
    except ValueError:
        changed = bool(
            prior_map is not None
            and (
                prior_map.get("identity_binding_changed")
                or (raw_uid, raw_command, argv_list, argv_status, argv_display)
                != (
                    prior_map.get("uid"),
                    prior_map.get("command"),
                    prior_map.get("argv"),
                    prior_map.get("argv_status"),
                    _extract_display_argv(
                        prior_map,
                        (
                            [str(x) for x in prior_map["argv"]]
                            if isinstance(prior_map.get("argv"), (list, tuple))
                            else []
                        ),
                        str(prior_map.get("argv_status", "ok")),
                    ),
                )
            )
        )
        if workload_reason is not None:
            return (
                "agent_workload",
                _make_history(
                    pid=raw_pid,
                    birth=raw_birth,
                    uid=raw_uid,
                    command=raw_command,
                    argv=argv_list,
                    argv_status=argv_status,
                    argv_display=argv_display,
                    status="revoked",
                    kind="agent_workload",
                    reason=workload_reason,
                    revoked=True,
                    revoked_reason=workload_reason,
                    identity_binding_changed=changed,
                    binding=None,
                    executable_status="workload",
                    prior_history=prior_map,
                ),
                workload_reason,
            )
        reason = "desktop_process_identity_missing"
        return (
            "unknown",
            _make_history(
                pid=raw_pid,
                birth=raw_birth,
                uid=raw_uid,
                command=raw_command,
                argv=argv_list,
                argv_status=argv_status,
                argv_display=argv_display,
                status="revoked",
                kind="unknown",
                reason=reason,
                revoked=True,
                revoked_reason=reason,
                identity_binding_changed=changed,
                binding=None,
                executable_status=reason,
                prior_history=prior_map,
            ),
            reason,
        )

    pid, birth, uid, command = key
    if history is not None:
        if not isinstance(history, Mapping):
            raise ValueError("process history must be a mapping or None")
        if history.get("pid") is not None and (
            history.get("pid"),
            history.get("birth"),
        ) != (pid, birth):
            raise ValueError("process history pid/birth does not match row")

    owner_uid = reg_dict.get("owner_uid")
    if workload_reason is not None:
        executable_status = "workload"
        canonical_command = command
        verified_identity = None
    else:
        executable_status, canonical_command, verified_identity = _resolve_executable_evidence(
            row, key, reg_dict, verified
        )

    is_system = (
        workload_reason is None
        and command.startswith(SYSTEM_EXECUTABLE_ROOTS)
        and ".." not in Path(command).parts
        and canonical_command not in verified
    )

    candidate_binding: dict[str, Any] | None = None
    if workload_reason is not None:
        intrinsic_kind = "agent_workload"
        intrinsic_reason = workload_reason
    elif executable_status == "executable_replaced":
        intrinsic_kind = "unknown"
        intrinsic_reason = "executable_replaced"
    elif is_system:
        intrinsic_kind = "macos_service"
        intrinsic_reason = "system_executable_root"
    elif executable_status in {
        "missing_or_noncanonical_executable",
        "symlink_or_noncanonical_executable",
        "desktop_binding_identity_mismatch",
    }:
        intrinsic_kind = "unknown"
        intrinsic_reason = executable_status
    elif uid != owner_uid:
        intrinsic_kind = "unknown"
        intrinsic_reason = "uid_mismatch"
    elif argv_status != "ok":
        intrinsic_kind = "unknown"
        intrinsic_reason = obs_error or "argv_identity_unavailable"
    elif executable_status != "ok" or verified_identity is None:
        intrinsic_kind = "unknown"
        intrinsic_reason = (
            executable_status
            if executable_status != "ok"
            else "unenrolled_or_unverified_executable"
        )
    else:
        role_kind = (
            "desktop_renderer"
            if "renderer" in str(verified_identity.get("role", ""))
            else "desktop_app"
        )
        intrinsic_kind = role_kind
        intrinsic_reason = "enrolled_verified_ui_role"
        candidate_binding = {
            "kind": role_kind,
            "reason": "enrolled_verified_ui_role",
            "entry": verified_identity["entry"],
            "application": verified_identity["application"],
            "role": verified_identity["role"],
            "identity": verified_identity,
        }

    if history is None:
        if intrinsic_kind == "agent_workload":
            return (
                "agent_workload",
                _make_history(
                    pid=pid,
                    birth=birth,
                    uid=uid,
                    command=command,
                    argv=argv_list,
                    argv_status=argv_status,
                    argv_display=argv_display,
                    status="revoked",
                    kind="agent_workload",
                    reason=intrinsic_reason,
                    revoked=True,
                    revoked_reason=intrinsic_reason,
                    identity_binding_changed=False,
                    binding=None,
                    executable_status=executable_status,
                ),
                intrinsic_reason,
            )
        if intrinsic_kind == "macos_service":
            return (
                "macos_service",
                _make_history(
                    pid=pid,
                    birth=birth,
                    uid=uid,
                    command=command,
                    argv=argv_list,
                    argv_status=argv_status,
                    argv_display=argv_display,
                    status="non_desktop",
                    kind="macos_service",
                    reason="system_executable_root",
                    revoked=False,
                    revoked_reason=None,
                    identity_binding_changed=False,
                    binding=None,
                    executable_status=executable_status,
                ),
                "system_executable_root",
            )
        if intrinsic_reason == "executable_replaced":
            return (
                "unknown",
                _make_history(
                    pid=pid,
                    birth=birth,
                    uid=uid,
                    command=command,
                    argv=argv_list,
                    argv_status=argv_status,
                    argv_display=argv_display,
                    status="revoked",
                    kind="unknown",
                    reason="executable_replaced",
                    revoked=True,
                    revoked_reason="executable_replaced",
                    identity_binding_changed=True,
                    binding=None,
                    executable_status=executable_status,
                ),
                "executable_replaced",
            )
        if intrinsic_reason != "enrolled_verified_ui_role":
            return (
                "unknown",
                _make_history(
                    pid=pid,
                    birth=birth,
                    uid=uid,
                    command=command,
                    argv=argv_list,
                    argv_status=argv_status,
                    argv_display=argv_display,
                    status="revoked",
                    kind="unknown",
                    reason=intrinsic_reason,
                    revoked=True,
                    revoked_reason=intrinsic_reason,
                    identity_binding_changed=False,
                    binding=None,
                    executable_status=executable_status,
                ),
                intrinsic_reason,
            )
        if admit and candidate_binding is not None:
            return (
                intrinsic_kind,
                _make_history(
                    pid=pid,
                    birth=birth,
                    uid=uid,
                    command=command,
                    argv=argv_list,
                    argv_status=argv_status,
                    argv_display=argv_display,
                    status="bound",
                    kind=intrinsic_kind,
                    reason="enrolled_verified_ui_role",
                    revoked=False,
                    revoked_reason=None,
                    identity_binding_changed=False,
                    binding=candidate_binding,
                    executable_status=executable_status,
                ),
                "enrolled_verified_ui_role",
            )
        return (
            "unknown",
            _make_history(
                pid=pid,
                birth=birth,
                uid=uid,
                command=command,
                argv=argv_list,
                argv_status=argv_status,
                argv_display=argv_display,
                status="unvalidated",
                kind="unknown",
                reason="process_identity_not_revalidated",
                revoked=False,
                revoked_reason=None,
                identity_binding_changed=False,
                binding=None,
                executable_status=executable_status,
            ),
            "process_identity_not_revalidated",
        )

    prior_status = str(history.get("status", "revoked"))
    prior_uid = history.get("uid")
    prior_command = history.get("command")
    prior_argv_status = (
        str(history["argv_status"]) if isinstance(history.get("argv_status"), str) else "ok"
    )
    prior_argv = (
        [str(x) for x in history["argv"]]
        if isinstance(history.get("argv"), (list, tuple))
        else (
            [prior_command]
            if isinstance(prior_command, str) and prior_command and prior_argv_status == "ok"
            else []
        )
    )
    prior_argv_display = _extract_display_argv(history, prior_argv, prior_argv_status)
    prior_kind = str(history.get("kind", "unknown"))
    prior_revoked_reason = str(history["revoked_reason"]) if history.get("revoked_reason") else None
    prior_changed = bool(history.get("identity_binding_changed"))
    prior_binding = (
        dict(history["binding"]) if isinstance(history.get("binding"), Mapping) else None
    )
    prior_exec_status = (
        str(history["executable_status"]) if history.get("executable_status") else None
    )
    same_argv = (
        argv_list == prior_argv
        and argv_status == prior_argv_status
        and argv_display == prior_argv_display
    ) or (
        is_system
        and prior_status == "non_desktop"
        and (argv_status != "ok" or prior_argv_status != "ok")
        and argv_display == prior_argv_display
    )
    same_identity = uid == prior_uid and command == prior_command and same_argv
    workload_tainted = intrinsic_kind == "agent_workload"

    if not same_identity:
        kind = "agent_workload" if workload_tainted else "unknown"
        if (
            prior_revoked_reason == "executable_replaced"
            or intrinsic_reason == "executable_replaced"
        ) and kind != "agent_workload":
            reason = "executable_replaced"
        elif intrinsic_kind == "agent_workload":
            reason = intrinsic_reason
        elif prior_status in {"bound", "unvalidated"} and intrinsic_reason in {
            "uid_mismatch",
            "missing_or_noncanonical_executable",
            "symlink_or_noncanonical_executable",
            "unenrolled_or_unverified_executable",
            "executable_replaced",
            "argv_identity_unavailable",
            "argv_identity_verification_failed",
        }:
            reason = intrinsic_reason
        else:
            reason = "identity_binding_changed"
        revoked_reason = (
            "executable_replaced"
            if (
                prior_revoked_reason == "executable_replaced"
                or intrinsic_reason == "executable_replaced"
            )
            else reason
        )
        return (
            kind,
            _make_history(
                pid=pid,
                birth=birth,
                uid=uid,
                command=command,
                argv=argv_list,
                argv_status=argv_status,
                argv_display=argv_display,
                status="revoked",
                kind=kind,
                reason=reason,
                revoked=True,
                revoked_reason=revoked_reason,
                identity_binding_changed=True,
                binding=prior_binding,
                executable_status=executable_status,
                prior_history=history,
            ),
            reason,
        )

    if prior_status == "revoked":
        kind = "agent_workload" if workload_tainted else "unknown"
        if (
            prior_revoked_reason == "executable_replaced"
            or intrinsic_reason == "executable_replaced"
        ) and kind != "agent_workload":
            reason = "executable_replaced"
        elif intrinsic_kind == "agent_workload":
            reason = intrinsic_reason
        elif prior_revoked_reason:
            reason = prior_revoked_reason
        else:
            reason = "identity_binding_changed"
        changed = (
            prior_changed
            or (intrinsic_kind == "agent_workload" and prior_kind != "agent_workload")
            or intrinsic_reason in {"executable_replaced", "enrolled_verified_ui_role"}
            or (
                prior_exec_status is not None
                and executable_status != prior_exec_status
                and prior_revoked_reason != "executable_replaced"
            )
        )
        revoked_reason = (
            "executable_replaced"
            if (
                prior_revoked_reason == "executable_replaced"
                or intrinsic_reason == "executable_replaced"
            )
            else reason
        )
        return (
            kind,
            _make_history(
                pid=pid,
                birth=birth,
                uid=uid,
                command=command,
                argv=argv_list,
                argv_status=argv_status,
                argv_display=argv_display,
                status="revoked",
                kind=kind,
                reason=reason,
                revoked=True,
                revoked_reason=revoked_reason,
                identity_binding_changed=changed,
                binding=prior_binding,
                executable_status=executable_status,
                prior_history=history,
            ),
            reason,
        )

    if prior_status == "non_desktop":
        if workload_tainted or intrinsic_kind != "macos_service" or prior_changed:
            kind = "agent_workload" if workload_tainted else "unknown"
            reason = (
                intrinsic_reason
                if workload_tainted
                else (
                    intrinsic_reason
                    if intrinsic_reason != "system_executable_root"
                    else "identity_binding_changed"
                )
            )
            return (
                kind,
                _make_history(
                    pid=pid,
                    birth=birth,
                    uid=uid,
                    command=command,
                    argv=argv_list,
                    argv_status=argv_status,
                    argv_display=argv_display,
                    status="revoked",
                    kind=kind,
                    reason=reason,
                    revoked=True,
                    revoked_reason=reason,
                    identity_binding_changed=True,
                    binding=None,
                    executable_status=executable_status,
                    prior_history=history,
                ),
                reason,
            )
        return (
            "macos_service",
            _make_history(
                pid=pid,
                birth=birth,
                uid=uid,
                command=command,
                argv=argv_list,
                argv_status=argv_status,
                argv_display=argv_display,
                status="non_desktop",
                kind="macos_service",
                reason="system_executable_root",
                revoked=False,
                revoked_reason=None,
                identity_binding_changed=False,
                binding=None,
                executable_status=executable_status,
                prior_history=history,
            ),
            "system_executable_root",
        )

    if prior_status == "unvalidated":
        if intrinsic_reason != "enrolled_verified_ui_role" or candidate_binding is None:
            kind = "agent_workload" if workload_tainted else "unknown"
            reason = intrinsic_reason
            return (
                kind,
                _make_history(
                    pid=pid,
                    birth=birth,
                    uid=uid,
                    command=command,
                    argv=argv_list,
                    argv_status=argv_status,
                    argv_display=argv_display,
                    status="revoked",
                    kind=kind,
                    reason=reason,
                    revoked=True,
                    revoked_reason=reason,
                    identity_binding_changed=True,
                    binding=None,
                    executable_status=executable_status,
                    prior_history=history,
                ),
                reason,
            )
        if admit:
            return (
                intrinsic_kind,
                _make_history(
                    pid=pid,
                    birth=birth,
                    uid=uid,
                    command=command,
                    argv=argv_list,
                    argv_status=argv_status,
                    argv_display=argv_display,
                    status="bound",
                    kind=intrinsic_kind,
                    reason="enrolled_verified_ui_role",
                    revoked=False,
                    revoked_reason=None,
                    identity_binding_changed=False,
                    binding=candidate_binding,
                    executable_status=executable_status,
                    prior_history=history,
                ),
                "enrolled_verified_ui_role",
            )
        return (
            "unknown",
            _make_history(
                pid=pid,
                birth=birth,
                uid=uid,
                command=command,
                argv=argv_list,
                argv_status=argv_status,
                argv_display=argv_display,
                status="unvalidated",
                kind="unknown",
                reason="process_identity_not_revalidated",
                revoked=False,
                revoked_reason=None,
                identity_binding_changed=False,
                binding=None,
                executable_status=executable_status,
                prior_history=history,
            ),
            "process_identity_not_revalidated",
        )

    # prior_status == "bound"
    if (
        intrinsic_reason != "enrolled_verified_ui_role"
        or candidate_binding is None
        or prior_binding is None
        or candidate_binding != prior_binding
    ):
        kind = "agent_workload" if workload_tainted else "unknown"
        reason = (
            intrinsic_reason
            if intrinsic_reason != "enrolled_verified_ui_role"
            else "desktop_binding_identity_mismatch"
        )
        return (
            kind,
            _make_history(
                pid=pid,
                birth=birth,
                uid=uid,
                command=command,
                argv=argv_list,
                argv_status=argv_status,
                argv_display=argv_display,
                status="revoked",
                kind=kind,
                reason=reason,
                revoked=True,
                revoked_reason=reason,
                identity_binding_changed=True,
                binding=prior_binding,
                executable_status=executable_status,
                prior_history=history,
            ),
            reason,
        )
    return (
        str(prior_binding["kind"]),
        _make_history(
            pid=pid,
            birth=birth,
            uid=uid,
            command=command,
            argv=argv_list,
            argv_status=argv_status,
            argv_display=argv_display,
            status="bound",
            kind=str(prior_binding["kind"]),
            reason="enrolled_verified_ui_role",
            revoked=False,
            revoked_reason=None,
            identity_binding_changed=False,
            binding=prior_binding,
            executable_status=executable_status,
            prior_history=history,
        ),
        "enrolled_verified_ui_role",
    )


class DesktopRegistry:
    """A frozen registry plus off-tick code and running-process evidence.

    ``runner`` exists to make verification hermetic in tests.  Production uses
    ``subprocess.run`` and invokes ``codesign`` and ``spctl`` only in
    :meth:`verify_all`; never in the tick-time classifier.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_REGISTRY,
        *,
        uid: int | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        trusted_roots: Iterable[str | Path] | None = None,
        require_seal: bool = True,
    ) -> None:
        self.path = Path(path).resolve(strict=True)
        self.require_seal = require_seal
        if require_seal:
            self._verify_seal(self.path)
        self.uid = os.getuid() if uid is None else uid
        self._runner = runner
        self._trusted_roots = tuple(
            _canonical(root)
            for root in (trusted_roots or ("/Applications", Path.home() / "Applications"))
            if Path(root).exists()
        )
        self.registry = self._load()
        self.permitted_uid = self.registry["owner_uid"]
        if self.permitted_uid != self.uid:
            raise ValueError("desktop registry permitted UID is not this owner UID")
        self._verified: dict[str, dict[str, Any]] = {}
        self._history: dict[tuple[int, str], dict[str, Any]] = {}
        self._history_log: list[dict[str, Any]] = []
        self._last_observation_ns: int | None = None
        self._proc_last_observation_ns: dict[tuple[int, str], int] = {}
        self._bindings: dict[tuple[int, str, int, str], dict[str, Any]] = {}
        # Bindings are evidence for samples already captured.  Revoking an
        # active binding therefore must not erase the snapshot that proves an
        # earlier interval; reducers distinguish it from a later invalid row
        # through that row's explicit identity-binding-change evidence.
        self._binding_history: dict[tuple[int, str, int, str], dict[str, Any]] = {}
        self._revoked_bindings: dict[tuple[int, str, int, str], str] = {}
        self._revoked_processes: dict[tuple[int, str], str] = {}

    @staticmethod
    def _verify_seal(path: Path) -> None:
        marker = path.with_name(path.name + ".complete")
        digest_file = path.with_name(path.name + ".sha256")
        if marker.read_text(encoding="utf-8") != "complete\n":
            raise ValueError("desktop registry completion marker missing")
        fields = digest_file.read_text(encoding="utf-8").strip().split()
        if (
            len(fields) != 2
            or fields[1] != path.name
            or fields[0] != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("desktop registry digest sidecar mismatch")

    def _load(self) -> dict[str, Any]:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._raw_registry = raw
        self._registry_inputs = [
            {"path": self.path.name, "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}
        ]
        base_name = raw.get("extends")
        if base_name:
            if not isinstance(base_name, str) or Path(base_name).name != base_name:
                raise ValueError("desktop registry extension has invalid base name")
            base_path = self.path.with_name(base_name)
            if self.require_seal:
                self._verify_seal(base_path)
            base_bytes = base_path.read_bytes()
            if hashlib.sha256(base_bytes).hexdigest() != raw.get("base_registry_sha256"):
                raise ValueError("desktop registry extension base digest mismatch")
            base = json.loads(base_bytes)
            self._registry_inputs.append(
                {"path": base_path.name, "sha256": hashlib.sha256(base_bytes).hexdigest()}
            )
            raw = {
                **base,
                "applications": [
                    *base.get("applications", []),
                    *raw.get("additional_applications", []),
                ],
                "signed_identities": {
                    **base.get("signed_identities", {}),
                    **raw.get("signed_identities", {}),
                },
                "extension": {"path": base_name, "sha256": raw["base_registry_sha256"]},
            }
        if raw.get("policy") != POLICY or not isinstance(raw.get("applications"), list):
            raise ValueError("desktop registry has wrong policy or applications")
        if type(raw.get("owner_uid")) is not int or raw["owner_uid"] < 0:
            raise ValueError("desktop registry owner UID is malformed")
        if not raw.get("permitted_use_declaration") or not raw.get("workload_exclusions"):
            raise ValueError("desktop registry lacks use declaration or workload exclusions")
        names: set[str] = set()
        for app in raw["applications"]:
            name = app.get("id")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("desktop registry application ids must be unique")
            names.add(name)
            if not isinstance(app.get("executables"), list) or not app["executables"]:
                raise ValueError("desktop registry application has no executable roles")
            for role in app["executables"]:
                if role.get("role") not in _ELIGIBLE_ROLES:
                    raise ValueError("desktop registry has an ineligible executable role")
        return raw

    def _run(self, argv: list[str]) -> str:
        result = self._runner(argv, capture_output=True, text=True, check=False, timeout=30)
        if result.returncode:
            raise ValueError(f"identity tool failed: {argv[0]}")
        return (result.stdout or "") + (result.stderr or "")

    def _entry_paths(self, app: dict[str, Any], role: dict[str, Any]) -> tuple[Path, Path]:
        bundle_text = app.get("bundle_path")
        relative = role.get("relative_executable_path")
        if not isinstance(bundle_text, str) or not isinstance(relative, str):
            raise ValueError("desktop registry executable path is malformed")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("desktop registry executable path traverses")
        bundle_raw = Path(bundle_text).expanduser()
        if ".." in bundle_raw.parts:
            raise ValueError("desktop registry bundle path traverses")
        bundle = _canonical(bundle_raw)
        if bundle_raw != bundle or bundle_raw.is_symlink():
            raise ValueError("desktop registry bundle path is symlinked or noncanonical")
        if not any(_under(bundle, root) for root in self._trusted_roots):
            raise ValueError("desktop registry bundle is outside permitted applications roots")
        executable_raw = bundle / relative
        executable = _canonical(executable_raw)
        if (
            executable_raw != executable
            or executable_raw.is_symlink()
            or not _under(executable, bundle)
        ):
            raise ValueError("desktop registry executable is symlinked or outside bundle")
        return bundle, executable

    @staticmethod
    def _plist(bundle: Path) -> dict[str, Any]:
        import plistlib

        with (bundle / "Contents" / "Info.plist").open("rb") as source:
            return plistlib.load(source)

    def _verify_role(self, app: dict[str, Any], role: dict[str, Any]) -> dict[str, Any]:
        bundle, executable = self._entry_paths(app, role)
        plist = self._plist(bundle)
        for key, plist_key in (
            ("bundle_id", "CFBundleIdentifier"),
            ("version", "CFBundleShortVersionString"),
            ("build", "CFBundleVersion"),
        ):
            if app.get(key) != plist.get(plist_key):
                raise ValueError(f"desktop registry {app['id']} {key} mismatch")
        # Verify the enrolled executable itself.  A recursive bundle check can
        # fail on unrelated Finder metadata in an otherwise signed Chrome
        # installation; that must not substitute for the exact-role check.
        self._run(["/usr/bin/codesign", "--verify", str(executable)])
        signature = self._run(["/usr/bin/codesign", "-dv", "--verbose=4", str(executable)])
        requirement_output = self._run(["/usr/bin/codesign", "-dr", "-", str(executable)])
        requirement = next(
            (line for line in requirement_output.splitlines() if line.startswith("designated => ")),
            "",
        )
        expected_team = app.get("team_identifier")
        if (
            "Signature=adhoc" in signature
            or "TeamIdentifier=not set" in signature
            or f"TeamIdentifier={expected_team}" not in signature
        ):
            raise ValueError(f"desktop registry {app['id']} team identity mismatch")
        sealed = self.registry.get("signed_identities", {}).get(role.get("sha256"), role)
        expected_requirement = sealed.get("designated_requirement")
        if not isinstance(expected_requirement, str) or requirement != expected_requirement:
            raise ValueError(f"desktop registry {app['id']} designated requirement mismatch")
        expected_cdhash = sealed.get("cdhash")
        if not isinstance(expected_cdhash, str) or f"CDHash={expected_cdhash}" not in signature:
            raise ValueError(f"desktop registry {app['id']} code-directory hash mismatch")
        expected_authorities = sealed.get("signing_authorities")
        if not isinstance(expected_authorities, list) or any(
            f"Authority={authority}" not in signature for authority in expected_authorities
        ):
            raise ValueError(f"desktop registry {app['id']} signing authority mismatch")
        if _sha256(executable) != role.get("sha256"):
            raise ValueError(f"desktop registry {app['id']} executable digest mismatch")
        stat = executable.stat()
        return {
            "entry": app["id"],
            "application": app["display_name"],
            "role": role["role"],
            "bundle_path": str(bundle),
            "executable": str(executable),
            "bundle_id": app["bundle_id"],
            "version": app["version"],
            "build": app["build"],
            "team_identifier": expected_team,
            "sha256": role["sha256"],
            "designated_requirement": expected_requirement,
            "cdhash": expected_cdhash,
            "signing_authorities": expected_authorities,
            "file_identity": {
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            },
        }

    def verify_all(self) -> dict[str, Any]:
        """Verify every sealed entry off tick and discard stale PID bindings."""
        verified: dict[str, dict[str, Any]] = {}
        checked_bundles: set[str] = set()
        for app in self.registry["applications"]:
            for role in app["executables"]:
                evidence = self._verify_role(app, role)
                # spctl is a bundle gate and is intentionally performed once.
                if evidence["bundle_path"] not in checked_bundles:
                    assessment = self._run(
                        ["/usr/sbin/spctl", "-a", "-vv", evidence["bundle_path"]]
                    )
                    if "accepted" not in assessment:
                        raise ValueError("desktop registry bundle lacks trusted spctl assessment")
                    checked_bundles.add(evidence["bundle_path"])
                verified[evidence["executable"]] = evidence
        self._verified = verified
        self._history.clear()
        self._history_log.clear()
        self._last_observation_ns = None
        self._bindings.clear()
        self._binding_history.clear()
        self._revoked_bindings.clear()
        self._revoked_processes.clear()
        return self.read_export()

    def _workload_reason(self, row: Mapping[str, Any]) -> str | None:
        return _intrinsic_workload_reason(row, self.registry, self._verified)

    def _probe_executable(
        self,
        command: str,
        *,
        uid: int,
        admit: bool,
        already_bound: bool,
    ) -> tuple[str, str, dict[str, Any] | None]:
        try:
            command_path = Path(command)
            resolved = str(_canonical(command_path))
        except (ValueError, OSError):
            return "missing_or_noncanonical_executable", command, None
        if command_path.is_symlink() or str(command_path) != resolved:
            return "symlink_or_noncanonical_executable", resolved, None
        identity = self._verified.get(resolved)
        if (
            not isinstance(identity, Mapping)
            or "role" not in identity
            or "file_identity" not in identity
        ):
            return "unenrolled_or_unverified_executable", resolved, None
        try:
            stat = Path(resolved).stat()
            current_file = {
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        except OSError:
            return "executable_replaced", resolved, dict(identity)
        if current_file != identity["file_identity"]:
            if current_file["size"] != identity["file_identity"].get("size"):
                return "executable_replaced", resolved, dict(identity)
            try:
                if _sha256(Path(resolved)) != identity.get("sha256"):
                    return "executable_replaced", resolved, dict(identity)
            except OSError:
                return "executable_replaced", resolved, dict(identity)
        elif admit and not already_bound and uid == self.permitted_uid:
            try:
                if _sha256(Path(resolved)) != identity.get("sha256"):
                    return "executable_replaced", resolved, dict(identity)
                recheck = Path(resolved).stat()
                recheck_file = {
                    "device": recheck.st_dev,
                    "inode": recheck.st_ino,
                    "size": recheck.st_size,
                    "mtime_ns": recheck.st_mtime_ns,
                    "ctime_ns": recheck.st_ctime_ns,
                }
                if recheck_file != current_file:
                    return "executable_replaced", resolved, dict(identity)
            except OSError:
                return "executable_replaced", resolved, dict(identity)
        return "ok", resolved, dict(identity)

    def normalize_row(
        self,
        row: Mapping[str, Any],
        *,
        admit: bool = False,
        history_before: Mapping[str, Any] | None = None,
        observation_ns: int | None = None,
    ) -> dict[str, Any]:
        """Collect off-tick filesystem evidence and normalize ``row`` to the exported schema."""
        if not isinstance(row, Mapping):
            raise ValueError("process row must be a mapping")
        if type(observation_ns) is int and observation_ns >= 0:
            obs_ns = observation_ns
            self._last_observation_ns = max(self._last_observation_ns or 0, obs_ns)
        elif type(row.get("observation_ns")) is int and row["observation_ns"] >= 0:
            obs_ns = int(row["observation_ns"])
            self._last_observation_ns = max(self._last_observation_ns or 0, obs_ns)
        elif type(row.get("time_ns")) is int and row["time_ns"] >= 0:
            obs_ns = int(row["time_ns"])
            self._last_observation_ns = max(self._last_observation_ns or 0, obs_ns)
        elif self._last_observation_ns is None:
            obs_ns = 0
            self._last_observation_ns = 0
        else:
            self._last_observation_ns += 1
            obs_ns = self._last_observation_ns

        argv_list, argv_status, obs_error = _extract_argv_evidence(row)
        argv_display = _extract_display_argv(row, argv_list, argv_status)
        workload_reason = _intrinsic_workload_reason(row, self.registry, self._verified)
        try:
            _, _, uid, command = self._key(row)
        except ValueError:
            executable_status = (
                "workload" if workload_reason is not None else "desktop_process_identity_missing"
            )
        else:
            override_status = row.get("executable_status")
            if workload_reason is not None:
                executable_status = "workload"
            elif isinstance(override_status, str) and override_status:
                executable_status = override_status
            else:
                already_bound = bool(
                    history_before is not None
                    and history_before.get("status") == "bound"
                    and history_before.get("uid") == uid
                    and history_before.get("command") == command
                    and history_before.get("argv") == argv_list
                    and history_before.get("argv_status") == argv_status
                    and history_before.get("argv_display") == argv_display
                )
                executable_status, _, _ = self._probe_executable(
                    command,
                    uid=uid,
                    admit=admit,
                    already_bound=already_bound,
                )

        normalized: dict[str, Any] = {
            **row,
            "admit": bool(admit),
            "executable_status": executable_status,
            "observation_ns": obs_ns,
            "argv": list(argv_list),
            "argv_display": argv_display,
        }
        if obs_error is not None:
            normalized["identity_observation_error"] = obs_error
        return normalized

    def history_snapshot(self) -> dict[tuple[int, str], dict[str, Any]]:
        """Return a deep-enough copy of the per-(pid, birth) state table."""
        return {key: dict(value) for key, value in self._history.items()}

    def history_for_row(
        self,
        row: Mapping[str, Any],
        snapshot: Mapping[tuple[int, str], dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Resolve history for ``row`` as of its capture time."""
        pid, birth = row.get("pid"), row.get("birth")
        if type(pid) is not int or pid <= 0 or not isinstance(birth, str) or not birth:
            return None
        proc_key = (pid, birth)
        if snapshot is not None:
            prior = snapshot.get(proc_key)
            return dict(prior) if isinstance(prior, Mapping) else None
        current = self._history.get(proc_key)
        if current is None:
            return None
        argv_list, argv_status, _ = _extract_argv_evidence(row)
        argv_display = _extract_display_argv(row, argv_list, argv_status)
        if (
            current.get("uid"),
            current.get("command"),
            current.get("argv"),
            current.get("argv_status"),
            current.get("argv_display"),
        ) == (row.get("uid"), row.get("command"), argv_list, argv_status, argv_display):
            return dict(current)
        for entry in reversed(self._history_log):
            after = entry.get("after")
            if (
                isinstance(after, Mapping)
                and (after.get("pid"), after.get("birth")) == proc_key
                and (
                    after.get("uid"),
                    after.get("command"),
                    after.get("argv"),
                    after.get("argv_status"),
                    after.get("argv_display"),
                )
                == (row.get("uid"), row.get("command"), argv_list, argv_status, argv_display)
            ):
                return dict(after)
        return None

    def _apply_history(
        self,
        history_before: Mapping[str, Any] | None,
        new_history: dict[str, Any],
        *,
        normalized_row: Mapping[str, Any] | None = None,
        decision: Mapping[str, Any] | None = None,
        observation_ns: int | None = None,
    ) -> None:
        pid, birth = new_history.get("pid"), new_history.get("birth")
        if type(pid) is not int or pid <= 0 or not isinstance(birth, str) or not birth:
            return
        proc_key = (pid, birth)
        last_for_proc = next(
            (
                entry
                for entry in reversed(self._history_log)
                if (entry.get("pid"), entry.get("birth")) == proc_key
            ),
            None,
        )
        should_append = False
        if last_for_proc is None:
            if history_before is None:
                should_append = True
        elif history_before == last_for_proc.get("after") and new_history != history_before:
            should_append = True

        if type(observation_ns) is int and observation_ns >= 0:
            prev_proc_obs = self._proc_last_observation_ns.get(proc_key)
            self._proc_last_observation_ns[proc_key] = (
                max(prev_proc_obs, observation_ns) if type(prev_proc_obs) is int else observation_ns
            )
            self._last_observation_ns = max(self._last_observation_ns or 0, observation_ns)

        if should_append:
            if type(observation_ns) is int and observation_ns >= 0:
                event_obs_ns = observation_ns
            elif self._last_observation_ns is not None:
                event_obs_ns = self._last_observation_ns
            else:
                event_obs_ns = 0
            if (
                last_for_proc is not None
                and type(last_for_proc.get("observation_ns")) is int
                and event_obs_ns < last_for_proc["observation_ns"]
            ):
                event_obs_ns = last_for_proc["observation_ns"]
            prev_proc_obs = self._proc_last_observation_ns.get(proc_key)
            if type(prev_proc_obs) is int and event_obs_ns < prev_proc_obs:
                event_obs_ns = prev_proc_obs
            self._proc_last_observation_ns[proc_key] = event_obs_ns
            self._last_observation_ns = max(self._last_observation_ns or 0, event_obs_ns)
            log_entry: dict[str, Any] = {
                "pid": pid,
                "birth": birth,
                "sequence": int(new_history.get("sequence", 0)),
                "observation_ns": event_obs_ns,
                "before": dict(history_before) if isinstance(history_before, Mapping) else None,
                "after": dict(new_history),
            }
            if isinstance(normalized_row, Mapping):
                log_entry["row"] = {**dict(normalized_row), "observation_ns": event_obs_ns}
            if isinstance(decision, Mapping):
                log_entry["classification"] = dict(decision)
            self._history_log.append(log_entry)
            self._history[proc_key] = dict(new_history)
        elif last_for_proc is None:
            self._history[proc_key] = dict(new_history)
        else:
            self._history[proc_key] = dict(last_for_proc["after"])

        uid, command = new_history.get("uid"), new_history.get("command")
        binding = new_history.get("binding")
        if (
            new_history.get("status") == "bound"
            and type(uid) is int
            and isinstance(command, str)
            and isinstance(binding, Mapping)
        ):
            bkey = (pid, birth, uid, command)
            stored_binding = {
                **dict(binding),
                "argv": list(new_history.get("argv", [command])),
                "argv_status": str(new_history.get("argv_status", "ok")),
                "argv_display": str(new_history.get("argv_display", command)),
            }
            self._bindings[bkey] = stored_binding
            self._binding_history[bkey] = dict(stored_binding)
        if new_history.get("status") == "revoked":
            rev_reason = str(
                new_history.get("revoked_reason")
                or new_history.get("reason")
                or "identity_binding_changed"
            )
            if new_history.get("identity_binding_changed"):
                self._revoked_processes.setdefault(proc_key, rev_reason)
            first_reason = self._revoked_processes.get(proc_key, rev_reason)
            matches = [key for key in self._bindings if key[:2] == proc_key]
            for key in matches:
                self._bindings.pop(key)
                self._revoked_bindings.setdefault(key, first_reason)

    def step_row(
        self,
        row: Mapping[str, Any],
        *,
        admit: bool,
        history_override: Any = _UNSET,
        commit: bool = True,
        observation_ns: int | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, Any] | None, dict[str, Any], str]:
        """Normalize ``row`` off-tick, evaluate via :func:`classify_row`, and commit history."""
        pid, birth = row.get("pid"), row.get("birth")
        if history_override is not _UNSET:
            history_before = (
                dict(history_override) if isinstance(history_override, Mapping) else None
            )
        elif type(pid) is int and pid > 0 and isinstance(birth, str) and birth:
            prior = self._history.get((pid, birth))
            history_before = dict(prior) if isinstance(prior, Mapping) else None
        else:
            history_before = None
        normalized = self.normalize_row(
            row,
            admit=admit,
            history_before=history_before,
            observation_ns=observation_ns,
        )
        kind, new_history, reason = classify_row(history_before, normalized, self)
        decision = decision_from_history(kind, new_history, reason)
        if commit:
            self._apply_history(
                history_before,
                new_history,
                normalized_row=normalized,
                decision=decision,
                observation_ns=normalized["observation_ns"],
            )
        return kind, decision, history_before, new_history, reason

    def invalidate_transition(self, row: Mapping[str, Any]) -> None:
        """Permanently revoke a sampled exec/UID transition, even if never admitted."""
        pid, birth = row.get("pid"), row.get("birth")
        if type(pid) is not int or pid <= 0 or not isinstance(birth, str) or not birth:
            return
        proc_key = (pid, birth)
        prior = self._history.get(proc_key)
        argv_list, argv_status, _ = _extract_argv_evidence(row)
        argv_display = _extract_display_argv(row, argv_list, argv_status)
        revoked = _make_history(
            pid=pid,
            birth=birth,
            uid=row.get("uid") if type(row.get("uid")) is int else None,
            command=row.get("command") if isinstance(row.get("command"), str) else None,
            argv=argv_list,
            argv_status=argv_status,
            argv_display=argv_display,
            status="revoked",
            kind=str(prior.get("kind", "unknown")) if isinstance(prior, Mapping) else "unknown",
            reason="identity_binding_changed",
            revoked=True,
            revoked_reason=(
                str(prior.get("revoked_reason") or "identity_binding_changed")
                if isinstance(prior, Mapping)
                else "identity_binding_changed"
            ),
            identity_binding_changed=True,
            binding=(
                dict(prior["binding"])
                if isinstance(prior, Mapping) and isinstance(prior.get("binding"), Mapping)
                else None
            ),
            executable_status=(
                str(prior.get("executable_status", "transition"))
                if isinstance(prior, Mapping)
                else "transition"
            ),
            prior_history=prior,
        )
        self._apply_history(prior, revoked)

    def admit_process(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Bind only an unbound, verified identity; revoked is terminal this attempt."""
        _, decision, _, _, _ = self.step_row(row, admit=True)
        return decision

    @staticmethod
    def _key(row: Mapping[str, Any]) -> tuple[int, str, int, str]:
        try:
            pid, uid, birth, command = row["pid"], row["uid"], row["birth"], row["command"]
            if type(pid) is not int or pid <= 0 or type(uid) is not int or uid < 0:
                raise ValueError
            if (
                not isinstance(birth, str)
                or not birth
                or not isinstance(command, str)
                or not command
            ):
                raise ValueError
            return pid, birth, uid, command
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("desktop process row lacks pid, birth, uid, or command") from exc

    def classify(self, row: Mapping[str, Any], *, allow_unbound: bool = False) -> dict[str, Any]:
        """Classify once through pure ``classify_row``, applying terminal revocation on failure."""
        _, decision, _, _, _ = self.step_row(row, admit=allow_unbound)
        return decision

    def read_export(self) -> dict[str, Any]:
        """Return only serializable policy, registry and cached binding/history evidence."""
        effective_bytes = canonical_bytes(self.registry)
        return {
            "policy": POLICY,
            "registry_sha256": hashlib.sha256(effective_bytes).hexdigest(),
            "registry_inputs": self._registry_inputs,
            "registry": self.registry,
            "verified": dict(self._verified),
            "process_history": [dict(self._history[key]) for key in sorted(self._history)],
            "history_log": [dict(entry) for entry in self._history_log],
            "bindings": [
                {
                    k: v
                    for k, v in self._bindings[key].items()
                    if k not in {"argv", "argv_status", "argv_display"}
                }
                | {"pid": key[0], "birth": key[1], "uid": key[2], "command": key[3]}
                for key in sorted(self._bindings)
            ],
            "binding_history": [
                {
                    k: v
                    for k, v in self._binding_history[key].items()
                    if k not in {"argv", "argv_status", "argv_display"}
                }
                | {"pid": key[0], "birth": key[1], "uid": key[2], "command": key[3]}
                for key in sorted(self._binding_history)
            ],
            "revoked_processes": [
                {"pid": pid, "birth": birth, "reason": reason}
                for (pid, birth), reason in sorted(self._revoked_processes.items())
            ],
            "revoked_bindings": [
                {
                    "pid": key[0],
                    "birth": key[1],
                    "uid": key[2],
                    "command": key[3],
                    "reason": reason,
                }
                for key, reason in sorted(self._revoked_bindings.items())
            ],
        }


def _reconstruct_exported_history(
    row: Mapping[str, Any], registry_export: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Reconstruct ``history`` for ``row`` from ``registry_export`` when not embedded."""
    if "history_before" in row:
        prior = row["history_before"]
        return dict(prior) if isinstance(prior, Mapping) else None
    try:
        key = DesktopRegistry._key(row)
    except ValueError:
        return None
    pid, birth, uid, command = key
    argv_list, argv_status, _ = _extract_argv_evidence(row)
    argv_display = _extract_display_argv(row, argv_list, argv_status)
    classification = row.get("classification")
    explicit_changed = isinstance(classification, Mapping) and bool(
        classification.get("identity_binding_changed")
    )
    reg_dict = (
        registry_export.get("registry", {})
        if isinstance(registry_export.get("registry"), Mapping)
        else {}
    )
    verified = (
        registry_export.get("verified", {})
        if isinstance(registry_export.get("verified"), Mapping)
        else {}
    )
    if not explicit_changed:
        for binding in [
            *(registry_export.get("bindings") or ()),
            *(registry_export.get("binding_history") or ()),
        ]:
            if not isinstance(binding, Mapping):
                continue
            if (
                binding.get("pid"),
                binding.get("birth"),
                binding.get("uid"),
                binding.get("command"),
            ) != key:
                continue
            identity = binding.get("identity")
            if (
                not isinstance(identity, Mapping)
                or verified.get(identity.get("executable")) != identity
                or not _valid_verified_identity(command, identity, reg_dict)
            ):
                return _make_history(
                    pid=pid,
                    birth=birth,
                    uid=uid,
                    command=command,
                    argv=argv_list,
                    argv_status=argv_status,
                    argv_display=argv_display,
                    status="revoked",
                    kind="unknown",
                    reason="desktop_binding_identity_mismatch",
                    revoked=True,
                    revoked_reason="desktop_binding_identity_mismatch",
                    identity_binding_changed=False,
                    binding=None,
                    executable_status="desktop_binding_identity_mismatch",
                )
            role_kind = (
                "desktop_renderer" if "renderer" in str(binding.get("role", "")) else "desktop_app"
            )
            reason = str(binding.get("reason", "enrolled_verified_ui_role"))
            b_dict = {
                "kind": role_kind,
                "reason": reason,
                "entry": binding["entry"],
                "application": binding["application"],
                "role": binding["role"],
                "identity": dict(identity),
            }
            bound_argv = [command]
            bound_display = command
            for entry in registry_export.get("history_log") or ():
                after = entry.get("after") if isinstance(entry, Mapping) else None
                if (
                    isinstance(after, Mapping)
                    and (
                        after.get("pid"),
                        after.get("birth"),
                        after.get("uid"),
                        after.get("command"),
                    )
                    == key
                    and after.get("status") == "bound"
                    and isinstance(after.get("argv"), list)
                ):
                    bound_argv = [str(x) for x in after["argv"]]
                    bound_display = str(after.get("argv_display") or " ".join(bound_argv))
                    break
            return _make_history(
                pid=pid,
                birth=birth,
                uid=uid,
                command=command,
                argv=bound_argv,
                argv_status="ok",
                argv_display=bound_display,
                status="bound",
                kind=role_kind,
                reason=reason,
                revoked=False,
                revoked_reason=None,
                identity_binding_changed=False,
                binding=b_dict,
                executable_status="ok",
            )
    for hist in registry_export.get("process_history") or ():
        if isinstance(hist, Mapping) and (hist.get("pid"), hist.get("birth")) == (
            pid,
            birth,
        ):
            return dict(hist)
    for rev in [
        *(registry_export.get("revoked_processes") or ()),
        *(registry_export.get("revoked_bindings") or ()),
    ]:
        if isinstance(rev, Mapping) and (rev.get("pid"), rev.get("birth")) == (
            pid,
            birth,
        ):
            rev_reason = str(rev.get("reason", "identity_binding_changed"))
            return _make_history(
                pid=pid,
                birth=birth,
                uid=rev.get("uid", uid) if type(rev.get("uid", uid)) is int else uid,
                command=(
                    rev.get("command", command)
                    if isinstance(rev.get("command", command), str)
                    else command
                ),
                argv=argv_list,
                argv_status=argv_status,
                argv_display=argv_display,
                status="revoked",
                kind="unknown",
                reason=rev_reason,
                revoked=True,
                revoked_reason=rev_reason,
                identity_binding_changed=True,
                binding=None,
                executable_status=rev_reason,
            )
    if explicit_changed and isinstance(classification, Mapping):
        cls_reason = str(classification.get("reason", "identity_binding_changed"))
        return _make_history(
            pid=pid,
            birth=birth,
            uid=uid,
            command=command,
            argv=argv_list,
            argv_status=argv_status,
            argv_display=argv_display,
            status="revoked",
            kind="unknown",
            reason=cls_reason,
            revoked=True,
            revoked_reason=cls_reason,
            identity_binding_changed=True,
            binding=None,
            executable_status=cls_reason,
        )
    return None


def classify_exported(row: Mapping[str, Any], registry_export: Mapping[str, Any]) -> dict[str, Any]:
    """Replay one desktop decision through pure :func:`classify_row`."""
    if not isinstance(registry_export, Mapping) or registry_export.get("policy") != POLICY:
        return {"kind": "unknown", "reason": "desktop_registry_policy_mismatch"}
    history = _reconstruct_exported_history(row, registry_export)
    admit = bool(row.get("admit", False)) if "history_before" in row else False
    kind, new_history, reason = classify_row(history, {**row, "admit": admit}, registry_export)
    return decision_from_history(kind, new_history, reason)
