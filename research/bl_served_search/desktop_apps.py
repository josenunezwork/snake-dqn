"""Sealed ordinary-desktop identity admission for ``classified-desktop-v1``.

This module deliberately does not inspect processes or execute signature tools
from ``classify``.  Expensive code-identity checks and PID/birth bindings are
made by the operator before a scored observation and cached in this object.
New PIDs require off-tick admission. An exec, UID transition, or revoked
identity is permanently ineligible for the rest of this registry attempt.
"""

from __future__ import annotations

import bisect
import copy
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from research.bl_k1.m1_background import SYSTEM_EXECUTABLE_ROOTS
from research.bl_paced_identity.events import canonical_bytes

POLICY = "classified-desktop-v1"
DEFAULT_REGISTRY = Path(__file__).with_name("DESKTOP_APP_REGISTRY_20261027_02.json")
_WORKLOAD_NAMES = frozenset(
    {"python", "python3", "node", "pytest", "uv", "codex", "codex-exec", "claude", "agy"}
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


@dataclass(frozen=True)
class History:
    """Complete, immutable state for one ``(pid, birth)`` identity stream."""

    pid: int | None = None
    birth: str | None = None
    fingerprint: tuple[Any, ...] = ()
    admitted_key: str | None = None
    revoked_reason: str | None = None
    identity_binding_changed: bool = False
    executable_identity: tuple[Any, ...] = ()


EMPTY_HISTORY = History()


class ReplayDecisions(dict[int, dict[str, Any]]):
    """Validated decisions plus O(log n) transition lookup by process/time."""

    transitions: dict[tuple[int, str], tuple[list[int], list[int]]]

    def __init__(self) -> None:
        super().__init__()
        self.transitions = {}


def _history(value: History | dict[str, Any] | None) -> History:
    if isinstance(value, History):
        return value
    if not isinstance(value, dict):
        return EMPTY_HISTORY
    fingerprint = value.get("fingerprint", ())
    return History(
        pid=value.get("pid") if type(value.get("pid")) is int else None,
        birth=value.get("birth") if isinstance(value.get("birth"), str) else None,
        fingerprint=_freeze(fingerprint) if isinstance(fingerprint, (list, tuple)) else (),
        admitted_key=(
            value.get("admitted_key") if isinstance(value.get("admitted_key"), str) else None
        ),
        revoked_reason=(
            value.get("revoked_reason") if isinstance(value.get("revoked_reason"), str) else None
        ),
        identity_binding_changed=value.get("identity_binding_changed") is True,
        executable_identity=_freeze(value.get("executable_identity", ())),
    )


def _history_json(value: History) -> dict[str, Any]:
    return _jsonable(asdict(value))


def _freeze(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _executable_identity(verification: dict[str, Any]) -> tuple[Any, ...]:
    value = verification.get("file_identity")
    if not isinstance(value, dict):
        return ()
    keys = ("device", "inode", "size", "mtime_ns", "ctime_ns")
    if any(type(value.get(key)) is not int for key in keys):
        return ()
    return tuple(value[key] for key in keys)


def _process_key(row: dict[str, Any]) -> tuple[int, str] | None:
    pid, birth = row.get("pid"), row.get("birth")
    return (
        (pid, birth) if type(pid) is int and pid > 0 and isinstance(birth, str) and birth else None
    )


def _fingerprint(row: dict[str, Any]) -> tuple[Any, ...] | None:
    key = _process_key(row)
    uid, command = row.get("uid"), row.get("command")
    if (
        key is None
        or type(uid) is not int
        or uid < 0
        or not isinstance(command, str)
        or not command
    ):
        return None
    argv = row.get("argv", ())
    if not isinstance(argv, (list, tuple)):
        return None
    return (uid, command, tuple(str(item) for item in argv))


def _workload_reason(row: dict[str, Any], registry_export: dict[str, Any]) -> str | None:
    if row.get("known_workload") or row.get("workload"):
        return "known_workload"
    command = str(row.get("command", ""))
    argv = " ".join(str(item) for item in row.get("argv", ()))
    for exclusion in registry_export.get("registry", {}).get("workload_exclusions", []):
        if exclusion.get("exact_executable") == command:
            return "workload_exclusion"
    # Enrolled GUI mains may carry these names; arbitrary command-line uses may not.
    verified = registry_export.get("verified", {})
    if command in verified and Path(command).name in {"Claude", "Codex"}:
        return None
    name = Path(command).name.lower()
    if (
        name in _WORKLOAD_NAMES
        or re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", name)
        or _WORKLOAD_TEXT.search(command)
        or _WORKLOAD_TEXT.search(argv)
    ):
        return "workload_exclusion"
    return None


def _enrolled_identity(identity: Any, command: str, registry: dict[str, Any]) -> bool:
    """Check exported verification evidence against the sealed registry bytes."""
    if not isinstance(identity, dict):
        return False
    for app in registry.get("applications", []):
        if app.get("id") != identity.get("entry"):
            continue
        for role in app.get("executables", []):
            if role.get("role") != identity.get("role") or role.get("role") not in _ELIGIBLE_ROLES:
                continue
            relative = Path(str(role.get("relative_executable_path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                return False
            sealed = registry.get("signed_identities", {}).get(role.get("sha256"), role)
            return (
                command == str(Path(str(app.get("bundle_path", ""))) / relative)
                and identity.get("executable") == command
                and identity.get("application") == app.get("display_name")
                and identity.get("bundle_path") == app.get("bundle_path")
                and identity.get("bundle_id") == app.get("bundle_id")
                and identity.get("version") == app.get("version")
                and identity.get("build") == app.get("build")
                and identity.get("team_identifier") == app.get("team_identifier")
                and identity.get("sha256") == role.get("sha256")
                and identity.get("designated_requirement") == sealed.get("designated_requirement")
                and identity.get("cdhash") == sealed.get("cdhash")
                and identity.get("signing_authorities") == sealed.get("signing_authorities")
            )
    return False


def classify_row(
    history_for_pid_birth: History | dict[str, Any] | None,
    row: dict[str, Any],
    registry_export: dict[str, Any],
) -> tuple[str, History, str]:
    """Pure policy transition used by collection and reducer replay.

    Filesystem/signature work belongs in ``row['verification']``.  This function
    reads only its arguments and returns the next complete history explicitly.
    """
    history = _history(history_for_pid_birth)
    key = _process_key(row)
    fingerprint = _fingerprint(row)
    if key is None or fingerprint is None:
        if key is not None:
            terminal = (
                history
                if history.revoked_reason
                else History(
                    pid=key[0],
                    birth=key[1],
                    fingerprint=history.fingerprint,
                    admitted_key=history.admitted_key,
                    revoked_reason="desktop_process_identity_missing",
                )
            )
            return (
                "unknown",
                terminal,
                terminal.revoked_reason or "desktop_process_identity_missing",
            )
        return "unknown", history, "desktop_process_identity_missing"
    if history.pid is None:
        history = History(pid=key[0], birth=key[1])
    elif (history.pid, history.birth) != key:
        return "unknown", history, "desktop_history_process_mismatch"
    workload = _workload_reason(row, registry_export)
    if history.fingerprint and history.fingerprint != fingerprint:
        changed = replace(
            history,
            revoked_reason=history.revoked_reason or "identity_binding_changed",
            identity_binding_changed=True,
        )
        if workload:
            return "agent_workload", changed, workload
        return (
            "unknown",
            changed,
            changed.revoked_reason or "identity_binding_changed",
        )
    # Workload evidence remains workload evidence after a terminal identity
    # revocation; the immutable revocation itself must still be retained.
    if history.revoked_reason:
        return (
            ("agent_workload", history, workload)
            if workload
            else ("unknown", history, history.revoked_reason)
        )
    verification = row.get("verification", {})
    observed_executable = (
        _executable_identity(verification) if isinstance(verification, dict) else ()
    )
    failure = _verification_failure(row, registry_export, history)
    if (
        not failure
        and history.executable_identity
        and history.executable_identity != observed_executable
    ):
        failure = "executable_replaced"
    if failure:
        enrolled = registry_export.get("verified", {}).get(row.get("command"))
        failed = History(
            pid=key[0],
            birth=key[1],
            fingerprint=fingerprint,
            admitted_key=history.admitted_key,
            revoked_reason=failure,
            identity_binding_changed=history.identity_binding_changed
            or bool(history.admitted_key)
            or (failure == "executable_replaced" and isinstance(enrolled, dict)),
            executable_identity=history.executable_identity or observed_executable,
        )
        return ("agent_workload", failed, workload) if workload else ("unknown", failed, failure)
    if workload:
        next_history = (
            history
            if history.fingerprint
            else History(
                pid=key[0],
                birth=key[1],
                fingerprint=fingerprint,
                executable_identity=observed_executable,
            )
        )
        return "agent_workload", next_history, workload
    base = History(
        pid=key[0],
        birth=key[1],
        fingerprint=fingerprint,
        admitted_key=history.admitted_key,
        revoked_reason=None,
        executable_identity=history.executable_identity or observed_executable,
    )
    command = str(row["command"])
    if command.startswith(SYSTEM_EXECUTABLE_ROOTS) and ".." not in Path(command).parts:
        return "macos_service", base, "system_executable_root"
    registry = registry_export.get("registry", {})
    if fingerprint[0] != registry.get("owner_uid"):
        return (
            "unknown",
            History(
                pid=key[0],
                birth=key[1],
                fingerprint=fingerprint,
                revoked_reason="uid_mismatch",
                executable_identity=observed_executable,
            ),
            "uid_mismatch",
        )
    identity = registry_export.get("verified", {}).get(command)
    if not _enrolled_identity(identity, command, registry):
        return "unknown", base, "unenrolled_or_unverified_executable"
    if base.admitted_key is None:
        if not row.get("admit"):
            return "unknown", base, "process_identity_not_revalidated"
        verification = row.get("verification", {})
        digest_ok = isinstance(verification, dict) and verification.get("digest") == identity.get(
            "sha256"
        )
        if not digest_ok:
            return (
                "unknown",
                History(
                    pid=key[0],
                    birth=key[1],
                    fingerprint=fingerprint,
                    revoked_reason="executable_replaced",
                    executable_identity=observed_executable,
                ),
                "executable_replaced",
            )
        base = History(
            pid=key[0],
            birth=key[1],
            fingerprint=fingerprint,
            admitted_key=command,
            executable_identity=observed_executable,
        )
    elif base.admitted_key != command:
        return (
            "unknown",
            replace(base, revoked_reason="identity_binding_changed", identity_binding_changed=True),
            "identity_binding_changed",
        )
    return (
        "desktop_renderer" if "renderer" in str(identity.get("role")) else "desktop_app",
        base,
        "enrolled_verified_ui_role",
    )


def decision_evidence(
    kind: str, history: History, reason: str, registry_export: dict[str, Any], index: int
) -> dict[str, Any]:
    """Build the only public decision representation for live and replay paths."""
    decision: dict[str, Any] = {
        "kind": kind,
        "reason": reason,
        "identity_binding_changed": history.identity_binding_changed,
        "history_index": index,
    }
    if history.admitted_key and kind in {"desktop_app", "desktop_renderer"}:
        identity = registry_export.get("verified", {}).get(history.admitted_key)
        if isinstance(identity, dict):
            decision |= {
                "entry": identity.get("entry"),
                "application": identity.get("application"),
                "role": identity.get("role"),
                "identity": copy.deepcopy(identity),
            }
    return decision


def _identity_projection(row: dict[str, Any]) -> dict[str, Any]:
    """The raw process facts that may select a cached transition event."""
    fields = (
        "pid",
        "birth",
        "uid",
        "command",
        "argv",
        "known_workload",
        "workload",
        "identity_observation_error",
    )
    return {field: copy.deepcopy(row.get(field)) for field in fields if field in row}


def _verification_failure(
    row: dict[str, Any], registry_export: dict[str, Any], history: History
) -> str | None:
    """Interpret captured filesystem facts; this is policy, never adapter logic."""
    if isinstance(row.get("identity_observation_error"), str) and row["identity_observation_error"]:
        return "identity_observation_failed"
    verification = row.get("verification", {})
    if not isinstance(verification, dict):
        return "verification_evidence_missing"
    if verification.get("ok") is not True:
        return "verification_evidence_missing"
    if verification.get("canonical_error"):
        return "missing_or_noncanonical_executable"
    if (
        not isinstance(verification.get("canonical"), str)
        or type(verification.get("is_symlink")) is not bool
        or not _executable_identity(verification)
    ):
        return "verification_evidence_missing"
    if verification["is_symlink"] or verification["canonical"] != row.get("command"):
        return "symlink_or_noncanonical_executable"
    identity = registry_export.get("verified", {}).get(row.get("command"))
    if isinstance(identity, dict):
        expected = identity.get("file_identity")
        observed = verification.get("file_identity")
        # The registry stat tuple belongs to the verification attempt.  A
        # fresh process generation is bound to its captured History identity.
        if (
            not history.executable_identity
            and not row.get("admit")
            and observed is not None
            and observed != expected
        ):
            return "executable_replaced"
        if row.get("admit") and not history.admitted_key:
            if verification.get("digest") != identity.get("sha256"):
                return "executable_replaced"
            after_digest = verification.get("file_identity_after_digest")
            if _executable_identity({"file_identity": after_digest}) != _executable_identity(
                verification
            ):
                return "executable_replaced"
    return None


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


class _RegistryVerification:
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
        self._histories: dict[tuple[int, str], History] = {}
        self._history_sequence: list[dict[str, Any]] = []

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
        return self.read_export()


class DesktopRegistry(_RegistryVerification):
    """Event-sourced adapter for :func:`classify_row`.

    It holds no classification state besides the immutable history records and
    append-only events that it exports.  Verification is prepared here, outside
    the pure policy transition, so reducer replay has the exact same evidence.
    """

    @staticmethod
    def _key(row: dict[str, Any]) -> tuple[int, str, int, str]:
        key = _process_key(row)
        uid, command = row.get("uid"), row.get("command")
        if (
            key is None
            or type(uid) is not int
            or uid < 0
            or not isinstance(command, str)
            or not command
        ):
            raise ValueError("desktop process row lacks pid, birth, uid, or command")
        return key[0], key[1], uid, command

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._base_export_cache: dict[str, Any] | None = None
        self._last_event_by_process: dict[tuple[int, str], int] = {}

    def verify_all(self) -> dict[str, Any]:
        if self._histories:
            raise ValueError("desktop registry cannot be reverified after identity history begins")
        result = super().verify_all()
        self._base_export_cache = None
        return result

    def _workload_reason(self, row: dict[str, Any]) -> str | None:
        return _workload_reason(row, {"registry": self.registry, "verified": self._verified})

    def _prepare(self, row: dict[str, Any], *, admit: bool) -> dict[str, Any]:
        prepared = copy.deepcopy(row)
        prepared["admit"] = admit
        current_history = self._histories.get(_process_key(row), EMPTY_HISTORY)
        if current_history.revoked_reason:
            prepared["verification"] = {"ok": True, "skipped": "already_revoked"}
            return prepared
        verification: dict[str, Any] = {"ok": True}
        command = row.get("command")
        if not isinstance(command, str) or not command:
            verification["canonical_error"] = "missing_command"
        else:
            try:
                raw = Path(command).expanduser()
                canonical = _canonical(raw)
                verification["canonical"] = str(canonical)
                verification["is_symlink"] = raw.is_symlink()
                stat = canonical.stat()
                verification["file_identity"] = {
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns,
                }
                identity = self._verified.get(command)
                if admit and current_history.admitted_key is None and identity is not None:
                    verification["digest"] = _sha256(canonical)
                    stat = canonical.stat()
                    verification["file_identity_after_digest"] = {
                        "device": stat.st_dev,
                        "inode": stat.st_ino,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "ctime_ns": stat.st_ctime_ns,
                    }
            except OSError:
                verification["canonical_error"] = "stat_or_resolution_failed"
        prepared["verification"] = verification
        return prepared

    def _export_base(self) -> dict[str, Any]:
        if self._base_export_cache is None:
            self._base_export_cache = {
                "policy": POLICY,
                "registry_sha256": hashlib.sha256(canonical_bytes(self.registry)).hexdigest(),
                "registry_inputs": self._registry_inputs,
                "registry": self.registry,
            }
        return self._base_export_cache | {"verified": self._verified}

    def observe(self, row: dict[str, Any], *, admit: bool = False) -> dict[str, Any]:
        prepared = self._prepare(row, admit=admit)
        key = _process_key(prepared)
        history = self._histories.get(key, EMPTY_HISTORY) if key is not None else EMPTY_HISTORY
        kind, next_history, reason = classify_row(history, prepared, self._export_base())
        if key is not None:
            self._histories[key] = next_history
        semantic = decision_evidence(kind, next_history, reason, self._export_base(), -1)
        previous_index = self._last_event_by_process.get(key) if key is not None else None
        previous = (
            self._history_sequence[previous_index]
            if isinstance(previous_index, int) and previous_index < len(self._history_sequence)
            else None
        )
        # Most process rows are steady-state.  Reuse the last decision only
        # when its raw identity projection is exactly the same; a revoked
        # process can otherwise carry a historical fingerprint from before an
        # exec and must not silently point at that old row.
        if (
            previous is not None
            and previous["history_after"] == _history_json(next_history)
            and (
                type(prepared.get("observation_ns")) is not int
                or type(previous["row"].get("observation_ns")) is int
            )
            and _identity_projection(previous["row"]) == _identity_projection(prepared)
            and {
                key: value
                for key, value in previous["classification"].items()
                if key != "history_index"
            }
            == {key: value for key, value in semantic.items() if key != "history_index"}
        ):
            return dict(previous["classification"])
        index = len(self._history_sequence)
        if key is not None and type(prepared.get("observation_ns")) is int and previous is not None:
            prior_time = previous["row"].get("observation_ns")
            if type(prior_time) is int and prepared["observation_ns"] < prior_time:
                raise ValueError("desktop observation time regressed")
        decision = decision_evidence(kind, next_history, reason, self._export_base(), index)
        event = {
            "index": index,
            "row": prepared,
            "history_before": _history_json(history),
            "history_after": _history_json(next_history),
            "classification": dict(decision),
        }
        self._history_sequence.append(event)
        if key is not None:
            self._last_event_by_process[key] = index
        return decision

    def observe_snapshot(self, row: dict[str, Any], *, admit: bool = False) -> dict[str, Any]:
        return {**row, "classification": self.observe(row, admit=admit)}

    def classify(self, row: dict[str, Any], *, allow_unbound: bool = False) -> dict[str, Any]:
        return self.observe(row, admit=allow_unbound)

    def admit_process(self, row: dict[str, Any]) -> dict[str, Any]:
        return self.observe(row, admit=True)

    def read_export(self) -> dict[str, Any]:
        exported = copy.deepcopy(self._export_base())
        histories = [_history_json(value) for _, value in sorted(self._histories.items())]
        exported |= {
            "history_sequence": copy.deepcopy(self._history_sequence),
            "histories": histories,
        }
        # Compatibility fields are derived views, not an independent state machine.
        historical = [value for value in self._histories.values() if value.admitted_key]
        exported["binding_history"] = [
            self._verified[value.admitted_key]
            | {
                "pid": value.pid,
                "birth": value.birth,
                "uid": value.fingerprint[0],
                "command": value.fingerprint[1],
            }
            for value in historical
        ]
        exported["bindings"] = [
            item
            for item, value in zip(exported["binding_history"], historical)
            if not value.revoked_reason
        ]
        exported["revoked_processes"] = [
            {"pid": value.pid, "birth": value.birth, "reason": value.revoked_reason}
            for value in self._histories.values()
            if value.revoked_reason
        ]
        exported["revoked_bindings"] = [
            {
                "pid": value.pid,
                "birth": value.birth,
                "uid": value.fingerprint[0],
                "command": value.fingerprint[1],
                "reason": value.revoked_reason,
            }
            for value in self._histories.values()
            if value.revoked_reason and value.admitted_key
        ]
        return exported


def replay_history(registry_export: dict[str, Any]) -> ReplayDecisions:
    """Validate every exported event and return its independently replayed decisions."""
    if registry_export.get("policy") != POLICY:
        raise ValueError("desktop registry policy mismatch")
    histories: dict[tuple[int, str], History] = {}
    decisions = ReplayDecisions()
    events = registry_export.get("history_sequence")
    if not isinstance(events, list):
        raise ValueError("desktop history sequence missing")
    for index, event in enumerate(events):
        if (
            not isinstance(event, dict)
            or event.get("index") != index
            or not isinstance(event.get("row"), dict)
        ):
            raise ValueError("desktop history event malformed")
        row = event["row"]
        key = _process_key(row)
        before = histories.get(key, EMPTY_HISTORY) if key is not None else EMPTY_HISTORY
        if event.get("history_before") != _history_json(before):
            raise ValueError("desktop history before mismatch")
        kind, after, reason = classify_row(before, row, registry_export)
        decision = decision_evidence(kind, after, reason, registry_export, index)
        if (
            event.get("history_after") != _history_json(after)
            or event.get("classification") != decision
        ):
            raise ValueError("desktop history replay mismatch")
        if key is not None:
            histories[key] = after
            observed_at = row.get("observation_ns")
            if type(observed_at) is int:
                times, indexes = decisions.transitions.setdefault(key, ([], []))
                if times and observed_at < times[-1]:
                    raise ValueError("desktop history event time regressed")
                times.append(observed_at)
                indexes.append(index)
        decisions[index] = decision
    expected = [_history_json(value) for _, value in sorted(histories.items())]
    if registry_export.get("histories") != expected:
        raise ValueError("desktop final histories mismatch")
    return decisions


def replay_reference(
    row: dict[str, Any],
    registry_export: dict[str, Any],
    decisions: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    classification = row.get("classification")
    index = classification.get("history_index") if isinstance(classification, dict) else None
    if type(index) is not int:
        return {"kind": "unknown", "reason": "desktop_history_reference_missing"}
    decisions = replay_history(registry_export) if decisions is None else decisions
    event = registry_export.get("history_sequence", [])[index] if index in decisions else None
    if not isinstance(event, dict):
        return {"kind": "unknown", "reason": "desktop_history_reference_invalid"}
    fields = (
        "pid",
        "birth",
        "uid",
        "command",
        "argv",
        "known_workload",
        "workload",
        "identity_observation_error",
    )
    if any(row.get(field) != event["row"].get(field) for field in fields):
        return {"kind": "unknown", "reason": "desktop_history_reference_row_mismatch"}
    observed_at = row.get("observation_ns")
    effective_at = event["row"].get("observation_ns")
    if type(observed_at) is int and type(effective_at) is int:
        if observed_at < effective_at:
            return {"kind": "unknown", "reason": "desktop_history_reference_before_effective"}
        key = _process_key(event["row"])
        if isinstance(decisions, ReplayDecisions) and key in decisions.transitions:
            times, indexes = decisions.transitions[key]
            position = bisect.bisect_right(times, observed_at) - 1
            if position < 0 or indexes[position] != index:
                return {"kind": "unknown", "reason": "desktop_history_reference_superseded"}
    elif type(observed_at) is int:
        return {"kind": "unknown", "reason": "desktop_history_reference_effective_time_missing"}
    return decisions[index]


def classify_exported(row: dict[str, Any], registry_export: dict[str, Any]) -> dict[str, Any]:
    """Replay the event referenced by ``row``; embedded classifications are not trusted."""
    try:
        return replay_reference(row, registry_export)
    except ValueError:
        return {"kind": "unknown", "reason": "desktop_history_replay_invalid"}
