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
from pathlib import Path
from typing import Any, Callable, Iterable

from research.bl_k1.m1_background import SYSTEM_EXECUTABLE_ROOTS
from research.bl_paced_identity.events import canonical_bytes

POLICY = "classified-desktop-v1"
DEFAULT_REGISTRY = Path(__file__).with_name("DESKTOP_APP_REGISTRY_20261027_02.json")
_WORKLOAD_NAMES = frozenset({"python", "python3", "node", "pytest", "uv", "codex", "claude"})
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
        self._bindings.clear()
        self._binding_history.clear()
        self._revoked_bindings.clear()
        self._revoked_processes.clear()
        return self.read_export()

    def _workload_reason(self, row: dict[str, Any]) -> str | None:
        if row.get("known_workload") or row.get("workload"):
            return "known_workload"
        command = str(row.get("command", ""))
        argv = " ".join(str(x) for x in row.get("argv", ()))
        for exclusion in self.registry["workload_exclusions"]:
            if exclusion.get("exact_executable") == command:
                return "workload_exclusion"
        # These signed GUI mains are eligible only through a verified binding.
        # Arbitrary enrolled Python/node/CLI identities never bypass workloads.
        if command in self._verified and Path(command).name in {"Claude", "Codex"}:
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

    def _binding_state(self, row: dict[str, Any]) -> str:
        """State machine: unbound -> bound -> revoked; revoked is terminal."""
        pid, birth = row.get("pid"), row.get("birth")
        if type(pid) is not int or not isinstance(birth, str):
            return "unbound"
        process = (pid, birth)
        if process in self._revoked_processes:
            return "revoked"
        matches = [key for key in self._bindings if key[:2] == process]
        if not matches:
            return "unbound"
        try:
            return "bound" if self._key(row) in matches else "changed"
        except ValueError:
            return "changed"

    def invalidate_transition(self, row: dict[str, Any]) -> None:
        """Permanently revoke a sampled exec/UID transition, even if never admitted."""
        self._invalidate_bindings(row, "identity_binding_changed", transition=True)

    def _invalidate_bindings(
        self, row: dict[str, Any], reason: str, *, transition: bool = False
    ) -> bool:
        """Revoke once, retaining the first reason and immutable admission history."""
        changed = self._binding_changed(row) or transition
        pid, birth = row.get("pid"), row.get("birth")
        if type(pid) is not int or not isinstance(birth, str):
            return False
        process = (pid, birth)
        if changed:
            self._revoked_processes.setdefault(process, reason)
        first_reason = self._revoked_processes.get(process, reason)
        matches = [key for key in self._bindings if key[:2] == process]
        for key in matches:
            self._bindings.pop(key)
            self._revoked_bindings.setdefault(key, first_reason)
        return changed

    def _binding_changed(self, row: dict[str, Any]) -> bool:
        """Whether failure must revoke an existing or already revoked process."""
        return self._binding_state(row) != "unbound"

    def _failure(self, row: dict[str, Any], reason: str, kind: str = "unknown") -> dict[str, Any]:
        changed = self._invalidate_bindings(row, reason)
        # Replacement remains identifiable on every subsequent point/interval
        # check, including after the executable is restored on disk.
        pid, birth = row.get("pid"), row.get("birth")
        first_reason = (
            self._revoked_processes.get((pid, birth))
            if type(pid) is int and isinstance(birth, str)
            else None
        )
        if first_reason == "executable_replaced" and kind != "agent_workload":
            reason = first_reason
        return {"kind": kind, "reason": reason, "identity_binding_changed": changed}

    def admit_process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Bind only an unbound, verified identity; revoked is terminal this attempt."""
        state = self._binding_state(row)
        decision = self.classify(row, allow_unbound=True)
        if decision["kind"] not in {"desktop_app", "desktop_renderer"} or state == "bound":
            return decision
        identity = decision["identity"]
        try:
            if _sha256(Path(str(row["command"]))) != identity["sha256"]:
                return self._failure(row, "executable_replaced")
        except OSError:
            return self._failure(row, "executable_replaced")
        # Recheck the cheap file identity after digest I/O before publishing.
        decision = self.classify(row, allow_unbound=True)
        if decision["kind"] not in {"desktop_app", "desktop_renderer"}:
            return decision
        key = self._key(row)
        self._bindings[key] = decision
        self._binding_history[key] = decision
        return decision

    @staticmethod
    def _key(row: dict[str, Any]) -> tuple[int, str, int, str]:
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

    def classify(self, row: dict[str, Any], *, allow_unbound: bool = False) -> dict[str, Any]:
        """Classify once from cached evidence, applying terminal revocation on failure."""
        workload = self._workload_reason(row)
        if workload:
            return self._failure(row, workload, "agent_workload")
        state = self._binding_state(row)
        if state == "revoked":
            return self._failure(row, "identity_binding_changed")
        try:
            key = self._key(row)
            command_path = Path(key[3])
            command = str(_canonical(command_path))
        except (ValueError, OSError):
            return self._failure(row, "missing_or_noncanonical_executable")
        if command_path.is_symlink() or str(command_path) != command:
            return self._failure(row, "symlink_or_noncanonical_executable")
        if key[2] != self.permitted_uid:
            return self._failure(row, "uid_mismatch")
        identity = self._verified.get(command)
        if identity is None:
            return self._failure(row, "unenrolled_or_unverified_executable")
        kind = "desktop_renderer" if "renderer" in identity["role"] else "desktop_app"
        decision = {
            "kind": kind,
            "reason": "enrolled_verified_ui_role",
            "entry": identity["entry"],
            "application": identity["application"],
            "role": identity["role"],
            "identity": identity,
        }
        if state == "changed" or (not allow_unbound and self._bindings.get(key) != decision):
            return self._failure(row, "process_identity_not_revalidated")
        try:
            stat = Path(command).stat()
            current_file = {
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        except OSError:
            return self._failure(row, "executable_replaced")
        if current_file != identity["file_identity"]:
            return self._failure(row, "executable_replaced")
        return decision

    def read_export(self) -> dict[str, Any]:
        """Return only serializable policy, registry and cached binding evidence."""
        effective_bytes = canonical_bytes(self.registry)
        return {
            "policy": POLICY,
            "registry_sha256": hashlib.sha256(effective_bytes).hexdigest(),
            "registry_inputs": self._registry_inputs,
            "registry": self.registry,
            "verified": dict(self._verified),
            "bindings": [
                self._bindings[key]
                | {"pid": key[0], "birth": key[1], "uid": key[2], "command": key[3]}
                for key in sorted(self._bindings)
            ],
            "binding_history": [
                self._binding_history[key]
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


def classify_exported(row: dict[str, Any], registry_export: dict[str, Any]) -> dict[str, Any]:
    """Replay one cached desktop decision without running identity tools.

    Reducers use this to prove an exported ``desktop_app`` row was present in
    the frozen off-tick binding cache.  It intentionally cannot manufacture a
    new admission: a different PID, birth, UID or executable is unknown.
    """
    if registry_export.get("policy") != POLICY:
        return {"kind": "unknown", "reason": "desktop_registry_policy_mismatch"}
    try:
        key = DesktopRegistry._key(row)
    except ValueError:
        return {"kind": "unknown", "reason": "desktop_process_identity_missing"}
    command = key[3]
    classification = row.get("classification")
    if isinstance(classification, dict) and classification.get("identity_binding_changed"):
        return {
            "kind": "unknown",
            "reason": str(classification.get("reason", "identity_binding_changed")),
            "identity_binding_changed": True,
        }
    registry = registry_export.get("registry", {})
    if row.get("known_workload") or row.get("workload"):
        return {"kind": "agent_workload", "reason": "known_workload"}
    for exclusion in registry.get("workload_exclusions", []):
        if exclusion.get("exact_executable") == command:
            return {"kind": "agent_workload", "reason": "workload_exclusion"}
    name = Path(command).name.lower()
    verified_gui_main = command in registry_export.get("verified", {}) and Path(command).name in {
        "Claude",
        "Codex",
    }
    if (
        (name in _WORKLOAD_NAMES and not verified_gui_main)
        or re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", name)
        or _WORKLOAD_TEXT.search(command)
    ):
        return {"kind": "agent_workload", "reason": "workload_exclusion"}
    if command.startswith(SYSTEM_EXECUTABLE_ROOTS) and ".." not in Path(command).parts:
        return {"kind": "macos_service", "reason": "system_executable_root"}
    if key[2] != registry.get("owner_uid"):
        return {
            "kind": "unknown",
            "reason": "desktop_registry_uid_mismatch",
            "identity_changed": True,
        }
    for exclusion in registry.get("workload_exclusions", []):
        if exclusion.get("exact_executable") == command:
            return {"kind": "agent_workload", "reason": "workload_exclusion"}
    for binding in [
        *registry_export.get("bindings", []),
        *registry_export.get("binding_history", []),
    ]:
        if (
            binding.get("pid"),
            binding.get("birth"),
            binding.get("uid"),
            binding.get("command"),
        ) != key:
            continue
        identity = binding.get("identity")
        if not isinstance(identity, dict):
            return {"kind": "unknown", "reason": "desktop_binding_identity_missing"}
        if registry_export.get("verified", {}).get(identity.get("executable")) != identity:
            return {"kind": "unknown", "reason": "desktop_binding_not_verified"}
        matching_role = None
        for app in registry.get("applications", []):
            if app.get("id") != binding.get("entry"):
                continue
            for role in app.get("executables", []):
                if role.get("role") == binding.get("role"):
                    matching_role = (app, role)
                    break
        if matching_role is None:
            return {"kind": "unknown", "reason": "desktop_binding_role_not_enrolled"}
        app, role = matching_role
        relative = Path(str(role.get("relative_executable_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            return {"kind": "unknown", "reason": "desktop_binding_role_path_malformed"}
        expected_executable = str(Path(str(app.get("bundle_path", ""))) / relative)
        sealed = registry.get("signed_identities", {}).get(role.get("sha256"), role)
        if (
            binding.get("application") != app.get("display_name")
            or identity.get("role") != role.get("role")
            or identity.get("sha256") != role.get("sha256")
            or identity.get("designated_requirement") != sealed.get("designated_requirement")
            or identity.get("cdhash") != sealed.get("cdhash")
            or identity.get("signing_authorities") != sealed.get("signing_authorities")
            or identity.get("executable") != command
            or command != expected_executable
            or identity.get("bundle_path") != app.get("bundle_path")
            or identity.get("bundle_id") != app.get("bundle_id")
            or identity.get("version") != app.get("version")
            or identity.get("build") != app.get("build")
            or identity.get("team_identifier") != app.get("team_identifier")
            or role.get("role") not in _ELIGIBLE_ROLES
        ):
            return {"kind": "unknown", "reason": "desktop_binding_identity_mismatch"}
        return {
            "kind": "desktop_renderer" if "renderer" in str(binding.get("role")) else "desktop_app",
            "reason": binding.get("reason"),
            "entry": binding.get("entry"),
            "application": binding.get("application"),
            "role": binding.get("role"),
            "identity": identity,
        }
    return {"kind": "unknown", "reason": "desktop_process_identity_not_bound"}
