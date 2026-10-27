"""Pure, fail-closed identity replay for prospective supervision evidence.

The observer is responsible for independently capturing the *actual* OS identity,
source bytes and application payloads. This verifier never operates a process and
makes no security claim against a malicious same-UID evidence producer. Births are
OS creation nanoseconds; event times are monotonic nanoseconds, in separate domains.
A sealed registry is not evidence that a matching launch or response occurred.
"""

from __future__ import annotations

import hashlib
import math
import posixpath
import re
from typing import Any

POLICY = "classified-supervised-v2"
MAX_PAYLOAD_BYTES = 65536
TEMPLATES = {"usage", "sessions", "read_state", "renew_lease", "heartbeat_upload"}
MODULES = {
    "tools.sandbox.supervisor",
    "tools.sandbox.supervisor_v2",
    "research.colab_cli.supervisor",
}
QUALIFIED_SOURCES = {
    "research/colab_cli/supervisor.py": (
        "773d700a5abde40069d3ded6e4b378bbaa35dc44218740d1f621da23160369a0"
    ),
    "research/colab_cli/cli_guard.py": (
        "fd41ea264860cd1dc215ac4f5c14acff31bd62210dcb7ba71ad7b4e231ecafdc"
    ),
    "research/colab_cli/policy.py": (
        "df0853ea3ca984a1bcfb5c78a8890467b4a66047ea56d269af3cc736475254e3"
    ),
}
CALLSITES = {
    "usage": "CLI.usage",
    "sessions": "CLI.sessions",
    "read_state": "CLI.renew_runtime_lease",
    "renew_lease": "CLI.renew_runtime_lease",
    "heartbeat_upload": "CLI._put_lease",
    "bookkeeping": "Broker.execute:monitor-bookkeeping",
}
ROOT_FIELDS = {
    "id",
    "uid",
    "pid",
    "birth_ns",
    "boot_id",
    "interpreter",
    "module",
    "argv",
    "repo",
    "cwd",
    "state_dir",
    "source_hashes",
    "startup_hook_hash",
    "manifest",
    "manifest_digest",
    "assignment",
    "session",
    "generation",
    "colab",
    "config",
    "cli_interpreter",
    "cli_source_hashes",
    "cli_startup_hook_hash",
    "operation_timeout_seconds",
}
PROCESS_FIELDS = {
    "uid",
    "pid",
    "birth_ns",
    "interpreter",
    "argv",
    "cwd",
    "source_hashes",
    "startup_hook_hash",
}


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _path(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("/")
        and "\x00" not in value
        and posixpath.normpath(value) == value
        and not value.startswith("//")
    )


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _sources(value: Any) -> bool:
    return (
        bool(value)
        and isinstance(value, dict)
        and all(_path(path) and _sha(sha) for path, sha in value.items())
    )


def process_id(identity: dict[str, Any]) -> str:
    """Return the exact high-resolution OS identity key (never just a PID)."""
    return "%s@%s" % (identity.get("pid"), identity.get("birth_ns"))


def rpc_request(template: str, generation: str, nonce: str) -> str:
    """Regenerate the qualified compact RPC wrapper; no arbitrary code is accepted.

    This is byte-for-byte the pinned supervisor's renew_runtime_lease/rpc grammar.
    The randomly generated RPC nonce is constrained separately from executable code.
    """
    if template not in {"read_state", "renew_lease"}:
        raise ValueError("unpermitted RPC template")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", generation):
        raise ValueError("invalid generation")
    if not re.fullmatch(r"SNAKE_RPC_[0-9a-f]{32}", nonce):
        raise ValueError("invalid RPC nonce")
    remote_root = "/content/snake-cli/" + generation
    operation = (
        f"read_state({remote_root!r},{generation!r},renew=True,compact=True)"
        if template == "read_state"
        else f"renew_lease({remote_root!r},{generation!r})"
    )
    code = (
        "import importlib.util\n"
        f"runtime_path={remote_root + '/runtime.py'!r}\n"
        "s=importlib.util.spec_from_file_location('snake_runtime',runtime_path)\n"
        "mod=importlib.util.module_from_spec(s)\ns.loader.exec_module(mod)\n"
        f"result=mod.{operation}"
    )
    return (
        "import json, traceback\ntry:\n"
        + "\n".join("    " + line for line in code.splitlines())
        + f"\n    print({nonce!r} + json.dumps({{'ok':True,'value':result}}))\n"
        + "except BaseException as exc:\n"
        + f"    print({nonce!r} + json.dumps({{'ok':False,'error':repr(exc)}}))\n"
    )


def heartbeat_request(generation: str) -> str:
    """Return exactly json.dump's qualified, non-secret generation heartbeat."""
    import json

    return json.dumps({"generation": generation, "renewal_clock": "contents-mtime"})


def _registry_errors(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if registry.get("policy") != POLICY:
        errors.append("policy mismatch")
    for field in ("ruling_hash", "policy_source_hash"):
        if not _sha(registry.get(field)):
            errors.append("missing/invalid " + field)
    if not isinstance(registry.get("boot_id"), str) or not registry["boot_id"]:
        errors.append("missing host boot identity")
    if not all(_integer(registry.get(x)) for x in ("capture_start_ns", "capture_end_ns")):
        errors.append("missing capture monotonic brackets")
    elif registry["capture_end_ns"] <= registry["capture_start_ns"]:
        errors.append("invalid capture monotonic brackets")
    roots = registry.get("roots")
    if not isinstance(roots, list) or not roots:
        return errors + ["registry roots missing"]
    ids: set[str] = set()
    pids: set[int] = set()
    for root in roots:
        if not isinstance(root, dict) or not ROOT_FIELDS <= root.keys():
            errors.append("incomplete root identity")
            continue
        label = str(root.get("id"))
        if not isinstance(root["id"], str) or not root["id"] or root["id"] in ids:
            errors.append("duplicate/invalid root id")
        ids.add(label)
        if not all(_integer(root[x]) for x in ("uid", "pid", "birth_ns")):
            errors.append(label + ": invalid UID/PID/high-resolution birth")
        if _integer(root["pid"]) and root["pid"] in pids:
            errors.append("duplicate root PID")
        if _integer(root["pid"]):
            pids.add(root["pid"])
        if root["boot_id"] != registry.get("boot_id"):
            errors.append(label + ": root host boot mismatch")
        paths = (
            "interpreter",
            "repo",
            "cwd",
            "state_dir",
            "manifest",
            "colab",
            "config",
            "cli_interpreter",
        )
        if not all(_path(root[x]) for x in paths) or root["cwd"] != root["repo"]:
            errors.append(label + ": canonical repository/cwd/path mismatch")
        if root["module"] not in MODULES or root["argv"] != [
            root["interpreter"],
            "-m",
            root["module"],
            "--state-dir",
            root["state_dir"],
            "--colab",
            root["colab"],
            "run",
            root["manifest"],
        ]:
            errors.append(label + ": exact nine-entry root argv mismatch")
        for field in ("source_hashes", "cli_source_hashes"):
            if not _sources(root[field]):
                errors.append(label + ": invalid " + field)
        sources = root["source_hashes"] if isinstance(root["source_hashes"], dict) else {}
        required_sources = {
            posixpath.join(root["repo"], root["module"].replace(".", "/") + ".py"),
            posixpath.join(root["repo"], "research/colab_cli/supervisor.py"),
            posixpath.join(root["repo"], "research/colab_cli/policy.py"),
            posixpath.join(root["repo"], "research/colab_cli/cli_guard.py"),
        }
        if root["module"] == "tools.sandbox.supervisor_v2":
            required_sources.update(
                posixpath.join(root["repo"], "tools/sandbox", name)
                for name in ("collector_v2.py", "supervision_accounting.py", "supervisor.py")
            )
        for relative, expected_sha in QUALIFIED_SOURCES.items():
            if sources.get(posixpath.join(root["repo"], relative)) != expected_sha:
                errors.append(label + ": qualified source version mismatch")
        if not required_sources <= sources.keys():
            errors.append(label + ": actual worktree source binding incomplete")
        cli_sources = root["cli_source_hashes"]
        hook = posixpath.join(posixpath.dirname(root["config"]), "cli-hook/sitecustomize.py")
        guard = posixpath.join(root["repo"], "research/colab_cli/cli_guard.py")
        if (
            not isinstance(cli_sources, dict)
            or not {root["colab"], hook, guard} <= cli_sources.keys()
        ):
            errors.append(label + ": CLI startup source binding incomplete")
        elif cli_sources[hook] != root["cli_startup_hook_hash"] or cli_sources[
            guard
        ] != sources.get(guard):
            errors.append(label + ": CLI guard/hook source mismatch")
        expected_hook = (
            "import os,runpy\ntry:\n    runpy.run_path("
            + repr(guard)
            + ", run_name='snake_cli_guard')\nexcept BaseException:\n    os._exit(70)\n"
        )
        if root["cli_startup_hook_hash"] != hashlib.sha256(expected_hook.encode()).hexdigest():
            errors.append(label + ": CLI startup hook hash mismatch")
        if not all(
            _sha(root[x]) for x in ("startup_hook_hash", "cli_startup_hook_hash", "manifest_digest")
        ):
            errors.append(label + ": invalid source/manifest digest")
        for field in ("assignment", "session", "generation"):
            if not isinstance(root[field], str) or not root[field] or "\x00" in root[field]:
                errors.append(label + ": invalid " + field)
        if not isinstance(root["generation"], str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}", root["generation"]
        ):
            errors.append(label + ": unsafe generation")
        if (
            not _number(root["operation_timeout_seconds"])
            or root["operation_timeout_seconds"] > 120
        ):
            errors.append(label + ": unbounded operation deadline")
    return errors


def _operation(root: dict[str, Any], event: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    template = event.get("template")
    request = event.get("request_text")
    prefix = [
        root["cli_interpreter"],
        root["colab"],
        "--auth",
        "oauth2",
        "--config",
        root["config"],
    ]
    timeout = event.get("timeout_seconds")
    if not _number(timeout) or timeout > root["operation_timeout_seconds"]:
        errors.append("operation deadline outside frozen bound")
    if template in {"usage", "sessions", "heartbeat_upload"} and (
        not _number(timeout) or timeout > 120
    ):
        errors.append("account/heartbeat deadline exceeds qualified read timeout")
    if template in {"usage", "sessions"}:
        expected = ""
        argv = prefix + [template]
    elif template in {"read_state", "renew_lease"}:
        try:
            expected = rpc_request(template, root["generation"], event.get("nonce", ""))
        except (TypeError, ValueError):
            expected = None
            errors.append("invalid RPC nonce/generation")
        if not _path(event.get("payload_path")):
            errors.append("noncanonical RPC script path")
        rpc_timeout = event.get("rpc_timeout_seconds")
        if not _number(rpc_timeout) or rpc_timeout > 120 or not _number(timeout):
            errors.append("invalid RPC kernel deadline")
        elif timeout != rpc_timeout:
            errors.append("compact polling must retain zero completion grace")
        argv = prefix + [
            "exec",
            "-s",
            root["session"],
            "-f",
            event.get("payload_path"),
            "--timeout",
            str(rpc_timeout),
        ]
    elif template == "heartbeat_upload":
        expected = heartbeat_request(root["generation"])
        if not _path(event.get("payload_path")):
            errors.append("noncanonical heartbeat path")
        argv = prefix + [
            "upload",
            "-s",
            root["session"],
            event.get("payload_path"),
            "/content/snake-cli/" + root["generation"] + "/lease-heartbeat.json",
        ]
    else:
        expected = None
        argv = []
        errors.append("unpermitted operation (ordinary workload)")
    if request != expected or not isinstance(request, str):
        errors.append("operation application request violates fixed template")
    else:
        payload = request.encode("utf-8")
        if (
            event.get("request_bytes") != len(payload)
            or event.get("request_digest") != hashlib.sha256(payload).hexdigest()
        ):
            errors.append("request byte/digest mismatch")
        if len(payload) > MAX_PAYLOAD_BYTES:
            errors.append("operation payload exceeds 64 KiB")
    return errors, argv


def _verify_identity(
    registry: dict[str, Any],
    events: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    *,
    final: bool = True,
) -> dict[str, Any]:
    """Replay a sealed registry, witnessed lifecycle journal and actual observations.

    Events have contiguous ``seq`` (starting at 1), ``mono_ns`` and ``kind``.
    Root identity/observation rows carry ``root_id,identity`` with every frozen
    ROOT_FIELD unchanged plus ``policy_source_hash``. Launch/exec rows carry
    PROCESS_FIELDS in ``identity``, ``parent`` birth-key and ``operation_id``.
    Operations start with a fixed request and bounded timeout; their ends carry
    response_bytes/digest plus independent request_capture/response_capture rows
    ``{bytes,digest,independent:true}``. Response bytes are not persisted: account
    output can change and may contain private data. Empty output still needs an
    independent captured size and SHA. Root observations and helper observations
    include monotonic times; helpers must still have a live, previously observed
    parent at launch. Reparenting observations never create ancestry.

    ``final=False`` permits ongoing operations/helpers for online prefix replay;
    final replay requires exits and independent payload completion. All results
    remain explicit evidence eligibility, not proof an observer captured reality.
    """
    if isinstance(registry, dict):
        registry = dict(registry)
        for key in ("start_ns", "end_ns"):
            observation = registry.get("observation_" + key)
            capture = registry.get("capture_" + key)
            if capture is None and observation is not None:
                registry["capture_" + key] = observation
            elif observation is not None and capture != observation:
                return {
                    "valid": False,
                    "errors": ["capture/observation brackets mismatch"],
                    "verified_roots": [],
                    "verified_processes": {},
                    "verified_operations": {},
                    "lifecycles": {},
                }
    errors = _registry_errors(registry) if isinstance(registry, dict) else ["registry not object"]
    if errors:
        return {
            "valid": False,
            "errors": errors,
            "verified_roots": [],
            "verified_processes": {},
            "verified_operations": {},
            "lifecycles": {},
        }
    roots = {root["id"]: root for root in registry["roots"]}
    root_keys = {process_id(root): root["id"] for root in roots.values()}
    processes: dict[str, dict[str, Any]] = {}
    operations: dict[str, dict[str, Any]] = {}
    observed_roots: set[str] = set()
    root_last_seen: dict[str, int] = {}
    root_observations: dict[str, list[int]] = {key: [] for key in roots}
    phases: dict[str, list[dict[str, Any]]] = {key: [] for key in roots}

    def provenance_valid(root: dict[str, Any], phase: str, row: dict[str, Any]) -> bool:
        source = posixpath.join(root["repo"], "research/colab_cli/supervisor.py")
        return (
            row.get("provenance")
            == {
                "source": source,
                "source_hash": root["source_hashes"][source],
                "callsite": CALLSITES.get(phase),
                "independent": True,
            }
            and phase in CALLSITES
        )

    event_order = [
        (row.get("mono_ns", -1), row.get("seq", 0), row) for row in events if isinstance(row, dict)
    ]
    if len(event_order) != len(events):
        errors.append("malformed journal event")
    previous_time = -1
    for index, (time_ns, seq, _) in enumerate(event_order, 1):
        if seq != index or not _integer(time_ns) or time_ns < previous_time:
            errors.append("dropped/out-of-order lifecycle journal")
        previous_time = time_ns if _integer(time_ns) else previous_time
    if any(row.get("kind") == "observation" for _, _, row in event_order):
        errors.append("independent observation supplied inside lifecycle journal")
    combined = [
        (time_ns, seq * 2, False, row)
        for time_ns, seq, row in event_order
        if _integer(time_ns) and _integer(seq)
    ]
    for index, row in enumerate(observations):
        if not isinstance(row, dict) or not _integer(row.get("mono_ns")):
            errors.append("malformed identity observation")
            continue
        combined.append((row["mono_ns"], -(10**12) + index, True, {**row, "kind": "observation"}))

    def check_root(row: dict[str, Any], time_ns: int, independent: bool) -> None:
        root_id = row.get("root_id")
        root = roots.get(root_id)
        actual = row.get("identity")
        if root is None or not isinstance(actual, dict):
            errors.append("unregistered/missing root identity")
            return
        if any(actual.get(field) != root[field] for field in ROOT_FIELDS):
            errors.append("root identity/source/assignment changed: " + root_id)
            return
        if row.get("policy_source_hash") != registry["policy_source_hash"]:
            errors.append("root policy source mismatch")
            return
        if independent:
            observed_roots.add(root_id)
            root_last_seen[root_id] = time_ns
            root_observations[root_id].append(time_ns)
        key = process_id(root)
        processes.setdefault(
            key,
            {
                "id": key,
                "identity": actual,
                "root_id": root_id,
                "operation_id": None,
                "op_id": None,
                "parent": None,
                "parent_id": None,
                "launch_ns": time_ns,
                "exit_ns": None,
            },
        )

    pending_phases = []
    for time_ns, _, independent, row in sorted(combined, key=lambda item: (item[0], item[1])):
        kind = row.get("kind")
        if kind == "root_identity" or (kind == "observation" and row.get("root_id") is not None):
            check_root(row, time_ns, independent)
        elif kind == "operation_start":
            op_id = row.get("operation_id")
            root = roots.get(row.get("root_id"))
            if (
                not isinstance(op_id, str)
                or not op_id
                or op_id in operations
                or root is None
                or root["id"] not in observed_roots
            ):
                errors.append("operation lacks unique ID/admitted root")
                continue
            problems, argv = _operation(root, row)
            if not provenance_valid(root, row.get("template"), row):
                problems.append("operation lacks independently captured qualified callsite")
            errors.extend(problems)
            operations[op_id] = {
                **row,
                "id": op_id,
                "start_ns": time_ns,
                "end_ns": None,
                "expected_argv": argv,
                "processes": [],
            }
        elif kind in {"launch", "exec"}:
            identity = row.get("identity")
            if not isinstance(identity, dict) or not PROCESS_FIELDS <= identity.keys():
                errors.append("incomplete helper OS identity")
                continue
            key = process_id(identity)
            parent = processes.get(row.get("parent"))
            operation = operations.get(row.get("operation_id"))
            root = roots.get(operation.get("root_id")) if operation is not None else None
            if (
                parent is None
                or parent["exit_ns"] is not None
                or root is None
                or parent["root_id"] != root["id"]
                or operation["end_ns"] is not None
            ):
                errors.append("unobserved/reparented/foreign helper operation chain")
                continue
            if (
                time_ns - root_last_seen.get(root["id"], -(10**20)) > 5_000_000_000
                or identity["uid"] != root["uid"]
                or identity["interpreter"] != root["cli_interpreter"]
                or identity["argv"] != operation["expected_argv"]
                or identity["cwd"] != root["cwd"]
                or identity["source_hashes"] != root["cli_source_hashes"]
                or identity["startup_hook_hash"] != root["cli_startup_hook_hash"]
            ):
                errors.append("helper executable/argv/source/UID not an allowed exact operation")
                continue
            if not all(_integer(identity[x]) for x in ("uid", "pid", "birth_ns")):
                errors.append("invalid helper PID/UID/birth")
                continue
            if key in root_keys or any(
                proc["identity"]["pid"] == identity["pid"] and other_key != key
                for other_key, proc in processes.items()
            ):
                errors.append("PID reuse/root exec requires fresh admission")
                continue
            if kind == "launch":
                if key in processes:
                    errors.append("duplicate witnessed launch")
                    continue
                launch_start = row.get("launch_start_ns")
                launch_end = row.get("launch_end_ns")
                if (
                    not _integer(launch_start)
                    or not _integer(launch_end)
                    or not operation["start_ns"] <= launch_start <= launch_end <= time_ns
                ):
                    errors.append("missing/invalid witnessed process launch bracket")
                    continue
                processes[key] = {
                    "id": key,
                    "identity": identity,
                    "root_id": root["id"],
                    "operation_id": operation["id"],
                    "op_id": operation["id"],
                    "parent": row["parent"],
                    "parent_id": row["parent"],
                    "launch_ns": launch_start,
                    "launch_start_ns": launch_start,
                    "launch_end_ns": launch_end,
                    "launch_observed_ns": time_ns,
                    "exit_ns": None,
                }
                operation["processes"].append(key)
            else:
                old = processes.get(key)
                if (
                    old is None
                    or old["exit_ns"] is not None
                    or old["identity"] != identity
                    or old["parent"] != row["parent"]
                    or old["operation_id"] != operation["id"]
                ):
                    errors.append("ambiguous/unregistered helper exec transition")
        elif kind == "observation":
            identity = row.get("identity")
            key = process_id(identity) if isinstance(identity, dict) else "missing"
            proc = processes.get(key)
            if proc is None or proc["exit_ns"] is not None or identity != proc["identity"]:
                errors.append("unregistered/changed helper observation")
        elif kind == "exit":
            key = process_id(row)
            proc = processes.get(key)
            if proc is None or proc["exit_ns"] is not None or key in root_keys:
                errors.append("unwitnessed/duplicate exit or root restart")
            else:
                proc["exit_ns"] = time_ns
                proc["exit_record"] = dict(row)
                for field in (
                    "final_cpu_ns",
                    "final_scope",
                    "descendants_complete",
                    "final_inclusions",
                ):
                    if field in row:
                        proc[field] = row[field]
        elif kind == "operation_end":
            operation = operations.get(row.get("operation_id"))
            if operation is None or operation["end_ns"] is not None:
                errors.append("unmatched/duplicate operation completion")
                continue
            operation["end_ns"] = time_ns
            if time_ns - operation["start_ns"] > operation["timeout_seconds"] * 1_000_000_000:
                errors.append("operation final evidence exceeds qualified deadline")
            if not operation["processes"]:
                errors.append("operation has no witnessed permitted helper")
            for key in operation["processes"]:
                if processes[key]["exit_ns"] is None:
                    errors.append("operation completed before helper exit")
            for side in ("request", "response"):
                if side == "request":
                    aggregate_request = operation["request_text"].encode("utf-8") * len(
                        operation["processes"]
                    )
                    size = len(aggregate_request)
                    sha = hashlib.sha256(aggregate_request).hexdigest()
                    operation["accounted_request_bytes"] = size
                else:
                    size = row.get("response_bytes")
                    sha = row.get("response_digest")
                capture = row.get(side + "_capture")
                if (
                    not _integer(size)
                    or not _sha(sha)
                    or not isinstance(capture, dict)
                    or capture != {"bytes": size, "digest": sha, "independent": True}
                ):
                    errors.append("missing/mismatched independent " + side + " payload coverage")
            request_size = operation.get("accounted_request_bytes")
            response_size = row.get("response_bytes")
            if (
                _integer(request_size)
                and _integer(response_size)
                and request_size + response_size > MAX_PAYLOAD_BYTES
            ):
                errors.append("request plus response exceeds 64 KiB")
            operation.update(
                {
                    field: row.get(field)
                    for field in (
                        "response_bytes",
                        "response_digest",
                        "request_capture",
                        "response_capture",
                    )
                }
            )
        elif kind == "root_phase":
            root = roots.get(row.get("root_id"))
            phase = row.get("phase")
            if root is None or not provenance_valid(root, phase, row):
                errors.append("unrecognized in-process phase/callsite is ordinary workload")
                continue
            start_ns, end_ns = row.get("start_ns"), row.get("end_ns")
            if (
                not _integer(start_ns)
                or not _integer(end_ns)
                or end_ns <= start_ns
                or end_ns > time_ns
            ):
                errors.append("invalid phase coverage brackets")
                continue
            pending_phases.append(row)
        else:
            errors.append("unrecognized lifecycle journal event")
    if observed_roots != set(roots):
        errors.append("root identity never independently observed")
    capture_start, capture_end = registry["capture_start_ns"], registry["capture_end_ns"]
    if not final:
        capture_end = max((time_ns for time_ns, _, _, _ in combined), default=capture_start)
    # Resolve spans only after all completion rows are indexed. Online prefixes
    # may cover a live operation up to the earlier capture/deadline boundary.
    for row in pending_phases:
        if row["phase"] != "bookkeeping":
            operation = operations.get(row.get("operation_id"))
            stop = None if operation is None else operation["end_ns"]
            if operation is not None and stop is None and not final:
                stop = min(
                    capture_end,
                    operation["start_ns"] + int(operation["timeout_seconds"] * 1_000_000_000),
                )
            if (
                operation is None
                or operation["root_id"] != row["root_id"]
                or row["phase"] != operation["template"]
                or operation["start_ns"] > row["start_ns"]
                or stop is None
                or stop < row["end_ns"]
            ):
                errors.append("in-process phase not covered by authenticated operation")
                continue
        phases[row["root_id"]].append(row)
    for operation in operations.values():
        if (
            operation["end_ns"] is None
            and capture_end - operation["start_ns"] > operation["timeout_seconds"] * 1_000_000_000
        ):
            errors.append("ongoing operation exceeds qualified deadline")
    for root_id in roots:
        seen = sorted(root_observations[root_id])
        if (
            not seen
            or seen[0] > capture_start
            or seen[-1] < capture_end
            or any(right - left > 5_000_000_000 for left, right in zip(seen, seen[1:]))
        ):
            errors.append("root identity coverage missing/gapped through final bracket: " + root_id)
        covered_until = capture_start
        for phase in sorted(phases[root_id], key=lambda row: row["start_ns"]):
            if phase["start_ns"] > covered_until:
                break
            covered_until = max(covered_until, phase["end_ns"])
        if covered_until < capture_end:
            errors.append("uncovered root phase defaults to ordinary workload: " + root_id)
    if final:
        if any(proc["exit_ns"] is None for key, proc in processes.items() if key not in root_keys):
            errors.append("missing final helper exit")
        if any(operation["end_ns"] is None for operation in operations.values()):
            errors.append("missing operation completion/payload evidence")
    return {
        "valid": not errors,
        "errors": errors,
        "verified_roots": [roots[key] for key in sorted(observed_roots)],
        "verified_processes": processes,
        "verified_operations": operations,
        "lifecycles": processes,
        "verified_phases": phases,
    }


def validate_registry(registry: dict[str, Any]) -> dict[str, Any]:
    """Validate the strict schema; declarations alone never establish observation."""
    try:
        if not isinstance(registry, dict):
            return {"valid": False, "errors": ["registry not object"]}
        normalized = dict(registry)
        for key in ("start_ns", "end_ns"):
            capture = normalized.get("capture_" + key)
            observation = normalized.get("observation_" + key)
            if capture is None and observation is not None:
                normalized["capture_" + key] = observation
            elif observation is not None and capture != observation:
                return {"valid": False, "errors": ["capture/observation brackets mismatch"]}
        errors = _registry_errors(normalized)
        return {"valid": not errors, "errors": errors}
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return {"valid": False, "errors": ["malformed registry identity data"]}


def verify_identity(
    registry: dict[str, Any],
    events: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    *,
    final: bool = True,
) -> dict[str, Any]:
    """Verify the documented strict schema, refusing malformed evidence as data."""
    try:
        return _verify_identity(registry, events, observations, final=final)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return {
            "valid": False,
            "errors": ["malformed identity/lifecycle evidence"],
            "verified_roots": [],
            "verified_processes": {},
            "verified_operations": {},
            "lifecycles": {},
        }
