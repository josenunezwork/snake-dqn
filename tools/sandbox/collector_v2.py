"""Prospective process-local subprocess collector; never attach to a live root.

Only the new supervisor_v2 entrypoint installs this facade. The qualified source
and the stdlib subprocess module remain unchanged. Every direct child, including
non-monitoring work, gets an operation and launch/exec/exit record. A direct wait4
final is descendant-inclusive: absent independent descendant/reaping evidence it
is explicitly incomplete and cannot open panel admission.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from research.bl_paced_identity.record import export_json, read_export
from research.bl_served_search import supervision_identity as identity
from tools.sandbox import supervision_accounting as accounting

VERSION = accounting.COLLECTOR_VERSION
POST_KILL_DRAIN_SECONDS = 0.25


class Collector:
    """Own immutable child evidence while preserving qualified control behavior."""

    def __init__(self, directory: Path, *, manifest_bytes: bytes | None = None) -> None:
        self.errors: list[str] = []
        self._lock = threading.RLock()
        self._active: set[str] = set()
        self.parent = self._birth(os.getpid())
        pid, birth = self.parent.split("@")
        self.root_id = f"pid-{pid}-birth-{birth}"
        context = {
            "version": VERSION,
            "policy": identity.POLICY,
            "root_process_id": self.parent,
            "capability": accounting.CAPABILITY,
            "sources": self.sources(),
            "startup": self.startup(manifest_bytes=manifest_bytes),
        }
        context_hash = accounting.sha256_bytes(accounting._canonical_json(context))
        self.journal = accounting.ImmutableJournal(directory, context_sha256=context_hash)
        export_json(directory / "collector.json", context)

    @staticmethod
    def startup(*, manifest_bytes: bytes | None = None) -> dict[str, Any]:
        """Publish non-secret identity before control starts, never attach later."""
        import psutil

        process = psutil.Process(os.getpid())
        argv = process.cmdline()
        cwd = Path(process.cwd()).resolve(strict=True)
        if cwd != Path(__file__).resolve().parents[2]:
            raise accounting.AccountingError("startup cwd differs from loaded collector checkout")
        exact_argv = len(argv) == 9 and argv[1:3] == ["-m", "tools.sandbox.supervisor_v2"]
        if manifest_bytes is not None and not exact_argv:
            raise accounting.AccountingError("startup manifest requires exact supervisor argv")
        result = {
            "pid": process.pid,
            "uid": process.uids().real,
            "birth_ns": round(process.create_time() * 1_000_000) * 1000,
            "argv": argv,
            "cwd": str(cwd),
            "interpreter": str(Path(process.exe()).resolve(strict=True)),
        }
        if exact_argv:
            path = Path(argv[-1]).resolve(strict=True)
            actual_bytes = path.read_bytes()
            if manifest_bytes is None or actual_bytes != manifest_bytes:
                raise accounting.AccountingError("startup manifest bytes changed or unavailable")
            result.update(
                manifest=str(path), manifest_sha256=accounting.sha256_bytes(manifest_bytes)
            )
            colab = Path(argv[6]).resolve(strict=True)
            result.update(
                colab=str(colab), colab_sha256=accounting.sha256_bytes(colab.read_bytes())
            )
        return result

    @staticmethod
    def sources() -> dict[str, str]:
        """Bind the exact future adapter/collector and qualified source closure."""
        repo = Path(__file__).resolve().parents[2]
        names = [
            "tools/sandbox/" + name
            for name in (
                "supervisor.py",
                "supervisor_v2.py",
                "collector_v2.py",
                "supervision_accounting.py",
                "launch_v2.sh",
                "source_delivery.py",
            )
        ]
        names += [
            "research/colab_cli/" + name
            for name in (
                "supervisor.py",
                "policy.py",
                "cli_guard.py",
                "admission.py",
                "canary.py",
                "release.py",
                "source.py",
            )
        ]
        names += ["tools/ops/dispatch.py", "tools/ops/dispatch.sh"]
        # Match the registry's full Python closure, including newly added files.
        names += [
            str(path.relative_to(repo))
            for directory in ("research/colab_cli", "tools/sandbox")
            for path in sorted((repo / directory).glob("*.py"))
        ]
        return {name: accounting.sha256_bytes((repo / name).read_bytes()) for name in names}

    @staticmethod
    def _birth(pid: int) -> str:
        import psutil

        return f"{pid}@{round(psutil.Process(pid).create_time() * 1_000_000) * 1000}"

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        """Journal failure invalidates evidence without interrupting safety work."""
        try:
            self.journal.append(kind, payload)
        except Exception as exc:
            with self._lock:
                self.errors.append(f"{kind}: {type(exc).__name__}")

    def metadata(self, argv: Any, timeout: Any) -> tuple[dict, bytes]:
        """Recognize only exact fixed requests at qualified Python callsites.

        This is wrapper provenance, never an independent OS observation. Unknown
        work is journaled with template='ordinary' so identity replay refuses it.
        """
        result = {
            "root_id": self.root_id,
            "template": "ordinary",
            "request_text": "",
            "request_bytes": 0,
            "request_digest": accounting.sha256_bytes(b""),
            "timeout_seconds": timeout,
        }
        ordinary = dict(result)
        frames = inspect.stack(context=0)
        try:
            calls = {
                f.function
                for f in frames
                if f.frame.f_globals.get("__name__") == "research.colab_cli.supervisor"
            }
            call = next(
                (
                    f.frame
                    for f in frames
                    if f.function == "_call_once"
                    and f.frame.f_globals.get("__name__") == "research.colab_cli.supervisor"
                ),
                None,
            )
            if call is None or not isinstance(argv, (tuple, list)):
                return result, b""
            cli = call.f_locals["self"]
            assignment = json.loads((cli.config.parent / "assignment.json").read_text())
            allocation = json.loads(Path(cli.assignment_record).read_text())
            if (
                assignment["session"] != cli.session
                or assignment["endpoint"] != allocation["endpoint"]
            ):
                return ordinary, b""
            generation = assignment["generation"]
            with Path(cli.binary).open("rb") as handle:
                shebang = shlex.split(handle.readline(4096)[2:].decode().strip())
            if len(shebang) != 1 or not Path(shebang[0]).is_absolute():
                return result, b""
            root = {
                "cli_interpreter": str(Path(shebang[0]).resolve()),
                "colab": str(Path(cli.binary).resolve()),
                "config": str(cli.config.resolve()),
                "session": cli.session,
                "generation": generation,
                "operation_timeout_seconds": 120,
            }
            # Qualified CLI uses a shebang path; normalize to its actual Python
            # argv only for template matching, while preserving actual spawn argv.
            normalized = [root["cli_interpreter"], *argv]
            request = ""
            command = argv[5] if len(argv) > 5 else ""
            if command in {"usage", "sessions"} and command in calls:
                template = command
            elif command == "upload" and "_put_lease" in calls:
                template = "heartbeat_upload"
                result["payload_path"] = argv[-2]
                request = Path(argv[-2]).read_text()
            elif command == "exec":
                request = Path(argv[9]).read_text()
                nonce = re.search(r"SNAKE_RPC_[0-9a-f]{32}", request)
                if nonce is None:
                    return ordinary, b""
                template = "read_state" if "result=mod.read_state(" in request else "renew_lease"
                # Use the pinned CLI's actual callsite contract: force=True on
                # renew_runtime_lease emits read_state in this qualified version.
                expected_fn = identity.CALLSITES[template].removeprefix("CLI.")
                if expected_fn not in calls:
                    return ordinary, b""
                result["payload_path"] = argv[9]
                result["nonce"] = nonce.group()
                result["rpc_timeout_seconds"] = (
                    int(argv[-1]) if re.fullmatch(r"[0-9]+", argv[-1]) else float(argv[-1])
                )
            else:
                return result, b""
            result.update(
                template=template,
                request_text=request,
                request_bytes=len(request.encode()),
                request_digest=accounting.sha256_bytes(request.encode()),
            )
            # _call_once clips a deadline by a few microseconds. Poll RPC grammar
            # keeps the frozen kernel timeout separately; actual call bound <= it.
            check = dict(result)
            if template in {"read_state", "renew_lease"}:
                if (
                    type(timeout) not in (int, float)
                    or not 0 < timeout <= result["rpc_timeout_seconds"]
                ):
                    return ordinary, b""
                check["timeout_seconds"] = result["rpc_timeout_seconds"]
                result["timeout_seconds"] = result["rpc_timeout_seconds"]
                result["subprocess_timeout_seconds"] = timeout
            problems, expected = identity._operation(root, check)
            if problems or normalized != expected:
                return ordinary, b""
            source = str(Path(call.f_globals["__file__"]).resolve())
            result["provenance"] = {
                "source": source,
                "source_hash": accounting.sha256_bytes(Path(source).read_bytes()),
                "callsite": identity.CALLSITES[template],
                "independent": True,
            }
            guard = str(Path(source).with_name("cli_guard.py"))
            hook = str(cli.config.parent / "cli-hook/sitecustomize.py")
            result["helper_sources"] = {
                name: accounting.sha256_bytes(Path(name).read_bytes())
                for name in (root["colab"], hook, guard)
            }
            result["helper_hook"] = result["helper_sources"][hook]
            return result, request.encode()
        except (KeyError, OSError, ValueError, TypeError, AttributeError):
            return ordinary, b""
        finally:
            del frames

    def run(
        self,
        args: Any,
        *,
        input: Any = None,
        capture_output: bool = False,
        timeout: float | None = None,
        check: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Use the stdlib run/communicate timeout sequence with our own Popen."""
        if input is not None:
            if kwargs.get("stdin") is not None:
                raise ValueError("stdin and input arguments may not both be used")
            kwargs["stdin"] = subprocess.PIPE
        if capture_output:
            if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
                raise ValueError("stdout/stderr may not be used with capture_output")
            kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pipes_requested = kwargs.get("stdout") == subprocess.PIPE and kwargs.get("stderr") in (
            subprocess.PIPE,
            subprocess.STDOUT,
        )
        # Capture raw pipe bytes before the qualified caller's text decoding and
        # newline translation, preserving exact application payload digests.
        text = kwargs.pop("text", None)
        universal = kwargs.pop("universal_newlines", None)
        if text is not None and universal is not None and bool(text) != bool(universal):
            raise subprocess.SubprocessError("Cannot disambiguate text/universal_newlines")
        encoding, errors = kwargs.pop("encoding", None), kwargs.pop("errors", None)
        text_mode = bool(text or universal or encoding or errors)
        encoding = encoding or (subprocess._text_encoding() if text_mode else None)
        input_bytes = (
            input.encode(encoding, errors or "strict") if text_mode and input is not None else input
        )
        arg_strings = [os.fspath(x) if isinstance(x, os.PathLike) else str(x) for x in args]
        oid = uuid.uuid4().hex
        metadata, request = self.metadata(arg_strings, timeout)
        self.emit("operation_start", {**metadata, "operation_id": oid})
        with self._lock:
            self._active.add(oid)
        launch_start = time.monotonic_ns()
        tracked = None
        output = error = None
        pipes_drained = False
        try:
            try:
                process = accounting.Wait4Popen(arg_strings, **kwargs)
            except BaseException as exc:
                self.errors.append("spawn/exec failed; final unavailable: " + type(exc).__name__)
                raise
            with process as child:
                try:
                    birth = self._birth(child.pid)
                    import psutil

                    proc = psutil.Process(child.pid)
                    # Successful Popen return witnesses exec via its CLOEXEC
                    # error pipe, even if the child is already a zombie. Resolve
                    # the exact shebang contract rather than losing fast children
                    # to a post-exit proc.exe()/cmdline() race.
                    executable = kwargs.get("executable") or arg_strings[0]
                    resolved = str(
                        Path(
                            shutil.which(
                                executable, path=(kwargs.get("env") or os.environ).get("PATH")
                            )
                            or executable
                        ).resolve()
                    )
                    actual_argv = [resolved, *arg_strings[1:]]
                    with Path(resolved).open("rb") as handle:
                        first = handle.readline(4096)
                    if first.startswith(b"#!"):
                        shebang = shlex.split(first[2:].decode().strip())
                        if len(shebang) != 1 or not Path(shebang[0]).is_absolute():
                            raise accounting.AccountingError("ambiguous shebang interpreter")
                        executable = str(Path(shebang[0]).resolve())
                        actual_argv = [executable, resolved, *arg_strings[1:]]
                    else:
                        executable = resolved
                    actual = {
                        "pid": child.pid,
                        "birth_ns": int(birth.split("@")[1]),
                        "uid": proc.uids().real,
                        "interpreter": executable,
                        "argv": actual_argv,
                        "cwd": str(Path(kwargs.get("cwd") or Path.cwd()).resolve()),
                        "source_hashes": metadata.get("helper_sources", {}),
                        "startup_hook_hash": metadata.get("helper_hook"),
                    }
                    tracked = accounting.TrackedChild(
                        oid,
                        child.pid,
                        actual["uid"],
                        tuple(arg_strings),
                        launch_start,
                        os.getpid(),
                        actual["birth_ns"],
                        actual,
                        self.parent,
                    )
                    self.emit(
                        "launch",
                        {
                            "operation_id": oid,
                            "parent": self.parent,
                            "identity": actual,
                            "launch_start_ns": launch_start,
                            "launch_end_ns": time.monotonic_ns(),
                        },
                    )
                    self.emit(
                        "exec", {"operation_id": oid, "parent": self.parent, "identity": actual}
                    )
                except Exception as exc:
                    self.errors.append("child identity unavailable: " + type(exc).__name__)
                    self.emit(
                        "launch",
                        {
                            "operation_id": oid,
                            "pid": child.pid,
                            "launch_start_ns": launch_start,
                            "identity_unavailable": True,
                        },
                    )
                    self.emit(
                        "exec",
                        {"operation_id": oid, "pid": child.pid, "identity_unavailable": True},
                    )
                try:
                    output, error = child.communicate(input_bytes, timeout=timeout)
                    pipes_drained = (
                        child.returncode is not None
                        and output is not None
                        and (kwargs.get("stderr") != subprocess.PIPE or error is not None)
                    )
                except subprocess.TimeoutExpired:
                    child.kill()
                    # Descendants may retain pipe FDs after the direct child
                    # dies. Bound cleanup, and never certify a post-kill drain.
                    try:
                        child.communicate(timeout=POST_KILL_DRAIN_SECONDS)
                    except subprocess.TimeoutExpired:
                        for pipe in (child.stdin, child.stdout, child.stderr):
                            if pipe is not None:
                                pipe.close()
                    raise
                except BaseException:
                    child.kill()
                    try:
                        child.wait()
                    except Exception as exc:
                        self.errors.append("wait4: " + type(exc).__name__)
                    raise
                finally:
                    try:
                        child.wait()
                    except Exception as exc:
                        self.errors.append("wait4: " + type(exc).__name__)
                    if child.final_rusage is None:
                        self.errors.append("missing final rusage")
                    if tracked is None:
                        self.emit(
                            "exit",
                            {
                                "operation_id": oid,
                                "pid": child.pid,
                                **(child.final_rusage or {}),
                                "final_cpu_ns": (
                                    sum(child.final_rusage.values())
                                    if child.final_rusage is not None
                                    else None
                                ),
                                "final_scope": "subtree",
                                "descendants_complete": False,
                                "identity_unavailable": True,
                                "returncode": child.returncode,
                            },
                        )
                    else:
                        try:
                            accounting.wait4_exit(self.journal, child, tracked)
                        except Exception as exc:
                            self.errors.append("final accounting: " + type(exc).__name__)
                            self.emit(
                                "exit",
                                {
                                    "operation_id": oid,
                                    "pid": tracked.pid,
                                    "birth_ns": tracked.birth_ns,
                                    "final_cpu_ns": None,
                                    "final_scope": "subtree",
                                    "descendants_complete": False,
                                    "accounting_unavailable": True,
                                    "returncode": child.returncode,
                                },
                            )
                code = child.returncode

            def decoded(value: bytes | None) -> str | bytes | None:
                if value is None or not text_mode:
                    return value
                return (
                    value.decode(encoding, errors or "strict")
                    .replace("\r\n", "\n")
                    .replace("\r", "\n")
                )

            stdout, stderr = decoded(output), decoded(error)
            if check and code:
                raise subprocess.CalledProcessError(code, args, output=stdout, stderr=stderr)
            return subprocess.CompletedProcess(args, code, stdout, stderr)
        finally:

            response_captured = pipes_requested and pipes_drained
            response = (output or b"") + (error or b"")
            if metadata["template"] in accounting.ALLOWED_TEMPLATES:
                with self._lock:
                    if not response_captured:
                        self.errors.append("permitted operation response not fully captured")
                    if len(request) + len(response) > accounting.MAX_APPLICATION_PAYLOAD_BYTES:
                        self.errors.append(
                            "permitted operation application payload exceeds 65536 bytes"
                        )
            try:
                accounting.record_operation_end(
                    self.journal,
                    oid,
                    request=request,
                    response=response,
                    response_captured=response_captured,
                )
            except Exception as exc:
                with self._lock:
                    self.errors.append("operation completion: " + type(exc).__name__)
            with self._lock:
                self._active.discard(oid)

    def check_output(self, args: Any, **kwargs: Any) -> Any:
        """Preserve check_output semantics without using stdlib's global Popen."""
        return self.run(args, stdout=subprocess.PIPE, check=True, **kwargs).stdout

    def seal(self) -> dict[str, Any]:
        """Seal evidence and an explicit incomplete descendant/observer status."""
        with self._lock:
            status = {
                "version": VERSION,
                "capability": accounting.CAPABILITY,
                "errors": list(self.errors),
                "outstanding_operations": sorted(self._active),
                "descendants_complete": False,
                "independent_observer_qualified": False,
            }
            status_hash = export_json(self.journal.directory / "status.json", status)
            self.journal.seal(status_sha256=status_hash)
            return status


class _SubprocessFacade:
    def __init__(self, collector: Collector) -> None:
        self.run = collector.run
        self.check_output = collector.check_output

    def __getattr__(self, name: str) -> Any:
        if name in {"Popen", "call", "check_call"}:
            raise RuntimeError("unsupported subprocess API in pinned collector source")
        return getattr(subprocess, name)


@contextmanager
def collector_route(collector: Collector) -> Iterator[None]:
    """Replace module references only in this future adapter's host process."""
    from research.colab_cli import admission, canary, cli_guard, release, source, supervisor
    from tools.ops import dispatch

    sources = collector.sources()
    for relative, expected in identity.QUALIFIED_SOURCES.items():
        if sources.get(relative) != expected:
            raise accounting.AccountingError("qualified supervisor source version mismatch")
    modules = [admission, canary, cli_guard, release, source, supervisor, dispatch]
    originals = [(module, module.subprocess) for module in modules]
    facade = _SubprocessFacade(collector)
    try:
        for module, _ in originals:
            module.subprocess = facade
        yield
    finally:
        for module, original in originals:
            module.subprocess = original


def read_collector(directory: Path) -> dict[str, Any]:
    """Load the sealed child journal for reducer input without granting admission."""
    context = read_export(directory / "collector.json")
    status = read_export(directory / "status.json")
    seal = read_export(directory / "COMPLETE.json")
    if (
        context.get("version") != VERSION
        or context.get("sources") != Collector.sources()
        or seal.get("collector_version") != VERSION
        or seal.get("context_sha256")
        != accounting.sha256_bytes(accounting._canonical_json(context))
        or seal.get("status_sha256") != accounting.sha256_bytes(accounting._canonical_json(status))
    ):
        raise accounting.AccountingError("collector version/source/context/status mismatch")
    return {
        "collector": context,
        "status": status,
        "supervision_journal": accounting.read_journal(directory),
    }
