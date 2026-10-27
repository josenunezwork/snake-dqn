"""Sparse-source fallback delivery that keeps bulk I/O out of the launch RPC.

This adapter is deliberately outside the qualified CLI.  It is enabled only by
the sandbox supervisor route after dispatch admission has accepted the exact
opt-in.  The foreground kernel RPC writes a small stdlib worker and starts it;
all archive validation and unpacking happens in that child.  Completion crosses
back to the host through a tiny, atomically-published JSON file read with
Contents, so polling never queues another kernel execution behind the worker.
"""

from __future__ import annotations

import ast
import base64
import json
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

OPT_IN = "background-sparse-v1"
_MARKER = "# SNAKE_SOURCE_DELIVERY"
_TERMINAL_NAME = ".snake-source-delivery-terminal.json"
_WORKER_NAME = ".snake-source-delivery-worker.py"
_MAX_WAIT_SECONDS = 120.0


class SourceDeliveryError(ValueError):
    """The foreground adapter cannot prove sparse source identity."""


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError) as exc:
        raise SourceDeliveryError("source delivery code has a nonliteral binding") from exc


def parse_delivery(code: str) -> tuple[str, str, dict[str, Any]] | None:
    """Accept only the qualified supervisor's known source-delivery template.

    Returning ``None`` leaves every non-source RPC on its original qualified
    route.  A marker with a near match is refused rather than guessed.
    """
    if _MARKER not in code.splitlines():
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SourceDeliveryError("malformed source delivery RPC") from exc
    # The qualified template has two imports, four assignments and exactly one
    # loader call.  Do not execute a marker-bearing near match under the
    # sandbox's alternate transport.
    if len(tree.body) != 8:
        raise SourceDeliveryError("unexpected source delivery template length")
    if not (
        isinstance(tree.body[0], ast.Import)
        and [(alias.name, alias.asname) for alias in tree.body[0].names]
        == [("importlib.util", None)]
        and isinstance(tree.body[1], ast.ImportFrom)
        and tree.body[1].module == "pathlib"
        and [(alias.name, alias.asname) for alias in tree.body[1].names] == [("Path", None)]
        and isinstance(tree.body[6], ast.Expr)
        and isinstance(tree.body[6].value, ast.Call)
        and isinstance(tree.body[6].value.func, ast.Attribute)
        and isinstance(tree.body[6].value.func.value, ast.Attribute)
        and isinstance(tree.body[6].value.func.value.value, ast.Name)
        and tree.body[6].value.func.value.value.id == "s"
        and tree.body[6].value.func.value.attr == "loader"
        and tree.body[6].value.func.attr == "exec_module"
        and len(tree.body[6].value.args) == 1
        and isinstance(tree.body[6].value.args[0], ast.Name)
        and tree.body[6].value.args[0].id == "mod"
    ):
        raise SourceDeliveryError("unexpected source delivery template")
    bindings: dict[str, Any] = {}
    operation: str | None = None
    for statement in tree.body[2:]:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if not isinstance(target, ast.Name):
                raise SourceDeliveryError("unsupported source delivery assignment")
            if target.id == "root":
                value = statement.value
                if not (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "Path"
                    and len(value.args) == 1
                ):
                    raise SourceDeliveryError("source delivery root is not Path(literal)")
                bindings["root"] = _literal(value.args[0])
            elif target.id == "plan":
                bindings["plan"] = _literal(statement.value)
            elif target.id == "s":
                value = statement.value
                if not (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Attribute)
                    and isinstance(value.func.value.value, ast.Name)
                    and value.func.value.value.id == "importlib"
                    and value.func.value.attr == "util"
                    and value.func.attr == "spec_from_file_location"
                    and len(value.args) == 2
                    and isinstance(value.args[0], ast.Constant)
                    and value.args[0].value == "snake_source"
                ):
                    raise SourceDeliveryError("unexpected source loader template")
            elif target.id == "mod":
                value = statement.value
                if not (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Attribute)
                    and isinstance(value.func.value.value, ast.Name)
                    and value.func.value.value.id == "importlib"
                    and value.func.value.attr == "util"
                    and value.func.attr == "module_from_spec"
                    and len(value.args) == 1
                    and isinstance(value.args[0], ast.Name)
                    and value.args[0].id == "s"
                ):
                    raise SourceDeliveryError("unexpected source module template")
            elif target.id == "result":
                value = statement.value
                if not (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name)
                    and value.func.value.id == "mod"
                    and value.func.attr in {"fetch", "install_fallback"}
                    and len(value.args) == 2
                    and all(isinstance(arg, ast.Name) for arg in value.args)
                    and [arg.id for arg in value.args] == ["root", "plan"]
                ):
                    raise SourceDeliveryError("unsupported source delivery operation")
                operation = value.func.attr
            else:
                raise SourceDeliveryError("unexpected source delivery binding")
    root, plan = bindings.get("root"), bindings.get("plan")
    if not isinstance(root, str) or not root.startswith("/content/") or not isinstance(plan, dict):
        raise SourceDeliveryError("source delivery lacks a safe root or plan")
    if (
        operation is None
        or not isinstance(plan.get("commit"), str)
        or not isinstance(plan.get("tree"), str)
    ):
        raise SourceDeliveryError("source delivery lacks pinned identity")
    return operation, root, plan


def _worker_program() -> str:
    """Return the remote stdlib worker source; imports happen after Popen."""
    return r"""import importlib.util, json, os, sys, traceback
from pathlib import Path
root = Path(sys.argv[1])
plan = json.loads(sys.argv[2])
terminal = root / ".snake-source-delivery-terminal.json"
temporary = root / ".snake-source-delivery-terminal.tmp"
print("SNAKE_SOURCE_DELIVERY child-start", flush=True)
try:
    spec = importlib.util.spec_from_file_location("snake_source", root / "source.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("source loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    delivery = module.install_fallback(root, plan)
    value = {"status": "completed", "delivery": delivery}
    print("SNAKE_SOURCE_DELIVERY child-completed", flush=True)
except BaseException as exc:
    value = {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()[-2000:]}
    print("SNAKE_SOURCE_DELIVERY child-failed", flush=True)
with temporary.open("x") as handle:
    json.dump(value, handle, sort_keys=True)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, terminal)
"""


def launch_code(root: str, plan: dict[str, Any]) -> str:
    """Build the only foreground RPC: write worker bytes, Popen, return PID."""
    encoded = base64.b64encode(_worker_program().encode()).decode()
    return (
        "# SNAKE_SOURCE_DELIVERY_BACKGROUND_LAUNCH\n"
        "import base64,subprocess,sys\nfrom pathlib import Path\n"
        f"root=Path({root!r})\nplan={json.dumps(plan, sort_keys=True)!r}\n"
        f"worker=root/{_WORKER_NAME!r}\nterminal=root/{_TERMINAL_NAME!r}\n"
        "if terminal.exists(): raise RuntimeError('source delivery terminal already exists')\n"
        f"worker.write_bytes(base64.b64decode({encoded!r}))\n"
        "p=subprocess.Popen([sys.executable,str(worker),str(root),plan],"
        "stdout=(root/'source-delivery.log').open('ab'),stderr=subprocess.STDOUT,"
        "start_new_session=True)\nresult={'pid':p.pid}"
    )


def _read_terminal(cli: Any, remote: str, deadline: float) -> dict[str, Any] | None:
    """Read a completion record using Contents only; absence remains pending."""
    from research.colab_cli import policy

    with tempfile.NamedTemporaryFile("wb") as handle:
        try:
            cli.call(
                "download",
                "-s",
                cli.session,
                remote,
                handle.name,
                timeout=min(policy.READ_SECONDS, max(0.1, deadline - time.monotonic())),
            )
        except Exception as exc:
            from research.colab_cli import policy

            message = str(exc).lower()
            if (
                isinstance(exc, FileNotFoundError)
                or "not found" in message
                or "no such file" in message
                or re.search(r"\bhttp\s*404\b", message) is not None
            ):
                return None
            if policy.transient_error(exc):
                return None
            raise SourceDeliveryError(
                f"source-delivery terminal Contents read failed: {exc}"
            ) from exc
        try:
            value = json.loads(Path(handle.name).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceDeliveryError("invalid source-delivery terminal record") from exc
    if not isinstance(value, dict):
        raise SourceDeliveryError("invalid source-delivery terminal type")
    return value


def install_background(
    cli: Any, root: str, plan: dict[str, Any], launch_rpc: Callable[..., Any]
) -> dict[str, Any]:
    """Spawn fallback install once, then wait only on Contents terminal state."""
    started = time.monotonic()
    control_deadline = getattr(cli._control_context, "deadline", float("inf"))
    session_deadline = cli.deadline if cli.deadline is not None else float("inf")
    priced_deadline = cli.priced_deadline if cli.priced_deadline is not None else float("inf")
    deadline = min(started + _MAX_WAIT_SECONDS, control_deadline, session_deadline, priced_deadline)
    if deadline <= started:
        raise SourceDeliveryError("background source install has no remaining staging budget")
    with cli.operation("source-delivery-launch"):
        launched = launch_rpc(launch_code(root, plan), timeout=min(300.0, deadline - started))
    if not isinstance(launched, dict) or not isinstance(launched.get("pid"), int):
        raise SourceDeliveryError("source delivery launch did not acknowledge child PID")
    remote = root + "/" + _TERMINAL_NAME
    while time.monotonic() < deadline:
        terminal = _read_terminal(cli, remote, deadline)
        if terminal is None:
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
            continue
        if terminal.get("status") == "failed":
            raise SourceDeliveryError(f"background source install failed: {terminal.get('error')}")
        delivery = terminal.get("delivery")
        if terminal.get("status") != "completed" or not isinstance(delivery, dict):
            raise SourceDeliveryError("background source install has invalid terminal state")
        if (
            delivery.get("transport") != "fallback"
            or delivery.get("commit") != plan["commit"]
            or delivery.get("tree") != plan["tree"]
            or delivery.get("inventory_verified") is not True
        ):
            raise SourceDeliveryError("background source install identity mismatch")
        return delivery
    raise SourceDeliveryError("background source install terminal timeout")


@contextmanager
def background_sparse_route(cli: Any) -> Iterator[None]:
    """Route only exact source-delivery RPCs for an admitted sandbox CLI."""
    original: Callable[..., Any] = cli.recoverable_rpc

    def routed(code: str, timeout: float = 300.0) -> Any:
        parsed = parse_delivery(code)
        if parsed is None:
            return original(code, timeout=timeout)
        operation, root, plan = parsed
        if operation == "fetch":
            # Git fetch can synchronously consume the staging control limit.
            # The already-qualified sparse archive is uploaded and checked next.
            return {
                "transport": "fallback_required",
                "reason": "sandbox background sparse delivery",
            }
        return install_background(cli, root, plan, original)

    cli.recoverable_rpc = routed
    try:
        yield
    finally:
        cli.recoverable_rpc = original
