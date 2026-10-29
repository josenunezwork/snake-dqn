"""Regression tests for the sandbox's background sparse-source adapter."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.ops import dispatch
from tools.sandbox import source_delivery


def _plan() -> dict:
    return {"commit": "a" * 40, "tree": "b" * 40, "files": {}}


class LocalContentsCLI:
    """Small local Contents/RPC stand-in that executes only launch code."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.session = "test-session"
        self.deadline = None
        self.priced_deadline = None
        self._control_context = SimpleNamespace(deadline=float("inf"))
        self.calls: list[str] = []
        self.rpc_elapsed: list[float] = []
        self.terminal_present_after_rpc: list[bool] = []
        self.operations: list[str] = []

    def operation(self, name: str):
        cli = self

        class Operation:
            def __enter__(self):
                cli.operations.append(name)

            def __exit__(self, *_args):
                return False

        return Operation()

    def recoverable_rpc(self, code: str, timeout: float = 300.0):
        self.calls.append(code)
        namespace: dict[str, object] = {}
        started = time.monotonic()
        exec(compile(code, "<local-launch>", "exec"), namespace)
        self.rpc_elapsed.append(time.monotonic() - started)
        self.terminal_present_after_rpc.append(
            (self.root / ".snake-source-delivery-terminal.json").exists()
        )
        return namespace["result"]

    def call(
        self, operation: str, _session_flag: str, _session: str, remote: str, local: str, **_kw
    ):
        assert operation == "download"
        path = Path(remote)
        if not path.is_file():
            raise FileNotFoundError(remote)
        shutil.copyfile(path, local)
        return ""


def _source(root: Path, body: str) -> None:
    root.mkdir()
    (root / "source.py").write_text(body, encoding="utf-8")


def test_background_launch_returns_before_slow_install_and_polls_contents(tmp_path: Path) -> None:
    root = tmp_path / "content"
    _source(
        root,
        "import time\n"
        "def install_fallback(root, plan):\n"
        "    time.sleep(2.5)\n"
        "    return {'transport':'fallback','commit':plan['commit'],'tree':plan['tree'],"
        "'inventory_verified':True}\n",
    )
    cli = LocalContentsCLI(root)
    started = time.monotonic()
    delivery = source_delivery.install_background(cli, str(root), _plan(), cli.recoverable_rpc)
    elapsed = time.monotonic() - started

    assert delivery["transport"] == "fallback"
    assert source_delivery.launch_code(str(root), _plan()) in cli.calls
    assert cli.operations == ["source-delivery-launch"]
    # The foreground execution itself only writes/spawns. The measured total
    # includes the intentionally slow child, proving polling did not rerun it.
    assert elapsed >= 2.0
    assert len(cli.calls) == 1
    assert cli.rpc_elapsed[0] < 0.25
    assert cli.terminal_present_after_rpc == [False]


def test_background_worker_error_is_propagated(tmp_path: Path) -> None:
    root = tmp_path / "content"
    _source(root, "def install_fallback(root, plan):\n    raise RuntimeError('bad archive')\n")
    cli = LocalContentsCLI(root)

    with pytest.raises(source_delivery.SourceDeliveryError, match="bad archive"):
        source_delivery.install_background(cli, str(root), _plan(), cli.recoverable_rpc)


def test_background_terminal_timeout_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "content"
    _source(root, "def install_fallback(root, plan):\n    return {}\n")
    cli = LocalContentsCLI(root)
    monkeypatch.setattr(source_delivery, "_MAX_WAIT_SECONDS", 0.01)

    with pytest.raises(source_delivery.SourceDeliveryError, match="timeout"):
        source_delivery.install_background(
            cli, str(root), _plan(), lambda *_args, **_kwargs: {"pid": 1}
        )


def test_corrupt_or_nonidentity_completion_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "content"
    _source(
        root,
        "def install_fallback(root, plan):\n"
        "    return {'transport':'fallback','commit':'wrong','tree':plan['tree'],"
        "'inventory_verified':True}\n",
    )
    cli = LocalContentsCLI(root)

    with pytest.raises(source_delivery.SourceDeliveryError, match="identity mismatch"):
        source_delivery.install_background(cli, str(root), _plan(), cli.recoverable_rpc)


def test_real_qualified_installer_refuses_corrupt_archive_in_background(tmp_path: Path) -> None:
    """The child imports the unmodified qualified installer and reports its refusal."""
    root = tmp_path / "content"
    root.mkdir()
    shutil.copyfile(Path("research/colab_cli/source.py"), root / "source.py")
    archive = root / "source.tar.gz"
    archive.write_bytes(b"not a gzip archive")
    plan = {
        "commit": "a" * 40,
        "tree": "b" * 40,
        "fallback_archive": "source.tar.gz",
        "fallback_sha256": "0" * 64,
        "files": {},
    }
    cli = LocalContentsCLI(root)

    with pytest.raises(
        source_delivery.SourceDeliveryError, match="fallback archive SHA-256 mismatch"
    ):
        source_delivery.install_background(cli, str(root), plan, cli.recoverable_rpc)


def test_expired_staging_cap_refuses_without_launch(tmp_path: Path) -> None:
    root = tmp_path / "content"
    _source(root, "def install_fallback(root, plan):\n    return {}\n")
    cli = LocalContentsCLI(root)
    cli._control_context.deadline = time.monotonic() - 0.01

    with pytest.raises(source_delivery.SourceDeliveryError, match="no remaining staging budget"):
        source_delivery.install_background(cli, str(root), _plan(), cli.recoverable_rpc)
    assert cli.calls == []


def test_exact_marker_routes_fetch_and_background_install_but_normal_rpc_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "content"
    _source(
        root,
        "def install_fallback(root, plan):\n"
        "    return {'transport':'fallback','commit':plan['commit'],'tree':plan['tree'],"
        "'inventory_verified':True}\n",
    )
    cli = LocalContentsCLI(root)
    original = cli.recoverable_rpc
    fetch = (
        "# SNAKE_SOURCE_DELIVERY\nimport importlib.util\nfrom pathlib import Path\n"
        f"root=Path('/content/test-source-route')\nplan={_plan()!r}\n"
        "s=importlib.util.spec_from_file_location('snake_source',root/'source.py')\n"
        "mod=importlib.util.module_from_spec(s)\ns.loader.exec_module(mod)\n"
        "result=mod.fetch(root,plan)"
    )
    install = fetch.replace("mod.fetch", "mod.install_fallback")
    original_parse = source_delivery.parse_delivery

    def local_parse(code: str):
        parsed = original_parse(code)
        if parsed is None:
            return None
        operation, _runtime_root, plan = parsed
        return operation, str(root), plan

    monkeypatch.setattr(source_delivery, "parse_delivery", local_parse)
    with source_delivery.background_sparse_route(cli):
        assert cli.recoverable_rpc(fetch)["transport"] == "fallback_required"
        assert cli.recoverable_rpc(install)["transport"] == "fallback"
        assert cli.recoverable_rpc("result={'ordinary': True}") == {"ordinary": True}
    assert cli.recoverable_rpc.__func__ is original.__func__


def test_parse_refuses_marker_bearing_near_match() -> None:
    with pytest.raises(source_delivery.SourceDeliveryError, match="template"):
        source_delivery.parse_delivery("# SNAKE_SOURCE_DELIVERY\nresult={'untrusted': True}")


def test_ambiguous_launch_is_not_replayed(tmp_path: Path) -> None:
    root = tmp_path / "content"
    _source(root, "def install_fallback(root, plan):\n    return {}\n")
    cli = LocalContentsCLI(root)
    attempts = 0

    def ambiguous(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("kernel acknowledgement unknown")

    with pytest.raises(RuntimeError, match="unknown"):
        source_delivery.install_background(cli, str(root), _plan(), ambiguous)
    assert attempts == 1


def test_expiry_while_waiting_stops_contents_polling(tmp_path: Path) -> None:
    root = tmp_path / "content"
    _source(root, "def install_fallback(root, plan):\n    return {}\n")
    cli = LocalContentsCLI(root)
    cli._control_context.deadline = time.monotonic() + 0.02

    def launched(*_args, **_kwargs):
        return {"pid": 1}

    started = time.monotonic()
    with pytest.raises(source_delivery.SourceDeliveryError, match="timeout"):
        source_delivery.install_background(cli, str(root), _plan(), launched)
    assert time.monotonic() - started < 0.20


def test_contents_http_404_is_pending_until_terminal_arrives(tmp_path: Path) -> None:
    root = tmp_path / "content"
    _source(root, "def install_fallback(root, plan):\n    return {}\n")
    cli = LocalContentsCLI(root)
    reads = 0
    terminal = {
        "status": "completed",
        "delivery": {
            "transport": "fallback",
            "commit": _plan()["commit"],
            "tree": _plan()["tree"],
            "inventory_verified": True,
        },
    }

    def launched(*_args, **_kwargs):
        return {"pid": 7}

    # The launch itself has no Contents call. Supply a terminal reader that
    # first reports the documented missing-file form, then returns completion.
    original_call = cli.call

    def contents_then_terminal(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 1:
            raise RuntimeError("HTTP 404: fixture completion is not yet visible")
        Path(args[-1]).write_text(json.dumps(terminal), encoding="utf-8")
        return ""

    cli.call = contents_then_terminal
    assert (
        source_delivery.install_background(cli, str(root), _plan(), launched)
        == terminal["delivery"]
    )
    cli.call = original_call


@pytest.mark.parametrize("value", [None, "wrong", True, 3])
def test_admission_refuses_invalid_background_source_delivery_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    packet = json.loads(
        Path("research/sandbox_a_resident/packet-train01-09/dispatch-manifest.json").read_text()
    )
    packet["sandbox_source_delivery"] = value

    with pytest.raises(dispatch.Refused, match="source delivery"):
        dispatch.require_sandbox_admission(packet, Path.cwd())


def test_admission_accepts_only_train01_background_source_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = json.loads(
        Path("research/sandbox_a_resident/packet-train01-09/dispatch-manifest.json").read_text()
    )
    packet["sandbox_source_delivery"] = source_delivery.OPT_IN
    bound = tmp_path / "bound-manifest.json"
    bound.write_text(json.dumps(packet), encoding="utf-8")
    from tools.sandbox import preflight

    monkeypatch.setattr(
        preflight,
        "preflight",
        lambda *_args, **_kwargs: {
            "study": "sandbox-a-train01",
            "outputs": {"broker_manifest": {"path": str(bound)}},
        },
    )
    monkeypatch.setattr(preflight, "run_focused_tests", lambda: None)
    monkeypatch.setattr(dispatch, "command", lambda *_args, **_kwargs: None)
    dispatch.require_sandbox_admission(packet, Path.cwd())
    packet["script"] = "research/sandbox_a_resident/job.sh"
    with pytest.raises(dispatch.Refused, match="exclusive to TRAIN-01"):
        dispatch.require_sandbox_admission(packet, Path.cwd())
