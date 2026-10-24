"""Deterministic C1 response-loss proof tests; no allocation or xdist."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

from research.colab_cli import cli_guard, supervisor
from research.colab_cli.failures import ClassifiedFailure, FailureClass
from research.colab_cli.rpc import request_code
from research.rpc_qualification import broker
from research.rpc_qualification.broker import InjectionObserver


class FakeCLI:
    """Small fake CLI preserving the post-send / same-ID distinction."""

    def __init__(self) -> None:
        self.call_records: list[dict[str, Any]] = []
        self.exec_uncertain = False
        self.oauth_python = "fake-python"
        self.config = Path("/tmp/fake-config.json")
        self.session = "fake-session"
        self.executions: dict[str, int] = {}
        self.values: dict[str, dict[str, str]] = {}
        self.renew_count = 0

        def original(*args: str, timeout: float) -> str:
            code = Path(args[args.index("-f") + 1]).read_text()
            request = re.search(r"# SNAKE_REQUEST_ID ([0-9a-f]+)", code)
            if request:
                request_id = request.group(1)
                self.executions.setdefault(request_id, 0)
                if not self.executions[request_id]:
                    self.executions[request_id] = 1
                    self.values[request_id] = {"unit": request_id}
            return "sent"

        self._call_once = original

    def remaining_seconds(self, allowance: float) -> float:
        return allowance

    def renew(self) -> None:
        self.renew_count += 1

    def rpc(self, code: str, timeout: float, *, request_id: str | None = None) -> dict[str, Any]:
        if self.exec_uncertain:
            raise supervisor.Refused("uncertain")
        if "injection-ready.json" in code:
            return {
                "ready": True,
                "value": {
                    "descriptors": [
                        {
                            "unit_id": f"u-{index}",
                            "path": f"/content/descriptor-{index}.json",
                            "request_id": f"{index:032x}",
                        }
                        for index in range(3)
                    ]
                },
            }
        if "execution_count':1" in code:
            descriptor = re.search(r"descriptor=Path\(([^)]+)\)", code)
            assert descriptor
            index = int(re.search(r"descriptor-(\d+)", descriptor.group(1)).group(1))
            return {
                "unit_id": f"u-{index}",
                "request_id": f"{index:032x}",
                "execution_count": 1,
                **{
                    key: {"path": f"/{key}-{index}", "sha256": "a" * 64, "size": 1}
                    for key in ("started", "done", "execution_started", "execution_done", "result")
                },
            }
        if "injection-proof.json" in code:
            return {"sha256": "b" * 64, "size": 1}
        assert request_id
        with tempfile.NamedTemporaryFile("w", suffix=".py") as handle:
            handle.write(f"# SNAKE_REQUEST_ID {request_id}\n" + code)
            handle.flush()
            try:
                self._call_once("exec", "-f", handle.name, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                self.call_records.append(
                    {"command": "exec", "status": "failed", "request_id": request_id}
                )
                self.exec_uncertain = True
                raise ClassifiedFailure(
                    FailureClass.CONTROL_RPC_TIMEOUT, "synthetic lost response"
                ) from exc
        self.call_records.append(
            {"command": "exec", "status": "completed", "request_id": request_id}
        )
        return self.values[request_id]


def test_post_send_loss_recovers_same_id_once_and_reissues_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeCLI()
    observer = InjectionObserver(tmp_path)
    planned = [f"{n:032x}" for n in range(3)]  # the fake runtime's descriptor IDs
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        broker,
        "download_file",
        lambda _cli, _remote, local, _size, _renew, **_kwargs: (
            Path(local).parent.mkdir(parents=True, exist_ok=True),
            Path(local).write_bytes(b"evidence"),
            "a" * 64,
        )[-1],
    )

    observer(
        "ready",
        {
            "cli": fake,
            "root": "/content/snake-cli/x",
            "manifest": {
                "env": {"RQ_MODE": "injection"},
                "rq": {"injection_manifest": {"request_ids": planned}},
            },
        },
    )

    proof = __import__("json").loads((tmp_path / "injection-proof.json").read_text())
    assert proof["status"] == "COMPLETE"
    # v2 §2.3: injected units use exactly the pre-dispatch request identities.
    assert [row["request_id"] for row in proof["injected_units"]] == planned
    assert len(proof["injected_units"]) == len(proof["injected_call_records"]) == 3
    assert all(row["injected"] is True for row in proof["injected_call_records"])
    assert set(fake.executions.values()) == {1}, "same request ID never re-executes its unit"
    assert fake.renew_count >= 6


def test_injection_failure_requires_independent_idle_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeCLI()
    observer = InjectionObserver(tmp_path)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *args, **kwargs: False)
    with pytest.raises(supervisor.Refused, match="independently prove idle"):
        observer._rpc_with_loss(fake, code="result={}", unit_id="u-0")
    assert fake._call_once.__name__ == "original", "adapter is restored before any later cleanup"


def test_only_the_post_send_record_is_tagged_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeCLI()
    fake.call_records.append({"command": "upload", "status": "failed", "failure_class": "UPLOAD"})
    observer = InjectionObserver(tmp_path)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *args, **kwargs: True)
    observer._rpc_with_loss(fake, code="result={}", unit_id="u-0")
    assert fake.call_records[0].get("injected") is None
    tagged = [row for row in fake.call_records if row.get("injected")]
    assert len(tagged) == 1 and tagged[0]["unit_id"] == "u-0"


def test_adapter_matches_the_real_request_code_header(tmp_path: Path) -> None:
    fake = FakeCLI()
    request_id = "a" * 32
    restore, state = InjectionObserver._arm_post_send_loss(fake, request_id)
    wrapped = request_code("result={'opaque': True}", request_id, "b" * 32, "SNAKE_RPC_test")
    path = tmp_path / "request.py"
    path.write_text(wrapped)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            fake._call_once("exec", "-f", str(path), timeout=1)
    finally:
        restore()
    assert state["post_send_loss"] is True
    assert fake.executions[request_id] == 1
