"""Keep offline source admission identical to the pre-compute execution gate."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from research.colab_cli import supervisor
from tools.ops import dispatch
from tools.sandbox import preflight, source_delivery
from tools.sandbox.supervisor import sandbox_admission_route

ROOT = Path(__file__).resolve().parents[3]
SUCCESSOR = "research/sandbox_a_resident/ct_h5000/job.sh"
SOURCE_ERROR = "background sparse source delivery is exclusive to TRAIN-01 and t34-cf-1028"
COMPLETION_ERROR = "training completion metadata is exclusive to TRAIN-01 and t34-cf-1028"


@pytest.mark.parametrize(
    "script,allowed",
    [
        ("research/sandbox_a_resident/train_job.sh", True),
        ("research/sandbox_a_resident/ct_job.sh", True),
        (SUCCESSOR, True),
        ("research/sandbox_a_resident/job.sh", False),
        ("research/sandbox_b_heads/run_job.sh", False),
        ("research/sandbox_b_heads/run_b2.sh", False),
        ("research/sandbox_b_heads/run_b3.sh", False),
        ("research/sandbox_a_resident/ct_h5000/other.sh", False),
        ("research/sandbox_a_resident/ct_h5000/job.sh.extra", False),
        ("research/sandbox_a_resident/ct_h5000/../ct_h5000/job.sh", False),
    ],
)
def test_validate_and_execute_share_exact_source_delivery_gate(
    monkeypatch: pytest.MonkeyPatch, script: str, allowed: bool
) -> None:
    """Stub transport, then exercise both real adapter entrypoints before I/O."""
    calls = []
    cli = object()
    broker = object()
    packet = {"script": script, "sandbox_source_delivery": source_delivery.OPT_IN}

    def validated(value, repo=None):
        calls.append("validate")
        return value

    def executed(self, actual_cli, value, *args, **kwargs):
        assert self is broker and actual_cli is cli and value is packet
        assert args == ("generation",) and kwargs == {"sentinel": True}
        calls.append("execute")
        return "result"

    @contextmanager
    def background(actual_cli):
        assert actual_cli is cli
        calls.append("enter")
        yield
        calls.append("exit")

    def no_provider(*args, **kwargs):
        raise AssertionError("admission must not call the provider")

    monkeypatch.setattr(supervisor, "validate", validated)
    monkeypatch.setattr(supervisor.Broker, "execute", executed)
    monkeypatch.setattr(supervisor.CLI, "_call_once", no_provider)
    monkeypatch.setattr(source_delivery, "background_sparse_route", background)
    with sandbox_admission_route():
        if allowed:
            assert supervisor.validate(packet, ROOT) is packet
            assert (
                supervisor.Broker.execute(broker, cli, packet, "generation", sentinel=True)
                == "result"
            )
        else:
            for action in (
                lambda: supervisor.validate(packet, ROOT),
                lambda: supervisor.Broker.execute(broker, cli, packet),
            ):
                with pytest.raises(ValueError) as error:
                    action()
                assert str(error.value) == SOURCE_ERROR
    assert calls == (["validate", "enter", "execute", "exit"] if allowed else [])
    assert supervisor.validate is validated
    assert supervisor.Broker.execute is executed


def test_non_opted_execution_remains_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    packet = {"script": "unrelated.sh"}
    monkeypatch.setattr(supervisor.Broker, "execute", lambda *args: calls.append(args))
    with sandbox_admission_route():
        supervisor.Broker.execute(None, None, packet)
    assert calls == [(None, None, packet)]


def test_smoke_gate_is_still_required_in_validation_and_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = {
        "script": "research/sandbox_a_resident/ct_cuda_canary_job.sh",
        "sandbox_source_delivery": source_delivery.OPT_IN,
    }

    def no_transport(*args, **kwargs):
        raise AssertionError("malformed smoke reached transport")

    monkeypatch.setattr(supervisor, "validate", no_transport)
    monkeypatch.setattr(supervisor.Broker, "execute", no_transport)
    with sandbox_admission_route():
        for action in (
            lambda: supervisor.validate(packet, ROOT),
            lambda: supervisor.Broker.execute(None, None, packet),
        ):
            with pytest.raises(ValueError, match="malformed C/T CUDA smoke manifest"):
                action()


@pytest.mark.parametrize("completion", [False, True])
def test_successor_reaches_bound_preflight_with_source_and_completion_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, completion: bool
) -> None:
    """The generic allow-lists defer successor completion validity to its preflight."""
    packet = json.loads(
        (
            ROOT / "research/sandbox_a_resident/packet-ct-h5000-1029-02/dispatch-manifest.json"
        ).read_text()
    )
    if completion:
        packet["completion_attempt"] = {"index": 1}
    bound = tmp_path / "bound.json"
    bound.write_text(json.dumps(packet))
    calls = []

    def bound_preflight(*args, **kwargs):
        return {
            "study": "t34-ct-h5000-1029",
            "outputs": {"broker_manifest": {"path": str(bound)}},
        }

    monkeypatch.setattr(preflight, "preflight", bound_preflight)
    monkeypatch.setattr(preflight, "run_focused_tests", lambda: None)
    monkeypatch.setattr(dispatch, "command", lambda *args, **kwargs: calls.append(args[0]))
    dispatch.require_sandbox_admission(packet, ROOT)
    assert calls[0][1:3] == ["-m", "research.sandbox_a_resident.ct_h5000.preflight"]
    # Admission must still refuse a packet changed beyond the bound manifest.
    packet["timeout_minutes"] += 1
    with pytest.raises(dispatch.Refused, match="differs from committed preflight binding"):
        dispatch.require_sandbox_admission(packet, ROOT)


@pytest.mark.parametrize(
    "script",
    [
        "research/sandbox_a_resident/job.sh",
        "research/sandbox_b_heads/run_job.sh",
        "research/sandbox_b_heads/run_b2.sh",
        "research/sandbox_b_heads/run_b3.sh",
    ],
)
def test_other_adopted_scripts_keep_completion_refusal(script: str) -> None:
    packet = json.loads(
        (
            ROOT / "research/sandbox_a_resident/packet-ct-h5000-1029-02/dispatch-manifest.json"
        ).read_text()
    )
    packet.update(script=script, completion_attempt={"index": 1})
    with pytest.raises(dispatch.Refused) as error:
        dispatch.require_sandbox_admission(packet, ROOT)
    assert str(error.value) == COMPLETION_ERROR
