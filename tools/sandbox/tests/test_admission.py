"""Admission-route tests for the adopted A100 development sandbox."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from tools.ops import dispatch
from tools.sandbox import preflight
from tools.sandbox.supervisor import sandbox_admission_route

ROOT = Path(__file__).resolve().parents[3]
PACKET = ROOT / "research/sandbox_b_heads/dispatch-manifest.json"
TEN_LINE_MANIFEST = ROOT / "research/sandbox_b_heads/sandbox-manifest.json"


def _packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch | None = None) -> dict:
    """Build a sandbox packet from the committed Job B packet, with no probe."""
    value = json.loads(PACKET.read_text())
    output_root = tmp_path / "sandbox-artifacts"
    if monkeypatch is not None:
        monkeypatch.setitem(dispatch.SANDBOX_OUTPUT_ROOTS, "sandbox-b-heads-v1", output_root)
    value["output_dir"] = str(output_root / "output")
    value.pop("canary", None)
    value["env"] = {
        key: item
        for key, item in value.get("env", {}).items()
        if not key.startswith("SNAKE_COLAB_CANARY")
    }
    value["sandbox_admission"] = {
        "schema": "standing-a100-development-sandbox-admission/v1",
        "manifest_path": str(TEN_LINE_MANIFEST),
    }
    # The owner removed the canary's measured-cost ceiling. Keep this fixture
    # valid if a packet revision has not yet raised its corresponding cap.
    budget = value.setdefault("budget", {})
    rate = float(budget.get("rate_units_per_hour", 8.0))
    timeout = float(value.get("timeout_minutes", 120.0))
    budget["unit_cap"] = max(float(budget.get("unit_cap", 0.0)), timeout * rate / 60 + 1)
    return value


def _b2_packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Build a route-only B-2 packet without reading development results."""
    value = _packet(tmp_path, monkeypatch)
    b2_root = tmp_path / "sandbox-b2-artifacts"
    monkeypatch.setitem(dispatch.SANDBOX_OUTPUT_ROOTS, "sandbox-b-heads-v2", b2_root)
    value["job_id"] = "sandbox-b-heads-v2"
    value["script"] = "research/sandbox_b_heads/run_b2.sh"
    value["output_dir"] = str(b2_root / "a100-attempt-02")
    return value


def _b3_packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Build a route-only B-3 packet without reading development results."""
    value = _packet(tmp_path, monkeypatch)
    b3_root = tmp_path / "sandbox-b3-artifacts"
    monkeypatch.setitem(dispatch.SANDBOX_OUTPUT_ROOTS, "sandbox-b-heads-v3", b3_root)
    value["job_id"] = "sandbox-b-heads-v3"
    value["script"] = "research/sandbox_b_heads/run_b3.sh"
    value["output_dir"] = str(b3_root / "a100-attempt-01")
    return value


def _bind_preflight(
    monkeypatch: pytest.MonkeyPatch,
    packet: dict,
    tmp_path: Path,
    *,
    original: dict | None = None,
) -> list[tuple[Path, bool]]:
    """Stub only the committed-manifest filesystem check for a route test."""
    broker_manifest = tmp_path / "broker-manifest.json"
    broker_manifest.write_text(json.dumps(original or packet), encoding="utf-8")
    calls: list[tuple[Path, bool]] = []

    def fake_preflight(path: Path, *, run_tests: bool = True, **_kwargs):
        calls.append((path, run_tests))
        return {
            "study": "sandbox-b-heads-v1",
            "outputs": {"broker_manifest": {"path": str(broker_manifest)}},
        }

    monkeypatch.setattr(preflight, "preflight", fake_preflight)
    return calls


def test_sandbox_validation_keeps_checks_and_uses_bounded_focused_test_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sandbox admission may outlast ordinary host validation, but remains bounded."""
    manifest = tmp_path / "packet.json"
    manifest.write_text(json.dumps({"sandbox_admission": {}}), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_command(args: list[str], **kwargs: object) -> None:
        captured["args"] = args
        captured.update(kwargs)

    monkeypatch.setattr(dispatch, "command", fake_command)
    dispatch.validate(ROOT, tmp_path, tmp_path / "state", manifest, tmp_path / "validate.log")

    assert captured["args"] == [
        str(ROOT / "tools/sandbox/launch.sh"),
        "--state-dir",
        str(tmp_path / "state"),
        "--colab",
        str(Path.home() / ".local/bin/colab"),
        "validate",
        str(manifest),
    ]
    assert captured["timeout_seconds"] == dispatch.SANDBOX_ADMISSION_VALIDATE_TIMEOUT_SECONDS
    assert captured["timeout_seconds"] > dispatch.COMMAND_TIMEOUT_SECONDS


def test_ordinary_validation_retains_standard_host_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The larger admission window cannot apply to ordinary qualified jobs."""
    manifest = tmp_path / "packet.json"
    manifest.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        dispatch,
        "command",
        lambda _args, **kwargs: captured.update(kwargs),
    )

    dispatch.validate(ROOT, tmp_path, tmp_path / "state", manifest, tmp_path / "validate.log")

    assert captured["timeout_seconds"] == dispatch.COMMAND_TIMEOUT_SECONDS


def test_extended_sandbox_timeout_still_respects_allocation_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An allocation lock deadline remains an upper bound for sandbox validation."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(dispatch.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        dispatch.subprocess,
        "run",
        lambda _args, **kwargs: captured.update(kwargs),
    )

    dispatch.command(
        ["true"],
        deadline=130.0,
        timeout_seconds=dispatch.SANDBOX_ADMISSION_VALIDATE_TIMEOUT_SECONDS,
    )

    assert captured["timeout"] == 30.0


def test_b2_is_limited_to_its_distinct_route_and_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _b2_packet(tmp_path, monkeypatch)
    calls = _bind_preflight(monkeypatch, packet, tmp_path)

    def b2_preflight(path: Path, *, run_tests: bool = True, **_kwargs):
        assert path == TEN_LINE_MANIFEST
        assert run_tests is True
        return {
            "study": "sandbox-b-heads-v2",
            "outputs": {"broker_manifest": {"path": str(tmp_path / "broker-manifest.json")}},
        }

    monkeypatch.setattr(preflight, "preflight", b2_preflight)
    # B-2's entrypoint is deliberately not present in this test's committed
    # baseline. The route still invokes the qualified device guard in a real
    # packet after the source-bound commit is published.
    from research.colab_cli import canary

    device_checks: list[dict] = []
    monkeypatch.setattr(canary, "job_device", lambda value, _repo: device_checks.append(value))
    dispatch.require_sandbox_admission(packet, ROOT)

    assert calls == []
    assert device_checks == [packet]
    assert set(packet["env"]) == {
        "SANDBOX_B_SALT",
        "SNAKE_DQN_DEVICE",
        "COLAB_CLI_REQUIREMENTS",
    }


def test_b2_cannot_use_the_original_b1_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _b2_packet(tmp_path, monkeypatch)
    packet["output_dir"] = str(tmp_path / "sandbox-artifacts" / "a100-attempt-02")

    with pytest.raises(dispatch.Refused, match="isolated development artifact root"):
        dispatch.require_sandbox_admission(packet, ROOT)


def test_b3_is_limited_to_its_distinct_route_and_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _b3_packet(tmp_path, monkeypatch)

    def b3_preflight(path: Path, *, run_tests: bool = True, **_kwargs):
        assert path == TEN_LINE_MANIFEST
        assert run_tests is True
        return {
            "study": "sandbox-b-heads-v3",
            "outputs": {"broker_manifest": {"path": str(tmp_path / "broker-manifest.json")}},
        }

    broker_manifest = tmp_path / "broker-manifest.json"
    broker_manifest.write_text(json.dumps(packet), encoding="utf-8")
    monkeypatch.setattr(preflight, "preflight", b3_preflight)
    from research.colab_cli import canary

    device_checks: list[dict] = []
    monkeypatch.setattr(canary, "job_device", lambda value, _repo: device_checks.append(value))
    dispatch.require_sandbox_admission(packet, ROOT)

    assert device_checks == [packet]
    packet["output_dir"] = str(tmp_path / "sandbox-b2-artifacts" / "a100-attempt-01")
    with pytest.raises(dispatch.Refused, match="isolated development artifact root"):
        dispatch.require_sandbox_admission(packet, ROOT)


def test_valid_sandbox_packet_is_admitted_without_family_canary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _packet(tmp_path, monkeypatch)
    calls = _bind_preflight(monkeypatch, packet, tmp_path)

    dispatch.require_sandbox_admission(packet, ROOT)

    assert "canary" not in packet
    assert calls == [(TEN_LINE_MANIFEST, True)]


@pytest.mark.parametrize("index", [0, 1])
def test_preferred_sandbox_accepts_only_derived_hardware_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int
) -> None:
    from research.compute.gpu_preference import prepare_preference, variants

    original = _packet(tmp_path, monkeypatch)
    prepare_preference(original)
    packet = variants(original)[index]
    _bind_preflight(monkeypatch, packet, tmp_path, original=original)
    dispatch.require_sandbox_admission(packet, ROOT)
    packet["env"]["SNAKE_DQN_DEVICE"] = "cpu"
    with pytest.raises(dispatch.Refused, match="differs from committed preflight binding"):
        dispatch.require_sandbox_admission(packet, ROOT)


def test_route_uses_real_supervisor_validate_and_only_mocks_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter must be exercised through the qualified supervisor validator."""
    packet = _packet(tmp_path, monkeypatch)
    calls = _bind_preflight(monkeypatch, packet, tmp_path)

    from research.colab_cli import supervisor

    with sandbox_admission_route():
        validated = supervisor.validate(packet, ROOT)

    assert validated["output_dir"] == packet["output_dir"]
    # The host adapter admits before input hashing, and the qualified validator
    # then rechecks the canary hook after its own normalization/defaults.
    assert calls == [(TEN_LINE_MANIFEST, True), (TEN_LINE_MANIFEST, True)]


def test_protected_sandbox_input_is_refused_before_qualified_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route must run sandbox path admission before supervisor input hashing."""
    from research.colab_cli import supervisor

    protected = tmp_path / "f5-input.npz"
    protected.write_bytes(b"read-bomb sentinel")
    packet = _packet(tmp_path, monkeypatch)
    packet["inputs"][0]["path"] = str(protected)
    hash_calls: list[Path] = []

    def hash_bomb(path: Path) -> str:
        hash_calls.append(path)
        raise AssertionError("qualified supervisor hashed a protected sandbox input")

    monkeypatch.setattr(supervisor, "sha256", hash_bomb)

    def no_preflight(*_args, **_kwargs):
        raise AssertionError("sandbox preflight should not run after protected-path refusal")

    monkeypatch.setattr(preflight, "preflight", no_preflight)
    with sandbox_admission_route():
        with pytest.raises(ValueError, match="protected"):
            supervisor.validate(packet, ROOT)

    assert hash_calls == []


def test_sandbox_refuses_registered_output_namespace_without_f5_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _packet(tmp_path, monkeypatch)
    packet["output_dir"] = str(tmp_path / "round-30" / "registered-output")

    with pytest.raises(dispatch.Refused, match="isolated development artifact root"):
        dispatch.require_sandbox_admission(packet, ROOT)


def test_sandbox_route_restores_original_canary_checker_and_keeps_receipt_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.colab_cli import canary

    ordinary_calls: list[dict] = []

    def original(manifest: dict, repo: Path) -> None:
        ordinary_calls.append(manifest)
        raise canary.CanaryError("dispatch requires a SHA-bound family canary receipt")

    monkeypatch.setattr(canary, "check", original)
    sandbox_calls: list[dict] = []

    def sandbox_admit(manifest: dict, _repo: Path) -> None:
        sandbox_calls.append(manifest)

    monkeypatch.setattr(dispatch, "require_sandbox_admission", sandbox_admit)
    ordinary = {"job_id": "ordinary"}
    sandbox = {"job_id": "sandbox", "sandbox_admission": {}}

    with sandbox_admission_route():
        canary.check(sandbox, ROOT)
        with pytest.raises(canary.CanaryError, match="SHA-bound family canary"):
            canary.check(ordinary, ROOT)

    assert sandbox_calls == [sandbox]
    assert ordinary_calls == [ordinary]
    assert canary.check is original


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("sandbox_admission"),
        lambda value: value["sandbox_admission"].update({"schema": "wrong/v1"}),
        lambda value: value["sandbox_admission"].update({"manifest_path": "relative.json"}),
        lambda value: value.update({"canary": {"mode": "probe"}}),
        lambda value: value["env"].update({"SNAKE_COLAB_CANARY": "1"}),
        lambda value: value["env"].update({"BL_P1_MODE": "y3-source"}),
        lambda value: value["source_paths"].append("research/bl_residual/f5_colab.sh"),
        lambda value: value["source_paths"].append("research/D4096/foreign.py"),
        lambda value: value["source_paths"].append("research/unrelated/foreign.py"),
        lambda value: value.update({"operational_admission": {"schema": "f5/v1"}}),
    ],
)
def test_sandbox_refuses_malformed_registered_or_protected_controls(
    tmp_path: Path, mutation, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _packet(tmp_path, monkeypatch)
    mutation(packet)

    with pytest.raises(ValueError):
        dispatch.require_sandbox_admission(packet, ROOT)


@pytest.mark.parametrize(
    "field",
    ["inputs", "source_paths", "commit", "required_exports"],
)
def test_sandbox_refuses_preflight_packet_or_input_mutation(
    tmp_path: Path, field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _packet(tmp_path, monkeypatch)
    original = copy.deepcopy(packet)
    if field == "inputs":
        packet["inputs"][0]["sha256"] = "0" * 64
    elif field == "source_paths":
        original["source_paths"].append("research/sandbox_b_heads/foreign.py")
    elif field == "commit":
        original["commit"] = "0" * 40
    else:
        original["required_exports"] = list(original["required_exports"]) + ["foreign.json"]
    _bind_preflight(monkeypatch, packet, tmp_path, original=original)

    with pytest.raises(dispatch.Refused, match="runtime packet differs"):
        dispatch.require_sandbox_admission(packet, ROOT)


def test_sandbox_refuses_invalid_exact_accounting_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _packet(tmp_path, monkeypatch)
    retry = copy.deepcopy(original)
    retry["job_id"] = f'{original["job_id"]}-retry-1'
    _bind_preflight(monkeypatch, retry, tmp_path, original=original)

    with pytest.raises(dispatch.Refused, match="runtime packet differs"):
        dispatch.require_sandbox_admission(retry, ROOT)


def test_sandbox_preserves_qualified_defaults_and_exact_accounting_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _packet(tmp_path, monkeypatch)
    retry = copy.deepcopy(original)
    retry["job_id"] = f'{original["job_id"]}-retry-1'
    retry["output_dir"] = original["output_dir"] + "-retry-1"
    for value in (original, retry):
        value["requirements"].pop("allowed_gpus", None)
        value["budget"].pop("reserve_units", None)
        value.pop("lease_seconds", None)
        value.pop("max_export_bytes", None)
    _bind_preflight(monkeypatch, retry, tmp_path, original=original)

    dispatch.require_sandbox_admission(retry, ROOT)


def test_qualified_supervisor_rejects_invalid_timeout_before_sandbox_admission(
    tmp_path: Path,
) -> None:
    from research.colab_cli import supervisor

    packet = _packet(tmp_path)
    packet["timeout_minutes"] = 0
    with pytest.raises(supervisor.Refused, match="timeout_minutes"):
        supervisor.validate(packet, ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output", "/tmp/big-lever/exp/f5-live"),
        ("session", "sandbox-D4096-live"),
    ],
)
def test_sandbox_waits_for_protected_active_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    """A sandbox cannot infer coexistence permission from a protected run."""
    home = tmp_path / "home"
    account = home / ".config/colab-cli/snake"
    account.mkdir(parents=True)
    session = value if field == "session" else "sandbox-live"
    record = {
        "endpoint": "protected-endpoint",
        "session": session,
        "rate_units_per_hour": 8.0,
        field: value,
    }
    (account / f"active-{session}.json").write_text(json.dumps(record), encoding="utf-8")
    manifest = tmp_path / "sandbox.json"
    manifest.write_text(
        json.dumps(
            {
                "job_id": "sandbox",
                "sandbox_admission": {},
                "budget": {"other_assignments": {}},
                "output_dir": str(tmp_path / "output"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(
        dispatch,
        "command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            _args[0], 0, stdout="[job] protected-endpoint | Hardware: A100\n", stderr=""
        ),
    )

    with pytest.raises(dispatch.Refused, match="protected F5/study assignment"):
        dispatch.refresh_assignments(manifest, tmp_path / "evidence.json", 10.0)
    assert not (tmp_path / "sandbox.live.json").exists()


def test_non_sandbox_refresh_keeps_existing_active_assignment_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    account = home / ".config/colab-cli/snake"
    account.mkdir(parents=True)
    record = {
        "endpoint": "owned-endpoint",
        "session": "ordinary-live",
        "rate_units_per_hour": 8.0,
    }
    (account / "active-ordinary-live.json").write_text(json.dumps(record), encoding="utf-8")
    manifest = tmp_path / "ordinary.json"
    manifest.write_text(
        json.dumps(
            {
                "job_id": "ordinary",
                "budget": {"other_assignments": {}},
                "output_dir": str(tmp_path / "output"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(
        dispatch,
        "command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            _args[0], 0, stdout="[job] owned-endpoint | Hardware: A100\n", stderr=""
        ),
    )

    refreshed = dispatch.refresh_assignments(manifest, tmp_path / "evidence.json", 10.0)

    assert json.loads(refreshed.read_text())["budget"]["other_assignments"] == {
        "owned-endpoint": 8.0
    }


def test_train01_admits_only_bounded_package_stall_timeout(monkeypatch):
    packet_path = ROOT / "research/sandbox_a_resident/packet-train01-06/dispatch-manifest.json"
    packet = json.loads(packet_path.read_text())
    monkeypatch.setattr(
        preflight,
        "preflight",
        lambda *a, **kw: {
            "study": "sandbox-a-train01",
            "outputs": {"broker_manifest": {"path": str(packet_path)}},
        },
    )
    calls = []
    monkeypatch.setattr(dispatch, "command", lambda *a, **kw: calls.append(a[0]))
    dispatch.require_sandbox_admission(packet, ROOT)
    assert packet["env"]["SANDBOX_A_STALL_TIMEOUT_SECONDS"] == "900"
    assert calls[0][2] == "research.sandbox_a_resident.training_preflight"


@pytest.mark.parametrize("timeout", ("0", "-1", "901", "inf", "nan", "disabled", 900))
def test_train01_refuses_disabled_or_excessive_stall_timeout(timeout):
    packet_path = ROOT / "research/sandbox_a_resident/packet-train01-06/dispatch-manifest.json"
    packet = json.loads(packet_path.read_text())
    packet["env"]["SANDBOX_A_STALL_TIMEOUT_SECONDS"] = timeout
    with pytest.raises(dispatch.Refused, match="stall timeout"):
        dispatch.require_sandbox_admission(packet, ROOT)


def test_stall_timeout_does_not_widen_other_sandbox_package_controls(tmp_path, monkeypatch):
    packet = _packet(tmp_path, monkeypatch)
    packet["env"]["SANDBOX_A_STALL_TIMEOUT_SECONDS"] = "900"
    with pytest.raises(dispatch.Refused, match="non-package controls"):
        dispatch.require_sandbox_admission(packet, ROOT)


@pytest.mark.parametrize("mutation", [None, "format", "packet", "missing"])
def test_train01_compact_export_controls_bind_exact_admitted_packet(monkeypatch, mutation):
    packet_path = ROOT / "research/sandbox_a_resident/packet-train01-15/dispatch-manifest.json"
    packet = json.loads(packet_path.read_text())
    packet["env"].update(
        SANDBOX_A_SNAPSHOT_FORMAT="compact", SANDBOX_A_PACKET_ID="packet-train01-15"
    )
    monkeypatch.setattr(
        preflight,
        "preflight",
        lambda *a, **kw: {
            "study": "sandbox-a-train01",
            "outputs": {"broker_manifest": {"path": str(packet_path)}},
        },
    )
    monkeypatch.setattr(dispatch, "read_json", lambda *_: packet)
    monkeypatch.setattr(dispatch, "command", lambda *a, **kw: None)
    if mutation == "format":
        packet["env"]["SANDBOX_A_SNAPSHOT_FORMAT"] = "raw"
    elif mutation == "packet":
        packet["env"]["SANDBOX_A_PACKET_ID"] = "packet-train01-16"
    elif mutation == "missing":
        del packet["env"]["SANDBOX_A_PACKET_ID"]
    if mutation:
        with pytest.raises(dispatch.Refused, match="compact export controls"):
            dispatch.require_sandbox_admission(packet, ROOT)
    else:
        dispatch.require_sandbox_admission(packet, ROOT)
