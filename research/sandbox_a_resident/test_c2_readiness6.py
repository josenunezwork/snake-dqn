"""Readiness-6 real observation twins and host-loss/reconciliation boundaries."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from research.colab_cli import supervisor
from research.sandbox_a_resident import training_prefix_recovery as prefix
from research.sandbox_a_resident import training_prefix_release as release
from research.sandbox_a_resident import training_resume as r
from research.sandbox_a_resident.test_training_prefix_recovery import (
    build,
    fixture,
    rewrite,
    verify,
)
from research.sandbox_a_resident.units import digest

HEADER = Path(__file__).parent / "fixtures/r6_readiness6/iteration-0208.json"


def release_capture(
    tmp: Path, output: Path, job: str, generation: str, *, other: bool = False
) -> Path:
    """Run the actual GET-only capture producer with the subprocess fake provider."""
    root = tmp / "release-provider"
    root.mkdir()
    account = root / "account"
    account.mkdir()
    if other:
        (account / "active-other-job.json").write_text(
            json.dumps({"endpoint": "other-endpoint", "session": "other-job"})
        )
        (account / "pending-unrelated-job.json").write_text("{}")
    remote = root / "remote"
    remote.mkdir()
    state = root / "provider-state.json"
    state.write_text(
        json.dumps(
            {
                "balance": 1000.0,
                "rate": 6.84 if other else 0.0,
                "sessions": (
                    {"other-endpoint": {"name": "other-job", "gpu": "A100"}} if other else {}
                ),
                "exec_count": {},
            }
        )
    )
    calls = root / "calls.jsonl"
    calls.touch()
    fake = root / "fake-colab"
    shutil.copyfile(Path(__file__).parents[2] / "tests/fixtures/fake_colab.py", fake)
    fake.chmod(0o700)
    stub = root / "stub/colab_cli"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text("")
    (stub / "client.py").write_text(
        "class Client:\n    def assign(self,*a,**k): pass\n"
        "    def _issue_request(self,*a,**k): pass\n"
    )
    assignment = root / "assignment.json"
    assignment.write_text(
        json.dumps(
            {
                "endpoint": "target-endpoint",
                "session": f"snake-{job}-{generation[:12]}",
                "generation": generation,
                "output": str(output.resolve()),
            }
        )
    )
    capture = root / "capture.json"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(supervisor, "ACCOUNT_ROOT", account)
        for key, val in {
            "FAKE_COLAB_STATE": str(state),
            "FAKE_COLAB_LOG": str(calls),
            "FAKE_COLAB_REMOTE": str(remote),
            "FAKE_COLAB_MODE": "normal",
            "SNAKE_COLAB_OAUTH_PYTHON": str(
                Path(__file__).parents[2] / "tests/fixtures/fake_colab_helper.py"
            ),
            "PYTHONPATH": str(stub.parent),
        }.items():
            mp.setenv(key, val)
        release.capture_prefix_release(
            assignment,
            output,
            job,
            generation,
            state_dir=root / "state",
            colab=str(fake),
            reconciliation_output=root / "reconciliation.json",
            capture_output=capture,
        )
    events = [json.loads(line) for line in calls.read_text().splitlines()]
    assert {row["argv"][4] for row in events} == {"usage", "sessions"}
    assert all(row["auth"] == "oauth2" for row in events)
    return capture


def test_real_r6_header_exact_fallback_admits_and_preserves_bytes() -> None:
    original = HEADER.read_bytes()
    provenance = json.loads((HEADER.parent / "provenance.json").read_text())
    assert len(original) == provenance["size"]
    assert digest(HEADER) == provenance["sha256"]
    value = json.loads(original)
    observation = value["metadata"]["metrics"]["compile"]["report"]["observation"]
    assert observation["status"] == "eager_fallback"
    assert observation["fallback_reason"] == r.OBSERVATION_EAGER_FALLBACK_REASON
    r._scan_json_faults(value, label="real_r6_header")
    r._scan_diagnostic_record_line(
        json.dumps(value),
        label="real_r6_header_line",
        expected_job_id=None,
        expected_generation=None,
    )
    assert json.loads(original) == value
    assert HEADER.read_bytes() == original


@pytest.mark.parametrize(
    "attack",
    [
        "different-reason",
        "reason-suffix",
        "different-status",
        "actor",
        "self-check",
        "root",
        "metrics",
        "nested-metadata",
        "dotted-key",
        "observation-extra-error",
        "observation-extra-string",
        "reason-without-status",
        "status-without-reason",
        "second-occurrence",
        "finite-false",
        "nan",
        "unknown-exception",
    ],
)
def test_real_header_fallback_exact_location_fault_twins(attack: str) -> None:
    value = json.loads(HEADER.read_text())
    observation = value["metadata"]["metrics"]["compile"]["report"]["observation"]
    pair = copy.deepcopy(observation)
    if attack == "different-reason":
        observation["fallback_reason"] = "NovelCompilerException('trip')"
    elif attack == "reason-suffix":
        observation["fallback_reason"] += " "
    elif attack == "different-status":
        observation["status"] = "compiled"
    elif attack == "actor":
        value["metadata"]["metrics"]["compile"]["report"]["actor"] = pair
    elif attack == "self-check":
        value["metadata"]["metrics"]["compile"]["self_check"] = pair
    elif attack == "root":
        value["diagnostic"] = pair
    elif attack == "metrics":
        value["metadata"]["metrics"]["other"] = pair
    elif attack == "nested-metadata":
        value = {"data": value}
    elif attack == "dotted-key":
        value = {"metadata.metrics.compile.report.observation": pair}
    elif attack == "observation-extra-error":
        observation["error"] = "NovelCompilerException('trip')"
    elif attack == "observation-extra-string":
        observation["note"] = r.OBSERVATION_EAGER_FALLBACK_REASON
    elif attack == "reason-without-status":
        observation.pop("status")
    elif attack == "status-without-reason":
        observation.pop("fallback_reason")
    elif attack == "second-occurrence":
        value["other"] = r.OBSERVATION_EAGER_FALLBACK_REASON
    elif attack == "finite-false":
        observation["finite"] = False
    elif attack == "nan":
        observation["loss"] = float("nan")
    else:
        value["error"] = "NovelScientificException('trip')"
    for action in (
        lambda: r._scan_json_faults(value, label="real_r6_header"),
        lambda: r._scan_diagnostic_record_line(
            json.dumps(value),
            label="real_r6_header_line",
            expected_job_id=None,
            expected_generation=None,
        ),
    ):
        with pytest.raises(r.ResumeRefused):
            action()


@pytest.mark.parametrize(
    "attack", ["alternate-clean-path", "canonical-symlink", "truncate", "change", "append"]
)
def test_canonical_full_log_fault_twins(tmp_path: Path, attack: str) -> None:
    fx = fixture(tmp_path)
    cert = tmp_path / "good.json"
    build(fx, cert)
    assert verify(fx, cert)["supervisor_log_full_sha256"] == digest(fx["log"])
    value = json.loads(cert.read_text())
    if attack in {"alternate-clean-path", "canonical-symlink"}:
        clone = tmp_path / "clean.log"
        shutil.copyfile(fx["log"], clone)
        with fx["log"].open("a") as stream:
            stream.write("FloatingPointError: late fault\nEXIT=1\n")
        value["supervisor_log_prefix"]["path"] = str(clone)
        if attack == "canonical-symlink":
            fx["log"].unlink()
            fx["log"].symlink_to(clone)
    elif attack == "truncate":
        fx["log"].write_bytes(fx["log"].read_bytes()[:-1])
    elif attack == "change":
        fx["log"].write_text(fx["log"].read_text().replace("PREFLIGHT_OK", "PREFLIGHT_NO"))
    else:
        with fx["log"].open("a") as stream:
            stream.write("healthy appended text\n")
    rewrite(cert, value)
    with pytest.raises(r.ResumeRefused, match="canonical|full size or SHA-256"):
        verify(fx, cert)
    if attack == "alternate-clean-path":
        fx["log"] = clone
        with pytest.raises(r.ResumeRefused, match="not canonical"):
            build(fx, tmp_path / "bad.json")


def test_get_only_reconciliation_allows_other_account_jobs(tmp_path: Path) -> None:
    fx = fixture(tmp_path)
    other = tmp_path / "other-capture"
    other.mkdir()
    fx["capture"] = release_capture(
        other, fx["output"], fx["manifest"]["job_id"], "b" * 32, other=True
    )
    assert build(fx, tmp_path / "good.json")["post_hoc_release_verified"] is True


@pytest.mark.parametrize("kind", ["runtime_timeout", "lease_expired"])
def test_host_loss_only_is_enforced_by_builder_and_verifier(tmp_path: Path, kind: str) -> None:
    fx = fixture(tmp_path)
    cert = tmp_path / "good.json"
    build(fx, cert)
    with pytest.raises(r.ResumeRefused, match="HOST-LOSS ONLY"):
        build(fx, tmp_path / "bad.json", kind)
    value = json.loads(cert.read_text())
    value["stop_kind"] = kind
    rewrite(cert, value)
    with pytest.raises(r.ResumeRefused, match="eligible stop"):
        verify(fx, cert)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_old_prefix_certificates_refuse(tmp_path: Path, version: str) -> None:
    fx = fixture(tmp_path)
    cert = tmp_path / "good.json"
    build(fx, cert)
    value = json.loads(cert.read_text())
    value["schema"] = "sandbox-a-train01-retained-export-prefix/" + version
    rewrite(cert, value)
    with pytest.raises(r.ResumeRefused, match="unsupported or malformed host ENOSPC"):
        verify(fx, cert)
    with pytest.raises(r.ResumeRefused, match="unsupported retained-prefix certificate"):
        prefix.verify_prefix_recovery(
            cert,
            fx["manifest"],
            digest(fx["output"] / "manifest.json"),
            output_root=fx["output"],
            source_control=fx["control"],
        )


@pytest.mark.parametrize(
    "attack",
    [
        "missing",
        "foreign-endpoint",
        "foreign-session",
        "target-active",
        "target-pending",
        "foreign-generation",
        "foreign-output",
        "record-hash",
        "wrong-producer",
        "wrong-operation",
        "different-status",
    ],
)
def test_reconciliation_binding_refuses_fault_twins(tmp_path: Path, attack: str) -> None:
    fx = fixture(tmp_path)
    cap = json.loads(fx["capture"].read_text())
    rec_path = Path(cap["broker_reconciliation"]["path"])
    record = json.loads(rec_path.read_text())
    assignment_path = Path(record["assignment_record"]["path"])
    assignment = json.loads(assignment_path.read_text())
    if attack == "missing":
        cap.pop("broker_reconciliation")
    elif attack == "foreign-endpoint":
        record["status"]["sessions"] = {"target-endpoint": "live"}
    elif attack == "foreign-session":
        record["status"]["sessions"] = {"renamed-endpoint": assignment["session"]}
    elif attack == "target-active":
        record["status"]["active"] = [assignment]
    elif attack == "target-pending":
        record["status"]["unresolved"] = [f"/account/pending-{assignment['session']}.json"]
    elif attack in {"foreign-generation", "foreign-output"}:
        assignment["generation" if attack == "foreign-generation" else "output"] = "foreign"
        assignment_path.write_text(json.dumps(assignment))
        record["assignment_record"].update(
            size=assignment_path.stat().st_size, sha256=digest(assignment_path)
        )
    elif attack == "record-hash":
        cap["broker_reconciliation"]["sha256"] = "0" * 64
    elif attack == "wrong-producer":
        record["producer"] = "hand-written assertion"
    elif attack == "wrong-operation":
        record["operation"] = "STOP"
    else:
        cap["status"] = {"sessions": {"different": "facts"}}
    if attack not in {"missing", "record-hash", "different-status"}:
        cap["status"] = record["status"]
        rewrite(rec_path, record)
        cap["broker_reconciliation"].update(size=rec_path.stat().st_size, sha256=digest(rec_path))
    rewrite(fx["capture"], cap)
    with pytest.raises(r.ResumeRefused):
        build(fx, tmp_path / "bad.json")
