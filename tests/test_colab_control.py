"""Focused identity-boundary tests for qualified Colab control classification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.compute import colab_control as control


def _record(tmp_path: Path, *, pid: int = 10, birth: str = "100") -> control.AdmissionRecord:
    (tmp_path / "research/colab_cli").mkdir(parents=True)
    supervisor = tmp_path / "research/colab_cli/supervisor.py"
    supervisor.write_text("# qualified\n")
    source = tmp_path / "research/colab_cli/source.py"
    source.write_text("# source\n")
    release = tmp_path / "research/colab_cli/release.py"
    release.write_text("# release\n")
    cli = tmp_path / "colab"
    cli.write_text("#!/bin/sh\n")
    qualification = tmp_path / "qualification.json"
    qualification.write_text(json.dumps({"tree": "d" * 40}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    raw = {
        "policy_id": control.POLICY_ID,
        "ruling_sha256": control.RULING_SHA256,
        "operational_pin": "3c5fc403" + "a" * 32,
        "pid": pid,
        "birth": birth,
        "executable": "/venv/python",
        "interpreter": "/venv/python",
        "import_root": str(tmp_path),
        "module": control.SUPERVISOR_MODULE,
        "argv": [
            "/venv/python",
            "-m",
            control.SUPERVISOR_MODULE,
            "--state-dir",
            str(Path("/tmp/state").resolve()),
            "--colab",
            str(cli),
            "run",
            str(manifest),
        ],
        "source_closure": {
            "research/colab_cli/supervisor.py": hashlib.sha256(supervisor.read_bytes()).hexdigest(),
            "research/colab_cli/source.py": hashlib.sha256(source.read_bytes()).hexdigest(),
            "research/colab_cli/release.py": hashlib.sha256(release.read_bytes()).hexdigest(),
        },
        "qualified_broker_source_closure": "d" * 40,
        "qualification_record": str(qualification),
        "qualification_record_sha256": hashlib.sha256(qualification.read_bytes()).hexdigest(),
        "operational_root": str(tmp_path),
        "dispatch_source_closure": {
            str(supervisor): hashlib.sha256(supervisor.read_bytes()).hexdigest()
        },
        "manifest": str(manifest),
        "cli_executable": str(cli),
        "cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(),
        "cli_interpreter": str(cli),
        "cli_package_closure": {str(cli): hashlib.sha256(cli.read_bytes()).hexdigest()},
        "oauth_interpreter": "/oauth/python",
        "mint_program_sha256": "c" * 64,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "job_id": "job-a",
        "state_path": str(Path("/tmp/state").resolve()),
        "output_path": "/tmp/output",
    }
    admission = tmp_path / "admission.json"
    helper = str(tmp_path / "tools/ops/colab_control_launch.py")
    launcher = str(tmp_path / "research/colab_cli/launch.sh")
    raw["bootstrap_argv"] = [
        "/venv/python",
        helper,
        "--admission",
        str(admission),
        "--broker",
        str(tmp_path),
        "--launcher",
        launcher,
        "--",
        launcher,
        "--control-nice0",
        *raw["argv"][3:],
    ]
    admission.write_text(json.dumps(raw))
    raw["bootstrap"] = {
        "pid": pid,
        "birth": birth,
        "cwd": str(tmp_path),
        "executable": "/venv/python",
        "argv": raw["bootstrap_argv"],
    }

    raw["admission_path"] = str(admission)
    raw["admission_sha256"] = hashlib.sha256(admission.read_bytes()).hexdigest()
    path = tmp_path / "admission.root.json"
    path.write_text(json.dumps(raw))
    return control._record(path, raw)


def _rows(record: control.AdmissionRecord) -> list[dict[str, object]]:
    return [
        {
            "pid": 10,
            "ppid": 1,
            "birth": "100",
            "executable": "/venv/python",
            "argv_vector": record.argv,
        },
        {
            "pid": 11,
            "ppid": 10,
            "birth": "101",
            "executable": "/usr/local/bin/colab",
            "argv_vector": (
                "/usr/local/bin/colab",
                "--auth",
                "oauth2",
                "--config",
                "/tmp/config",
                "sessions",
            ),
        },
        {
            "pid": 12,
            "ppid": 10,
            "birth": "102",
            "executable": "/venv/python",
            "argv_vector": ("/venv/python", "-c", "print('numerical work')"),
        },
    ]


def test_admits_exact_supervisor_and_cli_but_leaves_numerical_child_visible(tmp_path: Path) -> None:
    record = _record(tmp_path)
    rows = _rows(record)
    rows[1]["executable"] = record.cli_executable
    rows[1]["argv_vector"] = (
        record.cli_executable,
        "--auth",
        "oauth2",
        "--config",
        str(Path("/tmp/state/job-a/sessions.json").resolve()),
        "sessions",
    )
    found = control.annotate(
        rows,
        {(record.pid, record.birth): record},
        cwd_reader=lambda _pid: str(tmp_path),
        source_checker=lambda _record, _cwd: True,
    )
    assert {pid: item.role for pid, item in found.items()} == {10: "supervisor", 11: "cli"}
    assert control.candidate_pids(_rows(record), {(record.pid, record.birth): record}) == {
        10,
        11,
        12,
    }


def test_readable_exec_replacement_is_ordinary_but_tree_remains_candidate(tmp_path: Path) -> None:
    record = _record(tmp_path)
    rows = _rows(record)
    rows[0]["argv_vector"] = ("/venv/python", "-c", "not supervisor")
    assert (
        control.annotate(
            rows,
            {(record.pid, record.birth): record},
            cwd_reader=lambda _pid: str(tmp_path),
            source_checker=lambda _record, _cwd: True,
        )
        == {}
    )
    assert control.candidate_pids(rows, {(record.pid, record.birth): record}) == {10, 11, 12}


def test_missing_native_identity_for_control_descendant_faults(tmp_path: Path) -> None:
    record = _record(tmp_path)
    rows = _rows(record)
    del rows[1]["birth"]
    with pytest.raises(control.IdentitySensorError, match="birth/executable"):
        control.annotate(
            rows,
            {(record.pid, record.birth): record},
            cwd_reader=lambda _pid: str(tmp_path),
            source_checker=lambda _record, _cwd: True,
        )


def test_argument_trick_and_stale_birth_are_not_admitted(tmp_path: Path) -> None:
    record = _record(tmp_path)
    rows = _rows(record)
    rows[1]["argv_vector"] = (
        "/usr/local/bin/colab",
        "--auth",
        "oauth2",
        "--config",
        "/tmp/config",
        "sessions;evil",
    )
    assert 11 not in control.annotate(
        rows,
        {(record.pid, record.birth): record},
        cwd_reader=lambda _pid: str(tmp_path),
        source_checker=lambda _record, _cwd: True,
    )
    rows[0]["birth"] = "reused"
    assert (
        control.annotate(
            rows,
            {(record.pid, record.birth): record},
            cwd_reader=lambda _pid: str(tmp_path),
            source_checker=lambda _record, _cwd: True,
        )
        == {}
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy_id", "forged-policy"),
        ("ruling_sha256", "0" * 64),
        ("operational_pin", "short"),
        ("manifest_sha256", "bad"),
        ("interpreter", "relative/python"),
    ],
)
def test_forged_record_fields_refuse_before_admission(
    tmp_path: Path, field: str, value: str
) -> None:
    record = _record(tmp_path)
    raw = json.loads(record.path.read_text())
    raw[field] = value
    with pytest.raises(ValueError):
        control._record(record.path, raw)


def test_pane_requires_live_bound_root_and_exact_pending_argv(tmp_path: Path) -> None:
    record = _record(tmp_path)
    raw = json.loads(record.path.read_text())
    raw["pane"] = {
        "pid": 9,
        "birth": "99",
        "executable": "/bin/bash",
        "cwd": str(tmp_path),
        "argv": ["/bin/bash", "-c", "checked", "colab-control-nice0"],
    }
    pending = json.loads(Path(record.admission_path).read_text())
    pending["pane_argv"] = raw["pane"]["argv"]
    Path(record.admission_path).write_text(json.dumps(pending))
    raw["admission_sha256"] = hashlib.sha256(Path(record.admission_path).read_bytes()).hexdigest()
    record.path.write_text(json.dumps(raw))
    pane_record = control._record(record.path, raw)
    rows = _rows(record)
    rows[0]["ppid"] = 9
    rows.append(
        {
            "pid": 9,
            "ppid": 1,
            "birth": "99",
            "executable": "/bin/bash",
            "argv_vector": tuple(raw["pane"]["argv"]),
        }
    )
    found = control.annotate(
        rows,
        {(pane_record.pid, pane_record.birth): pane_record},
        cwd_reader=lambda _pid: str(tmp_path),
        source_checker=lambda _record, _cwd: True,
    )
    assert found[9].role == "pane"


@pytest.mark.parametrize("phase", ["bootstrap", "launch_shell"])
def test_exact_launch_phases_are_admitted_and_exec_replacement_revokes(tmp_path, phase):
    record = _record(tmp_path)
    rows = _rows(record)
    raw = json.loads(record.path.read_text())
    if phase == "bootstrap":
        rows[0]["argv_vector"] = tuple(raw["bootstrap"]["argv"])
    else:
        rows[0]["executable"] = "/bin/bash"
        rows[0]["argv_vector"] = (
            "/bin/bash",
            str(tmp_path / "research/colab_cli/launch.sh"),
            "--control-nice0",
            *record.argv[3:],
        )
        rows.append(
            {
                "pid": 13,
                "ppid": 10,
                "birth": "103",
                "executable": record.interpreter,
                "argv_vector": (
                    record.interpreter,
                    str(tmp_path / "tools/ops/colab_control_launch.py"),
                    "--verify-launch",
                    str(record.path),
                    "--pid",
                    "10",
                    "--",
                    *record.argv[3:],
                ),
            }
        )
    found = control.annotate(
        rows,
        {(record.pid, record.birth): record},
        cwd_reader=lambda _: str(tmp_path),
        source_checker=lambda *_: True,
    )
    assert found[10].role == phase
    if phase == "launch_shell":
        assert found[13].role == "launch_verifier"
    rows[0]["argv_vector"] = (record.interpreter, "-c", "pass")
    assert not control.annotate(
        rows,
        {(record.pid, record.birth): record},
        cwd_reader=lambda _: str(tmp_path),
        source_checker=lambda *_: True,
    )


def test_readable_witness_mismatch_is_ordinary_but_unreadable_live_witness_faults(tmp_path):
    record = _record(tmp_path)
    raw = json.loads(record.path.read_text())
    raw["job_id"] = "wrong-job"
    record.path.write_text(json.dumps(raw))
    changed = control._record(record.path, raw)
    assert (
        control.annotate(
            _rows(changed),
            {(changed.pid, changed.birth): changed},
            cwd_reader=lambda _: str(tmp_path),
            source_checker=lambda *_: True,
        )
        == {}
    )
    Path(record.admission_path).unlink()
    with pytest.raises(control.IdentitySensorError):
        control.annotate(
            _rows(record),
            {(record.pid, record.birth): record},
            cwd_reader=lambda _: str(tmp_path),
            source_checker=lambda *_: True,
        )
