"""The optional host launcher stays bounded and preserves process evidence."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from research.colab_queue.core import validate_job

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "research/q0q1_20261017/run_host_contracts.sh"
JOB = ROOT / "research/q0q1_20261017/jobs_overnight/overnight-host-contracts-a100.json"


def test_host_contract_job_schema():
    job = validate_job(json.loads(JOB.read_text()))
    assert job["requirements"] == {"gpu": "A100", "min_vcpu": 2, "est_minutes": 15}
    assert job["depends_on"] == []
    assert job["timeout_minutes"] == 20
    assert job["priority"] == 0
    assert job["env"] == {}
    assert job["script"] == str(LAUNCHER.relative_to(ROOT))
    assert job["output_dir"] == "overnight-20261018/host-contracts-a100/attempt-1"


@pytest.mark.parametrize(
    "status,skipped,xml", [(0, 0, True), (0, 1, True), (7, 0, True), (124, 0, False)]
)
def test_host_launcher_receipt(tmp_path, status, skipped, xml):
    out = tmp_path / "output"
    out.mkdir()
    stub = tmp_path / "timeout"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "out = pathlib.Path(os.environ['QUEUE_OUTPUT_DIR'])\n"
        "data = dict(args=sys.argv[1:], env=dict(os.environ))\n"
        "(out / 'invocation.json').write_text(json.dumps(data))\n"
        + (
            "(out / 'host-contracts.xml').write_text('<testsuites><testsuite><testcase>"
            + ("<skipped/>" if skipped else "")
            + "</testcase></testsuite></testsuites>')\n"
            if xml
            else ""
        )
        + "print('durations and stdout')\nprint('stderr', file=sys.stderr)\n"
        + f"sys.exit({status})\n"
    )
    stub.chmod(0o755)
    env = {
        **os.environ,
        "QUEUE_OUTPUT_DIR": str(out),
        "PATH": f"{tmp_path}:{Path(sys.executable).parent}:{os.environ['PATH']}",
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert result.returncode == status, result.stderr
    invocation = json.loads((out / "invocation.json").read_text())
    assert invocation["args"] == [
        "--signal=TERM",
        "--kill-after=30s",
        "900s",
        "/content/t16-colab-venv/bin/python",
        "-m",
        "pytest",
        "-q",
        "-ra",
        "--durations=15",
        "-o",
        f"cache_dir={out / 'pytest-cache'}",
        f"--basetemp={out / 'pytest-tmp'}",
        f"--junitxml={out / 'host-contracts.xml'}",
        "tests/test_q1_trainer.py",
        "tests/test_q1_collector_oracles.py",
        "tests/test_q1_qualification.py",
        "tests/test_q0_audits.py",
        "tests/test_q0_episode_metrics.py",
        "tests/test_t22_random_safe.py",
        "tests/test_t24_v8_bodymap.py",
    ]
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["exit_code"] == status
    assert receipt["skip_count"] == (skipped if xml else None)
    assert receipt["status"] == ("PASS" if status == skipped == 0 and xml else "INCOMPLETE")
    assert receipt["elapsed_seconds"] >= 0
    assert len(receipt["identity"]["commit"]) == 40
    assert set(receipt["identity"]["packages"]) == {"torch", "numpy", "numba", "llvmlite"}
    for key, value in receipt["identity"]["environment"].items():
        assert value == ("cpu" if key == "SNAKE_DQN_DEVICE" else "1")
        assert invocation["env"][key] == value
    for name, digest in receipt["artifacts_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert (out / "receipt.json.sha256").read_text().split()[0] == hashlib.sha256(
        (out / "receipt.json").read_bytes()
    ).hexdigest()
    # A reused attempt cannot overwrite the prior receipt.
    previous = (out / "receipt.json").read_bytes()
    assert subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True).returncode != 0
    assert (out / "receipt.json").read_bytes() == previous
