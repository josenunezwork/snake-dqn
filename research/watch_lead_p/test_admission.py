"""P's narrow L4 route, launch/exec binding and immutable packet checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from research.watch_lead_p.admission import exclusion_audit
from research.watch_lead_p.protocol import SEEDS, STUDY
from tools.ops import dispatch
from tools.sandbox import preflight

ROOT = Path(__file__).resolve().parents[2]


def test_sparse_source_includes_nested_requirements() -> None:
    paths = json.loads((ROOT / "research/watch_lead_p/runtime-sources.json").read_text())
    assert "requirements.txt" in paths
    assert "research/watch_lead_p/requirements.txt" in paths
    assert (ROOT / "research/watch_lead_p/requirements.txt").read_text().splitlines()[0] == (
        "-r ../../requirements.txt"
    )


def test_registry_exclusion_audit() -> None:
    value = exclusion_audit()
    assert value["seeds"] == list(SEEDS)
    assert len(set(SEEDS)) == 32
    assert value["collisions"] == []
    assert not set(SEEDS) & set(value["excluded_engineering_seeds"])
    assert value["train01_monitor_nomination_count"] == 1056


@pytest.mark.parametrize("change", [None, "gpu", "output", "source", "env"])
def test_p_route_is_narrow(tmp_path, monkeypatch, change) -> None:
    # This route-only fixture inherits B's historical pin. The launch test
    # below and the sealed packet dry-run check the real P entrypoint pin.
    monkeypatch.setattr("research.colab_cli.canary.job_device", lambda *a: "cpu")
    value = json.loads((ROOT / "research/sandbox_b_heads/dispatch-manifest.json").read_text())
    value.pop("canary", None)
    value["script"] = "research/watch_lead_p/run_job.sh"
    value["job_id"] = "p-route-test"
    value["requirements"]["gpu"] = "L4"
    value["requirements"]["allowed_gpus"] = ["L4"]
    value["env"] = {"SNAKE_DQN_DEVICE": "cpu"}
    value["source_paths"] = ["research/watch_lead_p/run.py"]
    value["output_dir"] = str(dispatch.SANDBOX_OUTPUT_ROOTS[STUDY] / "unused-test-output")
    value["sandbox_admission"] = {
        "schema": "standing-a100-development-sandbox-admission/v1",
        "manifest_path": str(tmp_path / "sandbox.json"),
    }
    original = tmp_path / "dispatch.json"
    original.write_text(json.dumps(value))
    monkeypatch.setattr(
        preflight,
        "preflight",
        lambda *a, **k: {"study": STUDY, "outputs": {"broker_manifest": {"path": str(original)}}},
    )
    if change == "gpu":
        value["requirements"]["allowed_gpus"] = ["A100"]
    elif change == "output":
        value["output_dir"] = str(tmp_path.resolve() / "foreign-root")
    elif change == "source":
        value["source_paths"] = ["research/foreign.py"]
    elif change == "env":
        value["env"]["SANDBOX_A_CONTROL_JSON"] = "{}"
    if change:
        with pytest.raises(dispatch.Refused):
            dispatch.require_sandbox_admission(value, ROOT)
    else:
        dispatch.require_sandbox_admission(value, ROOT)


def test_actual_launch_exec_binds_python_module_and_uploaded_inputs(tmp_path) -> None:
    """Exec the real shell entrypoint against a non-computing Python stand-in."""
    import os

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("b8-s3.pth", "frp3-s12.pth", "verified-ledger.json"):
        (inputs / name).write_text("fixture")
    binary = tmp_path / "bin"
    binary.mkdir()
    log = tmp_path / "exec.json"
    standin = binary / "python"
    standin.write_text(
        "#!/usr/bin/env python3\nimport os,sys,json\nfrom pathlib import Path\n"
        "Path(os.environ['EXEC_LOG']).write_text(json.dumps({'args':sys.argv[1:],"
        "'nice':os.getpriority(os.PRIO_PROCESS,0),"
        "'env':{k:os.environ.get(k) for k in ['SNAKE_DQN_DEVICE','OMP_NUM_THREADS',"
        "'MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS',"
        "'OMP_THREAD_LIMIT','TORCHINDUCTOR_COMPILE_THREADS']}}))\n"
    )
    standin.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{binary}:{os.environ['PATH']}",
        COLAB_CLI_ROOT=str(tmp_path / "runtime"),
        COLAB_INPUT_DIR=str(inputs),
        EXEC_LOG=str(log),
    )
    subprocess.run(
        ["bash", str(ROOT / "research/watch_lead_p/run_job.sh")],
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=15,
    )
    result = json.loads(log.read_text())
    assert result["args"] == [
        "-m",
        "research.watch_lead_p.run",
        "run",
        "--ledger",
        str(inputs / "verified-ledger.json"),
        "--inputs",
        str(inputs),
        "--output",
        str(tmp_path / "runtime/output/watch-lead"),
    ]
    assert result["nice"] >= 15
    assert result["env"].pop("SNAKE_DQN_DEVICE") == "cpu"
    assert set(result["env"].values()) == {"1"}
    assert (tmp_path / "saved_snakes/b8_s3_t7_20261016.pth").read_text() == "fixture"


def test_p_qualified_supervisor_exec_uses_source_adapter(monkeypatch) -> None:
    from contextlib import contextmanager

    from research.colab_cli import supervisor
    from tools.sandbox import supervisor as adapter

    calls = []

    @contextmanager
    def route(cli):
        calls.append(cli)
        yield

    monkeypatch.setattr("tools.sandbox.source_delivery.background_sparse_route", route)
    monkeypatch.setattr(supervisor.Broker, "execute", lambda *a, **k: "bound-execute")
    with adapter.sandbox_admission_route():
        assert (
            supervisor.Broker.execute(
                None,
                "cli",
                {
                    "script": "research/watch_lead_p/run_job.sh",
                    "sandbox_source_delivery": "background-sparse-v1",
                },
            )
            == "bound-execute"
        )
    assert calls == ["cli"]
