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
    science = json.loads((ROOT / "research/watch_lead_p/science-sources.json").read_text())
    assert len(paths) == len(set(paths))
    assert len(science) == len(set(science))
    assert "research/watch_lead_p/science-sources.json" in paths
    assert set(science).issubset(paths)
    assert "requirements.txt" in paths
    assert "research/watch_lead_p/requirements.txt" in paths
    assert (ROOT / "research/watch_lead_p/requirements.txt").read_text().splitlines()[0] == (
        "-r ../../requirements.txt"
    )


def test_seal_rejects_invalid_pin_before_creating_packet(tmp_path, monkeypatch) -> None:
    from research.watch_lead_p import seal

    monkeypatch.setattr(seal, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="full commit SHA"):
        seal.seal("invalid", tmp_path / "broker", tmp_path / "runtime", tmp_path / "benchmark")
    assert list(tmp_path.iterdir()) == []


def test_registry_exclusion_audit() -> None:
    value = exclusion_audit()
    assert value["seeds"] == list(SEEDS)
    assert len(set(SEEDS)) == 32
    assert value["collisions"] == []
    assert not set(SEEDS) & set(value["excluded_engineering_seeds"])
    assert value["train01_monitor_nomination_count"] == 1056


@pytest.mark.parametrize("change", [None, "gpu", "output", "source", "env"])
def test_p_route_is_narrow(tmp_path, monkeypatch, change) -> None:
    monkeypatch.setattr(preflight, "verify_committed_manifest", lambda *a: None)
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


@pytest.fixture
def completion_packet(tmp_path, monkeypatch, request):
    """Real committed miniature operator workflow; synthetic outcomes only."""
    import shutil

    from research.bl_served_search.exports import export_json
    from research.watch_lead_p import resume, seal
    from research.watch_lead_p.protocol import ARMS, FRAMES

    repo = (tmp_path / "repo").resolve()
    # Each parameterized case owns and removes its checkout, including on failure.
    # Retaining dozens of full clones previously exhausted the Mac disk.
    request.addfinalizer(lambda: shutil.rmtree(repo, ignore_errors=True))
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(repo)],
        check=True,
    )
    # Keep the complete pinned Git tree, but materialize only this fixture's
    # dependency closure. Full checkouts per case exceed dispatch's 180s guard.
    sources = json.loads((ROOT / "research/watch_lead_p/runtime-sources.json").read_text())
    patterns = [
        *sources,
        "research/watch_lead_p/",
        "research/colab_cli/",
        "tools/ops/",
        "tools/sandbox/",
        ".gitignore",
    ]
    subprocess.run(
        ["git", "-C", str(repo), "sparse-checkout", "set", "--no-cone", "--stdin"],
        input="".join(f"/{name}\n" for name in patterns),
        text=True,
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", "--detach", "HEAD"], check=True)
    # Exercise the current implementation before and after its eventual commit.
    for name in [*sources, "tools/ops/dispatch.py"]:
        shutil.copyfile(ROOT / name, repo / name)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    git("add", "research/watch_lead_p", "tools/ops/dispatch.py")
    git("-c", "core.hooksPath=/dev/null", "commit", "--allow-empty", "-qm", "fixture sources")
    pin = git("rev-parse", "HEAD")
    inputs = tmp_path / "inputs"
    paths = []
    recipe_sha256 = resume.recipe_fingerprint()
    for index in range(getattr(request, "param", 1)):
        path = inputs / "units" / f"session-{index:02d}.json"
        arms = {
            a: dict(initial_state="same", frames=FRAMES, seed=SEEDS[index], deaths=10, mass=100)
            for a in ARMS
        }
        export_json(
            path,
            dict(
                namespace=resume.NAMESPACE,
                allocation=resume.ALLOCATION,
                unit_id=resume.unit_id(index),
                allocation_index=index,
                recipe_sha256=recipe_sha256,
                seed=SEEDS[index],
                arms=arms,
            ),
        )
        paths.append(path)
    ledger = tmp_path / "ledger.json"
    resume.make_ledger(paths, ledger)
    forecast = tmp_path / "benchmark.json"
    forecast.write_text(
        json.dumps(
            dict(
                fixture_index=32,
                excluded_from_all_32_sessions=True,
                workers=12,
                projected_seconds=12000,
            )
        )
    )
    monkeypatch.setattr(seal, "ROOT", repo)
    monkeypatch.setattr(seal, "ARTIFACTS", tmp_path / "artifacts")
    seal.seal(pin, repo, repo, forecast, 90, ledger, inputs)
    git("add", "research/watch_lead_p/packet-90")
    git("-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture completion packet")
    monkeypatch.setattr(preflight, "REPO_ROOT", repo)
    monkeypatch.setattr(
        preflight,
        "_run",
        lambda command, **kw: subprocess.check_output(command, cwd=repo, text=True).strip(),
    )
    return repo, repo / "research/watch_lead_p/packet-90", inputs, paths


@pytest.mark.parametrize("completion_packet", [31], indirect=True)
def test_completion_packet_preflight_upload_skip_and_union(
    completion_packet, tmp_path, monkeypatch
):
    import concurrent.futures
    import shutil

    from research.colab_cli import supervisor
    from research.watch_lead_p import resume, run
    from research.watch_lead_p.protocol import FRAMES

    repo, packet, inputs, prior = completion_packet
    preflight.preflight(
        packet / "sandbox.json",
        registry_path=repo / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json",
        run_tests=False,
    )
    manifest = json.loads((packet / "dispatch-manifest.json").read_text())
    assert manifest["required_exports"] == [
        "watch-lead/inventory.json",
        "watch-lead/units/session-31.json",
    ]
    # Run the actual dispatch admission route with its recursive tests already
    # owned by the outer suite; all source, packet and input checks remain real.
    real_preflight = preflight.preflight
    monkeypatch.setattr(
        preflight,
        "preflight",
        lambda path, **kw: real_preflight(
            path, registry_path=repo / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json", run_tests=False
        ),
    )
    monkeypatch.setitem(dispatch.SANDBOX_OUTPUT_ROOTS, STUDY, tmp_path / "artifacts")
    monkeypatch.setattr(dispatch, "HERE", repo / "tools/ops")
    from tools.sandbox.supervisor import sandbox_admission_route

    with sandbox_admission_route():
        manifest = supervisor.validate(manifest, repo=repo)
    dispatch.require_sandbox_admission(manifest, repo)
    # The qualified staging path creates nested parents before uploading. Replay
    # its exact destination mapping locally; validate has admitted every dest.
    staged = tmp_path / "staged"
    for record in manifest["inputs"]:
        target = staged / record["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record["path"], target)
    assert len(resume.read_ledger(staged / "verified-ledger.json", staged)) == 31
    called = []

    def arm_worker(index, arm):
        called.append((index, arm))
        return dict(initial_state="same", frames=FRAMES, seed=SEEDS[index], deaths=10, mass=100)

    monkeypatch.setattr(run, "worker", arm_worker)

    class Pool(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, **kw):
            super().__init__(max_workers=2)
            self._processes = {}

    monkeypatch.setattr(run.concurrent.futures, "ProcessPoolExecutor", Pool)
    output = tmp_path / "completed"
    run.execute(staged / "verified-ledger.json", staged, output)
    assert sorted(called) == [(31, "P0"), (31, "P48"), (31, "bare")]
    assert sorted(p.name for p in (output / "units").glob("*.json")) == ["session-31.json"]
    report = resume.aggregate(
        [*prior, output / "units/session-31.json"], tmp_path / "decision.json"
    )
    assert report["decision"] == "STOP this delayed controller"
    assert len(report["inputs"]) == 32


@pytest.mark.parametrize(
    "corruption", ["seal-empty", "seal-uncommitted", "ledger-path", "dest-missing", "all-complete"]
)
@pytest.mark.parametrize("completion_packet", [31], indirect=True)
def test_completion_admission_rejects_broken_bindings(
    completion_packet, tmp_path, monkeypatch, corruption
):
    from research.watch_lead_p.admission import validate_inputs
    from tools.sandbox.registry import load_registry

    repo, packet, _, _ = completion_packet
    payload = json.loads((packet / "sandbox.json").read_text())
    if corruption == "seal-uncommitted":
        (packet / "seal.json").write_text("{}")
        with pytest.raises(preflight.PreflightError, match="committed HEAD"):
            validate_inputs(payload, load_registry())
        return
    # Other cases target semantic checks after committed-byte verification.
    monkeypatch.setattr(preflight, "verify_committed_manifest", lambda *a: None)
    if corruption == "seal-empty":
        (packet / "seal.json").write_text("{}")
    elif corruption == "ledger-path":
        record = next(r for r in payload["inputs"] if r["kind"] == "verified-ledger")
        record["path"] = str(tmp_path / "foreign-ledger.json")
    else:
        manifest = json.loads((packet / "dispatch-manifest.json").read_text())
        if corruption == "dest-missing":
            manifest["inputs"][-1].pop("dest")
            (packet / "dispatch-manifest.json").write_text(json.dumps(manifest))
        else:
            value = json.loads((packet / "verified-ledger.json").read_text())
            value["units"].append(value["units"][0])
            (packet / "verified-ledger.json").write_text(json.dumps(value))
        from research.bl_served_search.exports import digest

        seal = json.loads((packet / "seal.json").read_text())
        for name in seal:
            seal[name] = digest(packet / name)
        (packet / "seal.json").write_text(json.dumps(seal))
    with pytest.raises(ValueError):
        validate_inputs(payload, load_registry())


def test_dispatch_verifies_inventory_once_and_rejects_unsafe_admitted_paths(tmp_path, monkeypatch):
    """Inventory membership never bypasses the independent path/prefix boundary."""
    monkeypatch.setattr(dispatch, "HERE", tmp_path / "tools/ops")
    value = json.loads(
        (ROOT / "research/watch_lead_p/packet-03/dispatch-manifest.json").read_text()
    )
    monkeypatch.setitem(dispatch.SANDBOX_OUTPUT_ROOTS, STUDY, tmp_path)
    value["output_dir"] = str(tmp_path / "unused")
    inventory = tmp_path / "research/watch_lead_p/runtime-sources.json"
    inventory.parent.mkdir(parents=True)
    called = []
    monkeypatch.setattr(preflight, "verify_committed_manifest", lambda path: called.append(path))
    for source in ("/absolute.py", "research/watch_lead_p/../evil.py", "evil.py"):
        inventory.write_text(json.dumps([source]))
        value["source_paths"] = [source]
        with pytest.raises(dispatch.Refused, match="engineering dependency"):
            dispatch.require_sandbox_admission(value, tmp_path)
    assert called == [inventory] * 3


def test_seal_rejects_forecast_above_base_before_writing(tmp_path, monkeypatch):
    from research.watch_lead_p import seal

    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            dict(
                fixture_index=32,
                excluded_from_all_32_sessions=True,
                workers=12,
                projected_seconds=14401,
            )
        )
    )
    monkeypatch.setattr(seal, "ROOT", ROOT)
    pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with pytest.raises(ValueError, match="base envelope"):
        seal.seal(pin, tmp_path, tmp_path, benchmark, 99)
    assert not (ROOT / "research/watch_lead_p/packet-99").exists()


@pytest.mark.parametrize("count", [0, 32])
def test_sealer_refuses_empty_or_complete_completion_ledger(tmp_path, count):
    from research.bl_served_search.exports import export_json
    from research.watch_lead_p import resume, seal

    paths = []
    for index in range(count):
        path = tmp_path / "units" / f"session-{index:02d}.json"
        export_json(path, {"sealed": True})
        paths.append(path)
    ledger = tmp_path / "ledger.json"
    resume.make_ledger(paths, ledger)
    pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with pytest.raises(ValueError, match="1..31 verified units"):
        seal.seal(pin, tmp_path, tmp_path, tmp_path / "unused", 99, ledger, tmp_path)
    assert not (ROOT / "research/watch_lead_p/packet-99").exists()


def _refresh_packet_seal(packet):
    from research.bl_served_search.exports import digest

    value = json.loads((packet / "seal.json").read_text())
    for name in value:
        value[name] = digest(packet / name)
    (packet / "seal.json").write_text(json.dumps(value))


@pytest.mark.parametrize("completion", [False, True])
@pytest.mark.parametrize("damage", ["missing", "empty", "duplicate", "misroute", "foreign-source"])
def test_all_packets_validate_upload_destinations(
    completion_packet, monkeypatch, completion, damage
):
    from research.watch_lead_p.admission import validate_inputs
    from research.watch_lead_p.resume import empty_ledger
    from tools.sandbox.registry import load_registry

    _, packet, _, _ = completion_packet
    payload = json.loads((packet / "sandbox.json").read_text())
    manifest = json.loads((packet / "dispatch-manifest.json").read_text())
    if not completion:
        payload["inputs"] = [r for r in payload["inputs"] if r["kind"] != "verified-unit"]
        manifest["inputs"] = manifest["inputs"][:4]
        manifest["required_exports"] = ["watch-lead/inventory.json"] + [
            f"watch-lead/units/session-{i:02d}.json" for i in range(32)
        ]
        (packet / "verified-ledger.json").write_text(json.dumps(empty_ledger()))
    upload = manifest["inputs"][3]
    if damage == "missing":
        upload.pop("dest")
    elif damage == "empty":
        upload["dest"] = ""
    elif damage == "duplicate":
        upload["dest"] = manifest["inputs"][0]["dest"]
    elif damage == "misroute":
        upload["dest"] = "wrong-ledger.json"
    else:
        upload["path"] = str(packet / "foreign.json")
    (packet / "dispatch-manifest.json").write_text(json.dumps(manifest))
    _refresh_packet_seal(packet)
    monkeypatch.setattr(preflight, "verify_committed_manifest", lambda *a: None)
    with pytest.raises(ValueError, match="destination"):
        validate_inputs(payload, load_registry())


@pytest.mark.parametrize(
    "damage", ["ledger-extra", "record-extra", "bool-index", "unit-id", "sidecar", "hash"]
)
def test_completion_offline_schema_and_actual_sidecars(completion_packet, monkeypatch, damage):
    from research.bl_served_search.exports import digest
    from research.watch_lead_p.admission import validate_inputs
    from tools.sandbox.registry import load_registry

    _, packet, _, _ = completion_packet
    payload = json.loads((packet / "sandbox.json").read_text())
    manifest = json.loads((packet / "dispatch-manifest.json").read_text())
    ledger = json.loads((packet / "verified-ledger.json").read_text())
    if damage == "ledger-extra":
        ledger["outcome"] = 1
    elif damage == "record-extra":
        ledger["units"][0]["outcome"] = 1
    elif damage == "bool-index":
        ledger["units"][0]["allocation_index"] = True
    elif damage == "unit-id":
        ledger["units"][0]["unit_id"] = "foreign"
    elif damage == "hash":
        ledger["units"][0]["sha256"] = "0" * 64
    else:
        original = next(
            r for r in manifest["inputs"] if r["dest"] == "units/session-00.json.sha256"
        )
        foreign = packet / "foreign.sha256"
        foreign.write_text("wrong bytes")
        admitted = next(r for r in payload["inputs"] if r.get("path") == original["path"])
        original["path"] = admitted["path"] = str(foreign)
        original["sha256"] = admitted["sha256"] = digest(foreign)
    (packet / "dispatch-manifest.json").write_text(json.dumps(manifest))
    (packet / "verified-ledger.json").write_text(json.dumps(ledger))
    _refresh_packet_seal(packet)
    monkeypatch.setattr(preflight, "verify_committed_manifest", lambda *a: None)
    with pytest.raises(ValueError):
        validate_inputs(payload, load_registry())


@pytest.mark.parametrize(
    "damage",
    [
        "science-duplicate",
        "runtime-duplicate",
        "missing-science",
        "missing-inventory",
        "packet-paths",
    ],
)
def test_admission_requires_complete_unique_sparse_science(completion_packet, monkeypatch, damage):
    from research.watch_lead_p import admission
    from tools.sandbox.registry import load_registry

    repo, packet, _, _ = completion_packet
    monkeypatch.setattr(admission, "ROOT", repo)
    science_path = repo / "research/watch_lead_p/science-sources.json"
    runtime_path = repo / "research/watch_lead_p/runtime-sources.json"
    science = json.loads(science_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    manifest = json.loads((packet / "dispatch-manifest.json").read_text())
    if damage == "science-duplicate":
        science.append(science[0])
    elif damage == "runtime-duplicate":
        runtime.append(runtime[0])
    elif damage == "missing-science":
        runtime.remove(science[0])
        manifest["source_paths"] = runtime
    elif damage == "missing-inventory":
        runtime.remove("research/watch_lead_p/science-sources.json")
        manifest["source_paths"] = runtime
    else:
        manifest["source_paths"] = manifest["source_paths"][1:]
    science_path.write_text(json.dumps(science))
    runtime_path.write_text(json.dumps(runtime))
    (packet / "dispatch-manifest.json").write_text(json.dumps(manifest))
    _refresh_packet_seal(packet)
    monkeypatch.setattr(preflight, "verify_committed_manifest", lambda *a: None)
    payload = json.loads((packet / "sandbox.json").read_text())
    with pytest.raises(ValueError, match="inventory"):
        admission.validate_inputs(payload, load_registry())


@pytest.mark.parametrize("state", ["working", "staged", "committed"])
@pytest.mark.parametrize(
    "name", ["research/watch_lead_p/science-sources.json", "research/watch_lead_p/live.py"]
)
def test_seal_refuses_science_drift_from_pin(completion_packet, tmp_path, state, name):
    from research.watch_lead_p import seal

    repo, _, _, _ = completion_packet
    pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    path = repo / name
    path.write_text(path.read_text() + "\n")
    if state in {"staged", "committed"}:
        subprocess.run(["git", "add", name], cwd=repo, check=True)
    if state == "committed":
        subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture drift"],
            cwd=repo,
            check=True,
        )
    with pytest.raises(ValueError, match="differs from runtime pin"):
        seal.seal(pin, tmp_path / "broker", tmp_path / "runtime", tmp_path / "benchmark.json", 91)
    assert not (repo / "research/watch_lead_p/packet-91").exists()


@pytest.mark.parametrize("completion_packet", [31], indirect=True)
def test_seal_normalizes_all_cli_paths_and_binds_pin(completion_packet, tmp_path, monkeypatch):
    from research.watch_lead_p import resume, seal

    repo, _, inputs, _ = completion_packet
    pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    monkeypatch.chdir(tmp_path)
    seal.seal(
        pin,
        Path("nested/../broker"),
        Path("./runtime"),
        Path("benchmark.json"),
        91,
        Path("ledger.json"),
        Path("inputs"),
    )
    packet = repo / "research/watch_lead_p/packet-91"
    manifest = json.loads((packet / "dispatch-manifest.json").read_text())
    assert manifest["sandbox_admission"]["manifest_path"] == str(
        tmp_path / "broker/research/watch_lead_p/packet-91/sandbox.json"
    )
    assert all(Path(r["path"]).is_absolute() for r in manifest["inputs"])
    assert json.loads((packet / "forecast.json").read_text())["receipt_path"] == str(
        tmp_path / "benchmark.json"
    )
    assert json.loads((packet / "verified-ledger.json").read_text())[
        "recipe_sha256"
    ] == resume.recipe_fingerprint(repo)
    text = (packet / "DISPATCH.txt").read_text()
    assert f"--cwd {tmp_path}/runtime" in text
    assert "nested/.." not in text
    assert len(resume.read_ledger(packet / "verified-ledger.json", inputs, recipe_root=repo)) == 31
