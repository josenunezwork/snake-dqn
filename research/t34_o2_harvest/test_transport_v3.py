"""Mechanical O2 recovery regression: exact pinned Bash path, no provider calls."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from research.t34_o2_harvest import analyze_v3, make_manifest_v3
from research.t34_o2_harvest.bootstrap_preflight_v3 import _blob, run_bootstrap

PACKAGE = Path(__file__).parent
PIN = "f19d3f39050adfedc6eaef001a3364318fbb28bc"


def _runtime_manifest() -> dict:
    return {
        "commit": PIN,
        "script": "research/t34_o2_harvest/launch_v2.sh",
        "env": {
            "SNAKE_DQN_DEVICE": "cpu",
            "T34_O2_MODE": "campaign",
            "T34_O2_WORKERS": "10",
            "T34_O2_START": "0",
            "T34_O2_COUNT": "130",
        },
    }


def test_exact_remote_bootstrap_and_frozen_launcher_reach_cpu_probe() -> None:
    result = run_bootstrap(_runtime_manifest())
    assert result["worker_environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert result["worker_environment"]["SNAKE_DQN_DEVICE"] == "cpu"
    assert result["root_guard_relocated_only"]


def test_original_v2_mask_reproduces_remote_assertion_with_explicit_diagnostic() -> None:
    manifest = _runtime_manifest()
    manifest["env"]["CUDA_VISIBLE_DEVICES"] = ""
    with pytest.raises(RuntimeError, match="(?s)CUDA must be visible.*AssertionError"):
        run_bootstrap(manifest, enforce_admission=False)
    with pytest.raises(ValueError, match="requires CUDA_VISIBLE_DEVICES to be unset"):
        run_bootstrap(manifest)


def test_wrong_cuda_wheel_has_explicit_bootstrap_diagnostic() -> None:
    with pytest.raises(RuntimeError, match="torch CUDA='12.8', L4 wheel expects 12.6"):
        run_bootstrap(_runtime_manifest(), torch_cuda="12.8")


def test_v3_analyzer_changes_only_transport_placement_block() -> None:
    from research.t34_o2_harvest.frozen_sources_v3 import analysis_adapter

    original = _blob(PIN, "research/t34_o2_harvest/analyze_v2.py")
    old_tree = ast.parse(original)
    new_tree = ast.parse(analysis_adapter(original))
    # Ignore module docstrings. Every other node, including every scientific
    # function and sole-look marker, must be exactly the frozen V2 AST.
    old_tree.body.pop(0)
    new_tree.body.pop(0)
    new_analyze = next(n for n in new_tree.body if getattr(n, "name", None) == "_analyze_frozen")
    new_analyze.name = "analyze"
    old_admit = next(n for n in old_tree.body if getattr(n, "name", None) == "_admit_job")
    new_admit = next(n for n in new_tree.body if getattr(n, "name", None) == "_admit_job")
    index = next(
        i
        for i, node in enumerate(old_admit.body)
        if isinstance(node, ast.If) and "job execution placement differs" in ast.unparse(node)
    )
    assert isinstance(new_admit.body[index], ast.Assign)
    assert "O2 V3 requires visible CUDA" in ast.unparse(new_admit.body[index + 1])
    new_admit.body[index : index + 2] = [old_admit.body[index]]
    assert ast.dump(old_tree) == ast.dump(new_tree)


def test_v3_manifest_keeps_frozen_design_and_uses_hash_bound_url_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.t34_o2_harvest import plan_v2

    registration = PACKAGE / "REGISTRATION_O2_V2.json"
    catalog = PACKAGE.parents[0] / "t34_exit_b8/PUBLIC_INPUTS.json"
    preflight = tmp_path / "receipt.json"
    preflight.write_text("{}")
    frozen = json.loads(registration.read_text())
    # Use a complete previously tested template; production preparation invokes
    # the original builder in its exact Git snapshot with real fixture timing.
    template = _runtime_manifest()
    template["env"]["CUDA_VISIBLE_DEVICES"] = ""
    template.update(source_paths=sorted(frozen["source_hashes"]))

    def prior_manifest(_registration, kwargs):
        chunk = frozen["chunks"][kwargs["chunk"]]
        result = copy.deepcopy(template)
        result.update(
            job_id=f"t34-o2-harvest-v2-{kwargs['chunk']:02d}",
            output_dir=str(kwargs["output_dir"]),
            o2={"count": chunk["count"]},
            budget={"unit_cap": chunk["caps"]["units_per_job"]},
            inputs=[
                {
                    "path": str(p.resolve()),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "dest": d,
                }
                for p, d in (
                    (registration, "registration.json"),
                    (catalog, "public-inputs.json"),
                    (preflight, "preflight-receipt.json"),
                )
            ],
        )
        return result

    monkeypatch.setattr(make_manifest_v3, "invoke", prior_manifest)
    results = []
    for chunk in range(10):
        kwargs = dict(
            registration=registration,
            catalog=catalog,
            preflight_receipt=preflight,
            output_dir=tmp_path / f"v3-chunk-{chunk:02d}",
            commit=PIN,
            chunk=chunk,
        )
        prior = prior_manifest(registration, kwargs)
        manifest = make_manifest_v3.make_manifest(**kwargs)
        expected = copy.deepcopy(prior)
        expected["job_id"] = f"t34-o2-harvest-v3-{chunk:02d}"
        del expected["env"]["CUDA_VISIBLE_DEVICES"]
        expected["inputs"] = manifest["inputs"]
        assert manifest == expected
        assert manifest["source_paths"] == sorted(frozen["source_hashes"])
        for row, path in zip(manifest["inputs"][:2], (registration, catalog), strict=True):
            assert "path" not in row
            assert row["url"].startswith(make_manifest_v3.RAW + "/")
            assert row["size_bytes"] == path.stat().st_size
            assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert len([i for i in manifest["inputs"] if "path" in i]) == 1
        results.append(manifest)
    assert sum(m["o2"]["count"] for m in results) == 1260
    assert sum(m["budget"]["unit_cap"] for m in results) == pytest.approx(plan_v2.caps()["units"])


def test_v3_analyzer_rejects_the_old_mask_without_consuming_a_look(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.t34_o2_harvest import frozen_sources_v3

    registration = frozen_sources_v3.require_registration(PACKAGE / "REGISTRATION_O2_V2.json")
    manifest = _runtime_manifest()
    chunk = registration["chunks"][0]
    caps = {**registration["caps"], **chunk["caps"]}
    manifest.update(
        requirements={"gpu": "L4"},
        timeout_minutes=chunk["caps"]["timeout_minutes"],
        budget={
            "unit_cap": caps["units_per_job"],
            "rate_units_per_hour": caps["l4_rate_units_per_hour"],
            "reserve_units": caps["reserve_units_per_job"],
        },
        inputs=[{"dest": "registration.json", "sha256": "r"}],
    )
    terminal = {"generation": "g", "status": "completed", "returncode": 0}
    receipt = dict(
        manifest_sha256="m",
        commit=PIN,
        status="PASS",
        stop_verified=True,
        generation="g",
        terminal=terminal,
    )
    values = {"receipt.json": receipt, "manifest.json": manifest, "terminal.json": terminal}
    scope = {"__name__": "o2_test_adapter"}
    adapter = frozen_sources_v3.analysis_adapter(
        _blob(PIN, "research/t34_o2_harvest/analyze_v2.py")
    )
    exec(compile(adapter, "pinned-o2-adapter", "exec"), scope)
    monkeypatch.setitem(scope, "read_json", lambda p: values[p.name])
    monkeypatch.setitem(scope, "sha256", lambda _: "m")
    monkeypatch.setitem(scope, "_registered_chunk", lambda *_: (chunk, caps))
    # Correct placement progresses to export evidence. No fake result admitted.
    with pytest.raises(analyze_v3.InvalidEvidence, match="immutable export"):
        scope["_admit_job"](tmp_path, registration, "r")
    manifest["env"]["CUDA_VISIBLE_DEVICES"] = ""
    with pytest.raises(analyze_v3.InvalidEvidence, match="requires visible CUDA"):
        scope["_admit_job"](tmp_path, registration, "r")
    assert not (PACKAGE / "LOOK_O2_V2.json").exists()


def test_analysis_uses_pinned_dependencies_and_the_original_external_look_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.t34_o2_harvest import frozen_sources_v3

    registration = PACKAGE / "REGISTRATION_O2_V2.json"
    frozen = json.loads(registration.read_text())
    marker = registration.with_name("LOOK_O2_V2.json")
    inspected = []

    def inspect_child(command, **kwargs):
        repo = Path(kwargs["cwd"])
        assert command[:3] == [frozen_sources_v3.sys.executable, "-I", "-c"]
        assert "_analyze_frozen" in command[3]
        original = (repo / "research/t34_o2_harvest/analyze_v2.py").read_text()
        adapter = (repo / "research/t34_o2_harvest/analyze_v3.py").read_text()
        assert adapter == frozen_sources_v3.analysis_adapter(original)
        # The executed adapter is constructed from pinned science, rather than
        # copied from the mutable current-checkout sibling analyzer.
        assert (
            hashlib.sha256(original.encode()).hexdigest()
            == frozen["source_hashes"]["research/t34_o2_harvest/analyze_v2.py"]
        )
        assert (repo / "research/t34_o2_harvest/REGISTRATION_O2_V2.json").resolve() == (
            registration.resolve()
        )
        # Dependencies from the original source pin, including the source that
        # later main changed, are loaded from this isolated filesystem.
        for name in (
            "research/t34_round0/analyze.py",
            "research/t34_o2_harvest/controller.py",
            "research/t3_rollout_labels_20261015/bank_engine.py",
        ):
            assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == (
                frozen["source_hashes"][name]
            )
        inspected.append(repo)
        return subprocess.CompletedProcess(command, 0, stdout='{"status":"probe"}', stderr="")

    # Intercept only the analysis child, retaining real Git archive/check calls.
    real_run = frozen_sources_v3.subprocess.run

    def run(command, **kwargs):
        if len(command) > 2 and command[1:3] == ["-I", "-c"]:
            return inspect_child(command, **kwargs)
        return real_run(command, **kwargs)

    monkeypatch.setattr(frozen_sources_v3.subprocess, "run", run)
    assert analyze_v3.analyze(registration, [tmp_path / "job"], tmp_path / "analysis") == {
        "status": "probe"
    }
    assert len(inspected) == 1
    assert not inspected[0].exists()
    assert not marker.exists()


def test_copied_registration_cannot_create_a_second_look(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.t34_o2_harvest import frozen_sources_v3

    copy_path = tmp_path / "registration-copy.json"
    copy_path.write_bytes((PACKAGE / "REGISTRATION_O2_V2.json").read_bytes())

    def no_archive(*_args, **_kwargs):
        pytest.fail("copied registration reached snapshot preparation")

    monkeypatch.setattr(frozen_sources_v3.subprocess, "check_output", no_archive)
    with pytest.raises(analyze_v3.InvalidEvidence, match="canonical frozen"):
        analyze_v3.analyze(copy_path, [], tmp_path / "out")
    with pytest.raises(ValueError, match="canonical frozen"):
        frozen_sources_v3.invoke(copy_path, analysis={"run_dirs": [], "output_dir": "unused"})
    assert not (tmp_path / "LOOK_O2_V2.json").exists()
    assert not (tmp_path / "out").exists()
