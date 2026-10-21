"""Use the real supervisor's offline export and validation boundaries."""

from pathlib import Path

import pytest

from research.bl_ppo.exports import export_json, verify_export
from research.bl_prefix.dispatch import execute, resolve_inputs
from research.bl_prefix.make_manifest import REPO, build, forecast, source_paths
from research.colab_cli.runtime import _exports


def test_actual_supervisor_sidecars_and_mutation_fence(tmp_path: Path) -> None:
    path = tmp_path / "output" / "e6.json"
    info = export_json(path, {"development_only": True})
    assert verify_export(path) == info["sha256"]
    cache = {}
    assert _exports(tmp_path, 1024, cache) == [
        {"path": path.name, "sha256": info["sha256"], "size": info["size"]}
    ]
    with pytest.raises(FileExistsError):
        export_json(path, {})
    path.write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        _exports(tmp_path, 1024, cache)


def test_incomplete_export_is_not_accepted(tmp_path: Path) -> None:
    path = tmp_path / "output" / "e7.json"
    export_json(path, {})
    Path(str(path) + ".sha256").unlink()
    with pytest.raises(RuntimeError, match="marker"):
        _exports(tmp_path, 1024)


def test_blocked_plan_never_creates_gameplay_output(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text('{"experiment":"e6","ready":false}')
    with pytest.raises(ValueError, match="NOT_DISPATCHABLE"):
        execute(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_incomplete_runner_cannot_exit_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules,
        "research.bl_prefix.e6",
        SimpleNamespace(
            run=lambda *args, **kwargs: {"status": "INCOMPLETE", "reason": "missing critic"}
        ),
    )
    path = tmp_path / "plan.json"
    path.write_text('{"experiment":"e6","ready":true}')
    with pytest.raises(RuntimeError, match="INCOMPLETE.*missing critic"):
        execute(path, tmp_path / "out")


@pytest.mark.parametrize("value", ["input:../secret", "input:/secret", "/Users/test/x"])
def test_runtime_input_path_refusal(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError):
        resolve_inputs(value, tmp_path)


def test_public_checkpoint_runtime_mapping_preserves_prefix_plan_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalog B8 name remains reachable through the plan's bound alias."""
    public = tmp_path / "public-checkpoints"
    (public / "checkpoints").mkdir(parents=True)
    b8 = public / "B8_s3_final.pth"
    roster = public / "checkpoints" / "best_apex_fs.pth"
    b8.write_bytes(b"b8")
    roster.write_bytes(b"roster")
    monkeypatch.setenv("BL_PUBLIC_INPUTS", "1")
    assert resolve_inputs("input:checkpoints/b8_s3_t7_20261016.pth", tmp_path) == str(b8.resolve())
    assert resolve_inputs("input:checkpoints/best_apex_fs.pth", tmp_path) == str(roster.resolve())


def test_forecast_charges_collection_secondary_and_confirmations() -> None:
    e6 = forecast("e6", 0.001)
    e7 = forecast("e7", 0.001)
    assert e6["world_step_upper_bound"] == 37_760_000
    assert e7["world_step_upper_bound"] == 24_960_000
    assert e6["unit_cap"] == 43.4 and e7["unit_cap"] == 28.5
    assert e6["estimated_units_at_2_per_hour"] > 0


def test_prefix_source_closure_carries_transitive_runtime_dependencies() -> None:
    paths = source_paths()
    assert "research/bl_prefix/dispatch.py" in paths
    assert "research/bl_ppo/exports.py" in paths
    assert "research/bl_ppo/model.py" in paths
    assert "research/apex_safety_20260926/dev_screen.py" in paths
    assert "research/bl_prefix/test_transport.py" not in paths
    assert "research/bl_prefix/run.sh" in paths
    assert "research/colab_cli" in paths


def test_manifest_uses_real_offline_supervisor_validation(tmp_path: Path) -> None:
    import subprocess

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    plan = tmp_path / "plan.json"
    plan.write_text('{"experiment":"e6","ready":false}')
    result = build("e6", plan, tmp_path / "fresh-output", commit)
    assert result["requirements"]["gpu"] == "L4"
    assert result["budget"]["unit_cap"] == 43.4
    assert result["timeout_minutes"] == 1041.6
    assert result["bl_prefix"]["ready"] is False
    assert result["bl_prefix"]["dispatch_status"] == "NOT_DISPATCHED"
    assert not (tmp_path / "fresh-output").exists()
    with pytest.raises(ValueError, match="forecast does not fit"):
        build("e6", plan, tmp_path / "fresh-output", commit, estimate_minutes=522)


def test_ready_plan_cannot_admit_unuploaded_mac_files(tmp_path: Path) -> None:
    from research.bl_prefix.readiness import local_inputs, validate_ready

    local = tmp_path / "valid-local-input.json"
    local.write_text("{}")
    with pytest.raises(ValueError, match="uploaded"):
        local_inputs(str(local), {})
    with pytest.raises(ValueError, match="delivery tags"):
        validate_ready({"experiment": "e6", "ready": True, "seed_audit": str(local)}, {})
    assert local_inputs("input:delivered.json", {"delivered.json": local}) == str(local)


def test_resource_readiness_cannot_bypass_scientific_gate(tmp_path: Path) -> None:
    """A fitting forecast is independently reviewable but cannot supply endpoints."""
    import subprocess

    from research.bl_prefix.make_manifest import LIMITS

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    measured = forecast("e6", 0.00001)
    timing = tmp_path / "forecast.json"
    export_json(timing, measured)
    plan = tmp_path / "plan.json"
    export_json(
        plan,
        {
            "experiment": "e6",
            "ready": False,
            "forecast": "input:forecast.json",
            "blocking_reasons": ["final R/F endpoints and critic receipts are absent"],
        },
    )
    result = build(
        "e6",
        plan,
        tmp_path / "fresh-output",
        commit,
        input_files=[("forecast.json", timing)],
        estimate_minutes=measured["estimated_allocation_minutes"],
    )
    assert result["bl_prefix"]["ready"] is False
    assert result["bl_prefix"]["resource_readiness"]["ready"] is True
    assert result["bl_prefix"]["resource_readiness"]["dispatch_ready"] is False
    assert result["timeout_minutes"] == LIMITS["e6"][2] <= 1440
    with pytest.raises(ValueError, match="NOT_DISPATCHABLE"):
        execute(plan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_manifest_rejects_tampered_resource_forecast(tmp_path: Path) -> None:
    import subprocess

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    record = forecast("e7", 0.00001)
    record["estimated_allocation_minutes"] = 0.01
    timing = tmp_path / "forecast.json"
    export_json(timing, record)
    plan = tmp_path / "plan.json"
    export_json(plan, {"experiment": "e7", "ready": False, "forecast": "input:forecast.json"})
    with pytest.raises(ValueError, match="resource forecast differs"):
        build("e7", plan, tmp_path / "output", commit, input_files=[("forecast.json", timing)])


def test_retired_operational_ids_cannot_reenter_science() -> None:
    from research.bl_prefix.operational_burns import BURNED_IDENTITIES, require_unburned

    with pytest.raises(ValueError, match="operationally burned"):
        require_unburned([1102882751176])
    require_unburned([1, 2, 3])
    assert len(BURNED_IDENTITIES) == 2
