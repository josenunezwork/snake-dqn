"""Use the real supervisor's offline export and validation boundaries."""

from pathlib import Path

import pytest

from research.bl_ppo.exports import export_json, verify_export
from research.bl_prefix.dispatch import execute, resolve_inputs
from research.bl_prefix.make_manifest import REPO, build, forecast
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


def test_forecast_charges_collection_secondary_and_confirmations() -> None:
    e6 = forecast("e6", 0.001)
    e7 = forecast("e7", 0.001)
    assert e6["world_step_upper_bound"] == 37_120_000
    assert e7["world_step_upper_bound"] == 24_960_000
    assert e6["unit_cap"] == 4 and e7["unit_cap"] == 20
    assert e6["estimated_units_at_2_per_hour"] > 0


def test_manifest_uses_real_offline_supervisor_validation(tmp_path: Path) -> None:
    import subprocess

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    plan = tmp_path / "plan.json"
    plan.write_text('{"experiment":"e6","ready":false}')
    result = build("e6", plan, tmp_path / "fresh-output", commit)
    assert result["requirements"]["gpu"] == "L4"
    assert result["budget"]["unit_cap"] == 4
    assert result["timeout_minutes"] == 96
    assert result["bl_prefix"]["ready"] is False
    assert result["bl_prefix"]["dispatch_status"] == "NOT_DISPATCHED"
    assert not (tmp_path / "fresh-output").exists()
    with pytest.raises(ValueError, match="forecast does not fit"):
        build("e6", plan, tmp_path / "fresh-output", commit, estimate_minutes=49)


def test_ready_plan_cannot_admit_unuploaded_mac_files(tmp_path: Path) -> None:
    from research.bl_prefix.readiness import local_inputs, validate_ready

    local = tmp_path / "valid-local-input.json"
    local.write_text("{}")
    with pytest.raises(ValueError, match="uploaded"):
        local_inputs(str(local), {})
    with pytest.raises(ValueError, match="delivery tags"):
        validate_ready({"experiment": "e6", "ready": True, "seed_audit": str(local)}, {})
    assert local_inputs("input:delivered.json", {"delivered.json": local}) == str(local)
