"""Orchestration contracts for the L4 native-ruler transport wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.bl_served_search import native_l4
from research.bl_served_search.exports import export_json
from research.colab_cli import runtime


def _export(path: Path) -> Path:
    export_json(path, {"status": "input"})
    return path


def _audit(_fixtures: Path, out: Path) -> None:
    export_json(out / "cuda-action-audit.json", {"status": "PASS"})


def test_main_runs_its_own_canary_then_promotes_complete_payloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixtures, k1 = _export(tmp_path / "fixtures.json"), _export(tmp_path / "k1.json")
    calls: list[list[str]] = []
    runtime_root = tmp_path / "runtime"
    output = runtime_root / "output"
    output.mkdir(parents=True)
    scanner_cache: dict = {}

    def fake_native(arguments: list[str]) -> None:
        calls.append(arguments)
        out = Path(arguments[arguments.index("--out") + 1])
        assert not out.exists(), "native_ruler requires a fresh output directory"
        out.mkdir()
        export_json(out / "ruler-receipt.json", {"arguments": arguments})
        # The live scanner may run during the entrypoint.  External staging
        # must not create then remove cached paths below the output root.
        assert runtime._exports(runtime_root, 1024**2, scanner_cache) == []

    monkeypatch.setattr(native_l4, "cuda_action_audit", _audit)
    monkeypatch.setattr(native_l4, "_invoke_native", fake_native)
    native_l4.run(fixtures=fixtures, k1=k1, out=output, max_seconds=1800, canary=False)

    assert len(calls) == 2
    assert "--canary" in calls[0]
    assert "--canary-receipt" in calls[1]
    assert (output / "ruler-receipt.json.complete").is_file()
    assert (output / "cuda-action-audit.json.complete").is_file()
    assert (output / "native-canary" / "ruler-receipt.json.complete").is_file()
    assert (output / "native-canary" / "cuda-action-audit.json.complete").is_file()
    assert runtime._exports(runtime_root, 1024**2, scanner_cache)
    assert runtime._exports(runtime_root, 1024**2, scanner_cache)
    assert not any(path.name.startswith(".native-l4-stage-") for path in output.rglob("*"))


def test_family_canary_uses_root_receipt_and_never_calls_full(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixtures, k1 = _export(tmp_path / "fixtures.json"), _export(tmp_path / "k1.json")
    calls: list[list[str]] = []

    def fake_native(arguments: list[str]) -> None:
        calls.append(arguments)
        out = Path(arguments[arguments.index("--out") + 1])
        assert not out.exists(), "native_ruler requires a fresh output directory"
        out.mkdir()
        export_json(out / "ruler-receipt.json", {"arguments": arguments})

    monkeypatch.setattr(native_l4, "cuda_action_audit", _audit)
    monkeypatch.setattr(native_l4, "_invoke_native", fake_native)
    output = tmp_path / "out"
    output.mkdir()
    native_l4.run(fixtures=fixtures, k1=k1, out=output, max_seconds=600, canary=True)

    assert len(calls) == 1
    assert "--canary" in calls[0]
    assert "--canary-receipt" not in calls[0]
    assert (output / "ruler-receipt.json.complete").is_file()
    assert not (output / "native-canary").exists()


@pytest.mark.parametrize("canary,cap", [(True, 601), (False, 1801)])
def test_caps_fail_before_any_runtime_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, canary: bool, cap: int
) -> None:
    fixtures, k1 = _export(tmp_path / "fixtures.json"), _export(tmp_path / "k1.json")
    monkeypatch.setattr(native_l4, "cuda_action_audit", pytest.fail)
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="invalid native L4 wrapper cap"):
        native_l4.run(fixtures=fixtures, k1=k1, out=output, max_seconds=cap, canary=canary)


def test_wrapper_script_uses_native_archive_validation_and_public_catalog() -> None:
    script = Path(native_l4.__file__).with_name("run_native_l4.sh").read_text()
    assert "--native-source" in script
    assert "research.bl_upload.inputs" in script
    assert "BL_RULER_FIXTURES_SHA256" in script
    assert "SNAKE_DQN_DEVICE=cpu" in script
