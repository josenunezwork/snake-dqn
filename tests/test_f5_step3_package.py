"""Offline package guards do not imply a measured runtime forecast."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.bl_residual import f5_step3_package as package
from research.bl_residual.f5_environment_reference import CAPTURE05_SIZE


def test_export_envelope_fits_the_known_raw_native_environment_size() -> None:
    assert package.MAX_EXPORT_BYTES == 64 * 1024 * 1024
    assert package.MAX_EXPORT_BYTES > CAPTURE05_SIZE


def test_finite_envelope_doubles_once_and_preserves_archive_guards() -> None:
    value = package.envelope()
    assert value["effective_timeout_minutes"] == 2 * value["base_timeout_minutes"]
    assert value["aggregate_effective_timeout_minutes"] == 3 * value["effective_timeout_minutes"]
    assert value["global_completion_rounds_maximum"] == 2
    assert value["archive_stage_base_hours"] == 96
    assert value["archive_stage_effective_hours"] == 192
    assert value["archive_shards"] == 16
    assert value["unknown_costs"]
    assert value["forecast_status"].startswith("HOLD_")


def test_package_refuses_existing_destination_before_reading_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fresh"):
        package.build(tmp_path)


def test_input_reference_refuses_symlink(tmp_path: Path) -> None:
    original = tmp_path / "input.json"
    original.write_text("{}")
    alias = tmp_path / "alias.json"
    alias.symlink_to(original)
    with pytest.raises(ValueError, match="unsafe"):
        package.reference(alias)
