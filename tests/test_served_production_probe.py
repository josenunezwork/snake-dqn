"""Focused checks for the isolated real-Hub production trace arm."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.bl_served_search import production_probe as probe


def test_settings_only_allows_frozen_or_smaller_smoke() -> None:
    assert probe._settings({"seed": 2, "warmup_ticks": 0, "measured_ticks": 2}) == (2, 0, 2, 24)
    with pytest.raises(ValueError, match="frozen"):
        probe._settings({"seed": 2, "fps": 12})
    with pytest.raises(ValueError, match="frozen"):
        probe._settings({"seed": 2, "measured_ticks": probe.MEASURED_TICKS + 1})


def test_frame_id_preserves_product_payload_shape() -> None:
    assert probe._frame_id({"frame": 7}) == 7
    assert probe._frame_id({"game": {"frame": 8}}) == 8
    assert probe._frame_id({"frame": "8"}) is None


@pytest.mark.integration
@pytest.mark.skipif(
    not all(
        Path("saved_snakes", name).is_file()
        for name in ("b8_s3_t7_20261016.pth", "frp3_m3_s12_u60000_20261005.pth")
    ),
    reason="actual Hub smoke requires the local released checkpoint artifacts",
)
def test_smoke_runs_actual_hub_and_loopback_websocket() -> None:
    result = probe.run_production_arm(
        {"arm": "production", "seed": 2026102601, "warmup_ticks": 0, "measured_ticks": 2},
        timeout_seconds=120,
    )
    assert result["clock_kind"] == "hub_relative_sleep_not_absolute"
    assert result["absolute_overrun_applicable"] is False
    assert len(result["trace"]) == 2
    assert all(row["spans"]["game_update"]["calls"] == 1 for row in result["trace"])
    assert len(result["frame_send"]) == 2
    assert result["engine_error"] is None
    assert sum(row["source"] == "connect_snapshot" for row in result["client_receive"]) == 1
    assert sum(row["source"] == "hub_broadcast" for row in result["client_receive"]) == 2
