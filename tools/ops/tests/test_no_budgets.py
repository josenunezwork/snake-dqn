"""Unit prices cannot shrink builder wall-clock envelopes or reject work."""

import json
from types import SimpleNamespace

import pytest

from tools.ops import flood_sparse_main, r2_packet, spend


def test_flood_prior_unit_spend_only_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(flood_sparse_main, "BASE", tmp_path)
    receipt = tmp_path / (
        "snake-dqn-artifacts/big-lever/exp/gpu-flood-sparse-main-test/main-output/receipt.json"
    )
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"accounting": {}, "stop_verified": True}))
    monkeypatch.setattr(
        spend,
        "verify_receipts",
        lambda _: [SimpleNamespace(path=receipt, sha256="a" * 64, elapsed_seconds=60, units=1000)],
    )
    evidence = flood_sparse_main.prior_spend()
    assert evidence["committed_units"] > 36
    assert evidence["committed_wall_minutes"] == 161
    assert '"WARNING": "FLOOD_UNIT_THRESHOLD"' in capsys.readouterr().err


def test_r2_exhausted_unit_allowance_keeps_original_wall_cap(tmp_path, capsys):
    first = {
        "job_id": "canary-01",
        "output_dir": str(tmp_path / "first"),
        "budget": {"rate_units_per_hour": 10},
        "timeout_minutes": 12,
        "requirements": {"est_minutes": 6},
        "env": {r2_packet.FAMILY.producer_key: "420"},
    }
    result = r2_packet.attempt_manifest(first, 1000, tmp_path / "second")
    assert result["timeout_minutes"] == 12
    assert result["budget"]["unit_cap"] == 0
    assert result["env"] == first["env"]
    assert '"WARNING": "CANARY_UNIT_THRESHOLD"' in capsys.readouterr().err
    with pytest.raises(ValueError, match="exactly one"):
        r2_packet.attempt_manifest(first, 0, tmp_path / "third", attempt=3)
