"""Focused contract tests for the round-14 offline critic diagnostic."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from research.bl_ppo.exports import export_bytes
from research.bl_ppo.offline_critic_diagnostic import _metrics, diagnose, load_sidecar, main


def _rows(size: int, *, heldout: bool) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7 if heldout else 6)
    features = rng.normal(size=(size, 257)).astype(np.float32)
    features[:, -1] = np.linspace(0.1, 0.9, size, dtype=np.float32)
    lengths = np.tile(np.array([850, 1050, 1300, 1450], dtype=np.float32), size // 4 + 1)[:size]
    # Return has the registered h signal plus a z*h term; the predeclared
    # interaction head should therefore improve the held-out fit.
    returns = (0.5 * features[:, 0] + 1.5 * features[:, -1] * features[:, 1]).astype(np.float32)
    world_start = 1000 if heldout else 0
    return {
        "features": features,
        "returns": returns,
        "lengths": lengths,
        "mixes": np.tile(np.array(["scripted", "frozen", "mixed", "scripted"]), size // 4 + 1)[
            :size
        ],
        "provenance": np.full(size, "natural"),
        "world_ids": np.arange(world_start, world_start + size, dtype=np.int64),
        "life_ids": np.zeros(size, dtype=np.int64),
    }


def _sidecar(path, *, fixture: bool = False) -> None:
    fit, heldout = _rows(24, heldout=False), _rows(24, heldout=True)
    payload = {f"fit_{key}": value for key, value in fit.items()}
    payload.update({f"heldout_{key}": value for key, value in heldout.items()})
    if fixture:
        payload["heldout_provenance"] = np.array(["fixture_late_body"] * 24)
    _write_sidecar(path, payload)


def _write_sidecar(path, payload: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    export_bytes(path, buffer.getvalue())


def test_diagnostic_is_fit_only_and_reports_mix_length_metrics(tmp_path) -> None:
    sidecar = tmp_path / "rows.npz"
    _sidecar(sidecar)
    fit, heldout, _ = load_sidecar(sidecar)
    result = diagnose(fit, heldout)
    assert result["purpose"] == "offline diagnostic only; no substitution into P1"
    assert result["heldout"]["natural_rows"] == 24
    assert (
        result["heldout_natural_late"]["interaction_z_h_hz"]["mse"]
        < result["heldout_natural_late"]["registered_z_h"]["mse"]
    )
    assert result["heldout_natural_by_mix_and_length"]["scripted"]["800_999"]["baseline_1_h_L"]


def test_fixture_rows_cannot_enter_offline_diagnostic(tmp_path) -> None:
    sidecar = tmp_path / "fixtures-only.npz"
    _sidecar(sidecar, fixture=True)
    with pytest.raises(ValueError, match="no natural heldout rows"):
        load_sidecar(sidecar)


def test_sidecar_refuses_bad_normalized_h_and_world_split_overlap(tmp_path) -> None:
    fit, heldout = _rows(24, heldout=False), _rows(24, heldout=True)
    payload = {f"fit_{key}": value for key, value in fit.items()}
    payload.update({f"heldout_{key}": value for key, value in heldout.items()})
    payload["fit_features"] = payload["fit_features"].copy()
    payload["fit_features"][0, -1] = 1.1
    bad_h = tmp_path / "bad-h.npz"
    _write_sidecar(bad_h, payload)
    with pytest.raises(ValueError, match="normalized h"):
        load_sidecar(bad_h)
    payload["fit_features"][0, -1] = 0.1
    payload["heldout_world_ids"] = payload["heldout_world_ids"].copy()
    payload["heldout_world_ids"][0] = payload["fit_world_ids"][0]
    overlap = tmp_path / "world-overlap.npz"
    _write_sidecar(overlap, payload)
    with pytest.raises(ValueError, match="worlds overlap"):
        load_sidecar(overlap)


def test_explained_variance_removes_constant_bias_as_gate_definition_requires() -> None:
    metrics = _metrics(np.array([1.0, 3.0, 5.0]), np.array([8.0, 10.0, 12.0]))
    assert metrics["explained_variance"] == pytest.approx(1.0)
    assert metrics["bias"] == pytest.approx(7.0)


def test_cli_writes_new_json_without_touching_p1(tmp_path) -> None:
    sidecar, output = tmp_path / "rows.npz", tmp_path / "diagnostic.json"
    _sidecar(sidecar)
    assert main(["--sidecar", str(sidecar), "--out", str(output)]) == 0
    assert json.loads(output.read_text())["schema"] == "bl-ppo-offline-critic-diagnostic/v1"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(["--sidecar", str(sidecar), "--out", str(output)])


def test_hash_checked_chunk_index_is_loaded_without_cross_split_leakage(tmp_path) -> None:
    source = tmp_path / "rows.npz"
    _sidecar(source)
    index = tmp_path / "late-body-rows-index.json"
    export_bytes(
        index,
        json.dumps(
            {
                "chunks": [
                    {
                        "path": source.name,
                        "sha256": source.with_suffix(source.suffix + ".sha256")
                        .read_text()
                        .split()[0],
                        "rows": 48,
                    }
                ]
            },
            sort_keys=True,
        ).encode(),
    )
    fit, heldout, binding = load_sidecar(index)
    assert len(fit["returns"]) == len(heldout["returns"]) == 24
    assert binding["index_sha256"]
    bad = tmp_path / "bad-index.json"
    export_bytes(bad, index.read_bytes().replace(b'"rows": 48', b'"rows": 47'))
    with pytest.raises(ValueError, match="row count mismatch"):
        load_sidecar(bad)
