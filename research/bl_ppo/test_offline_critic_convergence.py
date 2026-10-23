"""Small fail-closed tests for the offline-only amendment 03 replay CLI."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from research.bl_ppo.exports import digest, export_bytes, export_json, verify_export
from research.bl_ppo.offline_critic_convergence import main, replay
from research.bl_ppo.offline_critic_diagnostic import diagnose, load_sidecar
from research.bl_ppo.test_offline_critic_diagnostic import _rows


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    fit, heldout = _rows(24, heldout=False), _rows(24, heldout=True)
    for rows in (fit, heldout):
        rows["features"][:, :256] -= rows["features"][:, :256].mean(axis=1, keepdims=True)
    fit["world_ids"] = np.arange(24, dtype=np.int64) * 5 + 1
    heldout["world_ids"] = np.arange(24, dtype=np.int64) * 5 + 1000
    arrays = {f"fit_{key}": value for key, value in fit.items()}
    arrays.update({f"heldout_{key}": value for key, value in heldout.items()})
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    sidecar, reference = tmp_path / "rows.npz", tmp_path / "diagnostic.json"
    export_bytes(sidecar, buffer.getvalue())
    fit, heldout, binding = load_sidecar(sidecar)
    diagnostic = diagnose(fit, heldout)
    diagnostic["input_binding"] = binding
    export_json(reference, diagnostic)
    return sidecar, reference


def test_replay_refuses_identity_even_when_synthetic_fit_reproduces(tmp_path) -> None:
    sidecar, reference = _inputs(tmp_path)
    original_hashes = digest(sidecar), digest(reference)
    result = replay(sidecar, reference)
    assert result["status"] == "FIT_ACCEPTANCE_FAIL"
    assert result["fit_acceptance"]["old_z_h_fit_mse"] >= 0.0
    assert result["fit_acceptance"]["checks"]["rank"] is False
    assert (digest(sidecar), digest(reference)) == original_hashes
    assert result["actor_transitions"] == result["provider_calls"] == 0


def test_replay_refuses_foreign_d1_binding(tmp_path) -> None:
    sidecar, reference = _inputs(tmp_path)
    value = json.loads(reference.read_text())
    value["input_binding"]["sidecar_sha256"] = "0" * 64
    foreign = tmp_path / "foreign.json"
    export_json(foreign, value)
    with pytest.raises(ValueError, match="input binding differs"):
        replay(sidecar, foreign)


def test_cli_preserves_inputs_exports_mismatch_and_refuses_overwrite(tmp_path) -> None:
    sidecar, reference = _inputs(tmp_path)
    output = tmp_path / "replay.json"
    argv = ["--sidecar", str(sidecar), "--reference", str(reference), "--out", str(output)]
    assert main(argv) == 1
    verify_export(output)
    before = digest(output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)
    assert digest(output) == before
