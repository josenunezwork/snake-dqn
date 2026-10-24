"""Small refusal checks at the actual R32 worker and reducer entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.bl_ppo.exports import digest
from research.r32_rehearsal import decisions, job, lifecycle, package


def test_worker_cannot_complete_a_failed_independent_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package.prepare_inputs(tmp_path / "inputs", tiny=True)
    monkeypatch.setattr(decisions, "run_case", lambda _: {"pass": False})
    with pytest.raises(ValueError, match="false-ready"):
        job.run_portable(tmp_path / "inputs", tmp_path / "output", mode="tiny")
    assert not (tmp_path / "output/receipt.json").exists()


@pytest.mark.parametrize("stop_verified", [False, True])
def test_reducer_refuses_unknown_stop_or_unbound_supervisor_exports(
    tmp_path: Path, stop_verified: bool
) -> None:
    inputs = tmp_path / "inputs"
    package.prepare_inputs(inputs, tiny=True)
    spec = json.loads((inputs / "spec.json").read_text())
    attempt = tmp_path / "attempt"
    exports = attempt / "exports"
    lifecycle.start_attempt(exports, spec)
    for index in range(3):
        lifecycle.publish_unit(exports, spec, index, {"synthetic": True})
    ledger = tmp_path / "ledger.json"
    lifecycle.build_ledger([exports], spec, ledger)
    manifest = {
        "commit": "a" * 40,
        "script": "research/r32_rehearsal/colab.sh",
        "env": {"R32_MODE": "main"},
        "inputs": [
            {
                "dest": "spec.json",
                "path": str(inputs / "spec.json"),
                "sha256": digest(inputs / "spec.json"),
            },
            {
                "dest": "verified-ledger.json",
                "path": str(inputs / "verified-ledger.json"),
                "sha256": digest(inputs / "verified-ledger.json"),
            },
        ],
    }
    (attempt / "manifest.json").write_text(json.dumps(manifest))
    receipt = attempt / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "stop_verified": stop_verified,
                "status": "PASS",
                "commit": "a" * 40,
                "manifest_sha256": digest(attempt / "manifest.json"),
                "elapsed_seconds": 1,
                "accounting": {"own_rate_units_per_hour": 0.08},
                "exports": [],
            }
        )
    )
    with pytest.raises(ValueError, match="STOP_UNVERIFIED|matching supervisor"):
        job.reduce_verified(inputs, ledger, tmp_path / "terminal.json", [receipt])
    assert not (inputs / "terminal-look.json").exists()
