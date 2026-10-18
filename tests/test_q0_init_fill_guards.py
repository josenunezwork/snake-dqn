"""Regression guards for reusing immutable pre-A2 fidelity evidence."""

from __future__ import annotations

import hashlib
import json

import pytest

from research.q0q1_20261017 import audit_b8_init as subject


def _source(tmp_path, monkeypatch):
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"test-b8")
    digest = hashlib.sha256(b8.read_bytes()).hexdigest()
    monkeypatch.setattr(subject, "B8_SHA256", digest)
    rows = [
        {
            "mix": ("scripted", "frozen", "mixed")[index % 3],
            "short_lt700": True,
            "long_ge800": True,
            "crowded": True,
            "open": True,
            "boost_eligible": True,
            "action_a": 1,
            "action_b": 1,
        }
        for index in range(4095)
    ]
    rows.append(subject.exact_tie_fixture())
    packet = {
        "b8": {"sha256": digest},
        "development_packet": {
            "fidelity_corpus": {
                "rows": rows,
                "bank_audit_pass": True,
                "candidate_observations_inspected": 684993,
                "excluded_trapped_observations": 17,
            }
        },
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(packet))
    return b8, path, packet


def test_no_fill_cannot_pass_by_crediting_legacy_global_open(tmp_path, monkeypatch):
    b8, path, _ = _source(tmp_path, monkeypatch)
    result = subject.run_observations(b8, path)
    assert result["status"] == "INCOMPLETE"
    assert result["missing_coverage"] == ["open"]
    assert result["observations"] == 4096
    assert result["additional_episodes"] == 0
    assert result["candidate_observations_inspected"] == 684993
    assert result["excluded_trapped_observations"] == 17


@pytest.mark.parametrize("error", ["action", "wrapper", "mask", "fallback"])
def test_invalid_saved_fixture_is_rejected_before_replacement(tmp_path, monkeypatch, error):
    b8, path, packet = _source(tmp_path, monkeypatch)
    fixture = packet["development_packet"]["fidelity_corpus"]["rows"][-1]
    if error == "action":
        fixture["action_b"] = 5
    else:
        fixture[
            {
                "wrapper": "wrapper_disagreement",
                "mask": "mask_disagreement",
                "fallback": "fallback_used",
            }[error]
        ] = True
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="disagreement|fail-closed"):
        subject.run_observations(b8, path)
