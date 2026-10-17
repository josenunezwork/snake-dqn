"""Tamper tests for the independent T15 evidence audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.t15_mixed_roster_20261017 import audit


def test_audit_pins_the_protocol_independently() -> None:
    assert audit._sha(audit.PROTOCOL) == audit.PROTOCOL_SHA256
    assert audit.PROTOCOL_SHA256 != ""


def test_audit_never_accepts_a_paced_pass_boolean_without_raw_recompute() -> None:
    attempt = {
        "requested_fps": 12,
        "actual_speed": 12,
        "speed_ceiling": 24,
        "total_ai_count": 6,
        "warmup": 200,
        "raw_tick_ms": [1000.0 / 12.0] * 120,
        "overruns": 0,
        "zero_overrun": True,
        "elapsed_seconds": 10.0,
        "actual_achieved_fps": 12.0,
        "roster": {"watch_roster": "mixed", "hero_id": 0},
        "pass": True,
    }
    failures = audit._paced_failures(attempt, 12, 24, 6)
    assert any("overrun" in failure for failure in failures)


def test_parity_digest_survives_sorted_json_roundtrip_and_detects_tampering() -> None:
    stream = [{"slot": 0, "state_sha256": "a" * 64, "q_sha256": "b" * 64, "action": 2}]
    round_tripped = json.loads(json.dumps(stream, sort_keys=True))
    assert audit._canonical_digest(stream) == audit._canonical_digest(round_tripped)
    altered = json.loads(json.dumps(stream, sort_keys=True))
    altered[0]["action"] = 1
    assert audit._canonical_digest(altered) != audit._canonical_digest(stream)


def test_audit_recomputes_scope_from_raw_roster_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    base = {
        "schema_version": "t15-mixed-watch-episode/v1",
        "seed": 1,
        "horizon": 5000,
        "frames_stepped": 5000,
        "roster": {
            "watch_roster": "mixed",
            "served": {"name": "b8-s3", "sha256": audit.B8_SHA256},
            "opponent_checkpoint": {"name": "frp3-s12", "sha256": audit.FRP3_SHA256},
            "veto": {
                "active": True,
                "scope": "watch_opponents",
                "wrapped_snake_ids": [1, 2, 3, 4, 5],
                "counters": {str(i): {} for i in range(1, 6)},
                "strict_receipt_checkpoint_match": True,
                "strict_receipt_wrapper_match": True,
                "wrapper": {
                    "method": audit.V8_METHOD,
                    "source_sha256": "c" * 64,
                },
            },
            "snakes": [
                {
                    "safety_veto": None,
                    "veto_method": None,
                    "policy_is_hero": True,
                    "policy_is_opponent": False,
                },
                *[
                    {
                        "veto_method": audit.V8_METHOD,
                        "veto_head_avoidance": True,
                        "veto_reference_lambda": None,
                        "policy_is_hero": False,
                        "policy_is_opponent": True,
                    }
                    for _ in range(5)
                ],
            ],
        },
    }
    monkeypatch.setattr(audit, "_watch_rows", lambda *_: [base])
    intent = {"banks": {"watch": {"seeds": [1]}}}
    result = audit._check_scope(Path("."), intent, "intent")
    assert result["pass"] is True
    tampered = json.loads(json.dumps(base))
    tampered["roster"]["veto"]["wrapped_snake_ids"] = [0, 1, 2, 3, 4]
    monkeypatch.setattr(audit, "_watch_rows", lambda *_: [tampered])
    result = audit._check_scope(Path("."), intent, "intent")
    assert result["pass"] is False
    assert any("scope" in failure for failure in result["failures"])


def test_continuous_audit_recomputes_dispatch_and_liveness_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation_ids = [[0], *([[]] * 4999)]
    observation_alive = [True, *([False] * 4999)]
    rows = [
        {
            "schema_version": "t15-mixed-watch-episode/v1",
            "seed": seed,
            "horizon": 5000,
            "frames_stepped": 5000,
            "last_error": None,
            "error": None,
            "no_errors": True,
            "identity_checked_frames": 5000,
            "hero": {
                "decisions": 1,
                "deaths": 0,
                "alive_frames": 1,
                "decision_length_histogram": {"5": 1},
                "decision_length_quantiles": {
                    "p50": 5.0,
                    "p95": 5.0,
                    "min": 5.0,
                    "max": 5.0,
                },
            },
            "dispatch": {
                "ids": [0],
                "observation_ids": observation_ids,
                "observation_alive": observation_alive,
            },
        }
        for seed in range(1, 26)
    ]
    monkeypatch.setattr(audit, "_watch_rows", lambda *_: rows)
    intent = {"banks": {"watch": {"seeds": list(range(1, 26))}}}
    result = audit._check_continuous(Path("."), intent, "intent")
    assert result["pass"] is True

    tampered = json.loads(json.dumps(rows))
    tampered[7]["dispatch"]["ids"] = [1]
    monkeypatch.setattr(audit, "_watch_rows", lambda *_: tampered)
    result = audit._check_continuous(Path("."), intent, "intent")
    assert result["pass"] is False
    assert any("dispatch ids" in failure for failure in result["failures"])


def test_controls_audit_rejects_keyword_complete_matrix_with_wrong_observed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = [
        "roster_unset",
        "roster_all_b8",
        "roster_mixed",
        "roster_unknown",
        "checkpoint_b8-s3",
        "checkpoint_frp3-s12",
        "checkpoint_champion",
        "checkpoint_not-a-checkpoint",
        "mixed_variant_unset",
        "mixed_variant_v2",
        "mixed_variant_v5",
        "mixed_variant_v7",
        "mixed_variant_v8",
        "mixed_variant_unknown",
        "mixed_watch_hero_off",
        "mixed_missing_pin",
        "mixed_unfilled_pin",
        "mixed_wrong_pin",
        "mixed_missing_bytes",
        "mixed_corrupt_bytes",
        "mixed_changed_wrapper_source",
        "mode_play_refuses_mixed",
        "train_refusal_b8",
        "mode_watch_rebuilds_mixed",
        "checkpoint_replacement_refused",
        "mixed_roster_count_6",
        "mixed_roster_count_8",
        "mixed_roster_count_12",
        "b8_fallback_chain",
        "checkpoint_replacement_success_b8",
        "play_opt_in_veto_champion",
        "play_opt_in_veto_frp3",
    ]
    rows = [
        {
            "name": name,
            "expected": {
                "scenario": (
                    "unknown roster unknown checkpoint missing unfilled wrong corrupt "
                    "unavailable wrapper mode rebuild replacement play train "
                    "all_b8 mixed watch_opponents"
                )
            },
            "observed": {"watch_roster": "all_b8", "hero_wrapped": False, "errors": []},
            "steps": 1,
            "no_errors": True,
            "pass": True,
        }
        for name in names
    ]
    monkeypatch.setattr(
        audit,
        "_envelope",
        lambda *args, **kwargs: {"data": {"cases": rows}},
    )
    result = audit._check_controls(
        Path("."),
        {"banks": {"controls": {"seeds": [1]}}},
        "intent",
    )
    assert result["pass"] is False
    assert any("roster_mixed" in failure for failure in result["failures"])


def test_receipt_tampering_is_invalid(tmp_path: Path) -> None:
    intent = tmp_path / "intent.json"
    intent.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": audit.RECEIPT_SCHEMA,
                "intent_sha256": audit._sha(intent),
                "files": {"intent.json": hashlib.sha256(b"tampered").hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(audit.Invalid):
        audit._receipt(tmp_path, audit._sha(intent))
