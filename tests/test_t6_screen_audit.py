"""Focused contracts for the refreshed T6 rounds-72/73 launch audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.t6_screen_20261017 import audit


def _base_receipt(seeds_72: list[int], seeds_73: list[int]) -> dict[str, object]:
    return {
        "pass": True,
        "checks": {
            "72": {"seeds": seeds_72, "registered_collisions": []},
            "73": {"seeds": seeds_73, "registered_collisions": []},
        },
    }


def _t7_banks(seed: int = 1_000) -> dict[str, object]:
    return {
        "banks": {
            f"s{index}": {"seeds": list(range(seed + index * 192, seed + (index + 1) * 192))}
            for index in range(5)
        }
    }


def _t7_receipt(repo: Path, banks: dict[str, object] | None = None) -> Path:
    path = repo / audit.T7_RECEIPT
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": audit.t7_seed_audit.SCHEMA,
                "namespace": audit.t7_seed_audit.BANK_NS,
                "pass": True,
                "banks": banks or _t7_banks()["banks"],
            }
        )
    )
    return path


def _documentary(*_args: object) -> tuple[tuple[Path, ...], list[dict[str, object]]]:
    return (), []


def test_audit_adds_running_t7_namespace_to_both_t6_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    banks = _t7_banks()
    _t7_receipt(tmp_path, banks["banks"])
    monkeypatch.setattr(
        audit.t6_seed_audit, "seed_bank_audit", lambda **_kwargs: _base_receipt([1], [2])
    )
    monkeypatch.setattr(audit, "_validated_documentary", _documentary)
    monkeypatch.setattr(audit.t7_seed_audit, "generate_banks", lambda: banks)
    monkeypatch.setattr(audit.t7_seed_audit, "seed_digest", lambda _seeds: "digest")
    monkeypatch.setattr(
        audit, "_additional_historical_namespaces", lambda _artifacts: ({"pass": True}, set())
    )

    receipt = audit.audit(tmp_path, artifacts=tmp_path / "artifacts", out=tmp_path / "current.json")

    assert receipt["pass"] is True
    assert receipt["excluded_output"] == str((tmp_path / "current.json").resolve())
    assert receipt["checks"]["72"]["t7_phase_r_namespace_collisions"] == []
    assert receipt["checks"]["73"]["t7_phase_r_namespace_collisions"] == []
    assert receipt["t7_phase_r_namespace"]["world_count"] == 960


def test_audit_fails_closed_when_a_t6_round_intersects_running_t7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    banks = _t7_banks()
    _t7_receipt(tmp_path, banks["banks"])
    monkeypatch.setattr(
        audit.t6_seed_audit, "seed_bank_audit", lambda **_kwargs: _base_receipt([1_000], [2])
    )
    monkeypatch.setattr(audit, "_validated_documentary", _documentary)
    monkeypatch.setattr(audit.t7_seed_audit, "generate_banks", lambda: banks)
    monkeypatch.setattr(audit.t7_seed_audit, "seed_digest", lambda _seeds: "digest")
    monkeypatch.setattr(
        audit, "_additional_historical_namespaces", lambda _artifacts: ({"pass": True}, set())
    )

    receipt = audit.audit(tmp_path)

    assert receipt["pass"] is False
    assert receipt["status"] == "FAIL"
    assert receipt["checks"]["72"]["t7_phase_r_namespace_collisions"] == [1_000]


def test_audit_fails_closed_without_a_passing_bound_t7_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        audit.t6_seed_audit, "seed_bank_audit", lambda **_kwargs: _base_receipt([1], [2])
    )
    monkeypatch.setattr(audit, "_validated_documentary", _documentary)
    monkeypatch.setattr(audit.t7_seed_audit, "generate_banks", _t7_banks)
    monkeypatch.setattr(audit.t7_seed_audit, "seed_digest", lambda _seeds: "digest")
    monkeypatch.setattr(
        audit, "_additional_historical_namespaces", lambda _artifacts: ({"pass": True}, set())
    )

    receipt = audit.audit(tmp_path)

    assert receipt["pass"] is False
    assert receipt["t7_phase_r_namespace"]["receipt"]["exists"] is False
    assert receipt["t7_phase_r_namespace"]["pass"] is False


def test_t7_namespace_rejects_duplicate_generated_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid = _t7_banks()
    _t7_receipt(tmp_path, invalid["banks"])
    invalid["banks"]["s1"]["seeds"] = invalid["banks"]["s0"]["seeds"]
    monkeypatch.setattr(audit.t7_seed_audit, "generate_banks", lambda: invalid)

    with pytest.raises(ValueError, match="duplicate"):
        audit._t7_namespace(tmp_path)


def test_audit_fails_closed_on_later_historical_namespace_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    banks = _t7_banks()
    _t7_receipt(tmp_path, banks["banks"])
    monkeypatch.setattr(
        audit.t6_seed_audit, "seed_bank_audit", lambda **_kwargs: _base_receipt([8], [9])
    )
    monkeypatch.setattr(audit, "_validated_documentary", _documentary)
    monkeypatch.setattr(audit.t7_seed_audit, "generate_banks", lambda: banks)
    monkeypatch.setattr(audit.t7_seed_audit, "seed_digest", lambda _seeds: "digest")
    monkeypatch.setattr(
        audit, "_additional_historical_namespaces", lambda _artifacts: ({"pass": True}, {9})
    )

    receipt = audit.audit(tmp_path)

    assert receipt["pass"] is False
    assert receipt["checks"]["73"]["additional_historical_namespace_collisions"] == [9]


def test_t7_receipt_must_bind_the_current_generated_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    banks = _t7_banks()
    _t7_receipt(tmp_path, banks["banks"])
    banks["banks"]["s0"]["seeds"][0] = 999_999
    monkeypatch.setattr(audit.t7_seed_audit, "generate_banks", lambda: banks)

    receipt, _seeds = audit._t7_namespace(tmp_path)

    assert receipt["pass"] is False
    assert receipt["receipt"]["roster_matches_generated"] is False
