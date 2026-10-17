"""Focused contracts for the T11 LOCAL30 seed freshness audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.t11_local30_20261017 import seed_audit


@pytest.fixture(autouse=True)
def _avoid_real_preregistration_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        seed_audit,
        "_frozen_receipt_exclusion",
        lambda _repo, output, _rosters: (((output,) if output is not None else ()), []),
    )


def test_candidate_rosters_are_fixed_96_world_disjoint_rounds() -> None:
    rosters = seed_audit.candidate_rosters()

    assert set(rosters) == {"74", "75"}
    assert all(len(row) == 96 for row in rosters.values())
    assert not (rosters["74"] & rosters["75"])


def test_prior_t7_banks_include_both_rejected_allocations() -> None:
    receipt, seeds = seed_audit._prior_t7_banks()

    assert receipt["rounds"] == [74, 75]
    assert receipt["world_count"] == 1_920
    assert len(seeds) == 1_920


def test_audit_fails_closed_when_visible_result_uses_a_reserved_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    research = tmp_path / "research"
    research.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    seed = min(seed_audit.candidate_rosters()["74"])
    (research / "observed.json").write_text(json.dumps({"world_seed": seed}))

    monkeypatch.setattr(
        seed_audit,
        "_registered_history",
        lambda _repo, _artifacts: ({"fixture": True}, set()),
    )
    monkeypatch.setattr(
        seed_audit,
        "_live_t7_namespace",
        lambda _repo: ({"pass": True, "world_count": 0}, set()),
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit, "t4e_collection_audit", lambda _path: ({"pass": True}, set())
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit, "snapshot_store_audit", lambda _path: ({"pass": True}, set())
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit,
        "training_namespace_audit",
        lambda _repo, _artifacts, _reserved: {"pass": True, "records": []},
    )

    result = seed_audit.audit(tmp_path, artifacts)

    assert result["pass"] is False
    assert result["checks"]["74"]["observed_json_collisions"] == [seed]


def test_audit_excludes_only_its_exact_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    research = tmp_path / "research"
    research.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    seed = min(seed_audit.candidate_rosters()["75"])
    receipt = artifacts / "t11" / "seed_audit.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({"world_seed": seed}))

    monkeypatch.setattr(
        seed_audit,
        "_registered_history",
        lambda _repo, _artifacts: ({"fixture": True}, set()),
    )
    monkeypatch.setattr(
        seed_audit,
        "_live_t7_namespace",
        lambda _repo: ({"pass": True, "world_count": 0}, set()),
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit, "t4e_collection_audit", lambda _path: ({"pass": True}, set())
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit, "snapshot_store_audit", lambda _path: ({"pass": True}, set())
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit,
        "training_namespace_audit",
        lambda _repo, _artifacts, _reserved: {"pass": True, "records": []},
    )

    result = seed_audit.audit(tmp_path, artifacts, receipt)
    assert result["pass"] is True

    sibling = receipt.parent / "actual.json"
    sibling.write_text(json.dumps({"world_seed": seed}))
    result = seed_audit.audit(tmp_path, artifacts, receipt)
    assert result["pass"] is False
    assert result["checks"]["75"]["observed_json_collisions"] == [seed]


def test_running_t7_namespace_collision_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    research = tmp_path / "research"
    research.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    seed = min(seed_audit.candidate_rosters()["74"])

    monkeypatch.setattr(
        seed_audit,
        "_registered_history",
        lambda _repo, _artifacts: ({"fixture": True}, set()),
    )
    monkeypatch.setattr(
        seed_audit,
        "_live_t7_namespace",
        lambda _repo: ({"pass": True, "world_count": 1}, {seed}),
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit, "t4e_collection_audit", lambda _path: ({"pass": True}, set())
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit, "snapshot_store_audit", lambda _path: ({"pass": True}, set())
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit,
        "training_namespace_audit",
        lambda _repo, _artifacts, _reserved: {"pass": True, "records": []},
    )

    result = seed_audit.audit(tmp_path, artifacts)

    assert result["pass"] is False
    assert result["checks"]["74"]["t7_running_namespace_collisions"] == [seed]


def test_metadata_only_round_use_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    research = tmp_path / "research"
    research.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (research / "prior.json").write_text(json.dumps({"round_index": 74, "note": "used"}))

    monkeypatch.setattr(
        seed_audit,
        "_registered_history",
        lambda _repo, _artifacts: ({"fixture": True}, set()),
    )
    monkeypatch.setattr(
        seed_audit,
        "_live_t7_namespace",
        lambda _repo: ({"pass": True, "world_count": 0}, set()),
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit, "t4e_collection_audit", lambda _path: ({"pass": True}, set())
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit, "snapshot_store_audit", lambda _path: ({"pass": True}, set())
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit,
        "training_namespace_audit",
        lambda _repo, _artifacts, _reserved: {"pass": True, "records": []},
    )

    result = seed_audit.audit(tmp_path, artifacts)

    assert result["pass"] is False
    assert result["text_scan"]["reserved_round_metadata"][str(research)] is True


def test_known_t7_round_metadata_does_not_impersonate_t11_development_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    research = tmp_path / "research"
    research.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (research / "t7.json").write_text(
        json.dumps({"namespace": seed_audit.t7_seed_audit.BANK_NS, "round_index": 74})
    )

    monkeypatch.setattr(
        seed_audit,
        "_registered_history",
        lambda _repo, _artifacts: ({"fixture": True}, set()),
    )
    monkeypatch.setattr(
        seed_audit,
        "_live_t7_namespace",
        lambda _repo: ({"pass": True, "world_count": 0}, set()),
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit, "t4e_collection_audit", lambda _path: ({"pass": True}, set())
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit, "snapshot_store_audit", lambda _path: ({"pass": True}, set())
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit,
        "training_namespace_audit",
        lambda _repo, _artifacts, _reserved: {"pass": True, "records": []},
    )

    result = seed_audit.audit(tmp_path, artifacts)

    assert result["pass"] is True
    assert result["text_scan"]["reserved_round_metadata"][str(research)] is False
    assert result["text_scan"]["round_metadata_scope"][str(research)]["known_t7_namespace"]


def test_mixed_jsonl_t7_and_unknown_round_metadata_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    research = tmp_path / "research"
    research.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (research / "mixed.jsonl").write_text(
        "\n".join(
            (
                json.dumps({"namespace": seed_audit.t7_seed_audit.BANK_NS, "round_index": 74}),
                json.dumps({"round_index": 75}),
            )
        )
        + "\n"
    )

    monkeypatch.setattr(
        seed_audit,
        "_registered_history",
        lambda _repo, _artifacts: ({"fixture": True}, set()),
    )
    monkeypatch.setattr(
        seed_audit,
        "_live_t7_namespace",
        lambda _repo: ({"pass": True, "world_count": 0}, set()),
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit, "t4e_collection_audit", lambda _path: ({"pass": True}, set())
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit, "snapshot_store_audit", lambda _path: ({"pass": True}, set())
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit,
        "training_namespace_audit",
        lambda _repo, _artifacts, _reserved: {"pass": True, "records": []},
    )

    result = seed_audit.audit(tmp_path, artifacts)

    assert result["pass"] is False
    assert result["text_scan"]["round_metadata_scope"][str(research)]["unknown_or_development"]


def test_hidden_and_ignored_seed_use_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    research = tmp_path / "research"
    hidden = research / ".hidden" / "used.json"
    ignored = research / "logs" / "used.json"
    hidden.parent.mkdir(parents=True)
    ignored.parent.mkdir()
    (research / ".gitignore").write_text("logs/\n")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    seed = min(seed_audit.candidate_rosters()["74"])
    hidden.write_text(json.dumps({"world_seed": seed}))
    ignored.write_text(json.dumps({"world_seed": seed}))

    monkeypatch.setattr(
        seed_audit,
        "_registered_history",
        lambda _repo, _artifacts: ({"fixture": True}, set()),
    )
    monkeypatch.setattr(
        seed_audit,
        "_live_t7_namespace",
        lambda _repo: ({"pass": True, "world_count": 0}, set()),
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit, "t4e_collection_audit", lambda _path: ({"pass": True}, set())
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit, "snapshot_store_audit", lambda _path: ({"pass": True}, set())
    )
    monkeypatch.setattr(
        seed_audit.t6_seed_audit,
        "training_namespace_audit",
        lambda _repo, _artifacts, _reserved: {"pass": True, "records": []},
    )

    result = seed_audit.audit(tmp_path, artifacts)

    assert result["pass"] is False
    assert result["checks"]["74"]["observed_json_collisions"] == [seed]
