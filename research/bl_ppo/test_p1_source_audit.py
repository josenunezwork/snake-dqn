"""Saved-export finalisation audit never manufactures a source result."""

from __future__ import annotations

import pytest

from research.bl_ppo.exports import export_json, export_torch
from research.bl_ppo.p1_source_audit import audit
from research.bl_ppo.recovery import SCHEMA


def _pack(tmp_path):
    pack = tmp_path / "pack"
    source = pack / "source-output"
    export_json(
        source / "receipt.json",
        {
            "status": "INCOMPLETE",
            "failure_class": "CHILD_NONZERO_EXIT",
            "stop_verified": True,
            "cleanup_error": None,
        },
    )
    for phase, actor, updates in (("origin", 0, 0), ("source-continuation", 49152, 3)):
        root = source / "exports" / phase
        counters = {
            "actor_transitions": actor,
            "actor_updates": updates,
            "actor_episode_boundaries": updates,
        }
        snapshot = export_torch(
            root / "pause.pt",
            {
                "schema": SCHEMA,
                "identity": {},
                "policy": {},
                "learner": {},
                "collector": {},
                "adapter": {},
                "rng": {},
                "counters": counters,
                "execution": {"phase": "actor"},
                "boundary": {
                    "requested_pause_at": 0,
                    "pause_at": 0,
                    "actual_pause_at": actor,
                    "kind": "exact" if phase == "origin" else "durability",
                },
            },
        )
        export_json(
            root / "training-status.json",
            {
                "status": "PAUSED",
                "critic_qualified": True,
                "counters": counters,
                "pause_at": 0,
                "actual_pause_at": actor,
                "requested_pause_at": 0,
                "snapshot": {**snapshot, "path": "pause.pt"},
                "action_trace": {"size": 1000},
            },
        )
    return pack


def test_audit_preserves_gate_and_refuses_to_invent_periodic_proof(tmp_path):
    pack = _pack(tmp_path)
    output = tmp_path / "audit.json"
    value = audit(pack, output)
    assert value["status"] == "HOLD_MISSING_OPERATIONAL_PROOF"
    assert value["critic_qualified"] is True
    assert value["origin_has_operational_label"] is False
    assert value["offline_complete_source_continuation"] is False
    assert value["provider_calls"] == 0
    assert not value["new_gameplay"]
    assert not value["refit_or_gate_rerun"]
    assert not (pack / "source-output/exports/receipt.json").exists()
    with pytest.raises(FileExistsError):
        audit(pack, output)


def test_audit_refuses_unverified_snapshot_bytes(tmp_path):
    pack = _pack(tmp_path)
    path = pack / "source-output/exports/origin/pause.pt"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest|SHA|hash"):
        audit(pack, tmp_path / "audit.json")
