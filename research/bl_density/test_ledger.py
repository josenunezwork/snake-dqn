from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.bl_density.ledger import SCHEMA, build_ledger, validate_ledger
from research.bl_ppo.exports import digest, export_json

NAMESPACE = "bl-density-round14b-20261022"


def test_build_ledger_uses_export_metadata_and_terminal_inventory(tmp_path: Path) -> None:
    episodes = tmp_path / "exports/density/episodes"
    episode = episodes / f"{NAMESPACE}_A_scripted_000.json"
    export_json(episode, {"outcome": "test payload builder must not inspect this"})
    inventory = tmp_path / "poll-000015.json"
    inventory.write_text(
        json.dumps(
            {
                "exports": [
                    {
                        "path": f"density/episodes/{episode.name}",
                        "sha256": digest(episode),
                        "size": episode.stat().st_size,
                    }
                ]
            }
        )
    )
    ledger = build_ledger(episodes, inventory, namespace=NAMESPACE)
    assert ledger == {
        "schema": SCHEMA,
        "namespace": NAMESPACE,
        "entries": [
            {
                "episode_id": f"{NAMESPACE}/A/scripted/000",
                "sha256": digest(episode),
                "size": episode.stat().st_size,
            }
        ],
    }


def test_ledger_validation_refuses_foreign_or_duplicate_entries() -> None:
    allocation = {"A": {"scripted": 1}}
    entry = {
        "episode_id": f"{NAMESPACE}/A/scripted/000",
        "sha256": "a" * 64,
        "size": 1,
    }
    with pytest.raises(ValueError, match="duplicate"):
        validate_ledger(
            {"schema": SCHEMA, "namespace": NAMESPACE, "entries": [entry, entry]},
            namespace=NAMESPACE,
            allocation=allocation,
        )
    with pytest.raises(ValueError, match="namespace"):
        validate_ledger(
            {"schema": SCHEMA, "namespace": "foreign", "entries": [entry]},
            namespace=NAMESPACE,
            allocation=allocation,
        )


def test_boolean_byte_size_is_not_valid_ledger_metadata() -> None:
    with pytest.raises(ValueError, match="id/hash/size"):
        validate_ledger(
            {
                "schema": SCHEMA,
                "namespace": NAMESPACE,
                "entries": [
                    {"episode_id": f"{NAMESPACE}/A/scripted/000", "sha256": "a" * 64, "size": True}
                ],
            },
            namespace=NAMESPACE,
            allocation={"A": {"scripted": 1}},
        )
