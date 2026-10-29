"""Evidence custody failures must refuse before an envelope can be assessed."""

import json
from pathlib import Path

import pytest

from research.bl_served_search.exports import export_json
from research.watch_lead_p_successor.assess_followup import inventory


def fixture(tmp_path: Path) -> Path:
    profile = tmp_path / "exports/profile"
    profile.mkdir(parents=True)
    export = export_json(profile / "followup.json", {"engineering_only": True})
    export["path"] = "profile/followup.json"
    (tmp_path / "receipt.json").write_text(json.dumps({"exports": [export]}))
    return tmp_path


@pytest.mark.parametrize(
    "damage", ["payload", "marker", "sidecar", "orphan", "duplicate", "missing"]
)
def test_inventory_refuses_damage(tmp_path: Path, damage: str) -> None:
    output = fixture(tmp_path)
    profile = output / "exports/profile"
    if damage == "payload":
        (profile / "followup.json").write_text("{}")
    elif damage == "marker":
        (profile / "followup.json.complete").unlink()
    elif damage == "sidecar":
        (profile / "followup.json.sha256").write_text("foreign  followup.json\n")
    elif damage == "orphan":
        (profile / "orphan.complete").touch()
    else:
        receipt = json.loads((output / "receipt.json").read_text())
        receipt["exports"] = receipt["exports"] * 2 if damage == "duplicate" else []
        (output / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        inventory(output)


def test_inventory_accepts_exact_triple_and_receipt(tmp_path: Path) -> None:
    assert len(inventory(fixture(tmp_path))) == 1


def test_inventory_refuses_symlink_payload(tmp_path: Path) -> None:
    output = fixture(tmp_path)
    payload = output / "exports/profile/followup.json"
    outside = output / "outside.json"
    payload.rename(outside)
    payload.symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe"):
        inventory(output)
