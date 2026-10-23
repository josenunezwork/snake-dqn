"""Resume skips complete schedules and refuses foreign or partial evidence."""

from pathlib import Path

import pytest

from research.bl_labels.exports import publish_json
from research.bl_labels.f1_units import build_ledger, empty_ledger, load_ledger

BANK = "a" * 64


def config(platform: str = "cpu") -> dict:
    return {
        "source_commit": "local-functional",
        "bank_sha256": BANK,
        "platform": platform,
        "schedule_label": "one",
        "status": "COMPLETE",
        "transport_probe": False,
        "root_count": 64,
    }


def test_empty_and_portable_completed_schedule_ledger(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    publish_json(empty, empty_ledger(BANK, "cpu"))
    assert load_ledger(empty, BANK, "cpu") == {}
    original = tmp_path / "original"
    publish_json(original / "units/config-one.json", config())
    ledger = tmp_path / "portable/ledger.json"
    build_ledger([original], ledger, BANK, "cpu")
    assert set(load_ledger(ledger, BANK, "cpu")) == {"one"}
    with pytest.raises(ValueError, match="foreign"):
        load_ledger(ledger, BANK, "l4")


@pytest.mark.parametrize("defect", ["platform", "bank", "partial", "probe", "duplicate"])
def test_no_foreign_or_unverified_schedule_is_published(tmp_path: Path, defect: str) -> None:
    unit = config()
    if defect == "platform":
        unit["platform"] = "l4"
    elif defect == "bank":
        unit["bank_sha256"] = "b" * 64
    elif defect == "partial":
        unit["root_count"] = 63
    elif defect == "probe":
        unit["transport_probe"] = True
    original = tmp_path / "original"
    publish_json(original / "units/config-one.json", unit)
    attempts = [original, original] if defect == "duplicate" else [original]
    with pytest.raises(ValueError):
        build_ledger(attempts, tmp_path / "union/ledger.json", BANK, "cpu")
    assert not (tmp_path / "union/ledger.json").exists()
