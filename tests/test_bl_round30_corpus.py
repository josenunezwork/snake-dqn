"""Closed-F4 exact-list guards and fixed-selector mechanics, synthetic only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.bl_round30 import admission, corpus, materialize
from research.bl_round30.protocol import AMENDMENT_SHA, SALT


def _provenance() -> dict:
    return {
        "kind": "closed_f4_training",
        "original_f4": True,
        "terminal_label": "STOP_FUNDING",
        "reuse_ruling_sha": AMENDMENT_SHA,
        "eligible_life_count": 960,
        "original_folds": [1, 2, 3, 4],
        "dose_allocation_sha": corpus.ALLOCATION_SHA,
        "admitted_life_set_sha": "500daa506069ca01450b3dd6dd272b6e965b4e8207cae2620070b91cf0d9a250",
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("original_f4", False),
        ("terminal_label", "GO"),
        ("reuse_ruling_sha", "a" * 64),
        ("eligible_life_count", 1152),
        ("original_folds", [0, 1, 2, 3, 4]),
        ("admitted_life_set_sha", "a" * 64),
    ],
)
def test_forged_or_overbroad_f4_exception_refuses(key: str, value: object) -> None:
    record = _provenance()
    admission._require_closed_f4(record)
    record[key] = value
    with pytest.raises(admission.AdmissionRefused):
        admission._require_closed_f4(record)


def test_foreign_provenance_refuses_even_when_f4_false() -> None:
    value = {
        "collection_authority": "test",
        "source_sha256": "a" * 64,
        "label_sha256": "a" * 64,
        "checkpoint_sha256": "a" * 64,
        "explicitly_reusable_development": True,
        "already_unsealed": True,
        "pins_current": True,
        **{
            key: False
            for key in (
                "f4",
                "protected",
                "strict",
                "retention",
                "natural_archive",
                "probe",
                "gameplay",
                "stale",
            )
        },
        "provenance": {"kind": "other_f4", "original_f4": True},
    }
    with pytest.raises(admission.AdmissionRefused, match="unsupported"):
        admission._require_corpus_authority(value)


def test_metadata_reader_refuses_forbidden_paths_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*args: object) -> bytes:
        raise AssertionError("payload opener reached")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    for name in ("d4096", "t34-round0-study-01"):
        with pytest.raises(admission.AdmissionRefused, match="forbidden"):
            corpus.metadata(tmp_path / name / "metadata.json", "a" * 64, [])


def test_exact_life_count_and_duplicate_nested_identity_refuse() -> None:
    for size in (192, 959, 961, 1152):
        with pytest.raises(admission.AdmissionRefused, match="exactly 960"):
            corpus.validate_life_inventory({"lives": [{}] * size})
    with pytest.raises(admission.AdmissionRefused, match="duplicate"):
        corpus.validate_life_inventory({"lives": [{"source_life_id": "same"}] * 960})


def test_parent_hash_has_fixed_salt_and_no_tranche_local_alias() -> None:
    for parent in ("d256-mac-watch6-001", "d1024-mac-watch6-001"):
        raw = hashlib.sha256((SALT + "\0" + parent).encode()).digest()
        expected = "validation" if int.from_bytes(raw, "big") % 10 < 2 else "train"
        assert admission.parent_split(parent, SALT) == expected
    assert "namespace-a/parent/0" != "namespace-b/parent/0"


def test_materializer_refuses_missing_approval_before_payload_opener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"schema": "invalid"}))
    approval = tmp_path / "approval.json"
    approval.write_text("{}")

    def forbidden(*args: object) -> bytes:
        raise AssertionError("payload opener reached")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(admission.AdmissionRefused):
        materialize.materialize(inventory, approval, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_deterministic_lossless_private_npz() -> None:
    import io

    import numpy as np

    table = {"z": np.array(["namespace/parent/0"]), "a": np.arange(20, dtype=np.float32)}
    first = materialize._npz_bytes(table)
    assert first == materialize._npz_bytes(dict(reversed(list(table.items()))))
    with np.load(io.BytesIO(first), allow_pickle=False) as loaded:
        assert loaded.files == ["a", "z"]
        for key in table:
            np.testing.assert_array_equal(table[key], loaded[key])


def test_fixed_selector_preserves_all_rows_and_global_identity() -> None:
    import numpy as np

    scalars = np.zeros(60, dtype=np.float32)
    scalars[1], scalars[3] = np.float32(10 / 1000), 1
    row = {
        "local": np.zeros((7, 31, 31), dtype=np.uint8),
        "global": np.zeros((4, 37, 37), dtype=np.uint8),
        "scalars": scalars,
        "q": np.zeros(6, dtype=np.float32),
        "legal": [True] * 6,
        "bare": 0,
        "status": ["EXACT", "UNKNOWN", None, "EXACT", "EXACT", "EXACT"],
        "died": [False, None, None, False, False, False],
        "mass": [10.0, None, None, 11.0, 12.0, 13.0],
        "length": 10,
        "decision": 7,
        "sampling_kind": "ordinary",
        "sampling_prob": 1.0,
    }
    lives = [
        {
            "source_life_id": f"namespace-{i}/parent-{i}/0",
            "global_life_index": i,
            "ultimate_parent_id": f"parent-{i}",
        }
        for i in range(2)
    ]
    table = materialize._table([(life, dict(row)) for life in lives])
    assert table["life_id"].tolist() == [life["source_life_id"] for life in lives]
    assert table["source_life_id"].tolist() == table["life_id"].tolist()
    assert table["global_life_index"].tolist() == [0, 1]
    assert len(set(table["row_id"].tolist())) == 2
    assert table["status"][0].tolist()[1:3] == ["UNKNOWN", "NOT_EVALUATED"]
    assert table["ordinary"].tolist() == [True, True]
    assert table["natural_stream"].tolist() == [True, True]
    duplicate = [(lives[0], dict(row)), (lives[0], dict(row))]
    with pytest.raises(ValueError, match="duplicate row"):
        materialize._table(duplicate)


def test_selector_source_pin_refuses_before_any_payload_read(tmp_path, monkeypatch) -> None:
    """Metadata acceptance cannot authorize a subsequently edited selector."""
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}")
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "approval_scope": "closed-f4-exact-960-metadata",
                "reviewer_role": "Codex-V",
                "selector_sha256": "a" * 64,
            }
        )
    )
    monkeypatch.setattr(materialize, "freeze_inventory", lambda _: object())
    monkeypatch.setattr(materialize, "require_codex_v_approval", lambda *_: None)
    monkeypatch.setattr(materialize, "admission_status", lambda _: "HOLD_R30_2")
    original = Path.read_bytes

    def guarded_read(path):
        if path != Path(materialize.__file__):
            raise AssertionError("payload opener reached before selector pin refusal")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(admission.AdmissionRefused, match="selector source differs"):
        materialize.materialize(inventory, approval, tmp_path / "output")
    assert not (tmp_path / "output").exists()
