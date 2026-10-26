"""Synthetic-only checks for explicit external seed metadata preparation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from research.bl_residual import f5_r10a_annex as original
from research.bl_residual.f5_r10a_closure_20261026 import external_collision_metadata as ext


def parse(raw: bytes, count: int) -> list[int]:
    return list(ext.integer_array(raw, count, time.monotonic() + 10))


def test_preserve_negative_and_full_uint64_domains() -> None:
    assert parse(b"[-659,0,9223372036854775808,18446744073709551615]", 4) == [
        -659,
        0,
        2**63,
        2**64 - 1,
    ]


@pytest.mark.parametrize(
    "raw,count",
    [
        (b"[1,1]", 2),
        (b"[2,1]", 2),
        (b"[1,]", 1),
        (b"[1.0]", 1),
        (b"[true]", 1),
        (b'{"seed":1,"outcome":7}', 1),
        (b"[18446744073709551616]", 1),
        (b"[1] trailing", 1),
        (b"[1]", 2),
        (b"[1,2]", 1),
        (b"[]", 1),
    ],
)
def test_strict_integer_metadata_only_schema(raw: bytes, count: int) -> None:
    with pytest.raises(ValueError):
        parse(raw, count)


def test_empty_integer_array_and_union_deduplication() -> None:
    assert parse(b" [ ] \n", 0) == []
    one = ext.integer_array(b"[-2,1,3]", 3, time.monotonic() + 10)
    two = ext.integer_array(b"[-2,0,3,18446744073709551615]", 4, time.monotonic() + 10)
    assert list(ext.union_values([one, two])) == [-2, 0, 1, 3, 2**64 - 1]


@pytest.mark.parametrize("token", ["D4096", "t34-round0-study-01"])
def test_protected_paths_refused_before_io(tmp_path: Path, token: str) -> None:
    with pytest.raises(ValueError, match="protected"):
        ext.safe_read(tmp_path / token / "training_exclusions.json", "0" * 64)


def test_hash_symlink_and_size_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "training_exclusions.json"
    path.write_bytes(b"[1]")
    with pytest.raises(ValueError, match="seal"):
        ext.safe_read(path, "0" * 64)
    with pytest.raises(ValueError, match="size"):
        ext.safe_read(path, ext.digest(path.read_bytes()), bound=2)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        ext.safe_read(link, ext.digest(path.read_bytes()))


def test_pruning_exactly_preserves_unchanged_audit_reasons() -> None:
    bank = original.allocate(0)
    historical = [
        {"parent_id": "raw-hit", "seed": bank[0]["seed"], "aliases": []},
        {"parent_id": "alias-hit", "seed": -7, "aliases": [bank[1]["parent_id"]]},
        {"parent_id": "irrelevant", "seed": -8, "aliases": ["foreign/parent"]},
    ]
    report = ext.audit_union(historical, [], rounds=2)
    assert report["rejected_rounds"][0]["reasons"] == original.collision_audit(bank, historical)
    assert report["provisional_round"] == 1
    assert report["accepted_round"] is None
    assert report["accepted_allocation_sha256"] is None
    assert report["all_684_executed"] is True


def test_internal_collision_checked_when_no_external_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    bank = original.allocate(0)
    bank[1]["seed"] = bank[0]["seed"]
    monkeypatch.setattr(original, "allocate", lambda _: bank)
    report = ext.audit_union([], [], rounds=1)
    assert report["status"] == "HOLD_NO_CLEAR_ROUND"
    assert report["rejected_rounds"][0]["reasons"][0]["kind"] == "internal_collision"


def test_error_is_redacted() -> None:
    secret = b"FORBIDDEN_OUTCOME_CANARY"
    with pytest.raises(ValueError) as failure:
        parse(b"[" + secret + b"]", 1)
    assert secret.decode() not in str(failure.value)


def test_publication_never_replaces_payload_or_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    ext.write_sealed(path, {"seed": 1})
    original_bytes = path.read_bytes()
    sidecar = Path(f"{path}.sha256")
    original_sidecar = sidecar.read_bytes()
    with pytest.raises(ValueError, match="existing"):
        ext.write_sealed(path, {"seed": 2})
    assert path.read_bytes() == original_bytes
    assert sidecar.read_bytes() == original_sidecar
    path.unlink()
    with pytest.raises(ValueError, match="existing"):
        ext.write_sealed(path, {"seed": 3})
    assert not path.exists()
    assert sidecar.read_bytes() == original_sidecar


def test_atomic_create_refuses_existing_destination(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_bytes(b"immutable")
    with pytest.raises(FileExistsError):
        ext.atomic_create(path, b"replacement")
    assert path.read_bytes() == b"immutable"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["metadata.json"]


@pytest.mark.parametrize("token", ["D4096", "t34-round0-study-01"])
def test_excluded_output_refused_before_mkdir_or_inputs(tmp_path: Path, token: str) -> None:
    output = tmp_path / token / "nested" / "output"
    with pytest.raises(ValueError, match="protected output"):
        ext.build(tmp_path, tmp_path, output, tmp_path / "nonexistent.json")
    assert not (tmp_path / token).exists()


def test_output_symlink_parent_and_existing_directory_refused(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    destination.mkdir()
    link = tmp_path / "link"
    link.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        ext.build(tmp_path, tmp_path, link / "output", tmp_path / "missing")
    assert not (destination / "output").exists()
    with pytest.raises(ValueError, match="existing output directory"):
        ext.build(tmp_path, tmp_path, destination, tmp_path / "missing")
    assert list(destination.iterdir()) == []


def test_completion_marker_binds_all_payloads_and_sidecars_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    register = repo / "research/t34_round0/SEED_AUDIT_ROUND0.json"
    register.parent.mkdir(parents=True)
    register.write_bytes(ext.canonical({"historical_frp_exclusions": []}))
    monkeypatch.setattr(ext, "REGISTER_SHA256", ext.digest(register.read_bytes()))
    monkeypatch.setattr(ext, "FROZEN", ())
    monkeypatch.setattr(ext, "load_repository", lambda _: ([], {}))
    bank = original.allocate(0)
    monkeypatch.setattr(original, "allocate", lambda _: bank)
    output = tmp_path / "fresh-output"
    calls = []
    real_write = ext.write_sealed

    def observe(path: Path, value: object) -> dict:
        calls.append(path.name)
        return real_write(path, value)

    monkeypatch.setattr(ext, "write_sealed", observe)
    report = ext.build(repo, tmp_path, output, tmp_path / "unused-catalog.json")
    assert report["disposition"] == "HOLD"
    assert calls[-2:] == ["external-collision-report.json", "COMPLETED.json"]
    marker = json.loads((output / "COMPLETED.json").read_bytes())
    expected_names = {"access-plan.json", "external-collision-report.json"}
    assert {binding["path"] for binding in marker["artifacts"]} == expected_names
    for binding in marker["artifacts"]:
        assert ext.digest((output / binding["path"]).read_bytes()) == binding["sha256"]
        sidecar = binding["sidecar"]
        assert ext.digest((output / sidecar["path"]).read_bytes()) == sidecar["sha256"]
