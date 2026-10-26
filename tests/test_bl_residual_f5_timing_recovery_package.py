"""Offline sealing tests for the F5 timing04 recovery package."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.bl_residual import f5_timing_recovery_package as package


def test_predecessor_environment_metadata_must_match_manifest(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_bytes(b'{"environment_policy": {}, "environment_capture": {}}')
    row = {
        "dest": "control.json",
        "path": str(path),
        "sha256": package.file_sha256(path),
        "size_bytes": path.stat().st_size,
    }
    reference = package._prior_environment_control({"inputs": [row]})
    assert reference["kind"] == "PINNED_V5_SUCCESS_COMPARISON_ONLY"
    path.write_bytes(b'{"environment_policy": {}, "environment_capture": {"changed": true}}')
    with pytest.raises(ValueError, match="hash or size differs"):
        package._prior_environment_control({"inputs": [row]})


def test_authority_is_canonical_and_covers_exact_transfer_rows() -> None:
    rows = [
        {"dest": "control.json", "path": "/tmp/control.json", "sha256": "a" * 64, "size_bytes": 9},
        {
            "dest": "b8.pth",
            "url": package.B8_URL,
            "sha256": package.B8_SHA256,
            "size_bytes": 6_562_005,
        },
    ]
    authority = package._authority("b" * 40, "c" * 40, rows)
    unsigned = dict(authority)
    supplied = unsigned.pop("sha256")
    assert supplied == package.hashlib.sha256(package.canonical(unsigned)).hexdigest()
    assert authority["inputs"] == [
        {"dest": "control.json", "size_bytes": 9, "transport": "upload", "sha256": "a" * 64},
        {
            "dest": "b8.pth",
            "size_bytes": 6_562_005,
            "transport": "https",
            "url": package.B8_URL,
            "sha256": package.B8_SHA256,
        },
    ]


def test_attempt04_inventory_reads_payload_metadata_and_stats_only(tmp_path: Path) -> None:
    private = {}
    for name in (
        "timing-fixtures.json",
        "champion_a5_freespace_20260621.pth",
        "best_apex_fs.pth",
        "best_apex_stage1_fs.pth",
        "best_apex_pre_fs.pth",
        "frp3_m3_s12_u60000_20261005.pth",
    ):
        path = tmp_path / name
        path.write_bytes(b"metadata-only-stat")
        private[name] = path
    manifest = {
        "inputs": [
            {
                "dest": "b8.pth",
                "url": package.B8_URL,
                "sha256": package.B8_SHA256,
                "size_bytes": 6_562_005,
            },
            *[
                {"dest": name, "path": str(path), "sha256": "d" * 64}
                for name, path in private.items()
            ],
        ]
    }
    rows = package._source_inputs(manifest)
    assert {row["dest"] for row in rows} == {row["dest"] for row in manifest["inputs"]}
    assert all(row["size_bytes"] == len(b"metadata-only-stat") for row in rows if "path" in row)
    private["best_apex_fs.pth"].unlink()
    try:
        package._source_inputs(manifest)
    except ValueError as exc:
        assert "absent or unsafe" in str(exc)
    else:  # pragma: no cover - assertion preserves the refusal boundary
        raise AssertionError("missing input must refuse")


def test_control_and_empty_ledger_bind_the_timing_recovery_identity() -> None:
    sources = {"research/bl_residual/f5_fit.py": "1" * 64}
    inputs = [
        {"dest": "b8.pth", "sha256": package.B8_SHA256, "size_bytes": 1},
        {"dest": "timing-fixtures.json", "sha256": "2" * 64, "size_bytes": 2},
    ]
    input_sha = "3" * 64
    control = package._control(
        sources,
        inputs,
        input_sha,
        "4" * 64,
        {"typed": "capture"},
        "5" * 64,
    )
    bound = package.identity(
        namespace=package.NAMESPACE,
        allocation=package.ALLOCATION_SHA256,
        source_sha256=control["identity"]["source_sha256"],
        environment_sha256=package._environment_sha256(),
        input_sha256=input_sha,
    )
    assert package.empty_ledger(bound)["units"] == []
    assert control["timing_input_sha256"] == input_sha
    assert control["qualification_fixture_sha256"] == "4" * 64
    assert control["ledger_sha256"] == "5" * 64
    assert "timing-fixtures.json" not in control["qualification_checkpoint_pool"]
