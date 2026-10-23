"""Tests for amendment-06's no-rebuild public HTTPS delivery adapter."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from research.bl_ppo import p1_origin_import
from research.bl_ppo.p1_repair_transport import (
    RELEASE_BASE_URL,
    public_input_preflight,
    public_rows,
)
from research.colab_cli import runtime
from research.colab_cli.canary import SCHEMA, family, input_inventory
from research.colab_cli.inputs import validate_public_input

pytest_plugins = ["research.bl_ppo.test_p1_origin_import"]


def test_public_rows_are_qualified_https_inventory(tmp_path) -> None:
    payload = b"immutable-origin-bytes"
    origin = tmp_path / "origin-inputs"
    origin.mkdir()
    source = origin / "origin.pt"
    source.write_bytes(payload)
    checksum = hashlib.sha256(payload).hexdigest()
    manifest = {
        "inputs": [
            {"path": str(source), "sha256": checksum, "dest": "origin.pt"},
            {"path": str(source), "sha256": checksum, "dest": "retained-copy.pt"},
        ]
    }

    converted = public_rows(
        manifest,
        [{"name": "origin.pt", "size_bytes": len(payload), "sha256": checksum}],
        RELEASE_BASE_URL,
    )

    row = converted["inputs"][0]
    assert row == {
        "url": f"{RELEASE_BASE_URL}/origin.pt",
        "sha256": checksum,
        "size_bytes": len(payload),
        "dest": "origin.pt",
    }
    assert validate_public_input(row) == {
        "url": row["url"],
        "sha256": checksum,
        "size_bytes": len(payload),
    }
    assert input_inventory(converted) == [
        {"dest": "origin.pt", "size": len(payload), "transport": "https", "sha256": checksum},
        {
            "dest": "retained-copy.pt",
            "size": len(payload),
            "transport": "upload",
            "sha256": checksum,
        },
    ]
    assert manifest["inputs"][0]["path"] == str(source)


def test_supervisor_validates_https_rows_and_real_p1_canary_check(tmp_path) -> None:
    """The qualified supervisor normalizes HTTPS rows and invokes the P1 canary gate."""
    payload = b"immutable-origin-bytes"
    checksum = hashlib.sha256(payload).hexdigest()
    row = {
        "url": f"{RELEASE_BASE_URL}/origin.pt",
        "sha256": checksum,
        "size_bytes": len(payload),
        "dest": "origin.pt",
    }
    canary = {
        "script": "research/bl_ppo/p1_colab.sh",
        "commit": "b491ef181df803f2a3e3aea4a81809e9ee054eb4",
        "job_id": "p1-transport-fixture",
        "output_dir": str(tmp_path / "fresh-output"),
        "timeout_minutes": 30,
        "device": "cuda:0",
        "lease_seconds": 180,
        "max_export_bytes": 1024,
        "required_exports": ["receipt.json"],
        "inputs": [row],
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 1, "est_minutes": 15},
        "budget": {
            "unit_cap": 2.0,
            "rate_units_per_hour": 1.54,
            "reserve_units": 0.4,
            "other_assignments": {},
        },
        "env": {
            "BL_P1_MODE": "y3-source",
            "BL_PPO_DEVICE": "cuda:0",
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": "research/bl_ppo",
        },
    }
    canary["canary"] = {"schema": SCHEMA, "mode": "probe", "family": family(canary)}
    from research.colab_cli.supervisor import validate

    normalized = validate(canary, Path(__file__).parents[2])
    assert normalized["inputs"] == [row]


def test_public_rows_refuse_unpinned_release_or_changed_origin_bytes(tmp_path) -> None:
    source_dir = tmp_path / "origin-inputs"
    source_dir.mkdir()
    source = source_dir / "origin.pt"
    source.write_bytes(b"changed")
    expected = hashlib.sha256(b"expected").hexdigest()
    manifest = {"inputs": [{"path": str(source), "sha256": expected, "dest": "origin.pt"}]}
    assets = [{"name": "origin.pt", "size_bytes": len(b"expected"), "sha256": expected}]

    with pytest.raises(ValueError, match="pinned source bytes"):
        public_rows(manifest, assets, RELEASE_BASE_URL)
    with pytest.raises(ValueError, match="immutable repository and tag"):
        public_rows(manifest, assets, "https://example.invalid/release")


def test_public_rows_allows_only_the_pinned_control_bundle_outside_origin_inputs(tmp_path) -> None:
    bundle = tmp_path / "p1-inputs.tar.gz"
    bundle.write_bytes(b"bundle")
    checksum = hashlib.sha256(bundle.read_bytes()).hexdigest()
    asset = {"name": bundle.name, "size_bytes": bundle.stat().st_size, "sha256": checksum}
    manifest = {"inputs": [{"path": str(bundle), "sha256": checksum, "dest": bundle.name}]}

    assert public_rows(manifest, [asset], RELEASE_BASE_URL)["inputs"][0]["url"].endswith(
        "/p1-inputs.tar.gz"
    )
    other = tmp_path / "not-an-origin-payload.bin"
    other.write_bytes(bundle.read_bytes())
    with pytest.raises(ValueError, match="pinned source bytes"):
        public_rows(
            {"inputs": [{"path": str(other), "sha256": checksum, "dest": other.name}]},
            [{**asset, "name": other.name}],
            RELEASE_BASE_URL,
        )


def test_public_input_preflight_writes_create_only_no_allocation_receipt(
    tmp_path, monkeypatch
) -> None:
    payload = b"https-public"
    checksum = hashlib.sha256(payload).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "url": f"{RELEASE_BASE_URL}/origin.pt",
                        "sha256": checksum,
                        "size_bytes": len(payload),
                        "dest": "origin.pt",
                    }
                ]
            }
        )
    )
    observed = {}

    def preflight(value):
        observed["manifest"] = value
        return [
            {
                "url": value["inputs"][0]["url"],
                "sha256": checksum,
                "size_bytes": len(payload),
                "attempt": 1,
                "elapsed_seconds": 0.5,
                "bytes_per_second": 24.0,
            }
        ]

    from research.colab_cli import supervisor

    monkeypatch.setattr(supervisor, "input_preflight", preflight)
    output = tmp_path / "public-input-preflight.json"
    receipt = public_input_preflight(manifest_path=manifest, output=output)

    assert observed["manifest"]["inputs"][0]["url"].startswith(RELEASE_BASE_URL)
    assert receipt["status"] == "PASS"
    assert receipt["total_bytes"] == len(payload)
    assert receipt["provider_allocations"] == receipt["provider_calls"] == 0
    assert (
        json.loads(output.read_text())["manifest"]["sha256"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    with pytest.raises(FileExistsError, match="must be fresh"):
        public_input_preflight(manifest_path=manifest, output=output)


class _Reply:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *unused) -> None:
        return None

    def read(self, size: int) -> bytes:
        value = self.payload[self.offset : self.offset + size]
        self.offset += len(value)
        return value


def test_origin_loader_refuses_changed_https_delivered_bytes(
    tmp_path, monkeypatch, tiny_p1
) -> None:
    """The import contract remains a second identity check after HTTPS fetch."""
    from research.bl_ppo.test_p1_origin_import import _make_import_fixture

    root, control_path, block, origin, _ = _make_import_fixture(tmp_path, monkeypatch, tiny_p1)
    runtime_root = tmp_path / "runtime-root"
    shutil.copytree(root, runtime_root / "inputs")
    changed = b"a transport-valid but origin-invalid snapshot"
    row = {
        "url": "https://github.com/josenunezwork/snake-dqn/releases/download/p1-y3-origin-20261023/"
        + origin.name,
        "sha256": hashlib.sha256(changed).hexdigest(),
        "size_bytes": len(changed),
        "dest": origin.name,
    }
    monkeypatch.setattr(runtime, "urlopen", lambda *args, **kwargs: _Reply(changed))

    assert runtime._fetch_public_inputs(runtime_root, [row]) == [
        {"dest": origin.name, "sha256": row["sha256"], "size_bytes": len(changed)}
    ]
    fetched_root = runtime_root / "inputs"
    assert (fetched_root / origin.name).read_bytes() == changed
    with pytest.raises(
        ValueError, match="export digest/name mismatch|origin snapshot digest mismatch"
    ):
        p1_origin_import.validate(
            json.loads((fetched_root / control_path.name).read_text()), fetched_root
        )
    assert block["origin_snapshot"]["sha256"] != row["sha256"]
