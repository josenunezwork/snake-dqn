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


@pytest.mark.parametrize(
    "base_url",
    [
        RELEASE_BASE_URL,
        "https://drive.usercontent.google.com/download?id=real_fixture_file_id"
        "&export=download&confirm=t",
    ],
)
def test_supervisor_validates_https_rows_and_real_p1_canary_check(tmp_path, base_url) -> None:
    """The qualified supervisor normalizes HTTPS rows and invokes the P1 canary gate."""
    payload = b"immutable-origin-bytes"
    checksum = hashlib.sha256(payload).hexdigest()
    row = {
        "url": (f"{base_url}/origin.pt" if base_url == RELEASE_BASE_URL else base_url),
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
    from research.bl_upload.qualification import _input_rows
    from research.colab_cli.canary import check

    check(normalized, Path(__file__).parents[2])
    assert input_inventory(normalized)[0]["transport"] == "https"
    assert _input_rows(normalized, Path(__file__).parents[2]) == []


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


@pytest.mark.parametrize("body_size", [64])
def test_drive_http200_html_warning_fails_both_ends_once_without_publication(
    tmp_path, monkeypatch, body_size
) -> None:
    """A small HTTP-200 interstitial cannot be mistaken for a pinned large asset."""
    from research.colab_cli.inputs import InputError, preflight

    warning = b"<!doctype html><title>Google Drive virus scan warning</title>"
    assert len(warning) < body_size
    row = {
        "url": "https://drive.usercontent.google.com/download?id=real_fixture_file_id"
        "&export=download&confirm=t",
        "sha256": hashlib.sha256(b"x" * body_size).hexdigest(),
        "size_bytes": body_size,
        "dest": "origin.pt",
    }
    calls = []

    def opener(request, **kwargs):
        calls.append(request.full_url)
        reply = _Reply(warning)
        reply.status = 200
        return reply

    with pytest.raises(InputError, match="size or SHA-256 mismatch"):
        preflight([row], opener=opener)
    assert calls == [row["url"]]
    assert list(tmp_path.rglob("*")) == []
    calls.clear()
    monkeypatch.setattr(runtime, "urlopen", opener)
    with pytest.raises(RuntimeError, match="size or SHA-256 mismatch"):
        runtime._fetch_public_inputs(tmp_path, [row])
    assert calls == [row["url"]]
    assert not (tmp_path / "inputs" / row["dest"]).exists()
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def _drive_pack_fixture(tmp_path, monkeypatch):
    from research.bl_ppo import p1_repair_transport as transport
    from research.bl_ppo.exports import digest, export_bytes, export_json
    from research.bl_ppo.p1_manifests import bundle_inputs
    from research.bl_ppo.p1_resume import empty_ledger

    source, output = tmp_path / "pack-03", tmp_path / "pack-04"
    source.mkdir()
    inputs = source / "inputs"
    inputs.mkdir()
    rows = []
    for prefix in ("p1", "canary"):
        namespace = f"bl_{prefix}_prior"
        ledger_name = "verified-ledger.json" if prefix == "p1" else "canary-verified-ledger.json"
        export_json(
            inputs / f"{prefix}-control.json",
            {
                "resume": {
                    "namespace": namespace,
                    "allocation": "same_worlds",
                    "skip_verified": True,
                }
            },
        )
        export_json(inputs / ledger_name, empty_ledger(namespace, "same_worlds"))
    for path in sorted(inputs.iterdir()):
        rows.append({"dest": path.name, "path": str(path), "sha256": digest(path)})
    bundle_inputs(source, rows)
    assets = []
    for index in range(9):
        name = "p1-inputs.tar.gz" if index == 0 else f"origin-{index}.pt"
        payload = (source / name).read_bytes() if index == 0 else f"origin-{index}".encode()
        path = source / "release-assets" / name
        export_bytes(path, payload)
        assets.append({"name": name, "sha256": digest(path), "size_bytes": len(payload)})
    export_json(source / "release-plan.json", {"assets": assets})
    for name in transport.DRIVE_MANIFESTS:
        export_json(
            source / name,
            {
                "scientific_admission": False,
                "job_id": "p1-test-oldtag",
                "output_dir": str(source / name.replace(".json", "-out")),
                "resume": {"namespace": "bl_p1_prior", "allocation": "same_worlds"},
                "inputs": [
                    {
                        "dest": a["name"],
                        "sha256": a["sha256"],
                        "size_bytes": a["size_bytes"],
                        "url": f"{RELEASE_BASE_URL}/{a['name']}",
                    }
                    for a in assets
                ],
            },
        )
    monkeypatch.setattr(transport, "drive_forecast", lambda *args: {"upload_forecast_seconds": 1})
    return source, output, assets


def test_drive_pack_new_attempt_and_exact_nine_url_swap(tmp_path, monkeypatch) -> None:
    from research.bl_ppo import p1_repair_transport as transport
    from research.bl_ppo.exports import digest

    source, output, assets = _drive_pack_fixture(tmp_path, monkeypatch)
    before = {
        str(path.relative_to(source)): digest(path) for path in source.rglob("*") if path.is_file()
    }
    prepared = transport.prepare_drive_pack(source=source, output=output, repo=tmp_path)
    assert prepared["bundle"]["regenerated"] is True
    assert prepared["bundle"]["before_sha256"] != prepared["bundle"]["after_sha256"]
    assert len(prepared["bundle_members_changed"]) == 8
    assert not (output / "canary-manifest.json").exists()
    urls = {
        a["name"]: f"https://drive.usercontent.google.com/download?id=file_id_{i:08d}"
        "&export=download&confirm=t"
        for i, a in enumerate(assets)
    }
    mapping = tmp_path / "urls.json"
    mapping.write_text(json.dumps(urls))
    with pytest.raises(ValueError, match="real file ID"):
        transport.build_drive_pack(
            source=source, urls_path=output / "drive-urls.template.json", output=output
        )
    assert not (output / "canary-manifest.json").exists()
    receipt = transport.build_drive_pack(source=source, urls_path=mapping, output=output)
    for name in transport.DRIVE_MANIFESTS:
        template = json.loads((output / "templates" / name).read_text())
        final = json.loads((output / name).read_text())
        changes = transport._manifest_diff(template, final)
        assert len(changes) == 9
        assert all(change["field"].endswith("/url") for change in changes)
        assert final["resume"]["allocation"] == "same_worlds"
        assert final["resume"]["namespace"] != "bl_p1_prior"
    assert receipt["url_changes_per_manifest"] == 9
    assert before == {
        str(path.relative_to(source)): digest(path) for path in source.rglob("*") if path.is_file()
    }
    with pytest.raises(FileExistsError, match="immutable output"):
        transport.build_drive_pack(source=source, urls_path=mapping, output=output)
    for bad_urls in ({}, {**urls, assets[0]["name"]: urls[assets[1]["name"]]}):
        with pytest.raises(ValueError):
            transport.drive_rows(template, bad_urls)
