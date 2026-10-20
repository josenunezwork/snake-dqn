"""Focused transport-only checks for T33's prepared L4 benchmark package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.t33_batched_teacher import make_manifest as manifests
from research.t33_batched_teacher import public_inputs


def _row(tmp_path: Path, *, name: str, dest: str) -> dict[str, object]:
    path = tmp_path / name
    path.write_bytes(name.encode())
    return {
        "name": name,
        "local_path": str(path),
        "dest": dest,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "public_url": None,
        "publication_required": True,
    }


def _inventory(tmp_path: Path) -> Path:
    rows = [
        _row(tmp_path, name="B8_s3_final.pth", dest="B8_s3_final.pth"),
        _row(
            tmp_path,
            name="champion_a5_freespace_20260621.pth",
            dest="checkpoints/champion_a5_freespace_20260621.pth",
        ),
        _row(tmp_path, name="best_apex_fs.pth", dest="checkpoints/best_apex_fs.pth"),
        _row(tmp_path, name="best_apex_stage1_fs.pth", dest="checkpoints/best_apex_stage1_fs.pth"),
        _row(tmp_path, name="best_apex_pre_fs.pth", dest="checkpoints/best_apex_pre_fs.pth"),
        _row(
            tmp_path,
            name="frp3_m3_s12_u60000_20261005.pth",
            dest="checkpoints/frp3_m3_s12_u60000_20261005.pth",
        ),
    ]
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({"inputs": rows}))
    return path


def _public_catalog(tmp_path: Path, inventory: Path) -> Path:
    rows = json.loads(inventory.read_text())["inputs"]
    catalog = [
        {key: row[key] for key in ("name", "dest", "size_bytes", "sha256", "public_url")}
        for row in rows
    ]
    path = tmp_path / "public-catalog.json"
    path.write_text(json.dumps({"inputs": catalog}))
    return path


def test_public_manifest_is_l4_only_and_blocked_with_catalog_only(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    manifest = manifests.make_manifest(
        commit="a" * 40,
        output_dir=tmp_path / "fresh-output",
        inventory_path=inventory,
        public_catalog_path=_public_catalog(tmp_path, inventory),
    )

    assert manifest["claim"] == "DRAFT_NO_ALLOCATION_AUTHORIZED"
    assert manifest["requirements"] == {
        "gpu": "L4",
        "allowed_gpus": ["L4"],
        "min_vcpu": 12,
        "est_minutes": 30,
    }
    assert manifest["timeout_minutes"] == 60
    assert manifest["budget"] == {"unit_cap": 2.0, "rate_units_per_hour": 2.0, "reserve_units": 0.4}
    assert manifest["transport"]["status"] == "PUBLICATION_BLOCKED"
    assert manifest["transport"]["dispatch_ready"] is False
    assert manifest["env"]["T33_INPUT_MODE"] == "public"
    assert [item["dest"] for item in manifest["inputs"]] == ["public-inputs.json"]
    with pytest.raises(manifests.ManifestRefused, match="publication is pending"):
        manifests.validate_manifest(manifest)


def test_local_upload_is_explicit_preflight_only(tmp_path: Path) -> None:
    manifest = manifests.make_manifest(
        commit="a" * 40,
        output_dir=tmp_path / "fresh-output",
        inventory_path=_inventory(tmp_path),
        local_upload_preflight=True,
    )

    assert manifest["claim"] == "LOCAL_UPLOAD_PREFLIGHT_ONLY"
    assert manifest["transport"]["dispatch_ready"] is False
    assert manifest["env"]["T33_INPUT_MODE"] == "local-preflight"
    assert len(manifest["inputs"]) == 6


def test_manifest_refuses_input_drift(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    first = json.loads(inventory.read_text())["inputs"][0]
    Path(str(first["local_path"])).write_bytes(b"changed")

    with pytest.raises(manifests.ManifestRefused, match="input size drift|input hash drift"):
        manifests.make_manifest(
            commit="a" * 40,
            output_dir=tmp_path / "fresh-output",
            inventory_path=inventory,
            local_upload_preflight=True,
        )


def test_local_supervisor_validation_is_used_without_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_validate(manifest: dict[str, object], repo: Path) -> dict[str, object]:
        captured.update(manifest=manifest, repo=repo)
        return {"validated": True}

    monkeypatch.setattr("research.colab_cli.supervisor.validate", fake_validate)
    result = manifests.validate_manifest({"job_id": "t33"}, repo=Path("/repo"))

    assert result == {"validated": True}
    assert captured == {"manifest": {"job_id": "t33"}, "repo": Path("/repo")}


def test_missing_public_url_refuses_before_any_downloader_call(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(Path(public_inputs.__file__).with_name("PUBLIC_INPUTS.json").read_text())
    called = False

    def opener(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("network opener must not run")

    with pytest.raises(public_inputs.PublicInputError, match="publication is pending"):
        public_inputs.materialize(catalog, tmp_path / "download", opener=opener)
    assert called is False


def test_public_downloader_uses_mocked_response_and_checks_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "B8_s3_final.pth"
    content = b"x" * 3
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(public_inputs, "EXPECTED", {name: (name, len(content), digest)})
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "name": key,
                        "dest": dest,
                        "size_bytes": size,
                        "sha256": sha,
                        "public_url": "https://example.invalid/" + key,
                    }
                    for key, (dest, size, sha) in public_inputs.EXPECTED.items()
                ]
            }
        )
    )

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            value, self.value = getattr(self, "value", content), b""
            return value

    paths = public_inputs.materialize(
        catalog, tmp_path / "download", opener=lambda *_args, **_kwargs: Response()
    )

    assert paths == [tmp_path / "download" / name]
    assert paths[0].read_bytes() == content
