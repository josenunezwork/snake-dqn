"""Metadata-only attempt-04 resolution and fail-closed publication regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.bl_ppo.exports import digest, export_bytes, export_json
from research.bl_residual import f4_d1024_package as package
from research.bl_residual import f4_ra_prepare as ra


def _replace(path: Path, value: dict) -> None:
    """Replace only synthetic temporary exports, including their sidecars."""
    for suffix in ("", ".sha256", ".complete"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    export_json(path, value)


@pytest.fixture()
def binding_world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    repo, pack = tmp_path / "repo", tmp_path / "d1024-pack-04"
    output = pack / "d1024-output-l4-04"
    paths = {
        "manifest": pack / "main.json",
        "supervisor_receipt": output / "receipt.json",
        "producer_receipt": output / "exports/fit/receipt.json",
    }
    export_json(paths["manifest"], {"output_dir": str(output), "commit": "a" * 40})
    for name in ("supervisor_receipt", "producer_receipt"):
        export_json(paths[name], {"synthetic": name})
    phase = repo / package.FIT_PUBLICATION_PATH
    export_json(phase, {"synthetic": "published phase"})
    binding = {
        "schema": "bl-residual-metadata-fit-binding/v1",
        "pack": str(pack),
        "fit_pin": "a" * 40,
        "phase": package.FIT_PUBLICATION_PATH,
        "phase_sha256": digest(phase),
        "files": {
            name: {"path": str(path), "size": path.stat().st_size, "sha256": digest(path)}
            for name, path in paths.items()
        },
    }
    binding_path = repo / package.FIT_BINDING_PATH
    export_json(binding_path, binding)
    published = {
        str(path.relative_to(repo)): path.read_bytes() for path in repo.rglob("*") if path.is_file()
    }

    def origin(argv: list[str], **_kwargs) -> bytes:
        return published[argv[-1].split(":", 1)[1]]

    monkeypatch.setattr(package, "ROOT", repo)
    monkeypatch.setattr(package.subprocess, "check_output", origin)
    return {
        "pack": pack,
        "output": output,
        "paths": paths,
        "binding": binding,
        "binding_path": binding_path,
        "published": published,
        "repo": repo,
    }


def _publish_binding(world: dict) -> None:
    _replace(world["binding_path"], world["binding"])
    for suffix in ("", ".sha256", ".complete"):
        world["published"][package.FIT_BINDING_PATH + suffix] = Path(
            str(world["binding_path"]) + suffix
        ).read_bytes()


def test_attempt_04_uses_published_sealed_manifest_without_attempt_01(binding_world: dict) -> None:
    world = binding_world
    assert package.published_fit_output(world["pack"]) == world["output"]
    assert not (world["pack"] / "d1024-output-l4-01").exists()


@pytest.mark.parametrize("name", ["manifest", "supervisor_receipt", "producer_receipt"])
def test_refuses_changed_bound_file(binding_world: dict, name: str) -> None:
    world = binding_world
    _replace(world["paths"][name], {"different": name})
    with pytest.raises(ValueError, match="published fit binding"):
        package.published_fit_output(world["pack"])


def test_refuses_unsealed_main_manifest(binding_world: dict) -> None:
    world = binding_world
    Path(str(world["paths"]["manifest"]) + ".complete").unlink()
    with pytest.raises(ValueError, match="missing or unsafe"):
        package.published_fit_output(world["pack"])


def test_refuses_locally_resealed_unpublished_binding(binding_world: dict) -> None:
    world = binding_world
    _replace(world["binding_path"], {**world["binding"], "pack": "/foreign/pack"})
    with pytest.raises(ValueError, match="origin/main"):
        package.published_fit_output(world["pack"])


@pytest.mark.parametrize("suffix", [".sha256", ".complete"])
def test_refuses_unpublished_binding_sidecar(binding_world: dict, suffix: str) -> None:
    world = binding_world
    world["published"][package.FIT_BINDING_PATH + suffix] = b"foreign sidecar\n"
    with pytest.raises(ValueError, match="origin/main"):
        package.published_fit_output(world["pack"])


@pytest.mark.parametrize("field", ["schema", "pack", "phase", "phase_sha256", "fit_pin"])
def test_refuses_binding_identity_mismatch(binding_world: dict, field: str) -> None:
    world = binding_world
    world["binding"][field] = "foreign"
    _publish_binding(world)
    with pytest.raises(ValueError, match="published fit binding"):
        package.published_fit_output(world["pack"])


def test_refuses_valid_but_different_fit_pin(binding_world: dict) -> None:
    world = binding_world
    world["binding"]["fit_pin"] = "b" * 40
    _publish_binding(world)
    with pytest.raises(ValueError, match="main output/source"):
        package.published_fit_output(world["pack"])


@pytest.mark.parametrize("field", ["path", "size"])
def test_published_unit_refuses_same_sha_with_different_path_or_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    from research.bl_residual.protocol import build_ledger

    inputs, fit = tmp_path / "inputs", tmp_path / "exports/fit"
    export_json(inputs / "control.json", {"namespace": "test", "allocation": "a", "doses": [1024]})
    export_json(fit / "allocation.json", {"namespace": "test", "allocation": "a"})
    checkpoint = fit / "fit-0001.pth"
    export_bytes(checkpoint, b"synthetic opaque fit; never deserialized")
    phase = tmp_path / package.FIT_PUBLICATION_PATH
    ledger = build_ledger([fit], phase, "test", "a")
    ledger["units"][0][field] = str(tmp_path / "foreign.pth") if field == "path" else 999
    _replace(phase, ledger)
    monkeypatch.setattr(package, "ROOT", tmp_path)
    monkeypatch.setattr(package, "require_publication", lambda *_: None)
    with pytest.raises(ValueError, match="published fit unit"):
        ra._published_unit(package.FIT_PUBLICATION_PATH, inputs, checkpoint, 1024)


@pytest.mark.parametrize("output_kind", ["attempt-01", "relative", "outside", "symlink"])
def test_refuses_manifest_output_disagreeing_with_binding(
    binding_world: dict, output_kind: str
) -> None:
    world = binding_world
    output = {
        "attempt-01": world["pack"] / "d1024-output-l4-01",
        "relative": Path("d1024-output-l4-04"),
        "outside": world["pack"].parent / "foreign-output",
        "symlink": world["pack"] / "alias",
    }[output_kind]
    if output_kind == "symlink":
        output.symlink_to(world["output"], target_is_directory=True)
    manifest = world["paths"]["manifest"]
    value = json.loads(manifest.read_text())
    _replace(manifest, {**value, "output_dir": str(output)})
    world["binding"]["files"]["manifest"] = {
        "path": str(manifest),
        "size": manifest.stat().st_size,
        "sha256": digest(manifest),
    }
    _publish_binding(world)
    with pytest.raises(ValueError, match="published fit binding"):
        package.published_fit_output(world["pack"])
