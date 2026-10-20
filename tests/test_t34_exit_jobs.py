"""Transport and entrypoint contracts for the unallocated T34 fit job."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.t34_exit_b8_fit import entrypoint, make_manifest, public_inputs, runner


def test_public_catalog_uses_release_archive_and_has_no_mac_paths() -> None:
    catalog = json.loads(Path(public_inputs.CATALOG_PATH).read_text())
    encoded = json.dumps(catalog)
    assert "/Users/" not in encoded
    assert [row["name"] for row in catalog["inputs"]] == ["B8_s3_final.pth", "league.tar.gz"]
    assert catalog["inputs"][1]["members"][0]["name"] == "champion_a5_freespace_20260621.pth"


def test_public_input_refuses_bad_member_before_download(tmp_path: Path) -> None:
    catalog = json.loads(Path(public_inputs.CATALOG_PATH).read_text())
    catalog["inputs"][1]["members"][0]["dest"] = "../escape.pth"
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    with pytest.raises(public_inputs.PublicInputError, match="unsafe public destination"):
        public_inputs.load_catalog(path)


def test_manifest_is_l4_only_and_remains_blocked(tmp_path: Path) -> None:
    manifest = make_manifest.make_manifest(commit="a" * 40, output_dir=tmp_path / "out")
    assert manifest["requirements"]["allowed_gpus"] == ["L4"]
    assert manifest["env"]["T34_FIT_DEVICE"] == "cuda"
    assert manifest["transport"]["dispatch_ready"] is False
    with pytest.raises(make_manifest.ManifestRefused, match="engine export"):
        make_manifest.validate_manifest(manifest)


def test_fake_preflight_calls_shared_adapter_for_real_update_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_runner(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"completed": True, "tripwire": False, "probes_ok": True}

    monkeypatch.setattr(runner, "run", fake_runner)
    report = entrypoint.run(
        exports=tmp_path / "unused",
        checkpoint=None,
        out_dir=tmp_path / "out",
        worlds=2,
        updates=200,
        device="cpu",
        fake_engine=True,
    )
    value = json.loads(report.read_text())
    assert captured["updates"] == 200
    assert captured["worlds"] == 2
    assert captured["fake_engine"] is True
    assert value["status"] == "LOCAL_FAKE_ENGINE_SMOKE_ONLY"
    assert (report.parent / "fit-preflight.json.complete").read_text() == "complete\n"
    assert (
        hashlib.sha256(report.read_bytes()).hexdigest()
        in (report.parent / "fit-preflight.json.sha256").read_text()
    )


def test_real_preflight_requires_complete_engine_export(tmp_path: Path) -> None:
    checkpoint = tmp_path / "B8_s3_final.pth"
    checkpoint.write_bytes(b"pinned")
    with pytest.raises(entrypoint.EntrypointRefused, match="engine export requires"):
        entrypoint.run(
            exports=tmp_path / "no-export",
            checkpoint=checkpoint,
            out_dir=tmp_path / "out",
            worlds=2,
            updates=200,
            device="cpu",
        )


def test_fit_mode_passes_epoch_fit_not_the_200_update_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    payload = exports / "payload.json"
    payload.write_text("{}")
    (exports / "payload.json.sha256").write_text(hashlib.sha256(payload.read_bytes()).hexdigest())
    (exports / "payload.json.complete").write_text("complete\n")
    checkpoint = tmp_path / "B8_s3_final.pth"
    checkpoint.write_bytes(b"pinned")
    captured: dict[str, object] = {}

    def fake_runner(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"completed": True, "tripwire": False, "probes_ok": True, "updates": 321}

    monkeypatch.setattr(runner, "run", fake_runner)
    binding = tmp_path / "binding.json"
    binding.write_text(
        json.dumps(
            {
                "approved": True,
                "source_factory": "x:y",
                "model_factory": "x:y",
                "probe_factory": "x:y",
                "preregistration_sha256": "a" * 64,
                "inputs": [
                    {
                        "name": "engine",
                        "dest": "engine-export/payload.json",
                        "size_bytes": 2,
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                        "url": "https://example.test/payload.json",
                        "complete_marker": True,
                    }
                ],
            }
        )
    )
    report = entrypoint.run(
        exports=exports,
        checkpoint=checkpoint,
        out_dir=tmp_path / "out",
        worlds=800,
        updates=200,
        device="cuda",
        mode="fit",
        epochs=5,
        binding_path=binding,
    )
    assert captured["updates"] == 200
    assert captured["epochs"] == 5
    assert json.loads(report.read_text())["updates"] == 321


def test_fit_shell_forces_cuda_for_colab_runtime() -> None:
    source = Path(entrypoint.__file__).with_name("fit.sh").read_text()
    assert '[[ "${T34_FIT_DEVICE:-cuda}" == "cuda" ]]' in source
    assert '--mode "${T34_FIT_MODE:-preflight}"' in source


def test_repeated_public_download_preserves_preexisting_verified_input(tmp_path):
    row = {
        "name": "engine",
        "dest": "payload.json",
        "size_bytes": 2,
        "sha256": hashlib.sha256(b"{}").hexdigest(),
        "url": "https://example.test/file",
    }
    target = tmp_path / "payload.json"
    target.write_bytes(b"{}")

    def must_not_contact_network(*args, **kwargs):
        raise AssertionError("existing destination must be refused before network")

    with pytest.raises(FileExistsError):
        public_inputs._download(row, tmp_path, must_not_contact_network, 1.0)
    assert target.read_bytes() == b"{}"


def test_existing_s1_archive_dot_prefix_is_supported(tmp_path):
    import io
    import tarfile

    archive = tmp_path / "league.tar.gz"
    payload = b"checkpoint"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("./model.pth")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    row = {
        "members": [
            {
                "name": "model.pth",
                "dest": "checkpoints/model.pth",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ]
    }
    paths = public_inputs._extract_members(row, archive, tmp_path / "inputs")
    assert paths[0].read_bytes() == payload


def test_full_fit_manifest_refuses_bad_epoch_before_any_runtime(tmp_path):
    binding = {
        "approved": True,
        "source_factory": "x:source",
        "probe_factory": "x:probe",
        "preregistration_sha256": "a" * 64,
        "inputs": [
            {
                "name": "engine",
                "dest": "engine-export/payload.json",
                "size_bytes": 2,
                "sha256": "a" * 64,
                "url": "https://example.test/payload",
                "complete_marker": True,
            }
        ],
        "worlds": 800,
        "epochs": 2,
        "job_id": "t34-fit-test-only",
        "fit_resources": {
            "est_minutes": 120,
            "timeout_minutes": 240,
            "unit_cap": 8.0,
            "rate_units_per_hour": 2.0,
            "reserve_units": 0.4,
        },
    }
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(binding))
    with pytest.raises(make_manifest.ManifestRefused, match="epochs"):
        make_manifest.make_manifest(
            commit="a" * 40, output_dir=tmp_path / "out", engine_binding_path=path, mode="fit"
        )
    binding["epochs"] = 5
    path.write_text(json.dumps(binding))
    manifest = make_manifest.make_manifest(
        commit="a" * 40, output_dir=tmp_path / "out", engine_binding_path=path, mode="fit"
    )
    assert manifest["env"]["T34_FIT_MODE"] == "fit"
    assert manifest["env"]["T34_FIT_WORLDS"] == "800"
    assert manifest["env"]["T34_FIT_EPOCHS"] == "5"
