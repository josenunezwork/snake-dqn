"""Focused v2 entrypoint and manifest contracts; no teacher rollout is run here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.t34_exit_b8 import make_manifest as manifests

DESTINATIONS = (
    "B8_s3_final.pth",
    "checkpoints/champion_a5_freespace_20260621.pth",
    "checkpoints/best_apex_fs.pth",
    "checkpoints/best_apex_stage1_fs.pth",
    "checkpoints/best_apex_pre_fs.pth",
    "checkpoints/frp3_m3_s12_u60000_20261005.pth",
)


@pytest.fixture(autouse=True)
def qualified_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manifests, "require_qualified_transport", lambda *_args: {})


def _catalogs(tmp_path: Path) -> tuple[Path, Path]:
    inputs = []
    for index, dest in enumerate(DESTINATIONS):
        source = tmp_path / f"input-{index}.pth"
        source.write_bytes(bytes([index]))
        inputs.append(
            {
                "name": source.name,
                "local_path": str(source),
                "dest": dest,
                "size_bytes": 1,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"inputs": inputs}))
    public = tmp_path / "public.json"
    public.write_text(
        json.dumps(
            {
                "inputs": [
                    {key: row[key] for key in ("name", "dest", "size_bytes", "sha256")}
                    for row in inputs
                ]
            }
        )
    )
    return inventory, public


def test_v2_l4_manifest_has_complete_sparse_source_closure(tmp_path: Path) -> None:
    inventory, public = _catalogs(tmp_path)
    manifest = manifests.make_manifest(
        kind="benchmark",
        kernel="v2",
        commit="a" * 40,
        output_dir=tmp_path / "fresh",
        inventory_path=inventory,
        public_catalog_path=public,
    )
    assert manifest["requirements"]["allowed_gpus"] == ["L4"]
    assert manifest["env"]["T34_KERNEL"] == "v2"
    assert manifest["env"]["T34_WORKERS"] == "2"
    assert {"research/t34_shadow", "research/t33_batched_teacher", "web/backend", "configs"} <= set(
        manifest["source_paths"]
    )
    assert all("/Users/" not in value for value in manifest["source_paths"])


def test_v2_cpu_probe_requires_observed_rate_and_records_hardware_shape(tmp_path: Path) -> None:
    inventory, public = _catalogs(tmp_path)
    kwargs = {
        "kind": "benchmark",
        "kernel": "v2",
        "cpu_runtime_probe": True,
        "commit": "a" * 40,
        "output_dir": tmp_path / "fresh",
        "inventory_path": inventory,
        "public_catalog_path": public,
    }
    with pytest.raises(manifests.ManifestRefused, match="observed positive provider rate"):
        manifests.make_manifest(**kwargs)
    manifest = manifests.make_manifest(**kwargs, cpu_rate_units_per_hour=0.25)
    assert manifest["requirements"] == {
        "gpu": "cpu",
        "allowed_gpus": ["cpu"],
        "min_vcpu": 2,
        "est_minutes": 15,
    }
    assert manifest["env"]["T34_RUNTIME_VARIANT"] == "cpu"
    assert manifest["env"]["T34_THREADS"] == "2"
    assert manifest["budget"]["rate_units_per_hour"] == 0.25


def test_v2_shells_select_kernel_and_mac_full_round_guard() -> None:
    bench = Path("research/t34_exit_b8/bench.sh").read_text()
    labels = Path("research/t34_exit_b8/label.sh").read_text()
    assert 'kernel="${T34_KERNEL:-v1}"' in bench
    assert "--cpu-runtime-probe" in bench
    assert "--mac-full-round" in bench
    assert 'kernel="${T34_KERNEL:-v1}"' in labels
    assert "--mac-full-round" in labels
