"""Focused v2 entrypoint and manifest contracts; no teacher rollout is run here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.t34_exit_b8 import benchmark
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


def test_v2_label_manifest_binds_measured_export_projection(tmp_path: Path) -> None:
    inventory, public = _catalogs(tmp_path)
    kwargs = {
        "kind": "label",
        "kernel": "v2",
        "commit": "a" * 40,
        "output_dir": tmp_path / "fresh",
        "inventory_path": inventory,
        "public_catalog_path": public,
        "round_id": "r1",
        "worlds": 100,
    }
    with pytest.raises(manifests.ManifestRefused, match="measured_export_bytes_per_world"):
        manifests.make_manifest(**kwargs)
    manifest = manifests.make_manifest(**kwargs, measured_export_bytes_per_world=1000)
    sizing = manifest["export_sizing"]
    assert manifest["max_export_bytes"] == 2 * 1024**3
    assert sizing["projected_bytes"] == 16 * 1024**2 + 125_000
    assert sizing["provisional_until_full_a5"] is True


@pytest.mark.parametrize("value", (None, 0, -1, True, float("nan")))
def test_v2_label_manifest_refuses_invalid_measured_export_size(
    tmp_path: Path, value: object
) -> None:
    inventory, public = _catalogs(tmp_path)
    with pytest.raises(manifests.ManifestRefused, match="measured_export_bytes_per_world"):
        manifests.make_manifest(
            kind="label",
            kernel="v2",
            commit="a" * 40,
            output_dir=tmp_path / "fresh",
            inventory_path=inventory,
            public_catalog_path=public,
            round_id="r1",
            worlds=2,
            measured_export_bytes_per_world=value,  # type: ignore[arg-type]
        )


def test_v2_label_manifest_refuses_export_projection_over_cap(tmp_path: Path) -> None:
    inventory, public = _catalogs(tmp_path)
    with pytest.raises(manifests.ManifestRefused, match="reduce chunk worlds"):
        manifests.make_manifest(
            kind="label",
            kernel="v2",
            commit="a" * 40,
            output_dir=tmp_path / "fresh",
            inventory_path=inventory,
            public_catalog_path=public,
            round_id="r1",
            worlds=2,
            measured_export_bytes_per_world=1024**3,
        )


def test_v2_shells_select_kernel_and_mac_full_round_guard() -> None:
    bench = Path("research/t34_exit_b8/bench.sh").read_text()
    labels = Path("research/t34_exit_b8/label.sh").read_text()
    assert 'kernel="${T34_KERNEL:-v1}"' in bench
    assert "--cpu-runtime-probe" in bench
    assert "--mac-full-round" in bench
    assert 'kernel="${T34_KERNEL:-v1}"' in labels
    assert "--mac-full-round" in labels


def _v2_report(namespace: str, **overrides: object) -> dict[str, object]:
    return {
        "schema": "t34-a5-benchmark/v2",
        "kernel": "v2",
        "complete": True,
        "preflight": False,
        "mac_full_round": False,
        "cpu_runtime_probe": False,
        "namespace": namespace,
        "recipe": {
            "workers": 1,
            "threads_per_worker": 5,
            "root_worlds_per_worker": 256,
            "horizon": 300,
            "device": "cuda",
        },
        "elapsed_seconds": 2.0,
        "active_row_steps": 7000,
        "resident_row_steps": 7000,
    } | overrides


def test_v2_a5_aggregate_requires_distinct_l4_worker_identities(tmp_path: Path) -> None:
    first, second = tmp_path / "lane0.json", tmp_path / "lane1.json"
    first.write_text(json.dumps(_v2_report("t34-exit-a5/v2/s0/a1/lane0")))
    second.write_text(json.dumps(_v2_report("t34-exit-a5/v2/s0/a1/lane1")))
    aggregate = benchmark.aggregate_lane_reports([first, second])
    assert aggregate["schema"] == "t34-a5-lane-aggregate/v2"
    assert aggregate["go_rule"]["met"] is True

    second.write_text(json.dumps(_v2_report("t34-exit-a5/v2/s0/a1/lane0")))
    with pytest.raises(ValueError, match="distinct worker"):
        benchmark.aggregate_lane_reports([first, second])


@pytest.mark.parametrize(
    "overrides",
    (
        {"mac_full_round": True},
        {
            "recipe": {
                "workers": 1,
                "threads_per_worker": 2,
                "root_worlds_per_worker": 2,
                "horizon": 300,
                "device": "cpu",
            }
        },
    ),
)
def test_v2_a5_aggregate_rejects_non_l4_worker_shape(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    first, second = tmp_path / "lane0.json", tmp_path / "lane1.json"
    first.write_text(json.dumps(_v2_report("one", **overrides)))
    second.write_text(json.dumps(_v2_report("two")))
    with pytest.raises(ValueError, match="CUDA L4 A5"):
        benchmark.aggregate_lane_reports([first, second])


def test_v2_zero_completion_keeps_per_world_work_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")

    def fake_runner(_kernel: str):
        return lambda _config: {
            "complete": False,
            "mixes": {"scripted": {"searched": 0, "expanded": 0}},
            "teacher_profile": {"active_row_steps": 0, "resident_row_steps": 0},
            "worlds": 0,
            "worlds_per_hour": 0.0,
            "decisions": 0,
            "decisions_per_second": 0.0,
            "export_bytes": 0,
            "export_bytes_per_world": 0.0,
            "configured_root_capacity": 2,
            "peak_root_worlds": 0,
            "search_batch_peak": 0,
            "invalid_reason": "shadow_control_mismatch",
            "unfinished_worlds": 2,
            "world_rate_scope": "complete_worlds_only",
            "root_inference_bucket_peak": 2,
            "branch_bucket_capacity": 32,
        }

    monkeypatch.setattr(benchmark, "_cpu_ceiling", lambda _threads: None)
    monkeypatch.setattr(benchmark, "_kernel_runner", fake_runner)
    monkeypatch.setattr(benchmark, "_load_policy", lambda _config: None)
    report = benchmark.run_benchmark(
        {
            "kernel": "v2",
            "preflight": True,
            "checkpoint": checkpoint,
            "output_dir": tmp_path / "worlds",
            "worlds": 2,
            "max_worlds": 2,
            "frames": 1,
            "seconds": 1,
            "threads": 1,
            "device": "cpu",
        }
    )
    assert report["executed_row_steps_per_world"] is None
    assert report["invalid_reason"] == "shadow_control_mismatch"
    assert report["unfinished_worlds"] == 2
