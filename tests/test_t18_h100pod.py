"""Focused, hermetic coverage for the T18 H100 pod preparation contract."""

from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from research.t16_colab import bundle
from research.t18_h100pod_20261017 import prepare


def _bundle_with_pinned_digest(monkeypatch, tmp_path):
    archive = tmp_path / "t16_colab_inputs.tar.gz"
    payload = tmp_path / "weights.pth"
    payload.write_bytes(b"weights")
    bundle.export_bundle(archive, {"init/B8_s3_final.pth": payload})
    monkeypatch.setattr(prepare, "BUNDLE_SHA256", bundle.sha256_file(archive))
    return archive


def _nvidia_smi(argv, **kwargs):
    assert argv[:1] == ["nvidia-smi"]
    return subprocess.CompletedProcess(argv, 0, "NVIDIA H100 80GB HBM3\n", "")


def test_prepare_verifies_extracts_and_writes_host_receipt(tmp_path, monkeypatch):
    archive = _bundle_with_pinned_digest(monkeypatch, tmp_path)
    monkeypatch.setattr(prepare.shutil, "which", lambda name: "/usr/bin/gcc")
    monkeypatch.setattr(prepare, "_vcpu_count", lambda: 16)
    monkeypatch.setattr(prepare, "_cpu_model", lambda: "AMD EPYC test")
    monkeypatch.setattr(prepare, "_shared_memory_available", lambda: prepare.MIN_SHM_BYTES)
    monkeypatch.setattr(
        prepare.run_smoke,
        "environment_info",
        lambda: {"gpu": "NVIDIA H100 80GB HBM3", "versions": {"torch": "2.9.1+cu126"}},
    )
    monkeypatch.setattr(
        prepare,
        "p0_ring_bytes",
        lambda: {
            "ring_rows_per_actor": 40_000,
            "actors": 2,
            "raw_two_actor_bytes": 995_920_016,
            "packed_two_actor_bytes": 264_800_016,
        },
    )

    receipt = prepare.prepare(archive, tmp_path / "in", tmp_path / "out", run=_nvidia_smi)

    saved = json.loads((tmp_path / "out/prepare_receipt.json").read_text())
    assert saved == receipt
    assert receipt["host"] == {
        "gpu_models": ["NVIDIA H100 80GB HBM3"],
        "vcpu_count": 16,
        "vcpu_model": "AMD EPYC test",
        "shared_memory_path": "/dev/shm",
        "shared_memory_available_bytes": prepare.MIN_SHM_BYTES,
    }
    assert receipt["bundle"]["sha256"] == bundle.sha256_file(archive)
    assert receipt["runtime"]["versions"]["torch"] == "2.9.1+cu126"
    assert receipt["p0_rings"]["raw_two_actor_bytes"] == 995_920_016
    assert (tmp_path / "in/t16-colab-inputs/init/B8_s3_final.pth").read_bytes() == b"weights"


def test_prepare_refuses_an_outer_digest_mismatch_before_extract(tmp_path, monkeypatch):
    archive = _bundle_with_pinned_digest(monkeypatch, tmp_path)
    monkeypatch.setattr(prepare, "BUNDLE_SHA256", hashlib.sha256(b"different").hexdigest())
    monkeypatch.setattr(prepare.shutil, "which", lambda name: "/usr/bin/gcc")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        prepare.prepare(archive, tmp_path / "in", tmp_path / "out", run=_nvidia_smi)

    assert not (tmp_path / "in/t16-colab-inputs").exists()


def test_prepare_requires_gcc_before_extracting(tmp_path, monkeypatch):
    archive = _bundle_with_pinned_digest(monkeypatch, tmp_path)
    monkeypatch.setattr(prepare.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="gcc is required"):
        prepare.prepare(archive, tmp_path / "in", tmp_path / "out", run=_nvidia_smi)

    assert not (tmp_path / "in/t16-colab-inputs").exists()


def test_prepare_requires_shared_memory_capacity_before_extract(tmp_path, monkeypatch):
    archive = _bundle_with_pinned_digest(monkeypatch, tmp_path)
    monkeypatch.setattr(prepare.shutil, "which", lambda name: "/usr/bin/gcc")
    monkeypatch.setattr(prepare, "_shared_memory_available", lambda: prepare.MIN_SHM_BYTES - 1)

    with pytest.raises(RuntimeError, match="requires at least"):
        prepare.prepare(archive, tmp_path / "in", tmp_path / "out", run=_nvidia_smi)

    assert not (tmp_path / "in/t16-colab-inputs").exists()


def test_prepare_refuses_a_runtime_that_fails_the_t16_cuda_contract(tmp_path, monkeypatch):
    archive = _bundle_with_pinned_digest(monkeypatch, tmp_path)
    monkeypatch.setattr(prepare.shutil, "which", lambda name: "/usr/bin/gcc")
    monkeypatch.setattr(prepare, "_shared_memory_available", lambda: prepare.MIN_SHM_BYTES)
    monkeypatch.setattr(
        prepare.run_smoke,
        "environment_info",
        lambda: (_ for _ in ()).throw(RuntimeError("CUDA runtime required")),
    )

    with pytest.raises(RuntimeError, match="CUDA runtime required"):
        prepare.prepare(archive, tmp_path / "in", tmp_path / "out", run=_nvidia_smi)

    assert not (tmp_path / "in/t16-colab-inputs").exists()


def test_commands_cover_the_four_p0_cells_then_smoke03():
    commands = json.loads(
        (
            prepare.Path(__file__).parents[1] / "research/t18_h100pod_20261017/commands.json"
        ).read_text()
    )
    assert commands["schema"] == "perf-podbench-commands/v1"
    rows = commands["commands"]
    assert [row["name"] for row in rows] == [
        "p0_raw_g0",
        "p0_raw_g3",
        "p0_pack_g0",
        "p0_pack_g3",
        "smoke03",
    ]
    for row in rows[:4]:
        argv = row["argv"]
        assert argv[:2] == ["python", "research/perf_learner_20261013/bench_learner.py"]
        assert argv[argv.index("--devices") + 1] == "cuda"
        assert argv[argv.index("--configs") + 1] == "t1_k2_c_ro,t1_bf16_k2_c_ro,t1_k4_c_ro"
        assert argv[argv.index("--ring-dir") + 1] == "/dev/shm"
        assert argv[argv.index("--init") + 1] == "{inputs}/init/B8_s3_final.pth"
    assert "--matrix" in rows[-1]["argv"] and "smoke03" in rows[-1]["argv"]


def test_n3_commands_are_isolated_from_the_existing_t18_workload():
    commands = json.loads(
        (
            prepare.Path(__file__).parents[1] / "research/t18_h100pod_20261017/commands_n3.json"
        ).read_text()
    )
    assert commands["schema"] == "perf-podbench-commands/v1"
    assert len(commands["commands"]) == 1
    row = commands["commands"][0]
    assert row["name"] == "n3_static_pool" and row["timeout_s"] == 1800
    assert row["argv"][row["argv"].index("--matrix") + 1] == "n3"


def test_p0_ring_bytes_match_the_committed_benchmark_defaults():
    rings = prepare.p0_ring_bytes()
    assert rings == {
        "ring_rows_per_actor": 40_000,
        "actors": 2,
        "raw_two_actor_bytes": 995_920_016,
        "packed_two_actor_bytes": 264_800_016,
    }
