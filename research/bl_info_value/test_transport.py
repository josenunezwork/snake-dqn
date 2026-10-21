"""Targeted tests for E3 source allocation and immutable supervisor exports."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from research.bl_info_value.catalog import check_path, choose_roots
from research.bl_info_value.io import export_bytes, verify_export, write_json, write_npz
from research.bl_info_value.manifests import forecast
from research.bl_info_value.resources import available_cpus, worker_count
from research.bl_info_value.run import execute, merge
from research.bl_info_value.upload_readiness import require_upload_readiness
from research.colab_cli.runtime import _exports


def test_every_export_has_supervisor_sidecars(tmp_path):
    output = tmp_path / "output"
    write_json(output / "receipt.json", {"status": "SMOKE_COMPLETE"})
    write_npz(output / "world.npz", {"x": np.arange(6)})
    export_bytes(output / "student.pth", b"checkpoint fixture")
    records = _exports(tmp_path, 1024**2)
    assert {r["path"] for r in records} == {"receipt.json", "world.npz", "student.pth"}
    for row in records:
        payload = output / row["path"]
        assert payload.with_name(payload.name + ".complete").is_file()
        assert payload.with_name(payload.name + ".sha256").read_text().strip() == row["sha256"]
        verify_export(payload)
    with pytest.raises(FileExistsError):
        export_bytes(output / "student.pth", b"overwrite")


def test_missing_or_changed_sidecar_refuses(tmp_path):
    payload = tmp_path / "output" / "receipt.json"
    write_json(payload, {"ok": True})
    payload.with_name(payload.name + ".sha256").unlink()
    with pytest.raises(RuntimeError, match="marker"):
        _exports(tmp_path, 1024)


def test_world_disjoint_balanced_selection_and_fill():
    rows = []
    for world in range(256):
        for decisive in (False, True):
            for root in range(5):
                rows.append(
                    dict(
                        source_key=f"world-{world}",
                        decision=root + 10 * int(decisive),
                        decisive=decisive,
                        frame=root + 1,
                    )
                )
    selected = choose_roots(rows)
    assert [sum(r["split"] == split for r in selected) for split in range(3)] == [192, 32, 32]
    assert len({r["key"] for r in selected}) == 256
    assert sum(len(r["roots"]) for r in selected) == 1536
    assert sum(row["decisive"] for world in selected for row in world["roots"]) == 768
    assert selected == choose_roots(list(reversed(rows)))


def test_duplicate_and_insufficient_roots_refuse():
    with pytest.raises(ValueError, match="insufficient"):
        choose_roots([])
    row = dict(source_key="same", decision=1, decisive=True)
    with pytest.raises(ValueError, match="duplicate"):
        choose_roots([row, row])


def test_sealed_path_rejected_without_open():
    with pytest.raises(ValueError, match="sealed"):
        check_path(__import__("pathlib").Path("/tmp/t34-round0-study-01/never-open"))


def test_merge_order_is_input_order():
    result = merge([({"x": np.array([2])}, {}), ({"x": np.array([1])}, {})])
    assert result["x"].tolist() == [2, 1]
    assert hashlib.sha256(result["x"].tobytes()).hexdigest()


def test_real_shell_uses_fresh_child_of_supervisor_output(tmp_path):
    (tmp_path / "output").mkdir()
    fake = tmp_path / "bin"
    fake.mkdir()
    python = fake / "python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(Path(__file__).with_name("job.sh"))],
        env={
            **os.environ,
            "PATH": str(fake) + os.pathsep + os.environ["PATH"],
            "COLAB_CLI_ROOT": str(tmp_path),
            "COLAB_INPUT_DIR": str(tmp_path / "inputs"),
            "E3_INPUT_SHA256": "a" * 64,
        },
        check=True,
        capture_output=True,
        text=True,
    )
    arguments = result.stdout.splitlines()
    assert arguments[arguments.index("--out") + 1] == str(tmp_path / "output" / "e3")
    assert not (tmp_path / "output" / "e3").exists()
    assert arguments[arguments.index("--device") + 1] == "cuda"
    assert arguments[arguments.index("--expected-workers") + 1] == "6"


@pytest.mark.parametrize("cpus,requested,expected", [(6, 6, 4), (8, 6, 6), (12, 12, 10), (4, 1, 1)])
def test_worker_headroom(cpus, requested, expected):
    assert worker_count(requested, cpus) == expected


@pytest.mark.parametrize("cpus,requested", [(2, 1), (1, 1), (8, 0), (8, 13)])
def test_worker_headroom_refuses_invalid_capacity(cpus, requested):
    with pytest.raises(ValueError, match="usable CPUs"):
        worker_count(requested, cpus)


def test_cpu_capacity_respects_affinity(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {2, 3, 4, 5}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    assert available_cpus() == 4


def test_priced_worker_count_refuses_affinity_shortfall(monkeypatch, tmp_path):
    """Host os.cpu_count may pass hardware admission while affinity is narrower."""
    monkeypatch.setattr("research.bl_info_value.run.available_cpus", lambda: 6)
    args = argparse.Namespace(
        work=tmp_path / "work", out=tmp_path / "output", workers=6, expected_workers=6
    )
    with pytest.raises(ValueError, match="priced worker count"):
        execute(args)
    assert not args.out.exists()
    assert not args.work.exists()


def test_fast_worlds_batch_progress_and_keep_scratch_out_of_exports(monkeypatch, tmp_path, capsys):
    """A completed backlog produces bounded control traffic and intact final arrays."""
    tasks = [dict(checkpoint="ckpt", source_npz="rows", plan=dict(specs=[])) for _ in range(8)]
    monkeypatch.setattr("research.bl_info_value.run.threads", lambda: None)
    monkeypatch.setattr("research.bl_info_value.run.lower_priority", lambda: 10)
    monkeypatch.setattr("research.bl_info_value.run.available_cpus", lambda: 8)
    monkeypatch.setattr("research.bl_info_value.run.time.perf_counter", lambda: 1.0)
    monkeypatch.setattr(
        "research.bl_info_value.run.unpack",
        lambda *args: dict(schema="bl-e3-input/v1", smoke=True, tasks=tasks),
    )
    for task in tasks:
        task["indices"] = [0]
    monkeypatch.setattr(
        "research.bl_info_value.run._reconstruct",
        lambda task: ({"action": np.array([2])}, {"elapsed_seconds": 1.0}),
    )
    monkeypatch.setattr("research.bl_info_value.fit.fit_students", lambda *args, **kwargs: {})
    output = tmp_path / "output"
    args = argparse.Namespace(
        work=tmp_path / "work",
        out=output / "e3",
        workers=1,
        expected_workers=None,
        archive=tmp_path / "unused",
        archive_sha256="a" * 64,
        smoke=True,
        epochs=2,
        device="cpu",
        check_parallel=False,
    )
    execute(args)
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["done"] for row in progress] == [1, 8]
    assert progress[-1]["eta_seconds"] == 0
    records = _exports(tmp_path, 1024**2)
    assert not any("worlds/" in row["path"] for row in records)
    assert len(list((args.work / "worlds").glob("*.npz"))) == 8
    with np.load(args.out / "dataset.npz") as dataset:
        assert dataset["action"].tolist() == [2] * 8
    assert len(json.loads((args.out / "timing.json").read_text())["world_metrics"]) == 8


def test_priority_inheritance_in_spawned_process():
    """Direct Python invocation also lowers both parent and spawn children."""
    result = subprocess.run(
        [
            __import__("sys").executable,
            "-c",
            "from concurrent.futures import ProcessPoolExecutor; "
            "import multiprocessing, os; "
            "from research.bl_info_value.resources import lower_priority; "
            "lower_priority(); "
            "pool=ProcessPoolExecutor(1, mp_context=multiprocessing.get_context('spawn')); "
            "print(os.nice(0), pool.submit(os.nice, 0).result()); pool.shutdown()",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    parent, child = map(int, result.stdout.split())
    assert parent >= 10 and child >= 10


def test_forecast_prices_real_batch_cost_and_reserve():
    timing = dict(
        worlds=3,
        roots=6,
        world_metrics=[dict(elapsed_seconds=3)] * 3,
        fits=dict(
            benchmark={
                arm: dict(batch32_update_seconds=0.01, batch192_inference_seconds=0.02)
                for arm in ("A", "B", "C")
            }
        ),
    )
    projected = forecast(timing)
    assert projected["usable_hours"] == 6.4
    assert projected["cap_units"] == 16
    assert projected["reserve_units"] == 3.2
    assert projected["fit_hours_cpu_upper_projection"] == pytest.approx(
        (0.01 * 2160 + 0.02 * 61) * 3 / 3600
    )


def test_e3_upload_preflight_binds_final_manifest_and_source_fallback(tmp_path):
    """The recovery package cannot publish a manifest without its own Z-9 receipt."""
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    inputs = tmp_path / "inputs.tar.gz"
    inputs.write_bytes(b"compact catalog")
    manifest = {
        "job_id": "bl-e3-upload-preflight-test",
        "commit": commit,
        "script": "research/bl_info_value/job.sh",
        "source_paths": [
            "src",
            "configs",
            "pyproject.toml",
            "research/colab_cli",
            "research/bl_info_value",
        ],
        "env": {"COLAB_CLI_REQUIREMENTS": "research/bl_info_value/requirements.txt"},
        "inputs": [
            {
                "path": str(inputs),
                "dest": "inputs.tar.gz",
                "sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
            }
        ],
    }
    receipt = tmp_path / "upload-preflight.json"
    result = require_upload_readiness(manifest, repo=repo, receipt_path=receipt)

    assert result["normal_forecast_seconds"] <= result["normal_budget_seconds"]
    assert result["fallback_total_bound_seconds"] <= result["transport"]["cap_seconds"]
    assert result["source_plan"]["fallback_bytes"] > 0
    assert json.loads(receipt.read_text()) == result
    with pytest.raises(ValueError, match="already exists"):
        require_upload_readiness(manifest, repo=repo, receipt_path=receipt)
