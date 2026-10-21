"""Targeted tests for E3 source allocation and immutable supervisor exports."""

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from research.bl_info_value.catalog import check_path, choose_roots
from research.bl_info_value.io import export_bytes, verify_export, write_json, write_npz
from research.bl_info_value.manifests import forecast
from research.bl_info_value.run import merge
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
