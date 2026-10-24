"""Light synthetic real-entrypoint proof; no factory data or gameplay."""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import torch

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_round30.job import preflight_inputs, run
from research.bl_round30.protocol import B8_SHA, JUDGMENT_SHA, NAMESPACE, require_series
from research.bl_round30.resume import build

B8 = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth")


def synthetic_inputs(root: Path) -> Path:
    root.mkdir()
    shutil.copyfile(B8, root / "b8.pth")
    control = {
        "schema": "r30-control/v1",
        "synthetic": True,
        "namespace": NAMESPACE + "-synthetic",
        "commit": "a" * 40,
        "ruling_sha256": JUDGMENT_SHA,
        "b8_sha256": B8_SHA,
    }
    export_json(root / "control.json", control)
    export_json(
        root / "verified-ledger.json",
        {
            "schema": "r30-job-ledger/v1",
            "namespace": control["namespace"],
            "allocation": digest(root / "control.json"),
            "units": [],
        },
    )
    return root


def test_real_entrypoint_three_tiny_units_and_exact_skip(tmp_path: Path) -> None:
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 15
    source = synthetic_inputs(tmp_path / "inputs")
    timings = []
    for index in range(3):
        started = time.monotonic()
        receipt = run(source, tmp_path / f"out-{index}", mode="smoke", device="cpu")
        timings.append(time.monotonic() - started)
        assert receipt["done"] == 1 and not receipt["heldout_outcomes_opened"]
        verify_export(tmp_path / f"out-{index}" / "unit-0000.pth")
    resume = tmp_path / "resume"
    resume.mkdir()
    ledger = build(source, [tmp_path / "out-0"], resume / "verified-ledger.json")
    assert len(ledger["units"]) == 1
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    for suffix in ("", ".sha256", ".complete"):
        shutil.copyfile(
            tmp_path / ("out-0/allocation.json" + suffix),
            interrupted / ("allocation.json" + suffix),
        )
    (interrupted / "unit-0000.pth").write_bytes(b"partial immutable attempt evidence")
    assert build(source, [interrupted], tmp_path / "empty/ledger.json")["units"] == []
    for name in ("control.json", "b8.pth"):
        shutil.copyfile(source / name, resume / name)
    receipt = run(resume, tmp_path / "skipped", mode="smoke", device="cpu")
    assert receipt["skipped"] == [0]
    assert digest(tmp_path / "out-0/unit-0000.pth") == digest(tmp_path / "skipped/unit-0000.pth")
    with pytest.raises(ValueError, match="duplicate"):
        build(source, [tmp_path / "out-0", tmp_path / "out-1"], tmp_path / "dupe/ledger.json")
    unit = torch.load(tmp_path / "out-0/unit-0000.pth", weights_only=False)
    assert unit["synthetic"] and unit["endpoints"]
    # Consumer really reloads the fused checkpoint and uses rowwise Mac actions.
    from research.bl_residual.job import synthetic_table
    from research.bl_round30.analyze import rows_from_table, rowwise_q
    from research.bl_round30.metrics import served_selection
    from src.model.ego2s_network import load_ego2s_checkpoint

    table = synthetic_table(load_ego2s_checkpoint(str(B8)))
    q = rowwise_q(unit["endpoints"]["adam"], table)
    rows = rows_from_table(table)
    assert len(
        [served_selection({**row, "q": values.tolist()}) for row, values in zip(rows, q)]
    ) == len(rows)
    print("R30 tiny real entrypoint seconds:", timings, flush=True)
    environment = {
        **os.environ,
        "COLAB_CLI_ROOT": str(tmp_path / "bash"),
        "COLAB_INPUT_DIR": str(source),
        "R30_DEVICE": "cpu",
        "R30_MODE": "smoke",
    }
    subprocess.run(
        ["/bin/bash", "research/bl_round30/colab.sh"], env=environment, check=True, timeout=25
    )
    verify_export(tmp_path / "bash/output/receipt.json")


def test_stale_pin_and_partial_export_refuse(tmp_path: Path) -> None:
    source = synthetic_inputs(tmp_path / "inputs")
    with pytest.raises(ValueError, match="foreign"):
        preflight_inputs(source, mode="r30-1", commit="b" * 40)
    export_json(tmp_path / "partial.json", {"synthetic": True})
    Path(str(tmp_path / "partial.json") + ".complete").unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_export(tmp_path / "partial.json")


def test_series_first_local_or_combined_cap_and_no_science_retry() -> None:
    value = {
        "ruling_sha256": JUDGMENT_SHA,
        "attempt": 0,
        "stage_hours": 0.2,
        "stage_units": 0.4,
        "combined_hours": 0.2,
        "combined_units": 0.4,
        "unencumbered_reservation_verified": True,
    }
    assert require_series(value, "r30-1")["remaining_units"] == pytest.approx(4.4)
    with pytest.raises(ValueError, match="cap"):
        require_series({**value, "combined_units": 14.4}, "r30-1")
    with pytest.raises(ValueError, match="completion"):
        require_series({**value, "attempt": 1}, "r30-1")
