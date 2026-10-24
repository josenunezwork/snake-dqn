"""Adversarial offline tests for the I0S Mac reducer; synthetic outcome-free trees."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests import i0s_synthetic_tree as tree
from tools import i0s_attempt as i0s
from tools import i0s_reduce as reduce

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (i0s.RULING.is_file() and i0s.ROUND36.is_file() and i0s.BUDGET_RULING.is_file()),
    reason="I0S ruling or round-36 judgment absent on this machine",
)

PASS = (0.40, [2600.0] * 3)


def validators(native: bool = True) -> dict:
    from research.gpu_sim_spike import integration_analyze, integration_units

    return {
        "valid_cell": integration_analyze.valid_cell,
        "native_valid": lambda report, commit: native and report.get("commit") == commit,
        "units": integration_units,
        "source_hashes": lambda: tree.SYNTHETIC_SOURCES,
    }


def run(
    tmp_path: Path,
    cells: dict,
    *,
    settle: tuple[int, ...] = (1,),
    second: dict | None = None,
    native: bool = True,
    attempt_one: str = "tree",
    **tree_kwargs: object,
) -> dict:
    path, packet = tmp_path / "r.jsonl", tmp_path / "packet"
    tree.make_register(path)
    tree.admit(path, 1, packet)
    output = i0s.attempt_dir(1, packet) / "main-output"
    if attempt_one == "mechanical":
        tree.receipt(output, failure_class="RPC_EXEC_POLL_TIMEOUT")
    else:
        tree.timeout_tree(output, cells, **tree_kwargs)
    if 1 in settle:
        tree.settle(path, 1)
    attempts = {1: i0s.attempt_outputs(1, packet)}
    if second is not None:
        i0s.attempt_dir(2, packet).mkdir(parents=True)
        if attempt_one == "mechanical":
            tree.release_old_i0(path)
            tree.admit(path, 2, packet)
        else:  # protocol breach: register it bypassing the predicate
            from tools import small_bench_register as register

            rows = register.read(path)
            event = register.reserve(rows, tree.package(2), i0s.reservation_id(2), tree.NOW)
            register.append(path, event)
        tree.timeout_tree(i0s.attempt_dir(2, packet) / "main-output", second, job_id=i0s.job_id(2))
        if 2 in settle:
            tree.settle(path, 2)
        attempts[2] = i0s.attempt_outputs(2, packet)
    from tools import small_bench_register as register

    return reduce.final(attempts, register.read(path), tmp_path / "work", validators(native=native))


def test_rc2_timeout_tree_rebuilds_ledger_and_is_incomplete(tmp_path: Path) -> None:
    result = run(tmp_path, {"cell-frozen-e1024": PASS, "cell-mixed-e1024": PASS})
    assert result["verdict"] == "INCOMPLETE"
    assert result["gating_verified"] == 2
    assert result["rebuilt_ledger"]["units"] == 2
    assert result["cells"]["cell-scripted-e1024"]["status"] == "MISSING"
    for unit in reduce.DESCRIPTIVE:
        assert result["cells"][unit]["status"] == "DESCRIPTIVE_NOT_RUN"
    exported = tmp_path / "packet/attempt-01/main-output/exports/verified-ledger.json"
    assert not exported.exists()


def test_failing_verified_cell_stops_before_incomplete(tmp_path: Path) -> None:
    result = run(tmp_path, {"cell-frozen-e1024": PASS, "cell-mixed-e1024": (0.60, [2600.0] * 3)})
    assert result["verdict"] == "STOP_SPEED"
    assert result["label"] == "STOP_SPEED (gating verified 2/3)"
    assert result["gating_failing"] == ["cell-mixed-e1024"]
    gate = result["cells"]["cell-mixed-e1024"]
    assert gate["median_bar_pass"] is False and gate["blocks_bar_pass"] is True


def test_single_slow_continuous_block_stops(tmp_path: Path) -> None:
    result = run(tmp_path, {"cell-frozen-e1024": (0.40, [2600.0, 1999.0, 2600.0])})
    assert result["verdict"] == "STOP_SPEED"
    assert result["cells"]["cell-frozen-e1024"]["blocks_bar_pass"] is False


def test_bar_boundaries_are_inclusive(tmp_path: Path) -> None:
    row = tree.cell_row("cell-frozen-e1024", 0.512, [2000.0] * 3)
    gate = reduce.gate(row)
    assert gate["median_bar_pass"] and gate["pass"]
    assert not reduce.gate(tree.cell_row("cell-frozen-e1024", 0.5121, [2000.0] * 3))["pass"]


def test_all_gating_pass_is_pass_even_without_descriptive(tmp_path: Path) -> None:
    cells = {unit: PASS for unit in reduce.GATING}
    cells["cell-frozen-e256"] = (0.2, None)
    result = run(tmp_path, cells, complete_ledger=True)
    assert result["verdict"] == "PASS_SPEED"
    assert result["cells"]["cell-frozen-e256"]["status"] == "VERIFIED_DESCRIPTIVE"
    assert result["cells"]["cell-mixed-e256"]["status"] == "DESCRIPTIVE_NOT_RUN"


def test_slow_descriptive_cell_never_decides(tmp_path: Path) -> None:
    cells = {unit: PASS for unit in reduce.GATING}
    cells["cell-frozen-e6"] = (9.0, None)
    assert run(tmp_path, cells)["verdict"] == "PASS_SPEED"


def test_invalid_outranks_stop(tmp_path: Path) -> None:
    result = run(
        tmp_path,
        {"cell-frozen-e1024": (0.9, [100.0] * 3)},
        failure="COMBINED_PARITY_MISMATCH",
    )
    assert result["verdict"] == "INVALID"


def test_tampered_unit_payload_is_invalid(tmp_path: Path) -> None:
    path, packet = tmp_path / "r.jsonl", tmp_path / "packet"
    tree.make_register(path)
    tree.admit(path, 1, packet)
    output = i0s.attempt_dir(1, packet) / "main-output"
    tree.timeout_tree(output, {"cell-frozen-e1024": PASS})
    payload = output / "exports/integration-units/cell-frozen-e1024.json"
    payload.write_text(payload.read_text().replace("0.4", "0.3", 1))
    tree.settle(path, 1)
    from tools import small_bench_register as register

    result = reduce.final({1: [output]}, register.read(path), tmp_path / "work", validators())
    assert result["verdict"] == "INVALID"


def test_foreign_namespace_unit_is_invalid(tmp_path: Path) -> None:
    foreign = f"i0-integration-{'0' * 40}-A100-main"
    result = run(tmp_path, {"cell-frozen-e1024": PASS}, namespace=foreign)
    assert result["verdict"] == "INVALID"


def test_malformed_published_cell_is_invalid(tmp_path: Path, monkeypatch) -> None:
    original = tree.cell_row

    def broken(unit: str, median: float, blocks=None) -> dict:
        row = original(unit, median, blocks)
        row["capacity"] = 1
        return row

    monkeypatch.setattr(tree, "cell_row", broken)
    assert run(tmp_path, {"cell-frozen-e1024": PASS})["verdict"] == "INVALID"


def test_missing_native_proof_cannot_stop_or_pass(tmp_path: Path) -> None:
    cells = {"cell-frozen-e1024": (0.9, [100.0] * 3), "cell-mixed-e1024": PASS}
    result = run(tmp_path, cells, native=False)
    assert result["verdict"] == "INCOMPLETE"
    assert result["cells"]["cell-frozen-e1024"]["status"] == "UNVERIFIED_PROOF_INCOMPLETE"


def test_rates_blinded_until_settled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="blinded"):
        run(tmp_path, {"cell-frozen-e1024": PASS}, settle=())


def test_integrity_never_opens_payloads(tmp_path: Path) -> None:
    output = tmp_path / "main-output"
    tree.timeout_tree(output, {"cell-frozen-e1024": PASS})
    payload = output / "exports/integration-units/cell-frozen-e1024.json"
    payload.write_text("{not json")  # consistent sidecars + receipt, unparseable payload
    Path(f"{payload}.sha256").write_text(f"{reduce.sha(payload)}  {payload.name}\n")
    tree.receipt(output)
    value = reduce.integrity([output])
    assert value["status"] == "INTEGRITY_PASS", value["errors"]
    assert value["outputs"][0]["unit_files"] == 1
    assert value["rates_computed"] is False and value["unit_payloads_opened"] is False


def test_integrity_flags_receipt_mismatch_and_bad_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "main-output"
    tree.timeout_tree(output, {"cell-frozen-e1024": PASS})
    extra = output / "exports/integration-units/cell-mixed-e1024.json"
    extra.write_text("{}")
    value = reduce.integrity([output])
    assert value["status"] == "INTEGRITY_FAIL"
    side = output / "exports/native-parity.json.sha256"
    side.write_text("0" * 64 + "  native-parity.json\n")
    assert any("sidecar" in error for error in reduce.integrity([output])["errors"])


def test_mechanical_first_attempt_is_never_scored(tmp_path: Path) -> None:
    result = run(
        tmp_path,
        {},
        attempt_one="mechanical",
        second={"cell-frozen-e1024": PASS},
        settle=(1, 2),
    )
    assert result["scored_attempt"] == 2
    assert result["attempts"][1]["scored"] is False
    assert result["verdict"] == "INCOMPLETE"


def test_retry_after_scored_attempt_is_not_scored(tmp_path: Path) -> None:
    result = run(
        tmp_path,
        {"cell-frozen-e1024": (0.9, [100.0] * 3)},
        second={unit: PASS for unit in reduce.GATING},
        settle=(1, 2),
    )
    assert result["scored_attempt"] == 1
    assert result["verdict"] == "STOP_SPEED"
    assert "outside the ruling predicate" in result["attempts"][2]["reason"]


def test_only_mechanical_attempt_is_incomplete(tmp_path: Path) -> None:
    result = run(tmp_path, {}, attempt_one="mechanical")
    assert result["scored_attempt"] is None
    assert result["verdict"] == "INCOMPLETE"


def test_third_attempt_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        reduce.select_scored({1: [], 2: [], 3: []})


def settled_packet(tmp_path: Path, cells: dict) -> tuple[Path, Path]:
    path, packet = tmp_path / "r.jsonl", tmp_path / "packet"
    tree.make_register(path)
    tree.admit(path, 1, packet)
    tree.timeout_tree(i0s.attempt_dir(1, packet) / "main-output", cells)
    tree.settle(path, 1)
    return path, packet


def test_final_runs_once_with_marker(tmp_path: Path) -> None:
    path, packet = settled_packet(tmp_path, {"cell-frozen-e1024": PASS})
    result = reduce.run_final(packet, path, validators())
    assert result["verdict"] == "INCOMPLETE"
    assert (packet / "REDUCTION.started").is_file() and (packet / "REDUCTION.json").is_file()
    with pytest.raises(ValueError, match="runs once"):
        reduce.run_final(packet, path, validators())


def test_crashed_final_cannot_be_repeated(tmp_path: Path) -> None:
    path, packet = settled_packet(tmp_path, {"cell-frozen-e1024": PASS})
    (packet / "REDUCTION.started").write_text("{}\n")  # a run that died mid-reduction
    with pytest.raises(ValueError, match="runs once"):
        reduce.run_final(packet, path, validators())
    assert not (packet / "REDUCTION-work").exists()


def test_blinded_refusal_leaves_no_marker(tmp_path: Path) -> None:
    path, packet = tmp_path / "r.jsonl", tmp_path / "packet"
    tree.make_register(path)
    tree.admit(path, 1, packet)
    tree.timeout_tree(i0s.attempt_dir(1, packet) / "main-output", {"cell-frozen-e1024": PASS})
    with pytest.raises(ValueError, match="blinded"):
        reduce.run_final(packet, path, validators())
    assert not (packet / "REDUCTION.started").exists()


@pytest.mark.parametrize("marker", ["REDUCTION.json", "REDUCTION.started", None])
def test_final_cli_refuses_relocation_and_repeat(tmp_path: Path, marker: str | None) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    output = packet / "REDUCTION.json"
    if marker:
        (packet / marker).write_text("{}")
    else:
        output = tmp_path / "elsewhere" / "R1.json"
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    run_ = subprocess.run(
        [
            str(ROOT / "venv/bin/python"),
            "-m",
            "tools.i0s_reduce",
            "final",
            "--packet",
            str(packet),
            "--register",
            str(tmp_path / "absent.jsonl"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    text = run_.stdout + run_.stderr
    assert run_.returncode != 0
    assert ("runs once" in text) if marker else ("only <packet>/REDUCTION.json" in text)
    assert not (tmp_path / "elsewhere").exists()


def test_reviewer_probe_native_invalid_attempt_is_not_mechanical_and_final_invalid(
    tmp_path: Path,
) -> None:
    from tools import small_bench_register as register

    path, packet = tmp_path / "r.jsonl", tmp_path / "packet"
    tree.make_register(path)
    tree.admit(path, 1, packet)
    tree.native_invalid_mechanical(i0s.attempt_dir(1, packet) / "main-output")
    tree.settle(path, 1)
    tree.release_old_i0(path)
    verdict = i0s.retry_predicate(register.read(path), packet, now=tree.NOW)
    assert not verdict["eligible"]
    assert verdict["attempt_1"]["mechanical_zero_unit"] is False
    i0s.attempt_dir(2, packet).mkdir(parents=True)
    with pytest.raises(i0s.Hold, match="retry predicate"):
        tree.admit(path, 2, packet)
    # Even if a retry were forced through, the attempt-1 INVALID decides the verdict.
    rows = register.read(path)
    register.append(path, register.reserve(rows, tree.package(2), i0s.reservation_id(2), tree.NOW))
    tree.timeout_tree(
        i0s.attempt_dir(2, packet) / "main-output",
        {unit: PASS for unit in reduce.GATING},
        job_id=i0s.job_id(2),
    )
    tree.settle(path, 2)
    result = reduce.run_final(packet, path, validators())
    assert result["verdict"] == "INVALID"
    assert any("native parity INVALID" in reason for reason in result["invalid_reasons"])


def test_valid_native_report_alone_makes_attempt_non_mechanical(tmp_path: Path) -> None:
    output = tmp_path / "main-output"
    tree.timeout_tree(output, {}, failure_class="RPC_EXEC_POLL_TIMEOUT")
    value = i0s.classify_attempt([output])
    assert value["mechanical_zero_unit"] is False and value["unit_exports"] == 0


def test_invalid_in_allocation_retry_output_is_seen(tmp_path: Path) -> None:
    path, packet = tmp_path / "r.jsonl", tmp_path / "packet"
    tree.make_register(path)
    tree.admit(path, 1, packet)
    base = i0s.attempt_dir(1, packet) / "main-output"
    tree.native_invalid_mechanical(base)
    tree.timeout_tree(Path(f"{base}-retry-1"), {unit: PASS for unit in reduce.GATING})
    tree.settle(path, 1)
    result = reduce.run_final(packet, path, validators())
    assert result["verdict"] == "INVALID"


def test_scored_attempt_reduces_the_output_that_produced(tmp_path: Path) -> None:
    path, packet = tmp_path / "r.jsonl", tmp_path / "packet"
    tree.make_register(path)
    tree.admit(path, 1, packet)
    base = i0s.attempt_dir(1, packet) / "main-output"
    tree.timeout_tree(base, {unit: PASS for unit in reduce.GATING})
    tree.receipt(Path(f"{base}-retry-1"), failure_class="ACCOUNTING")  # no exports
    tree.settle(path, 1)
    result = reduce.run_final(packet, path, validators())
    assert result["verdict"] == "PASS_SPEED"


def test_foreign_failure_marker_is_invalid(tmp_path: Path) -> None:
    from research.gpu_sim_spike import integration_units

    path, packet = tmp_path / "r.jsonl", tmp_path / "packet"
    tree.make_register(path)
    tree.admit(path, 1, packet)
    output = i0s.attempt_dir(1, packet) / "main-output"
    tree.timeout_tree(output, {unit: PASS for unit in reduce.GATING})
    integration_units.publish_json(
        output / "exports/integration-failure.json",
        {"status": "INCOMPLETE", "commit": "0" * 40, "source_sha256": {}},
    )
    tree.receipt(output)
    tree.settle(path, 1)
    result = reduce.run_final(packet, path, validators())
    assert result["verdict"] == "INVALID"
    assert any("foreign" in reason for reason in result["invalid_reasons"])


@pytest.mark.skipif(not i0s.RUN.is_dir(), reason="qualified I0 checkout absent")
def test_frozen_validators_reduce_rc2_tree_in_qualified_checkout(tmp_path: Path) -> None:
    env = dict(
        os.environ,
        PYTHONPATH=f"{i0s.RUN}:{ROOT}",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        SNAKE_DQN_DEVICE="cpu",
    )
    run_ = subprocess.run(
        [
            str(ROOT / "venv/bin/python"),
            str(ROOT / "tests/i0s_synthetic_tree.py"),
            str(tmp_path),
        ],
        cwd=i0s.RUN,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    result = json.loads(run_.stdout.strip().splitlines()[-1])
    assert result["verdict"] == "INCOMPLETE"
    assert result["gating_verified"] == 2
    assert result["rebuilt_ledger"]["units"] == 2
    assert result["invalid_reasons"] == [] and result["incomplete_reasons"] == []
