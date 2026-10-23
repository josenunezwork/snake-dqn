"""Fast synthetic-only R1/R2 tests; no workers, game, or real attempt inputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.bl_k1 import m1, m1_carry
from research.bl_served_search.exports import export_json, verify_export


def tick(finished: int = 100, overrun: bool = True, update: float = 0.001) -> dict:
    return {"finished_ns": finished, "overrun": overrun, "update_seconds": update}


def sample(visible: int = 100, seconds: float = 5.1, **fields: object) -> dict:
    return {
        "visible_ns": visible,
        "S_seconds": seconds,
        "phase": "warm",
        "fixture_id": "frozen-800",
        **fields,
    }


def attempt(
    root: Path,
    *,
    index: int = 2,
    ticks: list[dict] | None = None,
    samples: list[dict] | None = None,
    host: list[dict] | None = None,
    discards: int = 1,
    verdict: str = "PASS",
    cell_verdict: str = "PASS",
    retain: int = 4,
) -> dict:
    """Write only synthetic immutable exports in pytest's temporary directory."""
    cells = [cell for cell in m1.cells() if cell["workers"] > 0]
    cell = cells[index]
    foreign = [{"pid": 1234, "cpu": 75}]
    host = (
        [
            {"monotonic_ns": 100, "foreign_heavy_processes": []},
            {"monotonic_ns": 200, "foreign_heavy_processes": foreign},
        ]
        if host is None
        else host
    )
    records = []
    for n in range(1, discards + 1):
        row = {
            **cell,
            "namespace": "synthetic-only",
            "summary": {"complete": False, "verdict": "INCOMPLETE"},
            "stop_reason": m1.RESTART_POLICY["trigger"],
            "epoch_ns": n,
            "host": host,
            "ticks": ticks if ticks is not None else [tick()],
            "samples": samples if samples is not None else [],
        }
        export_json(root / f"{cell['cell_id']}.discarded-{n}.json", row)
        records.append(
            {
                "cell_id": cell["cell_id"],
                "n": n,
                "stop_reason": row["stop_reason"],
                "foreign_heavy_processes": host[-1]["foreign_heavy_processes"],
                "started_ns": 0,
                "finished_ns": 300,
                "ticks": len(row["ticks"]),
                "samples_count": len(row["samples"]),
            }
        )
    receipt = {
        "schema": m1.SCHEMA,
        "namespace": "synthetic-only",
        "verdict": verdict,
        "cells": [
            {
                "cell_id": c["cell_id"],
                "verdict": cell_verdict if i == index else "PASS",
                "complete": cell_verdict not in ("INCOMPLETE", "INVALID") or i != index,
                "update": {"p95_seconds": 0.001},
                "per_fixture": {"frozen-800": {"max_seconds": 1.0}},
            }
            for i, c in enumerate(cells[:retain])
        ],
        "discarded_runs": records,
        # This path must never be read; receipt is the retained authority.
        "completed_unit_exports": {"sentinel": {"path": "/forbidden/retained.json"}},
    }
    export_json(root / "receipt.json", receipt)
    return receipt


def test_astra_overrun_foreign_stop_clean_pass_is_blocked(tmp_path: Path) -> None:
    original = attempt(tmp_path)
    result = m1_carry.reduce_attempt(tmp_path)
    assert result["verdict"] == "BLOCKED"
    assert result["cells"][2]["verdict"] == "BLOCKED"
    assert result["cells"][2]["carried_reasons"] == [
        "carried prefix violation from discarded run 1"
    ]
    assert result["cells"][2]["update"] == original["cells"][2]["update"]
    assert result["cells"][2]["per_fixture"] == original["cells"][2]["per_fixture"]
    assert json.loads((tmp_path / "receipt.json").read_text()) == original


def test_tail_overrun_is_disclosed_never_carried(tmp_path: Path) -> None:
    attempt(tmp_path, ticks=[tick(101)])
    result = m1_carry.reduce_attempt(tmp_path)
    assert result["verdict"] == "PASS"
    assert result["verdict_label"] == "PASS under the restart protocol, 1 discards"
    disclosure = result["discarded_run_disclosures"][0]
    assert disclosure["total_prefix_overruns"] == 1
    assert disclosure["carried_overruns"] == 0


@pytest.mark.parametrize(
    "index,visible,seconds,fields,blocked",
    [
        (2, 100, 5.1, {}, True),
        (3, 100, 5.1, {}, True),
        (2, 101, 5.1, {}, False),
        (2, 100, 5.0, {}, False),
        (1, 100, 5.1, {}, False),
        (0, 100, 5.1, {}, False),
        (2, 100, 5.1, {"phase": "cold"}, False),
        (2, 100, 5.1, {"fixture_id": "mixed-800"}, False),
        (2, 100, 5.1, {"error": "INVALID"}, False),
    ],
)
def test_exact_doom_conditions(
    tmp_path: Path, index: int, visible: int, seconds: float, fields: dict, blocked: bool
) -> None:
    attempt(
        tmp_path,
        index=index,
        ticks=[tick(overrun=False)],
        samples=[sample(visible, seconds, **fields)],
    )
    result = m1_carry.reduce_attempt(tmp_path)
    assert result["verdict"] == ("BLOCKED" if blocked else "PASS")
    assert result["discarded_run_disclosures"][0]["carried_doom"] == int(blocked)


def test_p95_only_is_reported_not_carried(tmp_path: Path) -> None:
    attempt(tmp_path, ticks=[tick(overrun=False, update=0.02)])
    result = m1_carry.reduce_attempt(tmp_path)
    assert result["verdict"] == "PASS"
    assert result["discarded_run_disclosures"][0]["prefix_update_p95_seconds"] == 0.02


def test_no_clean_host_nothing_carried(tmp_path: Path) -> None:
    attempt(
        tmp_path,
        samples=[sample()],
        host=[{"monotonic_ns": 200, "foreign_heavy_processes": [{"pid": 8}]}],
    )
    result = m1_carry.reduce_attempt(tmp_path)
    disclosure = result["discarded_run_disclosures"][0]
    assert result["verdict"] == "PASS"
    assert disclosure["T_clean_ns"] is None
    assert disclosure["carried_doom"] == disclosure["carried_overruns"] == 0
    assert disclosure["total_prefix_overruns"] == 1
    assert disclosure["prefix_per_fixture_max_S_seconds"] == {"frozen-800": 5.1}


@pytest.mark.parametrize("status", ["BLOCKED", "INCOMPLETE", "INVALID"])
def test_monotone_verdicts_preserve_higher_precedence(tmp_path: Path, status: str) -> None:
    attempt(tmp_path, verdict=status, cell_verdict=status)
    result = m1_carry.reduce_attempt(tmp_path)
    assert result["verdict"] == status
    assert result["cells"][2]["verdict"] == status


@pytest.mark.parametrize("status", ["INCOMPLETE", "INVALID"])
def test_attempt_failure_precedence_survives_cell_carry(tmp_path: Path, status: str) -> None:
    attempt(tmp_path, verdict=status)
    result = m1_carry.reduce_attempt(tmp_path)
    assert result["verdict"] == status
    assert result["cells"][2]["verdict"] == "BLOCKED"


def test_all_discards_and_retained_disclosures(tmp_path: Path) -> None:
    attempt(tmp_path, index=0, discards=3, samples=[sample(), sample(phase="cold", seconds=9)])
    result = m1_carry.reduce_attempt(tmp_path)
    assert len(result["discarded_run_disclosures"]) == 3
    assert result["retained_cell_disclosures"][0]["restarts"] == 3
    assert not result["retained_cell_disclosures"][0]["coordinator_cold"]
    assert all(d["run_duration_seconds"] is None for d in result["retained_cell_disclosures"])
    for disclosure in result["discarded_run_disclosures"]:
        assert disclosure["run_duration_ns"] == 300
        assert disclosure["trigger_processes"] == [{"pid": 1234, "cpu": 75}]
        assert disclosure["ticks"] == 1
        assert disclosure["samples_count"] == 2
        assert disclosure["prefix_per_fixture_max_S_seconds"] == {"frozen-800": 9}
        assert disclosure["prefix_warm_per_fixture_max_S_seconds"] == {"frozen-800": 5.1}
    assert not result["discarded_timings_pooled"]


def test_no_discards_coordinator_cold_and_no_other_reads(tmp_path: Path, monkeypatch) -> None:
    attempt(tmp_path, discards=0)
    read = m1_carry.l1._read_completed_json
    paths = []

    def tracked(path: Path) -> tuple[dict, str]:
        paths.append(path.name)
        return read(path)

    monkeypatch.setattr(m1_carry.l1, "_read_completed_json", tracked)
    result = m1_carry.reduce_attempt(tmp_path)
    assert paths == ["receipt.json"]
    assert result["verdict"] == "PASS"
    assert result["retained_cell_disclosures"][0]["coordinator_cold"]
    assert not result["retained_cell_disclosures"][1]["coordinator_cold"]


def test_only_permitted_exports_read_with_discards(tmp_path: Path, monkeypatch) -> None:
    attempt(tmp_path)
    read = m1_carry.l1._read_completed_json
    paths = []

    def tracked(path: Path) -> tuple[dict, str]:
        assert path.parent == tmp_path
        paths.append(path.name)
        return read(path)

    monkeypatch.setattr(m1_carry.l1, "_read_completed_json", tracked)
    m1_carry.reduce_attempt(tmp_path)
    assert set(paths) == {"receipt.json", "cpu-w4-k4.discarded-1.json"}


def test_clean_cutoff_is_max_and_p95_uses_nearest_rank(tmp_path: Path) -> None:
    attempt(
        tmp_path,
        ticks=[tick(150, True, 0.001)] * 19 + [tick(201, False, 0.9)],
        host=[
            {"monotonic_ns": 150, "foreign_heavy_processes": []},
            {"monotonic_ns": 100, "foreign_heavy_processes": []},
            {"monotonic_ns": 200, "foreign_heavy_processes": [{"pid": 1234}]},
        ],
    )
    result = m1_carry.reduce_attempt(tmp_path)
    disclosure = result["discarded_run_disclosures"][0]
    assert disclosure["T_clean_ns"] == 150
    assert disclosure["carried_overruns"] == 19
    assert disclosure["prefix_update_p95_seconds"] == 0.001


def test_unretained_discard_still_disclosed(tmp_path: Path) -> None:
    attempt(tmp_path, retain=1, verdict="INCOMPLETE")
    result = m1_carry.reduce_attempt(tmp_path)
    assert result["verdict"] == "INCOMPLETE"
    assert result["discarded_run_disclosures"][0]["cell_id"] == "cpu-w4-k4"


@pytest.mark.parametrize("part", ["receipt.json", "receipt.json.complete", "receipt.json.sha256"])
def test_missing_receipt_components_refused(tmp_path: Path, part: str) -> None:
    attempt(tmp_path)
    (tmp_path / part).unlink()
    with pytest.raises(ValueError):
        m1_carry.reduce_attempt(tmp_path)


@pytest.mark.parametrize("corruption", ["missing", "changed", "unrecorded", "bad_count"])
def test_discard_audit_failures_refused(tmp_path: Path, corruption: str) -> None:
    receipt = attempt(tmp_path)
    path = tmp_path / "cpu-w4-k4.discarded-1.json"
    if corruption == "missing":
        path.unlink()
    elif corruption == "changed":
        path.write_text("{}")
    elif corruption == "unrecorded":
        export_json(tmp_path / "cpu-w4-k4.discarded-2.json", {})
    else:
        receipt = copy.deepcopy(receipt)
        receipt["discarded_runs"][0]["ticks"] = 999
        for suffix in ("", ".complete", ".sha256"):
            Path(str(tmp_path / "receipt.json") + suffix).unlink()
        export_json(tmp_path / "receipt.json", receipt)
    with pytest.raises(ValueError):
        m1_carry.reduce_attempt(tmp_path)


def test_cli_create_only_and_verified_output(tmp_path: Path, capsys) -> None:
    attempt(tmp_path)
    m1_carry.main([str(tmp_path)])
    path = tmp_path / "receipt.carried.json"
    checksum = verify_export(path)
    assert json.loads(capsys.readouterr().out)["sha256"] == checksum
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        m1_carry.main([str(tmp_path)])
    assert path.read_bytes() == original


@pytest.mark.parametrize("control_status", ["BLOCKED", "INVALID", "INCOMPLETE"])
def test_remedy_control_carry_is_disclosed_without_blocking(
    tmp_path: Path, control_status: str
) -> None:
    """R2 retains control violations without changing the worker admission bar."""
    receipt = attempt(tmp_path, discards=0)
    control = {"cell_id": "cpu-w0-k0-pre", "workers": 0, "max_pending": 0, "device": "cpu"}
    post = {**control, "cell_id": "cpu-w0-k0-post"}
    row = {
        **control,
        "namespace": "synthetic-only",
        "summary": {"complete": False, "verdict": "INCOMPLETE"},
        "stop_reason": m1.RESTART_POLICY["trigger"],
        "epoch_ns": 1,
        "host": [
            {"monotonic_ns": 100, "foreign_heavy_processes": []},
            {"monotonic_ns": 200, "foreign_heavy_processes": [{"pid": 1234}]},
        ],
        "ticks": [tick()],
        "samples": [],
    }
    # Include any frozen control-only fields (such as tick count).
    row.update(next(c for c in m1.cells() if c["cell_id"] == control["cell_id"]))
    export_json(tmp_path / "cpu-w0-k0-pre.discarded-1.json", row)
    receipt["cells"] = [
        {**control, "verdict": "PASS", "complete": True},
        *receipt["cells"],
        {**post, "verdict": control_status, "complete": True},
    ]
    receipt["remedy_attempt"] = True
    receipt["discarded_runs"] = [
        {
            "cell_id": control["cell_id"],
            "n": 1,
            "stop_reason": row["stop_reason"],
            "foreign_heavy_processes": [{"pid": 1234}],
            "started_ns": 0,
            "finished_ns": 300,
            "ticks": 1,
            "samples_count": 0,
        }
    ]
    for suffix in ("", ".complete", ".sha256"):
        Path(str(tmp_path / "receipt.json") + suffix).unlink()
    export_json(tmp_path / "receipt.json", receipt)
    result = m1_carry.reduce_attempt(tmp_path)
    assert result["verdict"] == ("PASS" if control_status == "BLOCKED" else control_status)
    assert result["cells"][0]["verdict"] == "BLOCKED"
    assert result["discarded_run_disclosures"][0]["carried_overruns"] == 1


def test_remedy_missing_controls_is_incomplete(tmp_path: Path) -> None:
    receipt = attempt(tmp_path, discards=0)
    receipt["remedy_attempt"] = True
    for suffix in ("", ".complete", ".sha256"):
        Path(str(tmp_path / "receipt.json") + suffix).unlink()
    export_json(tmp_path / "receipt.json", receipt)
    assert m1_carry.reduce_attempt(tmp_path)["verdict"] == "INCOMPLETE"


@pytest.mark.parametrize("discards", [0, 1])
def test_branch_b_effective_admission_preserves_r2_carry(tmp_path: Path, discards: int) -> None:
    """Effective B admission survives R2; a registered prefix violation still blocks."""
    receipt = attempt(tmp_path, discards=discards)
    for cell in receipt["cells"]:
        cell["branch_a_verdict"] = "BLOCKED"
    controls = [
        {"cell_id": cell["cell_id"], "verdict": "PASS", "complete": True}
        for cell in m1.cells()
        if cell["workers"] == 0
    ]
    receipt["cells"] = [controls[0], *receipt["cells"], controls[1]]
    receipt["remedy_attempt"] = True
    receipt["branch_b"] = {"enabled": True, "eligible": True}
    for suffix in ("", ".complete", ".sha256"):
        Path(str(tmp_path / "receipt.json") + suffix).unlink()
    export_json(tmp_path / "receipt.json", receipt)
    result = m1_carry.reduce_attempt(tmp_path)
    assert result["verdict"] == ("BLOCKED" if discards else "PASS")
    assert all(
        cell["branch_a_verdict"] == "BLOCKED"
        for cell in result["cells"]
        if "branch_a_verdict" in cell
    )
