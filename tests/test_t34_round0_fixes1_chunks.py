"""Round 0 FIX-FIRST chunk stratification contracts; no gameplay runs."""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from research.t34_round0.bank import ordered_rows
from research.t34_round0.common import PACKAGE, read_json
from research.t34_round0.make_manifest import dispatch_deadline, size_round


def test_interleaved_dispatch_order_preserves_frozen_ids_and_every_stratum() -> None:
    """Derive execution order without changing the hash-bound mix-major bank."""
    bank_rows = read_json(PACKAGE / "BANK_ROUND0.json")["rows"]
    rows = ordered_rows(bank_rows)

    assert [row["world_id"] for row in rows] != [row["world_id"] for row in bank_rows]
    assert {row["world_id"] for row in rows} == {row["world_id"] for row in bank_rows}
    assert len(rows) == 240
    assert all(
        Counter(row["mix"] for row in rows[start : start + 10])
        == {"scripted": 4, "frozen": 3, "mixed": 3}
        for start in range(0, len(rows), 10)
    )


def test_sized_chunks_end_on_strata_and_each_has_registered_mix_ratio() -> None:
    """Sizing never cuts an interleaved 4:3:3 stratum in half."""
    sizing = size_round(
        240,
        teacher_worlds_per_hour=54.4,
        bare_seconds_per_world=9.0,
        metric_seconds_per_pair=4.0,
        cpu_slowdown_factor=3.0,
    )
    rows = ordered_rows(read_json(PACKAGE / "BANK_ROUND0.json")["rows"])

    assert [(chunk["start"], chunk["count"]) for chunk in sizing["chunks"]] == [(0, 150), (150, 90)]
    for chunk in sizing["chunks"]:
        start, count = int(chunk["start"]), int(chunk["count"])
        assert count % 10 == 0
        assert Counter(row["mix"] for row in rows[start : start + count]) == {
            "scripted": count * 4 // 10,
            "frozen": count * 3 // 10,
            "mixed": count * 3 // 10,
        }


def test_ratification_dispatch_deadline_is_exactly_48_utc_hours() -> None:
    assert dispatch_deadline("2026-10-20T16:15:00-07:00") == 1792710900.0


def test_missed_dispatch_deadline_is_kill_before_any_runtime_admission() -> None:
    from research.t34_round0.dispatch import HistoryError, require_dispatch_deadline

    require_dispatch_deadline(100.0, now=100.0)
    with pytest.raises(HistoryError, match="deadline has expired: KILL"):
        require_dispatch_deadline(100.0, now=100.1)


def test_cap_terminal_uses_persisted_supervisor_terminal_and_never_outcomes(tmp_path: Path) -> None:
    from research.t34_round0.common import sha256
    from research.t34_round0.dispatch import _cap_expired, _runtime_cap_expired

    job = tmp_path / "job"
    record = {
        "schema": "t34-round0-cap-expiry/v1",
        "status": "INCONCLUSIVE",
        "no_extension": True,
        "operational_reason": "registered chunk wall cap expired",
        "preflight": False,
        "run_identity": {"commit": "a" * 40, "runtime_id": "job-1"},
    }
    exported = job / "exports/round0/cap_expiry.json"
    exported.parent.mkdir(parents=True)
    exported.write_text(json.dumps(record))
    terminal = {
        "job_id": "job-1",
        "generation": "g-1",
        "reason": "timeout",
        "exports": [{"path": "round0/cap_expiry.json", "sha256": sha256(exported)}],
    }
    (job / "terminal.json").write_text(json.dumps(terminal))
    receipt = {"job_id": "job-1", "generation": "g-1", "stop_verified": True}

    assert _cap_expired(job, receipt, {"commit": "a" * 40})
    assert _runtime_cap_expired(receipt, job)


@pytest.mark.parametrize(
    "error",
    [
        "Refused('whole-session wall cap reached (including setup/export)')",
        "Refused('priced session unit cap reached')",
        "Refused('priced session unit cap reached during calibration')",
        "Refused('priced session unit cap reached while awaiting account broker')",
        "Refused('deadline reached while awaiting account broker')",
        "Refused('whole-session deadline reached before helper')",
        "Refused('whole-session deadline reached before CLI call')",
        "AccountingError('priced session unit cap reached')",
    ],
)
def test_registered_broker_cap_refusals_are_inconclusive_only(error: str) -> None:
    from research.t34_round0.dispatch import _broker_cap_expired

    receipt = {"status": "INCOMPLETE", "stop_verified": True, "cleanup_error": None, "error": error}
    assert _broker_cap_expired(receipt)
    assert not _broker_cap_expired({**receipt, "error": "Refused('unregistered timeout')"})


def test_run_sh_rejects_manifest_process_mismatch_before_runtime_work(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "research/t34_round0/run.sh"
    env = {
        **os.environ,
        "COLAB_CLI_ROOT": str(tmp_path / "runtime"),
        "COLAB_INPUT_DIR": str(tmp_path / "inputs"),
        "T34_INPUT_CATALOG": "PUBLIC_INPUTS.json",
        "T34_ROUND0_BANK": "BANK_ROUND0.json",
        "T34_ROUND0_AUDIT": "SEED_AUDIT_ROUND0.json",
        "T34_ROUND0_RATIFICATION": "RATIFICATION.json",
        "T34_ROUND0_DISPATCH_ADMISSION": "DISPATCH_ADMISSION.json",
        "T34_ROUND0_RUNTIME_ID": "test-runtime",
        "T34_ROUND0_START": "0",
        "T34_ROUND0_COUNT": "10",
        "T34_ROUND0_COMMIT": "a" * 40,
        "T34_ROUND0_PROCESSES": "11",
    }
    result = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "requires K=12" in result.stderr


def test_proven_first_cap_prevents_export_failure_rerun(tmp_path: Path) -> None:
    import json

    from research.t34_round0.common import sha256
    from research.t34_round0.dispatch import _mechanical_incomplete, _settlement

    path = tmp_path / "exports/round0/incomplete.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": "t34-round0-incomplete/v1",
                "status": "INCOMPLETE",
                "mechanical_rerun_eligible": True,
                "preflight": False,
                "operational_reason": "export or serialization failure",
                "run_identity": {"commit": "a" * 40, "runtime_id": "job-1"},
            }
        )
    )
    terminal = {
        "job_id": "job-1",
        "generation": "g-1",
        "reason": None,
        "exports": [{"path": "round0/incomplete.json", "sha256": sha256(path)}],
    }
    receipt = {
        "job_id": "job-1",
        "generation": "g-1",
        "terminal": terminal,
        "status": "FAIL",
        "stop_verified": True,
        "cleanup_error": None,
        "error": "Refused('priced session unit cap reached')",
    }
    registration = {"commit": "a" * 40}
    assert _mechanical_incomplete(tmp_path, receipt, registration)
    assert _settlement(tmp_path, receipt, registration, 360)[0] == "INCONCLUSIVE"
