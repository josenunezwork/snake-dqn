"""Offline package invariants; no provider or real corpus fixtures."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from research.bl_residual import f5_package as package


def test_source_inventory_recurses_relative_imports_and_package_initializers(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    files = {
        "research/__init__.py": "",
        "research/bl_residual/__init__.py": "",
        "research/bl_residual/entry.py": "from research.model import Model\n",
        "research/model/__init__.py": "from .base import Model\n",
        "research/model/base.py": "from .checkpoint_manager import Checkpoint\nclass Model: pass\n",
        "research/model/checkpoint_manager.py": (
            "from .checkpoint_io import load\nclass Checkpoint: pass\n"
        ),
        "research/model/checkpoint_io.py": "def load(): pass\n",
    }
    for relative, contents in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)

    inventory = package.source_inventory(repo, roots=("research/bl_residual/entry.py",))
    assert set(files) <= set(inventory)
    before = inventory["research/model/checkpoint_io.py"]
    (repo / "research/model/checkpoint_io.py").write_text("def load(): return 'mutated'\n")
    updated = package.source_inventory(repo, roots=("research/bl_residual/entry.py",))
    assert updated["research/model/checkpoint_io.py"] != before


def test_capture_attempt_two_retains_elapsed_and_debit(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "job_id": "bl-f5-w-capture-repair-22e86986-01",
                "stop_verified": True,
                "failure_class": "CHILD_NONZERO_EXIT",
                "exports": [],
                "elapsed_seconds": 421.83044687495567,
                "accounting": {"last_crosscheck": {"observed_account_debit_units": 0.12}},
            }
        )
    )
    value = package.capture_remainder(receipt)
    assert value["remaining_capture_seconds"] == 178.16955312504433
    assert value["remaining_aggregate_seconds"] == 13978.169553125044
    assert value["prior_accounting"]["last_crosscheck"]["observed_account_debit_units"] == 0.12
    assert value["admission"] == "HOLD"
    assert len(value["hold_reasons"]) == 2
    assert value["allowance_reset"] is False
    source_hash = package.digest(receipt)
    assert value["prior_capture_receipt"]["sha256"] == source_hash
    for key, bad in (
        ("stop_verified", False),
        ("elapsed_seconds", -1),
        ("elapsed_seconds", float("nan")),
        ("exports", ["science"]),
        ("accounting", None),
    ):
        changed = json.loads(receipt.read_text())
        changed[key] = bad
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="prior capture"):
            package.capture_remainder(path)


def _stopped_capture(job_id: str, elapsed: float, debit: float, observed: float) -> dict:
    return {
        "job_id": job_id,
        "stop_verified": True,
        "failure_class": "CHILD_NONZERO_EXIT",
        "exports": [],
        "elapsed_seconds": elapsed,
        "accounting": {
            "last_crosscheck": {
                "attributable_account_debit_units": debit,
                "observed_account_debit_units": observed,
            },
            "raw_usage_preserved": True,
        },
    }


def test_capture_history_binds_both_stopped_attempts_and_discloses_all_debits(
    tmp_path: Path,
) -> None:
    first = tmp_path / "attempt-01.json"
    second = tmp_path / "attempt-02.json"
    first.write_text(
        json.dumps(_stopped_capture("bl-f5-w-capture-repair-22e86986-01", 421.0, 0.12, 0.12))
    )
    second.write_text(
        json.dumps(_stopped_capture("bl-f5-w-capture-repair-22e86986-02", 558.0, 0.10, 0.10))
    )
    value = package.capture_history([first, second])
    assert value["cumulative_actual_elapsed_seconds"] == 979.0
    assert value["cumulative_accounting_debit_units"] == pytest.approx(0.22)
    assert value["cumulative_observed_account_debit_units"] == pytest.approx(0.22)
    assert value["attempts"][1]["raw_accounting"]["raw_usage_preserved"] is True
    assert value["budget_status"] == "DISCLOSED_NOT_AN_ADMISSION_BLOCKER"
    with pytest.raises(ValueError, match="two or three published"):
        package.capture_history([first])
    second.write_text(json.dumps(_stopped_capture("foreign", 558.0, 0.10, 0.10)))
    with pytest.raises(ValueError, match="stopped non-science"):
        package.capture_history([first, second])


def test_exact_owner_ruling_binds_a_30_minute_capture_runaway_timeout(tmp_path: Path) -> None:
    ruling = tmp_path / "OWNER_RULING_COLAB_BUDGET_20261025.txt"
    ruling.write_text("different")
    with pytest.raises(ValueError, match="exact known"):
        package.capture_owner_override(ruling)
    value = package.capture_owner_override(package.OWNER_RULING_CAPTURE_PATH)
    assert value is not None
    assert value["binding"]["sha256"] == package.OWNER_RULING_CAPTURE_SHA
    assert value["capture_runaway_timeout_minutes"] == 30
    assert "current account" in value["rate_status"]


def test_rebuild_cannot_omit_published_failed_attempt(tmp_path: Path) -> None:
    receipt = tmp_path / (
        "research/bl_residual/ledger/F5_W_CAPTURE_AFTER_22e86986_REPAIR_01/"
        "output-01/receipt.json"
    )
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="requires explicit --prior-capture-receipt"):
        package.build(tmp_path / "new-packet", repo=tmp_path)
    assert not (tmp_path / "new-packet").exists()


def draft() -> dict:
    return {
        "scientific_admission": False,
        "commit": "a" * 40,
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 6, "est_minutes": 60},
        "timeout_minutes": 200,
        "device": "cuda:0",
        "resume": {
            "unit_kind": "paired-fit",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "f5": {"transfer_seconds": 150, "staging_seconds": 300},
    }


def test_draft_is_reviewable_and_cannot_claim_admission() -> None:
    value = draft()
    assert len(package.validate_draft_manifest(value)) == 5
    value["scientific_admission"] = True
    with pytest.raises(ValueError, match="refuse operational"):
        package.validate_draft_manifest(value)


def test_full_timing_cannot_inherit_transport_forecast_or_resume_exemption() -> None:
    from research.colab_cli import supervisor

    transport = {
        **draft(),
        "job_id": "f5-timing-envelope-test",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=package.ROOT, text=True
        ).strip(),
        "script": "research/bl_residual/f5_colab.sh",
        "budget": {"rate_units_per_hour": 2, "unit_cap": 100, "reserve_units": 0},
        "requirements": {"gpu": "L4", "min_vcpu": 6, "est_minutes": 15},
        "timeout_minutes": 30,
        "resume_exemption": {"kind": "short-canary", "timeout_minutes": 30},
    }
    original = copy.deepcopy(transport)
    proposal = package.hold_timing_envelope(transport)
    assert transport == original
    assert proposal["requirements"]["est_minutes"] is None
    assert proposal["timeout_minutes"] is None
    assert "resume" not in proposal and "resume_exemption" not in proposal
    with pytest.raises(supervisor.Refused, match="est_minutes must be numeric"):
        supervisor.validate(proposal, repo=package.ROOT)


@pytest.mark.parametrize(
    "key,value",
    [
        ("timeout_minutes", 480),
        ("device", "mps"),
        ("commit", "a" * 8),
        ("requirements", {"gpu": "A100"}),
        ("resume", {"skip_verified": False}),
        ("f5", {"transfer_seconds": 300, "staging_seconds": 300}),
    ],
)
def test_no_widened_timeout_placement_transfer_or_resume(key: str, value: object) -> None:
    candidate = copy.deepcopy(draft())
    candidate[key] = value
    with pytest.raises(ValueError):
        package.validate_draft_manifest(candidate)


def test_one_difference_and_original_denominator() -> None:
    record = package.one_difference()
    assert record["differences"] == ["qualifying_ordinary_B8_survivor_pair_weight"]
    assert record["C"]["qualifying_ordinary_B8_survivor_pair_weight"] == 1
    assert record["T"]["qualifying_ordinary_B8_survivor_pair_weight"] == 4
    assert record["T"]["recipe"]["updates"] == 2000
    assert record["C"]["survival_denominator"] == "original eligible pair count"


def test_prohibited_and_symlink_paths_refuse_before_open(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prohibited"):
        package.safe_metadata(tmp_path / "d4096" / "anything.json")
    with pytest.raises(ValueError, match="prohibited"):
        package.safe_metadata(tmp_path / "t34-round0-study-01" / "anything.json")
    real = tmp_path / "real.json"
    real.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        package.safe_metadata(link)


def test_bound_metadata_digest_mismatch_refuses(tmp_path: Path) -> None:
    path = tmp_path / "governance.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="digest mismatch"):
        package.binding(path, "0" * 64)


def test_final_canary_binds_every_science_byte_and_producer(tmp_path: Path) -> None:
    """The final transport probe must upload the eventual science files exactly."""
    names = {
        "control.json",
        "b8.pth",
        "selected-training.tar.xz",
        "f5-batch-plan.pt",
        "acceptance.json",
        "verified-ledger.json",
        "timing-fixtures.json",
    }
    from research.bl_ppo.exports import export_bytes

    immutable = {"selected-training.tar.xz", "f5-batch-plan.pt", "acceptance.json"}
    rows = []
    for name in sorted(names):
        path = tmp_path / name
        if name in immutable:
            export_bytes(path, b"synthetic inventory only: " + name.encode())
        else:
            path.write_bytes(b"synthetic inventory only: " + name.encode())
        rows.append({"path": str(path), "dest": name, "sha256": package.digest(path)})
    for name in sorted(immutable):
        for suffix in (".sha256", ".complete"):
            path = tmp_path / (name + suffix)
            rows.append({"path": str(path), "dest": name + suffix, "sha256": package.digest(path)})
    science = {
        "inputs": rows,
        "commit": "a" * 40,
        "source_paths": ["source.py"],
        "script": "research/bl_residual/f5_colab.sh",
        "required_exports": ["paired/fit-C.pth", "paired/fit-W.pth"],
    }
    probe = {**science, "inputs": []}
    result = package.bind_qualification_inventory(science, probe)
    assert result["inputs"] == rows
    assert set(result["required_exports"]) == {
        "paired/fit-C.pth",
        "paired/fit-W.pth",
        "qualification/qualification.json",
    }
    assert len(result["science_upload_inventory"]) == len(names) + 6
    assert science["inputs"] == rows
    science["inputs"] = rows[:-1]
    with pytest.raises(ValueError, match="incomplete"):
        package.bind_qualification_inventory(science, probe)
    science["inputs"] = rows
    Path(rows[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        package.bind_qualification_inventory(science, probe)


def test_capture_history_carries_attempt03_diagnostics_and_refuses_science(tmp_path: Path) -> None:
    paths = []
    for attempt in range(1, 4):
        value = _stopped_capture(
            f"bl-f5-w-capture-repair-22e86986-{attempt:02d}", 100.0 * attempt, 0.1, 0.1
        )
        if attempt == 3:
            value["exports"] = [
                {"path": name, "sha256": "a" * 64, "size": 123}
                for name in ("environment-raw.json", "environment-diagnostic.json")
            ]
        path = tmp_path / f"attempt-{attempt}.json"
        path.write_text(json.dumps(value))
        paths.append(path)
    history = package.capture_history(paths)
    assert len(history["attempts"]) == 3
    assert history["cumulative_actual_elapsed_seconds"] == 600.0
    assert history["cumulative_accounting_debit_units"] == pytest.approx(0.3)
    value["exports"][0]["path"] = "environment.json"
    paths[-1].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="stopped non-science"):
        package.capture_history(paths)
