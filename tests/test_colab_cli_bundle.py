"""Focused contracts for frozen Colab experiment bundles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from research.colab_cli import bundle


def plan() -> dict:
    return {
        "schema": bundle.SCHEMA,
        "experiment": "i0",
        "allocation_id": "allocation-01",
        "source": {
            "commit": "a" * 40,
            "source_sha256": "a" * 64,
            "toolchain_sha256": "b" * 64,
            "dependency_lock_sha256": "c" * 64,
        },
        "setup": [],
        "cells": [
            {
                "name": "baseline",
                "variant": "baseline",
                "cell": "e1024",
                "repeat": "r1",
                "command": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; "
                    "Path(__import__('os').environ['COLAB_BUNDLE_CELL_DIR'], "
                    "'result.json').write_text('{}')",
                ],
                "payload": "result.json",
            }
        ],
    }


def test_bundle_ledger_binds_full_frozen_plan() -> None:
    value = plan()
    ledger = bundle.empty_ledger(value)
    changed = plan()
    changed["cells"][0]["command"] = [sys.executable, "-c", "pass"]
    with pytest.raises(bundle.BundleError, match="foreign bundle resume ledger"):
        bundle._ledger(changed, ledger)


def test_run_skips_mac_verified_unit_without_remote_payload(tmp_path: Path) -> None:
    value = plan()
    ledger = bundle.empty_ledger(value)
    item = value["cells"][0]
    ledger["units"] = [{"namespace": bundle.namespace(value, item), "sha256": "d" * 64, "size": 1}]
    plan_path = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    plan_path.write_text(json.dumps(value))
    ledger_path.write_text(json.dumps(ledger))

    result = bundle.run(plan_path, tmp_path / "fresh-output", ledger_path)

    assert result["cells"] == [
        {"name": "baseline", "namespace": "i0/baseline/e1024/r1", "status": "SKIPPED_VERIFIED"}
    ]
    assert not (tmp_path / "fresh-output" / "cells").exists()


def test_duplicate_namespace_refuses_before_setup() -> None:
    value = plan()
    duplicate = dict(value["cells"][0], name="same-unit")
    value["cells"].append(duplicate)
    with pytest.raises(bundle.BundleError, match="duplicate experiment/variant/cell/repeat"):
        bundle.validate(value)


def test_distinct_experiments_share_setup_but_keep_independent_resume_rows(tmp_path: Path) -> None:
    value = plan()
    value["cells"].append(dict(value["cells"][0], name="r1-cell", experiment="r1"))
    plan_path, ledger_path = tmp_path / "plan.json", tmp_path / "ledger.json"
    plan_path.write_text(json.dumps(value))
    ledger_path.write_text(json.dumps(bundle.empty_ledger(value)))
    output = tmp_path / "output"

    result = bundle.run(plan_path, output, ledger_path)
    verified = bundle.verify_ledger(plan_path, output, ledger_path, tmp_path / "verified.json")

    assert [row["namespace"] for row in result["cells"]] == [
        "i0/baseline/e1024/r1",
        "r1/baseline/e1024/r1",
    ]
    assert {row["namespace"] for row in verified["units"]} == {
        "i0/baseline/e1024/r1",
        "r1/baseline/e1024/r1",
    }
    assert all(row["status"] == "COMPLETED_PENDING_MAC_ACCEPTANCE" for row in result["cells"])


def test_shared_setup_two_cells_and_mac_ledger_scans_each_payload(tmp_path: Path) -> None:
    value = plan()
    setup_marker = tmp_path / "setup-count"
    value["setup"] = [
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(setup_marker)!r}).write_text('once')",
        ]
    ]
    second = dict(value["cells"][0], name="candidate", variant="r1", repeat="r2")
    value["cells"].append(second)
    plan_path, input_ledger = tmp_path / "plan.json", tmp_path / "input-ledger.json"
    plan_path.write_text(json.dumps(value))
    input_ledger.write_text(json.dumps(bundle.empty_ledger(value)))

    result = bundle.run(plan_path, tmp_path / "output", input_ledger)
    verified = bundle.verify_ledger(
        plan_path, tmp_path / "output", input_ledger, tmp_path / "verified.json"
    )

    assert setup_marker.read_text() == "once"
    assert [row["status"] for row in result["cells"]] == [
        "COMPLETED_PENDING_MAC_ACCEPTANCE",
        "COMPLETED_PENDING_MAC_ACCEPTANCE",
    ]
    assert len(verified["units"]) == 2


def test_failed_cell_restores_before_independent_continuation(tmp_path: Path) -> None:
    value = plan()
    restored = tmp_path / "restored"
    value["cells"][0].update(
        command=[sys.executable, "-c", "raise SystemExit(1)"],
        on_failure="continue_after_isolation",
        restore=[
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(restored)!r}).write_text('ok')",
            ]
        ],
        isolation=[sys.executable, "-c", "pass"],
    )
    value["cells"].append(
        dict(
            value["cells"][0],
            name="independent",
            variant="r2",
            repeat="r2",
            on_failure="close",
            restore=[],
            isolation=None,
            command=plan()["cells"][0]["command"],
        )
    )
    plan_path, ledger_path = tmp_path / "plan.json", tmp_path / "ledger.json"
    plan_path.write_text(json.dumps(value))
    ledger_path.write_text(json.dumps(bundle.empty_ledger(value)))

    result = bundle.run(plan_path, tmp_path / "output", ledger_path)

    assert restored.read_text() == "ok"
    assert [row["status"] for row in result["cells"]] == [
        "FAILED",
        "COMPLETED_PENDING_MAC_ACCEPTANCE",
    ]


def test_foreign_or_duplicate_mac_ledger_refuses(tmp_path: Path) -> None:
    value = plan()
    ledger = bundle.empty_ledger(value)
    unit = {"namespace": bundle.namespace(value, value["cells"][0]), "sha256": "d" * 64, "size": 1}
    ledger["units"] = [unit, dict(unit)]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(value))
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger))
    with pytest.raises(bundle.BundleError, match="foreign or duplicate"):
        bundle.verify_ledger(plan_path, tmp_path, path, tmp_path / "out.json")


def test_failed_isolation_prevents_unrelated_continuation(tmp_path: Path) -> None:
    import subprocess

    value = plan()
    value["cells"][0].update(
        command=[sys.executable, "-c", "raise SystemExit(1)"],
        on_failure="continue_after_isolation",
        restore=[[sys.executable, "-c", "pass"]],
        isolation=[sys.executable, "-c", "raise SystemExit(2)"],
    )
    value["cells"].append(dict(plan()["cells"][0], name="unrelated", variant="next"))
    frozen, ledger = tmp_path / "plan.json", tmp_path / "ledger.json"
    frozen.write_text(json.dumps(value))
    ledger.write_text(json.dumps(bundle.empty_ledger(value)))
    with pytest.raises(subprocess.CalledProcessError):
        bundle.run(frozen, tmp_path / "output", ledger)
    assert not (tmp_path / "output/cells/unrelated").exists()


def test_dependency_pass_is_uploaded_and_binds_parent_bytes(tmp_path: Path, monkeypatch) -> None:
    value = plan()
    uploaded = tmp_path / "inputs"
    uploaded.mkdir()
    identity = bundle.namespace(value, value["cells"][0])
    verdict = {
        "status": "PASS",
        "namespace": identity,
        "payload_sha256": "d" * 64,
        "source": value["source"],
    }
    admission = uploaded / "parent-pass.json"
    admission.write_text(json.dumps(verdict))
    value["dependency_admissions"] = {
        "baseline": {
            "path": "/Users/never-present/parent-pass.json",
            "dest": "parent-pass.json",
            "sha256": bundle.digest(admission),
        }
    }
    value["cells"].append(
        dict(value["cells"][0], name="dependent", variant="child", depends_on=["baseline"])
    )
    frozen, ledger = tmp_path / "plan.json", tmp_path / "ledger.json"
    frozen.write_text(json.dumps(value))
    verified = bundle.empty_ledger(value)
    verified["units"] = [{"namespace": identity, "sha256": "d" * 64, "size": 1}]
    ledger.write_text(json.dumps(verified))
    monkeypatch.setenv("COLAB_INPUT_DIR", str(uploaded))
    result = bundle.run(frozen, tmp_path / "out", ledger)
    assert [row["status"] for row in result["cells"]] == [
        "SKIPPED_VERIFIED",
        "COMPLETED_PENDING_MAC_ACCEPTANCE",
    ]
    verdict["payload_sha256"] = "e" * 64
    admission.write_text(json.dumps(verdict))
    value["dependency_admissions"]["baseline"]["sha256"] = bundle.digest(admission)
    with pytest.raises(bundle.BundleError, match="payload or frozen source"):
        bundle._accepted_dependencies(value, {identity: verified["units"][0]})


def test_mac_union_refuses_duplicate_and_foreign_complete_units(tmp_path: Path) -> None:
    value = plan()
    frozen, empty = tmp_path / "plan.json", tmp_path / "empty.json"
    frozen.write_text(json.dumps(value))
    empty.write_text(json.dumps(bundle.empty_ledger(value)))
    bundle.run(frozen, tmp_path / "out", empty)
    verified = tmp_path / "verified.json"
    bundle.verify_ledger(frozen, tmp_path / "out", empty, verified)
    with pytest.raises(bundle.BundleError, match="duplicate completed"):
        bundle.verify_ledger(frozen, tmp_path / "out", verified, tmp_path / "duplicate.json")
    (tmp_path / "out/foreign.bin.complete").write_text("complete")
    with pytest.raises(bundle.BundleError, match="foreign completed"):
        bundle.verify_ledger(frozen, tmp_path / "out", empty, tmp_path / "foreign.json")
