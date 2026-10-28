"""Packet-08 preparation gates; CUDA execution remains operator-owned."""

from __future__ import annotations

import ast
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

from research.sandbox_a_resident import ct_progress, ct_resume
from research.sandbox_a_resident.ct_cuda_canary import fixture_control
from research.sandbox_a_resident.ct_task import CTControl, ct_seed_allocations


def _row(stage: str = "initializing") -> dict:
    return {
        "schema": "t34-cf-progress/v1",
        "study": "t34-cf-1028",
        "stage": stage,
        "elapsed_sec": 0.0,
        "latest_durable_stage": 0,
    }


def test_ast_inventory_covers_every_emit_literal_formatted_stage_and_callback_key() -> None:
    tree = ast.parse((ct_progress.HERE / "ct_run.py").read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "emit"
    ]
    inventory = ct_progress.progress_inventory()
    assert len(calls) == len(inventory) == 15
    for call in calls:
        pattern = ct_progress._stage_pattern(call.args[0])
        assert pattern in inventory
        if len(call.args) > 1 and isinstance(call.args[1], ast.Dict):
            assert ct_progress._keys(call.args[1]) <= inventory[pattern]
    for stage, module in ct_progress.CALLBACKS.items():
        callback = ast.parse((ct_progress.HERE / module).read_text())
        emitters = [
            n
            for n in ast.walk(callback)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "log_progress"
        ]
        assert emitters
        for emitter in emitters:
            assert ct_progress._keys(emitter.args[0]) <= inventory[re.escape(stage)]
    assert {
        "step",
        "minibatches",
        "searched_kl",
        "forbidden_margin",
        "weighted_anchor_kl",
        "phase",
        "arm_elapsed_sec",
    } <= inventory[re.escape("fit-progress")]


def test_real_fit_progress_and_dynamic_stage_shapes() -> None:
    row = _row("fit-progress")
    row.update(
        phase="fit_full_bv8_plus_topology",
        step=2000,
        minibatches=2000,
        searched_kl=-0.00001,
        forbidden_margin=-0.25,
        weighted_anchor_kl=0.0,
        arm_elapsed_sec=20.0,
    )
    ct_resume._scan_job_tail("CT_PROGRESS " + json.dumps(row, sort_keys=True), label="fixture")
    for stage in ("stage-0002-scoring-start", "stage-0009-published", "stage-0010-fit-start"):
        row = _row(stage)
        if stage.endswith("scoring-start"):
            row.update(chunk_index=0, roots=2)
        elif stage.endswith("published"):
            row.update(chunk_index=7, wall_sec=23)
        ct_progress.validate_progress(row)


@pytest.mark.parametrize(
    "fault",
    [
        "AssertionError",
        "LookupError",
        "KeyboardInterrupt",
        "SystemExit",
        "custom.BareException",
        "CustomFault: bad",
        "Traceback (most recent call last)",
        "Exception in thread Thread-1",
        "Traceback (most recent call last)\nOdd",
    ],
)
def test_exception_evidence_with_no_message_and_custom_classes(fault: str) -> None:
    with pytest.raises(ct_resume.CTResumeRefused, match="scientific/non-mechanical"):
        ct_resume._refuse_if_fault_text(fault, field_name="free-text log")


@pytest.mark.parametrize(
    "cause",
    [
        "priced session unit cap reached",
        "priced session unit cap reached during calibration",
        "priced session unit cap reached while awaiting account broker",
    ],
)
def test_only_exact_baseline_priced_unit_causes_are_restored(cause: str) -> None:
    for value in (cause, f"Refused({cause!r})", f"SupervisorError({cause!r})"):
        assert ct_resume._mechanical_supervisor_error(value)
        assert not ct_resume._mechanical_supervisor_error(value + "; a different cause")
    assert not ct_resume._mechanical_supervisor_error(cause + " later")


def test_frozen_pricing_outlasts_both_wall_forecast_and_effective_wall() -> None:
    root = Path(__file__).parent / "packet-ct-1028-07/reseal-02"
    manifest = json.loads((root / "dispatch-manifest.json").read_text())
    cap, rate = manifest["budget"]["unit_cap"], manifest["budget"]["rate_units_per_hour"]
    assert cap == 193 and rate == 8.0
    assert manifest["requirements"]["est_minutes"] == 720
    assert manifest["timeout_minutes"] == 1440
    assert cap * 60 / rate == 1447.5
    assert cap >= 2 * rate * (720 / 60)
    assert cap * 60 / 5.3 > 1440


def test_byte_cut_tail_requires_a_validated_full_line_suffix_and_redacts_values() -> None:
    line = "CT_PROGRESS " + json.dumps(_row(), sort_keys=True)
    text = (line + "\n") * 50
    tail = text.encode()[-4096:].decode()
    assert not tail.startswith("CT_PROGRESS")

    @ct_resume._evidence_session
    def scan() -> None:
        ct_resume._scan_job_tail(text, label="full history")
        ct_resume._EVIDENCE_CACHE.get()["progress_history_lines"] = {line}
        ct_resume._scan_job_tail(tail, label="byte-cut tail")
        corrupted = "x" + tail[1:]
        with pytest.raises(ct_resume.CTResumeRefused, match="unrecognized job_log_tail"):
            ct_resume._scan_job_tail(corrupted, label="byte-cut tail")

    scan()
    with pytest.raises(ct_resume.CTResumeRefused) as caught:
        ct_resume._scan_job_tail('CT_PROGRESS {"hero_deaths": 987654}', label="poll")
    assert "987654" not in str(caught.value)


def test_canary_seed_domains_are_disjoint_from_science_and_each_other() -> None:
    ranges = [
        ct_seed_allocations(ctrl)
        for ctrl in (
            CTControl(),
            fixture_control("a" * 40),
            fixture_control("a" * 40, signal_probe=True),
        )
    ]
    for i, domain in enumerate(ranges):
        for other in ranges[i + 1 :]:
            for a in domain:
                for b in other:
                    assert a["offset"] + a["count"] <= b["offset"] or (
                        b["offset"] + b["count"] <= a["offset"]
                    )


def test_false_or_inconsistent_shutdown_sla_still_refuses() -> None:
    row = {
        "schema": "t34-cf-sigusr1-marker/v1",
        "study": "t34-cf-1028",
        "allocation": "a" * 64,
        "status": "PREEMPTED_SIGUSR1",
        "interrupted_phase": "stage-0001-collection",
        "latest_durable_stage": 1,
        "received_unix": 1.0,
        "shutdown_elapsed_sec": 0.1,
        "within_15s_sla": True,
    }
    ct_resume._scan_job_tail("CT_SIGUSR1_CLEAN_EXIT " + json.dumps(row), label="stop")
    for update in ({"within_15s_sla": False}, {"shutdown_elapsed_sec": 15.1}):
        with pytest.raises(ct_resume.CTResumeRefused, match="unrecognized job_log_tail"):
            ct_resume._scan_job_tail(
                "CT_SIGUSR1_CLEAN_EXIT " + json.dumps({**row, **update}), label="stop"
            )


@pytest.mark.parametrize("mutation", ["tree", "source-path-type", "deep-json", "deep-repr"])
def test_source_and_diagnostic_failures_are_clean_refusals(tmp_path: Path, mutation: str) -> None:
    if mutation == "deep-json":
        with pytest.raises(ct_resume.CTResumeRefused, match="nested diagnostic"):
            ct_resume._load_json(
                "[" * (sys.getrecursionlimit() + 100) + "0" + "]" * (sys.getrecursionlimit() + 100)
            )
        return
    if mutation == "deep-repr":
        assert not ct_resume._mechanical_supervisor_error(
            "Refused(" + "[" * 2000 + "0" + "]" * 2000 + ")"
        )
        return
    from research.sandbox_a_resident.test_ct_real_producers import GOLDENS, _source_binding

    output = tmp_path / "output"
    shutil.copytree(GOLDENS / "wall-cap/output", output)
    control = json.loads((GOLDENS / "control.json").read_text())
    binding = _source_binding(output, control)
    path = output / ("staging/source-plan.json" if mutation == "tree" else "manifest.json")
    value = json.loads(path.read_text())
    if mutation == "tree":
        value["tree"] = "f" * 40
    else:
        value["source_paths"] = [123]
    path.chmod(0o600)
    path.write_text(json.dumps(value))

    @ct_resume._evidence_session
    def verify() -> None:
        ct_resume._EVIDENCE_CACHE.get()["source_archive_binding"] = binding
        ct_resume._verify_staged_source_archive(output / "staging/source.tar.gz", output)

    with pytest.raises(ct_resume.CTResumeRefused, match="unverified retained staged source"):
        verify()


@pytest.mark.parametrize("match_by", ["hash", "job", "path"])
def test_supersession_hash_job_and_path_are_independent(tmp_path: Path, match_by: str) -> None:
    import hashlib

    from tools.ops.packet_supersession import require_current_packet

    registry = tmp_path / "tools/ops"
    registry.mkdir(parents=True)
    raw = b'{"job_id":"same-job"}'
    row = {
        "packet_dir": "retired/reseal-01",
        "job_ids": ["same-job"],
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "successor": "next-packet",
        "reason": "test",
    }
    (registry / "packet-supersessions.json").write_text(
        json.dumps(
            {
                "schema": "snake-packet-supersessions/v1",
                "superseded": [row],
            }
        )
    )
    path = (
        tmp_path / ("retired/reseal-01" if match_by == "path" else "new") / "dispatch-manifest.json"
    )
    manifest = {"job_id": "same-job" if match_by == "job" else "different"}
    with pytest.raises(ValueError, match="packet superseded"):
        require_current_packet(
            manifest,
            path,
            root=tmp_path,
            original_bytes=raw if match_by == "hash" else b"reserialized",
        )


def test_unusual_export_name_must_stay_under_bound_output(tmp_path: Path) -> None:
    @ct_resume._evidence_session
    def verify() -> None:
        ct_resume._EVIDENCE_CACHE.get()["bound_output_dirs"] = {str(tmp_path / "output")}
        with pytest.raises(ct_resume.CTResumeRefused, match="unbound retained evidence"):
            ct_resume._candidate_search_roots(tmp_path / "external-exports")
        roots = ct_resume._candidate_search_roots(tmp_path / "output/external-exports")
        assert tmp_path / "output" in roots

    verify()


def test_ledger_progress_covers_real_namespaced_compact_unit() -> None:
    row = _row("ledger-verified")
    row.update(verified_stages=[1], compact_units=["a100-dev/t34-cf-1028/stages/stage-0001"])
    ct_progress.validate_progress(row)


@pytest.mark.parametrize("bad", [None, {}, {"env": None}, {"inputs": [None]}])
def test_malformed_canary_route_is_a_clean_refusal(bad: dict | None) -> None:
    from research.sandbox_a_resident.ct_canary_admission import require_smoke

    with pytest.raises(ValueError, match="malformed C/T CUDA smoke"):
        require_smoke(bad)


def test_large_checkpoint_hook_cannot_run_on_science_control(tmp_path: Path) -> None:
    from research.sandbox_a_resident.ct_run import run_ct_experiment

    with pytest.raises(ValueError, match="restricted to tiny fixtures"):
        run_ct_experiment(
            CTControl(),
            b8_checkpoint=tmp_path / "unused",
            frp3_checkpoint=tmp_path / "unused",
            champion_checkpoint=tmp_path / "unused",
            output=tmp_path / "output",
            fixture_stage_padding_bytes=1,
        )
    assert not (tmp_path / "output").exists()
