"""Unit and integration tests for Watch6 V3 rev 2 confirmation (all 5 Astra ruling points)."""

from __future__ import annotations

import copy
import gzip
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from research.colab_cli import canary
from research.watch6_search_efficacy import analyze as v2_analyze
from research.watch6_search_efficacy import protocol as v2_p
from research.watch6_search_efficacy_v3 import (
    admission,
    analyze,
    costs,
    dispatch,
    final_look,
    finalize,
    freshness,
    mirror,
)
from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3 import (
    ratify,
    recover,
    run,
    uploads,
)
from tools.ops.package_spec import (
    Input,
    digest,
    empty_ledger,
    publish_json,
    stage_inputs,
    unit_id,
    verify_immutable,
)

VALID_TRANSPORT_TRACEBACK = (
    "Refused('colab exec failed (1): Traceback (most recent call last):\\n"
    '  File "/opt/colab/cli/client.py", line 142, in execute_rpc\\n'
    "    response = self._http.post(url, timeout=30.0)\\n"
    '  File "/opt/colab/venv/lib/python3.12/site-packages/httpx/_client.py", line 1144, in post\\n'
    '    return self.request("POST", url, timeout=timeout)\\n'
    "httpx.ReadTimeout: The read operation timed out')"
)


def _synthetic_v3_rows(*, death_delta: int = -2, relative: float = 0.0) -> list[dict]:
    primary_alloc = p.allocation()
    result = []
    for index in range(p.COUNT):
        alloc = copy.deepcopy(primary_alloc[index])
        bare = {
            "frames": p.HORIZON,
            "mirror_identity_frames": p.HORIZON,
            "mass_sum": (500 + index * 20) * p.HORIZON,
            "deaths": 4 + index % 2,
        }
        dense = {
            **bare,
            "mass_sum": bare["mass_sum"] * (1 + relative),
            "deaths": bare["deaths"] + death_delta + (index % 2 if death_delta else 0),
        }
        result.append(
            {
                "schema": "watch6-search-pair/v3",
                "namespace": p.NAMESPACE,
                "science": p.identity(),
                "seed_kind": "primary",
                "allocation": alloc,
                "fixture": False,
                "unit_id": unit_id(p.NAMESPACE, "watch6", index),
                "arms": {"bare": bare, "exact_stop": dense},
            }
        )
    return result


def _synthetic_cost_report() -> dict:
    unit_seconds = [
        2875.628143575,
        2465.707615435,
        2877.332813413,
        2884.5381825629997,
        2782.20794508,
        1834.582514666,
        1162.4141118939997,
        2251.030796088,
        1779.7341660670008,
        505.75122514099985,
        521.022599889,
        2602.3025635050008,
        580.661589585,
        544.2436565339999,
        535.9960847250004,
        2874.661794685,
        545.610884795,
        2711.3811861120003,
        2825.4309084419997,
        527.3378031069997,
        503.98841117400025,
        2804.95671338,
        3315.239382726999,
        1295.9985010719993,
    ]
    native = {
        "audited_rows": 0,
        "boost_base_rows": 0,
        "calls": 1500,
        "fallback_rows": 0,
        "head_changes": 0,
        "kernel_rows": 1500,
        "long_rows": 0,
        "no_spacious": 0,
        "noncandidate_rows": 0,
        "overrides": 3,
        "python_calls": 0,
        "replayed_rows": 0,
        "root_stock_rows": 0,
        "rows": 1500,
        "stock_rows": 0,
        "tail_release_rows": 0,
        "tie_overrides": 0,
        "tie_rows": 0,
        "v5_changes": 0,
        "v7_changes": 0,
    }
    profiles = []
    for workers in (4, 6, 8):
        rows = [
            {
                "index": idx,
                "arms": {
                    "bare": {
                        "main_seconds_per_frame": 0.04428428550334956,
                        "mirror_identity_frames": 300,
                    },
                    "exact_stop": {
                        "main_seconds_per_frame": 0.04443272172334522,
                        "mirror_identity_frames": 300,
                    },
                },
                "branches": [
                    {
                        "seed": seed,
                        "seconds": 4.0,
                        "branch_seconds_upper": 4.0,
                        "branches": 1,
                        "frames_executed": [300],
                        "native_v8": native,
                    }
                    for seed in p.FIXTURE_SEEDS
                ],
            }
            for idx in range(workers)
        ]
        profiles.append(
            {
                "workers": workers,
                "branch_seconds_upper": 4.0,
                "main_seconds_per_frame": {
                    "bare": 0.04428428550334956,
                    "exact_stop": 0.04443272172334522,
                },
                "rows": rows,
            }
        )
    return {
        "schema": "watch6-empirical-costs/v3",
        "science": p.identity(),
        "measurement_science": costs.BASE_SCIENCE,
        "measurement_pin": costs.BASE_PIN,
        "canary": {
            "schema": "watch6-target-performance/v1",
            "science": costs.BASE_SCIENCE,
            "cpu_count": 12,
            "fixture": True,
            "horizon": 300,
            "profiles": profiles,
        },
        "canary_receipt": {"path": "/tmp/canary/receipt.json", "sha256": "a" * 64},
        "banks": [
            {
                "namespace": "t34-watch-replay-fix-20261030-bank01",
                "workers": 4,
                "horizon": 6000,
                "units": [
                    {"index": i, "seconds": unit_seconds[i], "frames": 6000, "search_roots": 500}
                    for i in range(12)
                ],
            },
            {
                "namespace": "t34-watch-replay-fix-20261030-bank02",
                "workers": 4,
                "horizon": 6000,
                "units": [
                    {
                        "index": i,
                        "seconds": unit_seconds[12 + i],
                        "frames": 6000,
                        "search_roots": 500,
                    }
                    for i in range(12)
                ],
            },
        ],
    }


def _write_approved_ruling(tmp_path: Path, verdict: str = "APPROVE") -> Path:
    ruling_md = tmp_path / "RULING_V3_APPROVED.md"
    ruling_md.write_text(
        "# Binding PRE-DATA ruling — Watch6 V3\n\n"
        f"Verdict: {verdict}\n"
        "STATUS: RATIFIED\n"
        "Judge: Astra · 2026-10-31\n",
        encoding="utf-8",
    )
    ruling_json = tmp_path / "astra-ruling.json"
    ratify.record_ruling(ruling_md, ruling_json)
    return ruling_json


def _write_attempt_output(
    out_dir: Path,
    *,
    shard: int,
    mode: str = "pass",
    status: str | None = None,
    failure_class: str | None = None,
    error: str | None = None,
    endpoint: str | None = "m-s-test-endpoint",
    stop_verified: bool = True,
    cleanup_error: str | None = None,
    export_slots: list[int] | None = None,
    pair_rows_by_slot: dict[int, dict] | None = None,
    prestart_retries_used: int = 0,
    compute_completion_attempt: int = 0,
    prior_receipts: list[dict] | None = None,
    completion_attempt: int | None = None,
    job_log_tail: str | None = None,
    supervisor_log_tail: str = "",
) -> Path:
    """Create a realistic supervisor.py output directory for tests.

    Modes:
    - "pass": full PASS attempt with hardware.json, poll-000000.json, and terminal.json (rc=0).
    - "prestart_staging": positive pre-1904 staging failure (warmup PASS, rate-calibration PASS,
      upload-measurement.json with staging_completed=False, no hardware.json, no poll-*.json).
    - "prestart_warmup": positive pre-1904 warmup failure (warmup-measurement.json INCOMPLETE).
    - "prestart_unassigned": pre-allocation refusal (endpoint=None, stop_verified=False).
    - "compute_transport": compute-started external transport failure (hardware.json + clean
      poll-000000.json + failed cli_calls + verified host/CLI transport traceback).
    - "bare_no_evidence": missing hardware.json AND missing pre-1904 positive proofs.
    - "child_fail": CHILD_NONZERO_EXIT with terminal returncode=1.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    priors = list(prior_receipts or [])
    tot_dispatches = 1 + prestart_retries_used + compute_completion_attempt

    manifest: dict = {
        "job_id": f"watch6-v3-shard-{shard:02d}-{out_dir.name}",
        "output_dir": str(out_dir),
        "commit": "c" * 40,
        "script": p.SCRIPT,
        "device": "cpu",
        "env": {
            "WATCH6_FIXTURE": "0",
            "WATCH6_SHARD": str(shard),
            "WATCH6_SHARDS": str(p.SHARDS),
            "WATCH6_WORKERS": str(p.WORKERS),
            "WATCH6_SCIENCE": p.identity(),
            "SNAKE_WATCH_SOURCE": "live",
        },
        "watch6_v3": {
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "primary_allocation": p.allocation(),
            "shard": shard,
            "phase": "primary",
            "prestart_retries_used": prestart_retries_used,
            "compute_completion_attempt": compute_completion_attempt,
            "total_dispatches_for_shard": tot_dispatches,
            "prior_receipts": priors,
        },
    }
    if completion_attempt is not None:
        manifest["completion_attempt"] = completion_attempt

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    cli_calls: list[dict] = []
    terminal_obj: dict | None = None

    if mode == "pass":
        eff_status = status or "PASS"
        eff_fcls = failure_class
        eff_err = error
        (out_dir / "warmup-measurement.json").write_text('{"status": "PASS"}\n')
        (out_dir / "rate-calibration.json").write_text('{"status": "PASS"}\n')
        (out_dir / "upload-measurement.json").write_text('{"elapsed_seconds": 65.0}\n')
        (out_dir / "source-delivery.json").write_text('{"transport": "github"}\n')
        (out_dir / "input-hashes.json").write_text("{}\n")
        (out_dir / "hardware.json").write_text('{"ready": true, "hardware": {"kind": "L4"}}\n')
        (out_dir / "poll-000000.json").write_text(
            json.dumps(
                {
                    "state": {
                        "status": "running",
                        "returncode": None,
                        "reason": None,
                        "error": None,
                        "unassign_error": None,
                    },
                    "terminal": None,
                    "job_log_tail": (
                        job_log_tail
                        if job_log_tail is not None
                        else f"WATCH6_V3_PROGRESS shard={shard} done=4/4 skipped=0 elapsed=100.0s\n"
                    ),
                    "supervisor_log_tail": supervisor_log_tail,
                }
            )
            + "\n"
        )
        terminal_obj = {"returncode": 0, "reason": None, "error": None}
        (out_dir / "terminal.json").write_text(json.dumps(terminal_obj) + "\n")
    elif mode == "prestart_staging":
        eff_status = status or "INCOMPLETE"
        eff_fcls = failure_class or "RPC_EXEC_POLL_TIMEOUT"
        eff_err = error or VALID_TRANSPORT_TRACEBACK
        (out_dir / "warmup-measurement.json").write_text('{"status": "PASS"}\n')
        (out_dir / "rate-calibration.json").write_text('{"status": "PASS"}\n')
        (out_dir / "upload-measurement.json").write_text(
            '{"elapsed_seconds": 120.0, "staging_completed": false}\n'
        )
        cli_calls = [{"subcommand": "upload", "returncode": 1}]
    elif mode == "prestart_warmup":
        eff_status = status or "INCOMPLETE"
        eff_fcls = failure_class or "SUPERVISOR"
        eff_err = error or "Refused('runtime readiness warmup deadline reached')"
        (out_dir / "warmup-measurement.json").write_text(
            '{"status": "INCOMPLETE", "error": "Refused(\'warmup deadline\')"}\n'
        )
    elif mode == "prestart_unassigned":
        endpoint = None
        stop_verified = False
        eff_status = status or "INCOMPLETE"
        eff_fcls = failure_class or "ACCOUNTING"
        eff_err = error or "AccountingError('account-wide assignment cap 6 (including notebooks)')"
    elif mode == "compute_transport":
        eff_status = status or "INCOMPLETE"
        eff_fcls = failure_class or "RPC_EXEC_POLL_TIMEOUT"
        eff_err = error if error is not None else VALID_TRANSPORT_TRACEBACK
        (out_dir / "warmup-measurement.json").write_text('{"status": "PASS"}\n')
        (out_dir / "rate-calibration.json").write_text('{"status": "PASS"}\n')
        (out_dir / "upload-measurement.json").write_text('{"elapsed_seconds": 65.0}\n')
        (out_dir / "source-delivery.json").write_text('{"transport": "github"}\n')
        (out_dir / "input-hashes.json").write_text("{}\n")
        (out_dir / "hardware.json").write_text('{"ready": true, "hardware": {"kind": "L4"}}\n')
        (out_dir / "poll-000000.json").write_text(
            json.dumps(
                {
                    "state": {
                        "status": "running",
                        "returncode": None,
                        "reason": None,
                        "error": None,
                        "unassign_error": None,
                    },
                    "terminal": None,
                    "job_log_tail": (
                        job_log_tail
                        if job_log_tail is not None
                        else f"WATCH6_V3_PROGRESS shard={shard} done=0/4 skipped=0 elapsed=60.0s\n"
                    ),
                    "supervisor_log_tail": supervisor_log_tail,
                }
            )
            + "\n"
        )
        cli_calls = [{"subcommand": "exec", "returncode": 1}]
    elif mode == "bare_no_evidence":
        eff_status = status or "INCOMPLETE"
        eff_fcls = failure_class or "SUPERVISOR"
        eff_err = error or VALID_TRANSPORT_TRACEBACK
    elif mode == "child_fail":
        eff_status = "INCOMPLETE"
        eff_fcls = "CHILD_NONZERO_EXIT"
        eff_err = error or "ValueError: mirror mismatch at frame 42"
        (out_dir / "hardware.json").write_text('{"ready": true, "hardware": {"kind": "L4"}}\n')
        terminal_obj = {"returncode": 1, "reason": "nonzero_exit", "error": eff_err}
        (out_dir / "terminal.json").write_text(json.dumps(terminal_obj) + "\n")
    else:
        raise ValueError(f"unknown helper mode: {mode}")

    exports = []
    for slot in export_slots or []:
        alloc = p.allocation()[slot]
        payload = (
            pair_rows_by_slot[slot]
            if pair_rows_by_slot and slot in pair_rows_by_slot
            else {"sealed": True, "index": slot}
        )
        pair_path = out_dir / "exports/efficacy" / f"pair-{slot:02d}.json"
        desc = publish_json(pair_path, payload)
        meta_desc = publish_json(
            pair_path.with_suffix(".meta.json"),
            {
                "namespace": p.NAMESPACE,
                "science": p.identity(),
                "seed_kind": "primary",
                "allocation": alloc,
                "fixture": False,
                "sha256": desc["sha256"],
                "size": desc["size"],
            },
        )
        exports.append(
            {
                "path": f"efficacy/pair-{slot:02d}.json",
                "sha256": desc["sha256"],
                "size": desc["size"],
            }
        )
        exports.append(
            {
                "path": f"efficacy/pair-{slot:02d}.meta.json",
                "sha256": meta_desc["sha256"],
                "size": meta_desc["size"],
            }
        )

    receipt = {
        "job_id": manifest["job_id"],
        "commit": manifest["commit"],
        "manifest_sha256": digest(manifest_path),
        "endpoint": endpoint,
        "status": eff_status,
        "failure_class": eff_fcls,
        "error": eff_err,
        "stop_verified": stop_verified,
        "cleanup_error": cleanup_error,
        "terminal": terminal_obj,
        "exports": exports,
        "transfers": [],
        "contents_transfers": [],
        "cli_calls": cli_calls,
        "control_operations": [],
    }
    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return receipt_path


def test_v3_ruling_point1_reproduces_astras_two_acceptances_and_refuses_both(
    tmp_path: Path,
) -> None:
    # --- Reproduction 1A: Compute-started RPC loss (poll-*.json + hardware.json, 0 exports) ---
    # Previously misclassified as pre-start because V3 rev 1 searched telemetry-*.json!
    r_compute_zero_exp = _write_attempt_output(
        tmp_path / "compute-zero-exports",
        shard=0,
        mode="compute_transport",
        export_slots=[],
    )
    rec_compute = json.loads(r_compute_zero_exp.read_text())
    assert admission.is_prestart_transport_failure(rec_compute, r_compute_zero_exp.parent) is False
    assert recover.is_prestart_transport_failure(rec_compute, r_compute_zero_exp.parent) is False
    assert (
        admission.is_external_transport_compute_failure(rec_compute, r_compute_zero_exp.parent)
        is True
    )

    # --- Reproduction 1B: Missing hardware.json without positive pre-1904 proof ---
    r_bare = _write_attempt_output(tmp_path / "bare-missing-hw", shard=0, mode="bare_no_evidence")
    rec_bare = json.loads(r_bare.read_text())
    assert admission.is_prestart_transport_failure(rec_bare, r_bare.parent) is False
    assert admission.is_external_transport_compute_failure(rec_bare, r_bare.parent) is False

    # Ambiguous post-staging failure at line 1904 (staging completed, non-UPLOAD_CAP, no hw.json):
    ambig_dir = tmp_path / "ambig-line1904"
    r_ambig = _write_attempt_output(ambig_dir, shard=0, mode="bare_no_evidence")
    (ambig_dir / "warmup-measurement.json").write_text('{"status": "PASS"}\n')
    (ambig_dir / "rate-calibration.json").write_text('{"status": "PASS"}\n')
    (ambig_dir / "upload-measurement.json").write_text('{"elapsed_seconds": 70.0}\n')
    (ambig_dir / "source-delivery.json").write_text('{"transport": "github"}\n')
    (ambig_dir / "input-hashes.json").write_text("{}\n")
    rec_ambig = json.loads(r_ambig.read_text())
    assert admission.is_prestart_transport_failure(rec_ambig, ambig_dir) is False

    # Post-staging UPLOAD_CAP before line 1904 IS positively pre-start:
    cap_dir = tmp_path / "post-staging-upload-cap"
    r_cap = _write_attempt_output(
        cap_dir,
        shard=0,
        mode="bare_no_evidence",
        failure_class="UPLOAD_CAP",
        error="Refused('total upload time cap reached after staging control')",
    )
    (cap_dir / "warmup-measurement.json").write_text('{"status": "PASS"}\n')
    (cap_dir / "rate-calibration.json").write_text('{"status": "PASS"}\n')
    (cap_dir / "upload-measurement.json").write_text('{"elapsed_seconds": 190.0}\n')
    (cap_dir / "source-delivery.json").write_text('{"transport": "github"}\n')
    (cap_dir / "input-hashes.json").write_text("{}\n")
    assert admission.is_prestart_transport_failure(json.loads(r_cap.read_text()), cap_dir) is True

    # Other genuine pre-1904 proofs (staging_completed=False, warmup INCOMPLETE, endpoint=None):
    for label, p_mode in (
        ("p-staging", "prestart_staging"),
        ("p-warmup", "prestart_warmup"),
        ("p-unassigned", "prestart_unassigned"),
    ):
        rp = _write_attempt_output(tmp_path / label, shard=0, mode=p_mode)
        assert admission.is_prestart_transport_failure(json.loads(rp.read_text()), rp.parent)

    # --- Reproduction 2A: In-memory / bare transport-labelled error WITHOUT traceback evidence ---
    in_memory_receipt = {
        "status": "INCOMPLETE",
        "stop_verified": True,
        "cleanup_error": None,
        "endpoint": "m-s-1",
        "failure_class": "RPC_EXEC_POLL_TIMEOUT",
        "error": "rpc poll timed out",
        "terminal": None,
    }
    assert admission.is_external_transport_compute_failure(in_memory_receipt, None) is False

    r_no_tb = _write_attempt_output(
        tmp_path / "compute-no-traceback",
        shard=0,
        mode="compute_transport",
        error="Refused('deadline reached during telemetry recovery')",
    )
    assert (
        admission.is_external_transport_compute_failure(
            json.loads(r_no_tb.read_text()), r_no_tb.parent
        )
        is False
    )

    # --- Reproduction 2B: Transport-labelled error carrying scientific/integrity/timeout faults ---
    poisoned_errors = [
        VALID_TRANSPORT_TRACEBACK[:-2] + "; ValueError: mirror mismatch at frame 88')",
        VALID_TRANSPORT_TRACEBACK[:-2] + "; non-finite NaN Q-value detected')",
        VALID_TRANSPORT_TRACEBACK[:-2] + "; AssertionError: digest mismatch')",
        VALID_TRANSPORT_TRACEBACK[:-2] + "; whole-session wall cap reached')",
        VALID_TRANSPORT_TRACEBACK[:-2] + "; faulthandler dump_traceback_later')",
        (
            "Refused('colab exec failed (1): Traceback (most recent call last):\\n"
            '  File "research/watch6_search_efficacy/engine.py", line 90, in run_arm\\n'
            '    raise TimeoutError("stalled")\\n'
            "TimeoutError: stalled')"
        ),
    ]
    for idx, bad_err in enumerate(poisoned_errors):
        for f_cls in ("RPC_EXEC_POLL_TIMEOUT", "SUPERVISOR"):
            rp_bad = _write_attempt_output(
                tmp_path / f"poisoned-{idx}-{f_cls}",
                shard=0,
                mode="compute_transport",
                failure_class=f_cls,
                error=bad_err,
            )
            assert (
                admission.is_external_transport_compute_failure(
                    json.loads(rp_bad.read_text()), rp_bad.parent
                )
                is False
            )

    # Clean receipt traceback BUT poll-000000.json job_log_tail contains child traceback/fault:
    rp_poll_fault = _write_attempt_output(
        tmp_path / "poll-log-scientific-fault",
        shard=0,
        mode="compute_transport",
        job_log_tail=(
            "Traceback (most recent call last):\n"
            '  File "research/watch6_search_efficacy/engine.py", line 112, in step\n'
            "ValueError: mirror mismatch\n"
        ),
    )
    assert (
        admission.is_external_transport_compute_failure(
            json.loads(rp_poll_fault.read_text()), rp_poll_fault.parent
        )
        is False
    )


def test_v3_ruling_point2_reserves_removed_unlimited_prestart_and_capped_compute(
    tmp_path: Path,
) -> None:
    assert not hasattr(p, "reserve_seed")
    assert not hasattr(p, "reserve_allocation")
    assert not hasattr(p, "MAX_RESERVE_SUBSTITUTIONS")

    primary = {r["seed"] for r in p.allocation()}
    fixtures = set(p.FIXTURE_SEEDS)
    assert len(primary) == p.COUNT == 96
    assert len(primary | fixtures) == 99
    assert not (primary | fixtures) & {r["seed"] for r in v2_p.allocation()}
    for ns, count in freshness.EXPLICIT_DISJOINT_NAMESPACES:
        assert not (primary | fixtures) & freshness._derive_namespace_seeds(ns, count)

    # V3 decide matches V2 decide bit-for-bit on 96 primary rows and refuses non-primary seed_kind:
    v2_rows = [
        {
            "namespace": v2_p.NAMESPACE,
            "science": v2_p.identity(),
            "allocation": alloc,
            "fixture": False,
            "unit_id": unit_id(v2_p.NAMESPACE, "watch6", idx),
            "arms": r["arms"],
        }
        for idx, (alloc, r) in enumerate(zip(v2_p.allocation(), _synthetic_v3_rows()))
    ]
    v2_res = v2_analyze.decide(v2_rows)
    v3_res = analyze.decide_v3(_synthetic_v3_rows())
    assert v3_res["verdict"] == v2_res["verdict"] == "GO_EFFICACY_ONLY"
    assert v3_res["death_delta_ci95"] == pytest.approx(v2_res["death_delta_ci95"])
    assert v3_res["mass_relative_ci95_fieller"] == pytest.approx(
        v2_res["mass_relative_ci95_fieller"]
    )
    assert v3_res["primary_pairs_count"] == 96
    assert v3_res["reserve_substitutions_count"] == 0

    bad_rows = _synthetic_v3_rows()
    bad_rows[5]["seed_kind"] = "reserve"
    with pytest.raises(ValueError, match="seed_kind"):
        analyze.decide_v3(bad_rows)

    base_pkg = tmp_path / "package-v3-01"
    base_pkg.mkdir()
    dummy_in = base_pkg / "dummy-input.bin"
    dummy_in.write_bytes(b"ok")
    (base_pkg / "canary.json").write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "path": str(dummy_in),
                        "dest": "input-metadata.tar.gz",
                        "sha256": digest(dummy_in),
                    }
                ]
            }
        )
    )
    (base_pkg / "output-canary").mkdir()
    (base_pkg / "output-canary/receipt.json").write_text(
        json.dumps({"status": "PASS", "stop_verified": True})
    )
    bench = tmp_path / "benchmark.json"
    bench.write_text(json.dumps({"science": p.identity()}))
    mir = tmp_path / "mac-mirror.json"
    mir.write_text(
        json.dumps(
            {
                "schema": "watch6-full-mirror/v3",
                "status": "PASS",
                "platform": "Darwin",
                "execution_identity": mirror.execution_identity(),
                "seed": p.FIXTURE_SEEDS[0],
                "horizon": p.HORIZON,
                "mirror_identity_frames": p.HORIZON,
                "mirror_mismatches": 0,
                "respawns_per_slot": [3, 3, 3, 3, 3, 3],
            }
        )
    )
    cost_path = tmp_path / "costs.json"
    cost_path.write_text(json.dumps(_synthetic_cost_report()))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=p.ROOT, text=True).strip()

    def fake_prepare(out: Path, *args, **kwargs) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        return {"shards": 24, "workers": 4, "base_minutes_per_shard": 209}

    # Unlimited verified pre-start retries (e.g. 8 pre-start retries > old cap of 6!):
    prestarts = [
        _write_attempt_output(tmp_path / f"prestart-{i}", shard=5, mode="prestart_staging")
        for i in range(8)
    ]
    with patch.object(recover, "prepare", side_effect=fake_prepare):
        plan_pre = recover.recover(
            base_package=base_pkg,
            receipts=prestarts,
            target_shards=[5],
            out=tmp_path / "rec-prestart-8",
            benchmark=bench,
            mirror_receipt=mir,
            empirical_costs=cost_path,
            commit=commit,
        )
        assert plan_pre["reused_initial_canary"] is True
        assert plan_pre["shard_meta"]["5"]["prestart_retries_used"] == 8
        assert plan_pre["shard_meta"]["5"]["compute_completion_attempt"] == 0
        assert plan_pre["shard_meta"]["5"]["total_dispatches_for_shard"] == 9

        # Partial-shard compute resumes: 1 and 2 failed compute attempts succeed; 3rd refuses!
        c1 = _write_attempt_output(
            tmp_path / "c1", shard=5, mode="compute_transport", export_slots=[5, 29]
        )
        c2 = _write_attempt_output(tmp_path / "c2", shard=5, mode="compute_transport")
        plan_c2 = recover.recover(
            base_package=base_pkg,
            receipts=[*prestarts, c1, c2],
            target_shards=[5],
            out=tmp_path / "rec-compute-2",
            benchmark=bench,
            mirror_receipt=mir,
            empirical_costs=cost_path,
            commit=commit,
        )
        assert plan_c2["verified_units_in_ledger"] == 2
        assert plan_c2["shard_meta"]["5"]["prestart_retries_used"] == 8
        assert plan_c2["shard_meta"]["5"]["compute_completion_attempt"] == 2

        c3 = _write_attempt_output(tmp_path / "c3", shard=5, mode="compute_transport")
        with pytest.raises(ValueError, match="exhausted compute completion cap"):
            recover.recover(
                base_package=base_pkg,
                receipts=[*prestarts, c1, c2, c3],
                target_shards=[5],
                out=tmp_path / "rec-compute-3-refused",
                benchmark=bench,
                mirror_receipt=mir,
                empirical_costs=cost_path,
                commit=commit,
            )


def test_v3_ruling_point3_staging_serialization_fails_closed_on_expiry_and_stale_manifests(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    ops_dir = tmp_path / "ops"
    (ops_dir / "manifests").mkdir(parents=True)
    (ops_dir / "logs").mkdir(parents=True)
    ruling_json = _write_approved_ruling(tmp_path)

    op_root = tmp_path / "operator"
    (op_root / "tools/ops").mkdir(parents=True)
    fake_dispatch = op_root / "tools/ops/dispatch.sh"
    called_marker = tmp_path / "dispatch-was-called.txt"
    fake_dispatch.write_text(
        f"#!/usr/bin/env bash\necho CALLED >> '{called_marker}'\nexit 0\n",
        encoding="utf-8",
    )
    fake_dispatch.chmod(0o755)

    # 1. Stale manifest (>600s old, backdated 2 hours!) MUST NOT be silently bypassed!
    stale_out = tmp_path / "stale-staging-out"
    stale_out.mkdir()
    stale_mf = ops_dir / "manifests/stale-job.json"
    stale_mf.write_text(json.dumps({"output_dir": str(stale_out)}), encoding="utf-8")
    two_hours_ago = time.time() - 7200.0
    os.utime(stale_mf, (two_hours_ago, two_hours_ago))
    assert dispatch.job_is_actively_staging(stale_mf, ops_dir) is True

    # 2. wait_other_jobs_staged expiry MUST fail CLOSED (never invoke tools/ops/dispatch.sh)
    #    and write durable watch6-staging-reconcile.json!
    m_wait = tmp_path / "m-wait.json"
    m_wait.write_text(json.dumps({"job_id": "j-wait", "output_dir": str(tmp_path / "out-wait")}))
    with patch.object(dispatch, "admit", return_value={"status": "PASS"}):
        with pytest.raises(RuntimeError, match="REFUSE: staging serialization wait expired"):
            dispatch.run_dispatch(
                [
                    "job-wait",
                    str(m_wait),
                    "--broker",
                    str(p.ROOT),
                    "--cwd",
                    str(p.ROOT),
                    "--operator-root",
                    str(op_root),
                    "--ruling-json",
                    str(ruling_json),
                ],
                ops_dir=ops_dir,
                staging_timeout=0.08,
                poll_seconds=0.02,
            )
    assert not called_marker.exists()
    assert (ops_dir / dispatch.RECONCILE_MARKER_NAME).is_file()

    # Reconcile refuses while stale-job is still unfinished!
    with pytest.raises(RuntimeError, match="REFUSE: cannot clear staging reconciliation fence"):
        dispatch.reconcile_staging(ops_dir)
    assert (ops_dir / dispatch.RECONCILE_MARKER_NAME).is_file()

    # Exited log with existing output_dir lacking upload-measurement/stopped receipt fails closed!
    (ops_dir / "logs/stale-job.log").write_text("EXIT=1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="REFUSE: exited ops manifest stale-job.json"):
        dispatch.reconcile_staging(ops_dir)

    # Once stale-job has a stop-verified receipt, reconcile_staging succeeds and clears fence:
    (stale_out / "receipt.json").write_text(
        json.dumps({"stop_verified": True, "cleanup_error": None}), encoding="utf-8"
    )
    rec_res = dispatch.reconcile_staging(ops_dir)
    assert rec_res["status"] == "RECONCILED"
    assert not (ops_dir / dispatch.RECONCILE_MARKER_NAME).exists()

    # 3. wait_own_job_staged expiry MUST fail CLOSED and fence any subsequent dispatch!
    py_extract = "import json,sys; print(json.load(open(sys.argv[1]))['output_dir'])"
    fake_dispatch.write_text(
        "#!/usr/bin/env bash\n"
        'NAME="$1"; MANIFEST="$2"\n'
        f'OUT=$(python3 -c "{py_extract}" "$MANIFEST")\n'
        'mkdir -p "$OUT"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    m_own = tmp_path / "m-own.json"
    m_own.write_text(json.dumps({"job_id": "j-own", "output_dir": str(tmp_path / "out-own")}))
    with patch.object(dispatch, "admit", return_value={"status": "PASS"}):
        with pytest.raises(RuntimeError, match="REFUSE: own staging wait expired"):
            dispatch.run_dispatch(
                [
                    "job-own",
                    str(m_own),
                    "--broker",
                    str(p.ROOT),
                    "--cwd",
                    str(p.ROOT),
                    "--operator-root",
                    str(op_root),
                    "--ruling-json",
                    str(ruling_json),
                ],
                ops_dir=ops_dir,
                staging_timeout=0.08,
                poll_seconds=0.02,
            )
    assert (ops_dir / dispatch.RECONCILE_MARKER_NAME).is_file()

    # Subsequent dispatch immediately refuses due to the reconciliation fence!
    m_next = tmp_path / "m-next.json"
    m_next.write_text(json.dumps({"job_id": "j-next", "output_dir": str(tmp_path / "out-next")}))
    with patch.object(dispatch, "admit", return_value={"status": "PASS"}):
        with pytest.raises(RuntimeError, match="REFUSE: unreconciled staging serialization fence"):
            dispatch.run_dispatch(
                [
                    "job-next",
                    str(m_next),
                    "--broker",
                    str(p.ROOT),
                    "--cwd",
                    str(p.ROOT),
                    "--operator-root",
                    str(op_root),
                    "--ruling-json",
                    str(ruling_json),
                ],
                ops_dir=ops_dir,
                staging_timeout=0.2,
                poll_seconds=0.02,
            )

    # Clear own-job staging and reconcile; then verify concurrent serialization & cap-6 passthrough:
    (tmp_path / "out-own/upload-measurement.json").write_text('{"elapsed_seconds": 50.0}\n')
    dispatch.reconcile_staging(ops_dir)

    events_file = tmp_path / "staging-events.txt"
    fake_dispatch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'NAME="$1"; MANIFEST="$2"\n'
        'OUT=$(python3 -c "import json,sys; '
        'print(json.load(open(sys.argv[1]))[\'output_dir\'])" "$MANIFEST")\n'
        'mkdir -p "$OUT"\n'
        f'echo "START:$NAME" >> "{events_file}"\n'
        '( sleep 0.2; echo "STAGED:$NAME" >> "'
        + str(events_file)
        + '"; echo "{}" > "$OUT/upload-measurement.json" ) &\n',
        encoding="utf-8",
    )
    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    m1.write_text(json.dumps({"job_id": "j1", "output_dir": str(tmp_path / "out1")}))
    m2.write_text(json.dumps({"job_id": "j2", "output_dir": str(tmp_path / "out2")}))
    results: list[int] = []

    def launch(name: str, mf: Path) -> None:
        results.append(
            dispatch.run_dispatch(
                [
                    name,
                    str(mf),
                    "--broker",
                    str(p.ROOT),
                    "--cwd",
                    str(p.ROOT),
                    "--operator-root",
                    str(op_root),
                    "--ruling-json",
                    str(ruling_json),
                ],
                ops_dir=ops_dir,
                poll_seconds=0.03,
            )
        )

    with patch.object(dispatch, "admit", return_value={"status": "PASS"}):
        t1 = threading.Thread(target=launch, args=("job-a", m1))
        t2 = threading.Thread(target=launch, args=("job-b", m2))
        t1.start()
        time.sleep(0.04)
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
    assert results == [0, 0]
    assert events_file.read_text().splitlines() == [
        "START:job-a",
        "STAGED:job-a",
        "START:job-b",
        "STAGED:job-b",
    ]

    fake_dispatch.write_text(
        "#!/usr/bin/env bash\n"
        'echo "REFUSE: account-wide assignment cap 6 (including notebooks)"\n'
        "exit 2\n",
        encoding="utf-8",
    )
    capfd.readouterr()
    m3 = tmp_path / "m3.json"
    m3.write_text(json.dumps({"job_id": "j3", "output_dir": str(tmp_path / "out3")}))
    with patch.object(dispatch, "admit", return_value={"status": "PASS"}):
        rc = dispatch.run_dispatch(
            [
                "job-c",
                str(m3),
                "--broker",
                str(p.ROOT),
                "--cwd",
                str(p.ROOT),
                "--operator-root",
                str(op_root),
                "--ruling-json",
                str(ruling_json),
            ],
            ops_dir=ops_dir,
            poll_seconds=0.03,
        )
    captured = capfd.readouterr()
    assert rc == 2
    assert (captured.out + captured.err).strip() == (
        "REFUSE: account-wide assignment cap 6 (including notebooks)"
    )


def _setup_full_24_shard_ops(tmp_path: Path) -> tuple[Path, Path, dict[int, dict]]:
    """Build a complete valid 24-shard ops tree (including pre-start & partial-resume chains)."""
    ops_dir = tmp_path / "ops"
    (ops_dir / "manifests").mkdir(parents=True)
    (ops_dir / "logs").mkdir(parents=True)
    ruling_json = _write_approved_ruling(tmp_path)

    # Canary: base output had pre-allocation ACCOUNTING refusal, -retry-1 passed!
    canary_base_out = tmp_path / "output-canary"
    canary_base_out.mkdir(parents=True)
    canary_mf = {
        "job_id": "watch6-v3-canary",
        "output_dir": str(canary_base_out),
        "commit": "c" * 40,
        "canary": {"schema": canary.SCHEMA, "mode": "probe", "family": p.SCRIPT},
        "env": {"WATCH6_FIXTURE": "1"},
        "watch6_v3": {
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "primary_allocation": p.allocation(),
        },
    }
    (canary_base_out / "manifest.json").write_text(json.dumps(canary_mf, sort_keys=True) + "\n")
    (canary_base_out / "receipt.json").write_text(
        json.dumps(
            {
                "job_id": "watch6-v3-canary",
                "commit": "c" * 40,
                "manifest_sha256": digest(canary_base_out / "manifest.json"),
                "endpoint": None,
                "status": "INCOMPLETE",
                "failure_class": "ACCOUNTING",
                "error": "AccountingError('cap 6')",
                "stop_verified": False,
                "cleanup_error": None,
                "terminal": None,
                "exports": [],
                "transfers": [],
                "contents_transfers": [],
                "cli_calls": [],
                "control_operations": [],
            }
        )
        + "\n"
    )
    canary_retry_out = Path(str(canary_base_out) + "-retry-1")
    canary_retry_out.mkdir(parents=True)
    canary_retry_mf = {**canary_mf, "output_dir": str(canary_retry_out)}
    (canary_retry_out / "manifest.json").write_text(
        json.dumps(canary_retry_mf, sort_keys=True) + "\n"
    )
    (canary_retry_out / "receipt.json").write_text(
        json.dumps(
            {
                "job_id": "watch6-v3-canary",
                "commit": "c" * 40,
                "manifest_sha256": digest(canary_retry_out / "manifest.json"),
                "endpoint": "m-s-canary",
                "status": "PASS",
                "failure_class": None,
                "error": None,
                "stop_verified": True,
                "cleanup_error": None,
                "terminal": {"returncode": 0, "reason": None, "error": None},
                "exports": [],
            }
        )
        + "\n"
    )
    (ops_dir / "manifests/watch6-v3-canary.json").write_text(json.dumps(canary_mf) + "\n")

    # Proven unallocated pre-dispatch refusal (exited log + no output dir):
    (ops_dir / "logs/watch6-v3-unalloc.log").write_text("EXIT=2\n")
    (ops_dir / "manifests/watch6-v3-unalloc.json").write_text(
        json.dumps(
            {
                "job_id": "unalloc",
                "output_dir": str(tmp_path / "never-created-dir"),
                "env": {"WATCH6_FIXTURE": "0"},
                "watch6_v3": {"namespace": p.NAMESPACE, "science": p.identity()},
            }
        )
    )

    all_rows = {r["allocation"]["index"]: r for r in _synthetic_v3_rows()}
    for s in range(p.SHARDS):
        if s == 5:
            continue
        r_path = _write_attempt_output(
            tmp_path / f"out-shard-{s:02d}",
            shard=s,
            mode="pass",
            export_slots=p.shard_indices(s),
            pair_rows_by_slot=all_rows,
        )
        (ops_dir / f"manifests/shard-{s:02d}.json").write_text(
            (r_path.parent / "manifest.json").read_text()
        )

    # Shard 5: 1 verified pre-start failure + 2 verified external-transport compute failures
    # (exporting [5] and [29]) + 1 passing completion exporting [53, 77]!
    s5_a1 = _write_attempt_output(tmp_path / "out-shard-05-a1", shard=5, mode="prestart_staging")
    p1 = [
        {"receipt_path": str(s5_a1), "receipt_sha256": digest(s5_a1), "classification": "prestart"}
    ]
    s5_a2 = _write_attempt_output(
        tmp_path / "out-shard-05-a2",
        shard=5,
        mode="compute_transport",
        export_slots=[5],
        pair_rows_by_slot=all_rows,
        prestart_retries_used=1,
        compute_completion_attempt=0,
        prior_receipts=p1,
    )
    p2 = [
        *p1,
        {
            "receipt_path": str(s5_a2),
            "receipt_sha256": digest(s5_a2),
            "classification": "external_transport_compute",
        },
    ]
    s5_a3 = _write_attempt_output(
        tmp_path / "out-shard-05-a3",
        shard=5,
        mode="compute_transport",
        export_slots=[29],
        pair_rows_by_slot=all_rows,
        prestart_retries_used=1,
        compute_completion_attempt=1,
        prior_receipts=p2,
    )
    p3 = [
        *p2,
        {
            "receipt_path": str(s5_a3),
            "receipt_sha256": digest(s5_a3),
            "classification": "external_transport_compute",
        },
    ]
    s5_a4 = _write_attempt_output(
        tmp_path / "out-shard-05-a4",
        shard=5,
        mode="pass",
        export_slots=[53, 77],
        pair_rows_by_slot=all_rows,
        prestart_retries_used=1,
        compute_completion_attempt=2,
        prior_receipts=p3,
    )
    for idx, rp in enumerate((s5_a1, s5_a2, s5_a3, s5_a4), 1):
        (ops_dir / f"manifests/shard-05-a{idx}.json").write_text(
            (rp.parent / "manifest.json").read_text()
        )
    return ops_dir, ruling_json, all_rows


def test_v3_ruling_point4_final_look_pre_outcome_lineage_failure_stop_and_hash_checks(
    tmp_path: Path,
) -> None:
    # 1. Valid complete 24-shard run passes all pre-outcome checks and takes the single look:
    ops_ok, ruling_ok, _ = _setup_full_24_shard_ops(tmp_path / "valid")
    decision = final_look.run_final_look(
        ops_dir=ops_ok, out_dir=tmp_path / "valid/look-out", ruling_json=ruling_ok
    )
    assert decision["verdict"] == "GO_EFFICACY_ONLY"
    assert decision["n"] == 96
    assert decision["primary_pairs_count"] == 96
    assert decision["reserve_substitutions_count"] == 0
    verify_immutable(tmp_path / "valid/look-out/final-verified-ledger.json")
    verify_immutable(tmp_path / "valid/look-out/efficacy-decision.json")

    # 2. Forbidden failure class on an earlier attempt (even though all 96 pairs exist!)
    #    MUST refuse BEFORE opening any pair-XX.json outcome!
    ops_forbid, ruling_forbid, _ = _setup_full_24_shard_ops(tmp_path / "forbid")
    a1_rec_path = tmp_path / "forbid/out-shard-05-a1/receipt.json"
    a1_rec = json.loads(a1_rec_path.read_text())
    a1_rec["failure_class"] = "CHILD_NONZERO_EXIT"
    a1_rec["error"] = "ValueError: mirror mismatch"
    a1_rec_path.write_text(json.dumps(a1_rec, sort_keys=True) + "\n")
    with patch.object(
        final_look, "analyze_verified_union", side_effect=AssertionError("OUTCOME OPENED!")
    ):
        with pytest.raises(ValueError, match="forbidden failure class"):
            final_look.run_final_look(
                ops_dir=ops_forbid,
                out_dir=tmp_path / "forbid/look-out",
                ruling_json=ruling_forbid,
            )

    # 3. Broken retry-cap lineage (deleted intermediate attempt manifest) MUST refuse pre-outcome!
    ops_gap, ruling_gap, _ = _setup_full_24_shard_ops(tmp_path / "gap")
    (ops_gap / "manifests/shard-05-a2.json").unlink()
    with patch.object(
        final_look, "analyze_verified_union", side_effect=AssertionError("OUTCOME OPENED!")
    ):
        with pytest.raises(ValueError, match="missing or mismatched prior receipt"):
            final_look.run_final_look(
                ops_dir=ops_gap, out_dir=tmp_path / "gap/look-out", ruling_json=ruling_gap
            )

    # 4. Exceeded compute completion cap (3 failed compute attempts via auto-retry + recover)
    #    MUST refuse pre-outcome!
    ops_cap, ruling_cap, rows_cap = _setup_full_24_shard_ops(tmp_path / "cap")
    # Add an extra auto-retry failed compute attempt between a3 and a4:
    auto_dir = tmp_path / "cap/auto-retry-s5"
    auto_out = auto_dir / "output"
    s5_a3_path = (tmp_path / "cap/out-shard-05-a3/receipt.json").resolve()
    s5_auto = _write_attempt_output(
        auto_out,
        shard=5,
        mode="compute_transport",
        prestart_retries_used=1,
        compute_completion_attempt=1,
        prior_receipts=json.loads((tmp_path / "cap/out-shard-05-a3/manifest.json").read_text())[
            "watch6_v3"
        ]["prior_receipts"],
        completion_attempt=1,
    )
    publish_json(auto_dir / "manifest.json", json.loads((auto_out / "manifest.json").read_text()))
    publish_json(
        auto_dir / "RETRY.json",
        {
            "schema": "ops-zero-export-retry/v1",
            "completion_attempt": 1,
            "max_completion_attempts": 2,
            "prior_manifest": {
                "path": str(ops_cap / "manifests/shard-05-a3.json"),
                "sha256": digest(ops_cap / "manifests/shard-05-a3.json"),
            },
            "prior_receipt": {"path": str(s5_a3_path), "sha256": digest(s5_a3_path)},
            "manifest_sha256": digest(auto_dir / "manifest.json"),
            "unallocated_proof": None,
        },
    )
    (ops_cap / "manifests/shard-05-auto.json").write_text((auto_dir / "manifest.json").read_text())
    # Re-point a4 as an auto-retry of s5_auto so chain is connected with 3 failed compute attempts:
    auto2_dir = tmp_path / "cap/auto-retry-s5-final"
    auto2_out = auto2_dir / "output"
    _write_attempt_output(
        auto2_out,
        shard=5,
        mode="pass",
        export_slots=[53, 77],
        pair_rows_by_slot=rows_cap,
        prestart_retries_used=1,
        compute_completion_attempt=1,
        prior_receipts=json.loads((auto_out / "manifest.json").read_text())["watch6_v3"][
            "prior_receipts"
        ],
        completion_attempt=2,
    )
    publish_json(auto2_dir / "manifest.json", json.loads((auto2_out / "manifest.json").read_text()))
    publish_json(
        auto2_dir / "RETRY.json",
        {
            "schema": "ops-zero-export-retry/v1",
            "completion_attempt": 2,
            "max_completion_attempts": 2,
            "prior_manifest": {
                "path": str(auto_dir / "manifest.json"),
                "sha256": digest(auto_dir / "manifest.json"),
            },
            "prior_receipt": {"path": str(s5_auto.resolve()), "sha256": digest(s5_auto)},
            "manifest_sha256": digest(auto2_dir / "manifest.json"),
            "unallocated_proof": None,
        },
    )
    (ops_cap / "manifests/shard-05-a4.json").write_text((auto2_dir / "manifest.json").read_text())
    with patch.object(
        final_look, "analyze_verified_union", side_effect=AssertionError("OUTCOME OPENED!")
    ):
        with pytest.raises(ValueError, match="exceeded compute completion retry cap"):
            final_look.run_final_look(
                ops_dir=ops_cap, out_dir=tmp_path / "cap/look-out", ruling_json=ruling_cap
            )

    # 5. Unstopped runtime on base dir hidden behind -retry-1 MUST refuse pre-outcome!
    ops_unstop, ruling_unstop, _ = _setup_full_24_shard_ops(tmp_path / "unstop")
    canary_base_rec = tmp_path / "unstop/output-canary/receipt.json"
    c_rec = json.loads(canary_base_rec.read_text())
    c_rec["endpoint"] = "leaked-endpoint"
    c_rec["stop_verified"] = False
    canary_base_rec.write_text(json.dumps(c_rec) + "\n")
    with patch.object(
        final_look, "analyze_verified_union", side_effect=AssertionError("OUTCOME OPENED!")
    ):
        with pytest.raises(ValueError, match="unstopped or unclean"):
            final_look.run_final_look(
                ops_dir=ops_unstop,
                out_dir=tmp_path / "unstop/look-out",
                ruling_json=ruling_unstop,
            )

    # 6. Hash tampering on pair sidecar MUST refuse pre-outcome!
    ops_hash, ruling_hash, _ = _setup_full_24_shard_ops(tmp_path / "hash")
    sha_file = tmp_path / "hash/out-shard-00/exports/efficacy/pair-00.json.sha256"
    os.chmod(sha_file, 0o644)
    sha_file.write_text("0" * 64 + "\n")
    with patch.object(
        final_look, "analyze_verified_union", side_effect=AssertionError("OUTCOME OPENED!")
    ):
        with pytest.raises(ValueError):
            final_look.run_final_look(
                ops_dir=ops_hash, out_dir=tmp_path / "hash/look-out", ruling_json=ruling_hash
            )


def test_v3_ruling_point5_ratify_parses_explicit_verdict_and_refuses_reject(
    tmp_path: Path,
) -> None:
    # 1. Exact Astra rejection format (and real RULING_V3.md on disk if present) MUST be refused!
    reject_md = tmp_path / "RULING_REJECT.md"
    reject_md.write_text(
        "# Binding PRE-DATA ruling — Watch6 V3\n\n"
        "Verdict: REJECT\n"
        "STATUS: NOT_RATIFIED\n"
        "Judge: Astra · 2026-10-31\n",
        encoding="utf-8",
    )
    out_reject = tmp_path / "reject-ruling.json"
    with pytest.raises(ValueError, match="REFUSE"):
        ratify.record_ruling(reject_md, out_reject)
    assert not out_reject.exists()

    real_rejection = Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/watch6-v3-ruling/RULING_V3.md"
    )
    if real_rejection.is_file():
        with pytest.raises(ValueError, match="REFUSE"):
            ratify.record_ruling(real_rejection, tmp_path / "real-reject.json")
        assert not (tmp_path / "real-reject.json").exists()

    # 2. Arbitrary nonempty Markdown without explicit Verdict line, or NOT_RATIFIED, refused!
    for idx, bad_text in enumerate(
        [
            "# RATIFIED BY ASTRA PRE-DATA\n",
            "Verdict: APPROVE\nSTATUS: NOT_RATIFIED\n",
            "Verdict: NOT_RATIFIED\n",
            "Verdict: APPROVE\nVerdict: REJECT\n",
        ]
    ):
        md = tmp_path / f"bad-{idx}.md"
        md.write_text(bad_text, encoding="utf-8")
        with pytest.raises(ValueError, match="REFUSE"):
            ratify.record_ruling(md, tmp_path / f"bad-{idx}.json")

    # 3. Explicit APPROVE and APPROVE-WITH-CONDITIONS succeed and re-verify on validate:
    for idx, good_verdict in enumerate(("APPROVE", "APPROVE-WITH-CONDITIONS")):
        md = tmp_path / f"good-{idx}.md"
        md.write_text(
            f"# Binding PRE-DATA ruling\n\nVerdict: {good_verdict}\nSTATUS: RATIFIED\n",
            encoding="utf-8",
        )
        out = tmp_path / f"good-{idx}.json"
        rec = ratify.record_ruling(md, out)
        assert rec["verdict"] == good_verdict
        assert ratify.validate_ruling_receipt(out)["verdict"] == good_verdict

    # 4. Forged astra-ruling.json pointing to a REJECT markdown is refused on validate:
    forged_json = tmp_path / "forged.json"
    forged_json.write_text(
        json.dumps(
            {
                "schema": ratify.SCHEMA,
                "status": "RATIFIED",
                "verdict": "APPROVE",
                "judge": "Astra",
                "implementer": "Argon",
                "namespace": p.NAMESPACE,
                "science": p.identity(),
                "registration_path": str(p.registration()),
                "registration_sha256": digest(p.registration()),
                "ruling_path": str(reject_md),
                "ruling_sha256": digest(reject_md),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="REFUSE"):
        ratify.validate_ruling_receipt(forged_json)


def test_v3_upload_budget_at_most_12_transfers_and_90_seconds(tmp_path: Path) -> None:
    report = _synthetic_cost_report()
    contract = tmp_path / "runtime-contract.json"
    publish_json(
        contract,
        {
            "schema": "watch6-v3-runtime-contract/v2",
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "primary_allocation": p.allocation(),
        },
    )
    ledger = tmp_path / "verified-ledger.json"
    publish_json(ledger, empty_ledger(p.NAMESPACE, "watch6", run.ledger_identity()))
    approval = tmp_path / "judge-approval.json"
    publish_json(
        approval,
        {
            "schema": "watch6-v3-protocol-binding/v2",
            "implementer": "Argon",
            "judge": "Astra",
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "registration_sha256": digest(p.registration()),
        },
    )
    mir = tmp_path / "mac-mirror.json"
    publish_json(
        mir,
        {
            "schema": "watch6-full-mirror/v3",
            "status": "PASS",
            "platform": "Darwin",
            "execution_identity": mirror.execution_identity(),
            "seed": p.FIXTURE_SEEDS[0],
            "horizon": p.HORIZON,
            "mirror_identity_frames": p.HORIZON,
            "mirror_mismatches": 0,
            "respawns_per_slot": [3, 4, 2, 1, 2, 3],
        },
    )
    cost_ev = tmp_path / "cost-evidence.json"
    publish_json(cost_ev, report)
    rel_bundle = tmp_path / "release-evidence.tar.gz"
    with (
        rel_bundle.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w") as tf,
    ):
        data = b"{}"
        ti = tarfile.TarInfo("b8-s3/receipt.json")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))

    saved = p.ROOT.parent / "snake-dqn/saved_snakes"
    inputs = [
        Input(contract, "runtime-contract.json"),
        Input(ledger, "verified-ledger.json"),
        Input(approval, "judge-approval.json"),
        Input(mir, "mac-mirror.json"),
        Input(cost_ev, "cost-evidence.json"),
        Input(rel_bundle, "release-evidence.tar.gz"),
        *uploads.checkpoint_inputs(saved, tmp_path / "checkpoint-parts"),
    ]
    packed = uploads.pack_inputs(
        stage_inputs(tmp_path / "staged", inputs), tmp_path / uploads.METADATA
    )
    assert len(packed) == 9  # 8 x 2 MiB checkpoint parts + 1 input-metadata.tar.gz

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=p.ROOT, text=True).strip()
    manifest = {
        "commit": commit,
        "script": p.SCRIPT,
        "source_paths": p.source_paths(),
        "max_export_bytes": 4 * 1024**2,
        "timeout_minutes": 30,
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 12, "est_minutes": 15},
        "canary": {"schema": canary.SCHEMA, "mode": "probe", "family": p.SCRIPT},
        "inputs": [{k: r[k] for k in ("path", "dest", "sha256")} for r in packed],
    }
    upload_check = uploads.check_uploads(manifest)
    assert upload_check["status"] == "PASS"
    assert upload_check["normal_transfer_count"] == 11
    assert upload_check["transfer_count"] == 12 <= uploads.MAX_TRANSFERS
    assert upload_check["estimated_transfer_seconds"] <= 90.0

    # 13th transfer (> 12) must refuse offline!
    extra = tmp_path / "extra.bin"
    extra.write_bytes(b"x")
    bad_manifest = {
        **manifest,
        "inputs": [
            *manifest["inputs"],
            {"path": str(extra), "dest": "extra.bin", "sha256": digest(extra)},
        ],
    }
    with pytest.raises(ValueError, match="upload transfer forecast refuses offline"):
        uploads.check_uploads(bad_manifest)


def test_v3_finalize_and_real_entrypoint_fixture(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    canary_receipt = pkg / "output-canary/receipt.json"
    canary_receipt.parent.mkdir()
    canary_receipt.write_text('{"status": "PASS"}\n')
    for s in range(p.SHARDS):
        manifest = {
            "job_id": f"watch6-v3-shard-{s:02d}",
            "output_dir": str(pkg / f"output-shard-{s:02d}"),
            "pending_qualification": True,
            "canary": {
                "schema": canary.SCHEMA,
                "receipt": {"path": str(canary_receipt), "sha256": "0" * 64},
            },
            "env": {"WATCH6_SHARD": str(s), "WATCH6_SHARDS": str(p.SHARDS)},
        }
        (pkg / f"shard-{s:02d}.pending.json").write_text(json.dumps(manifest))

    with patch.object(finalize, "admit", return_value={"status": "PASS"}):
        sealed = finalize.finalize(pkg, shard=5, idempotent=True)
        assert sealed == [pkg / "shard-05.json"]
        assert finalize.finalize(pkg, shard=5, idempotent=True) == [pkg / "shard-05.json"]

    ledger = tmp_path / "verified-ledger.json"
    publish_json(ledger, empty_ledger(p.NAMESPACE, "watch6", run.ledger_identity()))
    env = {
        **os.environ,
        "SNAKE_DQN_DEVICE": "cpu",
        "SNAKE_WATCH_SOURCE": "live",
        "SNAKE_CKPT_DIR": str(p.ROOT.parent / "snake-dqn/saved_snakes"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.watch6_search_efficacy_v3.run",
            "--ledger",
            str(ledger),
            "--out",
            str(tmp_path / "fixture-out"),
            "--workers",
            "1",
            "--fixture",
            "--fixture-horizon",
            "4",
        ],
        cwd=p.ROOT,
        env=env,
        check=True,
    )
    for idx in range(3):
        verify_immutable(tmp_path / "fixture-out" / f"fixture-pair-{idx:02d}.json")
        verify_immutable(tmp_path / "fixture-out" / f"fixture-pair-{idx:02d}.meta.json")
    assert list((tmp_path / "fixture-out").glob("pair-*.json")) == []


def _write_unallocated_503_attempt(
    ops_dir: Path,
    *,
    name: str,
    out_dir: Path,
    gpu: str = "H100",
    prefer: list[str] | None = None,
    write_unavailable_proof: bool = True,
    refusal_kind: str = "503",
    custom_error: str | None = None,
    endpoint: str | None = None,
    stop_verified: bool = False,
    cleanup_error: str | None = "ALLOCATION_UNRESOLVED: retained pending record",
) -> tuple[Path, Path]:
    """Create a minimal operational fixture matching real H100/A100 colab new 503 receipts."""
    (ops_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (ops_dir / "logs").mkdir(parents=True, exist_ok=True)
    (ops_dir / "dispatch" / name).mkdir(parents=True, exist_ok=True)
    (ops_dir / "alloc.lock").touch()
    out_dir.mkdir(parents=True, exist_ok=True)

    req: dict[str, object] = {"gpu": gpu, "allowed_gpus": [gpu], "est_minutes": 20}
    if prefer is not None:
        req["prefer"] = list(prefer)
    manifest = {
        "job_id": name,
        "commit": "d" * 40,
        "output_dir": str(out_dir),
        "requirements": req,
    }
    ops_mf = ops_dir / "manifests" / f"{name}.json"
    ops_mf.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    exec_mf = out_dir / "manifest.json"
    exec_mf.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    slug = "".join(ch if ch.isalnum() or ch in "_-" else "-" for ch in name.lower())
    session = f"snake-{slug}-0123456789ab"
    if custom_error is not None:
        err = custom_error
    elif refusal_kind == "precondition":
        detail = (
            f"colab new failed (1): [colab] Creating session '{session}'...\n"
            f"{dispatch.ALLOCATION_UNAVAILABLE}\n"
        )
        err = f"Refused({detail!r})"
    else:
        detail = (
            "colab new failed (1): Traceback (most recent call last):\n"
            '  File "/opt/colab/cli/assign.py", line 88, in create_session\n'
            "ColabRequestError: Failed to issue request POST "
            "https://colab.research.google.com/tun/m/assign"
            f"?nbh=4db4240f_c63c_4ea0_9e8c_e72bdd05bf09........&variant=GPU&accelerator={gpu}: "
            "Service Unavailable\n"
        )
        err = f"Refused({detail!r})"

    receipt = {
        "job_id": name,
        "commit": "d" * 40,
        "session": session,
        "manifest_sha256": digest(exec_mf),
        "status": "INCOMPLETE",
        "failure_class": "SUPERVISOR",
        "endpoint": endpoint,
        "accounting": {"allocated_endpoint": endpoint},
        "terminal": None,
        "stop_verified": stop_verified,
        "cleanup_error": cleanup_error,
        "error": err,
        "exports": [],
        "transfers": [],
        "contents_transfers": [],
        "control_operations": [],
        "cli_calls": [
            {"command": "version", "status": "completed"},
            {"command": "sessions", "status": "completed"},
            {"command": "usage", "status": "completed"},
            {"command": "new", "status": "failed"},
        ],
    }
    rec_path = out_dir / "receipt.json"
    rec_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    (ops_dir / "logs" / f"{name}.log").write_text(
        f"{err}\nEXIT=1\nOPS_ALLOCATION_FAILURE\n", encoding="utf-8"
    )

    if write_unavailable_proof:
        proof = {
            "schema": "colab-allocation-unavailable/v1",
            "classification": "ALLOCATION_UNAVAILABLE",
            "requested_accelerator": gpu,
            "mode": "LIVE",
            "endpoint_ever_assigned": False,
            "vm_compute_started": False,
            "recovery": {"endpoint": None},
            "receipt": {"path": str(rec_path), "sha256": digest(rec_path)},
        }
        (ops_dir / "dispatch" / name / "unavailable.json").write_text(
            json.dumps(proof, sort_keys=True) + "\n", encoding="utf-8"
        )
    return ops_mf, rec_path


def _write_fallback_sibling(
    ops_dir: Path,
    *,
    primary_name: str,
    primary_out_dir: Path,
    sibling_state: str,
) -> Path:
    """Create minimal -a100-fallback sibling matching real dispatch_preferred shapes.

    sibling_state:
    - "uploaded_and_stopped" (Shape A: upload-measurement.json + stop_verified receipt)
    - "stopped_pre_upload" (Shape B: stop_verified receipt, no upload-measurement.json)
    - "exited_pre_outdir" (Shape C: exited .log, no <out>-a100-fallback dir)
    - "aborted_pre_start" (Shape D/F: dispatch reservation + manifest, no .log, no out dir)
    - "in_flight_uploading" (Active upload: out dir + active .log without EXIT=, no upload/receipt)
    - "exited_unreconciled" (Broken sibling: exited .log + out dir without upload/stopped receipt)
    """
    fb_name = dispatch._fallback_sibling_name(primary_name)
    fb_out = Path(str(primary_out_dir).rstrip("/") + dispatch.FALLBACK_SUFFIX)
    (ops_dir / "dispatch" / fb_name).mkdir(parents=True, exist_ok=True)
    fb_manifest = {
        "job_id": fb_name,
        "commit": "d" * 40,
        "output_dir": str(fb_out),
        "requirements": {"gpu": "A100", "allowed_gpus": ["A100"], "est_minutes": 20},
    }
    fb_mf_path = ops_dir / "manifests" / f"{fb_name}.json"
    fb_mf_path.write_text(json.dumps(fb_manifest, sort_keys=True) + "\n", encoding="utf-8")

    if sibling_state == "uploaded_and_stopped":
        fb_out.mkdir(parents=True, exist_ok=True)
        (fb_out / "manifest.json").write_text(json.dumps(fb_manifest, sort_keys=True) + "\n")
        (fb_out / "upload-measurement.json").write_text('{"elapsed_seconds": 41.2}\n')
        (fb_out / "receipt.json").write_text(
            json.dumps(
                {
                    "job_id": fb_name,
                    "commit": "d" * 40,
                    "manifest_sha256": digest(fb_out / "manifest.json"),
                    "endpoint": "m-s-a100-ok",
                    "status": "PASS",
                    "stop_verified": True,
                    "cleanup_error": None,
                },
                sort_keys=True,
            )
            + "\n"
        )
        (ops_dir / "logs" / f"{fb_name}.log").write_text("EXIT=0\n", encoding="utf-8")
    elif sibling_state == "stopped_pre_upload":
        fb_out.mkdir(parents=True, exist_ok=True)
        (fb_out / "manifest.json").write_text(json.dumps(fb_manifest, sort_keys=True) + "\n")
        (fb_out / "receipt.json").write_text(
            json.dumps(
                {
                    "job_id": fb_name,
                    "commit": "d" * 40,
                    "manifest_sha256": digest(fb_out / "manifest.json"),
                    "endpoint": "m-s-a100-warmup-fail",
                    "status": "INCOMPLETE",
                    "failure_class": "SUPERVISOR",
                    "stop_verified": True,
                    "cleanup_error": None,
                },
                sort_keys=True,
            )
            + "\n"
        )
        (ops_dir / "logs" / f"{fb_name}.log").write_text("EXIT=1\n", encoding="utf-8")
    elif sibling_state == "exited_pre_outdir":
        (ops_dir / "logs" / f"{fb_name}.log").write_text(
            "preflight admission failed\nEXIT=1\n", encoding="utf-8"
        )
    elif sibling_state == "aborted_pre_start":
        (ops_dir / "logs" / f"{fb_name}.validate").write_text("VALIDATED\n", encoding="utf-8")
    elif sibling_state == "in_flight_uploading":
        fb_out.mkdir(parents=True, exist_ok=True)
        (fb_out / "rate-calibration.json").write_text('{"status": "PASS"}\n')
        (ops_dir / "logs" / f"{fb_name}.log").write_text(
            "OPS_ALLOCATION_SUCCESS\nstaging inputs...\n", encoding="utf-8"
        )
    elif sibling_state == "exited_unreconciled":
        fb_out.mkdir(parents=True, exist_ok=True)
        (ops_dir / "logs" / f"{fb_name}.log").write_text("EXIT=1\n", encoding="utf-8")
    else:
        raise ValueError(f"unknown sibling_state: {sibling_state}")
    return fb_mf_path


def test_v3_historical_22_manifest_shapes_pass_and_reconcile_clears_fence(
    tmp_path: Path,
) -> None:
    from research.compute import gpu_preference

    for sample_name in (
        "critic-rescore-20261029-01",
        "x" * 49,
        "x" * 50,
        "very-long-watch6-job-identifier-exceeding-sixty-four-chars-20261031-attempt-01",
    ):
        assert dispatch._fallback_sibling_name(sample_name) == gpu_preference.sibling_id(
            sample_name
        )

    ops_dir = tmp_path / "ops"
    pkgs = tmp_path / "packages"

    # Shape A (12/22 real manifests, including critic-rescore, doom-canary, spatial-ruler,
    # gpu-pref-acceptance-e, and ops/auto-retry/spatial-ruler-retry-*):
    mf_a1, _ = _write_unallocated_503_attempt(
        ops_dir,
        name="critic-rescore-20261029-01",
        out_dir=pkgs / "critic-rescore-20261029-attempt-01/output",
        prefer=["H100", "A100"],
        write_unavailable_proof=True,
    )
    fb_a1 = _write_fallback_sibling(
        ops_dir,
        primary_name="critic-rescore-20261029-01",
        primary_out_dir=pkgs / "critic-rescore-20261029-attempt-01/output",
        sibling_state="uploaded_and_stopped",
    )

    mf_a_auto, _ = _write_unallocated_503_attempt(
        ops_dir,
        name="spatial-ruler-retry-4d8ae3a7bc",
        out_dir=ops_dir / "auto-retry/spatial-ruler-retry-4d8ae3a7bc/output",
        prefer=["H100", "A100"],
        write_unavailable_proof=True,
    )
    fb_a_auto = _write_fallback_sibling(
        ops_dir,
        primary_name="spatial-ruler-retry-4d8ae3a7bc",
        primary_out_dir=ops_dir / "auto-retry/spatial-ruler-retry-4d8ae3a7bc/output",
        sibling_state="uploaded_and_stopped",
    )

    # Shape B (2/22 real manifests: gpu-pref-acceptance-20261028-b, sandbox-ct-h5000-20261029-02):
    mf_b, _ = _write_unallocated_503_attempt(
        ops_dir,
        name="gpu-pref-acceptance-20261028-b",
        out_dir=pkgs / "gpu-pref-b/output",
        prefer=["H100", "A100"],
        write_unavailable_proof=True,
    )
    fb_b = _write_fallback_sibling(
        ops_dir,
        primary_name="gpu-pref-acceptance-20261028-b",
        primary_out_dir=pkgs / "gpu-pref-b/output",
        sibling_state="stopped_pre_upload",
    )

    # Shape C (1/22 real manifest: sandbox-a-train01-20261028-s2-02):
    mf_c, _ = _write_unallocated_503_attempt(
        ops_dir,
        name="sandbox-a-train01-20261028-s2-02",
        out_dir=pkgs / "train01-s2-02/output",
        prefer=["H100", "A100"],
        write_unavailable_proof=True,
    )
    fb_c = _write_fallback_sibling(
        ops_dir,
        primary_name="sandbox-a-train01-20261028-s2-02",
        primary_out_dir=pkgs / "train01-s2-02/output",
        sibling_state="exited_pre_outdir",
    )

    # Shape D (2/22 real manifests: sandbox-ct-cf-20261028-14-r3, t34-spatial-amend07-02-ruler):
    mf_d, _ = _write_unallocated_503_attempt(
        ops_dir,
        name="t34-spatial-amend07-02-ruler",
        out_dir=pkgs / "spatial-amend07/ruler-output",
        prefer=["H100", "A100"],
        write_unavailable_proof=True,
    )
    fb_d = _write_fallback_sibling(
        ops_dir,
        primary_name="t34-spatial-amend07-02-ruler",
        primary_out_dir=pkgs / "spatial-amend07/ruler-output",
        sibling_state="aborted_pre_start",
    )

    # Shape E (1/22 real manifest: t34-doom-canary-transport01-h100 standalone H100 + unavailable):
    mf_e, _ = _write_unallocated_503_attempt(
        ops_dir,
        name="t34-doom-canary-transport01-h100",
        out_dir=pkgs / "doom-canary-transport01-h100/output",
        prefer=None,
        write_unavailable_proof=True,
    )

    # Shape F (4/22 real manifests: 3 H100-first without unavailable.json + unstarted fallback,
    # and 1 standalone A100 503 without unavailable.json):
    mf_f_pref, _ = _write_unallocated_503_attempt(
        ops_dir,
        name="t34-spatial-speed2-01-ruler",
        out_dir=pkgs / "spatial-speed2/ruler-output",
        prefer=["H100", "A100"],
        write_unavailable_proof=False,
        refusal_kind="503",
    )
    fb_f_pref = _write_fallback_sibling(
        ops_dir,
        primary_name="t34-spatial-speed2-01-ruler",
        primary_out_dir=pkgs / "spatial-speed2/ruler-output",
        sibling_state="aborted_pre_start",
    )
    mf_f_a100, _ = _write_unallocated_503_attempt(
        ops_dir,
        name="sandbox-ct-cf-20261028-14",
        out_dir=pkgs / "ct-cf-14/output",
        gpu="A100",
        prefer=None,
        write_unavailable_proof=False,
        refusal_kind="precondition",
    )

    # Standalone pre-start aborted single-job dispatch reservation (no .log, no output_dir):
    aborted_single = "watch6-source02-shard-6-retry-c355417c6b"
    (ops_dir / "dispatch" / aborted_single).mkdir(parents=True)
    (ops_dir / "logs" / f"{aborted_single}.validate").write_text("FAILED\n", encoding="utf-8")
    mf_aborted_single = ops_dir / "manifests" / f"{aborted_single}.json"
    mf_aborted_single.write_text(
        json.dumps(
            {
                "job_id": aborted_single,
                "commit": "d" * 40,
                "output_dir": str(pkgs / "never-created-single"),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    for mf in (
        mf_a1,
        fb_a1,
        mf_a_auto,
        fb_a_auto,
        mf_b,
        fb_b,
        mf_c,
        fb_c,
        mf_d,
        fb_d,
        mf_e,
        mf_f_pref,
        fb_f_pref,
        mf_f_a100,
        mf_aborted_single,
    ):
        assert dispatch.job_is_actively_staging(mf, ops_dir) is False

    # Pre-seed watch6-staging-reconcile.json and verify CLI --reconcile-staging clears it:
    fence_path = dispatch._write_reconcile_fence(
        ops_dir,
        reason="exited_manifest_missing_staging_or_stop_evidence",
        job_name="critic-rescore-20261029-01",
        details={"manifest": str(mf_a1)},
    )
    assert fence_path.is_file()

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.watch6_search_efficacy_v3.dispatch",
            "--reconcile-staging",
            "--ops-dir",
            str(ops_dir),
        ],
        cwd=p.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    reconciled = json.loads(proc.stdout)
    assert reconciled["status"] == "RECONCILED"
    assert reconciled["cleared_fence"] is True
    assert reconciled["checked_manifests"] == 15
    assert not fence_path.exists()

    # Subsequent live dispatch over this populated historical ops_dir proceeds cleanly:
    ruling_json = _write_approved_ruling(tmp_path)
    op_root = tmp_path / "operator"
    (op_root / "tools/ops").mkdir(parents=True)
    fake_dispatch = op_root / "tools/ops/dispatch.sh"
    fake_dispatch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'OUT=$(python3 -c "import json,sys; '
        'print(json.load(open(sys.argv[1]))[\'output_dir\'])" "$2")\n'
        'mkdir -p "$OUT"\n'
        'echo \'{"elapsed_seconds": 12.0}\' > "$OUT/upload-measurement.json"\n',
        encoding="utf-8",
    )
    fake_dispatch.chmod(0o755)
    canary_mf = tmp_path / "v3-canary.json"
    canary_mf.write_text(
        json.dumps({"job_id": "watch6-v3-canary-live", "output_dir": str(tmp_path / "v3-out")})
    )
    with patch.object(dispatch, "admit", return_value={"status": "PASS"}):
        rc = dispatch.run_dispatch(
            [
                "watch6-v3-canary-live",
                str(canary_mf),
                "--broker",
                str(p.ROOT),
                "--cwd",
                str(p.ROOT),
                "--operator-root",
                str(op_root),
                "--ruling-json",
                str(ruling_json),
            ],
            ops_dir=ops_dir,
            poll_seconds=0.02,
        )
    assert rc == 0
    assert not fence_path.exists()


def test_v3_staging_blocks_inflight_uploads_and_fences_leaked_or_ambiguous_attempts(
    tmp_path: Path,
) -> None:
    import fcntl

    # 1. Genuinely in-flight primary upload (even backdated 6 hours!) MUST block:
    ops_inflight = tmp_path / "ops-inflight"
    (ops_inflight / "manifests").mkdir(parents=True)
    (ops_inflight / "logs").mkdir(parents=True)
    (ops_inflight / "dispatch/inflight-primary").mkdir(parents=True)
    (ops_inflight / "alloc.lock").touch()
    inflight_out = tmp_path / "inflight-out"
    inflight_out.mkdir()
    (inflight_out / "rate-calibration.json").write_text('{"status": "PASS"}\n')
    (ops_inflight / "logs/inflight-primary.log").write_text("uploading tar...\n", encoding="utf-8")
    inflight_mf = ops_inflight / "manifests/inflight-primary.json"
    inflight_mf.write_text(
        json.dumps({"job_id": "inflight-primary", "output_dir": str(inflight_out)}) + "\n"
    )
    six_hours_ago = time.time() - 21600.0
    os.utime(inflight_mf, (six_hours_ago, six_hours_ago))
    assert dispatch.job_is_actively_staging(inflight_mf, ops_inflight) is True

    dispatch._write_reconcile_fence(
        ops_inflight, reason="manual", job_name="inflight-primary", details={}
    )
    with pytest.raises(RuntimeError, match="manifests still staging"):
        dispatch.reconcile_staging(ops_inflight)
    assert (ops_inflight / dispatch.RECONCILE_MARKER_NAME).is_file()

    # 2. Genuinely in-flight -a100-fallback upload after H100 503 primary MUST block BOTH manifests:
    ops_fb = tmp_path / "ops-fb-inflight"
    mf_h100, _ = _write_unallocated_503_attempt(
        ops_fb,
        name="h100-then-a100-uploading",
        out_dir=tmp_path / "fb-inflight/output",
        prefer=["H100", "A100"],
        write_unavailable_proof=True,
    )
    mf_a100 = _write_fallback_sibling(
        ops_fb,
        primary_name="h100-then-a100-uploading",
        primary_out_dir=tmp_path / "fb-inflight/output",
        sibling_state="in_flight_uploading",
    )
    assert dispatch.job_is_actively_staging(mf_h100, ops_fb) is True
    assert dispatch.job_is_actively_staging(mf_a100, ops_fb) is True
    with pytest.raises(RuntimeError, match="REFUSE: staging serialization wait expired"):
        dispatch.wait_other_jobs_staged(ops_fb, "watch6-v3-next", timeout=0.06, poll_seconds=0.02)
    assert (ops_fb / dispatch.RECONCILE_MARKER_NAME).is_file()

    # Once A100 fallback writes upload-measurement.json, both return False and reconcile clears:
    (tmp_path / "fb-inflight/output-a100-fallback/upload-measurement.json").write_text(
        '{"elapsed_seconds": 33.5}\n'
    )
    assert dispatch.job_is_actively_staging(mf_h100, ops_fb) is False
    assert dispatch.job_is_actively_staging(mf_a100, ops_fb) is False
    assert dispatch.reconcile_staging(ops_fb)["cleared_fence"] is True
    assert not (ops_fb / dispatch.RECONCILE_MARKER_NAME).exists()

    # 3. Pre-start dispatch currently holding ops/alloc.lock MUST block until lock releases:
    ops_locked = tmp_path / "ops-locked"
    (ops_locked / "manifests").mkdir(parents=True)
    (ops_locked / "logs").mkdir(parents=True)
    (ops_locked / "dispatch/validating-now").mkdir(parents=True)
    lock_file = ops_locked / "alloc.lock"
    lock_file.touch()
    mf_validating = ops_locked / "manifests/validating-now.json"
    mf_validating.write_text(
        json.dumps({"job_id": "validating-now", "output_dir": str(tmp_path / "uncreated-val")})
        + "\n"
    )
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl,sys,time; f=open(sys.argv[1],'r+'); "
            "fcntl.flock(f, fcntl.LOCK_SH); sys.stdout.write('LOCKED\\n'); sys.stdout.flush(); "
            "time.sleep(10)",
            str(lock_file),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "LOCKED"
        assert dispatch.job_is_actively_staging(mf_validating, ops_locked) is True
    finally:
        holder.terminate()
        holder.wait(timeout=5)
    assert dispatch.job_is_actively_staging(mf_validating, ops_locked) is False

    # Active sandbox allocation-lease.json also blocks even if alloc.lock is unlocked:
    lease_path = ops_locked / "dispatch/validating-now/allocation-lease.json"
    lease_path.write_text('{"schema": "sandbox-allocation-lease/v1"}\n')
    assert dispatch.job_is_actively_staging(mf_validating, ops_locked) is True
    lease_path.unlink()
    assert dispatch.job_is_actively_staging(mf_validating, ops_locked) is False
    _ = fcntl

    # 4. Leaked primary runtime (endpoint != None, stop_verified=False) MUST fence even when
    #    -a100-fallback sibling succeeded!
    ops_leak = tmp_path / "ops-leak"
    mf_leak, _ = _write_unallocated_503_attempt(
        ops_leak,
        name="leaked-primary",
        out_dir=tmp_path / "leak/output",
        prefer=["H100", "A100"],
        write_unavailable_proof=True,
        endpoint="m-s-leaked-h100",
        stop_verified=False,
    )
    _write_fallback_sibling(
        ops_leak,
        primary_name="leaked-primary",
        primary_out_dir=tmp_path / "leak/output",
        sibling_state="uploaded_and_stopped",
    )
    with pytest.raises(RuntimeError, match="REFUSE: exited ops manifest leaked-primary.json"):
        dispatch.job_is_actively_staging(mf_leak, ops_leak)
    assert (ops_leak / dispatch.RECONCILE_MARKER_NAME).is_file()

    # 5. Ambiguous colab new client timeout (endpoint=None, retained pending record, NOT server 503
    #    refusal and NO LIVE unavailable.json) MUST fence even if sibling succeeded!
    ops_ambig = tmp_path / "ops-ambig"
    mf_ambig, _ = _write_unallocated_503_attempt(
        ops_ambig,
        name="ambig-timeout-primary",
        out_dir=tmp_path / "ambig/output",
        prefer=["H100", "A100"],
        write_unavailable_proof=False,
        custom_error="Refused('colab new timed out after 60s: TimeoutExpired')",
    )
    _write_fallback_sibling(
        ops_ambig,
        primary_name="ambig-timeout-primary",
        primary_out_dir=tmp_path / "ambig/output",
        sibling_state="uploaded_and_stopped",
    )
    with pytest.raises(
        RuntimeError, match="REFUSE: exited ops manifest ambig-timeout-primary.json"
    ):
        dispatch.job_is_actively_staging(mf_ambig, ops_ambig)
    assert (ops_ambig / dispatch.RECONCILE_MARKER_NAME).is_file()

    # 6. Exited unreconciled -a100-fallback sibling MUST fence when scanning primary too:
    ops_bad_sib = tmp_path / "ops-bad-sib"
    mf_good_prim, _ = _write_unallocated_503_attempt(
        ops_bad_sib,
        name="good-prim-bad-sib",
        out_dir=tmp_path / "bad-sib/output",
        prefer=["H100", "A100"],
        write_unavailable_proof=True,
    )
    _write_fallback_sibling(
        ops_bad_sib,
        primary_name="good-prim-bad-sib",
        primary_out_dir=tmp_path / "bad-sib/output",
        sibling_state="exited_unreconciled",
    )
    with pytest.raises(RuntimeError, match="REFUSE: exited ops manifest good-prim-bad-sib.json"):
        dispatch.job_is_actively_staging(mf_good_prim, ops_bad_sib)
    assert (ops_bad_sib / dispatch.RECONCILE_MARKER_NAME).is_file()

    # 7. Forged/DRY_RUN or hash-mismatched unavailable.json is rejected on ambiguous receipt:
    ops_forged_u = tmp_path / "ops-forged-unav"
    mf_forged_u, rec_forged_u = _write_unallocated_503_attempt(
        ops_forged_u,
        name="forged-unav-job",
        out_dir=tmp_path / "forged-unav/output",
        prefer=None,
        write_unavailable_proof=True,
        custom_error="Refused('colab new timed out after 60s: TimeoutExpired')",
    )
    unav_file = ops_forged_u / "dispatch/forged-unav-job/unavailable.json"
    unav_obj = json.loads(unav_file.read_text())
    unav_obj["mode"] = "DRY_RUN"
    unav_file.write_text(json.dumps(unav_obj) + "\n")
    with pytest.raises(RuntimeError, match="REFUSE: exited ops manifest forged-unav-job.json"):
        dispatch.job_is_actively_staging(mf_forged_u, ops_forged_u)
    unav_obj["mode"] = "LIVE"
    unav_obj["receipt"] = {"path": str(rec_forged_u), "sha256": "0" * 64}
    unav_file.write_text(json.dumps(unav_obj) + "\n")
    with pytest.raises(RuntimeError, match="REFUSE: exited ops manifest forged-unav-job.json"):
        dispatch.job_is_actively_staging(mf_forged_u, ops_forged_u)

    # 8. wait_own_job_staged holds lock while -retry-1 sibling is still uploading after base retry:
    ops_retry = tmp_path / "ops-own-retry"
    (ops_retry / "manifests").mkdir(parents=True)
    (ops_retry / "logs").mkdir(parents=True)
    own_base = tmp_path / "own-retry-out"
    own_base.mkdir()
    (own_base / "receipt.json").write_text(
        json.dumps({"endpoint": "m-s-acct", "stop_verified": True, "cleanup_error": None}) + "\n"
    )
    (ops_retry / "logs/own-r.log").write_text("EXIT=1\nOPS_ALLOCATION_RETRY\n")
    own_r1 = Path(str(own_base) + "-retry-1")
    own_r1.mkdir()
    (ops_retry / "logs/own-r-retry-1.log").write_text("uploading on retry-1...\n")
    (ops_retry / "manifests/own-r.json").write_text(
        json.dumps({"job_id": "own-r", "output_dir": str(own_base)}) + "\n"
    )
    (ops_retry / "manifests/own-r-retry-1.json").write_text(
        json.dumps({"job_id": "own-r-retry-1", "output_dir": str(own_r1)}) + "\n"
    )
    with pytest.raises(RuntimeError, match="REFUSE: own staging wait expired"):
        dispatch.wait_own_job_staged(
            ops_retry,
            "own-r",
            str(own_base),
            ops_retry / "logs/own-r.log",
            timeout=0.05,
            poll_seconds=0.02,
        )
    (ops_retry / dispatch.RECONCILE_MARKER_NAME).unlink()
    (own_r1 / "upload-measurement.json").write_text('{"elapsed_seconds": 19.0}\n')
    dispatch.wait_own_job_staged(
        ops_retry,
        "own-r",
        str(own_base),
        ops_retry / "logs/own-r.log",
        timeout=0.2,
        poll_seconds=0.02,
    )
