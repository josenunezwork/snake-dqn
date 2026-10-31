"""Unit and integration tests for Watch6 V3 confirmation failure handling and V2 parity."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
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
    resume,
    run,
)
from tools.ops.package_spec import digest, empty_ledger, publish_json, unit_id, verify_immutable


def _synthetic_v3_rows(
    *,
    death_delta: int = -2,
    relative: float = 0.0,
    reserve_slots: set[int] | None = None,
) -> list[dict]:
    reserve_slots = reserve_slots or set()
    primary_alloc = p.allocation()
    reserve_alloc = p.reserve_allocation()
    result = []
    for index in range(p.COUNT):
        kind = "reserve" if index in reserve_slots else "primary"
        alloc = copy.deepcopy(reserve_alloc[index] if kind == "reserve" else primary_alloc[index])
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
                "seed_kind": kind,
                "allocation": alloc,
                "fixture": False,
                "unit_id": unit_id(p.NAMESPACE, "watch6", index),
                "arms": {"bare": bare, "exact_stop": dense},
            }
        )
    return result


def _synthetic_cost_report() -> dict:
    # Uses the exact measured L4 W4 numbers from retained costs-source02.json
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
        rows = []
        for idx in range(workers):
            rows.append(
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
            )
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


def _write_attempt_output(
    out_dir: Path,
    *,
    shard: int,
    status: str = "PASS",
    failure_class: str | None = None,
    error: str | None = None,
    stop_verified: bool = True,
    cleanup_error: str | None = None,
    write_hardware: bool = True,
    write_progress: bool = True,
    write_terminal: dict | None = None,
    export_slots: list[int] | None = None,
    phase: str = "primary",
    reserve_slots: list[int] | None = None,
    reserve_lineage: dict | None = None,
    pair_rows_by_slot: dict[int, dict] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    reserve_slots = sorted(reserve_slots or [])
    manifest = {
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
            "shard": shard,
            "phase": phase,
            "reserve_slots": reserve_slots,
            "reserve_lineage": reserve_lineage or {},
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    if write_hardware:
        (out_dir / "hardware.json").write_text(
            json.dumps({"ready": True, "hardware": {"kind": "L4"}}) + "\n"
        )
    if write_progress:
        (out_dir / "telemetry-000001.json").write_text(
            json.dumps({"stdout_tail": f"WATCH6_V3_PROGRESS shard={shard} done=0/4"}) + "\n"
        )
    if write_terminal is not None:
        (out_dir / "terminal.json").write_text(json.dumps(write_terminal) + "\n")

    exports = []
    for slot in export_slots or []:
        kind = "reserve" if slot in reserve_slots else "primary"
        alloc = p.reserve_allocation()[slot] if kind == "reserve" else p.allocation()[slot]
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
                "seed_kind": kind,
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
        "status": status,
        "failure_class": failure_class,
        "error": error,
        "stop_verified": stop_verified,
        "cleanup_error": cleanup_error,
        "terminal": write_terminal,
        "exports": exports,
    }
    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return receipt_path


def test_v3_seed_disjointness_and_sealed_export_guard() -> None:
    primary = {r["seed"] for r in p.allocation()}
    reserve = {r["seed"] for r in p.reserve_allocation()}
    fixtures = set(p.FIXTURE_SEEDS)
    assert len(primary) == p.COUNT == 96
    assert len(reserve) == p.COUNT == 96
    assert len(primary | reserve | fixtures) == 195

    v2_seeds = {r["seed"] for r in v2_p.allocation()}
    assert not (primary | reserve | fixtures) & v2_seeds

    for ns, count in freshness.EXPLICIT_DISJOINT_NAMESPACES:
        assert not (primary | reserve | fixtures) & freshness._derive_namespace_seeds(ns, count)

    with pytest.raises(ValueError, match="sealed export"):
        freshness._assert_not_sealed_export(
            Path("/tmp/output-shard-0/exports/efficacy/operational.json")
        )
    with pytest.raises(ValueError, match="sealed export"):
        freshness._assert_not_sealed_export(Path("/tmp/any/pair-00.json"))


def test_v3_empirical_forecast_24x4_w4() -> None:
    report = _synthetic_cost_report()
    forecast = costs.empirical_forecast(report)
    assert forecast["shards"] == 24
    assert forecast["workers"] == 4
    assert forecast["pairs_per_shard"] == 4
    assert forecast["waves_per_shard"] == 1
    assert forecast["base_minutes_per_shard"] == 209
    assert forecast["effective_minutes"] == 418
    assert (
        costs.empirical_forecast(
            report, {"shards": 24, "workers": 4, "base_minutes_per_shard": 209}
        )["base_minutes_per_shard"]
        == 209
    )
    with pytest.raises(ValueError, match="frozen 24x4"):
        costs.empirical_forecast(report, {"shards": 6, "workers": 4, "base_minutes_per_shard": 209})


def test_v3_decide_matches_v2_and_validates_reserve_lineage() -> None:
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
    assert v3_res["death_delta_per_1k"] == pytest.approx(v2_res["death_delta_per_1k"])
    assert v3_res["death_delta_ci95"] == pytest.approx(v2_res["death_delta_ci95"])
    assert v3_res["mass_relative_difference"] == pytest.approx(v2_res["mass_relative_difference"])
    assert v3_res["mass_relative_ci95_fieller"] == pytest.approx(
        v2_res["mass_relative_ci95_fieller"]
    )
    assert v3_res["primary_pairs_count"] == 96
    assert v3_res["reserve_substitutions_count"] == 0

    reserve_rows = _synthetic_v3_rows(reserve_slots={5, 29})
    res_with_reserve = analyze.decide_v3(reserve_rows, reserve_verified_indices={5, 29})
    assert res_with_reserve["verdict"] == "GO_EFFICACY_ONLY"
    assert res_with_reserve["primary_pairs_count"] == 94
    assert res_with_reserve["reserve_substitutions_count"] == 2
    assert res_with_reserve["reserve_substituted_indices"] == [5, 29]

    with pytest.raises(ValueError, match="without verified reserve lineage"):
        analyze.decide_v3(reserve_rows, reserve_verified_indices={5})
    with pytest.raises(ValueError, match="exceed cap"):
        analyze.decide_v3(
            _synthetic_v3_rows(reserve_slots=set(range(17))),
            reserve_verified_indices=set(range(17)),
        )


def test_v3_runtime_admission_and_ratify_fence(tmp_path: Path) -> None:
    report = _synthetic_cost_report()
    publish_json(
        tmp_path / "verified-ledger.json",
        empty_ledger(p.NAMESPACE, "watch6", run.ledger_identity()),
    )
    publish_json(
        tmp_path / "runtime-contract.json",
        {
            "schema": "watch6-v3-runtime-contract/v1",
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "primary_allocation": p.allocation(),
            "reserve_allocation": p.reserve_allocation(),
            "reserve_slots": [],
        },
    )
    publish_json(
        tmp_path / "mac-mirror.json",
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
    publish_json(
        tmp_path / "judge-approval.json",
        {
            "schema": "watch6-v3-protocol-binding/v1",
            "implementer": "Argon",
            "judge": "Astra",
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "registration_sha256": digest(p.registration()),
        },
    )
    publish_json(tmp_path / "cost-evidence.json", report)
    env = {
        "WATCH6_SCIENCE": p.identity(),
        "SNAKE_WATCH_SOURCE": "live",
        "WATCH6_SHARD": "0",
        "WATCH6_FIXTURE": "0",
        "WATCH6_WORKERS": "4",
        "WATCH6_SHARDS": "24",
        "WATCH6_BASE_MINUTES": "209",
        "WATCH6_MEASUREMENTS": json.dumps(report, sort_keys=True),
        "WATCH6_FROZEN_LAYOUT": json.dumps(
            {"workers": 4, "shards": 24, "base_minutes_per_shard": 209}, sort_keys=True
        ),
    }
    assert admission.verify_runtime(tmp_path, env)["reserve_slots"] == []
    with pytest.raises(ValueError, match="24 shards, W4"):
        admission.verify_runtime(tmp_path, {**env, "WATCH6_SHARDS": "6"})
    with pytest.raises(ValueError, match="24 shards, W4"):
        admission.verify_runtime(tmp_path, {**env, "WATCH6_WORKERS": "8"})

    ruling_out = tmp_path / "astra-ruling.json"
    with pytest.raises(ValueError, match="REFUSE: missing ratified Astra"):
        ratify.validate_ruling_receipt(ruling_out)
    ruling_md = tmp_path / "RULING_V3.md"
    ruling_md.write_text("# RATIFIED BY ASTRA PRE-DATA\n")
    ratify.record_ruling(ruling_md, ruling_out)
    assert ratify.validate_ruling_receipt(ruling_out)["status"] == "RATIFIED"
    ruling_md.write_text("# MUTATED AFTER BINDING\n")
    with pytest.raises(ValueError, match="REFUSE: Astra pre-data ruling receipt"):
        ratify.validate_ruling_receipt(ruling_out)


def test_v3_serialized_staging_lock_and_exact_refusal_passthrough(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    ops_dir = tmp_path / "ops"
    (ops_dir / "manifests").mkdir(parents=True)
    (ops_dir / "logs").mkdir(parents=True)

    ruling_md = tmp_path / "RULING_V3.md"
    ruling_md.write_text("RATIFIED\n")
    ruling_json = tmp_path / "astra-ruling.json"
    ratify.record_ruling(ruling_md, ruling_json)

    op_root = tmp_path / "operator"
    (op_root / "tools/ops").mkdir(parents=True)
    fake_dispatch = op_root / "tools/ops/dispatch.sh"

    # Already-staged computing job must NOT block new dispatches.
    computing_out = tmp_path / "computing-out"
    computing_out.mkdir()
    (computing_out / "upload-measurement.json").write_text('{"elapsed_seconds": 40.0}\n')
    (ops_dir / "manifests/computing-job.json").write_text(
        json.dumps({"output_dir": str(computing_out)})
    )
    assert not dispatch.job_is_actively_staging(ops_dir / "manifests/computing-job.json", ops_dir)

    # Active staging job DOES report actively staging until upload-measurement.json appears.
    staging_out = tmp_path / "staging-out"
    staging_out.mkdir()
    (ops_dir / "manifests/staging-job.json").write_text(
        json.dumps({"output_dir": str(staging_out)})
    )
    assert dispatch.job_is_actively_staging(ops_dir / "manifests/staging-job.json", ops_dir)
    (staging_out / "upload-measurement.json").write_text('{"elapsed_seconds": 35.0}\n')
    assert not dispatch.job_is_actively_staging(ops_dir / "manifests/staging-job.json", ops_dir)

    # Concurrent staging serialization test: Job B cannot invoke operator dispatch.sh
    # until Job A's background runtime writes upload-measurement.json.
    events_file = tmp_path / "staging-events.txt"
    fake_dispatch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'NAME="$1"; MANIFEST="$2"\n'
        'OUT=$(python3 -c "import json,sys; '
        'print(json.load(open(sys.argv[1]))[\'output_dir\'])" "$MANIFEST")\n'
        'mkdir -p "$OUT"\n'
        f'echo "START:$NAME" >> "{events_file}"\n'
        '( sleep 0.25; echo "STAGED:$NAME" >> "'
        + str(events_file)
        + '"; echo "{}" > "$OUT/upload-measurement.json" ) &\n'
    )
    fake_dispatch.chmod(0o755)

    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    m1.write_text(json.dumps({"job_id": "j1", "output_dir": str(tmp_path / "out1")}))
    m2.write_text(json.dumps({"job_id": "j2", "output_dir": str(tmp_path / "out2")}))

    results: list[int] = []

    def launch(name: str, mf: Path) -> None:
        rc = dispatch.run_dispatch(
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
            poll_seconds=0.05,
        )
        results.append(rc)

    with patch.object(dispatch, "admit", return_value={"status": "PASS"}):
        t1 = threading.Thread(target=launch, args=("job-a", m1))
        t2 = threading.Thread(target=launch, args=("job-b", m2))
        t1.start()
        time.sleep(0.05)
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

    assert results == [0, 0]
    order = events_file.read_text().splitlines()
    assert order == ["START:job-a", "STAGED:job-a", "START:job-b", "STAGED:job-b"]

    # Pre-allocation cap-6 refusal passthrough must emit ZERO extra stdout/stderr lines.
    fake_dispatch.write_text(
        "#!/usr/bin/env bash\n"
        'echo "REFUSE: account-wide assignment cap 6 (including notebooks)"\n'
        "exit 2\n"
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
            poll_seconds=0.05,
        )
    captured = capfd.readouterr()
    assert rc == 2
    assert (captured.out + captured.err).strip() == (
        "REFUSE: account-wide assignment cap 6 (including notebooks)"
    )


def test_v3_outcome_blind_ledger_and_fixture_rejection(tmp_path: Path) -> None:
    receipt_path = _write_attempt_output(
        tmp_path / "out-s0",
        shard=0,
        status="INCOMPLETE",
        failure_class="RPC_EXEC_POLL_TIMEOUT",
        error="rpc poll timed out",
        export_slots=[0, 24],
    )
    u0 = tmp_path / "out-s0/exports/efficacy/pair-00.json"
    u24 = tmp_path / "out-s0/exports/efficacy/pair-24.json"
    ledger = resume.build([u24, u0], [receipt_path])
    assert [row["index"] for row in ledger["units"]] == [0, 24]

    fixture_path = tmp_path / "out-s0/exports/efficacy/fixture-pair-00.json"
    publish_json(fixture_path, {"fixture": True})
    with pytest.raises(ValueError, match="non-scientific or fixture"):
        resume.build([fixture_path], [receipt_path])

    unstopped = _write_attempt_output(
        tmp_path / "out-s1", shard=1, stop_verified=False, export_slots=[1]
    )
    with pytest.raises(ValueError, match="stop not verified"):
        resume.build([tmp_path / "out-s1/exports/efficacy/pair-01.json"], [unstopped])


def _mock_recovery_env(tmp_path: Path) -> tuple[Path, Path, Path, Path, str]:
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
    return base_pkg, bench, mir, cost_path, commit


def test_v3_tier1_prestart_retries_beyond_two_reuse_canary_and_enforce_cap6(
    tmp_path: Path,
) -> None:
    base_pkg, bench, mir, cost_path, commit = _mock_recovery_env(tmp_path)
    receipts = [
        _write_attempt_output(
            tmp_path / f"prestart-{i}",
            shard=5,
            status="INCOMPLETE",
            failure_class="SUPERVISOR",
            error="total upload time cap reached after staging control",
            write_hardware=False,
            write_progress=False,
        )
        for i in range(6)
    ]

    captured_calls: list[dict] = []

    def fake_prepare(out: Path, *args, **kwargs) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        captured_calls.append(kwargs)
        return {"shards": 24, "workers": 4, "base_minutes_per_shard": 209}

    with patch.object(recover, "prepare", side_effect=fake_prepare):
        # After 3 pre-start failures (exceeding V2's rigid 2-retry limit!), V3 allows retry #4
        # without burning any compute completions and reuses the initial canary receipt!
        plan3 = recover.recover(
            base_package=base_pkg,
            receipts=receipts[:3],
            target_shards=[5],
            out=tmp_path / "rec-after-3",
            benchmark=bench,
            mirror_receipt=mir,
            empirical_costs=cost_path,
            commit=commit,
        )
        assert plan3["reused_initial_canary"] is True
        assert plan3["reserve_slots"] == []
        assert plan3["shard_meta"]["5"] == {
            "prestart_retries_used": 3,
            "compute_completion_attempt": 0,
            "total_dispatches_for_shard": 4,
        }
        assert captured_calls[-1]["pending_canary"] is False

        # Up to 6 pre-start failures allowed:
        plan6 = recover.recover(
            base_package=base_pkg,
            receipts=receipts[:6],
            target_shards=[5],
            out=tmp_path / "rec-after-6",
            benchmark=bench,
            mirror_receipt=mir,
            empirical_costs=cost_path,
            commit=commit,
        )
        assert plan6["shard_meta"]["5"]["prestart_retries_used"] == 6

        # 7th pre-start failure exceeds MAX_PRESTART_RETRIES_PER_SHARD=6 -> refuses!
        seventh = _write_attempt_output(
            tmp_path / "prestart-6",
            shard=5,
            status="INCOMPLETE",
            failure_class="UPLOAD_CAP",
            error="staging exceeded 300s",
            write_hardware=False,
            write_progress=False,
        )
        with pytest.raises(ValueError, match="pre-start retries 7 exceed cap 6"):
            recover.recover(
                base_package=base_pkg,
                receipts=[*receipts, seventh],
                target_shards=[5],
                out=tmp_path / "rec-after-7",
                benchmark=bench,
                mirror_receipt=mir,
                empirical_costs=cost_path,
                commit=commit,
            )


def test_v3_tier2_partial_shard_resume_and_tier3_mcar_reserve_boundary(tmp_path: Path) -> None:
    base_pkg, bench, mir, cost_path, commit = _mock_recovery_env(tmp_path)

    # Attempt 1 on shard 5 exports slots 5 and 29, then loses RPC connection mid-compute.
    r1 = _write_attempt_output(
        tmp_path / "c1",
        shard=5,
        status="INCOMPLETE",
        failure_class="RPC_EXEC_POLL_TIMEOUT",
        error="recoverable rpc poll timed out",
        export_slots=[5, 29],
    )

    def fake_prepare(out: Path, *args, **kwargs) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        return {"shards": 24, "workers": 4, "base_minutes_per_shard": 209}

    with patch.object(recover, "prepare", side_effect=fake_prepare):
        # Tier 2 resume #1: seals 2 verified pairs into ledger, keeps primary seeds for [53, 77].
        plan_t2 = recover.recover(
            base_package=base_pkg,
            receipts=[r1],
            target_shards=[5],
            out=tmp_path / "rec-t2-1",
            benchmark=bench,
            mirror_receipt=mir,
            empirical_costs=cost_path,
            commit=commit,
        )
        assert plan_t2["verified_units_in_ledger"] == 2
        assert plan_t2["reserve_slots"] == []
        assert plan_t2["requires_short_recovery_canary"] is True
        assert plan_t2["shard_meta"]["5"]["compute_completion_attempt"] == 1

        # Attempts 2 and 3 on shard 5 also fail mid-compute on external transport loss (0 exports).
        r2 = _write_attempt_output(
            tmp_path / "c2",
            shard=5,
            status="INCOMPLETE",
            failure_class="TRANSPORT_TIMEOUT",
            error="transport read timeout",
        )
        r3 = _write_attempt_output(
            tmp_path / "c3",
            shard=5,
            status="INCOMPLETE",
            failure_class="RUNTIME_LOST",
            error="runtime disappeared",
        )

        # Tier 3 MCAR reserve: activates reserve seeds ONLY for unexported slots [53, 77]!
        plan_t3 = recover.recover(
            base_package=base_pkg,
            receipts=[r1, r2, r3],
            target_shards=[5],
            out=tmp_path / "rec-t3",
            benchmark=bench,
            mirror_receipt=mir,
            empirical_costs=cost_path,
            commit=commit,
        )
        assert plan_t3["verified_units_in_ledger"] == 2
        assert plan_t3["reserve_slots"] == [53, 77]
        assert plan_t3["shard_meta"]["5"]["compute_completion_attempt"] == 0

        # Non-MCAR refusal 1: wall-clock timeout MUST NEVER trigger reserve substitution!
        r3_wall = _write_attempt_output(
            tmp_path / "c3-wall",
            shard=5,
            status="INCOMPLETE",
            failure_class="SUPERVISOR",
            error="whole-session wall cap reached (including setup/export)",
        )
        with pytest.raises(ValueError, match="non-MCAR / child / wall-timeout"):
            recover.recover(
                base_package=base_pkg,
                receipts=[r1, r2, r3_wall],
                target_shards=[5],
                out=tmp_path / "rec-wall-refused",
                benchmark=bench,
                mirror_receipt=mir,
                empirical_costs=cost_path,
                commit=commit,
            )

        # Non-MCAR refusal 2: CHILD_NONZERO_EXIT MUST NEVER trigger reserve substitution!
        r3_child = _write_attempt_output(
            tmp_path / "c3-child",
            shard=5,
            status="INCOMPLETE",
            failure_class="CHILD_NONZERO_EXIT",
            error="child exited with 1",
            write_terminal={"returncode": 1},
        )
        with pytest.raises(ValueError, match="non-MCAR / child / wall-timeout"):
            recover.recover(
                base_package=base_pkg,
                receipts=[r1, r2, r3_child],
                target_shards=[5],
                out=tmp_path / "rec-child-refused",
                benchmark=bench,
                mirror_receipt=mir,
                empirical_costs=cost_path,
                commit=commit,
            )

        # Reserve failure refusal: if the reserve compute attempt itself fails, refuse 2nd reserve!
        lineage_5 = {
            "shards": {
                "5": [{"receipt_path": str(x), "receipt_sha256": digest(x)} for x in (r1, r2, r3)]
            }
        }
        r4_res_fail = _write_attempt_output(
            tmp_path / "c4-res",
            shard=5,
            status="INCOMPLETE",
            failure_class="RUNTIME_LOST",
            error="runtime disappeared",
            phase="reserve",
            reserve_slots=[53, 77],
            reserve_lineage=lineage_5,
        )
        with pytest.raises(ValueError, match="exhausted its single reserve attempt"):
            recover.recover(
                base_package=base_pkg,
                receipts=[r1, r2, r3, r4_res_fail],
                target_shards=[5],
                out=tmp_path / "rec-res2-refused",
                benchmark=bench,
                mirror_receipt=mir,
                empirical_costs=cost_path,
                commit=commit,
            )


def test_v3_finalize_idempotent_single_shard(tmp_path: Path) -> None:
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
        assert json.loads(sealed[0].read_text())["canary"]["receipt"]["sha256"] == digest(
            canary_receipt
        )
        assert "pending_qualification" not in json.loads(sealed[0].read_text())
        # Second call with --idempotent succeeds without overwriting:
        assert finalize.finalize(pkg, shard=5, idempotent=True) == [pkg / "shard-05.json"]
        with pytest.raises(ValueError, match="already exists"):
            finalize.finalize(pkg, shard=5, idempotent=False)


def test_v3_final_look_end_to_end_and_incomplete_guard(tmp_path: Path) -> None:
    ops_dir = tmp_path / "ops"
    (ops_dir / "manifests").mkdir(parents=True)

    ruling_md = tmp_path / "RULING_V3.md"
    ruling_md.write_text("RATIFIED BY ASTRA\n")
    ruling_json = tmp_path / "astra-ruling.json"
    ratify.record_ruling(ruling_md, ruling_json)

    # Canary manifest + stopped receipt (with -retry-1 accounting retry dir!).
    canary_base_out = tmp_path / "output-canary"
    canary_retry_out = Path(str(canary_base_out) + "-retry-1")
    canary_retry_out.mkdir(parents=True)
    canary_mf = {
        "job_id": "watch6-v3-canary",
        "output_dir": str(canary_base_out),
        "canary": {"schema": canary.SCHEMA, "mode": "probe", "family": p.SCRIPT},
        "env": {"WATCH6_FIXTURE": "1"},
        "watch6_v3": {"namespace": p.NAMESPACE, "science": p.identity()},
    }
    (canary_retry_out / "manifest.json").write_text(json.dumps(canary_mf))
    (canary_retry_out / "receipt.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "stop_verified": True,
                "cleanup_error": None,
                "manifest_sha256": digest(canary_retry_out / "manifest.json"),
                "exports": [],
            }
        )
    )
    (ops_dir / "manifests/watch6-v3-canary.json").write_text(json.dumps(canary_mf))

    # Proven unallocated pre-dispatch refusal entry is safely ignored.
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

    all_rows = {r["allocation"]["index"]: r for r in _synthetic_v3_rows(reserve_slots={53, 77})}

    # Shards 0..23 except shard 5 complete on primary attempt 1:
    for s in range(p.SHARDS):
        if s == 5:
            continue
        r_path = _write_attempt_output(
            tmp_path / f"out-shard-{s:02d}",
            shard=s,
            export_slots=p.shard_indices(s),
            pair_rows_by_slot=all_rows,
        )
        (ops_dir / f"manifests/shard-{s:02d}.json").write_text(
            (r_path.parent / "manifest.json").read_text()
        )

    # Shard 5: 3 external-transport failed primary compute attempts (attempt 1 exported [5, 29]),
    # before reserve attempt completes [53, 77].
    s5_r1 = _write_attempt_output(
        tmp_path / "out-shard-05-a1",
        shard=5,
        status="INCOMPLETE",
        failure_class="RPC_EXEC_POLL_TIMEOUT",
        error="poll timeout",
        export_slots=[5, 29],
        pair_rows_by_slot=all_rows,
    )
    s5_r2 = _write_attempt_output(
        tmp_path / "out-shard-05-a2",
        shard=5,
        status="INCOMPLETE",
        failure_class="TRANSPORT_TIMEOUT",
        error="transport timeout",
    )
    s5_r3 = _write_attempt_output(
        tmp_path / "out-shard-05-a3",
        shard=5,
        status="INCOMPLETE",
        failure_class="RUNTIME_LOST",
        error="runtime lost",
    )
    for idx, rp in enumerate((s5_r1, s5_r2, s5_r3), 1):
        (ops_dir / f"manifests/shard-05-a{idx}.json").write_text(
            (rp.parent / "manifest.json").read_text()
        )

    # Before shard 5's reserve attempt runs (94/96 pairs), final_look refuses outcome-blind!
    with pytest.raises(ValueError, match="INVALID_OR_INCOMPLETE: 94/96 pairs verified"):
        final_look.run_final_look(
            ops_dir=ops_dir, out_dir=tmp_path / "look-incomplete", ruling_json=ruling_json
        )
    assert (tmp_path / "look-incomplete/incomplete-status.json").is_file()
    assert not (tmp_path / "look-incomplete/efficacy-decision.json").exists()

    # Now shard 5's reserve attempt exports [53, 77]:
    lineage_5 = {
        "shards": {
            "5": [
                {"receipt_path": str(rp), "receipt_sha256": digest(rp)}
                for rp in (s5_r1, s5_r2, s5_r3)
            ]
        }
    }
    s5_res = _write_attempt_output(
        tmp_path / "out-shard-05-reserve",
        shard=5,
        export_slots=[53, 77],
        phase="reserve",
        reserve_slots=[53, 77],
        reserve_lineage=lineage_5,
        pair_rows_by_slot=all_rows,
    )
    (ops_dir / "manifests/shard-05-reserve.json").write_text(
        (s5_res.parent / "manifest.json").read_text()
    )

    decision = final_look.run_final_look(
        ops_dir=ops_dir, out_dir=tmp_path / "look-complete", ruling_json=ruling_json
    )
    assert decision["verdict"] == "GO_EFFICACY_ONLY"
    assert decision["n"] == 96
    assert decision["primary_pairs_count"] == 94
    assert decision["reserve_substitutions_count"] == 2
    assert decision["reserve_substituted_indices"] == [53, 77]
    verify_immutable(tmp_path / "look-complete/final-verified-ledger.json")
    verify_immutable(tmp_path / "look-complete/efficacy-decision.json")


def test_v3_real_entrypoint_fixture_emits_fixture_pair_only(tmp_path: Path) -> None:
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
