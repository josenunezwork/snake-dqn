"""Unit and integration tests for research/watch6_alllen_screen."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import t

from research.watch6_alllen_screen import (
    analyze,
    engine,
    prepare,
    protocol,
    resume,
    run,
    seal_main,
    seed_audit,
)
from tools.ops.package_spec import digest, empty_ledger, publish_json, unit_id
from web.backend.watch_replay import B8_SHA, FRP3_SHA


def _saved_snakes_dir() -> Path | None:
    for candidate in (
        protocol.ROOT.parent / "snake-dqn" / "saved_snakes",
        protocol.ROOT / "saved_snakes",
    ):
        if (candidate / "b8_s3_t7_20261016.pth").is_file() and (
            candidate / "frp3_m3_s12_u60000_20261005.pth"
        ).is_file():
            return candidate
    return None


def _synthetic_arm_summary(
    arm: str,
    *,
    horizon: int,
    mass_integral: float,
    deaths: int,
    roots_lt600: int = 0,
    roots_ge600: int = 0,
    interventions_lt600: int = 0,
    interventions_ge600: int = 0,
) -> dict[str, Any]:
    spec = protocol.ARM_SPECS[arm]
    mass_sum = int(round(mass_integral * horizon))
    roots = roots_lt600 + roots_ge600
    interventions = interventions_lt600 + interventions_ge600
    return {
        "arm": arm,
        "min_length": spec["min_length"],
        "interval": spec["interval"],
        "frames": horizon,
        "alive_frames": horizon - deaths * 5,
        "mass_sum": mass_sum,
        "mass_integral": mass_sum / horizon,
        "deaths": deaths,
        "deaths_per_1k": deaths * 1000.0 / horizon,
        "lives": 1 + deaths,
        "decisions": horizon - deaths * 5,
        "boost_decisions": 12,
        "max_length": 750,
        "alive_length_median": mass_integral,
        "alive_length_p95": mass_integral * 1.4,
        "death_lengths": [300] * deaths,
        "short_body_deaths_lt600": deaths,
        "long_body_deaths_ge600": 0,
        "late_large_body_deaths_ge800": 0,
        "search_roots": roots,
        "search_roots_lt600": roots_lt600,
        "search_roots_ge600": roots_ge600,
        "interventions": interventions,
        "interventions_lt600": interventions_lt600,
        "interventions_ge600": interventions_ge600,
        "branches_evaluated": roots * 2,
        "branches_skipped": roots * 4,
        "native_v8_kernel_rows": roots * 600,
        "respawns_per_slot": [deaths, 1, 1, 0, 2, 1],
        "mirror_identity_frames": 0,
    }


def _synthetic_units(
    *,
    count: int = protocol.COUNT,
    horizon: int = protocol.HORIZON,
    namespace: str = protocol.NAMESPACE,
    l1_mass_delta: float = 20.0,
    l1_death_delta: int = 0,
    fixture: bool = False,
) -> list[dict[str, Any]]:
    contract = {
        **protocol.allocation(namespace, count, horizon, audit_live_mirror=False),
        "commit": "a" * 40,
    }
    cid = protocol.contract_identity(contract)
    rows = []
    for item in contract["rows"]:
        idx = item["index"]
        base_mass = 380.0 + (idx % 9) * 7.0
        l2_mass = base_mass + 12.0 + (idx % 5) * 1.5
        l1_mass = l2_mass + l1_mass_delta + ((idx % 3) - 1) * 2.0
        base_deaths = 3 + (idx % 2)
        l2_deaths = max(0, base_deaths - 1)
        l1_deaths = max(0, l2_deaths + l1_death_delta)
        rows.append(
            {
                "schema": protocol.UNIT_SCHEMA,
                "lane": protocol.LANE,
                "world": contract["world"],
                "namespace": namespace,
                "unit_id": unit_id(namespace, protocol.UNIT_KIND, idx),
                "allocation": {"index": idx, "seed": item["seed"], "horizon": horizon},
                "contract_identity": cid,
                "source_commit": contract["commit"],
                "checkpoint_shas": {"hero": B8_SHA, "opponents": FRP3_SHA},
                "fixture": fixture,
                "arm_execution_order": list(protocol.arm_order_for_index(idx)),
                "roster": {"hero": "b8-s3", "opponents": ["frp3-s12"] * 5},
                "arms": {
                    "bare": _synthetic_arm_summary(
                        "bare", horizon=horizon, mass_integral=base_mass, deaths=base_deaths
                    ),
                    "l2_prime_600": _synthetic_arm_summary(
                        "l2_prime_600",
                        horizon=horizon,
                        mass_integral=l2_mass,
                        deaths=l2_deaths,
                        roots_ge600=400,
                        interventions_ge600=60,
                    ),
                    "exact_stop_alllen": _synthetic_arm_summary(
                        "exact_stop_alllen",
                        horizon=horizon,
                        mass_integral=l1_mass,
                        deaths=l1_deaths,
                        roots_lt600=720,
                        roots_ge600=420,
                        interventions_lt600=140,
                        interventions_ge600=65,
                    ),
                },
            }
        )
    return rows


def test_protocol_allocation_and_arm_permutations() -> None:
    alloc = protocol.allocation()
    assert alloc["count"] == 48
    assert alloc["horizon"] == 6000
    assert alloc["arms"] == ["bare", "l2_prime_600", "exact_stop_alllen"]
    seeds = [row["seed"] for row in alloc["rows"]]
    assert len(set(seeds)) == 48

    counts: dict[tuple[str, str, str], int] = {}
    for idx in range(48):
        order = protocol.arm_order_for_index(idx)
        assert set(order) == set(protocol.ARMS)
        counts[order] = counts.get(order, 0) + 1
    assert len(counts) == 6
    assert set(counts.values()) == {8}

    paths = set(protocol.source_paths())
    assert "research/watch6_alllen_screen" in paths
    assert "research/bl_served_search/exact_stop.py" in paths
    assert "research/watch_replay/engine.py" in paths
    assert "research/watch_replay/product.py" in paths
    assert "tools/ops/package_spec.py" in paths


def test_seed_audit_disjointness_and_never_opens_exports(tmp_path: Path) -> None:
    fake_artifacts = tmp_path / "snake-dqn-artifacts"
    for idx in range(7):
        inv = fake_artifacts / f"frp-{idx}" / "namespaces" / "training_exclusions.json"
        inv.parent.mkdir(parents=True)
        inv.write_text(json.dumps([100 + idx, 200 + idx]), encoding="utf-8")

    # Plant an own-seed inside forbidden exports/ and pair-*.json paths: audit must
    # strictly skip them without opening or flagging them.
    own_seed = protocol.seed_for(protocol.NAMESPACE, 0)
    forbidden_export = (
        fake_artifacts / "t34-sol" / "watch6-efficacy-fix" / "exports" / "pair-000000.json"
    )
    forbidden_export.parent.mkdir(parents=True)
    forbidden_export.write_text(json.dumps({"seed": own_seed}), encoding="utf-8")
    loose_pair = fake_artifacts / "other" / "pair-000001.json"
    loose_pair.parent.mkdir(parents=True)
    loose_pair.write_text(json.dumps({"seed": own_seed}), encoding="utf-8")

    report = seed_audit.audit(repo=protocol.ROOT, artifacts=fake_artifacts, extra_worktrees=[])
    assert report["status"] == "CLEAR"
    assert len(report["main_seeds"]) == 48

    # Now plant a collision inside a visible training_exclusions.json: audit MUST refuse.
    bad_inv = fake_artifacts / "frp-0" / "namespaces" / "training_exclusions.json"
    bad_inv.write_text(json.dumps([own_seed]), encoding="utf-8")
    with pytest.raises(ValueError, match="seed disjointness audit failed"):
        seed_audit.audit(repo=protocol.ROOT, artifacts=fake_artifacts, extra_worktrees=[])


def test_analyze_paired_cis_and_decision_rule() -> None:
    adopt_rows = _synthetic_units(l1_mass_delta=20.0, l1_death_delta=0)
    res_adopt = analyze.analyze(adopt_rows)
    assert res_adopt["decision_rule"]["adopt_l_ge_1_for_new_banks"] is True
    assert res_adopt["decision_rule"]["decision"] == "ADOPT_EXACT_STOP_L_GE_1_FOR_NEW_BANKS"

    inc = res_adopt["comparisons"]["exact_stop_alllen_vs_l2_prime_600"]
    diffs = np.array(
        [
            r["arms"]["exact_stop_alllen"]["mass_sum"] / 6000.0
            - r["arms"]["l2_prime_600"]["mass_sum"] / 6000.0
            for r in adopt_rows
        ]
    )
    crit = float(t.ppf(0.975, 47))
    expected_ci = [
        float(diffs.mean() - crit * diffs.std(ddof=1) / np.sqrt(48)),
        float(diffs.mean() + crit * diffs.std(ddof=1) / np.sqrt(48)),
    ]
    assert np.allclose(inc["mass_delta_ci95"], expected_ci)

    # Mass delta >= +15, but deaths/1k worse (> 0) -> KEEP L>=600
    worse_deaths = _synthetic_units(l1_mass_delta=25.0, l1_death_delta=1)
    res_worse_deaths = analyze.analyze(worse_deaths)
    assert res_worse_deaths["decision_rule"]["mass_bar_met"] is True
    assert res_worse_deaths["decision_rule"]["deaths_bar_met"] is False
    assert res_worse_deaths["decision_rule"]["adopt_l_ge_1_for_new_banks"] is False
    assert res_worse_deaths["decision_rule"]["decision"] == "KEEP_SERVED_L2_PRIME_L_GE_600"

    # Deaths/1k no worse, but mass delta < +15 -> KEEP L>=600
    small_mass = _synthetic_units(l1_mass_delta=10.0, l1_death_delta=-1)
    res_small_mass = analyze.analyze(small_mass)
    assert res_small_mass["decision_rule"]["mass_bar_met"] is False
    assert res_small_mass["decision_rule"]["deaths_bar_met"] is True
    assert res_small_mass["decision_rule"]["adopt_l_ge_1_for_new_banks"] is False

    # Incomplete union or fixture rows refused
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        analyze.analyze(adopt_rows[:-1])
    with pytest.raises(ValueError, match="foreign namespace, fixture"):
        analyze.analyze(_synthetic_units(fixture=True))


def test_run_and_resume_ledger_skip_and_foreign_refusal(tmp_path: Path) -> None:
    contract = {
        **protocol.allocation("t34-test-alllen-resume", 4, 20, audit_live_mirror=False),
        "commit": "b" * 40,
    }
    calls: list[tuple[int, str]] = []

    def fake_run_arm(
        seed: int,
        arm: str,
        check: Any,
        *,
        horizon: int = 20,
        audit_live_mirror: bool = False,
        include_raw: bool = False,
        on_frame: Any = None,
    ) -> dict[str, Any]:
        check()
        if on_frame is not None:
            on_frame(horizon)
        calls.append((seed, arm))
        return {
            **_synthetic_arm_summary(
                arm,
                horizon=horizon,
                mass_integral=300.0 + (20.0 if arm == "exact_stop_alllen" else 0.0),
                deaths=1,
                roots_lt600=4 if arm == "exact_stop_alllen" else 0,
            ),
            "wall_seconds": 0.01,
            "roster": {"hero": "b8-s3", "opponents": ["frp3-s12"] * 5},
        }

    # Attempt 1 completes only indices 0 and 1 before mechanical interruption
    partial_contract = {**contract, "count": 2, "rows": contract["rows"][:2]}
    ledger0 = tmp_path / "ledger0.json"
    publish_json(
        ledger0,
        empty_ledger(
            contract["namespace"], protocol.UNIT_KIND, protocol.contract_identity(contract)
        ),
    )

    attempt1_out = tmp_path / "attempt1"
    attempt1_out.mkdir()
    publish_json(attempt1_out / "contract.json", contract)
    with patch.object(engine, "run_arm", side_effect=fake_run_arm):
        for row in partial_contract["rows"]:
            run.generate((contract, row, str(attempt1_out)))
    assert len(calls) == 6  # 2 units * 3 arms

    # Build verified Mac resume ledger over attempt1
    ledger1_data = resume.build(contract, [attempt1_out])
    assert [u["index"] for u in ledger1_data["units"]] == [0, 1]
    ledger1 = tmp_path / "ledger1.json"
    publish_json(ledger1, ledger1_data)

    # Duplicate export across roots must be refused
    with pytest.raises(ValueError, match="duplicate"):
        resume.build(contract, [attempt1_out, attempt1_out])

    # Foreign contract must be refused
    foreign_contract = {**contract, "commit": "c" * 40}
    with pytest.raises(ValueError, match="foreign"):
        resume.build(foreign_contract, [attempt1_out])

    # Attempt 2 skips indices 0 and 1 and runs only indices 2 and 3
    calls.clear()
    attempt2_out = tmp_path / "attempt2"
    with patch.object(engine, "run_arm", side_effect=fake_run_arm):
        completed_rows = run.run(contract, ledger1, attempt2_out, workers=1)
    assert [r["index"] for r in completed_rows] == [2, 3]
    assert len(calls) == 6  # only 2 remaining units * 3 arms

    # Union of attempt1 + attempt2 passes analyze
    union_files = analyze.collect_unit_files([attempt1_out, attempt2_out])
    assert len(union_files) == 4
    readout = analyze.analyze(
        [json.loads(p.read_text(encoding="utf-8")) for p in union_files],
        expected_namespace=contract["namespace"],
        expected_count=4,
        expected_horizon=20,
        allow_fixture=True,
    )
    assert readout["decision_rule"]["adopt_l_ge_1_for_new_banks"] is True


def test_seal_main_resolves_retry_chain(tmp_path: Path) -> None:
    ops = tmp_path / "ops"
    (ops / "manifests").mkdir(parents=True)
    out_retry = tmp_path / "canary_retry_out"
    out_retry.mkdir()
    receipt_path = out_retry / "receipt.json"
    receipt_path.write_text(json.dumps({"status": "PASS", "stop_verified": True}), encoding="utf-8")

    retry_manifest = ops / "manifests" / "canary-auto-1-retry-1.json"
    retry_manifest.write_text(json.dumps({"output_dir": str(out_retry)}), encoding="utf-8")
    queue = {
        "jobs": [
            {
                "id": "canary-01",
                "status": "FAILED",
                "retry_id": "canary-auto-1",
                "argv": ["dispatch.sh", "canary-01", "/unused.json"],
                "manifest": "/unused.json",
            },
            {
                "id": "canary-auto-1",
                "status": "PASS",
                "argv": ["dispatch.sh", "canary-auto-1", "/base.json"],
                "manifest": "/base.json",
            },
        ]
    }
    (ops / "ready-queue.json").write_text(json.dumps(queue), encoding="utf-8")
    resolved = seal_main.resolve_canary_receipt(ops, "canary-01")
    assert resolved == receipt_path

    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "main-unsealed.json").write_text(
        json.dumps({"job_id": "main-01", "script": protocol.SCRIPT}), encoding="utf-8"
    )
    with patch.object(seal_main.canary, "check") as mock_check:
        sealed_path = seal_main.seal(packet, protocol.ROOT, ops=ops, canary_id="canary-01")
        mock_check.assert_called_once()
    sealed_data = json.loads(sealed_path.read_text(encoding="utf-8"))
    assert sealed_data["canary"]["receipt"] == {
        "path": str(receipt_path),
        "sha256": digest(receipt_path),
    }
    with pytest.raises(FileExistsError):
        seal_main.seal(packet, protocol.ROOT, ops=ops, canary_id="canary-01")


def test_engine_exact_parity_with_watch_replay_and_alllen_search() -> None:
    saved = _saved_snakes_dir()
    if saved is None:
        pytest.skip("product checkpoints unavailable in test environment")
    from research.watch_replay.engine import run_arm as replay_run_arm

    seed = 42424242
    horizon = 11
    with patch.dict(os.environ, {"SNAKE_CKPT_DIR": str(saved), "SNAKE_DQN_DEVICE": "cpu"}):
        ref_bare = replay_run_arm(seed, False, lambda: None, horizon=horizon)
        our_bare = engine.run_arm(
            seed, "bare", lambda: None, horizon=horizon, audit_live_mirror=False, include_raw=True
        )
        assert our_bare["raw"]["frames"] == ref_bare["frames"]
        assert our_bare["raw"]["decisions"] == ref_bare["decisions"]
        assert our_bare["search_roots"] == 0

        ref_l2 = replay_run_arm(seed, True, lambda: None, horizon=horizon)
        our_l2 = engine.run_arm(
            seed,
            "l2_prime_600",
            lambda: None,
            horizon=horizon,
            audit_live_mirror=False,
            include_raw=True,
        )
        assert our_l2["raw"]["frames"] == ref_l2["frames"]
        assert our_l2["raw"]["decisions"] == ref_l2["decisions"]
        assert our_l2["search_roots"] == len(ref_l2["batches"]) == 0

        our_l1_fast = engine.run_arm(
            seed,
            "exact_stop_alllen",
            lambda: None,
            horizon=horizon,
            audit_live_mirror=False,
            include_raw=True,
        )
        our_l1_audited = engine.run_arm(
            seed,
            "exact_stop_alllen",
            lambda: None,
            horizon=horizon,
            audit_live_mirror=True,
            include_raw=True,
        )
        # Decisions 0, 5, 10 fire at L=10 >= 1 with interval=5 -> 3 H300 roots!
        assert our_l1_fast["search_roots"] == 3
        assert our_l1_fast["search_roots_lt600"] == 3
        assert our_l1_fast["search_roots_ge600"] == 0
        assert our_l1_fast["native_v8_kernel_rows"] > 0
        assert our_l1_audited["mirror_identity_frames"] == horizon
        assert our_l1_fast["raw"]["frames"] == our_l1_audited["raw"]["frames"]
        assert our_l1_fast["raw"]["decisions"] == our_l1_audited["raw"]["decisions"]

        def strip_rng(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{k: v for k, v in row.items() if k != "root_sha256"} for row in rows]

        assert strip_rng(our_l1_fast["raw"]["batches"]) == strip_rng(
            our_l1_audited["raw"]["batches"]
        )


def test_dense_trigger_respawn_reset_and_l600_vs_l1_boundary() -> None:
    from research.t3_rollout_labels_20261015.trigger import DenseTrigger

    trig_600 = DenseTrigger(min_length=600, interval=5)
    trig_1 = DenseTrigger(min_length=1, interval=5)

    assert trig_600.fires(0, 599, True) is False
    assert trig_1.fires(0, 10, True) is True
    assert trig_1.fires(1, 10, True) is False
    assert trig_1.fires(5, 10, True) is True
    assert trig_600.fires(5, 600, True) is True
    assert trig_600.fires(6, 600, True) is False

    # Respawn resets trigger cooldown immediately
    trig_1_respawn = DenseTrigger(min_length=1, interval=5)
    assert trig_1_respawn.fires(7, 10, True) is True


def test_prepare_packet_schema_and_canary_inventory_parity(tmp_path: Path) -> None:
    from research.colab_cli import canary

    commit = "d" * 40
    benchmark = tmp_path / "benchmark.json"
    publish_json(
        benchmark,
        {
            "commit": commit,
            "branch_timings": [{"seed": 1, "seconds": 0.40, "branches": 1}],
        },
    )
    fake_root = tmp_path / "workspace" / "repo"
    (fake_root / "web" / "backend").mkdir(parents=True)
    (fake_root / "web" / "backend" / "served_checkpoint_pins.json").write_text(
        json.dumps({"checkpoints": {}}), encoding="utf-8"
    )
    fake_saved = tmp_path / "workspace" / "snake-dqn" / "saved_snakes"
    fake_saved.mkdir(parents=True)
    for ckpt in ("b8_s3_t7_20261016.pth", "frp3_m3_s12_u60000_20261005.pth"):
        (fake_saved / ckpt).write_bytes(b"ckpt-bytes-" + ckpt.encode())

    def fake_git(cmd: list[str], **_kwargs: Any) -> str:
        if "branch" in cmd:
            return "  origin/main\n"
        return ""

    out = tmp_path / "packet-01"
    with (
        patch.object(prepare.p, "ROOT", fake_root),
        patch.object(prepare.subprocess, "check_output", side_effect=fake_git),
        patch.object(prepare, "release_bundle", side_effect=lambda b, _p: b.write_bytes(b"tar-gz")),
    ):
        forecast = prepare.prepare(out, commit, benchmark, skip_seed_audit=True)

    assert forecast["effective_timeout_minutes"] == 2 * forecast["base_minutes"]
    assert (out / "canary.json").is_file()
    assert (out / "main-unsealed.json").is_file()
    assert not (out / "main.json").exists()

    canary_m = json.loads((out / "canary.json").read_text(encoding="utf-8"))
    main_m = json.loads((out / "main-unsealed.json").read_text(encoding="utf-8"))

    assert canary.is_canary(canary_m) is True
    assert canary_m["timeout_minutes"] == 30
    assert canary_m["device"] == main_m["device"] == "cpu"
    assert canary_m["requirements"]["gpu"] == main_m["requirements"]["gpu"] == "L4"
    assert canary_m["requirements"]["min_vcpu"] == main_m["requirements"]["min_vcpu"] == 12
    assert canary.input_inventory(canary_m) == canary.input_inventory(main_m)
    assert set(main_m["required_exports"]) <= set(canary_m["required_exports"])
    assert main_m["resume"] == {
        "unit_kind": protocol.UNIT_KIND,
        "ledger_path": "inputs/main-ledger.json",
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }
    meta = main_m["watch6_alllen_screen"]
    assert main_m["timeout_minutes"] == 2 * meta["base_minutes"]
    assert main_m["budget"]["unit_cap"] == round(2.0 * meta["base_unit_cap"], 4)
    assert main_m["budget"]["reserve_units"] == round(2.0 * meta["base_reserve_units"], 4)
