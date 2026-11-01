"""Unit and integration tests for research/watch6_alllen_screen."""

from __future__ import annotations

import io
import json
import os
import tarfile
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
    prepare_resume,
    protocol,
    resume,
    run,
    seal_main,
    seed_audit,
    uploads,
)
from tools.ops import ready_queue
from tools.ops.package_spec import (
    Input,
    digest,
    empty_ledger,
    publish_json,
    stage_inputs,
    unit_id,
    verify_immutable,
)
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
    commit: str = "a" * 40,
) -> list[dict[str, Any]]:
    contract = {
        **protocol.allocation(namespace, count, horizon, audit_live_mirror=False),
        "commit": commit,
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


def _make_fake_public_catalog(root: Path, b8_bytes: bytes, frp3_bytes: bytes) -> tuple[
    Path,
    bytes,
    dict[str, tuple[str, int]],
    dict[str, tuple[int, str]],
    dict[str, tuple[str, int, str]],
]:
    b8_sha = __import__("hashlib").sha256(b8_bytes).hexdigest()
    frp3_sha = __import__("hashlib").sha256(frp3_bytes).hexdigest()
    league_buf = io.BytesIO()
    with tarfile.open(fileobj=league_buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("league/frp3_m3_s12_u60000_20261005.pth")
        info.size = len(frp3_bytes)
        tar.addfile(info, io.BytesIO(frp3_bytes))
    league_bytes = league_buf.getvalue()
    league_sha = __import__("hashlib").sha256(league_bytes).hexdigest()

    catalog = json.loads(
        (protocol.ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json").read_text(encoding="utf-8")
    )
    for asset in catalog["assets"]:
        if asset["name"] == "B8_s3_final.pth":
            asset.update(sha256=b8_sha, size_bytes=len(b8_bytes))
        elif asset["name"] == "league.tar.gz":
            asset.update(sha256=league_sha, size_bytes=len(league_bytes))
    for item in catalog["inputs"]:
        if item["name"] == "B8_s3_final.pth":
            item.update(sha256=b8_sha, size_bytes=len(b8_bytes))
        elif item["name"] == "frp3_m3_s12_u60000_20261005.pth":
            item.update(sha256=frp3_sha, size_bytes=len(frp3_bytes))
    cat_path = root / "research" / "t34_exit_b8" / "PUBLIC_INPUTS.json"
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    cat_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    pins_path = root / "web" / "backend" / "served_checkpoint_pins.json"
    pins_path.parent.mkdir(parents=True, exist_ok=True)
    pins_path.write_text(
        json.dumps(
            {
                "checkpoints": {
                    "b8-s3": {"checkpoint_sha256": b8_sha},
                    "frp3-s12": {"checkpoint_sha256": frp3_sha},
                }
            }
        ),
        encoding="utf-8",
    )
    fake_ckpts = {
        "b8_s3_t7_20261016.pth": ("b8-s3", len(b8_bytes)),
        "frp3_m3_s12_u60000_20261005.pth": ("frp3-s12", len(frp3_bytes)),
    }
    fake_assets = {
        "B8_s3_final.pth": (len(b8_bytes), b8_sha),
        "league.tar.gz": (len(league_bytes), league_sha),
    }
    fake_members = dict(uploads.public_inputs.EXPECTED_MEMBERS)
    fake_members["B8_s3_final.pth"] = ("B8_s3_final.pth", len(b8_bytes), b8_sha)
    fake_members["frp3_m3_s12_u60000_20261005.pth"] = (
        "checkpoints/frp3_m3_s12_u60000_20261005.pth",
        len(frp3_bytes),
        frp3_sha,
    )
    return cat_path, league_bytes, fake_ckpts, fake_assets, fake_members


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
    assert "research/t34_exit_b8/public_inputs.py" in paths
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

    own_seed = protocol.seed_for(protocol.NAMESPACE, 0)
    forbidden_export = (
        fake_artifacts / "t34-sol" / "watch6-efficacy-fix" / "exports" / "pair-000000.json"
    )
    forbidden_export.parent.mkdir(parents=True)
    forbidden_export.write_text(json.dumps({"seed": own_seed}), encoding="utf-8")
    loose_pair = fake_artifacts / "other" / "pair-000001.json"
    loose_pair.parent.mkdir(parents=True)
    loose_pair.write_text(json.dumps({"seed": own_seed}), encoding="utf-8")
    for own_dir in ("argon-alllen-screen-fix1", "argon-alllen-a100"):
        own_fix_bench = fake_artifacts / "t34-sol" / own_dir / "preflight" / "benchmark.json"
        own_fix_bench.parent.mkdir(parents=True)
        own_fix_bench.write_text(
            json.dumps({"seed": protocol.BRANCH_BENCHMARK_SEEDS[0]}), encoding="utf-8"
        )

    report = seed_audit.audit(repo=protocol.ROOT, artifacts=fake_artifacts, extra_worktrees=[])
    assert report["status"] == "CLEAR"
    assert len(report["main_seeds"]) == 48

    bad_inv = fake_artifacts / "frp-0" / "namespaces" / "training_exclusions.json"
    bad_inv.write_text(json.dumps([own_seed]), encoding="utf-8")
    with pytest.raises(ValueError, match="seed disjointness audit failed"):
        seed_audit.audit(repo=protocol.ROOT, artifacts=fake_artifacts, extra_worktrees=[])


def test_analyze_paired_cis_cross_commit_merge_and_decision_rule() -> None:
    l4_rows = _synthetic_units(l1_mass_delta=20.0, l1_death_delta=0, commit="5" * 40)[:9]
    a100_rows = _synthetic_units(l1_mass_delta=20.0, l1_death_delta=0, commit="f" * 40)[9:]
    merged_rows = l4_rows + a100_rows

    res_adopt = analyze.analyze(merged_rows)
    assert res_adopt["source_commits"] == ["5" * 40, "f" * 40]
    assert len(res_adopt["contract_sha256s"]) == 2
    assert res_adopt["decision_rule"]["adopt_l_ge_1_for_new_banks"] is True
    assert res_adopt["decision_rule"]["decision"] == "ADOPT_EXACT_STOP_L_GE_1_FOR_NEW_BANKS"

    inc = res_adopt["comparisons"]["exact_stop_alllen_vs_l2_prime_600"]
    diffs = np.array(
        [
            r["arms"]["exact_stop_alllen"]["mass_sum"] / 6000.0
            - r["arms"]["l2_prime_600"]["mass_sum"] / 6000.0
            for r in merged_rows
        ]
    )
    crit = float(t.ppf(0.975, 47))
    expected_ci = [
        float(diffs.mean() - crit * diffs.std(ddof=1) / np.sqrt(48)),
        float(diffs.mean() + crit * diffs.std(ddof=1) / np.sqrt(48)),
    ]
    assert np.allclose(inc["mass_delta_ci95"], expected_ci)

    # Tampered contract_identity on a cross-commit unit must be refused
    bad_commit_row = json.loads(json.dumps(merged_rows))
    bad_commit_row[0]["source_commit"] = "9" * 40
    with pytest.raises(ValueError, match="foreign namespace, fixture, or corrupted"):
        analyze.analyze(bad_commit_row)

    worse_deaths = _synthetic_units(l1_mass_delta=25.0, l1_death_delta=1)
    res_worse_deaths = analyze.analyze(worse_deaths)
    assert res_worse_deaths["decision_rule"]["adopt_l_ge_1_for_new_banks"] is False
    assert res_worse_deaths["decision_rule"]["decision"] == "KEEP_SERVED_L2_PRIME_L_GE_600"

    small_mass = _synthetic_units(l1_mass_delta=10.0, l1_death_delta=-1)
    res_small_mass = analyze.analyze(small_mass)
    assert res_small_mass["decision_rule"]["adopt_l_ge_1_for_new_banks"] is False

    with pytest.raises(ValueError, match="incomplete or duplicate"):
        analyze.analyze(merged_rows[:-1])
    with pytest.raises(ValueError, match="foreign namespace, fixture"):
        analyze.analyze(_synthetic_units(fixture=True))


def test_sharded_resume_cross_commit_ledger_and_partial_transfer_skip(tmp_path: Path) -> None:
    prior_commit = "5" * 40
    new_commit = "e" * 40
    prior_contract = {
        **protocol.allocation("t34-test-alllen-resume", 8, 20, audit_live_mirror=False),
        "commit": prior_commit,
    }
    new_contract = {**prior_contract, "commit": new_commit}
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

    # L4 attempt completes indices 0 and 2, plus leaves an incomplete partial unit-000003.json
    # WITHOUT a .complete marker (simulating mid-transfer Ctrl-C stop).
    l4_out = tmp_path / "l4_units"
    l4_out.mkdir()
    publish_json(l4_out / "contract.json", prior_contract)
    with patch.object(engine, "run_arm", side_effect=fake_run_arm):
        for idx in (0, 2):
            run.generate((prior_contract, prior_contract["rows"][idx], str(l4_out)))
    (l4_out / "unit-000003.json").write_text('{"partial": true}\n', encoding="utf-8")

    # Without allowed_commits, cross-commit is refused by default
    with pytest.raises(ValueError, match="foreign"):
        resume.build(new_contract, [l4_out])

    # With allowed_commits={prior_commit, new_commit}, indices 0 and 2 are verified, while
    # partial unit 3 (lacking .complete) is excluded so a shard will recompute it!
    ledger_data = resume.build(new_contract, [l4_out], allowed_commits={prior_commit, new_commit})
    assert [u["index"] for u in ledger_data["units"]] == [0, 2]
    ledger_path = tmp_path / "main-ledger.json"
    publish_json(ledger_path, ledger_data)

    # Duplicate across roots is still refused
    with pytest.raises(ValueError, match="duplicate"):
        resume.build(new_contract, [l4_out, l4_out], allowed_commits={prior_commit, new_commit})

    # Run 3 A100 shards over the remaining 6 unverified indices [1, 3, 4, 5, 6, 7]
    calls.clear()
    shard_dirs: list[Path] = []
    shard_indices: list[list[int]] = []
    with patch.object(engine, "run_arm", side_effect=fake_run_arm):
        for shard_idx in range(3):
            s_out = tmp_path / f"shard_{shard_idx}"
            rows = run.run(
                new_contract,
                ledger_path,
                s_out,
                workers=1,
                shard_index=shard_idx,
                shard_count=3,
            )
            shard_dirs.append(s_out)
            shard_indices.append([r["index"] for r in rows])

    assert shard_indices == [[1, 5], [3, 6], [4, 7]]
    assert len(calls) == 6 * 3  # 6 remaining units * 3 arms each, zero overlap

    # Final readout merges l4_out (ignoring partial unit-000003.json without .complete) + 3 shards
    union_files = analyze.collect_unit_files([l4_out, *shard_dirs])
    assert len(union_files) == 8
    readout = analyze.analyze(
        [json.loads(p.read_text(encoding="utf-8")) for p in union_files],
        expected_namespace=new_contract["namespace"],
        expected_count=8,
        expected_horizon=20,
        allow_fixture=True,
    )
    assert readout["source_commits"] == [prior_commit, new_commit]
    assert readout["decision_rule"]["adopt_l_ge_1_for_new_banks"] is True


def test_seal_main_and_resolve_queue_unit_dirs_follow_retry_chains(tmp_path: Path) -> None:
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
    assert analyze.resolve_queue_unit_dirs(ops, ["canary-01"]) == [out_retry / "exports" / "units"]

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

    trig_1_respawn = DenseTrigger(min_length=1, interval=5)
    assert trig_1_respawn.fires(7, 10, True) is True


def test_measured_2h_forecast_and_prepare_resume_3shards_a100(tmp_path: Path) -> None:
    from research.colab_cli import canary

    prior_commit = "5" * 40
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
    b8_bytes = b"ckpt-b8-payload"
    frp3_bytes = b"ckpt-frp3-payload"
    _, _, fake_ckpts, fake_assets, fake_members = _make_fake_public_catalog(
        fake_root, b8_bytes, frp3_bytes
    )

    def fake_git(cmd: list[str], **kwargs: Any) -> Any:
        if "branch" in cmd:
            return "  origin/main\n"
        if "show" in cmd:
            return b"# pinned control\n"
        return "" if kwargs.get("text") else b""

    def fake_source_prepare(_root: Path, _manifest: dict[str, Any], staging: Path) -> None:
        (staging / "source-plan.json").write_bytes(b"{}")
        (staging / "source.tar.gz").write_bytes(b"sparse-fallback")

    with (
        patch.object(prepare.p, "ROOT", fake_root),
        patch.object(prepare_resume.p, "ROOT", fake_root),
        patch.object(uploads.p, "ROOT", fake_root),
        patch.object(uploads, "CHECKPOINTS", fake_ckpts),
        patch.object(uploads.public_inputs, "EXPECTED_ASSETS", fake_assets),
        patch.object(uploads.public_inputs, "EXPECTED_MEMBERS", fake_members),
        patch.object(prepare.subprocess, "check_output", side_effect=fake_git),
        patch.object(prepare_resume.subprocess, "check_output", side_effect=fake_git),
        patch.object(uploads.subprocess, "check_output", side_effect=fake_git),
        patch.object(uploads.source, "prepare", side_effect=fake_source_prepare),
        patch.object(prepare, "release_bundle", side_effect=lambda b, _p: b.write_bytes(b"tar-gz")),
        patch.object(
            prepare_resume, "release_bundle", side_effect=lambda b, _p: b.write_bytes(b"tar-gz")
        ),
    ):
        # 1) Unsplit 48-unit L4 W4 packet is now REFUSED under the fixed ~2h/unit/worker forecast!
        with pytest.raises(ValueError, match="base forecast exceeds 11h"):
            prepare.prepare(tmp_path / "unsplit", commit, benchmark, skip_seed_audit=True)

        # 2) Pre-init resume_root and verify ready_queue detects A100 before manifests exist
        resume_root = tmp_path / "resume-01"
        prepare_resume.init_resume_dir(resume_root, shards=3, gpu="A100")
        assert (
            ready_queue.row_gpu(
                {"manifest": str(resume_root / "packet-canary" / "canary.json"), "argv": []}
            )
            == "A100"
        )
        for s_idx in range(3):
            assert (
                ready_queue.row_gpu(
                    {
                        "manifest": str(resume_root / f"packet-shard{s_idx}" / "main.json"),
                        "argv": [],
                    }
                )
                == "A100"
            )

        # 3) Simulate L4 prior output with 9 completed units and verified stop
        prior_out = tmp_path / "packet-02" / "output-main"
        prior_units = prior_out / "exports" / "units"
        prior_units.mkdir(parents=True)
        prior_contract = {
            **protocol.allocation(
                protocol.NAMESPACE, protocol.COUNT, protocol.HORIZON, audit_live_mirror=False
            ),
            "commit": prior_commit,
        }
        publish_json(prior_units / "contract.json", prior_contract)
        for idx in range(9):
            publish_json(prior_units / f"unit-{idx:06d}.json", {"stub": idx})
        prior_manifest_path = prior_out / "manifest.json"
        prior_manifest_path.write_text(
            json.dumps({"job_id": "watch6-alllen-main-02", "commit": prior_commit}),
            encoding="utf-8",
        )
        # Unstopped prior receipt must be refused!
        (prior_out / "receipt.json").write_text(
            json.dumps(
                {
                    "manifest_sha256": digest(prior_manifest_path),
                    "status": "INCOMPLETE",
                    "stop_verified": False,
                    "cleanup_error": "still running",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="verified stop"):
            prepare_resume.build_resume_packets(
                resume_root,
                commit=commit,
                benchmark=benchmark,
                prior_outputs=[prior_out],
                skip_seed_audit=True,
            )

        # Once verified stop is recorded, dispatch-time build_resume_packets succeeds!
        (prior_out / "receipt.json").write_text(
            json.dumps(
                {
                    "manifest_sha256": digest(prior_manifest_path),
                    "status": "INCOMPLETE",
                    "stop_verified": True,
                    "cleanup_error": None,
                    "error": "KeyboardInterrupt()",
                }
            ),
            encoding="utf-8",
        )
        plan = prepare_resume.build_resume_packets(
            resume_root,
            commit=commit,
            benchmark=benchmark,
            prior_outputs=[prior_out],
            shards=3,
            gpu="A100",
            workers=6,
            skip_seed_audit=True,
        )

    assert plan["verified_count"] == 9
    assert plan["remaining_count"] == 39
    assert [s["assigned_count"] for s in plan["shard_plans"]] == [13, 13, 13]
    # 13 units at W=6 -> 3 waves * (7200 * 4.1444/4.0180)/60 + 25 = ceil(371.32 + 25) = 397 min
    assert [s["base_minutes"] for s in plan["shard_plans"]] == [397, 397, 397]
    assert [s["timeout_minutes"] for s in plan["shard_plans"]] == [794, 794, 794]

    union_assigned = [idx for s in plan["shard_plans"] for idx in s["assigned_indices"]]
    assert len(union_assigned) == 39
    assert set(union_assigned) == set(range(9, 48))

    canary_m = json.loads(
        (resume_root / "packet-canary" / "canary.json").read_text(encoding="utf-8")
    )
    assert canary.is_canary(canary_m) is True
    assert canary_m["requirements"]["gpu"] == "A100"
    for s_idx in range(3):
        shard_m = json.loads(
            (resume_root / f"packet-shard{s_idx}" / "main-unsealed.json").read_text(
                encoding="utf-8"
            )
        )
        assert shard_m["requirements"]["gpu"] == "A100"
        assert shard_m["env"]["WATCH6_ALLLEN_WORKERS"] == "6"
        assert shard_m["env"]["WATCH6_ALLLEN_SHARD_INDEX"] == str(s_idx)
        assert shard_m["env"]["WATCH6_ALLLEN_SHARD_COUNT"] == "3"
        assert canary.input_inventory(canary_m) == canary.input_inventory(shard_m)


def test_real_upload_budget_le8_le60s_and_https_roundtrip(tmp_path: Path) -> None:
    import subprocess as sp

    from research.watch_replay.prepare import release_bundle

    commit = sp.check_output(["git", "rev-parse", "HEAD"], cwd=protocol.ROOT, text=True).strip()
    build = tmp_path / "build"
    build.mkdir()

    inputs: list[Input] = []
    for label, ns, cnt, hor, mirror in (
        ("main", protocol.NAMESPACE, protocol.COUNT, protocol.HORIZON, False),
        (
            "canary",
            protocol.CANARY_NAMESPACE,
            protocol.CANARY_COUNT,
            protocol.CANARY_HORIZON,
            True,
        ),
    ):
        contract = {**protocol.allocation(ns, cnt, hor, audit_live_mirror=mirror), "commit": commit}
        publish_json(build / f"{label}-contract.json", contract)
        publish_json(
            build / f"{label}-ledger.json",
            empty_ledger(
                contract["namespace"], protocol.UNIT_KIND, protocol.contract_identity(contract)
            ),
        )
        inputs.extend(
            [
                Input(build / f"{label}-contract.json", f"{label}-contract.json"),
                Input(build / f"{label}-ledger.json", f"{label}-ledger.json"),
            ]
        )

    pins = json.loads(
        (protocol.ROOT / "web/backend/served_checkpoint_pins.json").read_text(encoding="utf-8")
    )
    release_bundle(build / "release-evidence.tar.gz", pins)
    inputs.append(Input(build / "release-evidence.tar.gz", "release-evidence.tar.gz"))
    inputs.append(
        Input(protocol.ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json", "public-inputs.json")
    )

    packed = uploads.pack_inputs(stage_inputs(build / "staged", inputs), build / uploads.METADATA)
    assert len(packed) == 1
    metadata_row = packed[0]
    assert metadata_row["dest"] == uploads.METADATA
    assert Path(metadata_row["path"]).stat().st_size <= uploads.CHUNK_BYTES

    tracked = set(
        sp.check_output(
            ["git", "ls-tree", "-r", "--name-only", commit], cwd=protocol.ROOT, text=True
        ).splitlines()
    )
    source_paths = [
        sp_item
        for sp_item in protocol.source_paths()
        if sp_item in tracked or any(t.startswith(sp_item + "/") for t in tracked)
    ]
    manifest = {
        "commit": commit,
        "script": protocol.SCRIPT,
        "source_paths": source_paths,
        "max_export_bytes": uploads.MAX_FILE_BYTES,
        "inputs": [{k: row[k] for k in ("path", "dest", "sha256")} for row in packed],
    }
    report = uploads.check_uploads(manifest)
    assert report["status"] == "PASS"
    assert report["normal_transfer_count"] == 3 <= 8
    assert report["transfer_count"] == 4 <= 8
    assert report["chunk_put_count"] <= 8
    assert report["normal_estimated_transfer_seconds"] <= 60.0
    assert report["estimated_transfer_seconds"] <= 60.0

    # Test HTTPS checkpoint materialization via mocked opener (no live network call)
    fake_root = tmp_path / "fake_repo"
    b8_bytes = b"mock-b8-checkpoint-bytes"
    frp3_bytes = b"mock-frp3-checkpoint-bytes"
    cat_path, league_bytes, fake_ckpts, fake_assets, fake_members = _make_fake_public_catalog(
        fake_root, b8_bytes, frp3_bytes
    )

    build2 = tmp_path / "build2"
    build2.mkdir()
    inputs2 = [item for item in inputs if item.destination != "public-inputs.json"] + [
        Input(cat_path, "public-inputs.json")
    ]
    with (
        patch.object(uploads.p, "ROOT", fake_root),
        patch.object(uploads, "CHECKPOINTS", fake_ckpts),
        patch.object(uploads.public_inputs, "EXPECTED_ASSETS", fake_assets),
        patch.object(uploads.public_inputs, "EXPECTED_MEMBERS", fake_members),
    ):
        packed2 = uploads.pack_inputs(
            stage_inputs(build2 / "staged", inputs2), build2 / uploads.METADATA
        )
        runtime_in = tmp_path / "colab_input"
        runtime_in.mkdir()
        os.link(Path(packed2[0]["path"]), runtime_in / uploads.METADATA)

        class _Resp(io.BytesIO):
            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *args: object) -> None:
                self.close()

        def fake_opener(url: str, timeout: int = 60) -> _Resp:
            assert timeout == 60
            return _Resp(b8_bytes if url.endswith("B8_s3_final.pth") else league_bytes)

        with pytest.raises(ValueError, match="SHA256"):
            uploads.stage_metadata(runtime_in, "0" * 64, opener=fake_opener)

        uploads.stage_metadata(runtime_in, packed2[0]["sha256"], opener=fake_opener)
        for name in uploads.SMALL_PAYLOADS:
            verify_immutable(runtime_in / name)
        assert (runtime_in / "checkpoints/b8_s3_t7_20261016.pth").read_bytes() == b8_bytes
        assert (runtime_in / "checkpoints/frp3_m3_s12_u60000_20261005.pth").read_bytes() == (
            frp3_bytes
        )

    # Budget refusals: >2 MiB unsplit local input, >8 transfers, and >60 s transfer estimate.
    over2m = tmp_path / "over2m.bin"
    over2m.write_bytes(b"x" * (uploads.CHUNK_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds 2 MiB single-chunk bound"):
        uploads.check_uploads(
            {
                **manifest,
                "inputs": [
                    *manifest["inputs"],
                    {"path": str(over2m), "dest": "big.bin", "sha256": "0" * 64},
                ],
            }
        )

    tiny = tmp_path / "tiny.bin"
    tiny.write_bytes(b"ok")
    with pytest.raises(ValueError, match="upload transfer forecast refuses offline"):
        uploads.check_uploads(
            {
                **manifest,
                "inputs": [
                    *manifest["inputs"],
                    *[
                        {"path": str(tiny), "dest": f"extra-{i}.bin", "sha256": "0" * 64}
                        for i in range(6)
                    ],
                ],
            }
        )
