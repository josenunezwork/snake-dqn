"""Focused unit tests for research/watch6_l1_confirm (protocol, stats, costs, sole final look)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.bl_served_search.b2_engine import FRP3
from research.bl_served_search.ceiling import B8_SHA256
from research.colab_cli import canary
from research.watch6_l1_confirm import (
    admission,
    analyze,
    costs,
    final_look,
    mirror,
)
from research.watch6_l1_confirm import protocol as p
from research.watch6_l1_confirm import (
    ratify,
    resume,
    uploads,
)
from research.watch6_l1_confirm.run import DEFAULT_LOCAL_SAVED_DIR
from tools.ops.package_spec import (
    Input,
    PackageRefused,
    digest,
    empty_ledger,
    publish_json,
    stage_inputs,
    verify_immutable,
)


def _synthetic_arm_row(
    arm: str,
    *,
    horizon: int = p.HORIZON,
    deaths: int = 2,
    mass_per_frame: float = 500.0,
    deaths_ge800: int = 0,
) -> dict:
    is_bare = arm == "bare"
    batches = 0 if is_bare else 1200
    return {
        "arm": arm,
        "frames": horizon,
        "alive_frames": horizon - deaths * 30,
        "mass_sum": int(round(mass_per_frame * horizon)),
        "deaths": deaths,
        "decisions": horizon - deaths * 30,
        "boost_decisions": 100,
        "action_counts": [200, 12000, 200, 30, 40, 30],
        "length_median": mass_per_frame,
        "alive_length_median": mass_per_frame,
        "length_p95": mass_per_frame * 1.2,
        "max_length": int(mass_per_frame * 1.4),
        "death_lengths": ([900] * deaths_ge800) + ([300] * max(0, deaths - deaths_ge800)),
        "short_body_deaths_lt600": max(0, deaths - deaths_ge800),
        "long_body_deaths_ge600": deaths_ge800,
        "late_large_body_deaths": deaths_ge800,
        "batches": batches,
        "batches_lt600": 0 if arm != "l1" else 600,
        "batches_ge600": batches if arm == "l2_prime" else (600 if arm == "l1" else 0),
        "interventions": 0 if is_bare else 150,
        "interventions_lt600": 0 if arm != "l1" else 80,
        "interventions_ge600": 0 if is_bare else 70,
        "branches_evaluated": 0 if is_bare else 1400,
        "branches_skipped": 0 if is_bare else 2200,
        "native_v8_kernel_rows": 0 if is_bare else 80000,
        "mirror_identity_frames": horizon,
        "respawns_per_slot": [deaths, 3, 2, 4, 1, 2],
        "timing": {"main_seconds": 10.0, "search_seconds": 0.0 if is_bare else 90.0},
    }


def _synthetic_panel(
    *,
    bare_deaths_fn=lambda i: 4 + (i % 3),
    l2p_deaths_fn=lambda i: 2 + (i % 4),
    l1_deaths_fn=lambda i: i % 2,
    bare_mass_fn=lambda i: 380.0 + (i % 7) * 5.0,
    l2p_mass_fn=lambda i: 610.0 + (i % 7) * 6.0,
    l1_mass_fn=lambda i: 700.0 + (i % 7) * 6.0,
    l2p_ge800_fn=lambda i: 0,
    l1_ge800_fn=lambda i: 0,
) -> list[dict]:
    alloc = p.allocation()
    panel: list[dict] = []
    for row in alloc:
        i = row["index"]
        panel.append(
            {
                "schema": "watch6-l1-world/v1",
                "namespace": p.NAMESPACE,
                "science": p.identity(),
                "seed_kind": "primary",
                "index": i,
                "seed": row["seed"],
                "horizon": p.HORIZON,
                "fixture": False,
                "record_mode": None,
                "arm_order": row["arm_order"],
                "roster": {
                    "hero_checkpoint_sha256": B8_SHA256,
                    "opponent_checkpoint_sha256": FRP3,
                    "opponent_count": 5,
                    "veto_variant": "v8",
                    "veto_lambda": 8.0,
                    "mechanics_version": 2,
                    "num_snakes": 6,
                },
                "arms": {
                    "bare": _synthetic_arm_row(
                        "bare",
                        deaths=bare_deaths_fn(i),
                        mass_per_frame=bare_mass_fn(i),
                        deaths_ge800=0,
                    ),
                    "l2_prime": _synthetic_arm_row(
                        "l2_prime",
                        deaths=l2p_deaths_fn(i),
                        mass_per_frame=l2p_mass_fn(i),
                        deaths_ge800=l2p_ge800_fn(i),
                    ),
                    "l1": _synthetic_arm_row(
                        "l1",
                        deaths=l1_deaths_fn(i),
                        mass_per_frame=l1_mass_fn(i),
                        deaths_ge800=l1_ge800_fn(i),
                    ),
                },
                "elapsed_seconds": 120.0,
            }
        )
    return panel


def test_protocol_allocation_permutation_balance_and_sharding() -> None:
    alloc = p.allocation()
    assert len(alloc) == 96
    seeds = [row["seed"] for row in alloc]
    assert len(set(seeds + p.FIXTURE_SEEDS)) == 99

    # Exact 16 worlds per 3-arm execution permutation overall; 12 on A100, 4 on L4.
    overall_counts = {perm: 0 for perm in p.ARM_ORDER_PERMUTATIONS}
    a100_counts = {perm: 0 for perm in p.ARM_ORDER_PERMUTATIONS}
    l4_counts = {perm: 0 for perm in p.ARM_ORDER_PERMUTATIONS}

    covered: list[int] = []
    for shard in range(p.SHARDS):
        spec = p.shard_spec(shard)
        indices = spec["indices"]
        covered.extend(indices)
        for idx in indices:
            assert p.owner_shard_for_index(idx) == shard
            perm = tuple(alloc[idx]["arm_order"])
            overall_counts[perm] += 1
            if spec["gpu"] == "A100":
                a100_counts[perm] += 1
            else:
                l4_counts[perm] += 1
        if spec["gpu"] == "A100":
            assert spec["workers"] == 6 and len(indices) == 6
            # Every single A100 shard has all 6 permutations once.
            assert {tuple(alloc[i]["arm_order"]) for i in indices} == set(p.ARM_ORDER_PERMUTATIONS)
        else:
            assert spec["workers"] == 4 and len(indices) == 4

    assert sorted(covered) == list(range(96))
    assert set(overall_counts.values()) == {16}
    assert set(a100_counts.values()) == {12}
    assert set(l4_counts.values()) == {4}


def test_science_identity_decoupled_from_ops_and_colab_cli() -> None:
    sci = set(p.science_paths())
    assert "research/watch6_l1_confirm/REGISTRATION_L1_V1.md" in sci
    assert "research/watch6_l1_confirm/protocol.py" in sci
    assert "research/watch6_l1_confirm/engine.py" in sci
    assert "research/watch6_l1_confirm/analyze.py" in sci
    assert "configs/mechanics_v2.yaml" in sci
    assert not any(path.startswith("tools/ops/") for path in sci)
    assert not any(path.startswith("research/colab_cli/") for path in sci)
    assert "web/backend/watch_replay.py" not in sci
    assert "web/backend/app.py" not in sci
    assert "research/watch6_l1_confirm/admission.py" not in sci
    assert "research/watch6_l1_confirm/prepare.py" not in sci
    assert "research/watch6_l1_confirm/finalize.py" not in sci
    assert "research/watch6_l1_confirm/final_look.py" not in sci
    assert "research/watch6_l1_confirm/AMENDMENT_L4_REASSIGN_01.json" not in sci
    assert "research/watch6_l1_confirm/INCIDENT_HOST_RESTART_01.json" not in sci
    assert p.MAX_COMPUTE_COMPLETIONS_PER_SHARD == 2
    assert p.identity() == "57ec544fe45f7a8a876dbc9363048468b63358e224b2ee8e11dff41401dcfbd3"


def test_analyze_fixed_sequence_gates_and_parameterized_alpha() -> None:
    # 1. Both Gate 1 and Gate 2 pass -> GO_L1_SUPERIOR (0 vs 0 L>=800 deaths zero-variance pass).
    panel_go = _synthetic_panel()
    res_go = analyze.analyze_worlds(panel_go, alpha=0.05)
    assert res_go["verdict"] == "GO_L1_SUPERIOR"
    assert res_go["gate1_l1_vs_bare"]["passed"] is True
    assert res_go["gate2_l1_vs_l2_prime"]["tested_in_fixed_sequence"] is True
    assert res_go["gate2_l1_vs_l2_prime"]["passed"] is True
    assert res_go["gate2_l1_vs_l2_prime"]["bars"]["session_d_ge800_per_1k_ub_le_pos_0p05"][
        "ci"
    ] == [0.0, 0.0]

    # 2. Parameterized alpha monotonicity: narrower alpha (0.001) produces strictly wider CIs.
    res_strict = analyze.analyze_worlds(panel_go, alpha=0.001)
    ci_05 = res_go["gate1_l1_vs_bare"]["bars"]["deaths_per_1k_ub_lt_0"]["ci"]
    ci_001 = res_strict["gate1_l1_vs_bare"]["bars"]["deaths_per_1k_ub_lt_0"]["ci"]
    assert (ci_001[1] - ci_001[0]) > (ci_05[1] - ci_05[0])
    assert res_strict["alpha"] == 0.001
    with pytest.raises(ValueError):
        analyze.analyze_worlds(panel_go, alpha=0.0)
    with pytest.raises(ValueError):
        analyze.analyze_worlds(panel_go, alpha=1.0)
    with pytest.raises(ValueError):
        analyze.analyze_worlds(panel_go, alpha=True)  # type: ignore[arg-type]

    # 3. Borderline effect that passes Gate 1 at alpha=0.10; INCONCLUSIVE_GATE1 on noisy null.
    rng = np.random.default_rng(20261101)
    border_noise = rng.integers(-2, 3, size=96)
    panel_border = _synthetic_panel(
        bare_deaths_fn=lambda i: int(5 + border_noise[i]),
        l1_deaths_fn=lambda i: int(max(0, 5 + border_noise[i] - (1 if i < 20 else 0))),
    )
    res_a10 = analyze.analyze_worlds(panel_border, alpha=0.10)
    res_a0001 = analyze.analyze_worlds(
        _synthetic_panel(
            bare_deaths_fn=lambda i: 4 + (i % 5),
            l1_deaths_fn=lambda i: 4 + ((i + 1) % 5),
        ),
        alpha=0.05,
    )
    assert res_a10["gate1_l1_vs_bare"]["passed"] is True
    assert res_a0001["verdict"] == "INCONCLUSIVE_GATE1"
    assert res_a0001["gate2_l1_vs_l2_prime"]["tested_in_fixed_sequence"] is False
    assert res_a0001["gate2_l1_vs_l2_prime"]["passed"] is False

    # 4. Fixed-sequence hierarchy: Gate 1 FAILS (l1 worse than bare) while l1 beats l2_prime!
    # Gate 2 MUST NOT be tested or pass!
    panel_g1_fail = _synthetic_panel(
        bare_deaths_fn=lambda i: i % 2,
        l2p_deaths_fn=lambda i: 6 + (i % 4),
        l1_deaths_fn=lambda i: 3 + (i % 3),
    )
    res_g1_fail = analyze.analyze_worlds(panel_g1_fail, alpha=0.05)
    assert res_g1_fail["verdict"] == "FAIL_GATE1"
    assert res_g1_fail["gate2_l1_vs_l2_prime"]["unconditional_bars_met_diagnostic"] is True
    assert res_g1_fail["gate2_l1_vs_l2_prime"]["tested_in_fixed_sequence"] is False
    assert res_g1_fail["gate2_l1_vs_l2_prime"]["passed"] is False
    assert res_g1_fail["gate2_l1_vs_l2_prime"]["outcome"] == "NOT_TESTED_GATE1_DID_NOT_PASS"

    # 5. Gate 1 passes, Gate 2 fails SOLELY on late large-body L>=800 death guard (> +0.05/1k).
    panel_ge800_trip = _synthetic_panel(
        l2p_ge800_fn=lambda _i: 0,
        l1_ge800_fn=lambda i: 1 + (i % 2),  # ~1.5 / 14.4k = +0.104/1k > +0.05/1k
    )
    res_ge800 = analyze.analyze_worlds(panel_ge800_trip, alpha=0.05)
    assert res_ge800["gate1_l1_vs_bare"]["passed"] is True
    assert res_ge800["verdict"] == "KEEP_L2_PRIME_GATE1_ONLY"
    assert (
        res_ge800["gate2_l1_vs_l2_prime"]["bars"]["session_d_ge800_per_1k_ub_le_pos_0p05"][
            "bar_met"
        ]
        is False
    )

    # 6. Degenerate constant superiority (-1 in every session) rejected for UB < 0.
    panel_degen = _synthetic_panel(
        bare_deaths_fn=lambda _i: 3,
        l1_deaths_fn=lambda _i: 2,
    )
    res_degen = analyze.analyze_worlds(panel_degen, alpha=0.05)
    assert res_degen["gate1_l1_vs_bare"]["passed"] is False


def test_costs_forecast_and_envelopes() -> None:
    report = costs.build_cost_report()
    forecast = costs.empirical_forecast(report)
    assert forecast["shards"] == 18
    assert forecast["count"] == 96
    a100 = forecast["per_gpu_envelopes"]["A100"]
    l4 = forecast["per_gpu_envelopes"]["L4"]
    assert a100["priced_ceiling_rate_u_per_h"] >= a100["observed_rate_u_per_h"]
    assert l4["priced_ceiling_rate_u_per_h"] >= l4["observed_rate_u_per_h"]
    assert a100["timeout_minutes_2x"] == 2 * a100["base_minutes"] <= 1320
    assert l4["timeout_minutes_2x"] == 2 * l4["base_minutes"] <= 1320
    reg = forecast["registered_plan"]
    assert reg["a100_worlds"] + reg["l4_worlds"] == 96
    for alt in forecast["alternative_6runtime_plans"]:
        assert alt["a100_worlds"] + alt["l4_worlds"] == 96


def test_uploads_pack_unpack_and_ratify(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    for name in uploads.SMALL_PAYLOADS:
        if name == "public-inputs.json":
            data = (p.ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json").read_bytes()
            (src_dir / name).write_bytes(data)
        else:
            publish_json(src_dir / name, {"name": name})
    inputs = [Input(src_dir / name, name) for name in uploads.SMALL_PAYLOADS]
    staged = stage_inputs(tmp_path / "staged", inputs)
    packed = uploads.pack_inputs(staged, tmp_path / uploads.METADATA)
    assert len(packed) == 1
    assert packed[0]["size_bytes"] <= uploads.CHUNK_BYTES

    unpack_dir = tmp_path / "unpacked"
    unpack_dir.mkdir()
    (unpack_dir / uploads.METADATA).write_bytes((tmp_path / uploads.METADATA).read_bytes())
    uploads.stage_metadata(unpack_dir, packed[0]["sha256"], materialize_checkpoints=False)
    for name in uploads.SMALL_PAYLOADS:
        assert (unpack_dir / f"{name}.complete").is_file()

    # Ratify parser & receipt check.
    bad_md = tmp_path / "bad_ruling.md"
    bad_md.write_text("# Ruling\nStatus: REJECTED\nVerdict: REJECT\n", encoding="utf-8")
    with pytest.raises(ValueError, match="REFUSE"):
        ratify.record_ruling(bad_md, tmp_path / "bad.json")

    good_md = tmp_path / "good_ruling.md"
    good_md.write_text(
        "# Watch6-L1-V1 Pre-Data Ruling\nStatus: RATIFIED\nVerdict: APPROVE\n",
        encoding="utf-8",
    )
    rec = ratify.record_ruling(good_md, tmp_path / "ruling.json", judge="Astra")
    assert rec["verdict"] == "APPROVE"
    assert ratify.validate_ruling_receipt(tmp_path / "ruling.json")["status"] == "RATIFIED"


def _make_metadata_archive(
    dest_dir: Path,
    *,
    mirror_science: str | None = None,
) -> dict:
    """Create a valid single-chunk Watch6-L1-V1 input-metadata.tar.gz for final_look tests."""
    src_dir = dest_dir / "src"
    src_dir.mkdir(parents=True)
    sci = p.identity() if mirror_science is None else mirror_science
    mirror_payload = {
        "schema": mirror.SCHEMA,
        "status": "PASS",
        "platform": "macOS-test",
        "namespace": p.NAMESPACE,
        "science": sci,
        "fixture": True,
        "seed": p.FIXTURE_SEEDS[0],
        "horizon": p.HORIZON,
        "total_mirrored_frames": p.HORIZON * len(p.ARMS),
        "mirror_mismatches": 0,
        "elapsed_seconds": 12.0,
        "arms": {
            arm: {
                "frames": p.HORIZON,
                "mirror_identity_frames": p.HORIZON,
                "mirror_mismatches": 0,
                "respawns_per_slot": [1, 1, 1, 1, 1, 1],
                "batches": 0 if arm == "bare" else 1200,
                "native_v8_kernel_rows": 0 if arm == "bare" else 90000,
                "wall_seconds": 4.0,
                "timing": {"main_seconds": 2.0, "search_seconds": 0.0 if arm == "bare" else 2.0},
            }
            for arm in p.ARMS
        },
    }
    for name in uploads.SMALL_PAYLOADS:
        if name == "public-inputs.json":
            data = (p.ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json").read_bytes()
            (src_dir / name).write_bytes(data)
            publish_json(src_dir / "public-inputs-check.json", {"ok": True})
            (src_dir / f"{name}.sha256").write_text(
                f"{digest(src_dir / name)}  {name}\n",
                encoding="utf-8",
            )
            (src_dir / f"{name}.complete").write_text("ok\n", encoding="utf-8")
        elif name == "mac-mirror.json":
            publish_json(src_dir / name, mirror_payload)
        else:
            publish_json(src_dir / name, {"name": name, "science": p.identity()})
    inputs = [Input(src_dir / name, name) for name in uploads.SMALL_PAYLOADS]
    staged = stage_inputs(dest_dir / "staged", inputs)
    packed = uploads.pack_inputs(staged, dest_dir / uploads.METADATA)
    return packed[0]


def test_trigger_clock_anchoring_on_empty_advisory_matches_v3_and_screen_engines(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Pin Finding 1: empty advisory on first eligible decision anchors 5-decision clock."""
    import torch

    from research.watch6_alllen_screen import engine as screen_engine
    from research.watch6_l1_confirm import engine as l1_engine
    from research.watch6_search_efficacy import engine as v3_engine

    dummy_hero_ckpt = tmp_path / "hero.pth"
    dummy_opp_ckpt = tmp_path / "opp.pth"
    dummy_hero_ckpt.write_bytes(b"hero")
    dummy_opp_ckpt.write_bytes(b"opp")
    q_vec = np.array([0.1, 0.9, 0.2, -0.5, -0.4, -0.3], dtype=np.float32)

    # 16-frame scripted trajectory covering two lives with a respawn at frame 10:
    # Each tuple is (choose_alive, choose_len, adv_any, post_alive, post_len).
    def _build_steps(low_len: int, eligible_len: int) -> list[tuple[bool, int, bool, bool, int]]:
        return [
            # Life 1: frame 1 below threshold (for l2_prime); frame 2 first eligible w/ EMPTY adv!
            (True, low_len, True, True, eligible_len),  # frame 1 (d=0)
            (True, eligible_len, False, True, eligible_len + 5),  # frame 2 (d=1, EMPTY adv)
            (True, eligible_len + 5, True, True, eligible_len + 10),  # frame 3 (d=2)
            (True, eligible_len + 10, True, True, eligible_len + 15),  # frame 4 (d=3)
            (True, eligible_len + 15, True, True, eligible_len + 20),  # frame 5 (d=4)
            (True, eligible_len + 20, True, True, eligible_len + 25),  # frame 6 (d=5)
            (True, eligible_len + 25, True, True, eligible_len + 30),  # frame 7 (d=6)
            (True, eligible_len + 30, True, False, 0),  # frame 8 (d=7, dies at end of step)
            (False, 0, False, False, 0),  # frame 9 (dead cooldown)
            # Life 2: respawns pre-choose on frame 10 at eligible_len w/ EMPTY advisory!
            (True, eligible_len, False, True, eligible_len + 5),  # frame 10 (d=8, EMPTY adv)
            (True, eligible_len + 5, True, True, eligible_len + 10),  # frame 11 (d=9)
            (True, eligible_len + 10, True, True, eligible_len + 15),  # frame 12 (d=10)
            (True, eligible_len + 15, True, True, eligible_len + 20),  # frame 13 (d=11)
            (True, eligible_len + 20, True, True, eligible_len + 25),  # frame 14 (d=12)
            (True, eligible_len + 25, True, True, eligible_len + 30),  # frame 15 (d=13)
            (True, eligible_len + 30, True, True, eligible_len + 35),  # frame 16 (d=14)
        ]

    class _FakeSnake:
        def __init__(self, idx: int, length: int) -> None:
            self.id = idx
            self.auto_respawn = True
            self.is_alive = True
            self.length = length
            self.last_action = 1

        def update(self, *_args, **_kwargs) -> None:
            return None

    class _FakeSim:
        def __init__(self, steps: list[tuple[bool, int, bool, bool, int]]) -> None:
            self._steps = steps
            self.tick = 0
            self.allow_respawn = True
            self.S = 6
            self.alive = np.ones((1, 6), dtype=bool)
            self.length = np.full((1, 6), steps[0][1], dtype=np.int32)
            self._adv_any = True
            self._transition_valid = np.zeros((1, 6), dtype=bool)
            self.last_step = None

        def get_advisory_action_mask(self) -> np.ndarray:
            mask = np.zeros((1, 6, 6), dtype=bool)
            if self._adv_any:
                mask[0, :, :3] = True
            return mask

        def get_resolved_action_mask(self) -> np.ndarray:
            return np.ones((1, 6, 6), dtype=bool)

        def get_transition_valid(self) -> np.ndarray:
            return self._transition_valid

        def step_with_policy(self, choose_fn) -> None:
            c_alive, c_len, adv_any, p_alive, p_len = self._steps[self.tick]
            self.alive[0, 0] = c_alive
            self.length[0, 0] = c_len
            self._adv_any = adv_any
            actions = choose_fn(self)
            self.last_step = (c_alive, c_len, p_alive, p_len, int(actions[0, 0]))
            self._transition_valid[0, 0] = c_alive
            self.alive[0, 0] = p_alive
            self.length[0, 0] = p_len
            self.tick += 1

    class _FakeSession:
        def __init__(self, steps: list[tuple[bool, int, bool, bool, int]]) -> None:
            sim = _FakeSim(steps)
            runtime = type(
                "_R",
                (),
                {
                    "prepare": staticmethod(lambda _s: None),
                    "observe_step": staticmethod(lambda _s: None),
                },
            )()
            self.world = type(
                "_W",
                (),
                {"sim": sim, "runtime": runtime, "hero": object(), "opponents": object()},
            )()
            snakes = [_FakeSnake(i, steps[0][1]) for i in range(6)]
            game = type("_G", (), {"snakes": snakes, "frame_death_causes": {}})()
            policy = type(
                "_P",
                (),
                {"_next_dispatch_q": staticmethod(lambda *_a, **_k: torch.tensor(q_vec))},
            )()
            self.live = type(
                "_L",
                (),
                {
                    "game": game,
                    "pathb": None,
                    "policy": policy,
                    "served_checkpoint": {"path": str(dummy_hero_ckpt)},
                    "watch_opponent_checkpoint": {"path": str(dummy_opp_ckpt)},
                    "snapshot": staticmethod(lambda: {}),
                },
            )()

        def _step_live(self, actions: np.ndarray) -> None:
            c_alive, c_len, p_alive, p_len, _ = self.world.sim.last_step
            hero = self.live.game.snakes[0]
            if c_alive:
                hero.is_alive = True
                hero.length = c_len
                hero.update()
                self.live.policy._next_dispatch_q()
                hero.last_action = int(actions[0, 0])
            hero.is_alive = p_alive
            hero.length = p_len
            self.live.game.frame_death_causes = {0: "body"} if (c_alive and not p_alive) else {}

        def close(self) -> None:
            return None

    active_steps: list[tuple[bool, int, bool, bool, int]] = []
    fired_roots: list[int] = []

    def _fake_open_watch(*_args, **_kwargs):
        return _FakeSession(active_steps)

    def _fake_dense_batch(_world, root_frame: int, _q, legal):
        fired_roots.append(int(root_frame))
        # Pick the highest True index in boolean mask `legal` (action 2 >= 2) to pin boolean
        # mask indexing (`bool(legal[chosen])`, NOT `chosen in legal` which fails for >=2!).
        legal_indices = np.flatnonzero(legal).tolist()
        return {
            "chosen": int(legal_indices[-1]),
            "branches_evaluated": 3,
            "branches_skipped": 0,
            "native_v8": {
                "kernel_rows": 900,
                "stock_rows": 0,
                "fallback_rows": 0,
                "noncandidate_rows": 0,
            },
            "outcomes": {},
        }

    def _fake_dispatch(sim, _mask, actions, _heroes, _opps):
        from research.bl_served_search.ceiling import _legal

        _, bare_act = _legal(q_vec, sim.get_advisory_action_mask()[0, 0])
        actions[:, :] = int(bare_act)

    monkeypatch.setattr("research.watch6_search_efficacy.product.open_watch", _fake_open_watch)
    monkeypatch.setattr(l1_engine, "open_watch", _fake_open_watch)
    monkeypatch.setattr(screen_engine, "open_watch", _fake_open_watch)
    monkeypatch.setattr(l1_engine, "dense_batch", _fake_dense_batch)
    monkeypatch.setattr(v3_engine, "dense_batch", _fake_dense_batch)
    monkeypatch.setattr(screen_engine, "dense_batch", _fake_dense_batch)
    monkeypatch.setattr(
        "research.t3_rollout_labels_20261015.bank_engine.prepared_q",
        lambda *_a, **_k: q_vec,
    )
    monkeypatch.setattr(v3_engine, "prepared_q", lambda *_a, **_k: q_vec)
    monkeypatch.setattr(screen_engine, "prepared_q", lambda *_a, **_k: q_vec)
    monkeypatch.setattr(
        "research.t15_mixed_roster_20261017.episodes._assert_mixed",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "research.t15_mixed_roster_20261017.episodes.roster_identity",
        lambda *_a, **_k: {"hero": "b8"},
    )
    monkeypatch.setattr("src.simd_env.ego_raster.build_ego_raster", lambda *_a, **_k: object())
    monkeypatch.setattr("src.simd_env.eval_engine._dispatch_actions", _fake_dispatch)

    # 1. l2_prime (min_length=600, interval=5) vs V3 confirmed engine:
    # Life 1 first eligible decision is d=1 (frame 2, empty advisory) -> fires at d=6 (root 6).
    # Life 2 first eligible decision is d=8 (frame 10, empty advisory) -> fires at d=13 (root 14).
    active_steps = _build_steps(low_len=500, eligible_len=600)
    fired_roots = []
    v3_engine.run_arm(123, True, lambda: None, horizon=16)
    v3_roots = list(fired_roots)

    fired_roots = []
    l2p_res = l1_engine.run_arm(123, "l2_prime", lambda: None, horizon=16)
    l2p_roots = list(fired_roots)

    assert v3_roots == [6, 14]
    assert l2p_roots == v3_roots
    assert l2p_res["batches"] == 2
    assert l2p_res["action_counts"][2] == 2

    # 2. l1 (min_length=1, interval=5) vs watch6_alllen_screen engine:
    # Frame 1 (d=0) has empty advisory at L=30 -> anchors at d=0, fires at d=5 (root 5);
    # Life 2 respawns at frame 10 (d=8, empty advisory) -> anchors at d=8, fires at d=13 (root 14).
    l1_steps = list(_build_steps(low_len=30, eligible_len=35))
    l1_steps[0] = (True, 30, False, True, 35)  # frame 1 (d=0, EMPTY advisory)
    l1_steps[1] = (True, 35, True, True, 40)  # frame 2 (d=1, non-empty advisory)
    active_steps = l1_steps

    fired_roots = []
    screen_engine.run_arm(
        123, "exact_stop_alllen", lambda: None, horizon=16, audit_live_mirror=True
    )
    screen_roots = list(fired_roots)

    fired_roots = []
    l1_res = l1_engine.run_arm(123, "l1", lambda: None, horizon=16)
    l1_roots = list(fired_roots)

    assert screen_roots == [5, 14]
    assert l1_roots == screen_roots
    assert l1_res["batches"] == 2
    assert l1_res["action_counts"][2] == 2

    # 3. Truly illegal chosen action (legal[5] == False) MUST raise ScientificFault.
    monkeypatch.setattr(
        l1_engine,
        "dense_batch",
        lambda *_a, **_k: {
            "chosen": 5,
            "branches_evaluated": 1,
            "branches_skipped": 0,
            "native_v8": {"kernel_rows": 300},
            "outcomes": {},
        },
    )
    with pytest.raises(l1_engine.ScientificFault, match="illegal exact-stop action selection"):
        l1_engine.run_arm(123, "l1", lambda: None, horizon=16)


def test_admission_failure_predicates_and_final_look_sole_look_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # 1. Pre-allocation 503/412 never-allocated receipt with retained pending record -> prestart.
    prealloc_dir = tmp_path / "prealloc"
    (prealloc_dir / "exports").mkdir(parents=True)
    # Inert macOS metadata (.DS_Store, ._.DS_Store) inside output_dir/exports must NOT poison it!
    (prealloc_dir / ".DS_Store").write_bytes(b"macos")
    (prealloc_dir / "exports/.DS_Store").write_bytes(b"macos")
    (prealloc_dir / "exports/._.DS_Store").write_bytes(b"macos")
    prealloc_rec = {
        "status": "INCOMPLETE",
        "endpoint": None,
        "stop_verified": False,
        "cleanup_error": admission.RETAINED_PENDING_CLEANUP_ERROR,
        "failure_class": "SUPERVISOR",
        "error": "Refused('colab new failed (1): 503 Service Unavailable')",
        "transfers": [],
        "contents_transfers": [],
        "control_operations": [],
        "exports": [],
        "terminal": None,
    }
    assert admission.is_prestart_transport_failure(prealloc_rec, prealloc_dir) is True

    # Non-inert dot-file (e.g. .partial.tmp) or symlink named .DS_Store MUST fail closed!
    sneaky_dot = prealloc_dir / "exports/.partial.tmp"
    sneaky_dot.write_bytes(b"partial")
    assert admission.is_prestart_transport_failure(prealloc_rec, prealloc_dir) is False
    sneaky_dot.unlink()

    (prealloc_dir / "exports/.DS_Store").unlink()
    (prealloc_dir / "exports/.DS_Store").symlink_to(prealloc_dir / ".DS_Store")
    assert admission.is_prestart_transport_failure(prealloc_rec, prealloc_dir) is False
    (prealloc_dir / "exports/.DS_Store").unlink()

    # 1b. Exact shard-01 pre-allocation refusal shape with local supervisor staging/ tree.
    shard01_dir = tmp_path / "output-shard-01"
    shard01_staging = shard01_dir / "staging"
    shard01_staging.mkdir(parents=True)
    (shard01_dir / "manifest.json").write_text(
        '{"job_id": "watch6-l1-shard-01"}\n', encoding="utf-8"
    )
    for fname in (
        "bootstrap.sh",
        "release.py",
        "runtime.py",
        "schema-validation.json",
        "source-plan.json",
        "source.py",
        "source.tar.gz",
    ):
        (shard01_staging / fname).write_bytes(f"staged:{fname}".encode("utf-8"))
    shard01_rec = {
        "accounting": None,
        "cleanup_error": None,
        "cli_calls": [],
        "commit": "1873b9025d5a563dcefaf7fcb3cda1e4e6e25306",
        "contents_transfers": [],
        "control_operations": [],
        "elapsed_seconds": 0.0017196668777614832,
        "endpoint": None,
        "error": "Refused('unresolved allocation/stop: reconcile pending records first')",
        "exec_poll_recovery_base_seconds": 600.0,
        "exec_poll_recovery_seconds": 1200.0,
        "exports": [],
        "failure_class": "SUPERVISOR",
        "generation": "88a230800e4a45b29049625fa4ab4457",
        "job_id": "watch6-l1-shard-01-l1v1-main-01",
        "manifest_sha256": digest(shard01_dir / "manifest.json"),
        "session": "snake-watch6-l1-shard-01-l1v1-main-01-88a230800e4a",
        "status": "INCOMPLETE",
        "stop_verified": False,
        "supervisor_sha256": "12da8937ce9cee01ddb5984a7e3a2eca1dde20bfc127921c230bcc524b172ec9",
        "terminal": None,
        "transfers": [],
        "usage_after": None,
        "usage_before": None,
        "warmup_seconds": 0.0,
    }
    (shard01_dir / "receipt.json").write_text(json.dumps(shard01_rec), encoding="utf-8")
    # (a) Exact shard-01 shape classifies as prestart (and not compute):
    assert admission.is_prestart_transport_failure(shard01_rec, shard01_dir) is True
    assert admission.is_external_transport_compute_failure(shard01_rec, shard01_dir) is False

    # (b) Extra/unknown file, missing file, subdirectory, symlink, or any exports/terminal/
    #     hardware/poll evidence in or alongside staging/ MUST fail closed:
    extra_staging_file = shard01_staging / "unknown-extra.txt"
    extra_staging_file.write_text("extra\n", encoding="utf-8")
    assert admission.is_prestart_transport_failure(shard01_rec, shard01_dir) is False
    extra_staging_file.unlink()

    (shard01_staging / "source.tar.gz").unlink()
    assert admission.is_prestart_transport_failure(shard01_rec, shard01_dir) is False
    (shard01_staging / "source.tar.gz").write_bytes(b"staged:source.tar.gz")

    extra_subdir = shard01_staging / "nested"
    extra_subdir.mkdir()
    assert admission.is_prestart_transport_failure(shard01_rec, shard01_dir) is False
    extra_subdir.rmdir()

    (shard01_staging / "runtime.py").unlink()
    (shard01_staging / "runtime.py").symlink_to(shard01_dir / "manifest.json")
    assert admission.is_prestart_transport_failure(shard01_rec, shard01_dir) is False
    (shard01_staging / "runtime.py").unlink()
    (shard01_staging / "runtime.py").write_bytes(b"staged:runtime.py")

    assert (
        admission.is_prestart_transport_failure(
            {
                **shard01_rec,
                "exports": [{"path": "efficacy/world-06.json", "sha256": "0" * 64, "size": 10}],
            },
            shard01_dir,
        )
        is False
    )
    (shard01_dir / "exports").mkdir()
    (shard01_dir / "exports/world-06.json").write_text("{}", encoding="utf-8")
    assert admission.is_prestart_transport_failure(shard01_rec, shard01_dir) is False
    (shard01_dir / "exports/world-06.json").unlink()
    (shard01_dir / "exports").rmdir()

    assert (
        admission.is_prestart_transport_failure(
            {**shard01_rec, "terminal": {"returncode": 1, "reason": "error", "error": "boom"}},
            shard01_dir,
        )
        is False
    )
    for forbidden_evidence in ("terminal.json", "hardware.json", "poll-000000.json"):
        ev_path = shard01_dir / forbidden_evidence
        ev_path.write_text("{}\n", encoding="utf-8")
        assert admission.is_prestart_transport_failure(shard01_rec, shard01_dir) is False
        ev_path.unlink()
    assert admission.is_prestart_transport_failure(shard01_rec, shard01_dir) is True

    # (c) An endpoint-assigned receipt with staging/ is unaffected:
    assert (
        admission.is_prestart_transport_failure(
            {**shard01_rec, "endpoint": "gpu-a100-s-assigned", "stop_verified": True},
            shard01_dir,
        )
        is False
    )
    assert (
        admission.is_external_transport_compute_failure(
            {**shard01_rec, "endpoint": "gpu-a100-s-assigned", "stop_verified": True},
            shard01_dir,
        )
        is False
    )

    # 2. Pre-workload-started public checkpoint fetch failure (rc=74, inert exports/) -> prestart.
    rc74_dir = tmp_path / "rc74"
    (rc74_dir / "exports").mkdir(parents=True)
    (rc74_dir / "exports/.DS_Store").write_bytes(b"macos")
    (rc74_dir / "hardware.json").write_text('{"cpu_count": 12}\n', encoding="utf-8")
    (rc74_dir / "job.log").write_text(
        "WATCH6_L1_PRECOMPUTE_PUBLIC_INPUT_FETCH_FAILED: URLError timeout\n",
        encoding="utf-8",
    )
    rc74_rec = {
        "status": "INCOMPLETE",
        "endpoint": "m-s-test",
        "stop_verified": True,
        "cleanup_error": None,
        "failure_class": "CHILD_NONZERO_EXIT",
        "exports": [],
        "terminal": {"returncode": 74, "reason": "nonzero_exit", "error": None},
    }
    assert admission.is_prestart_transport_failure(rc74_rec, rc74_dir) is True

    # 3. Lease-expired & clean 2x wall-timeout compute interruptions are retryable;
    #    faulthandler 900s watchdog and ScientificFault rc=86 remain forbidden.
    lease_dir = tmp_path / "lease"
    lease_dir.mkdir()
    (lease_dir / "hardware.json").write_text('{"cpu_count": 12}\n', encoding="utf-8")
    (lease_dir / "poll-001.json").write_text(
        json.dumps(
            {
                "state": {"status": "running", "returncode": None, "reason": None, "error": None},
                "terminal": None,
                "job_log_tail": "WATCH6_L1_PROGRESS shard=0/18 done=2/6",
                "supervisor_log_tail": "",
            }
        ),
        encoding="utf-8",
    )
    lease_rec = {
        "status": "INCOMPLETE",
        "endpoint": "m-s-lease",
        "stop_verified": True,
        "cleanup_error": None,
        "failure_class": "CHILD_NONZERO_EXIT",
        "terminal": {"returncode": 143, "reason": "lease_expired", "error": None},
    }
    assert admission.is_external_transport_compute_failure(lease_rec, lease_dir) is True

    timeout_rec = {
        **lease_rec,
        "terminal": {"returncode": 75, "reason": "timeout", "error": None},
    }
    assert admission.is_prestart_transport_failure(timeout_rec, lease_dir) is False
    assert admission.is_external_transport_compute_failure(timeout_rec, lease_dir) is True

    sup_wall_rec = {
        "status": "INCOMPLETE",
        "endpoint": "m-s-supwall",
        "stop_verified": True,
        "cleanup_error": None,
        "failure_class": "SUPERVISOR",
        "error": "Refused('whole-session wall cap reached (including setup/export)')",
        "terminal": None,
    }
    assert admission.is_prestart_transport_failure(sup_wall_rec, lease_dir) is False
    assert admission.is_external_transport_compute_failure(sup_wall_rec, lease_dir) is True

    sci_fault_rec = {
        **lease_rec,
        "terminal": {"returncode": 86, "reason": "timeout", "error": None},
    }
    assert admission.is_external_transport_compute_failure(sci_fault_rec, lease_dir) is False

    watchdog_dir = tmp_path / "watchdog"
    watchdog_dir.mkdir()
    (watchdog_dir / "hardware.json").write_text('{"cpu_count": 12}\n', encoding="utf-8")
    (watchdog_dir / "job.log").write_text(
        "Timeout (0:15:00):\nThread 0x0001 (most recent call first):\n",
        encoding="utf-8",
    )
    assert admission.is_external_transport_compute_failure(timeout_rec, watchdog_dir) is False

    # 4. Direct test of SoleFinalLookViolation runtime audit hook.
    dummy_world = tmp_path / "world-00.json"
    publish_json(dummy_world, {"hello": "world"})
    with final_look.pre_outcome_open_guard():
        # Raw binary SHA-256 digesting via package_spec.digest is allowed:
        assert len(digest(dummy_world)) == 64
        # Any actual read of world-00.json raises SoleFinalLookViolation!
        with pytest.raises(final_look.SoleFinalLookViolation):
            dummy_world.read_text(encoding="utf-8")

    # 5. Full synthetic 18-shard (12 A100 + 6 L4) final_look end-to-end test + operational guards.
    ops_dir = tmp_path / "ops"
    mf_dir = ops_dir / "manifests"
    mf_dir.mkdir(parents=True)

    meta_input = _make_metadata_archive(tmp_path / "meta-valid")
    meta_inputs_field = [
        {k: meta_input[k] for k in ("path", "dest", "sha256")},
    ]

    # Foreign-namespace symlinks + .DS_Store in ops/manifests/ must be ignored/reported, not fatal!
    (mf_dir / ".DS_Store").write_bytes(b"macos")
    foreign_target = tmp_path / "foreign-v3-shard.json"
    foreign_target.write_text(
        json.dumps({"job_id": "watch6-v3-shard-00", "watch6_v3": {"namespace": "v3"}}),
        encoding="utf-8",
    )
    (mf_dir / "watch6-v3-foreign-symlink.json").symlink_to(foreign_target)
    (mf_dir / "watch6-v3-broken-symlink.json").symlink_to(tmp_path / "nonexistent-v3.json")

    ruling_md = tmp_path / "RULING_L1_V1.md"
    ruling_md.write_text("# Ruling\nStatus: RATIFIED\nVerdict: APPROVE\n", encoding="utf-8")
    ruling_json = tmp_path / "ruling-l1-v1.json"
    ratify.record_ruling(ruling_md, ruling_json, judge="Astra")

    # Write passing A100 and L4 canary attempts.
    for gpu, c_shard in (("A100", 0), ("L4", 12)):
        c_out = tmp_path / f"output-canary-{gpu.lower()}"
        c_out.mkdir()
        c_mf = {
            "job_id": f"watch6-l1-canary-{gpu.lower()}",
            "commit": "5213531207cb5b0e0baf747aaae0ee5c37901c7f",
            "output_dir": str(c_out),
            "inputs": meta_inputs_field,
            "requirements": {"gpu": gpu, "allowed_gpus": [gpu], "min_vcpu": 12},
            "canary": {"schema": canary.SCHEMA, "mode": "probe", "family": "fam"},
            "env": {
                "WATCH6_FIXTURE": "1",
                "WATCH6_SHARDS": str(p.SHARDS),
                "WATCH6_SHARD": str(c_shard),
                "WATCH6_WORKERS": str(p.shard_spec(c_shard)["workers"]),
                "WATCH6_METADATA_SHA256": meta_input["sha256"],
            },
            "watch6_l1_v1": {
                "namespace": p.NAMESPACE,
                "science": p.identity(),
                "primary_allocation": p.allocation(),
            },
        }
        (c_out / "manifest.json").write_text(json.dumps(c_mf), encoding="utf-8")
        c_rec = {
            "job_id": c_mf["job_id"],
            "commit": c_mf["commit"],
            "manifest_sha256": digest(c_out / "manifest.json"),
            "status": "PASS",
            "endpoint": f"m-s-canary-{gpu.lower()}",
            "stop_verified": True,
            "cleanup_error": None,
            "failure_class": None,
            "error": None,
            "terminal": {"returncode": 0, "reason": None, "error": None},
            "exports": [],
        }
        (c_out / "receipt.json").write_text(json.dumps(c_rec), encoding="utf-8")
        (mf_dir / f"canary-{gpu.lower()}.json").write_text(json.dumps(c_mf), encoding="utf-8")

    panel = _synthetic_panel()
    alloc = p.allocation()

    # Shard 0 linear chain: prestart -> wall-timeout compute (slot 0) -> PASS (slots 1..5).
    s0_pre = tmp_path / "shard-00-att1"
    (s0_pre / "staging").mkdir(parents=True)
    for fname in admission.ALLOWED_PREALLOC_STAGING_FILES:
        (s0_pre / "staging" / fname).write_bytes(f"staged:{fname}".encode("utf-8"))
    s0_spec = p.shard_spec(0)
    base_s0_mf = {
        "job_id": "watch6-l1-shard-00-a1",
        "commit": "5213531207cb5b0e0baf747aaae0ee5c37901c7f",
        "output_dir": str(s0_pre),
        "inputs": meta_inputs_field,
        "requirements": {"gpu": "A100", "allowed_gpus": ["A100"], "min_vcpu": 12},
        "env": {
            "WATCH6_FIXTURE": "0",
            "WATCH6_SHARDS": str(p.SHARDS),
            "WATCH6_SHARD": "0",
            "WATCH6_WORKERS": str(s0_spec["workers"]),
            "WATCH6_METADATA_SHA256": meta_input["sha256"],
        },
        "watch6_l1_v1": {
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "primary_allocation": alloc,
            "prestart_retries_used": 0,
            "compute_completion_attempt": 0,
            "total_dispatches_for_shard": 1,
            "prior_receipts": [],
        },
    }
    (s0_pre / "manifest.json").write_text(json.dumps(base_s0_mf), encoding="utf-8")
    s0_pre_rec = {
        **prealloc_rec,
        "job_id": base_s0_mf["job_id"],
        "commit": base_s0_mf["commit"],
        "manifest_sha256": digest(s0_pre / "manifest.json"),
    }
    (s0_pre / "receipt.json").write_text(json.dumps(s0_pre_rec), encoding="utf-8")
    (mf_dir / "shard-00-a1.json").write_text(json.dumps(base_s0_mf), encoding="utf-8")

    s0_comp = tmp_path / "shard-00-att2"
    s0_comp_eff = s0_comp / "exports/efficacy"
    s0_comp_eff.mkdir(parents=True)
    (s0_comp / "hardware.json").write_text('{"cpu_count": 12}\n', encoding="utf-8")
    w0_path = s0_comp_eff / "world-00.json"
    w0_desc = publish_json(w0_path, panel[0])
    publish_json(
        w0_path.with_suffix(".meta.json"),
        {
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "seed_kind": "primary",
            "allocation": alloc[0],
            "fixture": False,
            "sha256": w0_desc["sha256"],
            "size": w0_desc["size"],
        },
    )
    s0_comp_mf = {
        **base_s0_mf,
        "job_id": "watch6-l1-shard-00-a2",
        "output_dir": str(s0_comp),
        "watch6_l1_v1": {
            **base_s0_mf["watch6_l1_v1"],
            "prestart_retries_used": 1,
            "compute_completion_attempt": 0,
            "total_dispatches_for_shard": 2,
            "prior_receipts": [
                {
                    "receipt_path": str((s0_pre / "receipt.json").resolve()),
                    "receipt_sha256": digest(s0_pre / "receipt.json"),
                    "classification": "prestart",
                }
            ],
        },
    }
    (s0_comp / "manifest.json").write_text(json.dumps(s0_comp_mf), encoding="utf-8")
    s0_comp_rec = {
        **timeout_rec,
        "job_id": s0_comp_mf["job_id"],
        "commit": s0_comp_mf["commit"],
        "manifest_sha256": digest(s0_comp / "manifest.json"),
        "exports": [
            {
                "path": "efficacy/world-00.json",
                "sha256": w0_desc["sha256"],
                "size": w0_desc["size"],
            }
        ],
    }
    (s0_comp / "receipt.json").write_text(json.dumps(s0_comp_rec), encoding="utf-8")
    (mf_dir / "shard-00-a2.json").write_text(json.dumps(s0_comp_mf), encoding="utf-8")

    for shard in range(p.SHARDS):
        spec = admission.effective_shard_spec(shard)
        s_out = tmp_path / f"shard-{shard:02d}-pass"
        eff_dir = s_out / "exports/efficacy"
        eff_dir.mkdir(parents=True)
        if shard == 0:
            (s_out / "exports/.DS_Store").write_bytes(b"macos")
            (eff_dir / ".DS_Store").write_bytes(b"macos")
        exports_rows: list[dict] = []
        indices_to_write = spec["indices"][1:] if shard == 0 else spec["indices"]
        for idx in indices_to_write:
            w_path = eff_dir / f"world-{idx:02d}.json"
            desc = publish_json(w_path, panel[idx])
            publish_json(
                w_path.with_suffix(".meta.json"),
                {
                    "namespace": p.NAMESPACE,
                    "science": p.identity(),
                    "seed_kind": "primary",
                    "allocation": alloc[idx],
                    "fixture": False,
                    "sha256": desc["sha256"],
                    "size": desc["size"],
                },
            )
            exports_rows.append(
                {
                    "path": f"efficacy/world-{idx:02d}.json",
                    "sha256": desc["sha256"],
                    "size": desc["size"],
                }
            )

        if shard == 0:
            w_bind = {
                "namespace": p.NAMESPACE,
                "science": p.identity(),
                "primary_allocation": alloc,
                "prestart_retries_used": 1,
                "compute_completion_attempt": 1,
                "total_dispatches_for_shard": 3,
                "prior_receipts": [
                    {
                        "receipt_path": str((s0_pre / "receipt.json").resolve()),
                        "receipt_sha256": digest(s0_pre / "receipt.json"),
                        "classification": "prestart",
                    },
                    {
                        "receipt_path": str((s0_comp / "receipt.json").resolve()),
                        "receipt_sha256": digest(s0_comp / "receipt.json"),
                        "classification": "external_transport_compute",
                    },
                ],
            }
        else:
            w_bind = {
                "namespace": p.NAMESPACE,
                "science": p.identity(),
                "primary_allocation": alloc,
                "prestart_retries_used": 0,
                "compute_completion_attempt": 0,
                "total_dispatches_for_shard": 1,
                "prior_receipts": [],
            }
            if "amendment_id" in spec:
                w_bind["hardware_amendment_id"] = spec["amendment_id"]
                w_bind["hardware_amendment_sha256"] = spec["amendment_sha256"]

        s_mf = {
            "job_id": f"watch6-l1-shard-{shard:02d}",
            "commit": "5213531207cb5b0e0baf747aaae0ee5c37901c7f",
            "output_dir": str(s_out),
            "inputs": meta_inputs_field,
            "requirements": {
                "gpu": spec["gpu"],
                "allowed_gpus": [spec["gpu"]],
                "min_vcpu": 12,
            },
            "env": {
                "WATCH6_FIXTURE": "0",
                "WATCH6_SHARDS": str(p.SHARDS),
                "WATCH6_SHARD": str(shard),
                "WATCH6_WORKERS": str(spec["workers"]),
                "WATCH6_METADATA_SHA256": meta_input["sha256"],
            },
            "watch6_l1_v1": w_bind,
        }
        (s_out / "manifest.json").write_text(json.dumps(s_mf), encoding="utf-8")
        s_rec = {
            "job_id": s_mf["job_id"],
            "commit": s_mf["commit"],
            "manifest_sha256": digest(s_out / "manifest.json"),
            "status": "PASS",
            "endpoint": f"m-s-shard-{shard:02d}",
            "stop_verified": True,
            "cleanup_error": None,
            "failure_class": None,
            "error": None,
            "terminal": {"returncode": 0, "reason": None, "error": None},
            "exports": exports_rows,
        }
        (s_out / "receipt.json").write_text(json.dumps(s_rec), encoding="utf-8")
        (mf_dir / f"shard-{shard:02d}.json").write_text(json.dumps(s_mf), encoding="utf-8")

    # Finding 5: final_look CLI must NOT accept --alpha, and run_final_look hard-codes alpha=0.05.
    monkeypatch.setattr(
        "sys.argv",
        [
            "final_look",
            "--ops-dir",
            str(ops_dir),
            "--out-dir",
            str(tmp_path / "bad-cli"),
            "--alpha",
            "0.1",
        ],
    )
    with pytest.raises(SystemExit):
        final_look.main()

    decision = final_look.run_final_look(
        ops_dir=ops_dir,
        out_dir=tmp_path / "final-out",
        ruling_json=ruling_json,
    )
    assert decision["verdict"] == "GO_L1_SUPERIOR"
    assert decision["alpha"] == 0.05
    assert decision["total_main_attempts"] == 20  # 18 PASS + 1 prestart + 1 wall-timeout compute
    acct = json.loads((tmp_path / "final-out/pre-outcome-accounting.json").read_text("utf-8"))
    assert len(acct["ignored_foreign_manifest_symlinks"]) == 2
    assert len(acct["mac_mirror_sha256s"]) == 1
    assert acct["hardware_amendment"] == {
        "amendment_id": admission.AMENDMENT_ID,
        "sha256": admission.AMENDMENT_SHA256,
        "shards": [8, 9, 10],
        "gpu": "L4",
        "workers": 6,
    }

    # Amended shard 08 on A100 (or W=4) or un-amended shard 11 on L4 MUST fail closed in final_look!
    for bad_shard, bad_gpu, bad_w in ((8, "A100", 6), (8, "L4", 4), (11, "L4", 6)):
        s_dir = tmp_path / f"shard-{bad_shard:02d}-pass"
        orig_mf = json.loads((s_dir / "manifest.json").read_text(encoding="utf-8"))
        orig_rec = json.loads((s_dir / "receipt.json").read_text(encoding="utf-8"))
        tampered_mf = {
            **orig_mf,
            "requirements": {"gpu": bad_gpu, "allowed_gpus": [bad_gpu], "min_vcpu": 12},
            "env": {**orig_mf["env"], "WATCH6_WORKERS": str(bad_w)},
        }
        (s_dir / "manifest.json").write_text(json.dumps(tampered_mf), encoding="utf-8")
        (s_dir / "receipt.json").write_text(
            json.dumps({**orig_rec, "manifest_sha256": digest(s_dir / "manifest.json")}),
            encoding="utf-8",
        )
        (mf_dir / f"shard-{bad_shard:02d}.json").write_text(
            json.dumps(tampered_mf),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="GPU/worker mismatch"):
            final_look.run_final_look(
                ops_dir=ops_dir,
                out_dir=tmp_path / f"final-out-bad-s{bad_shard}-{bad_gpu}-{bad_w}",
                ruling_json=ruling_json,
            )
        (s_dir / "manifest.json").write_text(json.dumps(orig_mf), encoding="utf-8")
        (s_dir / "receipt.json").write_text(json.dumps(orig_rec), encoding="utf-8")
        (mf_dir / f"shard-{bad_shard:02d}.json").write_text(json.dumps(orig_mf), encoding="utf-8")

    # Finding 4a: A symlinked Watch6-L1-V1 manifest MUST fail closed!
    l1_symlink = mf_dir / "shard-17-symlink.json"
    l1_symlink.symlink_to(mf_dir / "shard-17.json")
    with pytest.raises(ValueError, match="symlink in Watch6-L1-V1 ops manifests"):
        final_look.run_final_look(
            ops_dir=ops_dir,
            out_dir=tmp_path / "final-out-l1-sym",
            ruling_json=ruling_json,
        )
    l1_symlink.unlink()

    # Finding 4b: Non-inert dot-file in exports/efficacy MUST fail closed!
    bad_export_dot = tmp_path / "shard-00-pass/exports/efficacy/.unexpected.tmp"
    bad_export_dot.write_bytes(b"tmp")
    with pytest.raises(ValueError, match="unexpected non-inert dot-file"):
        final_look.run_final_look(
            ops_dir=ops_dir,
            out_dir=tmp_path / "final-out-bad-dot",
            ruling_json=ruling_json,
        )
    bad_export_dot.unlink()

    # Finding 3: Foreign-identity mac-mirror.json inside input-metadata.tar.gz MUST fail closed!
    foreign_meta = _make_metadata_archive(tmp_path / "meta-foreign", mirror_science="0" * 64)
    s17_out = tmp_path / "shard-17-pass"
    orig_s17_mf = json.loads((s17_out / "manifest.json").read_text(encoding="utf-8"))
    orig_s17_rec = json.loads((s17_out / "receipt.json").read_text(encoding="utf-8"))
    bad_s17_mf = {
        **orig_s17_mf,
        "inputs": [{k: foreign_meta[k] for k in ("path", "dest", "sha256")}],
        "env": {**orig_s17_mf["env"], "WATCH6_METADATA_SHA256": foreign_meta["sha256"]},
    }
    (s17_out / "manifest.json").write_text(json.dumps(bad_s17_mf), encoding="utf-8")
    (s17_out / "receipt.json").write_text(
        json.dumps({**orig_s17_rec, "manifest_sha256": digest(s17_out / "manifest.json")}),
        encoding="utf-8",
    )
    (mf_dir / "shard-17.json").write_text(json.dumps(bad_s17_mf), encoding="utf-8")
    with pytest.raises(ValueError, match="mac-mirror.json"):
        final_look.run_final_look(
            ops_dir=ops_dir,
            out_dir=tmp_path / "final-out-foreign-mirror",
            ruling_json=ruling_json,
        )
    (s17_out / "manifest.json").write_text(json.dumps(orig_s17_mf), encoding="utf-8")
    (s17_out / "receipt.json").write_text(json.dumps(orig_s17_rec), encoding="utf-8")
    (mf_dir / "shard-17.json").write_text(json.dumps(orig_s17_mf), encoding="utf-8")

    # Forking shard 17 by adding a second root attempt must fail pre-outcome with 0 world opens!
    fork_out = tmp_path / "shard-17-fork"
    fork_out.mkdir()
    fork_mf = json.loads((mf_dir / "shard-17.json").read_text(encoding="utf-8"))
    fork_mf["job_id"] = "watch6-l1-shard-17-fork"
    fork_mf["output_dir"] = str(fork_out)
    (fork_out / "manifest.json").write_text(json.dumps(fork_mf), encoding="utf-8")
    fork_rec = {
        **prealloc_rec,
        "job_id": fork_mf["job_id"],
        "commit": fork_mf["commit"],
        "manifest_sha256": digest(fork_out / "manifest.json"),
    }
    (fork_out / "receipt.json").write_text(json.dumps(fork_rec), encoding="utf-8")
    (mf_dir / "shard-17-fork.json").write_text(json.dumps(fork_mf), encoding="utf-8")

    with pytest.raises(ValueError, match="INVALID_OR_INCOMPLETE"):
        final_look.run_final_look(
            ops_dir=ops_dir,
            out_dir=tmp_path / "final-out-fork",
            ruling_json=ruling_json,
        )
    assert not (tmp_path / "final-out-fork/l1-confirm-decision.json").exists()
    assert len(resume.WORLD_NAME_RE.findall("world-00.json")) == 1
    assert empty_ledger(p.NAMESPACE, "watch6", {"science": p.identity()})["units"] == []


@pytest.mark.skipif(
    not DEFAULT_LOCAL_SAVED_DIR.is_dir(),
    reason="local saved_snakes directory not present",
)
def test_engine_record_none_matches_snapshot_recording_bit_identical(monkeypatch) -> None:
    from research.watch6_l1_confirm.engine import run_arm

    monkeypatch.setenv("SNAKE_CKPT_DIR", str(DEFAULT_LOCAL_SAVED_DIR))
    monkeypatch.setenv("SNAKE_WATCH_SOURCE", "live")

    recorded_frames: list[dict] = []
    res_none = run_arm(p.FIXTURE_SEEDS[0], "l1", lambda: None, horizon=6, record=None)
    res_snap = run_arm(
        p.FIXTURE_SEEDS[0],
        "l1",
        lambda: None,
        horizon=6,
        record=recorded_frames.append,
    )
    assert len(recorded_frames) == 7  # initial frame 0 + 6 stepped frames
    assert res_none["mirror_identity_frames"] == 6
    assert res_none["batches"] == 2  # interval=5 fires at decision 1 and decision 6
    assert res_none["native_v8_kernel_rows"] > 0
    for key in (
        "frames",
        "alive_frames",
        "mass_sum",
        "deaths",
        "decisions",
        "boost_decisions",
        "action_counts",
        "batches",
        "interventions",
        "branches_evaluated",
        "branches_skipped",
        "native_v8_kernel_rows",
        "mirror_identity_frames",
        "respawns_per_slot",
        "roster",
    ):
        assert res_none[key] == res_snap[key], f"mismatch on {key}"


def test_freshness_audit_clear_and_collision_detection(tmp_path: Path) -> None:
    from research.watch6_l1_confirm import freshness

    clean_artifacts = tmp_path / "artifacts"
    clean_artifacts.mkdir()
    report = freshness.audit(artifacts_dir=clean_artifacts)
    assert report["status"] == "CLEAR"
    assert report["primary_count"] == 96
    assert report["fixture_count"] == 3

    # Inject a primary seed into visible artifact metadata -> must raise collision ValueError.
    leaked_seed = p.allocation()[0]["seed"]
    (clean_artifacts / "leaked-manifest.json").write_text(
        json.dumps({"seed": leaked_seed}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="collision"):
        freshness.audit(artifacts_dir=clean_artifacts)


@pytest.mark.integration
@pytest.mark.skipif(
    not DEFAULT_LOCAL_SAVED_DIR.is_dir(),
    reason="local saved_snakes directory not present",
)
def test_job_sh_fixture_end_to_end_with_runtime_and_canary_manifest_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Execute job.sh's fixture path end-to-end with ONLY runtime + canary manifest env."""
    import subprocess
    import sys

    from research.watch6_l1_confirm import prepare

    monkeypatch.setattr(prepare, "_require_published_pin", lambda _commit: None)
    monkeypatch.setattr(prepare, "admit", lambda _manifest, **_kw: {"status": "PASS"})

    sci = p.identity()
    benchmark_path = tmp_path / "benchmark.json"
    publish_json(
        benchmark_path,
        {
            "schema": "watch6-l1-mac-preflight/v1",
            "status": "PASS",
            "science": sci,
            "fixture": True,
        },
    )
    mirror_path = tmp_path / "mac-mirror.json"
    publish_json(
        mirror_path,
        {
            "schema": mirror.SCHEMA,
            "status": "PASS",
            "platform": "macOS-test",
            "namespace": p.NAMESPACE,
            "science": sci,
            "fixture": True,
            "seed": p.FIXTURE_SEEDS[0],
            "horizon": p.HORIZON,
            "total_mirrored_frames": p.HORIZON * len(p.ARMS),
            "mirror_mismatches": 0,
            "elapsed_seconds": 12.0,
            "arms": {
                arm: {
                    "frames": p.HORIZON,
                    "mirror_identity_frames": p.HORIZON,
                    "mirror_mismatches": 0,
                    "respawns_per_slot": [1, 1, 1, 1, 1, 1],
                    "batches": 0 if arm == "bare" else 1200,
                    "native_v8_kernel_rows": 0 if arm == "bare" else 90000,
                    "wall_seconds": 4.0,
                    "timing": {
                        "main_seconds": 2.0,
                        "search_seconds": 0.0 if arm == "bare" else 2.0,
                    },
                }
                for arm in p.ARMS
            },
        },
    )
    costs_path = tmp_path / "costs.json"
    publish_json(costs_path, costs.build_cost_report())

    prep_dir = tmp_path / "prepared"
    prepare.prepare(
        prep_dir,
        benchmark_path,
        "cf07791be3f8bc36f8f7f8c671712d984f162655",
        mirror_path,
        costs_path,
        main_shards=False,
    )

    canary_manifest = json.loads((prep_dir / "canary-a100.json").read_text(encoding="utf-8"))
    runtime_root = tmp_path / "runtime"
    inputs_dir = runtime_root / "inputs"
    output_dir = runtime_root / "output"
    inputs_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    for item in canary_manifest["inputs"]:
        dest = inputs_dir / item["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(item["path"]).read_bytes())

    runtime_env = {
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
        "COLAB_INPUT_DIR": str(inputs_dir),
        "COLAB_OUTPUT_DIR": str(output_dir),
        **{str(k): str(v) for k, v in canary_manifest["env"].items()},
    }
    proc = subprocess.run(
        ["bash", str(p.ROOT / p.SCRIPT)],
        cwd=p.ROOT,
        env=runtime_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"job.sh failed ({proc.returncode}):\n{proc.stderr}\n{proc.stdout}"

    for rel in canary_manifest["required_exports"]:
        export_path = output_dir / rel
        assert export_path.is_file(), f"missing required export {rel}"
        assert export_path.with_suffix(export_path.suffix + ".sha256").is_file()
        assert export_path.with_suffix(export_path.suffix + ".complete").is_file()
        verify_immutable(export_path)

    verify_immutable(output_dir / "workload-started.json")
    for idx in range(2):
        verify_immutable(output_dir / f"efficacy/fixture-world-{idx:02d}.json")
        verify_immutable(output_dir / f"efficacy/fixture-world-{idx:02d}.meta.json")

    op = json.loads((output_dir / "efficacy/operational.json").read_text(encoding="utf-8"))
    assert op["schema"] == "watch6-l1-operational/v1"
    assert op["namespace"] == p.NAMESPACE
    assert op["science"] == sci
    assert op["fixture"] is True
    assert op["skipped_verified_total"] == 0
    assert op["completed_indices"] == [0, 1]

    perf = json.loads((output_dir / "performance.json").read_text(encoding="utf-8"))
    assert perf["schema"] == "watch6-l1-canary-performance/v1"
    assert perf["science"] == sci
    assert perf["fixture"] is True
    assert 300 in perf["frames_executed"]
    assert perf["native_v8"]["kernel_rows"] > 0
    assert perf["native_v8"]["stock_rows"] == 0
    assert perf["native_v8"]["fallback_rows"] == 0
    assert perf["native_v8"]["noncandidate_rows"] == 0


def test_hardware_reassignment_amendment_l4_shards_08_09_10(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify AMENDMENT_L4_REASSIGN_01 moves shards 8..10 to L4 W=6 with zero science change."""
    from research.watch6_l1_confirm import finalize, prepare

    # 1. Science identity and registration SHA-256 are completely unchanged.
    assert p.identity() == "57ec544fe45f7a8a876dbc9363048468b63358e224b2ee8e11dff41401dcfbd3"
    assert (
        digest(p.registration())
        == "8b46779ff52d0c9648ff8de00c1893f77beb9bd99eb4a4e51afbe0bdf36800a0"
    )

    # 2. protocol.shard_spec remains untouched; effective_shard_spec reassigns only shards 8, 9, 10.
    amendment = admission.validate_hardware_amendment()
    assert amendment["amendment_id"] == "AMENDMENT_L4_REASSIGN_01"
    assert amendment["shards"] == [8, 9, 10]
    assert amendment["gpu"] == "L4"
    assert amendment["workers"] == 6
    assert amendment["sha256"] == admission.AMENDMENT_SHA256

    for s in range(p.SHARDS):
        raw_spec = p.shard_spec(s)
        eff_spec = admission.effective_shard_spec(s)
        assert eff_spec["indices"] == raw_spec["indices"]
        assert eff_spec["world_count"] == raw_spec["world_count"]
        if s in (8, 9, 10):
            assert raw_spec["gpu"] == "A100" and raw_spec["workers"] == 6
            assert eff_spec["gpu"] == "L4" and eff_spec["workers"] == 6
            assert eff_spec["world_count"] == 6 and len(eff_spec["indices"]) == 6
            assert eff_spec["amendment_id"] == admission.AMENDMENT_ID
            assert eff_spec["amendment_sha256"] == admission.AMENDMENT_SHA256
        elif s < 12:
            assert raw_spec["gpu"] == "A100" and raw_spec["workers"] == 6
            assert eff_spec["gpu"] == "A100" and eff_spec["workers"] == 6
            assert "amendment_id" not in eff_spec
        else:
            assert raw_spec["gpu"] == "L4" and raw_spec["workers"] == 4
            assert eff_spec["gpu"] == "L4" and eff_spec["workers"] == 4
            assert "amendment_id" not in eff_spec

    # 3. prepare + finalize for shards 8, 9, 10 with L4 canary receipt and L4 2x cost envelope.
    commit = "bc11385b4ec0ae3c9e7f035c68dd5ce61dbc76f3"
    monkeypatch.setattr(prepare, "_require_published_pin", lambda _commit: None)
    monkeypatch.setattr(canary, "check", lambda *_a, **_k: {"status": "PASS"})
    monkeypatch.setattr(
        admission,
        "check_uploads",
        lambda _m: {"status": "PASS", "upload_count": 1, "total_bytes": 1024},
    )

    sci = p.identity()
    benchmark_path = tmp_path / "benchmark.json"
    publish_json(
        benchmark_path,
        {
            "schema": "watch6-l1-mac-preflight/v1",
            "status": "PASS",
            "science": sci,
            "fixture": True,
        },
    )
    mirror_path = tmp_path / "mac-mirror.json"
    publish_json(
        mirror_path,
        {
            "schema": mirror.SCHEMA,
            "status": "PASS",
            "platform": "macOS-test",
            "namespace": p.NAMESPACE,
            "science": sci,
            "fixture": True,
            "seed": p.FIXTURE_SEEDS[0],
            "horizon": p.HORIZON,
            "total_mirrored_frames": p.HORIZON * len(p.ARMS),
            "mirror_mismatches": 0,
            "elapsed_seconds": 12.0,
            "arms": {
                arm: {
                    "frames": p.HORIZON,
                    "mirror_identity_frames": p.HORIZON,
                    "mirror_mismatches": 0,
                    "respawns_per_slot": [1, 1, 1, 1, 1, 1],
                    "batches": 0 if arm == "bare" else 1200,
                    "native_v8_kernel_rows": 0 if arm == "bare" else 90000,
                    "wall_seconds": 4.0,
                    "timing": {
                        "main_seconds": 2.0,
                        "search_seconds": 0.0 if arm == "bare" else 2.0,
                    },
                }
                for arm in p.ARMS
            },
        },
    )
    costs_path = tmp_path / "costs.json"
    publish_json(costs_path, costs.build_cost_report())
    ruling_md = tmp_path / "RULING_L1_V1.md"
    ruling_md.write_text("# Ruling\nStatus: RATIFIED\nVerdict: APPROVE\n", encoding="utf-8")
    ruling_json = tmp_path / "ruling-l1-v1.json"
    ratify.record_ruling(ruling_md, ruling_json, judge="Astra")

    prep_dir = tmp_path / "prepared-l4-move"
    summary = prepare.prepare(
        prep_dir,
        benchmark_path,
        commit,
        mirror_path,
        costs_path,
        main_shards=True,
        pending_canary=True,
        target_shards=[8, 9, 10],
        canary_gpus=["L4"],
    )
    assert summary["shards"] == 18
    assert sorted(p_mf.name for p_mf in prep_dir.glob("shard-*.pending.json")) == [
        "shard-08.pending.json",
        "shard-09.pending.json",
        "shard-10.pending.json",
    ]
    assert (prep_dir / "canary-l4.json").is_file()
    assert not (prep_dir / "canary-a100.json").exists()

    # Create a passing L4 canary receipt bound to canary-l4.json and finalize shards 8..10.
    canary_l4_mf = json.loads((prep_dir / "canary-l4.json").read_text(encoding="utf-8"))
    canary_l4_out = Path(canary_l4_mf["output_dir"])
    (canary_l4_out / "exports").mkdir(parents=True, exist_ok=True)
    publish_json(
        canary_l4_out / "exports/performance.json",
        {
            "schema": "watch6-l1-canary-performance/v1",
            "science": sci,
            "fixture": True,
            "frames_executed": [300],
            "native_v8": {
                "kernel_rows": 9000,
                "stock_rows": 0,
                "fallback_rows": 0,
                "noncandidate_rows": 0,
            },
        },
    )
    (canary_l4_out / "manifest.json").write_text(
        (prep_dir / "canary-l4.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    canary_l4_rec_path = canary_l4_out / "receipt.json"
    canary_l4_rec = {
        "job_id": canary_l4_mf["job_id"],
        "commit": commit,
        "manifest_sha256": digest(canary_l4_out / "manifest.json"),
        "status": "PASS",
        "endpoint": "m-s-canary-l4",
        "stop_verified": True,
        "cleanup_error": None,
        "failure_class": None,
        "error": None,
        "terminal": {"returncode": 0, "reason": None, "error": None},
        "exports": [],
    }
    canary_l4_rec_path.write_text(json.dumps(canary_l4_rec), encoding="utf-8")

    fin_res = finalize.finalize(
        prep_dir,
        canary_l4_receipt=canary_l4_rec_path,
    )
    assert sorted(fin_res["finalized_manifests"]) == [
        "shard-08.json",
        "shard-09.json",
        "shard-10.json",
    ]

    for shard in (8, 9, 10):
        mf = json.loads((prep_dir / f"shard-{shard:02d}.json").read_text(encoding="utf-8"))
        assert mf["requirements"] == {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": 540,
        }
        assert mf["timeout_minutes"] == 1080
        assert mf["budget"] == {
            "rate_units_per_hour": 2.5,
            "unit_cap": 45.0,
            "reserve_units": 9.0,
            "other_assignments": {},
        }
        assert mf["env"]["WATCH6_SHARD"] == str(shard)
        assert mf["env"]["WATCH6_WORKERS"] == "6"
        assert mf["env"]["WATCH6_BASE_MINUTES"] == "540"
        assert mf["canary"]["receipt"]["path"] == str(canary_l4_rec_path.resolve())
        assert mf["canary"]["receipt"]["sha256"] == digest(canary_l4_rec_path)
        assert mf["watch6_l1_v1"]["gpu"] == "L4"
        assert mf["watch6_l1_v1"]["workers"] == 6
        assert mf["watch6_l1_v1"]["base_minutes"] == 540
        assert mf["watch6_l1_v1"]["base_unit_cap"] == 22.5
        assert mf["watch6_l1_v1"]["base_reserve_units"] == 4.5
        assert mf["watch6_l1_v1"]["indices"] == p.shard_indices(shard)
        assert mf["watch6_l1_v1"]["hardware_amendment_id"] == admission.AMENDMENT_ID
        assert mf["watch6_l1_v1"]["hardware_amendment_sha256"] == admission.AMENDMENT_SHA256
        # Full non-pending admission succeeds:
        admission.admit(mf, pending_qualification=False)

    # 4. Amended shard 08 rejected on any other GPU or worker count or base_minutes in admit().
    mf08 = json.loads((prep_dir / "shard-08.json").read_text(encoding="utf-8"))
    for bad_gpu in ("A100", "T4"):
        bad_mf = json.loads(json.dumps(mf08))
        bad_mf["requirements"]["gpu"] = bad_gpu
        bad_mf["requirements"]["allowed_gpus"] = [bad_gpu]
        bad_mf["watch6_l1_v1"]["gpu"] = bad_gpu
        with pytest.raises(ValueError, match="requires exact >=12 vCPU L4 W=6 placement"):
            admission.admit(bad_mf, pending_qualification=False)

    for bad_w in (4, 8):
        bad_mf = json.loads(json.dumps(mf08))
        bad_mf["env"]["WATCH6_WORKERS"] = str(bad_w)
        bad_mf["watch6_l1_v1"]["workers"] = bad_w
        with pytest.raises(ValueError, match="requires exact >=12 vCPU L4 W=6 placement"):
            admission.admit(bad_mf, pending_qualification=False)

    bad_env_mf = json.loads(json.dumps(mf08))
    bad_env_mf["watch6_l1_v1"]["base_minutes"] = 570
    bad_env_mf["timeout_minutes"] = 1140
    bad_env_mf["requirements"]["est_minutes"] = 570
    bad_env_mf["env"]["WATCH6_BASE_MINUTES"] = "570"
    with pytest.raises(ValueError, match="invalid base_minutes envelope"):
        admission.admit(bad_env_mf, pending_qualification=False)

    # 5. Un-amended shards still require their registered GPU and worker count.
    prep_unamended = tmp_path / "prepared-unamended"
    prepare.prepare(
        prep_unamended,
        benchmark_path,
        commit,
        mirror_path,
        costs_path,
        main_shards=True,
        pending_canary=True,
        target_shards=[7, 11, 12],
        canary_gpus=["A100", "L4"],
    )
    for s_a100 in (7, 11):
        mf_s = json.loads(
            (prep_unamended / f"shard-{s_a100:02d}.pending.json").read_text(encoding="utf-8")
        )
        assert mf_s["requirements"]["gpu"] == "A100"
        assert mf_s["env"]["WATCH6_WORKERS"] == "6"
        assert "hardware_amendment_id" not in mf_s["watch6_l1_v1"]
        bad_l4 = json.loads(json.dumps(mf_s))
        bad_l4["requirements"]["gpu"] = "L4"
        bad_l4["requirements"]["allowed_gpus"] = ["L4"]
        with pytest.raises(ValueError, match="requires exact >=12 vCPU A100 W=6 placement"):
            admission.admit(bad_l4, pending_qualification=True)
        bad_amend_tag = json.loads(json.dumps(mf_s))
        bad_amend_tag["watch6_l1_v1"]["hardware_amendment_id"] = admission.AMENDMENT_ID
        with pytest.raises(ValueError, match="un-amended shard"):
            admission.admit(bad_amend_tag, pending_qualification=True)

    mf_s12 = json.loads((prep_unamended / "shard-12.pending.json").read_text(encoding="utf-8"))
    assert mf_s12["requirements"]["gpu"] == "L4"
    assert mf_s12["env"]["WATCH6_WORKERS"] == "4"
    bad_s12_w6 = json.loads(json.dumps(mf_s12))
    bad_s12_w6["env"]["WATCH6_WORKERS"] = "6"
    with pytest.raises(ValueError, match="requires exact >=12 vCPU L4 W=4 placement"):
        admission.admit(bad_s12_w6, pending_qualification=True)
    bad_s12_a100 = json.loads(json.dumps(mf_s12))
    bad_s12_a100["requirements"]["gpu"] = "A100"
    bad_s12_a100["requirements"]["allowed_gpus"] = ["A100"]
    with pytest.raises(ValueError, match="requires exact >=12 vCPU L4 W=4 placement"):
        admission.admit(bad_s12_a100, pending_qualification=True)

    # 6. resume.verify_bound_receipt and resume.build accept L4 W=6 on shard 08 and reject others.
    s08_dir = tmp_path / "exec-shard-08"
    (s08_dir / "exports/efficacy").mkdir(parents=True)
    panel = _synthetic_panel()
    alloc = p.allocation()
    idx48 = p.shard_indices(8)[0]
    w48_path = s08_dir / f"exports/efficacy/world-{idx48:02d}.json"
    w48_desc = publish_json(w48_path, panel[idx48])
    publish_json(
        w48_path.with_suffix(".meta.json"),
        {
            "namespace": p.NAMESPACE,
            "science": sci,
            "seed_kind": "primary",
            "allocation": alloc[idx48],
            "fixture": False,
            "sha256": w48_desc["sha256"],
            "size": w48_desc["size"],
        },
    )
    (s08_dir / "manifest.json").write_text(json.dumps(mf08), encoding="utf-8")
    s08_rec_path = s08_dir / "receipt.json"
    s08_rec = {
        "job_id": mf08["job_id"],
        "commit": mf08["commit"],
        "manifest_sha256": digest(s08_dir / "manifest.json"),
        "status": "INCOMPLETE",
        "endpoint": "m-s-s08",
        "stop_verified": True,
        "cleanup_error": None,
        "failure_class": "CHILD_NONZERO_EXIT",
        "terminal": {"returncode": 75, "reason": "timeout", "error": None},
        "exports": [
            {
                "path": f"efficacy/world-{idx48:02d}.json",
                "sha256": w48_desc["sha256"],
                "size": w48_desc["size"],
            }
        ],
    }
    s08_rec_path.write_text(json.dumps(s08_rec), encoding="utf-8")
    verified_mf, verified_rec = resume.verify_bound_receipt(s08_rec_path)
    assert verified_mf["requirements"]["gpu"] == "L4"
    assert verified_rec["job_id"] == mf08["job_id"]
    ledger_payload = resume.build([w48_path], [s08_rec_path])
    assert [u["index"] for u in ledger_payload["units"]] == [idx48]

    # Reject shard 08 receipt if manifest had A100 or W=4, and reject shard 11 receipt if on L4:
    for bad_patch in (
        {"requirements": {"gpu": "A100", "allowed_gpus": ["A100"], "min_vcpu": 12}},
        {"env": {**mf08["env"], "WATCH6_WORKERS": "4"}},
    ):
        bad_s08_mf = {**mf08, **bad_patch}
        (s08_dir / "manifest.json").write_text(json.dumps(bad_s08_mf), encoding="utf-8")
        s08_rec_path.write_text(
            json.dumps({**s08_rec, "manifest_sha256": digest(s08_dir / "manifest.json")}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="GPU/worker placement mismatch"):
            resume.verify_bound_receipt(s08_rec_path)

    bad_s11_mf = {
        **mf08,
        "env": {**mf08["env"], "WATCH6_SHARD": "11"},
    }
    (s08_dir / "manifest.json").write_text(json.dumps(bad_s11_mf), encoding="utf-8")
    s08_rec_path.write_text(
        json.dumps({**s08_rec, "manifest_sha256": digest(s08_dir / "manifest.json")}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="GPU/worker placement mismatch"):
        resume.verify_bound_receipt(s08_rec_path)

    # Restore valid s08 manifest/receipt.
    (s08_dir / "manifest.json").write_text(json.dumps(mf08), encoding="utf-8")
    s08_rec_path.write_text(json.dumps(s08_rec), encoding="utf-8")

    # 7. Amendment with extra shard [8, 9, 10, 11], wrong hash, or missing file is rejected.
    orig_amend_data = json.loads(admission.AMENDMENT_PATH.read_text(encoding="utf-8"))
    extra_shard_path = tmp_path / "amendment-extra-shard.json"
    extra_payload = {**orig_amend_data, "shards": [8, 9, 10, 11]}
    extra_shard_path.write_text(json.dumps(extra_payload), encoding="utf-8")
    extra_sha = digest(extra_shard_path)

    # Wrong hash against pinned AMENDMENT_SHA256 rejected:
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        admission.validate_hardware_amendment(extra_shard_path)

    # Even if hash were matched, extra shard [8, 9, 10, 11] is rejected by schema/shard guard:
    with pytest.raises(ValueError, match="invalid hardware reassignment amendment payload"):
        admission.validate_hardware_amendment(extra_shard_path, expected_sha256=extra_sha)

    # Missing amendment file rejected:
    with pytest.raises(ValueError, match="missing or symlinked hardware reassignment amendment"):
        admission.validate_hardware_amendment(tmp_path / "nonexistent-amendment.json")

    # Monkeypatching AMENDMENT_PATH to tampered file rejects effective_shard_spec, admit,
    # resume.verify_bound_receipt, and final_look.run_final_look:
    monkeypatch.setattr(admission, "AMENDMENT_PATH", extra_shard_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        admission.effective_shard_spec(8)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        admission.admit(mf08, pending_qualification=False)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resume.verify_bound_receipt(s08_rec_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        final_look.run_final_look(
            ops_dir=tmp_path,
            out_dir=tmp_path / "bad-amend-final",
            ruling_json=ruling_json,
        )


def test_host_interrupted_incident_accepted_and_refusals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Test synthetic shard-11 host-interrupted incident acceptance, recovery, and refusals."""
    import hashlib

    from research.watch6_l1_confirm import recover

    # Build a synthetic fixture tree + synthetic incident file (NOT the real ops/ tree).
    incident_dir = tmp_path / "incident_cfg"
    incident_dir.mkdir(parents=True)
    fixture_incident_path = incident_dir / "INCIDENT_HOST_RESTART_01.json"

    ops_dir = tmp_path / "ops"
    mf_dir = ops_dir / "manifests"
    mf_dir.mkdir(parents=True)
    meta_input = _make_metadata_archive(tmp_path / "meta-inc")
    meta_inputs_field = [{k: meta_input[k] for k in ("path", "dest", "sha256")}]

    ruling_md = tmp_path / "RULING_L1_V1.md"
    ruling_md.write_text("# Ruling\nStatus: RATIFIED\nVerdict: APPROVE\n", encoding="utf-8")
    ruling_json = tmp_path / "ruling-l1-v1.json"
    ratify.record_ruling(ruling_md, ruling_json, judge="Astra")

    # Passing A100 and L4 canary receipts.
    canary_recs: dict[str, Path] = {}
    for gpu, c_shard in (("A100", 0), ("L4", 12)):
        c_out = tmp_path / f"canary-{gpu.lower()}-out"
        (c_out / "exports").mkdir(parents=True)
        publish_json(
            c_out / "exports/performance.json",
            {
                "schema": "watch6-l1-canary-performance/v1",
                "science": p.identity(),
                "fixture": True,
                "frames_executed": [300],
                "native_v8": {
                    "kernel_rows": 9000,
                    "stock_rows": 0,
                    "fallback_rows": 0,
                    "noncandidate_rows": 0,
                },
            },
        )
        c_mf = {
            "job_id": f"watch6-l1-canary-{gpu.lower()}",
            "commit": "1873b9025d5a563dcefaf7fcb3cda1e4e6e25306",
            "output_dir": str(c_out),
            "inputs": meta_inputs_field,
            "requirements": {"gpu": gpu, "allowed_gpus": [gpu], "min_vcpu": 12},
            "canary": {"schema": canary.SCHEMA, "mode": "probe", "family": "fam"},
            "env": {
                "WATCH6_FIXTURE": "1",
                "WATCH6_SHARDS": str(p.SHARDS),
                "WATCH6_SHARD": str(c_shard),
                "WATCH6_WORKERS": str(p.shard_spec(c_shard)["workers"]),
                "WATCH6_METADATA_SHA256": meta_input["sha256"],
            },
            "watch6_l1_v1": {
                "namespace": p.NAMESPACE,
                "science": p.identity(),
                "primary_allocation": p.allocation(),
            },
        }
        (c_out / "manifest.json").write_text(json.dumps(c_mf), encoding="utf-8")
        c_rec = {
            "job_id": c_mf["job_id"],
            "commit": c_mf["commit"],
            "manifest_sha256": digest(c_out / "manifest.json"),
            "status": "PASS",
            "endpoint": f"m-s-canary-{gpu.lower()}",
            "stop_verified": True,
            "cleanup_error": None,
            "failure_class": None,
            "error": None,
            "terminal": {"returncode": 0, "reason": None, "error": None},
            "exports": [],
        }
        c_rec_path = c_out / "receipt.json"
        c_rec_path.write_text(json.dumps(c_rec), encoding="utf-8")
        (mf_dir / f"canary-{gpu.lower()}.json").write_text(json.dumps(c_mf), encoding="utf-8")
        canary_recs[gpu] = c_rec_path

    panel = _synthetic_panel()
    alloc = p.allocation()

    # Shard 11 attempt 0: prestart failure (kernel-creation GuardError before ready/hardware.json).
    s11_att0 = tmp_path / "shard-11-att0"
    (s11_att0 / "staging").mkdir(parents=True)
    for fname in admission.ALLOWED_PREALLOC_STAGING_FILES:
        (s11_att0 / "staging" / fname).write_bytes(f"staged:{fname}".encode("utf-8"))
    (s11_att0 / "warmup-measurement.json").write_text(
        '{"status": "INCOMPLETE", "error": "GuardError: kernel creation already attempted"}\n',
        encoding="utf-8",
    )
    s11_spec = admission.effective_shard_spec(11)
    s11_att0_mf = {
        "job_id": "watch6-l1v1-shard-11",
        "commit": "1873b9025d5a563dcefaf7fcb3cda1e4e6e25306",
        "output_dir": str(s11_att0),
        "inputs": meta_inputs_field,
        "requirements": {"gpu": "A100", "allowed_gpus": ["A100"], "min_vcpu": 12},
        "env": {
            "WATCH6_FIXTURE": "0",
            "WATCH6_SHARDS": str(p.SHARDS),
            "WATCH6_SHARD": "11",
            "WATCH6_WORKERS": str(s11_spec["workers"]),
            "WATCH6_METADATA_SHA256": meta_input["sha256"],
        },
        "watch6_l1_v1": {
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "primary_allocation": alloc,
            "prestart_retries_used": 0,
            "compute_completion_attempt": 0,
            "total_dispatches_for_shard": 1,
            "prior_receipts": [],
        },
    }
    (s11_att0 / "manifest.json").write_text(json.dumps(s11_att0_mf), encoding="utf-8")
    s11_att0_rec = {
        "job_id": s11_att0_mf["job_id"],
        "commit": s11_att0_mf["commit"],
        "manifest_sha256": digest(s11_att0 / "manifest.json"),
        "status": "INCOMPLETE",
        "endpoint": "m-s-s11-att0",
        "stop_verified": True,
        "cleanup_error": None,
        "failure_class": "SUPERVISOR",
        "error": (
            "Refused('colab exec failed (1):\\n"
            "GuardError: kernel creation already attempted; refusing uncertain retry\\n')"
        ),
        "cli_calls": [{"command": "exec", "status": "failed"}],
        "terminal": None,
        "exports": [],
    }
    (s11_att0 / "receipt.json").write_text(json.dumps(s11_att0_rec), encoding="utf-8")
    (mf_dir / "watch6-l1v1-shard-11.json").write_text(json.dumps(s11_att0_mf), encoding="utf-8")

    # Shard 11 attempt 1: receipt-less host-interrupted compute completion attempt #1 of 2.
    s11_retry_root = ops_dir / "auto-retry/watch6-l1v1-shard-11-retry-synth"
    s11_att1 = s11_retry_root / "output"
    (s11_att1 / "exports").mkdir(parents=True)
    (s11_att1 / "staging").mkdir(parents=True)
    for fname in admission.ALLOWED_PREALLOC_STAGING_FILES:
        (s11_att1 / "staging" / fname).write_bytes(f"staged:{fname}".encode("utf-8"))
    publish_json(s11_att1 / "exports/performance.json", {"frames_executed": [300]})
    publish_json(s11_att1 / "exports/workload-started.json", {"started": True})
    (s11_att1 / "hardware.json").write_text(
        '{"cpu_count": 12, "gpu": "A100", "ready": true}\n',
        encoding="utf-8",
    )

    # Write 1083 contiguous poll files poll-000000.json .. poll-001082.json.
    for i in range(admission.INCIDENT_POLL_COUNT):
        poll_payload = {
            "poll": i,
            "state": {
                "status": "running",
                "returncode": None,
                "reason": None,
                "error": None,
                "updated_at": (
                    float(admission.INCIDENT_LAST_POLL_UPDATED_AT)
                    if i == admission.INCIDENT_POLL_COUNT - 1
                    else float(admission.INCIDENT_LAST_POLL_UPDATED_AT - 1000 + i)
                ),
            },
            "terminal": None,
        }
        (s11_att1 / f"poll-{i:06d}.json").write_text(
            json.dumps(poll_payload),
            encoding="utf-8",
        )

    s11_att1_mf = {
        **s11_att0_mf,
        "job_id": "watch6-l1v1-shard-11-retry-synth",
        "completion_attempt": 1,
        "output_dir": str(s11_att1),
    }
    publish_json(s11_retry_root / "manifest.json", s11_att1_mf)
    s11_att1_mf_text = (s11_retry_root / "manifest.json").read_text(encoding="utf-8")
    (s11_att1 / "manifest.json").write_text(s11_att1_mf_text, encoding="utf-8")
    (mf_dir / "watch6-l1v1-shard-11-retry-synth.json").write_text(
        s11_att1_mf_text,
        encoding="utf-8",
    )
    retry_json_payload = {
        "schema": "ops-zero-export-retry/v1",
        "completion_attempt": 1,
        "max_completion_attempts": 1,
        "prior_manifest": {
            "path": str((mf_dir / "watch6-l1v1-shard-11.json").resolve()),
            "sha256": digest(mf_dir / "watch6-l1v1-shard-11.json"),
        },
        "prior_receipt": {
            "path": str((s11_att0 / "receipt.json").resolve()),
            "sha256": digest(s11_att0 / "receipt.json"),
        },
        "manifest_sha256": digest(s11_att1 / "manifest.json"),
        "unallocated_proof": None,
        "ledger_action": "preserve original uploaded ledger; zero new exports",
    }
    publish_json(s11_retry_root / "RETRY.json", retry_json_payload)

    sup_log_path = ops_dir / "logs/watch6-l1v1-shard-11-retry-synth.log"
    sup_log_path.parent.mkdir(parents=True)
    sup_log_path.write_text(
        "".join(f'{{"event": "poll", "poll": {i}}}\n' for i in range(1083)),
        encoding="utf-8",
    )

    sessions_out_text = "No active sessions.\n"
    fixture_incident = {
        "schema": admission.INCIDENT_SCHEMA,
        "incident_id": admission.INCIDENT_ID,
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "registration_sha256": digest(p.registration()),
        "shard": 11,
        "compute_completion_attempt": 1,
        "classification": "external_transport_compute",
        "attempt_output_dir": str(s11_att1.resolve()),
        "attempt_manifest_sha256": digest(s11_att1 / "manifest.json"),
        "manifest_commit": "1873b9025d5a563dcefaf7fcb3cda1e4e6e25306",
        "gpu": "A100",
        "workers": 6,
        "hardware_json_sha256": digest(s11_att1 / "hardware.json"),
        "last_poll": {
            "name": "poll-001082.json",
            "sha256": digest(s11_att1 / "poll-001082.json"),
            "state_status": "running",
            "updated_at": admission.INCIDENT_LAST_POLL_UPDATED_AT,
        },
        "poll_count": admission.INCIDENT_POLL_COUNT,
        "supervisor_log": {
            "path": str(sup_log_path.resolve()),
            "sha256": digest(sup_log_path),
            "lines": 1083,
            "last_write_pt": "2026-11-02T17:34:46",
        },
        "host_boot_epoch": admission.INCIDENT_HOST_BOOT_EPOCH,
        "sessions_evidence": {
            "captured_pt": "2026-11-02T18:05:00",
            "command": "colab sessions",
            "output_sha256": hashlib.sha256(sessions_out_text.encode("utf-8")).hexdigest(),
            "output_text": sessions_out_text,
        },
        "exports_listing": list(admission.INCIDENT_EXPORTS_LISTING),
        "world_outputs": 0,
        "ruling": admission.INCIDENT_RULING,
        "ruling_sha256": "78bcb2ff6f682ce73f422743acea1b8e06ce23fb1005bcedea034b17058dda5b",
    }
    fixture_incident_path.write_text(
        json.dumps(fixture_incident, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fixture_incident_sha = digest(fixture_incident_path)
    monkeypatch.setattr(admission, "INCIDENT_PATH", fixture_incident_path)
    monkeypatch.setattr(admission, "INCIDENT_SHA256", fixture_incident_sha)

    # 1. Incident is accepted for the synthetic shard-11 receipt-less attempt:
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is True

    # 2. Test recover.build_recovery_packet on shard 11 after attempt 0 + receipt-less attempt 1:
    #    It must count 1 prestart + 1 compute failure (next attempt is completion #2 of 2).
    monkeypatch.setattr(
        "research.watch6_l1_confirm.prepare._require_published_pin", lambda _c: None
    )
    monkeypatch.setattr(canary, "check", lambda *_a, **_k: {"status": "PASS"})
    monkeypatch.setattr(
        admission,
        "check_uploads",
        lambda _m: {"status": "PASS", "upload_count": 1, "total_bytes": 1024},
    )
    benchmark_path = tmp_path / "benchmark.json"
    publish_json(
        benchmark_path,
        {
            "schema": "watch6-l1-mac-preflight/v1",
            "status": "PASS",
            "science": p.identity(),
            "fixture": True,
        },
    )
    mirror_path = tmp_path / "mac-mirror.json"
    publish_json(
        mirror_path,
        {
            "schema": mirror.SCHEMA,
            "status": "PASS",
            "platform": "macOS-test",
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "fixture": True,
            "seed": p.FIXTURE_SEEDS[0],
            "horizon": p.HORIZON,
            "total_mirrored_frames": p.HORIZON * len(p.ARMS),
            "mirror_mismatches": 0,
            "elapsed_seconds": 12.0,
            "arms": {
                arm: {
                    "frames": p.HORIZON,
                    "mirror_identity_frames": p.HORIZON,
                    "mirror_mismatches": 0,
                    "respawns_per_slot": [1, 1, 1, 1, 1, 1],
                    "batches": 0 if arm == "bare" else 1200,
                    "native_v8_kernel_rows": 0 if arm == "bare" else 90000,
                    "wall_seconds": 4.0,
                    "timing": {
                        "main_seconds": 2.0,
                        "search_seconds": 0.0 if arm == "bare" else 2.0,
                    },
                }
                for arm in p.ARMS
            },
        },
    )
    costs_path = tmp_path / "costs.json"
    publish_json(costs_path, costs.build_cost_report())

    rec_out = tmp_path / "recover-s11"
    rec_pkt = recover.build_recovery_packet(
        prior_receipts_by_shard={11: [s11_att0 / "receipt.json", s11_att1]},
        all_stopped_main_receipts=[s11_att0 / "receipt.json", s11_att1],
        out_dir=rec_out,
        benchmark=benchmark_path,
        commit="1873b9025d5a563dcefaf7fcb3cda1e4e6e25306",
        mirror_receipt=mirror_path,
        empirical_costs=costs_path,
        canary_receipts_by_gpu={"A100": canary_recs["A100"]},
        pending_canary=False,
    )
    assert rec_pkt["shards"] == 18
    rec_mf = json.loads((rec_out / "shard-11.json").read_text(encoding="utf-8"))
    assert rec_mf["watch6_l1_v1"]["prestart_retries_used"] == 1
    assert rec_mf["watch6_l1_v1"]["compute_completion_attempt"] == 1
    assert rec_mf["watch6_l1_v1"]["total_dispatches_for_shard"] == 3
    assert len(rec_mf["watch6_l1_v1"]["prior_receipts"]) == 2
    assert rec_mf["watch6_l1_v1"]["prior_receipts"][0] == {
        "receipt_path": str((s11_att0 / "receipt.json").resolve()),
        "receipt_sha256": digest(s11_att0 / "receipt.json"),
        "classification": "prestart",
    }
    assert rec_mf["watch6_l1_v1"]["prior_receipts"][1] == {
        "receipt_path": str(fixture_incident_path.resolve()),
        "receipt_sha256": fixture_incident_sha,
        "classification": "external_transport_compute",
        "incident_id": admission.INCIDENT_ID,
        "incident_sha256": fixture_incident_sha,
        "attempt_output_dir": str(s11_att1.resolve()),
    }

    # 3. Complete all 18 shards (with shard 11 finishing on completion #2 of 2) and run final_look:
    for shard in range(p.SHARDS):
        spec = admission.effective_shard_spec(shard)
        s_out = tmp_path / f"shard-{shard:02d}-pass"
        eff_dir = s_out / "exports/efficacy"
        eff_dir.mkdir(parents=True)
        indices_to_write = spec["indices"]
        exports_rows: list[dict] = []
        for idx in indices_to_write:
            w_path = eff_dir / f"world-{idx:02d}.json"
            desc = publish_json(w_path, panel[idx])
            publish_json(
                w_path.with_suffix(".meta.json"),
                {
                    "namespace": p.NAMESPACE,
                    "science": p.identity(),
                    "seed_kind": "primary",
                    "allocation": alloc[idx],
                    "fixture": False,
                    "sha256": desc["sha256"],
                    "size": desc["size"],
                },
            )
            exports_rows.append(
                {
                    "path": f"efficacy/world-{idx:02d}.json",
                    "sha256": desc["sha256"],
                    "size": desc["size"],
                }
            )
        if shard == 11:
            s_mf = {
                **rec_mf,
                "output_dir": str(s_out),
                "inputs": meta_inputs_field,
                "env": {
                    **rec_mf["env"],
                    "WATCH6_METADATA_SHA256": meta_input["sha256"],
                },
            }
        else:
            w_bind = {
                "namespace": p.NAMESPACE,
                "science": p.identity(),
                "primary_allocation": alloc,
                "prestart_retries_used": 0,
                "compute_completion_attempt": 0,
                "total_dispatches_for_shard": 1,
                "prior_receipts": [],
            }
            if "amendment_id" in spec:
                w_bind["hardware_amendment_id"] = spec["amendment_id"]
                w_bind["hardware_amendment_sha256"] = spec["amendment_sha256"]
            s_mf = {
                "job_id": f"watch6-l1v1-shard-{shard:02d}",
                "commit": "1873b9025d5a563dcefaf7fcb3cda1e4e6e25306",
                "output_dir": str(s_out),
                "inputs": meta_inputs_field,
                "requirements": {
                    "gpu": spec["gpu"],
                    "allowed_gpus": [spec["gpu"]],
                    "min_vcpu": 12,
                },
                "env": {
                    "WATCH6_FIXTURE": "0",
                    "WATCH6_SHARDS": str(p.SHARDS),
                    "WATCH6_SHARD": str(shard),
                    "WATCH6_WORKERS": str(spec["workers"]),
                    "WATCH6_METADATA_SHA256": meta_input["sha256"],
                },
                "watch6_l1_v1": w_bind,
            }
        (s_out / "manifest.json").write_text(json.dumps(s_mf), encoding="utf-8")
        s_rec = {
            "job_id": s_mf["job_id"],
            "commit": s_mf["commit"],
            "manifest_sha256": digest(s_out / "manifest.json"),
            "status": "PASS",
            "endpoint": f"m-s-shard-{shard:02d}",
            "stop_verified": True,
            "cleanup_error": None,
            "failure_class": None,
            "error": None,
            "terminal": {"returncode": 0, "reason": None, "error": None},
            "exports": exports_rows,
        }
        (s_out / "receipt.json").write_text(json.dumps(s_rec), encoding="utf-8")
        (mf_dir / f"shard-{shard:02d}-pass.json").write_text(json.dumps(s_mf), encoding="utf-8")

    decision = final_look.run_final_look(
        ops_dir=ops_dir,
        out_dir=tmp_path / "final-inc-ok",
        ruling_json=ruling_json,
    )
    assert decision["verdict"] == "GO_L1_SUPERIOR"
    assert decision["total_main_attempts"] == 20  # 18 PASS + s11 att0 + s11 host-interrupted att1
    assert decision["incident"] == {
        "incident_id": admission.INCIDENT_ID,
        "id": admission.INCIDENT_ID,
        "sha256": fixture_incident_sha,
        "attempt_output_dir": str(s11_att1.resolve()),
        "attempt_dir": str(s11_att1.resolve()),
        "shard": 11,
        "compute_completion_attempt": 1,
        "classification": "external_transport_compute",
    }
    acct = json.loads((tmp_path / "final-inc-ok/pre-outcome-accounting.json").read_text("utf-8"))
    assert acct["host_interrupted_incident"] == decision["incident"]
    assert acct["incident"] == decision["incident"]

    # 4. Refusal (a): A second receipt-less directory (or other shard) or second INCIDENT_* file.
    second_receiptless = tmp_path / "second-receiptless"
    second_receiptless.mkdir()
    (second_receiptless / "manifest.json").write_text(s11_att1_mf_text, encoding="utf-8")
    assert admission.is_host_interrupted_compute_attempt(second_receiptless) is False
    second_mf = {**s11_att1_mf, "output_dir": str(second_receiptless)}
    second_mf_path = mf_dir / "shard-11-second-receiptless.json"
    second_mf_path.write_text(json.dumps(second_mf), encoding="utf-8")
    with pytest.raises(ValueError, match=r"unfinished runtime \(missing receipt\.json\)"):
        final_look.run_final_look(
            ops_dir=ops_dir,
            out_dir=tmp_path / "final-bad-second-dir",
            ruling_json=ruling_json,
        )
    second_mf_path.unlink()

    second_incident_file = incident_dir / "INCIDENT_HOST_RESTART_02.json"
    second_incident_file.write_text("{}", encoding="utf-8")
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is False
    with pytest.raises(ValueError, match="multiple INCIDENT_"):
        admission.validate_host_interrupted_incident()
    second_incident_file.unlink()
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is True

    orig_root = p.ROOT
    fake_root = tmp_path / "fake-pkg-root"
    fake_pkg_dir = fake_root / "research/watch6_l1_confirm"
    fake_pkg_dir.mkdir(parents=True)
    (fake_pkg_dir / "INCIDENT_HOST_RESTART_01.json").write_bytes(fixture_incident_path.read_bytes())
    (fake_pkg_dir / "INCIDENT_HOST_RESTART_02.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(p, "ROOT", fake_root)
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is False
    with pytest.raises(ValueError, match=r"expected exactly one INCIDENT_\* file"):
        admission.validate_host_interrupted_incident()
    monkeypatch.setattr(p, "ROOT", orig_root)
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is True

    # 5. Refusal (b): Tampered incident file (without updating INCIDENT_SHA256).
    orig_incident_bytes = fixture_incident_path.read_bytes()
    tampered_incident = {**fixture_incident, "world_outputs": 1}
    fixture_incident_path.write_text(json.dumps(tampered_incident), encoding="utf-8")
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is False
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        admission.validate_host_interrupted_incident()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        final_look.run_final_look(
            ops_dir=ops_dir,
            out_dir=tmp_path / "final-bad-tampered-inc",
            ruling_json=ruling_json,
        )
    fixture_incident_path.write_bytes(orig_incident_bytes)
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is True

    # 6. Refusal (c): Extra file or .DS_Store under exports/.
    extra_export = s11_att1 / "exports/extra-file.txt"
    extra_export.write_text("unexpected\n", encoding="utf-8")
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is False
    with pytest.raises(ValueError, match=r"unfinished runtime \(missing receipt\.json\)"):
        final_look.run_final_look(
            ops_dir=ops_dir,
            out_dir=tmp_path / "final-bad-extra-export",
            ruling_json=ruling_json,
        )
    extra_export.unlink()
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is True

    ds_store_export = s11_att1 / "exports/.DS_Store"
    ds_store_export.write_bytes(b"macos")
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is False
    with pytest.raises(ValueError, match=r"unfinished runtime \(missing receipt\.json\)"):
        final_look.run_final_look(
            ops_dir=ops_dir,
            out_dir=tmp_path / "final-bad-ds-store-export",
            ruling_json=ruling_json,
        )
    ds_store_export.unlink()
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is True

    # 7. Refusal (d): Stray world-66.json under exports/efficacy/ AND outside efficacy/ (os.walk).
    (s11_att1 / "exports/efficacy").mkdir(parents=True)
    stray_w66 = s11_att1 / "exports/efficacy/world-66.json"
    stray_w66.write_text("{}", encoding="utf-8")
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is False
    with pytest.raises(ValueError, match=r"unfinished runtime \(missing receipt\.json\)"):
        final_look.run_final_look(
            ops_dir=ops_dir,
            out_dir=tmp_path / "final-bad-stray-w66",
            ruling_json=ruling_json,
        )
    stray_w66.unlink()
    (s11_att1 / "exports/efficacy").rmdir()
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is True

    stray_w66_outside = s11_att1 / "staging/world-66.json"
    stray_w66_outside.write_text("{}", encoding="utf-8")
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is False
    with pytest.raises(ValueError, match=r"unfinished runtime \(missing receipt\.json\)"):
        final_look.run_final_look(
            ops_dir=ops_dir,
            out_dir=tmp_path / "final-bad-stray-w66-outside",
            ruling_json=ruling_json,
        )
    stray_w66_outside.unlink()
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is True

    # 8. Refusal (e): Poll-count mismatch and non-contiguous poll names.
    extra_poll = s11_att1 / "poll-001083.json"
    extra_poll.write_text('{"poll": 1083, "state": {"status": "running"}}', encoding="utf-8")
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is False
    with pytest.raises(ValueError, match=r"unfinished runtime \(missing receipt\.json\)"):
        final_look.run_final_look(
            ops_dir=ops_dir,
            out_dir=tmp_path / "final-bad-poll-count",
            ruling_json=ruling_json,
        )
    extra_poll.unlink()
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is True

    poll_500 = s11_att1 / "poll-000500.json"
    poll_499a = s11_att1 / "poll-000499a.json"
    poll_500.rename(poll_499a)
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is False
    with pytest.raises(ValueError, match=r"unfinished runtime \(missing receipt\.json\)"):
        final_look.run_final_look(
            ops_dir=ops_dir,
            out_dir=tmp_path / "final-bad-poll-gap",
            ruling_json=ruling_json,
        )
    poll_499a.rename(poll_500)
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is True

    # 9. Refusal (f): Supervisor log-sha mismatch.
    orig_log_bytes = sup_log_path.read_bytes()
    sup_log_path.write_bytes(orig_log_bytes + b'{"event": "extra"}\n')
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is False
    with pytest.raises(ValueError, match=r"unfinished runtime \(missing receipt\.json\)"):
        final_look.run_final_look(
            ops_dir=ops_dir,
            out_dir=tmp_path / "final-bad-log-sha",
            ruling_json=ruling_json,
        )
    sup_log_path.write_bytes(orig_log_bytes)
    assert admission.is_host_interrupted_compute_attempt(s11_att1) is True

    # 10. Refusal (g): Non-canonical prior_receipts representation (attempt_manifest_sha256 or
    #     missing incident fields) refused in both admission and final_look.
    canonical_priors = rec_mf["watch6_l1_v1"]["prior_receipts"]
    bad_sha_priors = [
        canonical_priors[0],
        {
            **canonical_priors[1],
            "receipt_sha256": fixture_incident["attempt_manifest_sha256"],
        },
    ]
    with pytest.raises(ValueError, match="host-interrupted prior SHA-256 mismatch"):
        admission.verify_prior_receipts_lineage(11, bad_sha_priors, 1, 1, 3)
    for missing_key in ("incident_id", "incident_sha256", "attempt_output_dir"):
        stripped_entry = {k: v for k, v in canonical_priors[1].items() if k != missing_key}
        with pytest.raises(ValueError, match="host-interrupted"):
            admission.verify_prior_receipts_lineage(
                11,
                [canonical_priors[0], stripped_entry],
                1,
                1,
                3,
            )

    s11_pass_out = tmp_path / "shard-11-pass"
    orig_s11_pass_mf_bytes = (s11_pass_out / "manifest.json").read_bytes()
    orig_s11_pass_rec_bytes = (s11_pass_out / "receipt.json").read_bytes()
    orig_s11_ops_mf_bytes = (mf_dir / "shard-11-pass.json").read_bytes()
    s11_pass_mf = json.loads(orig_s11_pass_mf_bytes.decode("utf-8"))
    bad_s11_pass_mf = {
        **s11_pass_mf,
        "watch6_l1_v1": {
            **s11_pass_mf["watch6_l1_v1"],
            "prior_receipts": bad_sha_priors,
        },
    }
    (s11_pass_out / "manifest.json").write_text(json.dumps(bad_s11_pass_mf), encoding="utf-8")
    s11_pass_rec = json.loads(orig_s11_pass_rec_bytes.decode("utf-8"))
    (s11_pass_out / "receipt.json").write_text(
        json.dumps({**s11_pass_rec, "manifest_sha256": digest(s11_pass_out / "manifest.json")}),
        encoding="utf-8",
    )
    (mf_dir / "shard-11-pass.json").write_text(json.dumps(bad_s11_pass_mf), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or mismatched prior receipt"):
        final_look.run_final_look(
            ops_dir=ops_dir,
            out_dir=tmp_path / "final-bad-prior-sha",
            ruling_json=ruling_json,
        )
    (s11_pass_out / "manifest.json").write_bytes(orig_s11_pass_mf_bytes)
    (s11_pass_out / "receipt.json").write_bytes(orig_s11_pass_rec_bytes)
    (mf_dir / "shard-11-pass.json").write_bytes(orig_s11_ops_mf_bytes)

    # 11. Refusal (h): Receipt carrying the "host_interrupted" marker key is not treated as the
    #     incident and is refused on disk.
    with pytest.raises(ValueError, match="forbidden failure class"):
        final_look._classify_attempt(
            s11_att1,
            {"status": "INCOMPLETE", "host_interrupted": True, "exports": [], "terminal": None},
        )
    assert (
        final_look._classify_attempt(
            s11_att0,
            {**s11_att0_rec, "host_interrupted": True},
            receipt_path=(s11_att0 / "receipt.json").resolve(),
            receipt_sha256=digest(s11_att0 / "receipt.json"),
        )
        == "prestart"
    )
    orig_att0_rec_bytes = (s11_att0 / "receipt.json").read_bytes()
    (s11_att0 / "receipt.json").write_text(
        json.dumps({**s11_att0_rec, "host_interrupted": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden host_interrupted key"):
        final_look._verify_attempt_stop_and_hashes(s11_att0)
    with pytest.raises(ValueError, match="forbidden host_interrupted key"):
        final_look.run_final_look(
            ops_dir=ops_dir,
            out_dir=tmp_path / "final-bad-marker-key",
            ruling_json=ruling_json,
        )
    (s11_att0 / "receipt.json").write_bytes(orig_att0_rec_bytes)


REAL_SHARD11_HOST_INTERRUPTED_DIR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/auto-retry/"
    "watch6-l1v1-shard-11-retry-1cf3c2c133/output"
)
REAL_SHARD11_SUPERVISOR_LOG = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/logs/"
    "watch6-l1v1-shard-11-retry-1cf3c2c133.log"
)


@pytest.mark.skipif(
    not (REAL_SHARD11_HOST_INTERRUPTED_DIR.is_dir() and REAL_SHARD11_SUPERVISOR_LOG.is_file()),
    reason="real shard-11 host-interrupted attempt evidence not present on this machine",
)
def test_real_shard11_host_interrupted_attempt_validates_read_only() -> None:
    """Read-only check that the real shard-11 attempt validates against INCIDENT_HOST_RESTART_01."""
    inc = admission.validate_host_interrupted_incident()
    assert inc["incident_id"] == admission.INCIDENT_ID
    assert inc["sha256"] == admission.INCIDENT_SHA256
    assert inc["shard"] == 11
    assert inc["compute_completion_attempt"] == 1
    assert inc["world_outputs"] == 0
    assert admission.is_host_interrupted_compute_attempt(REAL_SHARD11_HOST_INTERRUPTED_DIR) is True


def test_shard03_prealloc_accounting_baseline_shape_and_refusals(tmp_path: Path) -> None:
    """Verify R7 pre-allocation refusal shape with accounting-baseline.json and all refusals."""
    shard03_dir = tmp_path / "output-shard-03"
    shard03_staging = shard03_dir / "staging"
    shard03_staging.mkdir(parents=True)
    (shard03_dir / "manifest.json").write_text(
        '{"job_id": "watch6-l1-shard-03-l1v1-main-01"}\n',
        encoding="utf-8",
    )
    baseline_payload = {
        "assignments": [
            {
                "endpoint": "gpu-a100-s-ass1c0-2ued36dh85ogk",
                "rate_units_per_hour": 6.73,
                "session": "?",
                "source": "broker.active",
            },
            {
                "endpoint": "gpu-a100-s-ass1c0-3fprojzw40j5k",
                "rate_units_per_hour": 6.73,
                "session": "?",
                "source": "broker.active",
            },
            {
                "endpoint": "gpu-a100-s-usc1b0-rccadq0b9wuq",
                "rate_units_per_hour": 6.73,
                "session": "?",
                "source": "broker.active",
            },
        ],
        "baseline_balance_units": 2108.15,
        "baseline_rate_units_per_hour": 20.19,
        "declared_other_rate_units_per_hour": 20.19,
        "unaccounted_endpoints": [],
    }
    baseline_path = shard03_dir / "accounting-baseline.json"
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")
    for fname in (
        "bootstrap.sh",
        "release.py",
        "runtime.py",
        "schema-validation.json",
        "source-plan.json",
        "source.py",
        "source.tar.gz",
    ):
        (shard03_staging / fname).write_bytes(f"staged:{fname}".encode("utf-8"))

    shard03_rec = {
        "accounting": {
            "accounting_rate_basis": "priced_informational",
            "allocated_endpoint": None,
            "assignments": baseline_payload["assignments"],
            "baseline_balance_units": 2108.15,
            "baseline_rate_units_per_hour": 20.19,
            "calibration": None,
            "declared_other_rate_units_per_hour": 20.19,
            "integrated_accounted_other_units": 0.0,
            "last_crosscheck": None,
            "observed_rate_units_per_hour": None,
            "own_rate_units_per_hour": 8.0,
            "priced_rate_units_per_hour": 8.0,
            "rate_measurement_unavailable": True,
            "unaccounted_endpoints": [],
            "unit_cap": 152.0,
        },
        "cleanup_error": admission.RETAINED_PENDING_CLEANUP_ERROR,
        "cli_calls": [
            {"attempt": 1, "command": "version", "operation": "control", "status": "completed"},
            {"attempt": 1, "command": "sessions", "operation": "control", "status": "completed"},
            {"attempt": 1, "command": "usage", "operation": "control", "status": "completed"},
            {"attempt": 1, "command": "new", "operation": "control", "status": "failed"},
        ],
        "commit": "1873b9025d5a563dcefaf7fcb3cda1e4e6e25306",
        "contents_transfers": [],
        "control_operations": [],
        "elapsed_seconds": 3.43,
        "endpoint": None,
        "error": (
            'Refused("colab new failed (1): [colab] Creating session '
            "'snake-watch6-l1-shard-03-l1v1-main-01-0bc089f05593'...\\n"
            '[colab] Allocation refused (precondition failed).")'
        ),
        "exec_poll_recovery_base_seconds": 600.0,
        "exec_poll_recovery_seconds": 1200.0,
        "exports": [],
        "failure_class": "SUPERVISOR",
        "generation": "0bc089f0559341ee98fa1c7d8b2eefd3",
        "job_id": "watch6-l1-shard-03-l1v1-main-01",
        "manifest_sha256": digest(shard03_dir / "manifest.json"),
        "session": "snake-watch6-l1-shard-03-l1v1-main-01-0bc089f05593",
        "status": "INCOMPLETE",
        "stop_verified": False,
        "supervisor_sha256": "12da8937ce9cee01ddb5984a7e3a2eca1dde20bfc127921c230bcc524b172ec9",
        "terminal": None,
        "transfers": [],
        "usage_after": None,
        "usage_before": {"assignments": 3.0, "balance": 2108.15, "rate": 20.19},
        "warmup_seconds": 0.0,
    }
    (shard03_dir / "receipt.json").write_text(json.dumps(shard03_rec), encoding="utf-8")

    # (a) Exact shard-03 shape classifies as prestart (and not compute):
    assert admission._prealloc_tree_clean(shard03_dir) is True
    assert admission.is_prestart_transport_failure(shard03_rec, shard03_dir) is True
    assert admission.is_external_transport_compute_failure(shard03_rec, shard03_dir) is False

    # (b.1) accounting-baseline.json replaced by a symlink -> False:
    ext_target = tmp_path / "external-accounting-baseline.json"
    ext_target.write_text(json.dumps(baseline_payload), encoding="utf-8")
    baseline_path.unlink()
    baseline_path.symlink_to(ext_target)
    assert admission.is_prestart_transport_failure(shard03_rec, shard03_dir) is False
    baseline_path.unlink()
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")
    assert admission.is_prestart_transport_failure(shard03_rec, shard03_dir) is True

    # (b.2) accounting-baseline.json placed under staging/ -> False:
    staged_baseline = shard03_staging / "accounting-baseline.json"
    staged_baseline.write_text(json.dumps(baseline_payload), encoding="utf-8")
    assert admission.is_prestart_transport_failure(shard03_rec, shard03_dir) is False
    baseline_path.unlink()
    assert admission.is_prestart_transport_failure(shard03_rec, shard03_dir) is False
    staged_baseline.unlink()
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")
    assert admission.is_prestart_transport_failure(shard03_rec, shard03_dir) is True

    # (b.3) Additional unknown top-level file (e.g. intent.json, or unallowed sidecars) -> False:
    for bad_top_name in (
        "intent.json",
        "accounting-baseline.json.sha256",
        "accounting-baseline.json.complete",
    ):
        bad_top = shard03_dir / bad_top_name
        bad_top.write_text("{}\n", encoding="utf-8")
        assert admission.is_prestart_transport_failure(shard03_rec, shard03_dir) is False
        bad_top.unlink()
    assert admission.is_prestart_transport_failure(shard03_rec, shard03_dir) is True

    # (b.4) hardware.json present -> False:
    hw_path = shard03_dir / "hardware.json"
    hw_path.write_text('{"cpu_count": 12}\n', encoding="utf-8")
    assert admission.is_prestart_transport_failure(shard03_rec, shard03_dir) is False
    hw_path.unlink()

    # (b.5) poll-0001.json present -> False:
    poll_path = shard03_dir / "poll-0001.json"
    poll_path.write_text('{"state": {"status": "running"}}\n', encoding="utf-8")
    assert admission.is_prestart_transport_failure(shard03_rec, shard03_dir) is False
    poll_path.unlink()

    # (b.6) usage-allocated.json present -> False:
    usage_alloc_path = shard03_dir / "usage-allocated.json"
    usage_alloc_path.write_text("{}\n", encoding="utf-8")
    assert admission.is_prestart_transport_failure(shard03_rec, shard03_dir) is False
    usage_alloc_path.unlink()
    assert admission.is_prestart_transport_failure(shard03_rec, shard03_dir) is True

    # (b.7) endpoint set to a string -> False:
    assert (
        admission.is_prestart_transport_failure(
            {**shard03_rec, "endpoint": "gpu-a100-s-assigned"},
            shard03_dir,
        )
        is False
    )

    # (b.8) transfers non-empty -> False:
    assert (
        admission.is_prestart_transport_failure(
            {**shard03_rec, "transfers": [{"status": "completed", "direction": "upload"}]},
            shard03_dir,
        )
        is False
    )

    # (b.9) failure_class CHILD_NONZERO_EXIT -> False:
    assert (
        admission.is_prestart_transport_failure(
            {**shard03_rec, "failure_class": "CHILD_NONZERO_EXIT"},
            shard03_dir,
        )
        is False
    )

    # (b.10) cleanup_error any other string -> False:
    assert (
        admission.is_prestart_transport_failure(
            {**shard03_rec, "cleanup_error": "STOP_FAILED: session still active"},
            shard03_dir,
        )
        is False
    )


REAL_L1V1_MAIN01_DIR = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/l1v1-main-01")


@pytest.mark.skipif(
    not (
        REAL_L1V1_MAIN01_DIR.is_dir()
        and (REAL_L1V1_MAIN01_DIR / "output-shard-03/receipt.json").is_file()
        and (REAL_L1V1_MAIN01_DIR / "output-shard-01/receipt.json").is_file()
        and (REAL_L1V1_MAIN01_DIR / "output-shard-05/receipt.json").is_file()
        and (REAL_L1V1_MAIN01_DIR / "output-shard-00").is_dir()
    ),
    reason="real l1v1-main-01 attempt directories not present on this machine",
)
def test_real_l1v1_main01_prealloc_shards_validate_read_only() -> None:
    """Read-only check on real l1v1-main-01 shard-03, -01, -05 (True) and shard-00 (False)."""
    with final_look.pre_outcome_open_guard():
        for shard_name in ("output-shard-03", "output-shard-01", "output-shard-05"):
            shard_dir = REAL_L1V1_MAIN01_DIR / shard_name
            mf = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
            rec = json.loads((shard_dir / "receipt.json").read_text(encoding="utf-8"))
            assert isinstance(mf.get("job_id"), str) and bool(mf["job_id"])
            assert admission._prealloc_tree_clean(shard_dir) is True
            assert admission.is_prestart_transport_failure(rec, shard_dir) is True

        shard00_dir = REAL_L1V1_MAIN01_DIR / "output-shard-00"
        assert admission._prealloc_tree_clean(shard00_dir) is False


def test_shard06_source_delivery_poll_deadline_shape_and_refusals(tmp_path: Path) -> None:
    """Verify R8 Proof 5b post-allocation source-delivery poll_rpc deadline shape and refusals."""
    shard06_dir = tmp_path / "output-shard-06"
    shard06_staging = shard06_dir / "staging"
    shard06_staging.mkdir(parents=True)
    (shard06_dir / "manifest.json").write_text(
        '{"job_id": "watch6-l1-shard-06-l1v1-main-01"}\n',
        encoding="utf-8",
    )
    (shard06_dir / "accounting-baseline.json").write_text(
        json.dumps(
            {
                "assignments": [],
                "baseline_balance_units": 1813.64,
                "baseline_rate_units_per_hour": 19.28,
                "declared_other_rate_units_per_hour": 19.28,
                "unaccounted_endpoints": [],
            }
        ),
        encoding="utf-8",
    )
    (shard06_dir / "warmup-measurement.json").write_text(
        json.dumps({"attempts": 1, "elapsed_seconds": 3.432823915965855, "status": "PASS"}),
        encoding="utf-8",
    )
    (shard06_dir / "rate-calibration.json").write_text(
        json.dumps(
            {
                "informational_rate_units_per_hour": 6.73,
                "interval_seconds": 5.0,
                "observed_rate_units_per_hour": 6.73,
                "priced_rate_units_per_hour": 8.0,
                "required_stable_reads": 3,
                "samples": [],
                "stability_units_per_hour": 0.01,
                "status": "PASS",
                "timeout_seconds": 730.0,
            }
        ),
        encoding="utf-8",
    )
    for sample_idx in range(3):
        (shard06_dir / f"rate-calibration-{sample_idx:03d}.json").write_text(
            json.dumps({"sample": sample_idx, "usage": {"assignments": 6.0, "rate": 26.01}}),
            encoding="utf-8",
        )
    upload_payload = {
        "elapsed_seconds": 177.84766391618177,
        "input_transfer_seconds": {"input-metadata.tar.gz": 3.3123799171298742},
        "setup_seconds": 2.7284136661328375,
        "source_delivery_seconds": 165.0086683749687,
        "staging_completed": False,
        "transfer_seconds": 10.109884751029313,
    }
    upload_path = shard06_dir / "upload-measurement.json"
    upload_path.write_text(json.dumps(upload_payload), encoding="utf-8")
    (shard06_dir / "usage-allocated.json").write_text(
        json.dumps({"assignments": 6.0, "balance": 1813.64, "rate": 26.01}),
        encoding="utf-8",
    )
    for fname in (
        "bootstrap.sh",
        "release.py",
        "runtime.py",
        "schema-validation.json",
        "source-plan.json",
        "source.py",
        "source.tar.gz",
    ):
        (shard06_staging / fname).write_bytes(f"staged:{fname}".encode("utf-8"))

    transfers = [
        {
            "direction": "upload",
            "remote": "/content/snake-cli/cc549514489f4e7fbb455d61e7181384/bootstrap.sh",
            "sha256": "0e8c752b7b4602c79415a57a3e14145ba3b9cc1685e7cb3114a4f40cb95d70d4",
            "size_bytes": 3244,
            "status": "verified",
        },
        {
            "direction": "upload",
            "remote": "/content/snake-cli/cc549514489f4e7fbb455d61e7181384/source.py",
            "sha256": "76dd3918da14090d412cfa51e3d9e7f82bb46290aa3f80262e8d723ebdc5080f",
            "size_bytes": 14511,
            "status": "verified",
        },
        {
            "direction": "upload",
            "remote": (
                "/content/snake-cli/cc549514489f4e7fbb455d61e7181384/inputs/input-metadata.tar.gz"
            ),
            "sha256": "cb72db0f9beee6cffab272791cfe7879d27b4c2b3785e1cf4f867755416ac8ff",
            "size_bytes": 299209,
            "status": "verified",
        },
    ]
    control_ops = [
        {"operation": "PUT", "status": "completed", "elapsed_seconds": 1.29},
        {"operation": "renewal-heartbeat", "status": "failed", "elapsed_seconds": 0.54},
        {"operation": "renewal", "status": "completed", "elapsed_seconds": 2.31},
        {"operation": "mkdir", "status": "completed", "elapsed_seconds": 2.73},
        {"operation": "PUT", "status": "completed", "elapsed_seconds": 1.19},
        {"operation": "verify", "status": "completed", "elapsed_seconds": 2.30},
        {"operation": "PUT", "status": "completed", "elapsed_seconds": 1.00},
        {"operation": "verify", "status": "completed", "elapsed_seconds": 2.30},
        {"operation": "PUT", "status": "completed", "elapsed_seconds": 1.49},
        {"operation": "renewal-heartbeat", "status": "completed", "elapsed_seconds": 1.00},
        {"operation": "verify", "status": "completed", "elapsed_seconds": 1.82},
        {"operation": "source-fetch", "status": "failed", "elapsed_seconds": 165.01},
    ]
    shard06_rec = {
        "cleanup_error": None,
        "cli_calls": [
            {"attempt": 1, "command": "new", "operation": "control", "status": "completed"},
            {"attempt": 1, "command": "exec", "operation": "source-fetch", "status": "failed"},
            {"attempt": 1, "command": "stop", "operation": "control", "status": "completed"},
            {"attempt": 1, "command": "sessions", "operation": "control", "status": "completed"},
            {"attempt": 1, "command": "usage", "operation": "control", "status": "completed"},
        ],
        "commit": "1873b9025d5a563dcefaf7fcb3cda1e4e6e25306",
        "contents_transfers": [],
        "control_operations": control_ops,
        "elapsed_seconds": 209.32530329190195,
        "endpoint": "gpu-a100-s-ass1c0-1slz54hp3kb8k",
        "error": "Refused('deadline reached during telemetry recovery')",
        "exports": [],
        "failure_class": "RPC_EXEC_POLL_TIMEOUT",
        "job_id": "watch6-l1-shard-06-l1v1-main-01",
        "manifest_sha256": digest(shard06_dir / "manifest.json"),
        "status": "INCOMPLETE",
        "stop_verified": True,
        "terminal": None,
        "transfers": transfers,
    }
    (shard06_dir / "receipt.json").write_text(json.dumps(shard06_rec), encoding="utf-8")

    # (a) Exact §1.3 shape classifies as prestart (and not compute):
    assert admission.is_prestart_transport_failure(shard06_rec, shard06_dir) is True
    assert admission.is_external_transport_compute_failure(shard06_rec, shard06_dir) is False

    # (b.1) error differing by one character -> False:
    assert (
        admission.is_prestart_transport_failure(
            {**shard06_rec, "error": "Refused('deadline reached during telemetry recoverz')"},
            shard06_dir,
        )
        is False
    )

    # (b.2) failure_class TRANSPORT_TIMEOUT -> False:
    assert (
        admission.is_prestart_transport_failure(
            {**shard06_rec, "failure_class": "TRANSPORT_TIMEOUT"},
            shard06_dir,
        )
        is False
    )

    # (b.3) traceback set to any string (or last_traceback_line set) -> False:
    assert (
        admission.is_prestart_transport_failure(
            {**shard06_rec, "traceback": "Traceback (most recent call last):\nTimeoutError\n"},
            shard06_dir,
        )
        is False
    )
    assert (
        admission.is_prestart_transport_failure(
            {**shard06_rec, "last_traceback_line": "TimeoutError: deadline"},
            shard06_dir,
        )
        is False
    )

    # (b.4) any transfer with status completed instead of verified -> False:
    bad_transfers = [transfers[0], {**transfers[1], "status": "completed"}, transfers[2]]
    assert (
        admission.is_prestart_transport_failure(
            {**shard06_rec, "transfers": bad_transfers},
            shard06_dir,
        )
        is False
    )

    # (b.5) a second failed non-heartbeat operation (e.g. mkdir) -> False:
    bad_ops_second_fail = [
        *control_ops[:-1],
        {"operation": "mkdir", "status": "failed", "elapsed_seconds": 1.0},
        control_ops[-1],
    ]
    assert (
        admission.is_prestart_transport_failure(
            {**shard06_rec, "control_operations": bad_ops_second_fail},
            shard06_dir,
        )
        is False
    )

    # (b.6) no failed source-fetch -> False:
    bad_ops_no_fetch_fail = [
        *control_ops[:-1],
        {"operation": "source-fetch", "status": "completed", "elapsed_seconds": 10.0},
    ]
    assert (
        admission.is_prestart_transport_failure(
            {**shard06_rec, "control_operations": bad_ops_no_fetch_fail},
            shard06_dir,
        )
        is False
    )

    # (b.7) staging_completed true or absent -> False:
    upload_path.write_text(
        json.dumps({**upload_payload, "staging_completed": True}),
        encoding="utf-8",
    )
    assert admission.is_prestart_transport_failure(shard06_rec, shard06_dir) is False
    upload_path.write_text(
        json.dumps({k: v for k, v in upload_payload.items() if k != "staging_completed"}),
        encoding="utf-8",
    )
    assert admission.is_prestart_transport_failure(shard06_rec, shard06_dir) is False
    upload_path.write_text(json.dumps(upload_payload), encoding="utf-8")
    assert admission.is_prestart_transport_failure(shard06_rec, shard06_dir) is True

    # (b.8) source_delivery_seconds 0 (or bool) -> False:
    upload_path.write_text(
        json.dumps({**upload_payload, "source_delivery_seconds": 0}),
        encoding="utf-8",
    )
    assert admission.is_prestart_transport_failure(shard06_rec, shard06_dir) is False
    upload_path.write_text(
        json.dumps({**upload_payload, "source_delivery_seconds": True}),
        encoding="utf-8",
    )
    assert admission.is_prestart_transport_failure(shard06_rec, shard06_dir) is False
    upload_path.write_text(json.dumps(upload_payload), encoding="utf-8")
    assert admission.is_prestart_transport_failure(shard06_rec, shard06_dir) is True

    # (b.9-b.12) source-delivery.json, input-hashes.json, hardware.json, poll-0001.json present:
    for forbidden_file in (
        "source-delivery.json",
        "input-hashes.json",
        "hardware.json",
        "poll-0001.json",
    ):
        f_path = shard06_dir / forbidden_file
        f_path.write_text("{}\n", encoding="utf-8")
        assert admission.is_prestart_transport_failure(shard06_rec, shard06_dir) is False
        f_path.unlink()
    assert admission.is_prestart_transport_failure(shard06_rec, shard06_dir) is True

    # (b.13) terminal set to a dict -> False:
    assert (
        admission.is_prestart_transport_failure(
            {**shard06_rec, "terminal": {"returncode": 1, "reason": "error", "error": "boom"}},
            shard06_dir,
        )
        is False
    )

    # (b.14) exports non-empty -> False:
    assert (
        admission.is_prestart_transport_failure(
            {
                **shard06_rec,
                "exports": [{"path": "efficacy/world-36.json", "sha256": "0" * 64, "size": 10}],
            },
            shard06_dir,
        )
        is False
    )

    # (b.15) unknown top-level file (e.g. intent.json) -> False:
    intent_path = shard06_dir / "intent.json"
    intent_path.write_text("{}\n", encoding="utf-8")
    assert admission.is_prestart_transport_failure(shard06_rec, shard06_dir) is False
    intent_path.unlink()
    assert admission.is_prestart_transport_failure(shard06_rec, shard06_dir) is True

    # (b.16) symlinked upload-measurement.json -> False:
    ext_upload = tmp_path / "external-upload-measurement.json"
    ext_upload.write_text(json.dumps(upload_payload), encoding="utf-8")
    upload_path.unlink()
    upload_path.symlink_to(ext_upload)
    assert admission.is_prestart_transport_failure(shard06_rec, shard06_dir) is False
    upload_path.unlink()
    upload_path.write_text(json.dumps(upload_payload), encoding="utf-8")
    assert admission.is_prestart_transport_failure(shard06_rec, shard06_dir) is True

    # (b.17) non-empty exports/ -> False:
    (shard06_dir / "exports").mkdir()
    (shard06_dir / "exports/world-36.json").write_text("{}\n", encoding="utf-8")
    assert admission.is_prestart_transport_failure(shard06_rec, shard06_dir) is False
    (shard06_dir / "exports/world-36.json").unlink()
    (shard06_dir / "exports").rmdir()
    assert admission.is_prestart_transport_failure(shard06_rec, shard06_dir) is True

    # (b.18) endpoint null -> False:
    assert (
        admission.is_prestart_transport_failure(
            {**shard06_rec, "endpoint": None},
            shard06_dir,
        )
        is False
    )

    # (b.19) stop_verified false -> False:
    assert (
        admission.is_prestart_transport_failure(
            {**shard06_rec, "stop_verified": False},
            shard06_dir,
        )
        is False
    )

    # (b.20) cleanup_error any string -> False:
    assert (
        admission.is_prestart_transport_failure(
            {**shard06_rec, "cleanup_error": "STOP_FAILED: session still active"},
            shard06_dir,
        )
        is False
    )


REAL_L1V1_RECOVERY01_DIR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/l1v1-recovery-01"
)
REAL_MAIN01_SHARD06_RECEIPT_SHA256 = (
    "25d20ed7d414e631b156b8fd6ee6f223fe78dc6d88cde38e5b2be35f47ef6589"
)
REAL_RECOVERY01_SHARD01_RECEIPT_SHA256 = (
    "9378b4c0fa81ebc9c73cba2425bd60d2c8bda20eca13e19ed8ef228435b5cc65"
)


@pytest.mark.skipif(
    not (
        REAL_L1V1_MAIN01_DIR.is_dir()
        and (REAL_L1V1_MAIN01_DIR / "output-shard-06/receipt.json").is_file()
        and (REAL_L1V1_MAIN01_DIR / "output-shard-01/receipt.json").is_file()
        and (REAL_L1V1_MAIN01_DIR / "output-shard-03/receipt.json").is_file()
        and (REAL_L1V1_MAIN01_DIR / "output-shard-05/receipt.json").is_file()
        and (REAL_L1V1_MAIN01_DIR / "output-shard-00/receipt.json").is_file()
        and REAL_L1V1_RECOVERY01_DIR.is_dir()
        and (REAL_L1V1_RECOVERY01_DIR / "output-shard-01/receipt.json").is_file()
    ),
    reason="real l1v1-main-01 / l1v1-recovery-01 attempt directories not present on this machine",
)
def test_real_l1v1_source_delivery_poll_deadline_shards_validate_read_only() -> None:
    """Read-only check on real main-01 shard-06 and recovery-01 shard-01 under open guard."""
    with final_look.pre_outcome_open_guard():
        for shard_dir, expected_sha in (
            (REAL_L1V1_MAIN01_DIR / "output-shard-06", REAL_MAIN01_SHARD06_RECEIPT_SHA256),
            (
                REAL_L1V1_RECOVERY01_DIR / "output-shard-01",
                REAL_RECOVERY01_SHARD01_RECEIPT_SHA256,
            ),
        ):
            receipt_path = (shard_dir / "receipt.json").resolve()
            actual_sha = digest(receipt_path)
            assert actual_sha == expected_sha
            rec = json.loads(receipt_path.read_text(encoding="utf-8"))
            assert admission.is_prestart_transport_failure(rec, shard_dir) is True
            assert admission.is_external_transport_compute_failure(rec, shard_dir) is False
            assert (
                final_look._classify_attempt(
                    shard_dir,
                    rec,
                    receipt_path=receipt_path,
                    receipt_sha256=actual_sha,
                )
                == "prestart"
            )

        for shard_name in ("output-shard-01", "output-shard-03", "output-shard-05"):
            pre_dir = REAL_L1V1_MAIN01_DIR / shard_name
            pre_rec = json.loads((pre_dir / "receipt.json").read_text(encoding="utf-8"))
            assert admission.is_prestart_transport_failure(pre_rec, pre_dir) is True

        shard00_dir = REAL_L1V1_MAIN01_DIR / "output-shard-00"
        shard00_rec = json.loads((shard00_dir / "receipt.json").read_text(encoding="utf-8"))
        assert admission.is_prestart_transport_failure(shard00_rec, shard00_dir) is False


def test_verify_ops_proof_immutable_accepts_and_refuses(tmp_path: Path) -> None:
    """Verify R9 verify_ops_proof_immutable accepts bare and two-field sidecars and refuses (b)."""
    payload = tmp_path / "RETRY.json"
    payload.write_text('{"schema": "ops-zero-export-retry/v1"}\n', encoding="utf-8")
    sidecar = Path(f"{payload}.sha256")
    marker = Path(f"{payload}.complete")
    marker.write_bytes(b"complete\n")
    sha = digest(payload)

    # (a.1) Bare-digest sidecar (`sha + "\n"`) accepted:
    sidecar.write_text(f"{sha}\n", encoding="utf-8")
    res_bare = admission.verify_ops_proof_immutable(payload)
    assert res_bare == {"path": str(payload), "sha256": sha, "size": payload.stat().st_size}
    assert res_bare["sha256"] == sha

    # (a.2) Two-field sidecar (`sha + "  " + name + "\n"`) accepted:
    sidecar.write_text(f"{sha}  {payload.name}\n", encoding="utf-8")
    res_two = admission.verify_ops_proof_immutable(payload)
    assert res_two == {"path": str(payload), "sha256": sha, "size": payload.stat().st_size}
    assert res_two["sha256"] == sha

    # (b.1) Wrong digest (one hex char changed):
    bad_sha = ("0" if sha[0] != "0" else "1") + sha[1:]
    sidecar.write_text(f"{bad_sha}\n", encoding="utf-8")
    with pytest.raises(PackageRefused, match="destination-sidecar mismatch"):
        admission.verify_ops_proof_immutable(payload)
    sidecar.write_text(f"{bad_sha}  {payload.name}\n", encoding="utf-8")
    with pytest.raises(PackageRefused, match="destination-sidecar mismatch"):
        admission.verify_ops_proof_immutable(payload)

    # (b.2) Two-field with a different basename:
    sidecar.write_text(f"{sha}  manifest.json\n", encoding="utf-8")
    with pytest.raises(PackageRefused, match="destination-sidecar mismatch"):
        admission.verify_ops_proof_immutable(payload)

    # (b.3) Uppercase hex:
    sidecar.write_text(f"{sha.upper()}\n", encoding="utf-8")
    with pytest.raises(PackageRefused, match="destination-sidecar mismatch"):
        admission.verify_ops_proof_immutable(payload)
    sidecar.write_text(f"{sha.upper()}  {payload.name}\n", encoding="utf-8")
    with pytest.raises(PackageRefused, match="destination-sidecar mismatch"):
        admission.verify_ops_proof_immutable(payload)

    # (b.4) Three tokens:
    sidecar.write_text(f"{sha}  {payload.name}  extra\n", encoding="utf-8")
    with pytest.raises(PackageRefused, match="destination-sidecar mismatch"):
        admission.verify_ops_proof_immutable(payload)

    # (b.5) Empty sidecar:
    sidecar.write_text("", encoding="utf-8")
    with pytest.raises(PackageRefused, match="destination-sidecar mismatch"):
        admission.verify_ops_proof_immutable(payload)
    sidecar.write_text("   \n", encoding="utf-8")
    with pytest.raises(PackageRefused, match="destination-sidecar mismatch"):
        admission.verify_ops_proof_immutable(payload)

    # (b.6) Sidecar of 1025 bytes (digest followed by padding):
    padded_1025 = sha + (" " * (admission.OPS_PROOF_SIDECAR_MAX_BYTES - len(sha))) + "\n"
    sidecar.write_text(padded_1025, encoding="utf-8")
    assert sidecar.stat().st_size == 1025
    with pytest.raises(PackageRefused, match="oversized sidecar"):
        admission.verify_ops_proof_immutable(payload)

    # Restore valid sidecar before testing missing/marker/symlink/dir cases:
    sidecar.write_text(f"{sha}\n", encoding="utf-8")
    assert admission.verify_ops_proof_immutable(payload)["sha256"] == sha

    # (b.7) Missing .sha256:
    sidecar.unlink()
    with pytest.raises(PackageRefused, match="missing or unsafe immutable export"):
        admission.verify_ops_proof_immutable(payload)
    sidecar.write_text(f"{sha}\n", encoding="utf-8")

    # (b.8) Missing .complete:
    marker.unlink()
    with pytest.raises(PackageRefused, match="missing or unsafe immutable export"):
        admission.verify_ops_proof_immutable(payload)

    # (b.9) Marker `complete` without newline:
    marker.write_bytes(b"complete")
    with pytest.raises(PackageRefused, match="invalid completion marker"):
        admission.verify_ops_proof_immutable(payload)

    # (b.10) Marker `done\n`:
    marker.write_bytes(b"done\n")
    with pytest.raises(PackageRefused, match="invalid completion marker"):
        admission.verify_ops_proof_immutable(payload)
    marker.write_bytes(b"complete\n")
    assert admission.verify_ops_proof_immutable(payload)["sha256"] == sha

    # (b.11) Symlinked payload:
    ext_payload = tmp_path / "ext-RETRY.json"
    ext_payload.write_bytes(payload.read_bytes())
    payload.unlink()
    payload.symlink_to(ext_payload)
    with pytest.raises(PackageRefused, match="missing or unsafe immutable export"):
        admission.verify_ops_proof_immutable(payload)
    payload.unlink()
    payload.write_bytes(ext_payload.read_bytes())
    assert admission.verify_ops_proof_immutable(payload)["sha256"] == sha

    # (b.12) Symlinked sidecar:
    ext_sidecar = tmp_path / "ext-RETRY.json.sha256"
    ext_sidecar.write_text(f"{sha}\n", encoding="utf-8")
    sidecar.unlink()
    sidecar.symlink_to(ext_sidecar)
    with pytest.raises(PackageRefused, match="missing or unsafe immutable export"):
        admission.verify_ops_proof_immutable(payload)
    sidecar.unlink()
    sidecar.write_text(f"{sha}\n", encoding="utf-8")
    assert admission.verify_ops_proof_immutable(payload)["sha256"] == sha

    # (b.13) Symlinked marker:
    ext_marker = tmp_path / "ext-RETRY.json.complete"
    ext_marker.write_bytes(b"complete\n")
    marker.unlink()
    marker.symlink_to(ext_marker)
    with pytest.raises(PackageRefused, match="missing or unsafe immutable export"):
        admission.verify_ops_proof_immutable(payload)
    marker.unlink()
    marker.write_bytes(b"complete\n")
    assert admission.verify_ops_proof_immutable(payload)["sha256"] == sha

    # (b.14) Payload is a directory:
    dir_payload = tmp_path / "dir-payload"
    dir_payload.mkdir()
    Path(f"{dir_payload}.sha256").write_text(f"{sha}\n", encoding="utf-8")
    Path(f"{dir_payload}.complete").write_bytes(b"complete\n")
    with pytest.raises(PackageRefused, match="missing or unsafe immutable export"):
        admission.verify_ops_proof_immutable(dir_payload)


REAL_AUTO_RETRY_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/auto-retry")
REAL_L1V1_AUTO_RETRY_DIRS: tuple[str, ...] = (
    "watch6-l1v1-rec02-shard-11-retry-91caab5f03",
    "watch6-l1v1-shard-01-retry-98afc26d66",
    "watch6-l1v1-shard-03-retry-17a0a5b808",
    "watch6-l1v1-shard-05-retry-52e04ef43c",
    "watch6-l1v1-shard-06-retry-2b8fab8e6a",
    "watch6-l1v1-shard-11-retry-1cf3c2c133",
    "watch6-l1v1-shard-rec01-canary-a100-retry-b4daebaa31",
    "watch6-l1v1-shard-rec01-shard-01-retry-86cf1c8e5d",
)


@pytest.mark.skipif(
    not (
        REAL_AUTO_RETRY_ROOT.is_dir()
        and all(
            (REAL_AUTO_RETRY_ROOT / d / name).is_file()
            for d in REAL_L1V1_AUTO_RETRY_DIRS
            for name in ("RETRY.json", "manifest.json")
        )
    ),
    reason="real ops/auto-retry/watch6-l1v1-* directories not present on this machine",
)
def test_real_l1v1_ops_auto_retry_proofs_validate_read_only() -> None:
    """Read-only check on all 16 real watch6-l1v1 auto-retry proof files under open guard."""
    with final_look.pre_outcome_open_guard():
        checked = 0
        for dir_name in REAL_L1V1_AUTO_RETRY_DIRS:
            retry_dir = REAL_AUTO_RETRY_ROOT / dir_name
            for fname in ("RETRY.json", "manifest.json"):
                target = retry_dir / fname
                sidecar_tokens = Path(f"{target}.sha256").read_text(encoding="utf-8").split()
                assert len(sidecar_tokens) == 1
                desc = admission.verify_ops_proof_immutable(target)
                assert desc["sha256"] == sidecar_tokens[0]
                assert desc["sha256"] == digest(target)
                checked += 1
        assert checked == 16

        shard06_retry_json = (
            REAL_AUTO_RETRY_ROOT / "watch6-l1v1-shard-06-retry-2b8fab8e6a/RETRY.json"
        )
        with pytest.raises(PackageRefused, match="destination-sidecar mismatch"):
            verify_immutable(shard06_retry_json)
