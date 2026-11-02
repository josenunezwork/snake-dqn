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
from tools.ops.package_spec import Input, digest, empty_ledger, publish_json, stage_inputs


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
    assert "research/watch6_l1_confirm/prepare.py" not in sci
    assert "research/watch6_l1_confirm/final_look.py" not in sci


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
    s0_pre.mkdir()
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
        spec = p.shard_spec(shard)
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
