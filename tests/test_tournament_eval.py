"""Tests for the repaired promotion gate (src/scripts/tournament_eval.py).

Covers the pure helpers (agent-spec parsing, opponent-mix construction, the
§5.2 promotion rule) and a 2-seed miniature end-to-end run that uses scripted
stand-ins for hero/baseline/opponents, so no checkpoint is needed.
"""

import argparse
import json
from pathlib import Path

import pytest

from src.scripts.eval_stats import paired_stats
from src.scripts.tournament_eval import (
    MIX_NAMES,
    build_mix_specs,
    combined_paired_stats,
    main,
    parse_agent_spec,
    parse_mix_list,
    promotion_decision,
)


@pytest.fixture(autouse=True)
def _temporary_default_snapshot_root(monkeypatch, tmp_path):
    """Keep CLI-default evaluation artifacts out of the repository during tests."""
    monkeypatch.setenv("SNAKE_DQN_EVAL_SNAPSHOT_DIR", str(tmp_path / "snapshots"))


class TestBackwardCompat:
    def test_ci95_alias_survives_for_ensemble_eval(self):
        # ensemble_eval.py imports ci95 from tournament_eval; keep it exported.
        from src.scripts.eval_stats import ci95_halfwidth
        from src.scripts.tournament_eval import ci95

        assert ci95 is ci95_halfwidth


def test_main_uses_snapshots_for_every_checkpoint_agent(setup_config, tmp_path, monkeypatch):
    """A later replacement of a source path cannot affect a seeded eval row."""
    import torch

    from src.scripts import tournament_eval as module

    candidate = tmp_path / "candidate.pth"
    baseline = tmp_path / "baseline.pth"
    opponent = tmp_path / "opponent.pth"
    config = tmp_path / "tiny_eval.yaml"
    original_config = "game:\n  width: 400\n  height: 300\n  num_snakes: 3\n"
    config.write_text(original_config)
    torch.save({"dqn_state_dict": {}}, candidate)
    baseline.write_bytes(candidate.read_bytes())
    opponent.write_bytes(candidate.read_bytes())
    source_paths = {str(candidate.resolve()), str(baseline.resolve()), str(opponent.resolve())}
    observed_specs = []

    def fake_run_mix(hero_spec, mix_specs, frames, seeds, engine):
        observed_specs.append((hero_spec, list(mix_specs)))
        candidate.write_bytes(b"replaced after inputs were frozen")
        return [
            {
                "seed": seed,
                "mass_integral": 1.0,
                "max_mass": 1.0,
                "kills": 0.0,
                "deaths": 0.0,
                "survival_fraction": 1.0,
                "probes": {
                    "death_cause": None,
                    "boost_frame_fraction": 0.0,
                    "food_eaten": 0,
                    "kill_opportunity_count": 0,
                    "entrapment_event": False,
                    "peak_length": 1,
                },
            }
            for seed in seeds
        ]

    monkeypatch.setattr(module, "run_mix", fake_run_mix)
    original_load_config = module.load_and_initialize_config

    def load_then_replace(path):
        loaded = original_load_config(path)
        config.write_text("game:\n  width: 300\n  height: 300\n  num_snakes: 2\n")
        return loaded

    monkeypatch.setattr(module, "load_and_initialize_config", load_then_replace)
    output = tmp_path / "result.json"
    snapshots = tmp_path / "snapshots"
    assert (
        main(
            [
                str(candidate),
                "--baseline",
                str(baseline),
                "--opponents",
                str(opponent),
                "--mixes",
                "frozen,scripted",
                "--frames",
                "1",
                "--seeds",
                "0",
                "--config",
                str(config),
                "--json-output",
                str(output),
                "--snapshot-dir",
                str(snapshots),
            ]
        )
        == 0
    )

    data = json.loads(output.read_text())
    receipt = json.loads(Path(data["evaluation_inputs"]["receipt"]).read_text())
    frozen_paths = {
        reference
        for hero_spec, mix_specs in observed_specs
        for kind, reference in [hero_spec, *mix_specs]
        if kind == "checkpoint"
    }
    assert not frozen_paths & source_paths
    assert len({entry["sha256"] for entry in receipt["checkpoint_snapshots"]}) == 1
    assert all(
        Path(path).read_bytes() != b"replaced after inputs were frozen" for path in frozen_paths
    )
    assert (
        receipt["config"]["sha256"]
        == __import__("hashlib").sha256(original_config.encode()).hexdigest()
    )
    assert data["baseline"] == str(baseline)
    assert data["opponent_pool"] == [str(opponent)]


def test_snapshot_mutation_during_eval_fails_before_writing_a_verdict(
    setup_config, tmp_path, monkeypatch
):
    """A rewritten frozen input invalidates the entire evaluation, not one row."""
    import torch

    from src.scripts import tournament_eval as module

    candidate = tmp_path / "candidate.pth"
    config = tmp_path / "tiny_eval.yaml"
    config.write_text("game:\n  width: 400\n  height: 300\n  num_snakes: 3\n")
    torch.save({"dqn_state_dict": {}}, candidate)

    def corrupting_run_mix(hero_spec, mix_specs, frames, seeds, engine):
        if hero_spec[0] == "checkpoint":
            frozen = Path(hero_spec[1])
            frozen.chmod(0o644)
            frozen.write_bytes(b"corrupted after rollout started")
        return [
            {
                "seed": seed,
                "mass_integral": 1.0,
                "max_mass": 1.0,
                "kills": 0.0,
                "deaths": 0.0,
                "survival_fraction": 1.0,
                "probes": {
                    "death_cause": None,
                    "boost_frame_fraction": 0.0,
                    "food_eaten": 0,
                    "kill_opportunity_count": 0,
                    "entrapment_event": False,
                    "peak_length": 1,
                },
            }
            for seed in seeds
        ]

    monkeypatch.setattr(module, "run_mix", corrupting_run_mix)
    output = tmp_path / "result.json"
    assert (
        main(
            [
                str(candidate),
                "--baseline",
                "scripted:greedy_food",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted,mixed",
                "--frames",
                "1",
                "--seeds",
                "0",
                "--config",
                str(config),
                "--json-output",
                str(output),
                "--snapshot-dir",
                str(tmp_path / "snapshots"),
            ]
        )
        == 2
    )
    assert not output.exists()


def _runtime_rows(spec, seeds):
    return [
        {
            "seed": seed,
            "mass_integral": 1.0,
            "max_mass": 1.0,
            "mean_mass_alive": 1.0,
            "kills": 0.0,
            "deaths": 0.0,
            "survival_fraction": 1.0,
            "probes": {
                "death_cause": None,
                "boost_frame_fraction": 0.0,
                "food_eaten": 0,
                "kill_opportunity_count": 0,
                "entrapment_event": False,
                "peak_length": 1,
            },
            "world_runtime_spec": spec.descriptor(),
            "world_runtime_spec_digest": spec.digest,
        }
        for seed in seeds
    ]


@pytest.mark.parametrize(
    ("policy", "capacity"),
    [("source-exact", 400), ("fresh-reset-horizon-bound", 5002)],
)
def test_explicit_runtime_spec_is_reused_and_bound_to_diagnostic_json(
    setup_config, tmp_path, monkeypatch, policy, capacity
):
    """The public diagnostic flag binds one typed allocation identity end-to-end."""
    from src.scripts import tournament_eval as module

    config = tmp_path / "promotion.yaml"
    config.write_text(
        "game:\n"
        "  width: 400\n"
        "  height: 300\n"
        "  num_snakes: 3\n"
        "  mechanics_version: 2\n"
        "  frame_rate: 1\n"
    )
    seen_specs = []

    def fake_run_mix(
        hero_spec,
        mix_specs,
        frames,
        seeds,
        engine,
        profile=None,
        mix_id="unspecified",
        *,
        world_runtime_spec=None,
    ):
        assert engine == "simd"
        assert profile is not None
        assert frames == 5000
        assert world_runtime_spec is not None
        seen_specs.append(world_runtime_spec)
        return _runtime_rows(world_runtime_spec, seeds)

    monkeypatch.setattr(module, "run_mix", fake_run_mix)
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "scripted:greedy_food",
                "--baseline",
                "scripted:random_safe",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted",
                "--frames",
                "5000",
                "--seeds",
                "0,1",
                "--config",
                str(config),
                "--engine",
                "simd",
                "--evaluation-profile",
                "promotion-v2-watch-rect",
                "--simd-body-storage-policy",
                policy,
                "--json-output",
                str(output),
            ]
        )
        == 0
    )
    assert len(seen_specs) == 2
    assert all(spec is seen_specs[0] for spec in seen_specs)
    assert seen_specs[0].body_storage_capacity == capacity
    data = json.loads(output.read_text())
    assert data["strict_authority"] is False
    assert data["authority"] == "diagnostic-only"
    assert data["world_runtime_spec"] == {
        "descriptor": seen_specs[0].descriptor(),
        "digest": seen_specs[0].digest,
    }
    receipt = json.loads(Path(data["evaluation_inputs"]["receipt"]).read_text())
    assert receipt["world_runtime_spec"] == data["world_runtime_spec"]


def test_explicit_runtime_spec_reaches_every_pilot_mix_and_is_bound_to_output(
    setup_config, tmp_path, monkeypatch
):
    from src.scripts import tournament_eval as module

    config = tmp_path / "promotion.yaml"
    config.write_text(
        "game:\n"
        "  width: 400\n"
        "  height: 300\n"
        "  num_snakes: 3\n"
        "  mechanics_version: 2\n"
        "  frame_rate: 1\n"
    )
    seen_specs = []

    def fake_run_mix(*args, world_runtime_spec=None, **kwargs):
        assert world_runtime_spec is not None
        seen_specs.append(world_runtime_spec)
        return _runtime_rows(world_runtime_spec, [0, 1])

    monkeypatch.setattr(module, "run_mix", fake_run_mix)
    output = tmp_path / "pilot.json"
    assert (
        main(
            [
                "--pilot",
                "--baseline",
                "scripted:greedy_food",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted,mixed",
                "--frames",
                "5000",
                "--seeds",
                "0,1",
                "--config",
                str(config),
                "--engine",
                "simd",
                "--evaluation-profile",
                "promotion-v2-watch-rect",
                "--simd-body-storage-policy",
                "source-exact",
                "--json-output",
                str(output),
            ]
        )
        == 0
    )
    assert len(seen_specs) == 2
    assert all(spec is seen_specs[0] for spec in seen_specs)
    assert seen_specs[0].body_storage_capacity == 400
    data = json.loads(output.read_text())
    assert data["mode"] == "pilot"
    assert data["world_runtime_spec"] == {
        "descriptor": seen_specs[0].descriptor(),
        "digest": seen_specs[0].digest,
    }
    for mix in data["pilot"]["per_mix"].values():
        assert all(row["world_runtime_spec"] == seen_specs[0].descriptor() for row in mix["runs"])
        assert all(row["world_runtime_spec_digest"] == seen_specs[0].digest for row in mix["runs"])
    receipt = json.loads(Path(data["evaluation_inputs"]["receipt"]).read_text())
    assert receipt["world_runtime_spec"] == data["world_runtime_spec"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--engine", "live", "--evaluation-profile", "promotion-v2-watch-rect"],
        ["--engine", "simd", "--evaluation-profile", "legacy-diagnostic"],
    ],
)
def test_simd_body_storage_policy_rejects_invalid_combinations_before_snapshots(
    monkeypatch, arguments
):
    from src.scripts import tournament_eval as module

    called = []
    monkeypatch.setattr(
        module.EvaluationArtifacts,
        "snapshot_config",
        lambda *args, **kwargs: called.append((args, kwargs)),
    )
    with pytest.raises(SystemExit) as exc:
        main([*arguments, "--simd-body-storage-policy", "source-exact"])
    assert exc.value.code == 2
    assert not called


def test_runtime_row_identity_mismatch_refuses_result_before_paired_statistics(
    setup_config, tmp_path, monkeypatch
):
    from src.scripts import tournament_eval as module

    config = tmp_path / "promotion.yaml"
    config.write_text(
        "game:\n"
        "  width: 400\n"
        "  height: 300\n"
        "  num_snakes: 3\n"
        "  mechanics_version: 2\n"
        "  frame_rate: 1\n"
    )

    calls = 0

    def mismatched_candidate_run_mix(*args, world_runtime_spec=None, **kwargs):
        nonlocal calls
        calls += 1
        rows = _runtime_rows(world_runtime_spec, [0])
        if calls == 2:
            rows[0]["world_runtime_spec_digest"] = "0" * 64
        return rows

    monkeypatch.setattr(module, "run_mix", mismatched_candidate_run_mix)
    monkeypatch.setattr(
        module,
        "paired_stats",
        lambda *args, **kwargs: pytest.fail("runtime identity must be checked before statistics"),
    )
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "scripted:greedy_food",
                "--baseline",
                "scripted:random_safe",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted",
                "--frames",
                "5000",
                "--seeds",
                "0",
                "--config",
                str(config),
                "--engine",
                "simd",
                "--evaluation-profile",
                "promotion-v2-watch-rect",
                "--simd-body-storage-policy",
                "source-exact",
                "--json-output",
                str(output),
            ]
        )
        == 2
    )
    assert calls == 2
    assert not output.exists()


class TestParseAgentSpec:
    def test_checkpoint_path_passthrough(self):
        assert parse_agent_spec("saved_snakes/foo.pth") == ("checkpoint", "saved_snakes/foo.pth")

    def test_scripted_kinds(self):
        assert parse_agent_spec("scripted:greedy_food") == ("scripted", "greedy_food")
        assert parse_agent_spec("scripted:random_safe") == ("scripted", "random_safe")

    def test_unknown_scripted_kind_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_agent_spec("scripted:chess_engine")


class TestParseMixList:
    def test_valid_mixes(self):
        assert parse_mix_list("frozen,scripted") == ["frozen", "scripted"]

    def test_unknown_mix_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_mix_list("frozen,quantum")

    def test_duplicates_are_deduped_order_preserving(self):
        assert parse_mix_list("scripted,frozen,scripted,frozen") == ["scripted", "frozen"]


class TestBuildMixSpecs:
    POOL = [("checkpoint", "a.pth"), ("checkpoint", "b.pth")]

    def test_frozen_round_robins_the_pool(self):
        specs = build_mix_specs("frozen", 5, self.POOL)
        assert specs == [self.POOL[0], self.POOL[1], self.POOL[0], self.POOL[1], self.POOL[0]]

    def test_scripted_is_all_greedy_anchors(self):
        specs = build_mix_specs("scripted", 4, self.POOL)
        assert specs == [("scripted", "greedy_food")] * 4

    def test_mixed_alternates_pool_and_random_safe(self):
        specs = build_mix_specs("mixed", 5, self.POOL)
        assert specs == [
            self.POOL[0],
            ("scripted", "random_safe"),
            self.POOL[1],
            ("scripted", "random_safe"),
            self.POOL[0],
        ]

    def test_frozen_requires_a_pool(self):
        with pytest.raises(ValueError):
            build_mix_specs("frozen", 3, [])

    def test_unknown_mix_rejected(self):
        with pytest.raises(ValueError):
            build_mix_specs("quantum", 3, self.POOL)


def _paired(significant: bool, regression: bool = False) -> dict:
    return {"significant": significant, "regression": regression}


class TestPromotionDecision:
    def test_promotes_on_two_significant_mixes_and_anchor_ok(self):
        decision = promotion_decision(
            {"frozen": _paired(True), "scripted": _paired(True), "mixed": _paired(False)},
            MIX_NAMES,
        )
        assert decision["promote"] is True
        assert decision["significant_mixes"] == ["frozen", "scripted"]

    def test_rejects_with_one_significant_mix(self):
        decision = promotion_decision(
            {"frozen": _paired(True), "scripted": _paired(False), "mixed": _paired(False)},
            MIX_NAMES,
        )
        assert decision["promote"] is False
        assert any(">= 2 required" in r for r in decision["reasons"])


class TestCombinedPairedStats:
    """The descriptive cross-mix CI must retain worlds as the sample unit."""

    @staticmethod
    def _runs(seed_scores):
        return [{"seed": seed, "mass_integral": score} for seed, score in seed_scores]

    def test_averages_mix_deltas_per_world_before_ci(self):
        candidate = {
            "scripted": self._runs([(10, 3.0), (20, 7.0)]),
            "mixed": self._runs([(10, 7.0), (20, 13.0)]),
        }
        baseline = {
            "scripted": self._runs([(10, 1.0), (20, 3.0)]),
            "mixed": self._runs([(10, 1.0), (20, 3.0)]),
        }

        combined = combined_paired_stats(candidate, baseline, ["scripted", "mixed"])
        expected = paired_stats([4.0, 7.0], [0.0, 0.0])

        assert combined == expected
        assert combined["n"] == 2
        assert combined["deltas"] == [4.0, 7.0]

    def test_unequal_or_misaligned_seed_sets_fail(self):
        candidate = {
            "scripted": self._runs([(10, 3.0), (20, 7.0)]),
            "mixed": self._runs([(10, 4.0)]),
        }
        baseline = {
            "scripted": self._runs([(10, 1.0), (30, 3.0)]),
            "mixed": self._runs([(10, 1.0)]),
        }

        with pytest.raises(ValueError, match="candidate/baseline world seed sets differ"):
            combined_paired_stats(candidate, baseline, ["scripted", "mixed"])

        baseline["scripted"] = self._runs([(10, 1.0), (20, 3.0)])
        with pytest.raises(ValueError, match="world seed sets differ across mixes"):
            combined_paired_stats(candidate, baseline, ["scripted", "mixed"])

    def test_rejects_on_scripted_anchor_regression(self):
        decision = promotion_decision(
            {
                "frozen": _paired(True),
                "mixed": _paired(True),
                "scripted": _paired(False, regression=True),
            },
            MIX_NAMES,
        )
        assert decision["promote"] is False
        assert any("scripted anchor" in r for r in decision["reasons"])

    def test_rejects_when_anchor_mix_missing(self):
        decision = promotion_decision(
            {"frozen": _paired(True), "mixed": _paired(True)},
            ["frozen", "mixed"],
        )
        assert decision["promote"] is False
        assert any("anchor" in r for r in decision["reasons"])

    def test_rejects_on_single_mix(self):
        decision = promotion_decision({"scripted": _paired(True)}, ["scripted"])
        assert decision["promote"] is False

    def test_duplicate_mix_entries_count_as_one_mix(self):
        """A repeated mix cannot satisfy the >= 2 distinct-mixes clause."""
        decision = promotion_decision({"scripted": _paired(True)}, ["scripted", "scripted"])
        assert decision["promote"] is False
        assert decision["significant_mixes"] == ["scripted"]
        assert any(">= 2 required" in r for r in decision["reasons"])


class TestEndToEndMiniature:
    """2-seed, tiny-frames gate run with scripted stand-ins (no checkpoints)."""

    @pytest.fixture
    def tiny_config(self, tmp_path):
        cfg = tmp_path / "tiny_eval.yaml"
        cfg.write_text(
            "game:\n"
            "  width: 400\n"
            "  height: 300\n"
            "  num_snakes: 3\n"
            "  initial_food: 30\n"
            "  max_food: 30\n"
        )
        return cfg

    def test_gate_loop_runs_and_writes_json(self, setup_config, tiny_config, tmp_path):
        out = tmp_path / "gate.json"
        rc = main(
            [
                "scripted:random_safe",
                "--baseline",
                "scripted:greedy_food",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted,mixed",
                "--frames",
                "120",
                "--seeds",
                "0,1",
                "--config",
                str(tiny_config),
                "--json-output",
                str(out),
                "--snapshot-dir",
                str(tmp_path / "snapshots"),
            ]
        )
        assert rc == 0  # without --gate the exit code is 0 regardless of verdict

        data = json.loads(out.read_text())
        assert data["baseline"] == "scripted:greedy_food"
        assert data["mixes"] == ["scripted", "mixed"]

        (candidate,) = data["candidates"]
        assert "error" not in candidate
        assert set(candidate["per_mix"]) == {"scripted", "mixed"}
        for mix in ("scripted", "mixed"):
            runs = candidate["per_mix"][mix]["runs"]
            assert [r["seed"] for r in runs] == [0, 1]
            for run in runs:
                assert 0.0 <= run["survival_fraction"] <= 1.0
                assert run["mass_integral"] >= 0.0
                # Integral over the FULL horizon is never above the alive-mean.
                assert run["mass_integral"] <= run["mean_mass_alive"] + 1e-9
                assert "boost_frame_fraction" in run["probes"]
            paired = candidate["per_mix"][mix]["paired"]
            assert paired["n"] == 2
            assert len(paired["deltas"]) == 2
        combined = candidate["combined_paired"]
        assert combined["n"] == 2
        assert "promote" in candidate["decision"]

    def test_gate_exit_code_reflects_decision(self, setup_config, tiny_config, tmp_path):
        # random_safe vs itself as baseline: deltas ~0, never significant on
        # 2 seeds -> the gate must REJECT (exit 1).
        rc = main(
            [
                "scripted:random_safe",
                "--baseline",
                "scripted:random_safe",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted,mixed",
                "--frames",
                "60",
                "--seeds",
                "0,1",
                "--config",
                str(tiny_config),
                "--gate",
                "--snapshot-dir",
                str(tmp_path / "snapshots"),
            ]
        )
        assert rc == 1

    def test_pilot_mode_recommends_seed_count(self, setup_config, tiny_config, tmp_path):
        out = tmp_path / "pilot.json"
        rc = main(
            [
                "--pilot",
                "--baseline",
                "scripted:greedy_food",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted",
                "--frames",
                "60",
                "--seeds",
                "0,1",
                "--config",
                str(tiny_config),
                "--json-output",
                str(out),
                "--snapshot-dir",
                str(tmp_path / "snapshots"),
            ]
        )
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["mode"] == "pilot"
        assert data["strict_authority"] is False
        assert data["authority"] == "diagnostic-only"
        mix = data["pilot"]["per_mix"]["scripted"]
        assert mix["mass_integral_mean"] > 0.0
        assert mix["recommended_seeds"] is None or mix["recommended_seeds"] >= 2


def _save_eval_checkpoints(tmp_path):
    """Write minimal vector and raster31v2 inference checkpoints."""
    import torch

    from src.model.apex_network import ApexNetwork
    from src.model.obs_spec import OBS_SPEC_KEY, RASTER31V2, RASTER31V2_SHAPES
    from src.model.raster_network import RasterDuelingNetwork

    vector_path = tmp_path / "vector.pth"
    # The tiny live config keeps the legacy 58-D state builder. The evaluator
    # must preserve that vector path while routing only raster checkpoints to
    # RasterServingPolicy.
    vector = ApexNetwork(input_size=58, hidden_size=32, output_size=6)
    torch.save(
        {
            "dqn_state_dict": vector.state_dict(),
            "input_size": 58,
            "hidden_size": 32,
            "output_size": 6,
        },
        vector_path,
    )

    raster_path = tmp_path / "raster.pth"
    raster = RasterDuelingNetwork(output_size=6)
    torch.save(
        {
            "dqn_state_dict": raster.state_dict(),
            OBS_SPEC_KEY: RASTER31V2,
            "output_size": 6,
            **RASTER31V2_SHAPES.to_metadata(),
        },
        raster_path,
    )
    return vector_path, raster_path


class TestLiveRasterEval:
    """The live bridge supports raster only for the separately rolled-out hero."""

    @pytest.fixture
    def tiny_config(self, tmp_path):
        cfg = tmp_path / "tiny_live_raster.yaml"
        cfg.write_text(
            "game:\n"
            "  width: 400\n"
            "  height: 300\n"
            "  num_snakes: 3\n"
            "  initial_food: 12\n"
            "  max_food: 12\n"
        )
        return cfg

    def test_raster_candidate_and_vector_baseline_share_live_cli(
        self, setup_config, tiny_config, tmp_path
    ):
        vector, raster = _save_eval_checkpoints(tmp_path)
        out = tmp_path / "raster_candidate.json"

        assert (
            main(
                [
                    str(raster),
                    "--baseline",
                    str(vector),
                    "--opponents",
                    "scripted:random_safe",
                    "--mixes",
                    "scripted,mixed",
                    "--frames",
                    "1",
                    "--seeds",
                    "0",
                    "--config",
                    str(tiny_config),
                    "--json-output",
                    str(out),
                ]
            )
            == 0
        )
        data = json.loads(out.read_text())
        assert data["engine"] == "live"
        assert "error" not in data["candidates"][0]

    def test_raster_baseline_rolls_out_separately_from_vector_candidate(
        self, setup_config, tiny_config, tmp_path
    ):
        vector, raster = _save_eval_checkpoints(tmp_path)
        out = tmp_path / "raster_baseline.json"

        assert (
            main(
                [
                    str(vector),
                    "--baseline",
                    str(raster),
                    "--opponents",
                    "scripted:random_safe",
                    "--mixes",
                    "scripted,mixed",
                    "--frames",
                    "1",
                    "--seeds",
                    "0",
                    "--config",
                    str(tiny_config),
                    "--json-output",
                    str(out),
                ]
            )
            == 0
        )
        assert "error" not in json.loads(out.read_text())["candidates"][0]

    def test_live_raster_opponent_fails_before_evaluation(
        self, setup_config, tiny_config, tmp_path
    ):
        vector, raster = _save_eval_checkpoints(tmp_path)

        with pytest.raises(SystemExit) as exc:
            main(
                [
                    str(vector),
                    "--baseline",
                    str(vector),
                    "--opponents",
                    str(raster),
                    "--mixes",
                    "frozen,scripted",
                    "--frames",
                    "1",
                    "--seeds",
                    "0",
                    "--config",
                    str(tiny_config),
                ]
            )
        assert exc.value.code == 2

    def test_unused_raster_pool_entry_is_allowed_for_scripted_only_mix(
        self, setup_config, tiny_config, tmp_path
    ):
        vector, raster = _save_eval_checkpoints(tmp_path)

        # The raw pool includes raster, but the requested scripted mix expands
        # only to scripted opponents, so no raster policy is ever installed in
        # a non-hero slot.
        assert (
            main(
                [
                    str(vector),
                    "--baseline",
                    str(vector),
                    "--opponents",
                    str(raster),
                    "--mixes",
                    "scripted",
                    "--frames",
                    "1",
                    "--seeds",
                    "0",
                    "--config",
                    str(tiny_config),
                ]
            )
            == 0
        )

    def test_raster_hero_keeps_identity_and_masks_unsafe_q_across_frames(
        self, setup_config, monkeypatch, tmp_path
    ):
        """Hero consumes its own Q row and never executes an unsafe Q argmax."""
        import torch

        from src.game.game_state_factory import (
            configure_eval_game_state,
            create_training_game_state,
        )
        from src.scripts.tournament_eval import _attach_agent

        _, raster = _save_eval_checkpoints(tmp_path)
        gs = create_training_game_state(eval_mode=False)
        try:
            cache = {}
            _attach_agent(gs, 0, ("checkpoint", str(raster)), seed=0, policy_cache=cache)
            for slot in range(1, len(gs.snakes)):
                _attach_agent(gs, slot, ("scripted", "random_safe"), seed=0, policy_cache=cache)
            gs._shared_policy = None
            configure_eval_game_state(gs)

            hero = gs.snakes[0]
            policy = hero.policy
            assert hero.ai is policy

            # The serving bridge sees a batch row per roster snake. Give row 0
            # an unsafe boost-right argmax (5) and a safe straight fallback (1);
            # other rows intentionally carry different Q values. This exposes a
            # queue shift as a returned non-hero row rather than merely checking
            # that a raster forward pass occurred.
            def fixed_rows(tensors):
                rows = torch.zeros((tensors["tactical"].shape[0], 6), dtype=torch.float32)
                rows[0] = torch.tensor([0.0, 10.0, 0.0, 0.0, 0.0, 100.0])
                if rows.shape[0] > 1:
                    rows[1:] = torch.tensor([80.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                return rows

            policy.agent.network = fixed_rows
            returned_rows = []
            original_next_q = policy._next_dispatch_q

            def record_next_q(state):
                q = original_next_q(state)
                returned_rows.append(q.detach().cpu().squeeze(0).tolist())
                return q

            policy._next_dispatch_q = record_next_q
            monkeypatch.setattr(hero, "_get_safe_actions", lambda other_snakes, **kwargs: [1])
            initial_direction = hero.direction

            gs.update(train_mode=True, learn=False, allow_respawn=True)
            gs.update(train_mode=True, learn=False, allow_respawn=True)

            assert returned_rows == [[0.0, 10.0, 0.0, 0.0, 0.0, 100.0]] * 2
            assert hero.direction == initial_direction  # action 1 is straight, not unsafe action 5.
            assert hero.is_boosting is False
        finally:
            gs.full_cleanup()


# ---------------------------------------------------------------------------
# --engine simd: batched eval engine (blueprint §5.4)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_v2_config(tmp_path):
    """Tiny mechanics-v2 arena the BatchSim eval engine can run."""
    cfg = tmp_path / "tiny_v2_eval.yaml"
    cfg.write_text(
        "game:\n"
        "  width: 400\n"
        "  height: 300\n"
        "  num_snakes: 3\n"
        "  initial_food: 30\n"
        "  max_food: 30\n"
        "  mechanics_version: 2\n"
    )
    return cfg


class TestSimdEvalEngine:
    """Unit tests for src.simd_env.eval_engine (used by --engine simd)."""

    def test_metrics_have_live_engine_keys_and_valid_ranges(self, setup_config, tiny_v2_config):
        from src.core.config_loader import apply_config_to_game_config, load_config
        from src.core.game_config import initialize_config
        from src.simd_env.eval_engine import run_simd_eval

        cfg = load_config(str(tiny_v2_config))
        initialize_config(cfg)
        apply_config_to_game_config(cfg)

        seeds = [0, 1, 2, 3]
        recs = run_simd_eval(
            ("scripted", "greedy_food"),
            [("scripted", "random_safe"), ("scripted", "random_safe")],
            frames=150,
            seeds=seeds,
            gamma=0.99,
        )
        assert [r["seed"] for r in recs] == seeds  # seed order == env order
        for r in recs:
            # Same headline keys the live rollout emits.
            for key in ("mass_integral", "max_mass", "kills", "deaths", "survival_fraction"):
                assert key in r
            assert 0.0 <= r["survival_fraction"] <= 1.0
            assert r["mass_integral"] >= 0.0
            # Integral over the full horizon never exceeds the alive-mean.
            assert r["mass_integral"] <= r["mean_mass_alive"] + 1e-9
            probes = r["probes"]
            assert 0.0 <= probes["boost_frame_fraction"] <= 1.0
            assert probes["food_eaten"] >= 0
            assert probes["peak_length"] >= 1

    def test_raster_checkpoint_runs_through_simd_engine(
        self, setup_config, tiny_v2_config, tmp_path
    ):
        # A raster (raster31v2) checkpoint builds a NetworkSimdPolicy and runs
        # through the fast gate — the P4 promotion path. Hermetic: an untrained
        # tiny raster net, saved with obs_spec, gated vs a scripted anchor.
        import torch

        from src.core.config_loader import apply_config_to_game_config, load_config
        from src.core.game_config import initialize_config
        from src.model.obs_spec import OBS_SPEC_KEY, RASTER31V2
        from src.model.raster_network import RasterDuelingNetwork
        from src.simd_env.eval_engine import (
            NetworkSimdPolicy,
            build_simd_policy,
            run_simd_eval,
        )

        cfg = load_config(str(tiny_v2_config))
        initialize_config(cfg)
        apply_config_to_game_config(cfg)

        net = RasterDuelingNetwork()
        ckpt = tmp_path / "raster.pth"
        torch.save({"dqn_state_dict": net.state_dict(), OBS_SPEC_KEY: RASTER31V2}, ckpt)

        assert isinstance(build_simd_policy(("checkpoint", str(ckpt)), seed=0), NetworkSimdPolicy)

        recs = run_simd_eval(
            ("checkpoint", str(ckpt)),
            [("scripted", "random_safe")],
            frames=60,
            seeds=[0, 1],
            gamma=0.997,
        )
        assert [r["seed"] for r in recs] == [0, 1]
        for r in recs:
            assert r["mass_integral"] >= 0.0
            assert 0.0 <= r["survival_fraction"] <= 1.0

    def test_greedy_food_hero_grows_by_eating(self, setup_config, tiny_v2_config):
        # The greedy-food hero should eat and survive on a food-rich tiny arena.
        from src.core.config_loader import apply_config_to_game_config, load_config
        from src.core.game_config import initialize_config
        from src.simd_env.eval_engine import run_simd_eval

        cfg = load_config(str(tiny_v2_config))
        initialize_config(cfg)
        apply_config_to_game_config(cfg)

        recs = run_simd_eval(
            ("scripted", "greedy_food"),
            [("scripted", "random_safe"), ("scripted", "random_safe")],
            frames=120,
            seeds=[0, 1],
        )
        assert all(r["probes"]["food_eaten"] > 0 for r in recs)
        assert all(r["survival_fraction"] == 1.0 for r in recs)

    def test_boost_frame_fraction_detects_a_boosting_hero(
        self, setup_config, tiny_v2_config, monkeypatch
    ):
        """A boost-emitting hero yields boost_frame_fraction > 0.

        The scripted anchors never boost (they emit actions 0-2), so the probe
        is coincidentally 0.0 for them and could not detect a broken boost
        signal. This injects a hero that grows toward food, then engages boost
        whenever the boost-straight mask bit is safe, and asserts the probe
        reports the boosting — the promotion gate's boost-abuse detector. This
        would report 0.0 if boosting were read from a field reset by step().
        """
        import numpy as np

        from src.core.config_loader import apply_config_to_game_config, load_config
        from src.core.game_config import initialize_config
        from src.simd_env import eval_engine
        from src.simd_env.eval_engine import (
            GreedyFoodSimdPolicy,
            SimdPolicy,
            run_simd_eval,
        )

        cfg = load_config(str(tiny_v2_config))
        initialize_config(cfg)
        apply_config_to_game_config(cfg)

        class BoostWhenAblePolicy(SimdPolicy):
            """Boost straight when that bit is safe; else greedily seek food."""

            def __init__(self, seed):
                self._greedy = GreedyFoodSimdPolicy()
                self._seed = seed

            def actions(self, masks, sim, slots):
                out = self._greedy.actions(masks, sim, slots)
                # Prefer boost-straight (action 4) whenever it is a safe move.
                boost_straight_safe = masks[:, 4]
                out = np.where(boost_straight_safe, 4, out)
                return out.astype(np.int64)

        real_build = eval_engine.build_simd_policy
        hero_spec = ("scripted", "boost_when_able")

        def fake_build(spec, seed):
            if spec == hero_spec:
                return BoostWhenAblePolicy(seed)
            return real_build(spec, seed)

        monkeypatch.setattr(eval_engine, "build_simd_policy", fake_build)

        recs = run_simd_eval(
            hero_spec,
            [("scripted", "random_safe"), ("scripted", "random_safe")],
            frames=200,
            seeds=[0, 1, 2, 3],
        )
        # At least one seed's hero must have grown to boost-eligible length and
        # actually boosted, so the probe registers it.
        assert any(r["probes"]["boost_frame_fraction"] > 0.0 for r in recs)
        assert all(0.0 <= r["probes"]["boost_frame_fraction"] <= 1.0 for r in recs)

    @pytest.mark.parametrize("input_size", [58, 61])
    def test_vector_checkpoint_agent_rejected(self, tmp_path, input_size):
        # Keep this routing guard hermetic: private champion files are absent
        # in clean worktrees, and both historical vector widths must reject.
        import torch

        from src.model.apex_network import ApexNetwork
        from src.simd_env.eval_engine import build_simd_policy

        ckpt = tmp_path / "vector.pth"
        network = ApexNetwork(input_size=input_size, hidden_size=32, output_size=6)
        torch.save(
            {
                "dqn_state_dict": network.state_dict(),
                "input_size": input_size,
                "hidden_size": 32,
                "output_size": 6,
            },
            ckpt,
        )
        with pytest.raises(ValueError, match="needs a raster model") as error:
            build_simd_policy(("checkpoint", str(ckpt)), seed=0)
        assert "Use --engine live" in str(error.value)

    def test_unknown_scripted_kind_raises(self):
        from src.simd_env.eval_engine import build_simd_policy

        with pytest.raises(ValueError):
            build_simd_policy(("scripted", "chess_engine"), seed=0)

    def test_empty_seeds_and_bad_frames_raise(self):
        from src.simd_env.eval_engine import run_simd_eval

        with pytest.raises(ValueError):
            run_simd_eval(("scripted", "random_safe"), [("scripted", "random_safe")], 100, [])
        with pytest.raises(ValueError):
            run_simd_eval(("scripted", "random_safe"), [("scripted", "random_safe")], 0, [0])


class TestSimdEngineEndToEnd:
    """main(--engine simd) drives the gate through the batched engine."""

    def test_gate_calibration_greedy_beats_random(
        self, setup_config, tiny_v2_config, tmp_path, capsys
    ):
        # The calibration ordering the blueprint requires (greedy_food anchor
        # beats random_safe with paired significance) must hold via --engine simd.
        out = tmp_path / "simd_gate.json"
        rc = main(
            [
                "scripted:greedy_food",
                "--baseline",
                "scripted:random_safe",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted,mixed",
                "--frames",
                "150",
                # A terminal hero makes the mass integral heavy-tailed (a seed
                # where the hero dies early scores ~0), so the t-interval needs
                # more than 4 paired seeds to resolve even this large an effect.
                "--seeds",
                "0,1,2,3,4,5,6,7",
                "--config",
                str(tiny_v2_config),
                "--engine",
                "simd",
                "--json-output",
                str(out),
            ]
        )
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["engine"] == "simd"
        assert data["mode"] == "eval"
        assert data["strict_authority"] is False
        assert data["authority"] == "diagnostic-only"
        assert "Legacy diagnostic paired results" in capsys.readouterr().out
        (candidate,) = data["candidates"]
        assert "error" not in candidate
        assert set(candidate["per_mix"]) == {"scripted", "mixed"}
        for mix in ("scripted", "mixed"):
            runs = candidate["per_mix"][mix]["runs"]
            assert [r["seed"] for r in runs] == [0, 1, 2, 3, 4, 5, 6, 7]
        # greedy_food should decisively beat random_safe -> promote.
        assert candidate["decision"]["promote"] is True

    def test_checkpoint_baseline_rejected_up_front(self, setup_config, tiny_v2_config):
        # A checkpoint baseline under --engine simd fails fast (argparse error).
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "scripted:greedy_food",
                    "--baseline",
                    "saved_snakes/champion_a5_freespace_20260621.pth",
                    "--opponents",
                    "scripted:random_safe",
                    "--mixes",
                    "scripted,mixed",
                    "--frames",
                    "40",
                    "--seeds",
                    "0,1",
                    "--config",
                    str(tiny_v2_config),
                    "--engine",
                    "simd",
                ]
            )
        assert exc.value.code == 2  # argparse p.error exit code

    def test_checkpoint_candidate_recorded_as_error_not_crash(
        self, setup_config, tiny_v2_config, tmp_path
    ):
        # A checkpoint CANDIDATE (not baseline/opponent) is recorded as a
        # per-candidate error under --engine simd; the run still completes.
        vector, _ = _save_eval_checkpoints(tmp_path)
        out = tmp_path / "simd_ckpt_cand.json"
        rc = main(
            [
                str(vector),
                "--baseline",
                "scripted:random_safe",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted,mixed",
                "--frames",
                "40",
                "--seeds",
                "0,1",
                "--config",
                str(tiny_v2_config),
                "--engine",
                "simd",
                "--json-output",
                str(out),
            ]
        )
        assert rc == 0
        data = json.loads(out.read_text())
        (candidate,) = data["candidates"]
        assert "error" in candidate
        assert "raster" in candidate["error"]
        assert candidate["decision"]["promote"] is False

    def test_live_engine_default_unchanged(self, setup_config, tiny_v2_config, tmp_path):
        # --engine live (default) still works and records engine=live.
        out = tmp_path / "live_gate.json"
        rc = main(
            [
                "scripted:random_safe",
                "--baseline",
                "scripted:greedy_food",
                "--opponents",
                "scripted:random_safe",
                "--mixes",
                "scripted,mixed",
                "--frames",
                "60",
                "--seeds",
                "0,1",
                "--config",
                str(tiny_v2_config),
                "--json-output",
                str(out),
            ]
        )
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["engine"] == "live"


def test_strict_runtime_uses_frozen_rows_and_real_artifact_validators(monkeypatch, tmp_path):
    """Strict execution consumes frozen evidence and persists exact E1 rows."""
    from src.scripts import tournament_eval as module
    from tests.test_strict_artifacts import ArtifactFixture

    fixture = ArtifactFixture(tmp_path)
    calls = []

    def fake_rollout(hero, opponents, frames, seed, *, profile, mix_id, **kwargs):
        index = len(calls) % len(fixture.rosters)
        row = fixture.rosters[index]
        calls.append((hero, tuple(opponents), frames, seed, profile.digest, mix_id))
        assert row["mix"] == mix_id
        assert row["world_seed"] == seed
        return fixture._record(row, 11.0 if len(calls) <= len(fixture.rosters) else 12.0)

    monkeypatch.setattr(module, "rollout", fake_rollout)
    final = tmp_path / "strict-final.json"
    rc = module.main(
        [
            "--strict-promotion-request",
            str(fixture.request_path),
            "--strict-promotion-receipt",
            str(final),
            "--strict-e0-receipt",
            str(fixture.e0_path),
            "--strict-pilot-artifact",
            str(fixture.pilot_path),
            "--strict-calibration-artifact",
            str(fixture.calibration_path),
            "--strict-serving-bundle",
            str(fixture.serving_path),
        ]
    )
    assert rc == 0
    assert len(calls) == 2 * len(fixture.rosters)
    e0 = json.loads(fixture.e0_path.read_text())
    snapshots = {item["sha256"]: item["snapshot_path"] for item in e0["checkpoint_snapshots"]}
    for index, row in enumerate(fixture.rosters):
        incumbent, candidate = calls[index], calls[index + len(fixture.rosters)]
        assert incumbent[0] == ("checkpoint", snapshots[fixture.incumbent["sha256"]])
        assert candidate[0] == ("checkpoint", snapshots[fixture.candidate["sha256"]])
        assert incumbent[1] == candidate[1]
        from src.evaluation.strict_promotion import scripted_agent

        anchors = {
            scripted_agent("greedy_food")["sha256"]: ("scripted", "greedy_food"),
            scripted_agent("random_safe")["sha256"]: ("scripted", "random_safe"),
        }
        expected_opponents = tuple(
            (
                ("checkpoint", snapshots[slot["member_sha256"]])
                if slot["member_sha256"] in snapshots
                else anchors[slot["member_sha256"]]
            )
            for slot in row["slots"]
        )
        assert incumbent[1] == expected_opponents
        assert candidate[1] == expected_opponents
    raw = json.loads(final.with_suffix(".raw-worlds.json").read_text())
    assert len(raw["records"]) == 2 * len(fixture.rosters)
    assert raw["records"][0]["record"] == fixture._record(fixture.rosters[0], 11.0)
    assert raw["records"][len(fixture.rosters)]["record"] == fixture._record(
        fixture.rosters[0], 12.0
    )
    assert json.loads(final.read_text())["strict_authority"] is True


def test_strict_runtime_freeze_failure_executes_zero_worlds(monkeypatch, tmp_path):
    from src.scripts import tournament_eval as module
    from tests.test_strict_artifacts import ArtifactFixture

    fixture = ArtifactFixture(tmp_path)
    fixture.request["engine"] = "simd"
    fixture.request_path.write_text(json.dumps(fixture.request))
    called = []
    monkeypatch.setattr(module, "rollout", lambda *args, **kwargs: called.append(args))
    with pytest.raises(SystemExit):
        module.main(
            [
                "--strict-promotion-request",
                str(fixture.request_path),
                "--strict-promotion-receipt",
                str(tmp_path / "final.json"),
                "--strict-e0-receipt",
                str(fixture.e0_path),
                "--strict-pilot-artifact",
                str(fixture.pilot_path),
                "--strict-calibration-artifact",
                str(fixture.calibration_path),
                "--strict-serving-bundle",
                str(fixture.serving_path),
            ]
        )
    assert not called


def test_strict_runtime_failure_archives_completed_records_without_authority(monkeypatch, tmp_path):
    from src.scripts import tournament_eval as module
    from tests.test_strict_artifacts import ArtifactFixture

    fixture = ArtifactFixture(tmp_path)
    calls = []

    def failing_rollout(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 2:
            raise RuntimeError("controlled failure")
        return fixture._record(fixture.rosters[0], 11.0)

    monkeypatch.setattr(module, "rollout", failing_rollout)
    final = tmp_path / "strict-final.json"
    argv = [
        "--strict-promotion-request",
        str(fixture.request_path),
        "--strict-promotion-receipt",
        str(final),
        "--strict-e0-receipt",
        str(fixture.e0_path),
        "--strict-pilot-artifact",
        str(fixture.pilot_path),
        "--strict-calibration-artifact",
        str(fixture.calibration_path),
        "--strict-serving-bundle",
        str(fixture.serving_path),
    ]
    with pytest.raises(SystemExit):
        module.main(argv)
    incident = json.loads(final.with_suffix(".incident.json").read_text())
    assert len(incident["completed_records"]) == 1
    assert incident["failing_compound_key"] == {
        "role": "incumbent",
        "mix": "frozen",
        "world_seed": fixture.rosters[1]["world_seed"],
        "checkpoint_sha256": fixture.incumbent["sha256"],
    }
    assert incident["error"] == "controlled failure"
    assert incident["strict_authority"] is False
    assert not final.exists() and not final.with_suffix(".raw-worlds.json").exists()


@pytest.mark.parametrize("existing_suffix", ("", ".raw-worlds.json", ".incident.json"))
def test_strict_runtime_refuses_existing_outputs_before_rollout(
    monkeypatch, tmp_path, existing_suffix
):
    from src.scripts import tournament_eval as module
    from tests.test_strict_artifacts import ArtifactFixture

    fixture = ArtifactFixture(tmp_path)
    final = tmp_path / "strict-final.json"
    existing = final if not existing_suffix else final.with_suffix(existing_suffix)
    existing.write_text("prior bytes")
    called = []
    monkeypatch.setattr(module, "rollout", lambda *args, **kwargs: called.append(args))
    argv = [
        "--strict-promotion-request",
        str(fixture.request_path),
        "--strict-promotion-receipt",
        str(final),
        "--strict-e0-receipt",
        str(fixture.e0_path),
        "--strict-pilot-artifact",
        str(fixture.pilot_path),
        "--strict-calibration-artifact",
        str(fixture.calibration_path),
        "--strict-serving-bundle",
        str(fixture.serving_path),
    ]
    with pytest.raises(SystemExit):
        module.main(argv)
    assert existing.read_text() == "prior bytes"
    assert not called


def test_strict_runtime_rejects_broken_symlink_output_before_rollout(monkeypatch, tmp_path):
    from src.scripts import tournament_eval as module
    from tests.test_strict_artifacts import ArtifactFixture

    fixture = ArtifactFixture(tmp_path)
    final = tmp_path / "strict-final.json"
    raw = final.with_suffix(".raw-worlds.json")
    raw.symlink_to(tmp_path / "missing-raw.json")
    called = []
    monkeypatch.setattr(module, "rollout", lambda *args, **kwargs: called.append(args))
    with pytest.raises(SystemExit):
        module.main(
            [
                "--strict-promotion-request",
                str(fixture.request_path),
                "--strict-promotion-receipt",
                str(final),
                "--strict-e0-receipt",
                str(fixture.e0_path),
                "--strict-pilot-artifact",
                str(fixture.pilot_path),
                "--strict-calibration-artifact",
                str(fixture.calibration_path),
                "--strict-serving-bundle",
                str(fixture.serving_path),
            ]
        )
    assert raw.is_symlink()
    assert not called


def test_strict_runtime_rejects_e0_swap_after_freeze_before_worlds(monkeypatch, tmp_path):
    """A frozen request cannot be redirected by replacing E0 after freeze."""
    from src.evaluation import strict_promotion
    from src.scripts import tournament_eval as module
    from tests.test_strict_artifacts import ArtifactFixture

    fixture = ArtifactFixture(tmp_path)
    original_freeze = strict_promotion.freeze_strict_request

    def freeze_then_swap(*args, **kwargs):
        token = original_freeze(*args, **kwargs)
        fixture.e0_path.write_text('{"swapped": true}', encoding="utf-8")
        return token

    monkeypatch.setattr(strict_promotion, "freeze_strict_request", freeze_then_swap)
    called = []
    monkeypatch.setattr(module, "rollout", lambda *args, **kwargs: called.append(args))
    final = tmp_path / "strict-final.json"
    with pytest.raises(SystemExit):
        module.main(
            [
                "--strict-promotion-request",
                str(fixture.request_path),
                "--strict-promotion-receipt",
                str(final),
                "--strict-e0-receipt",
                str(fixture.e0_path),
                "--strict-pilot-artifact",
                str(fixture.pilot_path),
                "--strict-calibration-artifact",
                str(fixture.calibration_path),
                "--strict-serving-bundle",
                str(fixture.serving_path),
            ]
        )
    assert not called
    assert not final.exists()
    assert not final.with_suffix(".raw-worlds.json").exists()


def test_strict_runtime_rejects_snapshot_swap_after_worlds(monkeypatch, tmp_path):
    """Post-world revalidation prevents a swapped snapshot from reaching raw evidence."""
    from src.scripts import tournament_eval as module
    from tests.test_strict_artifacts import ArtifactFixture

    fixture = ArtifactFixture(tmp_path)
    calls = []

    def mutate_after_first_world(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            fixture.candidate_file.write_bytes(b"candidate swapped after freeze")
        return fixture._record(fixture.rosters[0], 11.0)

    monkeypatch.setattr(module, "rollout", mutate_after_first_world)
    final = tmp_path / "strict-final.json"
    with pytest.raises(SystemExit):
        module.main(
            [
                "--strict-promotion-request",
                str(fixture.request_path),
                "--strict-promotion-receipt",
                str(final),
                "--strict-e0-receipt",
                str(fixture.e0_path),
                "--strict-pilot-artifact",
                str(fixture.pilot_path),
                "--strict-calibration-artifact",
                str(fixture.calibration_path),
                "--strict-serving-bundle",
                str(fixture.serving_path),
            ]
        )
    incident = json.loads(final.with_suffix(".incident.json").read_text())
    assert incident["strict_authority"] is False
    assert incident["phase"] == "post-world-integrity"
    assert incident["failing_compound_key"] is None
    assert "strict frozen input changed" in incident["error"]
    assert len(incident["completed_records"]) == 2 * len(fixture.rosters)
    assert not final.exists() and not final.with_suffix(".raw-worlds.json").exists()


def test_strict_runtime_rejects_diagnostic_override_before_worlds(monkeypatch, tmp_path):
    from src.evaluation import strict_promotion
    from src.scripts import tournament_eval as module
    from tests.test_strict_artifacts import ArtifactFixture

    fixture = ArtifactFixture(tmp_path)
    called = []
    frozen = []
    monkeypatch.setattr(module, "rollout", lambda *args, **kwargs: called.append(args))
    monkeypatch.setattr(
        strict_promotion,
        "freeze_strict_request",
        lambda *args, **kwargs: frozen.append((args, kwargs)),
    )
    with pytest.raises(SystemExit):
        module.main(
            [
                "--strict-promotion-request",
                str(fixture.request_path),
                "--strict-promotion-receipt",
                str(tmp_path / "strict-final.json"),
                "--strict-e0-receipt",
                str(fixture.e0_path),
                "--strict-pilot-artifact",
                str(fixture.pilot_path),
                "--strict-calibration-artifact",
                str(fixture.calibration_path),
                "--strict-serving-bundle",
                str(fixture.serving_path),
                "--simd-body-storage-pol",
                "source-exact",
            ]
        )
    assert not called
    assert not frozen
