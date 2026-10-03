"""Producer of the v8 web-serving run (research/apex_veto_v8_serving_20261003/serving_run.py).

Pins, seeds, env handling and guards; tiny-horizon episodes on the real champion. Every
test that calls ``main`` first makes every episode runner fatal, so no test can play a
served episode through the harness entry point.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from research.apex_veto_serving_20261001 import serving_run as v2run  # noqa: E402
from research.apex_veto_v7_serving_20261002 import serving_run as v7run  # noqa: E402
from research.apex_veto_v8_serving_20261003 import serving_audit as audit_mod  # noqa: E402
from research.apex_veto_v8_serving_20261003 import serving_run as run_mod  # noqa: E402
from src.evaluation.safety_veto_v7 import SpacePreferenceCounters  # noqa: E402
from src.evaluation.safety_veto_v8 import SpaceAndHeadCounters, SpaceAndHeadVeto  # noqa: E402
from web.backend import safety_veto_serving as serving  # noqa: E402

HERE = Path(run_mod.__file__).resolve().parent
needs_champion = pytest.mark.skipif(
    not run_mod.CHAMPION_PATH.is_file(), reason="champion checkpoint absent"
)
FATAL_RUNNERS = (
    "run_watch_episode",
    "run_play_episode",
    "play_session",
    "run_parity_probe",
    "default_check",
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    from src.core import game_config

    for key in run_mod.schema.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    prev = game_config._current_config
    yield
    game_config._current_config = prev


@pytest.fixture
def fatal_runners(monkeypatch):
    def fatal(*_args, **_kwargs):
        raise AssertionError("a harness episode runner was called from a test")

    import torch

    for name in FATAL_RUNNERS:
        monkeypatch.setattr(run_mod, name, fatal)
    monkeypatch.setenv("SNAKE_DQN_DEVICE", os.environ.get("SNAKE_DQN_DEVICE", "cpu"))
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


class TestPins:
    def test_protocol_pins_equal_the_current_bytes(self):
        sha = hashlib.sha256((HERE / "protocol.md").read_bytes()).hexdigest()
        assert run_mod.PROTOCOL_SHA256 == audit_mod.PROTOCOL_SHA256 == sha

    def test_v8_pins_agree_across_hook_harness_and_audit(self):
        assert run_mod.V8_SOURCE_SHA256S == serving.V8_STRICT_RECEIPT_SOURCE_SHA256S
        assert audit_mod.V8_SOURCE_SHA256S == serving.V8_STRICT_RECEIPT_SOURCE_SHA256S
        assert run_mod.V8_STRICT_RECEIPT_SHA256 == serving.V8_STRICT_RECEIPT_SHA256
        assert audit_mod.V8_STRICT_RECEIPT_SHA256 == serving.V8_STRICT_RECEIPT_SHA256
        assert run_mod.V8_STRICT_INTENT_SHA256 == serving.V8_STRICT_INTENT_SHA256
        assert audit_mod.V8_STRICT_INTENT_SHA256 == serving.V8_STRICT_INTENT_SHA256
        assert run_mod.V8_STRICT_AUDIT_REPORT_SHA256 == serving.V8_STRICT_AUDIT_REPORT_SHA256
        assert audit_mod.V8_STRICT_AUDIT_REPORT_SHA256 == serving.V8_STRICT_AUDIT_REPORT_SHA256
        assert run_mod.V8_METHOD == audit_mod.V8_METHOD == serving.V8_STRICT_RECEIPT_METHOD
        assert run_mod.V8_LAMBDA == serving.V8_LAMBDA == 8.0
        assert run_mod.CHAMPION_SHA256 == audit_mod.CHAMPION_SHA256
        assert run_mod.CHAMPION_SHA256 == serving.V8_STRICT_RECEIPT_CHECKPOINT_SHA256
        identity = serving.wrapper_identity_v8()
        assert audit_mod.wrapper_failures(identity, "tree") == []
        assert identity["descriptor"] == audit_mod.DESCRIPTOR
        assert audit_mod.DESCRIPTOR == SpaceAndHeadVeto(8.0).descriptor()
        assert audit_mod.V7_METHOD == run_mod.V7_METHOD == serving.V7_STRICT_RECEIPT_METHOD

    def test_design_matches_the_producer_and_the_v7_lane(self):
        real, smoke = audit_mod.DESIGN[False], audit_mod.DESIGN[True]
        assert real["counts"] == run_mod.COUNTS == v7run.COUNTS
        assert run_mod.COUNTS == {"watch": 25, "play": 25, "parity": 2}
        assert smoke["counts"] == run_mod.SMOKE_COUNTS == v7run.SMOKE_COUNTS
        assert real["domain"] == run_mod.SEED_DOMAIN == "apex-veto-v8-web-serving-v1"
        assert smoke["domain"] == run_mod.SMOKE_SEED_DOMAIN
        assert run_mod.HORIZON == v7run.HORIZON == 5000
        assert (real["horizon"], real["parity_horizon"]) == (run_mod.HORIZON, run_mod.HORIZON)
        assert (smoke["horizon"], smoke["parity_horizon"]) == (
            run_mod.SMOKE_HORIZON,
            run_mod.SMOKE_PARITY_HORIZON,
        )
        assert audit_mod.RELEASE_ENV == run_mod.RELEASE_ENV
        assert run_mod.RELEASE_ENV["SNAKE_SERVE_VETO_VARIANT"] == "v8"
        assert audit_mod.DEFAULT_CASE_WATCH == run_mod.DEFAULT_CASE_WATCH
        assert audit_mod.OFF_CASE == run_mod.OFF_CASE
        assert set(run_mod.DEFAULT_CASES) == set(audit_mod.DEFAULT_CASE_WATCH) | {
            audit_mod.OFF_CASE
        }

    def test_schema_keys_match_the_writers(self):
        schema = run_mod.schema
        counters = SpaceAndHeadCounters().to_dict()
        ints = [k for k in counters if k not in schema.DIAGNOSTIC_TIMING_KEYS]
        assert tuple(ints) == schema.DIAGNOSTIC_KEYS
        assert set(schema.DIAGNOSTIC_TIMING_KEYS) == set(counters) - set(ints)
        record = SpaceAndHeadVeto(8.0).diagnostics_record()
        assert set(record) == set(counters) | {
            schema.HEAD_AVOIDANCE_KEY,
            schema.REFERENCE_LAMBDA_KEY,
            schema.V7_NESTED_KEY,
            schema.V7_PROBE_COUNTERS_KEY,
        }
        v7 = SpacePreferenceCounters().to_dict()
        assert schema.V7_DIAGNOSTIC_KEYS == v7run.schema.DIAGNOSTIC_KEYS
        assert set(schema.V7_DIAGNOSTIC_KEYS) <= set(v7)
        assert schema.V5_DIAGNOSTIC_KEYS == v7run.schema.V5_DIAGNOSTIC_KEYS
        assert schema.COUNTER_KEYS == v2run.schema.COUNTER_KEYS
        assert tuple(record[schema.V7_PROBE_COUNTERS_KEY]) == schema.COUNTER_KEYS
        assert schema.ENV_KEYS == (serving.ENV_WATCH_HERO, serving.ENV_PLAY_AI, serving.ENV_VARIANT)


class TestSeeds:
    def test_real_and_smoke_banks_are_fresh(self):
        for domain, counts in (
            (run_mod.SEED_DOMAIN, run_mod.COUNTS),
            (run_mod.SMOKE_SEED_DOMAIN, run_mod.SMOKE_COUNTS),
        ):
            seeds = run_mod.world_seeds(domain, counts)
            assert seeds == run_mod.world_seeds(domain, counts)
            report = run_mod.seed_report(seeds)
            assert report["disjoint"] is True and report["v8_earlier_overlap"] == []

    @pytest.mark.parametrize(
        "domain,purpose",
        [
            ("apex-veto-v7-web-serving-v1", "watch"),
            ("apex-veto-v7-web-serving-smoke-v1", "parity"),
            ("apex-veto-v8-strict-final-v1", "worlds"),
            ("apex-veto-v8-strict-dev-v1", "worlds"),
            ("apex-veto-v8-screen-v1", "worlds"),
        ],
    )
    def test_report_catches_an_earlier_seed(self, domain, purpose):
        from research.apex_safety_20260926 import dev_screen

        seed = dev_screen.uint32_seed(domain, purpose, 3)
        report = run_mod.seed_report({"watch": [seed], "play": [7]})
        assert report["disjoint"] is False
        assert 7 in report["v8_earlier_overlap"] and seed in report["v8_earlier_overlap"]

    def test_report_catches_a_played_v8_strict_seed(self):
        observed = run_mod.v8_strict_observed_seeds()
        assert len(observed) == 2 and all(observed.values())
        seed = next(iter(observed.values()))[-1]
        report = run_mod.seed_report({"watch": [seed]})
        assert report["disjoint"] is False and seed in report["v8_earlier_overlap"]

    def test_duplicate_seeds_are_refused(self):
        seeds = run_mod.world_seeds(run_mod.SEED_DOMAIN, {"watch": 1})["watch"]
        assert run_mod.seed_report({"watch": seeds, "play": seeds})["disjoint"] is False


class TestEnv:
    def test_serving_env_sets_exactly_and_restores(self, monkeypatch):
        monkeypatch.setenv(serving.ENV_PLAY_AI, "1")
        monkeypatch.setenv(serving.ENV_WATCH_HERO, "0")
        with run_mod.serving_env(run_mod.RELEASE_ENV):
            assert os.environ.get(serving.ENV_VARIANT) == "v8"
            assert serving.ENV_WATCH_HERO not in os.environ
            assert serving.ENV_PLAY_AI not in os.environ
            flags = serving.ServingVetoFlags.from_env()
            assert flags == serving.ServingVetoFlags(True, False, "v8")
        assert os.environ[serving.ENV_PLAY_AI] == "1"
        assert os.environ[serving.ENV_WATCH_HERO] == "0"
        assert serving.ENV_VARIANT not in os.environ


class TestMainGuards:
    def test_existing_out_is_refused(self, fatal_runners, tmp_path):
        with pytest.raises(RuntimeError, match="create-only"):
            run_mod.main(["--smoke", "--out", str(tmp_path)])

    def test_smoke_never_writes_under_the_artifact_root(self, fatal_runners):
        out = run_mod.ARTIFACT_ROOT / "never-created-v8-serving-smoke"
        with pytest.raises(RuntimeError, match="artifact root"):
            run_mod.main(["--smoke", "--out", str(out)])
        assert not out.exists()

    def test_non_champion_checkpoint_is_refused(self, fatal_runners, tmp_path):
        other = tmp_path / "other.pth"
        other.write_bytes(b"not the champion")
        with pytest.raises(RuntimeError, match="pinned champion"):
            run_mod.main(["--smoke", "--out", str(tmp_path / "o"), "--checkpoint", str(other)])
        assert not (tmp_path / "o").exists()

    def test_non_smoke_needs_clean_tree_and_pinned_protocol(self, fatal_runners, monkeypatch):
        monkeypatch.setattr(run_mod, "git_state", lambda: {"commit": "x", "dirty": True})

        class Args:
            smoke, checkpoint = False, run_mod.CHAMPION_PATH

        with pytest.raises(RuntimeError, match="clean git tree"):
            run_mod.build_intent(Args(), run_mod.CHAMPION_SHA256)
        monkeypatch.setattr(run_mod, "git_state", lambda: {"commit": "x", "dirty": False})
        monkeypatch.setattr(run_mod, "PROTOCOL_SHA256", "0" * 64)
        with pytest.raises(RuntimeError, match="pinned sha256"):
            run_mod.build_intent(Args(), run_mod.CHAMPION_SHA256)

    @pytest.mark.parametrize(
        "name,value",
        [
            ("V8_STRICT_RECEIPT_CHECKPOINT_SHA256", "0" * 64),
            ("V8_STRICT_RECEIPT_SHA256", "0" * 64),
            ("V8_STRICT_INTENT_SHA256", "0" * 64),
            ("V8_STRICT_AUDIT_REPORT_SHA256", "0" * 64),
            ("V8_LAMBDA", 4.0),
        ],
    )
    def test_changed_hook_pins_refuse_before_any_episode(
        self, fatal_runners, monkeypatch, name, value
    ):
        monkeypatch.setattr(serving, name, value)

        class Args:
            smoke, checkpoint = True, run_mod.CHAMPION_PATH

        with pytest.raises(RuntimeError, match="v8"):
            run_mod.build_intent(Args(), run_mod.CHAMPION_SHA256)


@needs_champion
class TestTinyEpisodesOnTheChampion:
    """The producer's runners on the real session (tiny horizons; called directly)."""

    def _ctx(self):
        return {
            "checkpoint": run_mod.CHAMPION_PATH,
            "checkpoint_sha256": run_mod.CHAMPION_SHA256,
            "intent_sha256": "0" * 64,
        }

    def test_watch_episode_serves_v8_on_the_hero(self):
        record = run_mod.run_watch_episode(0, 11, 30, self._ctx())
        assert record["status"] == "complete", record["error"]
        assert set(record) == set(run_mod.schema.EPISODE_KEYS)
        veto, hero = record["veto"], record["served"]["hero_id"]
        assert set(veto) == set(run_mod.schema.VETO_KEYS)
        assert veto["variant"] == "v8" and veto["wrapped_snake_ids"] == [hero]
        assert record["snakes_with_veto"] == [hero]
        assert audit_mod.wrapper_failures(record["wrapper"], "w") == []
        sid = str(hero)
        assert veto["per_snake"][sid]["decisions"] == record["decision_frames"]["per_snake"][sid]
        assert (
            audit_mod.diagnostics_failures(
                veto["diagnostics_per_snake"][sid], veto["per_snake"][sid], "w"
            )
            == []
        )
        assert veto["diagnostics_total"] == audit_mod.diagnostics_total(
            veto["diagnostics_per_snake"]
        )
        assert record["env"] == run_mod.RELEASE_ENV
        diag = veto["diagnostics_per_snake"][sid]
        assert diag["head_avoidance"] is True and diag["reference_lambda"] is None
        assert serving.ENV_VARIANT not in os.environ  # the env is restored

    def test_play_episode_is_served_unwrapped(self):
        ctx = self._ctx()
        sess = run_mod.play_session(ctx)
        record = run_mod.run_play_episode(sess, 0, 12, 30, ctx)
        assert record["status"] == "complete", record["error"]
        assert record["veto"]["active"] is False and record["snakes_with_veto"] == []
        assert record["veto"]["reason"] == audit_mod.PLAY_REASON
        assert record["decision_frames"] == {"per_snake": {}, "total": 0}
        assert record["episode"]["inputs_sent"] >= 1

    def test_parity_probe_matches_the_v8_rollout(self):
        probe = run_mod.run_parity_probe(0, 13, 30, self._ctx())
        assert probe["status"] == "complete", probe["error"]
        assert set(probe) == set(run_mod.schema.PARITY_KEYS)
        assert probe["first_divergence_frame"] is None and probe["frames_compared"] == 30
        assert probe["trace_sha256_equal"] and probe["veto_counters_equal"]
        assert probe["diagnostics_equal"]
        assert probe["session"]["wrapper_method"] == audit_mod.V8_METHOD
        assert probe["rollout"]["wrapper_method"] == audit_mod.V8_METHOD

    def test_default_check_passes_on_this_tree(self, monkeypatch):
        # S6 is a PRE-release precondition (v7 released, v8 opt-in). After a v8 release flip,
        # pin the pre-release default here so this keeps testing the harness logic.
        monkeypatch.setattr(serving, "VARIANT_RELEASED_DEFAULT", "v7")
        doc = run_mod.default_check(run_mod.CHAMPION_PATH)
        assert run_mod.default_check_failures(doc) == [] and doc["pass"] is True
        assert doc["builds"]["empty_env"]["modes"]["watch"]["method"] == run_mod.V7_METHOD
        assert doc["builds"]["variant_v5"]["modes"]["watch"]["method"] == run_mod.V5_METHOD
        assert doc["builds"]["variant_v2"]["modes"]["watch"]["method"] == run_mod.V2_METHOD
        assert all(key not in os.environ for key in run_mod.schema.ENV_KEYS)
