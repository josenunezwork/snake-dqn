"""Producer of the frp3-s12 checkpoint-swap web-serving run
(research/frp3_checkpoint_serving_20261005/serving_run.py) and the owner pin tool.

Pins, seeds, env handling, guards and the pin tool; tiny-horizon episodes on the real FRP-v3
checkpoint under a test pin. Every test that calls ``main`` first makes every episode runner
fatal, so no test can play a served episode through the harness entry point.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from research.apex_veto_v8_serving_20261003 import serving_run as v8run  # noqa: E402
from research.frp3_checkpoint_serving_20261005 import fill_strict_pin  # noqa: E402
from research.frp3_checkpoint_serving_20261005 import serving_audit as audit_mod  # noqa: E402
from research.frp3_checkpoint_serving_20261005 import serving_run as run_mod  # noqa: E402
from web.backend import safety_veto_serving as serving  # noqa: E402
from web.backend import served_checkpoint as registry  # noqa: E402

HERE = Path(run_mod.__file__).resolve().parent
MAIN_CHAMPION = run_mod.CHAMPION_PATH
needs_checkpoints = pytest.mark.skipif(
    not (MAIN_CHAMPION.is_file() and run_mod.CHECKPOINT_ARTIFACT_PATH.is_file()),
    reason="champion or FRP-v3 s12 checkpoint absent",
)
FATAL_RUNNERS = (
    "run_watch_episode",
    "run_play_episode",
    "play_session",
    "run_parity_probe",
    "default_check",
    "acquire_slot",
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


@pytest.fixture
def champion_default(monkeypatch):
    import web.backend.session as session_module

    monkeypatch.setattr(session_module, "DEFAULT_CHECKPOINT", str(MAIN_CHAMPION))


class TestPins:
    def test_protocol_pins_equal_the_current_bytes(self):
        sha = hashlib.sha256((HERE / "protocol.md").read_bytes()).hexdigest()
        assert run_mod.PROTOCOL_SHA256 == audit_mod.PROTOCOL_SHA256 == sha

    def test_identity_pins_agree_across_registry_harness_audit_and_tool(self):
        entry = registry.REGISTRY[run_mod.CHECKPOINT_NAME]
        for sha in (
            run_mod.CHECKPOINT_SHA256,
            audit_mod.CHECKPOINT_SHA256,
            fill_strict_pin.CHECKPOINT_SHA256,
            registry.FRP3_S12_SHA256,
        ):
            assert sha == entry.sha256
        assert run_mod.CHECKPOINT_FILENAME == audit_mod.CHECKPOINT_FILENAME == entry.filename
        assert str(run_mod.CHECKPOINT_ARTIFACT_PATH) == audit_mod.CHECKPOINT_ARTIFACT_PATH
        assert audit_mod.CHECKPOINT_ARTIFACT_PATH == entry.artifact_path
        assert run_mod.CHECKPOINT_NAME == audit_mod.CHECKPOINT_NAME == fill_strict_pin.NAME
        assert run_mod.V8_METHOD == audit_mod.V8_METHOD == registry.V8_METHOD
        assert fill_strict_pin.V8_METHOD == serving.V8_STRICT_RECEIPT_METHOD == run_mod.V8_METHOD
        assert run_mod.V8_SOURCE_SHA256S == serving.V8_STRICT_RECEIPT_SOURCE_SHA256S
        assert audit_mod.V8_SOURCE_SHA256S == serving.V8_STRICT_RECEIPT_SOURCE_SHA256S
        assert run_mod.CHAMPION_SHA256 == audit_mod.CHAMPION_SHA256 == registry.CHAMPION_SHA256
        assert run_mod.PROFILE_DIGEST == audit_mod.PROFILE_DIGEST
        from research.apex_safety_20260926 import dev_screen

        assert run_mod.PROFILE_DIGEST == dev_screen.PROFILE_DIGEST
        assert run_mod.SMOKE_PIN_ROOT == audit_mod.SMOKE_PIN_ROOT
        assert audit_mod.STRICT_FILES == registry.STRICT_FILES
        assert audit_mod.PINS_FILE == Path(registry.PINS_PATH)
        assert audit_mod.wrapper_failures(serving.wrapper_identity_v8(), "tree") == []

    def test_design_matches_the_producer_and_the_v8_lane(self):
        real, smoke = audit_mod.DESIGN[False], audit_mod.DESIGN[True]
        assert real["counts"] == run_mod.COUNTS == v8run.COUNTS
        assert smoke["counts"] == run_mod.SMOKE_COUNTS == v8run.SMOKE_COUNTS
        assert real["domain"] == run_mod.SEED_DOMAIN == "frp3-checkpoint-web-serving-v1"
        assert smoke["domain"] == run_mod.SMOKE_SEED_DOMAIN
        assert run_mod.HORIZON == v8run.HORIZON == 5000
        assert (real["horizon"], real["parity_horizon"]) == (run_mod.HORIZON, run_mod.HORIZON)
        assert (smoke["horizon"], smoke["parity_horizon"]) == (
            run_mod.SMOKE_HORIZON,
            run_mod.SMOKE_PARITY_HORIZON,
        )
        assert audit_mod.RELEASE_ENV == run_mod.RELEASE_ENV
        assert run_mod.RELEASE_ENV == {
            "SNAKE_SERVE_CHECKPOINT": "frp3-s12",
            "SNAKE_SERVE_VETO_WATCH_HERO": None,
            "SNAKE_SERVE_VETO_PLAY_AI": None,
            "SNAKE_SERVE_VETO_VARIANT": None,
        }
        assert audit_mod.DEFAULT_CASE_EXPECT == run_mod.DEFAULT_CASE_EXPECT
        assert audit_mod.DEFAULT_CASE_ENV == run_mod.DEFAULT_CASES
        assert audit_mod.REFUSED_CASES == run_mod.REFUSED_CASES

    def test_schema_extends_the_v8_schema(self):
        schema = run_mod.schema
        v8 = v8run.schema
        assert schema.SCHEMA != v8.SCHEMA and schema.STUDY_ID != v8.STUDY_ID
        assert set(schema.EPISODE_KEYS) == set(v8.EPISODE_KEYS) | {"served_checkpoint"}
        assert schema.VETO_KEYS == v8.VETO_KEYS and schema.COUNTER_KEYS == v8.COUNTER_KEYS
        assert schema.DIAGNOSTIC_KEYS == v8.DIAGNOSTIC_KEYS
        assert schema.ENV_KEYS == (registry.ENV_CHECKPOINT,) + tuple(v8.ENV_KEYS)
        choice = registry.ServedCheckpointChoice(requested=None)
        assert tuple(choice.to_dict()) == schema.SERVED_CHECKPOINT_KEYS


class TestSeeds:
    def test_real_and_smoke_banks_are_fresh(self):
        for domain, counts in (
            (run_mod.SEED_DOMAIN, run_mod.COUNTS),
            (run_mod.SMOKE_SEED_DOMAIN, run_mod.SMOKE_COUNTS),
        ):
            seeds = run_mod.world_seeds(domain, counts)
            report = run_mod.seed_report(seeds)
            assert report["disjoint"] is True and report["frp3_earlier_overlap"] == []
            assert report["strict_observed_checked"] is False

    @pytest.mark.parametrize(
        "domain,purpose",
        [
            ("apex-veto-v8-web-serving-v1", "watch"),
            ("apex-veto-v8-web-serving-smoke-v1", "parity"),
            ("apex-veto-v8-strict-final-v1", "worlds"),
        ],
    )
    def test_report_catches_an_earlier_seed(self, domain, purpose):
        from research.apex_safety_20260926 import dev_screen

        seed = dev_screen.uint32_seed(domain, purpose, 3)
        report = run_mod.seed_report({"watch": [seed], "play": [7]})
        assert report["disjoint"] is False
        assert 7 in report["frp3_earlier_overlap"] and seed in report["frp3_earlier_overlap"]

    def test_report_catches_an_frp3_excluded_seed(self):
        seed = run_mod.frp3_exclusions()[-1]
        report = run_mod.seed_report({"watch": [seed]})
        assert report["disjoint"] is False and seed in report["frp3_earlier_overlap"]

    def test_report_catches_a_strict_run_seed(self, tmp_path):
        root = tmp_path / "strict"
        (root / "output").mkdir(parents=True)
        (root / "intent.json").write_text(json.dumps({"banks": {"final": [123456789]}}))
        (root / "output" / "rosters.json").write_text(json.dumps([{"world_seed": 987654321}]))
        pin = {"strict_receipt": {"root": str(root), "intent": {"path": "intent.json"}}}
        for seed in (123456789, 987654321):
            report = run_mod.seed_report({"watch": [seed]}, pin)
            assert report["strict_observed_checked"] is True
            assert report["disjoint"] is False and seed in report["frp3_earlier_overlap"]

    def test_report_catches_a_strict_bank_domain_seed(self, tmp_path):
        from research.apex_safety_20260926 import dev_screen

        (tmp_path / "intent.json").write_text(
            json.dumps({"banks": {"x": [11]}, "namespaces": {"final": "frp3-strict-final-v9"}})
        )
        pin = {"strict_receipt": {"root": str(tmp_path), "intent": {"path": "intent.json"}}}
        assert run_mod.strict_domains(pin) == ["frp3-strict-final-v9"]
        seed = dev_screen.uint32_seed("frp3-strict-final-v9", "worlds", 5)
        report = run_mod.seed_report({"watch": [seed]}, pin)
        assert report["disjoint"] is False and seed in report["frp3_earlier_overlap"]

    def test_strict_intent_without_seeds_is_refused(self, tmp_path):
        (tmp_path / "intent.json").write_text(json.dumps({"nothing": 1}))
        pin = {"strict_receipt": {"root": str(tmp_path), "intent": {"path": "intent.json"}}}
        with pytest.raises(RuntimeError, match="no seeds"):
            run_mod.seed_report({"watch": [5_000_000]}, pin)


class TestEnvAndPins:
    def test_serving_env_sets_exactly_and_restores(self, monkeypatch):
        monkeypatch.setenv(serving.ENV_PLAY_AI, "1")
        monkeypatch.setenv(registry.ENV_CHECKPOINT, "champion")
        with run_mod.serving_env(run_mod.RELEASE_ENV):
            assert os.environ[registry.ENV_CHECKPOINT] == "frp3-s12"
            assert serving.ENV_PLAY_AI not in os.environ
            assert serving.ENV_VARIANT not in os.environ
            assert serving.ServingVetoFlags.from_env() == serving.ServingVetoFlags(
                True, False, "v8"
            )
        assert os.environ[serving.ENV_PLAY_AI] == "1"
        assert os.environ[registry.ENV_CHECKPOINT] == "champion"

    def test_smoke_pin_validates_and_unfilled_does_not(self):
        assert registry.strict_pin_problems("frp3-s12", run_mod.smoke_pins()) == []
        assert registry.strict_pin_problems("frp3-s12", run_mod.unfilled_pins())

    def test_pins_override_restores(self):
        saved, before = registry.PINS_PATH, registry.v8_gated_checkpoint_sha256s()
        with run_mod.pins_override(run_mod.smoke_pins()):
            assert registry.PINS_PATH != saved
            assert registry.v8_gated_checkpoint_sha256s() == [run_mod.CHECKPOINT_SHA256]
        with run_mod.pins_override(run_mod.unfilled_pins()):
            assert registry.v8_gated_checkpoint_sha256s() == []
        assert registry.PINS_PATH == saved
        assert registry.v8_gated_checkpoint_sha256s() == before

    @needs_checkpoints
    def test_real_run_refuses_while_the_pin_is_unfilled(self):
        with run_mod.pins_override(run_mod.unfilled_pins()):
            with pytest.raises(RuntimeError, match="not filled/valid"):
                run_mod.resolve_identity(smoke=False)

    @needs_checkpoints
    def test_smoke_resolves_the_swap_under_the_placeholder(self):
        identity = run_mod.resolve_identity(smoke=True)
        assert identity["choice"]["name"] == "frp3-s12"
        assert identity["choice"]["sha256"] == run_mod.CHECKPOINT_SHA256
        assert identity["pin"]["strict_receipt"]["root"] == run_mod.SMOKE_PIN_ROOT
        assert identity["pin"]["checkpoint_sha256"] == run_mod.CHECKPOINT_SHA256


class TestGuards:
    def test_existing_out_is_refused(self, fatal_runners, tmp_path):
        with pytest.raises(RuntimeError, match="create-only"):
            run_mod.main(["--smoke", "--out", str(tmp_path)])

    def test_smoke_never_writes_under_the_artifact_root(self, fatal_runners):
        out = run_mod.ARTIFACT_ROOT / "never-created-frp3-serving-smoke"
        with pytest.raises(RuntimeError, match="artifact root"):
            run_mod.main(["--smoke", "--out", str(out)])
        assert not out.exists()

    @needs_checkpoints
    def test_real_run_without_pin_refuses_before_any_slot(self, fatal_runners, tmp_path):
        with run_mod.pins_override(run_mod.unfilled_pins()):
            with pytest.raises(RuntimeError, match="not filled/valid"):
                run_mod.main(["--out", str(tmp_path / "o")])
        assert not (tmp_path / "o").exists()

    @needs_checkpoints
    def test_real_run_needs_power_and_lid(self, fatal_runners, monkeypatch, tmp_path):
        monkeypatch.setattr(run_mod, "resolve_identity", lambda smoke: {"run_pins": None})
        monkeypatch.setattr(run_mod, "on_ac_power", lambda: True)
        monkeypatch.setattr(run_mod, "lid_open", lambda: False)
        with pytest.raises(RuntimeError, match="open lid"):
            run_mod.main(["--out", str(tmp_path / "o")])

    def test_guard_waits_then_fails(self):
        readings = iter([False, False, True])
        slept = []
        guard = run_mod.make_guard(
            ac=lambda: next(readings), lid=lambda: True, sleep=slept.append, poll=1.0
        )
        guard()
        assert slept == [1.0, 1.0]
        stuck = run_mod.make_guard(
            ac=lambda: True, lid=lambda: False, sleep=lambda s: None, poll=10.0, max_pause=25.0
        )
        with pytest.raises(RuntimeError, match="unsafe"):
            stuck()

    @pytest.mark.parametrize(
        "text,expected",
        [
            ('| "AppleClamshellState" = No', True),
            ('| "AppleClamshellState" = Yes', False),
            ('| "AppleClamshellState" = Maybe', None),
            ("", None),
        ],
    )
    def test_clamshell_parser(self, text, expected):
        assert run_mod.parse_clamshell(text) is expected

    @needs_checkpoints
    def test_non_smoke_intent_needs_clean_tree_and_pinned_protocol(
        self, monkeypatch, champion_default
    ):
        identity = run_mod.resolve_identity(smoke=True)
        monkeypatch.setattr(
            run_mod, "seed_report", lambda s, p: {"disjoint": True, "strict_observed_checked": True}
        )
        monkeypatch.setattr(run_mod, "git_state", lambda: {"commit": "x", "dirty": True})

        class Args:
            smoke = False

        with run_mod.pins_override(identity["run_pins"]):
            with pytest.raises(RuntimeError, match="clean git tree"):
                run_mod.build_intent(Args(), identity)
            monkeypatch.setattr(run_mod, "git_state", lambda: {"commit": "x", "dirty": False})
            monkeypatch.setattr(run_mod, "PROTOCOL_SHA256", "f" * 64)
            with pytest.raises(RuntimeError, match="pinned sha256"):
                run_mod.build_intent(Args(), identity)

    @needs_checkpoints
    @pytest.mark.parametrize(
        "name,value",
        [
            ("V8_STRICT_RECEIPT_CHECKPOINT_SHA256", "0" * 64),
            ("V8_LAMBDA", 4.0),
            ("VARIANT_RELEASED_DEFAULT", "v7"),
        ],
    )
    def test_changed_hook_pins_refuse(self, monkeypatch, champion_default, name, value):
        identity = run_mod.resolve_identity(smoke=True)
        monkeypatch.setattr(serving, name, value)

        class Args:
            smoke = True

        with run_mod.pins_override(identity["run_pins"]):
            with pytest.raises(RuntimeError, match="v8"):
                run_mod.build_intent(Args(), identity)

    @needs_checkpoints
    def test_flipped_checkpoint_default_refuses(self, monkeypatch, champion_default):
        identity = run_mod.resolve_identity(smoke=True)
        monkeypatch.setattr(registry, "CHECKPOINT_RELEASED_DEFAULT", "frp3-s12")

        class Args:
            smoke = True

        with run_mod.pins_override(identity["run_pins"]):
            with pytest.raises(RuntimeError, match="released default"):
                run_mod.build_intent(Args(), identity)

    def test_missing_champion_refuses(self, monkeypatch, tmp_path):
        import web.backend.session as session_module

        monkeypatch.setattr(session_module, "DEFAULT_CHECKPOINT", str(tmp_path / "absent.pth"))
        identity = {"pin": None, "choice": {}, "run_pins": None}
        monkeypatch.setattr(run_mod, "seed_report", lambda s, p: {"disjoint": True})

        class Args:
            smoke = True

        with pytest.raises(RuntimeError, match="champion is missing"):
            run_mod.build_intent(Args(), identity)


def write_strict_root(tmp_path, checkpoint_sha=None, audit_status="PASS", outcome="STRICT_PASS"):
    """A strict root in the sequential strict template's shapes (as the v8 strict run wrote)."""
    root = tmp_path / "strict"
    (root / "output" / "audit").mkdir(parents=True)
    intent = root / "intent.json"
    intent.write_text(
        json.dumps(
            {
                "arms": {
                    "candidate": {
                        "checkpoint_sha256": checkpoint_sha or run_mod.CHECKPOINT_SHA256,
                        "method": run_mod.V8_METHOD,
                    },
                    "incumbent": {"checkpoint_sha256": run_mod.CHAMPION_SHA256},
                },
                "banks": {"final": [123456789]},
            }
        )
    )
    report = root / "output" / "audit" / "audit.json"
    report.write_text(
        json.dumps({"status": audit_status, "recomputed": {"expected_outcome": outcome}})
    )
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    (root / "output" / "receipt.json").write_text(
        json.dumps(
            {
                "decision": "STOP_PASS",
                "intent_sha256": sha(intent),
                "audit_report_sha256": sha(report),
            }
        )
    )
    return root


class TestFillStrictPin:
    def pins_copy(self, tmp_path):
        doc = json.loads(Path(registry.PINS_PATH).read_text())
        doc["checkpoints"]["frp3-s12"]["strict_receipt"] = None
        path = tmp_path / "pins.json"
        path.write_text(json.dumps(doc))
        return path

    def args(self, root, pins, *extra):
        return [
            "--root",
            str(root),
            "--receipt",
            "output/receipt.json",
            "--intent",
            "intent.json",
            "--audit-report",
            "output/audit/audit.json",
            "--pins",
            str(pins),
            *extra,
        ]

    def test_fills_once_and_validates(self, tmp_path):
        root, pins = write_strict_root(tmp_path), self.pins_copy(tmp_path)
        assert fill_strict_pin.main(self.args(root, pins, "--dry-run")) == 0
        assert json.loads(pins.read_text())["checkpoints"]["frp3-s12"]["strict_receipt"] is None
        assert fill_strict_pin.main(self.args(root, pins)) == 0
        doc = json.loads(pins.read_text())
        assert registry.strict_pin_problems("frp3-s12", doc) == []
        pin = doc["checkpoints"]["frp3-s12"]["strict_receipt"]
        assert pin["root"] == str(root.resolve())
        assert (
            pin["intent"]["sha256"]
            == hashlib.sha256((root / "intent.json").read_bytes()).hexdigest()
        )
        with pytest.raises(SystemExit, match="already filled"):
            fill_strict_pin.main(self.args(root, pins))

    @pytest.mark.parametrize(
        "kwargs,needle",
        [
            ({"checkpoint_sha": "0" * 64}, "candidate.checkpoint_sha256"),
            ({"checkpoint_sha": run_mod.CHAMPION_SHA256}, "candidate.checkpoint_sha256"),
            ({"audit_status": "FAIL"}, "status is not PASS"),
            ({"outcome": "STRICT_FAIL"}, "expected_outcome"),
        ],
    )
    def test_refuses_bad_evidence(self, tmp_path, kwargs, needle):
        root = write_strict_root(tmp_path, **kwargs)
        with pytest.raises(SystemExit, match=needle):
            fill_strict_pin.main(self.args(root, self.pins_copy(tmp_path)))

    def test_refuses_a_receipt_that_binds_other_bytes(self, tmp_path):
        root = write_strict_root(tmp_path)
        (root / "intent.json").write_text((root / "intent.json").read_text() + " ")
        with pytest.raises(SystemExit, match="receipt.intent_sha256"):
            fill_strict_pin.main(self.args(root, self.pins_copy(tmp_path)))

    def test_refuses_missing_files_and_unsafe_paths(self, tmp_path):
        root, pins = write_strict_root(tmp_path), self.pins_copy(tmp_path)
        (root / "output" / "receipt.json").unlink()
        with pytest.raises(SystemExit, match="missing"):
            fill_strict_pin.main(self.args(root, pins))
        for bad in (str(root / "intent.json"), "../strict/intent.json"):
            with pytest.raises(SystemExit, match="relative"):
                fill_strict_pin.build_pin(root, "output/receipt.json", bad, "x")


@needs_checkpoints
class TestTinyEpisodesOnTheSwap:
    """The producer's runners on the real session under a test pin (tiny horizons)."""

    @pytest.fixture(autouse=True)
    def _pins(self, champion_default):
        with run_mod.pins_override(run_mod.smoke_pins()):
            yield

    def _ctx(self):
        return {
            "checkpoint": run_mod.CHECKPOINT_ARTIFACT_PATH,
            "checkpoint_sha256": run_mod.CHECKPOINT_SHA256,
            "intent_sha256": "0" * 64,
            "guard": lambda: None,
        }

    def test_watch_episode_serves_the_swap_behind_v8(self):
        record = run_mod.run_watch_episode(0, 11, 30, self._ctx())
        assert record["status"] == "complete", record["error"]
        assert set(record) == set(run_mod.schema.EPISODE_KEYS)
        assert audit_mod.served_checkpoint_failures(record["served_checkpoint"], "w") == []
        assert record["checkpoint"]["session_sha256"] == run_mod.CHECKPOINT_SHA256
        veto, hero = record["veto"], record["served"]["hero_id"]
        assert veto["variant"] == "v8" and veto["wrapped_snake_ids"] == [hero]
        assert veto["strict_receipt_checkpoint_match"] is True and veto["reason"] is None
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
        assert all(key not in os.environ for key in run_mod.schema.ENV_KEYS)

    def test_play_episode_is_served_unwrapped_with_frp3_opponents(self):
        ctx = self._ctx()
        sess = run_mod.play_session(ctx)
        record = run_mod.run_play_episode(sess, 0, 12, 30, ctx)
        assert record["status"] == "complete", record["error"]
        assert record["checkpoint"]["session_sha256"] == run_mod.CHECKPOINT_SHA256
        assert record["veto"]["active"] is False and record["snakes_with_veto"] == []
        assert record["veto"]["reason"] == audit_mod.PLAY_REASON
        assert record["decision_frames"] == {"per_snake": {}, "total": 0}

    def test_parity_probe_matches_the_simd_rollout(self):
        probe = run_mod.run_parity_probe(0, 13, 30, self._ctx())
        assert probe["status"] == "complete", probe["error"]
        assert set(probe) == set(run_mod.schema.PARITY_KEYS)
        assert probe["first_divergence_frame"] is None, probe["divergence"]
        assert probe["frames_compared"] == 30
        assert probe["trace_sha256_equal"] and probe["veto_counters_equal"]
        assert probe["diagnostics_equal"]
        assert probe["session"]["wrapper_method"] == audit_mod.V8_METHOD
        assert probe["simd"]["wrapper_method"] == audit_mod.V8_METHOD
        assert probe["simd"]["provenance"] == audit_mod.SIMD_PROVENANCE
        assert probe["session"]["checkpoint_sha256"] == run_mod.CHECKPOINT_SHA256

    def test_default_check_passes_on_this_tree(self):
        doc = run_mod.default_check(run_mod.smoke_pins())
        assert run_mod.default_check_failures(doc) == [] and doc["pass"] is True
        builds = doc["builds"]
        assert builds["empty_env"]["modes"]["watch"]["method"] == run_mod.V8_METHOD
        assert "unfilled placeholder" in builds["frp3_pin_unfilled"]["served_checkpoint"]["reason"]
        assert "under variant v7" in builds["frp3_rollback_variant_v7"]["modes"]["watch"]["reason"]
        assert all(key not in os.environ for key in run_mod.schema.ENV_KEYS)
