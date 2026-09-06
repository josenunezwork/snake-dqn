"""Pin ``src/scripts/train_pqn.py``'s operational contract with the sweep.

The sweep orchestrator treats ``train_rc == 2`` as its ONLY signal that a run
diverged (:func:`src.scripts.sweep.is_clean`), but tests/test_sweep.py stubs the
train command out entirely — it proves sweep *reacts* to rc=2, never that
train_pqn *emits* it. These tests own the producer side of that contract: the
tripwire -> exit-2 mapping, the "always leave a final checkpoint" guarantee the
sweep's checkpoint discovery relies on, and ``build_config``'s precedence.

The trainer is stubbed (no GPU, no sim, no real training) for the CLI/loop
plumbing. The resume tests are the exception: they drive a REAL (tiny)
:class:`PQNTrainer` through save -> restore, because the thing under test is
whether restored weights/optimizer/odometers actually survive the round trip.
"""

import json

import pytest
import torch
import yaml
from pydantic import ValidationError

from src.core.device_manager import DeviceManager
from src.model.obs_spec import RASTER31V2_SHAPES
from src.scripts import sweep, train_pqn
from src.training.pqn_trainer import PQNConfig, PQNTelemetry, PQNTrainer, TripwireError


@pytest.fixture(autouse=True)
def _isolate_device():
    """``main`` calls DeviceManager.override_device; don't leak it to other tests."""
    yield
    DeviceManager.reset_for_testing()


def _telemetry(update: int, agent_steps: int) -> PQNTelemetry:
    """A finite, well-formed telemetry row.

    Constructed as the real dataclass (not a stand-in) so a field added to
    PQNTelemetry fails loudly here instead of being silently masked.
    """
    return PQNTelemetry(
        update=update,
        agent_steps=agent_steps,
        loss=0.5,
        grad_norm=1.0,
        mean_abs_q=2.0,
        max_abs_q=3.0,
        epsilon=0.5,
        mean_reward=0.01,
        action_entropy=1.5,
        kills_per_ep=0.0,
        boost_fraction=0.1,
        pool_size=0,
    )


class _FakeNetwork(torch.nn.Module):
    """Stands in for RasterDuelingNetwork; ``main`` only reprs it and counts params."""

    def __init__(self):
        super().__init__()
        self.loaded_state = None

    def get_num_parameters(self):
        return {"total": 1234}

    def load_state_dict(self, state_dict, strict=True):
        # Records rather than loads: this stand-in has no parameters, so the real
        # nn.Module implementation would reject any incoming key.
        self.loaded_state = state_dict


class _FakeOptimizer:
    """Stands in for the trainer's Adam; only ``load_state_dict`` is exercised."""

    def __init__(self):
        self.loaded_state = None

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict


class FakeTrainer:
    """Minimal stand-in for PQNTrainer exposing only what main/train_loop touch."""

    STEPS_PER_UPDATE = 100

    def __init__(self, config, device=None, trip_after=None):
        self.config = config
        self.device = device
        self.trip_after = trip_after
        self.agent_steps = 0
        self.update_idx = 0
        self.updates = 0
        self.network = _FakeNetwork()
        self.optimizer = _FakeOptimizer()
        self.saved = []

    def epsilon(self) -> float:
        return 0.5

    def update(self) -> PQNTelemetry:
        if self.trip_after is not None and self.updates >= self.trip_after:
            raise TripwireError("max|Q| 1e9 exceeds alarm 1000.0")
        self.updates += 1
        self.agent_steps += self.STEPS_PER_UPDATE
        return _telemetry(self.updates, self.agent_steps)

    def save_checkpoint(self, path: str) -> None:
        self.saved.append(path)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("stub-checkpoint")


def _install_trainer(monkeypatch, trip_after=None):
    """Swap the module-global PQNTrainer that ``main`` resolves at call time."""
    made = []

    def _factory(config, device=None):
        trainer = FakeTrainer(config, device=device, trip_after=trip_after)
        made.append(trainer)
        return trainer

    monkeypatch.setattr(train_pqn, "PQNTrainer", _factory)
    return made


def _run(tmp_path, monkeypatch, *, trip_after=None, extra=()):
    made = _install_trainer(monkeypatch, trip_after=trip_after)
    rc = train_pqn.main(
        [
            "--device",
            "cpu",
            "--out-dir",
            str(tmp_path),
            "--total-steps",
            "1000",
            "--envs",
            "2",
            "--snakes",
            "2",
            *extra,
        ]
    )
    return rc, made[0]


class TestTripwireExitCode:
    """The halt-and-flag contract sweep.is_clean depends on."""

    def test_tripwire_returns_rc_2(self, tmp_path, monkeypatch):
        rc, trainer = _run(tmp_path, monkeypatch, trip_after=3)

        assert rc == 2
        assert trainer.updates == 3, "the loop must halt on the tripwire, not continue"

    def test_tripwire_rc_matches_sweeps_constant(self, tmp_path, monkeypatch):
        """Asserted at one point so the two halves of the contract cannot drift."""
        rc, _ = _run(tmp_path, monkeypatch, trip_after=1)

        assert rc == sweep.TRIPWIRE_RC

    def test_clean_run_returns_rc_0(self, tmp_path, monkeypatch):
        rc, trainer = _run(tmp_path, monkeypatch)

        assert rc == 0
        assert trainer.agent_steps >= 1000

    def test_sweep_rejects_a_tripped_run_and_accepts_a_clean_one(self, tmp_path, monkeypatch):
        tripped_rc, _ = _run(tmp_path / "a", monkeypatch, trip_after=2)
        clean_rc, _ = _run(tmp_path / "b", monkeypatch)

        assert not sweep.is_clean({"train_rc": tripped_rc, "mass_integral": 99.0})
        assert sweep.is_clean({"train_rc": clean_rc, "mass_integral": 1.0})

    def test_tripwire_on_the_first_update_still_returns_rc_2(self, tmp_path, monkeypatch):
        """An immediate divergence has an empty history; rc must not degrade to 0."""
        rc, trainer = _run(tmp_path, monkeypatch, trip_after=0)

        assert rc == 2
        assert trainer.updates == 0
        assert (tmp_path / "history.jsonl").read_text() == ""


class TestArtifactsOnHalt:
    """sweep's checkpoint discovery needs a final checkpoint even on a tripwire."""

    def test_final_checkpoint_saved_even_on_tripwire(self, tmp_path, monkeypatch):
        rc, trainer = _run(tmp_path, monkeypatch, trip_after=2)

        assert rc == 2
        assert (tmp_path / "latest_pqn.pth").exists()
        assert str(tmp_path / "latest_pqn.pth") in trainer.saved

    def test_history_jsonl_written_and_parseable(self, tmp_path, monkeypatch):
        _run(tmp_path, monkeypatch, trip_after=3)

        lines = (tmp_path / "history.jsonl").read_text().splitlines()
        assert len(lines) == 3
        last = json.loads(lines[-1])
        assert last["update"] == 3
        assert last["agent_steps"] == 300
        assert last["max_abs_q"] == 3.0

    def test_out_dir_is_created(self, tmp_path, monkeypatch):
        nested = tmp_path / "runs" / "pqn_x"
        rc, _ = _run(nested, monkeypatch)

        assert rc == 0
        assert (nested / "latest_pqn.pth").exists()


class TestBuildConfigPrecedence:
    """CLI flag beats --config file beats PQNConfig default."""

    def test_cli_flag_beats_config_file(self, tmp_path, monkeypatch):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"pqn": {"num_envs": 64, "rollout_len": 24}}))

        made = _install_trainer(monkeypatch)
        train_pqn.main(
            [
                "--device",
                "cpu",
                "--out-dir",
                str(tmp_path),
                "--total-steps",
                "1",
                "--config",
                str(path),
                "--envs",
                "8",
            ]
        )
        config = made[0].config

        assert config.num_envs == 8, "explicit CLI flag must win"
        assert config.rollout_len == 24, "config file must win over the PQNConfig default"
        assert config.gamma == PQNConfig().gamma, "unset knobs keep the dataclass default"

    def test_config_file_never_splats_none_over_defaults(self, tmp_path, monkeypatch):
        """An almost-empty pqn: block must not blank out PQNConfig's defaults."""
        path = tmp_path / "c.yaml"
        path.write_text("pqn:\n  num_envs: 16\n")

        made = _install_trainer(monkeypatch)
        train_pqn.main(
            [
                "--device",
                "cpu",
                "--out-dir",
                str(tmp_path),
                "--total-steps",
                "1",
                "--config",
                str(path),
            ]
        )
        config = made[0].config
        default = PQNConfig()

        assert config.num_envs == 16
        for field in ("lr", "gamma", "lambda_", "eps_start", "eps_end", "hero_frac", "seed"):
            assert getattr(config, field) == getattr(default, field), field

    def test_no_flip_augment_survives_the_none_filter(self, tmp_path, monkeypatch):
        """store_const False must not be dropped by build_config's ``is not None`` test."""
        made = _install_trainer(monkeypatch)
        train_pqn.main(
            [
                "--device",
                "cpu",
                "--out-dir",
                str(tmp_path),
                "--total-steps",
                "1",
                "--no-flip-augment",
            ]
        )

        assert made[0].config.flip_augment is False

    def test_no_self_play_collapses_the_pool(self, tmp_path, monkeypatch):
        made = _install_trainer(monkeypatch)
        train_pqn.main(
            ["--device", "cpu", "--out-dir", str(tmp_path), "--total-steps", "1", "--no-self-play"]
        )
        config = made[0].config

        assert config.pool_capacity == 0
        assert config.hero_frac == 1.0

    def test_no_self_play_beats_a_config_file_pool(self, tmp_path, monkeypatch):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"pqn": {"pool_capacity": 10, "hero_frac": 0.8}}))

        made = _install_trainer(monkeypatch)
        train_pqn.main(
            [
                "--device",
                "cpu",
                "--out-dir",
                str(tmp_path),
                "--total-steps",
                "1",
                "--config",
                str(path),
                "--no-self-play",
            ]
        )
        config = made[0].config

        assert config.pool_capacity == 0
        assert config.hero_frac == 1.0


class TestConfigBlockIsValidated:
    """--config routes through PQNSettingsSchema; it is not a second, weaker reader."""

    def test_scientific_notation_is_coerced_to_float(self, tmp_path):
        """YAML 1.1 parses bare ``5e-4`` as a str; PQNConfig would carry it through.

        Written as raw text on purpose — yaml.safe_dump normalizes 5e-4 to 0.0005
        and cannot reproduce this.
        """
        path = tmp_path / "c.yaml"
        path.write_text("pqn:\n  lr: 5e-4\n  max_abs_q_alarm: 1e3\n  eps_end: 2e-2\n")

        config = PQNConfig(**train_pqn._load_config_overrides(str(path)))

        assert isinstance(config.lr, float)
        assert config.lr == pytest.approx(5e-4)
        assert isinstance(config.max_abs_q_alarm, float)
        assert config.max_abs_q_alarm == pytest.approx(1e3)
        assert isinstance(config.eps_end, float)
        assert config.eps_end == pytest.approx(2e-2)

    def test_a_str_threshold_would_break_the_tripwire_compare(self, tmp_path):
        """The coercion above is load-bearing: pqn_trainer compares max|Q| > alarm."""
        path = tmp_path / "c.yaml"
        path.write_text("pqn:\n  max_abs_q_alarm: 1e3\n")

        config = PQNConfig(**train_pqn._load_config_overrides(str(path)))

        assert (1e9 > config.max_abs_q_alarm) is True

    def test_upward_epsilon_schedule_is_rejected(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("pqn:\n  eps_start: 0.1\n  eps_end: 0.5\n")

        with pytest.raises(ValidationError, match="eps_end must not exceed"):
            train_pqn._load_config_overrides(str(path))

    def test_out_of_range_value_is_rejected(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("pqn:\n  hero_frac: 1.5\n")

        with pytest.raises(ValidationError, match="hero_frac"):
            train_pqn._load_config_overrides(str(path))

    def test_typo_under_pqn_is_rejected_not_ignored(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("pqn:\n  rollout_lenn: 24\n")

        with pytest.raises(ValidationError, match="rollout_lenn"):
            train_pqn._load_config_overrides(str(path))

    def test_vector_only_sections_are_still_ignored(self, tmp_path):
        """A full training config must pass without erroring on non-PQN sections."""
        path = tmp_path / "c.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "game": {"mechanics_version": 2, "arena_type": "circular", "num_snakes": 6},
                    "rewards": {"version": 2},
                    "apex": {"batch_size": 512},
                    "network": {"input_size": 61},
                    "pqn": {"num_envs": 32},
                }
            )
        )

        overrides = train_pqn._load_config_overrides(str(path))

        assert overrides["mechanics_version"] == 2
        assert overrides["arena_type"] == "circular"
        assert overrides["num_snakes"] == 6
        assert overrides["reward_version"] == 2
        assert overrides["num_envs"] == 32

    def test_pqn_block_wins_over_the_game_block_for_shared_keys(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"game": {"num_snakes": 6}, "pqn": {"num_snakes": 4}}))

        assert train_pqn._load_config_overrides(str(path))["num_snakes"] == 4

    def test_missing_pqn_block_yields_no_pqn_overrides(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"game": {"mechanics_version": 2}}))

        assert train_pqn._load_config_overrides(str(path)) == {"mechanics_version": 2}


# ---------------------------------------------------------------------------
# Resume (blueprint §4.4: spot/preemptible rentals, "resume under contract v2")
# ---------------------------------------------------------------------------
def _tiny_config(**overrides) -> PQNConfig:
    """A real-but-cheap config; eps_decay_steps is small so ε moves measurably."""
    params = dict(
        num_envs=1,
        num_snakes=2,
        rollout_len=4,
        minibatches=1,
        minibatch_size=8,
        eps_decay_steps=50,
        seed=1234,
    )
    params.update(overrides)
    return PQNConfig(**params)


def _trained_checkpoint(tmp_path, updates=3, **cfg_overrides):
    """Run a real trainer for a few updates and save; return (trainer, path)."""
    trainer = PQNTrainer(_tiny_config(**cfg_overrides))
    trainer.train(updates)
    path = tmp_path / "latest_pqn.pth"
    trainer.save_checkpoint(str(path))
    return trainer, path


class TestResumeRoundTrip:
    """A resumed run continues the SAME run: weights, optimizer, odometers, ε."""

    def test_resume_restores_weights_optimizer_and_odometers(self, tmp_path):
        saved, path = _trained_checkpoint(tmp_path)

        # A different seed so any restored value cannot coincidentally match.
        fresh = PQNTrainer(_tiny_config(seed=999))
        blob = train_pqn.load_pqn_resume_checkpoint(str(path), fresh.cfg)
        train_pqn.apply_resume_checkpoint(fresh, blob)

        assert fresh.update_idx == saved.update_idx
        assert fresh.agent_steps == saved.agent_steps
        assert fresh.agent_steps > 0, "the fixture must actually have trained"

        for (name, want), (_, got) in zip(
            saved.network.state_dict().items(), fresh.network.state_dict().items()
        ):
            assert torch.equal(want, got), name

        want_opt = saved.optimizer.state_dict()["state"]
        got_opt = fresh.optimizer.state_dict()["state"]
        assert got_opt, "Adam moments must be restored, not empty"
        assert set(want_opt) == set(got_opt)
        for pid, want_slot in want_opt.items():
            for key, want_val in want_slot.items():
                got_val = got_opt[pid][key]
                if torch.is_tensor(want_val):
                    assert torch.equal(want_val, got_val), f"param {pid} {key}"
                else:
                    assert want_val == got_val, f"param {pid} {key}"

    def test_resumed_epsilon_does_not_snap_back_to_eps_start(self, tmp_path):
        """The trap: ε is derived from agent_steps, so a weights-only resume would
        re-inject maximum exploration into a trained policy."""
        saved, path = _trained_checkpoint(tmp_path)

        fresh = PQNTrainer(_tiny_config(seed=999))
        assert fresh.epsilon() == fresh.cfg.eps_start, "a fresh trainer starts at eps_start"

        blob = train_pqn.load_pqn_resume_checkpoint(str(path), fresh.cfg)
        train_pqn.apply_resume_checkpoint(fresh, blob)

        assert fresh.epsilon() == pytest.approx(saved.epsilon())
        assert fresh.epsilon() < fresh.cfg.eps_start

    def test_a_real_checkpoint_satisfies_the_resume_contract(self, tmp_path):
        """Guards save<->load drift: whatever checkpoint_state writes must validate."""
        trainer = PQNTrainer(_tiny_config())

        train_pqn.validate_pqn_resume_checkpoint_config(trainer.checkpoint_state(), trainer.cfg)

    def test_resumed_trainer_keeps_training_from_the_restored_step(self, tmp_path):
        saved, path = _trained_checkpoint(tmp_path)

        fresh = PQNTrainer(_tiny_config(seed=999))
        train_pqn.apply_resume_checkpoint(
            fresh, train_pqn.load_pqn_resume_checkpoint(str(path), fresh.cfg)
        )
        before = fresh.agent_steps
        tel = fresh.update()

        assert tel.update == saved.update_idx, "update index continues, not restarts"
        assert tel.agent_steps > before


class TestResumeContract:
    """Refuse to silently continue a run under a different objective."""

    def _blob(self, tmp_path, **cfg_overrides):
        trainer = PQNTrainer(_tiny_config(**cfg_overrides))
        return trainer.checkpoint_state()

    def test_mismatched_gamma_is_rejected(self, tmp_path):
        blob = self._blob(tmp_path, gamma=0.9)

        with pytest.raises(ValueError, match="gamma"):
            train_pqn.validate_pqn_resume_checkpoint_config(blob, _tiny_config(gamma=0.997))

    def test_mismatched_lambda_is_rejected(self, tmp_path):
        blob = self._blob(tmp_path, lambda_=0.5)

        with pytest.raises(ValueError, match="lambda"):
            train_pqn.validate_pqn_resume_checkpoint_config(blob, _tiny_config(lambda_=0.65))

    def test_mismatched_mechanics_version_is_rejected(self, tmp_path):
        blob = self._blob(tmp_path, mechanics_version=1)

        with pytest.raises(ValueError, match="mechanics_version"):
            train_pqn.validate_pqn_resume_checkpoint_config(blob, _tiny_config(mechanics_version=2))

    def test_mismatched_reward_version_is_rejected(self, tmp_path):
        blob = self._blob(tmp_path, reward_version=1)

        with pytest.raises(ValueError, match="reward_version"):
            train_pqn.validate_pqn_resume_checkpoint_config(blob, _tiny_config(reward_version=2))

    def test_a_vector61_apex_checkpoint_is_rejected(self, tmp_path):
        """The legacy champion is not a PQN run; resuming from it must not 'work'."""
        legacy = {
            "dqn_state_dict": {},
            "optimizer_state_dict": {},
            "gamma": PQNConfig().gamma,
            "input_size": 61,
        }
        path = tmp_path / "champion.pth"
        torch.save(legacy, path)

        with pytest.raises(RuntimeError, match="algo"):
            train_pqn.load_pqn_resume_checkpoint(str(path), _tiny_config())

    def test_missing_state_dict_is_rejected(self, tmp_path):
        blob = self._blob(tmp_path)
        del blob["optimizer_state_dict"]
        path = tmp_path / "partial.pth"
        torch.save(blob, path)

        with pytest.raises(RuntimeError, match="optimizer_state_dict"):
            train_pqn.load_pqn_resume_checkpoint(str(path), _tiny_config())

    def test_missing_file_is_reported_before_any_training(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
            train_pqn.load_pqn_resume_checkpoint(str(tmp_path / "nope.pth"), _tiny_config())

    def test_non_dict_payload_is_rejected(self, tmp_path):
        path = tmp_path / "junk.pth"
        torch.save([1, 2, 3], path)

        with pytest.raises(RuntimeError, match="must be a dict"):
            train_pqn.load_pqn_resume_checkpoint(str(path), _tiny_config())

    def test_no_resume_requested_returns_none(self):
        assert train_pqn.load_pqn_resume_checkpoint(None, _tiny_config()) is None


def _resumable_blob(**over):
    """A minimal contract-valid PQN checkpoint (no trainer needed)."""
    state = {
        "dqn_state_dict": {"marker": torch.zeros(1)},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "obs_spec": "raster31v2",
        "algo": "pqn",
        "output_size": 6,
        "gamma": PQNConfig().gamma,
        "lambda": PQNConfig().lambda_,
        "mechanics_version": PQNConfig().mechanics_version,
        "reward_version": PQNConfig().reward_version,
        "update_counter": 42,
        "agent_steps": 700,
    }
    state.update(RASTER31V2_SHAPES.to_metadata())
    state.update(over)
    return state


class TestResumeCLI:
    """--resume plumbing: odometer continues, history appends, budget is a target.

    The blob is hand-built for speed; TestResumeRoundTrip pins it against a real
    checkpoint so the two cannot drift.
    """

    def _run_resume(self, tmp_path, monkeypatch, *, blob=None, total_steps="1000"):
        path = tmp_path / "latest_pqn.pth"
        torch.save(blob if blob is not None else _resumable_blob(), path)
        made = _install_trainer(monkeypatch)
        rc = train_pqn.main(
            [
                "--device",
                "cpu",
                "--out-dir",
                str(tmp_path),
                "--total-steps",
                total_steps,
                "--resume",
                str(path),
            ]
        )
        return rc, made[0]

    def test_resume_does_not_restart_the_odometer_at_zero(self, tmp_path, monkeypatch):
        rc, trainer = self._run_resume(tmp_path, monkeypatch)

        assert rc == 0
        assert trainer.update_idx == 42
        # 700 restored + 3 x 100 to cross the 1000 target — NOT 10 updates from 0.
        assert trainer.updates == 3
        assert trainer.agent_steps == 1000

    def test_resume_loads_weights_and_optimizer_into_the_trainer(self, tmp_path, monkeypatch):
        _, trainer = self._run_resume(tmp_path, monkeypatch)

        assert trainer.network.loaded_state is not None, "weights must be restored"
        assert "marker" in trainer.network.loaded_state
        assert trainer.optimizer.loaded_state is not None, "Adam state must be restored"

    def test_resume_appends_to_history_instead_of_truncating_it(self, tmp_path, monkeypatch):
        history = tmp_path / "history.jsonl"
        history.parent.mkdir(parents=True, exist_ok=True)
        history.write_text(json.dumps({"update": 1, "agent_steps": 100}) + "\n")

        self._run_resume(tmp_path, monkeypatch)

        lines = history.read_text().splitlines()
        assert len(lines) == 4, "the pre-preemption row must survive"
        assert json.loads(lines[0])["update"] == 1
        assert json.loads(lines[-1])["update"] == 3

    def test_a_fresh_run_still_truncates_history(self, tmp_path, monkeypatch):
        """Append mode is resume-only; a fresh run into a reused dir starts clean."""
        history = tmp_path / "history.jsonl"
        history.parent.mkdir(parents=True, exist_ok=True)
        history.write_text(json.dumps({"update": 99, "agent_steps": 9}) + "\n")

        _run(tmp_path, monkeypatch, trip_after=2)

        lines = history.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["update"] == 1

    def test_an_already_complete_budget_runs_no_updates_but_still_checkpoints(
        self, tmp_path, monkeypatch
    ):
        rc, trainer = self._run_resume(tmp_path, monkeypatch, total_steps="700")

        assert rc == 0
        assert trainer.updates == 0
        assert (tmp_path / "latest_pqn.pth").exists()

    def test_a_contract_violating_resume_aborts_before_the_trainer_is_built(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "bad.pth"
        torch.save(_resumable_blob(gamma=0.5), path)
        made = _install_trainer(monkeypatch)

        with pytest.raises(RuntimeError, match="gamma"):
            train_pqn.main(
                [
                    "--device",
                    "cpu",
                    "--out-dir",
                    str(tmp_path),
                    "--total-steps",
                    "1000",
                    "--resume",
                    str(path),
                ]
            )

        assert made == [], "the sim/network must not be built for a doomed resume"

    def test_no_resume_flag_leaves_the_trainer_untouched(self, tmp_path, monkeypatch):
        _, trainer = _run(tmp_path, monkeypatch)

        assert trainer.network.loaded_state is None
        assert trainer.optimizer.loaded_state is None
        assert trainer.update_idx == 0
