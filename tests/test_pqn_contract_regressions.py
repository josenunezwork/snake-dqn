"""Public-path regression coverage for P1 PQN provenance and continuation rules."""

from __future__ import annotations

import copy
from dataclasses import fields

import numpy as np
import pytest
import torch

from src.core.device_manager import DeviceManager
from src.core.runtime_contract import RunProvenance, canonical_digest
from src.model.obs_spec import RASTER31V3
from src.scripts import train_pqn
from src.training.pqn_trainer import PQNConfig, PQNTrainer


def _config(**overrides) -> PQNConfig:
    values = {"num_envs": 1, "num_snakes": 1, "rollout_len": 1, "minibatches": 1}
    values.update(overrides)
    return PQNConfig(**values)


def _checkpoint(config: PQNConfig) -> dict:
    return PQNTrainer(config).checkpoint_state()


def test_legacy_and_corrected_v3_checkpoints_record_their_distinct_mask_resolution():
    legacy = _checkpoint(_config())
    corrected = _checkpoint(_config(recipe="corrected-v3", obs_spec=RASTER31V3, flip_augment=False))

    assert legacy["action_mask_contract"]["resolution"] == "advisory_only"
    assert corrected["action_mask_contract"]["resolution"] == "row_local_intersection_else_legal"
    assert legacy["action_mask_contract_digest"] != corrected["action_mask_contract_digest"]


@pytest.mark.parametrize(
    ("descriptor", "mutation"),
    [
        ("target_contract", ("trapped_bootstrap", -999.0)),
        ("target_contract", ("validity", "all_rows")),
        ("sampler_contract", ("flip_augment", False)),
        ("sampler_contract", ("sgd_seed", 191)),
        ("reward_contract", ("terminal_potential", 1.0)),
        ("optimizer_contract", ("td_target_clip", 5.0)),
    ],
)
def test_continuation_rejects_tampered_descriptor_before_state_installation(
    tmp_path, descriptor, mutation
):
    config = _config()
    payload = _checkpoint(config)
    payload[descriptor] = copy.deepcopy(payload[descriptor])
    payload[descriptor][mutation[0]] = mutation[1]
    _resign(payload, descriptor)
    path = tmp_path / "tampered.pth"
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match=descriptor):
        train_pqn.load_pqn_resume_checkpoint(str(path), config, mode="continuation")


def test_continuation_rejects_optimizer_beta_and_weight_decay_changes_before_installation(tmp_path):
    config = _config()
    payload = _checkpoint(config)
    payload["optimizer_state_dict"] = copy.deepcopy(payload["optimizer_state_dict"])
    group = payload["optimizer_state_dict"]["param_groups"][0]
    group["betas"] = (0.5, 0.5)
    group["weight_decay"] = 0.25
    path = tmp_path / "optimizer-semantics.pth"
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="optimizer param_group"):
        train_pqn.load_pqn_resume_checkpoint(str(path), config, mode="continuation")


def test_exact_mode_is_rejected_and_weights_only_resets_optimizers_and_odometers(tmp_path):
    config = _config()
    source = PQNTrainer(config)
    payload = source.checkpoint_state()
    path = tmp_path / "weights.pth"
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="exact PQN resume is unsupported"):
        train_pqn.load_pqn_resume_checkpoint(None, config, mode="exact")

    loaded = train_pqn.load_pqn_resume_checkpoint(str(path), config, mode="weights-only")
    target = PQNTrainer(config)
    target.agent_steps = 123
    target.update_idx = 9
    before = copy.deepcopy(target.network.state_dict())
    with pytest.raises(ValueError, match="fresh trainer"):
        train_pqn.apply_resume_checkpoint(target, loaded, mode="weights-only")
    assert all(
        torch.equal(value, target.network.state_dict()[key]) for key, value in before.items()
    )
    target = PQNTrainer(config)
    train_pqn.apply_resume_checkpoint(target, loaded, mode="weights-only")
    assert target._resume_mode == "weights-only"
    assert target._resume_state["environment"] == "fresh"
    assert target.agent_steps == target.update_idx == target._action_collapse_streak == 0
    assert not target.optimizer.state


def _resign(payload, descriptor):
    """Keep hashes internally consistent: expected semantics must reject the forgery."""
    digest = canonical_digest(payload[descriptor])
    payload[descriptor + "_digest"] = digest
    key = descriptor.removesuffix("_contract") + "_digest"
    if key in payload["run_provenance"]:
        payload["run_provenance"][key] = digest
    payload.update(RunProvenance(**payload["run_provenance"]).to_metadata())


@pytest.mark.parametrize("component", ["target", "sampler", "optimizer", "reward"])
def test_provenance_links_to_actual_descriptor(tmp_path, component):
    config = _config()
    payload = _checkpoint(config)
    assert (
        payload["run_provenance"][component + "_digest"] == payload[component + "_contract_digest"]
    )
    payload["run_provenance"][component + "_digest"] = "0" * 64
    payload.update(RunProvenance(**payload["run_provenance"]).to_metadata())
    path = tmp_path / "crosslink.pth"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="run_provenance"):
        train_pqn.load_pqn_resume_checkpoint(str(path), config)


@pytest.mark.parametrize("mutation", ["state", "contract", "both"])
def test_actual_optimizer_and_declared_optimizer_each_match_recipe(tmp_path, mutation):
    config = _config()
    payload = _checkpoint(config)
    if mutation in {"state", "both"}:
        payload["optimizer_state_dict"]["param_groups"][0]["maximize"] = True
    if mutation in {"contract", "both"}:
        payload["optimizer_contract"]["param_groups"][0]["maximize"] = True
        _resign(payload, "optimizer_contract")
    path = tmp_path / "optimizer.pth"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="optimizer"):
        train_pqn.load_pqn_resume_checkpoint(str(path), config)


def test_missing_sampler_fact_cannot_be_resigned_as_valid(tmp_path):
    config = _config()
    payload = _checkpoint(config)
    del payload["sampler_contract"]["sgd_seed"]
    _resign(payload, "sampler_contract")
    path = tmp_path / "incomplete.pth"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="sampler_contract"):
        train_pqn.load_pqn_resume_checkpoint(str(path), config)


class _CapturedConstruction(Exception):
    pass


def _capture_cli(monkeypatch, argv):
    captured = {}

    def intercept(config, device=None):
        captured.update(config=config, device=device)
        raise _CapturedConstruction

    monkeypatch.setattr(train_pqn, "PQNTrainer", intercept)
    try:
        with pytest.raises(_CapturedConstruction):
            train_pqn.main(argv)
    finally:
        DeviceManager.reset_for_testing()
    return captured


@pytest.mark.parametrize(
    "cli,env,yaml_device,source",
    [
        (None, "cpu", "mps", "env"),
        ("cpu", "mps", "mps", "cli"),
        (None, None, "cpu", "config"),
    ],
)
def test_device_execution_and_metadata_use_one_yaml_read(
    monkeypatch, tmp_path, cli, env, yaml_device, source
):
    path = tmp_path / "device.yaml"
    path.write_text(f"hardware:\n  device: {yaml_device}\n")
    if env is None:
        monkeypatch.delenv("SNAKE_DQN_DEVICE", raising=False)
    else:
        monkeypatch.setenv("SNAKE_DQN_DEVICE", env)
    loads = []
    real_load = train_pqn.load_config

    def tracked_load(*args, **kwargs):
        loads.append(args)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(train_pqn, "load_config", tracked_load)
    args = ["--config", str(path)] + (["--device", cli] if cli else [])
    captured = _capture_cli(monkeypatch, args)
    assert len(loads) == 1
    cfg = captured["config"]
    assert str(captured["device"]) == cfg.effective_device == cfg.requested_device == "cpu"
    assert cfg.field_sources["requested_device"] == source


def test_complete_field_sources_and_recipe_config_cli_precedence(monkeypatch, tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text("pqn:\n  recipe: corrected-v3\n  gamma: 0.9\n")
    cfg = _capture_cli(
        monkeypatch, ["--config", str(path), "--device", "cpu", "--no-self-play", "--seed", "0"]
    )["config"]
    assert set(cfg.field_sources) == {field.name for field in fields(PQNConfig)} - {"field_sources"}
    assert cfg.field_sources["gamma"] == "config"
    assert cfg.field_sources["obs_spec"] == cfg.field_sources["flip_augment"] == "recipe"
    assert cfg.field_sources["hero_frac"] == cfg.field_sources["pool_capacity"] == "cli"
    assert cfg.field_sources["lr"] == "default"
    assert cfg.field_sources["seed"] == "cli" and cfg.seed == 0


def test_omitted_cli_seed_requests_entropy_once(monkeypatch):
    requests = []
    real_initialize = train_pqn.initialize_run_seed

    def tracked_initialize(seed):
        requests.append(seed)
        return real_initialize(7341 if seed is None else seed)

    monkeypatch.setattr(train_pqn, "initialize_run_seed", tracked_initialize)
    cfg = _capture_cli(monkeypatch, ["--device", "cpu"])["config"]
    assert requests == [None]
    assert cfg.seed == 7341 and cfg.field_sources["seed"] == "entropy"


@pytest.mark.parametrize("args", [["--mechanics-version", "3"], ["--arena-type", "circular"]])
def test_unsupported_world_rejected_before_model(monkeypatch, args):
    def forbidden(*args, **kwargs):
        pytest.fail("network must not be constructed for an unsupported world")

    monkeypatch.setattr(train_pqn, "PQNTrainer", forbidden)
    with pytest.raises(ValueError):
        train_pqn.main(args + ["--device", "cpu"])


@pytest.mark.parametrize("corrected", [False, True])
def test_runtime_advisory_empty_mask_matches_declared_recipe(monkeypatch, corrected):
    config = _config(
        **(
            {"recipe": "corrected-v3", "obs_spec": RASTER31V3, "flip_augment": False}
            if corrected
            else {}
        )
    )
    trainer = PQNTrainer(config)
    advisory = np.zeros((1, 1, 6), dtype=bool)
    legal = np.array([[[True, True, True, False, False, False]]])
    monkeypatch.setattr(trainer.sim, "get_action_mask", lambda: advisory)
    monkeypatch.setattr(trainer.sim, "get_resolved_action_mask", lambda: legal)
    _, mask = trainer._current_obs()
    assert np.array_equal(mask.cpu().numpy(), legal if corrected else advisory)


@pytest.mark.parametrize("mechanics,snakes,applies", [(1, 6, False), (2, 2, False), (2, 6, True)])
def test_floor_metadata_matches_simulator_applicability(mechanics, snakes, applies):
    trainer = PQNTrainer(_config(mechanics_version=mechanics, num_snakes=snakes))
    checkpoint = trainer.checkpoint_state()
    assert checkpoint["target_contract"]["population_floor"] is applies
    assert checkpoint["runtime_contract"]["population_floor"] is applies
    trainer.sim.alive[:] = False
    assert bool(trainer.sim.population_floor_reached().all()) is applies


@pytest.mark.parametrize("mutation", ["pool", "rng"])
def test_weights_only_rejects_nonfresh_pool_and_rng_before_mutation(mutation):
    trainer = PQNTrainer(_config())
    payload = trainer.checkpoint_state()
    if mutation == "pool":
        trainer.pool.add_snapshot(trainer.network)
    else:
        trainer.rng.random()
    with pytest.raises(ValueError, match="fresh trainer"):
        train_pqn.apply_resume_checkpoint(trainer, payload, mode="weights-only")


@pytest.mark.parametrize("kwargs", [{"mechanics_version": 1}, {"flip_augment": True}])
def test_direct_corrected_config_cannot_mislabel_conflicting_recipe(kwargs):
    values = {"recipe": "corrected-v3", "obs_spec": RASTER31V3, "flip_augment": False}
    values.update(kwargs)
    with pytest.raises(ValueError, match="corrected-v3"):
        _config(**values)
