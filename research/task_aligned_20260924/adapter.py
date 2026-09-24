"""Explicit CZ learner reuse and native mixed-roster collector.

Importing this module does not import numerical libraries, read artifacts, or run
games. Construction/loading/update functions are numerical operations and require
the caller's new-stage admission. No historical experiment module is executed.
"""

from __future__ import annotations

import ast
import copy
import functools
import hashlib
import inspect
import json
import textwrap
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PARENT_SEEDS = (2026096101, 2026096102, 2026096103)
PARENT_MARK = 4608
PARENT_ROOT = Path(__file__).resolve().parents[2].parent / (
    "snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-enemy-access/training"
)
PARENT_HASHES = {
    2026096101: "35fb390b18567f00766fbb9c09d4ce8da89beb2e1a50a47059b31c1469df09cc",
    2026096102: "abcb6506a0c956f7637c2eb154b1abdb85040cec951c1eba7d78b015f7bc0cf8",
    2026096103: "2c77908e9e94fb759481fa2c2fba0d124e3973f2fa95651c0fc1270acc9487f0",
}
SCHEDULE = ("solo", "s2", "s2", "s6", "s6")
ROSTERS = {"solo": 1, "s2": 2, "s6": 6}
PROFILE_OPPONENTS = {
    "solo": (),
    "s2": ("greedy_food",),
    "s6": ("greedy_food", "greedy_food", "greedy_food", "random_safe", "random_safe"),
}
LIFECYCLE = "task-aligned-native-watch-hero-terminal-mixed-roster/v1"
MAX_ROLLOUTS = 39060
ENVIRONMENTS = 16
ROLLOUT_LENGTH = 16
TRAIN_HORIZON = 1024
OBSERVATION_PROGRESS_HORIZON = 256  # Parent normalization, not episode horizon.
INPUT_CONTRACT = {
    "id": "enemy_access_native_raster31v3/v1",
    "arm": "enemy_visible",
    "obs_spec": "raster31v3",
    "q_scale": 0.1,
    "kept_tactical_channels": [0, 1, 2, 3, 4, 5],
    "kept_strategic_channels": [],
    "kept_scalar_indices": [0, 1, 12, 13, 14],
    "other_tactical_strategic_and_scalars": "zero",
    "raw_input_immutable": True,
    "mask": "external_native_legal_six_unchanged",
    "parameter_count_unchanged": True,
}


@dataclass
class PhysicalCounters:
    """Completed physical work; previews never count as completed native frames."""

    native_frames: int = 0
    hero_transitions: int = 0
    opponent_transitions: int = 0
    model_forwards: int = 0
    model_forward_rows: int = 0
    checkpoint_loads: int = 0
    checkpoint_files_hashed: int = 0
    optimizer_updates: int = 0
    sampled_transition_draws: int = 0
    completed_rollouts: int = 0
    preview_calls: int = 0
    episode_starts: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(vars(self))


@dataclass(frozen=True)
class ParentSpec:
    seed: int
    report_path: Path
    report_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str


@dataclass
class LoadedParent:
    model: Any
    native: dict[str, Any]
    config: dict[str, Any]
    metadata: dict[str, Any]
    counters: PhysicalCounters


def file_sha256(path: Path) -> str:
    """Stream a bound file; callers must account for checkpoint hashing separately."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parent_spec(seed: int, *, root: Path = PARENT_ROOT) -> ParentSpec:
    """Read named completed report metadata only; never open the checkpoint here."""
    if seed not in PARENT_SEEDS:
        raise ValueError("The complete three-parent CZ cohort is fixed")
    directory = root / f"enemy_visible-seed{seed}"
    report = directory / "report.json"
    value = json.loads(report.read_text())
    row = _validate_parent_report(value, seed)
    path = Path(row["path"])
    if path != directory / "checkpoint_4608.pth" or row["sha256"] != PARENT_HASHES[seed]:
        raise ValueError("CZ final checkpoint identity differs from the completed report")
    return ParentSpec(seed, report, file_sha256(report), path, row["sha256"])


def _validate_parent_report(report: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    expected = {
        "schema": "cz-pqn-enemy-access-training/v1",
        "status": "COMPLETE",
        "seed": seed,
        "arm": "enemy_visible",
        "starting_update": 4096,
        "final_update": 4608,
        "updates": 512,
        "promotion_eligible": False,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Completed CZ parent report identity differs")
    provenance = report["provenance"]
    cfg = report["config"]
    if (
        cfg != provenance["config"]
        or provenance["seed"] != seed
        or provenance["arm"] != "enemy_visible"
        or provenance["input_contract"] != INPUT_CONTRACT
        or provenance["loss_objective"] != {"loss": "half_mse", "coefficient": 0.5}
        or [row["mark"] for row in report["snapshots"]] != [4096, 4352, 4608]
    ):
        raise ValueError("CZ parent transform, objective, or lineage differs")
    required = {
        "recipe": "corrected-v3",
        "obs_spec": "raster31v3",
        "num_envs": 16,
        "num_snakes": 2,
        "rollout_len": 16,
        "sgd_epochs": 1,
        "minibatch_size": 256,
        "pad_sgd_batches": True,
        "gamma": 0.997,
        "lambda_": 0.65,
        "lr": 0.0005,
        "adam_eps": 0.00015,
        "eps_start": 0.1,
        "eps_end": 0.1,
        "grad_clip": 10.0,
        "ambient_food_reward_coefficient": 0.1,
        "living_mass_reward_coefficient": 0.0,
        "mechanics_version": 2,
        "reward_version": 2,
        "flip_augment": False,
        "episode_completion_mode": "hero_death_or_frame_cap_s2_v1",
        "decision_phase_mode": "watch_pre_move_v1",
        "max_frames": 256,
        "episode_reset_mode": "per_env_autoreset_v1",
        "episode_seed_mode": "derived_env_episode_v1",
        "pool_capacity": 0,
        "pool_admission_mode": "disabled_v1",
        "hero_frac": 0.0,
    }
    if any(cfg.get(key) != value for key, value in required.items()):
        raise ValueError("CZ inherited numerical recipe differs")
    return report["snapshots"][-1]


def state_digest(value: Any) -> str:
    """CZ's value-based tensor/Adam digest; independent of pickle storage IDs."""
    import torch

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(repr(("tensor", str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict[")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
            digest.update(b"]")
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(repr((type(item).__name__, item)).encode())
        else:
            raise TypeError(f"Unsupported state value: {type(item)}")

    visit(value)
    return digest.hexdigest()


@functools.lru_cache(None)
def model_type() -> type:
    """Exact CZ parameter layout and restriction, with no input-weight surgery."""
    import torch

    from src.model.raster_network import RasterDuelingNetwork

    class CZVisibleNetwork(RasterDuelingNetwork):
        def __init__(self) -> None:
            super().__init__()
            self.arm = "enemy_visible"
            self.obs_spec = "raster31v3"
            self.q_scale = 0.1
            self.input_transform = copy.deepcopy(INPUT_CONTRACT)

        def forward(self, obs: Any, strategic: Any = None, scalars: Any = None) -> Any:
            tactical, strategic, scalars = self._unpack(obs, strategic, scalars)
            if (
                tuple(tactical.shape[1:]) != (9, 31, 31)
                or tuple(strategic.shape[1:]) != (3, 25, 25)
                or tuple(scalars.shape[1:]) != (26,)
            ):
                raise ValueError("CZ raster input geometry differs")
            tac, strat, scal = (torch.zeros_like(value) for value in (tactical, strategic, scalars))
            tac[:, :6] = tactical[:, :6]
            scal[:, [0, 1, 12, 13, 14]] = scalars[:, [0, 1, 12, 13, 14]]
            return super().forward(tac, strat, scal) * 0.1

    return CZVisibleNetwork


def make_model() -> Any:
    import torch

    with torch.random.fork_rng(devices=[]):
        return model_type()()


def _validate_layout(model: Any, native: Mapping[str, Any], mark: int) -> None:
    from src.training.pqn_trainer import PQNTrainer

    named = list(model.named_parameters())
    state = native["dqn_state_dict"]
    optimizer = native["optimizer_state_dict"]
    groups = optimizer["param_groups"]
    if (
        len(named) != 30
        or list(state) != [name for name, _ in named]
        or len(groups) != 1
        or groups[0]["params"] != list(range(30))
    ):
        raise ValueError("CZ parameter or Adam ordering differs")
    for ident, (name, parameter) in enumerate(named):
        row = optimizer["state"][ident]
        if (
            state[name].shape != parameter.shape
            or state[name].dtype != parameter.dtype
            or set(row) != {"step", "exp_avg", "exp_avg_sq"}
            or float(row["step"]) != mark
        ):
            raise ValueError("CZ model or Adam age differs")
        for key in ("exp_avg", "exp_avg_sq"):
            if row[key].shape != parameter.shape or row[key].dtype != parameter.dtype:
                raise ValueError("CZ Adam moment shape differs")
    if PQNTrainer._nonfinite_tensor_count(native["dqn_state_dict"]) or (
        PQNTrainer._nonfinite_tensor_count(optimizer)
    ):
        raise ValueError("CZ parent contains nonfinite learner state")
    if groups[0].get("amsgrad") or groups[0]["lr"] != 0.0005 or groups[0]["eps"] != 0.00015:
        raise ValueError("CZ Adam recipe differs")


def load_parent(spec: ParentSpec, device: str = "cpu") -> LoadedParent:
    """One authenticated checkpoint deserialization; fresh runtime is constructed later."""
    import torch

    from src.core.runtime_contract import RunProvenance, canonical_digest

    counters = PhysicalCounters()
    if spec.seed not in PARENT_SEEDS or spec.checkpoint_sha256 != PARENT_HASHES[spec.seed]:
        raise ValueError("Unregistered CZ parent")
    if file_sha256(spec.report_path) != spec.report_sha256:
        raise ValueError("Frozen parent report changed")
    report = json.loads(spec.report_path.read_text())
    row = _validate_parent_report(report, spec.seed)
    if str(spec.checkpoint_path) != row["path"] or spec.checkpoint_sha256 != row["sha256"]:
        raise ValueError("Parent report/checkpoint binding differs")
    if file_sha256(spec.checkpoint_path) != spec.checkpoint_sha256:
        raise ValueError("Frozen parent checkpoint changed")
    counters.checkpoint_files_hashed += 1
    payload = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=False)
    counters.checkpoint_loads += 1
    native = payload["native_pqn"]
    if (
        payload.get("schema") != "cz-pqn-enemy-access-checkpoint/v1"
        or payload.get("mark") != PARENT_MARK
        or payload.get("q_scale") != 0.1
        or payload.get("promotion_eligible") is not False
        or payload.get("provenance") != report["provenance"]
        or native.get("update_counter") != PARENT_MARK
        or native.get("agent_steps") != row["agent_steps"]
        or native.get("restorable_environment_state") is not False
    ):
        raise ValueError("CZ checkpoint/report crosslink differs")
    provenance = RunProvenance.from_metadata(native)
    contract = native["optimizer_contract"]
    if (
        contract["version"] != "pqn-adam-half-mse-v1"
        or contract["loss"] != {"name": "mse", "coefficient": 0.5, "reduction": "mean"}
        or canonical_digest(contract) != native["optimizer_contract_digest"]
        or provenance.optimizer_digest != native["optimizer_contract_digest"]
    ):
        raise ValueError("CZ optimizer contract differs")
    model = make_model()
    _validate_layout(model, native, PARENT_MARK)
    for state_key, digest_key in (
        ("dqn_state_dict", "network_state_digest"),
        ("optimizer_state_dict", "optimizer_state_digest"),
    ):
        if (
            state_digest(native[state_key]) != row[digest_key]
            or payload[digest_key] != row[digest_key]
        ):
            raise ValueError("CZ state value digest differs")
    model.load_state_dict(native["dqn_state_dict"], strict=True)
    model.to(device).eval()
    return LoadedParent(
        model,
        copy.deepcopy(native),
        dict(report["config"]),
        {"seed": spec.seed, **row, "report_sha256": spec.report_sha256},
        counters,
    )


def _seed(run_seed: int, namespace: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"task-aligned/v1/{run_seed}/{namespace}".encode()).digest()[:8], "big"
    )


def sim_config(
    config: Mapping[str, Any], envs: int, snakes: int, horizon: int = TRAIN_HORIZON
) -> Any:
    from src.simd_env.batch_sim import BatchSimConfig

    fields = (
        "game_width",
        "game_height",
        "segment_size",
        "wall_thickness",
        "initial_food",
        "max_food",
        "min_boost_length",
        "boost_length_cost_frames",
        "mechanics_version",
        "gamma",
        "max_capacity",
        "arena_type",
        "frame_rate",
        "kill_scale",
        "death_value",
    )
    return BatchSimConfig(
        num_envs=envs,
        num_snakes=snakes,
        body_storage_capacity=2 * horizon + 400,
        **{key: config[key] for key in fields},
    )


def make_sim(
    config: Mapping[str, Any], seeds: Sequence[int], snakes: int, horizon: int = TRAIN_HORIZON
) -> Any:
    from src.simd_env.eval_engine import _TerminalHeroBatchSim

    if snakes not in (1, 2, 6):
        raise ValueError("Only explicit S1/S2/S6 physical rosters are admitted")
    return _TerminalHeroBatchSim(
        sim_config(config, len(seeds), snakes, horizon),
        seeds=list(seeds),
        train_mode=False,
        allow_respawn=True,
    )


def hero_observations(sim: Any, config: Mapping[str, Any], device: Any) -> tuple[dict, Any]:
    """Featurize the actual prepared world and select slot zero without changing masks."""
    import torch

    from src.model.raster_network import raster_tensors_from_obs
    from src.simd_env.featurizer import build_observations, obs_inputs_from_batch_sim

    masks = sim.get_resolved_action_mask()
    inputs = obs_inputs_from_batch_sim(
        sim,
        max_frames=OBSERVATION_PROGRESS_HORIZON,
        starvation_max=config["starvation_max"],
        max_length=config["max_length"],
    )
    values = raster_tensors_from_obs(
        build_observations(inputs, mask=masks, obs_spec="raster31v3"), device=device
    )
    return (
        {key: value.reshape(sim.E, sim.S, *value.shape[1:])[:, 0] for key, value in values.items()},
        torch.as_tensor(masks[:, 0], dtype=torch.bool, device=device),
    )


def scripted_actions(sim: Any, kinds: Sequence[str], run_seed: int) -> Any:
    """CZ greedy opponent plus stateless per-world/frame random normal-action policy."""
    import numpy as np

    from src.simd_env.eval_engine import GreedyFoodSimdPolicy

    if len(kinds) != sim.S - 1 or any(kind not in ("greedy_food", "random_safe") for kind in kinds):
        raise ValueError("Opponent roster differs from declared scripted kinds")
    actions = np.ones((sim.E, sim.S), dtype=np.int64)
    masks = sim.get_resolved_action_mask()
    for slot, kind in enumerate(kinds, 1):
        live = np.flatnonzero(sim.get_alive()[:, slot])
        slots = np.column_stack((live, np.full(len(live), slot, dtype=np.int64)))
        if kind == "greedy_food" and len(slots):
            actions[live, slot] = GreedyFoodSimdPolicy().actions(masks[live, slot], sim, slots)
        elif kind == "random_safe":
            for env in live:
                choices = np.flatnonzero(masks[env, slot, :3])
                if len(choices):
                    key = f"scripted/{sim.get_world_seeds()[env]}/{int(sim.frame[env])}/{slot}"
                    actions[env, slot] = choices[_seed(run_seed, key) % len(choices)]
    return actions


class _PreviewReady(Exception):
    pass


def prepared_preview(
    sim: Any, active: Any, config: Mapping[str, Any], device: Any
) -> tuple[dict, Any]:
    """Use a disposable clone; preserve live food/respawn/RNG state exactly."""
    preview = copy.deepcopy(sim)
    captured = []

    def capture(world: Any) -> Any:
        captured.append(hero_observations(world, config, device))
        raise _PreviewReady()

    try:
        preview.step_with_policy(capture, active_env_mask=active)
    except _PreviewReady:
        pass
    if len(captured) != 1:
        raise RuntimeError("Prepared Watch preview did not reach its selector")
    return captured[0]


@dataclass
class Runtime:
    profile: str
    sim: Any
    episode_ids: Any
    finished: Any
    action_rngs: list[Any]
    run_seed: int


def make_runtime(profile: str, config: Mapping[str, Any], run_seed: int) -> Runtime:
    import numpy as np

    seeds = [_seed(run_seed, f"{profile}/world/{env}/0") for env in range(ENVIRONMENTS)]
    return Runtime(
        profile,
        make_sim(config, seeds, ROSTERS[profile]),
        np.zeros(ENVIRONMENTS, dtype=np.int64),
        np.zeros(ENVIRONMENTS, dtype=bool),
        [
            np.random.default_rng(_seed(run_seed, f"{profile}/action/{env}/0"))
            for env in range(ENVIRONMENTS)
        ],
        run_seed,
    )


def collect_rollout(trainer: Any, runtime: Runtime) -> dict[str, Any]:
    """Collect only hero tensors; actual physical roster is always retained separately."""
    import numpy as np
    import torch

    from src.training.pqn_selfplay import HERO_POLICY_ID

    sim, cfg = runtime.sim, trainer.parent_config
    reset = np.flatnonzero(runtime.finished)
    if len(reset):
        runtime.episode_ids[reset] += 1
        seeds = [
            _seed(runtime.run_seed, f"{runtime.profile}/world/{env}/{runtime.episode_ids[env]}")
            for env in reset
        ]
        sim.reset_envs(runtime.finished, seeds=seeds)
        for env in reset:
            runtime.action_rngs[env] = np.random.default_rng(
                _seed(
                    runtime.run_seed, f"{runtime.profile}/action/{env}/{runtime.episode_ids[env]}"
                )
            )
        runtime.finished[reset] = False
        trainer.counters.episode_starts += len(reset)
    rows: list[dict[str, Any]] = []
    for _ in range(ROLLOUT_LENGTH):
        active = ~runtime.finished
        active &= sim.frame < TRAIN_HORIZON
        captured: dict[str, Any] = {}

        def choose(world: Any) -> Any:
            obs, mask = hero_observations(world, cfg, trainer.device)
            with torch.no_grad():
                q = trainer.network(obs["tactical"], obs["strategic"], obs["scalars"])
            masked = torch.where(mask, q, torch.full_like(q, torch.finfo(q.dtype).min))
            greedy = masked.argmax(1).cpu().numpy()
            actions = scripted_actions(world, PROFILE_OPPONENTS[runtime.profile], runtime.run_seed)
            actions[:, 0] = greedy
            for env in np.flatnonzero(active & world.get_alive()[:, 0]):
                rng = runtime.action_rngs[env]
                if rng.random() < 0.1:
                    choices = np.flatnonzero(mask[env].cpu().numpy())
                    if not len(choices):
                        raise RuntimeError("Alive hero has no resolved action")
                    actions[env, 0] = choices[rng.integers(len(choices))]
            captured.update(obs=obs, mask=mask, q=q, actions=actions[:, 0].copy())
            return actions

        sim.step_with_policy(choose, active_env_mask=active)
        trainer.counters.native_frames += int(active.sum())
        events = sim.get_step_events()
        valid = sim.get_transition_valid()[:, 0] & active
        trainer.counters.hero_transitions += int(valid.sum())
        trainer.counters.opponent_transitions += int(
            (sim.get_transition_valid()[:, 1:] & active[:, None]).sum()
        )
        base_reward = sim.get_reward()[:, 0].copy()
        ambient = events["ambient_food_ate"][:, 0].copy()
        overlay = 0.1 * (ambient & valid & sim.get_alive()[:, 0])
        rows.append(
            {
                **captured,
                "valid": valid,
                "done": sim.get_done()[:, 0].copy(),
                "rewards": base_reward + overlay,
                "ambient_food": ambient,
                "base_rewards": base_reward,
                "ambient_rewards": overlay,
                "boost": sim.get_boosted_this_step()[:, 0].copy(),
            }
        )
        runtime.finished |= ~sim.get_alive()[:, 0] | (sim.frame >= TRAIN_HORIZON)
    # Live time limits truncate: prepare their genuine next Watch decision on
    # the clone as well. A terminal hero never supplies a bootstrap transition.
    final_obs, final_mask = prepared_preview(sim, sim.get_alive()[:, 0], cfg, trainer.device)
    trainer.counters.preview_calls += 1
    masks = [row["mask"] for row in rows]
    roll: dict[str, Any] = {
        key: torch.stack([row["obs"][key] for row in rows], dim=0).unsqueeze(2)
        for key in ("tactical", "strategic", "scalars")
    }
    for target, source in (
        ("actions", "actions"),
        ("valid", "valid"),
        ("dones", "done"),
        ("rewards", "rewards"),
        ("boost", "boost"),
    ):
        roll[target] = np.stack([row[source] for row in rows], axis=0)[:, :, None]
    roll.update(
        trapped=np.zeros_like(roll["valid"]),
        hero_q=torch.stack([row["q"] for row in rows]).unsqueeze(2),
        next_mask=torch.stack(masks[1:] + [final_mask]).unsqueeze(2),
        final_obs={key: value.unsqueeze(1) for key, value in final_obs.items()},
        policy_ids=np.full((ENVIRONMENTS, 1), HERO_POLICY_ID, dtype=np.int64),
        profile=runtime.profile,
        epsilon=0.1,
    )
    return roll


def _half_mse_sgd(canonical: Any) -> Callable:
    """Clone the canonical function's globals; do not monkeypatch module state."""
    original = canonical.PQNTrainer._sgd
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    seams = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "F"
        and node.func.attr == "smooth_l1_loss"
    ]
    if len(seams) != 1 or seams[0].keywords or len(seams[0].args) != 2:
        raise ValueError("Canonical SGD loss seam changed; new qualification required")

    class HalfMSE:
        def __getattr__(self, name: str) -> Any:
            return getattr(canonical.F, name)

        def smooth_l1_loss(self, prediction: Any, target: Any) -> Any:
            return 0.5 * canonical.F.mse_loss(prediction, target)

    namespace = dict(original.__globals__, F=HalfMSE())
    return types.FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )


def build_trainer(
    parent: LoadedParent,
    runtime_seed: int,
    device: str = "cpu",
    rollouts: int = MAX_ROLLOUTS,
    *,
    initialize_runtimes: bool = True,
) -> Any:
    """Construct new runtimes and restore one full CZ learner; no game/update is run."""
    import numpy as np
    import torch

    import src.training.pqn_trainer as canonical

    if type(rollouts) is not int or not 0 < rollouts <= MAX_ROLLOUTS or rollouts % 5:
        raise ValueError("Freeze a positive complete-cycle rollout dose at or below 39060")
    if type(runtime_seed) is not int or runtime_seed in PARENT_SEEDS:
        raise ValueError("An explicit fresh runtime seed is required")
    sgd = _half_mse_sgd(canonical)

    class TaskAlignedTrainer(canonical.PQNTrainer):
        # We reuse only the numerical kernel, numeric guards and tripwire helpers.
        # Deliberately never call the stock constructor/collector/checkpoint writer.
        _sgd = sgd

        def __init__(self) -> None:
            self.parent_config = copy.deepcopy(parent.config)
            kernel = dict(parent.config, num_envs=ENVIRONMENTS, num_snakes=1)
            self.cfg = types.SimpleNamespace(**kernel)
            self.device = torch.device(device)
            self.network = parent.model.to(self.device)
            self.network.train()
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0005, eps=0.00015)
            self.optimizer.load_state_dict(copy.deepcopy(parent.native["optimizer_state_dict"]))
            self.update_idx = int(parent.native["update_counter"])
            self.agent_steps = int(parent.native["agent_steps"])
            for key in (
                "action_collapse_streak",
                "action_collapse_evidence_samples",
                "action_collapse_raw_action_mode",
            ):
                setattr(self, "_" + key, parent.native[key])
            self.sgd_rng = np.random.default_rng(_seed(runtime_seed, "optimizer"))
            self.counters = parent.counters
            self.runtime_seed = runtime_seed
            self.planned_rollouts = rollouts
            self.new_rollouts = 0
            self.starting_agent_steps = self.agent_steps
            self._last_sgd_sampling = {}
            self.parent_metadata = copy.deepcopy(parent.metadata)
            self.runtimes: dict[str, Runtime] = {}
            self.per_profile = {name: {"rollouts": 0, "hero_transitions": 0} for name in ROSTERS}
            self.refresh_numeric_recovery_state()
            if (
                state_digest(self.network.state_dict()) != parent.metadata["network_state_digest"]
                or state_digest(self.optimizer.state_dict())
                != parent.metadata["optimizer_state_digest"]
            ):
                raise ValueError("Full CZ learner restore changed model or Adam values")

            def counted(module: Any, inputs: Any, output: Any) -> None:
                self.counters.model_forwards += 1
                self.counters.model_forward_rows += int(output.shape[0])

            self._counter_hook = self.network.register_forward_hook(counted)
            self._optimizer_counter_hook = self.optimizer.register_step_post_hook(
                lambda optimizer, args, kwargs: self._count_optimizer_step()
            )
            if initialize_runtimes:
                self.initialize_runtimes()

        def _count_optimizer_step(self) -> None:
            self.counters.optimizer_updates += 1

        def initialize_runtimes(self) -> None:
            if self.runtimes or self.new_rollouts:
                raise RuntimeError("Fresh runtimes may be initialized exactly once")
            self.runtimes = {
                name: make_runtime(name, self.parent_config, runtime_seed) for name in ROSTERS
            }
            self.counters.episode_starts += ENVIRONMENTS * len(ROSTERS)

        def update(self) -> dict[str, Any]:
            if self.new_rollouts >= self.planned_rollouts:
                raise RuntimeError("Frozen rollout dose complete")
            if set(self.runtimes) != set(ROSTERS):
                raise RuntimeError("Initialize the three fresh physical runtimes before update")
            profile = SCHEDULE[self.new_rollouts % len(SCHEDULE)]
            if self.device.type == "mps":
                torch.mps.synchronize()
            started = time.monotonic()
            roll = collect_rollout(self, self.runtimes[profile])
            if self.device.type == "mps":
                torch.mps.synchronize()
            collected = time.monotonic()
            targets = self._compute_targets(roll)
            loss, grad, mean_q, max_q, entropy = self._sgd(roll, targets)
            if self.device.type == "mps":
                torch.mps.synchronize()
            optimized = time.monotonic()
            sampling = dict(self._last_sgd_sampling)
            n = int(roll["valid"].sum())
            if not 0 < n <= ENVIRONMENTS * ROLLOUT_LENGTH or sampling != {
                "eligible_hero_transitions": n,
                "sampled_transition_draws": n,
                "unique_sampled_transitions": n,
                "optimizer_steps": 1,
            }:
                raise RuntimeError("One exact-coverage native update contract failed")
            self.counters.sampled_transition_draws += n
            self.counters.completed_rollouts += 1
            self.agent_steps += n
            raw = roll["actions"][roll["valid"]]
            histogram = np.bincount(raw, minlength=6)
            probabilities = histogram / histogram.sum()
            raw_entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
            mode = int(histogram.argmax())
            streak, samples = self._update_action_collapse_evidence(0.1, raw_entropy, n, mode)
            tel = types.SimpleNamespace(
                loss=loss,
                grad_norm=grad,
                max_abs_q=max_q,
                update=self.update_idx,
                raw_action_entropy=raw_entropy,
                action_entropy=entropy,
                epsilon=0.1,
                action_collapse_streak=streak,
                action_collapse_evidence_samples=samples,
            )
            self._check_tripwires(tel)
            self.update_idx += 1
            self.new_rollouts += 1
            self.per_profile[profile]["rollouts"] += 1
            self.per_profile[profile]["hero_transitions"] += n
            if self.agent_steps - self.starting_agent_steps > 10_000_000:
                raise RuntimeError("Fresh hero-transition cap exceeded")
            return {
                "profile": profile,
                "update": self.update_idx,
                "loss": loss,
                "grad_norm": grad,
                "mean_abs_q": mean_q,
                "max_abs_q": max_q,
                "new_rollouts": self.new_rollouts,
                "agent_steps": self.agent_steps,
                "collection_seconds": collected - started,
                "target_sgd_seconds": optimized - collected,
                "update_seconds": optimized - started,
                **sampling,
                "counters": self.counters.as_dict(),
            }

        def checkpoint_state(self) -> dict[str, Any]:
            return {
                "schema": "task-aligned-cz-full-state/v1",
                "parent": self.parent_metadata,
                "input_contract": copy.deepcopy(INPUT_CONTRACT),
                "q_scale": 0.1,
                "dqn_state_dict": self._clone_for_recovery(self.network.state_dict()),
                "optimizer_state_dict": self._clone_for_recovery(self.optimizer.state_dict()),
                "update_counter": self.update_idx,
                "agent_steps": self.agent_steps,
                "action_collapse_streak": self._action_collapse_streak,
                "action_collapse_evidence_samples": self._action_collapse_evidence_samples,
                "action_collapse_raw_action_mode": self._action_collapse_raw_action_mode,
                "parent_config": self.parent_config,
                "task_contract": task_contract(),
                "runtime_seed": self.runtime_seed,
                "planned_rollouts": self.planned_rollouts,
                "completed_rollouts": self.new_rollouts,
                "per_profile": self.per_profile,
                "runtime_episode_counts": {
                    name: [int(value) for value in runtime.episode_ids]
                    for name, runtime in self.runtimes.items()
                },
                "counters": self.counters.as_dict(),
                "restorable_environment_state": False,
                "resume_semantics": "full-learner-state-fresh-runtime-not-exact-resume",
                "promotion_eligible": False,
            }

        def save_checkpoint(self, path: str | Path) -> None:
            path = Path(path)
            with path.open("xb") as stream:
                torch.save(self.checkpoint_state(), stream)

        def close(self) -> None:
            self._counter_hook.remove()
            self._optimizer_counter_hook.remove()
            self.runtimes.clear()

        def used_world_seeds(self) -> list[int]:
            """Return every world actually constructed, including unfinished episodes."""
            return [
                _seed(self.runtime_seed, f"{name}/world/{env}/{episode}")
                for name, runtime in self.runtimes.items()
                for env, final_episode in enumerate(runtime.episode_ids)
                for episode in range(int(final_episode) + 1)
            ]

    return TaskAlignedTrainer()


def task_contract() -> dict[str, Any]:
    """Actual-world metadata, deliberately separate from singleton hero tensor layout."""
    return {
        "version": LIFECYCLE,
        "physical_rosters": dict(ROSTERS),
        "scheduled_rollout_cycle": list(SCHEDULE),
        "num_envs_per_runtime": ENVIRONMENTS,
        "rollout_length": ROLLOUT_LENGTH,
        "training_episode_horizon": TRAIN_HORIZON,
        "body_storage_capacity": 2 * TRAIN_HORIZON + 400,
        "observation_progress_horizon": OBSERVATION_PROGRESS_HORIZON,
        "opponents": {key: list(value) for key, value in PROFILE_OPPONENTS.items()},
        "sim_train_mode": False,
        "opponents_respawn": True,
        "hero_respawn": False,
        "reset": "per-env-at-next-selected-runtime-rollout-boundary",
        "death": "terminal-reward-only",
        "live_cap": "prepared-successor-bootstrap",
        "inactive_lanes": "native-active-env-mask",
        "kernel_tensor_slots": 1,
        "loss": {"name": "mse", "coefficient": 0.5, "reduction": "mean"},
        "ambient_food_overlay": 0.1,
        "sgd_epochs": 1,
        "reward_change": False,
        "fresh_environment_and_rng": True,
        "enemy_weight_reset": False,
        "random_safe_rule": "sha256-seed-world-frame-slot-mod-normal-allowed/v1",
        "training_random_safe_seed": "runtime_seed",
        "evaluation_random_safe_seed": 0,
    }


def load_candidate(
    path: Path, expected_sha256: str, device: str = "cpu", *, allow_partial: bool = False
) -> LoadedParent:
    """Load this package's explicit learner state; never route through a stock loader."""
    import torch

    counts = PhysicalCounters()
    if file_sha256(path) != expected_sha256:
        raise ValueError("Candidate checkpoint hash differs")
    counts.checkpoint_files_hashed += 1
    payload = torch.load(path, map_location="cpu", weights_only=False)
    counts.checkpoint_loads += 1
    if (
        payload.get("schema") != "task-aligned-cz-full-state/v1"
        or payload.get("input_contract") != INPUT_CONTRACT
        or payload.get("q_scale") != 0.1
        or payload.get("task_contract") != task_contract()
        or payload.get("restorable_environment_state") is not False
        or payload.get("promotion_eligible") is not False
        or payload["parent"]["seed"] not in PARENT_SEEDS
        or payload["parent"]["sha256"] != PARENT_HASHES[payload["parent"]["seed"]]
        or payload["update_counter"] != PARENT_MARK + payload["completed_rollouts"]
        or (not allow_partial and payload["completed_rollouts"] != payload["planned_rollouts"])
    ):
        raise ValueError("Candidate learner/collector contract differs")
    model = make_model()
    _validate_layout(model, payload, payload["update_counter"])
    model.load_state_dict(payload["dqn_state_dict"], strict=True)
    model.to(device).eval()
    metadata = {
        "seed": payload["parent"]["seed"],
        "sha256": expected_sha256,
        "mark": payload["update_counter"],
        "path": str(path),
        "network_state_digest": state_digest(payload["dqn_state_dict"]),
        "optimizer_state_digest": state_digest(payload["optimizer_state_dict"]),
    }
    return LoadedParent(model, payload, payload["parent_config"], metadata, counts)


def evaluation_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Pure metadata for the exact new serving task, shared by all controller roles."""
    fields = (
        "game_width",
        "game_height",
        "segment_size",
        "wall_thickness",
        "initial_food",
        "max_food",
        "min_boost_length",
        "boost_length_cost_frames",
        "mechanics_version",
        "gamma",
        "max_capacity",
        "arena_type",
        "frame_rate",
        "kill_scale",
        "death_value",
        "starvation_max",
        "max_length",
    )
    return {
        "version": "task-aligned-watch-evaluation/v1",
        "world": {key: config[key] for key in fields},
        "input_contract": copy.deepcopy(INPUT_CONTRACT),
        "decision_phase": "watch_pre_move_v1",
        "observation_progress_horizon": OBSERVATION_PROGRESS_HORIZON,
        "body_storage_capacity": "2*requested_horizon+400",
        "sim_train_mode": False,
        "hero_respawn": False,
        "opponents_respawn": True,
        "mass": "post-transition-logical-length-times-alive",
        "encounter": "pre-action-hero-head-any-live-enemy-body-cell-L1-at-most-16",
        "random_safe": "sha256-seed0-world-frame-slot-mod-normal-allowed/v1",
        "greedy_food": "native-GreedyFoodSimdPolicy",
        "allowed_rosters": [1, 6],
        "inference_batch_size": 32,
        "inference_device": "cpu",
        "inference_dtype": "float32",
    }


def initial_descriptor(sim: Any, world_seed: int, env: int = 0) -> dict[str, Any]:
    """Bind actual initial geometry/food/RNG before any controller decision."""
    snakes = [
        {
            "body": [list(cell) for cell in sim.get_bodies(env, slot)],
            "heading": int(sim.direction[env, slot]),
            "length": int(sim.length[env, slot]),
            "alive": bool(sim.alive[env, slot]),
        }
        for slot in range(sim.S)
    ]
    rng_state = repr(sim._rngs[env]._rng.getstate()).encode()
    return {
        "world_seed": world_seed,
        "hero": snakes[0],
        "opponents": snakes[1:],
        "food": [list(cell) for cell in sim.get_food(env)],
        "rng_sha256": hashlib.sha256(rng_state).hexdigest(),
    }


def run_batch_episodes(
    model: Any,
    config: Mapping[str, Any],
    requests: Sequence[Any],
    *,
    anchor: str | None = None,
    device: str = "cpu",
    meter: PhysicalCounters | None = None,
    decision_sink: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[Any], dict[str, int]]:
    """Independent native worlds with batched CPU inference and shared-forward accounting."""
    import numpy as np
    import torch

    from research.task_aligned_20260924.evaluation import EpisodeTrace, FrameMetrics
    from src.simd_env.eval_engine import GreedyFoodSimdPolicy

    if not requests or len(requests) > 32:
        raise ValueError("Evaluation batch requires 1..32 declared worlds")
    first = requests[0]
    if (
        any(
            (row.profile, row.opponents, row.horizon)
            != (first.profile, first.opponents, first.horizon)
            for row in requests
        )
        or len({row.world_seed for row in requests}) != len(requests)
        or len(first.opponents) not in (0, 5)
        or first.horizon <= 0
    ):
        raise ValueError("Batch worlds must be unique and share one declared profile")
    if anchor not in (None, "greedy_food", "random_safe") or (
        (model is None) != (anchor is not None)
    ):
        raise ValueError("Exactly one learned or scripted hero controller is required")
    sim = make_sim(
        config, [row.world_seed for row in requests], 1 + len(first.opponents), first.horizon
    )
    if meter is not None:
        meter.episode_starts += len(requests)
    descriptors = [initial_descriptor(sim, row.world_seed, env) for env, row in enumerate(requests)]
    frames: list[list[Any]] = [[] for _ in requests]
    keys = (
        "native_frames",
        "hero_transitions",
        "model_forwards",
        "checkpoint_loads",
        "optimizer_updates",
    )
    counts = [{key: 0 for key in keys} for _ in requests]
    shared = {key: 0 for key in keys}
    active = np.ones(len(requests), dtype=bool)
    for _ in range(first.horizon):
        before: dict[str, Any] = {}

        def choose(world: Any) -> Any:
            masks = world.get_resolved_action_mask()
            actions = scripted_actions(world, first.opponents, 0)
            live = np.flatnonzero(active & world.get_alive()[:, 0])
            heads = world.get_heads()[:, 0]
            before["encounter"] = [
                any(
                    abs(int(cell[0]) - int(heads[env, 0])) + abs(int(cell[1]) - int(heads[env, 1]))
                    <= 16
                    for slot in range(1, world.S)
                    if world.get_alive()[env, slot]
                    for cell in world.get_bodies(env, slot)
                )
                for env in range(world.E)
            ]
            if anchor == "greedy_food":
                slots = np.column_stack((live, np.zeros(len(live), dtype=np.int64)))
                actions[live, 0] = GreedyFoodSimdPolicy().actions(masks[live, 0], world, slots)
            elif anchor == "random_safe":
                for env in live:
                    choices = np.flatnonzero(masks[env, 0, :3])
                    if len(choices):
                        key = f"scripted/{requests[env].world_seed}/{int(world.frame[env])}/0"
                        actions[env, 0] = choices[_seed(0, key) % len(choices)]
            else:
                obs, mask = hero_observations(world, config, torch.device(device))
                with torch.inference_mode():
                    q = model(obs["tactical"], obs["strategic"], obs["scalars"])
                shared["model_forwards"] += 1
                if meter is not None:
                    meter.model_forwards += 1
                    meter.model_forward_rows += len(requests)
                if not bool(torch.isfinite(q).all()) or not bool(mask[live].any(dim=1).all()):
                    raise ValueError("Nonfinite batched Q or empty live hero mask")
                actions[:, 0] = (
                    torch.where(mask, q, torch.full_like(q, -torch.inf)).argmax(1).cpu().numpy()
                )
            if decision_sink is not None and model is not None:
                for env in live:
                    decision_sink(
                        {
                            "world_seed": requests[env].world_seed,
                            "frame": int(world.frame[env]),
                            "q": q[env].detach().cpu().numpy().copy(),
                            "mask": masks[env, 0].copy(),
                            "action": int(actions[env, 0]),
                        }
                    )
            if np.any(~masks[live, 0, actions[live, 0]]):
                raise RuntimeError("Batched hero selected outside resolved mask")
            return actions

        sim.step_with_policy(choose, active_env_mask=active)
        events, alive = sim.get_step_events(), sim.get_alive()[:, 0]
        valid = sim.get_transition_valid()[:, 0]
        if meter is not None:
            meter.native_frames += int(active.sum())
            meter.hero_transitions += int((valid & active).sum())
            meter.opponent_transitions += int(sim.get_transition_valid()[active, 1:].sum())
        for env in np.flatnonzero(active):
            if not valid[env]:
                raise RuntimeError("Evaluation executed a non-transition hero frame")
            counts[env]["native_frames"] += 1
            counts[env]["hero_transitions"] += 1
            frames[env].append(
                FrameMetrics(
                    alive=bool(alive[env]),
                    mass=float(sim.get_lengths()[env, 0]) if alive[env] else 0.0,
                    ambient_food=int(events["ambient_food_ate"][env, 0]),
                    boost=bool(sim.get_boosted_this_step()[env, 0]),
                    encounter=bool(before["encounter"][env]),
                )
            )
        active &= alive
        if not bool(active.any()):
            break
    traces = [
        EpisodeTrace(frames[env], descriptors[env], counts[env]) for env in range(len(requests))
    ]
    return traces, shared


def run_episode(
    model: Any,
    config: Mapping[str, Any],
    request: Any,
    *,
    anchor: str | None = None,
    device: str = "cpu",
) -> Any:
    """Single-world facade over the same batched lifecycle implementation."""
    traces, shared = run_batch_episodes(model, config, [request], anchor=anchor, device=device)
    trace = traces[0]
    trace.counters = {key: trace.counters[key] + shared[key] for key in shared}
    return trace


def evaluation_runtime(
    spec: Mapping[str, Any], policy: Mapping[str, Any], device: str = "cpu"
) -> Any:
    """Factory for evaluation.py's CLI; hashes and geometry are checked before gameplay."""
    from research.task_aligned_20260924.evaluation import EvaluationRuntime, canonical_hash

    role = policy["role"]
    initial = {
        key: 0
        for key in (
            "native_frames",
            "hero_transitions",
            "model_forwards",
            "checkpoint_loads",
            "optimizer_updates",
        )
    }
    model = None
    anchor = None
    if device != "cpu":
        raise ValueError("Evaluation uses frozen CPU float32 serving")
    if role == "parent":
        binding = policy.get("parent_binding")
        if binding is None:
            source = parent_spec(policy["lineage"])
        else:
            source = ParentSpec(
                seed=int(binding["seed"]),
                report_path=Path(binding["report_path"]),
                report_sha256=binding["report_sha256"],
                checkpoint_path=Path(binding["checkpoint_path"]),
                checkpoint_sha256=binding["checkpoint_sha256"],
            )
        if source.checkpoint_sha256 != policy["checkpoint_sha256"]:
            raise ValueError("Evaluation parent policy differs")
        loaded = load_parent(source, device)
        model, config = loaded.model, loaded.config
        initial["checkpoint_loads"] = loaded.counters.checkpoint_loads
    elif role == "candidate":
        loaded = load_candidate(
            Path(policy["checkpoint_path"]), policy["checkpoint_sha256"], device
        )
        if loaded.metadata["seed"] != policy["lineage"]:
            raise ValueError("Candidate lineage differs")
        model, config = loaded.model, loaded.config
        initial["checkpoint_loads"] = loaded.counters.checkpoint_loads
    elif role == "anchor":
        anchor = policy["name"]
        config = dict(spec.get("parent_config", spec.get("source_config", {})))
    else:
        raise ValueError("Unknown evaluation controller")
    digest = canonical_hash(evaluation_contract(config))
    if digest != spec["contract_sha256"]:
        raise ValueError("Evaluation world/serving contract differs")
    return EvaluationRuntime(
        run_episode=lambda request: run_episode(
            model, config, request, anchor=anchor, device=device
        ),
        contract_sha256=digest,
        initial_counters=initial,
        run_batch=lambda requests: run_batch_episodes(
            model, config, requests, anchor=anchor, device=device
        ),
    )


def evaluation_factory(spec: Mapping[str, Any], policy: Mapping[str, Any]) -> Any:
    """Two-argument factory retained for evaluation.py's generic CLI."""
    return evaluation_runtime(spec, policy, device="cpu")


def qualification_evaluation_probe(
    source: LoadedParent,
    spec: Mapping[str, Any],
    qualification_seed: int,
    out: Path,
    heartbeat: Any,
) -> dict[str, Any]:
    """Bounded new serving throughput and E32/E1 parity probe; never trains.

    Reconstructs the pristine parent from its already loaded tensor dictionary.
    At most 18,560 new hero frames: 3 learned E32/H128 cells, 6 scripted
    E32/H32 cells, and 128 E32-vs-E1 parity frames. No checkpoint is opened.
    """
    import numpy as np

    from research.task_aligned_20260924.evaluation import (
        EpisodeRequest,
        canonical_hash,
        save_report,
    )

    model = make_model()
    model.load_state_dict(source.native["dqn_state_dict"], strict=True)
    model.to("cpu").eval()
    meter = PhysicalCounters()
    timings = []
    held_worlds = {int(seed) for values in spec["world_seeds"].values() for seed in values}

    def requests_for(profile: str, horizon: int, namespace: str) -> list[Any]:
        seeds = [
            _seed(qualification_seed, f"eval-qualification/{namespace}/{i}") for i in range(32)
        ]
        if len(set(seeds)) != 32 or held_worlds.intersection(seeds):
            raise ValueError("Qualification worlds overlap or collide with held worlds")
        return [
            EpisodeRequest(seed, profile, tuple(spec["profiles"][profile]), horizon)
            for seed in seeds
        ]

    for controller in ("parent", "greedy_food", "random_safe"):
        for profile in ("solo", "food_pressure", "mixed"):
            horizon = 128 if controller == "parent" else 32
            requests = requests_for(profile, horizon, f"throughput/{profile}")
            heartbeat.progress = {
                "phase": "qualification_evaluation_throughput",
                "controller": controller,
                "profile": profile,
            }
            started = time.perf_counter()
            traces, shared = run_batch_episodes(
                model if controller == "parent" else None,
                source.config,
                requests,
                anchor=None if controller == "parent" else controller,
                device="cpu",
                meter=meter,
            )
            elapsed = time.perf_counter() - started
            observed_steps = max(len(trace.frames) for trace in traces)
            if observed_steps <= 0:
                raise RuntimeError("Vacuous evaluation throughput probe")
            # Normalize to physical batch steps actually observed, even if all
            # heroes died before the requested probe horizon.
            timings.append(
                {
                    "controller": controller,
                    "profile": profile,
                    "requested_horizon": horizon,
                    "observed_batch_steps": observed_steps,
                    "seconds": elapsed,
                    "forecast_h3000_seconds": elapsed * 3000 / observed_steps,
                    "model_forwards": shared["model_forwards"],
                }
            )

    heartbeat.progress = {"phase": "qualification_evaluation_batch_parity"}
    requests = requests_for("mixed", 2, "parity")
    batched_decisions: dict[tuple[int, int], dict[str, Any]] = {}

    def remember(row: dict[str, Any]) -> None:
        batched_decisions[(row["world_seed"], row["frame"])] = row

    batched, _ = run_batch_episodes(
        model, source.config, requests, meter=meter, decision_sink=remember
    )
    checked_decisions = 0

    def compare(row: dict[str, Any]) -> None:
        nonlocal checked_decisions
        expected = batched_decisions[(row["world_seed"], row["frame"])]
        if (
            not np.allclose(row["q"], expected["q"], rtol=1e-5, atol=1e-6)
            or not np.array_equal(row["mask"], expected["mask"])
            or row["action"] != expected["action"]
        ):
            raise RuntimeError("E32/E1 prepared Q, mask, or greedy-action parity failed")
        checked_decisions += 1

    for index, request in enumerate(requests):
        singles, _ = run_batch_episodes(
            model, source.config, [request], meter=meter, decision_sink=compare
        )
        if batched[index].initial_descriptor != singles[0].initial_descriptor or list(
            batched[index].frames
        ) != list(singles[0].frames):
            raise RuntimeError(
                f"E32/E1 evaluation trajectory mismatch at qualification world {index}"
            )
    if checked_decisions != len(batched_decisions) or checked_decisions < 32:
        raise RuntimeError("Vacuous or incomplete E32/E1 decision parity")
    heartbeat.progress = {"phase": "qualification_evaluation_serialization"}
    columns = {
        "alive": [True] * 3000,
        "mass": [123.0] * 3000,
        "ambient_food": [0] * 3000,
        "boost": [False] * 3000,
        "encounter": [True] * 3000,
    }
    synthetic = {
        "schema_version": 1,
        "synthetic_serialization_fixture_only": True,
        "cases": [
            {
                "frame_columns": columns,
                "world_seed": i,
                "initial_descriptor": batched[0].initial_descriptor,
            }
            for i in range(32)
        ],
    }
    started = time.perf_counter()
    for cell in range(3):
        save_report(out / f"serialization-fixture-{cell}.json", synthetic)
    serialization_seconds = time.perf_counter() - started
    parent_seconds = sum(
        row["forecast_h3000_seconds"] for row in timings if row["controller"] == "parent"
    )
    anchor_seconds = sum(
        row["forecast_h3000_seconds"] for row in timings if row["controller"] != "parent"
    )
    learned_forecast = 2 * (parent_seconds + serialization_seconds) + 60
    anchors_forecast = 2 * (anchor_seconds + 2 * serialization_seconds) + 60
    result = {
        "schema_version": 1,
        "timings": timings,
        "serialization_three_cells_seconds": serialization_seconds,
        "fixed_process_load_overhead_seconds": 60,
        "batch_parity_decisions": checked_decisions,
        "q_parity_rtol": 1e-5,
        "q_parity_atol": 1e-6,
        "batch_parity_worlds": 32,
        "batch_parity_horizon": 2,
        "batch_parity": True,
        "learned_job_forecast_seconds": learned_forecast,
        "six_anchor_cells_forecast_seconds": anchors_forecast,
        "fits": learned_forecast <= 900 and anchors_forecast <= 900,
        "forecast_caveat": (
            "short fresh probes; fixed 2x headroom is not a long-body runtime guarantee"
        ),
        "contract_sha256": canonical_hash(evaluation_contract(source.config)),
        "physical_counters": meter.as_dict(),
    }
    path = out / "evaluation-throughput.json"
    with path.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {**result, "physical_counters": meter}
