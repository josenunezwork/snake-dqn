"""One fixed fresh-experience child of the completed 6102 learner.

Import is standard-library only. All numerical work and artifact access are
explicit calls owned by a separately admitted stage harness. The qualified
collector/kernel is reused, but its original-CZ checkpoint writer is not.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

STUDY_ID = "task-aligned-continuation-20260924"
SCHEMA = "task-aligned-continuation-full-state/v1"
QUALIFICATION_SCHEMA = "task-aligned-continuation-disposable-state/v1"
SOURCE_SEED = 2026096102
SOURCE_UPDATE = 35613
SOURCE_ROLLOUTS = 31005
SCIENCE_ROLLOUTS = 31005
QUALIFICATION_ROLLOUTS = 10
FINAL_UPDATE = SOURCE_UPDATE + SCIENCE_ROLLOUTS
SOURCE_SHA256 = "8d67915ef1ea5dbb0804ffd8bf9874dedb27c95878bacdd5b5e5ae77682aa46f"
SOURCE_PATH = Path(__file__).resolve().parents[2].parent / (
    "snake-dqn-artifacts/ongoing-research-20260913/"
    "task-aligned-cz-integration-20260924/admission-v2/run/"
    "train_2026096102/output/checkpoint_rollout_31005.pth"
)
RNG_NAMESPACE = "task-aligned-continuation/v1"


def source_binding() -> dict[str, Any]:
    """Fixed identity; does not inspect the source artifact."""
    return {
        "path": str(SOURCE_PATH),
        "sha256": SOURCE_SHA256,
        "seed": SOURCE_SEED,
        "global_update": SOURCE_UPDATE,
        "completed_rollouts": SOURCE_ROLLOUTS,
    }


def effective_runtime_seed(runtime_seed: int, *, qualification: bool = False) -> int:
    """Domain-separate the new runtime and optimizer streams from earlier studies."""
    if type(runtime_seed) is not int or runtime_seed < 0:
        raise ValueError("Declare a nonnegative integer fresh runtime seed")
    role = "qualification" if qualification else "science"
    key = f"{RNG_NAMESPACE}/{role}/runtime/{runtime_seed}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def load_source(device: str = "mps") -> Any:
    """Authenticate and deserialize the one fixed completed candidate, exactly once."""
    from research.task_aligned_20260924 import adapter

    source = adapter.load_candidate(SOURCE_PATH, SOURCE_SHA256, device=device)
    _validate_source(source)
    return source


def _validate_source(source: Any) -> None:
    from research.task_aligned_20260924 import adapter

    native = source.native
    if (
        source.metadata.get("seed") != SOURCE_SEED
        or source.metadata.get("sha256") != SOURCE_SHA256
        or Path(source.metadata.get("path", "")).resolve() != SOURCE_PATH.resolve()
        or source.metadata.get("mark") != SOURCE_UPDATE
        or native.get("schema") != "task-aligned-cz-full-state/v1"
        or native.get("update_counter") != SOURCE_UPDATE
        or native.get("completed_rollouts") != SOURCE_ROLLOUTS
        or native.get("planned_rollouts") != SOURCE_ROLLOUTS
        or native.get("parent", {}).get("seed") != SOURCE_SEED
        or native.get("parent", {}).get("sha256") != adapter.PARENT_HASHES[SOURCE_SEED]
        or native.get("task_contract") != adapter.task_contract()
        or native.get("input_contract") != adapter.INPUT_CONTRACT
        or source.config != native.get("parent_config")
        or native.get("restorable_environment_state") is not False
    ):
        raise ValueError("Continuation must start from the exact completed6102 source")
    if type(native.get("agent_steps")) is not int or native["agent_steps"] <= 0:
        raise ValueError("Missing cumulative source hero clock")
    # Reusing an already-trained LoadedParent would silently fork the wrong state.
    if any(
        getattr(source.counters, name)
        for name in ("hero_transitions", "optimizer_updates", "completed_rollouts")
    ):
        raise ValueError("Source object already consumed by a learner")


def continuation_contract() -> dict[str, Any]:
    """Pure metadata over the qualified physical and numerical contracts."""
    from research.task_aligned_20260924 import adapter

    return {
        "study_id": STUDY_ID,
        "source": source_binding(),
        "science_rollouts": SCIENCE_ROLLOUTS,
        "final_global_update": FINAL_UPDATE,
        "task_contract": adapter.task_contract(),
        "input_contract": copy.deepcopy(adapter.INPUT_CONTRACT),
        "rng_namespace": RNG_NAMESPACE,
        "source_model_reset": False,
        "source_adam_reset": False,
        "enemy_weight_reset": False,
        "historical_rollout_replay": False,
        "exact_runtime_resume": False,
        "checkpoint_selection": "fixed-final-only",
        "promotion_eligible": False,
    }


class ContinuationLearner:
    """Delegate the unchanged collector/update; own child lineage and serialization."""

    def __init__(
        self,
        source: Any,
        runtime_seed: int,
        device: str,
        initialize_runtimes: bool,
        qualification: bool,
    ) -> None:
        from research.task_aligned_20260924 import adapter

        _validate_source(source)
        self.qualification = bool(qualification)
        self.declared_runtime_seed = runtime_seed
        self.effective_runtime_seed = effective_runtime_seed(
            runtime_seed, qualification=qualification
        )
        if self.effective_runtime_seed == source.native["runtime_seed"]:
            raise ValueError("Fresh runtime seed collides with the source runtime")
        self.source_identity = {
            **source_binding(),
            "agent_steps": source.native["agent_steps"],
            "network_state_digest": source.metadata["network_state_digest"],
            "optimizer_state_digest": source.metadata["optimizer_state_digest"],
            "runtime_seed": source.native["runtime_seed"],
        }
        self.original_cz_parent = copy.deepcopy(source.native["parent"])
        self.inherited_physical_counters = copy.deepcopy(source.native["counters"])
        self.source_tripwire = {
            key: source.native[key]
            for key in (
                "action_collapse_streak",
                "action_collapse_evidence_samples",
                "action_collapse_raw_action_mode",
            )
        }
        self.child_lineage = (
            f"{STUDY_ID}/6102/{'qualification' if qualification else 'science'}/{runtime_seed}"
        )
        dose = QUALIFICATION_ROLLOUTS if qualification else SCIENCE_ROLLOUTS
        self._trainer = adapter.build_trainer(
            source,
            self.effective_runtime_seed,
            device,
            rollouts=dose,
            initialize_runtimes=initialize_runtimes,
        )
        self._saved = False
        self._assert_clocks()

    def __getattr__(self, name: str) -> Any:
        # Expose existing network, optimizer, counters, runtimes and target kernel
        # for the stage harness and its bounded restoration qualification.
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._trainer, name)

    def _assert_clocks(self) -> None:
        trainer = self._trainer
        n = trainer.new_rollouts
        if (
            trainer.update_idx != SOURCE_UPDATE + n
            or trainer.agent_steps
            != self.source_identity["agent_steps"] + trainer.counters.hero_transitions
            or trainer.counters.completed_rollouts != n
            or trainer.counters.optimizer_updates != n
            or trainer.counters.sampled_transition_draws != trainer.counters.hero_transitions
        ):
            raise RuntimeError("Continuation cumulative/update/physical clocks disagree")
        if trainer.counters.hero_transitions > n * 256:
            raise RuntimeError("Continuation exceeds fixed fresh-row capacity")

    def update(self) -> dict[str, Any]:
        if self._saved:
            raise RuntimeError("A published endpoint cannot receive further updates")
        row = self._trainer.update()
        self._assert_clocks()
        return {
            **row,
            "child_lineage": self.child_lineage,
            "source_global_update": SOURCE_UPDATE,
            "source_agent_steps": self.source_identity["agent_steps"],
            "additional_agent_steps": self.counters.hero_transitions,
            "qualification_only": self.qualification,
        }

    def restoration_evidence(self) -> dict[str, Any]:
        """Numerical state comparisons; caller budgets this admitted inspection."""
        from research.task_aligned_20260924 import adapter

        if self.new_rollouts != 0:
            raise RuntimeError("Restoration evidence must precede disposable/science updates")
        self._assert_clocks()
        evidence = {
            "source": copy.deepcopy(self.source_identity),
            "model_digest": adapter.state_digest(self.network.state_dict()),
            "optimizer_digest": adapter.state_digest(self.optimizer.state_dict()),
            "update_counter": self.update_idx,
            "agent_steps": self.agent_steps,
            "tripwire": {key: getattr(self._trainer, "_" + key) for key in self.source_tripwire},
            "declared_runtime_seed": self.declared_runtime_seed,
            "effective_runtime_seed": self.effective_runtime_seed,
            "qualification_only": self.qualification,
        }
        if (
            evidence["model_digest"] != self.source_identity["network_state_digest"]
            or evidence["optimizer_digest"] != self.source_identity["optimizer_state_digest"]
            or evidence["tripwire"] != self.source_tripwire
        ):
            raise RuntimeError("Restoration changed model, Adam, or tripwire evidence")
        return evidence

    def _state(self) -> dict[str, Any]:
        self._assert_clocks()
        trainer = self._trainer
        clone = trainer._clone_for_recovery
        return {
            "schema": QUALIFICATION_SCHEMA if self.qualification else SCHEMA,
            "study_id": STUDY_ID,
            "child_lineage": self.child_lineage,
            "qualification_only": self.qualification,
            "source_checkpoint": copy.deepcopy(self.source_identity),
            "original_cz_parent": copy.deepcopy(self.original_cz_parent),
            "inherited_physical_counters": copy.deepcopy(self.inherited_physical_counters),
            "continuation_contract": continuation_contract(),
            "source_config": copy.deepcopy(trainer.parent_config),
            "dqn_state_dict": clone(trainer.network.state_dict()),
            "optimizer_state_dict": clone(trainer.optimizer.state_dict()),
            "update_counter": trainer.update_idx,
            "agent_steps": trainer.agent_steps,
            "additional_agent_steps": trainer.counters.hero_transitions,
            "action_collapse_streak": trainer._action_collapse_streak,
            "action_collapse_evidence_samples": trainer._action_collapse_evidence_samples,
            "action_collapse_raw_action_mode": trainer._action_collapse_raw_action_mode,
            "planned_rollouts": trainer.planned_rollouts,
            "completed_rollouts": trainer.new_rollouts,
            "source_completed_rollouts": SOURCE_ROLLOUTS,
            "cumulative_rollouts_since_cz": SOURCE_ROLLOUTS + trainer.new_rollouts,
            "declared_runtime_seed": self.declared_runtime_seed,
            "effective_runtime_seed": self.effective_runtime_seed,
            "runtime_episode_counts": {
                name: [int(value) for value in runtime.episode_ids]
                for name, runtime in trainer.runtimes.items()
            },
            "per_profile": copy.deepcopy(trainer.per_profile),
            "counters": trainer.counters.as_dict(),
            "restorable_environment_state": False,
            "resume_semantics": "full-learner-state-fresh-runtime-not-exact-resume",
            "historical_rollout_replay": False,
            "promotion_eligible": False,
        }

    def snapshot_state(self) -> dict[str, Any]:
        """Disposable qualification payload; never a science endpoint or selection mark."""
        if not self.qualification:
            raise RuntimeError("Intermediate snapshots are qualification-only")
        return self._state()

    def checkpoint_state(self) -> dict[str, Any]:
        if self.qualification or self.new_rollouts != SCIENCE_ROLLOUTS:
            raise RuntimeError("Science endpoint is final-only at exactly31005 fresh rollouts")
        return self._state()

    def save_checkpoint(self, path: str | Path) -> None:
        """Create the sole final child; never write the source or a partial endpoint."""
        import torch

        if self._saved:
            raise RuntimeError("The final child was already written")
        path = Path(path)
        if path.resolve() == SOURCE_PATH.resolve():
            raise ValueError("The original6102 checkpoint is immutable")
        payload = self.checkpoint_state()
        with path.open("xb") as handle:
            torch.save(payload, handle)
        self._saved = True

    def used_world_seeds(self) -> list[int]:
        return self._trainer.used_world_seeds()

    def close(self) -> None:
        self._trainer.close()


def build_continuation(
    source: Any,
    runtime_seed: int,
    device: str = "mps",
    *,
    initialize_runtimes: bool = True,
    qualification: bool = False,
) -> ContinuationLearner:
    """Restore one fixed source; use31005 science or10 disposable qualification updates."""
    if type(qualification) is not bool:
        raise ValueError("Qualification mode must be explicit boolean")
    return ContinuationLearner(source, runtime_seed, device, initialize_runtimes, qualification)


def _load_state(path: Path, expected_sha256: str, device: str, *, qualification: bool) -> Any:
    import torch
    from research.task_aligned_20260924 import adapter

    counts = adapter.PhysicalCounters()
    if adapter.file_sha256(path) != expected_sha256:
        raise ValueError("Continuation checkpoint hash differs")
    counts.checkpoint_files_hashed += 1
    payload = torch.load(path, map_location="cpu", weights_only=False)
    counts.checkpoint_loads += 1
    source = payload.get("source_checkpoint", {})
    n = payload.get("completed_rollouts")
    expected_dose = QUALIFICATION_ROLLOUTS if qualification else SCIENCE_ROLLOUTS
    if (
        payload.get("schema") != (QUALIFICATION_SCHEMA if qualification else SCHEMA)
        or payload.get("study_id") != STUDY_ID
        or payload.get("qualification_only") is not qualification
        or payload.get("continuation_contract") != continuation_contract()
        or any(source.get(key) != value for key, value in source_binding().items())
        or payload.get("original_cz_parent", {}).get("seed") != SOURCE_SEED
        or payload.get("original_cz_parent", {}).get("sha256") != adapter.PARENT_HASHES[SOURCE_SEED]
        or type(n) is not int
        or not 0 <= n <= expected_dose
        or (not qualification and n != SCIENCE_ROLLOUTS)
        or payload.get("planned_rollouts") != expected_dose
        or payload.get("update_counter") != SOURCE_UPDATE + n
        or payload.get("source_completed_rollouts") != SOURCE_ROLLOUTS
        or payload.get("cumulative_rollouts_since_cz") != SOURCE_ROLLOUTS + n
        or payload.get("effective_runtime_seed")
        != effective_runtime_seed(payload["declared_runtime_seed"], qualification=qualification)
        or payload.get("restorable_environment_state") is not False
        or payload.get("historical_rollout_replay") is not False
        or payload.get("promotion_eligible") is not False
    ):
        raise ValueError("Continuation lineage or fixed dose differs")
    counters = payload["counters"]
    if set(counters) != set(adapter.PhysicalCounters.__dataclass_fields__) or any(
        type(value) is not int or value < 0 for value in counters.values()
    ):
        raise ValueError("Malformed continuation physical counters")
    fresh = payload["additional_agent_steps"]
    if (
        type(fresh) is not int
        or not 0 <= fresh <= n * 256
        or payload["agent_steps"] != source["agent_steps"] + fresh
        or counters["hero_transitions"] != fresh
        or counters["sampled_transition_draws"] != fresh
        or counters["optimizer_updates"] != n
        or counters["completed_rollouts"] != n
    ):
        raise ValueError("Continuation saved clocks disagree")
    model = adapter.make_model()
    adapter._validate_layout(model, payload, SOURCE_UPDATE + n)
    model.load_state_dict(payload["dqn_state_dict"], strict=True)
    model.to(device).eval()
    metadata = {
        "seed": SOURCE_SEED,
        "child_lineage": payload["child_lineage"],
        "sha256": expected_sha256,
        "path": str(path),
        "mark": SOURCE_UPDATE + n,
        "source_checkpoint": source_binding(),
        "qualification_only": qualification,
        "network_state_digest": adapter.state_digest(payload["dqn_state_dict"]),
        "optimizer_state_digest": adapter.state_digest(payload["optimizer_state_dict"]),
    }
    return adapter.LoadedParent(model, payload, payload["source_config"], metadata, counts)


def load_endpoint(path: Path, expected_sha256: str, device: str = "cpu") -> Any:
    """Authenticated final child for frozen serving; rejects all disposable snapshots."""
    return _load_state(Path(path), expected_sha256, device, qualification=False)


def load_qualification_snapshot(path: Path, expected_sha256: str, device: str = "cpu") -> Any:
    """Explicit bounded roundtrip loader, never accepted by science serving."""
    return _load_state(Path(path), expected_sha256, device, qualification=True)
