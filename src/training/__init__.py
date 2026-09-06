"""Lightweight package exports for training modules."""

from importlib import import_module

__all__ = [
    # Base infrastructure
    "BaseReplayBuffer",
    "PrioritizedReplayBuffer",
    "TensorBoardLogger",
    "BaseDQNPolicy",
    "MultiStepBuffer",
    # Ape-X Policy
    "ApexPolicy",
    # Ape-X Learner
    "ApexLearner",
    "ApexLearnerConfig",
    "create_apex_learner",
    # Ape-X Actor (distributed)
    "ApexActor",
    # Ape-X Buffer (distributed replay buffer)
    "BufferProcess",
    "ActorBufferClient",
    "LearnerBufferClient",
    "LocalApexBuffer",
    "SharedPrioritizedBuffer",
    "get_default_capacity",
    # Curriculum learning
    "CurriculumManager",
    "CurriculumPhase",
    # SumTree
    "SumTree",
    # Behavioral probes
    "BehaviorProbes",
]

_LAZY_EXPORTS = {
    "BaseReplayBuffer": ("src.training.base_buffer", "BaseReplayBuffer"),
    "PrioritizedReplayBuffer": ("src.training.replay_buffer", "PrioritizedReplayBuffer"),
    "TensorBoardLogger": ("src.training.tensorboard_logger", "TensorBoardLogger"),
    "BaseDQNPolicy": ("src.training.base_dqn_policy", "BaseDQNPolicy"),
    "MultiStepBuffer": ("src.training.multistep_buffer", "MultiStepBuffer"),
    "ApexPolicy": ("src.training.apex_policy", "ApexPolicy"),
    "ApexLearner": ("src.training.apex_learner", "ApexLearner"),
    "ApexLearnerConfig": ("src.training.apex_learner", "ApexLearnerConfig"),
    "create_apex_learner": ("src.training.apex_learner", "create_apex_learner"),
    "ApexActor": ("src.training.apex_actor", "ApexActor"),
    "BufferProcess": ("src.training.apex_buffer", "BufferProcess"),
    "ActorBufferClient": ("src.training.apex_buffer", "ActorBufferClient"),
    "LearnerBufferClient": ("src.training.apex_buffer", "LearnerBufferClient"),
    "LocalApexBuffer": ("src.training.apex_buffer", "LocalApexBuffer"),
    "SharedPrioritizedBuffer": ("src.training.apex_buffer", "SharedPrioritizedBuffer"),
    "get_default_capacity": ("src.training.apex_buffer", "get_default_capacity"),
    "CurriculumManager": ("src.training.curriculum", "CurriculumManager"),
    "CurriculumPhase": ("src.training.curriculum", "CurriculumPhase"),
    "SumTree": ("src.training.sum_tree", "SumTree"),
    "BehaviorProbes": ("src.training.behavior_probes", "BehaviorProbes"),
}


def __getattr__(name: str) -> object:
    """Load package-level exports only when callers ask for them."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
