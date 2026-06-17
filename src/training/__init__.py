"""Lightweight package exports for training modules."""

from importlib import import_module

__all__ = [
    # Base infrastructure
    "BaseReplayBuffer",
    "PrioritizedReplayBuffer",
    "UniformReplayBuffer",
    "MetricsTracker",
    "TensorBoardLogger",
    "BaseDQNPolicy",
    "MultiStepBuffer",
    "OnlineTrainer",
    # Ape-X Policy
    "ApexPolicy",
    # Ape-X priority utilities
    "compute_td_error",
    "compute_td_error_double_dqn",
    "td_error_to_priority",
    "compute_importance_weights",
    "update_priorities_batch",
    "compute_priority_statistics",
    "BetaScheduler",
    "PriorityStatistics",
    "compute_actor_priorities",
    "compute_learner_priorities",
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
    "create_apex_buffer",
    "get_default_capacity",
    # Curriculum learning
    "CurriculumManager",
    "CurriculumPhase",
    # SumTree and Sequence Buffer
    "SumTree",
    "SequenceReplayBuffer",
]

_LAZY_EXPORTS = {
    "BaseReplayBuffer": ("src.training.base_buffer", "BaseReplayBuffer"),
    "PrioritizedReplayBuffer": ("src.training.replay_buffer", "PrioritizedReplayBuffer"),
    "UniformReplayBuffer": ("src.training.replay_buffer", "UniformReplayBuffer"),
    "MetricsTracker": ("src.training.metrics_tracker", "MetricsTracker"),
    "TensorBoardLogger": ("src.training.tensorboard_logger", "TensorBoardLogger"),
    "BaseDQNPolicy": ("src.training.base_dqn_policy", "BaseDQNPolicy"),
    "MultiStepBuffer": ("src.training.multistep_buffer", "MultiStepBuffer"),
    "OnlineTrainer": ("src.training.online_trainer", "OnlineTrainer"),
    "ApexPolicy": ("src.training.apex_policy", "ApexPolicy"),
    "compute_td_error": ("src.training.apex_priorities", "compute_td_error"),
    "compute_td_error_double_dqn": (
        "src.training.apex_priorities",
        "compute_td_error_double_dqn",
    ),
    "td_error_to_priority": ("src.training.apex_priorities", "td_error_to_priority"),
    "compute_importance_weights": (
        "src.training.apex_priorities",
        "compute_importance_weights",
    ),
    "update_priorities_batch": ("src.training.apex_priorities", "update_priorities_batch"),
    "compute_priority_statistics": (
        "src.training.apex_priorities",
        "compute_priority_statistics",
    ),
    "BetaScheduler": ("src.training.apex_priorities", "BetaScheduler"),
    "PriorityStatistics": ("src.training.apex_priorities", "PriorityStatistics"),
    "compute_actor_priorities": ("src.training.apex_priorities", "compute_actor_priorities"),
    "compute_learner_priorities": ("src.training.apex_priorities", "compute_learner_priorities"),
    "ApexLearner": ("src.training.apex_learner", "ApexLearner"),
    "ApexLearnerConfig": ("src.training.apex_learner", "ApexLearnerConfig"),
    "create_apex_learner": ("src.training.apex_learner", "create_apex_learner"),
    "ApexActor": ("src.training.apex_actor", "ApexActor"),
    "BufferProcess": ("src.training.apex_buffer", "BufferProcess"),
    "ActorBufferClient": ("src.training.apex_buffer", "ActorBufferClient"),
    "LearnerBufferClient": ("src.training.apex_buffer", "LearnerBufferClient"),
    "LocalApexBuffer": ("src.training.apex_buffer", "LocalApexBuffer"),
    "SharedPrioritizedBuffer": ("src.training.apex_buffer", "SharedPrioritizedBuffer"),
    "create_apex_buffer": ("src.training.apex_buffer", "create_apex_buffer"),
    "get_default_capacity": ("src.training.apex_buffer", "get_default_capacity"),
    "CurriculumManager": ("src.training.curriculum", "CurriculumManager"),
    "CurriculumPhase": ("src.training.curriculum", "CurriculumPhase"),
    "SumTree": ("src.training.sum_tree", "SumTree"),
    "SequenceReplayBuffer": ("src.training.sequence_buffer", "SequenceReplayBuffer"),
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
