"""ExIt-B8 deterministic verify-then-expand teacher package."""

from research.t34_exit_b8.engine import (
    ActionOutcome,
    DecisionDiagnostic,
    ExitTeacher,
    ExitTeacherConfig,
    TriggerConfig,
    mix_action,
)
from research.t34_exit_b8.policy import B8Policy, PolicyEvaluation, RolloutPolicy, action_grid

__all__ = [
    "ActionOutcome",
    "B8Policy",
    "DecisionDiagnostic",
    "ExitTeacher",
    "ExitTeacherConfig",
    "PolicyEvaluation",
    "RolloutPolicy",
    "TriggerConfig",
    "action_grid",
    "mix_action",
]
