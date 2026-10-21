"""The exact T3-dense acting rule over bare B8, on the Round 0 world harness.

Rule (frozen; ``research/t3_rollout_labels_20261015/ceiling.py:103-140``, dense mode):

* trigger: T3's own ``DenseTrigger(min_length=600)`` (imported, unchanged): the
  clock starts at the first live decision with L >= 600 and fires every 5th
  episode-local decision thereafter, advancing through later length dips; no Q
  threshold; at most one batch per 5 decisions;
* candidates: every advisory-legal action (boosts included); an empty advisory
  mask means no batch and bare B8's normal-only fallback;
* outcome: each candidate is rolled D = min(300, H - decision) frames on a cloned
  world (cloned RNG, real opponents with their v8 carry, B8 greedy continuation,
  eps 0, no hero veto) through the audited ``ExitTeacher._rollout_merged`` engine;
* selection: maximum alive transitions; ties -> B8's argmax, then highest Q,
  then lowest index. All-die delays DO execute (the T3 rule).

The engine is the T34 merged-batch continuation (``research/t34_exit_b8/engine.py``),
not T3's ``batched_rollout``, because the latter only supports stateless
scripted anchors and Round 0-D includes frozen/mixed rosters. A5-D fidelity
checks the substitution against historical T3 endpoints on 8 census worlds.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from research.t3_rollout_labels_20261015.trigger import DenseTrigger
from research.t34_exit_b8.engine import ExitTeacher
from research.t34_exit_b8.faults import IllegalActionFault

MIN_LENGTH = 600
INTERVAL = 5
DEPTH = 300
RULE_ID = "t3-dense-ceiling/v1 (ceiling.py:103-140, dense)"


def select_action(
    greedy: int, q: np.ndarray, actions: list[int], alive_steps: dict[int, int]
) -> int:
    """T3's choice: longest alive; ties keep B8 argmax, then highest Q, then lowest index."""
    best = max(alive_steps[action] for action in actions)
    tied = [action for action in actions if alive_steps[action] == best]
    if greedy in tied:
        return int(greedy)
    return int(max(tied, key=lambda action: (float(q[action]), -action)))


def label_kind(greedy: int, chosen: int, survives: dict[int, bool]) -> str:
    """Amendment 2 §1 roles: only horizon rescues are decisive; delays weigh ~0."""
    if survives.get(greedy, False):
        return "confirm"
    if survives[chosen]:
        return "decisive"
    return "delay" if chosen != greedy else "no_rescue"


class DenseT3Controller:
    """Stateful per-episode controller; one instance per hero episode."""

    def __init__(self, teacher: ExitTeacher, *, horizon: int, min_length: int = MIN_LENGTH) -> None:
        if horizon < 1 or min_length < 0:
            raise ValueError("invalid dense controller horizon or minimum length")
        self.teacher = teacher
        self.horizon = int(horizon)
        self.min_length = int(min_length)
        self.trigger = DenseTrigger(min_length=self.min_length, interval=INTERVAL)
        self.last_batch = -INTERVAL
        self.base_config = teacher.config
        self.stats = {
            "decisions": 0,
            "trigger_eligible": 0,
            "batches": 0,
            "action_changes": 0,
            "delay_changes": 0,
            "rescue_changes": 0,
            "empty_advisory_triggers": 0,
            "rollout_rows": 0,
            "rollout_row_steps": 0,
        }

    def decide(self, world: Any, decision: int) -> tuple[int, dict[str, Any] | None]:
        """Return the executed hero action at a prepared decision and its label, if searched."""
        sim = world.sim
        evaluation = self.teacher.evaluate(world)
        q = np.asarray(evaluation.q, dtype=np.float32)
        greedy = int(evaluation.greedy_action())
        alive = bool(sim.alive[0, 0])
        length = int(sim.length[0, 0]) if alive else 0
        advisory = np.asarray(sim.get_advisory_action_mask()[0, 0], dtype=bool)
        self.stats["decisions"] += 1
        fires = self.trigger.fires(decision, length, alive)
        self.stats["trigger_eligible"] += int(fires)
        if not fires or decision - self.last_batch < INTERVAL:
            return greedy, None
        if not advisory.any():
            self.stats["empty_advisory_triggers"] += 1
            return greedy, None
        actions = [int(action) for action in np.flatnonzero(advisory)]
        depth = min(DEPTH, self.horizon - int(decision))
        if depth < 1:
            return greedy, None
        self.teacher.config = dataclasses.replace(self.base_config, horizon=depth)
        try:
            outcomes = self.teacher._rollout_merged([world], [actions])[0]
        finally:
            self.teacher.config = self.base_config
        alive_steps = {action: int(outcomes[action].alive_steps) for action in actions}
        survives = {action: bool(outcomes[action].survives) for action in actions}
        chosen = select_action(greedy, q, actions, alive_steps)
        if chosen not in actions or not 0 <= chosen < 6:
            raise IllegalActionFault("dense controller selected a non-advisory action")
        kind = label_kind(greedy, chosen, survives)
        self.last_batch = int(decision)
        self.stats["batches"] += 1
        self.stats["rollout_rows"] += len(actions)
        self.stats["rollout_row_steps"] += sum(int(outcomes[a].row_steps) for a in actions)
        if chosen != greedy:
            self.stats["action_changes"] += 1
            self.stats["rescue_changes"] += int(kind == "decisive")
            self.stats["delay_changes"] += int(kind == "delay")
        label = {
            "decision": int(decision),
            "length": length,
            "student_action": greedy,
            "action": chosen,
            "executed_action": chosen,
            "forbidden": [
                a for a in actions if not survives[a] and survives[chosen] and a != chosen
            ],
            "outcomes": {
                str(action): {
                    "survives": survives[action],
                    "alive_steps": alive_steps[action],
                    "mean_mass": float(outcomes[action].mass) / depth,
                    "horizon": depth,
                    "censored": False,
                    "tested": True,
                }
                for action in actions
            },
            "reason": "t3_dense",
            "expanded": True,
            "label_kind": kind,
            "delay_only": kind == "delay",
            "decisive": kind == "decisive",
            "kl_anchor_only": kind == "confirm",
            "q": [float(value) for value in q],
            "legal": [bool(value) for value in advisory],
            "horizon": depth,
        }
        if kind == "delay":
            label["label_weight"] = 0.0
        return chosen, label
