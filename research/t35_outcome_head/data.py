"""Targets and world-level folds from T34 ExIt v2 world exports.

Two target families per labelled row (one row per hero decision while alive):

* **teacher**: the exported per-action 300-step search outcomes
  (``survive``, ``mean_mass``) for every ``tested`` action. Only searched rows
  carry them.
* **own** (control arm, expert 3 lever f): the hero's own continuation in the
  recorded trajectory, for the executed action only. ``survive`` means the hero
  completed 300 further steps; ``mean_mass`` is the mean hero length over the
  next 300 decisions with 0 after death. Rows whose 300-step window runs past
  the end of a world the hero survived are censored and excluded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HORIZON = 300
N_ACTIONS = 6


@dataclass(frozen=True)
class WorldTargets:
    world_id: str
    decisions: np.ndarray  # (N,) int64
    searched: np.ndarray  # (N,) bool: any action tested
    decisive: np.ndarray  # (N,) bool
    a_star: np.ndarray  # (N,) int64, -1 when absent
    student_action: np.ndarray  # (N,) int64
    executed_action: np.ndarray  # (N,) int64
    legal: np.ndarray  # (N, 6) bool
    q_export: np.ndarray  # (N, 6) float32 (NaN when absent)
    t_tested: np.ndarray  # (N, 6) bool
    t_survive: np.ndarray  # (N, 6) float32
    t_mass: np.ndarray  # (N, 6) float32
    t_alive_steps: np.ndarray  # (N, 6) int64
    own_valid: np.ndarray  # (N,) bool (not censored)
    own_survive: np.ndarray  # (N,) float32
    own_mass: np.ndarray  # (N,) float32

    @property
    def n(self) -> int:
        return int(self.decisions.shape[0])


def discover_worlds(root: str | Path) -> list[Path]:
    """All complete ``world-*.json`` exports below ``root`` in a stable order."""
    paths = sorted(Path(root).rglob("world-*.json"))
    return [p for p in paths if p.with_name(p.name + ".complete").is_file()]


def own_continuation(
    lengths: Sequence[int], frames: int, horizon: int = HORIZON
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(valid, survive, mean_mass)`` of the recorded hero continuation.

    ``lengths[d]`` is the hero length at decision ``d``; decisions are
    contiguous from 0 while the hero is alive. ``D = len(lengths)``. If
    ``D < frames`` the hero died during step ``D - 1``; from decision ``t`` it
    then completed ``D - 1 - t`` steps. If ``D == frames`` it survived the world
    and windows that would run past ``frames`` are censored.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    n = int(lengths.shape[0])
    died = n < int(frames)
    t = np.arange(n)
    if died:
        alive_steps = (n - 1) - t
        valid = np.ones(n, dtype=bool)
    else:
        alive_steps = np.full(n, horizon)
        valid = (int(frames) - t) >= horizon
    survive = (alive_steps >= horizon).astype(np.float32)
    # mean over k = 1..horizon of length at decision t + k (0 after death).
    padded = np.zeros(n + horizon + 1, dtype=np.float64)
    padded[:n] = lengths
    if not died and n:
        padded[n:] = lengths[-1]  # only reached by censored rows
    csum = np.concatenate([[0.0], np.cumsum(padded)])
    mass = (csum[t + 1 + horizon] - csum[t + 1]) / horizon
    return valid, survive, mass.astype(np.float32)


def world_targets(world: Mapping[str, Any]) -> WorldTargets:
    """Extract both target families from one loaded export."""
    labels = list(world["labels"])
    decisions = np.asarray([int(lab["decision"]) for lab in labels], dtype=np.int64)
    if not np.array_equal(decisions, np.arange(len(labels))):
        raise ValueError(f"{world['world_id']}: hero labels are not contiguous decisions")
    n = len(labels)
    t_tested = np.zeros((n, N_ACTIONS), dtype=bool)
    t_survive = np.zeros((n, N_ACTIONS), dtype=np.float32)
    t_mass = np.zeros((n, N_ACTIONS), dtype=np.float32)
    t_alive = np.zeros((n, N_ACTIONS), dtype=np.int64)
    legal = np.ones((n, N_ACTIONS), dtype=bool)
    q = np.full((n, N_ACTIONS), np.nan, dtype=np.float32)
    a_star = np.full(n, -1, dtype=np.int64)
    for i, lab in enumerate(labels):
        tested = [bool(v) for v in lab.get("tested", [False] * N_ACTIONS)]
        for a in range(N_ACTIONS):
            if tested[a]:
                out = lab["outcomes"][a]
                t_tested[i, a] = True
                t_survive[i, a] = float(bool(out["survive"]))
                t_mass[i, a] = float(out["mean_mass"])
                t_alive[i, a] = int(out["alive_steps"])
        if "legal" in lab:
            legal[i] = np.asarray(lab["legal"], dtype=bool)
        if lab.get("q") is not None:
            q[i] = np.asarray(lab["q"], dtype=np.float32)
        if lab.get("a_star") is not None:
            a_star[i] = int(lab["a_star"])
    searched = t_tested.any(axis=1)
    decisive = np.asarray([bool(lab.get("decisive", False)) for lab in labels]) & searched
    if np.any(a_star[decisive] < 0):
        raise ValueError(f"{world['world_id']}: decisive row without a_star")
    valid, own_s, own_m = own_continuation(
        [int(lab["length"]) for lab in labels], int(world["frames"])
    )
    return WorldTargets(
        world_id=str(world["world_id"]),
        decisions=decisions,
        searched=searched,
        decisive=decisive,
        a_star=a_star,
        student_action=np.asarray([int(lab["student_action"]) for lab in labels], np.int64),
        executed_action=np.asarray([int(w[0]) for w in world["actions"][:n]], np.int64),
        legal=legal,
        q_export=q,
        t_tested=t_tested,
        t_survive=t_survive,
        t_mass=t_mass,
        t_alive_steps=t_alive,
        own_valid=valid,
        own_survive=own_s,
        own_mass=own_m,
    )


def assign_folds(world_ids: Sequence[str], has_searched: Sequence[bool], k: int = 5) -> dict:
    """Deterministic world-level folds, balanced on worlds with searched rows.

    Worlds are ordered by sha256(world_id) within each stratum and dealt
    round-robin into ``k`` folds; no row of a world ever crosses folds.
    """

    def key(w: str) -> str:
        return hashlib.sha256(w.encode()).hexdigest()

    folds: dict[str, int] = {}
    for flag in (True, False):
        stratum = sorted((w for w, s in zip(world_ids, has_searched) if bool(s) is flag), key=key)
        for i, w in enumerate(stratum):
            folds[w] = i % k
    return folds
