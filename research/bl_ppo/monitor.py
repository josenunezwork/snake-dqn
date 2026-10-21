"""Isolated E8 ordinary-state drift bank and operational policy probes.

The bank is populated from rows already emitted by the real collector during the
critic-only phase.  It never calls the collector, samples an action, or touches
the training RNG.  For recurrent R rows it retains the *actual* pre-forward GRU
state, rather than silently evaluating a zero-state approximation.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ORDINARY_BANK_SIZE = 10_000
MONITOR_SCHEMA = "bl-ppo-e8-monitor/v1"


def _field(raw: Any, name: str) -> np.ndarray:
    value = raw[name] if isinstance(raw, Mapping) else getattr(raw, name)
    return np.asarray(value)


def _hidden_rows(hidden: Any, count: int) -> np.ndarray | None:
    if hidden is None:
        return None
    value = np.asarray(hidden, dtype=np.float32)
    if value.shape == (1, count, 256):
        return value[0]
    if value.shape == (count, 1, 256):
        return value[:, 0]
    if value.shape == (count, 256):
        return value
    raise ValueError("actual recurrent hidden state must be [1,N,256], [N,1,256], or [N,256]")


class FixedOrdinaryBank:
    """Deterministic reservoir of exactly 10k actual critic-refit observations.

    ``offer`` is intentionally a data-only operation: callers pass the raw
    collector batch and the adapter's pre-forward hidden state.  The source
    declaration is frozen on the first call, so a later policy cannot quietly
    nominate a different state distribution for the drift diagnostic.
    """

    def __init__(self, *, capacity: int = ORDINARY_BANK_SIZE, seed: int = 20261021) -> None:
        if capacity < 1:
            raise ValueError("ordinary-bank capacity must be positive")
        self.capacity, self.seed = int(capacity), int(seed)
        self._rng = np.random.default_rng(seed)
        self.seen = 0
        self._frozen = False
        self._source: dict[str, str] | None = None
        self._rows: list[dict[str, np.ndarray]] = []
        self._sha256: str | None = None

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def sha256(self) -> str:
        if not self._frozen or self._sha256 is None:
            raise RuntimeError("ordinary bank is not frozen")
        return self._sha256

    def offer(
        self,
        raw: Any,
        *,
        hidden_before: Any,
        actor_mode: str,
        actor_sha256: str,
        context_kind: str,
    ) -> None:
        """Reservoir-sample an actual collector batch during critic refit.

        ``actor_mode`` must attest the initial frozen stochastic actor.  R uses
        ``actual_collector_hidden`` and stores its pre-forward state; F uses
        ``feedforward_no_hidden``.  Those explicit labels make an unavailable
        recurrent reconstruction a hard failure instead of a misleading drift
        report.
        """
        if self._frozen:
            raise RuntimeError("cannot add rows after ordinary bank is frozen")
        if actor_mode != "frozen_initial_stochastic" or not actor_sha256:
            raise ValueError("ordinary bank requires declared frozen initial stochastic actor")
        local, global_, scalars = (
            _field(raw, "local"),
            _field(raw, "global_"),
            _field(raw, "scalars"),
        )
        mask, reset = _field(raw, "mask"), _field(raw, "reset")
        frame, length = _field(raw, "frame"), _field(raw, "length")
        count = len(local)
        if not count or any(
            len(x) != count for x in (global_, scalars, mask, reset, frame, length)
        ):
            raise ValueError("ordinary-bank raw fields must be non-empty and aligned")
        hidden = _hidden_rows(hidden_before, count)
        if hidden is None and context_kind != "feedforward_no_hidden":
            raise ValueError("missing recurrent context; refusing zero-state drift measurement")
        if hidden is not None and context_kind != "actual_collector_hidden":
            raise ValueError("recurrent drift bank requires declared actual collector context")
        source = {
            "actor_mode": actor_mode,
            "actor_sha256": actor_sha256,
            "context_kind": context_kind,
        }
        if self._source is None:
            self._source = source
        elif self._source != source:
            raise ValueError("ordinary bank source declaration changed")
        for index in range(count):
            row = {
                "local": np.ascontiguousarray(local[index]).copy(),
                "global": np.ascontiguousarray(global_[index]).copy(),
                "scalars": np.ascontiguousarray(scalars[index]).copy(),
                "mask": np.ascontiguousarray(mask[index], dtype=bool).copy(),
                "reset": np.asarray(bool(reset[index]), dtype=bool),
                "remaining_horizon": np.asarray(
                    (5000 - float(frame[index])) / 5000.0, dtype=np.float32
                ),
                "length": np.asarray(int(length[index]), dtype=np.int32),
            }
            if hidden is not None:
                row["hidden"] = np.ascontiguousarray(hidden[index]).copy()
            self.seen += 1
            if len(self._rows) < self.capacity:
                self._rows.append(row)
            else:
                replace = int(self._rng.integers(0, self.seen))
                if replace < self.capacity:
                    self._rows[replace] = row

    def freeze(self) -> dict[str, Any]:
        if self._frozen:
            return self.metadata()
        if len(self._rows) != self.capacity or self._source is None:
            raise ValueError(f"ordinary bank requires exactly {self.capacity} rows before freeze")
        digest = hashlib.sha256()
        digest.update(
            json.dumps({"schema": MONITOR_SCHEMA, "source": self._source}, sort_keys=True).encode()
        )
        for row in self._rows:
            for key in sorted(row):
                value = np.ascontiguousarray(row[key])
                digest.update(key.encode() + b"\0" + str(value.dtype).encode() + b"\0")
                digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
                digest.update(value.tobytes())
        self._sha256, self._frozen = digest.hexdigest(), True
        return self.metadata()

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": MONITOR_SCHEMA,
            "capacity": self.capacity,
            "seen_actual_refit_rows": self.seen,
            "selected_rows": len(self._rows),
            "frozen": self._frozen,
            "sha256": self._sha256,
            "source": None if self._source is None else dict(self._source),
        }

    def rows(self) -> Sequence[dict[str, np.ndarray]]:
        if not self._frozen:
            raise RuntimeError("ordinary bank must be frozen before evaluation")
        return tuple(self._rows)

    def state_dict(self) -> dict[str, Any]:
        """Bounded recovery payload; callers export it with immutable sidecars."""
        return {
            "metadata": self.metadata(),
            "rows": list(self._rows),
            "rng_state": self._rng.bit_generator.state,
        }


def run_fixed_bank(
    policy: Any,
    bank: FixedOrdinaryBank,
    diagnostics: Any,
    *,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Evaluate policy/B8 drift on frozen actual contexts and update diagnostics.

    The policy stays in its prior train/eval mode.  Masked actions receive a
    finite sentinel so :meth:`E8Diagnostics.observe_fixed_bank` can reject NaNs
    while KL is still computed over the legal action support only.
    """
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = bank.rows()
    was_training = bool(policy.training)
    policy.eval()
    all_policy, all_b8, lengths = [], [], []
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                chunk = rows[start : start + batch_size]

                def tensor(key: str, dtype: Any = None) -> torch.Tensor:
                    return torch.as_tensor(
                        np.stack([r[key] for r in chunk]), device=device, dtype=dtype
                    )

                local, global_, scalars = tensor("local"), tensor("global"), tensor("scalars")
                mask = tensor("mask", torch.bool)
                reset = tensor("reset", torch.bool)
                remaining = tensor("remaining_horizon", torch.float32)
                hidden = None
                if "hidden" in chunk[0]:
                    hidden = tensor("hidden", torch.float32).unsqueeze(0)
                logits, value, _ = policy(
                    local, global_, scalars, hidden, reset, remaining_horizon=remaining
                )
                if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
                    raise FloatingPointError("nonfinite monitor logits/value")
                teacher = policy.teacher(local, global_, scalars) / policy.temperature
                if not torch.isfinite(teacher).all():
                    raise FloatingPointError("nonfinite monitor teacher logits")
                sentinel = torch.finfo(logits.dtype).min / 4
                all_policy.append(logits.masked_fill(~mask, sentinel).cpu().numpy())
                all_b8.append(teacher.masked_fill(~mask, sentinel).cpu().numpy())
                lengths.append(np.asarray([int(r["length"]) for r in chunk], dtype=np.int32))
    finally:
        policy.train(was_training)
    diagnostics.observe_fixed_bank(
        policy_logits=np.concatenate(all_policy),
        b8_logits=np.concatenate(all_b8),
        lengths=np.concatenate(lengths),
    )
    return {"ordinary_bank": bank.metadata(), "evaluated_rows": len(rows)}


@dataclass(frozen=True)
class ProbeRoot:
    """A pre-registered excluded root for an isolated stochastic/greedy probe."""

    seed: int
    roster: Sequence[tuple[str, str]]
    mix: str


def run_paired_mass_probe(
    policy: Any,
    roots: Sequence[ProbeRoot],
    driver_factory: Callable[..., Any],
    *,
    horizon: int = 5000,
    rng_seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run excluded, matched real simulator worlds under greedy and stochastic π.

    Sampling uses an isolated NumPy generator, so these genuine mass returns do
    not advance PyTorch's training RNG or interact with the on-policy collector.
    The returned transition count is a chargeable operational cost, never folded
    into PPO counters.
    """
    import torch

    if not roots or horizon < 1:
        raise ValueError("probe needs roots and a positive horizon")
    rng = np.random.default_rng(rng_seed)
    # Driver construction imports simulator code and may consume process-global
    # RNGs.  Preserve all three training streams around the entire probe, not
    # merely around stochastic action selection.
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = bool(policy.training)
    policy.eval()
    rows = []
    try:
        with torch.no_grad():
            for root in roots:
                values: dict[str, float] = {}
                steps: dict[str, int] = {}
                for mode in ("stochastic", "greedy"):
                    driver, hidden, reset, mass, count = (
                        driver_factory(root.seed, root.roster, horizon),
                        None,
                        True,
                        0.0,
                        0,
                    )
                    done = False
                    while count < horizon:
                        obs = driver.observe_hero()
                        local = torch.as_tensor(obs["local"], device=device)
                        global_ = torch.as_tensor(obs["global"], device=device)
                        scalars = torch.as_tensor(obs["scalars"], device=device)
                        mask = torch.as_tensor(obs["mask"], device=device, dtype=torch.bool)
                        remaining = torch.tensor([(horizon - count) / horizon], device=device)
                        reset_t = torch.tensor([reset], device=device, dtype=torch.bool)
                        logits, value, hidden = policy(
                            local, global_, scalars, hidden, reset_t, remaining_horizon=remaining
                        )
                        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
                            raise FloatingPointError("nonfinite mass-probe logits/value")
                        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 4)
                        if mode == "greedy":
                            action = int(logits.argmax(dim=-1).item())
                        else:
                            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                            action = int(rng.choice(6, p=probabilities))
                        step = driver.step(action)
                        mass += float(step["mass"])
                        count += 1
                        reset = bool(step.get("done", False))
                        if reset:
                            done = True
                            break
                    if not done:
                        raise RuntimeError(
                            "probe driver failed to terminate at its registered horizon"
                        )
                    values[mode], steps[mode] = mass / horizon, count
                rows.append(
                    {
                        "seed": root.seed,
                        "mix": root.mix,
                        "stochastic_mass_integral": values["stochastic"],
                        "greedy_mass_integral": values["greedy"],
                        "stochastic_minus_greedy": values["stochastic"] - values["greedy"],
                        "stochastic_steps": steps["stochastic"],
                        "greedy_steps": steps["greedy"],
                    }
                )
    finally:
        policy.train(was_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    return {
        "schema": MONITOR_SCHEMA,
        "kind": "excluded_actual_stochastic_vs_greedy_mass_probe",
        "roots": len(rows),
        "horizon": horizon,
        "rng_seed": rng_seed,
        "rows": rows,
        "mean_stochastic_minus_greedy": float(
            np.mean([r["stochastic_minus_greedy"] for r in rows])
        ),
        "charged_simulator_transitions": int(
            sum(r["stochastic_steps"] + r["greedy_steps"] for r in rows)
        ),
        "excluded_from_training": True,
    }
