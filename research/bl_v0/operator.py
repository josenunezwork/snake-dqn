"""V0's kernel-only latency-ladder operator.

This module deliberately leaves the frozen T3 selector untouched.  It calls
``ceiling._step`` and temporarily substitutes only its imported
``bank_engine.rollout_all`` dependency.  C1 is therefore the unmodified H300
path; C2 changes only evaluation order and stops only after an H300 survivor.

C3/C4 run a real, serial H64 prefix for every evaluated first action.  Their
H300 decisions are supplied by the C1 cache, which makes them an oracle-latency
measurement rather than a new controller.  The dummy leaf is CPU float32,
128->64->64->6, and has exactly 12,806 parameters.
"""

from __future__ import annotations

import contextlib
import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from research.bl_k1.native import RowStats, configure
from research.bl_served_search import ceiling
from research.bl_served_search.ruler_operator import CARRY_FIELDS, ExactT3Operator
from research.bl_served_search.ruler_timing import STAGES, StageTimer
from research.t3_rollout_labels_20261015.trigger import DenseTrigger

HORIZON = 300
PREFIX = 64
CELL_MODES = ("all", "stop", "prefix64_oracle", "prefix64_oracle_stop")


def _cache_key(fixture: Mapping[str, Any]) -> str:
    """Use the complete root identity, not a fixture label, for C1 oracle reuse."""
    root = fixture.get("root_sha256")
    if not isinstance(root, str) or len(root) != 64:
        raise ValueError("V0 fixture lacks the immutable root sha256")
    return root


def _ordered_actions(legal: Sequence[int], bare: int, q: np.ndarray) -> list[int]:
    """T3-compatible exact-stop order: bare, then Q descending/action ascending."""
    if bare not in legal:
        raise RuntimeError("B8 bare action is not in the searchable advisory mask")
    others = [int(action) for action in legal if int(action) != bare]
    return [int(bare), *sorted(others, key=lambda action: (-float(q[action]), action))]


def _attestation(stats: RowStats, mix: str) -> dict[str, Any]:
    """Report observed branch-hook rows without fabricating scripted coverage."""
    values = stats.as_dict()
    required = mix in ("frozen", "mixed")
    return {
        "required": required,
        "status": (
            "ATTESTED"
            if required
            and int(values["kernel_rows"]) > 0
            and int(values["stock_rows"]) == 0
            and int(values["noncandidate_rows"]) == 0
            and int(values["fallback_rows"]) == 0
            and int(values["replayed_rows"]) == 0
            and int(values["rows"]) == int(values["kernel_rows"])
            else ("NOT_APPLICABLE_SCRIPTED" if not required else "INVALID")
        ),
        **values,
    }


@dataclass
class _BatchTrace:
    q: np.ndarray | None = None
    evaluated: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    prefix_outcomes: dict[int, dict[str, Any]] = field(default_factory=dict)
    leaf_features: list[np.ndarray] = field(default_factory=list)
    leaf_featurization_seconds: float = 0.0
    leaf_forward_seconds: float = 0.0
    dummy_leaf_calls: int = 0
    dummy_leaf_batch_size: int = 0


class _FixedDummyLeaf:
    """Fixed CPU FP32 dummy leaf; parameter count is asserted at construction."""

    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.model = (
            torch.nn.Sequential(
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 6),
            )
            .cpu()
            .float()
            .eval()
        )
        with torch.no_grad():
            # Fixed, non-random weights keep the charged work stable across runs.
            offset = 0
            for parameter in self.model.parameters():
                count = parameter.numel()
                values = torch.arange(offset, offset + count, dtype=torch.float32).reshape_as(
                    parameter
                )
                parameter.copy_(((values % 97.0) - 48.0) / 97.0)
                offset += count
        if sum(parameter.numel() for parameter in self.model.parameters()) != 12_806:
            raise AssertionError("V0 dummy leaf parameter count drifted")

    def __call__(self, rows: np.ndarray) -> None:
        rows = np.asarray(rows, dtype=np.float32)
        self.calls += 1
        self.batch_sizes.append(int(rows.shape[0]))
        with self.torch.no_grad():
            self.model(self.torch.from_numpy(rows))


def _project_leaf(obs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Deterministically select 128 equally spaced float32 values from real bv8 inputs.

    The projection is intentionally simple and fixed: concatenate the real
    ``local``, ``global`` and ``scalars`` arrays in that order, then take 128
    equally spaced inclusive indices.  It is a latency charge, never a model.
    """
    values = np.concatenate(
        [
            np.asarray(obs["local"], dtype=np.float32).reshape(-1),
            np.asarray(obs["global"], dtype=np.float32).reshape(-1),
            np.asarray(obs["scalars"], dtype=np.float32).reshape(-1),
        ]
    )
    if values.size < 128:
        raise RuntimeError("real bv8 leaf observation is too small for V0 projection")
    indices = np.linspace(0, values.size - 1, num=128, dtype=np.int64)
    return values[indices].astype(np.float32, copy=False)


class V0Operator(ExactT3Operator):
    """ExactT3Operator subclass implementing C1--C4 without shared-source edits."""

    def __init__(self, mode: str = "all", *, oracle_cache: dict[str, Any] | None = None):
        if mode not in CELL_MODES:
            raise ValueError(f"unknown V0 cell mode: {mode}")
        super().__init__(native_mode="kernel")
        self.mode = mode
        self.oracle_cache = oracle_cache if oracle_cache is not None else {}
        self._dummy_leaf = _FixedDummyLeaf() if mode.startswith("prefix64") else None
        self._active_timer: StageTimer | None = None

    @contextlib.contextmanager
    def _patched_rollouts(self, fixture: Mapping[str, Any], trace: _BatchTrace) -> Iterator[None]:
        """Scope the alternative evaluation only to a single frozen selector call."""
        from research.t3_rollout_labels_20261015 import bank_engine

        original_rollout = bank_engine.rollout_all
        original_q = bank_engine.prepared_q
        key = _cache_key(fixture)

        def capture_q(world: Any, obs: Mapping[str, np.ndarray]) -> np.ndarray:
            q = np.asarray(original_q(world, obs), dtype=np.float32)
            trace.q = q.copy()
            return q

        def routed(
            world: Any, legal: Sequence[int], depth: int, decision: int, **kwargs: Any
        ) -> dict:
            if trace.q is None:
                raise RuntimeError("V0 rollout started before exact B8 Q was prepared")
            if self.mode == "all":
                outputs = original_rollout(world, legal, depth, decision, **kwargs)
                trace.evaluated = [int(action) for action in legal]
                return outputs
            if self.mode == "stop":
                return self._stopped_h300(
                    original_rollout, world, legal, depth, decision, trace, **kwargs
                )
            return self._prefix_oracle(world, legal, decision, trace, key, **kwargs)

        bank_engine.prepared_q = capture_q
        bank_engine.rollout_all = routed
        try:
            yield
        finally:
            bank_engine.rollout_all = original_rollout
            bank_engine.prepared_q = original_q

    @staticmethod
    def _outcome_with_state(outcome: Mapping[str, Any], state: str) -> dict[str, Any]:
        return {**dict(outcome), "evaluation": state}

    def _stopped_h300(
        self,
        rollout: Callable[..., dict[int, dict[str, Any]]],
        world: Any,
        legal: Sequence[int],
        depth: int,
        decision: int,
        trace: _BatchTrace,
        **kwargs: Any,
    ) -> dict[int, dict[str, Any]]:
        assert trace.q is not None
        # The bare action is the legal masked argmax.  It cannot be inferred from
        # an arbitrary legal ordering, so calculate it from the prepared Q here.
        bare = int(
            max(
                (int(action) for action in legal),
                key=lambda action: (float(trace.q[action]), -action),
            )
        )
        order = _ordered_actions(legal, bare, trace.q)
        outcomes: dict[int, dict[str, Any]] = {}
        for action in order:
            outcome = rollout(world, [action], depth, decision, **kwargs)[int(action)]
            outcomes[int(action)] = self._outcome_with_state(outcome, "EVALUATED")
            trace.evaluated.append(int(action))
            if bool(outcome["survives"]):
                break
        trace.skipped = [int(action) for action in legal if int(action) not in outcomes]
        # Ceiling._choose only needs evaluated outcomes: a full-depth survivor is
        # maximal, and all branches are present in the all-die case.
        return outcomes

    def _prefix_oracle(
        self,
        world: Any,
        legal: Sequence[int],
        decision: int,
        trace: _BatchTrace,
        key: str,
        **_: Any,
    ) -> dict[int, dict[str, Any]]:
        if key not in self.oracle_cache:
            raise RuntimeError("C3/C4 requires cached C1 H300 outcomes for this root")
        assert trace.q is not None
        cached = self.oracle_cache[key]
        oracle = cached["outcomes"]
        cached_q = np.asarray(cached.get("q"), dtype=np.float32)
        if cached_q.shape != trace.q.shape or not np.array_equal(cached_q, trace.q):
            raise RuntimeError("C3/C4 B8 Q differs from C1 cache for the immutable root")
        if set(oracle) != {int(action) for action in legal}:
            raise RuntimeError("C3/C4 legal action set differs from the C1 oracle cache")
        bare = int(
            max(
                (int(action) for action in legal),
                key=lambda action: (float(trace.q[action]), -action),
            )
        )
        order = _ordered_actions(legal, bare, trace.q)
        selected = (
            order if self.mode == "prefix64_oracle" else self._stop_order_from_oracle(order, oracle)
        )
        trace.evaluated = list(selected)
        trace.skipped = [int(action) for action in legal if int(action) not in selected]
        for action in selected:
            prefix, feature = self._prefix(world, int(action), decision)
            trace.prefix_outcomes[int(action)] = prefix
            trace.leaf_features.append(feature)
        if len(trace.leaf_features) != len(legal):
            # The registered charge shape remains all legal actions, even in C4.
            # Padding is a documented fixed zero row, not a fabricated evaluated
            # leaf feature.  The real-prefix records above remain the evidence.
            trace.leaf_features.extend(
                [np.zeros(128, np.float32) for _ in range(len(legal) - len(trace.leaf_features))]
            )
        started = time.perf_counter()
        assert self._dummy_leaf is not None
        calls_before = self._dummy_leaf.calls
        trace.dummy_leaf_batch_size = len(legal)
        if self._active_timer is None:
            self._dummy_leaf(np.stack(trace.leaf_features[: len(legal)]))
        else:
            with self._active_timer.measure("leaf_forward"):
                self._dummy_leaf(np.stack(trace.leaf_features[: len(legal)]))
        trace.leaf_forward_seconds += time.perf_counter() - started
        trace.dummy_leaf_calls += self._dummy_leaf.calls - calls_before
        return {
            int(action): self._outcome_with_state(oracle[int(action)], "EVALUATED")
            for action in selected
        }

    @staticmethod
    def _stop_order_from_oracle(
        order: Sequence[int], oracle: Mapping[int, Mapping[str, Any]]
    ) -> list[int]:
        selected: list[int] = []
        for action in order:
            selected.append(int(action))
            if bool(oracle[int(action)]["survives"]):
                break
        return selected

    def _prefix(self, world: Any, action: int, decision: int) -> tuple[dict[str, Any], np.ndarray]:
        """Run a real exact H64 continuation and featurize its real terminal state."""
        from research.bl_k1.native import branch_scope
        from research.t3_rollout_labels_20261015.bank_engine import choose_and_step, step_prepared
        from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

        # Match rollout_all's scoped kernel routing.  Root selection remains
        # stock; only these continuation rows may contribute kernel counters.
        with branch_scope(world.mix):
            branch = world.clone()
            for policy in [branch.hero, *branch.opponents]:
                for name in CARRY_FIELDS:
                    if hasattr(policy, name):
                        setattr(policy, name, copy.deepcopy(getattr(policy, name)))
            executed = 0
            step_prepared(branch, action)
            executed += 1
            while executed < PREFIX and bool(branch.sim.alive[0, 0]):
                choose_and_step(branch, capture_obs=False)
                executed += 1
        started = time.perf_counter()
        if self._active_timer is None:
            obs = build_ego_raster(
                branch.sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
            )
            feature = _project_leaf(obs)
        else:
            # Nested accounting subtracts this real feature work from the
            # parent ``observation`` timer, preserving exclusive stage totals.
            with self._active_timer.measure("leaf_featurization"):
                obs = build_ego_raster(
                    branch.sim,
                    EgoRasterConfig(version="bv8"),
                    backend="numba",
                    rows=np.array([[0, 0]]),
                )
                feature = _project_leaf(obs)
        # Attribution is accumulated by caller via this small tagged tuple.
        elapsed = time.perf_counter() - started
        # Prefix outcomes are diagnostic and do not participate in H300 choice.
        outcome = {
            "survives": bool(branch.sim.alive[0, 0]) and executed == PREFIX,
            "frames_executed": executed,
            "prefix_horizon": PREFIX,
            "leaf_featurization_seconds": elapsed,
        }
        return outcome, feature

    def batch(self, fixture: dict[str, Any], cell: str | None = None) -> dict[str, Any]:
        """Run one registered cell; the shared instance retains the C1 cache.

        The runner uses C1--C4 labels.  The constructor's lower-case modes are
        retained for direct focused tests and backwards-readable local calls.
        """
        aliases = {
            "C1": "all",
            "C2": "stop",
            "C3": "prefix64_oracle",
            "C4": "prefix64_oracle_stop",
        }
        requested = aliases.get(cell or self.mode, cell or self.mode)
        if requested not in CELL_MODES:
            raise ValueError(f"unknown V0 cell mode: {cell}")
        previous = self.mode
        self.mode = requested
        if self.mode.startswith("prefix64") and self._dummy_leaf is None:
            self._dummy_leaf = _FixedDummyLeaf()
        try:
            return self._batch(fixture)
        finally:
            self.mode = previous

    def _batch(self, fixture: dict[str, Any]) -> dict[str, Any]:
        stats = RowStats()
        trace = _BatchTrace()
        started = time.perf_counter()
        with configure("kernel", stats), StageTimer() as timer:
            timer.seconds["leaf_featurization"] = 0.0
            timer.seconds["leaf_forward"] = 0.0
            self._active_timer = timer
            try:
                with timer.measure("clone_digest"):
                    root = self._restore(fixture)
                with self._patched_rollouts(fixture, trace):
                    result = ceiling._step(
                        root,
                        int(fixture["root_frame"]),
                        DenseTrigger(min_length=600, interval=5),
                        dense=True,
                        horizon=int(fixture["root_frame"]) + HORIZON,
                    )
            finally:
                self._active_timer = None
        if not result.get("fired"):
            raise RuntimeError("V0 ruler root did not fire the unchanged dense trigger")
        if self.mode == "all":
            key = _cache_key(fixture)
            cached = self.oracle_cache.get(key)
            payload = {
                "outcomes": copy.deepcopy(result["outcomes"]),
                "q": trace.q.tolist() if trace.q is not None else None,
            }
            if cached is not None and cached != payload:
                raise RuntimeError("repeated C1 exact outcomes differ for the same immutable root")
            self.oracle_cache[key] = payload
        trace.leaf_featurization_seconds = float(timer.seconds["leaf_featurization"])
        trace.leaf_forward_seconds = float(timer.seconds["leaf_forward"])
        outcomes = result["outcomes"]
        semantic = {
            "fixture_id": fixture["fixture_id"],
            "bare": int(result["bare"]),
            "chosen": int(result["chosen"]),
            "fired": True,
            "outcomes": {str(key): value for key, value in sorted(outcomes.items())},
            "branches_evaluated": trace.evaluated,
            "branches_skipped": [
                {"action": action, "state": "NOT_EVALUATED"} for action in trace.skipped
            ],
            "branch_frames": {
                str(action): int(value.get("frames_executed", 0))
                for action, value in sorted(
                    (trace.prefix_outcomes if trace.prefix_outcomes else outcomes).items()
                )
            },
            "prefix_outcomes": {
                str(key): value for key, value in sorted(trace.prefix_outcomes.items())
            },
            "branches": [
                {
                    "action": int(action),
                    "state": "EVALUATED",
                    "frames_executed": int(
                        (trace.prefix_outcomes if trace.prefix_outcomes else outcomes)[
                            int(action)
                        ].get("frames_executed", 0)
                    ),
                }
                for action in trace.evaluated
            ]
            + [
                {"action": int(action), "state": "NOT_EVALUATED", "frames_executed": 0}
                for action in trace.skipped
            ],
        }
        stages = {key: float(timer.seconds[key]) for key in STAGES}
        stages.update(
            leaf_featurization=float(trace.leaf_featurization_seconds),
            leaf_forward=float(trace.leaf_forward_seconds),
            submit_visible=0.0,
            submit_consumed=0.0,
        )
        native = {"mode": "kernel", **stats.as_dict()}
        dummy = {
            "calls": trace.dummy_leaf_calls,
            "batch_size": (
                len(trace.evaluated)
                if self.mode == "prefix64_oracle"
                else (
                    len(trace.evaluated) + len(trace.skipped)
                    if self.mode == "prefix64_oracle_stop"
                    else 0
                )
            ),
            "parameters": 12_806 if self.mode.startswith("prefix64") else 0,
            "dtype": "float32" if self.mode.startswith("prefix64") else None,
            "device": "cpu" if self.mode.startswith("prefix64") else None,
        }
        if self.mode.startswith("prefix64") and self._dummy_leaf is not None:
            if dummy["calls"] != 1:
                raise RuntimeError("V0 oracle cell must execute exactly one dummy leaf call")
            if (
                not self._dummy_leaf.batch_sizes
                or self._dummy_leaf.batch_sizes[-1] != dummy["batch_size"]
            ):
                raise RuntimeError(
                    "V0 dummy leaf call did not use the registered legal-action batch"
                )
            dummy["torch_intraop_threads"] = self._dummy_leaf.torch.get_num_threads()
            dummy["torch_interop_threads"] = self._dummy_leaf.torch.get_num_interop_threads()
        return {
            "cell": self.mode,
            "semantic": semantic,
            "stage_seconds": stages,
            "leaf_call_wall_seconds": float(trace.leaf_forward_seconds),
            "dummy_leaf": dummy,
            "operator_wall_seconds": time.perf_counter() - started,
            "native_v8": native,
            "execution_attestation": _attestation(stats, str(fixture["mix"])),
        }


def create_operator() -> V0Operator:
    """Default factory is C1; the runner constructs the other registered cells."""
    return V0Operator()
