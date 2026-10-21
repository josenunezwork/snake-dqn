"""Held-out development probe for the ratified fit's epoch selection/tripwires.

``probe_factory`` returns ``hook(student_model, epoch) -> ProbeResult`` exactly
as ``research.t34_exit_b8_fit.diagnostics`` expects: fresh development worlds
(never gate data), each played to completion (death or H) by the current
student with the frozen v8 opponents, recording at every decision the student's
action, the *previous* student's action on the same observation, and the hero
length.  ``survival`` is the fraction of worlds alive at H.

Determinism / parallelism.  Every world is played independently on CPU with
batch-one fp32 forwards of CPU copies of both models, under its own Python and
NumPy global RNG streams seeded from the world seed.  A world's trace therefore
does not depend on which other worlds share its process, so ``workers=K``
(spawned processes, round-robin world shards) returns exactly the serial result
(tested).  The probe is host-bound (simulator + v8 opponents), so CPU is not a
loss on the L4; the fit itself still runs on ``--device``.

Production shape is 32 worlds x H5000 (validated by the ratified
``validate_probe``).  Smaller shapes exist only for local smoke wiring and are
stamped as such by :mod:`research.t34_claim_a.pipeline`.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import multiprocessing as mp
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from research.t34_exit_b8_fit.diagnostics import ProbeResult

PROBE_FACTORY = "research.t34_claim_a.probe:probe_factory"
PRODUCTION_WORLDS = 32
PRODUCTION_HORIZON = 5000
PROGRESS_SECONDS = 110.0
_LAST: list["DevProbe"] = []
# The pipeline registers a per-epoch callback here (partial receipts).
_ON_CALL: list[Callable[[dict[str, Any]], None]] = []
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def _progress(event: Mapping[str, Any]) -> None:
    print(json.dumps({"phase": "probe", **event}, sort_keys=True), flush=True)


def _q(model: torch.nn.Module, obs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Batch-one CPU fp32 forward (partition-independent by construction)."""
    from research.t34_claim_a.student import needs_full_sight
    from src.model.ego2s_network import obs_tensors

    tensors = obs_tensors(obs, torch.device("cpu"))
    with torch.no_grad():
        if needs_full_sight(model):
            values = model(*tensors, torch.from_numpy(np.ascontiguousarray(obs["full_sight"])))
        else:
            values = model(*tensors)
    return values.float().numpy().astype(np.float32)[0]


def play_world(
    row: Mapping[str, Any],
    carrier: str,
    model: torch.nn.Module,
    previous: torch.nn.Module,
    horizon: int,
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Play one development world to death or ``horizon`` with the current student."""
    from research.t34_claim_a.student import _evaluation, full_sight_planes, needs_full_sight
    from research.t34_exit_b8.policy import B8Policy, action_grid
    from research.t34_exit_b8.trajectory import build_world, prepare_world
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    seed = int(row.get("world_seed", row.get("seed", 0)))
    ambient_py, ambient_np = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        sight = needs_full_sight(model) or needs_full_sight(previous)
        dispatcher = B8Policy(device="cpu")
        world = build_world(dict(row), carrier)
        root = np.asarray([[0, 0]], dtype=np.int64)
        actions, prior_actions, lengths = [], [], []
        forward = 0.0
        alive = True
        for step in range(int(horizon)):
            prepare_world(world)
            obs = dict(
                build_ego_raster(
                    world.sim, EgoRasterConfig(version="bv8"), backend="numba", rows=root
                )
            )
            if sight:
                obs["full_sight"] = full_sight_planes(world.sim, root)
            tick = perf_counter()
            current, earlier = _q(model, obs), _q(previous, obs)
            forward += perf_counter() - tick
            action = _evaluation(world.sim, current).greedy_action()
            prior_actions.append(_evaluation(world.sim, earlier).greedy_action())
            actions.append(action)
            lengths.append(int(world.sim.length[0, 0]))
            world.sim.step(action_grid(world, action, cache=dispatcher))
            world.runtime.observe_step(world.sim)
            if progress is not None:
                progress(step)
            if not bool(world.sim.alive[0, 0]):
                alive = False
                break
    finally:
        random.setstate(ambient_py)
        np.random.set_state(ambient_np)
    return {
        "world_id": str(row["world_id"]),
        "mix": str(row.get("mix", "")),
        "horizon": int(horizon),
        "complete": True,
        "survived": alive,
        "actions": actions,
        "previous_actions": prior_actions,
        "lengths": lengths,
        "forward_seconds": forward,
    }


def _worker_init() -> None:
    for key in _THREAD_ENV:
        os.environ[key] = "1"
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    torch.set_num_threads(1)
    import numba

    numba.config.reload_config()
    numba.set_num_threads(1)


def _play_shard(
    rank: int,
    rows: Sequence[Mapping[str, Any]],
    carrier: str,
    model: torch.nn.Module,
    previous: torch.nn.Module,
    horizon: int,
    epoch: int,
) -> list[dict[str, Any]]:
    """Worker entry: play a round-robin shard serially, with its own progress lines."""
    last = [perf_counter()]
    results = []
    for index, row in enumerate(rows):

        def tick(step: int, index: int = index) -> None:
            if perf_counter() - last[0] >= PROGRESS_SECONDS:
                _progress(
                    {"epoch": epoch, "worker": rank, "world": index, "of": len(rows), "step": step}
                )
                last[0] = perf_counter()

        results.append(play_world(row, carrier, model, previous, horizon, tick))
    return results


def _cpu_copy(model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(model).to("cpu", dtype=torch.float32).eval()


@dataclass
class DevProbe:
    """Student rollouts over a fixed development world bank (serial or K processes)."""

    carrier_checkpoint: Path
    previous_model: torch.nn.Module
    rows: Sequence[Mapping[str, Any]]
    horizon: int = PRODUCTION_HORIZON
    workers: int = 1
    calls: list[dict[str, Any]] = field(default_factory=list)
    on_call: Callable[[dict[str, Any]], None] | None = None

    def __post_init__(self) -> None:
        if not 1 <= int(self.horizon) <= PRODUCTION_HORIZON:
            raise ValueError("probe horizon must lie in [1, 5000]")
        if type(self.workers) is not int or self.workers < 1:
            raise ValueError("probe workers must be a positive integer")
        ids = [str(row["world_id"]) for row in self.rows]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("probe needs unique development world ids")
        if any("/probe" not in world_id for world_id in ids):
            raise ValueError("probe worlds must come from a dedicated probe namespace")

    def _play(self, model: torch.nn.Module, previous: torch.nn.Module, epoch: int) -> list[dict]:
        carrier = str(self.carrier_checkpoint)
        workers = min(self.workers, len(self.rows))
        if workers == 1:
            return _play_shard(0, self.rows, carrier, model, previous, self.horizon, epoch)
        shards = [list(self.rows[rank::workers]) for rank in range(workers)]
        context = mp.get_context("spawn")
        results: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
        started = perf_counter()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=_worker_init
        ) as pool:
            futures = {
                pool.submit(
                    _play_shard, rank, shard, carrier, model, previous, self.horizon, epoch
                ): rank
                for rank, shard in enumerate(shards)
            }
            pending = set(futures)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=PROGRESS_SECONDS,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    results[futures[future]] = future.result()
                _progress(
                    {
                        "epoch": epoch,
                        "shards_done": workers - len(pending),
                        "shards": workers,
                        "elapsed_seconds": round(perf_counter() - started, 1),
                    }
                )
        merged: dict[str, dict[str, Any]] = {}
        for shard in results:
            for trace in shard:
                merged[trace["world_id"]] = trace
        return [merged[str(row["world_id"])] for row in self.rows]

    def __call__(self, model: torch.nn.Module, epoch: int) -> ProbeResult:
        started = perf_counter()
        was_training = model.training
        current, previous = _cpu_copy(model), _cpu_copy(self.previous_model)
        _progress({"epoch": int(epoch), "event": "start", "workers": self.workers})
        traces = self._play(current, previous, int(epoch))
        if was_training:
            model.train()
        forward = float(sum(trace.pop("forward_seconds") for trace in traces))
        survival = float(np.mean([trace["survived"] for trace in traces]))
        call = {
            "epoch": int(epoch),
            "seconds": perf_counter() - started,
            "forward_seconds": forward,
            "decisions": int(sum(len(t["actions"]) for t in traces)),
            "survival": survival,
            "worlds": len(traces),
            "horizon": int(self.horizon),
            "workers": min(self.workers, len(self.rows)),
        }
        self.calls.append(call)
        _progress({"event": "done", **call})
        if self.on_call is not None:
            self.on_call(call)
        return ProbeResult(survival=survival, trajectories=traces, is_gate_data=False)


def probe_rows(namespace: str, worlds: int, mixes: Sequence[str] | None = None) -> list[dict]:
    """Development world bank from the ratified prep-world generator."""
    from research.t34_exit_b8.trajectory import make_world_rows

    if "/probe" not in namespace:
        raise ValueError("probe namespace must contain '/probe' (never a label or gate bank)")
    config: dict[str, Any] = {"namespace": namespace}
    if mixes:
        config["mixes"] = tuple(mixes)
    return make_world_rows(config, int(worlds))


def probe_factory(*, binding: Mapping[str, Any], previous_model: torch.nn.Module) -> DevProbe:
    """``module:function`` named by ``binding['probe_factory']``."""
    settings = dict(binding.get("claim_a") or {})
    probe = dict(settings.get("probe") or {})
    carrier = settings.get("carrier_checkpoint")
    if not carrier:
        raise ValueError("binding.claim_a.carrier_checkpoint is required for the probe")
    namespace = str(probe.get("namespace") or "")
    if not namespace:
        raise ValueError("binding.claim_a.probe.namespace must be preregistered")
    rows = probe_rows(
        namespace, int(probe.get("worlds", PRODUCTION_WORLDS)), probe.get("mixes") or None
    )
    hook = DevProbe(
        carrier_checkpoint=Path(str(carrier)),
        previous_model=previous_model,
        rows=rows,
        horizon=int(probe.get("horizon", PRODUCTION_HORIZON)),
        workers=int(probe.get("workers", 1)),
        on_call=_ON_CALL[-1] if _ON_CALL else None,
    )
    _LAST.append(hook)
    return hook


def last_probe() -> DevProbe | None:
    return _LAST[-1] if _LAST else None


def validate_smoke_probe(probe: ProbeResult, *, worlds: int, horizon: int) -> dict[str, Any]:
    """The ratified ``validate_probe`` checks at a declared non-production shape.

    Used only by the local smoke (stamped ``SMOKE_PROBE_SHAPE``); production
    runs keep the ratified 32-world / H5000 validator untouched.
    """
    from research.t34_exit_b8_fit.diagnostics import trajectory_diagnostics

    if probe.is_gate_data:
        raise ValueError("selection probe must never use gate data")
    if len(probe.trajectories) != worlds:
        raise ValueError(f"required {worlds} smoke probe worlds, got {len(probe.trajectories)}")
    old, new, length = [], [], []
    for record in probe.trajectories:
        if int(record.get("horizon", 0)) != horizon or not bool(record.get("complete", False)):
            raise ValueError("smoke probe requires complete trajectories at its declared horizon")
        old.extend(record["previous_actions"])
        new.extend(record["actions"])
        length.extend(record["lengths"])
    return {
        "survival": float(probe.survival),
        "smoke_probe_shape": {"worlds": worlds, "horizon": horizon},
        **trajectory_diagnostics(
            previous_actions=np.asarray(old), actions=np.asarray(new), lengths=np.asarray(length)
        ),
    }
