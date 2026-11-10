"""Batched ``open-stop-h32`` certificate on ``CTDeviceSim`` / ``DeviceSim`` and gradient check.

Implements Step B1a (``M0-GPU-SITL-Preflight``) from
``docs/research/advice/2026-11-09_live_search_reopen_registration_argon.md`` §3.2:
1. Device-native batched ``open-stop-h32`` rollout (`6 * E` branches × `H = 32` steps)
   and terminal openness certificate (`alive` through `H = 32`,
   `v8_best_safe_area >= min(max(4, L), 160)`, and `max_pocket_r >= L` for `L >= 160`)
   mirroring ``research/watch_replay/open_stop.py``.
2. Single-backward-pass gradient check of the registration's margin loss
   ``max(0, delta - (Q(s, a*) - Q(s, a0)))`` with ``delta = 0.0632`` on unfrozen
   ``Ego2sNet`` initialized from ``b8-s3``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from numba import njit

from research.bl_served_search import ceiling
from research.gpu_sim_spike import ego_device, r2_cache, v8_device
from research.gpu_sim_spike.device_policy import DevicePolicy
from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.ego_device import (
    _ttl_by_owner,
    build_observation,
)
from research.gpu_sim_spike.v8_device import v8_decide as gpu_v8_decide
from research.gpu_sim_spike.vector_device import (
    FEATURE_ROW_CHUNK,
    FREE_SPACE_BFS_CAP,
    FREE_SPACE_MIN_CAP,
    DeviceVectorRuntime,
)
from research.sandbox_a_resident.device_sim import _ENV_TENSOR_FIELDS
from research.watch_replay.open_stop import (
    _CLOSED,
    OPEN_STOP_BRANCH_HORIZON,
    OPEN_STOP_CRN_NAMESPACE,
    OPEN_STOP_POCKET_MIN_LENGTH,
    OPEN_STOP_SELECTION_SUFFIXES,
    OPEN_STOP_V8_LAMBDA,
    OPEN_STOP_V8_NEED_CAP,
    OPEN_STOP_V8_NEED_FLOOR,
    branch_suffix_seed,
    evaluation_order_nb,
)
from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint

RESCUE_MARGIN_DELTA_DEFAULT = 0.3002
ORDINARY_MARGIN_DELTA_DEFAULT = 0.0632
MARGIN_DELTA_DEFAULT = RESCUE_MARGIN_DELTA_DEFAULT

_ORIG_V8_FLOOD = v8_device._flood
_ORIG_EGO_COMPONENTS = ego_device._component_sizes


@njit(cache=True)
def _bfs_flood_cpu_nb(
    start: np.ndarray,
    blocked: np.ndarray,
    cap: np.ndarray,
    has_release: bool,
    release: np.ndarray,
    limit: int,
) -> np.ndarray:
    """Exact cardinal FIFO BFS matching ``v8_device._flood`` / ``flood_cuda.flood`` on CPU."""
    n, a, _ = start.shape
    h, w = blocked.shape[2], blocked.shape[3]
    out = np.zeros((n, a), dtype=np.int64)
    qx = np.empty(h * w, dtype=np.int32)
    qy = np.empty(h * w, dtype=np.int32)
    qd = np.empty(h * w, dtype=np.int32)
    visited = np.zeros((h, w), dtype=np.int32)
    stamp = 0
    for i in range(n):
        for j in range(a):
            c = int(cap[i, j])
            if c <= 0:
                continue
            sx = int(start[i, j, 0])
            sy = int(start[i, j, 1])
            if sx < 0 or sx >= w or sy < 0 or sy >= h:
                continue
            if blocked[i, j, sy, sx]:
                continue
            if has_release and release[i, j, sy, sx] > 0:
                continue
            if c == 1 or limit <= 0:
                out[i, j] = 1
                continue
            stamp += 1
            visited[sy, sx] = stamp
            qx[0] = sx
            qy[0] = sy
            qd[0] = 0
            head = 0
            tail = 1
            while head < tail and tail < c:
                x = qx[head]
                y = qy[head]
                d = qd[head]
                head += 1
                nd = d + 1
                if nd > limit:
                    continue
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if visited[ny, nx] != stamp and not blocked[i, j, ny, nx]:
                            if not has_release or release[i, j, ny, nx] <= nd:
                                visited[ny, nx] = stamp
                                qx[tail] = nx
                                qy[tail] = ny
                                qd[tail] = nd
                                tail += 1
                                if tail >= c:
                                    break
            out[i, j] = tail if tail < c else c
    return out


def _cpu_queue_flood(
    start: torch.Tensor,
    blocked: torch.Tensor,
    cap: torch.Tensor,
    release: torch.Tensor | None = None,
    *,
    max_steps: int | None = None,
) -> torch.Tensor:
    """CPU queue fallback for ``v8_device._flood``; delegates to CUDA/FakeTensor when not CPU."""
    if start.device.type != "cpu" or not v8_device._host_scalar_safe(start):
        return _ORIG_V8_FLOOD(start, blocked, cap, release, max_steps=max_steps)
    h, w = blocked.shape[-2:]
    limit = min(h * w - 1, max_steps if max_steps is not None else v8_device._AREA_MAX)
    start_np = np.ascontiguousarray(start.numpy(), dtype=np.int64)
    blocked_np = np.ascontiguousarray(blocked.numpy(), dtype=np.bool_)
    cap_np = np.ascontiguousarray(cap.numpy(), dtype=np.int64)
    if release is None:
        rel_np = np.empty((1, 1, 1, 1), dtype=np.int64)
        res = _bfs_flood_cpu_nb(start_np, blocked_np, cap_np, False, rel_np, int(limit))
    else:
        rel_np = np.ascontiguousarray(release.numpy(), dtype=np.int64)
        res = _bfs_flood_cpu_nb(start_np, blocked_np, cap_np, True, rel_np, int(limit))
    return torch.from_numpy(res)


@contextmanager
def _maybe_cpu_fast_flood(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    prev_flood = v8_device._flood
    prev_comp = ego_device._component_sizes

    def _cpu_queue_components(sim: Any, rows: torch.Tensor, ttl: torch.Tensor) -> torch.Tensor:
        if sim.device.type != "cpu" or not ego_device._host_scalar_safe(rows):
            return _ORIG_EGO_COMPONENTS(sim, rows, ttl)
        n_rows, h, w = rows.shape[0], int(sim.grid_h), int(sim.grid_w)
        env, snake = rows[:, 0], rows[:, 1]
        cardinal = ego_device._cardinal(sim)
        heads, direction = sim.heads()[env, snake], sim.direction[env, snake]
        rel = torch.arange(3, device=sim.device)[None, :] - 1
        move = cardinal[(direction[:, None] + rel) % 4]
        start = heads[:, None, :] + move
        v_max = h * w
        blocked = ~(ttl.amax(dim=1) <= 1)
        blocked_rows = blocked[env][:, None].expand(-1, 3, -1, -1)
        cap = torch.full((n_rows, 3), v_max, dtype=torch.long, device=sim.device)
        sizes = _cpu_queue_flood(start, blocked_rows, cap, max_steps=v_max)
        return torch.where(sim.alive[env, snake, None], sizes, torch.zeros_like(sizes))

    v8_device._flood = _cpu_queue_flood
    ego_device._component_sizes = _cpu_queue_components
    try:
        yield
    finally:
        v8_device._flood = prev_flood
        ego_device._component_sizes = prev_comp


_VECTOR_RUNTIME_ENV_FIELDS: tuple[str, ...] = (
    "controlled",
    "_carry_state",
    "_carry_mask",
    "_carry_valid",
    "_alive_post",
    "sel_state",
    "sel_mask",
    "sel_ready",
    "sel_fresh",
    "prev_dist",
    "prev_id",
)


class FastDeviceVectorRuntime(DeviceVectorRuntime):
    """Bit-identical ``DeviceVectorRuntime`` subclass that skips redundant floods on CPU and CUDA.

    In eager rollout / ``open-stop-h32`` branch evaluation (outside CUDA graphs):
    - ``prepare()`` skips ``features()`` when ``fresh`` has zero active rows
      (i.e. all controlled slots are carried from ``observe_step``), and when
      ``fresh`` is non-empty, computes features and free-space floods only on
      the ``fresh`` rows.
    - ``observe_step()`` computes features and free-space floods only on the
      ``capture = controlled & alive`` rows (excluding slot 0 hero and dead slots).
    - ``_free_space_for_rows()`` delegates to ``v8_device._flood`` so ``cpu_fast_flood=True``
      uses the Numba FIFO queue on CPU, ``cpu_fast_flood=False`` uses the pure-Torch
      wavefront flood on CPU, and CUDA uses ``flood_cuda.flood``.
    """

    def _free_space_for_rows(self, sim: Any, rows: torch.Tensor) -> torch.Tensor:
        assert self.params is not None
        if not len(rows):
            return torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        er, sr = rows[:, 0], rows[:, 1]
        if r2_cache.enabled(sim):
            occupied_all = r2_cache.all_occupancy(sim)
            heads_all = sim.heads()
        else:
            cells, live_body, _, _ = self._body_view(sim)
            e = int(sim.E)
            occ_u8 = torch.zeros((e, sim.grid_h, sim.grid_w), dtype=torch.uint8, device=self.device)
            all_cells = cells.reshape(e, -1, 2)
            all_live = live_body.reshape(e, -1)
            ox, oy = all_cells[..., 0], all_cells[..., 1]
            inb = all_live & (ox >= 0) & (ox < sim.grid_w) & (oy >= 0) & (oy < sim.grid_h)
            occ_u8.flatten(1).scatter_reduce_(
                1,
                (oy.clamp(0, sim.grid_h - 1) * sim.grid_w + ox.clamp(0, sim.grid_w - 1)),
                inb.to(torch.uint8),
                "amax",
                include_self=True,
            )
            occupied_all = occ_u8 != 0
            heads_all = cells[..., 0, :]

        n = len(rows)
        heads = heads_all[er, sr]
        direction = sim.direction.to(self.device, dtype=torch.long)[er, sr]
        length = sim.length.to(self.device)[er, sr]
        turns, cardinal = self._turns_and_cardinal(sim)
        dirs = (direction[..., None] + turns) % 4
        starts = heads[..., None, :] + cardinal[dirs]
        blocked = occupied_all[er][:, None].expand(n, 3, -1, -1)

        cap = torch.minimum(
            torch.full_like(length, FREE_SPACE_BFS_CAP),
            torch.clamp(length * 2, min=FREE_SPACE_MIN_CAP),
        ).to(torch.long)
        count = v8_device._flood(
            starts,
            blocked,
            cap[:, None].expand(n, 3),
            max_steps=FREE_SPACE_BFS_CAP - 1,
        )
        return self._normalize_free(count, cap)

    def _features_for_rows(self, sim: Any, rows: torch.Tensor) -> torch.Tensor:
        assert self.params is not None
        fs_rows = self._free_space_for_rows(sim, rows) if self.params.use_free_space else None
        fs_full = None
        if fs_rows is not None:
            fs_full = torch.zeros((sim.E, sim.S, 3), dtype=torch.float32, device=self.device)
            fs_full[rows[:, 0], rows[:, 1]] = fs_rows
        if len(rows) <= FEATURE_ROW_CHUNK:
            return self._features_rows(sim, rows, update_memory=True, free_space=fs_full)
        parts = []
        for start in range(0, len(rows), FEATURE_ROW_CHUNK):
            stop = min(start + FEATURE_ROW_CHUNK, len(rows))
            parts.append(
                self._features_rows(
                    sim,
                    rows[start:stop],
                    update_memory=True,
                    free_space=fs_full,
                )
            )
        return torch.cat(parts, dim=0)

    def prepare(self, sim: Any, rows_mask: torch.Tensor | None = None) -> None:
        if not v8_device._host_scalar_safe(sim.alive):
            super().prepare(sim, rows_mask=rows_mask)
            return
        self._check()
        assert (
            self.controlled is not None
            and self._carry_valid is not None
            and self._alive_post is not None
        )
        assert (
            self.sel_state is not None
            and self.sel_mask is not None
            and self.sel_ready is not None
            and self.sel_fresh is not None
        )
        env_mask = (
            torch.ones(sim.E, dtype=torch.bool, device=self.device)
            if rows_mask is None
            else torch.as_tensor(rows_mask, device=self.device, dtype=torch.bool)
        )
        if env_mask.shape != (sim.E,):
            raise ValueError(f"rows_mask must have shape {(sim.E,)}, got {tuple(env_mask.shape)}")
        affected = self.controlled & env_mask[:, None]
        alive = sim.alive.to(self.device, dtype=torch.bool)
        respawned = (alive & ~self._alive_post).any(1)
        ready = affected & alive
        carried = ready & self._carry_valid & ~respawned[:, None]
        fresh = ready & ~carried
        next_state = torch.where(
            affected[..., None], torch.zeros_like(self.sel_state), self.sel_state
        )
        next_mask = torch.where(affected[..., None], torch.zeros_like(self.sel_mask), self.sel_mask)
        next_state = torch.where(carried[..., None], self._carry_state, next_state)
        next_mask = torch.where(carried[..., None], self._carry_mask, next_mask)
        if bool(fresh.any().item()):
            fresh_rows = fresh.nonzero(as_tuple=False)
            states_sub = self._features_for_rows(sim, fresh_rows)
            next_state = next_state.clone()
            next_state[fresh_rows[:, 0], fresh_rows[:, 1]] = states_sub
            masks = self._mask(sim)
            next_mask = torch.where(fresh[..., None], masks, next_mask)
        self.sel_state = next_state
        self.sel_mask = next_mask
        self._carry_valid = torch.where(
            affected, torch.zeros_like(self._carry_valid), self._carry_valid
        )
        self.sel_ready = torch.where(affected, ready, self.sel_ready)
        self.sel_fresh = torch.where(affected, fresh, self.sel_fresh)
        self.fresh_rows = self.fresh_rows + fresh.sum()
        self.carried_rows = self.carried_rows + carried.sum()

    def observe_step(self, sim: Any, rows_mask: torch.Tensor | None = None) -> None:
        if not v8_device._host_scalar_safe(sim.alive):
            super().observe_step(sim, rows_mask=rows_mask)
            return
        self._check()
        assert (
            self.controlled is not None
            and self._carry_valid is not None
            and self._alive_post is not None
        )
        env_mask = (
            torch.ones(sim.E, dtype=torch.bool, device=self.device)
            if rows_mask is None
            else torch.as_tensor(rows_mask, device=self.device, dtype=torch.bool)
        )
        affected = self.controlled & env_mask[:, None]
        done = getattr(sim, "_last_done", torch.zeros_like(sim.alive)).to(
            self.device, dtype=torch.bool
        )
        reset = done & affected
        self.prev_dist = torch.where(
            reset, torch.full_like(self.prev_dist, torch.inf), self.prev_dist
        )
        self.prev_id = torch.where(reset, torch.full_like(self.prev_id, -1), self.prev_id)
        alive = sim.alive.to(self.device, dtype=torch.bool)
        capture = affected & alive
        if bool(capture.any().item()):
            cap_rows = capture.nonzero(as_tuple=False)
            states_sub = self._features_for_rows(sim, cap_rows)
            self._carry_state = self._carry_state.clone()
            self._carry_state[cap_rows[:, 0], cap_rows[:, 1]] = states_sub
            masks = self._mask(sim)
            self._carry_mask = torch.where(capture[..., None], masks, self._carry_mask)
        self._carry_valid = torch.where(affected, capture, self._carry_valid)
        self._alive_post = torch.where(env_mask[:, None], alive, self._alive_post)

    def repeat_interleave(self, repeats: int) -> "FastDeviceVectorRuntime":
        out = super().repeat_interleave(repeats)
        out.__class__ = FastDeviceVectorRuntime
        return out


def sync_runtime_from_cpu(
    cpu_runtime: Any,
    device_policy: DevicePolicy,
    *,
    vector_prepared: bool,
) -> None:
    """Copy a CPU :class:`Vector61Runtime` state into ``device_policy.vector_runtime``."""
    rt = device_policy.vector_runtime
    dev = device_policy.device
    rt._carry_state.copy_(
        torch.as_tensor(cpu_runtime._carry_state, dtype=torch.float32, device=dev)
    )
    rt._carry_mask.copy_(torch.as_tensor(cpu_runtime._carry_mask, dtype=torch.bool, device=dev))
    rt._carry_valid.copy_(torch.as_tensor(cpu_runtime._carry_valid, dtype=torch.bool, device=dev))
    rt._alive_post.copy_(torch.as_tensor(cpu_runtime._alive_post, dtype=torch.bool, device=dev))
    rt.sel_state.copy_(torch.as_tensor(cpu_runtime.sel_state, dtype=torch.float32, device=dev))
    rt.sel_mask.copy_(torch.as_tensor(cpu_runtime.sel_mask, dtype=torch.bool, device=dev))
    rt.sel_ready.copy_(torch.as_tensor(cpu_runtime.sel_ready, dtype=torch.bool, device=dev))
    rt.sel_fresh.copy_(torch.as_tensor(cpu_runtime.sel_fresh, dtype=torch.bool, device=dev))
    rt.prev_dist.copy_(
        torch.as_tensor(cpu_runtime.memory.prev_dist, dtype=torch.float32, device=dev)
    )
    rt.prev_id.copy_(torch.as_tensor(cpu_runtime.memory.prev_id, dtype=torch.long, device=dev))
    device_policy.vector_prepared = bool(vector_prepared)


@torch.no_grad()
def compact_sim_and_policy(
    sim: DeviceSim,
    policy: DevicePolicy,
    keep_rows: torch.Tensor,
) -> None:
    """Compact ``sim`` and ``policy`` in-place to the 1-D row indices ``keep_rows``."""
    keep_rows = keep_rows.to(device=sim.device, dtype=torch.long).reshape(-1)
    new_e = int(keep_rows.numel())
    for name in _ENV_TENSOR_FIELDS:
        val = getattr(sim, name, None)
        if isinstance(val, torch.Tensor):
            setattr(sim, name, val.index_select(0, keep_rows))
    sim.E = new_e
    r2_cache.bump(sim)

    rt = policy.vector_runtime
    for name in _VECTOR_RUNTIME_ENV_FIELDS:
        val = getattr(rt, name, None)
        if isinstance(val, torch.Tensor):
            setattr(rt, name, val.index_select(0, keep_rows))
    rt.E = new_e


@torch.no_grad()
def evaluate_terminal_openness_device(sim: DeviceSim) -> dict[str, torch.Tensor]:
    """Evaluate the batched ``open-stop`` terminal openness predicate on ``sim.device``.

    Mirrors :func:`research.watch_replay.open_stop.evaluate_terminal_openness` across
    all ``E`` worlds of ``sim`` (`cpu` or `cuda`) without host round-trips:
    - ``v8_need = min(max(4, length), 160)``
    - ``v8_best_safe_area = max_{a: advisory[a]} area(a)`` from ``v8_decide(..., features=True)``
    - ``v8_open = (v8_best_safe_area >= v8_need)``
    - ``max_pocket_r = max_{a: advisory[a]} region_size(a % 3)`` from ``_component_sizes``
    - ``pocket_open = (max_pocket_r >= length) if length >= 160 else True``
    - ``un_enclosed = v8_open & pocket_open``
    """
    e_count = int(sim.E)
    device = sim.device
    if e_count == 0:
        empty_b = torch.zeros(0, dtype=torch.bool, device=device)
        empty_i = torch.zeros(0, dtype=torch.long, device=device)
        return {
            "alive": empty_b,
            "length": empty_i,
            "advisory_any": empty_b,
            "v8_need": empty_i,
            "v8_best_safe_area": empty_i,
            "v8_open": empty_b,
            "max_pocket_r": empty_i,
            "pocket_open": empty_b,
            "un_enclosed": empty_b,
        }

    alive = sim.alive[:, 0].clone()
    zeros_i = torch.zeros(e_count, dtype=torch.long, device=device)
    raw_length = sim.length[:, 0].to(torch.long)
    length = torch.where(alive, raw_length, zeros_i)
    v8_need = torch.where(
        alive,
        torch.clamp(length, min=int(OPEN_STOP_V8_NEED_FLOOR), max=int(OPEN_STOP_V8_NEED_CAP)),
        zeros_i,
    )

    advisory = sim.advisory_mask()[:, 0] & alive.unsqueeze(1)
    advisory_any = advisory.any(dim=1)

    hero_rows = torch.stack(
        (
            torch.arange(e_count, device=device, dtype=torch.long),
            torch.zeros(e_count, device=device, dtype=torch.long),
        ),
        dim=1,
    )
    q_zeros = torch.zeros((e_count, 6), dtype=torch.float32, device=device)
    v8_res = gpu_v8_decide(
        sim,
        hero_rows,
        q_zeros,
        advisory,
        float(OPEN_STOP_V8_LAMBDA),
        features=True,
    )
    areas = torch.where(
        advisory,
        v8_res.area.to(torch.long),
        torch.zeros((e_count, 6), dtype=torch.long, device=device),
    )
    v8_best_safe_area = torch.where(advisory_any, areas.max(dim=1).values, zeros_i)
    v8_open = advisory_any & (v8_best_safe_area >= v8_need)

    reg = ego_device._component_sizes(sim, hero_rows, _ttl_by_owner(sim)).to(torch.long)
    dir_idx = torch.tensor([0, 1, 2, 0, 1, 2], device=device, dtype=torch.long)
    reg_actions = reg.index_select(1, dir_idx)
    legal_reg = torch.where(
        advisory,
        reg_actions,
        torch.zeros((e_count, 6), dtype=torch.long, device=device),
    )
    max_pocket_r = torch.where(advisory_any, legal_reg.max(dim=1).values, zeros_i)
    pocket_open = advisory_any & torch.where(
        length >= int(OPEN_STOP_POCKET_MIN_LENGTH),
        max_pocket_r >= length,
        torch.ones(e_count, dtype=torch.bool, device=device),
    )
    un_enclosed = v8_open & pocket_open

    return {
        "alive": alive,
        "length": length,
        "advisory_any": advisory_any,
        "v8_need": v8_need,
        "v8_best_safe_area": v8_best_safe_area,
        "v8_open": v8_open,
        "max_pocket_r": max_pocket_r,
        "pocket_open": pocket_open,
        "un_enclosed": un_enclosed,
    }


def openness_row_to_dict(openness: dict[str, torch.Tensor], row: int) -> dict[str, Any]:
    """Convert row ``row`` of :func:`evaluate_terminal_openness_device` to CPU dict format."""
    return {
        "length": int(openness["length"][row].item()),
        "advisory_any": bool(openness["advisory_any"][row].item()),
        "v8_need": int(openness["v8_need"][row].item()),
        "v8_best_safe_area": int(openness["v8_best_safe_area"][row].item()),
        "max_pocket_r": int(openness["max_pocket_r"][row].item()),
        "un_enclosed": bool(openness["un_enclosed"][row].item()),
        "v8_open": bool(openness["v8_open"][row].item()),
        "pocket_open": bool(openness["pocket_open"][row].item()),
    }


@torch.no_grad()
def batched_open_stop_h32(
    sim: DeviceSim,
    policy: DevicePolicy,
    *,
    world_seeds: Sequence[int],
    decisions: Sequence[int],
    suffix_id: int = OPEN_STOP_SELECTION_SUFFIXES[0],
    branch_horizon: int = OPEN_STOP_BRANCH_HORIZON,
    namespace: str = OPEN_STOP_CRN_NAMESPACE,
    compact_dead: bool = True,
    enable_r2: bool = True,
    cpu_fast_flood: bool = True,
    return_host_records: bool = True,
) -> dict[str, Any]:
    """Run batched ``open-stop-h32`` over ``R = sim.E`` roots × ``6`` actions on ``sim.device``.

    For each root ``r in 0..R-1`` and hero action ``a in 0..5``:
    - Clones the prepared root state into ``6 * R`` branches on ``sim.device``.
    - Re-keys ``branch_sim.rng_key`` per root with :func:`branch_suffix_seed`
      (action-independent CRN across all 6 actions of root ``r``) and zeros
      ``branch_sim.rng_counter``.
    - Steps ``a`` on step 1 against the ``5x frp3-s12 + v8`` opponent continuation.
    - Rolls steps ``2..branch_horizon`` under the ``b8-s3`` hero continuation and
      ``5x frp3-s12 + v8`` opponent continuation.
    - Certifies surviving branches at ``H = branch_horizon`` with
      :func:`evaluate_terminal_openness_device`.
    """
    r_count = int(sim.E)
    if len(world_seeds) != r_count or len(decisions) != r_count:
        raise ValueError(
            f"world_seeds ({len(world_seeds)}) and decisions ({len(decisions)}) "
            f"must match sim.E ({r_count})"
        )
    if branch_horizon < 1:
        raise ValueError("branch_horizon must be >= 1")

    with _maybe_cpu_fast_flood(bool(cpu_fast_flood and sim.device.type == "cpu")):
        return _batched_open_stop_h32_impl(
            sim,
            policy,
            world_seeds=world_seeds,
            decisions=decisions,
            suffix_id=suffix_id,
            branch_horizon=branch_horizon,
            namespace=namespace,
            compact_dead=compact_dead,
            enable_r2=enable_r2,
            return_host_records=return_host_records,
        )


@torch.no_grad()
def _batched_open_stop_h32_impl(
    sim: DeviceSim,
    policy: DevicePolicy,
    *,
    world_seeds: Sequence[int],
    decisions: Sequence[int],
    suffix_id: int,
    branch_horizon: int,
    namespace: str,
    compact_dead: bool,
    enable_r2: bool,
    return_host_records: bool,
) -> dict[str, Any]:
    r_count = int(sim.E)
    device = sim.device
    root_sim = sim.clone()
    root_policy = policy.repeat_interleave(1)
    if root_policy.vector_runtime is not None:
        root_policy.vector_runtime.__class__ = FastDeviceVectorRuntime
    if enable_r2:
        r2_cache.enable(root_sim)
    if not root_sim._prepared:
        root_sim.prepare()
    if not root_policy.vector_prepared:
        root_policy.vector_runtime.prepare(root_sim)
        root_policy.vector_prepared = True

    root_q_t, root_advisory_t, root_bare_t = root_policy.evaluate(root_sim)
    fallback = ~root_advisory_t.any(dim=1, keepdim=True)
    fallback_mask = torch.tensor(
        [[True, True, True, False, False, False]], dtype=torch.bool, device=device
    )
    root_legal_t = torch.where(
        fallback,
        fallback_mask.expand_as(root_advisory_t),
        root_advisory_t,
    )

    b_count = r_count * 6
    branch_sim = root_sim.repeat_interleave(6)
    branch_policy = root_policy.repeat_interleave(6)
    if branch_policy.vector_runtime is not None:
        branch_policy.vector_runtime.__class__ = FastDeviceVectorRuntime
    if enable_r2:
        r2_cache.enable(branch_sim)

    # Hero is terminal inside the H=32 continuation rollout (never respawns mid-branch).
    branch_sim.terminal_slots[:, 0] = True

    root_keys: list[int] = []
    for ws, dec in zip(world_seeds, decisions):
        raw_seed = branch_suffix_seed(namespace, int(ws), int(dec), suffix_id=int(suffix_id))
        key = int(raw_seed) & 0x7FFF_FFFF_FFFF_FFFF
        if key == 0:
            key = 1
        root_keys.append(key)
    keys_t = torch.tensor(root_keys, dtype=torch.long, device=device).repeat_interleave(6)
    branch_sim.rng_key.copy_(keys_t)
    branch_sim.rng_counter.zero_()

    # Step 1 (h = 0): forced hero action 0..5 per root.
    forced_actions = torch.arange(6, dtype=torch.long, device=device).repeat(r_count)
    full_actions = branch_policy.actions(branch_sim, forced_actions)
    branch_sim.step(full_actions)
    branch_policy.after_step(branch_sim)

    frames_executed = torch.ones(b_count, dtype=torch.long, device=device)
    alive_every_step = branch_sim.alive[:, 0].clone()
    frames_survived = alive_every_step.to(torch.long)
    final_length = torch.where(
        alive_every_step,
        branch_sim.length[:, 0].to(torch.long),
        torch.zeros(b_count, dtype=torch.long, device=device),
    )

    active_idx = torch.arange(b_count, dtype=torch.long, device=device)

    for _step in range(1, int(branch_horizon)):
        if compact_dead:
            keep_mask = branch_sim.alive[:, 0]
            if not bool(keep_mask.any().item()):
                break
            if not bool(keep_mask.all().item()):
                keep_rows = keep_mask.nonzero(as_tuple=False).squeeze(1)
                compact_sim_and_policy(branch_sim, branch_policy, keep_rows)
                active_idx = active_idx.index_select(0, keep_rows)

            branch_sim.prepare()
            branch_policy.vector_runtime.prepare(branch_sim)
            branch_policy.vector_prepared = True
            _, _, hero_act = branch_policy.evaluate(branch_sim)
            step_actions = branch_policy.actions(branch_sim, hero_act)
            branch_sim.step(step_actions)
            branch_policy.after_step(branch_sim)

            still_alive = branch_sim.alive[:, 0]
            frames_executed[active_idx] += 1
            alive_every_step[active_idx] &= still_alive
            frames_survived[active_idx] += still_alive.to(torch.long)
            final_length[active_idx] = torch.where(
                still_alive,
                branch_sim.length[:, 0].to(torch.long),
                torch.zeros_like(final_length[active_idx]),
            )
        else:
            was_alive = alive_every_step.clone()
            if not bool(was_alive.any().item()):
                break
            branch_sim.prepare()
            branch_policy.vector_runtime.prepare(branch_sim)
            branch_policy.vector_prepared = True
            _, _, hero_act = branch_policy.evaluate(branch_sim)
            step_actions = branch_policy.actions(branch_sim, hero_act)
            branch_sim.step(step_actions)
            branch_policy.after_step(branch_sim)

            now_alive = branch_sim.alive[:, 0] & was_alive
            frames_executed = torch.where(was_alive, frames_executed + 1, frames_executed)
            alive_every_step = now_alive
            frames_survived = torch.where(now_alive, frames_survived + 1, frames_survived)
            final_length = torch.where(
                now_alive,
                branch_sim.length[:, 0].to(torch.long),
                torch.where(was_alive, torch.zeros_like(final_length), final_length),
            )

    raw_survives = (frames_executed == int(branch_horizon)) & alive_every_step

    # Evaluate terminal openness on surviving branches in one device batch.
    if compact_dead:
        surv_mask_local = branch_sim.alive[:, 0] & (
            frames_executed.index_select(0, active_idx) == int(branch_horizon)
        )
        if bool(surv_mask_local.any().item()):
            if not bool(surv_mask_local.all().item()):
                surv_rows = surv_mask_local.nonzero(as_tuple=False).squeeze(1)
                compact_sim_and_policy(branch_sim, branch_policy, surv_rows)
                active_idx = active_idx.index_select(0, surv_rows)
            openness_local = evaluate_terminal_openness_device(branch_sim)
        else:
            active_idx = torch.zeros(0, dtype=torch.long, device=device)
            openness_local = evaluate_terminal_openness_device(branch_sim)
    else:
        surv_rows = raw_survives.nonzero(as_tuple=False).squeeze(1)
        if int(surv_rows.numel()) > 0:
            compact_sim_and_policy(branch_sim, branch_policy, surv_rows)
            active_idx = surv_rows
            openness_local = evaluate_terminal_openness_device(branch_sim)
        else:
            active_idx = torch.zeros(0, dtype=torch.long, device=device)
            openness_local = evaluate_terminal_openness_device(branch_sim)

    un_enclosed_full = torch.zeros(b_count, dtype=torch.bool, device=device)
    v8_area_full = torch.zeros(b_count, dtype=torch.long, device=device)
    pocket_r_full = torch.zeros(b_count, dtype=torch.long, device=device)
    if int(active_idx.numel()) > 0:
        un_enclosed_full[active_idx] = openness_local["un_enclosed"]
        v8_area_full[active_idx] = openness_local["v8_best_safe_area"]
        pocket_r_full[active_idx] = openness_local["max_pocket_r"]
    certified_full = raw_survives & un_enclosed_full

    certified_mat = certified_full.view(r_count, 6)
    raw_survives_mat = raw_survives.view(r_count, 6)
    v8_area_mat = v8_area_full.view(r_count, 6)
    pocket_r_mat = pocket_r_full.view(r_count, 6)
    safe_mask = certified_mat & root_legal_t
    unsafe_mask = root_legal_t & ~safe_mask

    row_arange = torch.arange(r_count, device=device, dtype=torch.long)
    bare_clamped = root_bare_t.clamp(0, 5)
    bare_is_safe = safe_mask[row_arange, bare_clamped]
    has_safe = safe_mask.any(dim=1)
    has_unsafe = unsafe_mask.any(dim=1)
    q_safe_masked = root_q_t.masked_fill(~safe_mask, -torch.inf)
    best_safe_q_act = q_safe_masked.argmax(dim=1)
    a_star_t = torch.where(
        bare_is_safe,
        bare_clamped,
        torch.where(has_safe, best_safe_q_act, torch.full_like(bare_clamped, -1)),
    )
    rescue_mask = has_safe & (~bare_is_safe) & has_unsafe

    safe_area_masked = torch.where(safe_mask, v8_area_mat, torch.full_like(v8_area_mat, -1))
    best_area_vals, best_area_act = safe_area_masked.max(dim=1)
    bare_area_vals = v8_area_mat[row_arange, bare_clamped]
    ordinary_mask = has_safe & bare_is_safe & has_unsafe & (best_area_vals > bare_area_vals)

    roots_out: list[dict[str, Any]] = []
    if return_host_records:
        openness_by_branch: dict[int, dict[str, Any]] = {}
        active_idx_cpu = active_idx.detach().cpu().tolist()
        for local_i, b_idx in enumerate(active_idx_cpu):
            openness_by_branch[int(b_idx)] = openness_row_to_dict(openness_local, local_i)

        root_q_np = root_q_t.detach().cpu().numpy()
        root_legal_np = root_legal_t.detach().cpu().numpy()
        root_advisory_np = root_advisory_t.detach().cpu().numpy()
        root_bare_np = root_bare_t.detach().cpu().numpy()
        raw_survives_cpu = raw_survives.detach().cpu().tolist()
        certified_cpu = certified_full.detach().cpu().tolist()
        frames_exec_cpu = frames_executed.detach().cpu().tolist()
        final_len_cpu = final_length.detach().cpu().tolist()

        for r in range(r_count):
            dec = int(decisions[r])
            ws = int(world_seeds[r])
            q_r = root_q_np[r]
            legal_r = root_legal_np[r]
            advisory_r = root_advisory_np[r]
            bare_r = int(root_bare_np[r])
            legal_indices = np.flatnonzero(legal_r).tolist()
            order_r = evaluation_order_nb(legal_indices, bare_r, q_r)

            all_outcomes: dict[int, dict[str, Any]] = {}
            for a in range(6):
                b_idx = r * 6 + a
                cert = bool(certified_cpu[b_idx])
                raw_s = bool(raw_survives_cpu[b_idx])
                f_exec = int(frames_exec_cpu[b_idx])
                f_surv = int(branch_horizon) if raw_s else max(0, f_exec - 1)
                open_info = (
                    openness_by_branch[b_idx]
                    if raw_s and b_idx in openness_by_branch
                    else {
                        **dict(_CLOSED),
                        "v8_open": False,
                        "pocket_open": False,
                    }
                )
                all_outcomes[a] = {
                    "action": int(a),
                    "legal": bool(legal_r[a]),
                    "status": "EVALUATED",
                    "suffix_id": int(suffix_id),
                    "survives": cert,
                    "certified": cert,
                    "raw_survives": raw_s,
                    "frames_executed": f_exec,
                    "frames_survived": f_surv,
                    "final_length": int(final_len_cpu[b_idx]) if raw_s else 0,
                    "terminal_openness": open_info,
                }

            # Mirror open_stop_batch's exact early-stop selection over `order_r`.
            evaluated_actions: list[int] = []
            early_stop_outcomes: dict[int, dict[str, Any]] = {}
            chosen_cert: int | None = None
            for a in order_r:
                evaluated_actions.append(int(a))
                early_stop_outcomes[int(a)] = dict(all_outcomes[int(a)])
                if all_outcomes[int(a)]["certified"]:
                    chosen_cert = int(a)
                    break
            if chosen_cert is not None:
                chosen = int(chosen_cert)
                stop_reason = "FIRST_CERTIFIED_SAFE"
            else:
                chosen = int(ceiling._choose(early_stop_outcomes, bare_r, q_r))
                stop_reason = "ALL_FAIL_FALLBACK"

            reported_outcomes: dict[int, dict[str, Any]] = {}
            for a in sorted(order_r):
                if a in early_stop_outcomes:
                    reported_outcomes[a] = early_stop_outcomes[a]
                else:
                    reported_outcomes[a] = {"status": "NOT_EVALUATED"}

            roots_out.append(
                {
                    "world_seed": ws,
                    "decision": dec,
                    "bare": bare_r,
                    "chosen": chosen,
                    "stop_reason": stop_reason,
                    "q": q_r.astype(float).tolist(),
                    "advisory": advisory_r.astype(bool).tolist(),
                    "legal": legal_r.astype(bool).tolist(),
                    "evaluation_order": evaluated_actions,
                    "full_order": [int(x) for x in order_r],
                    "evaluated_actions": evaluated_actions,
                    "all_outcomes": all_outcomes,
                    "outcomes": reported_outcomes,
                }
            )

    return {
        "roots": roots_out,
        "certified_matrix": certified_mat.detach().cpu().numpy() if return_host_records else None,
        "raw_survives_matrix": (
            raw_survives_mat.detach().cpu().numpy() if return_host_records else None
        ),
        "legal_mask": root_legal_t,
        "safe_mask": safe_mask,
        "unsafe_mask": unsafe_mask,
        "certified": certified_mat,
        "raw_survives": raw_survives_mat,
        "v8_best_safe_area": v8_area_mat,
        "max_pocket_r": pocket_r_mat,
        "root_q": root_q_t,
        "bare_actions": root_bare_t,
        "a_star_actions": a_star_t,
        "re_rank_target_actions": best_area_act,
        "rescue_mask": rescue_mask,
        "ordinary_mask": ordinary_mask,
    }


def freeze_ego2s_value_head(net: Ego2sNet) -> dict[str, int]:
    """Freeze ``value.weight`` and ``value.bias`` on ``Ego2sNet`` for pairwise hinge training.

    In ``Ego2sNet``, ``Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))``. Any within-state
    pairwise action difference ``Q(s, a*) - Q(s, a_unsafe)`` cancels ``V(s)`` identically,
    giving ``value.*`` exact zero gradient while served play and all registered metrics
    depend only on within-state action ordering.
    """
    frozen_tensors = 0
    frozen_scalars = 0
    trainable_tensors = 0
    trainable_scalars = 0
    for name, param in net.named_parameters():
        numel = int(param.numel())
        if name.startswith("value."):
            param.requires_grad_(False)
            param.grad = None
            frozen_tensors += 1
            frozen_scalars += numel
        else:
            param.requires_grad_(True)
            trainable_tensors += 1
            trainable_scalars += numel
    return {
        "frozen_param_tensors": int(frozen_tensors),
        "frozen_scalar_params": int(frozen_scalars),
        "trainable_param_tensors": int(trainable_tensors),
        "trainable_scalar_params": int(trainable_scalars),
    }


def margin_loss_gradient_check(
    b8_checkpoint: str,
    *,
    sim: DeviceSim | None = None,
    legal_mask: torch.Tensor | None = None,
    safe_mask: torch.Tensor | None = None,
    unsafe_mask: torch.Tensor | None = None,
    a_star: Sequence[int] | torch.Tensor | None = None,
    a_0: Sequence[int] | torch.Tensor | None = None,
    delta: float = RESCUE_MARGIN_DELTA_DEFAULT,
    freeze_value_head: bool = True,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run one backward pass of the registered rescue hinge loss on ``Ego2sNet``.

    Computes ``max(0, delta - (Q(s, a*(s)) - max_{a in legal(s) \\ A_safe(s)} Q(s, a)))``
    with ``delta = 0.3002`` on roots that have both safe and unsafe legal actions,
    freezing ``value.weight`` and ``value.bias`` via :func:`freeze_ego2s_value_head`
    and asserting finite, non-zero gradients on every non-value parameter tensor.
    """
    dev = torch.device(device)
    net = load_ego2s_checkpoint(str(b8_checkpoint), device=dev)
    if not isinstance(net, Ego2sNet):
        raise TypeError(f"Expected Ego2sNet from {b8_checkpoint}, got {type(net)!r}")
    local_channels = int(net.config["local_channels"])
    n_scalars = int(net.config["n_scalars"])
    net.train()
    if freeze_value_head:
        freeze_info = freeze_ego2s_value_head(net)
    else:
        for p in net.parameters():
            p.requires_grad_(True)
        freeze_info = {
            "frozen_param_tensors": 0,
            "frozen_scalar_params": 0,
            "trainable_param_tensors": sum(1 for _ in net.parameters()),
            "trainable_scalar_params": sum(int(p.numel()) for p in net.parameters()),
        }
    net.zero_grad(set_to_none=True)

    if sim is not None:
        total_roots = int(sim.E)
        hero_rows = torch.stack(
            (
                torch.arange(total_roots, device=sim.device, dtype=torch.long),
                torch.zeros(total_roots, device=sim.device, dtype=torch.long),
            ),
            dim=1,
        )
        with torch.no_grad():
            advisory = sim.advisory_mask()[:, 0]
            fallback = ~advisory.any(dim=1, keepdim=True)
            fallback_mask = torch.tensor(
                [[True, True, True, False, False, False]],
                dtype=torch.bool,
                device=sim.device,
            )
            default_legal = torch.where(fallback, fallback_mask.expand_as(advisory), advisory)
            features = (
                gpu_v8_decide(
                    sim,
                    hero_rows,
                    torch.zeros((total_roots, 6), device=sim.device),
                    advisory,
                    float(OPEN_STOP_V8_LAMBDA),
                    features=True,
                )
                .feature_planes()
                .reshape(total_roots, 30)
            )
            tac_all, strat_all, scalars_all = build_observation(sim, hero_rows, features)
            tac_all = tac_all.to(dev)
            strat_all = strat_all.to(dev)
            scalars_all = scalars_all.to(dev)
            legal_all = (
                legal_mask.to(device=dev, dtype=torch.bool)
                if legal_mask is not None
                else default_legal.to(device=dev, dtype=torch.bool)
            )
    else:
        gen = torch.Generator(device="cpu").manual_seed(20261109)
        total_roots = 16
        tac_all = torch.randint(
            0, 256, (total_roots, local_channels, 31, 31), generator=gen, dtype=torch.uint8
        ).to(dev)
        strat_all = torch.randint(
            0, 256, (total_roots, 4, 37, 37), generator=gen, dtype=torch.uint8
        ).to(dev)
        scalars_all = torch.rand((total_roots, n_scalars), generator=gen).to(dev)
        legal_all = (
            legal_mask.to(device=dev, dtype=torch.bool)
            if legal_mask is not None
            else torch.ones((total_roots, 6), dtype=torch.bool, device=dev)
        )

    with torch.no_grad():
        q_init = net(tac_all, strat_all, scalars_all)
        q_legal_init = q_init.masked_fill(~legal_all, -torch.inf)
        bare_all = (
            torch.as_tensor(a_0, dtype=torch.long, device=dev)
            if a_0 is not None
            else q_legal_init.argmax(dim=1)
        )
        if safe_mask is not None:
            safe_all = safe_mask.to(device=dev, dtype=torch.bool) & legal_all
        elif a_star is not None:
            astar_given = torch.as_tensor(a_star, dtype=torch.long, device=dev)
            safe_all = torch.zeros_like(legal_all)
            valid_given = (astar_given >= 0) & (astar_given < 6)
            valid_rows = valid_given.nonzero(as_tuple=False).squeeze(1)
            safe_all[valid_rows, astar_given[valid_rows]] = True
            safe_all &= legal_all
        else:
            # Construct a synthetic rescue partition per sample: top legal Q action is unsafe,
            # second-best legal Q action is certified safe.
            q_second = q_legal_init.clone()
            all_idx = torch.arange(total_roots, device=dev, dtype=torch.long)
            q_second[all_idx, bare_all] = -torch.inf
            second_best = q_second.argmax(dim=1)
            safe_all = torch.zeros_like(legal_all)
            safe_all[all_idx, second_best] = True
            safe_all &= legal_all

        unsafe_all = (
            (unsafe_mask.to(device=dev, dtype=torch.bool) & legal_all & ~safe_all)
            if unsafe_mask is not None
            else (legal_all & ~safe_all)
        )
        both_safe_and_unsafe = safe_all.any(dim=1) & unsafe_all.any(dim=1)
        if not bool(both_safe_and_unsafe.any().item()):
            raise ValueError(
                "margin_loss_gradient_check requires at least one root with both safe and "
                "unsafe legal actions"
            )
        all_idx = torch.arange(total_roots, device=dev, dtype=torch.long)
        bare_is_safe_all = safe_all[all_idx, bare_all.clamp(0, 5)]
        rescue_candidates = both_safe_and_unsafe & (~bare_is_safe_all)
        selected_mask = (
            rescue_candidates if bool(rescue_candidates.any().item()) else both_safe_and_unsafe
        )
        selected_rows = selected_mask.nonzero(as_tuple=False).squeeze(1)

    e_count = int(selected_rows.numel())
    tac = tac_all.index_select(0, selected_rows)
    strat = strat_all.index_select(0, selected_rows)
    scalars = scalars_all.index_select(0, selected_rows)
    safe_sel = safe_all.index_select(0, selected_rows)
    unsafe_sel = unsafe_all.index_select(0, selected_rows)

    q = net(tac, strat, scalars)
    batch_idx = torch.arange(e_count, device=dev, dtype=torch.long)
    if a_star is not None:
        astar_full = torch.as_tensor(a_star, dtype=torch.long, device=dev)
        astar_t = astar_full.index_select(0, selected_rows)
        invalid_star = (astar_t < 0) | (astar_t >= 6) | (~safe_sel[batch_idx, astar_t.clamp(0, 5)])
        if bool(invalid_star.any().item()):
            fallback_star = q.detach().masked_fill(~safe_sel, -torch.inf).argmax(dim=1)
            astar_t = torch.where(invalid_star, fallback_star, astar_t)
    else:
        astar_t = q.detach().masked_fill(~safe_sel, -torch.inf).argmax(dim=1)

    q_unsafe_masked = q.masked_fill(~unsafe_sel, -torch.inf)
    q_best_unsafe, a_unsafe_t = q_unsafe_masked.max(dim=1)
    q_star = q[batch_idx, astar_t]
    margins = float(delta) - (q_star - q_best_unsafe)
    per_sample_loss = torch.clamp(margins, min=0.0)
    active_samples = int((per_sample_loss > 0).sum().item())
    loss = per_sample_loss.mean()
    loss.backward()

    param_records: list[dict[str, Any]] = []
    all_trainable_grads_present = True
    all_grads_finite = True
    total_param_tensors = 0
    total_scalar_params = 0
    zero_param_tensors = 0
    zero_scalar_params = 0
    non_value_zero_param_tensors = 0
    non_value_zero_scalar_params = 0
    value_head_scalar_params = 0
    value_head_frozen_tensors = 0

    for name, param in net.named_parameters():
        total_param_tensors += 1
        numel = int(param.numel())
        total_scalar_params += numel
        is_value = name.startswith("value.")
        if is_value:
            value_head_scalar_params += numel
            if not param.requires_grad:
                value_head_frozen_tensors += 1
        grad = param.grad
        if grad is None:
            if param.requires_grad:
                all_trainable_grads_present = False
                all_grads_finite = False
                if not is_value:
                    non_value_zero_param_tensors += 1
                    non_value_zero_scalar_params += numel
            param_records.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "numel": numel,
                    "requires_grad": bool(param.requires_grad),
                    "has_grad": False,
                    "all_finite": not bool(param.requires_grad),
                    "all_zero": True,
                    "zero_elements": numel,
                    "grad_l2_norm": None,
                    "grad_max_abs": None,
                }
            )
            zero_param_tensors += 1
            zero_scalar_params += numel
            continue

        is_finite = bool(torch.isfinite(grad).all().item())
        if not is_finite:
            all_grads_finite = False
        zero_elems = int((grad == 0).sum().item())
        is_all_zero = bool(zero_elems == numel)
        if is_all_zero:
            zero_param_tensors += 1
        zero_scalar_params += zero_elems

        if not is_value:
            if is_all_zero:
                non_value_zero_param_tensors += 1
            non_value_zero_scalar_params += zero_elems

        param_records.append(
            {
                "name": name,
                "shape": list(param.shape),
                "numel": numel,
                "requires_grad": bool(param.requires_grad),
                "has_grad": True,
                "all_finite": is_finite,
                "all_zero": is_all_zero,
                "zero_elements": zero_elems,
                "grad_l2_norm": float(grad.detach().norm(p=2).item()),
                "grad_max_abs": float(grad.detach().abs().max().item()),
            }
        )

    all_non_value_grads_finite_and_nonzero = bool(
        all_trainable_grads_present
        and all_grads_finite
        and non_value_zero_param_tensors == 0
        and active_samples > 0
    )
    if not all_non_value_grads_finite_and_nonzero:
        raise AssertionError(
            "Registered hinge gradient check failed: "
            f"all_trainable_grads_present={all_trainable_grads_present}, "
            f"all_grads_finite={all_grads_finite}, "
            f"non_value_zero_param_tensors={non_value_zero_param_tensors}, "
            f"active_samples={active_samples}"
        )

    return {
        "loss_form": "registered_rescue_pairwise_hinge_vs_best_unsafe_legal",
        "delta": float(delta),
        "total_input_roots": int(total_roots),
        "selected_root_indices": [int(x) for x in selected_rows.detach().cpu().tolist()],
        "selected_a_star": [int(x) for x in astar_t.detach().cpu().tolist()],
        "selected_a_best_unsafe": [int(x) for x in a_unsafe_t.detach().cpu().tolist()],
        "batch_size": int(e_count),
        "active_margin_samples": int(active_samples),
        "loss_value": float(loss.detach().item()),
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "value_head_frozen": bool(freeze_value_head and value_head_frozen_tensors == 2),
        "freeze_summary": freeze_info,
        "all_grads_present": bool(all_trainable_grads_present),
        "all_grads_finite": bool(all_grads_finite),
        "all_non_value_grads_finite_and_nonzero": bool(all_non_value_grads_finite_and_nonzero),
        "total_param_tensors": int(total_param_tensors),
        "total_scalar_params": int(total_scalar_params),
        "exact_zero_param_tensors": int(zero_param_tensors),
        "exact_zero_scalar_params": int(zero_scalar_params),
        "value_head_zero_param_tensors": int(zero_param_tensors - non_value_zero_param_tensors),
        "value_head_scalar_params": int(value_head_scalar_params),
        "non_value_zero_param_tensors": int(non_value_zero_param_tensors),
        "non_value_zero_scalar_params": int(non_value_zero_scalar_params),
        "dueling_value_cancellation_note": (
            "In Ego2sNet, Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a')). "
            "The pairwise hinge Q(s, a*) - max_{a in legal\\safe} Q(s, a) cancels V(s) "
            "identically; freeze_ego2s_value_head freezes value.weight [1, 256] and "
            "value.bias [1] (257 scalars) while all 24 non-value parameter tensors "
            "(1,638,006 scalars) receive finite non-zero gradients."
        ),
        "parameters": param_records,
    }
