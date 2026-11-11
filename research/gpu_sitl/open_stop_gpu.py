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

import copy
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
from numba import njit

from research.bl_served_search import ceiling
from research.gpu_sim_spike import ego_device, r2_cache, v8_device
from research.gpu_sim_spike.anchors_device import choose_anchor
from research.gpu_sim_spike.device_policy import DevicePolicy, _masked_argmax
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
from src.core.mechanics_constants import HEADON_SIZE_RATIO
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
    """CPU queue fallback for ``v8_device._flood``; raises if ever called on CUDA."""
    if start.device.type == "cuda":
        raise RuntimeError(
            "_cpu_queue_flood must never be invoked on a CUDA tensor; "
            "CUDA runs must use native flood_cuda kernels"
        )
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


DEFAULT_CUDA_ARCH_LIST = "8.0;8.9;9.0"
REQUIRED_CUDA_ARCHS = ("8.0", "8.9", "9.0")
_CREATED_EXT_DIRS: set[str] = set()
_LAST_SCOPED_BUILD_KEY: tuple[str, str] | None = None


def _resolve_scoped_torch_extensions_dir(arch_tag: str) -> str:
    """Return an architecture-scoped TORCH_EXTENSIONS_DIR so single-arch builds cannot
    collide with or reuse stale multi-arch or different-arch extension builds."""
    base_ext_dir = os.environ.get("TORCH_EXTENSIONS_DIR", "").strip()
    subdir_name = f"gpu_sitl_{arch_tag}"
    if not base_ext_dir:
        default_root = (
            Path.home()
            / ".cache"
            / "torch_extensions"
            / f"py{sys.version_info.major}{sys.version_info.minor}_{torch.version.cuda or 'cpu'}"
        )
        scoped_path = default_root / subdir_name
    else:
        base_path = Path(base_ext_dir)
        if base_path.name == subdir_name:
            scoped_path = base_path
        elif base_path.name.startswith("gpu_sitl_sm_"):
            scoped_path = base_path.parent / subdir_name
        else:
            scoped_path = base_path / subdir_name
    scoped_str = str(scoped_path)
    if scoped_str not in _CREATED_EXT_DIRS:
        scoped_path.mkdir(parents=True, exist_ok=True)
        _CREATED_EXT_DIRS.add(scoped_str)
    return scoped_str


def _load_event_cuda_extension() -> Any:
    """Pre-load ``research.gpu_sim_spike.event_cuda._EXTENSION`` under the active single-arch
    scope so ``event_cuda.append_fifo`` never compiles lazily for 3 architectures on the first
    snake death."""
    from torch.utils import cpp_extension

    from research.gpu_sim_spike import event_cuda

    if event_cuda._EXTENSION is not None:
        return event_cuda._EXTENSION
    if (
        not hasattr(torch._C, "_cuda_getDeviceCount")
        and getattr(cpp_extension.load, "__module__", "") == "torch.utils.cpp_extension"
    ):
        return object()
    root = Path(event_cuda.__file__).resolve().parent
    event_cuda._EXTENSION = cpp_extension.load(
        name="snake_dqn_a0_event_v1",
        sources=[str(root / "event_cuda.cpp"), str(root / "event_cuda.cu")],
        extra_cuda_cflags=["-O3", "--threads", "2"],
        extra_cflags=["-O3"],
        verbose=False,
    )
    return event_cuda._EXTENSION


def _load_move_cuda_extension() -> Any:
    """Pre-load ``research.gpu_sim_spike.move_cuda._EXTENSION`` under the active single-arch
    scope so ``CTDeviceSim`` with ``move_backend='indexed'`` never compiles lazily on step 1."""
    from torch.utils import cpp_extension

    from research.gpu_sim_spike import move_cuda

    if move_cuda._EXTENSION is not None:
        return move_cuda._EXTENSION
    if (
        not hasattr(torch._C, "_cuda_getDeviceCount")
        and getattr(cpp_extension.load, "__module__", "") == "torch.utils.cpp_extension"
    ):
        return object()
    root = Path(move_cuda.__file__).resolve().parent
    move_cuda._EXTENSION = cpp_extension.load(
        name="snake_dqn_move_v1",
        sources=[str(root / "move_cuda.cpp"), str(root / "move_cuda.cu")],
        extra_cuda_cflags=["-O3", "--fmad=false", "--threads", "2"],
        extra_cflags=["-O3"],
        verbose=False,
    )
    return move_cuda._EXTENSION


def ensure_cuda_initialized(device: torch.device | str) -> torch.device:
    """Normalize ``device`` and initialize CUDA + ``CUDACachingAllocator`` when targeting CUDA.

    In PyTorch 2.9 (`torch/cuda/memory.py:374-376`, `torch/_utils.py:831-875`),
    ``torch.cuda.reset_peak_memory_stats(dev)`` calls ``_get_device_index(dev, optional=True)``.
    When ``dev`` carries an explicit index (``torch.device('cuda', 0)``), ``_get_device_index``
    returns ``0`` directly without calling ``_get_current_device_index()`` ->
    ``torch.cuda.current_device()`` -> ``torch.cuda._lazy_init()``. Calling
    ``torch._C._cuda_resetPeakMemoryStats(0)`` before ``torch._C._cuda_init()`` then fails in
    ``CUDACachingAllocator::resetPeakStats`` with ``RuntimeError: Invalid device argument``
    because ``device_allocator`` is still empty. Initializing CUDA here before any indexed
    device/memory/RNG call prevents that failure across smoke, harvest, and pilot paths.
    """
    dev = torch.device(device)
    if dev.type in {"cuda", "mps"} and dev.index is None:
        dev = torch.device(dev.type, 0)
    if dev.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.set_float32_matmul_precision("highest")
        if torch.cuda.is_available():
            if (
                hasattr(torch._C, "_cuda_getDeviceCount")
                or getattr(torch.cuda.init, "__module__", "") != "torch.cuda"
            ):
                torch.cuda.init()
                if (
                    hasattr(torch._C, "_cuda_setDevice")
                    or getattr(torch.cuda.set_device, "__module__", "") != "torch.cuda"
                ):
                    torch.cuda.set_device(dev)
    return dev


def normalize_device(device: torch.device | str) -> torch.device:
    """Normalize ``cuda`` -> ``cuda:0`` and ``mps`` -> ``mps:0`` and initialize CUDA if active."""
    return ensure_cuda_initialized(device)


def verify_cuda_native_extensions(device: torch.device | str) -> dict[str, Any]:
    """Fail closed on CUDA if native ``flood_cuda`` or ``vector_geometry`` is unavailable,
    if ``TORCH_CUDA_ARCH_LIST`` omits A100/L4/H100 (8.0/8.9/9.0), if the active GPU capability
    from ``torch.cuda.get_device_capability`` is not in ``REQUIRED_CUDA_ARCHS``, or if CPU
    flood overrides are active.

    Scopes ``TORCH_CUDA_ARCH_LIST`` to the active device's single compute capability and
    ``TORCH_EXTENSIONS_DIR`` to an arch-tagged subdirectory for the actual
    ``cpp_extension.load`` build, then restores the declared ``TORCH_CUDA_ARCH_LIST``.
    """
    global _LAST_SCOPED_BUILD_KEY
    arch_list_raw = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if not arch_list_raw:
        os.environ["TORCH_CUDA_ARCH_LIST"] = DEFAULT_CUDA_ARCH_LIST
        arch_list_raw = DEFAULT_CUDA_ARCH_LIST
    arch_tokens = {
        tok.strip().removesuffix("+PTX").removesuffix("a")
        for part in arch_list_raw.replace(",", ";").split(";")
        for tok in part.split()
        if tok.strip()
    }
    missing_archs = [arch for arch in REQUIRED_CUDA_ARCHS if arch not in arch_tokens]
    dev = ensure_cuda_initialized(device)
    if dev.type != "cuda":
        return {
            "device": str(dev),
            "cuda_active": False,
            "cuda_initialized": False,
            "flood_cuda_loaded": False,
            "vector_geometry_loaded": False,
            "event_cuda_loaded": False,
            "torch_cuda_arch_list": arch_list_raw,
            "declared_torch_cuda_arch_list": arch_list_raw,
            "effective_build_cuda_arch": None,
            "device_capability": None,
            "scoped_torch_extensions_dir": None,
            "arch_list_covers_a100_l4_h100": len(missing_archs) == 0,
            "cpu_overrides_absent": bool(
                v8_device._flood is _ORIG_V8_FLOOD
                and ego_device._component_sizes is _ORIG_EGO_COMPONENTS
            ),
        }
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    if missing_archs:
        raise RuntimeError(
            f"TORCH_CUDA_ARCH_LIST={arch_list_raw!r} is missing required GPU architectures "
            f"{missing_archs} (must cover A100 8.0, L4 8.9, and H100 9.0)"
        )
    if (
        v8_device._flood is not _ORIG_V8_FLOOD
        or ego_device._component_sizes is not _ORIG_EGO_COMPONENTS
    ):
        raise RuntimeError(
            "CPU flood/component overrides are active while running on a CUDA device; "
            "silent CPU fallback is forbidden"
        )
    geom_env = os.environ.get("GPU_SIM_VECTOR_GEOMETRY", "").strip().lower()
    flood_env = os.environ.get("GPU_SIM_VECTOR_FLOOD", "").strip().lower()
    if geom_env not in {"native", "exact"}:
        raise RuntimeError(
            f"GPU_SIM_VECTOR_GEOMETRY must be 'native' or 'exact' on CUDA, got {geom_env!r}"
        )
    if flood_env != "static":
        raise RuntimeError(f"GPU_SIM_VECTOR_FLOOD must be 'static' on CUDA, got {flood_env!r}")

    cap_raw = torch.cuda.get_device_capability(dev)
    cap_major, cap_minor = int(cap_raw[0]), int(cap_raw[1])
    effective_build_arch = f"{cap_major}.{cap_minor}"
    if effective_build_arch not in REQUIRED_CUDA_ARCHS or effective_build_arch not in arch_tokens:
        raise RuntimeError(
            f"Unsupported CUDA compute capability {effective_build_arch!r} from "
            f"torch.cuda.get_device_capability({dev!r}); expected one of "
            f"{REQUIRED_CUDA_ARCHS} covered by TORCH_CUDA_ARCH_LIST={arch_list_raw!r}"
        )

    arch_tag = f"sm_{cap_major}{cap_minor}"
    scoped_ext_dir = _resolve_scoped_torch_extensions_dir(arch_tag)
    build_key = (effective_build_arch, scoped_ext_dir)

    from research.gpu_sim_spike import event_cuda, flood_cuda, move_cuda, vector_geometry

    has_real_cuda = bool(
        hasattr(torch._C, "_cuda_getDeviceCount") and torch._C._cuda_getDeviceCount() > 0
    )
    is_cold_arch = _LAST_SCOPED_BUILD_KEY != build_key
    if is_cold_arch:
        if hasattr(flood_cuda._load, "cache_clear"):
            flood_cuda._load.cache_clear()
        if hasattr(vector_geometry._load_exact_extension, "cache_clear"):
            vector_geometry._load_exact_extension.cache_clear()
        if _LAST_SCOPED_BUILD_KEY is not None:
            flood_cuda._EXTENSION = None
            flood_cuda._ERROR = None
            vector_geometry._EXTENSION = None
            event_cuda._EXTENSION = None
            move_cuda._EXTENSION = None

    os.environ["TORCH_EXTENSIONS_DIR"] = scoped_ext_dir
    os.environ["TORCH_CUDA_ARCH_LIST"] = effective_build_arch
    prev_max_jobs = os.environ.get("MAX_JOBS")
    move_ext: Any = object()
    try:
        if is_cold_arch and has_real_cuda:
            if prev_max_jobs is None:
                os.environ["MAX_JOBS"] = "3"
            loaders: tuple[Callable[[], Any], ...] = (
                flood_cuda._load,
                vector_geometry._load_exact_extension,
                _load_event_cuda_extension,
                _load_move_cuda_extension,
            )
            with ThreadPoolExecutor(max_workers=len(loaders)) as pool:
                futures = [pool.submit(fn) for fn in loaders]
                flood_ext, vec_ext, event_ext, move_ext = [fut.result() for fut in futures]
        else:
            flood_ext = flood_cuda._load()
            vec_ext = vector_geometry._load_exact_extension()
            event_ext = _load_event_cuda_extension()
            if has_real_cuda:
                move_ext = _load_move_cuda_extension()
    finally:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list_raw
        if prev_max_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = prev_max_jobs

    if flood_ext is None or vec_ext is None or event_ext is None or move_ext is None:
        raise RuntimeError(
            f"Native CUDA extensions failed to load (flood_cuda={flood_ext!r}, "
            f"vector_geometry={vec_ext!r}, event_cuda={event_ext!r}, move_cuda={move_ext!r}); "
            "silent fallback is forbidden"
        )
    _LAST_SCOPED_BUILD_KEY = build_key
    return {
        "device": str(dev),
        "cuda_active": True,
        "cuda_initialized": bool(
            torch.cuda.is_initialized() if hasattr(torch._C, "_cuda_getDeviceCount") else True
        ),
        "flood_cuda_loaded": True,
        "vector_geometry_loaded": True,
        "event_cuda_loaded": True,
        "move_cuda_loaded": bool(has_real_cuda),
        "gpu_sim_vector_geometry": geom_env,
        "gpu_sim_vector_flood": flood_env,
        "torch_cuda_arch_list": arch_list_raw,
        "declared_torch_cuda_arch_list": arch_list_raw,
        "effective_build_cuda_arch": effective_build_arch,
        "device_capability": [cap_major, cap_minor],
        "scoped_torch_extensions_dir": scoped_ext_dir,
        "arch_list_covers_a100_l4_h100": True,
        "cpu_overrides_absent": True,
    }


def select_a_star_nb_on_device(
    root_q_t: torch.Tensor,
    safe_mask: torch.Tensor,
    bare_actions: torch.Tensor,
) -> torch.Tensor:
    """Compute ``a^*(s)`` on device matching :func:`evaluation_order_nb` (`open-stop-h32-nb`).

    Order of preference among legal certified-safe actions (``safe_mask``):
    1. ``bare`` action ``a_b8(s)`` if it is certified safe.
    2. Highest-Q certified-safe non-boost action in ``{0, 1, 2}`` (ties broken by lowest index).
    3. Highest-Q certified-safe boost action in ``{3, 4, 5}`` (ties broken by lowest index).
    4. ``-1`` if no legal action is certified safe.
    """
    r_count = int(root_q_t.shape[0])
    device = root_q_t.device
    safe_mask = safe_mask.to(device=device, dtype=torch.bool)
    row_arange = torch.arange(r_count, device=device, dtype=torch.long)
    bare_clamped = bare_actions.to(device=device, dtype=torch.long).clamp(0, 5)
    bare_is_safe = safe_mask[row_arange, bare_clamped]
    has_safe = safe_mask.any(dim=1)

    safe_nb_mask = safe_mask.clone()
    safe_nb_mask[:, 3:] = False
    has_safe_nb = safe_nb_mask.any(dim=1, keepdim=True)
    effective_safe_mask = torch.where(has_safe_nb, safe_nb_mask, safe_mask)
    best_safe_nb_act = root_q_t.masked_fill(~effective_safe_mask, -torch.inf).argmax(dim=1)
    return torch.where(
        bare_is_safe,
        bare_clamped,
        torch.where(has_safe, best_safe_nb_act, torch.full_like(bare_clamped, -1)),
    )


@contextmanager
def _maybe_cpu_fast_flood(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    prev_flood = v8_device._flood
    prev_comp = ego_device._component_sizes

    def _cpu_queue_components(sim: Any, rows: torch.Tensor, ttl: torch.Tensor) -> torch.Tensor:
        if sim.device.type == "cuda":
            raise RuntimeError(
                "_cpu_queue_components must never be invoked on a CUDA simulator; "
                "CUDA runs must use native flood_cuda.components"
            )
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

    def _body_view(self, sim: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cached_sim = getattr(self, "_cached_body_sim", None)
        cached_view = getattr(self, "_cached_body_view", None)
        if cached_sim is sim and cached_view is not None:
            return cached_view
        return super()._body_view(sim)

    def _free_space_for_rows(self, sim: Any, rows: torch.Tensor) -> torch.Tensor:
        assert self.params is not None
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.device).reshape(-1, 2)
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
                (oy.clamp(0, sim.grid_h - 1) * sim.grid_w + ox.clamp(0, sim.grid_w - 1)).to(
                    torch.long
                ),
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

        cap = torch.minimum(
            torch.full_like(length, FREE_SPACE_BFS_CAP),
            torch.clamp(length * 2, min=FREE_SPACE_MIN_CAP),
        ).to(torch.long)

        if (
            self.device is not None
            and self.device.type == "cuda"
            and os.environ.get("GPU_SIM_VECTOR_FLOOD", "").strip().lower() == "static"
            and type(starts).__name__ != "FakeTensor"
            and int(sim.E) > 0
            and v8_device._flood is _ORIG_V8_FLOOD
        ):
            from research.gpu_sim_spike import flood_cuda

            starts_full = torch.full((sim.E, sim.S, 3, 2), -1, dtype=torch.long, device=self.device)
            cap_full = torch.ones((sim.E, sim.S), dtype=torch.long, device=self.device)
            starts_full[er, sr] = starts
            cap_full[er, sr] = cap
            count = flood_cuda.static_capped(starts_full, occupied_all.contiguous(), cap_full)[
                er, sr
            ]
            return self._normalize_free(count, cap)

        blocked = occupied_all[er][:, None].expand(n, 3, -1, -1)
        count = v8_device._flood(
            starts,
            blocked,
            cap[:, None].expand(n, 3),
            max_steps=FREE_SPACE_BFS_CAP,
        )
        return self._normalize_free(count, cap)

    def _features_for_rows(self, sim: Any, rows: torch.Tensor) -> torch.Tensor:
        assert self.params is not None
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.device).reshape(-1, 2)
        if not len(rows):
            return torch.zeros((0, self.params.input_size), dtype=torch.float32, device=self.device)
        if self.device is not None and self.device.type == "cuda":
            fs_rows = self._free_space_for_rows(sim, rows) if self.params.use_free_space else None
            fs_full = None
            if fs_rows is not None:
                fs_full = torch.zeros((sim.E, sim.S, 3), dtype=torch.float32, device=self.device)
                fs_full[rows[:, 0], rows[:, 1]] = fs_rows
            return self._features_rows(sim, rows, update_memory=True, free_space=fs_full)

        self._cached_body_sim = sim
        self._cached_body_view = super()._body_view(sim)
        try:
            fs_rows = self._free_space_for_rows(sim, rows) if self.params.use_free_space else None
            fs_full = None
            if fs_rows is not None:
                fs_full = torch.zeros((sim.E, sim.S, 3), dtype=torch.float32, device=self.device)
                fs_full[rows[:, 0], rows[:, 1]] = fs_rows
            chunk_sz = max(int(FEATURE_ROW_CHUNK), 256)
            if len(rows) <= chunk_sz:
                return self._features_rows(sim, rows, update_memory=True, free_space=fs_full)
            parts = []
            for start in range(0, len(rows), chunk_sz):
                stop = min(start + chunk_sz, len(rows))
                parts.append(
                    self._features_rows(
                        sim,
                        rows[start:stop],
                        update_memory=True,
                        free_space=fs_full,
                    )
                )
            return torch.cat(parts, dim=0)
        finally:
            self._cached_body_sim = None
            self._cached_body_view = None

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


def _covered_by_other_r2(sim: Any) -> torch.Tensor:
    """Return ``(E, S)`` count of cells covered by at least one slot ``!= s`` under R2."""

    def compute() -> torch.Tensor:
        slot_occ = r2_cache.slot_occupancy(sim)
        occ_cnt = r2_cache.occupancy_counts(sim)
        total_occ = (occ_cnt != 0).flatten(1).sum(1)
        unique_own = (slot_occ & (occ_cnt[:, None] == 1)).flatten(2).sum(2)
        return total_occ[:, None] - unique_own

    return r2_cache.memo(sim, "covered_by_other", compute)


def _world_opp_reach(sim: Any, r2: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(reach_w, rvalid_w_base, bound_w)`` over ``(E, S, 7, ...)`` once per state."""

    def compute() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = sim.bodies.device
        oh_w = sim.heads()
        od_w = sim.direction
        ol_w = sim.length
        oa_w = sim.alive & (sim.seg_count > 0)
        od3_w = (od_w[:, :, None] + torch.arange(3, device=device)[None, None, :] - 1) % 4
        table_x, table_y = v8_device._deltas(device, r2)
        odelta_w = torch.stack((table_x[od3_w], table_y[od3_w]), -1)
        reach1_w, reach2_w = oh_w[:, :, None] + odelta_w, oh_w[:, :, None] + 2 * odelta_w
        reach_w = torch.cat((oh_w[:, :, None], reach1_w, reach2_w), 2)
        rvalid_w_base = torch.cat(
            (
                oa_w[:, :, None],
                oa_w[:, :, None].expand(-1, -1, 3),
                (oa_w & (ol_w >= int(sim.cfg.min_boost_length)))[:, :, None].expand(-1, -1, 3),
            ),
            2,
        )
        bound_w = (ol_w.clamp_min(1) + 1)[:, :, None].expand_as(rvalid_w_base)
        return reach_w, rvalid_w_base, bound_w

    if r2:
        return r2_cache.memo(sim, "v8_world_opp_reach", compute)
    return compute()


def _post_geom_with_table(
    sim: Any,
    env: torch.Tensor,
    slot: torch.Tensor,
    actions: torch.Tensor,
    heads: torch.Tensor,
    r2: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lightweight ``(head_pos, first_pos, lpost, boosted)`` matching ``v8_device._post_body``."""
    length_a = sim.length[env, slot][:, None].expand_as(actions)
    bf_a = sim.boost_frames[env, slot][:, None].expand_as(actions)
    heading = sim.direction[env, slot][:, None]
    hd_a = (heading + actions.remainder(3) - 1) % 4
    table_x, table_y = v8_device._deltas(env.device, r2)
    delta_a = torch.stack((table_x[hd_a], table_y[hd_a]), -1)
    boosted = (actions >= 3) & (length_a >= int(sim.cfg.min_boost_length))
    lpost = torch.where(
        boosted & (bf_a + 1 >= int(sim.cfg.boost_length_cost_frames)) & (length_a > 1),
        length_a - 1,
        length_a,
    )
    heads3 = heads[:, None, :]
    head_pos = heads3 + delta_a * torch.where(boosted, 2, 1)[..., None]
    first_pos = torch.where(boosted[..., None], heads3 + delta_a, heads3.expand_as(head_pos))
    return head_pos, first_pos, lpost, boosted


def _compute_release_area_active_lanes(
    sim: Any,
    env: torch.Tensor,
    slot: torch.Tensor,
    other_occ: torch.Tensor,
    area_cap: torch.Tensor,
    head_pos: torch.Tensor,
    first_pos: torch.Tensor,
    lpost: torch.Tensor,
    boosted: torch.Tensor,
    *,
    act_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact v7 tail-release flood over active ``(row, action)`` lanes only.

    Skips out-of-bounds and ``other_occ``-blocked starts (exact ``0``), deduplicates
    unboosted ``a >= 3`` columns against ``a - 3``, and buckets active lanes by
    ``area_cap <= 256`` vs ``area_cap > 256`` so ``6%`` mature spliced heroes do not
    inflate ``span`` or ``max_steps`` across the ``94%`` online short-snake lanes.
    """
    device = head_pos.device
    ns = int(head_pos.shape[0])
    if ns == 0:
        return torch.zeros((0, 6), dtype=torch.long, device=device)
    h, w = int(sim.grid_h), int(sim.grid_w)
    if not v8_device._host_scalar_safe(head_pos):
        r2 = r2_cache.enabled(sim)
        span = r2_cache.window(sim) if r2 else int(sim.cap)
        own_c, own_v = v8_device._own_cells(sim, env, slot, span)
        actions6 = torch.arange(6, device=device)[None].expand(ns, -1)
        pc_s, pv_s, lp_s, _ = v8_device._post_body(
            sim, env, slot, actions6, own_c, own_v, span if r2 else None, r2
        )
        i_idx = torch.arange(span, device=device)[None, None, :]
        rel_values = (lp_s[:, :, None] - i_idx + 1).clamp_min(0)
        rel_map = torch.zeros((ns, 6, h * w), dtype=torch.long, device=device)
        px, py = pc_s[..., 0], pc_s[..., 1]
        good = pv_s & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        pind = py.clamp(0, h - 1) * w + px.clamp(0, w - 1)
        rel_map.scatter_reduce_(
            2,
            pind,
            torch.where(good, rel_values, torch.zeros_like(rel_values)),
            reduce="amax",
            include_self=True,
        )
        hi = head_pos[..., 1].clamp(0, h - 1) * w + head_pos[..., 0].clamp(0, w - 1)
        rel_map.scatter_(
            2, hi[..., None], torch.zeros((ns, 6, 1), dtype=torch.long, device=device)
        )
        hp_eff = (
            torch.where(act_mask[..., None], head_pos, torch.full_like(head_pos, -1))
            if act_mask is not None
            else head_pos
        )
        return v8_device._flood(
            hp_eff,
            other_occ[:, None].expand(-1, 6, -1, -1),
            area_cap[:, None].expand(-1, 6),
            rel_map.reshape(ns, 6, h, w),
            max_steps=int(v8_device._AREA_MAX),
        )

    hx, hy = head_pos[..., 0], head_pos[..., 1]
    in_b = (hx >= 0) & (hx < w) & (hy >= 0) & (hy < h)
    hi_all = hy.clamp(0, h - 1) * w + hx.clamp(0, w - 1)
    oo_at_head = other_occ.reshape(ns, h * w).gather(1, hi_all)
    valid_start = in_b & (~oo_at_head)
    if act_mask is not None:
        valid_start = valid_start & act_mask

    dup_of_normal = (~boosted[:, 3:]) & valid_start[:, 3:] & valid_start[:, :3]
    compute_mask = valid_start.clone()
    compute_mask[:, 3:] &= ~dup_of_normal

    area_out = torch.zeros((ns, 6), dtype=torch.long, device=device)
    lane_r, lane_a = compute_mask.nonzero(as_tuple=True)
    m_total = int(lane_r.numel())
    if m_total > 0:

        def _eval_lane_group(gr: torch.Tensor, ga: torch.Tensor) -> torch.Tensor:
            gm = int(gr.numel())
            if gm == 0:
                return torch.zeros(0, dtype=torch.long, device=device)
            env_m = env.index_select(0, gr)
            slot_m = slot.index_select(0, gr)
            hp_m = head_pos[gr, ga]
            fp_m = first_pos[gr, ga]
            lp_m = lpost[gr, ga]
            bst_m = boosted[gr, ga]
            len_m = sim.length[env_m, slot_m]
            cnt_m = sim.seg_count[env_m, slot_m]
            span_m = min(int(sim.cap), max(2, int(cnt_m.max().item()) + 2))
            ln_m = torch.minimum(cnt_m + 1, len_m)
            ln_m = torch.where(bst_m, torch.minimum(ln_m + 1, len_m), ln_m)
            ln_m = torch.minimum(ln_m, lp_m)
            nh_m = torch.where(bst_m, torch.full_like(len_m, 2), torch.ones_like(len_m))
            i_idx = torch.arange(span_m, device=device)[None, :]
            old_idx = (i_idx - nh_m[:, None]).clamp_min(0)
            ring_m = (sim.head_ptr[env_m, slot_m][:, None] - old_idx) % sim.cap
            old_m = sim.bodies[env_m[:, None], slot_m[:, None], ring_m]
            is_i0 = i_idx == 0
            is_i1_bst = (i_idx == 1) & bst_m[:, None]
            pc_m = torch.where(
                is_i0[..., None],
                hp_m[:, None, :],
                torch.where(is_i1_bst[..., None], fp_m[:, None, :], old_m),
            )
            pv_m = i_idx < ln_m[:, None]
            rel_dtype = torch.int32 if device.type == "mps" else torch.long
            rel_vals = (lp_m[:, None] - i_idx + 1).clamp_min(0).to(rel_dtype)
            rel_flat = torch.zeros((gm, h * w), dtype=rel_dtype, device=device)
            px, py = pc_m[..., 0], pc_m[..., 1]
            good = pv_m & (px >= 0) & (px < w) & (py >= 0) & (py < h)
            pind = py.clamp(0, h - 1) * w + px.clamp(0, w - 1)
            rel_flat.scatter_reduce_(
                1,
                pind,
                torch.where(good, rel_vals, torch.zeros_like(rel_vals)),
                reduce="amax",
                include_self=True,
            )
            hi_m = hp_m[:, 1].clamp(0, h - 1) * w + hp_m[:, 0].clamp(0, w - 1)
            rel_flat.scatter_(
                1, hi_m[:, None], torch.zeros((gm, 1), dtype=rel_dtype, device=device)
            )
            rel_grid = rel_flat.to(torch.long).reshape(gm, 1, h, w)
            oo_m = other_occ.index_select(0, gr)
            ac_m = area_cap.index_select(0, gr)
            ac_max = max(1, int(ac_m.max().item()))
            if device.type == "mps":
                tight_max = min(int(v8_device._AREA_MAX), max(64, ((ac_max + 63) // 64) * 64))
            else:
                tight_max = min(int(v8_device._AREA_MAX), ac_max)
            return v8_device._flood(
                hp_m[:, None, :],
                oo_m[:, None, :, :],
                ac_m[:, None],
                rel_grid,
                max_steps=tight_max,
            ).squeeze(1)

        ac_lanes = area_cap.index_select(0, lane_r)
        small_mask = ac_lanes <= 256
        n_small = int(small_mask.sum().item())
        if 0 < n_small < m_total:
            sr = lane_r[small_mask]
            sa = lane_a[small_mask]
            lr = lane_r[~small_mask]
            la = lane_a[~small_mask]
            area_out[sr, sa] = _eval_lane_group(sr, sa)
            area_out[lr, la] = _eval_lane_group(lr, la)
        else:
            area_out[lane_r, lane_a] = _eval_lane_group(lane_r, lane_a)

    if act_mask is None:
        area_out[:, 3:] = torch.where(dup_of_normal, area_out[:, :3], area_out[:, 3:])
    else:
        area_out[:, 3:] = torch.where(
            dup_of_normal & act_mask[:, 3:], area_out[:, :3], area_out[:, 3:]
        )
    return area_out


def fast_v8_decide(
    sim: Any,
    rows: torch.Tensor,
    q: torch.Tensor,
    mask: torch.Tensor,
    lam: float = 8.0,
    features: bool = False,
    *,
    slot_layout: str | None = None,
    _only_action: bool = False,
    _only_features: bool = False,
) -> v8_device.V8DeviceResult:
    """Bit-identical faster ``v8_decide`` for hero and batched opponent decisions.

    Eliminates wasted materializations and flood launches without changing any
    field of :class:`V8DeviceResult` (unless internal ``_only_action`` /
    ``_only_features`` caller hints skip unused diagnostic fields):
    1. On CUDA with ``slot_layout`` in ``("hero", "opp_slot_major")`` and unpatched
       ``v8_device._flood``, uses ``flood_cuda.static_capped`` directly on ``(E, H, W)``
       occupancy without materializing ``(N, H, W)`` ``all_occ`` or ``(N, 3, H, W)`` grids.
    2. Computes lightweight ``(head_pos, first_pos, lpost, boosted)`` first and builds
       post-move body tails + ``landing`` floods ONLY on rows with active boost moves.
    3. Reuses world-level opponent reachability ``(E, S, 7, 2)`` across hero and all 5
       opponent slots instead of recomputing ``5x`` per step.
    4. Evaluates v7 tail-release ``area`` only on active in-bounds unblocked unique lanes
       (bucketed by ``area_cap <= 256`` vs ``> 256``), and when ``_only_features=True``
       skips unused ``float64`` ``_rerank`` passes on ``q = zeros``.
    """
    q = q.to(dtype=torch.float32)
    device, n = q.device, int(q.shape[0])
    rows = rows.to(device=device, dtype=torch.long)
    if n == 0:
        return gpu_v8_decide(sim, rows, q, mask, lam, features=features)
    env, slot = rows[:, 0], rows[:, 1]
    h, w = int(sim.grid_h), int(sim.grid_w)
    r2 = r2_cache.enabled(sim)
    span = r2_cache.window(sim) if r2 else int(sim.cap)

    old_cells_all: torch.Tensor | None = None
    old_valid_all: torch.Tensor | None = None
    other_occ_eager: torch.Tensor | None = None
    if not r2:
        old_cells_all, old_valid_all = v8_device._body_cells(sim, env)
        other_valid = old_valid_all & (
            torch.arange(sim.S, device=device)[None, :, None] != slot[:, None, None]
        )
        other_occ_eager = v8_device._paint(old_cells_all, other_valid, h, w)

    heads = v8_device._heads(sim, env, slot)
    direction = sim.direction[env, slot]
    rel3 = torch.arange(3, device=device)[None]
    hd3 = (direction[:, None] + rel3 - 1) % 4
    table_x, table_y = v8_device._deltas(device, r2)
    delta3 = torch.stack((table_x[hd3], table_y[hd3]), -1)
    starts = heads[:, None] + delta3
    length = sim.length[env, slot]
    cap = (2 * length).clamp(v8_device._CAP_MIN, v8_device._CAP_MAX)
    need = length.clamp_min(1).minimum(cap)

    e_sim = int(sim.E)
    s_sim = int(sim.S)
    can_use_static_cuda = (
        device.type == "cuda"
        and v8_device._flood is _ORIG_V8_FLOOD
        and type(starts).__name__ != "FakeTensor"
        and e_sim > 0
    )
    if can_use_static_cuda and slot_layout == "hero" and n == e_sim:
        from research.gpu_sim_spike import flood_cuda

        occ_world = (
            r2_cache.all_occupancy(sim)
            if r2
            else v8_device._paint(
                *v8_device._body_cells(sim, torch.arange(e_sim, device=device)), h, w
            )
        )
        counts = flood_cuda.static_capped(
            starts[:, None], occ_world.contiguous(), cap[:, None]
        ).squeeze(1)
    elif (
        can_use_static_cuda
        and slot_layout == "opp_slot_major"
        and s_sim > 1
        and n == e_sim * (s_sim - 1)
    ):
        from research.gpu_sim_spike import flood_cuda

        s_opp = s_sim - 1
        occ_world = (
            r2_cache.all_occupancy(sim)
            if r2
            else v8_device._paint(
                *v8_device._body_cells(sim, torch.arange(e_sim, device=device)), h, w
            )
        )
        starts_es = starts.view(s_opp, e_sim, 3, 2).transpose(0, 1).contiguous()
        cap_es = cap.view(s_opp, e_sim).transpose(0, 1).contiguous()
        counts = (
            flood_cuda.static_capped(starts_es, occ_world.contiguous(), cap_es)
            .transpose(0, 1)
            .reshape(n, 3)
        )
    else:
        if r2:
            all_occ = r2_cache.all_occupancy(sim)[env]
        else:
            assert old_cells_all is not None and old_valid_all is not None
            all_occ = v8_device._paint(old_cells_all, old_valid_all, h, w)
        counts_max = (
            int(v8_device._CAP_MAX)
            if not v8_device._host_scalar_safe(cap)
            else min(int(v8_device._CAP_MAX), max(1, int(cap.max().item())))
        )
        counts = v8_device._flood(
            starts,
            all_occ[:, None].expand(-1, 3, -1, -1),
            cap[:, None].expand(-1, 3),
            max_steps=counts_max,
        )

    spacious = counts >= need[:, None]
    legal = mask.bool()
    legal = torch.where(legal.any(1, keepdim=True), legal, torch.arange(6, device=device)[None] < 3)
    actions6 = torch.arange(6, device=device)[None].expand(n, -1)

    head_pos, first_pos, lpost, boosted = _post_geom_with_table(
        sim, env, slot, actions6, heads, r2
    )

    boost3 = boosted[:, 3:]
    land_col_mask = (boost3 & legal[:, 3:] & spacious) if _only_action else boost3
    if not v8_device._host_scalar_safe(land_col_mask):
        own_cells, own_valid = v8_device._own_cells(sim, env, slot, span)
        post_cells, post_valid, _, _ = v8_device._post_body(
            sim, env, slot, actions6, own_cells, own_valid, span if r2 else None, r2
        )
        oo_all = r2_cache.other_occupancy(sim, env, slot) if r2 else other_occ_eager
        assert oo_all is not None
        post_tail3 = v8_device._paint(
            post_cells[:, 3:, 1:].reshape(n * 3, span - 1, 2),
            post_valid[:, 3:, 1:].reshape(n * 3, span - 1),
            h,
            w,
        ).reshape(n, 3, h, w)
        landing_block3 = oo_all[:, None] | post_tail3
        head_pos3_masked = torch.where(
            land_col_mask[..., None], head_pos[:, 3:], torch.full_like(head_pos[:, 3:], -1)
        )
        landing3 = v8_device._flood(
            head_pos3_masked,
            landing_block3,
            cap[:, None].expand(-1, 3),
            max_steps=v8_device._CAP_MAX,
        )
        landing = torch.where(
            boost3, landing3, torch.full_like(landing3, v8_device._LANDING_NO_BOOST)
        )
    else:
        landing = torch.full((n, 3), v8_device._LANDING_NO_BOOST, dtype=torch.long, device=device)
        land_rows = land_col_mask.any(dim=1).nonzero(as_tuple=False).squeeze(1)
        nl = int(land_rows.numel())
        if nl > 0:
            env_l = env.index_select(0, land_rows)
            slot_l = slot.index_select(0, land_rows)
            cnt_l = sim.seg_count[env_l, slot_l]
            span_l = min(int(sim.cap), max(2, int(cnt_l.max().item()) + 2))
            own_c_l, own_v_l = v8_device._own_cells(sim, env_l, slot_l, span_l)
            act3_l = torch.arange(3, 6, device=device)[None].expand(nl, -1)
            pc_l, pv_l, _, _ = v8_device._post_body(
                sim, env_l, slot_l, act3_l, own_c_l, own_v_l, span_l, r2
            )
            oo_l = (
                r2_cache.other_occupancy(sim, env_l, slot_l)
                if r2
                else other_occ_eager.index_select(0, land_rows)  # type: ignore[union-attr]
            )
            post_tail3_l = v8_device._paint(
                pc_l[:, :, 1:].reshape(nl * 3, span_l - 1, 2),
                pv_l[:, :, 1:].reshape(nl * 3, span_l - 1),
                h,
                w,
            ).reshape(nl, 3, h, w)
            landing_block3_l = oo_l[:, None] | post_tail3_l
            col_l = land_col_mask.index_select(0, land_rows)
            hp3_l = head_pos.index_select(0, land_rows)[:, 3:]
            hp3_masked_l = torch.where(col_l[..., None], hp3_l, torch.full_like(hp3_l, -1))
            cap_l = cap.index_select(0, land_rows)
            cap_l_max = min(int(v8_device._CAP_MAX), max(1, int(cap_l.max().item())))
            landing3_l = v8_device._flood(
                hp3_masked_l,
                landing_block3_l,
                cap_l[:, None].expand(-1, 3),
                max_steps=cap_l_max,
            )
            bst3_l = boost3.index_select(0, land_rows)
            landing[land_rows] = torch.where(
                bst3_l, landing3_l, torch.full_like(landing3_l, v8_device._LANDING_NO_BOOST)
            )

    landing_ok3 = (landing == v8_device._LANDING_NO_BOOST) | (landing >= need[:, None])
    elig = legal & torch.cat((spacious, spacious & landing_ok3), dim=1)

    if r2:
        covered = _covered_by_other_r2(sim)[env, slot]
    else:
        assert other_occ_eager is not None
        covered = other_occ_eager.flatten(1).sum(1)
    area_cap = (
        (2 * length)
        .clamp_min(v8_device._CAP_MIN)
        .minimum(torch.full_like(length, v8_device._AREA_MAX))
        .minimum((h * w - covered).clamp_min(1))
    )

    from src.core.mechanics_constants import HEADON_SIZE_RATIO

    paths = torch.stack((head_pos, first_pos), dim=2)
    path_valid = (
        torch.arange(2, device=device)[None, None, :] < torch.where(boosted, 2, 1)[:, :, None]
    )
    if (
        slot_layout == "opp_slot_major"
        and e_sim > 0
        and s_sim > 1
        and n == e_sim * (s_sim - 1)
    ):
        s_opp = s_sim - 1
        reach_w, rvalid_w_base, bound_w = _world_opp_reach(sim, r2)
        opp_mask = (
            torch.arange(1, s_sim, device=device)[:, None]
            != torch.arange(s_sim, device=device)[None, :]
        )
        paths_es = paths.view(s_opp, e_sim, 6, 2, 2).permute(1, 0, 2, 3, 4)
        equal_es = (
            (paths_es[:, :, :, :, None, None, :] == reach_w[:, None, None, None, :, :, :]).all(-1)
            & rvalid_w_base[:, None, None, None, :, :]
            & opp_mask[None, :, None, None, :, None]
        )
        b_es = bound_w[:, None, None, None, :, :].expand_as(equal_es)
        bmax_point = (
            torch.where(equal_es, b_es, torch.full_like(b_es, -1))
            .amax(dim=(-1, -2))
            .permute(1, 0, 2, 3)
            .reshape(n, 6, 2)
        )
    elif slot_layout == "hero" and e_sim > 0 and n == e_sim:
        reach_w, rvalid_w_base, bound_w = _world_opp_reach(sim, r2)
        opp_hero = (torch.arange(s_sim, device=device) != 0)[None, :, None]
        rvalid_h = rvalid_w_base & opp_hero
        equal_h = (paths[:, :, :, None, None, :] == reach_w[:, None, None, :, :, :]).all(
            -1
        ) & rvalid_h[:, None, None]
        bmax_point = torch.where(
            equal_h, bound_w[:, None, None], torch.full_like(bound_w[:, None, None], -1)
        ).amax(dim=(-1, -2))
    else:
        reach_w, rvalid_w_base, bound_w = _world_opp_reach(sim, r2)
        opp = torch.arange(s_sim, device=device)[None, :] != slot[:, None]
        reach = reach_w[env]
        rvalid = rvalid_w_base[env] & opp[:, :, None]
        bound = bound_w[env]
        equal = (paths[:, :, :, None, None, :] == reach[:, None, None, :, :, :]).all(
            -1
        ) & rvalid[:, None, None]
        bmax_point = torch.where(
            equal, bound[:, None, None], torch.full_like(bound[:, None, None], -1)
        ).amax(dim=(-1, -2))

    waived_point = (int(getattr(sim.cfg, "mechanics_version", 1)) == 2) & (
        lpost[:, :, None] >= float(HEADON_SIZE_RATIO) * bmax_point
    )
    risky = ((bmax_point >= 0) & ~waived_point & path_valid).any(dim=2)
    waived = (waived_point & (bmax_point >= 0) & path_valid).any(dim=2) & ~risky

    if features and _only_features and v8_device._host_scalar_safe(q):
        oo_f = r2_cache.other_occupancy(sim, env, slot) if r2 else other_occ_eager
        assert oo_f is not None
        area = _compute_release_area_active_lanes(
            sim, env, slot, oo_f, area_cap, head_pos, first_pos, lpost, boosted
        )
        z = torch.zeros(n, dtype=torch.long, device=device)
        return v8_device.V8DeviceResult(
            z,
            z,
            z,
            z,
            z,
            z,
            counts,
            cap,
            need,
            area_cap,
            area,
            landing,
            risky,
            waived,
            elig,
        )

    base = q.masked_fill(~legal, -1e30).argmax(1)
    v5, outcome = v8_device._choice(q, elig, base)
    base_v2_eligible = legal.gather(1, base[:, None]).squeeze(1) & spacious.gather(
        1, base.remainder(3)[:, None]
    ).squeeze(1)
    base_boost_bad = (
        (base >= 3)
        & base_v2_eligible
        & ~landing_ok3.gather(1, (base - 3).clamp_min(0)[:, None]).squeeze(1)
    )
    same_normal = legal.gather(1, base.remainder(3)[:, None]).squeeze(1)
    v5 = torch.where(base_boost_bad & same_normal, base.remainder(3), v5)
    outcome = torch.where(
        base_boost_bad & same_normal, torch.full_like(outcome, v8_device._VETOED), outcome
    )
    reason = torch.where(
        base_boost_bad & same_normal,
        torch.full_like(base, v8_device._R_SAME_DIR),
        torch.where(
            base_boost_bad & (outcome == v8_device._NO_SPACIOUS),
            torch.full_like(base, v8_device._R_LANDING_NONE),
            torch.where(
                base_boost_bad,
                torch.full_like(base, v8_device._R_LANDING_V2),
                torch.where(
                    elig.gather(1, base[:, None]).squeeze(1),
                    torch.full_like(base, v8_device._R_KEPT),
                    torch.full_like(base, v8_device._R_V2),
                ),
            ),
        ),
    )

    safe = elig & ~risky
    speed = v5 >= 3
    candidates = elig & ((actions6 >= 3) == speed[:, None])
    candidates.scatter_(1, v5[:, None], True)
    do_v7 = (
        (outcome != v8_device._NO_SPACIOUS)
        & (reason != v8_device._R_SAME_DIR)
        & (float(lam) != 0.0)
        & (candidates.sum(1) >= 2)
    )

    if features or not v8_device._host_scalar_safe(q):
        oo_all_f = r2_cache.other_occupancy(sim, env, slot) if r2 else other_occ_eager
        assert oo_all_f is not None
        area_full = _compute_release_area_active_lanes(
            sim, env, slot, oo_all_f, area_cap, head_pos, first_pos, lpost, boosted
        )
        area = area_full
    else:
        same_v5 = safe & ((actions6 >= 3) == (v5[:, None] >= 3)) & (actions6 != v5[:, None])
        alt_v5 = torch.where(
            same_v5.any(1, keepdim=True), same_v5, safe & (actions6 != v5[:, None])
        )
        anc_v5 = q.masked_fill(~alt_v5, -1e30).argmax(1)
        hcand_v5 = alt_v5 & ((actions6 >= 3) == (anc_v5[:, None] >= 3))
        rep_v5 = (
            (outcome != v8_device._NO_SPACIOUS)
            & risky.gather(1, v5[:, None]).squeeze(1)
            & alt_v5.any(1)
        )
        need_area = do_v7 | ((~do_v7) & rep_v5 & (float(lam) != 0.0) & (hcand_v5.sum(1) >= 2))
        area_full = torch.zeros((n, 6), dtype=torch.long, device=device)
        sub_r = need_area.nonzero(as_tuple=False).squeeze(1)
        if int(sub_r.numel()) > 0:
            env_s = env.index_select(0, sub_r)
            slot_s = slot.index_select(0, sub_r)
            oo_s = (
                r2_cache.other_occupancy(sim, env_s, slot_s)
                if r2
                else other_occ_eager.index_select(0, sub_r)  # type: ignore[union-attr]
            )
            act_needed = (candidates | safe).index_select(0, sub_r)
            area_full[sub_r] = _compute_release_area_active_lanes(
                sim,
                env_s,
                slot_s,
                oo_s,
                area_cap.index_select(0, sub_r),
                head_pos.index_select(0, sub_r),
                first_pos.index_select(0, sub_r),
                lpost.index_select(0, sub_r),
                boosted.index_select(0, sub_r),
                act_mask=act_needed,
            )
        area = torch.full_like(area_full, v8_device._LANDING_UNKNOWN)

    reranked = v8_device._rerank(q, candidates, v5, area_full, area_cap, lam)
    v7 = torch.where(do_v7, reranked, v5)
    v7_outcome = torch.where(
        do_v7 & (v7 != v5), torch.full_like(outcome, v8_device._VETOED), outcome
    )
    same_v7 = safe & ((actions6 >= 3) == (v7[:, None] >= 3)) & (actions6 != v7[:, None])
    alternate = torch.where(same_v7.any(1, keepdim=True), same_v7, safe & (actions6 != v7[:, None]))
    anchor = q.masked_fill(~alternate, -1e30).argmax(1)
    hcand = alternate & ((actions6 >= 3) == (anchor[:, None] >= 3))
    head_best = v8_device._rerank(q, hcand, anchor, area_full, area_cap, lam)
    replace_mask = (
        (v7_outcome != v8_device._NO_SPACIOUS)
        & risky.gather(1, v7[:, None]).squeeze(1)
        & alternate.any(1)
    )
    final = torch.where(replace_mask, head_best, v7)
    final_outcome = torch.where(
        replace_mask, torch.full_like(outcome, v8_device._VETOED), v7_outcome
    )
    return v8_device.V8DeviceResult(
        final,
        base,
        v5,
        v7,
        final_outcome,
        reason,
        counts,
        cap,
        need,
        area_cap,
        area,
        landing,
        risky,
        waived,
        elig,
    )


_ORIG_DEVICE_POLICY_EVALUATE = DevicePolicy.evaluate
_ORIG_DEVICE_POLICY_ACTIONS = DevicePolicy.actions


class FastDevicePolicy(DevicePolicy):
    """Bit-identical faster :class:`DevicePolicy` batching all 5 opponent slots per step."""

    @torch.inference_mode()
    def evaluate(self, sim: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.reference_bridge is not None
            or DevicePolicy.evaluate is not _ORIG_DEVICE_POLICY_EVALUATE
        ):
            return super().evaluate(sim)
        e = int(sim.E)
        rows = torch.stack(
            (
                torch.arange(e, device=self.device, dtype=torch.long),
                torch.zeros(e, dtype=torch.long, device=self.device),
            ),
            dim=1,
        )
        with sim.stage("v8_floods"):
            legal = sim.advisory_mask()[:, 0]
        with sim.stage("v8_floods"):
            features = (
                fast_v8_decide(
                    sim,
                    rows,
                    torch.zeros((e, 6), dtype=torch.float32, device=self.device),
                    legal,
                    self.v8_lambda,
                    features=True,
                    slot_layout="hero",
                    _only_features=True,
                )
                .feature_planes()
                .reshape(e, 30)
            )
        with sim.stage("observation_paint"):
            observation = build_observation(sim, rows, features)
        with sim.stage("b8_forward"):
            q = self.b8_network(*observation).float()
        return q, legal, _masked_argmax(q, legal)

    @torch.inference_mode()
    def actions(self, sim: Any, hero_actions: torch.Tensor | None = None) -> torch.Tensor:
        s_sim = int(sim.S)
        if (
            self.reference_bridge is not None
            or DevicePolicy.actions is not _ORIG_DEVICE_POLICY_ACTIONS
            or len(self.opponent_networks) != s_sim - 1
            or any(k is not None for k in self.anchor_kinds)
            or any(net is None for net in self.opponent_networks)
        ):
            return super().actions(sim, hero_actions)
        if hero_actions is None:
            _, _, hero_actions = self.evaluate(sim)
        e = int(sim.E)
        s_opp = s_sim - 1
        out = torch.ones((e, s_sim), dtype=torch.long, device=self.device)
        out[:, 0] = hero_actions
        if e == 0 or s_opp <= 0:
            return out
        if not self.vector_prepared:
            with sim.stage("vector61"):
                self.vector_runtime.prepare(sim)
            self.vector_prepared = True
        e_ar = torch.arange(e, dtype=torch.long, device=self.device)
        s_ar = torch.arange(1, s_sim, dtype=torch.long, device=self.device)
        opp_rows = torch.stack(
            (e_ar.repeat(s_opp), s_ar.repeat_interleave(e)),
            dim=1,
        )
        with sim.stage("vector61"):
            states_all, masks_all = self.vector_runtime.selection(opp_rows)
        with sim.stage("opponent_forwards"):
            q_parts = [
                net(states_all[k * e : (k + 1) * e]).float()
                for k, net in enumerate(self.opponent_networks)
            ]
            q_all = torch.cat(q_parts, dim=0)
        with sim.stage("v8_floods"):
            opp_act = fast_v8_decide(
                sim,
                opp_rows,
                q_all,
                masks_all,
                self.v8_lambda,
                features=False,
                slot_layout="opp_slot_major",
                _only_action=True,
            ).action
        out[:, 1:] = opp_act.reshape(s_opp, e).transpose(0, 1)
        return out

    def repeat_interleave(self, repeats: int) -> "FastDevicePolicy":
        out = super().repeat_interleave(repeats)
        out.__class__ = FastDevicePolicy
        return out  # type: ignore[return-value]


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


def _module_off_device(mod: torch.nn.Module | None, target_dev: torch.device) -> bool:
    if mod is None:
        return False
    for p in mod.parameters():
        if p.device != target_dev:
            return True
    for b in mod.buffers():
        if b.device != target_dev:
            return True
    return False


def move_device_policy_to_device(
    policy: DevicePolicy,
    device: str | torch.device,
    *,
    e_count: int | None = None,
) -> DevicePolicy:
    """Ensure ``policy``, its networks/tables, and ``policy.vector_runtime`` live on ``device``."""
    target_dev = normalize_device(device)
    if target_dev.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.set_float32_matmul_precision("highest")

    policy.device = target_dev
    if _module_off_device(policy.b8_network, target_dev):
        policy.b8_network = copy.deepcopy(policy.b8_network).to(target_dev).eval()
    if any(_module_off_device(net, target_dev) for net in policy.opponent_networks):
        memo: dict[int, torch.nn.Module] = {}
        new_nets = []
        for net in policy.opponent_networks:
            if net is not None:
                if id(net) not in memo:
                    memo[id(net)] = (
                        copy.deepcopy(net).to(target_dev).eval()
                        if _module_off_device(net, target_dev)
                        else net
                    )
                net = memo[id(net)]
            new_nets.append(net)
        policy.opponent_networks = tuple(new_nets)
    if any(isinstance(t, torch.Tensor) and t.device != target_dev for t in policy.anchor_tables):
        policy.anchor_tables = tuple(
            t.to(target_dev) if isinstance(t, torch.Tensor) else t for t in policy.anchor_tables
        )

    rt = policy.vector_runtime
    if rt is not None:
        rt.device = target_dev
        for attr in (
            *_VECTOR_RUNTIME_ENV_FIELDS,
            "_cpu_sector_lut",
            "_free_ratio",
            "fresh_rows",
            "carried_rows",
        ):
            val = getattr(rt, attr, None)
            if isinstance(val, torch.Tensor) and val.device != target_dev:
                target_dtype = (
                    torch.float32
                    if target_dev.type == "mps" and val.dtype == torch.float64
                    else val.dtype
                )
                setattr(rt, attr, val.to(device=target_dev, dtype=target_dtype))
        if (
            e_count is not None
            and isinstance(getattr(rt, "controlled", None), torch.Tensor)
            and int(rt.controlled.shape[0]) != int(e_count)
        ):
            src_e = int(rt.controlled.shape[0])
            keep_rows = torch.arange(int(e_count), device=target_dev, dtype=torch.long) % max(
                1, src_e
            )
            for attr in _VECTOR_RUNTIME_ENV_FIELDS:
                val = getattr(rt, attr, None)
                if isinstance(val, torch.Tensor):
                    setattr(rt, attr, val.index_select(0, keep_rows).clone())
            rt.E = int(e_count)
        rt.__class__ = FastDeviceVectorRuntime
    if type(policy) is DevicePolicy:
        policy.__class__ = FastDevicePolicy
    return policy


def assert_sim_and_policy_on_device(
    sim: DeviceSim,
    policy: DevicePolicy,
    expected_device: str | torch.device | None = None,
) -> None:
    """Fail closed if any tensor or module in ``sim`` or ``policy`` is off ``expected_device``."""
    dev = (
        normalize_device(expected_device)
        if expected_device is not None
        else normalize_device(sim.device)
    )
    if sim.device != dev:
        raise RuntimeError(f"sim.device mismatch: expected {dev}, got {sim.device}")
    for name in _ENV_TENSOR_FIELDS:
        val = getattr(sim, name, None)
        if isinstance(val, torch.Tensor) and val.device != dev:
            raise RuntimeError(f"sim.{name} device mismatch: expected {dev}, got {val.device}")
    if policy.device != dev:
        raise RuntimeError(f"policy.device mismatch: expected {dev}, got {policy.device}")
    for pname, p in policy.b8_network.named_parameters():
        if p.device != dev:
            raise RuntimeError(
                f"policy.b8_network.{pname} device mismatch: expected {dev}, got {p.device}"
            )
    for bname, b in policy.b8_network.named_buffers():
        if b.device != dev:
            raise RuntimeError(
                f"policy.b8_network.{bname} device mismatch: expected {dev}, got {b.device}"
            )
    for idx, net in enumerate(policy.opponent_networks):
        if net is None:
            continue
        for pname, p in net.named_parameters():
            if p.device != dev:
                raise RuntimeError(
                    f"policy.opponent_networks[{idx}].{pname} device mismatch: "
                    f"expected {dev}, got {p.device}"
                )
        for bname, b in net.named_buffers():
            if b.device != dev:
                raise RuntimeError(
                    f"policy.opponent_networks[{idx}].{bname} device mismatch: "
                    f"expected {dev}, got {b.device}"
                )
    for idx, tbl in enumerate(policy.anchor_tables):
        if isinstance(tbl, torch.Tensor) and tbl.device != dev:
            raise RuntimeError(
                f"policy.anchor_tables[{idx}] device mismatch: expected {dev}, got {tbl.device}"
            )
    rt = policy.vector_runtime
    if rt is not None:
        if rt.device != dev:
            raise RuntimeError(
                f"policy.vector_runtime.device mismatch: expected {dev}, got {rt.device}"
            )
        for attr in (
            *_VECTOR_RUNTIME_ENV_FIELDS,
            "_cpu_sector_lut",
            "_free_ratio",
            "fresh_rows",
            "carried_rows",
        ):
            val = getattr(rt, attr, None)
            if isinstance(val, torch.Tensor) and val.device != dev:
                raise RuntimeError(
                    f"policy.vector_runtime.{attr} device mismatch: "
                    f"expected {dev}, got {val.device}"
                )
        if int(rt.controlled.shape[0]) != int(sim.E):
            raise RuntimeError(
                f"policy.vector_runtime.controlled E={int(rt.controlled.shape[0])} "
                f"!= sim.E={int(sim.E)}"
            )


@torch.no_grad()
def compact_sim_and_policy(
    sim: DeviceSim,
    policy: DevicePolicy,
    keep_rows: torch.Tensor,
) -> None:
    """Compact ``sim`` and ``policy`` in-place to the 1-D row indices ``keep_rows``."""
    target_dev = normalize_device(sim.device)
    sim.device = target_dev
    move_device_policy_to_device(policy, target_dev)
    keep_rows = keep_rows.to(device=target_dev, dtype=torch.long).reshape(-1)
    new_e = int(keep_rows.numel())
    for name in _ENV_TENSOR_FIELDS:
        val = getattr(sim, name, None)
        if isinstance(val, torch.Tensor):
            setattr(sim, name, val.to(target_dev).index_select(0, keep_rows))
    sim.E = new_e
    r2_cache.bump(sim)

    rt = policy.vector_runtime
    if rt is not None:
        for name in _VECTOR_RUNTIME_ENV_FIELDS:
            val = getattr(rt, name, None)
            if isinstance(val, torch.Tensor):
                setattr(rt, name, val.to(target_dev).index_select(0, keep_rows))
        rt.E = new_e
        rt.__class__ = FastDeviceVectorRuntime
    assert_sim_and_policy_on_device(sim, policy, target_dev)


@torch.no_grad()
def _compact_sim_and_policy_fast(
    sim: DeviceSim,
    policy: DevicePolicy,
    keep_rows: torch.Tensor,
) -> None:
    """Fast inner-loop compaction when ``sim`` and ``policy`` are already validated on-device."""
    keep_rows = keep_rows.reshape(-1)
    new_e = int(keep_rows.shape[0])
    for name in _ENV_TENSOR_FIELDS:
        val = getattr(sim, name, None)
        if isinstance(val, torch.Tensor):
            setattr(sim, name, val.index_select(0, keep_rows))
    sim.E = new_e
    r2_cache.bump(sim)

    rt = policy.vector_runtime
    if rt is not None:
        for name in _VECTOR_RUNTIME_ENV_FIELDS:
            val = getattr(rt, name, None)
            if isinstance(val, torch.Tensor):
                setattr(rt, name, val.index_select(0, keep_rows))
        rt.E = new_e


@torch.no_grad()
def concat_sims_and_policies(
    sims: Sequence[DeviceSim],
    policies: Sequence[DevicePolicy],
) -> tuple[DeviceSim, DevicePolicy]:
    """Concatenate multiple ``(sim, policy)`` snapshots along world dimension ``dim=0``."""
    if not sims or len(sims) != len(policies):
        raise ValueError("sims and policies must be non-empty sequences of equal length")
    target_dev = normalize_device(sims[0].device)
    if len(sims) == 1:
        out_sim = sims[0].clone()
        out_sim.device = target_dev
        out_pol = policies[0].repeat_interleave(1)
        move_device_policy_to_device(out_pol, target_dev, e_count=int(out_sim.E))
        assert_sim_and_policy_on_device(out_sim, out_pol, target_dev)
        return out_sim, out_pol

    any_sim_prepared = any(bool(s._prepared) for s in sims)
    any_pol_prepared = any(bool(p.vector_prepared) for p in policies)
    norm_sims: list[DeviceSim] = []
    norm_pols: list[DevicePolicy] = []
    for s_in, p_in in zip(sims, policies):
        s_cur = s_in.clone()
        s_cur.device = target_dev
        for name in _ENV_TENSOR_FIELDS:
            v = getattr(s_cur, name, None)
            if isinstance(v, torch.Tensor) and v.device != target_dev:
                setattr(s_cur, name, v.to(target_dev))
        p_cur = p_in.repeat_interleave(1)
        move_device_policy_to_device(p_cur, target_dev)
        if (any_sim_prepared or any_pol_prepared) and not s_cur._prepared:
            s_cur.prepare()
        if any_pol_prepared and p_cur.vector_runtime is not None and not p_cur.vector_prepared:
            p_cur.vector_runtime.prepare(s_cur)
            p_cur.vector_prepared = True
        norm_sims.append(s_cur)
        norm_pols.append(p_cur)

    out_sim = norm_sims[0].clone()
    out_sim.device = target_dev
    out_pol = norm_pols[0].repeat_interleave(1)
    move_device_policy_to_device(out_pol, target_dev)

    total_e = sum(int(s.E) for s in norm_sims)
    for name in _ENV_TENSOR_FIELDS:
        vals = [getattr(s, name, None) for s in norm_sims]
        if all(isinstance(v, torch.Tensor) for v in vals):
            setattr(out_sim, name, torch.cat([v.to(target_dev) for v in vals], dim=0))
    out_sim.E = total_e
    out_sim._prepared = bool(any_sim_prepared or any_pol_prepared)
    r2_cache.bump(out_sim)

    out_rt = out_pol.vector_runtime
    if out_rt is not None and all(p.vector_runtime is not None for p in norm_pols):
        for name in _VECTOR_RUNTIME_ENV_FIELDS:
            vals = [getattr(p.vector_runtime, name, None) for p in norm_pols]
            if all(isinstance(v, torch.Tensor) for v in vals):
                setattr(out_rt, name, torch.cat([v.to(target_dev) for v in vals], dim=0))
        out_rt.E = total_e
        out_rt.__class__ = FastDeviceVectorRuntime
    out_pol.vector_prepared = bool(any_pol_prepared)
    assert_sim_and_policy_on_device(out_sim, out_pol, target_dev)
    return out_sim, out_pol


@torch.inference_mode()
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
    device = normalize_device(sim.device)
    if device.type == "cuda":
        verify_cuda_native_extensions(device)
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
    h, w = int(sim.grid_h), int(sim.grid_w)
    r2 = r2_cache.enabled(sim)
    v8_best_safe_area = zeros_i.clone()
    max_pocket_r = zeros_i.clone()
    adv_rows = (
        torch.arange(e_count, device=device, dtype=torch.long)
        if not v8_device._host_scalar_safe(advisory_any)
        else advisory_any.nonzero(as_tuple=False).squeeze(1)
    )
    na = int(adv_rows.numel())
    if na > 0:
        env_a = adv_rows
        slot_a = torch.zeros(na, dtype=torch.long, device=device)
        if r2:
            oo_a = r2_cache.other_occupancy(sim, env_a, slot_a)
            covered_a = _covered_by_other_r2(sim)[env_a, slot_a]
        else:
            old_c_a, old_v_a = v8_device._body_cells(sim, env_a)
            other_v_a = old_v_a & (torch.arange(sim.S, device=device)[None, :, None] != 0)
            oo_a = v8_device._paint(old_c_a, other_v_a, h, w)
            covered_a = oo_a.flatten(1).sum(1)
        len_a = sim.length[env_a, slot_a]
        ac_a = (
            (2 * len_a)
            .clamp_min(v8_device._CAP_MIN)
            .minimum(torch.full_like(len_a, v8_device._AREA_MAX))
            .minimum((h * w - covered_a).clamp_min(1))
        )
        actions6_a = torch.arange(6, device=device)[None].expand(na, -1)
        heads_a = v8_device._heads(sim, env_a, slot_a)
        hp_a, fp_a, lp_a, bst_a = _post_geom_with_table(sim, env_a, slot_a, actions6_a, heads_a, r2)
        adv_sub = advisory.index_select(0, adv_rows)
        areas_a = _compute_release_area_active_lanes(
            sim,
            env_a,
            slot_a,
            oo_a,
            ac_a,
            hp_a,
            fp_a,
            lp_a,
            bst_a,
            act_mask=adv_sub,
        )
        areas_masked = torch.where(
            adv_sub,
            areas_a.to(torch.long),
            torch.zeros((na, 6), dtype=torch.long, device=device),
        )
        best_a = areas_masked.max(dim=1).values
        v8_best_safe_area[adv_rows] = torch.where(
            advisory_any.index_select(0, adv_rows), best_a, torch.zeros_like(best_a)
        )

        if device.type == "mps":
            slots_t = torch.arange(sim.cap, device=device)[None, None, :]
            off_t = (sim.head_ptr[..., None] - slots_t) % sim.cap
            ttl_v = (sim.length[..., None] - off_t).clamp_min(0).to(torch.int32)
            act_t = sim.alive[..., None] & (off_t < sim.seg_count[..., None])
            bx, by = sim.bodies[..., 0], sim.bodies[..., 1]
            val_t = act_t & (bx >= 0) & (bx < w) & (by >= 0) & (by < h)
            fidx_t = by.clamp(0, h - 1) * w + bx.clamp(0, w - 1)
            ttl_flat = torch.zeros((e_count, int(sim.S), h * w), dtype=torch.int32, device=device)
            ttl_flat.scatter_reduce_(
                2,
                fidx_t,
                torch.where(val_t, ttl_v, torch.zeros_like(ttl_v)),
                reduce="amax",
                include_self=True,
            )
            ttl_grid = ttl_flat.to(torch.long).reshape(e_count, int(sim.S), h, w)
        else:
            ttl_grid = _ttl_by_owner(sim)
        reg_a = ego_device._component_sizes(
            sim, hero_rows.index_select(0, adv_rows), ttl_grid
        ).to(torch.long)
        dir_idx = torch.tensor([0, 1, 2, 0, 1, 2], device=device, dtype=torch.long)
        reg_actions_a = reg_a.index_select(1, dir_idx)
        legal_reg_a = torch.where(
            adv_sub,
            reg_actions_a,
            torch.zeros((na, 6), dtype=torch.long, device=device),
        )
        pkt_a = legal_reg_a.max(dim=1).values
        max_pocket_r[adv_rows] = torch.where(
            advisory_any.index_select(0, adv_rows), pkt_a, torch.zeros_like(pkt_a)
        )

    v8_open = advisory_any & (v8_best_safe_area >= v8_need)
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


def _repeat_sim_fast(src: DeviceSim, repeats: int) -> DeviceSim:
    """Repeat ``src`` along world dimension ``dim=0`` in one allocation without pre-cloning."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if repeats == 1:
        return src.clone()
    out = object.__new__(type(src))
    old_e = int(src.E)
    for key, value in src.__dict__.items():
        if isinstance(value, torch.Tensor):
            if value.ndim >= 1 and value.shape[0] == old_e:
                setattr(out, key, value.repeat_interleave(repeats, dim=0))
            else:
                setattr(out, key, value.clone())
        else:
            setattr(out, key, copy.copy(value))
    out.E = old_e * repeats
    out._timing = type(src._timing)(out.device)
    out.exact_mechanics = src.exact_mechanics
    r2_cache.inherit(out, src)
    return out


@torch.no_grad()
def batched_open_stop_h32(
    sim: DeviceSim,
    policy: DevicePolicy,
    *,
    world_seeds: Sequence[int],
    decisions: Sequence[int],
    suffix_id: int = OPEN_STOP_SELECTION_SUFFIXES[0],
    suffix_ids: Sequence[int] | None = None,
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
    if suffix_ids is not None and len(suffix_ids) != r_count:
        raise ValueError(f"suffix_ids ({len(suffix_ids)}) must match sim.E ({r_count})")
    if branch_horizon < 1:
        raise ValueError("branch_horizon must be >= 1")

    if sim.device.type == "cuda":
        verify_cuda_native_extensions(sim.device)
    with _maybe_cpu_fast_flood(bool(cpu_fast_flood and sim.device.type == "cpu")):
        out = _batched_open_stop_h32_impl(
            sim,
            policy,
            world_seeds=world_seeds,
            decisions=decisions,
            suffix_id=suffix_id,
            suffix_ids=suffix_ids,
            branch_horizon=branch_horizon,
            namespace=namespace,
            compact_dead=compact_dead,
            enable_r2=enable_r2,
            return_host_records=return_host_records,
        )
    with torch.inference_mode(False):
        return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}


@torch.inference_mode()
def _batched_open_stop_h32_impl(
    sim: DeviceSim,
    policy: DevicePolicy,
    *,
    world_seeds: Sequence[int],
    decisions: Sequence[int],
    suffix_id: int,
    suffix_ids: Sequence[int] | None,
    branch_horizon: int,
    namespace: str,
    compact_dead: bool,
    enable_r2: bool,
    return_host_records: bool,
) -> dict[str, Any]:
    r_count = int(sim.E)
    device = normalize_device(sim.device)
    use_r2 = bool(enable_r2 and device.type != "mps")
    can_eval_in_place = (
        sim.device == device
        and bool(sim._prepared)
        and bool(policy.vector_prepared)
        and (not use_r2 or r2_cache.enabled(sim))
    )
    if can_eval_in_place:
        root_sim = sim
        root_policy = policy
        if type(root_policy) is DevicePolicy:
            root_policy = policy.repeat_interleave(1)
            root_policy.__class__ = FastDevicePolicy
        assert_sim_and_policy_on_device(root_sim, root_policy, device)
        root_alive = root_sim.alive[:, 0].to(device=device, dtype=torch.bool).clone()
    else:
        root_sim = sim.clone()
        root_sim.device = device
        root_alive = root_sim.alive[:, 0].to(device=device, dtype=torch.bool).clone()
        root_sim.terminal_slots[:, 0] = True
        root_policy = policy.repeat_interleave(1)
        move_device_policy_to_device(root_policy, device)
        assert_sim_and_policy_on_device(root_sim, root_policy, device)
        if use_r2:
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
    branch_sim = _repeat_sim_fast(root_sim, 6)
    branch_policy = root_policy.repeat_interleave(6)
    if type(branch_policy) is DevicePolicy:
        branch_policy.__class__ = FastDevicePolicy
    if branch_policy.vector_runtime is not None:
        branch_policy.vector_runtime.__class__ = FastDeviceVectorRuntime
    if use_r2:
        r2_cache.enable(branch_sim)

    # Hero is terminal inside the H=32 continuation rollout (never respawns mid-branch).
    branch_sim.terminal_slots[:, 0] = True

    root_keys: list[int] = []
    for idx_r, (ws, dec) in enumerate(zip(world_seeds, decisions)):
        suf_r = int(suffix_ids[idx_r]) if suffix_ids is not None else int(suffix_id)
        raw_seed = branch_suffix_seed(namespace, int(ws), int(dec), suffix_id=suf_r)
        key = int(raw_seed) & 0x7FFF_FFFF_FFFF_FFFF
        if key == 0:
            key = 1
        root_keys.append(key)
    keys_t = torch.tensor(root_keys, dtype=torch.long, device=device).repeat_interleave(6)
    branch_sim.rng_key.copy_(keys_t)
    branch_sim.rng_counter.zero_()

    def _after_branch_step(is_final_step: bool) -> None:
        if is_final_step:
            branch_policy.vector_prepared = False
            return
        if (
            compact_dead
            and branch_policy.reference_bridge is None
            and branch_policy.vector_runtime is not None
        ):
            with branch_sim.stage("vector61"):
                branch_policy.vector_runtime.observe_step(
                    branch_sim, rows_mask=branch_sim.alive[:, 0]
                )
            branch_policy.vector_prepared = False
        else:
            branch_policy.after_step(branch_sim)

    # Step 1 (h = 0): forced hero action 0..5 per root.
    forced_actions = torch.arange(6, dtype=torch.long, device=device).repeat(r_count)
    full_actions = branch_policy.actions(branch_sim, forced_actions)
    branch_sim.step(full_actions)
    _after_branch_step(int(branch_horizon) == 1)

    frames_executed = torch.ones(b_count, dtype=torch.long, device=device)
    alive_every_step = branch_sim.alive[:, 0].clone() & root_alive.repeat_interleave(6)
    frames_survived = alive_every_step.to(torch.long)
    final_length = torch.where(
        alive_every_step,
        branch_sim.length[:, 0].to(torch.long),
        torch.zeros(b_count, dtype=torch.long, device=device),
    )

    active_idx = torch.arange(b_count, dtype=torch.long, device=device)

    for _step in range(1, int(branch_horizon)):
        is_last = _step == int(branch_horizon) - 1
        if compact_dead:
            keep_mask = branch_sim.alive[:, 0]
            alive_count = int(keep_mask.sum().item())
            if alive_count == 0:
                break
            if alive_count < int(branch_sim.E):
                keep_rows = keep_mask.nonzero(as_tuple=False).squeeze(1)
                _compact_sim_and_policy_fast(branch_sim, branch_policy, keep_rows)
                active_idx = active_idx.index_select(0, keep_rows)

            branch_sim.prepare()
            branch_policy.vector_runtime.prepare(branch_sim)
            branch_policy.vector_prepared = True
            _, _, hero_act = branch_policy.evaluate(branch_sim)
            step_actions = branch_policy.actions(branch_sim, hero_act)
            branch_sim.step(step_actions)
            _after_branch_step(is_last)

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
            _after_branch_step(is_last)

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
        surv_count = int(surv_mask_local.sum().item())
        if surv_count > 0:
            if surv_count < int(branch_sim.E):
                surv_rows = surv_mask_local.nonzero(as_tuple=False).squeeze(1)
                _compact_sim_and_policy_fast(branch_sim, branch_policy, surv_rows)
                active_idx = active_idx.index_select(0, surv_rows)
            openness_local: dict[str, torch.Tensor] = evaluate_terminal_openness_device(branch_sim)
        else:
            active_idx = torch.zeros(0, dtype=torch.long, device=device)
            openness_local = {}
    else:
        surv_rows = raw_survives.nonzero(as_tuple=False).squeeze(1)
        if int(surv_rows.numel()) > 0:
            compact_sim_and_policy(branch_sim, branch_policy, surv_rows)
            active_idx = surv_rows
            openness_local = evaluate_terminal_openness_device(branch_sim)
        else:
            active_idx = torch.zeros(0, dtype=torch.long, device=device)
            openness_local = {}

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
    a_star_t = select_a_star_nb_on_device(root_q_t, safe_mask, bare_clamped)
    rescue_mask = has_safe & (~bare_is_safe) & has_unsafe
    ordinary_mask = bare_is_safe

    roots_out: list[dict[str, Any]] = []
    if return_host_records:
        openness_by_branch: dict[int, dict[str, Any]] = {}
        active_idx_cpu = active_idx.detach().cpu().tolist()
        if active_idx_cpu:
            op_len = openness_local["length"].detach().cpu().tolist()
            op_adv = openness_local["advisory_any"].detach().cpu().tolist()
            op_need = openness_local["v8_need"].detach().cpu().tolist()
            op_area = openness_local["v8_best_safe_area"].detach().cpu().tolist()
            op_v8 = openness_local["v8_open"].detach().cpu().tolist()
            op_pkt = openness_local["max_pocket_r"].detach().cpu().tolist()
            op_pktopen = openness_local["pocket_open"].detach().cpu().tolist()
            op_unenc = openness_local["un_enclosed"].detach().cpu().tolist()
            for local_i, b_idx in enumerate(active_idx_cpu):
                openness_by_branch[int(b_idx)] = {
                    "length": int(op_len[local_i]),
                    "advisory_any": bool(op_adv[local_i]),
                    "v8_need": int(op_need[local_i]),
                    "v8_best_safe_area": int(op_area[local_i]),
                    "v8_open": bool(op_v8[local_i]),
                    "max_pocket_r": int(op_pkt[local_i]),
                    "pocket_open": bool(op_pktopen[local_i]),
                    "un_enclosed": bool(op_unenc[local_i]),
                }

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
                    "suffix_id": int(suffix_ids[r]) if suffix_ids is not None else int(suffix_id),
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
    dev = normalize_device(device)
    if dev.type == "cuda":
        verify_cuda_native_extensions(dev)
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
        alive_mask = sim.alive[:, 0].to(device=dev, dtype=torch.bool).clone()
        if not sim._prepared:
            sim = sim.clone()
            sim.terminal_slots[:, 0] = True
            sim.prepare()
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
                fast_v8_decide(
                    sim,
                    hero_rows,
                    torch.zeros((total_roots, 6), device=sim.device),
                    advisory,
                    float(OPEN_STOP_V8_LAMBDA),
                    features=True,
                    slot_layout="hero",
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
        alive_mask = torch.ones(total_roots, dtype=torch.bool, device=dev)
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
        both_safe_and_unsafe = safe_all.any(dim=1) & unsafe_all.any(dim=1) & alive_mask
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
    bare_sel = bare_all.index_select(0, selected_rows)
    if a_star is not None:
        astar_full = torch.as_tensor(a_star, dtype=torch.long, device=dev)
        astar_t = astar_full.index_select(0, selected_rows)
        invalid_star = (astar_t < 0) | (astar_t >= 6) | (~safe_sel[batch_idx, astar_t.clamp(0, 5)])
        if bool(invalid_star.any().item()):
            fallback_star = select_a_star_nb_on_device(q.detach(), safe_sel, bare_sel)
            astar_t = torch.where(invalid_star, fallback_star, astar_t)
    else:
        astar_t = select_a_star_nb_on_device(q.detach(), safe_sel, bare_sel)

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
