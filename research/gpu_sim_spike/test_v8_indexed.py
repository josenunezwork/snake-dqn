"""CPU construction oracle for the optional native v8 direct-ring bundle.

CUDA compilation/execution belongs to the A100 packet.  This test proves the
direct ring formula independently on the Mac and binds it to the frozen tensor
maps consumed by :func:`v8_decide`, including duplicated selected worlds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.v8_device import _body_cells, _paint, _post_body
from src.simd_env.v8_nb import V8Result
from src.simd_env.v8_nb import v8_decide as cpu_v8_decide


def _fixture() -> DeviceSim:
    from research.gpu_sim_spike.test_v8_device import _self_overlap_fixture

    return DeviceSim.from_reference(_self_overlap_fixture(), device="cpu")


def _indexed_oracle(sim: DeviceSim, rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Straight indexed-ring construction, intentionally independent of v8 helpers."""
    n, h, w, cap = rows.shape[0], sim.grid_h, sim.grid_w, sim.cap
    all_occ = torch.zeros((n, h, w), dtype=torch.bool)
    other_occ = torch.zeros_like(all_occ)
    post_tail = torch.zeros((n, 6, h, w), dtype=torch.bool)
    release = torch.zeros((n, 6, h, w), dtype=torch.long)
    lpost = torch.zeros((n, 6), dtype=torch.long)
    boosted = torch.zeros((n, 6), dtype=torch.bool)
    dx, dy = (0, 1, 0, -1), (-1, 0, 1, 0)
    for r, (env_t, observer_t) in enumerate(rows.tolist()):
        for snake in range(sim.S):
            if not bool(sim.alive[env_t, snake]):
                continue
            for k in range(int(sim.seg_count[env_t, snake])):
                ring = (int(sim.head_ptr[env_t, snake]) - k) % cap
                x, y = (int(v) for v in sim.bodies[env_t, snake, ring])
                if 0 <= x < w and 0 <= y < h:
                    all_occ[r, y, x] = True
                    if snake != observer_t:
                        other_occ[r, y, x] = True
        count = int(sim.seg_count[env_t, observer_t])
        length = int(sim.length[env_t, observer_t])
        hp = int(sim.head_ptr[env_t, observer_t])
        hx, hy = (int(v) for v in sim.bodies[env_t, observer_t, hp])
        heading = int(sim.direction[env_t, observer_t])
        bf = int(sim.boost_frames[env_t, observer_t])
        for action in range(6):
            is_boost = action >= 3 and length >= int(sim.cfg.min_boost_length)
            ln = min(count + 1, length)
            if is_boost:
                ln = min(ln + 1, length)
            post_len = length - int(
                is_boost and bf + 1 >= int(sim.cfg.boost_length_cost_frames) and length > 1
            )
            ln = min(ln, post_len)
            lpost[r, action] = post_len
            boosted[r, action] = is_boost
            hd = (heading + action % 3 - 1) % 4
            nh = 2 if is_boost else 1
            for i in range(ln):
                if i < nh:
                    x, y = hx + dx[hd] * (nh - i), hy + dy[hd] * (nh - i)
                else:
                    ring = (hp - (i - nh)) % cap
                    x, y = (int(v) for v in sim.bodies[env_t, observer_t, ring])
                if 0 <= x < w and 0 <= y < h:
                    if i:
                        post_tail[r, action, y, x] = True
                    release[r, action, y, x] = max(int(release[r, action, y, x]), post_len - i + 1)
    return all_occ, other_occ, post_tail, release, lpost, boosted


def _tensor_baseline(sim: DeviceSim, rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The unchanged tensor construction, factored only for oracle comparison."""
    env, slot = rows[:, 0], rows[:, 1]
    n, h, w = rows.shape[0], sim.grid_h, sim.grid_w
    old_cells, old_valid = _body_cells(sim, env)
    own_cells = old_cells[torch.arange(n), slot]
    own_valid = old_valid[torch.arange(n), slot]
    all_occ = _paint(old_cells, old_valid, h, w)
    other = old_valid & (torch.arange(sim.S)[None, :, None] != slot[:, None, None])
    other_occ = _paint(old_cells, other, h, w)
    actions = torch.arange(6)[None].expand(n, -1)
    post_cells, post_valid, lpost, boosted = _post_body(
        sim, env, slot, actions, own_cells, own_valid
    )
    post_tail = _paint(
        post_cells[:, :, 1:].reshape(n * 6, sim.cap - 1, 2),
        post_valid[:, :, 1:].reshape(n * 6, sim.cap - 1),
        h,
        w,
    ).reshape(n, 6, h, w)
    i = torch.arange(sim.cap)[None, None, :]
    values = (lpost[:, :, None] - i + 1).clamp_min(0)
    release = torch.zeros((n, 6, h * w), dtype=torch.long)
    px, py = post_cells[..., 0], post_cells[..., 1]
    good = post_valid & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    index = py.clamp(0, h - 1) * w + px.clamp(0, w - 1)
    release.scatter_reduce_(2, index, torch.where(good, values, torch.zeros_like(values)), "amax")
    return all_occ, other_occ, post_tail, release.reshape(n, 6, h, w), lpost, boosted


def test_indexed_ring_construction_matches_cpu_tensor_baseline_with_duplicate_rows() -> None:
    sim = _fixture()
    # Duplicate and shuffled selected worlds ensure output ownership stays in
    # caller row order rather than collapsing by environment identity.
    rows = torch.tensor([[0, 0], [0, 1], [0, 0]], dtype=torch.long)
    actual = _indexed_oracle(sim, rows)
    expected = _tensor_baseline(sim, rows)
    for got, want in zip(actual, expected):
        assert got.dtype == want.dtype
        assert got.numpy().tobytes() == want.numpy().tobytes()


def _cuda_canary_reference() -> object:
    """Adversarial valid source state for native CUDA admission, not a fixture oracle."""
    from research.gpu_sim_spike.test_v8_device import _self_overlap_fixture

    sim = _self_overlap_fixture()
    # Force the selected ring across index zero.  Slot 2 remains populated but
    # dead, which proves liveness filtering rather than relying on zero fill.
    cells = ((7, 6), (6, 6), (5, 6), (5, 7), (6, 7), (7, 7), (8, 7), (8, 6), (9, 6))
    sim.head_ptr[0, 0] = 1
    sim.seg_count[0, 0] = len(cells)
    sim.length[0, 0] = len(cells)
    for k, cell in enumerate(cells):
        sim.bodies[0, 0, (1 - k) % sim.cap] = cell
    sim.alive[0, 2] = False
    return sim


def _assert_v8_fields(native: object, reference: V8Result) -> None:
    for name in V8Result.__dataclass_fields__:
        got = getattr(native, name).detach().cpu().numpy()
        want = getattr(reference, name)
        assert got.tobytes() == want.tobytes(), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native indexed admission needs CUDA")
def test_cuda_indexed_construction_and_v8_diagnostics_match_cpu_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission canary: six constructed maps and all v8 outputs must be bytes-equal."""
    from research.gpu_sim_spike import v8_indexed
    from research.gpu_sim_spike.v8_device import v8_decide

    source = _cuda_canary_reference()
    rows_cpu = torch.tensor([[0, 0], [0, 1], [0, 0]], dtype=torch.long)
    cpu_sim = DeviceSim.from_reference(source, device="cpu")
    expected_maps = _indexed_oracle(cpu_sim, rows_cpu)
    cuda_sim = DeviceSim.from_reference(source, device="cuda")
    actual_maps = v8_indexed.construct(cuda_sim, rows_cpu.to("cuda"))
    torch.cuda.synchronize()
    for got, want in zip(actual_maps, expected_maps):
        assert got.detach().cpu().numpy().tobytes() == want.numpy().tobytes()

    q = np.array(
        [[0.7, 0.4, 0.6, 0.8, 0.2, 0.5], [0.0, 1.0, 0.0, 0.0, 9.0, 0.0], [1.0] * 6],
        dtype=np.float32,
    )
    mask = np.array([[True] * 6, [True, False, True, True, False, True], [False] * 6], dtype=bool)
    canonical = cpu_v8_decide(source, rows_cpu.numpy(), q, mask, features=True)
    monkeypatch.setenv("GPU_SIM_V8_INDEXED", "1")
    native = v8_decide(
        cuda_sim,
        rows_cpu.to("cuda"),
        torch.as_tensor(q, device="cuda"),
        torch.as_tensor(mask, device="cuda"),
        features=True,
    )
    torch.cuda.synchronize()
    _assert_v8_fields(native, canonical)
