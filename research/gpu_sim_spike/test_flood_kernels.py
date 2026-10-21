"""Independent adversarial checks for the bounded device flood kernels."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.gpu_sim_spike import flood_cuda
from research.gpu_sim_spike.ego_device import _component_sizes
from research.gpu_sim_spike.v8_device import _flood


def _bfs_count(
    start: tuple[int, int], blocked: np.ndarray, cap: int, release: np.ndarray | None = None
) -> int:
    """Standalone cardinal FIFO oracle; intentionally shares no device helpers."""
    h, w = blocked.shape
    sx, sy = start
    if not (0 <= sx < w and 0 <= sy < h) or blocked[sy, sx]:
        return 0
    if release is not None and release[sy, sx] > 0:
        return 0
    todo = deque([(sx, sy, 0)])
    seen = {(sx, sy)}
    while todo and len(seen) < cap:
        x, y, distance = todo.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) in seen:
                continue
            if blocked[ny, nx] or (release is not None and distance + 1 < release[ny, nx]):
                continue
            seen.add((nx, ny))
            todo.append((nx, ny, distance + 1))
            if len(seen) == cap:
                break
    return len(seen)


def _ego_sim(allowed: np.ndarray) -> tuple[SimpleNamespace, torch.Tensor]:
    h, w = allowed.shape
    sim = SimpleNamespace()
    sim.E, sim.S, sim.grid_h, sim.grid_w, sim.device = 1, 1, h, w, torch.device("cpu")
    sim.alive = torch.tensor([[True]])
    sim.direction = torch.tensor([[1]])  # right: starts at x+1 for straight
    sim._heads = torch.tensor([[[0, 0]]])
    sim.heads = lambda: sim._heads
    # ttl > 1 is blocked by the device component contract.
    ttl = torch.where(
        torch.as_tensor(allowed)[None, None],
        torch.zeros((1, 1, h, w), dtype=torch.long),
        torch.full((1, 1, h, w), 2, dtype=torch.long),
    )
    return sim, ttl


def test_component_labels_match_independent_bfs_on_long_serpentine_and_repeated_rows() -> None:
    # One-cell-wide winding corridor has a graph diameter far beyond H+W.
    h, w = 35, 33
    allowed = np.zeros((h, w), dtype=bool)
    for y in range(0, h, 2):
        allowed[y, :] = True
        if y + 1 < h:
            allowed[y + 1, -1 if (y // 2) % 2 == 0 else 0] = True
    sim, ttl = _ego_sim(allowed)
    # Head points right into the first corridor cell; duplicate rows prove the
    # one-per-world labelling is reused rather than re-flooded per action.
    rows = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
    got = _component_sizes(sim, rows, ttl).numpy()
    expected = _bfs_count((1, 0), ~allowed, allowed.size)
    assert expected > h + w
    assert np.array_equal(got[:, 1], np.array([expected, expected]))
    assert np.array_equal(got[:, (0, 2)], np.zeros((2, 2), dtype=np.int64))


@pytest.mark.parametrize("h,w", ((1, 1), (1, 5), (5, 1)))
def test_component_labels_handle_singleton_edges_blocked_oob_and_dead(h: int, w: int) -> None:
    allowed = np.ones((h, w), dtype=bool)
    sim, ttl = _ego_sim(allowed)
    rows = torch.tensor([[0, 0]], dtype=torch.long)
    got = _component_sizes(sim, rows, ttl).numpy()
    # The head is at (0, 0), direction right; all in-bounds candidate starts
    # share the same component, while out-of-bounds starts remain zero.
    assert got.shape == (1, 3)
    expected = np.array([_bfs_count(start, ~allowed, h * w) for start in ((0, -1), (1, 0), (0, 1))])
    np.testing.assert_array_equal(got[0], expected)
    sim.alive[0, 0] = False
    assert not _component_sizes(sim, rows, ttl).any()


def test_component_labels_random_maps_match_independent_bfs() -> None:
    rng = np.random.default_rng(711)
    h, w = 11, 13
    for _ in range(20):
        allowed = rng.random((h, w)) > 0.28
        allowed[0, 1] = True
        sim, ttl = _ego_sim(allowed)
        got = _component_sizes(sim, torch.tensor([[0, 0]]), ttl).numpy()[0]
        expected = _bfs_count((1, 0), ~allowed, h * w)
        assert got[1] == expected


def test_component_labels_needs_rigorous_cardinal_bound_for_arbitrary_label_order() -> None:
    # This connected 15-cell counterexample remains split after seven local
    # hook/shortcut rounds.  It guards the V-1 cap instead of assuming a
    # logarithmic bound from pointer jumping alone.
    allowed = np.array(
        [
            [cell == "." for cell in line]
            for line in (
                ".##.###.",
                "..#...##",
                ".#.##.#.",
                "#..##..#",
                ".#.###..",
                ".#.#.##.",
                ".#...#.#",
                "...##..#",
            )
        ],
        dtype=bool,
    )
    sim, ttl = _ego_sim(allowed)
    for y, x in zip(*np.where(allowed)):
        if x == 0:
            continue
        # Direction right with a straight action starts at (x, y).
        sim._heads[0, 0] = torch.tensor([x - 1, y])
        sim.direction[0, 0] = 1
        got = _component_sizes(sim, torch.tensor([[0, 0]]), ttl).numpy()[0, 1]
        assert got == _bfs_count((x, y), ~allowed, allowed.size)


def test_v8_wavefront_matches_fifo_caps_and_delayed_release_without_waiting() -> None:
    # A winding corridor forces more than H+W hops; each cap must stop at the
    # CPU cardinal-component count, independent of FIFO discovery order.
    h, w = 91, 91
    open_cells = np.zeros((h, w), dtype=bool)
    for y in range(0, h, 2):
        open_cells[y, :] = True
        if y + 1 < h:
            open_cells[y + 1, -1 if (y // 2) % 2 == 0 else 0] = True
    blocked = torch.as_tensor(~open_cells)[None, None]
    start = torch.tensor([[[0, 0]]])
    for cap in (160, 4096):
        got = _flood(start, blocked, torch.tensor([[cap]]), max_steps=cap)
        assert got.item() == _bfs_count((0, 0), ~open_cells, cap)

    # The only edge out of the source opens at distance 9. FIFO has no waiting
    # queue, so the source's empty first frontier remains permanently empty.
    blocked_np = np.ones((3, 4), dtype=bool)
    blocked_np[1, 0:3] = False
    release_np = np.zeros_like(blocked_np, dtype=np.int64)
    release_np[1, 1] = 9
    got = _flood(
        torch.tensor([[[0, 1]]]),
        torch.as_tensor(blocked_np)[None, None],
        torch.tensor([[4096]]),
        torch.as_tensor(release_np)[None, None],
        max_steps=4096,
    )
    assert got.item() == _bfs_count((0, 1), blocked_np, 4096, release_np) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA toolkit/runtime")
def test_native_cuda_frontier_matches_fifo_across_tiles_and_delayed_release() -> None:
    """CUDA proof exercises queue dedupe, tile boundaries, and no-wait release.

    It is intentionally not a FakeTensor test: a CUDA E1 job must compile and
    execute the extension.  Development Macs skip it, while an L4 preflight
    treats a missing native extension as a preparation failure rather than
    measuring the Python implementation.
    """
    assert flood_cuda.available(), "CUDA preflight must fail if native E1 kernels do not compile"
    h, w = 19, 23  # crosses the kernel's 256-cell initialization tile.
    blocked_np = np.ones((h, w), dtype=bool)
    blocked_np[2, 1:22] = False
    blocked_np[3:18, 21] = False
    blocked_np[17, 2:22] = False
    release_np = np.zeros_like(blocked_np, dtype=np.int64)
    release_np[2, 2] = 2
    # The immediate delayed neighbour leaves the FIFO empty, proving it is not
    # requeued later merely because its release TTL has elapsed.
    blocked_np[8, 1:4] = False
    release_np[8, 2] = 9
    starts = torch.tensor([[[1, 2]], [[1, 8]]], device="cuda", dtype=torch.long)
    blocked = torch.as_tensor(blocked_np, device="cuda")[None, None].expand(2, -1, -1, -1)
    release = torch.as_tensor(release_np, device="cuda")[None, None].expand(2, -1, -1, -1)
    cap = torch.tensor([[4096], [4096]], device="cuda", dtype=torch.long)
    got = _flood(starts, blocked, cap, release).cpu().numpy().reshape(-1)
    expected = np.array(
        [
            _bfs_count((1, 2), blocked_np, 4096, release_np),
            _bfs_count((1, 8), blocked_np, 4096, release_np),
        ]
    )
    assert np.array_equal(got, expected)
    # A failed early release attempt must not mark that cell seen: the longer
    # route arrives at its release distance and must enqueue it exactly once.
    retry_blocked = torch.tensor([[[[False, False, False], [False, False, True]]]], device="cuda")
    retry_release = torch.tensor([[[[0, 3, 0], [0, 0, 0]]]], device="cuda", dtype=torch.long)
    retry = _flood(
        torch.tensor([[[0, 0]]], device="cuda"),
        retry_blocked,
        torch.tensor([[4096]], device="cuda"),
        retry_release,
    )
    assert retry.cpu().item() == 5
    # A non-default stream catches extensions which launch on stream 0 rather
    # than PyTorch's current stream.  The event dependency makes the result
    # observably wrong if the guard/stream contract is ignored.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream_result = _flood(starts[:1], blocked[:1], cap[:1], release[:1])
    stream.synchronize()
    assert stream_result.cpu().item() == expected[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA toolkit/runtime")
def test_native_cuda_components_match_serpentine_oracle_across_tiles() -> None:
    """The union-find kernel must not inherit a short propagation bound."""
    assert flood_cuda.available(), "CUDA preflight must fail if native E1 kernels do not compile"
    h, w = 37, 41
    allowed = np.zeros((h, w), dtype=bool)
    for y in range(0, h, 2):
        allowed[y, :] = True
        if y + 1 < h:
            allowed[y + 1, -1 if (y // 2) % 2 == 0 else 0] = True
    sim, ttl = _ego_sim(allowed)
    sim.device = torch.device("cuda")
    sim.alive = sim.alive.cuda()
    sim.direction = sim.direction.cuda()
    sim._heads = sim._heads.cuda()
    got = (
        _component_sizes(sim, torch.tensor([[0, 0]], device="cuda"), ttl.cuda()).cpu().numpy()[0, 1]
    )
    assert got == _bfs_count((1, 0), ~allowed, allowed.size)

    # Separate components, a dead row, an OOB start, and duplicate/shuffled
    # rows exercise the native gather after its one-per-world union-find pass.
    split = np.zeros((19, 23), dtype=bool)
    split[1:18, 1:11] = True
    split[2:5, 15:20] = True
    split_sim, split_ttl = _ego_sim(split)
    split_sim.device = torch.device("cuda")
    split_sim.alive = torch.tensor([[True]], device="cuda")
    split_sim.direction = torch.tensor([[1]], device="cuda")
    split_sim._heads = torch.tensor([[[0, 9]]], device="cuda")
    rows = torch.tensor([[0, 0], [0, 0]], device="cuda")
    result = _component_sizes(split_sim, rows, split_ttl.cuda()).cpu().numpy()
    assert np.array_equal(result[:, 1], np.array([170, 170]))
    assert not result[:, (0, 2)].any()
    split_sim.alive.zero_()
    assert not _component_sizes(split_sim, rows[:1], split_ttl.cuda()).cpu().any()
