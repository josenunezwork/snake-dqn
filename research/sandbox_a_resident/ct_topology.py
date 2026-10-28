"""Frozen dynamic-topology approximations and geometric fixture validation for C/T (T arm).

Implements Ruling §4 ("For T, freeze full-board wall/occupancy, own TTL, enemy TTL
and food planes, plus six action-conditioned bounded reach/TTL-slack summaries,
before collection; publish normalization, orientation, boost treatment, BFS horizon
and missing-value rules in its short manifest. Validate these features on
deterministic geometric fixtures. Call them dynamic-topology approximations:
TTL<=arrival ignores later growth, boosts, new body and opponent reactions, and is
not exact future reachability.")
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numba
import numpy as np
import torch

BFS_HORIZON = 128
ENEMY_TTL_NORM_CAP = 1000.0
ILLEGAL_BOOST_SENTINEL = -1.0
IMMEDIATE_COLLISION_VALUE = 0.0
REACH_WEIGHT = 0.8
SLACK_WEIGHT = 0.2
QUANT_MAX = 65535

TOPOLOGY_SPEC: dict[str, Any] = {
    "schema": "dynamic-topology-approximations/v1",
    "name": "dynamic-topology approximations",
    "disclaimer": (
        "Dynamic-topology approximations only: time-expanded TTL<=arrival assumes "
        "unit-rate tail decay and ignores later food growth, future boosts, newly laid "
        "body segments, and opponent reactions; it is not exact future reachability. "
        "Diagnostic T-arm input only; authorizes no serving obs-spec change."
    ),
    "planes": {
        "shape_production": [4, 83, 145],
        "storage_dtype": "uint16",
        "forward_scale": "float32 = uint16 / 65535.0 in [0.0, 1.0]",
        "channels": [
            {
                "index": 0,
                "name": "wall_occupancy",
                "rule": (
                    "1.0 (65535) if occupied by any living snake segment, 0.0 (0) if free; "
                    "conv pads exterior border with 1.0 (wall)."
                ),
            },
            {
                "index": 1,
                "name": "own_ttl",
                "rule": (
                    "clamp(own_ttl_steps(y, x) / max(1, hero_length), 0.0, 1.0) where "
                    "own_ttl_steps = max(0, hero_length - hero_seg_count) + (hero_seg_count - k); "
                    "hero head has 1.0 (65535), empty 0.0."
                ),
            },
            {
                "index": 2,
                "name": "enemy_ttl",
                "rule": (
                    "clamp(max_{s in 1..5 alive} enemy_ttl_steps(s, y, x) / 1000.0, "
                    "0.0, 1.0), empty 0.0."
                ),
            },
            {
                "index": 3,
                "name": "food",
                "rule": "0.0 (0) empty, 0.5 (32768) ambient food, 1.0 (65535) corpse-tagged food.",
            },
        ],
    },
    "orientation": {
        "board_planes": (
            "Canonical arena-aligned lattice (y in [0, grid_h), x in [0, grid_w)), "
            "+x Right/East, +y Down/South; hero head uniquely marked at own_ttl == 1.0."
        ),
        "action_summaries": (
            "Hero ego-relative action order "
            "[0:left, 1:straight, 2:right, 3:boost_left, 4:boost_straight, 5:boost_right]."
        ),
    },
    "summaries": {
        "count": 6,
        "dtype": "float32",
        "bfs_horizon": BFS_HORIZON,
        "formula": (
            "0.8 * (reachable_cells / (grid_h * grid_w)) + "
            "0.2 * mean_{u in reachable}(min(arrival_step(u) - total_ttl(u), 128) / 128.0)"
        ),
        "normal_actions_0_2": (
            "Step d=1 to c1 = head + dir(a). If c1 out-of-bounds or total_ttl(c1) > 1, "
            "value = 0.0. Else 4-neighbor time-expanded BFS from (c1, d=1) up to d <= 128 "
            "admitting neighbor u at step d iff in-bounds and max(own_ttl(u), enemy_ttl(u)) <= d."
        ),
        "boost_treatment_3_5": (
            "If hero not alive or hero_length < min_boost_length, missing-value sentinel = -1.0. "
            "Else step d=1 traverses c1 = head + dir(a) and c2 = head + 2*dir(a), popping "
            "pop1 = 2 + 1[boost_frames + 1 >= boost_length_cost_frames and hero_length > 1] "
            "own tail segments and 1 enemy tail segment. "
            "If c1 or c2 is out-of-bounds, enemy_ttl > 1, or own_ttl > pop1, value = 0.0. "
            "Else time-expanded BFS from (c2, d=1) up to d <= 128 with remaining own_ttl "
            "shifted by -(pop1 - 1)."
        ),
        "missing_value_rules": {
            "illegal_boost_or_dead_boost": ILLEGAL_BOOST_SENTINEL,
            "immediate_step1_collision_or_dead_normal": IMMEDIATE_COLLISION_VALUE,
        },
    },
}

_DX = np.array([0, 1, 0, -1], dtype=np.int64)
_DY = np.array([-1, 0, 1, 0], dtype=np.int64)
_TURNS = np.array([-1, 0, 1, -1, 0, 1], dtype=np.int64)


_ZLIB_BLOCK_RAW_BYTES = 16 * 1024 * 1024
_ZLIB_CODEC_SCHEMA = "ct-zlib-tensor/v1"


def compress_array_to_u8_tensor(
    arr: np.ndarray,
    *,
    byte_shuffle: bool = False,
    level: int = 6,
) -> dict[str, Any]:
    """Losslessly compress a NumPy array into chunked CPU ``uint8`` ``torch.Tensor`` leaves."""
    contig = np.ascontiguousarray(arr)
    itemsize = int(contig.dtype.itemsize)
    do_shuffle = bool(byte_shuffle and itemsize > 1 and contig.size > 0)
    raw_u8 = contig.view(np.uint8).reshape(-1)
    total_bytes = int(raw_u8.size)
    # Align chunk step to itemsize so per-chunk byte shuffle is self-contained.
    block_bytes = (_ZLIB_BLOCK_RAW_BYTES // max(1, itemsize)) * max(1, itemsize)
    chunks: list[torch.Tensor] = []
    for start in range(0, total_bytes, block_bytes):
        sub = raw_u8[start : min(total_bytes, start + block_bytes)]
        if do_shuffle:
            sub = np.ascontiguousarray(sub.reshape(-1, itemsize).T).reshape(-1)
        comp = zlib.compress(memoryview(sub), level=int(level))
        chunks.append(torch.frombuffer(bytearray(comp), dtype=torch.uint8))
    return {
        "codec": _ZLIB_CODEC_SCHEMA,
        "dtype": contig.dtype.str,
        "shape": [int(d) for d in contig.shape],
        "byte_shuffle": do_shuffle,
        "nbytes": total_bytes,
        "chunks": chunks,
    }


def decompress_u8_tensor_to_array(block: Mapping[str, Any]) -> np.ndarray:
    """Losslessly reconstruct the exact NumPy array from ``compress_array_to_u8_tensor``."""
    if block.get("codec") != _ZLIB_CODEC_SCHEMA:
        raise ValueError(f"foreign compressed array codec: {block.get('codec')!r}")
    dt = np.dtype(str(block["dtype"]))
    shape = tuple(int(d) for d in block["shape"])
    expected_bytes = int(block["nbytes"])
    do_shuffle = bool(block.get("byte_shuffle", False))
    itemsize = int(dt.itemsize)
    if expected_bytes != int(np.prod(shape, dtype=np.int64)) * itemsize:
        raise ValueError("compressed array header shape/nbytes mismatch")
    out_u8 = np.empty(expected_bytes, dtype=np.uint8)
    cursor = 0
    for chunk_t in block["chunks"]:
        if not isinstance(chunk_t, torch.Tensor) or chunk_t.dtype != torch.uint8:
            raise ValueError("compressed chunk must be a uint8 torch.Tensor")
        raw = zlib.decompress(memoryview(chunk_t.detach().cpu().contiguous().numpy()))
        n = len(raw)
        if cursor + n > expected_bytes:
            raise ValueError("decompressed byte stream exceeds declared nbytes")
        sub = np.frombuffer(raw, dtype=np.uint8)
        if do_shuffle:
            if n % itemsize != 0:
                raise ValueError("misaligned byte-shuffled block")
            sub = np.ascontiguousarray(sub.reshape(itemsize, -1).T).reshape(-1)
        out_u8[cursor : cursor + n] = sub
        cursor += n
    if cursor != expected_bytes:
        raise ValueError("decompressed byte stream shorter than declared nbytes")
    return out_u8.view(dt).reshape(shape)


def _plane0_matches_p1_or_p2(planes_u16: np.ndarray) -> bool:
    expected_p0 = np.where(
        (planes_u16[:, 1] > 0) | (planes_u16[:, 2] > 0),
        np.uint16(QUANT_MAX),
        np.uint16(0),
    )
    return bool(np.array_equal(planes_u16[:, 0], expected_p0))


@numba.njit(cache=True)
def _encode_csr_row_deltas_u16_nb(offsets: np.ndarray, flat_indices: np.ndarray) -> np.ndarray:
    out = np.empty(flat_indices.shape[0], dtype=np.uint16)
    n_rows = offsets.shape[0] - 1
    for r in range(n_rows):
        s = offsets[r]
        e = offsets[r + 1]
        if e > s:
            prev = np.uint16(0)
            for p in range(s, e):
                cur = flat_indices[p]
                out[p] = cur - prev if p > s else cur
                prev = cur
    return out


@numba.njit(cache=True)
def _decode_csr_row_deltas_u16_nb(offsets: np.ndarray, row_deltas: np.ndarray) -> np.ndarray:
    out = np.empty(row_deltas.shape[0], dtype=np.uint16)
    n_rows = offsets.shape[0] - 1
    for r in range(n_rows):
        s = offsets[r]
        e = offsets[r + 1]
        if e > s:
            acc = np.uint16(0)
            for p in range(s, e):
                acc = np.uint16(acc + row_deltas[p]) if p > s else row_deltas[p]
                out[p] = acc
    return out


@dataclass
class SparseTopologyBatch:
    """Lossless CSR-packed sparse representation of (N, 4, H, W) uint16 planes + (N, 6) summaries.

    When ``p0_derived=True`` (satisfied identically by all ``compute_topology_features``
    outputs where Plane 0 ``wall_occupancy`` equals ``65535 * ((Plane 1 > 0) | (Plane 2 > 0))``),
    only Planes 1..3 nonzeros are stored and Plane 0 is reconstructed losslessly during
    unpacking, halving both RAM and serialized storage.
    """

    num_rows: int
    grid_h: int
    grid_w: int
    offsets: np.ndarray  # (N + 1,) int64
    flat_indices: np.ndarray  # (nnz,) uint16
    values: np.ndarray  # (nnz,) uint16
    summaries: np.ndarray  # (N, 6) float32
    p0_derived: bool = False

    @property
    def nbytes(self) -> int:
        return int(
            self.offsets.nbytes
            + self.flat_indices.nbytes
            + self.values.nbytes
            + self.summaries.nbytes
        )

    def slice_rows(self, indices: np.ndarray | Sequence[int]) -> "SparseTopologyBatch":
        """Extract a subset/reordering of rows preserving ``p0_derived``."""
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        N = int(idx.size)
        if N == 0:
            return SparseTopologyBatch(
                num_rows=0,
                grid_h=self.grid_h,
                grid_w=self.grid_w,
                offsets=np.zeros(1, dtype=np.int64),
                flat_indices=np.zeros(0, dtype=np.uint16),
                values=np.zeros(0, dtype=np.uint16),
                summaries=np.zeros((0, 6), dtype=np.float32),
                p0_derived=self.p0_derived,
            )
        counts = self.offsets[idx + 1] - self.offsets[idx]
        new_offsets = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(counts, out=new_offsets[1:])
        total_nnz = int(new_offsets[-1])
        new_flat = np.empty(total_nnz, dtype=np.uint16)
        new_vals = np.empty(total_nnz, dtype=np.uint16)
        cursor = 0
        for r_i in idx.tolist():
            s = int(self.offsets[r_i])
            e = int(self.offsets[r_i + 1])
            span = e - s
            if span > 0:
                new_flat[cursor : cursor + span] = self.flat_indices[s:e]
                new_vals[cursor : cursor + span] = self.values[s:e]
                cursor += span
        return SparseTopologyBatch(
            num_rows=N,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            offsets=new_offsets,
            flat_indices=new_flat,
            values=new_vals,
            summaries=np.ascontiguousarray(self.summaries[idx], dtype=np.float32),
            p0_derived=self.p0_derived,
        )

    def unpack_rows(
        self, row_indices: np.ndarray | Sequence[int], device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Losslessly reconstruct (B, 4, H, W) float32/int32-ready tensors and (B, 6) summaries."""
        idx = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        B = idx.shape[0]
        hw = self.grid_h * self.grid_w
        plane_elems = 4 * hw
        if plane_elems > 65536:
            raise ValueError("SparseTopologyBatch uint16 flat index requires 4*H*W <= 65536")
        dense = np.zeros((B, plane_elems), dtype=np.uint16)
        _unpack_csr_u16_nb(
            idx,
            self.offsets,
            self.flat_indices,
            self.values,
            dense,
            bool(self.p0_derived),
            int(hw),
        )
        dense_4d = dense.reshape(B, 4, self.grid_h, self.grid_w)
        sums = np.ascontiguousarray(self.summaries[idx], dtype=np.float32)
        # Convert uint16 -> float32 * (1 / 65535.0) deterministically for CPU & CUDA conv kernels.
        planes_t = torch.from_numpy(dense_4d.astype(np.float32) * np.float32(1.0 / QUANT_MAX))
        sums_t = torch.from_numpy(sums)
        if device is not None:
            planes_t = planes_t.to(device, non_blocking=True)
            sums_t = sums_t.to(device, non_blocking=True)
        return planes_t, sums_t

    def unpack_raw_u16(self, row_indices: np.ndarray | Sequence[int]) -> np.ndarray:
        """Reconstruct exact (B, 4, H, W) uint16 array for lossless round-trip verification."""
        idx = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        B = idx.shape[0]
        hw = self.grid_h * self.grid_w
        dense = np.zeros((B, 4 * hw), dtype=np.uint16)
        _unpack_csr_u16_nb(
            idx,
            self.offsets,
            self.flat_indices,
            self.values,
            dense,
            bool(self.p0_derived),
            int(hw),
        )
        return dense.reshape(B, 4, self.grid_h, self.grid_w)

    def to_state_dict(self) -> dict[str, Any]:
        counts_u32 = np.diff(self.offsets).astype(np.uint32)
        row_deltas_u16 = _encode_csr_row_deltas_u16_nb(self.offsets, self.flat_indices)
        values_deltas = np.empty_like(self.values, dtype=np.uint16)
        if self.values.size > 0:
            values_deltas[0] = self.values[0]
            values_deltas[1:] = (
                self.values[1:].astype(np.uint32) - self.values[:-1].astype(np.uint32)
            ).astype(np.uint16)
        return {
            "schema": "ct-sparse-topology-batch/v2",
            "num_rows": int(self.num_rows),
            "grid_h": int(self.grid_h),
            "grid_w": int(self.grid_w),
            "p0_derived": bool(self.p0_derived),
            "counts": compress_array_to_u8_tensor(counts_u32, byte_shuffle=True),
            "flat_index_deltas": compress_array_to_u8_tensor(row_deltas_u16, byte_shuffle=True),
            "values_deltas": compress_array_to_u8_tensor(values_deltas, byte_shuffle=True),
            "summaries": compress_array_to_u8_tensor(self.summaries, byte_shuffle=True),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "SparseTopologyBatch":
        schema = state.get("schema")
        if schema == "ct-sparse-topology-batch/v1":
            return cls(
                num_rows=int(state["num_rows"]),
                grid_h=int(state["grid_h"]),
                grid_w=int(state["grid_w"]),
                offsets=state["offsets"].detach().cpu().numpy().astype(np.int64),
                flat_indices=state["flat_indices"].detach().cpu().numpy().astype(np.uint16),
                values=state["values"].detach().cpu().numpy().astype(np.uint16),
                summaries=state["summaries"].detach().cpu().numpy().astype(np.float32),
                p0_derived=False,
            )
        if schema != "ct-sparse-topology-batch/v2":
            raise ValueError("foreign SparseTopologyBatch schema")
        num_rows = int(state["num_rows"])
        counts_u32 = decompress_u8_tensor_to_array(state["counts"]).astype(np.int64)
        if counts_u32.shape != (num_rows,):
            raise ValueError("corrupt SparseTopologyBatch v2 counts shape")
        offsets = np.zeros(num_rows + 1, dtype=np.int64)
        if num_rows > 0:
            np.cumsum(counts_u32, out=offsets[1:])
        deltas = decompress_u8_tensor_to_array(state["flat_index_deltas"]).astype(np.uint16)
        flat_indices = _decode_csr_row_deltas_u16_nb(offsets, deltas)
        if "values_deltas" in state:
            v_deltas = decompress_u8_tensor_to_array(state["values_deltas"]).astype(np.uint64)
            values = np.cumsum(v_deltas, dtype=np.uint64).astype(np.uint16)
        else:
            values = decompress_u8_tensor_to_array(state["values"]).astype(np.uint16)
        summaries = decompress_u8_tensor_to_array(state["summaries"]).astype(np.float32)
        if int(offsets[-1]) != int(flat_indices.size) or int(values.size) != int(flat_indices.size):
            raise ValueError("corrupt SparseTopologyBatch v2 nnz length")
        return cls(
            num_rows=num_rows,
            grid_h=int(state["grid_h"]),
            grid_w=int(state["grid_w"]),
            offsets=offsets,
            flat_indices=flat_indices,
            values=values,
            summaries=summaries,
            p0_derived=bool(state["p0_derived"]),
        )


@numba.njit(cache=True)
def _unpack_csr_u16_nb(
    row_indices: np.ndarray,
    offsets: np.ndarray,
    flat_indices: np.ndarray,
    values: np.ndarray,
    out_dense: np.ndarray,
    p0_derived: bool,
    hw: int,
) -> None:
    B = row_indices.shape[0]
    three_hw = 3 * hw
    for b in range(B):
        r = row_indices[b]
        start = offsets[r]
        end = offsets[r + 1]
        for p in range(start, end):
            col = flat_indices[p]
            val = values[p]
            out_dense[b, col] = val
            if p0_derived and val > 0 and hw <= col < three_hw:
                out_dense[b, col % hw] = np.uint16(65535)


def pack_sparse_topology(planes_u16: np.ndarray, summaries_f32: np.ndarray) -> SparseTopologyBatch:
    """Losslessly CSR-pack (N, 4, H, W) uint16 planes and (N, 6) float32 summaries."""
    if planes_u16.ndim != 4 or planes_u16.shape[1] != 4 or planes_u16.dtype != np.uint16:
        raise ValueError("planes_u16 must be uint16 with shape (N, 4, H, W)")
    N, _, H, W = planes_u16.shape
    hw = H * W
    if 4 * hw > 65536:
        raise ValueError("4 * H * W exceeds uint16 flat index capacity")
    if summaries_f32.shape != (N, 6):
        raise ValueError("summaries_f32 must have shape (N, 6)")
    p0_derived = N == 0 or _plane0_matches_p1_or_p2(planes_u16)
    if p0_derived and N > 0:
        sub = planes_u16[:, 1:4].reshape(N, 3 * hw)
        nz_rows, nz_sub_cols = np.nonzero(sub)
        nz_cols = nz_sub_cols + hw
        vals = sub[nz_rows, nz_sub_cols].astype(np.uint16)
    else:
        flat = planes_u16.reshape(N, 4 * hw)
        nz_rows, nz_cols = np.nonzero(flat)
        vals = flat[nz_rows, nz_cols].astype(np.uint16)
    counts = np.bincount(nz_rows, minlength=N).astype(np.int64)
    offsets = np.empty(N + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return SparseTopologyBatch(
        num_rows=N,
        grid_h=H,
        grid_w=W,
        offsets=offsets,
        flat_indices=nz_cols.astype(np.uint16),
        values=vals,
        summaries=np.ascontiguousarray(summaries_f32, dtype=np.float32),
        p0_derived=p0_derived,
    )


class IncrementalSparseTopologyBuilder:
    """Append sparse topology rows incrementally during collection without dense O(N) peak."""

    def __init__(self, grid_h: int, grid_w: int) -> None:
        if 4 * grid_h * grid_w > 65536:
            raise ValueError("4 * grid_h * grid_w exceeds uint16 capacity")
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self._counts: list[int] = []
        self._idx_chunks: list[np.ndarray] = []
        self._val_chunks: list[np.ndarray] = []
        self._sum_chunks: list[np.ndarray] = []
        self._p0_derived: bool = True
        self._nnz_total: int = 0

    def __len__(self) -> int:
        return len(self._counts)

    @property
    def nbytes(self) -> int:
        n_rows = len(self._counts)
        return int((n_rows + 1) * 8 + self._nnz_total * 4 + n_rows * 24)

    def _downgrade_existing_to_explicit_p0(self) -> None:
        """Materialize Plane-0 entries on stored rows if a custom non-derivable row arrives."""
        if not self._p0_derived:
            return
        if self._counts:
            prev = self.build()
            dense_prev = prev.unpack_raw_u16(np.arange(prev.num_rows, dtype=np.int64))
            explicit_prev = pack_sparse_topology(dense_prev, prev.summaries)
            if explicit_prev.p0_derived:
                # Force explicit 4-plane packing.
                hw = self.grid_h * self.grid_w
                flat = dense_prev.reshape(prev.num_rows, 4 * hw)
                nz_r, nz_c = np.nonzero(flat)
                cnts = np.bincount(nz_r, minlength=prev.num_rows)
                self._counts = [int(c) for c in cnts.tolist()]
                self._idx_chunks = [nz_c.astype(np.uint16)]
                self._val_chunks = [flat[nz_r, nz_c].astype(np.uint16)]
                self._sum_chunks = [prev.summaries]
                self._nnz_total = int(nz_c.size)
            else:
                cnts = np.diff(explicit_prev.offsets)
                self._counts = [int(c) for c in cnts.tolist()]
                self._idx_chunks = [explicit_prev.flat_indices]
                self._val_chunks = [explicit_prev.values]
                self._sum_chunks = [explicit_prev.summaries]
                self._nnz_total = int(explicit_prev.flat_indices.size)
        self._p0_derived = False

    def append_batch(self, planes_u16: np.ndarray, summaries_f32: np.ndarray) -> None:
        if planes_u16.shape[0] == 0:
            return
        N = planes_u16.shape[0]
        hw = self.grid_h * self.grid_w
        if self._p0_derived and not _plane0_matches_p1_or_p2(planes_u16):
            self._downgrade_existing_to_explicit_p0()
        if self._p0_derived:
            sub = planes_u16[:, 1:4].reshape(N, 3 * hw)
            nz_r, nz_sub_c = np.nonzero(sub)
            nz_c = (nz_sub_c + hw).astype(np.uint16)
            vals = sub[nz_r, nz_sub_c].astype(np.uint16)
        else:
            flat = planes_u16.reshape(N, 4 * hw)
            nz_r, nz_c_raw = np.nonzero(flat)
            nz_c = nz_c_raw.astype(np.uint16)
            vals = flat[nz_r, nz_c_raw].astype(np.uint16)
        counts = np.bincount(nz_r, minlength=N)
        self._counts.extend(int(c) for c in counts.tolist())
        self._idx_chunks.append(nz_c)
        self._val_chunks.append(vals)
        self._sum_chunks.append(np.ascontiguousarray(summaries_f32, dtype=np.float32))
        self._nnz_total += int(nz_c.size)

    def build(self) -> SparseTopologyBatch:
        N = len(self._counts)
        offsets = np.zeros(N + 1, dtype=np.int64)
        if N > 0:
            offsets[1:] = np.cumsum(np.asarray(self._counts, dtype=np.int64))
        flat_idx = (
            np.concatenate(self._idx_chunks) if self._idx_chunks else np.zeros(0, dtype=np.uint16)
        )
        vals = (
            np.concatenate(self._val_chunks) if self._val_chunks else np.zeros(0, dtype=np.uint16)
        )
        sums = (
            np.concatenate(self._sum_chunks, axis=0)
            if self._sum_chunks
            else np.zeros((0, 6), dtype=np.float32)
        )
        return SparseTopologyBatch(
            num_rows=N,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            offsets=offsets,
            flat_indices=flat_idx,
            values=vals,
            summaries=sums,
            p0_derived=self._p0_derived,
        )


def _quantize_u16(val: float) -> int:
    if val <= 0.0:
        return 0
    if val >= 1.0:
        return QUANT_MAX
    return int(np.floor(val * QUANT_MAX + 0.5))


def _time_expanded_bfs_ref(
    own_ttl: np.ndarray,
    enemy_ttl: np.ndarray,
    start_x: int,
    start_y: int,
    own_shift: int,
    horizon: int,
) -> float:
    H, W = own_ttl.shape
    dist = np.full((H, W), -1, dtype=np.int64)
    qx = np.empty(H * W, dtype=np.int64)
    qy = np.empty(H * W, dtype=np.int64)
    dist[start_y, start_x] = 1
    qx[0] = start_x
    qy[0] = start_y
    head = 0
    tail = 1
    slack_sum = 0.0
    # At start cell (d = 1), compute its slack too:
    eff_own0 = own_ttl[start_y, start_x] - own_shift
    if eff_own0 < 0:
        eff_own0 = 0
    tot0 = eff_own0 if eff_own0 >= enemy_ttl[start_y, start_x] else enemy_ttl[start_y, start_x]
    s0 = 1 - tot0
    if s0 < 0:
        s0 = 0
    if s0 > horizon:
        s0 = horizon
    slack_sum += float(s0) / float(horizon)

    while head < tail:
        cx = int(qx[head])
        cy = int(qy[head])
        head += 1
        d_next = int(dist[cy, cx]) + 1
        if d_next > horizon:
            continue
        for dir_i in range(4):
            nx = cx + int(_DX[dir_i])
            ny = cy + int(_DY[dir_i])
            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                continue
            if dist[ny, nx] >= 0:
                continue
            eff_own = int(own_ttl[ny, nx]) - own_shift
            if eff_own < 0:
                eff_own = 0
            en = int(enemy_ttl[ny, nx])
            tot = eff_own if eff_own >= en else en
            if tot <= d_next:
                dist[ny, nx] = d_next
                qx[tail] = nx
                qy[tail] = ny
                tail += 1
                sl = d_next - tot
                if sl > horizon:
                    sl = horizon
                slack_sum += float(sl) / float(horizon)
    reach_frac = float(tail) / float(H * W)
    mean_slack = slack_sum / float(tail)
    return float(np.float32(REACH_WEIGHT * reach_frac + SLACK_WEIGHT * mean_slack))


def compute_topology_features_reference(
    sim: Any, world_rows: Sequence[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-Python/NumPy exact reference oracle for geometric fixture verification."""
    rows = list(range(int(sim.E))) if world_rows is None else [int(r) for r in world_rows]
    N = len(rows)
    H, W = int(sim.grid_h), int(sim.grid_w)
    S, cap = int(sim.S), int(sim.cap)
    min_boost_len = int(sim.cfg.min_boost_length)
    boost_cost_frames = int(sim.cfg.boost_length_cost_frames)

    bodies = sim.bodies.detach().cpu().numpy()
    head_ptr = sim.head_ptr.detach().cpu().numpy()
    seg_count = sim.seg_count.detach().cpu().numpy()
    length = sim.length.detach().cpu().numpy()
    alive = sim.alive.detach().cpu().numpy()
    direction = sim.direction.detach().cpu().numpy()
    boost_frames = sim.boost_frames.detach().cpu().numpy()
    food_cells = sim.food_cells.detach().cpu().numpy()
    food_present = sim.food_present.detach().cpu().numpy()
    corpse_food = sim.corpse_food.detach().cpu().numpy()

    planes = np.zeros((N, 4, H, W), dtype=np.uint16)
    summaries = np.zeros((N, 6), dtype=np.float32)

    for out_i, e in enumerate(rows):
        own_ttl = np.zeros((H, W), dtype=np.int64)
        enemy_ttl = np.zeros((H, W), dtype=np.int64)

        if alive[e, 0]:
            sc0 = int(seg_count[e, 0])
            len0 = int(length[e, 0])
            hp0 = int(head_ptr[e, 0])
            slack0 = max(0, len0 - sc0)
            denom0 = float(max(1, len0))
            for k in range(sc0):
                ring = (hp0 - k) % cap
                cx, cy = int(bodies[e, 0, ring, 0]), int(bodies[e, 0, ring, 1])
                if 0 <= cx < W and 0 <= cy < H:
                    ttl_val = slack0 + (sc0 - k)
                    if ttl_val > own_ttl[cy, cx]:
                        own_ttl[cy, cx] = ttl_val
            for y in range(H):
                for x in range(W):
                    if own_ttl[y, x] > 0:
                        planes[out_i, 0, y, x] = QUANT_MAX
                        planes[out_i, 1, y, x] = _quantize_u16(
                            min(1.0, float(own_ttl[y, x]) / denom0)
                        )

        for s in range(1, S):
            if not alive[e, s]:
                continue
            scs = int(seg_count[e, s])
            lens = int(length[e, s])
            hps = int(head_ptr[e, s])
            slacks = max(0, lens - scs)
            for k in range(scs):
                ring = (hps - k) % cap
                cx, cy = int(bodies[e, s, ring, 0]), int(bodies[e, s, ring, 1])
                if 0 <= cx < W and 0 <= cy < H:
                    ttl_val = slacks + (scs - k)
                    if ttl_val > enemy_ttl[cy, cx]:
                        enemy_ttl[cy, cx] = ttl_val
        for y in range(H):
            for x in range(W):
                if enemy_ttl[y, x] > 0:
                    planes[out_i, 0, y, x] = QUANT_MAX
                    planes[out_i, 2, y, x] = _quantize_u16(
                        min(1.0, float(enemy_ttl[y, x]) / ENEMY_TTL_NORM_CAP)
                    )

        for f_idx in range(food_present.shape[1]):
            if not food_present[e, f_idx]:
                continue
            fx, fy = int(food_cells[e, f_idx, 0]), int(food_cells[e, f_idx, 1])
            if 0 <= fx < W and 0 <= fy < H:
                val_u16 = QUANT_MAX if corpse_food[e, f_idx] else 32768
                if val_u16 > planes[out_i, 3, fy, fx]:
                    planes[out_i, 3, fy, fx] = val_u16

        if not alive[e, 0]:
            summaries[out_i, :3] = IMMEDIATE_COLLISION_VALUE
            summaries[out_i, 3:] = ILLEGAL_BOOST_SENTINEL
            continue

        hp0 = int(head_ptr[e, 0])
        hx, hy = int(bodies[e, 0, hp0, 0]), int(bodies[e, 0, hp0, 1])
        hdir = int(direction[e, 0])
        hlen = int(length[e, 0])
        hbframes = int(boost_frames[e, 0])
        can_boost = hlen >= min_boost_len
        pop1_boost = 2 + (1 if (hbframes + 1 >= boost_cost_frames and hlen > 1) else 0)

        for a in range(6):
            cand_dir = (hdir + int(_TURNS[a])) % 4
            dx, dy = int(_DX[cand_dir]), int(_DY[cand_dir])
            if a < 3:
                c1x, c1y = hx + dx, hy + dy
                if c1x < 0 or c1x >= W or c1y < 0 or c1y >= H:
                    summaries[out_i, a] = IMMEDIATE_COLLISION_VALUE
                    continue
                if own_ttl[c1y, c1x] > 1 or enemy_ttl[c1y, c1x] > 1:
                    summaries[out_i, a] = IMMEDIATE_COLLISION_VALUE
                    continue
                summaries[out_i, a] = _time_expanded_bfs_ref(
                    own_ttl, enemy_ttl, c1x, c1y, 0, BFS_HORIZON
                )
            else:
                if not can_boost:
                    summaries[out_i, a] = ILLEGAL_BOOST_SENTINEL
                    continue
                c1x, c1y = hx + dx, hy + dy
                c2x, c2y = hx + 2 * dx, hy + 2 * dy
                if (
                    c1x < 0
                    or c1x >= W
                    or c1y < 0
                    or c1y >= H
                    or c2x < 0
                    or c2x >= W
                    or c2y < 0
                    or c2y >= H
                ):
                    summaries[out_i, a] = IMMEDIATE_COLLISION_VALUE
                    continue
                if (
                    enemy_ttl[c1y, c1x] > 1
                    or own_ttl[c1y, c1x] > pop1_boost
                    or enemy_ttl[c2y, c2x] > 1
                    or own_ttl[c2y, c2x] > pop1_boost
                ):
                    summaries[out_i, a] = IMMEDIATE_COLLISION_VALUE
                    continue
                summaries[out_i, a] = _time_expanded_bfs_ref(
                    own_ttl, enemy_ttl, c2x, c2y, pop1_boost - 1, BFS_HORIZON
                )

    return planes, summaries


@numba.njit(cache=True)
def _compute_topology_nb(
    rows: np.ndarray,
    bodies: np.ndarray,
    head_ptr: np.ndarray,
    seg_count: np.ndarray,
    length: np.ndarray,
    alive: np.ndarray,
    direction: np.ndarray,
    boost_frames: np.ndarray,
    food_cells: np.ndarray,
    food_present: np.ndarray,
    corpse_food: np.ndarray,
    H: int,
    W: int,
    min_boost_len: int,
    boost_cost_frames: int,
    planes: np.ndarray,
    summaries: np.ndarray,
) -> None:
    N = rows.shape[0]
    S = alive.shape[1]
    cap = bodies.shape[2]
    F = food_present.shape[1]
    dx_arr = np.array([0, 1, 0, -1], dtype=np.int64)
    dy_arr = np.array([-1, 0, 1, 0], dtype=np.int64)
    turns_arr = np.array([-1, 0, 1, -1, 0, 1], dtype=np.int64)

    own_ttl = np.empty((H, W), dtype=np.int64)
    enemy_ttl = np.empty((H, W), dtype=np.int64)
    dist = np.empty((H, W), dtype=np.int64)
    qx = np.empty(H * W, dtype=np.int64)
    qy = np.empty(H * W, dtype=np.int64)

    for out_i in range(N):
        e = rows[out_i]
        own_ttl.fill(0)
        enemy_ttl.fill(0)

        if alive[e, 0]:
            sc0 = seg_count[e, 0]
            len0 = length[e, 0]
            hp0 = head_ptr[e, 0]
            slack0 = len0 - sc0
            if slack0 < 0:
                slack0 = 0
            denom0 = float(len0 if len0 > 1 else 1)
            for k in range(sc0):
                ring = (hp0 - k) % cap
                cx = bodies[e, 0, ring, 0]
                cy = bodies[e, 0, ring, 1]
                if 0 <= cx < W and 0 <= cy < H:
                    ttl_val = slack0 + (sc0 - k)
                    if ttl_val > own_ttl[cy, cx]:
                        own_ttl[cy, cx] = ttl_val
            for y in range(H):
                for x in range(W):
                    tv = own_ttl[y, x]
                    if tv > 0:
                        planes[out_i, 0, y, x] = 65535
                        ratio = float(tv) / denom0
                        if ratio >= 1.0:
                            q = 65535
                        elif ratio <= 0.0:
                            q = 0
                        else:
                            q = int(np.floor(ratio * 65535.0 + 0.5))
                        planes[out_i, 1, y, x] = np.uint16(q)

        for s in range(1, S):
            if not alive[e, s]:
                continue
            scs = seg_count[e, s]
            lens = length[e, s]
            hps = head_ptr[e, s]
            slacks = lens - scs
            if slacks < 0:
                slacks = 0
            for k in range(scs):
                ring = (hps - k) % cap
                cx = bodies[e, s, ring, 0]
                cy = bodies[e, s, ring, 1]
                if 0 <= cx < W and 0 <= cy < H:
                    ttl_val = slacks + (scs - k)
                    if ttl_val > enemy_ttl[cy, cx]:
                        enemy_ttl[cy, cx] = ttl_val
        for y in range(H):
            for x in range(W):
                tv = enemy_ttl[y, x]
                if tv > 0:
                    planes[out_i, 0, y, x] = 65535
                    ratio = float(tv) / 1000.0
                    if ratio >= 1.0:
                        q = 65535
                    elif ratio <= 0.0:
                        q = 0
                    else:
                        q = int(np.floor(ratio * 65535.0 + 0.5))
                    planes[out_i, 2, y, x] = np.uint16(q)

        for f_idx in range(F):
            if not food_present[e, f_idx]:
                continue
            fx = food_cells[e, f_idx, 0]
            fy = food_cells[e, f_idx, 1]
            if 0 <= fx < W and 0 <= fy < H:
                val_u16 = np.uint16(65535 if corpse_food[e, f_idx] else 32768)
                if val_u16 > planes[out_i, 3, fy, fx]:
                    planes[out_i, 3, fy, fx] = val_u16

        if not alive[e, 0]:
            for a in range(3):
                summaries[out_i, a] = np.float32(0.0)
            for a in range(3, 6):
                summaries[out_i, a] = np.float32(-1.0)
            continue

        hp0 = head_ptr[e, 0]
        hx = bodies[e, 0, hp0, 0]
        hy = bodies[e, 0, hp0, 1]
        hdir = direction[e, 0]
        hlen = length[e, 0]
        hbframes = boost_frames[e, 0]
        can_boost = hlen >= min_boost_len
        pop1_boost = 2 + (1 if (hbframes + 1 >= boost_cost_frames and hlen > 1) else 0)

        for a in range(6):
            cand_dir = (hdir + turns_arr[a]) % 4
            ddx = dx_arr[cand_dir]
            ddy = dy_arr[cand_dir]
            start_x = 0
            start_y = 0
            own_shift = 0
            if a < 3:
                c1x = hx + ddx
                c1y = hy + ddy
                if c1x < 0 or c1x >= W or c1y < 0 or c1y >= H:
                    summaries[out_i, a] = np.float32(0.0)
                    continue
                if own_ttl[c1y, c1x] > 1 or enemy_ttl[c1y, c1x] > 1:
                    summaries[out_i, a] = np.float32(0.0)
                    continue
                start_x = c1x
                start_y = c1y
                own_shift = 0
            else:
                if not can_boost:
                    summaries[out_i, a] = np.float32(-1.0)
                    continue
                c1x = hx + ddx
                c1y = hy + ddy
                c2x = hx + 2 * ddx
                c2y = hy + 2 * ddy
                if (
                    c1x < 0
                    or c1x >= W
                    or c1y < 0
                    or c1y >= H
                    or c2x < 0
                    or c2x >= W
                    or c2y < 0
                    or c2y >= H
                ):
                    summaries[out_i, a] = np.float32(0.0)
                    continue
                if (
                    enemy_ttl[c1y, c1x] > 1
                    or own_ttl[c1y, c1x] > pop1_boost
                    or enemy_ttl[c2y, c2x] > 1
                    or own_ttl[c2y, c2x] > pop1_boost
                ):
                    summaries[out_i, a] = np.float32(0.0)
                    continue
                start_x = c2x
                start_y = c2y
                own_shift = pop1_boost - 1

            dist.fill(-1)
            dist[start_y, start_x] = 1
            qx[0] = start_x
            qy[0] = start_y
            q_head = 0
            q_tail = 1

            eff_own0 = own_ttl[start_y, start_x] - own_shift
            if eff_own0 < 0:
                eff_own0 = 0
            en0 = enemy_ttl[start_y, start_x]
            tot0 = eff_own0 if eff_own0 >= en0 else en0
            s0 = 1 - tot0
            if s0 < 0:
                s0 = 0
            if s0 > 128:
                s0 = 128
            slack_sum = float(s0) / 128.0

            while q_head < q_tail:
                cx = qx[q_head]
                cy = qy[q_head]
                q_head += 1
                d_next = dist[cy, cx] + 1
                if d_next > 128:
                    continue
                for dir_i in range(4):
                    nx = cx + dx_arr[dir_i]
                    ny = cy + dy_arr[dir_i]
                    if nx < 0 or nx >= W or ny < 0 or ny >= H:
                        continue
                    if dist[ny, nx] >= 0:
                        continue
                    eff_own = own_ttl[ny, nx] - own_shift
                    if eff_own < 0:
                        eff_own = 0
                    en = enemy_ttl[ny, nx]
                    tot = eff_own if eff_own >= en else en
                    if tot <= d_next:
                        dist[ny, nx] = d_next
                        qx[q_tail] = nx
                        qy[q_tail] = ny
                        q_tail += 1
                        sl = d_next - tot
                        if sl > 128:
                            sl = 128
                        slack_sum += float(sl) / 128.0

            reach_frac = float(q_tail) / float(H * W)
            mean_slack = slack_sum / float(q_tail)
            summaries[out_i, a] = np.float32(0.8 * reach_frac + 0.2 * mean_slack)


def compute_topology_features(
    sim: Any, world_rows: Sequence[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute frozen (N, 4, H, W) uint16 planes and (N, 6) float32 summaries via Numba."""
    if world_rows is None:
        rows_arr = np.arange(int(sim.E), dtype=np.int64)
        rows_t = None
    else:
        rows_arr = np.asarray(list(world_rows), dtype=np.int64)
        rows_t = torch.as_tensor(rows_arr, dtype=torch.long, device=sim.device)

    N = int(rows_arr.shape[0])
    H, W = int(sim.grid_h), int(sim.grid_w)
    planes = np.zeros((N, 4, H, W), dtype=np.uint16)
    summaries = np.zeros((N, 6), dtype=np.float32)
    if N == 0:
        return planes, summaries

    # Transfer ONLY selected world rows and ONLY up to max_seg_count ring span
    # instead of copying (1024, 6, 10400, 2) on every anchor tick!
    if rows_t is not None:
        seg_sub = sim.seg_count.index_select(0, rows_t)
        hp_sub = sim.head_ptr.index_select(0, rows_t)
        len_sub = sim.length.index_select(0, rows_t)
        alive_sub = sim.alive.index_select(0, rows_t)
        dir_sub = sim.direction.index_select(0, rows_t)
        bf_sub = sim.boost_frames.index_select(0, rows_t)
        fc_sub = sim.food_cells.index_select(0, rows_t)
        fp_sub = sim.food_present.index_select(0, rows_t)
        cf_sub = sim.corpse_food.index_select(0, rows_t)
        max_seg = max(1, int(seg_sub.max().item()))
        k = torch.arange(max_seg, device=sim.device)[None, None, :]
        rings = (hp_sub[..., None] - k) % int(sim.cap)
        bodies_raw = sim.bodies.index_select(0, rows_t)
        # Re-normalize compact ring so head_ptr = max_seg - 1 and k in 0..sc-1 sits at max_seg-1-k
        ordered = bodies_raw.gather(2, rings[..., None].expand(-1, -1, -1, 2))
        bodies_compact = torch.flip(ordered, dims=(2,)).detach().cpu().numpy()
        hp_compact = np.full((N, int(sim.S)), max_seg - 1, dtype=np.int64)
        local_rows = np.arange(N, dtype=np.int64)
        _compute_topology_nb(
            local_rows,
            bodies_compact,
            hp_compact,
            seg_sub.detach().cpu().numpy(),
            len_sub.detach().cpu().numpy(),
            alive_sub.detach().cpu().numpy(),
            dir_sub.detach().cpu().numpy(),
            bf_sub.detach().cpu().numpy(),
            fc_sub.detach().cpu().numpy(),
            fp_sub.detach().cpu().numpy(),
            cf_sub.detach().cpu().numpy(),
            H,
            W,
            int(sim.cfg.min_boost_length),
            int(sim.cfg.boost_length_cost_frames),
            planes,
            summaries,
        )
        return planes, summaries

    max_seg = max(1, int(sim.seg_count.max().item()))
    k = torch.arange(max_seg, device=sim.device)[None, None, :]
    rings = (sim.head_ptr[..., None] - k) % int(sim.cap)
    ordered = sim.bodies.gather(2, rings[..., None].expand(-1, -1, -1, 2))
    bodies_compact = torch.flip(ordered, dims=(2,)).detach().cpu().numpy()
    hp_compact = np.full((int(sim.E), int(sim.S)), max_seg - 1, dtype=np.int64)
    _compute_topology_nb(
        rows_arr,
        bodies_compact,
        hp_compact,
        sim.seg_count.detach().cpu().numpy(),
        sim.length.detach().cpu().numpy(),
        sim.alive.detach().cpu().numpy(),
        sim.direction.detach().cpu().numpy(),
        sim.boost_frames.detach().cpu().numpy(),
        sim.food_cells.detach().cpu().numpy(),
        sim.food_present.detach().cpu().numpy(),
        sim.corpse_food.detach().cpu().numpy(),
        H,
        W,
        int(sim.cfg.min_boost_length),
        int(sim.cfg.boost_length_cost_frames),
        planes,
        summaries,
    )
    return planes, summaries


def validate_geometric_fixtures() -> dict[str, Any]:
    """Validate frozen dynamic-topology approximations on deterministic geometric fixtures."""
    from research.sandbox_a_resident.ct_mechanics import CTDeviceSim
    from src.simd_env.batch_sim import BatchSimConfig

    cfg = BatchSimConfig(
        num_envs=3,
        num_snakes=6,
        game_width=600,  # 30 cells wide
        game_height=400,  # 20 cells high
        segment_size=20,
        wall_thickness=20,
        initial_food=2,
        max_food=4,
        max_capacity=256,
        body_storage_capacity=256,
        mechanics_version=1,
    )
    sim = CTDeviceSim(cfg, device="cpu", train_mode=False, allow_respawn=True)
    sim.bodies.zero_()
    sim.head_ptr.zero_()
    sim.seg_count.zero_()
    sim.length.fill_(1)
    sim.alive.fill_(False)
    sim.direction.fill_(1)
    sim.boost_frames.zero_()
    sim.food_present.fill_(False)
    sim.corpse_food.fill_(False)
    sim.food_count.zero_()

    def place_snake(
        env: int,
        slot: int,
        cells_head_to_tail: list[tuple[int, int]],
        direction: int,
        length_override: int | None = None,
    ) -> None:
        n = len(cells_head_to_tail)
        sim.alive[env, slot] = True
        sim.direction[env, slot] = direction
        sim.seg_count[env, slot] = n
        sim.length[env, slot] = length_override if length_override is not None else n
        sim.head_ptr[env, slot] = n - 1
        for k, (cx, cy) in enumerate(cells_head_to_tail):
            sim.bodies[env, slot, n - 1 - k, 0] = cx
            sim.bodies[env, slot, n - 1 - k, 1] = cy

    # Env 0: Corridor bifurcation test (TTL <= arrival vs sealed high-TTL trap)!
    # Hero head at (5, 10) facing East (direction=1):
    # - Action 0 (Left = North to y=9): enters a 4-cell pocket sealed by enemy high-TTL wall
    #   (TTL ~50 >> arrival=4) -> tiny trapped summary!
    # - Action 1 (Straight = East to (6,10)): immediate collision (TTL=50 > 1) -> exact 0.0!
    # - Action 2 (Right = South to y=11): enters a 4-cell tunnel plugged at (5, 14) by own tail
    #   with TTL = 3 <= arrival_step = 4 -> unwinds into open 30x20 board!
    hero_env0 = [
        (5, 10),  # k=0 head, ttl=11
        (4, 10),  # k=1, ttl=10
        (3, 10),  # k=2, ttl=9
        (3, 11),  # k=3, ttl=8
        (3, 12),  # k=4, ttl=7
        (3, 13),  # k=5, ttl=6
        (3, 14),  # k=6, ttl=5
        (4, 14),  # k=7, ttl=4
        # k=8, ttl=3 -> reached from (5,10) South at d=4 >= ttl=3 -> OPENS!
        (5, 14),
        (6, 13),  # k=9, ttl=2 (right wall of tunnel near bottom)
        (6, 12),  # k=10, ttl=1
    ]
    place_snake(0, 0, hero_env0, direction=1)
    # Seal right wall of South tunnel at (6,11), seal North pocket {(5,9),(5,8)}, block East (6,10)
    enemy_wall = (
        [
            (6, 10),  # blocks Straight immediately!
            (
                6,
                11,
            ),  # seals East side of South tunnel so South MUST pass through own tail at (5,14)!
            (4, 9),
            (4, 8),
            (4, 7),
            (5, 7),
            (6, 7),
            (6, 8),
            (6, 9),  # completely seals North pocket {(5,9), (5,8)}!
        ]
        + [(20, y) for y in range(1, 19)]
        + [(21, y) for y in range(18, 0, -1)]
    )
    place_snake(0, 1, enemy_wall, direction=0)
    # Add 1 ambient food at (10, 10) and 1 corpse food at (12, 12)
    sim.food_cells[0, 0] = torch.tensor([10, 10])
    sim.food_present[0, 0] = True
    sim.corpse_food[0, 0] = False
    sim.food_cells[0, 1] = torch.tensor([12, 12])
    sim.food_present[0, 1] = True
    sim.corpse_food[0, 1] = True
    sim.food_count[0] = 2

    # Env 1: Illegal boost sentinel test (length=3 < min_boost_length=5) + top-wall OOB collision!
    # Hero head at (10, 0) facing North (dir=0) -> Straight (1) goes to (10, -1) OOB -> 0.0!
    # Boost actions 3, 4, 5 must all equal -1.0!
    place_snake(1, 0, [(10, 0), (10, 1), (10, 2)], direction=0)

    # Env 2: Legal boost 2-cell traversal over 2-pop own tail!
    # Hero length = 6 (>= 5), head at (10, 10) facing East (direction=1):
    # Put own tail k=4 (ttl = 2) at c1 = (11, 10)!
    # Normal Straight (action 1) moves 1 cell to (11, 10): own_ttl=2 > 1 -> BLOCKED (0.0)!
    # Boost Straight (action 4) moves 2 cells through (11, 10) to (12, 10), popping pop1=2 tail
    # segments, so own_ttl(11,10)=2 <= pop1=2 -> LEGAL & OPEN (> 0.3)!
    hero_env2 = [
        (10, 10),  # k=0, ttl=6
        (10, 11),  # k=1, ttl=5
        (11, 11),  # k=2, ttl=4
        (12, 11),  # k=3, ttl=3
        (
            11,
            10,
        ),  # k=4, ttl=2 -> blocks normal step (ttl=2 > 1), clears on boost (ttl=2 <= pop1=2)!
        (12, 12),  # k=5, ttl=1
    ]
    place_snake(2, 0, hero_env2, direction=1)

    ref_planes, ref_sums = compute_topology_features_reference(sim)
    nb_planes, nb_sums = compute_topology_features(sim)

    np.testing.assert_array_equal(
        ref_planes, nb_planes, err_msg="Numba vs reference planes mismatch"
    )
    np.testing.assert_allclose(
        ref_sums, nb_sums, atol=1e-6, rtol=1e-6, err_msg="Numba vs reference summaries mismatch"
    )

    # Verify sparse CSR round-trip is 100% bitwise lossless:
    packed = pack_sparse_topology(nb_planes, nb_sums)
    unpacked_u16 = packed.unpack_raw_u16([0, 1, 2])
    np.testing.assert_array_equal(unpacked_u16, nb_planes, err_msg="sparse round-trip lossy")

    # Fixture 0 assertions:
    assert nb_planes[0, 1, 10, 5] == QUANT_MAX, "hero head must have own_ttl == 65535"
    assert nb_planes[0, 3, 10, 10] == 32768, "ambient food must quantize to 32768"
    assert nb_planes[0, 3, 12, 12] == QUANT_MAX, "corpse food must quantize to 65535"
    sealed_north = float(nb_sums[0, 0])
    blocked_east = float(nb_sums[0, 1])
    unwinding_south = float(nb_sums[0, 2])
    assert blocked_east == 0.0, f"immediate collision must be 0.0, got {blocked_east}"
    assert (
        0.0 < sealed_north < 0.02
    ), f"sealed 2-cell pocket expected tiny reach, got {sealed_north}"
    assert (
        unwinding_south > 0.35
    ), f"unwinding tail corridor must flood open board, got {unwinding_south}"
    assert unwinding_south > 20.0 * sealed_north

    # Fixture 1 assertions:
    assert float(nb_sums[1, 1]) == 0.0, "OOB wall collision must be 0.0"
    assert float(nb_sums[1, 0]) > 0.2 and float(nb_sums[1, 2]) > 0.2
    np.testing.assert_array_equal(
        nb_sums[1, 3:6],
        np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        err_msg="illegal boost must use -1.0 sentinel",
    )

    # Fixture 2 assertions (normal blocked by own_ttl=2 vs boost clearing own_ttl=2 via 2-seg pop!):
    assert (
        float(nb_sums[2, 1]) == 0.0
    ), f"normal step into own_ttl=2 must be 0.0, got {nb_sums[2, 1]}"
    assert (
        float(nb_sums[2, 4]) > 0.35
    ), f"boost step popping 2 tail segs must clear own_ttl=2, got {nb_sums[2, 4]}"

    return {
        "schema": "t34-cf-topology-fixtures/v1",
        "status": "PASS",
        "sealed_north_summary": sealed_north,
        "unwinding_south_summary": unwinding_south,
        "boost_pop2_clear_summary": float(nb_sums[2, 4]),
    }
