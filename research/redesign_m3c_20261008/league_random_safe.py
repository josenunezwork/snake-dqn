"""Batched exact implementation of the scripted ``random_safe`` anchor.

The public anchor hashes a small ASCII key for every decision.  This module
keeps that protocol byte-for-byte, but caches the immutable
``version|kind|world_seed|slot|`` prefix and hashes all selected rows in one
Numba call.  It is intentionally pool-local: the canonical Python anchor
remains the reference implementation.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
from numba import njit

from src.evaluation.anchors import SCRIPTED_ANCHOR_VERSION, watch_anchor_frame

_PREFIX_CAPACITY = 96
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_SHA256_K = np.array(
    [
        0x428A2F98,
        0x71374491,
        0xB5C0FBCF,
        0xE9B5DBA5,
        0x3956C25B,
        0x59F111F1,
        0x923F82A4,
        0xAB1C5ED5,
        0xD807AA98,
        0x12835B01,
        0x243185BE,
        0x550C7DC3,
        0x72BE5D74,
        0x80DEB1FE,
        0x9BDC06A7,
        0xC19BF174,
        0xE49B69C1,
        0xEFBE4786,
        0x0FC19DC6,
        0x240CA1CC,
        0x2DE92C6F,
        0x4A7484AA,
        0x5CB0A9DC,
        0x76F988DA,
        0x983E5152,
        0xA831C66D,
        0xB00327C8,
        0xBF597FC7,
        0xC6E00BF3,
        0xD5A79147,
        0x06CA6351,
        0x14292967,
        0x27B70A85,
        0x2E1B2138,
        0x4D2C6DFC,
        0x53380D13,
        0x650A7354,
        0x766A0ABB,
        0x81C2C92E,
        0x92722C85,
        0xA2BFE8A1,
        0xA81A664B,
        0xC24B8B70,
        0xC76C51A3,
        0xD192E819,
        0xD6990624,
        0xF40E3585,
        0x106AA070,
        0x19A4C116,
        0x1E376C08,
        0x2748774C,
        0x34B0BCB5,
        0x391C0CB3,
        0x4ED8AA4A,
        0x5B9CCA4F,
        0x682E6FF3,
        0x748F82EE,
        0x78A5636F,
        0x84C87814,
        0x8CC70208,
        0x90BEFFFA,
        0xA4506CEB,
        0xBEF9A3F7,
        0xC67178F2,
    ],
    dtype=np.uint32,
)


@njit(cache=True, nogil=True)
def _rotr(value: np.uint32, count: int) -> np.uint32:
    return np.uint32((value >> count) | (value << (32 - count)))


@njit(cache=True, nogil=True)
def _append_signed_decimal(buf: np.ndarray, start: int, value: np.int64) -> int:
    """Append the exact decimal spelling of an int64, including INT64_MIN."""
    pos = start
    unsigned = np.uint64(value)
    if value < 0:
        buf[pos] = ord("-")
        pos += 1
        # Avoid negating INT64_MIN in signed arithmetic.
        unsigned = np.uint64(-(value + 1)) + np.uint64(1)
    digits_start = pos
    if unsigned == 0:
        buf[pos] = ord("0")
        return pos + 1
    while unsigned != 0:
        buf[pos] = ord("0") + np.uint8(unsigned % np.uint64(10))
        pos += 1
        unsigned //= np.uint64(10)
    left = digits_start
    right = pos - 1
    while left < right:
        byte = buf[left]
        buf[left] = buf[right]
        buf[right] = byte
        left += 1
        right -= 1
    return pos


@njit(cache=True, nogil=True)
def _digest_first8_mod(prefix: np.ndarray, prefix_len: int, frame: np.int64, count: int) -> int:
    """SHA-256(prefix + decimal(frame)), returning big-endian digest[:8] % count."""
    block = np.zeros(128, dtype=np.uint8)
    for i in range(prefix_len):
        block[i] = prefix[i]
    message_len = _append_signed_decimal(block, prefix_len, frame)
    block[message_len] = 0x80
    blocks = 1 if message_len < 56 else 2
    bit_len = np.uint64(message_len) * np.uint64(8)
    last = blocks * 64 - 8
    for i in range(8):
        block[last + i] = np.uint8(bit_len >> (56 - 8 * i))

    h0 = np.uint32(0x6A09E667)
    h1 = np.uint32(0xBB67AE85)
    h2 = np.uint32(0x3C6EF372)
    h3 = np.uint32(0xA54FF53A)
    h4 = np.uint32(0x510E527F)
    h5 = np.uint32(0x9B05688C)
    h6 = np.uint32(0x1F83D9AB)
    h7 = np.uint32(0x5BE0CD19)
    w = np.empty(64, dtype=np.uint32)
    for chunk in range(blocks):
        offset = chunk * 64
        for i in range(16):
            base = offset + 4 * i
            w[i] = (
                (np.uint32(block[base]) << 24)
                | (np.uint32(block[base + 1]) << 16)
                | (np.uint32(block[base + 2]) << 8)
                | np.uint32(block[base + 3])
            )
        for i in range(16, 64):
            s0 = _rotr(w[i - 15], 7) ^ _rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
            s1 = _rotr(w[i - 2], 17) ^ _rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
            w[i] = np.uint32(w[i - 16] + s0 + w[i - 7] + s1)
        a, b, c, d = h0, h1, h2, h3
        e, f, g, h = h4, h5, h6, h7
        for i in range(64):
            s1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
            choose = (e & f) ^ ((~e) & g)
            temp1 = np.uint32(h + s1 + choose + _SHA256_K[i] + w[i])
            s0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
            majority = (a & b) ^ (a & c) ^ (b & c)
            temp2 = np.uint32(s0 + majority)
            h, g, f, e = g, f, e, np.uint32(d + temp1)
            d, c, b, a = c, b, a, np.uint32(temp1 + temp2)
        h0 = np.uint32(h0 + a)
        h1 = np.uint32(h1 + b)
        h2 = np.uint32(h2 + c)
        h3 = np.uint32(h3 + d)
        h4 = np.uint32(h4 + e)
        h5 = np.uint32(h5 + f)
        h6 = np.uint32(h6 + g)
        h7 = np.uint32(h7 + h)

    # First eight digest bytes, interpreted just as int.from_bytes(..., "big").
    first8 = (np.uint64(h0) << 32) | np.uint64(h1)
    return int(first8 % np.uint64(count))


@njit(cache=True, nogil=True)
def _actions_kernel(
    prefixes: np.ndarray,
    prefix_lengths: np.ndarray,
    frames: np.ndarray,
    masks: np.ndarray,
    rows: np.ndarray,
    out: np.ndarray,
) -> None:
    for index in range(rows.shape[0]):
        env = rows[index, 0]
        slot = rows[index, 1]
        count = 0
        for action in range(3):
            if masks[index, action]:
                count += 1
        if count == 0:
            out[index] = 1
            continue
        # Watch decisions use the zero-based transition frame, while GridBatchSim
        # has already incremented its public prepared-frame counter.
        pick = _digest_first8_mod(
            prefixes[env, slot], prefix_lengths[env, slot], frames[env] - 1, count
        )
        seen = 0
        for action in range(3):
            if masks[index, action]:
                if seen == pick:
                    out[index] = action
                    break
                seen += 1


class BatchedRandomSafePolicy:
    """Pool-local scripted anchor that hashes valid rows in a single Numba call.

    ``world_seeds`` is a property so :class:`StaticPool`'s existing segment
    rebind assignment rebuilds the immutable prefixes.  Only signed-int64
    seeds, slots and prepared frames use the compiled path.  Python's canonical
    hashlib implementation remains the safe fallback for out-of-range values.
    """

    batched_rows = True

    def __init__(self, world_seeds: Sequence[int], slots_per_world: int = 6) -> None:
        self._slots_per_world = int(slots_per_world)
        if self._slots_per_world < 1:
            raise ValueError("slots_per_world must be positive")
        self.__world_seeds: tuple[int, ...] = ()
        self._compiled = False
        self._prefixes = np.empty((0, 0, _PREFIX_CAPACITY), dtype=np.uint8)
        self._prefix_lengths = np.empty((0, 0), dtype=np.int16)
        self.world_seeds = world_seeds

    @property
    def world_seeds(self) -> tuple[int, ...]:
        return self._world_seeds

    @world_seeds.setter
    def world_seeds(self, value: Sequence[int]) -> None:
        self._world_seeds = value

    @property
    def _world_seeds(self) -> tuple[int, ...]:
        return self.__world_seeds

    @_world_seeds.setter
    def _world_seeds(self, value: Sequence[int]) -> None:
        self.__world_seeds = tuple(int(seed) for seed in value)
        self._rebuild_prefixes()

    def rebind(self, world_seeds: Sequence[int], slots_per_world: int | None = None) -> None:
        """Rebuild cached prefixes for a segment whose world or slot shape changed."""
        if slots_per_world is not None:
            slots = int(slots_per_world)
            if slots < 1:
                raise ValueError("slots_per_world must be positive")
            self._slots_per_world = slots
        self.world_seeds = world_seeds

    def _rebuild_prefixes(self) -> None:
        if any(seed < _INT64_MIN or seed > _INT64_MAX for seed in self._world_seeds):
            self._compiled = False
            return
        prefixes = np.zeros(
            (len(self._world_seeds), self._slots_per_world, _PREFIX_CAPACITY), dtype=np.uint8
        )
        lengths = np.zeros((len(self._world_seeds), self._slots_per_world), dtype=np.int16)
        for env, seed in enumerate(self._world_seeds):
            for slot in range(self._slots_per_world):
                prefix = f"{SCRIPTED_ANCHOR_VERSION}|random_safe|{seed}|{slot}|".encode("ascii")
                if len(prefix) >= _PREFIX_CAPACITY - 20:
                    self._compiled = False
                    return
                prefixes[env, slot, : len(prefix)] = np.frombuffer(prefix, dtype=np.uint8)
                lengths[env, slot] = len(prefix)
        self._prefixes = prefixes
        self._prefix_lengths = lengths
        self._compiled = True

    @staticmethod
    def _legacy_action(seed: int, slot: int, prepared_frame: int, mask: np.ndarray) -> int:
        safe = [action for action in range(3) if mask[action]]
        if not safe:
            return 1
        frame = watch_anchor_frame(prepared_frame)
        key = f"{SCRIPTED_ANCHOR_VERSION}|random_safe|{seed}|{slot}|{frame}"
        pick = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % len(safe)
        return safe[pick]

    def actions(self, masks: np.ndarray, sim, slots: np.ndarray) -> np.ndarray:
        """Return canonical random-safe actions in the input-row order."""
        rows = np.ascontiguousarray(np.asarray(slots, dtype=np.int64).reshape(-1, 2))
        resolved = np.ascontiguousarray(np.asarray(masks, dtype=np.bool_))
        if resolved.shape != (len(rows), 6):
            raise ValueError(f"expected masks shaped ({len(rows)}, 6), got {resolved.shape}")
        out = np.ones(len(rows), dtype=np.int64)
        if not len(rows):
            return out
        frame_values = np.asarray(sim.frame)
        compiled = (
            self._compiled
            and np.issubdtype(frame_values.dtype, np.signedinteger)
            and rows[:, 0].min() >= 0
            and rows[:, 0].max() < len(self._world_seeds)
            and rows[:, 1].min() >= 0
            and rows[:, 1].max() < self._slots_per_world
        )
        if compiled:
            frames = np.ascontiguousarray(frame_values, dtype=np.int64)
            if np.any(frames[rows[:, 0]] < 1):
                raise ValueError("prepared_frame must be at least 1")
            _actions_kernel(self._prefixes, self._prefix_lengths, frames, resolved, rows, out)
            return out
        for index, (env, slot) in enumerate(rows):
            out[index] = self._legacy_action(
                self._world_seeds[int(env)], int(slot), int(frame_values[int(env)]), resolved[index]
            )
        return out
