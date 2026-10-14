"""Shared read-only demonstration store for ``train_m3c --demo-mmap PATH`` (perf-ram).

Every run with demonstrations (``--obs-version b``) keeps a private 300k-row buffer of
decompressed M2b shards (3.71 GB raw, 0.97 GB with ``--obs-pack``) and decompresses one
shard per target sync. This module decompresses every *training* row (hold-out worlds
excluded, exactly :meth:`DemoStream.load`) of the requested rounds ONCE into a single file;
each run maps it read-only and keeps only a slot -> row index (8 bytes per slot), so the
rows live once in the host's page cache however many runs read them.

Size: rounds 10-12 hold 3,725,000 training rows: 12.0 GB packed, 46 GB raw. The packed
store therefore pays off against ``--obs-pack``'s private 0.97 GB buffers from about 12
concurrent runs per host (its pages are clean file cache: the kernel can drop the ones no
buffer slot points at, so the resident part is smaller); a raw store only makes sense with
``--obs-pack`` off AND more than 12 runs. RAM model: ``research/perf_ram_20261013``.

File layout: ``MAGIC`` (8 bytes), header length (uint64 LE), a JSON header (fields with
dtype / per-row shape / offset, row count, the shard table ``key -> [start, rows, bytes]``,
the packed layout or null), then one C-contiguous array per field at a 4096-aligned offset.
Planes are stored packed (:mod:`obs_pack`) when built with a layout, raw otherwise; the
other fields keep :class:`DemoStream`'s buffer dtypes, so a gather returns the same bytes.

:meth:`SharedDemoStore.open_or_build` serializes builders with an exclusive ``flock`` on
``PATH.lock``; a build writes ``PATH.tmp<pid>`` and renames it, so a reader never sees a
partial file. Opening checks that every shard the run will use is in the store with the
file size it was built from.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from research.redesign_m3c_20261008.train_m3c import (
    DEMO_KEYS,
    DemoStream,
    demo_fields,
    demo_key,
)

MAGIC = b"M3CDEMO1"
ALIGN = 4096
VERSION = 1


def _fields(pack) -> List[Tuple[str, Tuple[int, ...], str]]:
    """:func:`demo_fields` (the DemoStream buffer's rows) with dtype names for the header."""
    return [(n, tuple(s), np.dtype(d).name) for n, s, d in demo_fields(pack)]


def _shard_paths(data: Path, rounds: Iterable[int]) -> List[Path]:
    paths: List[Path] = []
    for r in rounds:
        paths += sorted((Path(data) / f"round-{r}").glob("chunk-*-part-*.npz"))
    return paths


class SharedDemoStore:
    """A read-only mapping of a built store (see the module docstring)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        fd = os.open(self.path, os.O_RDONLY)
        try:
            self._mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)
        if self._mm[:8] != MAGIC:
            raise ValueError(f"{self.path}: not a demonstration store")
        (hlen,) = struct.unpack("<Q", self._mm[8:16])
        self.header = json.loads(self._mm[16 : 16 + hlen].decode())
        if self.header.get("version") != VERSION:
            raise ValueError(f"{self.path}: store version {self.header.get('version')}")
        from research.redesign_m3c_20261008.obs_pack import PackLayout

        self.pack = PackLayout.from_dict(self.header["pack"])
        self.rows = int(self.header["rows"])
        self.shards: Dict[str, Tuple[int, int, int]] = {
            k: tuple(v) for k, v in self.header["shards"].items()
        }
        self.arrays: Dict[str, np.ndarray] = {}
        for f in self.header["fields"]:
            a = np.ndarray(
                (self.rows,) + tuple(f["shape"]),
                dtype=np.dtype(f["dtype"]),
                buffer=self._mm,
                offset=int(f["offset"]),
            )
            self.arrays[f["name"]] = a  # read-only (the mapping is ACCESS_READ)

    @property
    def nbytes(self) -> int:
        return len(self._mm)

    def shard_range(self, path: Path) -> Tuple[int, int]:
        start, n, _ = self.shards[demo_key(path)]
        return int(start), int(n)

    def check(self, paths: Sequence[Path]) -> None:
        """Every shard in ``paths`` is in the store, built from a file of the same size."""
        bad = []
        for p in paths:
            entry = self.shards.get(demo_key(p))
            if entry is None or int(entry[2]) != Path(p).stat().st_size:
                bad.append(demo_key(p))
        if bad:
            raise SystemExit(
                f"--demo-mmap {self.path}: {len(bad)} shard(s) missing or changed since the "
                f"build (e.g. {bad[0]}); remove the store to rebuild it"
            )

    def close(self) -> None:
        self.arrays = {}
        try:
            self._mm.close()
        except BufferError:  # a caller still holds a view; the mapping dies with it
            pass

    # ------------------------------------------------------------------ build
    @staticmethod
    def build(path: Path, data: Path, rounds: Sequence[int], pack=None, log=print) -> None:
        """Decompress (and pack) every training row of ``rounds`` into ``path`` (atomic)."""
        path = Path(path)
        paths = _shard_paths(data, rounds)
        if not paths:
            raise SystemExit("no demonstration shards")
        t0 = time.time()
        counts = []
        for p in paths:  # pass 1: rows per shard (reads only world_seed)
            with np.load(p) as z:
                counts.append(int((z["world_seed"] % 10 != 0).sum()))
        rows = int(sum(counts))
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        fields = _fields(pack)
        header = {
            "version": VERSION,
            "rows": rows,
            "pack": None if pack is None else pack.to_dict(),
            "rounds": [int(r) for r in rounds],
            "held_out_rule": "world_seed % 10 == 0 excluded (DemoStream.load)",
            "shards": {
                demo_key(p): [int(s), int(n), int(p.stat().st_size)]
                for p, s, n in zip(paths, starts, counts)
            },
            "fields": [],
        }
        # offsets need the header length: size it with placeholder offsets, then fix
        body: List[dict] = [
            {"name": n, "shape": list(sh), "dtype": dt, "offset": 0} for n, sh, dt in fields
        ]
        header["fields"] = body
        hlen_guess = len(json.dumps(header).encode()) + 64 * len(body) + 64
        off = -(-(16 + hlen_guess) // ALIGN) * ALIGN
        for f in body:
            f["offset"] = off
            nb = rows * int(np.prod(f["shape"], dtype=np.int64)) * np.dtype(f["dtype"]).itemsize
            off = -(-(off + nb) // ALIGN) * ALIGN
        total = off
        raw = json.dumps(header).encode()
        if 16 + len(raw) > body[0]["offset"]:
            raise AssertionError("store header overflow")
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
        try:
            with open(tmp, "wb") as fh:
                fh.truncate(total)
                fh.write(MAGIC + struct.pack("<Q", len(raw)) + raw)
            out = {
                f["name"]: np.memmap(
                    tmp,
                    mode="r+",
                    dtype=np.dtype(f["dtype"]),
                    offset=int(f["offset"]),
                    shape=(rows,) + tuple(f["shape"]),
                )
                for f in body
            }
            for i, (p, s, n) in enumerate(zip(paths, starts, counts)):
                part = DemoStream.load(p)
                if len(part["a_v8"]) != n:
                    raise AssertionError(f"{p}: row count changed during the build")
                sl = slice(int(s), int(s) + n)
                if pack is not None:
                    from research.redesign_m3c_20261008.obs_pack import pack as pack_rows

                    out["packed"][sl] = pack_rows(part["local"], part["global"], pack)
                else:
                    out["local"][sl] = part["local"]
                    out["global"][sl] = part["global"]
                for k in DEMO_KEYS[2:]:
                    out[k][sl] = part[k]
                if log and (i + 1) % 50 == 0:
                    log(f"demo store: {i + 1}/{len(paths)} shards ({time.time() - t0:.0f} s)")
            for a in out.values():
                a.flush()
            del out
            with open(tmp, "rb+") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        if log:
            log(
                f"demo store: built {path} ({rows} rows, {total / 1e9:.2f} GB, "
                f"{'packed' if pack is not None else 'raw'}) in {time.time() - t0:.0f} s"
            )

    @classmethod
    def open_or_build(
        cls, path: Path, data: Path, rounds: Sequence[int], pack=None, log=print
    ) -> "SharedDemoStore":
        """Open ``path``, building it first (under an exclusive host lock) if it is missing.

        An existing store is used as built (packed or raw: the gathered rows are the same)."""
        import fcntl

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_name(path.name + ".lock"), "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if not path.exists():
                    cls.build(path, data, rounds, pack, log=log)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        store = cls(path)
        store.check(_shard_paths(data, rounds))
        return store


def store_nbytes(rows: int, pack: Optional[object]) -> int:
    """File bytes of a store with ``rows`` rows (planes packed with ``pack`` or raw)."""
    total = 0
    for _, shape, dt in _fields(pack):
        total += int(rows) * int(np.prod(shape, dtype=np.int64)) * np.dtype(dt).itemsize
    return total
