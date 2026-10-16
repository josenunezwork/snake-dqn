"""Validated immutable rollout-label bank and its surviving-set margin loss."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class LabelBank:
    """In-memory `.npz` shards sampled with the learner's existing NumPy RNG."""

    def __init__(self, directory: Path, nscal: int, pack=None) -> None:
        shards = sorted(Path(directory).glob("*.npz"))
        if not shards:
            raise ValueError(f"label bank has no .npz shards: {directory}")
        rows = []
        for path in shards:
            with np.load(path) as archive:
                keys = set(archive.files)
                required = {"scalars", "mask", "surviving", "action", "weight"}
                if not required <= keys or not ({"packed"} <= keys or {"local", "global"} <= keys):
                    raise ValueError(f"invalid label-bank shard {path}")
                row = {key: archive[key].copy() for key in archive.files}
            n = len(row["scalars"])
            if (
                row["scalars"].shape != (n, nscal)
                or row["scalars"].dtype != np.float32
                or not np.isfinite(row["scalars"]).all()
                or row["mask"].shape != (n, 6)
                or row["mask"].dtype != np.bool_
            ):
                raise ValueError(f"invalid scalar/mask shape in {path}")
            if (
                row["surviving"].shape != (n, 6)
                or row["surviving"].dtype != np.bool_
                or not row["surviving"].any(axis=1).all()
            ):
                raise ValueError(f"invalid surviving set in {path}")
            if (
                row["action"].shape != (n,)
                or not np.issubdtype(row["action"].dtype, np.integer)
                or row["weight"].shape != (n,)
                or row["weight"].dtype != np.float32
            ):
                raise ValueError(f"invalid action/weight shape in {path}")
            if not np.isfinite(row["weight"]).all() or (row["weight"] < 0).any():
                raise ValueError(f"invalid weight in {path}")
            if (row["action"] < 0).any() or (row["action"] > 5).any():
                raise ValueError(f"invalid action in {path}")
            # v2 keeps the raw outcome set even when its selected label set prefers
            # non-boost actions.  T4a must classify counterexamples from this raw
            # provenance rather than from the preferred label set.
            if "surviving_original" in row and (
                row["surviving_original"].shape != (n, 6)
                or row["surviving_original"].dtype != np.bool_
                or not row["surviving_original"].any(axis=1).all()
            ):
                raise ValueError(f"invalid original surviving set in {path}")
            if "packed" in row:
                if (
                    pack is None
                    or row["packed"].dtype != np.uint8
                    or row["packed"].shape != (n, int(pack.nbytes))
                ):
                    raise ValueError(f"packed shard/layout mismatch in {path}")
                from research.redesign_m3c_20261008.obs_pack import pack_ref, unpack_ref

                row["local"], row["global"] = unpack_ref(row["packed"], pack)
                if not np.array_equal(pack_ref(row["local"], row["global"], pack), row["packed"]):
                    raise ValueError(f"non-canonical packed observations in {path}")
            elif (
                row["local"].dtype != np.uint8
                or row["global"].dtype != np.uint8
                or row["local"].shape != (n, 7, 31, 31)
                or row["global"].shape != (n, 4, 37, 37)
            ):
                raise ValueError(f"invalid local/global observations in {path}")
            rows.append(row)
        self.local = np.concatenate([r["local"] for r in rows])
        self.global_ = np.concatenate([r["global"] for r in rows])
        self.scalars = np.concatenate([r["scalars"] for r in rows]).astype(np.float32, copy=False)
        self.mask = np.concatenate([r["mask"] for r in rows]).astype(bool, copy=False)
        self.surviving = np.concatenate([r["surviving"] for r in rows]).astype(bool, copy=False)
        self.surviving_original = np.concatenate(
            [r.get("surviving_original", r["surviving"]) for r in rows]
        ).astype(bool, copy=False)
        self.action = np.concatenate([r["action"] for r in rows]).astype(np.int64, copy=False)
        self.weight = np.concatenate([r["weight"] for r in rows]).astype(np.float32, copy=False)
        self.size = len(self.action)
        if self.size == 0:
            raise ValueError("label bank has no rows")
        # A counter row is one where B8's chosen action itself survived.  The
        # preferred v2 label set can deliberately omit that action, so derive
        # this from the preserved raw outcome provenance.
        self.counter = self.surviving_original[np.arange(self.size), self.action]
        self._counter_indices = np.flatnonzero(self.counter)
        self._decisive_indices = np.flatnonzero(~self.counter)

    def sample(self, rng, n: int, counter_ratio: float | None = None) -> dict:
        """Sample rows, optionally enforcing a counter:decisive ratio.

        ``None`` intentionally retains the original one-call uniform sampler so the
        default learner path consumes the identical RNG stream.
        """
        if counter_ratio is None:
            indices = rng.integers(self.size, size=n)
        else:
            self.validate_counter_ratio(counter_ratio)
            if n < 0:
                raise ValueError("sample size must be >= 0")
            if n == 0:
                indices = np.empty(0, dtype=np.int64)
                return self._rows(indices)
            n_counter = int(round(n * (1.0 - 1.0 / (counter_ratio + 1.0))))
            n_decisive = n - n_counter
            if n_counter and not len(self._counter_indices):
                raise ValueError("label bank has no counter rows for requested counter ratio")
            if n_decisive and not len(self._decisive_indices):
                raise ValueError("label bank has no decisive rows for requested counter ratio")
            parts = []
            if n_counter:
                parts.append(
                    self._counter_indices[rng.integers(len(self._counter_indices), size=n_counter)]
                )
            if n_decisive:
                parts.append(
                    self._decisive_indices[
                        rng.integers(len(self._decisive_indices), size=n_decisive)
                    ]
                )
            indices = np.concatenate(parts) if len(parts) > 1 else parts[0]
        return self._rows(indices)

    def validate_counter_ratio(self, counter_ratio: float) -> None:
        """Check that a requested two-sided sampler can draw each required side."""
        if not np.isfinite(counter_ratio) or counter_ratio < 0:
            raise ValueError("counter ratio must be finite and >= 0")
        if counter_ratio > 0 and not len(self._counter_indices):
            raise ValueError("label bank has no counter rows for requested counter ratio")
        if not len(self._decisive_indices):
            raise ValueError("label bank has no decisive rows for requested counter ratio")

    def _rows(self, indices: np.ndarray) -> dict:
        return {
            "local": self.local[indices],
            "global": self.global_[indices],
            "scalars": self.scalars[indices],
            "mask": self.mask[indices],
            "surviving": self.surviving[indices],
            "action": self.action[indices],
            "surviving_original": self.surviving_original[indices],
            "counter": self.counter[indices],
            "weight": self.weight[indices],
        }


def surviving_margin(torch, q, surviving, weight, margin: float):
    """Weighted all-action DQfD margin toward a non-empty surviving action set."""
    survive = surviving.bool()
    per_row = (q + (~survive).to(q.dtype) * margin).max(1).values
    per_row = per_row - torch.where(survive, q, torch.full_like(q, -float("inf"))).max(1).values
    return (per_row * weight).mean()
