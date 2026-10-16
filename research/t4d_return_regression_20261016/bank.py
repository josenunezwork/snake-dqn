"""Validated immutable branch-return bank used by the T4d learner.

The collector stores one row for every rolled-out legal action.  ``S`` is deliberately
raw (before ``kappa * (1 - gamma)``), so the manifest binds the discount and reward
semantics that make it meaningful.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.redesign_m3c_20261008.obs_pack import PackLayout, unpack_ref

REQUIRED = frozenset(
    {
        "state_packed",
        "state_scalars",
        "action",
        "T",
        "died",
        "S",
        "L0",
        "L_T",
        "terminal_packed",
        "terminal_scalars",
        "terminal_mask",
        "row_weight",
        "family",
    }
)


def realized_returns(T, died, S, L0, L_T, gamma: float, kappa: float) -> np.ndarray:
    """Return the realised branch target before a survivor's value bootstrap.

    This is exactly the mass-reward plus PBRS telescoping identity.  ``S`` remains
    raw so it can be inspected independently of the target construction.
    """
    T = np.asarray(T, dtype=np.int64)
    died = np.asarray(died, dtype=bool)
    S = np.asarray(S, dtype=np.float64)
    L0 = np.asarray(L0, dtype=np.float64)
    L_T = np.asarray(L_T, dtype=np.float64)
    if not (np.isfinite(gamma) and 0.0 < gamma <= 1.0 and np.isfinite(kappa)):
        raise ValueError("gamma must be in (0, 1] and kappa finite")
    if np.any(T < 1) or not (
        np.isfinite(S).all() and np.isfinite(L0).all() and np.isfinite(L_T).all()
    ):
        raise ValueError("invalid branch return fields")
    base = kappa * (1.0 - gamma) * S - L0 / 10.0
    survivor = np.power(gamma, T) * (L_T / 10.0)
    death = -3.0 * np.power(gamma, T - 1)
    return base + np.where(died, death, survivor)


class ReturnBank:
    """All validated T4d bank rows, sampled with the learner's NumPy RNG."""

    def __init__(self, location: Path, nscal: int) -> None:
        location = Path(location)
        paths = [location] if location.is_file() else sorted(location.glob("*.npz"))
        if not paths:
            raise ValueError(f"return bank has no .npz shards: {location}")
        if location.is_dir():
            try:
                sidecar = json.loads((location / "manifest.json").read_text())
                expected = sidecar["shards"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError("return bank needs a valid immutable manifest.json") from exc
            if not isinstance(expected, dict) or set(expected) != {p.name for p in paths}:
                raise ValueError("return-bank shard set does not exactly match manifest")
            for path in paths:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if expected[path.name] != digest:
                    raise ValueError(f"return-bank shard hash mismatch: {path.name}")
            sidecar = sidecar
        else:
            try:
                sidecar = json.loads((location.parent / "manifest.json").read_text())
                expected = sidecar["shards"]
                digest = hashlib.sha256(location.read_bytes()).hexdigest()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError("return bank file needs a valid sibling manifest.json") from exc
            if expected != {location.name: digest}:
                raise ValueError("return-bank file hash does not match manifest")
        rows: list[dict[str, np.ndarray]] = []
        manifest: dict[str, Any] | None = None
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                if not REQUIRED <= set(archive.files):
                    raise ValueError(f"invalid return-bank shard {path}: missing required fields")
                if "manifest_json" not in archive.files:
                    raise ValueError(f"invalid return-bank shard {path}: missing manifest_json")
                try:
                    raw_manifest = archive["manifest_json"]
                    parsed = json.loads(str(raw_manifest.item()))
                except (ValueError, TypeError, AttributeError) as exc:
                    raise ValueError(f"invalid return-bank manifest in {path}") from exc
                if manifest is None:
                    manifest = parsed
                elif parsed != manifest:
                    raise ValueError(f"return-bank manifests disagree: {path}")
                names = set(REQUIRED)
                if "b8_action" in archive.files:
                    names.add("b8_action")
                rows.append({name: archive[name].copy() for name in names})
        assert manifest is not None
        if any(sidecar.get(key) != manifest[key] for key in ("schema_version", "layout")):
            raise ValueError("return-bank shard manifest disagrees with manifest.json")
        self.manifest = manifest
        self._validate_manifest(nscal)
        self._validate_rows(rows, nscal)
        self.state_packed = np.concatenate([r["state_packed"] for r in rows])
        self.state_scalars = np.concatenate([r["state_scalars"] for r in rows])
        self.action = np.concatenate([r["action"] for r in rows]).astype(np.int64, copy=False)
        self.T = np.concatenate([r["T"] for r in rows]).astype(np.int64, copy=False)
        self.died = np.concatenate([r["died"] for r in rows]).astype(bool, copy=False)
        self.S = np.concatenate([r["S"] for r in rows]).astype(np.float64, copy=False)
        self.L0 = np.concatenate([r["L0"] for r in rows]).astype(np.int64, copy=False)
        self.L_T = np.concatenate([r["L_T"] for r in rows]).astype(np.int64, copy=False)
        self.terminal_packed = np.concatenate([r["terminal_packed"] for r in rows])
        self.terminal_scalars = np.concatenate([r["terminal_scalars"] for r in rows])
        self.terminal_mask = np.concatenate([r["terminal_mask"] for r in rows]).astype(
            bool, copy=False
        )
        self.weight = np.concatenate([r["row_weight"] for r in rows]).astype(np.float32, copy=False)
        self.family = np.concatenate([r["family"] for r in rows]).astype(np.uint8, copy=False)
        if any("b8_action" in row for row in rows) and not all("b8_action" in row for row in rows):
            raise ValueError("return-bank shards inconsistently provide b8_action")
        self.b8_action = (
            np.concatenate([r["b8_action"] for r in rows]).astype(np.int64, copy=False)
            if all("b8_action" in row for row in rows)
            else None
        )
        self.size = len(self.action)
        if not self.size:
            raise ValueError("return bank has no rows")
        self.layout = PackLayout.from_dict(manifest["layout"])
        self.local, self.global_ = unpack_ref(self.state_packed, self.layout)
        self.terminal_local, self.terminal_global = unpack_ref(self.terminal_packed, self.layout)
        from research.redesign_m3c_20261008.obs_pack import pack_ref

        if not np.array_equal(pack_ref(self.local, self.global_, self.layout), self.state_packed):
            raise ValueError("return-bank state observations are non-canonical")
        if not np.array_equal(
            pack_ref(self.terminal_local, self.terminal_global, self.layout), self.terminal_packed
        ):
            raise ValueError("return-bank terminal observations are non-canonical")
        # Only contrast mode requires branch grouping.  Keep legacy level-mode loader
        # acceptance and sampling byte-for-byte compatible with existing banks.
        self._groups: tuple[np.ndarray, ...] | None = None
        self._groups_for_row: list[np.ndarray] | None = None

    def _validate_manifest(self, nscal: int) -> None:
        m = self.manifest
        needed = {
            "schema_version",
            "gamma",
            "kappa",
            "phi",
            "layout",
            "packing_version",
            "reward_semantics",
        }
        if not needed <= set(m) or m["schema_version"] not in (1, 2):
            raise ValueError("unsupported return-bank manifest")
        if m["packing_version"] != "ego2s-bv8/obs_pack-v1" or m["phi"] != "length/10,dead=0":
            raise ValueError("return-bank packing or potential is incompatible")
        if m["reward_semantics"] != "mass+pbrs+death_no_overlays_or_kill":
            raise ValueError("return-bank reward semantics are incompatible")
        if not (
            np.isfinite(m["gamma"]) and 0.0 < float(m["gamma"]) <= 1.0 and np.isfinite(m["kappa"])
        ):
            raise ValueError("return-bank gamma/kappa is invalid")
        layout = PackLayout.from_dict(m["layout"])
        if layout is None:
            raise ValueError("return-bank layout is missing")
        if nscal != 60:
            raise ValueError("return bank requires the bv8 60-scalar observation")

    def _validate_rows(self, rows: list[dict[str, np.ndarray]], nscal: int) -> None:
        layout = PackLayout.from_dict(self.manifest["layout"])
        assert layout is not None
        for row in rows:
            n = len(row["action"])
            exact = {
                "state_packed": ((n, layout.nbytes), np.uint8),
                "state_scalars": ((n, nscal), np.float32),
                "T": ((n,), np.int32),
                "died": ((n,), np.bool_),
                "S": ((n,), np.float64),
                "L0": ((n,), np.int32),
                "L_T": ((n,), np.int32),
                "terminal_packed": ((n, layout.nbytes), np.uint8),
                "terminal_scalars": ((n, nscal), np.float32),
                "terminal_mask": ((n, 6), np.bool_),
                "row_weight": ((n,), np.float32),
                "family": ((n,), np.uint8),
            }
            for name, (shape, dtype) in exact.items():
                if row[name].shape != shape or row[name].dtype != dtype:
                    raise ValueError(f"invalid return-bank {name} shape or dtype")
            if row["action"].shape != (n,) or not np.issubdtype(row["action"].dtype, np.integer):
                raise ValueError("invalid return-bank action")
            if "b8_action" in row and (
                row["b8_action"].shape != (n,)
                or not np.issubdtype(row["b8_action"].dtype, np.integer)
                or (row["b8_action"] < 0).any()
                or (row["b8_action"] > 5).any()
            ):
                raise ValueError("invalid return-bank b8_action")
            if (row["action"] < 0).any() or (row["action"] > 5).any() or (row["T"] < 1).any():
                raise ValueError("invalid return-bank action or horizon")
            if not all(
                np.isfinite(row[x]).all()
                for x in ("state_scalars", "terminal_scalars", "S", "row_weight")
            ):
                raise ValueError("non-finite return-bank numeric field")
            families = (0, 1)
            if self.manifest.get("provenance", {}).get("status") == "FORMAT_SMOKE_INITIAL_STATES":
                families = (0, 1, 2)
            if (row["S"] < 0).any() or not np.isin(row["family"], families).all():
                raise ValueError("invalid return-bank raw mass sum or family")
            if (row["row_weight"] <= 0).any() or (row["L0"] < 0).any() or (row["L_T"] < 0).any():
                raise ValueError("invalid return-bank weight or length")
            if row["terminal_mask"][row["died"]].any():
                raise ValueError("dead return-bank rows must have empty terminal masks")
            dead = row["died"]
            if dead.any() and (row["L_T"][dead] != 0).any():
                raise ValueError("dead return-bank rows must have terminal length zero")
            if dead.any() and (
                row["terminal_packed"][dead].any() or row["terminal_scalars"][dead].any()
            ):
                raise ValueError("dead return-bank rows must have zero terminal observations")

    @property
    def gamma(self) -> float:
        return float(self.manifest["gamma"])

    @property
    def kappa(self) -> float:
        return float(self.manifest["kappa"])

    def _build_groups(self) -> tuple[np.ndarray, ...]:
        """Group all recorded legal branches by their canonical source observation.

        The contrast target's reference is selected from the *current* bootstrapped returns,
        so sampling must retain every recorded branch for a source state.  Store only index
        arrays after construction; keeping packed/scalar bytes as dictionary keys would double
        an already sizeable immutable bank in memory.
        """
        groups: dict[bytes, list[int]] = {}
        for index, (packed, scalars) in enumerate(zip(self.state_packed, self.state_scalars)):
            digest = hashlib.sha256(packed.tobytes() + scalars.tobytes()).digest()
            groups.setdefault(digest, []).append(index)
        result: list[np.ndarray] = []
        for indices in groups.values():
            first = indices[0]
            for index in indices[1:]:
                if not (
                    np.array_equal(self.state_packed[first], self.state_packed[index])
                    and np.array_equal(self.state_scalars[first], self.state_scalars[index])
                ):
                    raise ValueError("return-bank source-state hash collision")
            actions = self.action[np.asarray(indices, dtype=np.int64)]
            if len(np.unique(actions)) != len(actions):
                raise ValueError("return-bank has duplicate action branch for a source state")
            result.append(np.asarray(indices, dtype=np.int64))
        if not result:
            raise ValueError("return-bank has no source-state groups")
        return tuple(result)

    def _contrast_groups(self) -> list[np.ndarray]:
        if self._groups_for_row is None:
            self._groups = self._build_groups()
            self._groups_for_row = [np.empty(0, dtype=np.int64) for _ in range(self.size)]
            for group in self._groups:
                for index in group:
                    self._groups_for_row[int(index)] = group
        return self._groups_for_row

    def sample(
        self, rng: np.random.Generator, n: int, *, kappa: float, grouped: bool = False
    ) -> dict[str, np.ndarray]:
        if n < 0:
            raise ValueError("sample size must be >= 0")
        if grouped:
            # Preserve legacy branch-uniform RNG sampling: support branches only provide
            # each selected row's target-derived a_ref, never extra loss rows.
            selected_indices = rng.integers(self.size, size=n)
            groups_for_row = self._contrast_groups()
            groups = [groups_for_row[int(index)] for index in selected_indices]
            indices = np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)
        else:
            indices = rng.integers(self.size, size=n)
            groups = []
        # MPS does not accept float64 uploads.  S stays float64 in the immutable
        # bank, while its closed-form realised target is deliberately computed on
        # CPU before the learner receives a float32 batch.
        realised = realized_returns(
            self.T[indices],
            self.died[indices],
            self.S[indices],
            self.L0[indices],
            self.L_T[indices],
            self.gamma,
            kappa,
        ).astype(np.float32)
        out = {
            "local": self.local[indices],
            "global": self.global_[indices],
            "scalars": self.state_scalars[indices],
            "action": self.action[indices],
            "T": self.T[indices],
            "died": self.died[indices],
            "realised": realised,
            "terminal_local": self.terminal_local[indices],
            "terminal_global": self.terminal_global[indices],
            "terminal_scalars": self.terminal_scalars[indices],
            "terminal_mask": self.terminal_mask[indices],
            "weight": self.weight[indices],
        }
        if grouped:
            # Kept on CPU by the learner: branch members are contiguous and the small
            # vector avoids unsupported/device-sync-heavy segmented reductions on MPS.
            out["group_sizes"] = np.asarray([len(group) for group in groups], dtype=np.int64)
            selected = np.zeros(len(indices), dtype=bool)
            selected[np.cumsum(np.r_[0, out["group_sizes"][:-1]])] = True
            # The selected input action is intentionally the first member of each copied
            # group; retain the branch-uniform source row rather than changing sampling.
            positions = [
                int(np.flatnonzero(group == original)[0])
                for group, original in zip(groups, selected_indices)
            ]
            selected[:] = False
            selected[np.cumsum(np.r_[0, out["group_sizes"][:-1]]) + np.asarray(positions)] = True
            out["selected"] = selected
        return out
