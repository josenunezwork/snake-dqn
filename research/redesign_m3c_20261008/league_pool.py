"""Immutable, static frozen-opponent pool for the opt-in M3-C league slice.

This deliberately contains no PFSP, publication, or cache policy.  It binds the
six permanent initial identities, validates their bytes before loading them, and
partitions frozen seat rows by exact policy identity.  The collector owns seat
permutations and replay ownership; this module never labels or emits a replay row.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

MAX_ELIGIBLE_POLICIES = 16
V8_METHOD = "free-space-veto/v8-space-and-head(lambda=8.0)"


class _PolicyIds(tuple):
    """Tuple that remains callable during the collector API transition."""

    def __call__(self) -> Tuple[str, ...]:
        return tuple(self)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of ``path`` without trusting checkpoint metadata."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PoolIdentity:
    """Pinned static policy identity; all non-scripted entries are byte-bound."""

    policy_id: str
    kind: str  # ``bv8``, ``vector61``, or ``scripted``
    checkpoint_path: Optional[str]
    checkpoint_sha256: Optional[str]
    obs_spec: str
    mask_config: str
    wrapper: str = "none"
    source_sha256s: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"bv8", "vector61", "scripted"}:
            raise ValueError(f"unknown pool policy kind {self.kind!r}")
        if not self.policy_id:
            raise ValueError("pool policy_id must not be empty")
        if self.kind == "scripted":
            if self.checkpoint_path is not None or self.checkpoint_sha256 is not None:
                raise ValueError("scripted policy must not carry checkpoint bytes")
        elif not self.checkpoint_path or not self.checkpoint_sha256:
            raise ValueError(f"{self.policy_id}: checkpoint path and sha256 are required")
        if self.wrapper != "none" and not self.source_sha256s:
            raise ValueError(f"{self.policy_id}: wrapped policy needs source sha256 pins")

    def record(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "kind": self.kind,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "obs_spec": self.obs_spec,
            "mask_config": self.mask_config,
            "wrapper": self.wrapper,
            "source_sha256s": dict(self.source_sha256s),
        }

    def validate(self, repo_root: Optional[Path] = None) -> None:
        """Fail closed if pinned policy or wrapper source bytes are absent/mismatched."""
        if self.kind != "scripted":
            path = Path(str(self.checkpoint_path))
            if not path.is_file():
                raise FileNotFoundError(f"{self.policy_id}: missing checkpoint {path}")
            actual = sha256_file(path)
            if actual != self.checkpoint_sha256:
                raise RuntimeError(
                    f"{self.policy_id}: checkpoint SHA mismatch: {actual} != "
                    f"{self.checkpoint_sha256}"
                )
        if self.source_sha256s:
            root = Path.cwd() if repo_root is None else Path(repo_root)
            for rel, expected in self.source_sha256s:
                path = root / rel
                if not path.is_file():
                    raise FileNotFoundError(f"{self.policy_id}: missing wrapper source {path}")
                actual = sha256_file(path)
                if actual != expected:
                    raise RuntimeError(
                        f"{self.policy_id}: wrapper SHA mismatch for {rel}: {actual} != {expected}"
                    )


def _v8_source_pins() -> Tuple[Tuple[str, str], ...]:
    """Use the served v8 receipt pins, not an unbound local wrapper import."""
    from web.backend.safety_veto_serving import V8_STRICT_RECEIPT_SOURCE_SHA256S

    return tuple(sorted(V8_STRICT_RECEIPT_SOURCE_SHA256S.items()))


def _source_pins(*paths: str) -> Tuple[Tuple[str, str], ...]:
    """Bind policy implementation bytes into the receipt at construction time."""
    return tuple((path, sha256_file(Path(path))) for path in paths)


def default_pool_identities(
    checkpoint_dir: Optional[str] = None, b8_checkpoint: Optional[str] = None
) -> Tuple[PoolIdentity, ...]:
    """The six permanent first-slice identities, with no dynamic snapshots.

    ``SNAKE_CKPT_DIR`` permits a transferred T16 bundle to retain its own absolute
    location.  Validation intentionally happens at actor startup, after the
    environment has been staged, rather than when this module is imported.
    """
    directory = Path(checkpoint_dir or os.environ.get("SNAKE_CKPT_DIR", "saved_snakes"))
    source_pins = _v8_source_pins()
    bv8_sources = _source_pins(
        "src/simd_env/ego2s_policy.py", "src/model/ego2s_network.py", "src/simd_env/ego_raster.py"
    )
    v61_sources = _source_pins(
        "src/simd_env/vector61_policy.py", "src/simd_env/vector61_featurizer.py"
    )
    script_sources = _source_pins("src/simd_env/fast_anchors.py")
    return (
        PoolIdentity(
            "b8-s3",
            "bv8",
            str(b8_checkpoint or directory / "b8_s3_t7_20261016.pth"),
            "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
            "ego2s-bv8",
            "resolved",
            "none",
            bv8_sources,
        ),
        PoolIdentity(
            "frp3-s12",
            "vector61",
            str(directory / "frp3_m3_s12_u60000_20261005.pth"),
            "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723",
            "vector61",
            "advisory",
            "none",
            v61_sources,
        ),
        PoolIdentity(
            "frp3-s12-v8",
            "vector61",
            str(directory / "frp3_m3_s12_u60000_20261005.pth"),
            "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723",
            "vector61",
            "advisory",
            V8_METHOD,
            tuple(sorted(set(v61_sources + source_pins))),
        ),
        PoolIdentity(
            "champion-v8",
            "vector61",
            str(directory / "champion_a5_freespace_20260621.pth"),
            "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93",
            "vector61",
            "advisory",
            V8_METHOD,
            tuple(sorted(set(v61_sources + source_pins))),
        ),
        PoolIdentity(
            "scripted:greedy_food",
            "scripted",
            None,
            None,
            "scripted",
            "resolved",
            "none",
            script_sources,
        ),
        PoolIdentity(
            "scripted:random_safe",
            "scripted",
            None,
            None,
            "scripted",
            "resolved",
            "none",
            script_sources,
        ),
    )


class StaticPool:
    """Validated frozen policies and deterministic policy-row dispatch."""

    def __init__(self, identities: Sequence[PoolIdentity], policies: Mapping[str, Any]) -> None:
        if not identities or len(identities) > MAX_ELIGIBLE_POLICIES:
            raise ValueError(f"static pool needs 1..{MAX_ELIGIBLE_POLICIES} identities")
        self.identities = tuple(identities)
        ids = tuple(identity.policy_id for identity in self.identities)
        if len(set(ids)) != len(ids):
            raise ValueError("static pool policy IDs must be unique")
        if set(ids) != set(policies):
            raise ValueError("static pool policies must exactly match identities")
        self.policies = dict(policies)

    @property
    def policy_ids(self) -> _PolicyIds:
        """Permanent ID order (also callable for the first collector draft)."""
        return _PolicyIds(identity.policy_id for identity in self.identities)

    def identity_records(self) -> list[Dict[str, Any]]:
        return [identity.record() for identity in self.identities]

    def groups_for(self, seats: np.ndarray, alive: np.ndarray) -> "OrderedDict[str, np.ndarray]":
        """Partition live frozen rows by ID in permanent identity order.

        ``seats[e, s]`` is ``None`` for a learner seat; it must be a declared pool
        ID otherwise.  This makes empty groups harmless and catches a remap error
        before an action can silently fall back to the wrong policy.
        """
        seats = np.asarray(seats, dtype=object)
        alive = np.asarray(alive, dtype=bool)
        if seats.shape != alive.shape:
            raise ValueError(f"seat/alive shape mismatch: {seats.shape} vs {alive.shape}")
        known = set(self.policy_ids)
        unknown = {str(x) for x in seats.ravel() if x is not None and x not in known}
        if unknown:
            raise ValueError(f"unknown frozen pool IDs in seats: {sorted(unknown)}")
        out: "OrderedDict[str, np.ndarray]" = OrderedDict()
        for policy_id in self.policy_ids:
            rows = np.argwhere((seats == policy_id) & alive).astype(np.int64, copy=False)
            if len(rows):
                out[policy_id] = rows
        return out

    def dispatch(
        self,
        prepared: Any,
        masks: np.ndarray,
        alive: np.ndarray,
        actions: np.ndarray,
        seats: np.ndarray,
    ) -> Dict[str, Any]:
        """Run each exact frozen policy once and scatter to its owned live rows only."""
        groups = self.groups_for(seats, alive)
        counts = {policy_id: int(len(rows)) for policy_id, rows in groups.items()}
        start = time.perf_counter()
        kinds = {identity.policy_id: identity.kind for identity in self.identities}
        gpu_seconds = 0.0
        for policy_id, rows in groups.items():
            # CUDA events are placed around only a BV8 policy group.  On-device event
            # elapsed time excludes the CPU/vector/script work between groups.
            gpu_events = _gpu_events() if kinds[policy_id] == "bv8" else None
            choice = np.asarray(
                self.policies[policy_id].actions(masks[rows[:, 0], rows[:, 1]], prepared, rows),
                dtype=np.int64,
            ).reshape(-1)
            gpu_seconds += _gpu_elapsed(gpu_events)
            if len(choice) != len(rows):
                raise RuntimeError(
                    f"{policy_id}: returned {len(choice)} actions for {len(rows)} rows"
                )
            actions[rows[:, 0], rows[:, 1]] = choice
        return {
            "frozen_forward_s": time.perf_counter() - start,
            "frozen_forward_gpu_s": gpu_seconds,
            "frozen_live_rows": counts,
        }


def _gpu_events():
    import torch

    if not torch.cuda.is_available():
        return None
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end


def _gpu_elapsed(events: Any) -> float:
    if events is None:
        return 0.0
    start, end = events
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / 1000.0


def build_static_pool(
    identities: Sequence[PoolIdentity], *, world_seeds: Sequence[int], runtime: Any, device: str
) -> StaticPool:
    """Validate and load the permanent pool.  Models are eval-only and immutable."""
    for identity in identities:
        identity.validate()
    from src.simd_env.ego2s_policy import Ego2sSimdPolicy
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    policies: Dict[str, Any] = {}
    for identity in identities:
        if identity.kind == "scripted":
            policies[identity.policy_id] = FastProfileAnchorSimdPolicy(
                identity.policy_id.split(":", 1)[1], world_seeds
            )
        elif identity.kind == "bv8":
            policies[identity.policy_id] = Ego2sSimdPolicy(
                str(identity.checkpoint_path), device=device, mask_mode=identity.mask_config
            )
            from src.model.ego2s_network import spec_for

            if spec_for(policies[identity.policy_id].net) != identity.obs_spec:
                raise RuntimeError(f"{identity.policy_id}: checkpoint obs spec does not match pin")
        else:
            policies[identity.policy_id] = Vector61SimdPolicy(
                str(identity.checkpoint_path),
                runtime,
                veto_slots=range(6) if identity.wrapper == V8_METHOD else (),
                veto_variant="v8" if identity.wrapper == V8_METHOD else "v2",
                veto_lambda=8.0 if identity.wrapper == V8_METHOD else None,
                forward="batched",
            )
            if policies[identity.policy_id].input_size != 61:
                raise RuntimeError(f"{identity.policy_id}: checkpoint is not vector61")
        _freeze_policy(policies[identity.policy_id])
    return StaticPool(identities, policies)


def build_default_static_pool(
    args: Any, ctx: Any, sim: Any, runtime: Any, device: Any
) -> StaticPool:
    """Collector-facing constructor for the permanent first-slice pool.

    ``args`` and ``ctx`` are accepted so the actor call site has one stable
    extension hook; only ``args.league_checkpoint_dir`` is consumed when present.
    The simulator's actual world seeds are passed to deterministic scripted
    anchors, preserving their established per-world action semantics.
    """
    del ctx
    directory = getattr(args, "league_checkpoint_dir", None)
    b8_checkpoint = getattr(args, "league_b8_checkpoint", None) or getattr(args, "init", None)
    seeds = sim.get_world_seeds()
    return build_static_pool(
        default_pool_identities(directory, b8_checkpoint),
        world_seeds=seeds,
        runtime=runtime,
        device=str(device),
    )


def _freeze_policy(policy: Any) -> None:
    """Enforce eval/no-grad tensor ownership without assuming a policy class."""
    for value in vars(policy).values():
        if hasattr(value, "eval") and hasattr(value, "parameters"):
            value.eval()
            for param in value.parameters():
                param.requires_grad_(False)
