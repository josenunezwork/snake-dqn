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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

MAX_ELIGIBLE_POLICIES = 16
V8_METHOD = "free-space-veto/v8-space-and-head(lambda=8.0)"


class LeagueVector61Policy:
    """Pool-local vector61 policy with the compiled, exact v8 wrapper.

    ``Vector61SimdPolicy`` already batches a checkpoint's rows and, when its
    runtime uses ``backend="numba"``, constructs vector61 states without a
    Python row loop.  Its v8 compatibility wrapper intentionally invokes the
    live hook one row at a time so it can retain serving diagnostics.  The
    training pool never consumes those diagnostics, while its many worlds make
    that Python loop the dominant actor cost.  This adapter keeps the policy's
    public continuation fields and delegates every non-v8 behavior unchanged;
    v8 uses the established decision-identical ``v8_decide`` kernel.

    It is deliberately local to the league.  Serving and strict-evaluation
    users retain their live-hook bookkeeping semantics.
    """

    def __init__(self, policy: Any) -> None:
        object.__setattr__(self, "_policy", policy)

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep cached-pool rebind and full-state restores on the real policy."""
        if name == "_policy":
            object.__setattr__(self, name, value)
        else:
            setattr(self._policy, name, value)

    def __getattr__(self, name: str) -> Any:
        """Expose the wrapped policy's immutable state and snapshot fields."""
        return getattr(self._policy, name)

    def actions(self, masks: np.ndarray, sim: Any, slots: np.ndarray) -> np.ndarray:
        """Return vector61 actions, compiling only the pool's v8 decision path."""
        if not self._policy.veto_slots or self._policy.veto_variant != "v8":
            return self._policy.actions(masks, sim, slots)

        from src.simd_env.v8_nb import v8_decide
        from src.simd_env.vector61_policy import check_live_rule_config

        del masks  # Vector61SimdPolicy intentionally acts on the runtime's advisory mask.
        slots = np.ascontiguousarray(np.asarray(slots, dtype=np.int64).reshape(-1, 2))
        if not len(slots):
            return np.empty(0, dtype=np.int64)
        states, mask = self._policy.runtime.selection(slots)
        if states.shape[1] != self._policy.input_size:
            raise ValueError(
                f"checkpoint input_size={self._policy.input_size} does not match the active "
                f"featurizer width {states.shape[1]}"
            )
        if getattr(self._policy, "_rule_config_checked", None) is not sim:
            check_live_rule_config(sim)
            self._policy._rule_config_checked = sim
        q = self._policy.q_values(states).cpu().numpy()
        result = v8_decide(sim, slots, q, mask, float(self._policy.veto_spec.lam))
        # Preserve the fields a full-state snapshot captures.  The old hook's
        # per-episode counters are diagnostics only and the trainer never reads
        # them; action semantics are the compiled kernel's contract.
        self._policy.last_veto = {
            "rows": slots.copy(),
            "frame": sim.frame[slots[:, 0]].copy(),
            "counts": result.counts,
            "need": result.need,
            "spacious": result.spacious,
            "base": result.base,
            "final": result.action,
        }
        return result.action.astype(np.int64, copy=False)


class LeagueBV8Policy:
    """Frozen B8 forward with one host transfer of actions per policy group.

    Raster construction stays on the actor CPU.  The Q forward, action mask and
    argmax remain on the model device.  CUDA timing deliberately starts only
    after the device tensors are prepared, so it measures the network/selection
    work rather than CPU featurization or host-to-device copies.
    """

    def __init__(self, checkpoint_path: str, device: str) -> None:
        import torch

        from src.model.ego2s_network import load_ego2s_checkpoint, spec_for
        from src.simd_env.ego2s_policy import ego_config_for

        self.net = load_ego2s_checkpoint(checkpoint_path, device).eval()
        self.device = next(self.net.parameters()).device
        self.checkpoint_path = str(checkpoint_path)
        self.ego_config = ego_config_for(self.net)
        self.obs_spec = spec_for(self.net)
        self.last_forward_gpu_s = 0.0
        # The learner's b-format raster and B8's bv8 raster share these 7/4/30
        # bytes exactly.  A frame-local cache lets the actor make that gather
        # once for learner+B8 rows; v8 scalars remain B8-only.
        self._base_ego_config = replace(self.ego_config, version="b")
        self._base_cache_slots: Optional[np.ndarray] = None
        self._base_cache_obs: Optional[Dict[str, np.ndarray]] = None
        self._base_cache_sim: Optional[Any] = None
        self._base_cache_frame: Optional[np.ndarray] = None
        for param in self.net.parameters():
            param.requires_grad_(False)
        self._torch = torch

    def actions(self, masks: np.ndarray, sim: Any, slots: np.ndarray) -> np.ndarray:
        from src.model.ego2s_network import obs_tensors
        from src.simd_env.ego_raster import build_ego_raster

        slots = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
        if not len(slots):
            self.clear_frame_cache()
            return np.empty(0, dtype=np.int64)
        if self.ego_config.version == "bv8":
            from src.simd_env.ego_raster_bv8 import v8_scalars

            if (
                self._base_cache_sim is sim
                and self._base_cache_slots is not None
                and np.array_equal(self._base_cache_slots, slots)
                and self._base_cache_frame is not None
                and np.array_equal(self._base_cache_frame, sim.frame[slots[:, 0]])
            ):
                base = self._base_cache_obs
                self._base_cache_slots = None
                self._base_cache_obs = None
                assert base is not None
            else:
                self._base_cache_slots = None
                self._base_cache_obs = None
                base = build_ego_raster(sim, self._base_ego_config, backend="numba", rows=slots)
            v8, _ = v8_scalars(sim, slots)
            obs = {
                "local": base["local"],
                "global": base["global"],
                "scalars": np.concatenate([base["scalars"], v8], axis=1),
            }
        else:
            self._base_cache_slots = None
            self._base_cache_obs = None
            obs = build_ego_raster(sim, self.ego_config, backend="numba", rows=slots)
        tensors = obs_tensors(obs, self.device)
        torch = self._torch
        mask = _normal_fallback_mask(
            torch.from_numpy(np.asarray(masks, dtype=np.bool_)).to(self.device)
        )
        events = _cuda_events_for(self.device)
        with torch.no_grad():
            q = self.net(*tensors)
            actions = torch.where(mask, q, torch.full_like(q, -torch.inf)).argmax(dim=1)
        self.last_forward_gpu_s = _cuda_elapsed(events)
        return actions.cpu().numpy().astype(np.int64, copy=False)

    def prime_with_learner_rows(
        self, sim: Any, learner_rows: np.ndarray, b8_rows: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Gather shared b-format rasters once and retain B8's frame-local slice.

        The row order is learner rows followed by B8 rows.  The numba raster
        kernel is row independent, so each slice is byte-identical to its old
        separate invocation.  V8 diagnostics are not included here: they are
        B8-specific and are still evaluated only for B8 rows in :meth:`actions`.
        """
        from src.simd_env.ego_raster import build_ego_raster

        if self.ego_config.version != "bv8":
            raise ValueError("only ego2s-bv8 B8 policies have reusable b-format raster bytes")
        learner_rows = np.ascontiguousarray(np.asarray(learner_rows, dtype=np.int64).reshape(-1, 2))
        b8_rows = np.ascontiguousarray(np.asarray(b8_rows, dtype=np.int64).reshape(-1, 2))
        both = np.concatenate([learner_rows, b8_rows], axis=0)
        all_obs = build_ego_raster(sim, self._base_ego_config, backend="numba", rows=both)
        n = len(learner_rows)
        self.clear_frame_cache()
        self._base_cache_slots = b8_rows.copy()
        self._base_cache_obs = {key: value[n:] for key, value in all_obs.items()}
        self._base_cache_sim = sim
        self._base_cache_frame = np.asarray(sim.frame[b8_rows[:, 0]], dtype=np.int64).copy()
        return {key: value[:n] for key, value in all_obs.items()}

    def clear_frame_cache(self) -> None:
        """Drop a transient raster slice at a segment/rebind boundary."""
        self._base_cache_slots = None
        self._base_cache_obs = None
        self._base_cache_sim = None
        self._base_cache_frame = None


def _normal_fallback_mask(mask: Any) -> Any:
    """Mirror legacy resolved-mask fallback without a host round trip."""
    empty = ~mask.any(dim=1)
    mask = mask.clone()
    mask[:, :3] |= empty[:, None]
    return mask


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
        "research/redesign_m3c_20261008/league_pool.py",
        "src/model/ego2s_network.py",
        "src/simd_env/ego_raster.py",
        "src/simd_env/ego_raster_bv8.py",
        "src/simd_env/v8_nb.py",
    )
    v61_sources = _source_pins(
        "research/redesign_m3c_20261008/league_pool.py",
        "src/simd_env/vector61_policy.py",
        "src/simd_env/vector61_featurizer.py",
        "src/simd_env/vector61_nb.py",
        "src/simd_env/v8_nb.py",
    )
    script_sources = _source_pins("src/simd_env/fast_anchors.py", "src/evaluation/anchors.py")
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
    def policy_ids(self) -> Tuple[str, ...]:
        """Permanent identity order used for deterministic sampling and dispatch."""
        return tuple(identity.policy_id for identity in self.identities)

    def identity_records(self) -> list[Dict[str, Any]]:
        return [identity.record() for identity in self.identities]

    def rebind(self, runtime: Any, world_seeds: Sequence[int]) -> None:
        """Attach segment-local state without reloading immutable frozen weights."""
        seeds = tuple(int(seed) for seed in world_seeds)
        for identity in self.identities:
            policy = self.policies[identity.policy_id]
            if identity.kind == "vector61":
                policy.runtime = runtime
                policy.veto_counters.clear()
                policy.boost_counters.clear()
                policy.live_vetoes.clear()
                policy.last_veto = None
            elif identity.kind == "scripted":
                policy._world_seeds = seeds
            elif identity.kind == "bv8":
                policy.clear_frame_cache()

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
        return self._dispatch(prepared, masks, alive, actions, seats)

    def prime_b8_with_learner_rows(
        self, prepared: Any, learner_rows: np.ndarray, alive: np.ndarray, seats: np.ndarray
    ) -> Optional[Dict[str, np.ndarray]]:
        """Reuse b-format raster bytes between the learner and the B8 pool group.

        ``None`` means no live B8 group, leaving the learner's established
        raster call unchanged.  The cache is consumed in B8's following
        dispatch and is invalidated on any mismatch, so no observation can
        survive a frame, death, or respawn boundary.
        """
        policy = self.policies["b8-s3"]
        if not isinstance(policy, LeagueBV8Policy):
            raise TypeError("b8-s3 pool policy must be LeagueBV8Policy")
        policy.clear_frame_cache()
        b8_rows = self.groups_for(seats, alive).get("b8-s3")
        if b8_rows is None or not len(learner_rows):
            return None
        from src.simd_env.ego_raster import EgoRasterConfig

        if policy._base_ego_config != EgoRasterConfig(version="b"):
            # A differently configured learner raster cannot be reused merely
            # because its arrays have compatible shapes.
            return None
        return policy.prime_with_learner_rows(prepared, learner_rows, b8_rows)

    def _dispatch(
        self,
        prepared: Any,
        masks: np.ndarray,
        alive: np.ndarray,
        actions: np.ndarray,
        seats: np.ndarray,
    ) -> Dict[str, Any]:
        """Shared one-call-per-policy dispatcher in permanent identity order."""
        groups = self.groups_for(seats, alive)
        counts = {policy_id: int(len(rows)) for policy_id, rows in groups.items()}
        start = time.perf_counter()
        gpu_seconds = 0.0
        for policy_id, rows in groups.items():
            choice = np.asarray(
                self.policies[policy_id].actions(masks[rows[:, 0], rows[:, 1]], prepared, rows),
                dtype=np.int64,
            ).reshape(-1)
            gpu_seconds += float(getattr(self.policies[policy_id], "last_forward_gpu_s", 0.0))
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


def _cuda_events_for(device: Any):
    import torch

    # The model device, rather than global CUDA availability, is authoritative:
    # a CPU/MPS policy on a CUDA-equipped host has zero CUDA work.
    if torch.device(device).type != "cuda":
        return None
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end


def _cuda_elapsed(events: Any) -> float:
    if events is None:
        return 0.0
    start, end = events
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / 1000.0


def build_static_pool(
    identities: Sequence[PoolIdentity],
    *,
    world_seeds: Sequence[int],
    runtime: Any,
    device: str,
    resident_bv8: Optional[LeagueBV8Policy] = None,
) -> StaticPool:
    """Validate and load the permanent pool.  Models are eval-only and immutable."""
    for identity in identities:
        identity.validate()
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    policies: Dict[str, Any] = {}
    for identity in identities:
        if identity.kind == "scripted":
            policies[identity.policy_id] = FastProfileAnchorSimdPolicy(
                identity.policy_id.split(":", 1)[1], world_seeds
            )
        elif identity.kind == "bv8":
            if identity.mask_config != "resolved":
                raise RuntimeError(f"{identity.policy_id}: LeagueBV8Policy requires resolved mask")
            if resident_bv8 is not None and resident_bv8.checkpoint_path == str(
                identity.checkpoint_path
            ):
                policies[identity.policy_id] = resident_bv8
            else:
                policies[identity.policy_id] = LeagueBV8Policy(
                    str(identity.checkpoint_path), device=device
                )
            if policies[identity.policy_id].obs_spec != identity.obs_spec:
                raise RuntimeError(f"{identity.policy_id}: checkpoint obs spec does not match pin")
        else:
            policy = Vector61SimdPolicy(
                str(identity.checkpoint_path),
                runtime,
                veto_slots=range(6) if identity.wrapper == V8_METHOD else (),
                veto_variant="v8" if identity.wrapper == V8_METHOD else "v2",
                veto_lambda=8.0 if identity.wrapper == V8_METHOD else None,
                forward="batched",
            )
            # Vector61SimdPolicy's loader follows the global DeviceManager.  The
            # initial league rule explicitly keeps these opponents on CPU.
            import torch

            policy.network = policy.network.to("cpu")
            policy.device = torch.device("cpu")
            if policy.input_size != 61:
                raise RuntimeError(f"{identity.policy_id}: checkpoint is not vector61")
            # Keep bare vector61 on its exact established implementation.  The
            # v8 pool entries are safe to compile: their diagnostics are unused
            # by M3-C, and v8_nb is action-parity pinned against the live hook.
            policies[identity.policy_id] = (
                LeagueVector61Policy(policy) if identity.wrapper == V8_METHOD else policy
            )
        _freeze_policy(policies[identity.policy_id])
    return StaticPool(identities, policies)


def build_default_static_pool(
    args: Any, ctx: Any, sim: Any, runtime: Any, device: Any
) -> StaticPool:
    """Collector-facing constructor for the permanent first-slice pool.

    ``args`` and ``ctx`` are accepted so the actor call site has one stable
    extension hook; ``league_checkpoint_dir`` and ``league_b8_checkpoint`` select
    the independently pinned pool artifacts, never the mutable learner init.
    The simulator's actual world seeds are passed to deterministic scripted
    anchors, preserving their established per-world action semantics.
    """
    del ctx
    directory = getattr(args, "league_checkpoint_dir", None)
    b8_checkpoint = getattr(args, "league_b8_checkpoint", None)
    seeds = sim.get_world_seeds()
    identities = default_pool_identities(directory, b8_checkpoint)
    for identity in identities:
        identity.validate()
    cached = getattr(args, "_league_static_pool", None)
    if cached is not None and cached.identities == identities:
        cached.rebind(runtime, seeds)
        return cached
    resident = getattr(args, "_league_b8_policy", None)
    pool = build_static_pool(
        identities,
        world_seeds=seeds,
        runtime=runtime,
        device=str(device),
        resident_bv8=resident,
    )
    args._league_b8_policy = pool.policies["b8-s3"]
    args._league_static_pool = pool
    return pool


def _freeze_policy(policy: Any) -> None:
    """Enforce eval/no-grad tensor ownership without assuming a policy class."""
    wrapped = getattr(policy, "_policy", None)
    if wrapped is not None:
        _freeze_policy(wrapped)
        return
    for value in vars(policy).values():
        if hasattr(value, "eval") and hasattr(value, "parameters"):
            value.eval()
            for param in value.parameters():
                param.requires_grad_(False)
