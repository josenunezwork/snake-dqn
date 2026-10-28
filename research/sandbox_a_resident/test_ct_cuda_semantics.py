"""CUDA device-consistency contracts checked on CPU via meta-device and logical-device traps.

Exercises device transitions across collection -> scoring -> zero-residual-init check ->
three-arm fitting without claiming execution on physical CUDA hardware.
Run explicitly via ``python -m research.sandbox_a_resident.ct_preflight <packet> --cuda-semantics``
and included in normal ``CT_FOCUSED_TESTS`` preflight.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten, tree_map

from research.sandbox_a_resident.ct_branching import MergedScoredRoots
from research.sandbox_a_resident.ct_fit import fit_all_three_arms
from research.sandbox_a_resident.ct_models import (
    CT_ARM_NAMES,
    CTPolicyArm,
    verify_zero_residual_init,
)
from research.sandbox_a_resident.ct_snapshots import LosslessBoundedSnapshotStore, RootMetadata
from research.sandbox_a_resident.ct_task import AnchorArchive, CTControl
from research.sandbox_a_resident.ct_topology import pack_sparse_topology
from src.model.ego2s_network import Ego2sNet

_BV8_CFG: dict[str, Any] = {
    "local_channels": 7,
    "n_scalars": 60,
    "n_out": 6,
}


def _normalize_logical_device(dev: Any) -> str | None:
    if dev is None:
        return None
    s = str(dev)
    if s in ("cuda", "cuda:0"):
        return "cuda:0"
    if s == "cpu":
        return "cpu"
    return s


class _LogicalTensor(torch.Tensor):
    @property
    def device(self) -> torch.device:  # type: ignore[override]
        return torch.device(getattr(self, "_logical_device_tag", "cpu"))

    def __deepcopy__(self, memo: dict[int, Any]) -> "_LogicalTensor":
        tag = getattr(self, "_logical_device_tag", "cpu")
        base = self.detach().clone().as_subclass(torch.Tensor)
        out = base.as_subclass(_LogicalTensor).requires_grad_(self.requires_grad)
        out._logical_device_tag = tag  # type: ignore[attr-defined]
        if isinstance(self, torch.nn.Parameter):
            param = torch.nn.Parameter(out, requires_grad=self.requires_grad)
            out = param  # type: ignore[assignment]
            out._logical_device_tag = tag  # type: ignore[attr-defined]
        memo[id(self)] = out
        return out


class _LogicalCudaTrapMode(TorchFunctionMode):
    """Intercepts torch ops on CPU tensors while enforcing strict cpu vs cuda:0 separation."""

    def __init__(self) -> None:
        super().__init__()
        self.cuda_op_count: int = 0

    def tag_of(self, t: torch.Tensor) -> str:
        return getattr(t, "_logical_device_tag", "cpu")

    def set_tag(self, t: torch.Tensor, tag: str) -> torch.Tensor:
        is_param = isinstance(t, torch.nn.Parameter)
        req_grad = t.requires_grad
        if tag == "cuda:0":
            if not isinstance(t, _LogicalTensor):
                t = t.as_subclass(_LogicalTensor)
            t._logical_device_tag = tag  # type: ignore[attr-defined]
            if is_param and not isinstance(t, torch.nn.Parameter):
                t = torch.nn.Parameter(t, requires_grad=req_grad)
                t._logical_device_tag = tag  # type: ignore[attr-defined]
        else:
            if isinstance(t, _LogicalTensor):
                t = t.as_subclass(torch.Tensor)
                if is_param:
                    t = torch.nn.Parameter(t, requires_grad=req_grad)
            t._logical_device_tag = tag  # type: ignore[attr-defined]
        return t

    def tag_module(self, mod: torch.nn.Module, tag: str) -> torch.nn.Module:
        for sub in mod.modules():
            for k, p in list(sub._parameters.items()):
                if p is not None:
                    sub._parameters[k] = self.set_tag(p, tag)  # type: ignore[assignment]
            for k, b in list(sub._buffers.items()):
                if b is not None:
                    sub._buffers[k] = self.set_tag(b, tag)
        return mod

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})

        if func is torch._has_compatible_shallow_copy_type:
            return (self.tag_of(args[0]) == self.tag_of(args[1])) and bool(func(*args, **kwargs))

        if func is torch.Tensor.cpu:
            out = func(*args, **kwargs)
            if isinstance(out, torch.Tensor):
                return self.set_tag(out, "cpu")
            return out

        if func is torch.Tensor.cuda:
            out = args[0].clone()
            return self.set_tag(out, "cuda:0")

        # Handle Tensor.to(...) device transitions.
        if func is torch.Tensor.to:
            self_t = args[0]
            cur_tag = self.tag_of(self_t)
            target_tag: str | None = None
            new_args = [self_t]
            for a in args[1:]:
                if isinstance(a, (torch.device, str)) and _normalize_logical_device(a) in (
                    "cpu",
                    "cuda:0",
                ):
                    target_tag = _normalize_logical_device(a)
                    new_args.append(torch.device("cpu"))
                elif isinstance(a, torch.Tensor):
                    target_tag = self.tag_of(a)
                    new_args.append(a)
                else:
                    new_args.append(a)
            if "device" in kwargs and kwargs["device"] is not None:
                norm = _normalize_logical_device(kwargs["device"])
                if norm in ("cpu", "cuda:0"):
                    target_tag = norm
                    kwargs["device"] = torch.device("cpu")
            out = func(*new_args, **kwargs)
            final_tag = target_tag if target_tag is not None else cur_tag
            if isinstance(out, torch.Tensor):
                return self.set_tag(out, final_tag)
            return out

        # Handle factory functions with explicit device=cuda/cuda:0.
        requested_tag: str | None = None
        if "device" in kwargs and kwargs["device"] is not None:
            norm = _normalize_logical_device(kwargs["device"])
            if norm in ("cpu", "cuda:0"):
                requested_tag = norm
                kwargs["device"] = torch.device("cpu")

        flat_items, _ = tree_flatten((args, kwargs))
        input_tensors = [x for x in flat_items if isinstance(x, torch.Tensor)]
        non_scalar_cpu_or_cuda = [
            t for t in input_tensors if not (t.ndim == 0 and self.tag_of(t) == "cpu")
        ]
        check_tensors = non_scalar_cpu_or_cuda if non_scalar_cpu_or_cuda else input_tensors
        seen_tags = {self.tag_of(t) for t in check_tensors}
        if len(seen_tags) > 1:
            raise RuntimeError(f"Cross-device tensor mix in {func}: {sorted(seen_tags)}")

        out = func(*args, **kwargs)
        inherited_tag = (
            requested_tag
            if requested_tag is not None
            else (next(iter(seen_tags)) if len(seen_tags) == 1 else "cpu")
        )
        if inherited_tag == "cuda:0":
            self.cuda_op_count += 1

        def _mark(x: Any) -> Any:
            if isinstance(x, torch.Tensor):
                return self.set_tag(x, inherited_tag)
            return x

        return tree_map(_mark, out)


def test_meta_device_propagation_across_all_three_ct_arms() -> None:
    """When b8_model is on a non-CPU device ('meta'), CTPolicyArm keeps all submodules there."""
    b8_meta = Ego2sNet(**_BV8_CFG).to("meta").eval()
    local = torch.empty((4, 7, 31, 31), dtype=torch.uint8, device="meta")
    glob = torch.empty((4, 4, 37, 37), dtype=torch.uint8, device="meta")
    scalars = torch.empty((4, 60), dtype=torch.float32, device="meta")
    planes = torch.empty((4, 4, 83, 145), dtype=torch.float32, device="meta")
    summaries = torch.empty((4, 6), dtype=torch.float32, device="meta")

    for arm_name in CT_ARM_NAMES:
        arm = CTPolicyArm(b8_meta, arm_name)
        param_devices = {p.device.type for p in arm.parameters()}
        assert param_devices == {"meta"}, f"{arm_name} leaked non-meta parameters: {param_devices}"
        logits, rep = arm.forward_with_features(
            local, glob, scalars, topo_planes=planes, topo_summaries=summaries
        )
        assert logits.device.type == "meta" and logits.shape == (4, 6)
        expected_rep_dim = 384 if arm_name == "full_bv8_plus_topology" else 256
        assert rep.device.type == "meta" and rep.shape == (4, expected_rep_dim)


def test_cross_device_trap_catches_raw_cpu_input_on_relocated_cuda_model() -> None:
    """Sanity-check that _LogicalCudaTrapMode raises RuntimeError on a genuine CPU/CUDA mix."""
    b8_model = Ego2sNet(**_BV8_CFG).eval()
    local_cpu = torch.zeros((2, 7, 31, 31), dtype=torch.uint8)
    glob_cpu = torch.zeros((2, 4, 37, 37), dtype=torch.uint8)
    scalars_cpu = torch.zeros((2, 60), dtype=torch.float32)

    trap = _LogicalCudaTrapMode()
    trap.tag_module(b8_model, "cuda:0")
    with trap:
        with pytest.raises(RuntimeError, match="Cross-device tensor mix"):
            _ = b8_model(local_cpu, glob_cpu, scalars_cpu)


def test_zero_residual_init_and_three_arm_fit_with_relocated_cuda_b8_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate collection moving b8_model to cuda:0, then running zero-init check + fit."""
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _seed: None)

    rng = np.random.default_rng(20261028)
    b8_model = Ego2sNet(**_BV8_CFG).eval()

    # Build a small synthetic CPU store + scored labels + anchor archive (as loaded/held on host).
    store = LosslessBoundedSnapshotStore(
        grid_h=83, grid_w=145, max_roots=8, max_bytes=32 * 1024 * 1024
    )
    n_roots = 6
    for i in range(n_roots):
        part = "train" if i < 4 else "assessment"
        kind = "death_lookback" if i % 2 == 0 else "ordinary"
        lb = 140 if kind == "death_lookback" else None
        meta = RootMetadata(
            root_index=i,
            root_uid=f"r-{i}",
            kind=kind,
            partition=part,
            world_id=i,
            mix="self",
            roster_index=i,
            life_id=0,
            tick=20,
            hero_length=700,
            death_event_id=i if kind == "death_lookback" else None,
            lookback=lb,
            death_tick=160 if kind == "death_lookback" else None,
            death_length=700 if kind == "death_lookback" else None,
            b8_action=1,
        )
        planes_u16 = np.zeros((4, 83, 145), dtype=np.uint16)
        planes_u16[0, 10, 10] = 65535
        planes_u16[1, 10, 10] = 5120
        sums_f32 = rng.uniform(-1.0, 1.0, size=(6,)).astype(np.float32)
        store.metadata.append(meta)
        store.snapshots.append(None)  # type: ignore[arg-type]
        store._locals.append(
            torch.from_numpy(rng.integers(0, 256, size=(7, 31, 31), dtype=np.uint8))
        )
        store._globs.append(
            torch.from_numpy(rng.integers(0, 256, size=(4, 37, 37), dtype=np.uint8))
        )
        store._scalars.append(torch.from_numpy(rng.standard_normal(size=(60,)).astype(np.float32)))
        store._q_b8.append(torch.from_numpy(rng.standard_normal(size=(6,)).astype(np.float32)))
        store._legal_masks.append(torch.tensor([True, True, True, False, False, False]))
        store._topo_builder.append_batch(planes_u16[None, ...], sums_f32[None, ...])

    q_sel = rng.uniform(0.2, 1.0, size=(n_roots, 6)).astype(np.float64)
    q_sel[:, 3:] = np.nan
    s_sel = np.clip(q_sel, 0.0, 1.0)
    m_sel = np.full((n_roots, 6), 700.0, dtype=np.float64)
    a_star = np.zeros(n_roots, dtype=np.int64)
    forb = np.zeros((n_roots, 6), dtype=bool)
    forb[:, 2] = True
    scored = MergedScoredRoots(
        num_roots=n_roots,
        legal_mask=np.tile(
            np.array([[True, True, True, False, False, False]], dtype=bool), (n_roots, 1)
        ),
        b8_action=np.ones(n_roots, dtype=np.int64),
        s_sel=s_sel,
        m_sel=m_sel,
        q_sel=q_sel,
        a_star=a_star,
        forbidden_sel=forb,
        s_score=s_sel.copy(),
        m_score=m_sel.copy(),
        q_score=q_sel.copy(),
        chunk_sha256s={0: "0" * 64},
        total_logical_branch_steps=100,
        total_executed_sim_steps=100,
        total_scoring_wall_sec=0.1,
    )

    n_anc = 8
    anc_planes = np.zeros((n_anc, 4, 83, 145), dtype=np.uint16)
    anc_planes[:, 0, 12, 12] = 65535
    anc_planes[:, 1, 12, 12] = 5120
    anc_sums = rng.uniform(-1.0, 1.0, size=(n_anc, 6)).astype(np.float32)
    anchors = AnchorArchive(
        num_rows=n_anc,
        world_id=np.array([0, 1, 2, 3, 0, 1, 4, 5], dtype=np.int32),
        mix_id=np.zeros(n_anc, dtype=np.int8),
        is_assessment=np.array([False, False, False, False, False, False, True, True], dtype=bool),
        life_id=np.zeros(n_anc, dtype=np.int32),
        tick=np.arange(n_anc, dtype=np.int32),
        hero_length=np.full(n_anc, 700, dtype=np.int32),
        local=torch.from_numpy(rng.integers(0, 256, size=(n_anc, 7, 31, 31), dtype=np.uint8)),
        glob=torch.from_numpy(rng.integers(0, 256, size=(n_anc, 4, 37, 37), dtype=np.uint8)),
        scalars=torch.from_numpy(rng.standard_normal(size=(n_anc, 60)).astype(np.float32)),
        legal_mask=torch.tensor([[True, True, True, False, False, False]] * n_anc),
        q_b8=torch.from_numpy(rng.standard_normal(size=(n_anc, 6)).astype(np.float32)),
        b8_action=np.ones(n_anc, dtype=np.int8),
        topology=pack_sparse_topology(anc_planes, anc_sums),
    )

    ctrl = CTControl(
        tiny=True,
        worlds=6,
        minibatches=2,
        searched_batch_size=2,
        anchor_batch_size=4,
    )
    world_partitions = ["train", "train", "train", "train", "assessment", "assessment"]

    trap = _LogicalCudaTrapMode()

    with trap:
        # Simulate collection in-place device move: b8_network.to("cuda:0").
        b8_model.to(torch.device("cuda:0"))
        assert all(trap.tag_of(p) == "cuda:0" for p in b8_model.parameters())

        # Direct call without device= (inferred from relocated model) and explicit device="cuda:0".
        check_inferred = verify_zero_residual_init(b8_model)
        assert check_inferred["status"] == "PASS"

        fit_res = fit_all_three_arms(
            control=ctrl,
            store=store,
            scored=scored,
            anchors=anchors,
            world_partitions=world_partitions,
            b8_model=b8_model,
            device="cuda",
        )
        assert fit_res is not None
        arm_preds, shared_meta = fit_res
        assert set(arm_preds.keys()) == set(CT_ARM_NAMES)
        assert shared_meta["zero_residual_init_verification"]["status"] == "PASS"
        assert shared_meta["zero_residual_init_verification"]["device"] == "cuda:0"
        assert trap.cuda_op_count > 100
