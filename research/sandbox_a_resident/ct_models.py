"""Three C/T policy arms and zero-initialized residual dynamic-topology branch.

Implements Ruling §4:
- ``frozen_phi``: frozen ``ego2s-bv8`` encoder ``phi_B8(s)`` with trainable linear +
  zero-initialized nonlinear MLP residual head (``head_lr = 1e-4``), starting at
  exact B8 output.
- ``full_bv8``: fully trainable ``ego2s-bv8`` trunk (``trunk_lr = 1e-5``) and head
  (``head_lr = 1e-4``), starting at exact B8 output.
- ``full_bv8_plus_topology`` (T arm): ``full_bv8`` plus ``TopologyEncoder``
  (``trunk_lr = 1e-5``) over frozen ``(4, H, W)`` dynamic-topology approximation
  planes, 6 action-conditioned reach/TTL-slack summaries, and hero scalars, joined
  through a unit-tested zero-initialized residual ``topo_head: Linear(128, 6)``
  (``head_lr = 1e-4``) so step-0 output equals exact old-path B8 output bitwise.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from research.sandbox_a_resident.training_model import B8Actor
from src.model.ego2s_network import Ego2sNet

CT_ARM_NAMES: tuple[str, ...] = (
    "frozen_phi",
    "full_bv8",
    "full_bv8_plus_topology",
)


def _require_bv8_net(model: Ego2sNet) -> None:
    if model.config.get("local_channels") != 7 or model.config.get("n_scalars") != 60:
        raise ValueError("C/T requires an ego2s-bv8 (7 local channels, 60 scalars) B8 model")
    if model.config.get("n_out") != 6:
        raise ValueError("C/T requires 6 relative actions")


class TopologyEncoder(nn.Module):
    """Encoder for frozen dynamic-topology approximations (planes + 6 BFS summaries)."""

    def __init__(self, n_scalars: int = 60, out_dim: int = 128) -> None:
        super().__init__()
        self.board_conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 10)),
            nn.Flatten(),
            nn.Linear(64 * 6 * 10, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.summary_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(128 + 64 + n_scalars, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        topo_planes: torch.Tensor,
        topo_summaries: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        if topo_planes.ndim != 4 or topo_planes.shape[1] != 4:
            raise ValueError(
                f"topo_planes must have shape (B, 4, H, W), got {tuple(topo_planes.shape)}"
            )
        if topo_summaries.ndim != 2 or topo_summaries.shape[1] != 6:
            raise ValueError(
                f"topo_summaries must have shape (B, 6), got {tuple(topo_summaries.shape)}"
            )
        planes_f = topo_planes.float()
        # Pad 1-cell exterior arena wall ring: channel 0 (wall_occupancy) = 1.0, others = 0.0.
        ch0 = F.pad(planes_f[:, 0:1], (1, 1, 1, 1), mode="constant", value=1.0)
        ch_rest = F.pad(planes_f[:, 1:4], (1, 1, 1, 1), mode="constant", value=0.0)
        padded = torch.cat((ch0, ch_rest), dim=1)
        b_feat = self.board_conv(padded)
        s_feat = self.summary_mlp(topo_summaries.float())
        return self.fuse(torch.cat((b_feat, s_feat, scalars.float()), dim=-1))


class CTPolicyArm(nn.Module):
    """Unified C/T policy wrapper supporting frozen_phi, full_bv8, full_bv8_plus_topology."""

    def __init__(self, b8_model: Ego2sNet, arm_name: str) -> None:
        super().__init__()
        if arm_name not in CT_ARM_NAMES:
            raise ValueError(f"unknown C/T arm {arm_name!r}; expected one of {CT_ARM_NAMES}")
        _require_bv8_net(b8_model)
        self.arm_name = arm_name
        self.config = dict(b8_model.config)

        self.local = copy.deepcopy(b8_model.local)
        self.glob = copy.deepcopy(b8_model.glob)
        self.fuse = copy.deepcopy(b8_model.fuse)
        self.adv = copy.deepcopy(b8_model.adv)

        ref_param = next(b8_model.parameters(), None)
        model_device = ref_param.device if ref_param is not None else torch.device("cpu")
        model_dtype = ref_param.dtype if ref_param is not None else torch.float32

        self.phi_mlp_hidden: nn.Sequential | None = None
        self.phi_mlp_out: nn.Linear | None = None
        self.topology_encoder: TopologyEncoder | None = None
        self.topo_head: nn.Linear | None = None

        if arm_name == "frozen_phi":
            for mod in (self.local, self.glob, self.fuse):
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad_(False)
            self.phi_mlp_hidden = nn.Sequential(
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
            )
            self.phi_mlp_out = nn.Linear(128, 6)
            nn.init.zeros_(self.phi_mlp_out.weight)
            nn.init.zeros_(self.phi_mlp_out.bias)
            self.phi_mlp_hidden = self.phi_mlp_hidden.to(device=model_device, dtype=model_dtype)
            self.phi_mlp_out = self.phi_mlp_out.to(device=model_device, dtype=model_dtype)
        elif arm_name == "full_bv8_plus_topology":
            self.topology_encoder = TopologyEncoder(
                n_scalars=int(self.config["n_scalars"]), out_dim=128
            )
            self.topo_head = nn.Linear(128, 6)
            nn.init.zeros_(self.topo_head.weight)
            nn.init.zeros_(self.topo_head.bias)
            self.topology_encoder = self.topology_encoder.to(device=model_device, dtype=model_dtype)
            self.topo_head = self.topo_head.to(device=model_device, dtype=model_dtype)

    def train(self, mode: bool = True) -> "CTPolicyArm":
        super().train(mode)
        if self.arm_name == "frozen_phi":
            self.local.eval()
            self.glob.eval()
            self.fuse.eval()
        return self

    def encode_bv8(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        loc_f = local.float() * (1.0 / 255.0)
        glb_f = glob.float() * (1.0 / 255.0)
        scl_f = scalars.float()
        if self.arm_name == "frozen_phi":
            with torch.no_grad():
                return self.fuse(torch.cat((self.local(loc_f), self.glob(glb_f), scl_f), dim=-1))
        return self.fuse(torch.cat((self.local(loc_f), self.glob(glb_f), scl_f), dim=-1))

    def forward_with_features(
        self,
        local: torch.Tensor,
        glob: torch.Tensor,
        scalars: torch.Tensor,
        topo_planes: torch.Tensor | None = None,
        topo_summaries: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, detached_representation)`` in B8 advantage coordinates."""
        h_bv8 = self.encode_bv8(local, glob, scalars)
        logits = self.adv(h_bv8)
        if self.arm_name == "frozen_phi":
            assert self.phi_mlp_hidden is not None and self.phi_mlp_out is not None
            h_mlp = self.phi_mlp_hidden(h_bv8)
            logits = logits + self.phi_mlp_out(h_mlp)
            return logits, h_bv8.detach()
        if self.arm_name == "full_bv8":
            return logits, h_bv8.detach()
        assert self.topology_encoder is not None and self.topo_head is not None
        if topo_planes is None or topo_summaries is None:
            raise ValueError("full_bv8_plus_topology requires topo_planes and topo_summaries")
        h_topo = self.topology_encoder(topo_planes, topo_summaries, scalars)
        logits = logits + self.topo_head(h_topo)
        rep = torch.cat((h_bv8.detach(), h_topo.detach()), dim=-1)
        return logits, rep

    def forward(
        self,
        local: torch.Tensor,
        glob: torch.Tensor,
        scalars: torch.Tensor,
        topo_planes: torch.Tensor | None = None,
        topo_summaries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits, _ = self.forward_with_features(
            local, glob, scalars, topo_planes=topo_planes, topo_summaries=topo_summaries
        )
        return logits

    def build_optimizer(self, *, trunk_lr: float = 1e-5, head_lr: float = 1e-4) -> torch.optim.Adam:
        """Create Adam with frozen trunk_lr=1e-5 and head_lr=1e-4 parameter groups."""
        if self.arm_name == "frozen_phi":
            assert self.phi_mlp_hidden is not None and self.phi_mlp_out is not None
            head_params = (
                list(self.adv.parameters())
                + list(self.phi_mlp_hidden.parameters())
                + list(self.phi_mlp_out.parameters())
            )
            return torch.optim.Adam([{"params": head_params, "lr": float(head_lr)}])

        trunk_params = (
            list(self.local.parameters())
            + list(self.glob.parameters())
            + list(self.fuse.parameters())
        )
        head_params = list(self.adv.parameters())
        if self.arm_name == "full_bv8_plus_topology":
            assert self.topology_encoder is not None and self.topo_head is not None
            trunk_params = trunk_params + list(self.topology_encoder.parameters())
            head_params = head_params + list(self.topo_head.parameters())
        return torch.optim.Adam(
            [
                {"params": trunk_params, "lr": float(trunk_lr)},
                {"params": head_params, "lr": float(head_lr)},
            ]
        )


def verify_zero_residual_init(
    b8_model: Ego2sNet,
    *,
    grid_h: int = 83,
    grid_w: int = 145,
    seed: int = 20261028,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Verify exact step-0 bitwise output parity with B8Actor and step-2 encoder grad flow."""
    ref_param = next(b8_model.parameters(), None)
    inferred_dev = ref_param.device if ref_param is not None else torch.device("cpu")
    exec_device = (
        torch.device("cuda:0" if str(device) == "cuda" else device)
        if device is not None
        else inferred_dev
    )
    b8_eval = copy.deepcopy(b8_model).to(exec_device).eval()

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    local = torch.randint(0, 256, (8, 7, 31, 31), generator=gen, dtype=torch.uint8).to(exec_device)
    glob = torch.randint(0, 256, (8, 4, 37, 37), generator=gen, dtype=torch.uint8).to(exec_device)
    scalars = torch.randn((8, 60), generator=gen, dtype=torch.float32).to(exec_device)
    planes_a = torch.rand((8, 4, grid_h, grid_w), generator=gen, dtype=torch.float32).to(
        exec_device
    )
    sums_a = (torch.rand((8, 6), generator=gen, dtype=torch.float32) * 2.0 - 1.0).to(exec_device)
    planes_b = torch.ones_like(planes_a)
    sums_b = torch.full_like(sums_a, -1.0)

    ref_actor = B8Actor(b8_model=b8_eval, temperature=1.0).to(exec_device).eval()
    with torch.no_grad():
        ref_out = ref_actor(local, glob, scalars)

    arm_checks: dict[str, bool] = {}
    for name in CT_ARM_NAMES:
        torch.manual_seed(int(seed))
        if exec_device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        arm = CTPolicyArm(b8_eval, name).to(exec_device).eval()
        with torch.no_grad():
            out_a = arm(local, glob, scalars, topo_planes=planes_a, topo_summaries=sums_a)
            out_b = arm(local, glob, scalars, topo_planes=planes_b, topo_summaries=sums_b)
        if not torch.equal(out_a, ref_out) or not torch.equal(out_b, ref_out):
            raise AssertionError(f"step-0 output mismatch vs exact B8Actor on arm={name}")
        arm_checks[name] = True

    # Verify T-arm zero-residual gradient mechanics:
    # Step 1: topo_head receives non-zero gradient while topology_encoder grad is exact 0.
    # Step 2 (after 1 optimizer update): topology_encoder receives non-zero gradient.
    torch.manual_seed(int(seed))
    if exec_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    t_arm = CTPolicyArm(b8_eval, "full_bv8_plus_topology").to(exec_device).train()
    opt = t_arm.build_optimizer(trunk_lr=1e-5, head_lr=1e-4)
    target = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1], dtype=torch.long, device=exec_device)

    opt.zero_grad()
    logits1 = t_arm(local, glob, scalars, topo_planes=planes_a, topo_summaries=sums_a)
    loss1 = F.cross_entropy(logits1, target)
    loss1.backward()
    assert t_arm.topo_head is not None and t_arm.topology_encoder is not None
    head_grad_step1 = float(t_arm.topo_head.weight.grad.detach().cpu().abs().max().item())
    enc_grad_step1 = max(
        float(p.grad.detach().cpu().abs().max().item())
        for p in t_arm.topology_encoder.parameters()
        if p.grad is not None
    )
    if head_grad_step1 <= 0.0 or enc_grad_step1 != 0.0:
        raise AssertionError(
            f"unexpected step-1 T-arm grads: head={head_grad_step1} enc={enc_grad_step1}"
        )
    opt.step()

    opt.zero_grad()
    logits2 = t_arm(local, glob, scalars, topo_planes=planes_a, topo_summaries=sums_a)
    loss2 = F.cross_entropy(logits2, target)
    loss2.backward()
    enc_grad_step2 = max(
        float(p.grad.detach().cpu().abs().max().item())
        for p in t_arm.topology_encoder.parameters()
        if p.grad is not None
    )
    if enc_grad_step2 <= 0.0:
        raise AssertionError("topology_encoder received zero gradient on step 2")

    return {
        "status": "PASS",
        "device": str(exec_device),
        "exact_step0_b8_identity": arm_checks,
        "t_arm_step1_head_grad_max": head_grad_step1,
        "t_arm_step1_encoder_grad_max": enc_grad_step1,
        "t_arm_step2_encoder_grad_max": enc_grad_step2,
    }
