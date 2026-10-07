"""Learner-side cost probe for the proposed ego2s dueling CNN (redesign scope).

Builds the draft network from the scoping doc (local 6x31x31 + global 4x37x37 +
12 scalars -> dueling V + A(6)) and times, per sample:

- ``act``   — no-grad forward at batch 512 (acting for 512 agent slots);
- ``train`` — a Double-DQN style update at batch 512: online forward on s and
  s', target forward on s', Huber loss, backward, Adam step.

Run on one CPU thread (``torch.set_num_threads(1)``) and, if available, on MPS.
Nothing here is evidence for a gate; it sizes the compute plan only.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn


class Ego2sNet(nn.Module):
    """Draft dueling CNN for the ``ego2s`` observation (not registered anywhere)."""

    def __init__(self) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.glob = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(nn.Linear(256 + 128 + 12, 256), nn.LayerNorm(256), nn.ReLU())
        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, 6)

    def forward(self, local, glob, scalars):
        h = self.fuse(torch.cat([self.local(local), self.glob(glob), scalars], dim=1))
        a = self.adv(h)
        return self.value(h) + a - a.mean(dim=1, keepdim=True)


def batch(n: int, device):
    return (
        torch.rand(n, 6, 31, 31, device=device),
        torch.rand(n, 4, 37, 37, device=device),
        torch.rand(n, 12, device=device),
    )


def sync(device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def probe(device, bs: int, reps: int) -> dict:
    net, tgt = Ego2sNet().to(device), Ego2sNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    s, s2 = batch(bs, device), batch(bs, device)
    act = torch.randint(0, 6, (bs, 1), device=device)
    rew = torch.rand(bs, device=device)

    def do_act():
        with torch.no_grad():
            net(*s)

    def do_train():
        q = net(*s).gather(1, act).squeeze(1)
        with torch.no_grad():
            a2 = net(*s2).argmax(1, keepdim=True)
            y = rew + 0.99 * tgt(*s2).gather(1, a2).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    out = {"device": str(device), "batch": bs, "params": sum(p.numel() for p in net.parameters())}
    for name, fn in (("act", do_act), ("train", do_train)):
        fn()
        sync(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        sync(device)
        dt = (time.perf_counter() - t0) / reps
        out[f"{name}_samples_per_s"] = round(bs / dt, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mps", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    print(json.dumps(probe(torch.device("cpu"), args.batch, args.reps)), flush=True)
    if args.mps and torch.backends.mps.is_available():
        print(json.dumps(probe(torch.device("mps"), args.batch, args.reps * 4)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
