#!/usr/bin/env python3
"""M4-B warm start: widen an ``ego2s-b`` student into an ``ego2s-bv8`` one (Q unchanged).

``ego2s-bv8`` appends the 30 v8 feature scalars after the 30 ``ego2s-b`` scalars, and
:class:`~src.model.ego2s_network.Ego2sNet` feeds ``cat(local 256, global 128, scalars)`` to
its first fuse layer. The widened network copies every weight and gives that layer 30 new
input columns (``384 + 30 .. 384 + 59``) set to zero, so its Q on a bv8 observation equals the
original Q on the b part of it exactly (pinned in ``tests/test_ego_raster_bv8.py``).

  ./venv/bin/python research/redesign_m4b_20261012/widen_bv8.py SRC_B.pth OUT_BV8.pth
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

N_LOCAL_FEATS = 256 + 128  # Ego2sNet local + global tower widths ahead of the scalars


def widen_net(net_b):
    """A new bv8 :class:`Ego2sNet` whose Q equals ``net_b``'s on the b part of the input."""
    import torch

    from src.model.ego2s_network import OBS_SHAPES, OBS_SPEC_B, OBS_SPEC_BV8, Ego2sNet, spec_for

    if spec_for(net_b) != OBS_SPEC_B:
        raise ValueError(f"expected an {OBS_SPEC_B} network, got {spec_for(net_b)}")
    nb = OBS_SHAPES[OBS_SPEC_B][1]
    channels, nbv8 = OBS_SHAPES[OBS_SPEC_BV8]
    wide = Ego2sNet(local_channels=channels, n_scalars=nbv8, n_out=net_b.config["n_out"])
    src = net_b.state_dict()
    dst = wide.state_dict()
    with torch.no_grad():
        for k, v in src.items():
            if k == "fuse.0.weight":
                w = torch.zeros_like(dst[k])
                w[:, : N_LOCAL_FEATS + nb] = v
                dst[k] = w
            else:
                if dst[k].shape != v.shape:
                    raise ValueError(f"unexpected shape change at {k}")
                dst[k] = v.clone()
    wide.load_state_dict(dst)
    return wide.eval()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path, help="ego2s-b checkpoint")
    ap.add_argument("out", type=Path, help="ego2s-bv8 checkpoint to write (create-only)")
    args = ap.parse_args(argv)
    import torch

    from src.model.ego2s_network import load_ego2s_checkpoint, save_ego2s_checkpoint

    if args.out.exists():
        print(f"refusing: {args.out} exists (create-only)", file=sys.stderr)
        return 2
    blob = torch.load(args.src, map_location="cpu", weights_only=False)
    net = load_ego2s_checkpoint(str(args.src), "cpu")
    wide = widen_net(net)
    meta = {
        "widened_from": str(args.src),
        "widened_from_sha256": hashlib.sha256(args.src.read_bytes()).hexdigest(),
        "widened_by": "research/redesign_m4b_20261012/widen_bv8.py",
        "src_meta": blob.get("meta", {}),
    }
    save_ego2s_checkpoint(args.out, wide, meta)
    print(f"wrote {args.out} (ego2s-bv8, fuse input {N_LOCAL_FEATS}+60, new columns zero)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
