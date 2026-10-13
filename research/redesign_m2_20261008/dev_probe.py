#!/usr/bin/env python3
"""Development probe (NOT the pre-registered check): student vs frp3-s12, both no veto.

Same engine and profile as ``ni_check.py`` but on DISTILLATION-namespace worlds reserved
for probing (``ni_spec.distill_seeds(..., round_index=50)``; no data round uses index 50)
so the pre-registered NI worlds stay untouched. Development information only: it never
decides whether a pre-registered check runs, and is never reported as the check's result.

  nice -n 10 ./venv/bin/python research/redesign_m2_20261008/dev_probe.py \\
      --student S.pth --mix scripted --worlds 8 --arm student --out probe.jsonl

``--v61-backend numba`` (opt-in; default: env ``SNAKE_V61_BACKEND``, else ``python``) runs
the vector61 opponents' / carrier's featurizer compiled (``src/simd_env/vector61_nb.py``).
The records are identical to the Python featurizer's (whole record and per-frame world
digest, every mix, no-veto advisory student, student + v8 and the v8 incumbent:
``tests/test_eval_v61_backend_identity.py``); a non-default backend is written into the
row as ``v61_backend``.
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m2_20261008 import ni_spec  # noqa: E402
from research.redesign_scope_20261007 import grid_h5000_identity as gid  # noqa: E402

PROBE_ROUND_INDEX = 50
V61_BACKENDS = ("python", "numba")  # == src.simd_env.vector61_policy.V61_BACKENDS
V61_BACKEND_ENV = "SNAKE_V61_BACKEND"  # == src.simd_env.vector61_policy.V61_BACKEND_ENV


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--student", default=None)
    ap.add_argument("--arm", choices=("student", "baseline"), required=True)
    ap.add_argument("--mix", choices=ni_spec.MIXES, required=True)
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--round-index", type=int, default=PROBE_ROUND_INDEX, help="M2b: 51; M3: 52")
    ap.add_argument("--veto", choices=("none", "v8"), default="none")
    ap.add_argument(
        "--ego2s-forward", choices=("batched", "rowwise"), default="batched", help="M3-C: rowwise"
    )
    ap.add_argument(
        "--ego2s-mask",
        choices=("resolved", "advisory"),
        default="resolved",
        help="no-veto student only; M3-C: advisory (the live selection rule, as Phase R)",
    )
    ap.add_argument(
        "--frames", type=int, default=None, help="default: the NI horizon (smoke tests only)"
    )
    ap.add_argument(
        "--v61-backend",
        choices=V61_BACKENDS,
        default=None,
        help="vector61 featurizer (default: env SNAKE_V61_BACKEND, else python); numba is "
        "record-identical (tests/test_eval_v61_backend_identity.py)",
    )
    return ap


def effective_v61_backend(requested: Optional[str]) -> str:
    """The backend ``Vector61Runtime`` will use (flag, else env, else ``python``)."""
    backend = requested or os.environ.get(V61_BACKEND_ENV) or "python"
    if backend not in V61_BACKENDS:
        raise SystemExit(f"vector61 backend must be one of {V61_BACKENDS}, got {backend!r}")
    return backend


def probe_seeds(worlds: int, round_index: int) -> List[int]:
    from research.redesign_m2_20261008 import ni_spec_m2b

    seeds = ni_spec.distill_seeds(worlds, round_index=round_index)
    assert not set(seeds) & (set(ni_spec.ni_seeds()) | set(ni_spec_m2b.ni_seeds()))
    return seeds


def probe_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """The hero / veto keyword arguments ``run_simd_eval`` gets for ``args``."""
    kwargs: Dict[str, Any] = {"hero_ego2s": args.student} if args.arm == "student" else {}
    if args.arm == "student":
        kwargs["ego2s_forward"] = args.ego2s_forward
        if args.veto == "none":
            kwargs["ego2s_mask"] = args.ego2s_mask
    if args.veto == "v8":
        kwargs.update(hero_safety_veto="v8", hero_safety_veto_lambda=8.0)
    return kwargs


def probe(
    args: argparse.Namespace, ctx: Optional[Mapping[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run one probe (one mix); returns ``(records, row)`` (``row`` = the jsonl line)."""
    from src.simd_env import eval_engine as ee

    if ctx is None:
        ctx = gid._context(1)
    profile = ctx["profile"]
    if args.frames:
        import dataclasses

        profile = dataclasses.replace(profile, scored_horizon=args.frames)
    backend = effective_v61_backend(args.v61_backend)
    seeds = probe_seeds(args.worlds, args.round_index)
    rows = {
        int(r["world_seed"]): [ctx["lookup"][s["member_sha256"]] for s in r["slots"]]
        for r in ctx["ds"]._design_rows(seeds)
        if r["mix"] == args.mix
    }
    t0 = time.perf_counter()
    recs = ee.run_simd_eval(
        ctx["lookup"][ni_spec.FRP3_S12[1]],
        rows[seeds[0]],
        args.frames or ni_spec.HORIZON,
        seeds,
        profile=profile,
        opponent_specs_by_world=rows,
        mix_id=args.mix,
        vector61=True,
        vector61_forward="rowwise",
        sim_engine="grid",
        vector61_backend=backend,
        **probe_kwargs(args),
    )
    row = {
        "arm": args.arm,
        "veto": args.veto,
        "round_index": args.round_index,
        "frames": args.frames or ni_spec.HORIZON,
        "ego2s_forward": args.ego2s_forward if args.arm == "student" else None,
        "ego2s_mask": args.ego2s_mask if args.arm == "student" and args.veto == "none" else None,
        "student": args.student,
        "student_sha256": (
            __import__("hashlib").sha256(Path(args.student).read_bytes()).hexdigest()
            if args.student
            else None
        ),
        "mix": args.mix,
        "wall": time.perf_counter() - t0,
        "mass": {int(r["seed"]): r["mass_integral"] for r in recs},
        "deaths": {int(r["seed"]): r["deaths"] for r in recs},
        "causes": {int(r["seed"]): r.get("death_cause") for r in recs},
    }
    if backend != "python":  # default rows keep their pre-existing keys
        row["v61_backend"] = backend
    return recs, row


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    torch.set_num_threads(1)
    _, row = probe(args)
    with args.out.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    m = list(row["mass"].values())
    print(
        json.dumps({"arm": args.arm, "veto": args.veto, "mix": args.mix, "mean": sum(m) / len(m)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
