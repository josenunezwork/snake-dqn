#!/usr/bin/env python3
"""Dev probe (development information only) for an eval-only seed ensemble of ego2s students.

Runs ``research/redesign_m2_20261008/dev_probe.py``'s ``probe()`` unchanged (student arm,
``--ego2s-forward rowwise --ego2s-mask advisory``, no veto) with the hero policy swapped for
:class:`research.ensemble_20261014.ensemble_policy.EnsembleEgo2sPolicy`, one mix after the
other in one process. Each output row is a dev-probe row plus ``ensemble`` (rule, members,
sha256), ``peak`` (per-world max mass) and ``diag`` (last-decision length per world, decisions
at length >= 800, Q-calibration statistics).

  ./venv/bin/python research/ensemble_20261014/ensemble_probe.py run  # niced (--nice 10) \\
      --members 'ART/redesign-m4b-mac-20261013/B8_s*/final.pth' --rule mean \\
      --mixes frozen,scripted,mixed --v61-backend numba --anchors fast --out OUT.jsonl

  # identity: a 1-member ensemble must reproduce that member's existing probe rows
  ./venv/bin/python research/ensemble_20261014/ensemble_probe.py identity OUT.jsonl REF.jsonl
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Iterator, List, Optional, Sequence  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.ensemble_20261014.ensemble_policy import (  # noqa: E402
    RULES,
    EnsembleEgo2sPolicy,
    members_from_globs,
)
from research.redesign_m2_20261008 import dev_probe  # noqa: E402


@contextmanager
def ensemble_hero(members: Sequence[str], rule: str, holder: Dict[str, Any]) -> Iterator[None]:
    """Make ``run_simd_eval``'s no-veto ego2s hero the ensemble (restored on exit)."""
    from src.simd_env import ego2s_policy as ep

    orig = ep.Ego2sSimdPolicy

    def factory(checkpoint_path, device="cpu", forward="batched", mask_mode="resolved"):
        if forward != "rowwise" or mask_mode != "advisory":
            raise RuntimeError("the ensemble probe runs rowwise / advisory only")
        pol = EnsembleEgo2sPolicy(members, rule=rule, forward=forward, mask_mode=mask_mode)
        holder["policy"] = pol
        return pol

    ep.Ego2sSimdPolicy = factory
    try:
        yield
    finally:
        ep.Ego2sSimdPolicy = orig


def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd_run(args: argparse.Namespace) -> int:
    import torch

    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    if args.nice:
        os.nice(args.nice)
    torch.set_num_threads(1)
    members = members_from_globs(args.members)
    for m in members:
        if not Path(m).is_file():
            raise SystemExit(f"missing member {m}")
    shas = [sha(m) for m in members]
    ctx = gid._context(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for mix in args.mixes.split(","):
        argv = [
            "--student", members[0], "--arm", "student", "--mix", mix,
            "--worlds", str(args.worlds), "--out", str(args.out),
            "--round-index", str(args.round_index), "--ego2s-forward", "rowwise",
            "--ego2s-mask", "advisory", "--anchors", args.anchors,
        ]  # fmt: skip
        if args.v61_backend:
            argv += ["--v61-backend", args.v61_backend]
        if args.frames:
            argv += ["--frames", str(args.frames)]
        pargs = dev_probe.build_parser().parse_args(argv)
        holder: Dict[str, Any] = {}
        t0 = time.perf_counter()
        with ensemble_hero(members, args.rule, holder):
            recs, row = dev_probe.probe(pargs, ctx)
        seeds = dev_probe.probe_seeds(pargs.worlds, pargs.round_index)
        row["student"] = f"ensemble:{args.rule}:K{len(members)}"
        row["student_sha256"] = None
        row["ensemble"] = {"rule": args.rule, "members": members, "member_sha256": shas}
        row["peak"] = {int(r["seed"]): r["max_mass"] for r in recs}
        row["diag"] = holder["policy"].diagnostics(seeds)
        row["tag"] = args.tag
        with args.out.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        m = list(row["mass"].values())
        d = list(row["deaths"].values())
        print(
            json.dumps(
                {
                    "tag": args.tag,
                    "mix": mix,
                    "mass": sum(m) / len(m),
                    "survival": sum(1 for x in d if x == 0) / len(d),
                    "wall": round(time.perf_counter() - t0, 1),
                }
            ),
            flush=True,
        )
    return 0


def cmd_identity(args: argparse.Namespace) -> int:
    def load(p: Path) -> Dict[str, dict]:
        rows: Dict[str, dict] = {}
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["mix"]] = r
        return rows

    out, ref = load(args.out), load(args.ref)
    report: Dict[str, Any] = {"out": str(args.out), "ref": str(args.ref), "mixes": {}}
    ok_all = True
    for mix, r in out.items():
        if mix not in ref:
            report["mixes"][mix] = "no reference row"
            ok_all = False
            continue
        same = {k: r[k] == ref[mix][k] for k in ("mass", "deaths", "causes")}
        report["mixes"][mix] = same
        ok_all &= all(same.values())
    report["identical"] = ok_all
    print(json.dumps(report, indent=1))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=1) + "\n")
    return 0 if ok_all else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--members", nargs="+", required=True, help="checkpoint paths or globs")
    r.add_argument("--rule", choices=RULES, default="mean")
    r.add_argument("--mixes", default="frozen,scripted,mixed")
    r.add_argument("--worlds", type=int, default=32)
    r.add_argument("--round-index", type=int, default=52)
    r.add_argument("--v61-backend", choices=dev_probe.V61_BACKENDS, default=None)
    r.add_argument("--anchors", choices=dev_probe.ANCHORS, default="python")
    r.add_argument("--frames", type=int, default=None, help="smoke tests only")
    r.add_argument("--tag", default="")
    r.add_argument("--nice", type=int, default=10, help="os.nice increment (0: none)")
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(fn=cmd_run)
    i = sub.add_parser("identity")
    i.add_argument("out", type=Path)
    i.add_argument("ref", type=Path)
    i.add_argument("--json-out", type=Path, default=None)
    i.set_defaults(fn=cmd_identity)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())


__all__: List[str] = ["ensemble_hero", "main"]
