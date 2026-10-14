#!/usr/bin/env python3
"""Training-actor throughput benchmark for ``train_m3c.actor_main`` (perf-actor, 2026-10-13).

Runs the real ``actor_main`` in-process against a message sink (no learner, no queue): the
sink times every frame message after ``--warm`` frames and stops the actor after
``--frames`` timed frames. Each (config, worlds, device) cell runs in its own subprocess,
one at a time (one CPU process plus, on MPS, the GPU).

Configs (``train_m3c`` flags):

* ``baseline``   ``--actor-impl reference --v61-backend python`` (the pre-perf-actor actor)
* ``vectorized`` ``--actor-impl vectorized --v61-backend python``
* ``numba``      ``--actor-impl vectorized --v61-backend numba``

Reported per cell: actor frames/s, hero transitions/s (hero decisions; each becomes one
n-step transition) and the actor's own per-frame timing split (``msg["timing"]``,
``env_s``). ``--grow-to L`` raises every snake's target length at segment start (bodies
then grow one cell per frame), to look at long-body cost.

Usage (re-run on a quiet machine; numbers taken while other jobs run are noisy)::

    ./venv/bin/python research/perf_actor_20261013/bench_actor.py \\
        --worlds 64,128,256 --devices cpu,mps --frames 150 --warm 30 \\
        --out research/perf_actor_20261013/results/bench_actor.json

    # one cell under cProfile (top functions by own time)
    ./venv/bin/python research/perf_actor_20261013/bench_actor.py --profile \\
        --configs baseline --worlds 64 --devices cpu --frames 100

    # perf-opp: the opponent-forward modes on the M5 recipe (bv8 + labels, 25% mixed);
    # --reps 4 repeats every worlds' cells, odd reps in reversed order (A B, B A, ...)
    ./venv/bin/python research/perf_actor_20261013/bench_actor.py --configs numba \\
        --worlds 128,256 --devices mps --opp-forwards per-group,merged-cpu,merged-device \\
        --reps 4 \\
        --extra "--mixed-frac 0.25 --obs-version bv8 --onpolicy-labels --label-frac 1.0 \\
                 --anchor-mode v8margin --v8-backend numba"
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STUDENT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m2b-20261008/student_m2b_final.pth"
)
CONFIGS = {
    "baseline": ("reference", "python"),
    "vectorized": ("vectorized", "python"),
    "numba": ("vectorized", "numba"),
}


class _Stop:
    def __init__(self) -> None:
        self.flag = False

    def is_set(self) -> bool:
        return self.flag

    def set(self) -> None:
        self.flag = True


class _Sink:
    """Stands in for the actor's queue: times frames, stops the actor when done."""

    def __init__(self, warm: int, frames: int, stop: _Stop, prof=None) -> None:
        self.warm, self.frames, self.stop, self.prof = warm, frames, stop, prof
        self.n = 0
        self.t0 = None
        self.t1 = None
        self.decisions = 0
        self.transitions = 0
        self.env_s = 0.0
        self.timing: Dict[str, float] = {}

    def put(self, msg: Dict[str, Any], timeout: float = 0.0) -> None:
        self.n += 1
        if self.n == self.warm:
            if self.prof is not None:
                self.prof.enable()
            self.t0 = time.perf_counter()
        elif self.n > self.warm:
            self.decisions += int(msg["decisions"])
            self.transitions += int(len(msg["s"]))
            self.env_s += float(msg["env_s"])
            for k, v in msg["timing"].items():
                self.timing[k] = self.timing.get(k, 0.0) + float(v)
        if self.n >= self.warm + self.frames:
            self.t1 = time.perf_counter()
            if self.prof is not None:
                self.prof.disable()
            self.stop.set()


def default_init() -> str:
    """The M2b student if present, else a random ego2s-b net (same forward cost)."""
    if STUDENT.exists():
        return str(STUDENT)
    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint

    path = Path(tempfile.gettempdir()) / "bench_actor_random_ego2s_b.pth"
    if not path.exists():
        import torch

        torch.manual_seed(0)
        save_ego2s_checkpoint(path, Ego2sNet(7, 30))
    return str(path)


def run_cell(
    config: str,
    worlds: int,
    device: str,
    frames: int,
    warm: int,
    init: str,
    grow_to: int,
    frozen_frac: float,
    seed: int,
    profile: bool = False,
    opp_forward: str = "per-group",
    extra: str = "",
) -> Dict[str, Any]:
    """One in-process ``actor_main`` run; returns rates and the timing split."""
    from research.redesign_m3c_20261008 import train_m3c as T

    impl, v61 = CONFIGS[config]
    argv = [
        "--seed",
        str(seed),
        "--init",
        init,
        "--demo-data",
        ".",
        "--out",
        "unused",
        "--actors",
        "1",
        "--worlds",
        str(worlds),
        "--frozen-frac",
        str(frozen_frac),
        "--segment-frames",
        str(warm + frames + 10),
        "--device",
        device,
        "--actor-impl",
        impl,
        "--v61-backend",
        v61,
        "--opp-forward",
        opp_forward,
        *shlex.split(extra),
    ]
    args = T.build_parser().parse_args(argv)
    if grow_to:
        orig = T.build_segment

        def grown(*a, **k):
            out = orig(*a, **k)
            sim = out[0]
            rng = np.random.default_rng(seed)
            sim.length[:, :] = rng.integers(grow_to // 4, grow_to + 1, size=sim.length.shape)
            return out

        T.build_segment = grown
    nscal = T.nscal_for(args)
    margin = worlds * 16
    ring_cap = worlds * 64 + margin
    net = T.load_init(args, "cpu")
    spec = T._weights_spec(net)
    n_weights = int(sum(n for _, _, n in spec))
    blocks = [
        T.Block.create(T.ring_nbytes(ring_cap, nscal), None, "ring0"),
        T.Block.create(8 + 8 * (n_weights + 1), None, "weights"),
        T.Block.create(32, None, "progress"),
    ]
    names = {"ring0": blocks[0].desc, "weights": blocks[1].desc, "progress": blocks[2].desc}
    wver = np.ndarray((1,), dtype=np.int64, buffer=blocks[1].buf, offset=0)
    wver[0] = 0
    wbuf = np.ndarray((n_weights + 1,), dtype=np.float64, buffer=blocks[1].buf, offset=8)
    progress = np.ndarray((4,), dtype=np.int64, buffer=blocks[2].buf, offset=0)
    progress[:] = 0
    T._publish_weights(net, wbuf, wver)
    del wver, wbuf, progress
    cfg = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "K": 1,
        "ring_cap": ring_cap,
        "ring_margin": margin,
        "n_weights": n_weights,
        "actor_device": device,
    }
    prof = None
    if profile:
        import cProfile

        prof = cProfile.Profile()
    stop = _Stop()
    sink = _Sink(warm, frames, stop, prof)
    try:
        T.actor_main(0, cfg, names, sink, stop)
    finally:
        for b in blocks:
            b.unlink()
            b.close()
    wall = sink.t1 - sink.t0
    n = sink.frames
    row = {
        "config": config,
        "actor_impl": impl,
        "v61_backend": v61,
        "worlds": worlds,
        "device": device,
        "frames": n,
        "warm": warm,
        "grow_to": grow_to,
        "frozen_frac": frozen_frac,
        "opp_forward": opp_forward,
        "extra": extra,
        "wall_s": wall,
        "frames_per_s": n / wall,
        "ms_per_frame": 1e3 * wall / n,
        "hero_transitions_per_s": sink.decisions / wall,
        "emitted_transitions": sink.transitions,
        "env_ms_per_frame": 1e3 * sink.env_s / n,
        "timing_ms_per_frame": {k: 1e3 * v / n for k, v in sink.timing.items()},
    }
    if prof is not None:
        import io
        import pstats

        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(45)
        row["profile_tottime"] = s.getvalue()
    return row


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (config, worlds, device, opp_forward): median and all reps of ms/frame."""
    cells: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        if not r.get("error"):
            key = tuple(r.get(k) for k in ("config", "worlds", "device", "opp_forward"))
            cells.setdefault(key, []).append(r)
    out = []
    for (config, worlds, device, opp), rs in cells.items():
        ms = [r["ms_per_frame"] for r in rs]
        opp_ms = [r["timing_ms_per_frame"].get("opponents_s", float("nan")) for r in rs]
        out.append(
            {
                "config": config,
                "worlds": worlds,
                "device": device,
                "opp_forward": opp,
                "reps": len(rs),
                "ms_per_frame_median": float(np.median(ms)),
                "ms_per_frame": ms,
                "opponents_ms_median": float(np.median(opp_ms)),
                "opponents_ms": opp_ms,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", default="baseline,vectorized,numba")
    ap.add_argument("--worlds", default="64,128,256")
    ap.add_argument("--devices", default="cpu,mps")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--warm", type=int, default=30)
    ap.add_argument("--grow-to", type=int, default=0)
    ap.add_argument("--frozen-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--profile", action="store_true", help="cProfile one in-process cell")
    ap.add_argument("--opp-forwards", default="per-group", help="train_m3c --opp-forward values")
    ap.add_argument("--extra", default="", help="more train_m3c flags for every cell (one string)")
    ap.add_argument(
        "--reps", type=int, default=1, help="repeat each worlds' cells; odd reps run reversed"
    )
    ap.add_argument("--cell", default=None, help=argparse.SUPPRESS)  # internal: one cell
    a = ap.parse_args()
    init = a.init or default_init()
    if a.cell or a.profile:
        cell = (a.cell or f"{a.configs},{a.worlds},{a.devices},{a.opp_forwards}").split(",")
        config, worlds, device = cell[:3]
        row = run_cell(
            config,
            int(worlds),
            device,
            a.frames,
            a.warm,
            init,
            a.grow_to,
            a.frozen_frac,
            a.seed,
            profile=a.profile,
            opp_forward=cell[3] if len(cell) > 3 else "per-group",
            extra=a.extra,
        )
        if a.profile:
            print(row.pop("profile_tottime"))
        print("CELL " + json.dumps(row), flush=True)
        return 0
    rows: List[Dict[str, Any]] = []
    for device in a.devices.split(","):
        if device == "mps":
            import torch

            if not torch.backends.mps.is_available():
                print("mps unavailable: skipped", flush=True)
                continue
        for worlds in [int(w) for w in a.worlds.split(",")]:
            pairs = [(c, o) for c in a.configs.split(",") for o in a.opp_forwards.split(",")]
            for rep, (config, opp) in [
                (r, p) for r in range(a.reps) for p in (pairs[::-1] if r % 2 else pairs)
            ]:
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--cell",
                    f"{config},{worlds},{device},{opp}",
                    "--frames",
                    str(a.frames),
                    "--warm",
                    str(a.warm),
                    "--grow-to",
                    str(a.grow_to),
                    "--frozen-frac",
                    str(a.frozen_frac),
                    "--seed",
                    str(a.seed),
                    "--init",
                    init,
                    "--extra=" + a.extra,
                ]
                r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
                line = [x for x in r.stdout.splitlines() if x.startswith("CELL ")]
                if r.returncode or not line:
                    row = {"config": config, "worlds": worlds, "device": device, "error": True}
                    row["opp_forward"] = opp
                    row["stderr"] = r.stderr[-1500:]
                else:
                    row = json.loads(line[-1][5:])
                row["rep"] = rep
                rows.append(row)
                print(
                    json.dumps(
                        {
                            k: row.get(k)
                            for k in ("config", "worlds", "device", "opp_forward", "rep")
                        }
                    )
                    + f" -> {row.get('frames_per_s', float('nan')):.1f} frames/s, "
                    f"{row.get('hero_transitions_per_s', float('nan')):.0f} hero tr/s",
                    flush=True,
                )
    result = {
        "schema": "perf-actor-bench/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "machine": platform.machine(),
        "note": "single actor process, in-process sink (no learner); noisy if the host is busy",
        "init": init,
        "extra": a.extra,
        "reps": a.reps,
        "summary": summarize(rows),
        "rows": rows,
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
