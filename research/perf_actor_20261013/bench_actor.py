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

perf-compile: ``--compile-actor MODE[,MODE]`` (``off`` = eager, ``default``,
``reduce-overhead``, ``max-autotune-no-cudagraphs``) runs every cell once per mode with
``train_m3c --compile-actor MODE`` (hero forward padded to ``--worlds`` rows). Compilation and
the eager probe happen before the first frame (outside the timed frames); the row carries
the actor's compile status (``compile_s``, probe max |dQ| / argmax flip rate, graphs,
``fallback``: a cell that fell back is timing eager code). The hero forward is
``timing_ms_per_frame["forward_s"]``.

Usage (re-run on a quiet machine; numbers taken while other jobs run are noisy)::

    ./venv/bin/python research/perf_actor_20261013/bench_actor.py \\
        --worlds 64,128,256 --devices cpu,mps --frames 150 --warm 30 \\
        --out research/perf_actor_20261013/results/bench_actor.json

    # CUDA pod: eager vs compiled hero forward (perf-compile)
    python research/perf_actor_20261013/bench_actor.py --configs numba --worlds 64,128,256 \\
        --devices cuda --compile-actor off,default,reduce-overhead --frames 300 --warm 30 \\
        --out bench_actor_cuda_compile.json

    # one cell under cProfile (top functions by own time)
    ./venv/bin/python research/perf_actor_20261013/bench_actor.py --profile \\
        --configs baseline --worlds 64 --devices cpu --frames 100

    # perf-profile: the B8 production actor (bv8 obs, v8 labels, mass reward, mixed worlds)
    # with long bodies, under pyinstrument, saved for research/perf_profile_20261013
    ./venv/bin/python research/perf_actor_20261013/bench_actor.py --profile --b8 \\
        --profiler pyinstrument --profile-dir /tmp/prof_long --configs numba --worlds 128 \\
        --devices mps --frames 150 --warm 400 --grow-to 1200

``--b8`` appends the B8 recipe's actor-side flags (:data:`B8_ARGS`); ``--train-args`` appends
any other ``train_m3c`` flags. Neither changes the default cells.
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STUDENT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m2b-20261008/student_m2b_final.pth"
)
#: The B8 recipe's actor-side ``train_m3c`` flags (``--b8``; the learner-only flags omitted).
B8_ARGS = (
    "--reward mass --mass-kappa 1.0 --kill-scale 0 --gamma-end 0.998 --gamma-hold 2000000 "
    "--gamma-anneal-end 8000000 --mixed-frac 0.25 --anchor-mode v8margin --anchor-hold 8000000 "
    "--anchor-floor-at 15000000 --onpolicy-labels --obs-version bv8 --label-frac 1.0"
)
PROFILERS = ("cprofile", "pyinstrument")
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
        self.session = None  # pyinstrument
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
                (self.prof.enable if hasattr(self.prof, "enable") else self.prof.start)()
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
                if hasattr(self.prof, "disable"):
                    self.prof.disable()
                else:
                    self.session = self.prof.stop()
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
    compile_actor: str = "off",
    train_args: str = "",
    profiler: str = "cprofile",
    profile_dir: Optional[Path] = None,
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
        "--compile-actor",
        compile_actor,
        *shlex.split(train_args),
    ]
    args = T.build_parser().parse_args(argv)
    built: list = []  # the segments' sims (read after the run: body sizes in the window)
    orig_build = T.build_segment

    def tracked(*a, **k):
        out = orig_build(*a, **k)
        sim = out[0]
        if grow_to:
            rng = np.random.default_rng(seed)
            sim.length[:, :] = rng.integers(grow_to // 4, grow_to + 1, size=sim.length.shape)
        built.append(sim)
        return out

    T.build_segment = tracked
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
    if profile and profiler == "cprofile":
        import cProfile

        prof = cProfile.Profile()
    elif profile:
        os.environ.setdefault("PYINSTRUMENT_IGNORE_OVERHEAD_WARNING", "1")
        from pyinstrument import Profiler

        prof = Profiler(interval=0.001, async_mode="disabled", use_timing_thread=True)
    stop = _Stop()
    sink = _Sink(warm, frames, stop, prof)
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):  # the actor prints its compile status
            T.actor_main(0, cfg, names, sink, stop)
    finally:
        T.build_segment = orig_build
        for b in blocks:
            b.unlink()
            b.close()
    wall = sink.t1 - sink.t0
    n = sink.frames
    segs = built[-1].seg_count[built[-1].alive] if built else np.zeros(0)
    body = {  # body segments of the live snakes at the end of the timed window
        "mean": float(segs.mean()) if len(segs) else 0.0,
        "p50": float(np.median(segs)) if len(segs) else 0.0,
        "max": int(segs.max()) if len(segs) else 0,
    }
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
        "train_args": train_args,
        "wall_s": wall,
        "frames_per_s": n / wall,
        "ms_per_frame": 1e3 * wall / n,
        "hero_transitions_per_s": sink.decisions / wall,
        "emitted_transitions": sink.transitions,
        "env_ms_per_frame": 1e3 * sink.env_s / n,
        "timing_ms_per_frame": {k: 1e3 * v / n for k, v in sink.timing.items()},
        "body_segments_end": body,
        "compile_actor": compile_actor,
    }
    for line in captured.getvalue().splitlines():
        if '"compile_actor"' in line:
            row["compile_status"] = json.loads(line)["compile_actor"]
    if prof is not None and profiler == "cprofile":
        import pstats

        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(45)
        row["profile_tottime"] = s.getvalue()
        if profile_dir:
            Path(profile_dir).mkdir(parents=True, exist_ok=True)
            prof.dump_stats(str(Path(profile_dir) / "actor0.pstats"))
    elif prof is not None:
        from pyinstrument.renderers import ConsoleRenderer

        row["profile_tottime"] = ConsoleRenderer(unicode=False, color=False).render(sink.session)
        if profile_dir:
            Path(profile_dir).mkdir(parents=True, exist_ok=True)
            from research.redesign_m3c_20261008.profile_m3c import save_session_compact

            save_session_compact(sink.session, Path(profile_dir) / "actor0.pyisession")
    if profile and profile_dir:
        meta = {
            "role": "actor0",
            "mode": profiler,
            "frames": n,
            "decisions": sink.decisions,
            "env_s": sink.env_s,
            "profiled_s": wall,
            "source": "bench_actor",
            **{k: v for k, v in row.items() if k not in ("profile_tottime",)},
            **sink.timing,
        }
        (Path(profile_dir) / "actor0.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return row


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
    ap.add_argument("--profile", action="store_true", help="profile one in-process cell")
    ap.add_argument("--profiler", choices=PROFILERS, default="cprofile")
    ap.add_argument("--profile-dir", type=Path, default=None, help="save the profile here")
    ap.add_argument("--b8", action="store_true", help="append the B8 actor flags (B8_ARGS)")
    ap.add_argument("--train-args", default="", help="extra train_m3c flags (one string)")
    ap.add_argument(
        "--compile-actor",
        default="off",
        help="perf-compile: comma list of hero-forward compile modes (off = eager)",
    )
    ap.add_argument("--note", default="", help="provenance note (e.g. quiet / noisy)")
    ap.add_argument("--cell", default=None, help=argparse.SUPPRESS)  # internal: one cell
    a = ap.parse_args()
    init = a.init or default_init()
    train_args = " ".join(x for x in (B8_ARGS if a.b8 else "", a.train_args) if x)
    if a.cell or a.profile:
        cmode0 = a.compile_actor.split(",")[0]  # --profile: one cell, the first mode
        cell = a.cell or f"{a.configs},{a.worlds},{a.devices},{cmode0}"
        config, worlds, device, cmode = cell.split(",")
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
            compile_actor=cmode,
            train_args=train_args,
            profiler=a.profiler,
            profile_dir=a.profile_dir,
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
        for worlds, config, cmode in [
            (int(w), c, m)
            for w in a.worlds.split(",")
            for c in a.configs.split(",")
            for m in a.compile_actor.split(",")
        ]:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--cell",
                f"{config},{worlds},{device},{cmode}",
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
                "--train-args=" + train_args,  # one token: the value may start with "--"
            ]
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            line = [x for x in r.stdout.splitlines() if x.startswith("CELL ")]
            if r.returncode or not line:
                row = {"config": config, "worlds": worlds, "device": device, "error": True}
                row["compile_actor"] = cmode
                row["stderr"] = r.stderr[-1500:]
            else:
                row = json.loads(line[-1][5:])
                if "[perf-compile]" in r.stderr:  # an eager fallback (message on stderr)
                    row["compile_stderr"] = r.stderr[-1500:]
            rows.append(row)
            print(
                json.dumps({k: row.get(k) for k in ("config", "worlds", "device", "compile_actor")})
                + f" -> {row.get('frames_per_s', float('nan')):.1f} frames/s, "
                f"{row.get('hero_transitions_per_s', float('nan')):.0f} hero tr/s",
                flush=True,
            )
    result = {
        "schema": "perf-actor-bench/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "machine": platform.machine(),
        "note": "single actor process, in-process sink (no learner); noisy if the host is busy"
        + (f"; {a.note}" if a.note else ""),
        "init": init,
        "rows": rows,
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
