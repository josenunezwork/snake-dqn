"""Opt-in ``torch.compile`` for the multi-actor ego2s trainer (perf-compile, 2026-10-13).

Nothing here runs unless ``--compile-learner`` / ``--compile-actor`` is set (default
``off``: the trainer never calls ``torch.compile``; ``tests/test_perf_compile.py`` checks).

* Learner (``--compile-learner MODE``, needs ``--learner-fast``): see
  :class:`research.redesign_m3c_20261008.learner_m3c.Learner`. Scope ``loss`` (default)
  compiles :func:`~research.redesign_m3c_20261008.learner_m3c.loss_terms` - the three
  forwards, the Double DQN target, the TD and anchor terms - so AOTAutograd also compiles
  its backward; scope ``forward`` compiles only the online / target network forwards.
* Actor (``--compile-actor MODE``): :class:`HeroForward`, the actor's hero Q forward.

**Shapes.** ``dynamic=False`` everywhere and every compiled call sees one of a few fixed
shapes, so there is no recompilation storm and ``reduce-overhead`` (CUDA graphs) can replay:

* the learner routes only *canonical* batches to the compiled function - all agent rows
  survived the stale-row re-check (``n == agent rows``) and the anchor block is either
  empty (bv8 before 10k labels) or full (``m == anchor rows``); anything else (a ring
  overrun, a dropped label row - 0 of 79k updates in the M5 L60 run) runs the SAME function
  eagerly. At most three shape keys exist per run.
* the actor pads the live hero rows (``<= --worlds``) to ``--worlds`` rows in a fixed
  staging block and keeps the first ``n`` Q rows. Ego2sNet rows are independent (per-row
  conv / LayerNorm), so a padding row never changes a real row's Q beyond kernel-level
  float reassociation.

Each learner / hero forward counts the graphs it compiled (:class:`GraphCount`; reported as
``graphs``, with dynamo's process-wide counters under ``dynamo_process``) and stops calling
the compiled code past ``GRAPH_LIMIT`` (a recompilation storm: eager fallback + message).

**Numerics.** Inductor fuses and reorders float ops: compiled Q equals eager Q only up to
float32 reassociation (``tests/test_perf_compile.py`` reports and bounds max |dQ| and the
argmax flip rate; compiled training stays within a tolerance of eager over N updates). It is
NOT bitwise.

**Workers.** The trainer calls :func:`limit_compile_workers` in each compiling process (the
actors compile in-process; the learner's Triton-compile pool is quiesced after compiling), so
K actors do not each keep a ``min(32, cores)``-process compile pool alive on a pod.

**Fallback.** If compiling or running the compiled code raises (e.g. an inductor backend
that does not support this device), the caller prints one clear message, records the reason
in :meth:`status` and continues eagerly; the failing batch is recomputed eagerly. The actor
also checks the compiled forward against eager on a probe batch first and falls back when
they disagree beyond ``PROBE_TOL``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

COMPILE_MODES = ("off", "default", "reduce-overhead", "max-autotune-no-cudagraphs")
LEARNER_SCOPES = ("loss", "forward")
#: probe check (actor): max |q_compiled - q_eager| <= PROBE_TOL * (1 + max |q_eager|)
PROBE_TOL = 1e-3
#: a learner / hero forward that compiles more graphs than this is in a recompilation storm:
#: it stops calling the compiled code and continues eagerly (expected: the learner <= 3 shape
#: keys, x 2 grad modes for scope ``forward``; the hero forward 1)
GRAPH_LIMIT = 8


def check_mode(mode: str) -> str:
    if mode not in COMPILE_MODES:
        raise ValueError(f"compile mode must be one of {COMPILE_MODES}, got {mode!r}")
    return mode


def uses_cudagraphs(mode: str, device) -> bool:
    return mode == "reduce-overhead" and str(device).startswith("cuda")


def dynamo_counters() -> Dict[str, int]:
    """Process-wide dynamo counters: compiled graphs, frames, graph breaks."""
    from torch._dynamo.utils import counters

    return {
        "graphs": int(counters["stats"]["unique_graphs"]),
        "frames_ok": int(counters["frames"]["ok"]),
        "frames_total": int(counters["frames"]["total"]),
        "graph_breaks": int(sum(counters["graph_break"].values())),
    }


def _graphs_and_breaks() -> Tuple[int, int]:
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"]), int(sum(counters["graph_break"].values()))


class GraphCount:
    """One caller's own compiled graphs / graph breaks (``with count: <compiled call>``).

    Dynamo's counters are process-wide (a bench runs many learners in one process), so this
    adds up their deltas around the caller's own compiled calls; calls are single-threaded,
    so a delta belongs to the call it brackets. ``storm``: more than ``GRAPH_LIMIT`` graphs.
    """

    def __init__(self) -> None:
        self.graphs = 0
        self.graph_breaks = 0
        self._at: Optional[Tuple[int, int]] = None

    def __enter__(self) -> "GraphCount":
        self._at = _graphs_and_breaks()
        return self

    def __exit__(self, *exc) -> bool:
        g, b = _graphs_and_breaks()
        self.graphs += g - self._at[0]
        self.graph_breaks += b - self._at[1]
        return False

    @property
    def storm(self) -> bool:
        return self.graphs > GRAPH_LIMIT

    def storm_error(self) -> RuntimeError:
        return RuntimeError(
            f"recompilation storm: {self.graphs} graphs compiled > GRAPH_LIMIT {GRAPH_LIMIT}"
        )

    def status(self) -> Dict[str, Any]:
        return {
            "graphs": self.graphs,
            "graph_breaks": self.graph_breaks,
            "recompile_storm": self.storm,
            "dynamo_process": dynamo_counters(),
        }


def compile_fn(torch, fn, mode: str):
    """``torch.compile(fn, mode=..., dynamic=False)`` (``fn``: a function or a module)."""
    check_mode(mode)
    if mode == "off":
        raise ValueError("compile_fn called with mode 'off'")
    if not hasattr(torch, "compile"):
        raise RuntimeError("this torch has no torch.compile")
    return torch.compile(fn, mode=mode, dynamic=False)


def limit_compile_workers(role: str) -> Dict[str, Any]:
    """Keep inductor's compile workers from multiplying across the trainer's processes.

    Every actor and the learner is its own process, and inductor's default is a pool of
    ``min(32, cores)`` Triton-compile worker processes per process (CUDA), kept alive after
    compiling: K actors x 32 idle workers on a pod. So actors (``role="actor"``: one small
    forward graph) compile in-process (``compile_threads = 1``) and the learner keeps the
    parallel pool but quiesces it after each compilation. An explicit
    ``TORCHINDUCTOR_COMPILE_THREADS`` / ``TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL`` wins.
    (On CPU / MPS inductor starts no worker pool either way; the effect is CUDA-only and
    untested here.) Returns the settings in force.
    """
    import torch._inductor.config as icfg

    if role not in ("actor", "learner"):
        raise ValueError(f"role must be actor or learner, got {role!r}")
    if role == "actor" and "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ:
        icfg.compile_threads = 1
    quiesce_supported = hasattr(icfg, "quiesce_async_compile_pool")  # not in torch 2.5.x
    if role == "learner" and "TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL" not in os.environ:
        if quiesce_supported:
            icfg.quiesce_async_compile_pool = True
        elif "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ:
            # No way to quiesce the pool after compiling: compile in-process instead, so K
            # learners on one pod do not each keep ~32 idle worker processes alive.
            icfg.compile_threads = 1
    return {
        "compile_threads": icfg.compile_threads,
        "quiesce_async_compile_pool": bool(getattr(icfg, "quiesce_async_compile_pool", False)),
        "quiesce_supported": quiesce_supported,
    }


def fallback_message(who: str, device, mode: str, exc: BaseException) -> str:
    dev = str(device)
    hint = ""
    if dev.startswith("mps"):
        hint = " (the MPS inductor backend is experimental in torch 2.x)"
    return (
        f"[perf-compile] {who}: torch.compile(mode={mode!r}) failed on {dev}{hint}: "
        f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''} "
        f"-> continuing EAGER (same math, no compile)"
    )


class _FlatStage:
    """One flat host block (pinned on CUDA) with typed numpy views, uploaded in one copy."""

    ALIGN = 64

    def __init__(self, torch, segs: Sequence[Tuple[str, tuple, Any]], pin: bool) -> None:
        self.layout = []
        off = 0
        for name, shape, dtype in segs:
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self.layout.append((name, tuple(shape), np.dtype(dtype), off, nbytes))
            off += -(-nbytes // self.ALIGN) * self.ALIGN
        if pin:
            self.host_t = torch.zeros(off, dtype=torch.uint8, pin_memory=True)
            self.flat = self.host_t.numpy()
        else:
            self.flat = np.zeros(off, dtype=np.uint8)
            self.host_t = torch.from_numpy(self.flat)
        self.v = {
            name: np.ndarray(shape, dtype=dtype, buffer=self.flat, offset=o)
            for name, shape, dtype, o, _ in self.layout
        }
        self.tdt = {
            np.dtype(np.uint8): torch.uint8,
            np.dtype(np.float32): torch.float32,
        }

    def upload(self, device, non_blocking: bool) -> Dict[str, Any]:
        dev = self.host_t.to(device, non_blocking=non_blocking)
        out = {}
        for name, shape, dtype, o, nbytes in self.layout:
            seg = dev[o : o + nbytes]
            if dtype != np.uint8:
                seg = seg.view(self.tdt[dtype])
            out[name] = seg.view(shape)
        return out


class HeroForward:
    """The actor's hero Q forward under ``torch.compile`` on a fixed ``rows`` batch.

    ``__call__(obs)`` takes the featurizer output for the ``n <= rows`` live hero rows and
    returns their ``(n, 6)`` Q on the device. The rows are copied into a fixed staging block
    (one host->device copy), the compiled net runs on all ``rows`` and the first ``n`` Q rows
    are returned: one static shape for the whole run. Call it under ``torch.no_grad()``.

    The network object is used in place: weight updates must be copied INTO its parameters
    (``load_state_dict`` on the device net), which the actor does when this class is on.
    """

    def __init__(self, torch, net, device, mode: str, rows: int, nscal: int, local, glob):
        check_mode(mode)
        self.torch, self.net, self.mode = torch, net, mode
        self.device = torch.device(device)
        self.rows = int(rows)
        pin = self.device.type == "cuda"
        self.non_blocking = pin
        self.stage = _FlatStage(
            torch,
            [
                ("local", (self.rows,) + tuple(local), np.uint8),
                ("global", (self.rows,) + tuple(glob), np.uint8),
                ("scalars", (self.rows, int(nscal)), np.float32),
            ],
            pin,
        )
        self.cudagraphs = uses_cudagraphs(mode, self.device)
        self.fallback: Optional[str] = None
        self.calls = 0
        self.compile_s = 0.0
        self.probe: Dict[str, float] = {}
        self.count = GraphCount()
        self.fn = compile_fn(torch, net, mode)
        self._probe()

    def _probe(self) -> None:
        """Compile now (warm-up) and check against eager on a random batch."""
        torch = self.torch
        r = np.random.default_rng(0)
        v = self.stage.v
        for k in ("local", "global"):
            v[k][:] = np.where(
                r.random(v[k].shape) < 0.15, r.integers(1, 256, v[k].shape, dtype=np.uint8), 0
            )
        v["scalars"][:] = r.standard_normal(v["scalars"].shape).astype(np.float32)
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                dv = self.stage.upload(self.device, self.non_blocking)
                ref = self.net(dv["local"], dv["global"], dv["scalars"]).float()
                with self.count:
                    if self.cudagraphs:
                        torch.compiler.cudagraph_mark_step_begin()
                    got = self.fn(dv["local"], dv["global"], dv["scalars"]).float()
                    for _ in range(2):  # CUDA graphs record on a later call: replay once too
                        if self.cudagraphs:
                            torch.compiler.cudagraph_mark_step_begin()
                        got = self.fn(dv["local"], dv["global"], dv["scalars"]).float().clone()
                d = float((got - ref).abs().max())
                scale = float(ref.abs().max())
                flips = float((got.argmax(1) != ref.argmax(1)).float().mean())
        except Exception as exc:  # noqa: BLE001
            self._fall_back(exc)
            return
        finally:
            self.compile_s = time.perf_counter() - t0
            v["local"][:] = 0
            v["global"][:] = 0
            v["scalars"][:] = 0
        self.probe = {"max_abs_diff": d, "q_scale": scale, "argmax_flip_rate": flips}
        if not d <= PROBE_TOL * (1.0 + scale):
            self._fall_back(RuntimeError(f"compiled Q differs from eager by {d:.3g}"))

    def _fall_back(self, exc: BaseException) -> None:
        self.fallback = (
            f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
        )
        print(fallback_message("actor hero forward", self.device, self.mode, exc), file=sys.stderr)
        self.fn = None

    def _eager(self, obs: Dict[str, np.ndarray]):
        return self.net(
            *[
                self.torch.from_numpy(np.ascontiguousarray(obs[k])).to(self.device)
                for k in ("local", "global", "scalars")
            ]
        )

    def __call__(self, obs: Dict[str, np.ndarray]):
        n = int(obs["local"].shape[0])
        if self.fn is None or n > self.rows:
            return self._eager(obs)
        v = self.stage.v
        v["local"][:n] = obs["local"]
        v["global"][:n] = obs["global"]
        v["scalars"][:n] = obs["scalars"]
        dv = self.stage.upload(self.device, self.non_blocking)
        try:
            with self.count:
                if self.cudagraphs:
                    self.torch.compiler.cudagraph_mark_step_begin()
                q = self.fn(dv["local"], dv["global"], dv["scalars"])[:n]
        except Exception as exc:  # noqa: BLE001
            self._fall_back(exc)
            return self._eager(obs)
        self.calls += 1
        if self.count.storm:  # (this call's result is fine; later calls run eagerly)
            self._fall_back(self.count.storm_error())
        return q.clone() if self.cudagraphs else q

    def status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "device": str(self.device),
            "rows": self.rows,
            "compiled": self.fn is not None,
            "fallback": self.fallback,
            "compile_s": round(self.compile_s, 3),
            "probe": self.probe,
            "calls": self.calls,
            **self.count.status(),
        }
