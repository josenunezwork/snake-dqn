#!/usr/bin/env python3
"""Per-run RAM of ``train_m3c``'s multi-actor trainer: model, measurements, codec cost (perf-ram).

Subcommands (all write JSON with ``--out``; Mac numbers taken while other runs share the
machine are labelled ``noisy``):

* ``model``   analytic bytes per component for production configs (default / ``--obs-pack``
  / ``--obs-pack --demo-mmap`` / smaller ``--replay`` / ``--demo-capacity``), the per-run
  total and how many runs fit in 125 / 167 / 200 GB hosts. Buffer sizes come from the
  trainer's own definitions (``ring_capacity``, ``ring_nbytes``, ``demo_fields``, the
  ``Replay`` / ``LabelBuffer`` dtypes, ``demo_store.store_nbytes``); the non-buffer part
  (interpreter, torch / CUDA, numba, simulator) is ``--other-gb`` (default: derived from
  the RTX 4090 pod telemetry by ``podtel``).
* ``podtel``  reads pod-runner ``telemetry.jsonl`` files: per-job plateau RSS (the runner
  SUMS VmRSS over learner + actors, so the shared rings are counted twice), the derived
  non-buffer memory per run, and the cgroup headroom change when jobs exit (true per-run
  memory).
* ``trainer`` short real ``train_m3c`` runs (learner + actors, tiny capacities, scratch
  output, file-backed rings in the scratch directory as on the pods) per variant, sampling
  every process; reports the run's footprint next to the model's buffers for the same small
  config. Linux: sum of PSS (shared pages split between their mappers). macOS (psutil may
  not read another process's USS there): sum of ``footprint -p`` phys_footprint (private
  dirty memory; file-backed pages excluded) + the file-backed rings + the touched store
  pages. (POSIX shm would count in the footprint of every process that touched it.)
* ``codec``   ``obs_pack`` cost: pack us/row (actor side) and unpack-into-staging vs the
  raw fancy-index gather for a learner batch.
* ``actor-state`` in-process: one actor's simulator + vector61 runtime + opponents + hero
  featurizer memory for 16 / 64 / 128 worlds (USS delta and numpy bytes).
* ``verify-data`` ``obs_pack`` round trip on every demonstration row (all rounds).

Re-run on a quiet machine / a CUDA pod (``trainer --device cuda``; compare
``--actor-device cpu`` vs ``cuda`` for the actors' CUDA context)::

    ./venv/bin/python research/perf_ram_20261013/bench_ram.py model \\
        --out research/perf_ram_20261013/results/model.json
    ./venv/bin/python research/perf_ram_20261013/bench_ram.py podtel <pod telemetry.jsonl ...>
    ./venv/bin/python research/perf_ram_20261013/bench_ram.py trainer --init STUDENT.pth \\
        --demo-data DATA --scratch /tmp/perf_ram --device cpu --out trainer.json
    ./venv/bin/python research/perf_ram_20261013/bench_ram.py codec --demo-data DATA
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m3c_20261008 import obs_pack as op  # noqa: E402
from research.redesign_m3c_20261008 import train_m3c as m3c  # noqa: E402

GB = 1e9
HOSTS_GB = (125, 167, 200)
#: gpu_run_v2.RAM_RESERVE_GB: the pod runner keeps this much free
RESERVE_GB = 8.0
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
STUDENT = ARTIFACTS / "redesign-m2b-20261008/student_m2b_final.pth"
DEMO_DATA = ARTIFACTS / "redesign-m2b-20261008/data"
#: training (non-held-out) demonstration rows of rounds 10, 11, 12 (counted 2026-10-13:
#: 1,845,000 + 920,000 + 960,000 of 4,160,000); recounted by ``model`` when --demo-data exists
TRAIN_ROWS_DEFAULT = 3_725_000
HOLDOUT_ROWS_DEFAULT = 800
#: Ego2sNet parameters (ego2s-b 1,630,583; bv8 1,638,263)
PARAMS = {"b": 1_630_583, "bv8": 1_638_263}
#: Non-buffer memory of one run on the RTX 4090 pods (CUDA learner + 2 CUDA actors, 64
#: worlds: interpreter, torch / CUDA, numba, simulators), from ``podtel`` over the 2026-10-08
#: and 2026-10-12 pod telemetry:
#: * SUMMED: per-process VmRSS summed (what the runner's per-job ``rss_gb`` shows) minus the
#:   buffers and the double-counted rings: median 4.09 GB over 45 plateaued b-jobs (bv8:
#:   4.27, 2 jobs). Shared library pages (libtorch / CUDA) are counted once per process.
#: * MARGINAL: what one more run really costs the host: cgroup headroom freed when a b-job
#:   exits (median 16.17 GB over 13 exits) minus its 13.33 GB of buffers = 2.84, rounded up.
#:   Planning uses this one. (Mac, CPU learner + 1 CPU actor, 16 worlds: ~0.5 GB private;
#:   presumably the CUDA contexts of the pods' learner and both actors are most of the rest -
#:   unmeasured; ``trainer --device cuda --actor-device cpu|cuda`` on a pod would tell.)
OTHER_SUMMED_GB = 4.1
OTHER_MARGINAL_GB = 2.9

REPLAY_ROW = sum(np.dtype(d).itemsize for d in (np.int64, np.int64, np.float32, np.float32))
REPLAY_ROW += np.dtype(np.int64).itemsize  # s, a, R, disc, nxt = 32 B
LABEL_ROW = 2 * np.dtype(np.int64).itemsize  # ids, a = 16 B


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _host() -> Dict[str, Any]:
    return {"platform": platform.platform(), "python": platform.python_version()}


# ----------------------------------------------------------------------------- model
def count_rows(data: Path, rounds) -> Dict[str, int]:
    """Training and first-round-12-shard held-out rows (reads ``world_seed`` only)."""
    train = 0
    for r in rounds:
        for p in sorted((data / f"round-{r}").glob("chunk-*-part-*.npz")):
            with np.load(p) as z:
                train += int((z["world_seed"] % 10 != 0).sum())
    first = sorted((data / "round-12").glob("chunk-*-part-*.npz"))[0]
    with np.load(first) as z:
        held = int((z["world_seed"] % 10 == 0).sum())
    return {"train_rows": train, "holdout_rows": held}


def run_bytes(
    *,
    obs: str = "b",
    worlds: int = 64,
    actors: int = 2,
    replay: int = 500_000,
    demo_capacity: int = 300_000,
    label_capacity: int = 300_000,
    labels: bool = True,
    queue_frames: int = 8,
    pack: bool = False,
    demo_mmap: bool = False,
    train_rows: int = TRAIN_ROWS_DEFAULT,
    holdout_rows: int = HOLDOUT_ROWS_DEFAULT,
    other_gb: Optional[float] = None,
) -> Dict[str, Any]:
    """Bytes per component of one multi-actor run (and the host-shared demonstration store)."""
    layout = op.train_layout() if pack else None
    nscal = m3c.OBS_NSCAL[obs]
    cap, _ = m3c.ring_capacity(replay, actors, worlds, queue_frames)
    comp: Dict[str, float] = {}
    comp["rings"] = actors * m3c.ring_nbytes(cap, nscal, layout)
    comp["replay"] = replay * REPLAY_ROW
    has_demos = obs == "b"
    shared = 0
    if has_demos and demo_mmap:
        comp["demo_buffer"] = demo_capacity * np.dtype(np.int64).itemsize  # slot -> row index
        from research.redesign_m3c_20261008.demo_store import store_nbytes

        shared = store_nbytes(train_rows, layout)
    elif has_demos:
        comp["demo_buffer"] = demo_capacity * m3c.demo_row_nbytes(layout)
    else:
        comp["demo_buffer"] = 0
    # loader transient: one decompressed shard (~9k training rows of ~10k) plus the keep-mask
    # copy np.load makes (raw planes), plus its packed form with --obs-pack; none with a store
    shard = 10_000 * m3c.demo_row_nbytes(None)
    comp["demo_loader_peak"] = (
        0 if (not has_demos or demo_mmap) else 2 * shard + (shard // 4 if pack else 0)
    )
    comp["labels"] = label_capacity * LABEL_ROW if (labels or obs == "bv8") else 0
    comp["holdout"] = holdout_rows * m3c.demo_row_nbytes(None)
    params = PARAMS[obs]
    # learner: online + target + grads + Adam m, v (fp32); actors: one copy each
    comp["models"] = 4 * params * (5 + actors)
    buffers = sum(comp.values())
    other = (OTHER_MARGINAL_GB if other_gb is None else other_gb) * GB
    total = buffers + other
    # what the pod runner's per-job rss_gb shows: VmRSS summed over the processes, so the
    # rings appear in the learner's RSS (pages it read) AND in each actor's (pages it wrote),
    # and the shared libraries in every process
    summed_rss = buffers + comp["rings"] + OTHER_SUMMED_GB * GB
    return {
        "obs": obs,
        "worlds": worlds,
        "actors": actors,
        "replay": replay,
        "demo_capacity": demo_capacity if has_demos else 0,
        "pack": pack,
        "demo_mmap": demo_mmap,
        "ring_rows_per_actor": cap,
        "ring_row_bytes": m3c.ring_nbytes(1, nscal, layout) - 8,
        "demo_row_bytes": m3c.demo_row_nbytes(layout) if has_demos else 0,
        "components_gb": {k: round(v / GB, 4) for k, v in comp.items()},
        "other_gb": round(other / GB, 3),
        "per_run_gb": round(total / GB, 3),
        "per_run_summed_rss_gb": round(summed_rss / GB, 3),
        "host_shared_gb": round(shared / GB, 3),
    }


def runs_per_host(per_run_gb: float, shared_gb: float = 0.0) -> Dict[str, int]:
    out = {}
    for h in HOSTS_GB:
        out[str(h)] = max(0, int((h - RESERVE_GB - shared_gb) // per_run_gb))
    return out


MODEL_CONFIGS: List[Dict[str, Any]] = [
    {"name": "b default", "obs": "b"},
    {"name": "b --obs-pack", "obs": "b", "pack": True},
    {"name": "b --obs-pack --demo-mmap", "obs": "b", "pack": True, "demo_mmap": True},
    {"name": "b --demo-mmap (raw store)", "obs": "b", "demo_mmap": True},
    {"name": "b --replay 250k [semantics]", "obs": "b", "replay": 250_000},
    {"name": "b --demo-capacity 150k [semantics]", "obs": "b", "demo_capacity": 150_000},
    {
        "name": "b --replay 250k --demo-capacity 150k [semantics]",
        "obs": "b",
        "replay": 250_000,
        "demo_capacity": 150_000,
    },
    {"name": "b --obs-pack --replay 250k [semantics]", "obs": "b", "pack": True, "replay": 250_000},
    {
        "name": "b --obs-pack --demo-capacity 150k [semantics]",
        "obs": "b",
        "pack": True,
        "demo_capacity": 150_000,
    },
    {"name": "bv8 default", "obs": "bv8"},
    {"name": "bv8 --obs-pack", "obs": "bv8", "pack": True},
    {"name": "bv8 --replay 250k [semantics]", "obs": "bv8", "replay": 250_000},
    {
        "name": "bv8 --obs-pack --replay 250k [semantics]",
        "obs": "bv8",
        "pack": True,
        "replay": 250_000,
    },
]


def cmd_model(a) -> Dict[str, Any]:
    rows = {"train_rows": TRAIN_ROWS_DEFAULT, "holdout_rows": HOLDOUT_ROWS_DEFAULT}
    if a.demo_data and Path(a.demo_data).exists():
        rows = count_rows(Path(a.demo_data), [10, 11, 12])
    out = []
    for c in MODEL_CONFIGS:
        kw = {k: v for k, v in c.items() if k != "name"}
        kw.setdefault("worlds", a.worlds)
        r = run_bytes(**kw, **rows, other_gb=a.other_gb)
        r["name"] = c["name"]
        r["runs_per_host"] = runs_per_host(r["per_run_gb"], r["host_shared_gb"])
        r["runs_per_host_if_store_evicted"] = runs_per_host(r["per_run_gb"])
        out.append(r)
    print(
        f"{'config':52s} {'rings':>6s} {'demo':>6s} {'other':>6s} {'run':>6s} "
        f"{'sumRSS':>7s} {'shared':>7s}  runs@125/167/200 GB"
    )
    for r in out:
        c = r["components_gb"]
        print(
            f"{r['name']:52s} {c['rings']:6.2f} {c['demo_buffer']:6.2f} {r['other_gb']:6.2f} "
            f"{r['per_run_gb']:6.2f} {r['per_run_summed_rss_gb']:7.2f} {r['host_shared_gb']:7.2f}"
            f"  {'/'.join(str(v) for v in r['runs_per_host'].values())}"
        )
    return {
        "kind": "perf-ram model",
        "created_utc": _now(),
        "rows": rows,
        "worlds": a.worlds,
        "reserve_gb": RESERVE_GB,
        "hosts_gb": list(HOSTS_GB),
        "other_gb_source": (
            "argument" if a.other_gb is not None else "OTHER_MARGINAL_GB (podtel cgroup exits)"
        ),
        "other_summed_gb": OTHER_SUMMED_GB,
        "configs": out,
    }


# ----------------------------------------------------------------------------- podtel
def cmd_podtel(a) -> Dict[str, Any]:
    """Per-job plateau summed RSS -> non-buffer memory; cgroup headroom deltas on job exit."""
    res: Dict[str, Any] = {"kind": "perf-ram podtel", "created_utc": _now(), "files": []}
    for f in a.files:
        rows = [json.loads(line) for line in open(f)]
        plateau: Dict[str, float] = {}
        for r in rows:
            for k, v in (r.get("jobs") or {}).items():
                if v.get("rss_gb"):
                    plateau[k] = max(plateau.get(k, 0.0), float(v["rss_gb"]))
        jobs = {}
        for k, rss in sorted(plateau.items()):
            obs = "bv8" if k.startswith("B8") else "b"
            model = run_bytes(obs=obs, labels=True, other_gb=0.0)
            buffers = model["per_run_gb"]  # rings + demo + small buffers, no "other"
            rings = model["components_gb"]["rings"]
            jobs[k] = {
                "obs": obs,
                "summed_rss_gb": round(rss, 2),
                # rings full in the learner AND the actors (else the run never plateaued)
                "plateau": bool(rss > 2 * rings),
                "other_summed_gb": round(rss - buffers - rings, 2),
                "buffers_gb": round(buffers, 2),
            }
        exits = []
        prev = None
        for r in rows:
            live = {k for k, v in (r.get("jobs") or {}).items() if v.get("rss_gb")}
            head = r.get("mem_headroom_gb")
            if prev is not None and head is not None and prev[1] is not None:
                gone = prev[0] - live
                if gone and live <= prev[0]:
                    exits.append(
                        {
                            "jobs_exited": sorted(gone),
                            "headroom_gain_gb": round(head - prev[1], 2),
                            "per_job_gb": round((head - prev[1]) / len(gone), 2),
                        }
                    )
            prev = (live, head)
        res["files"].append({"file": str(f), "jobs": jobs, "cgroup_exits": exits})
        print(f"== {f}")
        for k, j in jobs.items():
            print(f"  {k:8s} {j}")
        for e in exits:
            print(f"  exit {e}")
    allj = [j for f in res["files"] for j in f["jobs"].values() if j["plateau"]]
    b_jobs = [j for j in allj if j["obs"] == "b"]
    v8_jobs = [j for j in allj if j["obs"] == "bv8"]
    exits = [e["per_job_gb"] for f in res["files"] for e in f["cgroup_exits"]]
    if b_jobs:
        b_buf = float(np.median([j["buffers_gb"] for j in b_jobs]))
        res["summary"] = {
            "plateau_jobs": len(allj),
            "b_jobs": len(b_jobs),
            "other_summed_gb_median_b": float(np.median([j["other_summed_gb"] for j in b_jobs])),
            "bv8_jobs": len(v8_jobs),
            "other_summed_gb_median_bv8": (
                float(np.median([j["other_summed_gb"] for j in v8_jobs])) if v8_jobs else None
            ),
            "cgroup_exits": len(exits),
            "cgroup_per_job_gb_median": float(np.median(exits)) if exits else None,
            "b_buffers_gb": b_buf,
            "other_marginal_gb": float(np.median(exits)) - b_buf if exits else None,
        }
        print("summary", json.dumps(res["summary"]))
    return res


# ----------------------------------------------------------------------------- trainer
VARIANTS = {
    "default": [],
    "pack": ["--obs-pack"],
    "pack_mmap": ["--obs-pack", "--demo-mmap", "{store}"],
}


def _tree(pid: int):
    import psutil

    try:
        p = psutil.Process(pid)
        return [p] + p.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}


def _phys_footprint(pid: int) -> Optional[float]:
    """macOS ``footprint -p PID`` phys_footprint in GB (None if unavailable)."""
    import re

    try:
        out = subprocess.run(
            ["footprint", "-p", str(pid)], capture_output=True, text=True, timeout=20
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"phys_footprint:\s+([\d.]+)\s*([KMG]?B)", out)
    return float(m.group(1)) * _UNITS[m.group(2)] / GB if m else None


def _mem(p) -> Optional[Dict[str, Any]]:
    import psutil

    try:
        name = " ".join(p.cmdline()[-2:])[-60:]
        d: Dict[str, Any] = {"rss": p.memory_info().rss / GB, "cmd": name}
        try:
            fi = p.memory_full_info()  # Linux: uss + pss; macOS: AccessDenied for other pids
            d["uss"] = fi.uss / GB
            if hasattr(fi, "pss"):
                d["pss"] = fi.pss / GB
        except psutil.AccessDenied:
            pass
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None
    if sys.platform == "darwin":
        d["phys_footprint"] = _phys_footprint(p.pid)
    return d


def _small_model(a, variant: str, store_rows: int) -> Dict[str, Any]:
    rows = {"train_rows": store_rows, "holdout_rows": HOLDOUT_ROWS_DEFAULT}
    return run_bytes(
        obs="b",
        worlds=a.worlds,
        actors=a.actors,
        replay=a.replay,
        demo_capacity=a.demo_capacity,
        labels=False,
        pack=variant != "default",
        demo_mmap=variant == "pack_mmap",
        other_gb=0.0,
        **rows,
    )


def cmd_trainer(a) -> Dict[str, Any]:
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    store = scratch / f"demo_store_r{a.demo_rounds.replace(',', '_')}_packed.bin"
    env = dict(os.environ)
    env.setdefault("NUMBA_CACHE_DIR", str(scratch / "nbcache"))
    store_rows = 0
    if "pack_mmap" in a.variants.split(","):  # build the store outside the measured run
        from research.redesign_m3c_20261008.demo_store import SharedDemoStore

        t = time.time()
        rounds = [int(r) for r in a.demo_rounds.split(",")]
        s = SharedDemoStore.open_or_build(store, a.demo_data, rounds, op.train_layout())
        store_rows, store_gb, build_s = s.rows, s.nbytes / GB, time.time() - t
        s.close()
        print(
            f"demo store {store}: {store_rows} rows, {store_gb:.2f} GB ({build_s:.0f} s)",
            flush=True,
        )
    res: Dict[str, Any] = {
        "kind": "perf-ram trainer",
        "created_utc": _now(),
        "host": _host(),
        "noisy": a.noisy,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "variants": {},
        "store": {"path": str(store), "rows": store_rows} if store_rows else None,
    }
    for variant in a.variants.split(","):
        out = scratch / f"run_{variant}_{int(time.time())}"
        opts = {
            "--seed": 0,
            "--init": a.init,
            "--demo-data": a.demo_data,
            "--demo-rounds": a.demo_rounds,
            "--out": out,
            "--actors": a.actors,
            "--worlds": a.worlds,
            "--replay": a.replay,
            "--demo-capacity": a.demo_capacity,
            "--demo-initial-shards": a.demo_initial_shards,
            "--transitions": a.transitions,
            "--warmup": a.warmup,
            "--checkpoint-every": 10**12,
            "--device": a.device,
            "--v61-backend": "numba",
        }
        argv = [sys.executable, str(REPO / "research/redesign_m3c_20261008/train_m3c.py")]
        for k, v in opts.items():
            argv += [k, str(v)]
        if a.actor_device:
            argv += ["--actor-device", a.actor_device]
        ring_dir = scratch / "rings"  # file-backed rings, as the pods run them (--ring-dir)
        ring_dir.mkdir(exist_ok=True)
        argv += ["--ring-dir", str(ring_dir)]
        argv += [x.format(store=store) for x in VARIANTS[variant]]
        t0 = time.time()
        log_path = scratch / f"{out.name}.log"
        with log_path.open("w") as fh:  # a file, not a pipe: a full pipe would stall the run
            proc = subprocess.Popen(argv, env=env, stdout=fh, stderr=subprocess.STDOUT)
        samples = []
        peak: Dict[str, Dict[str, float]] = {}
        keys = ("rss", "uss", "pss", "phys_footprint")
        while proc.poll() is None:
            procs = _tree(proc.pid)
            snap = []
            for p in procs:
                m = _mem(p)
                if m is None or "resource_tracker" in m["cmd"]:
                    continue
                role = "learner" if p.pid == proc.pid else f"child{p.pid}"
                m["role"] = role
                snap.append(m)
                pk = peak.setdefault(role, {k: 0.0 for k in keys})
                for k in keys:
                    pk[k] = max(pk[k], m.get(k) or 0.0)
            if snap:
                samples.append({"t": round(time.time() - t0, 1), "procs": snap})
            time.sleep(a.sample_every)
        log = log_path.read_text(errors="replace")
        model = _small_model(a, variant, store_rows)
        rings_gb = model["components_gb"]["rings"]
        # --demo-mmap: the store pages behind the buffer's slots are resident file cache
        # (shared by every run on the host; counted once here)
        store_touched = (
            min(a.demo_capacity, store_rows) * m3c.demo_row_nbytes(op.train_layout()) / GB
            if variant == "pack_mmap"
            else 0.0
        )
        # plateau: the last full sample (learner + every actor alive)
        full = [s for s in samples if len(s["procs"]) >= 1 + a.actors]
        last = full[-1]["procs"] if full else []
        on_linux = sys.platform.startswith("linux")
        if on_linux:  # PSS splits the file-backed rings / store pages between their mappers
            footprint = sum(p.get("pss", 0.0) for p in last)
            rule = "sum PSS"
        else:  # phys_footprint excludes file-backed pages: add the rings and the store pages
            footprint = sum(p.get("phys_footprint") or 0.0 for p in last)
            footprint += rings_gb + store_touched
            rule = "sum phys_footprint + file-backed rings + touched store pages"
        # steady buffers: the model's (other_gb=0) minus the demo loader's transient shard
        # (no shard rotation in a run shorter than one target period)
        buffers_gb = (
            model["per_run_gb"] + store_touched - model["components_gb"]["demo_loader_peak"]
        )
        res["variants"][variant] = {
            "argv": argv[2:],
            "returncode": proc.returncode,
            "wall_s": round(time.time() - t0, 1),
            "tail": log.strip().splitlines()[-3:],
            "model_small": model,
            "store_touched_gb": round(store_touched, 3),
            "plateau": last,
            "peak": peak,
            "footprint_gb": round(footprint, 3),
            "footprint_rule": rule,
            "summed_rss_gb": round(sum(p["rss"] for p in last), 3),
            "model_buffers_steady_gb": round(buffers_gb, 3),
            "non_buffer_gb": round(footprint - buffers_gb, 3),
            "samples": samples[:: max(1, len(samples) // 60)],
        }
        print(
            f"{variant:10s} rc={proc.returncode} footprint {footprint:.3f} GB "
            f"(model buffers {buffers_gb:.3f} GB -> non-buffer {footprint - buffers_gb:.3f} GB); "
            f"summed RSS {res['variants'][variant]['summed_rss_gb']:.3f} GB",
            flush=True,
        )
        if not a.keep:
            shutil.rmtree(out, ignore_errors=True)
    res["summary"] = trainer_summary(res)
    return res


def trainer_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    """Measured footprint differences between variants vs the model's buffer differences."""
    v = res["variants"]
    out: Dict[str, Any] = {}
    for a_, b_ in (("default", "pack"), ("pack", "pack_mmap"), ("default", "pack_mmap")):
        if a_ in v and b_ in v:
            out[f"{a_}->{b_}"] = {
                "measured_saving_gb": round(v[a_]["footprint_gb"] - v[b_]["footprint_gb"], 3),
                "model_saving_gb": round(
                    v[a_]["model_buffers_steady_gb"] - v[b_]["model_buffers_steady_gb"], 3
                ),
            }
    out["non_buffer_gb"] = {k: x["non_buffer_gb"] for k, x in v.items()}
    return out


# ----------------------------------------------------------------------------- codec
def cmd_codec(a) -> Dict[str, Any]:
    layout = op.layout_for_arena(145, 83, 6)
    if a.demo_data and Path(a.demo_data).exists():
        p = sorted((Path(a.demo_data) / "round-12").glob("chunk-*-part-*.npz"))[0]
        with np.load(p) as z:
            L, G = np.ascontiguousarray(z["local"]), np.ascontiguousarray(z["global"])
        source = str(p)
    else:
        sys.path.insert(0, str(REPO / "tests"))
        from test_obs_pack import valid_rows  # noqa: E402

        L, G = valid_rows(np.random.default_rng(0), 4000)
        source = "synthetic valid rows"
    op.pack(L[:8], G[:8], layout)  # compile / load the cached kernels
    P = op.pack(L, G, layout)
    n = a.batch_rows
    oL = np.empty((n,) + op.LOCAL_SHAPE, np.uint8)
    oG = np.empty((n,) + op.GLOBAL_SHAPE, np.uint8)
    dst = np.arange(n)
    rounds = []
    for rep in range(a.rounds):
        t = time.perf_counter()
        op.pack(L, G, layout)
        t_pack = time.perf_counter() - t
        rows = np.random.default_rng(rep).integers(0, len(P), n)
        t = time.perf_counter()
        for _ in range(a.reps):
            op.unpack_into(P, rows, oL, oG, dst, layout)
        t_unpack = (time.perf_counter() - t) / a.reps
        t = time.perf_counter()
        for _ in range(a.reps):
            oL[...] = L[rows]
            oG[...] = G[rows]
        t_raw = (time.perf_counter() - t) / a.reps
        rounds.append(
            {
                "pack_us_per_row": t_pack / len(L) * 1e6,
                "unpack_ms_per_batch": t_unpack * 1e3,
                "raw_gather_ms_per_batch": t_raw * 1e3,
            }
        )
    med = {k: float(np.median([r[k] for r in rounds])) for k in rounds[0]}
    print(json.dumps({"source": source, "rows": len(L), "batch_rows": n, **med}))
    return {
        "kind": "perf-ram codec",
        "created_utc": _now(),
        "host": _host(),
        "noisy": a.noisy,
        "source": source,
        "rows": len(L),
        "batch_rows": n,
        "packed_bytes_per_row": layout.nbytes,
        "raw_bytes_per_row": op.RAW_OBS_BYTES,
        "median": med,
        "rounds": rounds,
    }


# ----------------------------------------------------------------------------- verify-data
def cmd_verify_data(a) -> Dict[str, Any]:
    """Pack -> unpack round trip on EVERY demonstration row (held-out rows too)."""
    layout = op.layout_for_arena(145, 83, 6)
    t0 = time.time()
    n = shards = heads_max = 0
    rects = set()
    for r in [int(x) for x in a.rounds.split(",")]:
        for path in sorted((Path(a.demo_data) / f"round-{r}").glob("chunk-*-part-*.npz")):
            with np.load(path) as z:
                L, G = z["local"], z["global"]
            P = op.pack(L, G, layout)  # raises ValueError on any invariant violation
            L2, G2 = op.unpack(P, np.arange(len(P)), layout)
            if not (np.array_equal(L2, L) and np.array_equal(G2, G)):
                raise SystemExit(f"{path}: round trip differs")
            heads = (L[:, 2] != 0).reshape(len(L), -1).sum(axis=1)
            heads_max = max(heads_max, int(heads.max()))
            g = P[:, layout.off_g : layout.off_g + 5]
            rects.update(map(tuple, np.unique(g[:, 3:5], axis=0).tolist()))
            n += len(L)
            shards += 1
    res = {
        "kind": "perf-ram verify-data",
        "created_utc": _now(),
        "rows": n,
        "shards": shards,
        "byte_identical": True,
        "seconds": round(time.time() - t0, 1),
        "max_enemy_head_pixels": heads_max,
        "head_cap": layout.head_cap,
        "global_rect_sizes_seen": sorted(rects),
    }
    print(json.dumps(res))
    return res


# ----------------------------------------------------------------------------- actor state
def _ndarray_bytes(obj, seen=None, depth=0) -> int:
    """numpy bytes reachable from ``obj`` (attributes / dicts / lists, a few levels deep)."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 4 or isinstance(obj, (types.ModuleType, type)):
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes) if obj.base is None else 0
    if isinstance(obj, dict):
        items = list(obj.values())
    elif isinstance(obj, (list, tuple)):
        items = list(obj)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        items = list(vars(obj).values())
    else:
        return 0
    return sum(_ndarray_bytes(x, seen, depth + 1) for x in items)


def cmd_actor_state(a) -> Dict[str, Any]:
    """In-process: one actor's simulator + featurizer + opponent state for ``--worlds``
    worlds (``build_segment`` as the actor calls it, random legal hero actions, the ego2s-b
    hero rows built every frame, an Ego2sNet hero forward). USS / RSS deltas of this
    process, plus the numpy bytes held by the simulator / runtime / policies."""
    import psutil
    import torch

    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.model.ego2s_network import Ego2sNet
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    torch.set_num_threads(1)
    me = psutil.Process()

    def mem() -> Dict[str, float]:
        fi = me.memory_full_info()
        return {"rss": fi.rss / GB, "uss": getattr(fi, "uss", fi.rss) / GB}

    out: Dict[str, Any] = {"kind": "perf-ram actor-state", "created_utc": _now(), "rows": []}
    out["host"], out["noisy"] = _host(), a.noisy
    net = Ego2sNet(7, 30).eval()
    out["baseline"] = mem()  # interpreter + torch + numpy + one network
    ctx = gid._context(1)
    out["after_context"] = mem()
    B = EgoRasterConfig(version="b")
    rng = np.random.default_rng(0)
    for w in [int(x) for x in a.worlds.split(",")]:
        argv = ["--seed", "0", "--init", "x", "--demo-data", ".", "--out", "/nonexistent"]
        argv += ["--worlds", str(w), "--v61-backend", "numba", "--mixed-frac", "0.25"]
        args = m3c.build_parser().parse_args(argv)
        before = mem()
        sim, runtime, policies, groups = m3c.build_segment(args, ctx, 0, 0.99)

        def choose(prepared):
            masks = prepared.get_resolved_action_mask()
            alive = prepared.get_alive()
            actions = np.ones((sim.E, prepared.S), dtype=np.int64)
            runtime.prepare(prepared)
            hero = np.argwhere(alive[:, :1])
            if len(hero):
                obs = build_ego_raster(prepared, B, backend="numba", rows=hero)
                with torch.no_grad():
                    net(*[torch.from_numpy(obs[k]) for k in ("local", "global", "scalars")])
                actions[hero[:, 0], 0] = m3c.sample_legal(rng, masks[hero[:, 0], 0])
            for spec_, cells in groups.items():
                live = cells[alive[cells[:, 0], cells[:, 1]]]
                if len(live):
                    mk = masks[live[:, 0], live[:, 1]]
                    actions[live[:, 0], live[:, 1]] = policies[spec_].actions(mk, prepared, live)
            return actions

        for _ in range(a.frames):
            sim.step_with_policy(choose)
            runtime.observe_step(sim)
        after = mem()
        row = {
            "worlds": w,
            "frames": a.frames,
            "delta_uss_gb": round(after["uss"] - before["uss"], 4),
            "delta_rss_gb": round(after["rss"] - before["rss"], 4),
            "numpy_gb": {
                "sim": round(_ndarray_bytes(sim) / GB, 4),
                "runtime": round(_ndarray_bytes(runtime) / GB, 4),
                "policies": round(_ndarray_bytes(policies) / GB, 4),
            },
            "process_after": after,
        }
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
        sim = runtime = policies = groups = None  # free before the next world count
        gc.collect()
    return out


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model")
    m.add_argument("--worlds", type=int, default=64)
    m.add_argument("--other-gb", type=float, default=None, help="default: OTHER_MARGINAL_GB")
    m.add_argument("--demo-data", default=str(DEMO_DATA))
    p = sub.add_parser("podtel")
    p.add_argument("files", nargs="+")
    t = sub.add_parser("trainer")
    t.add_argument("--init", type=Path, default=STUDENT)
    t.add_argument("--demo-data", type=Path, default=DEMO_DATA)
    t.add_argument("--demo-rounds", default="12")
    t.add_argument("--scratch", required=True, help="run outputs + the demo store (deleted runs)")
    t.add_argument("--variants", default="default,pack,pack_mmap")
    t.add_argument("--actors", type=int, default=1)
    t.add_argument("--worlds", type=int, default=16)
    t.add_argument("--replay", type=int, default=40_000)
    t.add_argument("--demo-capacity", type=int, default=40_000)
    t.add_argument("--demo-initial-shards", type=int, default=5)
    t.add_argument("--transitions", type=int, default=100_000)
    t.add_argument("--warmup", type=int, default=5_000)
    t.add_argument("--device", default="cpu")
    t.add_argument("--actor-device", default=None)
    t.add_argument("--sample-every", type=float, default=1.0)
    t.add_argument("--keep", action="store_true", help="keep the run directories")
    t.add_argument("--noisy", action="store_true", help="label: the machine was busy")
    c = sub.add_parser("codec")
    c.add_argument("--demo-data", default=str(DEMO_DATA))
    c.add_argument("--batch-rows", type=int, default=896, help="384 x 2 (s, s') + 128 anchor")
    c.add_argument("--rounds", type=int, default=5)
    c.add_argument("--reps", type=int, default=20)
    c.add_argument("--noisy", action="store_true")
    v = sub.add_parser("verify-data")
    v.add_argument("--demo-data", default=str(DEMO_DATA))
    v.add_argument("--rounds", default="10,11,12")
    s_ = sub.add_parser("actor-state")
    s_.add_argument("--worlds", default="16,64,128")
    s_.add_argument("--frames", type=int, default=200)
    s_.add_argument("--noisy", action="store_true")
    for s in (m, p, t, c, s_, v):
        s.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    cmds = {
        "model": cmd_model,
        "podtel": cmd_podtel,
        "trainer": cmd_trainer,
        "codec": cmd_codec,
        "actor-state": cmd_actor_state,
        "verify-data": cmd_verify_data,
    }
    res = cmds[a.cmd](a)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
