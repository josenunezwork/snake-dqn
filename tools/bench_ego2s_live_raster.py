"""Latency and payload benchmark for live Watch ego2s hero_raster streaming.

Measures >=1,500 live Watch ticks at N=6 (CPU, 1 thread) comparing:
  1. main unsubscribed (baseline)
  2. branch unsubscribed (raster_subscribers == 0)
  3. branch subscribed (raster_subscribers == 1)

Bars:
  - Unsubscribed full-tick p95/p99 within +0.10 ms of main
  - Subscribed build_frame overhead p95 <= 1.5 ms
  - Subscribed frame payload <= 32 KiB (32,768 B)
  - 0 ticks > 41.67 ms (24 fps ceiling)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Callable, Dict, List, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)


def _pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q))


def _stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(arr)),
        "p50": _pct(arr, 50),
        "p95": _pct(arr, 95),
        "p99": _pct(arr, 99),
        "max": float(np.max(arr)),
    }


def _find_b8_checkpoint() -> str:
    candidates = [
        REPO_ROOT / "saved_snakes" / "b8_s3_t7_20261016.pth",
        REPO_ROOT.parent / "snake-dqn" / "saved_snakes" / "b8_s3_t7_20261016.pth",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    raise FileNotFoundError("Could not locate b8_s3_t7_20261016.pth")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_main_build_frame() -> Callable:
    """Load the exact main-branch web/backend/serialize.py::build_frame via git show."""
    src = subprocess.check_output(
        ["git", "show", "main:web/backend/serialize.py"],
        cwd=str(REPO_ROOT),
        text=True,
    )
    mod = types.ModuleType("web_backend_serialize_main")
    mod.__file__ = str(REPO_ROOT / "web/backend/serialize.py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod.build_frame


def _run_arm(
    sess,
    build_fn: Callable,
    *,
    subscribed: bool,
    ticks: int,
    reset_every: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one deterministic arm of ``ticks`` live Watch steps + build_frame."""
    _seed_everything(seed)
    sess.reset_game()
    sess.set_raster_stream("bench", subscribed)
    gc.collect()

    step_ms = np.empty(ticks, dtype=np.float64)
    bf_ms = np.empty(ticks, dtype=np.float64)
    full_ms = np.empty(ticks, dtype=np.float64)
    frame_bytes = np.empty(ticks, dtype=np.float64)
    raster_bytes = np.empty(ticks, dtype=np.float64)

    for t in range(ticks):
        if reset_every > 0 and t > 0 and (t % reset_every) == 0:
            sess.reset_game()
        t0 = time.perf_counter()
        sess.step()
        t1 = time.perf_counter()
        frame = build_fn(sess)
        t2 = time.perf_counter()

        step_ms[t] = (t1 - t0) * 1000.0
        bf_ms[t] = (t2 - t1) * 1000.0
        full_ms[t] = (t2 - t0) * 1000.0

        hr = frame.get("hero_raster")
        if subscribed:
            assert hr is not None
            raster_bytes[t] = len(json.dumps(hr, separators=(",", ":")).encode("utf-8"))
        else:
            assert hr is None
            raster_bytes[t] = 0.0
        frame_bytes[t] = len(json.dumps(frame, separators=(",", ":")).encode("utf-8"))

    return step_ms, bf_ms, full_ms, frame_bytes, raster_bytes


def run_benchmark(
    ticks: int = 1800,
    warmup: int = 200,
    reset_every: int = 300,
    seed: int = 1337,
) -> Dict[str, object]:
    from web.backend import served_checkpoint as registry
    from web.backend.serialize import build_frame
    from web.backend.session import GameSession

    b8_path = _find_b8_checkpoint()
    registry.SAVED_DIR = str(Path(b8_path).parent)
    main_build_frame = _load_main_build_frame()

    _seed_everything(seed)
    sess = GameSession(checkpoint=b8_path)
    assert len(sess.game.snakes) == 6, f"Expected N=6 snakes, got {len(sess.game.snakes)}"

    # Warmup Numba kernels + PyTorch conv trunk
    for _ in range(warmup):
        sess.step()
        main_build_frame(sess)
        sess.set_raster_stream("bench", False)
        build_frame(sess)
        sess.set_raster_stream("bench", True)
        build_frame(sess)

    main_step, main_bf, main_full, main_fbytes, _ = _run_arm(
        sess,
        main_build_frame,
        subscribed=False,
        ticks=ticks,
        reset_every=reset_every,
        seed=seed,
    )
    unsub_step, unsub_bf, unsub_full, unsub_fbytes, _ = _run_arm(
        sess,
        build_frame,
        subscribed=False,
        ticks=ticks,
        reset_every=reset_every,
        seed=seed,
    )
    sub_step, sub_bf, sub_full, sub_fbytes, raster_bytes = _run_arm(
        sess,
        build_frame,
        subscribed=True,
        ticks=ticks,
        reset_every=reset_every,
        seed=seed,
    )

    overhead_bf = sub_bf - unsub_bf
    p95_delta_vs_main = _pct(unsub_full, 95) - _pct(main_full, 95)
    p99_delta_vs_main = _pct(unsub_full, 99) - _pct(main_full, 99)
    sub_overhead_p95 = _pct(overhead_bf, 95)
    max_sub_frame_bytes = int(np.max(sub_fbytes))
    max_raster_bytes = int(np.max(raster_bytes))
    ticks_over_41_67_main = int(np.sum(main_full > 41.67))
    ticks_over_41_67_unsub = int(np.sum(unsub_full > 41.67))
    ticks_over_41_67_sub = int(np.sum(sub_full > 41.67))

    result = {
        "config": {
            "ticks": ticks,
            "warmup": warmup,
            "reset_every": reset_every,
            "num_snakes": 6,
            "seed": seed,
            "checkpoint": os.path.basename(b8_path),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "torch_num_threads": torch.get_num_threads(),
        },
        "main_step_ms": _stats(main_step),
        "unsubscribed_step_ms": _stats(unsub_step),
        "subscribed_step_ms": _stats(sub_step),
        "main_build_frame_ms": _stats(main_bf),
        "unsubscribed_build_frame_ms": _stats(unsub_bf),
        "subscribed_build_frame_ms": _stats(sub_bf),
        "subscribed_build_frame_overhead_ms": _stats(overhead_bf),
        "main_full_tick_ms": _stats(main_full),
        "unsubscribed_full_tick_ms": _stats(unsub_full),
        "subscribed_full_tick_ms": _stats(sub_full),
        "unsubscribed_frame_bytes": _stats(unsub_fbytes),
        "subscribed_frame_bytes": _stats(sub_fbytes),
        "hero_raster_bytes": _stats(raster_bytes),
        "bars": {
            "unsub_full_tick_p95_delta_vs_main_ms": p95_delta_vs_main,
            "unsub_full_tick_p99_delta_vs_main_ms": p99_delta_vs_main,
            "sub_build_frame_overhead_p95_ms": sub_overhead_p95,
            "sub_payload_max_bytes": max_sub_frame_bytes,
            "sub_payload_max_kib": max_sub_frame_bytes / 1024.0,
            "hero_raster_max_bytes": max_raster_bytes,
            "hero_raster_max_kib": max_raster_bytes / 1024.0,
            "ticks_over_41_67_ms_main": ticks_over_41_67_main,
            "ticks_over_41_67_ms_unsubscribed": ticks_over_41_67_unsub,
            "ticks_over_41_67_ms_subscribed": ticks_over_41_67_sub,
            "pass_unsub_p95_within_0_10ms": bool(p95_delta_vs_main <= 0.10),
            "pass_unsub_p99_within_0_10ms": bool(p99_delta_vs_main <= 0.10),
            "pass_sub_overhead_p95_le_1_50ms": bool(sub_overhead_p95 <= 1.50),
            "pass_payload_le_32kib": bool(max_sub_frame_bytes <= 32768),
            "pass_zero_ticks_over_41_67ms": bool(
                ticks_over_41_67_unsub == 0 and ticks_over_41_67_sub == 0
            ),
        },
    }
    result["bars"]["all_passed"] = all(
        v for k, v in result["bars"].items() if k.startswith("pass_")
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=1800)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--reset-every", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    res = run_benchmark(
        ticks=args.ticks,
        warmup=args.warmup,
        reset_every=args.reset_every,
        seed=args.seed,
    )
    text = json.dumps(res, indent=2)
    print(text)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    if not res["bars"]["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
