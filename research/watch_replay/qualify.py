"""Mac 24fps serving qualification for Watch replay controller filtering and 132-session bank.

Evaluates the T15 / ROADMAP §2.1 item 3 Watch replay qualification bars on this Mac:
  - Confirmed 132-session inventory parity:
      (a) SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l2-prime over watch-replay-bank-live-168
      (b) Direct serving over watch-replay-bank-live-132
  - Cold readiness < 1.0 s (GameSession() + initial snapshot(), plus FastAPI/WS /ws/stream
    first frame, while cold .watch.gz verification starts concurrently; cold first-verified
    .watch.gz completion time is also measured and reported separately).
  - 200 warmup + 1,500 unpaced Mac ticks and 200 warmup + 120 paced-24fps Mac ticks while
    cold .watch.gz verification runs concurrently:
      * update_ms p95 <= 11.688 ms (1.5x T15 Watch6 7.792 ms update p95)
      * step_ms / tick_ms p99 <= 41.67 ms (1000 / 24 ms processing budget)
      * 0 / 120 paced-24fps overruns (tick_ms < 41.6667 ms)
    measured for BOTH live mixed-Watch6 b8-s3 fallback ticks (cold-start worst case) and
    active recorded .watch.gz playback + rotation ticks.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence
from unittest.mock import patch

import numpy as np
import torch

from research.t15_mixed_roster_20261017.episodes import _assert_mixed, new_session
from web.backend import watch_replay as wr
from web.backend.watch_replay import (
    ENV_REPLAY_CONTROLLER,
    REPLAY_CONTROLLER_ALL,
    Recording,
    ReplayBank,
    peek_header,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BANK_168 = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/watch-replay-bank-live-168"
)
DEFAULT_BANK_132 = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/watch-replay-bank-live-132"
)

WARMUP_TICKS = 200
UNPACED_TICKS = 1500
PACED_TICKS = 120
SUPPORTED_CONTROLLERS = ("exact-stop-l2-prime", "exact-stop-l1")
TARGET_CONTROLLER = "exact-stop-l2-prime"
EXPECTED_CONFIRMED_COUNT = 132
EXPECTED_UNCONFIRMED_L1_COUNT = 36
EXPECTED_DEFAULT_COUNTS = {
    "exact-stop-l2-prime": EXPECTED_CONFIRMED_COUNT,
    "exact-stop-l1": EXPECTED_UNCONFIRMED_L1_COUNT,
}

BAR_COLD_READY_SECONDS = 1.0
BAR_UPDATE_P95_MS = 11.688  # 1.5 * 7.792 ms (T15 Watch6 mixed update p95)
BAR_STEP_P99_MS = 41.67  # 24 fps frame budget (1000 / 24 = 41.6667 ms)
BAR_PACED_BUDGET_MS = 1000.0 / 24.0
BAR_PACED_OVERRUNS = 0


def _percentile(ordered: Sequence[float], q: float) -> float:
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return float(
        ordered[low]
        if low == high
        else ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    )


def distribution(samples: Sequence[float]) -> dict[str, float | int]:
    if not samples:
        raise RuntimeError("no latency samples")
    ordered = sorted(float(x) for x in samples)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
        "mean": float(statistics.fmean(ordered)),
    }


def _cmd_text(argv: Sequence[str]) -> str:
    try:
        return subprocess.check_output(argv, cwd=REPO_ROOT, text=True, timeout=10).strip()
    except Exception as exc:
        return f"unavailable ({exc})"


def _current_nice_level() -> int | str:
    try:
        return int(os.getpriority(os.PRIO_PROCESS, 0))
    except Exception:
        return _cmd_text(["ps", "-o", "nice=", "-p", str(os.getpid())])


def collect_environment() -> dict[str, Any]:
    thread_keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OMP_THREAD_LIMIT",
        "SNAKE_DQN_DEVICE",
    )
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _cmd_text(["git", "rev-parse", "HEAD"]),
        "git_branch": _cmd_text(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status_porcelain": _cmd_text(["git", "status", "--porcelain"]),
        "worktree": str(REPO_ROOT),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_num_threads": int(torch.get_num_threads()),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "cpu_brand": _cmd_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "power_status": _cmd_text(["pmset", "-g", "batt"]),
        "nice_level": _current_nice_level(),
        "load_average_start": [float(x) for x in os.getloadavg()],
        "thread_env": {key: os.environ.get(key) for key in thread_keys},
        "port_8000_untouched": True,
    }


def scan_bank_headers(root: Path, *, product: bool = True) -> dict[str, Any]:
    """Inspect line-1 headers and sidecar SHA-256s of every .watch.gz in root."""
    started = time.perf_counter()
    paths = sorted(root.glob("*.watch.gz"))
    counts: Counter[str] = Counter()
    units_by_controller: dict[str, dict[str, dict[str, Any]]] = {}
    sizes: list[int] = []
    for path in paths:
        header = peek_header(path, product=product)
        sha_tokens = Path(str(path) + ".sha256").read_text().split()
        if len(sha_tokens) != 2 or sha_tokens[1] != path.name:
            raise ValueError(f"malformed sha256 sidecar for {path}")
        ctrl = str(header["controller"])
        uid = str(header["unit_id"])
        size_bytes = int(path.stat().st_size)
        counts[ctrl] += 1
        sizes.append(size_bytes)
        bucket = units_by_controller.setdefault(ctrl, {})
        if uid in bucket:
            raise ValueError(f"duplicate unit_id {uid!r} in {root}")
        bucket[uid] = {
            "file": path.name,
            "sha256": sha_tokens[0],
            "seed": int(header["seed"]),
            "horizon": int(header["horizon"]),
            "size_bytes": size_bytes,
        }
    elapsed = time.perf_counter() - started
    return {
        "root": str(root),
        "total_files": len(paths),
        "by_controller": dict(sorted(counts.items())),
        "compressed_size_bytes": (
            {
                "min": min(sizes),
                "median": int(statistics.median(sizes)),
                "max": max(sizes),
                "total": sum(sizes),
            }
            if sizes
            else {"min": 0, "median": 0, "max": 0, "total": 0}
        ),
        "line1_scan_seconds": elapsed,
        "units_by_controller": units_by_controller,
    }


def prove_line1_filter_skips_non_target_payloads(
    mixed_bank: Path,
    controller: str = TARGET_CONTROLLER,
    *,
    product: bool = True,
) -> dict[str, Any]:
    """Prove ReplayBank skips all non-target files on line 1 with 0 payload verifications."""
    if controller not in SUPPORTED_CONTROLLERS:
        raise ValueError(
            f"unsupported controller {controller!r}; expected one of {SUPPORTED_CONTROLLERS}"
        )
    verified_controllers: list[str] = []
    peeked_controllers: Counter[str] = Counter()
    original_verify = wr.verify_payload
    original_peek = wr.peek_header
    non_target_paths = {
        path
        for path in sorted(mixed_bank.glob("*.watch.gz"))
        if str(original_peek(path, product=product).get("controller")) != controller
    }

    def spy_peek(path: Path, *, product: bool = product) -> dict:
        header = original_peek(path, product=product)
        peeked_controllers[str(header["controller"])] += 1
        return header

    def spy_verify(payload: bytes, **kwargs: Any) -> dict:
        kwargs["product"] = product
        header = original_verify(payload, **kwargs)
        verified_controllers.append(str(header["controller"]))
        return header

    class NonTargetFirstRng:
        """Visit every non-target recording FIRST (in reverse order) before any target recording."""

        def shuffle(self, items: list[Path]) -> None:
            items.reverse()
            items.sort(key=lambda path: 0 if path in non_target_paths else 1)

    started = time.perf_counter()
    with (
        patch.object(wr, "peek_header", spy_peek),
        patch.object(wr, "verify_payload", spy_verify),
        patch.object(wr.random, "SystemRandom", NonTargetFirstRng),
    ):
        bank = ReplayBank(
            mixed_bank,
            controller=controller,
            shuffle_discovery=True,
        )
        try:
            deadline = time.monotonic() + 30.0
            picked = None
            while time.monotonic() < deadline:
                picked = bank.pick()
                if picked is not None or not bank.pending:
                    break
                time.sleep(0.01)
            if picked is None:
                raise RuntimeError(f"ReplayBank failed to pick {controller}: {bank.error}")
            elapsed = time.perf_counter() - started
            picked_info = picked.info()
        finally:
            bank.close()

    non_target_peeked = sum(
        count for ctrl, count in peeked_controllers.items() if ctrl != controller
    )
    return {
        "controller_filter": controller,
        "non_target_headers_peeked_and_skipped_before_first_pick": int(non_target_peeked),
        "non_target_peeked_by_controller": {
            ctrl: int(count)
            for ctrl, count in sorted(peeked_controllers.items())
            if ctrl != controller
        },
        "l1_headers_peeked_and_skipped_before_first_pick": int(
            peeked_controllers.get("exact-stop-l1", 0)
        ),
        "l2_prime_headers_peeked_and_skipped_before_first_pick": int(
            peeked_controllers.get("exact-stop-l2-prime", 0)
        ),
        "non_target_verify_payload_calls": sum(1 for c in verified_controllers if c != controller),
        "target_verify_payload_calls": sum(1 for c in verified_controllers if c == controller),
        "first_picked_unit_id": picked_info["unit_id"],
        "first_picked_controller": picked_info["controller"],
        "first_picked_sha256": picked_info["sha256"],
        "elapsed_seconds": elapsed,
    }


def prove_line1_filter_skips_l1_payloads(
    bank_168: Path,
    controller: str = TARGET_CONTROLLER,
    *,
    product: bool = True,
) -> dict[str, Any]:
    """Backward-compatible wrapper around prove_line1_filter_skips_non_target_payloads."""
    return prove_line1_filter_skips_non_target_payloads(
        bank_168,
        controller=controller,
        product=product,
    )


def audit_banks(
    bank_168: Path | None,
    bank_132: Path,
    controller: str = TARGET_CONTROLLER,
    *,
    product: bool = True,
) -> dict[str, Any]:
    if controller not in SUPPORTED_CONTROLLERS:
        raise ValueError(
            f"unsupported controller {controller!r}; expected one of {SUPPORTED_CONTROLLERS}"
        )
    scan_pure = scan_bank_headers(bank_132, product=product)
    target_from_pure = scan_pure["units_by_controller"].get(controller, {})
    comparable_pure = {
        uid: (meta["sha256"], meta["seed"], meta["horizon"])
        for uid, meta in target_from_pure.items()
    }
    other_controllers_in_pure = [k for k in scan_pure["by_controller"] if k != controller]
    pure_clean = (
        len(comparable_pure) > 0
        and scan_pure["total_files"] == len(comparable_pure)
        and not other_controllers_in_pure
    )

    if bank_168 is None:
        return {
            "target_controller": controller,
            "mixed_bank": None,
            "pure_bank": {k: v for k, v in scan_pure.items() if k != "units_by_controller"},
            "live_168": None,
            "live_132": {k: v for k, v in scan_pure.items() if k != "units_by_controller"},
            "target_count_in_mixed": None,
            "non_target_count_in_mixed": 0,
            "target_count_in_pure": len(comparable_pure),
            "other_controllers_in_pure": other_controllers_in_pure,
            "other_controllers_in_132": other_controllers_in_pure,
            "filtered_mixed_equals_pure_exact_sha_seed_unit_set": pure_clean,
            "filtered_168_equals_132_exact_sha_seed_unit_set": pure_clean,
            "line1_reverse_discovery_filter_proof": None,
        }

    scan_168 = scan_bank_headers(bank_168, product=product)
    l2_from_168 = scan_168["units_by_controller"].get("exact-stop-l2-prime", {})
    l1_from_168 = scan_168["units_by_controller"].get("exact-stop-l1", {})
    target_from_168 = scan_168["units_by_controller"].get(controller, {})

    comparable_168 = {
        uid: (meta["sha256"], meta["seed"], meta["horizon"])
        for uid, meta in target_from_168.items()
    }
    non_target_count_in_mixed = scan_168["total_files"] - len(comparable_168)
    exact_match = pure_clean and comparable_168 == comparable_pure
    if bank_168.resolve() == DEFAULT_BANK_168.resolve():
        exact_match = (
            exact_match
            and len(l2_from_168) == EXPECTED_CONFIRMED_COUNT
            and len(l1_from_168) == EXPECTED_UNCONFIRMED_L1_COUNT
            and len(comparable_pure) == EXPECTED_DEFAULT_COUNTS[controller]
        )
    filter_proof = prove_line1_filter_skips_non_target_payloads(
        bank_168,
        controller=controller,
        product=product,
    )
    return {
        "target_controller": controller,
        "mixed_bank": {k: v for k, v in scan_168.items() if k != "units_by_controller"},
        "pure_bank": {k: v for k, v in scan_pure.items() if k != "units_by_controller"},
        "live_168": {k: v for k, v in scan_168.items() if k != "units_by_controller"},
        "live_132": {k: v for k, v in scan_pure.items() if k != "units_by_controller"},
        "target_count_in_mixed": len(comparable_168),
        "non_target_count_in_mixed": non_target_count_in_mixed,
        "target_count_in_pure": len(comparable_pure),
        "confirmed_l2_prime_count_in_168": len(l2_from_168),
        "unconfirmed_l1_count_in_168": len(l1_from_168),
        "confirmed_l2_prime_count_in_132": len(
            scan_pure["units_by_controller"].get("exact-stop-l2-prime", {})
        ),
        "exact_stop_l1_count_in_pure": len(
            scan_pure["units_by_controller"].get("exact-stop-l1", {})
        ),
        "other_controllers_in_pure": other_controllers_in_pure,
        "other_controllers_in_132": other_controllers_in_pure,
        "filtered_mixed_equals_pure_exact_sha_seed_unit_set": exact_match,
        "filtered_168_equals_132_exact_sha_seed_unit_set": exact_match,
        "line1_reverse_discovery_filter_proof": filter_proof,
    }


def measure_cold_readiness(bank_dir: Path, controller: str) -> dict[str, Any]:
    """Measure fresh-process cold GameSession + HTTP/WS readiness while cold verification starts."""
    env = dict(os.environ)
    for key in (
        "SNAKE_SERVE_CHECKPOINT",
        "SNAKE_SERVE_VETO_WATCH_HERO",
        "SNAKE_SERVE_VETO_PLAY_AI",
        "SNAKE_SERVE_VETO_VARIANT",
    ):
        env.pop(key, None)
    env["SNAKE_SERVE_WATCH_ROSTER"] = "mixed"
    env["SNAKE_WATCH_SOURCE"] = "replay"
    env["SNAKE_WATCH_REPLAY_DIR"] = str(bank_dir)
    env[ENV_REPLAY_CONTROLLER] = controller
    env["SNAKE_SCORES_DB"] = ":memory:"
    env["SNAKE_DQN_DEVICE"] = "cpu"

    session_probe_code = """
import json, time
from unittest.mock import patch
t_proc = time.perf_counter()
import web.backend.session as sm
t_import = time.perf_counter()
with patch.object(sm, "served_resolution"), patch.object(sm, "watch_tick"):
    t_s0 = time.perf_counter()
    s = sm.GameSession()
    s._release_monitoring = False
    snap0 = s.snapshot()
    t_ready = time.perf_counter()
    clean_mixed = s._is_clean_mixed_watch6_b8()
    bank_pending_at_ready = bool(s._replay_bank is not None and s._replay_bank.pending)
    initial_watch_source = snap0["session"].get("watch_source")
    first_replay = None
    deadline = time.perf_counter() + 90.0
    ticks_until_replay = 0
    while time.perf_counter() < deadline:
        s.step()
        ticks_until_replay += 1
        snap = s.snapshot()
        if snap["session"].get("watch_source") == "replay":
            first_replay = {
                "activated_seconds_from_session_start": time.perf_counter() - t_s0,
                "ticks_until_activation": ticks_until_replay,
                "replay_info": snap["session"]["replay"],
            }
            break
        time.sleep(1.0 / 24.0)
    s.close()
print("QUAL_JSON:" + json.dumps({
    "import_seconds": t_import - t_proc,
    "session_construct_plus_snapshot_seconds": t_ready - t_s0,
    "process_import_plus_session_ready_seconds": t_ready - t_proc,
    "clean_mixed_watch6_b8": clean_mixed,
    "bank_pending_at_ready": bank_pending_at_ready,
    "initial_watch_source": initial_watch_source,
    "protocol_version": snap0["protocol_version"],
    "mechanics_version": snap0["mechanics_version"],
    "num_snakes": len(snap0["snakes"]),
    "first_verified_replay": first_replay,
}))
"""
    raw_session = subprocess.check_output(
        [sys.executable, "-c", session_probe_code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        timeout=150,
    )
    session_line = next(
        line[len("QUAL_JSON:") :]
        for line in raw_session.splitlines()
        if line.startswith("QUAL_JSON:")
    )
    session_metrics = json.loads(session_line)

    ws_probe_code = """
import json, time
from unittest.mock import patch
from fastapi.testclient import TestClient
import web.backend.app as app_mod
import web.backend.session as sm
with patch.object(sm, "served_resolution"), patch.object(sm, "watch_tick"):
    t0 = time.perf_counter()
    with TestClient(app_mod.app) as client:
        health = client.get("/api/health")
        t_health = time.perf_counter()
        with client.websocket_connect("/ws/stream") as ws:
            frame = ws.receive_json()
            t_ws = time.perf_counter()
print("QUAL_WS_JSON:" + json.dumps({
    "health_status": health.status_code,
    "health_ready_seconds": t_health - t0,
    "ws_first_frame_ready_seconds": t_ws - t0,
    "ws_frame_type": frame.get("type"),
    "ws_watch_roster": frame.get("session", {}).get("watch_roster"),
    "ws_watch_source_at_boot": frame.get("session", {}).get("watch_source"),
    "ws_num_snakes": len(frame.get("snakes", [])),
}))
"""
    raw_ws = subprocess.check_output(
        [sys.executable, "-c", ws_probe_code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        timeout=60,
    )
    ws_line = next(
        line[len("QUAL_WS_JSON:") :]
        for line in raw_ws.splitlines()
        if line.startswith("QUAL_WS_JSON:")
    )
    ws_metrics = json.loads(ws_line)

    cold_pass = (
        float(session_metrics["session_construct_plus_snapshot_seconds"]) < BAR_COLD_READY_SECONDS
        and float(ws_metrics["ws_first_frame_ready_seconds"]) < BAR_COLD_READY_SECONDS
        and int(ws_metrics["health_status"]) == 200
        and bool(session_metrics["clean_mixed_watch6_b8"])
        and session_metrics["first_verified_replay"] is not None
        and session_metrics["first_verified_replay"]["replay_info"]["controller"] == controller
    )
    return {
        "session_probe": session_metrics,
        "http_ws_probe": ws_metrics,
        "cold_readiness_bar_seconds": BAR_COLD_READY_SECONDS,
        "pass": cold_pass,
    }


class ConcurrentColdVerifier:
    """Continuously run cold ReplayBank-equivalent line-1 filter + uncached Recording verify."""

    def __init__(self, bank_dir: Path, controller: str) -> None:
        self.bank_dir = bank_dir
        self.controller = controller
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self.frames_validated = 0
        self.recordings_verified = 0
        self.headers_filtered_out = 0
        self.verification_durations: list[float] = []
        self.errors: list[str] = []

        all_paths = sorted(bank_dir.glob("*.watch.gz"))
        target_by_size = []
        other_paths = []
        for path in all_paths:
            hdr = peek_header(path, product=True)
            if controller == REPLAY_CONTROLLER_ALL or hdr["controller"] == controller:
                target_by_size.append(path)
            else:
                other_paths.append(path)
        if not target_by_size:
            raise ValueError(f"no recordings for controller {controller!r} in {bank_dir}")
        target_by_size.sort(key=lambda p: p.stat().st_size, reverse=True)
        # Interleave non-target files with the heaviest target .watch.gz files
        # so both line-1 filtering and worst-case frame validation run concurrently.
        schedule: list[Path] = []
        top_heavy = target_by_size[:8]
        for idx, heavy in enumerate(top_heavy):
            if other_paths:
                schedule.append(other_paths[idx % len(other_paths)])
            schedule.append(heavy)
        self._schedule = schedule
        self._thread = threading.Thread(
            target=self._run, name="qual-cold-watch-gz-verifier", daemon=True
        )

    def snapshot_counters(self) -> dict[str, int]:
        with self._lock:
            return {
                "frames_validated": self.frames_validated,
                "recordings_verified": self.recordings_verified,
                "headers_filtered_out": self.headers_filtered_out,
                "errors": len(self.errors),
            }

    def _run(self) -> None:
        idx = 0
        while not self.stop.is_set():
            path = self._schedule[idx % len(self._schedule)]
            idx += 1
            try:
                if self.controller != REPLAY_CONTROLLER_ALL:
                    hdr = peek_header(path, product=True)
                    if hdr["controller"] != self.controller:
                        with self._lock:
                            self.headers_filtered_out += 1
                        continue
                t0 = time.perf_counter()
                rec = Recording(path, product=True, cache=None, stop=self.stop)
                if (
                    self.controller != REPLAY_CONTROLLER_ALL
                    and rec.header["controller"] != self.controller
                ):
                    raise ValueError(f"unexpected controller {rec.header['controller']}")
                elapsed = time.perf_counter() - t0
                with self._lock:
                    self.recordings_verified += 1
                    self.verification_durations.append(elapsed)
            except InterruptedError:
                return
            except Exception as exc:
                with self._lock:
                    self.errors.append(f"{path.name}: {exc}")
                return

    @contextlib.contextmanager
    def active(self) -> Iterator["ConcurrentColdVerifier"]:
        original_validate = wr.validate_frame
        verifier_thread_ident: int | None = None

        def counting_validate(frame: dict, index: int) -> None:
            original_validate(frame, index)
            if threading.get_ident() == verifier_thread_ident:
                with self._lock:
                    self.frames_validated += 1

        with patch.object(wr, "validate_frame", counting_validate):
            self._thread.start()
            verifier_thread_ident = self._thread.ident
            # Wait until cold frame validation is actively mid-flight before starting ticks.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if self.snapshot_counters()["frames_validated"] >= 32:
                    break
                time.sleep(0.005)
            try:
                yield self
            finally:
                self.stop.set()
                self._thread.join(timeout=3.0)


def _timed_live_tick(session: Any) -> tuple[float, float, float]:
    """Return (update_ms, step_ms, step_plus_snapshot_ms) for one live Watch6 tick."""
    original_update = session.game.update
    update_ms = 0.0

    def timed_update(*args: Any, **kwargs: Any) -> Any:
        nonlocal update_ms
        t0 = time.perf_counter()
        try:
            return original_update(*args, **kwargs)
        finally:
            update_ms = (time.perf_counter() - t0) * 1000.0

    session.game.update = timed_update
    t_start = time.perf_counter()
    try:
        before = int(session.game.frame)
        session.step()
        t_after_step = time.perf_counter()
        snap = session.snapshot()
        t_after_snap = time.perf_counter()
        if int(session.game.frame) != before + 1 or session.last_error:
            raise RuntimeError(f"live Watch6 tick failed: {session.last_error!r}")
        if snap["session"].get("watch_roster") != "mixed" or len(snap["snakes"]) != 6:
            raise RuntimeError("live Watch6 snapshot lost mixed 6-snake contract")
    finally:
        session.game.update = original_update
    return (
        update_ms,
        (t_after_step - t_start) * 1000.0,
        (t_after_snap - t_start) * 1000.0,
    )


def measure_live_fallback_under_cold_verification(
    bank_dir: Path, controller: str, seed: int = 17001
) -> dict[str, Any]:
    """Measure 1,500 unpaced + 120 paced-24fps live mixed-Watch6 ticks under cold verification."""
    verifier = ConcurrentColdVerifier(bank_dir, controller)
    with verifier.active():
        unpaced_session = new_session(seed=seed, num_snakes=6)
        try:
            _assert_mixed(unpaced_session, 6)
            if not unpaced_session._is_clean_mixed_watch6_b8():
                raise RuntimeError("unpaced qualification session failed _is_clean_mixed_watch6_b8")
            for _ in range(WARMUP_TICKS):
                _timed_live_tick(unpaced_session)
            c_before_unpaced = verifier.snapshot_counters()
            updates_ms, steps_ms, ticks_ms = [], [], []
            for _ in range(UNPACED_TICKS):
                u_ms, s_ms, t_ms = _timed_live_tick(unpaced_session)
                updates_ms.append(u_ms)
                steps_ms.append(s_ms)
                ticks_ms.append(t_ms)
            c_after_unpaced = verifier.snapshot_counters()
        finally:
            unpaced_session.close()

        paced_session = new_session(seed=seed + 100, num_snakes=6)
        try:
            _assert_mixed(paced_session, 6)
            paced_session.set_speed(120)
            actual_speed = float(paced_session.speed)
            speed_ceiling = float(paced_session.speed_ceiling)
            if actual_speed != 24.0 or speed_ceiling != 24.0:
                raise RuntimeError(
                    f"expected Watch6 24.0 fps ceiling, got speed={actual_speed} "
                    f"ceiling={speed_ceiling}"
                )
            for _ in range(WARMUP_TICKS):
                _timed_live_tick(paced_session)
            budget_s = 1.0 / actual_speed
            budget_ms = budget_s * 1000.0
            c_before_paced = verifier.snapshot_counters()
            p_updates_ms, p_steps_ms, p_ticks_ms = [], [], []
            overruns = 0
            t_paced_start = time.perf_counter()
            for _ in range(PACED_TICKS):
                u_ms, s_ms, t_ms = _timed_live_tick(paced_session)
                p_updates_ms.append(u_ms)
                p_steps_ms.append(s_ms)
                p_ticks_ms.append(t_ms)
                if t_ms >= budget_ms:
                    overruns += 1
                time.sleep(budget_s)
            paced_elapsed = time.perf_counter() - t_paced_start
            c_after_paced = verifier.snapshot_counters()
        finally:
            paced_session.close()

    if verifier.errors:
        raise RuntimeError(f"concurrent cold verifier raised errors: {verifier.errors}")

    frames_during_unpaced = (
        c_after_unpaced["frames_validated"] - c_before_unpaced["frames_validated"]
    )
    frames_during_paced = c_after_paced["frames_validated"] - c_before_paced["frames_validated"]
    unpaced_update_dist = distribution(updates_ms)
    unpaced_step_dist = distribution(steps_ms)
    unpaced_tick_dist = distribution(ticks_ms)
    paced_update_dist = distribution(p_updates_ms)
    paced_step_dist = distribution(p_steps_ms)
    paced_tick_dist = distribution(p_ticks_ms)

    passed = (
        float(unpaced_update_dist["p95"]) <= BAR_UPDATE_P95_MS
        and float(unpaced_step_dist["p99"]) <= BAR_STEP_P99_MS
        and float(unpaced_tick_dist["p99"]) <= BAR_STEP_P99_MS
        and overruns == BAR_PACED_OVERRUNS
        and float(paced_tick_dist["p99"]) <= BAR_STEP_P99_MS
        and frames_during_unpaced > 0
        and frames_during_paced > 0
    )
    return {
        "warmup_ticks": WARMUP_TICKS,
        "unpaced_1500": {
            "ticks": UNPACED_TICKS,
            "update_ms": unpaced_update_dist,
            "step_ms": unpaced_step_dist,
            "tick_ms": unpaced_tick_dist,
            "concurrent_frames_validated": frames_during_unpaced,
        },
        "paced_24fps_120": {
            "requested_fps": 120,
            "actual_speed_fps": actual_speed,
            "speed_ceiling_fps": speed_ceiling,
            "budget_ms": budget_ms,
            "ticks": PACED_TICKS,
            "update_ms": paced_update_dist,
            "step_ms": paced_step_dist,
            "tick_ms": paced_tick_dist,
            "overruns": overruns,
            "zero_overrun": overruns == 0,
            "elapsed_seconds": paced_elapsed,
            "concurrent_frames_validated": frames_during_paced,
        },
        "concurrent_verifier_totals": {
            "recordings_verified": verifier.recordings_verified,
            "frames_validated": verifier.frames_validated,
            "headers_filtered_out": verifier.headers_filtered_out,
            "mean_single_recording_verify_seconds": (
                float(statistics.fmean(verifier.verification_durations))
                if verifier.verification_durations
                else None
            ),
        },
        "pass": passed,
    }


def measure_replay_playback_under_cold_verification(
    bank_dir: Path, controller: str
) -> dict[str, Any]:
    """Measure 1,500 unpaced + 120 paced-24fps active .watch.gz playback + rotation ticks."""
    import web.backend.session as sm

    saved_env = {
        key: os.environ.get(key)
        for key in (
            "SNAKE_SERVE_WATCH_ROSTER",
            "SNAKE_SERVE_CHECKPOINT",
            "SNAKE_SERVE_VETO_WATCH_HERO",
            "SNAKE_SERVE_VETO_PLAY_AI",
            "SNAKE_SERVE_VETO_VARIANT",
            "SNAKE_WATCH_SOURCE",
            "SNAKE_WATCH_REPLAY_DIR",
            ENV_REPLAY_CONTROLLER,
        )
    }
    try:
        for key in saved_env:
            os.environ.pop(key, None)
        os.environ["SNAKE_SERVE_WATCH_ROSTER"] = "mixed"
        os.environ["SNAKE_WATCH_SOURCE"] = "replay"
        os.environ["SNAKE_WATCH_REPLAY_DIR"] = str(bank_dir)
        os.environ[ENV_REPLAY_CONTROLLER] = controller

        with patch.object(sm, "served_resolution"), patch.object(sm, "watch_tick"):
            session = sm.GameSession()
        session._release_monitoring = False
        try:
            if session._replay_bank is None or not session._is_clean_mixed_watch6_b8():
                raise RuntimeError("replay bank did not activate on clean mixed Watch6 b8-s3")
            # Wait for recording #1 to activate and recording #2 to finish prefetching so we
            # can also verify nonblocking rotation across distinct confirmed unit_ids.
            deadline = time.monotonic() + 120.0
            while time.monotonic() < deadline:
                if session._watch_replay is None:
                    session.step()
                if session._watch_replay is not None and not session._replay_bank._ready.empty():
                    break
                time.sleep(0.01)
            if session._watch_replay is None or session._replay_bank._ready.empty():
                raise RuntimeError("timed out waiting for active + prefetched Watch recordings")

            live_update_calls = 0
            original_update = session.game.update

            def forbid_live_update(*args: Any, **kwargs: Any) -> Any:
                nonlocal live_update_calls
                live_update_calls += 1
                return original_update(*args, **kwargs)

            session.game.update = forbid_live_update
            observed_controllers: Counter[str] = Counter()
            observed_units: list[str] = []

            def timed_replay_tick() -> tuple[float, float]:
                t0 = time.perf_counter()
                session.step()
                t1 = time.perf_counter()
                snap = session.snapshot()
                t2 = time.perf_counter()
                if snap["session"].get("watch_source") != "replay":
                    raise RuntimeError("Watch dropped out of replay unexpectedly")
                rep = snap["session"]["replay"]
                observed_controllers[str(rep["controller"])] += 1
                if not observed_units or observed_units[-1] != rep["unit_id"]:
                    observed_units.append(str(rep["unit_id"]))
                return (t1 - t0) * 1000.0, (t2 - t0) * 1000.0

            verifier = ConcurrentColdVerifier(bank_dir, controller)
            with verifier.active():
                for _ in range(WARMUP_TICKS):
                    timed_replay_tick()
                c_before_unpaced = verifier.snapshot_counters()
                steps_ms, ticks_ms = [], []
                for idx in range(UNPACED_TICKS):
                    if idx == UNPACED_TICKS // 2:
                        # Exercise nonblocking rotation mid-window under concurrent cold verify.
                        t_rot0 = time.perf_counter()
                        prev_uid = session._watch_replay.header["unit_id"]
                        session.reset_game()
                        rotation_ms = (time.perf_counter() - t_rot0) * 1000.0
                        next_uid = session._watch_replay.header["unit_id"]
                        if prev_uid == next_uid:
                            raise RuntimeError(f"immediate unit repeat on rotation: {prev_uid}")
                    s_ms, t_ms = timed_replay_tick()
                    steps_ms.append(s_ms)
                    ticks_ms.append(t_ms)
                c_after_unpaced = verifier.snapshot_counters()

                session.set_speed(120)
                actual_speed = float(session.speed)
                budget_s = 1.0 / actual_speed
                budget_ms = budget_s * 1000.0
                for _ in range(WARMUP_TICKS):
                    timed_replay_tick()
                c_before_paced = verifier.snapshot_counters()
                p_steps_ms, p_ticks_ms = [], []
                overruns = 0
                for _ in range(PACED_TICKS):
                    s_ms, t_ms = timed_replay_tick()
                    p_steps_ms.append(s_ms)
                    p_ticks_ms.append(t_ms)
                    if t_ms >= budget_ms:
                        overruns += 1
                    time.sleep(budget_s)
                c_after_paced = verifier.snapshot_counters()
        finally:
            session.close()
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    if verifier.errors:
        raise RuntimeError(f"concurrent cold verifier raised errors: {verifier.errors}")

    unpaced_step_dist = distribution(steps_ms)
    unpaced_tick_dist = distribution(ticks_ms)
    paced_step_dist = distribution(p_steps_ms)
    paced_tick_dist = distribution(p_ticks_ms)
    non_target = sum(v for k, v in observed_controllers.items() if k != controller)
    frames_unpaced = c_after_unpaced["frames_validated"] - c_before_unpaced["frames_validated"]
    frames_paced = c_after_paced["frames_validated"] - c_before_paced["frames_validated"]

    passed = (
        live_update_calls == 0
        and non_target == 0
        and len(set(observed_units)) >= 2
        and float(unpaced_step_dist["p99"]) <= BAR_STEP_P99_MS
        and float(unpaced_tick_dist["p99"]) <= BAR_STEP_P99_MS
        and overruns == BAR_PACED_OVERRUNS
        and float(paced_tick_dist["p99"]) <= BAR_STEP_P99_MS
        and frames_unpaced > 0
        and frames_paced > 0
    )
    return {
        "live_update_calls_during_replay": live_update_calls,
        "observed_controllers": dict(observed_controllers),
        "non_target_controller_frames": non_target,
        "distinct_units_played": observed_units,
        "mid_stream_reset_rotation_ms": rotation_ms,
        "unpaced_1500": {
            "ticks": UNPACED_TICKS,
            "step_ms": unpaced_step_dist,
            "tick_ms": unpaced_tick_dist,
            "concurrent_frames_validated": frames_unpaced,
        },
        "paced_24fps_120": {
            "actual_speed_fps": actual_speed,
            "budget_ms": budget_ms,
            "ticks": PACED_TICKS,
            "step_ms": paced_step_dist,
            "tick_ms": paced_tick_dist,
            "overruns": overruns,
            "zero_overrun": overruns == 0,
            "concurrent_frames_validated": frames_paced,
        },
        "pass": passed,
    }


def qualify_target(label: str, bank_dir: Path, controller: str) -> dict[str, Any]:
    cold = measure_cold_readiness(bank_dir, controller)
    live_fallback = measure_live_fallback_under_cold_verification(bank_dir, controller)
    replay_playback = measure_replay_playback_under_cold_verification(bank_dir, controller)

    strict_controller_bar = {
        "bar": f"0 non-{controller} frames & 0 live update calls",
        "target_controller": controller,
        "measured_non_target_frames": replay_playback["non_target_controller_frames"],
        "measured_live_updates": replay_playback["live_update_calls_during_replay"],
        "pass": (
            int(replay_playback["non_target_controller_frames"]) == 0
            and int(replay_playback["live_update_calls_during_replay"]) == 0
        ),
    }
    bars: dict[str, Any] = {
        "cold_session_ready_lt_1s": {
            "bar": f"< {BAR_COLD_READY_SECONDS:.3f} s",
            "measured": cold["session_probe"]["session_construct_plus_snapshot_seconds"],
            "pass": float(cold["session_probe"]["session_construct_plus_snapshot_seconds"])
            < BAR_COLD_READY_SECONDS,
        },
        "cold_http_ws_first_frame_lt_1s": {
            "bar": f"< {BAR_COLD_READY_SECONDS:.3f} s",
            "measured": cold["http_ws_probe"]["ws_first_frame_ready_seconds"],
            "pass": float(cold["http_ws_probe"]["ws_first_frame_ready_seconds"])
            < BAR_COLD_READY_SECONDS,
        },
        "live_fallback_update_p95_le_11_688ms": {
            "bar": f"<= {BAR_UPDATE_P95_MS:.3f} ms",
            "measured": live_fallback["unpaced_1500"]["update_ms"]["p95"],
            "pass": float(live_fallback["unpaced_1500"]["update_ms"]["p95"]) <= BAR_UPDATE_P95_MS,
        },
        "live_fallback_step_p99_le_41_67ms": {
            "bar": f"<= {BAR_STEP_P99_MS:.2f} ms",
            "measured_step_ms_p99": live_fallback["unpaced_1500"]["step_ms"]["p99"],
            "measured_tick_ms_p99": live_fallback["unpaced_1500"]["tick_ms"]["p99"],
            "pass": (
                float(live_fallback["unpaced_1500"]["step_ms"]["p99"]) <= BAR_STEP_P99_MS
                and float(live_fallback["unpaced_1500"]["tick_ms"]["p99"]) <= BAR_STEP_P99_MS
            ),
        },
        "live_fallback_paced_24fps_zero_overruns": {
            "bar": "0 / 120 overruns (< 41.667 ms)",
            "measured_overruns": live_fallback["paced_24fps_120"]["overruns"],
            "measured_max_tick_ms": live_fallback["paced_24fps_120"]["tick_ms"]["max"],
            "pass": int(live_fallback["paced_24fps_120"]["overruns"]) == 0,
        },
        "replay_playback_step_p99_le_41_67ms": {
            "bar": f"<= {BAR_STEP_P99_MS:.2f} ms",
            "measured_step_ms_p99": replay_playback["unpaced_1500"]["step_ms"]["p99"],
            "measured_tick_ms_p99": replay_playback["unpaced_1500"]["tick_ms"]["p99"],
            "pass": (
                float(replay_playback["unpaced_1500"]["step_ms"]["p99"]) <= BAR_STEP_P99_MS
                and float(replay_playback["unpaced_1500"]["tick_ms"]["p99"]) <= BAR_STEP_P99_MS
            ),
        },
        "replay_playback_paced_24fps_zero_overruns": {
            "bar": "0 / 120 overruns (< 41.667 ms)",
            "measured_overruns": replay_playback["paced_24fps_120"]["overruns"],
            "measured_max_tick_ms": replay_playback["paced_24fps_120"]["tick_ms"]["max"],
            "pass": int(replay_playback["paced_24fps_120"]["overruns"]) == 0,
        },
        "replay_playback_strict_target_controller_only": strict_controller_bar,
    }
    if controller == "exact-stop-l2-prime":
        bars["replay_playback_strict_l2_prime_only"] = strict_controller_bar
    target_pass = (
        cold["pass"]
        and live_fallback["pass"]
        and replay_playback["pass"]
        and all(item["pass"] for item in bars.values())
    )
    return {
        "label": label,
        "bank_dir": str(bank_dir),
        "controller_env": controller,
        "cold_readiness": cold,
        "live_fallback_under_concurrent_cold_verification": live_fallback,
        "replay_playback_under_concurrent_cold_verification": replay_playback,
        "bars": bars,
        "pass": target_pass,
    }


def run_qualification(
    bank_168: Path | None = DEFAULT_BANK_168,
    bank_132: Path = DEFAULT_BANK_132,
    *,
    controller: str = TARGET_CONTROLLER,
    bank: Path | None = None,
    mixed_bank: Path | None = None,
) -> dict[str, Any]:
    if controller not in SUPPORTED_CONTROLLERS:
        raise ValueError(
            f"unsupported controller {controller!r}; expected one of {SUPPORTED_CONTROLLERS}"
        )
    pure_bank = bank if bank is not None else bank_132
    resolved_mixed_bank = mixed_bank if mixed_bank is not None else bank_168

    started = time.perf_counter()
    env_meta = collect_environment()
    inventory = audit_banks(resolved_mixed_bank, pure_bank, controller=controller)
    target_pure = qualify_target("pure_bank_direct", pure_bank, controller)
    targets: dict[str, Any] = {
        "pure_bank_direct": target_pure,
        "live_132_direct": target_pure,
    }
    if resolved_mixed_bank is not None:
        target_mixed = qualify_target("mixed_bank_filtered", resolved_mixed_bank, controller)
        targets["mixed_bank_filtered"] = target_mixed
        targets["live_168_filtered"] = target_mixed
    else:
        target_mixed = None

    env_meta["load_average_end"] = [float(x) for x in os.getloadavg()]
    env_meta["total_qualification_seconds"] = time.perf_counter() - started

    filter_proof = inventory["line1_reverse_discovery_filter_proof"]
    filter_ok = (
        True
        if filter_proof is None
        else (
            int(filter_proof["non_target_verify_payload_calls"]) == 0
            and int(filter_proof["non_target_headers_peeked_and_skipped_before_first_pick"])
            == int(inventory["non_target_count_in_mixed"])
        )
    )
    overall_pass = (
        bool(inventory["filtered_mixed_equals_pure_exact_sha_seed_unit_set"])
        and filter_ok
        and bool(target_pure["pass"])
        and (True if target_mixed is None else bool(target_mixed["pass"]))
    )
    return {
        "schema": "t34-watch-replay-mac-qualification/v1",
        "verdict": "QUALIFIED" if overall_pass else "NOT_QUALIFIED",
        "qualified_for_controller": controller,
        "pure_bank_dir": str(pure_bank),
        "mixed_bank_dir": None if resolved_mixed_bank is None else str(resolved_mixed_bank),
        "bars_contract": {
            "cold_readiness_seconds_lt": BAR_COLD_READY_SECONDS,
            "unpaced_ticks": UNPACED_TICKS,
            "paced_24fps_ticks": PACED_TICKS,
            "warmup_ticks": WARMUP_TICKS,
            "update_p95_ms_le": BAR_UPDATE_P95_MS,
            "step_p99_ms_le": BAR_STEP_P99_MS,
            "paced_24fps_overruns_eq": BAR_PACED_OVERRUNS,
            "paced_24fps_budget_ms": BAR_PACED_BUDGET_MS,
            "reference_doc": "docs/research/t15_mixed_roster_qualification_2026-10-17.md",
            "t15_watch6_baseline_update_p95_ms": 7.792,
            "t15_watch6_baseline_full_tick_p95_ms": 8.635,
        },
        "environment": env_meta,
        "bank_inventory_audit": inventory,
        "targets": targets,
        "not_measured": [
            (
                "Real browser Canvas/DOM rendering and WebGL/2D paint latency "
                "(measured FastAPI/TestClient HTTP+WebSocket DTO serialization and delivery, "
                "not Chromium/Safari GPU frame compositing)."
            ),
            (
                "Multi-hour full-bank endurance (verified all line-1 headers + sidecars, "
                "adversarial non-target line-1 skip proof, cold readiness, 1,500 unpaced + 120 "
                "paced-24fps ticks under continuous worst-case concurrent .watch.gz "
                "verification, and mid-stream nonblocking session rotation)."
            ),
            (
                "Live port-8000 running snake-web server process "
                "(intentionally left untouched per operational constraints)."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        choices=list(SUPPORTED_CONTROLLERS),
        default=TARGET_CONTROLLER,
        help="Replay controller to qualify (default: exact-stop-l2-prime).",
    )
    parser.add_argument(
        "--bank",
        "--bank-132",
        dest="bank",
        type=Path,
        default=DEFAULT_BANK_132,
        help="Controller-pure bank directory to qualify directly.",
    )
    parser.add_argument(
        "--mixed-bank",
        "--bank-168",
        dest="mixed_bank",
        type=Path,
        default=DEFAULT_BANK_168,
        help="Mixed bank directory for filtered-over-mixed parity and serving qualification.",
    )
    parser.add_argument(
        "--no-mixed-bank",
        action="store_true",
        help="Qualify only --bank without a mixed bank parity target.",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    mixed = None if args.no_mixed_bank else args.mixed_bank
    report = run_qualification(
        mixed,
        args.bank,
        controller=args.controller,
        bank=args.bank,
        mixed_bank=mixed,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    mixed_pass = (
        report["targets"]["mixed_bank_filtered"]["pass"]
        if "mixed_bank_filtered" in report["targets"]
        else None
    )
    pure_pass = report["targets"]["pure_bank_direct"]["pass"]
    print(
        f"verdict={report['verdict']} "
        f"controller={args.controller} "
        f"mixed_pass={mixed_pass} "
        f"pure_pass={pure_pass} "
        f"out={args.json_out}"
    )
    if report["verdict"] != "QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
