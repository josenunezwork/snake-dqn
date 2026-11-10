"""Mac engineering gate and bank tooling for opt-in long Watch sessions (H=14,400).

This module implements:
1. Bank verification and atomic installation helpers for both default ``H=6,000``
   and opt-in ``H=14,400`` product banks without mutating the served default bank.
2. Seed-disjointness verification between the 132-session ``H=6,000`` bank
   (``t34-watch-replay-l1-os-20261106-bank0{1,2,3}-v1`` and
   ``watch-replay-bank-live-132-l1os-v1``) and the 12-session ``H=14,400`` G4 pilot
   (``watch-replay-bank-long-12-l1os-h14400-pilot-v1``).
3. The 0-unit Mac engineering gate (``python -m tools.watch_long_sessions gate``):
   - Generates two short local sessions (``H=6``) through the existing
     ``research/watch_replay/run_l1_os.py::run_unit`` writer and verifies
     mid-session checkpointing + bit-identical deterministic resume.
   - Builds a synthetic ``H=14,400`` product bank through the exact canonical
     gzip writer (``canonical``, ``gzip.GzipFile(mtime=0, compresslevel=6)``,
     ``publish_immutable``) at realistic late-game frame sizes and measures:
     * Compressed, decoded, and max-line byte sizes against ``MAX_COMPRESSED``
       (128 MiB), ``MAX_LONG_DECODED`` (1,228.8 MiB; ``MAX_DECODED`` remains
       512 MiB for default ``H=6,000``), and ``MAX_LINE`` (2 MiB).
     * Peak heap delta (``tracemalloc``), pinned anonymous ``mmap`` bytes, and
       process RSS (``ru_maxrss``) across active + prefetched ``ReplayBank`` slots.
     * Synchronous ``verify_payload`` time, cooperative background prefetch
       ``verify_payload(..., stop=Event())`` time vs the 600 s (10 min at 24 fps)
       playback window, cold first-pick latency, and non-blocking rotation.
     * Default ``H=6,000`` byte-identical behavior and <1 ms line-1 header skip
       when ``H=14,400`` files coexist in a bank directory with ``allow_long=False``.
   - Scans the installed 132-session ``H=6,000`` bank (when present) and computes
     both linear (2.40x) and late-game-weighted superlinear extrapolations
     to ``H=14,400``.
   - Emits the exact specification for the 12-session ``H=14,400`` G4 pilot package.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import resource
import shutil
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

from research.watch_replay import seed_audit as replay_seed_audit
from research.watch_replay.prepare_l1_os import G4_RATE, L1_OS_BANK_NAMESPACES
from research.watch_replay.run import CONTROLLERS, allocation, identity
from tools.ops.package_spec import publish_immutable, unit_id
from web.backend import watch_replay
from web.backend.watch_replay import (
    B8_SHA,
    DEFAULT_HORIZON,
    FRP3_SHA,
    LONG_HORIZON,
    MAX_COMPRESSED,
    MAX_DECODED,
    MAX_LINE,
    MAX_LONG_DECODED,
    PRODUCT_HORIZONS,
    SCHEMA,
    Recording,
    ReplayBank,
    canonical,
    max_decoded_bytes,
    peek_header,
    resolve_replay_horizons,
    validate_frame,
    validate_header,
    verify_payload,
)

PRODUCT_CONTROLLER = "exact-stop-l1-os"

ENV_INSTALLED_H6000_BANK = "SNAKE_WATCH_INSTALLED_H6000_BANK"
REPO_ROOT = Path(os.path.abspath(__file__)).parents[1]
INSTALLED_H6000_BANK_DEFAULT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/watch-replay-bank-live-132-l1os-installed"
)


def _resolve_default_installed_bank() -> Path:
    """Return the default installed H=6,000 bank path, allowing an env override."""
    override = os.environ.get(ENV_INSTALLED_H6000_BANK, "").strip()
    return Path(override) if override else INSTALLED_H6000_BANK_DEFAULT


def _repo_relpath(path: Path) -> str:
    """Format a path relative to REPO_ROOT using portable POSIX separators."""
    base = os.path.abspath(REPO_ROOT)
    target = os.path.abspath(path if path.is_absolute() else REPO_ROOT / path)
    return Path(os.path.relpath(target, base)).as_posix()


INSTALLED_H6000_NAMESPACE = "watch-replay-bank-live-132-l1os-v1"
INSTALLED_H6000_COUNT = 132
INSTALLED_H6000_SUBBANK_COUNT = 44

PILOT_NAMESPACE = "watch-replay-bank-long-12-l1os-h14400-pilot-v1"
PILOT_COUNT = 12
PILOT_HORIZON = LONG_HORIZON
PILOT_CONTROLLER = PRODUCT_CONTROLLER
PILOT_CHECKPOINT_FRAMES = 1800
PILOT_MAX_SESSION_BYTES = 96 * 1024**2  # 96 MiB (raised from 40 MiB for H=14,400)


def long_allocation(
    namespace: str,
    count: int,
    controller: str = PRODUCT_CONTROLLER,
    horizon: int = LONG_HORIZON,
) -> dict[str, Any]:
    """Deterministic seed table for Watch sessions up to H=14,400.

    ``research/watch_replay/run.py::allocation`` is frozen at ``horizon <= 6000``
    because ``run.py`` is in ``watch6_l1_confirm``'s pinned science paths. This helper
    uses the identical SHA-256 seed derivation and contract schema while allowing
    ``1 <= horizon <= 14,400``.
    """
    if not namespace or "/" in namespace or " " in namespace:
        raise ValueError("namespace must be a single path token")
    if count <= 0 or count > 256:
        raise ValueError("count must be in 1..256")
    if controller not in CONTROLLERS:
        raise ValueError(f"unsupported controller {controller!r}")
    if not 1 <= horizon <= LONG_HORIZON:
        raise ValueError("invalid horizon")
    if horizon <= DEFAULT_HORIZON and count <= 128:
        return allocation(namespace, count, controller, horizon)
    rows = []
    for index in range(count):
        material = f"{namespace}/{index}".encode()
        seed = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
        rows.append({"index": index, "seed": seed})
    return {
        "schema": SCHEMA,
        "namespace": namespace,
        "controller": controller,
        "horizon": horizon,
        "world": "watch6-mechanics-v2-continuous-respawn",
        "protocol_version": 2,
        "checkpoints": {"hero": B8_SHA, "opponents": FRP3_SHA},
        "rows": rows,
    }


def build_synthetic_frame(
    tick: int,
    *,
    seed: int = 1770658101,
    snake_lengths: tuple[int, ...] = (380, 210, 150, 120, 90, 80),
) -> dict[str, Any]:
    """Build a protocol-valid Watch v2 frame with configurable snake body lengths."""
    snakes = []
    for s_idx, length in enumerate(snake_lengths):
        # Model realistic frame-to-frame snake motion: trajectories shift every 12 ticks,
        # matching real Watch gzip locality (~45-55 MiB compressed for H=14,400) while
        # producing ~37.3 KiB late-game lines (matching installed bank 31-42 KiB/frame).
        phase = tick // 12
        segments = [
            [
                100.0 + ((phase + s_idx * 19 + j * 3) % 120) * (10.0 + ((j + s_idx) % 11) / 7.0),
                80.0
                + ((phase * 3 + s_idx * 13 + j * 5) % 65) * (10.0 + ((j * 3 + s_idx) % 13) / 9.0),
            ]
            for j in range(length)
        ]
        snakes.append(
            {
                "id": s_idx,
                "segments": segments,
                "head": segments[0] if segments else None,
                "length": int(length),
                "alive": True,
                "boosting": False,
                "color": [0, 255, 136] if s_idx == 0 else [255, 85, 85],
                "is_hero": s_idx == 0,
                "name": f"Snake {s_idx}",
            }
        )
    food_count = 100 if sum(snake_lengths) >= 400 else 30
    food_phase = tick // 8
    food = [
        [
            50.0 + ((food_phase * 7 + k * 11) % 130) * (10.0 + (k % 7) / 3.0),
            50.0 + ((food_phase * 5 + k * 17) % 70) * (10.0 + (k % 5) / 7.0),
        ]
        for k in range(food_count)
    ]
    action = tick % 6
    return {
        "type": "frame",
        "protocol_version": 2,
        "frame": tick,
        "mechanics_version": 2,
        "paused": False,
        "viewer_count": 0,
        "obs_spec": "ego2s-bv8",
        "architecture": "CNN",
        "checkpoint_name": "b8_s3_t7_20261016.pth",
        "serving_contract": None,
        "play": None,
        "hero_raster": None,
        "netviz": None,
        "arena": {
            "width": 1450,
            "height": 830,
            "segment": 10,
            "wall": 10,
            "arena_type": "rectangular",
        },
        "snakes": snakes,
        "food": food,
        "stats": {
            "alive": 6,
            "best_length": int(snake_lengths[0]),
            "deaths": 0,
            "epsilon": 0.0,
            "food_eaten": max(0, int(snake_lengths[0]) - 3),
            "frame": tick,
            "kills": 0,
            "loss": None,
        },
        "session": {
            "mode": "watch",
            "watch_roster": "mixed",
            "hero_id": 0,
            "playing": True,
            "seed": seed,
            "fps": 24,
            "safety_veto": {
                "active": True,
                "variant": "v8",
                "controller": PRODUCT_CONTROLLER,
            },
        },
        "inspector": {
            "action_labels": [
                "Straight",
                "Left",
                "Right",
                "Boost Straight",
                "Boost Left",
                "Boost Right",
            ],
            "chosen": action,
            "executed_action": action,
            "replay_bare_action": action,
            "replay_search_override": False,
            "q_values": [0.42, 0.31, -0.12, 0.05, -0.20, -0.35],
            "state": [0.125] * 60,
            "free_space": [0.85, 0.72, 0.64],
        },
    }


_SYNTHETIC_FRAME_STREAM_CACHE: dict[
    tuple[int, tuple[int, ...], int], tuple[bytes, int, int, float]
] = {}


def write_synthetic_session(
    out_dir: Path,
    *,
    namespace: str,
    count: int,
    index: int,
    horizon: int,
    controller: str = PRODUCT_CONTROLLER,
    commit: str = "a" * 40,
    fixture: bool | None = None,
    snake_lengths: tuple[int, ...] = (380, 210, 150, 120, 90, 80),
    unique_frames: int = 64,
) -> dict[str, Any]:
    """Write a canonical `.watch.gz` session through the exact product writer."""
    decoded_limit = max_decoded_bytes(allow_long=(horizon == LONG_HORIZON), horizon=horizon)
    contract = {
        **long_allocation(namespace, count, controller, horizon),
        "commit": commit,
    }
    row = contract["rows"][index]
    is_fixture = (horizon not in PRODUCT_HORIZONS) if fixture is None else fixture
    stats = {
        "hero_mass_integral": float(snake_lengths[0]),
        "hero_deaths": 1,
        "max_hero_length": int(snake_lengths[0]),
        "decisions": int(horizon),
        "search_roots": int(horizon // 5),
        "interventions": int(horizon // 20),
        "mirror_identity_frames": int(horizon),
    }
    header = {
        "schema": SCHEMA,
        "protocol_version": 2,
        "world": "watch6-mechanics-v2-continuous-respawn",
        "unit_id": unit_id(namespace, "sessions", row["index"]),
        "index": row["index"],
        "seed": row["seed"],
        "horizon": horizon,
        "controller": controller,
        "checkpoint_shas": {"hero": B8_SHA, "opponents": FRP3_SHA},
        "source_commit": commit,
        "allocation": identity(contract),
        "fixture": is_fixture,
        "stats": stats,
    }
    header_bytes = canonical(header)
    header_gz = gzip.compress(header_bytes, compresslevel=6, mtime=0)

    cache_key = (horizon, tuple(snake_lengths), unique_frames)
    cached = _SYNTHETIC_FRAME_STREAM_CACHE.get(cache_key)
    if cached is None:
        num_templates = max(1, min(horizon, unique_frames))
        template_parts: list[list[bytes]] = []
        for k in range(num_templates):
            sample = build_synthetic_frame(
                k + 1,
                seed=1770658101,
                snake_lengths=snake_lengths,
            )
            if k == 0:
                validate_frame(sample, 1)
            sample["frame"] = 0
            sample["stats"]["frame"] = 0
            parts = canonical(sample).split(b'"frame":0')
            if len(parts) != 3:
                raise RuntimeError("expected exactly 2 frame markers in canonical template")
            template_parts.append(parts)

        frames_decoded = 0
        frames_peak_line = 0
        t_compress = time.perf_counter()
        buffer = io.BytesIO()
        with gzip.GzipFile(
            fileobj=buffer, mode="wb", filename="", mtime=0, compresslevel=6
        ) as zipped:
            chunk: list[bytes] = []
            for frame_idx in range(horizon):
                p0, p1, p2 = template_parts[frame_idx % num_templates]
                tag = f'"frame":{frame_idx + 1}'.encode("ascii")
                line = p0 + tag + p1 + tag + p2
                frames_decoded += len(line)
                if len(line) > frames_peak_line:
                    frames_peak_line = len(line)
                if len(line) > MAX_LINE or frames_decoded > decoded_limit:
                    raise ValueError("synthetic replay exceeds decoded/line limit")
                chunk.append(line)
                if len(chunk) == 256:
                    zipped.write(b"".join(chunk))
                    chunk.clear()
            if chunk:
                zipped.write(b"".join(chunk))
        frames_gz = buffer.getvalue()
        compress_s = time.perf_counter() - t_compress
        cached = (frames_gz, frames_decoded, frames_peak_line, compress_s)
        _SYNTHETIC_FRAME_STREAM_CACHE[cache_key] = cached
    else:
        frames_gz, frames_decoded, frames_peak_line, compress_s = cached

    decoded_bytes = len(header_bytes) + frames_decoded
    peak_line_bytes = max(len(header_bytes), frames_peak_line)
    if peak_line_bytes > MAX_LINE or decoded_bytes > decoded_limit:
        raise ValueError("synthetic replay exceeds decoded/line limit")

    started = time.perf_counter()
    payload = header_gz + frames_gz
    if len(payload) > MAX_COMPRESSED:
        raise ValueError("synthetic replay exceeds MAX_COMPRESSED")
    path = out_dir / f"{row['index']:06d}.watch.gz"
    published = publish_immutable(path, payload)
    write_seconds = max(time.perf_counter() - started, compress_s)
    return {
        **published,
        "index": row["index"],
        "seed": row["seed"],
        "horizon": horizon,
        "allocation": header["allocation"],
        "unit_id": header["unit_id"],
        "compressed_bytes": len(payload),
        "decoded_bytes": decoded_bytes,
        "peak_line_bytes": peak_line_bytes,
        "write_seconds": write_seconds,
    }


def _verify_bank_directory_with_rows(
    bank_dir: Path,
    *,
    expected_horizon: int | None = None,
    allow_long: bool = True,
    full_payload: bool = False,
    verify_sha: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify all `.watch.gz` recordings in a bank directory and return summary + rows."""
    direct_files = sorted(bank_dir.glob("*.watch.gz"))
    if direct_files:
        sessions_dir = bank_dir
        files = direct_files
    elif (bank_dir / "sessions").is_dir():
        sessions_dir = bank_dir / "sessions"
        files = sorted(sessions_dir.glob("*.watch.gz"))
    else:
        sessions_dir = bank_dir
        files = []
    if not files:
        raise ValueError(f"no .watch.gz files found in {sessions_dir}")
    horizons = frozenset({expected_horizon}) if expected_horizon is not None else None
    seen_unit_ids: set[str] = set()
    seen_alloc_indices: set[tuple[str, int]] = set()
    seen_seeds: set[int] = set()
    allocations: set[str] = set()
    controllers: set[str] = set()
    found_horizons: set[int] = set()
    compressed_sizes: list[int] = []
    verified_rows: list[dict[str, Any]] = []

    started = time.perf_counter()
    for path in files:
        sha_path = Path(str(path) + ".sha256")
        complete_path = Path(str(path) + ".complete")
        if not sha_path.is_file() or not complete_path.is_file():
            raise ValueError(f"missing sidecar for {path.name}")
        sha_tokens = sha_path.read_text(encoding="utf-8").split()
        if (
            len(sha_tokens) != 2
            or sha_tokens[1] != path.name
            or len(sha_tokens[0]) != 64
            or any(ch not in "0123456789abcdef" for ch in sha_tokens[0])
        ):
            raise ValueError(f"invalid .sha256 sidecar format for {path.name}")
        size = path.stat().st_size
        compressed_sizes.append(size)
        if full_payload:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if sha_tokens != [digest, path.name]:
                raise ValueError(f"SHA mismatch on {path.name}")
            header = verify_payload(
                data,
                product=True,
                allow_long=allow_long,
                horizons=horizons,
            )
        else:
            if verify_sha:
                with path.open("rb") as fh:
                    digest = hashlib.file_digest(fh, "sha256").hexdigest()
                if sha_tokens != [digest, path.name]:
                    raise ValueError(f"SHA mismatch on {path.name}")
            header = peek_header(
                path,
                product=True,
                allow_long=allow_long,
                horizons=horizons,
            )
        uid = str(header["unit_id"])
        alloc = str(header.get("allocation", ""))
        idx = int(header["index"])
        seed = int(header["seed"])
        if uid in seen_unit_ids:
            raise ValueError(f"duplicate unit_id {uid} in {bank_dir}")
        if (alloc, idx) in seen_alloc_indices:
            raise ValueError(f"duplicate allocation/index {(alloc, idx)} in {bank_dir}")
        if seed in seen_seeds:
            raise ValueError(f"duplicate seed {seed} in {bank_dir}")
        seen_unit_ids.add(uid)
        seen_alloc_indices.add((alloc, idx))
        seen_seeds.add(seed)
        allocations.add(alloc)
        controllers.add(str(header["controller"]))
        found_horizons.add(int(header["horizon"]))
        verified_rows.append(
            {
                "name": path.name,
                "unit_id": uid,
                "allocation": header.get("allocation"),
                "index": idx,
                "seed": seed,
                "horizon": int(header["horizon"]),
                "compressed_bytes": size,
            }
        )
    elapsed = time.perf_counter() - started
    sorted_sizes = sorted(compressed_sizes)
    n = len(sorted_sizes)
    summary = {
        "bank_dir": str(bank_dir),
        "sessions_dir": str(sessions_dir),
        "count": n,
        "horizons": sorted(found_horizons),
        "controllers": sorted(controllers),
        "allocations": sorted(allocations),
        "total_compressed_bytes": sum(sorted_sizes),
        "min_compressed_bytes": sorted_sizes[0],
        "median_compressed_bytes": sorted_sizes[n // 2],
        "mean_compressed_bytes": sum(sorted_sizes) / n,
        "p95_compressed_bytes": sorted_sizes[min(n - 1, int(n * 0.95))],
        "max_compressed_bytes": sorted_sizes[-1],
        "verified_files": [row["name"] for row in verified_rows],
        "full_payload_verified": full_payload,
        "elapsed_seconds": elapsed,
    }
    return summary, verified_rows


def verify_bank_directory(
    bank_dir: Path,
    *,
    expected_horizon: int | None = None,
    allow_long: bool = True,
    full_payload: bool = False,
    verify_sha: bool = False,
) -> dict[str, Any]:
    """Verify all `.watch.gz` recordings in a bank directory."""
    summary, _ = _verify_bank_directory_with_rows(
        bank_dir,
        expected_horizon=expected_horizon,
        allow_long=allow_long,
        full_payload=full_payload,
        verify_sha=verify_sha,
    )
    return summary


def install_long_bank(
    source_dir: Path,
    dest_dir: Path,
    *,
    expected_count: int = PILOT_COUNT,
    expected_horizon: int = LONG_HORIZON,
    full_payload: bool = True,
    require_contract: bool = True,
) -> dict[str, Any]:
    """Verify and atomically install an opt-in long-session Watch bank."""
    if dest_dir.exists():
        raise FileExistsError(f"destination already exists: {dest_dir}")
    summary, verified_rows = _verify_bank_directory_with_rows(
        source_dir,
        expected_horizon=expected_horizon,
        allow_long=True,
        full_payload=full_payload,
        verify_sha=True,
    )
    if summary["count"] != expected_count:
        raise ValueError(f"expected {expected_count} sessions, found {summary['count']}")
    if summary["horizons"] != [expected_horizon]:
        raise ValueError(f"expected horizon {[expected_horizon]}, found {summary['horizons']}")
    if summary["controllers"] != [PRODUCT_CONTROLLER]:
        raise ValueError(
            f"expected controller {[PRODUCT_CONTROLLER]}, found {summary['controllers']}"
        )
    if len(summary["allocations"]) != 1 or not summary["allocations"][0]:
        raise ValueError(f"expected single non-empty allocation, found {summary['allocations']}")
    src_sessions = Path(summary["sessions_dir"])
    contract_src = source_dir / "contract.json"
    if not contract_src.is_file() and (src_sessions / "contract.json").is_file():
        contract_src = src_sessions / "contract.json"
    contract_verified = False
    if contract_src.is_file():
        try:
            contract = json.loads(contract_src.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid contract.json: {exc}") from exc
        if not isinstance(contract, dict):
            raise ValueError("contract.json must be a JSON object")
        rows_raw = contract.get("rows")
        try:
            contract_alloc = identity(contract)
            contract_pairs = (
                sorted((int(r["index"]), int(r["seed"])) for r in rows_raw)
                if isinstance(rows_raw, list)
                else None
            )
        except Exception as exc:
            raise ValueError(f"invalid contract.json fields: {exc}") from exc
        verified_pairs = sorted((int(r["index"]), int(r["seed"])) for r in verified_rows)
        if (
            contract.get("controller") != PRODUCT_CONTROLLER
            or contract.get("horizon") != expected_horizon
            or contract_pairs is None
            or len(contract_pairs) != expected_count
            or any(r.get("allocation") != contract_alloc for r in verified_rows)
            or contract_pairs != verified_pairs
        ):
            raise ValueError("contract.json mismatch against verified bank recordings")
        contract_verified = True
    elif require_contract:
        raise ValueError(f"missing contract.json in {source_dir} (pass --no-contract to override)")
    staging = dest_dir.parent / f".{dest_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for name in summary["verified_files"]:
            gz_path = src_sessions / name
            sha_path = Path(str(gz_path) + ".sha256")
            complete_path = Path(str(gz_path) + ".complete")
            for item in (gz_path, sha_path, complete_path):
                target = staging / item.name
                try:
                    os.link(item, target)
                except OSError:
                    shutil.copy2(item, target)
        if contract_src.is_file():
            shutil.copy2(contract_src, staging / "contract.json")
        os.replace(staging, dest_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        **summary,
        "contract_verified": contract_verified,
        "installed_dir": str(dest_dir),
    }


def _scan_bank_disk_seeds(bank_dir: Path) -> set[int]:
    """Scan top-level and sessions/ subdirectory .watch.gz headers and return their seed union."""
    seeds: set[int] = set()
    if not bank_dir.is_dir():
        return seeds
    scan_dirs = [bank_dir]
    if (bank_dir / "sessions").is_dir():
        scan_dirs.append(bank_dir / "sessions")
    for scan_dir in scan_dirs:
        for path in sorted(scan_dir.glob("*.watch.gz")):
            hdr = peek_header(path, product=True)
            seeds.add(int(hdr["seed"]))
    return seeds


def audit_pilot_seeds(
    *,
    pilot_namespace: str = PILOT_NAMESPACE,
    pilot_count: int = PILOT_COUNT,
    pilot_horizon: int = PILOT_HORIZON,
    installed_namespace: str = INSTALLED_H6000_NAMESPACE,
    installed_count: int = INSTALLED_H6000_COUNT,
    installed_bank_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify that the 12-session H=14,400 pilot seeds are disjoint from all installed H=6,000
    bank seeds and all historical Watch/T34 study namespaces."""
    default_bank_dir = _resolve_default_installed_bank()
    if installed_bank_dir is None:
        installed_bank_dir = default_bank_dir
    pilot_contract = long_allocation(
        pilot_namespace, pilot_count, PRODUCT_CONTROLLER, pilot_horizon
    )
    pilot_seeds = [row["seed"] for row in pilot_contract["rows"]]
    pilot_seed_set = set(pilot_seeds)
    if len(pilot_seed_set) != pilot_count:
        raise ValueError("duplicate internal seeds in pilot namespace")

    # 1. Check the 3 installed 44-session L1-OS sub-banks (3 x 44 = 132 sessions).
    subbank_seeds: set[int] = set()
    for ns in L1_OS_BANK_NAMESPACES:
        sub_contract = allocation(
            ns, INSTALLED_H6000_SUBBANK_COUNT, PRODUCT_CONTROLLER, DEFAULT_HORIZON
        )
        subbank_seeds.update(row["seed"] for row in sub_contract["rows"])

    # 2. Check the single-namespace alias (watch-replay-bank-live-132-l1os-v1).
    alias_contract = long_allocation(
        installed_namespace, installed_count, PRODUCT_CONTROLLER, DEFAULT_HORIZON
    )
    alias_seeds = {row["seed"] for row in alias_contract["rows"]}

    # 3. Check all prior namespaces registered in research/watch_replay/seed_audit.py.
    prior_namespace_seeds: set[int] = set()
    for ns, cap in (
        *replay_seed_audit.KNOWN_NAMESPACE_PREFIXES,
        *replay_seed_audit.LATER_STUDY_NAMESPACES,
        *replay_seed_audit.BANK_V2_NAMESPACES,
    ):
        for idx in range(cap):
            prior_namespace_seeds.add(replay_seed_audit.seed_for(ns, idx))

    # 4. If the installed bank (and default 132-session bank when distinct) is present on disk,
    # verify against headers across both top-level and sessions/ layouts.
    disk_installed_seeds = _scan_bank_disk_seeds(installed_bank_dir)
    same_as_default = os.path.abspath(installed_bank_dir) == os.path.abspath(default_bank_dir) or (
        installed_bank_dir.exists()
        and default_bank_dir.exists()
        and installed_bank_dir.resolve() == default_bank_dir.resolve()
    )
    default_bank_seeds = (
        set(disk_installed_seeds) if same_as_default else _scan_bank_disk_seeds(default_bank_dir)
    )
    default_bank_present = default_bank_dir.is_dir() and len(default_bank_seeds) > 0

    all_compared = (
        subbank_seeds
        | alias_seeds
        | prior_namespace_seeds
        | disk_installed_seeds
        | default_bank_seeds
    )
    overlap = sorted(all_compared.intersection(pilot_seed_set))
    if overlap:
        raise ValueError(f"seed collision between pilot and prior/installed banks: {overlap}")
    return {
        "pilot_namespace": pilot_namespace,
        "pilot_count": pilot_count,
        "pilot_horizon": pilot_horizon,
        "pilot_allocation": identity(pilot_contract),
        "pilot_rows": pilot_contract["rows"],
        "installed_bank_dir": _repo_relpath(installed_bank_dir),
        "installed_subbank_namespaces": list(L1_OS_BANK_NAMESPACES),
        "installed_subbank_seed_count": len(subbank_seeds),
        "installed_disk_seed_count": len(disk_installed_seeds),
        "default_bank_dir": _repo_relpath(default_bank_dir),
        "default_bank_present": default_bank_present,
        "default_bank_seed_count": len(default_bank_seeds),
        "installed_namespace": installed_namespace,
        "installed_count": installed_count,
        "installed_allocation": identity(alias_contract),
        "total_prior_seeds_checked": len(all_compared),
        "seed_overlap_count": len(overlap),
        "disjoint_from_installed_132": len(overlap) == 0,
    }


def build_pilot_package_spec(
    *,
    installed_bank_dir: Path | None = None,
) -> dict[str, Any]:
    """Return the exact specification for the 12-session H=14,400 G4 pilot package."""
    seed_audit = audit_pilot_seeds(installed_bank_dir=installed_bank_dir)
    return {
        "package_dir": "research/watch_replay_long_pilot",
        "purpose": (
            "Exploration-lane 12-session H=14,400 G4 pilot Watch replay bank (exact-stop-l1-os)"
        ),
        "why_new_package": (
            "research/watch_replay/{engine.py,product.py,run.py,uploads.py} are bound by "
            "research/watch6_l1_confirm/protocol.py::science_paths() and research/ is frozen "
            "for this engineering gate. In addition, research/watch_replay/prepare_l1_os.py "
            "hard-requires contract['horizon'] == 6000 and enforces MAX_SESSION_BYTES = 40 MiB "
            "(whereas H=14,400 p95 compressed size is ~54.6-64.0 MiB and max is ~68.6-80.4 MiB)."
        ),
        "namespace": PILOT_NAMESPACE,
        "count": PILOT_COUNT,
        "horizon": PILOT_HORIZON,
        "controller": PILOT_CONTROLLER,
        "allocation": seed_audit["pilot_allocation"],
        "seed_disjointness": {
            "compared_against_installed_subbanks": seed_audit["installed_subbank_namespaces"],
            "compared_against_alias": INSTALLED_H6000_NAMESPACE,
            "installed_bank_dir": seed_audit["installed_bank_dir"],
            "installed_disk_seed_count": seed_audit["installed_disk_seed_count"],
            "default_bank_dir": seed_audit["default_bank_dir"],
            "default_bank_present": seed_audit["default_bank_present"],
            "default_bank_seed_count": seed_audit["default_bank_seed_count"],
            "total_prior_seeds_checked": seed_audit["total_prior_seeds_checked"],
            "overlap_count": seed_audit["seed_overlap_count"],
            "disjoint": seed_audit["disjoint_from_installed_132"],
            "pilot_seeds": [r["seed"] for r in seed_audit["pilot_rows"]],
        },
        "checkpoints": {
            "mechanism": (
                "research/world_checkpoint (job.py / job.sh) + research/watch_replay/run_l1_os.py "
                "intra-session checkpointing (save_checkpoint / load_latest_checkpoint)"
            ),
            "cadence_frames": PILOT_CHECKPOINT_FRAMES,
            "expected_checkpoints_per_session": [1800, 3600, 5400, 7200, 9000, 10800, 12600],
            "signals_handled": ["SIGTERM", "SIGUSR1"],
            "max_work_lost_minutes_on_interruption": 45.0,
            "resume_determinism": (
                "Bit-identical frames_sha256 verified on resume before frame 14,400 publication"
            ),
        },
        "size_caps": {
            "max_session_compressed_bytes": PILOT_MAX_SESSION_BYTES,
            "max_reader_compressed_bytes": MAX_COMPRESSED,
            "max_reader_default_decoded_bytes": MAX_DECODED,
            "max_reader_decoded_bytes": MAX_LONG_DECODED,
            "max_reader_line_bytes": MAX_LINE,
        },
        "inputs": [
            {
                "remote_path": "inputs/contract.json",
                "description": (
                    "Frozen long_allocation("
                    "namespace='watch-replay-bank-long-12-l1os-h14400-pilot-v1', "
                    "count=12, controller='exact-stop-l1-os', horizon=14400) + pinned commit SHA"
                ),
            },
            {
                "remote_path": "inputs/verified-ledger.json",
                "description": (
                    "Mac-verified ledger (empty units=[] on attempt 0; "
                    "populated from path/size/sha256 only)"
                ),
            },
            {
                "remote_path": "inputs/watch-replay-metadata.tar.gz",
                "description": (
                    "Stage-metadata bundle (research/watch_replay/uploads.py::pack_inputs) "
                    "plus public checkpoint pins for b8_s3_t7_20261016.pth and "
                    "frp3_m3_s12_u60000_20261005.pth"
                ),
                "checkpoint_shas": {"hero": B8_SHA, "opponents": FRP3_SHA},
            },
        ],
        "manifest_resume_declaration": {
            "resume": {
                "unit_kind": "sessions",
                "ledger_path": "inputs/verified-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            }
        },
        "compute_and_expected_units": {
            "work_type": (
                "CPU-only (SNAKE_DQN_DEVICE=cpu); G4 (48 vCPU) for 12-worker concurrent pilot"
            ),
            "sessions": 12,
            "frames_per_session": 14400,
            "total_frames": 172800,
            "colab_g4_48vcpu_pilot": {
                "runtime_tier": "G4 (48 vCPU)",
                "workers": 12,
                "threads_per_worker": 1,
                "waves": 1,
                "rate_units_per_hour_g4": G4_RATE,
                "rate_source_1": (
                    "research/watch_replay/prepare_l1_os.py (G4_RATE = 10.05 u/h; 48-worker "
                    "calibration across 19 clean-start exact-stop-l1-os H=6,000 sessions "
                    "measured mean 2.289 h, max 2.726 h per session + 25 min staging)"
                ),
                "rate_source_2": (
                    "docs/research/advice/2026-11-06_long_sessions_astra.md §3 (owner-supplied "
                    "6.0-8.0 worker-hours/session at H=14,400 on G4 at 10.05 u/h accounting for "
                    "superlinear late-game exact-search scaling min(max(300, L), 1200) "
                    "for L >= 800)"
                ),
                "linear_2_4x_mean_wall_hours": 5.91,
                "linear_2_4x_max_wall_hours": 6.96,
                "linear_2_4x_expected_units_range": [59.4, 69.9],
                "superlinear_expected_wall_hours_range": [6.0, 8.0],
                "expected_units_forecast_range": [60.3, 80.4],
                "expected_units_forecast_midpoint": 70.4,
                "base_unit_cap": 80.4,
                "effective_unit_cap_2x_rule": 160.8,
                "effective_reserve_units_2x_rule": 32.2,
                "base_timeout_minutes": 480,
                "effective_timeout_minutes_2x_rule": 960,
            },
        },
    }


def extrapolate_from_installed_bank(
    installed_dir: Path | None = None,
    *,
    sample_sessions_for_frame_curve: int = 6,
) -> dict[str, Any]:
    """Measure the installed H=6,000 bank and extrapolate compressed/decoded sizes to H=14,400."""
    if installed_dir is None:
        installed_dir = _resolve_default_installed_bank()
    if not installed_dir.exists():
        return {"installed_bank_available": False, "path": str(installed_dir)}
    summary = verify_bank_directory(
        installed_dir,
        expected_horizon=DEFAULT_HORIZON,
        allow_long=False,
        full_payload=False,
    )
    sessions_dir = Path(summary["sessions_dir"])
    files = sorted(sessions_dir.glob("*.watch.gz"))
    size_pairs = sorted((p.stat().st_size, p) for p in files)
    median_path = size_pairs[len(size_pairs) // 2][1]
    max_path = size_pairs[-1][1]

    def _measure_decoded(path: Path) -> dict[str, Any]:
        total = 0
        max_line = 0
        lines = 0
        with gzip.open(path, "rb") as gz:
            for line in gz:
                total += len(line)
                if len(line) > max_line:
                    max_line = len(line)
                lines += 1
        return {
            "name": path.name,
            "compressed_bytes": path.stat().st_size,
            "decoded_bytes": total,
            "max_line_bytes": max_line,
            "lines": lines,
        }

    median_decoded = _measure_decoded(median_path)
    max_decoded = _measure_decoded(max_path)

    if sample_sessions_for_frame_curve <= 1 or len(size_pairs) <= 1:
        sampled = [size_pairs[-1][1]]
    else:
        sampled = [
            size_pairs[round(i * (len(size_pairs) - 1) / (sample_sessions_for_frame_curve - 1))][1]
            for i in range(sample_sessions_for_frame_curve)
        ]
    early_bytes = 0
    mid_bytes = 0
    late_bytes = 0
    for path in sampled:
        with gzip.open(path, "rb") as gz:
            gz.readline()  # header
            for idx, line in enumerate(gz, start=1):
                n = len(line)
                if idx <= 2000:
                    early_bytes += n
                elif idx <= 4000:
                    mid_bytes += n
                else:
                    late_bytes += n
    denom = float(len(sampled) * 2000)
    early_avg = early_bytes / denom
    mid_avg = mid_bytes / denom
    late_avg = late_bytes / denom
    overall_avg = (early_avg + mid_avg + late_avg) / 3.0
    late_to_overall_ratio = late_avg / overall_avg if overall_avg > 0 else 1.0

    linear_scale = LONG_HORIZON / DEFAULT_HORIZON  # 2.40
    # Frames 1..6000 follow overall_avg; frames 6001..14400 (8,400 frames) run in the
    # late-game steady state (with an additional 1.10x safety margin for rare tail lengths).
    superlinear_scale = (
        (DEFAULT_HORIZON * overall_avg) + ((LONG_HORIZON - DEFAULT_HORIZON) * late_avg * 1.10)
    ) / (DEFAULT_HORIZON * overall_avg)

    return {
        "installed_bank_available": True,
        "installed_h6000": {
            "count": summary["count"],
            "min_compressed_mib": round(summary["min_compressed_bytes"] / 1024**2, 3),
            "median_compressed_mib": round(summary["median_compressed_bytes"] / 1024**2, 3),
            "mean_compressed_mib": round(summary["mean_compressed_bytes"] / 1024**2, 3),
            "p95_compressed_mib": round(summary["p95_compressed_bytes"] / 1024**2, 3),
            "max_compressed_mib": round(summary["max_compressed_bytes"] / 1024**2, 3),
            "median_session_decoded_mib": round(median_decoded["decoded_bytes"] / 1024**2, 3),
            "median_session_max_line_kib": round(median_decoded["max_line_bytes"] / 1024, 2),
            "max_session_decoded_mib": round(max_decoded["decoded_bytes"] / 1024**2, 3),
            "max_session_max_line_kib": round(max_decoded["max_line_bytes"] / 1024, 2),
        },
        "frame_size_growth_profile": {
            "sampled_sessions": [p.name for p in sampled],
            "early_1_2000_bytes_per_frame": round(early_avg, 1),
            "mid_2001_4000_bytes_per_frame": round(mid_avg, 1),
            "late_4001_6000_bytes_per_frame": round(late_avg, 1),
            "overall_1_6000_bytes_per_frame": round(overall_avg, 1),
            "late_to_overall_ratio": round(late_to_overall_ratio, 4),
        },
        "extrapolated_h14400": {
            "linear_scale_factor": round(linear_scale, 3),
            "superlinear_late_weighted_scale_factor": round(superlinear_scale, 3),
            "compressed_mib_linear": {
                "mean": round(summary["mean_compressed_bytes"] * linear_scale / 1024**2, 2),
                "p95": round(summary["p95_compressed_bytes"] * linear_scale / 1024**2, 2),
                "max": round(summary["max_compressed_bytes"] * linear_scale / 1024**2, 2),
            },
            "compressed_mib_superlinear": {
                "mean": round(summary["mean_compressed_bytes"] * superlinear_scale / 1024**2, 2),
                "p95": round(summary["p95_compressed_bytes"] * superlinear_scale / 1024**2, 2),
                "max": round(summary["max_compressed_bytes"] * superlinear_scale / 1024**2, 2),
            },
            "decoded_mib_linear": {
                "median": round(median_decoded["decoded_bytes"] * linear_scale / 1024**2, 2),
                "max": round(max_decoded["decoded_bytes"] * linear_scale / 1024**2, 2),
            },
            "decoded_mib_superlinear": {
                "median": round(median_decoded["decoded_bytes"] * superlinear_scale / 1024**2, 2),
                "max": round(max_decoded["decoded_bytes"] * superlinear_scale / 1024**2, 2),
            },
            "max_reader_default_decoded_cap_mib": round(MAX_DECODED / 1024**2, 1),
            "max_reader_long_decoded_cap_mib": round(MAX_LONG_DECODED / 1024**2, 1),
            "decoded_headroom_below_long_cap_mib": {
                "linear_max": round(
                    (MAX_LONG_DECODED - max_decoded["decoded_bytes"] * linear_scale) / 1024**2, 2
                ),
                "superlinear_max": round(
                    (MAX_LONG_DECODED - max_decoded["decoded_bytes"] * superlinear_scale) / 1024**2,
                    2,
                ),
            },
            "fits_reader_max_compressed_128_mib": (
                summary["max_compressed_bytes"] * superlinear_scale < MAX_COMPRESSED
            ),
            "fits_reader_max_long_decoded_1228_8_mib": (
                max_decoded["decoded_bytes"] * superlinear_scale < MAX_LONG_DECODED
            ),
            "exceeds_legacy_prepare_40_mib_cap": (
                summary["p95_compressed_bytes"] * linear_scale > 40 * 1024**2
            ),
            "pilot_package_session_cap_mib": round(PILOT_MAX_SESSION_BYTES / 1024**2, 1),
        },
    }


def run_short_local_generator_check(
    *,
    horizon: int = 6,
    count: int = 2,
) -> dict[str, Any]:
    """Generate 2 short sessions locally with the real `run_l1_os.run_unit` generator
    and verify mid-session checkpointing + bit-identical resume."""
    from research.watch_replay import run_l1_os

    contract = {
        **long_allocation(
            "watch-long-session-mac-gate-short-v1", count, PRODUCT_CONTROLLER, horizon
        ),
        "commit": "b" * 40,
    }
    started = time.perf_counter()
    prev_ckpt_dir = os.environ.get("SNAKE_CKPT_DIR")
    if not prev_ckpt_dir:
        os.environ["SNAKE_CKPT_DIR"] = str(Path(__file__).resolve().parents[1] / "saved_snakes")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_uninterrupted = Path(tmp) / "uninterrupted"
            out_resumed = Path(tmp) / "resumed"
            out_uninterrupted.mkdir(parents=True, exist_ok=True)
            out_resumed.mkdir(parents=True, exist_ok=True)

            # Generate 2 short sessions with the existing real generator; checkpoint session 0.
            ckpt_frame = max(1, horizon // 2)
            unit0 = run_l1_os.run_unit(
                contract,
                contract["rows"][0],
                out_uninterrupted,
                lambda: None,
                checkpoint_frames=frozenset({ckpt_frame}),
            )
            unit1 = run_l1_os.run_unit(
                contract,
                contract["rows"][1],
                out_uninterrupted,
                lambda: None,
            )

            ckpt_file = (
                out_uninterrupted / "checkpoints" / "000000" / f"frame-{ckpt_frame:06d}.json"
            )
            if not ckpt_file.is_file():
                raise RuntimeError(f"expected mid-session checkpoint at {ckpt_file}")

            # Resume session 0 from the checkpoint in a fresh output dir and prove bit-identity.
            resumed0 = run_l1_os.run_unit(
                contract,
                contract["rows"][0],
                out_resumed,
                lambda: None,
                checkpoint_dirs=[out_uninterrupted / "checkpoints"],
            )

            p0 = out_uninterrupted / "000000.watch.gz"
            p1 = out_uninterrupted / "000001.watch.gz"
            p0_resumed = out_resumed / "000000.watch.gz"

            sha0 = hashlib.sha256(p0.read_bytes()).hexdigest()
            sha0_resumed = hashlib.sha256(p0_resumed.read_bytes()).hexdigest()
            if sha0 != sha0_resumed or resumed0.get("resumed_from_frame") != ckpt_frame:
                raise RuntimeError("resumed session is not bit-identical to uninterrupted session")

            rec0 = Recording(p0, product=False)
            rec1 = Recording(p1, product=False)
            try:
                frames0 = 1
                while rec0.step():
                    frames0 += 1
                frames1 = 1
                while rec1.step():
                    frames1 += 1
            finally:
                rec0.close()
                rec1.close()
    finally:
        if prev_ckpt_dir is None:
            os.environ.pop("SNAKE_CKPT_DIR", None)
        else:
            os.environ["SNAKE_CKPT_DIR"] = prev_ckpt_dir

    elapsed = time.perf_counter() - started
    return {
        "generator_module": "research.watch_replay.run_l1_os.run_unit",
        "controller": PRODUCT_CONTROLLER,
        "sessions_generated": count,
        "horizon": horizon,
        "checkpoint_frame_tested": ckpt_frame,
        "resumed_from_frame": resumed0.get("resumed_from_frame"),
        "resumed_bit_identical": sha0 == sha0_resumed,
        "uninterrupted_sha256": sha0,
        "resumed_sha256": sha0_resumed,
        "frames_replayed": [frames0, frames1],
        "unit0_compressed_bytes": unit0["size"],
        "unit1_compressed_bytes": unit1["size"],
        "elapsed_seconds": round(elapsed, 3),
    }


def _pick_ready(bank: ReplayBank, timeout_s: float = 30.0) -> Recording | None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        rec = bank.pick()
        if rec is not None or not bank.pending:
            return rec
        time.sleep(0.005)
    raise TimeoutError("timed out waiting for ReplayBank.pick()")


def run_h14400_scale_synthetic_gate() -> dict[str, Any]:
    """Build a synthetic H=14,400 product bank + H=6,000 default bank and measure memory,
    verification/prefetch time, rotation, and default H=6,000 byte-identity."""
    rss_divisor = float(1024**2) if sys.platform == "darwin" else 1024.0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        default_dir = root / "bank_h6000"
        long_dir = root / "bank_h14400"
        mixed_dir = root / "bank_mixed"

        # 1. Build 2 default H=6,000 sessions in default_dir.
        h6000_s0 = write_synthetic_session(
            default_dir,
            namespace="watch-gate-default-h6000-v1",
            count=2,
            index=0,
            horizon=DEFAULT_HORIZON,
            snake_lengths=(260, 150, 110, 90, 70, 60),
        )
        h6000_s1 = write_synthetic_session(
            default_dir,
            namespace="watch-gate-default-h6000-v1",
            count=2,
            index=1,
            horizon=DEFAULT_HORIZON,
            snake_lengths=(260, 150, 110, 90, 70, 60),
        )

        # 2. Build 3 full H=14,400 product sessions in long_dir with realistic late-game
        # line sizes (~36.8 KiB/frame, matching installed H=6,000 bank 31-42 KiB/frame).
        late_game_lengths = (380, 210, 150, 120, 90, 80)
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        h14400_s0 = write_synthetic_session(
            long_dir,
            namespace=PILOT_NAMESPACE,
            count=3,
            index=0,
            horizon=LONG_HORIZON,
            snake_lengths=late_game_lengths,
        )
        h14400_s1 = write_synthetic_session(
            long_dir,
            namespace=PILOT_NAMESPACE,
            count=3,
            index=1,
            horizon=LONG_HORIZON,
            snake_lengths=late_game_lengths,
        )
        h14400_s2 = write_synthetic_session(
            long_dir,
            namespace=PILOT_NAMESPACE,
            count=3,
            index=2,
            horizon=LONG_HORIZON,
            snake_lengths=late_game_lengths,
        )

        p_long_0 = long_dir / "000000.watch.gz"

        # 3. Verify synchronous (unthrottled) verify_payload time on the full H=14,400 recording
        # without tracemalloc overhead so timing reflects true production speed.
        raw_bytes_0 = p_long_0.read_bytes()
        t0 = time.perf_counter()
        hdr0 = verify_payload(raw_bytes_0, product=True, allow_long=True)
        sync_verify_s0 = time.perf_counter() - t0
        del raw_bytes_0

        # 4. Measure production cold first-pick time on ReplayBank(long_dir, allow_long=True)
        # with the exact default COOPERATIVE_YIELD_* settings (no overriding of
        # COOPERATIVE_YIELD_* and no tracemalloc active).
        t_prod_cold = time.perf_counter()
        prod_bank = ReplayBank(long_dir, shuffle_discovery=False, allow_long=True)
        try:
            prod_rec = _pick_ready(prod_bank, timeout_s=180.0)
            cold_pick_production_seconds = time.perf_counter() - t_prod_cold
            assert prod_rec is not None and prod_rec.header["horizon"] == LONG_HORIZON
            prod_rec.close()
        finally:
            prod_bank.close()

        playback_window_seconds = LONG_HORIZON / 24.0  # 600.0 s at 24 fps

        # 5. Verify that default validate_header / verify_payload / Recording / ReplayBank
        # reject H=14,400 when allow_long=False (byte-identical default contract).
        default_rejected_header = False
        try:
            validate_header(hdr0, product=True)
        except ValueError:
            default_rejected_header = True

        default_rejected_peek = False
        try:
            peek_header(p_long_0, product=True)
        except ValueError:
            default_rejected_peek = True

        # 6. Mixed directory test: place H=6,000 (000000) and H=14,400 (000001) in mixed_dir.
        # Verify that default ReplayBank(allow_long=False) skips H=14,400 on line 1 in <5 ms
        # without full payload decompression or warnings, and serves only H=6,000.
        mixed_dir.mkdir(parents=True, exist_ok=True)
        for item in default_dir.glob("000000.watch.gz*"):
            shutil.copy2(item, mixed_dir / item.name)
        for item in long_dir.glob("000001.watch.gz*"):
            shutil.copy2(item, mixed_dir / item.name)

        t_skip = time.perf_counter()
        unselected_skip_check = watch_replay._is_unselected_long_header(
            mixed_dir / "000001.watch.gz"
        )
        skip_header_ms = (time.perf_counter() - t_skip) * 1000.0

        orig_wait = watch_replay.COOPERATIVE_YIELD_MIN_WAIT_SECONDS
        orig_duty = watch_replay.COOPERATIVE_YIELD_DUTY_FACTOR
        try:
            # Production cold-first-pick with default COOPERATIVE_YIELD_* was already measured
            # in step 4 above. Use minimal yield sleep during the full 3-slot tracemalloc
            # verification run so tracemalloc's ~3x CPU overhead does not multiply sleep time.
            watch_replay.COOPERATIVE_YIELD_MIN_WAIT_SECONDS = 0.00005
            watch_replay.COOPERATIVE_YIELD_DUTY_FACTOR = 0.05

            verify_calls_mixed: list[int] = []
            orig_verify = watch_replay.verify_payload

            def _counting_verify(payload: Any, *args: Any, **kwargs: Any) -> dict:
                verify_calls_mixed.append(len(payload))
                return orig_verify(payload, *args, **kwargs)

            watch_replay.verify_payload = _counting_verify  # type: ignore[assignment]
            try:
                mixed_default_bank = ReplayBank(mixed_dir, shuffle_discovery=False)
                try:
                    mixed_rec = _pick_ready(mixed_default_bank)
                    assert mixed_rec is not None
                    mixed_default_horizon = mixed_rec.header["horizon"]
                    try:
                        # With only 1 eligible H=6,000 unit, a second pick returns None once
                        # discovery finishes skipping the H=14,400 file on line 1.
                        second_mixed = _pick_ready(mixed_default_bank)
                        assert second_mixed is None
                    finally:
                        mixed_rec.close()
                finally:
                    mixed_default_bank.close()
            finally:
                watch_replay.verify_payload = orig_verify  # type: ignore[assignment]
            # Only the single H=6,000 file was decompressed/verified; the H=14,400 file was
            # skipped on line 1 without calling verify_payload.
            mixed_skipped_without_verify = len(verify_calls_mixed) == 1

            # 7. Full verify_payload + 3-slot Recording lifecycle under tracemalloc:
            # Start tracemalloc BEFORE constructing ReplayBank, and run the real verify_payload
            # on all 3 H=14,400 slots (000000, 000001, 000002) without skipping slots 2 and 3.
            long_full_verifies = 0

            def _counting_long_verify(payload: Any, *args: Any, **kwargs: Any) -> dict:
                nonlocal long_full_verifies
                long_full_verifies += 1
                return orig_verify(payload, *args, **kwargs)

            watch_replay.verify_payload = _counting_long_verify  # type: ignore[assignment]
            try:
                tracemalloc.start()
                active_rec: Recording | None = None
                long_bank: ReplayBank | None = None
                try:
                    t_cold_pick = time.perf_counter()
                    long_bank = ReplayBank(long_dir, shuffle_discovery=False, allow_long=True)
                    active_rec = _pick_ready(long_bank, timeout_s=180.0)
                    cold_pick_unthrottled_seconds = time.perf_counter() - t_cold_pick
                    assert active_rec is not None
                    first_horizon = active_rec.header["horizon"]
                    # Wait for the background worker to finish full 14,400-frame verify_payload
                    # on both prefetch queue slots (000001 and 000002, maxsize=2) so all 3
                    # H=14,400 recordings are verified and pinned simultaneously in ReplayBank.
                    deadline = time.perf_counter() + 180.0
                    while time.perf_counter() < deadline and not long_bank._ready.full():
                        time.sleep(0.01)
                    assert long_full_verifies == 3
                    queued_recs = list(long_bank._ready.queue)
                    ready_count = len(queued_recs)
                    pinned_mmap_bytes = (
                        len(active_rec._mmap) if active_rec._mmap is not None else 0
                    ) + sum(
                        len(r._mmap) for r in queued_recs if getattr(r, "_mmap", None) is not None
                    )

                    # Step 500 frames and rotate non-blockingly to the next H=14,400 session.
                    for _ in range(500):
                        ok = active_rec.step()
                        assert ok is True and active_rec.frame["type"] == "frame"
                    t_rot = time.perf_counter()
                    rot1 = long_bank.pick()
                    rot1_ms = (time.perf_counter() - t_rot) * 1000.0
                    assert rot1 is not None
                    try:
                        second_horizon = rot1.header["horizon"]
                        current_mem, peak_heap_delta = tracemalloc.get_traced_memory()
                    finally:
                        rot1.close()
                finally:
                    if active_rec is not None:
                        active_rec.close()
                    if long_bank is not None:
                        long_bank.close()
                    tracemalloc.stop()
                rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            finally:
                watch_replay.verify_payload = orig_verify  # type: ignore[assignment]

            # 8. Separate long_root test (default bank in default_dir + long bank in long_dir).
            dual_bank = ReplayBank(
                default_dir,
                shuffle_discovery=False,
                long_root=long_dir,
                horizons=frozenset({LONG_HORIZON}),
            )
            try:
                dual_rec = _pick_ready(dual_bank, timeout_s=120.0)
                assert dual_rec is not None
                try:
                    dual_horizon_only_long = dual_rec.header["horizon"]
                finally:
                    dual_rec.close()
            finally:
                dual_bank.close()
        finally:
            watch_replay.COOPERATIVE_YIELD_MIN_WAIT_SECONDS = orig_wait
            watch_replay.COOPERATIVE_YIELD_DUTY_FACTOR = orig_duty

    return {
        "synthetic_h14400_session0": {
            "compressed_mib": round(h14400_s0["compressed_bytes"] / 1024**2, 3),
            "decoded_mib": round(h14400_s0["decoded_bytes"] / 1024**2, 3),
            "peak_line_kib": round(h14400_s0["peak_line_bytes"] / 1024, 2),
            "write_seconds": round(h14400_s0["write_seconds"], 3),
            "within_max_compressed_128_mib": h14400_s0["compressed_bytes"] <= MAX_COMPRESSED,
            "within_max_long_decoded_1228_8_mib": h14400_s0["decoded_bytes"] <= MAX_LONG_DECODED,
            "within_max_line_2048_kib": h14400_s0["peak_line_bytes"] <= MAX_LINE,
        },
        "verification_and_prefetch_timing": {
            "sync_verify_h14400_seconds": round(sync_verify_s0, 3),
            "sync_verify_per_frame_ms": round(sync_verify_s0 * 1000.0 / LONG_HORIZON, 4),
            "cold_first_pick_production_seconds": round(cold_pick_production_seconds, 3),
            "cold_first_pick_tracemalloc_unthrottled_seconds": round(
                cold_pick_unthrottled_seconds, 3
            ),
            "playback_duration_24fps_seconds": playback_window_seconds,
            "prefetch_headroom_ratio": round(
                playback_window_seconds / max(0.001, cold_pick_production_seconds), 2
            ),
            "warm_rotation_ms": round(rot1_ms, 4),
            "ready_queue_depth_reached": ready_count,
            "full_verify_payload_slots_verified": long_full_verifies,
        },
        "memory": {
            "pinned_mmap_3_slots_mib": round(pinned_mmap_bytes / 1024**2, 3),
            "peak_python_heap_delta_mib": round(peak_heap_delta / 1024**2, 3),
            "current_python_heap_delta_mib": round(current_mem / 1024**2, 3),
            "ru_maxrss_before_mib": round(rss_before / rss_divisor, 2),
            "ru_maxrss_after_mib": round(rss_after / rss_divisor, 2),
        },
        "default_h6000_byte_identity_and_isolation": {
            "default_rejects_h14400_header": default_rejected_header,
            "default_rejects_h14400_peek": default_rejected_peek,
            "unselected_long_header_detected": unselected_skip_check,
            "unselected_long_header_skip_ms": round(skip_header_ms, 3),
            "mixed_dir_default_served_horizon": mixed_default_horizon,
            "mixed_dir_skipped_long_without_verify": mixed_skipped_without_verify,
            "opt_in_first_horizon": first_horizon,
            "opt_in_rotated_horizon": second_horizon,
            "long_only_dual_bank_horizon": dual_horizon_only_long,
            "h6000_sessions_verified": [h6000_s0["index"], h6000_s1["index"]],
            "h14400_sessions_verified": [
                h14400_s0["index"],
                h14400_s1["index"],
                h14400_s2["index"],
            ],
        },
    }


def run_mac_engineering_gate(
    *,
    installed_bank_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute the complete 0-unit Mac engineering gate for opt-in H=14,400 Watch sessions."""
    started = time.perf_counter()
    if installed_bank_dir is None:
        installed_bank_dir = _resolve_default_installed_bank()
    short_gen = run_short_local_generator_check(horizon=6, count=2)
    synth_gate = run_h14400_scale_synthetic_gate()
    extrapolation = extrapolate_from_installed_bank(installed_bank_dir)
    pilot_spec = build_pilot_package_spec(installed_bank_dir=installed_bank_dir)

    s0 = synth_gate["synthetic_h14400_session0"]
    timing = synth_gate["verification_and_prefetch_timing"]
    mem = synth_gate["memory"]
    iso = synth_gate["default_h6000_byte_identity_and_isolation"]

    checks = {
        "short_local_generator_2_sessions_ok": (
            short_gen["sessions_generated"] == 2
            and short_gen["frames_replayed"] == [6, 6]
            and short_gen["resumed_bit_identical"] is True
        ),
        "synthetic_h14400_within_reader_bounds": (
            s0["within_max_compressed_128_mib"]
            and s0["within_max_long_decoded_1228_8_mib"]
            and s0["within_max_line_2048_kib"]
            and 31.0 <= s0["peak_line_kib"] <= 42.5
        ),
        "prefetch_faster_than_playback_window": (
            timing["cold_first_pick_production_seconds"] < timing["playback_duration_24fps_seconds"]
            and timing["prefetch_headroom_ratio"] >= 2.0
            and timing["full_verify_payload_slots_verified"] == 3
        ),
        "memory_bounded_and_streaming": (
            mem["peak_python_heap_delta_mib"] < 32.0
            and 0.0 < mem["pinned_mmap_3_slots_mib"] < 3 * (MAX_COMPRESSED / 1024**2)
        ),
        "default_h6000_isolation_and_byte_identity": (
            iso["default_rejects_h14400_header"] is True
            and iso["default_rejects_h14400_peek"] is True
            and iso["unselected_long_header_detected"] is True
            and iso["unselected_long_header_skip_ms"] < 25.0
            and iso["mixed_dir_default_served_horizon"] == DEFAULT_HORIZON
            and iso["mixed_dir_skipped_long_without_verify"] is True
            and iso["opt_in_first_horizon"] == LONG_HORIZON
            and iso["opt_in_rotated_horizon"] == LONG_HORIZON
            and iso["long_only_dual_bank_horizon"] == LONG_HORIZON
        ),
        "default_h6000_bank_present": (
            pilot_spec["seed_disjointness"]["default_bank_present"] is True
        ),
        "pilot_seeds_disjoint_from_installed_132": (
            pilot_spec["seed_disjointness"]["disjoint"] is True
        ),
    }
    if extrapolation.get("installed_bank_available"):
        ext = extrapolation["extrapolated_h14400"]
        checks["extrapolated_installed_bank_within_reader_bounds"] = (
            ext["fits_reader_max_compressed_128_mib"] is True
            and ext["fits_reader_max_long_decoded_1228_8_mib"] is True
        )

    passed = all(checks.values())
    return {
        "verdict": "PASS" if passed else "FAIL",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "checks": checks,
        "short_local_generator_check": short_gen,
        "synthetic_h14400_gate": synth_gate,
        "installed_bank_extrapolation": extrapolation,
        "pilot_package_spec": pilot_spec,
        "resolved_env_horizons": [
            sorted(resolve_replay_horizons()[0] or ()),
            sorted(resolve_replay_horizons(allow_long=True)[0] or ()),
            sorted(resolve_replay_horizons(horizons="14400")[0] or ()),
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    gate_p = sub.add_parser(
        "gate", help="Run the 0-u Mac engineering gate for H=14,400 Watch sessions"
    )
    gate_p.add_argument(
        "--installed-bank",
        type=Path,
        default=None,
        help="Path to the installed 132-session H=6,000 bank for empirical extrapolation",
    )

    verify_p = sub.add_parser("verify-bank", help="Verify a Watch replay bank directory")
    verify_p.add_argument("bank_dir", type=Path)
    verify_p.add_argument("--horizon", type=int, choices=(6000, 14400), default=None)
    verify_p.add_argument("--disallow-long", action="store_true")
    verify_p.add_argument("--full-payload", action="store_true")

    install_p = sub.add_parser(
        "install-long-bank", help="Verify and atomically install an H=14,400 bank"
    )
    install_p.add_argument("source_dir", type=Path)
    install_p.add_argument("dest_dir", type=Path)
    install_p.add_argument("--count", type=int, default=PILOT_COUNT)
    install_p.add_argument("--horizon", type=int, default=LONG_HORIZON)
    install_p.add_argument(
        "--no-contract",
        action="store_true",
        help="Allow installing when contract.json is absent",
    )

    sub.add_parser(
        "pilot-spec", help="Print the 12-session H=14,400 G4 pilot package specification"
    )

    args = parser.parse_args(argv)
    if args.cmd == "gate":
        report = run_mac_engineering_gate(installed_bank_dir=args.installed_bank)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["verdict"] == "PASS" else 1
    if args.cmd == "verify-bank":
        res = verify_bank_directory(
            args.bank_dir,
            expected_horizon=args.horizon,
            allow_long=not args.disallow_long,
            full_payload=args.full_payload,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    if args.cmd == "install-long-bank":
        res = install_long_bank(
            args.source_dir,
            args.dest_dir,
            expected_count=args.count,
            expected_horizon=args.horizon,
            full_payload=True,
            require_contract=not args.no_contract,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    if args.cmd == "pilot-spec":
        print(json.dumps(build_pilot_package_spec(), indent=2, sort_keys=True))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
