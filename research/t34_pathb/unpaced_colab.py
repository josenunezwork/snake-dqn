"""Create-only CPU collection for T34's descriptive unpaced Colab arm.

This module deliberately has no decision or pacing role.  It runs exactly the
registered terminal and product rows in a shard, writes one immutable record
per world, and leaves the analyzer to consume those records as optional
descriptive evidence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Mapping
from unittest.mock import patch

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

from research.t34_exit_b8.public_inputs import EXPECTED_MEMBERS  # noqa: E402
from research.t34_exit_b8.public_inputs import _load as load_public_catalog  # noqa: E402
from research.t34_pathb import adapter_v2, retention_v2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "t34-unpaced-world/v1"
SHARD_SCHEMA = "t34-unpaced-shard/v1"
GLOBAL_UNIT_CAP = 42.533128
GLOBAL_RESERVE_UNITS = 7.088855
SMOKE_SECONDS_PER_WORLD = 362.498247


class UnpacedRefused(RuntimeError):
    """The descriptive arm lacks a portable, registered execution binding."""


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UnpacedRefused(f"invalid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise UnpacedRefused(f"expected JSON object: {path}")
    return value


def _runtime_registration(registration: Mapping[str, Any], commit: str) -> None:
    """Check portable registration bindings without reading a Mac artifact path."""
    if (
        registration.get("schema") != "t34-pathb-v2-registration/v1"
        or registration.get("ratified") is not True
        or registration.get("ratified_by") != "Claude"
        or registration.get("near_boundary_precedence_acknowledged") is not True
        or registration.get("mac_arms") != list(retention_v2.MAC_ARMS)
        or registration.get("unpaced_colab") != retention_v2.unpaced_colab_policy()
    ):
        raise UnpacedRefused("production requires the ratified unpaced Colab amendment")
    if (ROOT / ".git").exists():
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if actual != commit:
            raise UnpacedRefused("runtime HEAD differs from the registered source commit")
    elif ROOT != Path(os.environ.get("COLAB_CLI_ROOT", "/nonexistent")) / "repo":
        raise UnpacedRefused("non-Git source is allowed only in the supervisor fallback bundle")
    # The qualified source delivery verifies the pinned commit/tree, including
    # the sparse fallback. This additional check binds every executable byte
    # even when fallback intentionally has no .git directory.
    if retention_v2.source_inventory(ROOT) != registration.get("source_hashes"):
        raise UnpacedRefused("runtime source inventory differs from registration")
    for label in ("bank", "seed_audit", "profile", "preregistration"):
        binding = registration.get(label, {})
        relative = Path(str(binding.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise UnpacedRefused(f"{label} must be a portable repository binding")
        target = ROOT / relative
        if not target.is_file() or _sha(target) != binding.get("sha256"):
            raise UnpacedRefused(f"registered {label} hash differs")
    bank = _load_object(ROOT / registration["bank"]["path"])
    for key in ("terminal_rows", "product_rows", "timing_row"):
        if registration.get(key) != bank.get(key):
            raise UnpacedRefused(f"registered {key} differs from the frozen bank")
    if (
        len(bank["terminal_rows"]) != 144
        or len(bank["product_rows"]) != 32
        or any(
            sum(row["mix"] == mix for row in bank["terminal_rows"]) != 48
            for mix in retention_v2.MIXES
        )
    ):
        raise UnpacedRefused("registered unpaced bank differs from N48/32")


def _hardware(nproc: int, workers: int) -> dict[str, Any]:
    import psutil

    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "nproc": nproc,
        "ram_bytes": int(psutil.virtual_memory().total),
        "process_workers": workers,
        "native_threads_per_worker": 1,
        "max_cpu_threads_per_world": 2,
    }


def _write_complete(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    with path.with_name(path.name + ".sha256").open("x", encoding="utf-8") as stream:
        stream.write(digest + "  " + path.name + "\n")
    with path.with_name(path.name + ".complete").open("x", encoding="utf-8") as stream:
        stream.write("complete\n")
    return digest


def _public_paths(catalog: Path, input_root: Path) -> dict[str, Path]:
    """Map every registered SHA to a verified runtime-relative public file."""
    rows, _assets = load_public_catalog(catalog)
    result: dict[str, Path] = {}
    for row in rows:
        digest = str(row["sha256"])
        expected = EXPECTED_MEMBERS.get(str(row["name"]))
        target = input_root / str(row["dest"])
        if (
            expected is None
            or not target.is_file()
            or target.stat().st_size != int(row["size_bytes"])
        ):
            raise UnpacedRefused(f"public checkpoint absent: {row['name']}")
        if _sha(target) != digest:
            raise UnpacedRefused(f"public checkpoint hash differs: {row['name']}")
        if digest in result:
            raise UnpacedRefused("public catalog repeats a checkpoint SHA")
        result[digest] = target
    if len(result) != 6:
        raise UnpacedRefused("public catalog must map all six registered checkpoints")
    return result


def _mapped_row(row: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    """Keep the registered row byte-for-byte except for runtime checkpoint locations."""
    mapped = json.loads(json.dumps(row))
    return mapped


def _registration_paths(
    registration: Mapping[str, Any], public: Mapping[str, Path]
) -> dict[str, Path]:
    rows = registration.get("checkpoints")
    if not isinstance(rows, list) or len(rows) != 6:
        raise UnpacedRefused("registered public path map needs exactly six checkpoints")
    result: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise UnpacedRefused("registered checkpoint entry is malformed")
        digest = str(row.get("sha256", ""))
        if digest not in public:
            raise UnpacedRefused("registered checkpoint SHA is missing from public inputs")
        result[row["path"]] = public[digest]
    return result


@contextlib.contextmanager
def _runtime_checkpoint_paths(paths: Mapping[str, Path]) -> Iterator[None]:
    """Temporarily redirect only runtime file lookup to verified public bytes.

    The serving registry and its pins remain unchanged.  The context replaces
    its lookup directory and the B8 census directory only while a world is
    constructed, so the simulator continues to receive the same checkpoint
    bytes and policy semantics as the registered Mac row.
    """
    from research.b8_death_census_20261014 import census
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from web.backend import served_checkpoint

    b8 = paths.get("3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8")
    if b8 is None:
        raise UnpacedRefused("public B8 s3 checkpoint missing")
    temporary = Path(tempfile.mkdtemp(prefix="t34-unpaced-inputs-", dir=b8.parent))
    saved = temporary / "saved_snakes"
    saved.mkdir()
    try:
        names = {spec[2]: name for name, spec in EXPECTED_MEMBERS.items()}
        names[adapter_v2.B8_SHA] = served_checkpoint.B8_S3_FILENAME
        for digest, name in names.items():
            if digest not in paths:
                raise UnpacedRefused(f"required runtime checkpoint absent: {digest}")
            os.link(paths[digest], saved / name)
        b8_dir = temporary / "b8"
        (b8_dir / "B8_s3").mkdir(parents=True)
        os.link(b8, b8_dir / "B8_s3" / "final.pth")
        with (
            patch.object(census, "B8_DIR", b8_dir),
            patch.object(gid, "CHECKPOINT_DIR", saved),
            patch.object(served_checkpoint, "SAVED_DIR", str(saved)),
        ):
            yield
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _collect_world(
    row: Mapping[str, Any],
    *,
    kind: str,
    registration: Mapping[str, Any],
    public_paths: Mapping[str, str],
    horizon: int,
) -> dict[str, Any]:
    paths = {digest: Path(path) for digest, path in public_paths.items()}
    registered_paths = _registration_paths(registration, paths)
    mapped = _mapped_row(row, paths)
    roster = mapped.get("roster")
    if isinstance(roster, dict):
        for spec in roster.get("specs", []):
            if spec[0] == "checkpoint":
                try:
                    spec[1] = str(registered_paths[str(spec[1])])
                except KeyError as exc:
                    raise UnpacedRefused(
                        "terminal roster path lacks registered public mapping"
                    ) from exc
    started, cpu_started = time.monotonic(), time.process_time()
    with _runtime_checkpoint_paths(paths):
        session = (
            adapter_v2.open_terminal(mapped, 0, "unpaced")
            if kind == "terminal"
            else adapter_v2.open_watch(mapped, 0, "unpaced")
        )
        try:
            progress_started = time.monotonic()

            def progress() -> None:
                nonlocal progress_started
                now = time.monotonic()
                if now - progress_started >= 120:
                    print(
                        json.dumps(
                            {
                                "phase": "unpaced_world",
                                "seed": int(row["seed"]),
                                "frames_done": session.frame,
                                "frames_total": horizon,
                                "elapsed_seconds": now - started,
                            }
                        ),
                        flush=True,
                    )
                    progress_started = now

            return retention_v2._collect(
                session,
                horizon=horizon,
                fps=0,
                started=started,
                cpu_started=cpu_started,
                check_cap=progress,
            )
        finally:
            session.close(wait=True)


def run_shard(
    *,
    registration_path: Path,
    catalog: Path,
    input_root: Path,
    out: Path,
    shard: int,
    shard_count: int,
    nproc: int | None = None,
    preflight: bool = False,
    development_rows: list[Mapping[str, Any]] | None = None,
    frames: int | None = None,
    source_commit: str | None = None,
) -> Path:
    """Run one deterministic modulo shard; preflight emits no efficacy rows."""
    if out.exists() or shard < 0 or shard_count < 1 or shard >= shard_count:
        raise UnpacedRefused("output must be fresh and shard must be within shard_count")
    registration = _load_object(registration_path)
    if not preflight and registration.get("ratified") is not True:
        raise UnpacedRefused("production unpaced collection requires ratified registration bytes")
    commit = source_commit or str(registration.get("source_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or commit != registration.get("source_commit"):
        raise UnpacedRefused("unpaced collection needs the registered source commit")
    if not preflight:
        _runtime_registration(registration, commit)
    else:
        from research.t34_pathb.cost_smoke_v2 import _preflight

        _preflight()
    public = _public_paths(catalog, input_root)
    if not preflight and frames is not None and int(frames) != 5000:
        raise UnpacedRefused("production unpaced horizon is fixed at 5000")
    horizon = int(registration.get("horizon", 5000)) if frames is None else int(frames)
    if horizon < 1:
        raise UnpacedRefused("registered horizon is invalid")
    if not preflight and horizon != 5000:
        raise UnpacedRefused("production registration horizon must be 5000")
    if preflight and horizon > 4:
        raise UnpacedRefused("preflight is limited to four frames")
    rows = [("terminal", row) for row in registration.get("terminal_rows", [])] + [
        ("product", row) for row in registration.get("product_rows", [])
    ]
    if not rows:
        raise UnpacedRefused("registration has no terminal/product rows")
    selected = [
        (index, kind, row) for index, (kind, row) in enumerate(rows) if index % shard_count == shard
    ]
    if preflight:
        if not isinstance(development_rows, list) or len(development_rows) != 2:
            raise UnpacedRefused("preflight requires exactly two development rows")
        selected = [
            (index, str(row.get("kind", "terminal")), row)
            for index, row in enumerate(development_rows)
        ]
        registered_seeds = {int(row["seed"]) for _kind, row in rows}
        registered_seeds.add(int(registration["timing_row"]["seed"]))
        if any(int(row["seed"]) in registered_seeds for _index, _kind, row in selected):
            raise UnpacedRefused(
                "preflight must provide development rows disjoint from registered banks"
            )
        if len({int(row["seed"]) for _index, _kind, row in selected}) != 2 or {
            kind for _index, kind, _row in selected
        } != {"terminal", "product"}:
            raise UnpacedRefused(
                "preflight requires one terminal and one product development world"
            )
    if not selected:
        raise UnpacedRefused("shard has no worlds")
    requested = int(nproc if nproc is not None else max(2, os.cpu_count() or 2))
    if not 1 <= requested <= min(64, os.cpu_count() or 1):
        raise UnpacedRefused("nproc exceeds available CPU capacity")
    if not preflight and requested < 2:
        raise UnpacedRefused("production Colab execution needs main plus at least one worker")
    workers = 1 if preflight else min(requested - 1, 10)
    if not 1 <= workers <= 64:
        raise UnpacedRefused("nproc must be within 1..64")
    out.mkdir(parents=True)
    started = time.monotonic()
    payload_paths: dict[int, dict[str, str]] = {}
    public_strings = {key: str(value) for key, value in public.items()}
    reg_sha = _sha(registration_path)
    bank_sha = registration.get("bank", {}).get("sha256")
    if not isinstance(bank_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", bank_sha):
        raise UnpacedRefused("registration lacks its bound bank SHA")

    world_costs: dict[int, float] = {}

    def publish(position: int, result: dict[str, Any], elapsed: float) -> None:
        index, kind, row = selected[position]
        world_costs[index] = elapsed
        if not preflight:
            name = f"{kind}-{index:03d}.json"
            record = {
                "schema": SCHEMA,
                "registration_sha256": reg_sha,
                "source_commit": commit,
                "bank_sha256": bank_sha,
                "world": row,
                "kind": kind,
                "mix": row.get("mix", "mixed"),
                "seed": int(row["seed"]),
                "horizon": horizon,
                "arm": "unpaced",
                "metrics": result,
            }
            digest = _write_complete(out / "worlds" / name, record)
            payload_paths[index] = {"path": f"worlds/{name}", "sha256": digest}
        done, elapsed_total = len(world_costs), time.monotonic() - started
        print(
            json.dumps(
                {
                    "phase": "unpaced_progress",
                    "done": done,
                    "total": len(selected),
                    "elapsed_seconds": elapsed_total,
                    "eta_seconds": elapsed_total / done * (len(selected) - done),
                }
            ),
            flush=True,
        )

    # Publish each completed world immediately. Only the final index is sorted
    # by registered position, so completion order never changes the row mapping.
    arguments = [
        (row, kind, registration, public_strings, horizon) for _index, kind, row in selected
    ]
    if workers == 1:
        for position, argument in enumerate(arguments):
            world_started = time.monotonic()
            publish(position, _collect_star(argument), time.monotonic() - world_started)
    else:
        import multiprocessing

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            pending = {
                pool.submit(_collect_star, argument): position
                for position, argument in enumerate(arguments)
            }
            while pending:
                done, _ = concurrent.futures.wait(
                    pending, timeout=120, return_when=concurrent.futures.FIRST_COMPLETED
                )
                if not done:
                    print(
                        json.dumps(
                            {
                                "phase": "unpaced_progress",
                                "done": len(world_costs),
                                "total": len(selected),
                                "elapsed_seconds": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
                for future in sorted(done, key=lambda item: pending[item]):
                    position = pending.pop(future)
                    result = future.result()
                    publish(position, result, 0.0)
    if preflight:
        _write_complete(
            out / "preflight.json",
            {
                "schema": "t34-unpaced-preflight/v1",
                "status": "COMPLETE",
                "world_count": len(world_costs),
                "hardware": _hardware(requested, workers),
                "frames_per_world": horizon,
                "world_elapsed_seconds": list(world_costs.values()),
                "elapsed_seconds": time.monotonic() - started,
                "metrics_persisted": False,
                "source_commit": commit,
            },
        )
        return out / "preflight.json"
    shard_value = {
        "schema": SHARD_SCHEMA,
        "registration_sha256": reg_sha,
        "source_commit": commit,
        "bank_sha256": bank_sha,
        "status": "COMPLETE",
        "hardware": _hardware(requested, workers),
        "cost": {
            "smoke_seconds_per_world": SMOKE_SECONDS_PER_WORLD,
            "elapsed_seconds": time.monotonic() - started,
        },
        "worlds": [payload_paths[index] for index in sorted(payload_paths)],
    }
    _write_complete(out / "shard.json", shard_value)
    return out / "shard.json"


def _collect_star(args: tuple[Any, ...]) -> dict[str, Any]:
    row, kind, registration, paths, horizon = args
    return _collect_world(
        row, kind=kind, registration=registration, public_paths=paths, horizon=horizon
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--nproc", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--development-rows", type=Path)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    result = run_shard(
        registration_path=args.registration,
        catalog=args.catalog,
        input_root=args.input_root,
        out=args.out,
        shard=args.shard,
        shard_count=args.shard_count,
        nproc=args.nproc,
        preflight=args.preflight,
        development_rows=(
            _load_object(args.development_rows).get("rows")
            if args.development_rows is not None
            else None
        ),
        frames=args.frames,
        source_commit=args.source_commit,
    )
    print(json.dumps({"result": str(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
