#!/usr/bin/env python3
"""Runtime adapter for one pre-registered S1 mature M3-C training job.

It is deliberately fail-closed: it accepts only the approved registration,
materializes immutable uploaded inputs, runs the real trainer, and exposes only
hash-verified, size-bounded immutable exports to the CLI runtime supervisor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from research.s1_mature_20261018.manifests import (
    APPROVAL,
    MAX_EXPORT_BYTES,
    TRAINING_DECISIONS,
    namespace,
    training_command,
)

B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            candidate = (destination / member.name).resolve()
            if (
                not candidate.is_relative_to(destination.resolve())
                or member.issym()
                or member.islnk()
            ):
                raise RuntimeError(f"unsafe archive member: {member.name}")
        tar.extractall(destination, filter="data")


def reassemble_and_extract(index_path: Path, exports_root: Path, destination: Path) -> Path:
    """Verify a manually selected transport snapshot and safely reconstruct it.

    This is deliberately only a recovery preparation primitive.  A new approved
    manifest is still required to run ``--resume-full-state``; it never retries
    or allocates a runtime itself.
    """
    index = json.loads(index_path.read_text())
    if index.get("schema") != "s1-chunked-export/v1" or destination.exists():
        raise RuntimeError("invalid export index or non-fresh recovery destination")
    chunks = index.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise RuntimeError("export index lacks chunks")
    archive = destination.with_suffix(".tar.gz")
    with archive.open("xb") as combined:
        for entry in chunks:
            name, digest = entry.get("path"), entry.get("sha256")
            if not isinstance(name, str) or Path(name).name != name or not isinstance(digest, str):
                raise RuntimeError("unsafe export chunk index")
            part = exports_root / name
            if not part.is_file() or sha256_file(part) != digest:
                raise RuntimeError(f"missing or changed export chunk: {name}")
            shutil.copyfileobj(part.open("rb"), combined)
    if sha256_file(archive) != index.get("archive_sha256"):
        archive.unlink(missing_ok=True)
        raise RuntimeError("reassembled archive digest mismatch")
    destination.mkdir()
    _safe_extract(archive, destination)
    manifests = list(destination.rglob("manifest.json"))
    if len(manifests) != 1 or not (manifests[0].parent / "state.pt").is_file():
        raise RuntimeError("recovery archive is not one complete full-state snapshot")
    return manifests[0].parent


def _complete_export(source: Path, destination: Path) -> None:
    """Copy an immutable file then atomically publish hash and completion marker."""
    if not source.is_file() or source.stat().st_size > MAX_EXPORT_BYTES:
        raise RuntimeError(f"export absent or exceeds 2 GiB: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    digest = sha256_file(destination)
    sidecar = destination.with_name(destination.name + ".sha256")
    sidecar.write_text(f"{digest}  {destination.name}\n")
    with sidecar.open("rb") as handle:
        os.fsync(handle.fileno())
    destination.with_name(destination.name + ".complete").touch(exist_ok=False)


def _archive_directory(source: Path, archive: Path) -> None:
    temporary = archive.with_suffix(archive.suffix + ".partial")
    with tarfile.open(temporary, "w:gz", compresslevel=6) as tar:
        tar.add(source, arcname=source.name, recursive=True)
    os.replace(temporary, archive)


def _publish_chunked(
    archive: Path, exports: Path, stem: str, *, chunk_size: int = 1536 * 1024**2
) -> None:
    """Publish a reassemblable archive as immutable chunks plus a checked index."""
    if chunk_size <= 0 or chunk_size > MAX_EXPORT_BYTES:
        raise RuntimeError("invalid transport chunk size")
    chunks: list[dict[str, object]] = []
    with archive.open("rb") as source:
        number = 0
        while True:
            block = source.read(chunk_size)
            if not block:
                break
            name = f"{stem}.part-{number:03d}"
            local = archive.with_name(name)
            local.write_bytes(block)
            _complete_export(local, exports / name)
            chunks.append({"path": name, "sha256": sha256_file(local), "size": len(block)})
            local.unlink()
            number += 1
    if not chunks:
        raise RuntimeError(f"empty transport archive: {archive}")
    index = archive.with_name(f"{stem}.index.json")
    index.write_text(
        json.dumps(
            {
                "schema": "s1-chunked-export/v1",
                "archive_sha256": sha256_file(archive),
                "chunks": chunks,
            },
            sort_keys=True,
        )
        + "\n"
    )
    _complete_export(index, exports / index.name)
    index.unlink()


def _snapshot_exports(full_state: Path, exports: Path, seen: set[str]) -> None:
    for manifest in sorted(full_state.glob("snapshot-*/manifest.json")):
        snapshot = manifest.parent
        key = snapshot.name
        if key in seen:
            continue
        # A manifest is committed last by AsyncCheckpointer.  Confirm its single state file digest.
        try:
            value = json.loads(manifest.read_text())
        except FileNotFoundError:
            continue  # Retention retired this snapshot before pin acquisition.
        item = value["files"][0]
        state = snapshot / item["name"]
        staging_root = full_state / f".transport-{key}"
        staging = staging_root / key
        staging.mkdir(parents=True, exist_ok=False)
        try:
            # Full-state retention may remove this directory while a multi-GiB
            # archive is being produced.  A hard link pins the verified inode.
            try:
                os.link(state, staging / item["name"])
            except FileNotFoundError:
                continue
            if sha256_file(staging / item["name"]) != item["sha256"]:
                raise RuntimeError(f"full-state checksum mismatch: {snapshot}")
            (staging / "manifest.json").write_text(json.dumps(value, sort_keys=True) + "\n")
            archive = full_state / f"{key}.tar.gz"
            _archive_directory(staging, archive)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)
        _publish_chunked(archive, exports, f"full-state-{key}")
        archive.unlink()
        seen.add(key)


def _decision_count(run_dir: Path) -> int:
    log = run_dir / "log.jsonl"
    if not log.is_file():
        return 0
    try:
        last = log.read_text().splitlines()[-1]
        return int(
            json.loads(last).get("learner_decisions", json.loads(last).get("transitions", 0))
        )
    except (IndexError, OSError, ValueError, json.JSONDecodeError):
        return 0


def _write_progress(
    path: Path, *, phase: str, started: float, child: subprocess.Popen[str] | None, run_dir: Path
) -> None:
    row = {
        "phase": phase,
        "done": _decision_count(run_dir),
        "total": TRAINING_DECISIONS,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "pid": child.pid if child else None,
    }
    path.write_text(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"progress": row}, sort_keys=True), flush=True)


def _replace_option(command: list[str], option: str, value: str) -> None:
    command[command.index(option) + 1] = value


def actual_recipe(summary: dict) -> dict:
    """Normalize recorded trainer arguments, without inventing missing settings."""
    args = summary["args"]
    recipe = {key: int(args[key]) for key in ("worlds", "actors", "learner_seats", "transitions")}
    for key in ("schedule_package", "collection_profile", "namespace"):
        recipe[key] = args[key]
    return recipe


def registered_halt(summary: dict) -> bool:
    """Separate registered scientific safety stops from operational failures."""
    reason = str(summary.get("halt", "")).lower()
    return any(
        marker in reason
        for marker in (
            "non-finite loss",
            "nonfinite loss",
            "non-finite q",
            "nonfinite q",
            "non-finite gradient",
            "nonfinite gradient",
            "action collapse",
            "illegal action",
            "overall greedy boost rate",
            "greedy boost rate below length 700",
        )
    )


def record_tripwire(run_dir: Path, summary: dict, arm: str, seed: int, trainer: Path) -> None:
    """Record a verified safety halt, retaining its raw evidence for Mac audit."""
    actors = int(summary["actors"])
    if (
        not registered_halt(summary)
        or summary.get("stopped_for_time")
        or summary.get("actor_exitcodes") != [0] * actors
        or summary.get("ring_overrun_rows") != 0
        or summary.get("nstep_accounting_reconciled") is not True
    ):
        raise RuntimeError("halt has incomplete operational/identity evidence")
    log = run_dir / "log.jsonl"
    if not log.exists():
        # An early learner tripwire can precede the first telemetry window.
        with log.open("x") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": "adapter_registered_tripwire",
                        "reason": summary["halt"],
                        "summary_sha256": sha256_file(run_dir / "summary.json"),
                    }
                )
                + "\n"
            )
    packet = {
        "schema": "s1-training-outcome/v1",
        "status": "TRIPWIRE_KILL",
        "arm": arm.upper(),
        "training_seed": seed,
        "b8_sha256": B8_SHA256,
        "namespace": summary["args"]["namespace"],
        "reason": summary["halt"],
        "raw_summary_halt": summary["halt"],
        "trainer_source_sha256": sha256_file(trainer),
        "summary_sha256": sha256_file(run_dir / "summary.json"),
        "log_sha256": sha256_file(log),
        "recipe": actual_recipe(summary),
        "counters": {
            key: summary[key]
            for key in (
                "learner_decisions",
                "emitted_replay_rows",
                "updates",
                "pending_update_debt",
            )
        },
    }
    with (run_dir / "outcome.json").open("x") as stream:
        json.dump(packet, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run(arm: str, seed: int, root: Path, *, smoke: bool = False) -> None:
    env = os.environ
    if not smoke and (
        env.get("S1_REGISTRATION_STATUS") != APPROVAL or env.get("S1_OWNER_RELEASE") != APPROVAL
    ):
        raise RuntimeError("DRAFT_PENDING_OWNER_REVIEW: S1 training dispatch is hard-blocked")
    if not smoke and env.get("S1_TRAINING_NAMESPACE") != namespace(arm, seed):
        raise RuntimeError("training namespace does not match the registered S1 arm and seed")
    input_dir = Path(env["COLAB_INPUT_DIR"])
    output = Path(env["COLAB_OUTPUT_DIR"])
    init, demo, league = (
        input_dir / "B8_s3_final.pth",
        input_dir / "demo.tar.gz",
        input_dir / "league.tar.gz",
    )
    parity_reference, parity_bank = (
        input_dir / "mac-parity-reference.json",
        input_dir / "parity-bank.json",
    )
    if not all(path.is_file() for path in (init, demo, league)) or (
        not smoke and not all(path.is_file() for path in (parity_reference, parity_bank))
    ):
        raise RuntimeError("missing registered B8, demo, league, or Mac parity input")
    if sha256_file(init) != B8_SHA256:
        raise RuntimeError("uploaded initializer is not the registered B8 comparator")
    try:
        seed_bank = json.loads((input_dir / "seed-bank.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("missing or invalid registered seed bank") from exc
    if seed not in seed_bank.get("seeds", []):
        raise RuntimeError("job seed is absent from the uploaded registered seed bank")
    work = root / "s1-work"
    work.mkdir(exist_ok=False)
    demo_dir, league_dir = work / "demo", work / "league"
    demo_dir.mkdir()
    league_dir.mkdir()
    _safe_extract(demo, demo_dir)
    _safe_extract(league, league_dir)
    if not (demo_dir / "margin.json").is_file():
        raise RuntimeError("demo archive does not contain margin.json")
    # The supervisor imports each output-relative file under its local exports/.
    # A second runtime exports/ prefix would produce exports/exports/ on the Mac.
    run_dir, full_state, exports = work / "run", work / "full-state", output
    started = time.monotonic()
    parity_receipt = work / "runtime-parity.json"
    if not smoke:
        parity_env = {
            **env,
            "SNAKE_CKPT_DIR": str(league_dir),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
        from research.q0q1_20261017.q0b_measure import validate_current_runtime

        packet = json.loads((input_dir / "q0b-l4.json").read_text())
        runtime_check = validate_current_runtime(packet, expected_hardware="L4")
        if runtime_check.get("pass") is not True:
            raise RuntimeError(f"current L4 runtime is not Q0b qualified: {runtime_check}")
        fixture_inputs = work / "fixture-inputs"
        (fixture_inputs / "init").mkdir(parents=True)
        os.symlink(league_dir, fixture_inputs / "checkpoints", target_is_directory=True)
        shutil.copyfile(init, fixture_inputs / "init/B8_s3_final.pth")
        parity_env["Q0Q1_INPUT_ROOT"] = str(fixture_inputs)
        for fixture in ("t22", "t24"):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "research.q0q1_20261017.q0b_cuda_fixtures",
                    "--fixture",
                    fixture,
                    "--out",
                    str(work / f"{fixture}-cuda.json"),
                ],
                cwd=root / "repo",
                env=parity_env,
                check=True,
                timeout=240,
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "research.q0q1_20261017.q1_parity",
                "--verify-reference",
                str(parity_reference),
                "--bank",
                str(parity_bank),
                "--b8",
                str(init),
                "--out",
                str(parity_receipt),
            ],
            cwd=root / "repo",
            env=parity_env,
            check=True,
            timeout=300,
        )
    command = [
        str(root / "venv/bin/python") if (root / "venv/bin/python").is_file() else sys.executable,
        *training_command(arm, seed)[1:],
    ]
    replacements = {
        "${S1_INIT_PATH}": str(init),
        "${S1_DEMO_DIR}": str(demo_dir),
        "${S1_RUN_DIR}": str(run_dir),
        "${S1_FULL_STATE_DIR}": str(full_state),
    }
    command = [replacements.get(token, token) for token in command]
    if smoke:
        # Qualification only: this is a real M3-C run but intentionally cannot
        # be mistaken for S1 data or a production manifest.
        _replace_option(command, "--transitions", "12001")
        _replace_option(command, "--device", "cpu")
        _replace_option(command, "--actor-device", "cpu")
        _replace_option(command, "--worlds", "8")
        _replace_option(command, "--actors", "2")
        _replace_option(command, "--learner-seats", "3")
        _replace_option(command, "--checkpoint-every", "6000")
        _replace_option(command, "--full-state-every", "6000")
        _replace_option(command, "--ring-dir", str(work / "rings"))
        _replace_option(command, "--namespace", f"s1-preflight-smoke/seed-{seed}")
        command.extend(["--replay", "2048", "--demo-capacity", "2048", "--warmup", "256"])
        (work / "rings").mkdir()
    child_env = {
        **env,
        "SNAKE_CKPT_DIR": str(league_dir),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
    }
    child = subprocess.Popen(
        command, cwd=root / "repo", env=child_env, text=True, start_new_session=True
    )
    shutdown_requested = False

    def request_checkpoint(_signum: int, _frame: object) -> None:
        """Convert supervisor SIGUSR1 into a trainer interrupt and final snapshot."""
        nonlocal shutdown_requested
        shutdown_requested = True
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)

    def terminate_child(_signum: int, _frame: object) -> None:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise KeyboardInterrupt

    previous_usr1 = signal.signal(signal.SIGUSR1, request_checkpoint)
    previous_term = signal.signal(signal.SIGTERM, terminate_child)
    previous_int = signal.signal(signal.SIGINT, terminate_child)
    seen: set[str] = set()
    progress = output / "progress.json"
    try:
        while child.poll() is None:
            _snapshot_exports(full_state, exports, seen)
            _write_progress(
                progress, phase="training", started=started, child=child, run_dir=run_dir
            )
            time.sleep(120)
        _snapshot_exports(full_state, exports, seen)
    finally:
        signal.signal(signal.SIGUSR1, previous_usr1)
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
    summary, final = run_dir / "summary.json", run_dir / "final.pth"
    if not summary.is_file():
        raise RuntimeError("training did not write summary.json")
    data = json.loads(summary.read_text())
    if parity_receipt.is_file():
        shutil.copyfile(parity_receipt, run_dir / "runtime-parity.json")
    if data.get("halt"):
        record_tripwire(
            run_dir, data, arm, seed, root / "repo/research/redesign_m3c_20261008/train_m3c.py"
        )
        final_archive = work / "s1-final.tar.gz"
        _archive_directory(run_dir, final_archive)
        _publish_chunked(final_archive, exports, "s1-final")
        return
    if child.returncode or not final.is_file():
        raise RuntimeError("training did not produce a clean final")
    expected_decisions = 12_001 if smoke else TRAINING_DECISIONS
    if (
        data.get("halt")
        or data.get("stopped_for_time")
        or int(data.get("learner_decisions", 0)) < expected_decisions
    ):
        raise RuntimeError("training did not produce a complete registered final")
    actors, seats = int(data.get("actors", 0)), int(data.get("m3c", {}).get("learner_seats", 0))
    rows, updates, debt = (
        int(data.get("emitted_replay_rows", -1)),
        int(data.get("updates", -1)),
        int(data.get("pending_update_debt", -1)),
    )
    worlds = int(data.get("args", {}).get("worlds", 0))
    if (
        data.get("actor_exitcodes") != [0] * actors
        or data.get("ring_overrun_rows") != 0
        or data.get("nstep_accounting_reconciled") is not True
        or rows != 256 * updates + debt
        or debt < 0
        or int(data.get("transition_budget_overshoot", -1)) > actors * worlds * seats - 1
        or data.get("full_state_complete") is not True
    ):
        raise RuntimeError("training counter, actor, ring, or full-state receipt is incomplete")
    provenance = {
        "schema": "s1-training-provenance/v1",
        "arm": arm.upper(),
        "training_seed": seed,
        "endpoint": expected_decisions,
        "checkpoint_sha256": sha256_file(final),
        "summary_sha256": sha256_file(summary),
        "init_sha256": sha256_file(init),
        "namespace": env["S1_TRAINING_NAMESPACE"],
        "source": "research/redesign_m3c_20261008/train_m3c.py",
        "trainer_source_sha256": sha256_file(
            root / "repo/research/redesign_m3c_20261008/train_m3c.py"
        ),
        "b8_sha256": sha256_file(init),
        "runtime_parity_receipt_sha256": (
            sha256_file(parity_receipt) if parity_receipt.is_file() else None
        ),
        "clean_exit": True,
        "tripwire_triggered": False,
        "recipe": actual_recipe(data),
        "actual_recipe": data["args"],
        "raw_summary": data,
        "counters": {
            "actual_learner_decisions": data["learner_decisions"],
            "registered_overshoot_bound": actors * worlds * seats - 1,
            **{
                key: data.get(key)
                for key in (
                    "learner_decisions",
                    "emitted_replay_rows",
                    "updates",
                    "pending_update_debt",
                    "transition_budget_overshoot",
                )
            },
        },
        "full_state_exports": sorted(seen),
    }
    (run_dir / "s1-training-provenance.json").write_text(
        json.dumps(provenance, sort_keys=True) + "\n"
    )
    final_archive = work / "s1-final.tar.gz"
    _archive_directory(run_dir, final_archive)
    _publish_chunked(final_archive, exports, "s1-final")
    _write_progress(progress, phase="complete", started=started, child=child, run_dir=run_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("n1", "n3"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("COLAB_CLI_ROOT", ".")))
    parser.add_argument("--smoke", action="store_true", help="local qualification fixture only")
    args = parser.parse_args()
    run(args.arm, args.seed, args.root.resolve(), smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
