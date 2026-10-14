#!/usr/bin/env python3
"""Mac-side dev probes that start as each checkpoint lands (M3-C; development only).

Watches ``landed.jsonl`` (written by ``gpu_run_v2.py`` after each sha-checked pull) and/or
local run directories (``--watch-dir``, rows of each run's ``manifest.jsonl``), and runs
``research/redesign_m2_20261008/dev_probe.py`` for every scheduled checkpoint x veto x mix
not yet probed: the no-veto arm first (the M3-C primary metric), then v8. Each probe is one
single-thread process under ``nice -n 10`` after the AC / lid / thermal guard; with
``--slots`` it also holds one shared CPU slot lock (compute policy). Results:
``<out>/<run>__<ckpt>__<veto>.jsonl`` (one row per mix, dev_probe's format). Idempotent:
re-running skips (run, ckpt, veto, mix) rows already written.

  ./venv/bin/python research/redesign_m3c_20261008/probe_watch.py \\
      --landed RUN_DIR/landed.jsonl --out RESULTS --worlds 16 --vetoes none --slots --procs 2
  (M3-C runs two watchers: intermediates ``--schedule 5000k,10000k,15000k --worlds 16
  --vetoes none`` and finals ``--schedule final --worlds 32 --vetoes none,v8``; they write
  different files because the checkpoint stem is part of the name.)

``--v61-backend numba`` (opt-in) passes ``dev_probe --v61-backend numba`` (compiled vector61
featurizer for the opponents / v8 carrier). Its records are identical to the default's
(``tests/test_eval_v61_backend_identity.py``), so an output file may hold rows of either
backend (each non-default row carries ``v61_backend``); resuming across backends is allowed.
``--anchors fast`` (opt-in) passes ``dev_probe --anchors fast`` (compiled, decision-identical
scripted anchors; rows carry ``anchors``), under the same record-identity rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEV_PROBE = REPO / "research/redesign_m2_20261008/dev_probe.py"
LOCK_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909")
MIXES = ("frozen", "scripted", "mixed")
MAX_FAILURES = 2  # a probe that failed twice is logged and skipped

Job = Tuple[str, str, str, str, str]  # run, ckpt stem, path, veto, mix
_CHECKPOINT_HASHES: Dict[Path, Tuple[int, int, str]] = {}


class ProbeIdentityError(ValueError):
    """A public probe output name would conflate incompatible checkpoints."""


def ckpt_stem(file: str) -> str:
    return Path(file).stem


def scheduled(file: str, schedule: Sequence[str]) -> bool:
    stem = ckpt_stem(file)
    return stem in schedule or stem.replace("ckpt_", "") in schedule


def landed_rows(landed: Optional[Path], watch_dirs: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    # A runner root directory: every attempt's landed.jsonl (attempt00/, attempt01/, ...).
    files = sorted(landed.glob("*/landed.jsonl")) if landed and landed.is_dir() else [landed]
    for landed in files:
        if not landed or not landed.exists():
            continue
        for line in landed.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            row = {"run": r["run"], "file": r["file"], "path": r["path"]}
            if r.get("sha256"):
                row["sha256"] = r["sha256"]
            rows.append(row)
    for d in watch_dirs:
        for man in sorted(Path(d).glob("*/manifest.jsonl")):
            for line in man.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                row = {
                    "run": man.parent.name,
                    "file": r["file"],
                    "path": str(man.parent / r["file"]),
                }
                if r.get("sha256"):
                    row["sha256"] = r["sha256"]
                rows.append(row)
    return rows


def result_path(out: Path, run: str, stem: str, veto: str) -> Path:
    return out / f"{run}__{stem}__{veto}.jsonl"


def _sha256_file(path: Path) -> str:
    """Hash a checkpoint, reusing the digest while its size and mtime are unchanged."""
    stat = path.stat()
    cached = _CHECKPOINT_HASHES.get(path)
    marker = (stat.st_size, stat.st_mtime_ns)
    if cached and cached[:2] == marker:
        return cached[2]
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    value = digest.hexdigest()
    _CHECKPOINT_HASHES[path] = marker + (value,)
    return value


def checkpoint_sha(row: Dict[str, str]) -> str:
    """Return the selected local checkpoint identity, checking its manifest witness."""
    path = Path(row["path"])
    declared = row.get("sha256")
    if declared and not re.fullmatch(r"[0-9a-f]{64}", declared):
        raise ProbeIdentityError(f"invalid manifest sha256 for {path}: {declared!r}")
    if path.exists():
        actual = _sha256_file(path)
        if declared and declared != actual:
            raise ProbeIdentityError(
                f"checkpoint {path} no longer matches its landed manifest sha256; "
                "do not reuse or append this probe output"
            )
        return actual
    if declared:
        # Puller validates landed sha256 before writing the row. A manifest-only watcher
        # may reach this branch briefly, but the witness is still enough to detect a
        # competing attempt before the local file lands.
        return declared
    raise ProbeIdentityError(f"cannot establish checkpoint identity for {path}: missing sha256")


def done_mixes(path: Path, student_sha256: str, veto: str, round_index: int) -> set:
    """Completed mixes, only when every existing row matches this exact probe contract."""
    got = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                mix = row["mix"]
                if row.get("student_sha256") != student_sha256:
                    raise ProbeIdentityError(
                        f"{path} was made with another checkpoint; "
                        "choose a distinct --out directory"
                    )
                if row.get("veto") != veto or row.get("round_index") != round_index:
                    raise ProbeIdentityError(
                        f"{path} has incompatible probe settings; choose a distinct --out directory"
                    )
                if row.get("ego2s_forward") != "rowwise":
                    raise ProbeIdentityError(
                        f"{path} lacks the required rowwise forward setting; "
                        "choose a distinct --out directory"
                    )
                expected_mask = "advisory" if veto == "none" else None
                if row.get("ego2s_mask") != expected_mask:
                    raise ProbeIdentityError(
                        f"{path} has incompatible ego2s mask settings; "
                        "choose a distinct --out directory"
                    )
                got.add(mix)
            except (ValueError, KeyError, TypeError) as exc:
                if isinstance(exc, ProbeIdentityError):
                    raise
                raise ProbeIdentityError(
                    f"{path} has an invalid existing probe row; choose a distinct --out directory"
                ) from exc
    return got


def pending_jobs(
    rows: Sequence[Dict[str, str]],
    out: Path,
    schedule: Sequence[str],
    vetoes: Sequence[str],
    running: set,
    round_index: int = 52,
) -> List[Job]:
    """Every scheduled (run, ckpt, veto, mix) not yet written or running; no-veto first."""
    seen = set()
    jobs: List[Job] = []
    identities = {}
    for r in rows:
        if scheduled(r["file"], schedule) and re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", r["run"]):
            stem = ckpt_stem(r["file"])
            output = (r["run"], stem)
            sha = checkpoint_sha(r)
            prior = identities.setdefault(output, sha)
            if prior != sha:
                raise ProbeIdentityError(
                    f"{result_path(out, r['run'], stem, 'none').parent / (r['run'] + '__' + stem)} "
                    "would combine checkpoints from different attempts; "
                    "choose a distinct --out directory"
                )
    for veto in vetoes:
        for r in rows:
            if not scheduled(r["file"], schedule):
                continue
            if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", r["run"]):
                continue
            stem = ckpt_stem(r["file"])
            have = done_mixes(
                result_path(out, r["run"], stem, veto),
                identities[(r["run"], stem)],
                veto,
                round_index,
            )
            for mix in MIXES:
                key = (r["run"], stem, veto, mix)
                if mix in have or key in running or key in seen:
                    continue
                seen.add(key)
                jobs.append((r["run"], stem, r["path"], veto, mix))
    return jobs


def probe_argv(
    job: Job,
    out: Path,
    worlds: int,
    round_index: int,
    slots: bool,
    v61_backend: Optional[str] = None,
    anchors: Optional[str] = None,
) -> List[str]:
    """dev_probe's argv for ``job``; ``v61_backend`` / ``anchors`` (opt-in) add
    ``--v61-backend`` / ``--anchors``."""
    run, stem, path, veto, mix = job
    argv = [
        sys.executable,
        str(DEV_PROBE),
        "--student",
        path,
        "--arm",
        "student",
        "--mix",
        mix,
        "--worlds",
        str(worlds),
        "--round-index",
        str(round_index),
        "--veto",
        veto,
        "--ego2s-forward",
        "rowwise",  # Phase R's forward mode
        "--ego2s-mask",
        "advisory",  # used by dev_probe for the no-veto hero only (Phase R's rule)
        "--out",
        str(result_path(out, run, stem, veto)),
    ]
    if v61_backend is not None:
        argv += ["--v61-backend", v61_backend]
    if anchors is not None:
        argv += ["--anchors", anchors]
    if slots:
        return [sys.executable, str(REPO / "research/redesign_m3_20261008/slotrun.py"), "--"] + argv
    return ["nice", "-n", "10"] + argv


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--landed", type=Path, default=None)
    ap.add_argument("--watch-dir", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--schedule", default="5000k,10000k,15000k")
    ap.add_argument("--vetoes", default="none,v8")
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--round-index", type=int, default=52)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--slots", action="store_true", help="hold a shared CPU slot lock per probe")
    ap.add_argument(
        "--v61-backend",
        choices=("python", "numba"),
        default=None,
        help="pass dev_probe --v61-backend (opt-in; numba is record-identical, "
        "tests/test_eval_v61_backend_identity.py); default: dev_probe's own default",
    )
    ap.add_argument(
        "--anchors",
        choices=("python", "fast"),
        default=None,
        help="pass dev_probe --anchors (opt-in; fast = compiled, decision-identical scripted "
        "anchors); default: dev_probe's own default",
    )
    ap.add_argument("--until", type=Path, default=None, help="exit when this file exists and idle")
    ap.add_argument("--once", action="store_true", help="one pass, then wait for the jobs, exit")
    args = ap.parse_args(argv)
    from research.redesign_m2_20261008.gen_data import guard_ok

    args.out.mkdir(parents=True, exist_ok=True)
    schedule = [s.strip() for s in args.schedule.split(",") if s.strip()]
    vetoes = [v.strip() for v in args.vetoes.split(",") if v.strip()]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", SNAKE_DQN_DEVICE="cpu")
    live: Dict[tuple, subprocess.Popen] = {}
    failures: Dict[tuple, int] = {}
    log = args.out / "probe_watch.log"
    while True:
        for key, proc in list(live.items()):
            if proc.poll() is not None:
                if proc.returncode != 0:
                    failures[key] = failures.get(key, 0) + 1
                with log.open("a") as fh:
                    fh.write(
                        json.dumps({"done": key, "rc": proc.returncode, "t": time.time()}) + "\n"
                    )
                del live[key]
        rows = landed_rows(args.landed, args.watch_dir)
        given_up = {k for k, n in failures.items() if n >= MAX_FAILURES}
        jobs = pending_jobs(
            rows, args.out, schedule, vetoes, set(live) | given_up, round_index=args.round_index
        )
        while jobs and len(live) < args.procs:
            problems = [] if args.slots else guard_ok()  # slotrun.py checks it itself
            if problems:
                with log.open("a") as fh:
                    fh.write(json.dumps({"waiting": problems, "t": time.time()}) + "\n")
                break
            job = jobs.pop(0)
            key = job[:2] + job[3:]
            argv_ = probe_argv(
                job,
                args.out,
                args.worlds,
                args.round_index,
                args.slots,
                args.v61_backend,
                args.anchors,
            )
            live[key] = subprocess.Popen(argv_, cwd=REPO, env=env, stdout=subprocess.DEVNULL)
            with log.open("a") as fh:
                fh.write(json.dumps({"started": key, "t": time.time()}) + "\n")
        idle = not live and not jobs
        if idle and (args.once or (args.until and args.until.exists())):
            return 0
        time.sleep(5 if args.once else 60)


if __name__ == "__main__":
    raise SystemExit(main())
