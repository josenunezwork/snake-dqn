"""L1 D0 driver: unpack excluded A5 exports -> diagnostic cache -> registered cells -> rule.

One process, run by ``d0.sh`` on one L4 runtime (or locally on a tiny fixture).
Phases, each timed into ``d0_timing.json`` (rewritten after every phase):

1. ``unpack``: verify the hash-bound tarball and every world digest (``inputs.unpack``);
2. ``cache``: replay each world through the ratified A2 regenerator (checkpoint and
   every saved state hash must match), keep every searched row and every
   ``anchor_stride``-th ``shadow_confirm`` row (registered stride 1 = the full pool);
3. ``report``: free-Q probes + the registered D0 cell set on world-level folds;
4. ``decide``: the frozen D0 rule (``decide.py``) -> ``D0_DECISION.json``.

Outputs land in ``--out-dir`` (exported); the cache stays in ``--work``.  Nothing
here is evidence about a student, and no weights are written or exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from . import inputs
from .cache import build_cache
from .decide import DecisionRefused, decide
from .harness import D0_CELLS, D0_FOLDS, D0_FREE_Q, D0_FREE_Q_STEPS, run_report

SCHEMA = "t34-claim-a-d0-run/v1"
REGISTERED_ANCHOR_STRIDE = 1


def _write(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def run(
    *,
    tarball: Path,
    tarball_sha256: str,
    checkpoint: Path,
    work: Path,
    out_dir: Path,
    workers: int,
    device: str,
    threads: int = 2,
    anchor_stride: int = REGISTERED_ANCHOR_STRIDE,
    expect_worlds: int = inputs.EXPECTED_A5_WORLDS,
    cells: tuple[str, ...] = D0_CELLS,
    free_q_steps: int = D0_FREE_Q_STEPS,
    folds: int = D0_FOLDS,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=False)
    timing: dict[str, Any] = {
        "schema": SCHEMA,
        "claim": "D0_DESIGN_DIAGNOSTIC_EXCLUDED_A5_NOT_EVIDENCE",
        "tarball_sha256": tarball_sha256,
        "anchor_stride": anchor_stride,
        "registered": anchor_stride == REGISTERED_ANCHOR_STRIDE
        and tuple(cells) == D0_CELLS
        and free_q_steps == D0_FREE_Q_STEPS
        and folds == D0_FOLDS,
        "device": device,
        "workers": workers,
        "phases": {},
        "status": "RUNNING",
    }
    timing_path = out_dir / "d0_timing.json"

    def phase(name: str, started: float, **extra: Any) -> None:
        timing["phases"][name] = {"seconds": round(time.perf_counter() - started, 3), **extra}
        _write(timing_path, timing)

    try:
        started = time.perf_counter()
        worlds = inputs.unpack(tarball, tarball_sha256, work / "exports")
        if len(worlds) != expect_worlds:
            raise inputs.InputsRefused(f"expected {expect_worlds} worlds, got {len(worlds)}")
        phase("unpack", started, worlds=len(worlds))

        started = time.perf_counter()
        manifest = build_cache(
            [work / "exports" / "worlds"],
            checkpoint,
            work / "cache",
            anchor_stride=anchor_stride,
            workers=workers,
            expect_worlds=expect_worlds,
        )
        (out_dir / "cache_manifest.json").write_text((work / "cache" / "manifest.json").read_text())
        phase(
            "cache",
            started,
            accepted=manifest["accepted"],
            rejected=len(manifest["rejected"]),
            totals=manifest["totals"],
        )
        if manifest["rejected"]:
            timing["status"] = "PRECONDITION_FAILED"
            timing["reason"] = "replay rejected worlds (hash or Q mismatch); see cache_manifest"
            _write(timing_path, timing)
            return timing

        started = time.perf_counter()
        report_path = out_dir / "d0_report.json"
        run_report(
            work / "cache",
            checkpoint,
            report_path,
            cells=cells,
            free_q=D0_FREE_Q,
            free_q_steps=free_q_steps,
            folds_k=folds,
            threads=threads,
            device=device,
        )
        phase("report", started)

        started = time.perf_counter()
        raw = report_path.read_bytes()
        try:
            decision = decide(json.loads(raw))
        except DecisionRefused as exc:
            decision = {"decision": "REFUSED", "reason": str(exc)}
        decision["report_sha256"] = hashlib.sha256(raw).hexdigest()
        decision["registered_run"] = timing["registered"]
        with (out_dir / "D0_DECISION.json").open("x") as stream:
            json.dump(decision, stream, indent=2, sort_keys=True)
            stream.write("\n")
        phase("decide", started, decision=decision["decision"])
        timing["status"] = "COMPLETE"
    except BaseException as exc:  # record, then re-raise for the supervisor
        timing["status"] = "FAILED"
        timing["error"] = f"{type(exc).__name__}: {exc}"
        _write(timing_path, timing)
        raise
    _write(timing_path, timing)
    return timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tarball", type=Path, required=True)
    parser.add_argument("--tarball-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--anchor-stride", type=int, default=REGISTERED_ANCHOR_STRIDE)
    parser.add_argument("--expect-worlds", type=int, default=inputs.EXPECTED_A5_WORLDS)
    args = parser.parse_args()
    timing = run(
        tarball=args.tarball,
        tarball_sha256=args.tarball_sha256,
        checkpoint=args.checkpoint,
        work=args.work,
        out_dir=args.out_dir,
        workers=args.workers,
        device=args.device,
        threads=args.threads,
        anchor_stride=args.anchor_stride,
        expect_worlds=args.expect_worlds,
    )
    print(json.dumps({"status": timing["status"], "phases": timing["phases"]}, sort_keys=True))


if __name__ == "__main__":
    main()
