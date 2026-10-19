"""Prepare one reviewed S1 operational rerun without allocating a Colab runtime.

This tool only writes a retry manifest and a binding record.  It never invokes
the supervisor, ``colab``, subprocesses, or evaluation.  The operator later
passes the record to ``run.py --continue`` after a fresh account snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from research.s1_mature_20261018 import admission, ledger


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _no_efficacy_artifacts(root: Path) -> None:
    forbidden = (
        "analysis.json",
        "analysis-input.json",
        "independent-audit.json",
        "mac-final-evaluation-receipt.json",
    )
    evaluation = root / "evaluation"
    if (
        any((root / name).exists() for name in forbidden)
        or list(root.glob("eval-*.json"))
        or (evaluation.exists() and any(evaluation.rglob("*")))
    ):
        raise ledger.Refused(
            "evaluation or analysis artifacts already exist; manual rerun is forbidden"
        )


def _verified_final_export(source: dict[str, Any]) -> bool:
    """Return true only for a small complete, internally verified final packet.

    A partial index/chunk set is operational evidence, not a final.  A completed
    but malformed packet is an integrity uncertainty and therefore refuses rerun.
    """
    exports = Path(source["output_dir"]) / "exports"
    index_path = exports / "s1-final.index.json"
    if not index_path.exists():
        return False
    try:
        if index_path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("final index exceeds local inspection bound")
        index = _read(index_path)
        chunks = index["chunks"]
        if (
            index.get("schema") != "s1-chunked-export/v1"
            or not isinstance(chunks, list)
            or not chunks
        ):
            raise ValueError("invalid final index")
        pieces: list[Path] = []
        total = 0
        for row in chunks:
            if not isinstance(row, dict):
                raise ValueError("final index chunk is malformed")
            name, digest = row.get("path"), row.get("sha256")
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("final index chunk path is unsafe")
            piece = exports / str(name)
            if not piece.is_file():
                return False
            total += piece.stat().st_size
            if total > 64 * 1024 * 1024:
                raise ValueError("completed final packet exceeds local inspection bound")
            if not isinstance(digest, str) or _digest(piece) != digest:
                raise ValueError("final chunk digest mismatch")
            pieces.append(piece)
        with tempfile.NamedTemporaryFile() as archive:
            for piece in pieces:
                archive.write(piece.read_bytes())
            archive.flush()
            if _digest(Path(archive.name)) != index.get("archive_sha256"):
                raise ValueError("final archive digest mismatch")
            with tarfile.open(archive.name, "r:*") as tar:
                names = {Path(member.name).name for member in tar.getmembers() if member.isfile()}
        required = {"final.pth", "target.pth", "summary.json", "s1-training-provenance.json"}
        if not required <= names:
            raise ValueError("completed final archive lacks required decision artifacts")
        return True
    except (OSError, KeyError, TypeError, ValueError, tarfile.TarError) as exc:
        raise ledger.Refused(f"final export is corrupt or ambiguous: {exc}") from exc


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(
    *,
    worker_ledger: ledger.Ledger,
    source_manifest: Path,
    receipt: Path,
    review: Path,
    manifest_out: Path,
    record_out: Path,
    output_dir: Path,
    state_dir: Path,
    artifact_root: Path,
) -> Path:
    """Reconcile a single lost attempt and write a fresh, non-launchable retry record."""
    source_manifest, receipt, review = (
        source_manifest.resolve(),
        receipt.resolve(),
        review.resolve(),
    )
    manifest_out, record_out, output_dir, state_dir, artifact_root = (
        manifest_out.resolve(),
        record_out.resolve(),
        output_dir.resolve(),
        state_dir.resolve(),
        artifact_root.resolve(),
    )
    if any(path.exists() for path in (manifest_out, record_out, output_dir, state_dir)):
        raise ledger.Refused("retry manifest, record, output, and state paths must all be fresh")
    _no_efficacy_artifacts(artifact_root)
    source = _read(source_manifest)
    arm = source.get("env", {}).get("S1_ARM")
    if arm not in {"n1", "n3"}:
        raise ledger.Refused("source manifest is not an S1 N1/N3 training job")
    source_job = source.get("job_id")
    if not isinstance(source_job, str):
        raise ledger.Refused("source manifest lacks its job id")
    if _verified_final_export(source):
        raise ledger.Refused("verified final export exists; manual rerun is forbidden")
    worker_ledger.reconcile_operational_failure(source_job, receipt, review)
    retry_job = f"retry-{arm}"
    retry = dict(source)
    retry["job_id"] = retry_job
    retry["output_dir"] = str(output_dir)
    retry["attempt"] = {"kind": "manual-operational-rerun", "of": source_job}
    admission.write_once(manifest_out, retry)
    record = {
        "schema": "s1-manual-operational-rerun/v1",
        "status": "PREPARED_NOT_LAUNCHED",
        "source_job_id": source_job,
        "retry_job_id": retry_job,
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": admission.sha256(source_manifest),
        },
        "retry_manifest": {"path": str(manifest_out), "sha256": admission.sha256(manifest_out)},
        "supervisor_receipt": {"path": str(receipt), "sha256": admission.sha256(receipt)},
        "operational_failure_review": {"path": str(review), "sha256": admission.sha256(review)},
        "state_dir": str(state_dir),
        "no_resume": True,
        "no_efficacy_look": True,
        "manual_operator_action_required": True,
    }
    admission.write_once(record_out, record)
    return record_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--record-out", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        prepare(
            worker_ledger=ledger.Ledger(args.ledger),
            source_manifest=args.source_manifest,
            receipt=args.receipt,
            review=args.review,
            manifest_out=args.manifest_out,
            record_out=args.record_out,
            output_dir=args.output_dir,
            state_dir=args.state_dir,
            artifact_root=args.artifact_root,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
