"""Claim A Colab manifests: student-checkpoint labels and the pipeline qualification.

``label`` wraps the ratified ``research.t34_exit_b8.make_manifest`` (which still
validates the pinned B8/league catalog, the v2 export sizing and the qualified
transport) and adds what that builder refuses: a **nondefault rollout student**,
bound by SHA-256 and size, delivered either as a public URL input or as a
supervisor-uploaded file at ``inputs/student/<name>``.  The runtime verifies the
same digest again before labelling (``label.sh``).

``pipeline`` builds the L4 qualification job for ``pipeline.sh`` (label ->
payload -> cache -> CUDA fit -> production probe), public inputs only.

Both builders are pure: they never allocate, and ``--validate`` runs only the
supervisor's local validator.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from research.t34_exit_b8 import make_manifest as ratified
from research.t34_exit_b8.qualification import QualificationRefused, require_qualified_transport

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/t34_claim_a"
FACTORY = "research.t34_claim_a.student:student_policy"
SOURCE_PATHS = sorted(set(ratified.V2_SOURCE_PATHS) | {PACKAGE, "research/t34_exit_b8_fit"})
L4_RATE_UNITS_PER_HOUR = 2.0


class ManifestRefused(ValueError):
    """A Claim A draft cannot become a supervisor job."""


def student_input(
    *,
    sha256: str,
    size_bytes: int,
    name: str,
    path: Path | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    """One hash-bound student checkpoint row (public URL xor uploaded local file)."""
    if not re.fullmatch(r"[0-9a-f]{64}", sha256 or ""):
        raise ManifestRefused("student sha256 must be a full lowercase SHA-256")
    if type(size_bytes) is not int or size_bytes <= 0:
        raise ManifestRefused("student size_bytes must be a positive integer")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}\.pth", name or ""):
        raise ManifestRefused("student name must be a plain .pth file name")
    if (path is None) == (url is None):
        raise ManifestRefused("student needs exactly one of a local path or a public https URL")
    dest = f"student/{name}"
    if url is not None:
        if not url.startswith("https://"):
            raise ManifestRefused("student URL must be https")
        return {"url": url, "sha256": sha256, "size_bytes": size_bytes, "dest": dest}
    source = Path(path).expanduser().resolve(strict=True)
    from research.t34_claim_a.student import sha256_file

    if source.stat().st_size != size_bytes or sha256_file(source) != sha256:
        raise ManifestRefused(f"student bytes differ from the declared identity: {source}")
    if size_bytes > 64 * 1024**2:
        raise ManifestRefused("uploaded student exceeds the 64 MiB per-file upload bound")
    return {"path": str(source), "sha256": sha256, "dest": dest}


def make_label_manifest(
    *,
    commit: str,
    output_dir: Path,
    student: Mapping[str, Any],
    round_id: str,
    shard: int,
    attempt: int,
    worlds: int,
    measured_export_bytes_per_world: int,
    triggers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A v2 CPU fan-out label job whose rollout policy is an explicit student."""
    base = ratified.make_manifest(
        kind="label",
        kernel="v2",
        commit=commit,
        output_dir=output_dir,
        round_id=round_id,
        shard=shard,
        attempt=attempt,
        worlds=worlds,
        triggers=triggers,
        measured_export_bytes_per_world=measured_export_bytes_per_world,
    )
    row = student_input(**student)
    if any(item.get("dest") == row["dest"] for item in base["inputs"]):
        raise ManifestRefused("student destination collides with a pinned input")
    manifest = json.loads(json.dumps(base))
    tag = row["sha256"][:8]
    manifest["job_id"] = f"t34-claim-a-{round_id}-s{shard}-a{attempt}-{tag}".lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", manifest["job_id"]):
        raise ManifestRefused("round/shard/attempt do not form a valid job id")
    manifest["script"] = f"{PACKAGE}/label.sh"
    manifest["inputs"].append(row)
    manifest["env"].update(
        {
            "T34_POLICY_FACTORY": FACTORY,
            "T34_STUDENT_CHECKPOINT": row["dest"],
            "T34_STUDENT_SHA256": row["sha256"],
            "T34_NAMESPACE": f"t34-claim-a/v2/{round_id}/s{shard}/a{attempt}",
        }
    )
    manifest["source_paths"] = SOURCE_PATHS
    manifest["student"] = {
        "sha256": row["sha256"],
        "size_bytes": student["size_bytes"],
        "transport": "public_url" if "url" in row else "supervisor_upload",
        "dest": row["dest"],
    }
    return manifest


def make_pipeline_manifest(
    *,
    commit: str,
    output_dir: Path,
    label_worlds: int = 24,
    heldout_decisive: int = 2,
    epochs: int = 4,
    est_minutes: int = 90,
    unit_cap: float | None = None,
) -> dict[str, Any]:
    """The Claim A L4 qualification: CPU fan-out labels, CUDA cached fit, 32xH5000 probe."""
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ManifestRefused("commit must be a full lowercase 40-character SHA")
    output_dir = output_dir.expanduser()
    if not output_dir.is_absolute():
        raise ManifestRefused("output_dir must be an absolute fresh Mac directory")
    if not 2 <= label_worlds <= 64 or not 1 <= heldout_decisive < label_worlds:
        raise ManifestRefused("qualification label worlds must lie in [2, 64]")
    if not 4 <= epochs <= 6:
        raise ManifestRefused("fit epochs must lie in [4, 6]")
    timeout = 2 * int(est_minutes)
    cap = float(unit_cap) if unit_cap is not None else timeout * L4_RATE_UNITS_PER_HOUR / 60
    if timeout * L4_RATE_UNITS_PER_HOUR / 60 > cap:
        raise ManifestRefused("whole-session timeout priced above the unit cap")
    catalog = ratified.PUBLIC_CATALOG_PATH.resolve(strict=True)
    try:
        require_qualified_transport(ROOT, commit)
    except QualificationRefused as exc:
        raise ManifestRefused(str(exc)) from exc
    return {
        "job_id": f"t34-claim-a-pipeline-qual-{commit[:8]}",
        "commit": commit,
        "script": f"{PACKAGE}/pipeline.sh",
        "claim": "PIPELINE_QUALIFICATION_TIMING_ONLY_NO_CANDIDATE",
        "transport": {"mode": "public_release_urls", "dispatch_ready": True},
        "env": {
            "PYTHONUNBUFFERED": "1",
            "T34_CA_LABEL_WORLDS": str(label_worlds),
            "T34_CA_HELDOUT_DECISIVE": str(heldout_decisive),
            "T34_CA_EPOCHS": str(epochs),
            "T34_CA_DEVICE": "cuda",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        },
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": int(est_minutes),
        },
        "timeout_minutes": timeout,
        "output_dir": str(output_dir),
        "inputs": [
            {
                "path": str(catalog),
                "sha256": ratified.sha256_file(catalog),
                "dest": "public-inputs.json",
            }
        ],
        "budget": {
            "unit_cap": cap,
            "rate_units_per_hour": L4_RATE_UNITS_PER_HOUR,
            "reserve_units": round(cap * 0.2, 3),
            "other_assignments": [],
        },
        "lease_seconds": 600,
        "max_export_bytes": 256 * 1024**2,
        "source_paths": SOURCE_PATHS,
    }


def validate_manifest(manifest: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    """Run only the supervisor's pure local validator (no allocation)."""
    if manifest.get("transport", {}).get("dispatch_ready") is False:
        raise ManifestRefused("public input publication is pending; manifest is not dispatch-ready")
    try:
        require_qualified_transport(repo, manifest.get("commit", ""))
    except QualificationRefused as exc:
        raise ManifestRefused(str(exc)) from exc
    from research.colab_cli.supervisor import validate

    return validate(manifest, repo=repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("label", "pipeline"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--validate", action="store_true")
    # label
    parser.add_argument("--round", default="r1")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--worlds", type=int)
    parser.add_argument("--measured-export-bytes-per-world", type=int)
    parser.add_argument("--triggers", type=json.loads)
    parser.add_argument("--student-path", type=Path)
    parser.add_argument("--student-url")
    parser.add_argument("--student-sha256")
    parser.add_argument("--student-size-bytes", type=int)
    parser.add_argument("--student-name", default="student.pth")
    # pipeline
    parser.add_argument("--label-worlds", type=int, default=24)
    parser.add_argument("--heldout-decisive", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--est-minutes", type=int, default=90)
    parser.add_argument("--unit-cap", type=float)
    args = parser.parse_args()
    commit = args.commit or ratified.git_head()
    if args.kind == "label":
        if args.worlds is None:
            raise SystemExit("--worlds is required for a label manifest")
        student: dict[str, Any] = {
            "sha256": args.student_sha256,
            "size_bytes": args.student_size_bytes,
            "name": args.student_name,
            "path": args.student_path,
            "url": args.student_url,
        }
        if args.student_path is not None and (
            args.student_sha256 is None or args.student_size_bytes is None
        ):
            from research.t34_claim_a.student import sha256_file

            student["sha256"] = student["sha256"] or sha256_file(args.student_path)
            student["size_bytes"] = student["size_bytes"] or args.student_path.stat().st_size
        manifest = make_label_manifest(
            commit=commit,
            output_dir=args.output_dir,
            student=student,
            round_id=args.round,
            shard=args.shard,
            attempt=args.attempt,
            worlds=args.worlds,
            measured_export_bytes_per_world=args.measured_export_bytes_per_world,
            triggers=args.triggers,
        )
    else:
        manifest = make_pipeline_manifest(
            commit=commit,
            output_dir=args.output_dir,
            label_worlds=args.label_worlds,
            heldout_decisive=args.heldout_decisive,
            epochs=args.epochs,
            est_minutes=args.est_minutes,
            unit_cap=args.unit_cap,
        )
    if args.validate:
        validate_manifest(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"manifest": str(args.out), "validated": args.validate}, sort_keys=True))


if __name__ == "__main__":
    main()
