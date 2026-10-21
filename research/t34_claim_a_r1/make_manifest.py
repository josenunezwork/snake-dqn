"""R1 labels-only chunk manifests (Colab CLI supervisor), built on the ratified label job.

Each chunk is the ratified ``research.t34_exit_b8`` v2 label job (pinned public
B8 s3 + league catalog, ``research/t34_exit_b8/label.sh``, CPU process fan-out on
an L4 runtime, beta 1, confirm interval 8) with four changes made here, not in
the ratified package:

* ``T34_NAMESPACE`` is the seed-audited Claim A R1 namespace of the chunk
  (``t34-claim-a/r1/{train|heldout}/s{k}/a{attempt}``), checked against
  ``SEED_AUDIT_R1.json`` (fresh, disjoint from Round 0 / 0-D / retention and the
  visible strict/serving/screen history);
* the job binds the **governing** Round 0 verdict (RULING_round0_go_file.md):
  ``<study_root>/amendment-boost-floor/decision.json`` with ``governing: true``,
  ``decision: GO``, ``evidence_authority: TIER_1_DEVELOPMENT``, its sealed
  ``phase_terminal.json`` (which must list that decision's SHA-256) and the
  amendment-ledger line on ``origin/main``.  A registered ``analysis/decision.json``
  is refused;
* ``budget.other_assignments`` is ``{}`` (a list is refused at admission);
* ``claim`` is ``R1_LABEL_DATA_ONLY_NOT_EVIDENCE``.

The builder is pure: it never allocates.  ``--validate`` runs only the
supervisor's local validator.  Lane-cap admission across chunks is
:func:`research.t34_claim_a_r1.plan.admissible`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from research.t34_exit_b8 import make_manifest as ratified

from .plan import CHUNK_WORLDS, LANE_CAP_UNITS, Chunk

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/t34_claim_a_r1"
DEFAULT_SEED_AUDIT = Path(__file__).with_name("SEED_AUDIT_R1.json")
# A5 v2 receipts: ~8.9 MB per exported world (research/t34_claim_a/README.md).
MEASURED_EXPORT_BYTES_PER_WORLD = 9_000_000
CLAIM = "R1_LABEL_DATA_ONLY_NOT_EVIDENCE"
AMEND_LEDGER = "research/t34_round0_amend/ledger/{root}.jsonl"


class ManifestRefused(ValueError):
    """An R1 chunk cannot become a supervisor job."""


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def published_ledger_entries(repo: Path, relative: str) -> list[dict[str, Any]]:
    """Ledger lines as published on ``origin/main`` (``git show``; no fetch)."""
    try:
        text = subprocess.run(
            ["git", "-C", str(repo), "show", f"origin/main:{relative}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise ManifestRefused(f"amendment ledger {relative} is not on origin/main") from exc
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def governing_round0(
    decision_path: Path,
    *,
    repo: Path = ROOT,
    ledger: Callable[[Path, str], list[dict[str, Any]]] = published_ledger_entries,
) -> dict[str, Any]:
    """Bind the published governing Round 0 GO; refuse anything else."""
    decision_path = Path(decision_path).resolve(strict=True)
    if decision_path.name != "decision.json" or decision_path.parent.name != (
        "amendment-boost-floor"
    ):
        raise ManifestRefused(
            "Round 0 GO must be <study_root>/amendment-boost-floor/decision.json "
            "(RULING_round0_go_file.md); the registered analysis/decision.json never governs"
        )
    decision = json.loads(decision_path.read_text())
    if decision.get("governing") is not True:
        raise ManifestRefused("decision.json is not the governing verdict (governing != true)")
    if decision.get("decision") != "GO":
        raise ManifestRefused(f"governing Round 0 verdict is {decision.get('decision')!r}, not GO")
    if decision.get("evidence_authority") != "TIER_1_DEVELOPMENT":
        raise ManifestRefused("governing GO must carry evidence_authority TIER_1_DEVELOPMENT")
    terminal_path = decision_path.parent / "phase_terminal.json"
    if not terminal_path.is_file():
        raise ManifestRefused("governing verdict has no sealed phase_terminal.json")
    terminal = json.loads(terminal_path.read_text())
    decision_sha = sha256_file(decision_path)
    if (terminal.get("files") or {}).get("decision.json") != decision_sha:
        raise ManifestRefused("phase_terminal.json does not seal this decision.json")
    terminal_sha = sha256_file(terminal_path)
    relative = AMEND_LEDGER.format(root=decision_path.parent.parent.name)
    if not any(e.get("terminal_sha256") == terminal_sha for e in ledger(repo, relative)):
        raise ManifestRefused("governing terminal is not in the published amendment ledger")
    return {
        "decision_path": str(decision_path),
        "decision_sha256": decision_sha,
        "phase_terminal_sha256": terminal_sha,
        "amendment_ledger": relative,
        "ruling": "docs/research/advice/2026-10-20_colab_lanes/RULING_round0_go_file.md",
    }


def _seed_audit(path: Path, namespace: str, count: int) -> dict[str, str]:
    from research.t34_claim_a.seed_audit import SeedAuditFailed, verify_receipt

    try:
        verify_receipt(json.loads(Path(path).read_text()), namespace, count)
    except (OSError, ValueError, SeedAuditFailed) as exc:
        raise ManifestRefused(f"seed audit does not cover {namespace} x{count}: {exc}") from exc
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))}


def make_chunk_manifest(
    *,
    commit: str,
    output_dir: Path,
    chunk: Chunk,
    round0: Mapping[str, Any],
    seed_audit: Path = DEFAULT_SEED_AUDIT,
) -> dict[str, Any]:
    """One 100-world R1 chunk as the ratified v2 label job, retargeted to its namespace."""
    audit = _seed_audit(seed_audit, chunk.namespace, chunk.worlds)
    base = ratified.make_manifest(
        kind="label",
        kernel="v2",
        commit=commit,
        output_dir=output_dir,
        round_id="r1",
        shard=chunk.shard,
        attempt=chunk.attempt,
        worlds=chunk.worlds,
        measured_export_bytes_per_world=MEASURED_EXPORT_BYTES_PER_WORLD,
    )
    manifest = json.loads(json.dumps(base))
    if manifest["script"] != "research/t34_exit_b8/label.sh":
        raise ManifestRefused("R1 must run the ratified v2 label entrypoint")
    env = manifest["env"]
    if env.get("T34_BETA") != "1" or env.get("T34_CONFIRM_INTERVAL") != "8":
        raise ManifestRefused("R1 is beta 1, confirm interval 8 (bare B8 + teacher)")
    env["T34_NAMESPACE"] = chunk.namespace
    manifest["job_id"] = chunk.job_id
    manifest["claim"] = CLAIM
    manifest["budget"] = dict(manifest["budget"], other_assignments={})
    if manifest["budget"]["unit_cap"] > LANE_CAP_UNITS:
        raise ManifestRefused("a single chunk cap exceeds the R1 lane cap")
    manifest["seed_audit"] = audit
    manifest["round0_governing"] = dict(round0)
    manifest["r1"] = {
        "split": chunk.split,
        "shard": chunk.shard,
        "attempt": chunk.attempt,
        "namespace": chunk.namespace,
        "worlds": CHUNK_WORLDS,
        "lane_cap_units": LANE_CAP_UNITS,
        "label_data_only": (
            "no fit, screen, selection or claim before the Path B disposition; "
            "not citable in the retention verdict (SUCCESS_CONTRACT §5 Amendment)"
        ),
    }
    return manifest


def validate_manifest(manifest: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    return ratified.validate_manifest(manifest, repo=repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "heldout"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--round0-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--seed-audit", type=Path, default=DEFAULT_SEED_AUDIT)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    chunk = Chunk(args.split, args.shard, args.attempt)
    manifest = make_chunk_manifest(
        commit=args.commit or ratified.git_head(),
        output_dir=args.output_dir,
        chunk=chunk,
        round0=governing_round0(args.round0_decision),
        seed_audit=args.seed_audit,
    )
    if args.validate:
        validate_manifest(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"manifest": str(args.out), "job_id": chunk.job_id}, sort_keys=True))


if __name__ == "__main__":
    main()
