"""Mac-side ratification and amendment verifier for the Watch6 V3 accounting adapter.

Implements Astra Ruling Points 1, 3, and 5:
- Verifies the immutable accounting amendment outside ``protocol.source_paths()``, including
  baseline commit/science, original registration hash, rulings 2/3 + shard-00 path ruling hashes,
  shard-00 proof/receipt hashes, complete 10-file adapter inventory hashes, and adapter commit.
- Reuses strict ``APPROVE`` / ``APPROVE-WITH-CONDITIONS`` verdict parsing; ``APPROVE-PATH``,
  ``REJECT``, ``NOT_RATIFIED``, missing, or conflicting verdicts are refused closed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.ratify import ALLOWED_VERDICTS, parse_ruling_verdict
from tools.ops.package_spec import digest, publish_json, verify_immutable
from tools.watch6_v3_accounting import (
    ADAPTER_ROOT,
    BASELINE_COMMIT,
    BASELINE_NAMESPACE,
    BASELINE_SCIENCE,
    REGISTRATION_V3_SHA256,
    RULING_2_REL,
    RULING_2_SHA256,
    RULING_3_REL,
    RULING_3_SHA256,
    SHARD00_EXEC_MANIFEST_SHA256,
    SHARD00_PATH_RULING_REL,
    SHARD00_PATH_RULING_SHA256,
    SHARD00_PROOF_DISPATCH_REL,
    SHARD00_PROOF_REL,
    SHARD00_PROOF_SHA256,
    SHARD00_RECEIPT_REL,
    SHARD00_RECEIPT_SHA256,
    adapter_commit,
    adapter_files_map,
    adapter_inventory_sha256,
    artifacts_root,
    default_amendment_path,
    default_ruling_json_path,
    verify_baseline_closure,
)

SCHEMA = "watch6-v3-accounting-ruling/v1"
_ADAPTER_COMMIT_LINE_RE = re.compile(
    r"adapter[_\s-]+commit[^0-9a-fA-F\n]*([0-9a-f]{40})\b",
    re.IGNORECASE,
)


def resolve_ruling_json(override: Path | None = None, *, repo: Path | None = None) -> Path:
    """Resolve the Mac-side accounting re-ratification receipt path."""
    if override is not None:
        return override.resolve()
    env = os.environ.get("WATCH6_V3_ACCOUNTING_RULING_JSON")
    if env:
        return Path(env).resolve()
    return default_ruling_json_path(repo).resolve()


def resolve_amendment_path(override: Path | None = None, *, repo: Path | None = None) -> Path:
    """Resolve the immutable accounting amendment Markdown path."""
    if override is not None:
        return override.resolve()
    env = os.environ.get("WATCH6_V3_ACCOUNTING_AMENDMENT")
    if env:
        return Path(env).resolve()
    return default_amendment_path(repo).resolve()


def _assert_outside_source_closure(path: Path, root: Path) -> None:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(root.resolve())
    except ValueError:
        return
    rel_str = rel.as_posix()
    for entry in p.source_paths():
        clean = entry.rstrip("/")
        if rel_str == clean or rel_str.startswith(clean + "/"):
            raise ValueError(
                f"REFUSE: accounting amendment {resolved} lies inside "
                f"hashed V3 source closure ({entry})"
            )


def _verify_frozen_artifacts_if_present(art_root: Path) -> None:
    expected_artifacts = (
        (RULING_2_REL, RULING_2_SHA256),
        (RULING_3_REL, RULING_3_SHA256),
        (SHARD00_PATH_RULING_REL, SHARD00_PATH_RULING_SHA256),
        (SHARD00_PROOF_REL, SHARD00_PROOF_SHA256),
        (SHARD00_PROOF_DISPATCH_REL, SHARD00_PROOF_SHA256),
        (SHARD00_RECEIPT_REL, SHARD00_RECEIPT_SHA256),
    )
    for rel, expected_sha in expected_artifacts:
        candidate = art_root / rel
        if candidate.is_symlink():
            raise ValueError(f"REFUSE: symlinked frozen reference artifact: {candidate}")
        if candidate.exists():
            if not candidate.is_file() or digest(candidate) != expected_sha:
                raise ValueError(
                    f"REFUSE: frozen reference artifact modified or mismatched: {candidate}"
                )


def verify_amendment_document(
    amendment_md: Path | None = None,
    *,
    repo: Path | None = None,
    require_clean_git: bool = True,
    expected_adapter_commit: str | None = None,
) -> dict:
    """Verify immutable accounting amendment sidecars and exact adapter/baseline hash bindings."""
    root = (repo or ADAPTER_ROOT).resolve()
    verify_baseline_closure(root)

    raw_target = (
        amendment_md if amendment_md is not None else resolve_amendment_path(None, repo=root)
    )
    if raw_target.is_symlink():
        raise ValueError(f"REFUSE: symlinked accounting amendment forbidden: {raw_target}")
    target = raw_target.resolve()
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"REFUSE: missing accounting amendment document: {target}")

    _assert_outside_source_closure(target, root)
    immutable_desc = verify_immutable(target)
    text = target.read_text(encoding="utf-8")

    commit_matches = list(dict.fromkeys(_ADAPTER_COMMIT_LINE_RE.findall(text)))
    if expected_adapter_commit is not None:
        bound_commit = expected_adapter_commit
    elif len(commit_matches) == 1:
        bound_commit = commit_matches[0]
    else:
        raise ValueError(
            f"REFUSE: accounting amendment {target} must bind exactly one 40-hex Adapter Commit"
        )

    verified_commit = adapter_commit(
        root,
        require_clean=require_clean_git,
        expected_commit=bound_commit,
    )
    files_map = adapter_files_map(root)
    inv_sha = adapter_inventory_sha256(root)

    required_tokens = (
        ("baseline_commit", BASELINE_COMMIT),
        ("baseline_science", BASELINE_SCIENCE),
        ("baseline_namespace", BASELINE_NAMESPACE),
        ("registration_v3_sha256", REGISTRATION_V3_SHA256),
        ("ruling_2_rel", RULING_2_REL),
        ("ruling_2_sha256", RULING_2_SHA256),
        ("ruling_3_rel", RULING_3_REL),
        ("ruling_3_sha256", RULING_3_SHA256),
        ("shard00_path_ruling_rel", SHARD00_PATH_RULING_REL),
        ("shard00_path_ruling_sha256", SHARD00_PATH_RULING_SHA256),
        ("shard00_proof_rel", SHARD00_PROOF_REL),
        ("shard00_proof_sha256", SHARD00_PROOF_SHA256),
        ("shard00_receipt_rel", SHARD00_RECEIPT_REL),
        ("shard00_receipt_sha256", SHARD00_RECEIPT_SHA256),
        ("shard00_exec_manifest_sha256", SHARD00_EXEC_MANIFEST_SHA256),
        ("adapter_commit", verified_commit),
        ("adapter_inventory_sha256", inv_sha),
    )
    for label, token in required_tokens:
        if token not in text:
            raise ValueError(
                f"REFUSE: accounting amendment {target} missing required {label} binding ({token})"
            )

    for rel_path, file_sha in sorted(files_map.items()):
        if rel_path not in text or file_sha not in text:
            raise ValueError(
                f"REFUSE: accounting amendment {target} missing exact hash binding for {rel_path}"
            )
        pattern = re.compile(re.escape(rel_path) + r"[^\n]*" + re.escape(file_sha))
        rev_pattern = re.compile(re.escape(file_sha) + r"[^\n]*" + re.escape(rel_path))
        if pattern.search(text) is None and rev_pattern.search(text) is None:
            raise ValueError(
                f"REFUSE: accounting amendment {target} does not pair {rel_path} with {file_sha}"
            )

    _verify_frozen_artifacts_if_present(artifacts_root(root))

    return {
        "amendment_path": str(target),
        "amendment_sha256": immutable_desc["sha256"],
        "adapter_commit": verified_commit,
        "adapter_inventory_sha256": inv_sha,
        "adapter_files": files_map,
    }


def record_ruling(
    ruling_md: Path,
    *,
    amendment_md: Path | None = None,
    out: Path | None = None,
    judge: str = "Astra",
    repo: Path | None = None,
    require_clean_git: bool = True,
) -> dict:
    """Bind Astra's re-ratification ruling to the immutable amendment and adapter hashes."""
    root = (repo or ADAPTER_ROOT).resolve()
    if ruling_md.is_symlink():
        raise ValueError(f"REFUSE: symlinked ruling markdown forbidden: {ruling_md}")
    resolved_ruling = ruling_md.resolve(strict=True)
    if (
        resolved_ruling.is_symlink()
        or not resolved_ruling.is_file()
        or resolved_ruling.stat().st_size == 0
    ):
        raise ValueError(f"ruling markdown must be a nonempty regular file: {resolved_ruling}")
    if judge not in ("Astra", "Fable"):
        raise ValueError("independent judge must be Astra (or Fable stand-in)")

    verdict = parse_ruling_verdict(resolved_ruling.read_text(encoding="utf-8"))
    amend = verify_amendment_document(
        amendment_md,
        repo=root,
        require_clean_git=require_clean_git,
    )
    target = resolve_ruling_json(out, repo=root)
    payload = {
        "schema": SCHEMA,
        "status": "RATIFIED",
        "verdict": verdict,
        "judge": judge,
        "implementer": "Argon",
        "independent_implementation_review": True,
        "namespace": BASELINE_NAMESPACE,
        "science": BASELINE_SCIENCE,
        "baseline_commit": BASELINE_COMMIT,
        "registration_path": str(
            (root / "research/watch6_search_efficacy_v3/REGISTRATION_V3.md").resolve()
        ),
        "registration_sha256": REGISTRATION_V3_SHA256,
        "ruling_2_sha256": RULING_2_SHA256,
        "ruling_3_sha256": RULING_3_SHA256,
        "path_ruling_sha256": SHARD00_PATH_RULING_SHA256,
        "shard00_proof_sha256": SHARD00_PROOF_SHA256,
        "shard00_receipt_sha256": SHARD00_RECEIPT_SHA256,
        "amendment_path": amend["amendment_path"],
        "amendment_sha256": amend["amendment_sha256"],
        "adapter_commit": amend["adapter_commit"],
        "adapter_inventory_sha256": amend["adapter_inventory_sha256"],
        "adapter_files": amend["adapter_files"],
        "ruling_path": str(resolved_ruling),
        "ruling_sha256": digest(resolved_ruling),
    }
    publish_json(target, payload)
    return payload


def validate_ruling_receipt(
    path: Path | None = None,
    *,
    amendment_override: Path | None = None,
    repo: Path | None = None,
    require_clean_git: bool = True,
) -> dict:
    """Fail closed if Astra's accounting re-ratification receipt is absent, unapproved, or stale."""
    root = (repo or ADAPTER_ROOT).resolve()
    raw_target = path if path is not None else resolve_ruling_json(None, repo=root)
    if raw_target.is_symlink():
        raise ValueError(f"REFUSE: symlinked accounting ruling receipt forbidden: {raw_target}")
    target = raw_target.resolve()
    if target.is_symlink() or not target.is_file():
        raise ValueError(
            f"REFUSE: missing ratified Astra accounting ruling receipt at {target} "
            "(run `python -m tools.watch6_v3_accounting.ratify --ruling <RULING.md>`)"
        )

    verify_immutable(target)
    receipt = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError(f"REFUSE: corrupt accounting ruling receipt at {target}")

    ruling_md = Path(str(receipt.get("ruling_path", "")))
    amendment_md = Path(str(receipt.get("amendment_path", "")))
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != "RATIFIED"
        or receipt.get("verdict") not in ALLOWED_VERDICTS
        or receipt.get("judge") not in ("Astra", "Fable")
        or receipt.get("implementer") != "Argon"
        or receipt.get("namespace") != BASELINE_NAMESPACE
        or receipt.get("namespace") != p.NAMESPACE
        or receipt.get("science") != BASELINE_SCIENCE
        or receipt.get("science") != p.identity()
        or receipt.get("baseline_commit") != BASELINE_COMMIT
        or receipt.get("registration_sha256") != REGISTRATION_V3_SHA256
        or receipt.get("ruling_2_sha256") != RULING_2_SHA256
        or receipt.get("ruling_3_sha256") != RULING_3_SHA256
        or receipt.get("path_ruling_sha256") != SHARD00_PATH_RULING_SHA256
        or receipt.get("shard00_proof_sha256") != SHARD00_PROOF_SHA256
        or receipt.get("shard00_receipt_sha256") != SHARD00_RECEIPT_SHA256
        or not ruling_md.is_absolute()
        or ruling_md.is_symlink()
        or not ruling_md.is_file()
        or digest(ruling_md) != receipt.get("ruling_sha256")
        or not amendment_md.is_absolute()
    ):
        raise ValueError(
            "REFUSE: Astra accounting ruling receipt is stale, invalid, or hash-mismatched"
        )

    if amendment_override is not None and amendment_md.resolve() != amendment_override.resolve():
        raise ValueError(
            f"REFUSE: ruling receipt amendment_path {amendment_md} "
            f"differs from {amendment_override}"
        )

    parsed_verdict = parse_ruling_verdict(ruling_md.read_text(encoding="utf-8"))
    if parsed_verdict != receipt.get("verdict"):
        raise ValueError("REFUSE: ruling markdown verdict does not match recorded receipt")

    rec_commit = receipt.get("adapter_commit")
    if not isinstance(rec_commit, str):
        raise ValueError("REFUSE: ruling receipt missing adapter_commit")

    amend = verify_amendment_document(
        amendment_md,
        repo=root,
        require_clean_git=require_clean_git,
        expected_adapter_commit=rec_commit,
    )
    if (
        receipt.get("amendment_sha256") != amend["amendment_sha256"]
        or receipt.get("adapter_commit") != amend["adapter_commit"]
        or receipt.get("adapter_inventory_sha256") != amend["adapter_inventory_sha256"]
        or receipt.get("adapter_files") != amend["adapter_files"]
    ):
        raise ValueError(
            "REFUSE: live adapter files or accounting amendment differ from ratified Astra receipt"
        )

    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ruling",
        type=Path,
        required=True,
        help=(
            "Path to Astra's re-ratification Markdown "
            "(must carry Verdict: APPROVE / APPROVE-WITH-CONDITIONS)"
        ),
    )
    parser.add_argument(
        "--amendment",
        type=Path,
        default=None,
        help="Path to immutable WATCH6_V3_ACCOUNTING_AMENDMENT_20261031.md",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for astra-accounting-ruling.json",
    )
    parser.add_argument("--judge", default="Astra", choices=("Astra", "Fable"))
    args = parser.parse_args()
    receipt = record_ruling(
        args.ruling,
        amendment_md=args.amendment,
        out=args.out,
        judge=args.judge,
    )
    validate_ruling_receipt(args.out, amendment_override=args.amendment)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
