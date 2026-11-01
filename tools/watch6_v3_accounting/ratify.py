"""Mac-side ratification and amendment verifier for the Watch6 V3 accounting adapter.

Implements Astra Ruling Points 1, 3, and 5:
- Verifies the immutable accounting amendment outside ``protocol.source_paths()``, including
  baseline commit, actual broker science identity computed from ``V3_BROKER_ROOT``, original
  registration hash, rulings 2/3 + shard-00 path ruling hashes, shard-00 proof/receipt hashes,
  complete 10-file adapter inventory hashes, and adapter commit.
- Reuses strict ``APPROVE`` / ``APPROVE-WITH-CONDITIONS`` verdict parsing; ``APPROVE-PATH``,
  ``REJECT``, ``NOT_RATIFIED``, missing, or conflicting verdicts are refused closed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from tools.watch6_v3_accounting import (
    ADAPTER_ROOT,
    AMENDMENT_V5_REL,
    AMENDMENT_V5_SHA256,
    AMENDMENT_V6_REL,
    AMENDMENT_V6_SHA256,
    BASELINE_COMMIT,
    BASELINE_NAMESPACE,
    BASELINE_SCIENCE,
    PINNED_SUPERVISOR_REL,
    PINNED_SUPERVISOR_SHA256,
    REGISTRATION_V3_SHA256,
    RULING_2_REL,
    RULING_2_SHA256,
    RULING_3_REL,
    RULING_3_SHA256,
    RULING_VERIFY_5_REL,
    RULING_VERIFY_5_SHA256,
    RULING_VERIFY_6_REL,
    RULING_VERIFY_6_SHA256,
    SHARD00_EXEC_MANIFEST_SHA256,
    SHARD00_PATH_RULING_REL,
    SHARD00_PATH_RULING_SHA256,
    SHARD00_PROOF_DISPATCH_REL,
    SHARD00_PROOF_REL,
    SHARD00_PROOF_SHA256,
    SHARD00_RECEIPT_REL,
    SHARD00_RECEIPT_SHA256,
    SHARD12_EVIDENCE_SIDECAR_REL,
    SHARD12_EVIDENCE_SIDECAR_SHA256,
    SHARD12_EXEC_MANIFEST_SHA256,
    SHARD12_LIVE_MANIFEST_SHA256,
    SHARD12_OPS_LOG_SHA256,
    SHARD12_OPS_MANIFEST_SHA256,
    SHARD12_PATH_RULING_REL,
    SHARD12_PATH_RULING_SHA256,
    SHARD12_RECEIPT_REL,
    SHARD12_RECEIPT_SHA256,
    V3_BROKER_ROOT,
    adapter_commit,
    adapter_files_map,
    adapter_inventory_sha256,
    artifacts_root,
    compute_broker_protocol,
    default_amendment_path,
    default_ruling_json_path,
    ensure_cli_dash_b,
    isolated_broker_execution,
    verify_baseline_closure,
)

# isort: split

from research.watch6_search_efficacy_v3.ratify import ALLOWED_VERDICTS, parse_ruling_verdict
from tools.ops.package_spec import digest, publish_json, verify_immutable

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


def _assert_outside_source_closure(
    path: Path,
    root: Path,
    source_paths: list[str] | None = None,
) -> None:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(root.resolve())
    except ValueError:
        return
    rel_str = rel.as_posix()
    if source_paths is not None:
        entries = source_paths
    else:
        raw_paths = compute_broker_protocol(V3_BROKER_ROOT)["source_paths"]
        entries = [str(item) for item in raw_paths]  # type: ignore[union-attr]
    for entry in entries:
        clean = entry.rstrip("/")
        if rel_str == clean or rel_str.startswith(clean + "/"):
            raise ValueError(
                f"REFUSE: accounting amendment {resolved} lies inside "
                f"hashed V3 source closure ({entry})"
            )


def _verify_frozen_artifacts_if_present(art_root: Path) -> None:
    expected_artifacts = (
        (RULING_2_REL, RULING_2_SHA256, False),
        (RULING_3_REL, RULING_3_SHA256, False),
        (SHARD00_PATH_RULING_REL, SHARD00_PATH_RULING_SHA256, False),
        (SHARD00_PROOF_REL, SHARD00_PROOF_SHA256, False),
        (SHARD00_PROOF_DISPATCH_REL, SHARD00_PROOF_SHA256, False),
        (SHARD00_RECEIPT_REL, SHARD00_RECEIPT_SHA256, False),
        (AMENDMENT_V5_REL, AMENDMENT_V5_SHA256, True),
        (RULING_VERIFY_5_REL, RULING_VERIFY_5_SHA256, False),
        (SHARD12_PATH_RULING_REL, SHARD12_PATH_RULING_SHA256, False),
        (SHARD12_RECEIPT_REL, SHARD12_RECEIPT_SHA256, False),
        (SHARD12_EVIDENCE_SIDECAR_REL, SHARD12_EVIDENCE_SIDECAR_SHA256, True),
        (AMENDMENT_V6_REL, AMENDMENT_V6_SHA256, True),
        (RULING_VERIFY_6_REL, RULING_VERIFY_6_SHA256, False),
    )
    for rel, expected_sha, require_sidecar in expected_artifacts:
        candidate = art_root / rel
        if candidate.is_symlink():
            raise ValueError(f"REFUSE: symlinked frozen reference artifact: {candidate}")
        if candidate.exists():
            if not candidate.is_file() or digest(candidate) != expected_sha:
                raise ValueError(
                    f"REFUSE: frozen reference artifact modified or mismatched: {candidate}"
                )
            if require_sidecar:
                verify_immutable(candidate)


def verify_amendment_document(
    amendment_md: Path | None = None,
    *,
    repo: Path | None = None,
    broker_root: Path | None = None,
    require_clean_git: bool = True,
    expected_adapter_commit: str | None = None,
) -> dict:
    """Verify immutable accounting amendment sidecars and exact adapter/broker hash bindings."""
    eff_broker = broker_root if broker_root is not None else V3_BROKER_ROOT
    with isolated_broker_execution("ratify-amendment", broker_root=eff_broker):
        adapter_repo = (repo or ADAPTER_ROOT).resolve()
        closure_info = verify_baseline_closure(eff_broker, require_head_at_baseline=True)
        resolved_broker = Path(closure_info["broker_root"])
        broker_proto = compute_broker_protocol(resolved_broker)
        raw_sp = broker_proto["source_paths"]
        source_paths = [str(item) for item in raw_sp]  # type: ignore[union-attr]
        broker_science = closure_info["science"]

        raw_target = (
            amendment_md
            if amendment_md is not None
            else resolve_amendment_path(None, repo=adapter_repo)
        )
        if raw_target.is_symlink():
            raise ValueError(f"REFUSE: symlinked accounting amendment forbidden: {raw_target}")
        target = raw_target.resolve()
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"REFUSE: missing accounting amendment document: {target}")

        _assert_outside_source_closure(target, adapter_repo, source_paths)
        _assert_outside_source_closure(target, resolved_broker, source_paths)
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
            adapter_repo,
            require_clean=require_clean_git,
            expected_commit=bound_commit,
        )
        files_map = adapter_files_map(adapter_repo)
        inv_sha = adapter_inventory_sha256(adapter_repo)

        sup_path = resolved_broker / PINNED_SUPERVISOR_REL
        if sup_path.is_symlink() or not sup_path.is_file():
            raise ValueError(f"REFUSE: missing pinned supervisor in broker: {sup_path}")
        if digest(sup_path) != PINNED_SUPERVISOR_SHA256:
            raise ValueError(
                f"REFUSE: broker supervisor.py hash {digest(sup_path)} != "
                f"{PINNED_SUPERVISOR_SHA256}"
            )

        required_tokens = (
            ("v3_broker_root", str(V3_BROKER_ROOT)),
            ("baseline_commit", BASELINE_COMMIT),
            ("baseline_science", broker_science),
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
            ("amendment_v5_rel", AMENDMENT_V5_REL),
            ("amendment_v5_sha256", AMENDMENT_V5_SHA256),
            ("ruling_verify_5_rel", RULING_VERIFY_5_REL),
            ("ruling_verify_5_sha256", RULING_VERIFY_5_SHA256),
            ("amendment_v6_rel", AMENDMENT_V6_REL),
            ("amendment_v6_sha256", AMENDMENT_V6_SHA256),
            ("ruling_verify_6_rel", RULING_VERIFY_6_REL),
            ("ruling_verify_6_sha256", RULING_VERIFY_6_SHA256),
            ("shard12_path_ruling_rel", SHARD12_PATH_RULING_REL),
            ("shard12_path_ruling_sha256", SHARD12_PATH_RULING_SHA256),
            ("shard12_evidence_sidecar_rel", SHARD12_EVIDENCE_SIDECAR_REL),
            ("shard12_evidence_sidecar_sha256", SHARD12_EVIDENCE_SIDECAR_SHA256),
            ("shard12_receipt_rel", SHARD12_RECEIPT_REL),
            ("shard12_receipt_sha256", SHARD12_RECEIPT_SHA256),
            ("shard12_exec_manifest_sha256", SHARD12_EXEC_MANIFEST_SHA256),
            ("shard12_ops_manifest_sha256", SHARD12_OPS_MANIFEST_SHA256),
            ("shard12_live_manifest_sha256", SHARD12_LIVE_MANIFEST_SHA256),
            ("shard12_ops_log_sha256", SHARD12_OPS_LOG_SHA256),
            ("pinned_supervisor_rel", PINNED_SUPERVISOR_REL),
            ("pinned_supervisor_sha256", PINNED_SUPERVISOR_SHA256),
            ("adapter_commit", verified_commit),
            ("adapter_inventory_sha256", inv_sha),
        )
        for label, token in required_tokens:
            if token not in text:
                raise ValueError(
                    f"REFUSE: accounting amendment {target} missing required {label} "
                    f"binding ({token})"
                )

        for rel_path, file_sha in sorted(files_map.items()):
            if rel_path not in text or file_sha not in text:
                raise ValueError(
                    f"REFUSE: accounting amendment {target} missing exact hash binding "
                    f"for {rel_path}"
                )
            pattern = re.compile(re.escape(rel_path) + r"[^\n]*" + re.escape(file_sha))
            rev_pattern = re.compile(re.escape(file_sha) + r"[^\n]*" + re.escape(rel_path))
            if pattern.search(text) is None and rev_pattern.search(text) is None:
                raise ValueError(
                    f"REFUSE: accounting amendment {target} does not pair "
                    f"{rel_path} with {file_sha}"
                )

        _verify_frozen_artifacts_if_present(artifacts_root(adapter_repo))

        return {
            "amendment_path": str(target),
            "amendment_sha256": immutable_desc["sha256"],
            "broker_root": str(resolved_broker),
            "science": broker_science,
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
    broker_root: Path | None = None,
    require_clean_git: bool = True,
) -> dict:
    """Bind Astra's re-ratification ruling to the amendment, broker, and adapter hashes."""
    eff_broker = broker_root if broker_root is not None else V3_BROKER_ROOT
    with isolated_broker_execution("ratify-record", broker_root=eff_broker):
        adapter_repo = (repo or ADAPTER_ROOT).resolve()
        closure_info = verify_baseline_closure(eff_broker, require_head_at_baseline=True)
        resolved_broker = Path(closure_info["broker_root"])
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
            repo=adapter_repo,
            broker_root=resolved_broker,
            require_clean_git=require_clean_git,
        )
        target = resolve_ruling_json(out, repo=adapter_repo)
        payload = {
            "schema": SCHEMA,
            "status": "RATIFIED",
            "verdict": verdict,
            "judge": judge,
            "implementer": "Argon",
            "independent_implementation_review": True,
            "broker_root": str(resolved_broker),
            "namespace": closure_info["namespace"],
            "science": closure_info["science"],
            "baseline_commit": closure_info["baseline_commit"],
            "registration_path": str(
                (
                    resolved_broker / "research/watch6_search_efficacy_v3/REGISTRATION_V3.md"
                ).resolve()
            ),
            "registration_sha256": closure_info["registration_sha256"],
            "ruling_2_sha256": RULING_2_SHA256,
            "ruling_3_sha256": RULING_3_SHA256,
            "path_ruling_sha256": SHARD00_PATH_RULING_SHA256,
            "shard00_proof_sha256": SHARD00_PROOF_SHA256,
            "shard00_receipt_sha256": SHARD00_RECEIPT_SHA256,
            "amendment_v5_sha256": AMENDMENT_V5_SHA256,
            "verify_5_ruling_sha256": RULING_VERIFY_5_SHA256,
            "amendment_v6_sha256": AMENDMENT_V6_SHA256,
            "verify_6_ruling_sha256": RULING_VERIFY_6_SHA256,
            "shard12_path_ruling_sha256": SHARD12_PATH_RULING_SHA256,
            "shard12_evidence_sidecar_sha256": SHARD12_EVIDENCE_SIDECAR_SHA256,
            "shard12_receipt_sha256": SHARD12_RECEIPT_SHA256,
            "pinned_supervisor_sha256": PINNED_SUPERVISOR_SHA256,
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
    broker_root: Path | None = None,
    require_clean_git: bool = True,
) -> dict:
    """Fail closed if Astra's accounting re-ratification receipt is absent, unapproved, or stale."""
    eff_broker = broker_root if broker_root is not None else V3_BROKER_ROOT
    with isolated_broker_execution("ratify-validate", broker_root=eff_broker):
        adapter_repo = (repo or ADAPTER_ROOT).resolve()
        closure_info = verify_baseline_closure(eff_broker, require_head_at_baseline=True)
        resolved_broker = Path(closure_info["broker_root"])
        broker_science = closure_info["science"]

        raw_target = path if path is not None else resolve_ruling_json(None, repo=adapter_repo)
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
        rec_broker_raw = receipt.get("broker_root")
        if not isinstance(rec_broker_raw, str) or Path(rec_broker_raw).resolve() != resolved_broker:
            raise ValueError(
                f"REFUSE: ruling receipt broker_root {rec_broker_raw!r} "
                f"differs from verified broker {resolved_broker}"
            )

        if (
            receipt.get("schema") != SCHEMA
            or receipt.get("status") != "RATIFIED"
            or receipt.get("verdict") not in ALLOWED_VERDICTS
            or receipt.get("judge") not in ("Astra", "Fable")
            or receipt.get("implementer") != "Argon"
            or receipt.get("namespace") != BASELINE_NAMESPACE
            or receipt.get("namespace") != closure_info["namespace"]
            or receipt.get("science") != BASELINE_SCIENCE
            or receipt.get("science") != broker_science
            or receipt.get("baseline_commit") != BASELINE_COMMIT
            or receipt.get("registration_sha256") != REGISTRATION_V3_SHA256
            or receipt.get("ruling_2_sha256") != RULING_2_SHA256
            or receipt.get("ruling_3_sha256") != RULING_3_SHA256
            or receipt.get("path_ruling_sha256") != SHARD00_PATH_RULING_SHA256
            or receipt.get("shard00_proof_sha256") != SHARD00_PROOF_SHA256
            or receipt.get("shard00_receipt_sha256") != SHARD00_RECEIPT_SHA256
            or receipt.get("amendment_v5_sha256") != AMENDMENT_V5_SHA256
            or receipt.get("verify_5_ruling_sha256") != RULING_VERIFY_5_SHA256
            or receipt.get("amendment_v6_sha256") != AMENDMENT_V6_SHA256
            or receipt.get("verify_6_ruling_sha256") != RULING_VERIFY_6_SHA256
            or receipt.get("shard12_path_ruling_sha256") != SHARD12_PATH_RULING_SHA256
            or receipt.get("shard12_evidence_sidecar_sha256") != SHARD12_EVIDENCE_SIDECAR_SHA256
            or receipt.get("shard12_receipt_sha256") != SHARD12_RECEIPT_SHA256
            or receipt.get("pinned_supervisor_sha256") != PINNED_SUPERVISOR_SHA256
            or not ruling_md.is_absolute()
            or ruling_md.is_symlink()
            or not ruling_md.is_file()
            or digest(ruling_md) != receipt.get("ruling_sha256")
            or not amendment_md.is_absolute()
        ):
            raise ValueError(
                "REFUSE: Astra accounting ruling receipt is stale, invalid, or hash-mismatched"
            )

        if (
            amendment_override is not None
            and amendment_md.resolve() != amendment_override.resolve()
        ):
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
            repo=adapter_repo,
            broker_root=resolved_broker,
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
                "REFUSE: live adapter files or accounting amendment differ from ratified "
                "Astra receipt"
            )

        return receipt


def main() -> None:
    ensure_cli_dash_b("tools.watch6_v3_accounting.ratify")
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
        help="Path to immutable WATCH6_V3_ACCOUNTING_AMENDMENT_V6_1_20261101.md",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for astra-accounting-ruling.json",
    )
    parser.add_argument(
        "--broker-root",
        "--broker",
        dest="broker_root",
        type=Path,
        default=V3_BROKER_ROOT,
        help="Path to clean V3 broker root at 357e3504",
    )
    parser.add_argument("--judge", default="Astra", choices=("Astra", "Fable"))
    args = parser.parse_args()
    receipt = record_ruling(
        args.ruling,
        amendment_md=args.amendment,
        out=args.out,
        judge=args.judge,
        broker_root=args.broker_root,
    )
    validate_ruling_receipt(
        args.out,
        amendment_override=args.amendment,
        broker_root=args.broker_root,
    )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
