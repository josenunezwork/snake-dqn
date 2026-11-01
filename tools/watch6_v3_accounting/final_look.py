"""Sole Mac final-look orchestrator for Watch6 V3 confirmation via the accounting adapter.

Implements Astra Ruling Point 7 (narrowly superseding Ruling 3's entrypoint condition; no
standalone ``analyze.py``):
1. Verifies Mac host, actual-byte baseline V3 closure equality on the real V3 broker root
   ``/Users/josenunez/Projects/ml/snake-dqn-argon-v3-confirm`` at ``357e3504`` (refusing the main
   checkout and any cross-root substitution), and Astra's ratified accounting ruling receipt
   binding the immutable amendment and 10-file adapter inventory.
2. Audits the complete shared ``<ops_dir>/manifests/*.json`` registry, retaining shard 00's
   PROVEN never-allocated attempt verbatim as ``prestart`` lineage while requiring clean
   ``stop_verified=True`` (`cleanup_error is None`) on all allocated attempts.
3. Enforces unbroken linear retry-cap lineage (`<=2` failed compute completions per shard),
   L4-only W4 placement (no A100 substitution/fallback), immutable sidecars/export inventories,
   and exact 96/96 primary-slot coverage BEFORE opening any outcome payload.
4. Any failed prerequisite raises ``INVALID_OR_INCOMPLETE`` without taking an outcome look.
5. Once all prerequisites pass, reads the 96 ``pair-XX.json`` payloads once and invokes the
   unchanged reducer ``research.watch6_search_efficacy_v3.analyze.decide_v3`` once.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path

from tools.watch6_v3_accounting import (
    ADAPTER_ROOT,
    BASELINE_COMMIT,
    BASELINE_SCIENCE,
    V3_BROKER_ROOT,
    bind_broker_protocol,
    ensure_cli_dash_b,
    isolated_broker_execution,
)

# isort: split

from research.colab_cli import canary
from research.watch6_search_efficacy_v3 import admission as v3_admission
from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.analyze import decide_v3
from tools.ops.package_spec import digest, publish_json, verify_immutable
from tools.watch6_v3_accounting.admission import (
    is_external_transport_compute_failure,
    is_prestart_transport_failure,
)
from tools.watch6_v3_accounting.dispatch import (
    DEFAULT_OPS_DIR,
    FALLBACK_SUFFIX,
    _fallback_sibling_name,
    _log_has_exit,
)
from tools.watch6_v3_accounting.proof import (
    has_forbidden_or_contradictory_output_artifacts,
    is_proven_never_allocated_attempt,
    verify_never_allocated_attempt,
    verify_no_receipt_unallocated_proof,
)
from tools.watch6_v3_accounting.ratify import validate_ruling_receipt
from tools.watch6_v3_accounting.resume import PAIR_NAME_RE
from tools.watch6_v3_accounting.resume import build as build_verified_ledger


@dataclass(frozen=True)
class _ShardAttempt:
    shard: int
    ops_name: str
    attempt_dir: Path
    receipt_path: Path
    receipt_sha256: str
    manifest: dict
    receipt: dict
    classification: str  # "pass" | "prestart" | "external_transport_compute"
    never_allocated_proof: dict | None = None


def _fail_prelook(message: str) -> ValueError:
    if "INVALID_OR_INCOMPLETE" in message:
        return ValueError(message)
    return ValueError(f"INVALID_OR_INCOMPLETE: {message}")


def _verify_auto_retry_sidecar(path: Path) -> None:
    """Verify RETRY.json / auto-retry manifest sidecars (1-col auto_retry.py or 2-col)."""
    for suffix in ("", ".sha256", ".complete"):
        target = Path(f"{path}{suffix}")
        if target.is_symlink() or not target.is_file():
            raise _fail_prelook(f"missing or unsafe auto-retry immutable file: {path}")
    if Path(f"{path}.complete").read_bytes() != b"complete\n":
        raise _fail_prelook(f"invalid auto-retry completion marker: {path}")
    observed = digest(path)
    tokens = Path(f"{path}.sha256").read_text(encoding="utf-8").split()
    if tokens not in ([observed], [observed, path.name]):
        raise _fail_prelook(f"auto-retry sidecar SHA256 mismatch: {path}")


def _is_valid_no_output_unallocated_proof(proof_path: Path, out_dir: Path) -> bool:
    return verify_no_receipt_unallocated_proof(proof_path, out_dir)


def _is_proven_unallocated_without_receipt(ops_dir: Path, name: str, out_dir: Path) -> bool:
    """True iff an ops manifest exited before creating receipt.json and never allocated."""
    if (out_dir / "receipt.json").exists() or (out_dir / "receipt.json").is_symlink():
        return False
    if has_forbidden_or_contradictory_output_artifacts(out_dir):
        return False
    log_path = ops_dir / "logs" / f"{name}.log"
    if (
        not out_dir.exists()
        and not out_dir.is_symlink()
        and not (ops_dir / "dispatch" / name).exists()
        and _log_has_exit(log_path)
    ):
        return True
    disp_dir = ops_dir / "dispatch" / name
    raw_candidates = [
        disp_dir / "unavailable.json",
        *(sorted(disp_dir.glob("unallocated*.json")) if disp_dir.is_dir() else []),
        *sorted(ops_dir.glob(f"unallocated-{name}*.json")),
    ]
    existing: list[Path] = []
    seen: set[Path] = set()
    for cand in raw_candidates:
        if cand.is_symlink():
            return False
        if cand.exists():
            if not cand.is_file():
                return False
            key = cand.resolve()
            if key not in seen:
                seen.add(key)
                existing.append(cand)
    if not existing:
        return False
    if len({digest(c) for c in existing}) != 1:
        return False
    return all(_is_valid_no_output_unallocated_proof(cand, out_dir) for cand in existing)


def _classify_attempt(
    attempt_dir: Path,
    receipt: dict,
    *,
    ops_dir: Path,
    ops_name: str,
) -> str:
    """Classify a V3 attempt outcome-blind; raise INVALID_OR_INCOMPLETE on any forbidden fault."""
    if receipt.get("status") == "PASS":
        terminal = receipt.get("terminal")
        if (
            receipt.get("stop_verified") is not True
            or receipt.get("cleanup_error") is not None
            or not isinstance(receipt.get("endpoint"), str)
            or not receipt.get("endpoint")
            or receipt.get("failure_class") is not None
            or receipt.get("error") is not None
            or not isinstance(terminal, dict)
            or terminal.get("returncode") != 0
            or terminal.get("reason") is not None
            or terminal.get("error") is not None
        ):
            raise _fail_prelook(f"malformed or unstopped PASS receipt/terminal in {attempt_dir}")
        return "pass"
    if is_prestart_transport_failure(receipt, attempt_dir, ops_dir=ops_dir, ops_name=ops_name):
        return "prestart"
    if is_external_transport_compute_failure(receipt, attempt_dir):
        return "external_transport_compute"
    raise _fail_prelook(
        f"forbidden failure class in {attempt_dir}: "
        f"status={receipt.get('status')!r} failure_class={receipt.get('failure_class')!r} "
        f"error={receipt.get('error')!r}"
    )


def _verify_attempt_stop_and_hashes(
    attempt_dir: Path,
    *,
    ops_dir: Path,
    ops_name: str,
) -> tuple[dict, dict, Path, str, dict | None]:
    """Verify receipt.json stop/cleanup state (or Ruling Point 4 proof) and manifest binding."""
    raw_rec = attempt_dir / "receipt.json"
    if raw_rec.is_symlink() or not raw_rec.is_file():
        raise _fail_prelook(f"unfinished V3 runtime (missing receipt.json): {attempt_dir}")
    receipt_path = raw_rec.resolve()
    exec_manifest_path = attempt_dir / "manifest.json"
    if exec_manifest_path.is_symlink() or not exec_manifest_path.is_file():
        raise _fail_prelook(f"missing executed manifest.json: {attempt_dir}")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    exec_manifest = json.loads(exec_manifest_path.read_text(encoding="utf-8"))
    if (
        digest(exec_manifest_path) != receipt.get("manifest_sha256")
        or receipt.get("job_id") != exec_manifest.get("job_id")
        or receipt.get("commit") != exec_manifest.get("commit")
        or exec_manifest.get("commit") != BASELINE_COMMIT
    ):
        raise _fail_prelook(f"executed manifest SHA256/identity/commit mismatch: {receipt_path}")

    req = exec_manifest.get("requirements", {})
    if req.get("gpu") != "L4" or req.get("allowed_gpus") != ["L4"] or "prefer" in req:
        raise _fail_prelook(f"non-L4 or A100-substituted V3 execution forbidden: {attempt_dir}")

    prealloc_clean_unassigned = (
        receipt.get("endpoint") is None
        and receipt.get("stop_verified") is False
        and receipt.get("cleanup_error") is None
        and not has_forbidden_or_contradictory_output_artifacts(
            attempt_dir,
            job_id=(
                exec_manifest.get("job_id")
                if isinstance(exec_manifest.get("job_id"), str)
                else None
            ),
            session=receipt.get("session") if isinstance(receipt.get("session"), str) else None,
        )
        and v3_admission.is_prestart_transport_failure(receipt, attempt_dir)
    )
    never_alloc_proof: dict | None = None
    if is_proven_never_allocated_attempt(
        receipt,
        attempt_dir,
        ops_dir=ops_dir,
        ops_name=ops_name,
    ):
        never_alloc_proof = verify_never_allocated_attempt(
            attempt_dir,
            receipt=receipt,
            ops_dir=ops_dir,
            ops_name=ops_name,
        )

    if not (prealloc_clean_unassigned or never_alloc_proof is not None) and (
        receipt.get("stop_verified") is not True or receipt.get("cleanup_error") is not None
    ):
        raise _fail_prelook(f"unstopped or unclean V3 runtime: {receipt_path}")

    return exec_manifest, receipt, receipt_path, digest(receipt_path), never_alloc_proof


def _verify_shard_lineage(
    shard: int,
    attempts: list[_ShardAttempt],
    unallocated_manifests: set[Path],
) -> list[_ShardAttempt]:
    """Verify complete, unbroken linear retry-cap lineage and <=2 compute completions for shard."""
    if not attempts:
        raise _fail_prelook(f"shard {shard} (slots {p.shard_indices(shard)}) has zero attempts")

    by_receipt_path = {att.receipt_path: att for att in attempts}
    parent_of: dict[Path, Path | None] = {}
    recover_history_of: dict[Path, list[tuple[Path, str, str]]] = {}

    for att in attempts:
        name_dir = att.attempt_dir.name
        retry_json = att.attempt_dir.parent / "RETRY.json"
        top_comp = att.manifest.get("completion_attempt", 0)
        if type(top_comp) is not int or not 0 <= top_comp <= p.MAX_COMPUTE_COMPLETIONS_PER_SHARD:
            raise _fail_prelook(
                f"invalid top-level completion_attempt={top_comp!r} "
                f"on shard {shard} ({att.attempt_dir})"
            )

        if name_dir.endswith("-retry-1"):
            base_dir = Path(str(att.attempt_dir)[: -len("-retry-1")]).resolve()
            base_receipt = (base_dir / "receipt.json").resolve()
            base_att = by_receipt_path.get(base_receipt)
            if base_att is None or base_att.classification != "prestart":
                raise _fail_prelook(
                    f"-retry-1 attempt {att.attempt_dir} lacks verified "
                    f"pre-start base receipt {base_receipt}"
                )
            parent_of[att.receipt_path] = base_receipt
        elif retry_json.is_file():
            _verify_auto_retry_sidecar(retry_json)
            _verify_auto_retry_sidecar(att.attempt_dir.parent / "manifest.json")
            proof = json.loads(retry_json.read_text(encoding="utf-8"))
            if (
                proof.get("schema") != "ops-zero-export-retry/v1"
                or proof.get("completion_attempt") != top_comp
                or not 1 <= top_comp <= p.MAX_COMPUTE_COMPLETIONS_PER_SHARD
                or proof.get("manifest_sha256") != digest(att.attempt_dir.parent / "manifest.json")
            ):
                raise _fail_prelook(
                    f"malformed auto-retry RETRY.json on shard {shard}: {retry_json}"
                )
            unalloc_ref = proof.get("unallocated_proof")
            prior_rec_info = proof.get("prior_receipt", {})
            prior_rec_path_raw = prior_rec_info.get("path", "")
            prior_rec_path = Path(prior_rec_path_raw).resolve() if prior_rec_path_raw else None
            if prior_rec_path is not None and prior_rec_path in by_receipt_path:
                prior_att = by_receipt_path[prior_rec_path]
                if prior_att.receipt_sha256 != prior_rec_info.get("sha256"):
                    raise _fail_prelook(
                        f"auto-retry prior_receipt SHA256 mismatch on shard {shard}"
                    )
                parent_of[att.receipt_path] = prior_rec_path
            elif unalloc_ref is not None:
                u_path = Path(unalloc_ref.get("path", ""))
                prior_mf = Path(proof.get("prior_manifest", {}).get("path", "")).resolve()
                if (
                    not u_path.is_file()
                    or digest(u_path) != unalloc_ref.get("sha256")
                    or prior_mf not in unallocated_manifests
                ):
                    raise _fail_prelook(
                        f"auto-retry unallocated_proof mismatch on shard {shard}: {retry_json}"
                    )
                parent_of[att.receipt_path] = None
            else:
                raise _fail_prelook(
                    f"auto-retry missing prior receipt {prior_rec_path} in shard {shard} lineage"
                )
        else:
            if top_comp != 0:
                raise _fail_prelook(
                    f"completion_attempt={top_comp} without RETRY.json "
                    f"on shard {shard} ({att.attempt_dir})"
                )
            binding = att.manifest.get("watch6_v3", {})
            pre_u = binding.get("prestart_retries_used")
            comp_u = binding.get("compute_completion_attempt")
            tot_d = binding.get("total_dispatches_for_shard")
            priors = binding.get("prior_receipts", [])
            if (
                type(pre_u) is not int
                or pre_u < 0
                or type(comp_u) is not int
                or not 0 <= comp_u <= p.MAX_COMPUTE_COMPLETIONS_PER_SHARD
                or type(tot_d) is not int
                or tot_d != 1 + pre_u + comp_u
                or not isinstance(priors, list)
                or len(priors) != pre_u + comp_u
            ):
                raise _fail_prelook(
                    f"invalid retry-cap counters on shard {shard} ({att.attempt_dir})"
                )
            if tot_d == 1:
                parent_of[att.receipt_path] = None
            else:
                history: list[tuple[Path, str, str]] = []
                for entry in priors:
                    if not isinstance(entry, dict):
                        raise _fail_prelook(f"malformed prior_receipts on shard {shard}")
                    pr_path = Path(entry.get("receipt_path", "")).resolve()
                    pr_sha = str(entry.get("receipt_sha256", ""))
                    pr_cls = str(entry.get("classification", ""))
                    pr_att = by_receipt_path.get(pr_path)
                    if (
                        pr_att is None
                        or pr_att.receipt_sha256 != pr_sha
                        or pr_att.classification != pr_cls
                    ):
                        raise _fail_prelook(
                            f"missing or mismatched prior receipt {pr_path} "
                            f"in shard {shard} lineage"
                        )
                    history.append((pr_path, pr_sha, pr_cls))
                parent_of[att.receipt_path] = history[-1][0]
                recover_history_of[att.receipt_path] = history

    roots = [r_path for r_path, parent in parent_of.items() if parent is None]
    if len(roots) != 1:
        raise _fail_prelook(f"shard {shard} requires exactly 1 root attempt, found {len(roots)}")

    children_of: dict[Path, list[Path]] = {att.receipt_path: [] for att in attempts}
    for child_path, parent_path in parent_of.items():
        if parent_path is not None:
            children_of[parent_path].append(child_path)
    if any(len(kids) > 1 for kids in children_of.values()):
        raise _fail_prelook(f"forked/concurrent attempt lineage detected on shard {shard}")

    chain: list[_ShardAttempt] = []
    cursor: Path | None = roots[0]
    visited: set[Path] = set()
    while cursor is not None:
        if cursor in visited:
            raise _fail_prelook(f"cycle in shard {shard} retry lineage")
        visited.add(cursor)
        chain.append(by_receipt_path[cursor])
        next_kids = children_of[cursor]
        cursor = next_kids[0] if next_kids else None

    if len(chain) != len(attempts):
        raise _fail_prelook(
            f"disconnected attempts on shard {shard} "
            f"({len(chain)}/{len(attempts)} in linear chain)"
        )

    for idx, att in enumerate(chain):
        if att.receipt_path in recover_history_of:
            expected_hist = [
                (prev.receipt_path, prev.receipt_sha256, prev.classification)
                for prev in chain[:idx]
            ]
            if recover_history_of[att.receipt_path] != expected_hist:
                raise _fail_prelook(f"recover prior_receipts gap or reordering on shard {shard}")

    for earlier in chain[:-1]:
        if earlier.classification == "pass":
            raise _fail_prelook(
                f"shard {shard} recorded extra attempt after PASS receipt {earlier.receipt_path}"
            )

    if chain[-1].classification != "pass":
        raise _fail_prelook(
            f"shard {shard} terminal attempt did not PASS "
            f"({chain[-1].classification}: {chain[-1].receipt_path})"
        )

    failed_compute_attempts = sum(
        1 for att in chain[:-1] if att.classification == "external_transport_compute"
    )
    if failed_compute_attempts > p.MAX_COMPUTE_COMPLETIONS_PER_SHARD:
        raise _fail_prelook(
            f"shard {shard} exceeded compute completion retry cap "
            f"({failed_compute_attempts} > {p.MAX_COMPUTE_COMPLETIONS_PER_SHARD})"
        )
    return chain


def _audit_pre_outcome(
    *,
    ops_dir: Path,
    out_dir: Path,
    ruling_json: Path | None,
    amendment_md: Path | None,
    adapter_repo: Path,
    resolved_broker: Path,
    closure_info: dict[str, str],
    require_clean_git: bool,
) -> tuple[dict, dict, list[Path], list[tuple[Path, str]], list[_ShardAttempt], list[Path], Path]:
    if platform.system() != "Darwin":
        raise _fail_prelook("final look must run on Mac")

    broker_science = closure_info["science"]
    ruling_receipt = validate_ruling_receipt(
        ruling_json,
        amendment_override=amendment_md,
        repo=adapter_repo,
        broker_root=resolved_broker,
        require_clean_git=require_clean_git,
    )

    ops_dir = ops_dir.resolve(strict=True)
    manifests_dir = ops_dir / "manifests"
    if not manifests_dir.is_dir():
        raise _fail_prelook(f"missing ops manifests directory: {manifests_dir}")

    canary_attempts: list[tuple[Path, str]] = []
    shard_attempts: dict[int, list[_ShardAttempt]] = {s: [] for s in range(p.SHARDS)}
    unallocated_manifests: set[Path] = set()
    seen_output_dirs: set[Path] = set()

    for mf_path in sorted(manifests_dir.glob("*.json")):
        if mf_path.name.endswith(".live.json"):
            continue
        if mf_path.is_symlink() or not mf_path.is_file():
            raise _fail_prelook(f"symlinked or irregular ops manifest in registry: {mf_path}")
        try:
            data = json.loads(mf_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _fail_prelook(f"corrupt ops manifest in shared registry: {mf_path}") from exc
        if not isinstance(data, dict):
            raise _fail_prelook(f"non-object ops manifest in shared registry: {mf_path}")

        binding = data.get("watch6_v3")
        if not isinstance(binding, dict) or binding.get("namespace") != p.NAMESPACE:
            if data.get("script") == p.SCRIPT:
                raise _fail_prelook(f"V3 script manifest missing watch6_v3 binding: {mf_path}")
            continue

        if binding.get("science") != broker_science or binding.get("science") != BASELINE_SCIENCE:
            raise _fail_prelook(f"foreign science identity under V3 namespace in {mf_path}")
        if data.get("commit") != BASELINE_COMMIT:
            raise _fail_prelook(f"foreign runtime commit under V3 namespace in {mf_path}")
        if any(k in binding for k in ("reserve_allocation", "reserve_slots", "reserve_lineage")):
            raise _fail_prelook(f"forbidden reserve fields in V3 manifest {mf_path}")

        req = data.get("requirements", {})
        name = mf_path.stem
        if (
            name.endswith(FALLBACK_SUFFIX)
            or req.get("gpu") != "L4"
            or req.get("allowed_gpus") != ["L4"]
            or "prefer" in req
        ):
            raise _fail_prelook(
                f"forbidden non-L4 or A100-substituted V3 manifest in registry: {mf_path}"
            )

        fb_name = _fallback_sibling_name(name)
        fb_out = Path(str(data["output_dir"]) + FALLBACK_SUFFIX)
        if (manifests_dir / f"{fb_name}.json").exists() or fb_out.exists() or fb_out.is_symlink():
            raise _fail_prelook(f"forbidden A100 fallback sibling detected for V3 job {name}")

        raw_out = Path(data["output_dir"]).resolve()
        retry_one_out = Path(str(data["output_dir"]) + "-retry-1").resolve()
        separate_retry_mf = (manifests_dir / f"{name}-retry-1.json").is_file()

        candidate_dirs: list[Path] = []
        if raw_out.exists() or raw_out.is_symlink():
            candidate_dirs.append(raw_out)
        if not separate_retry_mf and (retry_one_out.exists() or retry_one_out.is_symlink()):
            candidate_dirs.append(retry_one_out)

        if not candidate_dirs:
            if _is_proven_unallocated_without_receipt(ops_dir, name, raw_out):
                unallocated_manifests.add(mf_path.resolve())
                if "auto-retry" in str(data["output_dir"]):
                    orig_auto_mf = (Path(data["output_dir"]).parent / "manifest.json").resolve()
                    if orig_auto_mf.is_file():
                        unallocated_manifests.add(orig_auto_mf)
                continue
            raise _fail_prelook(f"unfinished or unverified V3 ops manifest {mf_path} ({raw_out})")

        for attempt_dir in candidate_dirs:
            if attempt_dir in seen_output_dirs:
                continue
            has_bound_receipt = (attempt_dir / "receipt.json").is_file() and (
                attempt_dir / "manifest.json"
            ).is_file()
            if not has_bound_receipt and _is_proven_unallocated_without_receipt(
                ops_dir, name, attempt_dir
            ):
                unallocated_manifests.add(mf_path.resolve())
                continue
            seen_output_dirs.add(attempt_dir)

            exec_manifest, receipt, receipt_path, receipt_sha, never_alloc_proof = (
                _verify_attempt_stop_and_hashes(
                    attempt_dir,
                    ops_dir=ops_dir,
                    ops_name=name,
                )
            )
            exec_binding = exec_manifest.get("watch6_v3", {})
            if (
                exec_binding.get("namespace") != p.NAMESPACE
                or exec_binding.get("science") != broker_science
                or exec_binding.get("science") != BASELINE_SCIENCE
                or exec_binding.get("primary_allocation") != p.allocation()
                or any(
                    k in exec_binding
                    for k in ("reserve_allocation", "reserve_slots", "reserve_lineage")
                )
            ):
                raise _fail_prelook(f"executed manifest allocation/science mismatch: {attempt_dir}")

            classification = _classify_attempt(
                attempt_dir,
                receipt,
                ops_dir=ops_dir,
                ops_name=name,
            )
            if (
                canary.is_canary(exec_manifest)
                or exec_manifest.get("env", {}).get("WATCH6_FIXTURE") == "1"
            ):
                canary_attempts.append((receipt_path, classification))
            else:
                env = exec_manifest.get("env", {})
                if int(env.get("WATCH6_SHARDS", -1)) != p.SHARDS:
                    raise _fail_prelook(f"wrong WATCH6_SHARDS in {attempt_dir}")
                shard = int(env.get("WATCH6_SHARD", -1))
                if shard not in range(p.SHARDS):
                    raise _fail_prelook(f"invalid shard {shard} in {attempt_dir}")
                shard_attempts[shard].append(
                    _ShardAttempt(
                        shard=shard,
                        ops_name=name,
                        attempt_dir=attempt_dir,
                        receipt_path=receipt_path,
                        receipt_sha256=receipt_sha,
                        manifest=exec_manifest,
                        receipt=receipt,
                        classification=classification,
                        never_allocated_proof=never_alloc_proof,
                    )
                )

    passing_canaries = [r_path for r_path, cls in canary_attempts if cls == "pass"]
    if not passing_canaries:
        raise _fail_prelook("missing passing V3 L4 canary receipt in ops manifests")

    ordered_main_attempts: list[_ShardAttempt] = []
    for shard in range(p.SHARDS):
        chain = _verify_shard_lineage(shard, shard_attempts[shard], unallocated_manifests)
        ordered_main_attempts.extend(chain)

    main_receipts = [att.receipt_path for att in ordered_main_attempts]
    unit_paths: list[Path] = []
    seen_slots: dict[int, Path] = {}

    for att in ordered_main_attempts:
        declared_pairs: set[Path] = set()
        for row in att.receipt.get("exports", []):
            rel = str(row.get("path", ""))
            if not rel.startswith("efficacy/"):
                continue
            fname = Path(rel).name
            match = PAIR_NAME_RE.fullmatch(fname)
            if match is None:
                continue
            slot = int(match.group(1))
            candidate = (att.attempt_dir / "exports" / rel).resolve(strict=True)
            declared_pairs.add(candidate)
            if slot in seen_slots:
                raise _fail_prelook(
                    f"duplicate exported slot {slot}: {seen_slots[slot]} vs {candidate}"
                )
            seen_slots[slot] = candidate
            unit_paths.append(candidate)

        efficacy_dir = att.attempt_dir / "exports/efficacy"
        if efficacy_dir.is_dir():
            for disk_file in sorted(efficacy_dir.iterdir()):
                if PAIR_NAME_RE.fullmatch(disk_file.name) is not None:
                    if disk_file.resolve() not in declared_pairs:
                        raise _fail_prelook(
                            f"unrecorded pair file on disk outside receipt exports: {disk_file}"
                        )

    out_dir.mkdir(parents=True, exist_ok=False)
    ledger = build_verified_ledger(
        unit_paths,
        main_receipts,
        ops_dir=ops_dir,
        broker_root=resolved_broker,
    )
    ledger_path = out_dir / "final-verified-ledger.json"
    publish_json(ledger_path, ledger)

    verified_slots = {row["index"] for row in ledger["units"]}
    if len(verified_slots) != p.COUNT:
        missing = sorted(set(range(p.COUNT)) - verified_slots)
        missing_by_shard = {
            str(s): [i for i in missing if i % p.SHARDS == s]
            for s in sorted({i % p.SHARDS for i in missing})
        }
        status = {
            "verdict": "INVALID_OR_INCOMPLETE",
            "verified_count": len(verified_slots),
            "expected_count": p.COUNT,
            "missing_indices": missing,
            "missing_by_shard": missing_by_shard,
            "ledger_path": str(ledger_path),
        }
        publish_json(out_dir / "incomplete-status.json", status)
        raise _fail_prelook(
            f"{len(verified_slots)}/{p.COUNT} pairs verified; "
            f"missing shards {missing_by_shard} (no outcome look taken)"
        )

    return (
        closure_info,
        ruling_receipt,
        passing_canaries,
        canary_attempts,
        ordered_main_attempts,
        unit_paths,
        ledger_path,
    )


def run_final_look(
    *,
    ops_dir: Path,
    out_dir: Path,
    ruling_json: Path | None = None,
    amendment_md: Path | None = None,
    repo: Path | None = None,
    broker_root: Path | None = None,
    require_clean_git: bool = True,
) -> dict:
    """Audit all V3 ops manifests/receipts/lineages pre-outcome, seal ledger, and take 1 look."""
    adapter_repo = (repo or ADAPTER_ROOT).resolve()
    target_broker = broker_root if broker_root is not None else V3_BROKER_ROOT
    with isolated_broker_execution("final_look", broker_root=target_broker):
        try:
            with bind_broker_protocol(target_broker, require_head_at_baseline=True) as closure_info:
                resolved_broker = Path(closure_info["broker_root"])
                (
                    closure_info,
                    ruling_receipt,
                    passing_canaries,
                    canary_attempts,
                    ordered_main_attempts,
                    unit_paths,
                    ledger_path,
                ) = _audit_pre_outcome(
                    ops_dir=ops_dir,
                    out_dir=out_dir,
                    ruling_json=ruling_json,
                    amendment_md=amendment_md,
                    adapter_repo=adapter_repo,
                    resolved_broker=resolved_broker,
                    closure_info=closure_info,
                    require_clean_git=require_clean_git,
                )

                # All pre-outcome prerequisites passed across all 24 shards / 96 slots.
                # Open the 96 verified pair-XX.json payloads ONCE and call unchanged decide_v3 ONCE.
                rows: list[dict] = []
                for path in sorted(unit_paths, key=lambda item: item.name):
                    verify_immutable(path)
                    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
                    if meta.get("seed_kind") != "primary":
                        raise _fail_prelook(
                            f"non-primary unit rejected before opening outcomes: {path}"
                        )
                    rows.append(json.loads(path.read_text(encoding="utf-8")))

                decision = decide_v3(rows)
        except ValueError as exc:
            raise _fail_prelook(str(exc)) from exc

        decision["verified_receipt_count"] = len(ordered_main_attempts)
        decision["astra_ruling"] = ruling_receipt
        decision["accounting_amendment_sha256"] = ruling_receipt["amendment_sha256"]
        decision["adapter_commit"] = ruling_receipt["adapter_commit"]
        decision["adapter_inventory_sha256"] = ruling_receipt["adapter_inventory_sha256"]
        decision["baseline_closure"] = closure_info
        decision["retained_never_allocated_attempts"] = [
            att.never_allocated_proof
            for att in ordered_main_attempts
            if att.never_allocated_proof is not None
        ]
        decision["canary_receipt_count"] = len(passing_canaries)
        decision["total_canary_attempts"] = len(canary_attempts)
        decision["total_main_attempts"] = len(ordered_main_attempts)
        decision["final_ledger_sha256"] = digest(ledger_path)
        publish_json(out_dir / "efficacy-decision.json", decision)
        return decision


def main() -> None:
    ensure_cli_dash_b("tools.watch6_v3_accounting.final_look")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-dir", "--ops", dest="ops_dir", type=Path, default=DEFAULT_OPS_DIR)
    parser.add_argument("--out-dir", "--out", dest="out_dir", type=Path, required=True)
    parser.add_argument(
        "--ruling-json", "--ruling-receipt", dest="ruling_json", type=Path, default=None
    )
    parser.add_argument("--amendment", dest="amendment_md", type=Path, default=None)
    parser.add_argument(
        "--broker-root",
        "--broker",
        dest="broker_root",
        type=Path,
        default=V3_BROKER_ROOT,
    )
    args = parser.parse_args()
    decision = run_final_look(
        ops_dir=args.ops_dir,
        out_dir=args.out_dir.resolve(),
        ruling_json=args.ruling_json,
        amendment_md=args.amendment_md,
        broker_root=args.broker_root,
    )
    print(
        json.dumps(
            {
                "verdict": decision["verdict"],
                "replay_serving_decision": decision["replay_serving_decision"],
                "n": decision["n"],
                "death_delta_per_1k": decision["death_delta_per_1k"],
                "death_delta_ci95": decision["death_delta_ci95"],
                "mass_relative_difference": decision["mass_relative_difference"],
                "mass_relative_ci95_fieller": decision["mass_relative_ci95_fieller"],
                "reserve_substitutions_count": decision["reserve_substitutions_count"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
