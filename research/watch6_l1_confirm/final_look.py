"""Sole Mac final-look orchestrator for Watch6-L1-V1 confirmation.

Enforces the sole-final-look rule with a runtime pre-outcome file-open guard:
before ANY `world-XX.json` outcome payload is parsed, audits all Watch6-L1-V1 attempts
across `<ops_dir>/manifests/*.json` (including `-retry-1` accounting retries, `auto-retry`
`RETRY.json` chains, and `recover.py` chains) and verifies for every shard 0..17
(every world 0..95):
1. Ratified independent pre-data ruling receipt (`Astra` or `Fable`).
2. `stop_verified` (`cleanup_error is None`, or proven never-allocated pre-assignment refusal)
   and `manifest_sha256` binding on every runtime.
3. Passing canary receipts for every hardware tier (`A100` and `L4`) used by main shards.
4. Zero forbidden failure classes and unbroken linear retry-cap lineage (`<= 2` failed
   external-transport compute attempts per shard; `0` forks; `0` attempts after `PASS`).
5. Immutable export/sidecar/receipt-inventory SHA-256 hashes and exact 96/96 primary bijection.
Only after all 5 pre-outcome gates pass and `pre-outcome-accounting.json` +
`final-verified-ledger.json` are sealed does `final_look` lift the open guard and take the 1 look.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from research.colab_cli import canary
from research.watch6_l1_confirm import protocol as p
from research.watch6_l1_confirm.admission import (
    RETAINED_PENDING_CLEANUP_ERROR,
    is_external_transport_compute_failure,
    is_prestart_transport_failure,
)
from research.watch6_l1_confirm.analyze import DEFAULT_ALPHA, analyze_verified_union
from research.watch6_l1_confirm.ratify import validate_ruling_receipt
from research.watch6_l1_confirm.resume import WORLD_NAME_RE
from research.watch6_l1_confirm.resume import build as build_verified_ledger
from tools.ops.package_spec import digest, publish_json, verify_immutable

DEFAULT_OPS_DIR = p.ROOT.parent / "snake-dqn-artifacts/ops"
_PACKAGE_SPEC_PATH = str((p.ROOT / "tools/ops/package_spec.py").resolve())

_GUARD_STATE = threading.local()
_AUDIT_HOOK_INSTALLED = False
_AUDIT_HOOK_LOCK = threading.Lock()


class SoleFinalLookViolation(RuntimeError):
    """Raised if any code attempts to read a world-XX.json payload before pre-outcome seal."""


def _audit_world_open(event: str, args: tuple) -> None:
    if event != "open" or not getattr(_GUARD_STATE, "active", False):
        return
    file_arg = args[0] if args else None
    mode_arg = args[1] if len(args) > 1 else "r"
    if not isinstance(file_arg, (str, os.PathLike)):
        return
    name = os.path.basename(os.fspath(file_arg))
    if WORLD_NAME_RE.fullmatch(name) is None:
        return
    # Allow ONLY raw binary SHA-256 hashing inside tools/ops/package_spec.py:digest()
    # (CPython FileIO strips 'b' and emits mode='r' in sys.audit("open", ...)).
    if mode_arg in ("r", "rb"):
        frame = sys._getframe(1)
        while frame is not None:
            code = frame.f_code
            if (
                code.co_name == "digest"
                and os.path.realpath(code.co_filename) == _PACKAGE_SPEC_PATH
            ):
                return
            frame = frame.f_back
    raise SoleFinalLookViolation(
        f"SOLE_FINAL_LOOK_VIOLATION: attempted to open outcome file {name!r} (mode={mode_arg!r}) "
        "before pre-outcome accounting certificate was sealed"
    )


def _ensure_audit_hook() -> None:
    global _AUDIT_HOOK_INSTALLED
    with _AUDIT_HOOK_LOCK:
        if not _AUDIT_HOOK_INSTALLED:
            sys.addaudithook(_audit_world_open)
            _AUDIT_HOOK_INSTALLED = True


@contextlib.contextmanager
def pre_outcome_open_guard():
    """Block opening any world-XX.json payload (except raw SHA-256 digesting) while active."""
    _ensure_audit_hook()
    prev = getattr(_GUARD_STATE, "active", False)
    _GUARD_STATE.active = True
    try:
        yield
    finally:
        _GUARD_STATE.active = prev


@dataclass(frozen=True)
class _ShardAttempt:
    shard: int
    gpu: str
    ops_name: str
    attempt_dir: Path
    receipt_path: Path
    receipt_sha256: str
    manifest: dict
    receipt: dict
    classification: str  # "pass" | "prestart" | "external_transport_compute"


def _log_has_exit(log_path: Path) -> bool:
    if not log_path.is_file() or log_path.is_symlink():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return any(
        tok in text
        for tok in ("EXIT_CODE=", "ALLOCATION_UNAVAILABLE", "Refused(", "412 Precondition", "503 ")
    )


def _is_valid_live_unallocated_proof(proof_path: Path, out_dir: Path) -> bool:
    if proof_path.is_symlink() or not proof_path.is_file():
        return False
    try:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if (
        not isinstance(proof, dict)
        or proof.get("schema") != "colab-allocation-unavailable/v1"
        or proof.get("mode") != "LIVE"
        or proof.get("endpoint_ever_assigned") is not False
        or proof.get("vm_compute_started") is not False
    ):
        return False
    receipt_file = out_dir / "receipt.json"
    if not receipt_file.exists():
        return True
    if receipt_file.is_symlink():
        return False
    try:
        rec = json.loads(receipt_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        rec.get("endpoint") is None
        and proof.get("receipt", {}).get("path") == str(receipt_file)
        and proof.get("receipt", {}).get("sha256") == digest(receipt_file)
    )


def _is_proven_unallocated(ops_dir: Path, name: str, out_dir: Path) -> bool:
    log_path = ops_dir / "logs" / f"{name}.log"
    if (
        not out_dir.exists()
        and not (ops_dir / "dispatch" / name).exists()
        and _log_has_exit(log_path)
    ):
        return True
    proof_candidates = [
        ops_dir / "dispatch" / name / "unavailable.json",
        *sorted(ops_dir.glob(f"unallocated-{name}*.json")),
    ]
    return any(_is_valid_live_unallocated_proof(cand, out_dir) for cand in proof_candidates)


def _classify_attempt(attempt_dir: Path, receipt: dict) -> str:
    if receipt.get("status") == "PASS":
        terminal = receipt.get("terminal")
        if (
            receipt.get("failure_class") is not None
            or receipt.get("error") is not None
            or not isinstance(terminal, dict)
            or terminal.get("returncode") != 0
            or terminal.get("reason") is not None
            or terminal.get("error") is not None
        ):
            raise ValueError(
                f"INVALID_OR_INCOMPLETE: malformed PASS receipt/terminal in {attempt_dir}"
            )
        return "pass"
    if is_prestart_transport_failure(receipt, attempt_dir):
        return "prestart"
    if is_external_transport_compute_failure(receipt, attempt_dir):
        return "external_transport_compute"
    raise ValueError(
        f"INVALID_OR_INCOMPLETE: forbidden failure class in {attempt_dir}: "
        f"status={receipt.get('status')!r} failure_class={receipt.get('failure_class')!r} "
        f"error={receipt.get('error')!r}"
    )


def _verify_attempt_stop_and_hashes(attempt_dir: Path) -> tuple[dict, dict, Path, str]:
    if attempt_dir.is_symlink():
        raise ValueError(f"INVALID_OR_INCOMPLETE: symlinked attempt_dir refused: {attempt_dir}")
    receipt_path = attempt_dir / "receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError(
            f"INVALID_OR_INCOMPLETE: unfinished runtime (missing receipt.json): {attempt_dir}"
        )
    exec_manifest_path = attempt_dir / "manifest.json"
    if exec_manifest_path.is_symlink() or not exec_manifest_path.is_file():
        raise ValueError(f"INVALID_OR_INCOMPLETE: missing executed manifest.json: {attempt_dir}")

    resolved_receipt = receipt_path.resolve()
    receipt = json.loads(resolved_receipt.read_text(encoding="utf-8"))
    exec_manifest = json.loads(exec_manifest_path.read_text(encoding="utf-8"))
    if (
        digest(exec_manifest_path) != receipt.get("manifest_sha256")
        or receipt.get("job_id") != exec_manifest.get("job_id")
        or receipt.get("commit") != exec_manifest.get("commit")
    ):
        raise ValueError(
            f"INVALID_OR_INCOMPLETE: executed manifest SHA256/identity mismatch: {resolved_receipt}"
        )

    prealloc_unassigned = (
        receipt.get("endpoint") is None
        and receipt.get("stop_verified") is False
        and receipt.get("cleanup_error") in (None, RETAINED_PENDING_CLEANUP_ERROR)
        and is_prestart_transport_failure(receipt, attempt_dir)
    )
    if not prealloc_unassigned and (
        receipt.get("stop_verified") is not True or receipt.get("cleanup_error") is not None
    ):
        raise ValueError(f"INVALID_OR_INCOMPLETE: unstopped or unclean runtime: {resolved_receipt}")

    return exec_manifest, receipt, resolved_receipt, digest(resolved_receipt)


def _verify_shard_lineage(
    shard: int,
    attempts: list[_ShardAttempt],
    unallocated_manifests: set[Path],
) -> list[_ShardAttempt]:
    if not attempts:
        raise ValueError(
            f"INVALID_OR_INCOMPLETE: shard {shard} (slots {p.shard_indices(shard)}) "
            "has zero attempts"
        )

    by_receipt_path = {att.receipt_path: att for att in attempts}
    parent_of: dict[Path, Path | None] = {}
    recover_history_of: dict[Path, list[tuple[Path, str, str]]] = {}

    for att in attempts:
        name_dir = att.attempt_dir.name
        retry_json = att.attempt_dir.parent / "RETRY.json"
        top_comp = att.manifest.get("completion_attempt", 0)
        if type(top_comp) is not int or not 0 <= top_comp <= p.MAX_COMPUTE_COMPLETIONS_PER_SHARD:
            raise ValueError(
                f"INVALID_OR_INCOMPLETE: invalid top-level completion_attempt={top_comp!r} "
                f"on shard {shard} ({att.attempt_dir})"
            )

        if name_dir.endswith("-retry-1"):
            base_dir = Path(str(att.attempt_dir)[: -len("-retry-1")]).resolve()
            base_receipt = (base_dir / "receipt.json").resolve()
            base_att = by_receipt_path.get(base_receipt)
            if base_att is None or base_att.classification != "prestart":
                raise ValueError(
                    f"INVALID_OR_INCOMPLETE: -retry-1 attempt {att.attempt_dir} lacks verified "
                    f"pre-start base receipt {base_receipt}"
                )
            parent_of[att.receipt_path] = base_receipt
        elif retry_json.is_file():
            verify_immutable(retry_json)
            verify_immutable(att.attempt_dir.parent / "manifest.json")
            proof = json.loads(retry_json.read_text(encoding="utf-8"))
            if (
                proof.get("schema") != "ops-zero-export-retry/v1"
                or proof.get("completion_attempt") != top_comp
                or not 1 <= top_comp <= p.MAX_COMPUTE_COMPLETIONS_PER_SHARD
                or proof.get("manifest_sha256") != digest(att.attempt_dir.parent / "manifest.json")
            ):
                raise ValueError(
                    f"INVALID_OR_INCOMPLETE: malformed auto-retry RETRY.json on shard {shard}: "
                    f"{retry_json}"
                )
            unalloc_ref = proof.get("unallocated_proof")
            prior_rec_info = proof.get("prior_receipt", {})
            prior_rec_path_raw = prior_rec_info.get("path", "")
            prior_rec_path = Path(prior_rec_path_raw).resolve() if prior_rec_path_raw else None
            if prior_rec_path is not None and prior_rec_path in by_receipt_path:
                prior_att = by_receipt_path[prior_rec_path]
                if prior_att.receipt_sha256 != prior_rec_info.get("sha256"):
                    raise ValueError(
                        "INVALID_OR_INCOMPLETE: auto-retry prior_receipt SHA256 mismatch "
                        f"on shard {shard}"
                    )
                parent_of[att.receipt_path] = prior_rec_path
            elif unalloc_ref is not None:
                u_path = Path(unalloc_ref.get("path", ""))
                prior_mf = Path(proof.get("prior_manifest", {}).get("path", "")).resolve()
                if (
                    u_path.is_symlink()
                    or not u_path.is_file()
                    or digest(u_path) != unalloc_ref.get("sha256")
                    or prior_mf not in unallocated_manifests
                ):
                    raise ValueError(
                        "INVALID_OR_INCOMPLETE: auto-retry unallocated_proof mismatch "
                        f"on shard {shard}: {retry_json}"
                    )
                parent_of[att.receipt_path] = None
            else:
                raise ValueError(
                    f"INVALID_OR_INCOMPLETE: auto-retry missing prior receipt {prior_rec_path} "
                    f"in shard {shard} lineage"
                )
        else:
            if top_comp != 0:
                raise ValueError(
                    f"INVALID_OR_INCOMPLETE: completion_attempt={top_comp} without RETRY.json "
                    f"on shard {shard} ({att.attempt_dir})"
                )
            binding = att.manifest.get("watch6_l1_v1", {})
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
                raise ValueError(
                    f"INVALID_OR_INCOMPLETE: invalid retry-cap counters on shard {shard} "
                    f"({att.attempt_dir})"
                )
            if tot_d == 1:
                parent_of[att.receipt_path] = None
            else:
                history: list[tuple[Path, str, str]] = []
                for entry in priors:
                    if not isinstance(entry, dict):
                        raise ValueError(
                            f"INVALID_OR_INCOMPLETE: malformed prior_receipts on shard {shard}"
                        )
                    pr_path = Path(entry.get("receipt_path", "")).resolve()
                    pr_sha = str(entry.get("receipt_sha256", ""))
                    pr_cls = str(entry.get("classification", ""))
                    pr_att = by_receipt_path.get(pr_path)
                    if (
                        pr_att is None
                        or pr_att.receipt_sha256 != pr_sha
                        or pr_att.classification != pr_cls
                    ):
                        raise ValueError(
                            f"INVALID_OR_INCOMPLETE: missing or mismatched prior receipt {pr_path} "
                            f"in shard {shard} lineage"
                        )
                    history.append((pr_path, pr_sha, pr_cls))
                parent_of[att.receipt_path] = history[-1][0]
                recover_history_of[att.receipt_path] = history

    roots = [r_path for r_path, parent in parent_of.items() if parent is None]
    if len(roots) != 1:
        raise ValueError(
            f"INVALID_OR_INCOMPLETE: shard {shard} requires exactly 1 root attempt, "
            f"found {len(roots)}"
        )

    children_of: dict[Path, list[Path]] = {att.receipt_path: [] for att in attempts}
    for child_path, parent_path in parent_of.items():
        if parent_path is not None:
            children_of[parent_path].append(child_path)
    if any(len(kids) > 1 for kids in children_of.values()):
        raise ValueError(
            f"INVALID_OR_INCOMPLETE: forked/concurrent attempt lineage detected on shard {shard}"
        )

    chain: list[_ShardAttempt] = []
    cursor: Path | None = roots[0]
    visited: set[Path] = set()
    while cursor is not None:
        if cursor in visited:
            raise ValueError(f"INVALID_OR_INCOMPLETE: cycle in shard {shard} retry lineage")
        visited.add(cursor)
        chain.append(by_receipt_path[cursor])
        next_kids = children_of[cursor]
        cursor = next_kids[0] if next_kids else None

    if len(chain) != len(attempts):
        raise ValueError(
            f"INVALID_OR_INCOMPLETE: disconnected attempts on shard {shard} "
            f"({len(chain)}/{len(attempts)} in linear chain)"
        )

    for idx, att in enumerate(chain):
        if att.receipt_path in recover_history_of:
            expected_hist = [
                (prev.receipt_path, prev.receipt_sha256, prev.classification)
                for prev in chain[:idx]
            ]
            if recover_history_of[att.receipt_path] != expected_hist:
                raise ValueError(
                    "INVALID_OR_INCOMPLETE: recover.py prior_receipts gap or reordering "
                    f"on shard {shard}"
                )

    for earlier in chain[:-1]:
        if earlier.classification == "pass":
            raise ValueError(
                f"INVALID_OR_INCOMPLETE: shard {shard} recorded extra attempt after PASS receipt "
                f"{earlier.receipt_path}"
            )

    if chain[-1].classification != "pass":
        raise ValueError(
            f"INVALID_OR_INCOMPLETE: shard {shard} terminal attempt did not PASS "
            f"({chain[-1].classification}: {chain[-1].receipt_path})"
        )

    failed_compute_attempts = sum(
        1 for att in chain[:-1] if att.classification == "external_transport_compute"
    )
    if failed_compute_attempts > p.MAX_COMPUTE_COMPLETIONS_PER_SHARD:
        raise ValueError(
            f"INVALID_OR_INCOMPLETE: shard {shard} exceeded compute completion retry cap "
            f"({failed_compute_attempts} > {p.MAX_COMPUTE_COMPLETIONS_PER_SHARD})"
        )
    return chain


def run_final_look(
    *,
    ops_dir: Path,
    out_dir: Path,
    ruling_json: Path | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Audit all Watch6-L1-V1 ops manifests/receipts/lineages pre-outcome, then take 1 sole look."""
    if platform.system() != "Darwin":
        raise ValueError("final look must run on Mac")

    with pre_outcome_open_guard():
        ruling_receipt = validate_ruling_receipt(ruling_json)

        ops_dir = ops_dir.resolve(strict=True)
        manifests_dir = ops_dir / "manifests"
        if not manifests_dir.is_dir() or manifests_dir.is_symlink():
            raise ValueError(f"missing or symlinked ops manifests directory: {manifests_dir}")

        canary_attempts_by_gpu: dict[str, list[tuple[Path, str]]] = {
            gpu: [] for gpu in p.TARGET_GPUS
        }
        shard_attempts: dict[int, list[_ShardAttempt]] = {s: [] for s in range(p.SHARDS)}
        unallocated_manifests: set[Path] = set()
        seen_output_dirs: set[Path] = set()

        for mf_path in sorted(manifests_dir.iterdir()):
            if mf_path.name.startswith("."):
                continue
            if mf_path.is_symlink():
                raise ValueError(f"INVALID_OR_INCOMPLETE: symlink in ops manifests: {mf_path}")
            if mf_path.suffix != ".json" or mf_path.name.endswith(".live.json"):
                continue
            try:
                data = json.loads(mf_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            binding = data.get("watch6_l1_v1")
            if not isinstance(binding, dict) or binding.get("namespace") != p.NAMESPACE:
                continue
            if binding.get("science") != p.identity():
                raise ValueError(
                    "INVALID_OR_INCOMPLETE: foreign science identity under L1-V1 namespace: "
                    f"{mf_path}"
                )
            if any(
                k in binding for k in ("reserve_allocation", "reserve_slots", "reserve_lineage")
            ):
                raise ValueError(f"INVALID_OR_INCOMPLETE: forbidden reserve fields in {mf_path}")

            name = mf_path.stem
            raw_out_unres = Path(data["output_dir"])
            if raw_out_unres.is_symlink():
                raise ValueError(f"INVALID_OR_INCOMPLETE: symlinked output_dir: {raw_out_unres}")
            raw_out = raw_out_unres.resolve()
            retry_one_unres = Path(str(data["output_dir"]) + "-retry-1")
            if retry_one_unres.is_symlink():
                raise ValueError(
                    f"INVALID_OR_INCOMPLETE: symlinked retry output_dir: {retry_one_unres}"
                )
            retry_one_out = retry_one_unres.resolve()
            separate_retry_mf = (manifests_dir / f"{name}-retry-1.json").is_file()

            candidate_dirs: list[Path] = []
            if raw_out.exists():
                candidate_dirs.append(raw_out)
            if not separate_retry_mf and retry_one_out.exists():
                candidate_dirs.append(retry_one_out)

            if not candidate_dirs:
                if _is_proven_unallocated(ops_dir, name, raw_out):
                    unallocated_manifests.add(mf_path.resolve())
                    if "auto-retry" in data["output_dir"]:
                        orig_auto_mf = (Path(data["output_dir"]).parent / "manifest.json").resolve()
                        if orig_auto_mf.is_file():
                            unallocated_manifests.add(orig_auto_mf)
                    continue
                raise ValueError(
                    "INVALID_OR_INCOMPLETE: unfinished or unverified Watch6-L1-V1 ops manifest "
                    f"{mf_path} ({raw_out})"
                )

            for attempt_dir in candidate_dirs:
                if attempt_dir in seen_output_dirs:
                    continue
                has_bound_receipt = (attempt_dir / "receipt.json").is_file() and (
                    attempt_dir / "manifest.json"
                ).is_file()
                if not has_bound_receipt and _is_proven_unallocated(ops_dir, name, attempt_dir):
                    unallocated_manifests.add(mf_path.resolve())
                    continue
                seen_output_dirs.add(attempt_dir)

                exec_manifest, receipt, receipt_path, receipt_sha = _verify_attempt_stop_and_hashes(
                    attempt_dir
                )
                exec_binding = exec_manifest.get("watch6_l1_v1", {})
                if (
                    exec_binding.get("namespace") != p.NAMESPACE
                    or exec_binding.get("science") != p.identity()
                    or exec_binding.get("primary_allocation") != p.allocation()
                    or any(
                        k in exec_binding
                        for k in ("reserve_allocation", "reserve_slots", "reserve_lineage")
                    )
                ):
                    raise ValueError(
                        "INVALID_OR_INCOMPLETE: executed manifest allocation/science mismatch: "
                        f"{attempt_dir}"
                    )

                gpu = str(exec_manifest.get("requirements", {}).get("gpu", ""))
                if gpu not in p.TARGET_GPUS:
                    raise ValueError(
                        f"INVALID_OR_INCOMPLETE: unsupported GPU {gpu!r} in {attempt_dir}"
                    )

                classification = _classify_attempt(attempt_dir, receipt)
                if (
                    canary.is_canary(exec_manifest)
                    or exec_manifest.get("env", {}).get("WATCH6_FIXTURE") == "1"
                ):
                    canary_attempts_by_gpu[gpu].append((receipt_path, classification))
                else:
                    env = exec_manifest.get("env", {})
                    if int(env.get("WATCH6_SHARDS", -1)) != p.SHARDS:
                        raise ValueError(
                            f"INVALID_OR_INCOMPLETE: wrong WATCH6_SHARDS in {attempt_dir}"
                        )
                    shard = int(env.get("WATCH6_SHARD", -1))
                    if shard not in range(p.SHARDS):
                        raise ValueError(
                            f"INVALID_OR_INCOMPLETE: invalid shard {shard} in {attempt_dir}"
                        )
                    spec = p.shard_spec(shard)
                    if gpu != spec["gpu"] or int(env.get("WATCH6_WORKERS", -1)) != spec["workers"]:
                        raise ValueError(
                            f"INVALID_OR_INCOMPLETE: shard {shard} GPU/worker mismatch "
                            f"in {attempt_dir}"
                        )
                    shard_attempts[shard].append(
                        _ShardAttempt(
                            shard=shard,
                            gpu=gpu,
                            ops_name=name,
                            attempt_dir=attempt_dir,
                            receipt_path=receipt_path,
                            receipt_sha256=receipt_sha,
                            manifest=exec_manifest,
                            receipt=receipt,
                            classification=classification,
                        )
                    )

        ordered_main_attempts: list[_ShardAttempt] = []
        for shard in range(p.SHARDS):
            chain = _verify_shard_lineage(shard, shard_attempts[shard], unallocated_manifests)
            ordered_main_attempts.extend(chain)

        used_gpus = sorted({att.gpu for att in ordered_main_attempts})
        passing_canaries_by_gpu: dict[str, list[str]] = {}
        for gpu in used_gpus:
            passing = [str(r_path) for r_path, cls in canary_attempts_by_gpu[gpu] if cls == "pass"]
            if not passing:
                raise ValueError(
                    f"INVALID_OR_INCOMPLETE: missing passing Watch6-L1-V1 {gpu} canary receipt"
                )
            passing_canaries_by_gpu[gpu] = passing

        main_receipts = [att.receipt_path for att in ordered_main_attempts]
        unit_paths: list[Path] = []
        seen_slots: dict[int, Path] = {}

        for att in ordered_main_attempts:
            declared_worlds: set[Path] = set()
            for row in att.receipt.get("exports", []):
                rel = str(row.get("path", ""))
                if not rel.startswith("efficacy/"):
                    continue
                fname = Path(rel).name
                match = WORLD_NAME_RE.fullmatch(fname)
                if match is None:
                    continue
                slot = int(match.group(1))
                candidate = att.attempt_dir / "exports" / rel
                if candidate.is_symlink():
                    raise ValueError(f"INVALID_OR_INCOMPLETE: symlinked world export: {candidate}")
                resolved_cand = candidate.resolve(strict=True)
                declared_worlds.add(resolved_cand)
                if slot in seen_slots:
                    raise ValueError(
                        f"INVALID_OR_INCOMPLETE: duplicate exported slot {slot}: "
                        f"{seen_slots[slot]} vs {resolved_cand}"
                    )
                seen_slots[slot] = resolved_cand
                unit_paths.append(resolved_cand)

            efficacy_dir = att.attempt_dir / "exports/efficacy"
            if efficacy_dir.is_dir():
                for disk_file in sorted(efficacy_dir.iterdir()):
                    if WORLD_NAME_RE.fullmatch(disk_file.name) is not None:
                        if disk_file.is_symlink() or disk_file.resolve() not in declared_worlds:
                            raise ValueError(
                                "INVALID_OR_INCOMPLETE: unrecorded world file on disk outside "
                                f"receipt exports: {disk_file}"
                            )

        out_dir.mkdir(parents=True, exist_ok=False)
        ledger = build_verified_ledger(unit_paths, main_receipts)
        ledger_path = out_dir / "final-verified-ledger.json"
        publish_json(ledger_path, ledger)

        verified_slots = {row["index"] for row in ledger["units"]}
        if len(verified_slots) != p.COUNT:
            missing = sorted(set(range(p.COUNT)) - verified_slots)
            missing_by_shard: dict[str, list[int]] = {}
            for idx in missing:
                sh = str(p.owner_shard_for_index(idx))
                missing_by_shard.setdefault(sh, []).append(idx)
            status = {
                "verdict": "INVALID_OR_INCOMPLETE",
                "verified_count": len(verified_slots),
                "expected_count": p.COUNT,
                "missing_indices": missing,
                "missing_by_shard": missing_by_shard,
                "ledger_path": str(ledger_path),
            }
            publish_json(out_dir / "incomplete-status.json", status)
            raise ValueError(
                f"INVALID_OR_INCOMPLETE: {len(verified_slots)}/{p.COUNT} worlds verified; "
                f"missing shards {missing_by_shard} (no outcome look taken)"
            )

        accounting_cert = {
            "schema": "watch6-l1-pre-outcome-accounting/v1",
            "status": "PRE_OUTCOME_ACCOUNTING_PASSED",
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "verified_worlds": len(verified_slots),
            "shards_verified": p.SHARDS,
            "passing_canaries_by_gpu": passing_canaries_by_gpu,
            "total_main_attempts": len(ordered_main_attempts),
            "final_ledger_sha256": digest(ledger_path),
            "outcome_files_opened_during_audit": 0,
        }
        accounting_path = out_dir / "pre-outcome-accounting.json"
        publish_json(accounting_path, accounting_cert)

    # Pre-outcome guard lifted ONLY after all 18 shards / 96 worlds passed and certificate sealed.
    decision = analyze_verified_union(unit_paths, main_receipts, alpha=alpha)
    decision["pre_data_ruling"] = ruling_receipt
    decision["pre_outcome_accounting_sha256"] = digest(accounting_path)
    decision["passing_canaries_by_gpu"] = passing_canaries_by_gpu
    decision["total_main_attempts"] = len(ordered_main_attempts)
    decision["final_ledger_sha256"] = digest(ledger_path)
    publish_json(out_dir / "l1-confirm-decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-dir", "--ops", dest="ops_dir", type=Path, default=DEFAULT_OPS_DIR)
    parser.add_argument("--out-dir", "--out", dest="out_dir", type=Path, required=True)
    parser.add_argument(
        "--ruling-json", "--ruling-receipt", dest="ruling_json", type=Path, default=None
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()
    decision = run_final_look(
        ops_dir=args.ops_dir,
        out_dir=args.out_dir.resolve(),
        ruling_json=args.ruling_json,
        alpha=args.alpha,
    )
    print(
        json.dumps(
            {
                "verdict": decision["verdict"],
                "replay_serving_decision": decision["replay_serving_decision"],
                "alpha": decision["alpha"],
                "n": decision["n"],
                "gate1_outcome": decision["gate1_l1_vs_bare"]["outcome"],
                "gate2_outcome": decision["gate2_l1_vs_l2_prime"]["outcome"],
                "reserve_substitutions_count": decision["reserve_substitutions_count"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
