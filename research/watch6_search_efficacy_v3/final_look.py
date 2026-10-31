"""Single Mac final-look orchestrator for Watch6 V3 confirmation.

Before opening ANY `pair-XX.json` outcome payload, audits every V3 attempt across
`<ops_dir>/manifests/*.json` (including `-retry-1` accounting retries, `auto-retry`
`RETRY.json` chains, and `recover.py` chains) and verifies for every shard 0..23
(every slot 0..95):
1. `stop_verified` (`cleanup_error is None`) and `manifest_sha256` binding on every runtime.
2. Zero forbidden failure classes on any attempt (only `PASS`, positively proven pre-start
   failure, or traceback-verified external-transport compute failure).
3. Complete, unbroken linear retry-cap lineage rooted at the initial packet with at most
   `MAX_COMPUTE_COMPLETIONS_PER_SHARD = 2` failed compute-started attempts (`<= 3` total
   compute-started attempts) per shard.
4. Immutable export/sidecar/receipt-inventory SHA256 hashes and exact 96/96 slot coverage.
Only after all 4 pre-outcome checks pass for every slot does `final_look` open outcomes.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path

from research.colab_cli import canary
from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.admission import (
    is_external_transport_compute_failure,
    is_prestart_transport_failure,
)
from research.watch6_search_efficacy_v3.analyze import analyze_verified_union
from research.watch6_search_efficacy_v3.dispatch import DEFAULT_OPS_DIR, _log_has_exit
from research.watch6_search_efficacy_v3.ratify import validate_ruling_receipt
from research.watch6_search_efficacy_v3.resume import PAIR_NAME_RE
from research.watch6_search_efficacy_v3.resume import build as build_verified_ledger
from tools.ops.package_spec import digest, publish_json, verify_immutable


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


def _is_valid_live_unallocated_proof(proof_path: Path, out_dir: Path) -> bool:
    if not proof_path.is_file():
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
    """True iff an ops manifest exited pre-output_dir or has a LIVE unallocated proof."""
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
    """Classify a stopped V3 attempt outcome-blind; raise ValueError on any forbidden failure."""
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
    """Verify receipt.json stop/cleanup state and executed manifest.json SHA256 binding."""
    receipt_path = (attempt_dir / "receipt.json").resolve()
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError(
            f"INVALID_OR_INCOMPLETE: unfinished V3 runtime (missing receipt.json): {attempt_dir}"
        )
    exec_manifest_path = attempt_dir / "manifest.json"
    if not exec_manifest_path.is_file() or exec_manifest_path.is_symlink():
        raise ValueError(f"INVALID_OR_INCOMPLETE: missing executed manifest.json: {attempt_dir}")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    exec_manifest = json.loads(exec_manifest_path.read_text(encoding="utf-8"))
    if (
        digest(exec_manifest_path) != receipt.get("manifest_sha256")
        or receipt.get("job_id") != exec_manifest.get("job_id")
        or receipt.get("commit") != exec_manifest.get("commit")
    ):
        raise ValueError(
            f"INVALID_OR_INCOMPLETE: executed manifest SHA256/identity mismatch: {receipt_path}"
        )

    prealloc_unassigned = (
        receipt.get("endpoint") is None
        and receipt.get("stop_verified") is False
        and receipt.get("cleanup_error") is None
        and is_prestart_transport_failure(receipt, attempt_dir)
    )
    if not prealloc_unassigned and (
        receipt.get("stop_verified") is not True or receipt.get("cleanup_error") is not None
    ):
        raise ValueError(f"INVALID_OR_INCOMPLETE: unstopped or unclean V3 runtime: {receipt_path}")

    return exec_manifest, receipt, receipt_path, digest(receipt_path)


def _verify_shard_lineage(
    shard: int,
    attempts: list[_ShardAttempt],
    unallocated_manifests: set[Path],
    ops_dir: Path,
) -> list[_ShardAttempt]:
    """Verify complete, unbroken linear retry-cap lineage and <=2 compute completions for shard."""
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
                    not u_path.is_file()
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
) -> dict:
    """Audit all V3 ops manifests/receipts/lineages pre-outcome, seal ledger, and take 1 look."""
    if platform.system() != "Darwin":
        raise ValueError("final look must run on Mac")
    ruling_receipt = validate_ruling_receipt(ruling_json)

    ops_dir = ops_dir.resolve(strict=True)
    manifests_dir = ops_dir / "manifests"
    if not manifests_dir.is_dir():
        raise ValueError(f"missing ops manifests directory: {manifests_dir}")

    canary_attempts: list[tuple[Path, str]] = []
    shard_attempts: dict[int, list[_ShardAttempt]] = {s: [] for s in range(p.SHARDS)}
    unallocated_manifests: set[Path] = set()
    seen_output_dirs: set[Path] = set()

    for mf_path in sorted(manifests_dir.glob("*.json")):
        if mf_path.name.endswith(".live.json"):
            continue
        data = json.loads(mf_path.read_text(encoding="utf-8"))
        binding = data.get("watch6_v3")
        if not isinstance(binding, dict) or binding.get("namespace") != p.NAMESPACE:
            continue
        if binding.get("science") != p.identity():
            raise ValueError(f"foreign science identity under V3 namespace in {mf_path}")
        if any(k in binding for k in ("reserve_allocation", "reserve_slots", "reserve_lineage")):
            raise ValueError(f"forbidden reserve fields in V3 manifest {mf_path}")

        name = mf_path.stem
        raw_out = Path(data["output_dir"]).resolve()
        retry_one_out = Path(str(data["output_dir"]) + "-retry-1").resolve()
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
                "INVALID_OR_INCOMPLETE: unfinished or unverified V3 ops manifest "
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
            exec_binding = exec_manifest.get("watch6_v3", {})
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

            classification = _classify_attempt(attempt_dir, receipt)
            if (
                canary.is_canary(exec_manifest)
                or exec_manifest.get("env", {}).get("WATCH6_FIXTURE") == "1"
            ):
                canary_attempts.append((receipt_path, classification))
            else:
                env = exec_manifest.get("env", {})
                if int(env.get("WATCH6_SHARDS", -1)) != p.SHARDS:
                    raise ValueError(f"INVALID_OR_INCOMPLETE: wrong WATCH6_SHARDS in {attempt_dir}")
                shard = int(env.get("WATCH6_SHARD", -1))
                if shard not in range(p.SHARDS):
                    raise ValueError(
                        f"INVALID_OR_INCOMPLETE: invalid shard {shard} in {attempt_dir}"
                    )
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
                    )
                )

    passing_canaries = [r_path for r_path, cls in canary_attempts if cls == "pass"]
    if not passing_canaries:
        raise ValueError(
            "INVALID_OR_INCOMPLETE: missing passing V3 L4 canary receipt in ops manifests"
        )

    ordered_main_attempts: list[_ShardAttempt] = []
    for shard in range(p.SHARDS):
        chain = _verify_shard_lineage(shard, shard_attempts[shard], unallocated_manifests, ops_dir)
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
                raise ValueError(
                    f"INVALID_OR_INCOMPLETE: duplicate exported slot {slot}: "
                    f"{seen_slots[slot]} vs {candidate}"
                )
            seen_slots[slot] = candidate
            unit_paths.append(candidate)

        efficacy_dir = att.attempt_dir / "exports/efficacy"
        if efficacy_dir.is_dir():
            for disk_file in sorted(efficacy_dir.iterdir()):
                if PAIR_NAME_RE.fullmatch(disk_file.name) is not None:
                    if disk_file.resolve() not in declared_pairs:
                        raise ValueError(
                            "INVALID_OR_INCOMPLETE: unrecorded pair file on disk outside "
                            f"receipt exports: {disk_file}"
                        )

    out_dir.mkdir(parents=True, exist_ok=False)
    ledger = build_verified_ledger(unit_paths, main_receipts)
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
        raise ValueError(
            f"INVALID_OR_INCOMPLETE: {len(verified_slots)}/{p.COUNT} pairs verified; "
            f"missing shards {missing_by_shard} (no outcome look taken)"
        )

    # All pre-outcome checks passed across all 24 shards / 96 slots; now take the single look.
    decision = analyze_verified_union(unit_paths, main_receipts)
    decision["astra_ruling"] = ruling_receipt
    decision["canary_receipt_count"] = len(passing_canaries)
    decision["total_canary_attempts"] = len(canary_attempts)
    decision["total_main_attempts"] = len(ordered_main_attempts)
    decision["final_ledger_sha256"] = digest(ledger_path)
    publish_json(out_dir / "efficacy-decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-dir", "--ops", dest="ops_dir", type=Path, default=DEFAULT_OPS_DIR)
    parser.add_argument("--out-dir", "--out", dest="out_dir", type=Path, required=True)
    parser.add_argument(
        "--ruling-json", "--ruling-receipt", dest="ruling_json", type=Path, default=None
    )
    args = parser.parse_args()
    decision = run_final_look(
        ops_dir=args.ops_dir,
        out_dir=args.out_dir.resolve(),
        ruling_json=args.ruling_json,
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
