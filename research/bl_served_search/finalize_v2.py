"""Open the one scientific look only from eight verified terminal assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.bl_served_search import analyze
from research.bl_served_search.chunks import CHUNK_COUNT
from research.bl_served_search.exports import digest, export_json, verify_export
from research.bl_served_search.provenance_v2 import FROZEN_BANK_SHA256
from research.bl_upload.qualification import _canonical, _sha256_bytes


def verified_chunk(plan_path: Path, index: int, receipt: dict) -> Path:
    """Follow prepared plan -> manifest -> terminal -> exported shard hashes."""
    verify_export(plan_path)
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "bl-served-search-wp1-plan/v2" or not 0 <= index < CHUNK_COUNT:
        raise ValueError("wrong root plan/chunk schema")
    directory = plan_path.parent / f"chunk-{index+1:02d}"
    manifest_path = directory / "manifest.json"
    expected = plan["manifests"][index + 1]
    verify_export(manifest_path)
    if digest(manifest_path) != expected["sha256"]:
        raise ValueError("manifest changed")
    manifest = json.loads(manifest_path.read_text())
    chunk_plan_path = directory / "plan.json"
    verify_export(chunk_plan_path)
    chunk_plan = json.loads(chunk_plan_path.read_text())
    bound = {row["dest"]: row["sha256"] for row in manifest["inputs"]}
    if bound.get("l2-plan.json") != digest(chunk_plan_path):
        raise ValueError("chunk plan differs from manifest input")
    bank_path = plan_path.parent / "evidence/bank.json"
    verify_export(bank_path)
    if (
        digest(bank_path) != FROZEN_BANK_SHA256
        or digest(bank_path) != plan["bank_sha256"]
        or chunk_plan["input_hashes"].get("bank.json") != digest(bank_path)
    ):
        raise ValueError("bank differs from pinned root/chunk plan")
    if (
        chunk_plan["analyzer_sha256"] != digest(Path(analyze.__file__))
        or manifest["commit"] != plan["commit"]
    ):
        raise ValueError("pinned analyzer/commit changed")
    transport_path = Path(expected["transport_receipt"])
    verify_export(transport_path)
    transport = json.loads(transport_path.read_text())
    if (
        digest(transport_path) != expected["transport_sha256"]
        or transport.get("manifest_sha256") != _sha256_bytes(_canonical(manifest))
        or transport.get("all_bytes_verified") is not True
    ):
        raise ValueError("transport proof does not bind manifest")
    if (
        receipt.get("schema") != "bl-served-search-terminal-chunk/v2"
        or receipt.get("chunk_index") != index
    ):
        raise ValueError("terminal chunk wrapper required")
    supervisor_path = Path(receipt["supervisor_receipt"])
    output = Path(manifest["output_dir"])
    if supervisor_path.resolve() != (output / "receipt.json").resolve():
        raise ValueError("foreign assignment output")
    if digest(supervisor_path) != receipt["supervisor_sha256"]:
        raise ValueError("supervisor receipt changed")
    supervisor = json.loads(supervisor_path.read_text())
    actual_manifest = output / "manifest.json"
    if (
        supervisor.get("manifest_sha256") != digest(actual_manifest)
        or json.loads(actual_manifest.read_text()) != manifest
    ):
        raise ValueError("supervisor manifest differs from prepared manifest")
    terminal = supervisor.get("terminal") or {}
    if (
        supervisor.get("status") != "PASS"
        or supervisor.get("stop_verified") is not True
        or supervisor.get("error")
        or supervisor.get("cleanup_error")
        or supervisor.get("commit") != plan["commit"]
        or supervisor.get("job_id") != manifest["job_id"]
        or not supervisor.get("endpoint")
        or not supervisor.get("generation")
        or terminal.get("generation") != supervisor["generation"]
        or terminal.get("returncode") != 0
        or terminal.get("status") not in {"completed", "success", "succeeded", "PASS"}
    ):
        raise ValueError("assignment not terminal/valid/released")
    shard = Path(receipt["shard"])
    relative = f"l2-shard-{index}.json"
    if shard.resolve() != (output / "exports" / relative).resolve():
        raise ValueError("foreign assignment shard")
    verify_export(shard)
    if digest(shard) != receipt["shard_sha256"]:
        raise ValueError("shard changed")
    exports = [row for row in supervisor.get("exports", []) if row.get("path") == relative]
    if (
        len(exports) != 1
        or exports[0].get("sha256") != digest(shard)
        or exports[0].get("size") != shard.stat().st_size
    ):
        raise ValueError("shard not bound to terminal supervisor exports")
    return shard


def seal(plan_path: Path, index: int, supervisor_path: Path, shard: Path) -> dict:
    """Create the required terminal wrapper only after verified release/export."""
    receipt = {
        "schema": "bl-served-search-terminal-chunk/v2",
        "chunk_index": index,
        "supervisor_receipt": str(supervisor_path.resolve()),
        "supervisor_sha256": digest(supervisor_path),
        "shard": str(shard.resolve()),
        "shard_sha256": digest(shard),
    }
    verified_chunk(plan_path, index, receipt)
    export_json(plan_path.parent / f"chunk-{index+1:02d}/receipt.json", receipt)
    return receipt


def finalize(plan_path: Path, out: Path) -> dict:
    """Reject faults without partial metrics; the scientific output is create-only."""
    if any(Path(str(out) + suffix).exists() for suffix in ("", ".sha256", ".complete")):
        raise FileExistsError("one-look output already exists")
    verify_export(plan_path)
    paths, errors = [], []
    for index in range(CHUNK_COUNT):
        try:
            receipt_path = plan_path.parent / f"chunk-{index+1:02d}/receipt.json"
            verify_export(receipt_path)
            paths.append(verified_chunk(plan_path, index, json.loads(receipt_path.read_text())))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"chunk-{index+1:02d}: {exc}")
    look_path = plan_path.parent / "look-opened.json"
    if look_path.exists():
        raise FileExistsError("registered one-look already opened")
    if not errors:
        rows, errors = analyze.load_rows(plan_path.parent / "evidence/bank.json", paths)
    if errors:
        result = {
            "schema": "bl-served-search-l2-analysis/v2",
            "status": "INCONCLUSIVE",
            "decision": {"label": "INCONCLUSIVE", "reason": "terminal chunk set invalid"},
            "errors": errors,
        }
    else:
        export_json(
            look_path,
            {
                "plan_sha256": digest(plan_path),
                "analyzer_sha256": digest(Path(analyze.__file__)),
                "shards": {str(path): digest(path) for path in paths},
            },
        )
        result = {"schema": "bl-served-search-l2-analysis/v2", **analyze.analyze(rows)}
    export_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seal-index", type=int)
    parser.add_argument("--supervisor-receipt", type=Path)
    parser.add_argument("--shard", type=Path)
    args = parser.parse_args()
    if args.seal_index is not None:
        if args.supervisor_receipt is None or args.shard is None:
            parser.error("sealing requires --supervisor-receipt and --shard")
        print(json.dumps(seal(args.plan, args.seal_index, args.supervisor_receipt, args.shard)))
    else:
        if args.out is None:
            parser.error("finalize requires create-only --out")
        print(json.dumps(finalize(args.plan, args.out)))


if __name__ == "__main__":
    main()
