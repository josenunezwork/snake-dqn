"""Finalize Watch6-L1-V1 pending main-shard manifests after passing A100/L4 canary receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.colab_cli import canary
from research.watch6_l1_confirm import protocol as p
from research.watch6_l1_confirm.admission import admit
from tools.ops.package_spec import digest, publish_json, verify_immutable


def _validate_canary_receipt(receipt_path: Path, expected_gpu: str) -> Path:
    receipt_path = receipt_path.resolve(strict=True)
    if receipt_path.is_symlink():
        raise ValueError(f"refusing symlinked canary receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_path = receipt_path.parent / "manifest.json"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or digest(manifest_path) != receipt.get("manifest_sha256")
    ):
        raise ValueError(f"canary receipt manifest_sha256 mismatch: {receipt_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "PASS"
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
        or manifest.get("requirements", {}).get("gpu") != expected_gpu
        or manifest.get("watch6_l1_v1", {}).get("namespace") != p.NAMESPACE
        or manifest.get("watch6_l1_v1", {}).get("science") != p.identity()
    ):
        raise ValueError(f"canary receipt for {expected_gpu} did not PASS cleanly: {receipt_path}")

    perf_path = receipt_path.parent / "exports/performance.json"
    verify_immutable(perf_path)
    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    counters = perf.get("native_v8", {})
    if (
        perf.get("schema") != "watch6-l1-canary-performance/v1"
        or perf.get("science") != p.identity()
        or perf.get("fixture") is not True
        or p.SEARCH_HORIZON not in perf.get("frames_executed", [])
        or counters.get("kernel_rows", 0) <= 0
        or any(
            counters.get(k, -1) != 0 for k in ("stock_rows", "fallback_rows", "noncandidate_rows")
        )
    ):
        raise ValueError(f"canary performance.json failed H300/native-v8 attestation: {perf_path}")
    return receipt_path


def finalize(
    packet_dir: Path,
    *,
    canary_a100_receipt: Path | None = None,
    canary_l4_receipt: Path | None = None,
    gpus: set[str] | None = None,
) -> dict:
    """Bind passing A100/L4 canary receipts into shard-XX.pending.json and emit shard-XX.json."""
    packet_dir = packet_dir.resolve(strict=True)
    target_gpus = set(gpus) if gpus is not None else set(p.TARGET_GPUS)
    if not target_gpus.issubset(set(p.TARGET_GPUS)):
        raise ValueError(f"unsupported GPUs {target_gpus}; expected subset of {p.TARGET_GPUS}")

    resolved_canaries: dict[str, Path] = {}
    for gpu, explicit in (("A100", canary_a100_receipt), ("L4", canary_l4_receipt)):
        if gpu not in target_gpus:
            continue
        cand = (
            explicit
            if explicit is not None
            else (packet_dir / f"output-canary-{gpu.lower()}/receipt.json")
        )
        resolved_canaries[gpu] = _validate_canary_receipt(cand, gpu)

    finalized: list[str] = []
    for pending_path in sorted(packet_dir.glob("shard-*.pending.json")):
        manifest = json.loads(pending_path.read_text(encoding="utf-8"))
        gpu = manifest.get("requirements", {}).get("gpu")
        if gpu not in target_gpus:
            continue
        qual = resolved_canaries[gpu]
        manifest.pop("pending_qualification", None)
        manifest["canary"] = {
            "schema": canary.SCHEMA,
            "receipt": {"path": str(qual), "sha256": digest(qual)},
        }
        canary.check(manifest, p.ROOT)
        upload_report = admit(manifest, pending_qualification=False)
        stem = pending_path.name[: -len(".pending.json")]
        out_manifest = packet_dir / f"{stem}.json"
        if out_manifest.exists():
            raise FileExistsError(f"refusing to overwrite finalized manifest: {out_manifest}")
        publish_json(packet_dir / f"{stem}-finalized-preflight.json", upload_report)
        out_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        finalized.append(out_manifest.name)

    return {
        "status": "FINALIZED",
        "packet_dir": str(packet_dir),
        "canaries_verified": {gpu: str(path) for gpu, path in resolved_canaries.items()},
        "finalized_manifests": finalized,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--canary-a100-receipt", type=Path, default=None)
    parser.add_argument("--canary-l4-receipt", type=Path, default=None)
    parser.add_argument("--gpus", nargs="*", choices=p.TARGET_GPUS, default=None)
    args = parser.parse_args()
    report = finalize(
        args.packet,
        canary_a100_receipt=args.canary_a100_receipt,
        canary_l4_receipt=args.canary_l4_receipt,
        gpus=set(args.gpus) if args.gpus else None,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
