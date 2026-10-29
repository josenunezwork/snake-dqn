"""Immutable stage-unit exports, verified ledger, and hash-bound compact XZ resume for C/T.

Implements Ruling §4 & §5 and JUDGMENT_RESUME_POLICY.md:
- Deterministic idempotent exported stage units under ``a100-dev/t34-cf-1028/stages``:
  * ``stage-0001``: complete Stage-1 collection + whole-world split + support report;
  * ``stage-0002 .. stage-0009``: 8 independent counterfactual H300 scoring chunks;
  * ``stage-0010``: 3-arm distillation fit + held-out ordered gate verdict.
- Atomic completion-last publication (payload + ``.sha256`` + ``.complete``, then
  JSON header + ``.sha256`` + ``.complete``).
- Ledger-bound compact ``.pt.xz`` + ``.pt.xz.manifest`` transport support for any
  stage snapshot without a hardcoded single-snapshot hash allowlist in Python source.
"""

from __future__ import annotations

import hashlib
import lzma
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from research.sandbox_a_resident import ct_json as json
from research.sandbox_a_resident.training_compact import (
    CODEC,
    XZ_PRESET,
    CompactResumeRefused,
    decode_stream,
    describe_state,
    verify_decoded_state,
)
from research.sandbox_a_resident.units import digest, publish, verified_descriptor

NAMESPACE = "a100-dev/t34-cf-1028/stages"
LEDGER_SCHEMA = "t34-cf-stage-ledger/v1"
SNAPSHOT_SCHEMA = "t34-cf-stage-snapshot/v1"
COMPACT_RESUME_FIELD = "compact_resume"
COMPACT_RESUME_SCHEMA = "t34-cf-compact-resume/v1"
COMPACT_MANIFEST_SCHEMA = "t34-cf-compact-snapshot/v1"
COMPACT_DERIVATIVE_SUFFIX = ".xz"
COMPACT_MANIFEST_SUFFIX = ".xz.manifest"
_READ_BYTES = 1024 * 1024
# units.publish() writes "<64-hex>\n" (65 B) + "complete\n" (9 B) = 74 B of sidecars per artifact.
PUBLISH_SIDECAR_BYTES = 65 + 9
# ct_job.sh caps two watchdog files at 1 MiB each and history at 4 MiB;
# 8 KiB covers all three sets of sidecars and final progress.json replacement.
STALL_DIAGNOSTIC_SEAL_RESERVE_BYTES = 6 * (1024 * 1024) + 8192
# Reserved tail inside max_total_export_bytes (384 MiB) for summary/sigusr1/fault + stall sealing.
TAIL_SUMMARY_AND_DIAGNOSTIC_RESERVE_BYTES = 2_228_224 + 4 * (1024 * 1024)


class StageExportOverflowError(RuntimeError):
    """Raised when a serialized stage snapshot or cumulative attempt exports exceed frozen caps."""


def allocation(control: Mapping[str, Any]) -> str:
    """Return frozen control digest independently of attempt path or job_id."""
    return hashlib.sha256(
        json.dumps(dict(control), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def unit_id(stage: int) -> str:
    """Return canonical unit identity for 1-indexed C/T stage ``stage``."""
    if type(stage) is not int or stage < 1:
        raise ValueError("C/T stage must be a positive integer >= 1")
    return f"{NAMESPACE}/stage-{stage:04d}"


def empty_ledger(control: Mapping[str, Any]) -> dict[str, Any]:
    """Return first-attempt empty verified ledger."""
    return {
        "schema": LEDGER_SCHEMA,
        "namespace": NAMESPACE,
        "allocation": allocation(control),
        "units": [],
    }


def _valid_descriptor_shape(desc: Any) -> bool:
    return (
        isinstance(desc, dict)
        and isinstance(desc.get("path"), str)
        and bool(desc["path"])
        and type(desc.get("size")) is int
        and desc["size"] >= 1
        and isinstance(desc.get("sha256"), str)
        and len(desc["sha256"]) == 64
        and all(ch in "0123456789abcdef" for ch in desc["sha256"])
    )


def validate_compact_resume(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate ledger-bound ``compact_resume`` entries against ledger units (no hardcoded hashes).

    Returns ``{unit_id: entry_dict}`` mapping each compact-transported unit to its
    bound ``original``, ``derivative``, and ``manifest`` descriptors.
    """
    json.require_finite(payload)
    if COMPACT_RESUME_FIELD not in payload:
        return {}
    block = payload[COMPACT_RESUME_FIELD]
    if (
        not isinstance(block, dict)
        or block.get("schema") != COMPACT_RESUME_SCHEMA
        or not isinstance(block.get("entries"), list)
        or not block["entries"]
    ):
        raise ValueError("malformed compact_resume block")

    units_list = payload.get("units")
    if not isinstance(units_list, list):
        raise ValueError("ledger missing units list")
    units_by_id = {
        row.get("id"): row
        for row in units_list
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }

    out: dict[str, dict[str, Any]] = {}
    for entry in block["entries"]:
        if not isinstance(entry, dict) or set(entry) != {
            "unit",
            "original",
            "derivative",
            "manifest",
        }:
            raise ValueError("malformed compact_resume entry")
        u_id = entry["unit"]
        if not isinstance(u_id, str) or u_id in out or u_id not in units_by_id:
            raise ValueError("compact_resume unit missing or duplicated in ledger")
        for role in ("original", "derivative", "manifest"):
            if not _valid_descriptor_shape(entry[role]):
                raise ValueError(f"malformed compact_resume {role} descriptor")
        unit_row = units_by_id[u_id]
        exports = unit_row.get("exports")
        if not isinstance(exports, list) or not exports:
            raise ValueError("ledger unit lacks exports for compact_resume binding")
        orig = entry["original"]
        expected_first = {
            "path": exports[0].get("path"),
            "size": exports[0].get("size"),
            "sha256": exports[0].get("sha256"),
        }
        if orig != expected_first:
            raise ValueError("compact_resume original differs from unit primary export")
        if entry["derivative"]["path"] != orig["path"] + COMPACT_DERIVATIVE_SUFFIX:
            raise ValueError("noncanonical compact derivative path")
        if entry["manifest"]["path"] != orig["path"] + COMPACT_MANIFEST_SUFFIX:
            raise ValueError("noncanonical compact manifest path")
        out[u_id] = entry
    return out


def encode_compact_stage_snapshot(original: Path, *, unit: str) -> dict[str, Any]:
    """Create-only Mac XZ encoding + per-tensor structure manifest for a verified stage ``.pt``."""
    import torch

    sidecar = Path(str(original) + ".sha256")
    complete = Path(str(original) + ".complete")
    if not (original.is_file() and sidecar.is_file() and complete.is_file()):
        raise CompactResumeRefused(f"incomplete original stage snapshot: {original.name}")
    if any(p.is_symlink() for p in (original, sidecar, complete)):
        raise CompactResumeRefused(f"symlinked stage snapshot: {original.name}")
    expected_sha = sidecar.read_text().strip().split()[0]
    if digest(original) != expected_sha:
        raise CompactResumeRefused(f"corrupt original stage snapshot: {original.name}")
    size = original.stat().st_size

    derivative = Path(str(original) + COMPACT_DERIVATIVE_SUFFIX)
    manifest_path = Path(str(original) + COMPACT_MANIFEST_SUFFIX)
    if manifest_path.exists() or Path(str(manifest_path) + ".partial").exists():
        raise FileExistsError(manifest_path)

    def _compress(src_path: Path, dst_path: Path) -> None:
        compressor = lzma.LZMACompressor(
            format=lzma.FORMAT_XZ, check=lzma.CHECK_SHA256, preset=XZ_PRESET
        )
        with src_path.open("rb") as src, dst_path.open("xb") as dst:
            for chunk in iter(lambda: src.read(16 * _READ_BYTES), b""):
                dst.write(compressor.compress(chunk))
            dst.write(compressor.flush())
            dst.flush()
            os.fsync(dst.fileno())

    encoded = publish(derivative, lambda dst: _compress(original, dst))
    decode_stream(derivative, lambda _b: None, expected_size=size, expected_sha256=expected_sha)
    state = torch.load(original, map_location="cpu", weights_only=False)
    manifest = {
        "schema": COMPACT_MANIFEST_SCHEMA,
        "unit": unit,
        "codec": CODEC,
        "original": {"name": original.name, "size": size, "sha256": expected_sha},
        "derivative": {
            "name": derivative.name,
            "size": encoded["size"],
            "sha256": encoded["sha256"],
        },
        **describe_state(state),
    }
    described = publish(
        manifest_path,
        lambda dst: dst.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n"),
    )
    return {"derivative": encoded, "manifest": described}


def _safe_runtime_path(input_root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts or not raw.parts or raw.parts[0] != "resume":
        raise CompactResumeRefused(f"unsafe compact resume runtime path: {relative!r}")
    return input_root / raw


def _verify_bound_file(path: Path, desc: Mapping[str, Any]) -> None:
    if path.is_symlink() or not path.is_file():
        raise CompactResumeRefused(f"missing compact resume file: {path.name}")
    if path.stat().st_size != desc["size"] or digest(path) != desc["sha256"]:
        raise CompactResumeRefused(f"compact resume file mismatch: {path.name}")


def materialize_compact_entries(
    ledger_payload: Mapping[str, Any], input_root: Path
) -> dict[str, dict[str, Any]]:
    """Stream-decode compact ``.pt.xz`` entries into ``input_root/resume/`` before load."""
    entries = validate_compact_resume(ledger_payload)
    manifests: dict[str, dict[str, Any]] = {}
    for u_id, entry in entries.items():
        deriv_path = _safe_runtime_path(input_root, entry["derivative"]["path"])
        man_path = _safe_runtime_path(input_root, entry["manifest"]["path"])
        orig_path = _safe_runtime_path(input_root, entry["original"]["path"])
        partial_path = orig_path.with_name(orig_path.name + ".partial")
        if any(p.exists() or p.is_symlink() for p in (orig_path, partial_path)):
            raise CompactResumeRefused(f"decoded target already exists: {orig_path.name}")
        _verify_bound_file(deriv_path, entry["derivative"])
        _verify_bound_file(man_path, entry["manifest"])
        man = json.loads(man_path.read_text())
        if (
            not isinstance(man, dict)
            or man.get("schema") != COMPACT_MANIFEST_SCHEMA
            or man.get("unit") != u_id
            or man.get("codec") != CODEC
            or man.get("original")
            != {
                "name": orig_path.name,
                "size": entry["original"]["size"],
                "sha256": entry["original"]["sha256"],
            }
            or man.get("derivative")
            != {
                "name": deriv_path.name,
                "size": entry["derivative"]["size"],
                "sha256": entry["derivative"]["sha256"],
            }
        ):
            raise CompactResumeRefused(f"compact manifest header mismatch for {u_id}")

        orig_path.parent.mkdir(parents=True, exist_ok=True)
        with partial_path.open("xb") as out_stream:
            try:
                decode_stream(
                    deriv_path,
                    out_stream.write,
                    expected_size=entry["original"]["size"],
                    expected_sha256=entry["original"]["sha256"],
                )
                out_stream.flush()
                os.fsync(out_stream.fileno())
            except BaseException:
                out_stream.close()
                if partial_path.exists():
                    partial_path.unlink()
                raise
        os.link(partial_path, orig_path)
        partial_path.unlink()
        manifests[u_id] = man
    return manifests


def load_verified_stage_payload(
    stage_row: Mapping[str, Any],
    *,
    compact_manifests: Mapping[str, Mapping[str, Any]] | None = None,
) -> Any:
    """Load one verified ``stage-XXXX.pt`` payload and verify compact tensor structure if bound."""
    import torch

    pt_exports = [e["path"] for e in stage_row["exports"] if str(e["path"]).endswith(".pt")]
    if len(pt_exports) != 1:
        raise ValueError("verified stage unit must contain exactly one .pt payload")
    state = torch.load(Path(pt_exports[0]), map_location="cpu", weights_only=False)
    u_id = stage_row["id"]
    if compact_manifests is not None and u_id in compact_manifests:
        verify_decoded_state(state, compact_manifests[u_id])
    return state


def recursive_export_bytes(root: Path, *, exclude: Path | None = None) -> int:
    """Return recursive byte sum across all regular files under ``root`` (sidecars/diagnostics)."""
    if not root.exists():
        return 0
    excluded_resolved = exclude.resolve() if (exclude is not None and exclude.exists()) else None
    total = 0
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            continue
        if not candidate.is_file():
            continue
        if excluded_resolved is not None and candidate.resolve() == excluded_resolved:
            continue
        total += candidate.stat().st_size
    return total


def publish_bounded_export(
    path: Path,
    writer: Callable[[Path], None],
    *,
    export_root: Path,
    max_total_bytes: int | None = None,
    reserved_tail_bytes: int = 0,
) -> dict[str, Any]:
    """Publish one completion-last artifact while enforcing recursive total-export byte cap."""
    if reserved_tail_bytes < 0:
        raise ValueError("reserved_tail_bytes must be >= 0")
    export_root.mkdir(parents=True, exist_ok=True)

    def _checked_writer(partial_dst: Path) -> None:
        writer(partial_dst)
        size = partial_dst.stat().st_size
        if max_total_bytes is not None:
            existing = recursive_export_bytes(export_root, exclude=partial_dst)
            artifact_bundle = size + PUBLISH_SIDECAR_BYTES
            projected_total = existing + artifact_bundle + int(reserved_tail_bytes)
            if projected_total > int(max_total_bytes):
                partial_dst.unlink(missing_ok=True)
                raise StageExportOverflowError(
                    f"recursive export total {projected_total} B (existing={existing} B, "
                    f"artifact_with_sidecars={artifact_bundle} B, "
                    f"reserved_tail={int(reserved_tail_bytes)} B) exceeds "
                    f"max_total_export_bytes={max_total_bytes}"
                )

    published = publish(path, _checked_writer)
    if max_total_bytes is not None:
        observed_total = recursive_export_bytes(export_root) + int(reserved_tail_bytes)
        if observed_total > int(max_total_bytes):
            raise StageExportOverflowError(
                f"post-publish recursive export total {observed_total} B exceeds "
                f"max_total_export_bytes={max_total_bytes}"
            )
    return published


def publish_stage_snapshot(
    output: Path,
    stage: int,
    writer: Callable[[Path], None],
    *,
    metadata: Mapping[str, Any],
    max_stage_bytes: int | None = None,
    max_total_bytes: int | None = None,
    reserved_tail_bytes: int = 0,
) -> dict[str, Any]:
    """Publish ``stage-XXXX.pt`` and completion-last ``stage-XXXX.json`` under exact recursive caps.

    - ``max_stage_bytes`` bounds the ``stage-XXXX.pt`` payload bytes.
    - ``max_total_bytes`` bounds ALL recursive regular files under ``output`` — including
      prior stages, nested ``stall-watchdog/*`` diagnostics, ``progress.json``, all six files
      published for this stage (``.pt`` + ``.sha256`` + ``.complete`` + ``.json`` + ``.sha256``
      + ``.complete``), plus ``reserved_tail_bytes`` reserved for final summary/diagnostic files.
    """
    if reserved_tail_bytes < 0:
        raise ValueError("reserved_tail_bytes must be >= 0")
    identity = unit_id(stage)
    output.mkdir(parents=True, exist_ok=True)
    pt_path = output / f"stage-{stage:04d}.pt"
    info_path = output / f"stage-{stage:04d}.json"
    meta_copy = dict(metadata)

    def _checked_writer(partial_dst: Path) -> None:
        writer(partial_dst)
        size = partial_dst.stat().st_size
        if max_stage_bytes is not None and size > int(max_stage_bytes):
            partial_dst.unlink(missing_ok=True)
            raise StageExportOverflowError(
                f"stage-{stage:04d}.pt size {size} exceeds max_stage_export_bytes={max_stage_bytes}"
            )
        if max_total_bytes is not None:
            existing_bytes = recursive_export_bytes(output, exclude=partial_dst)
            projected_descriptor = {
                "id": identity,
                "path": str(pt_path.resolve()),
                "size": size,
                "sha256": "0" * 64,
            }
            projected_envelope = {
                "schema": SNAPSHOT_SCHEMA,
                "id": identity,
                "stage": int(stage),
                "snapshot": projected_descriptor,
                "metadata": meta_copy,
            }
            envelope_size = len(
                (json.dumps(projected_envelope, sort_keys=True) + "\n").encode("utf-8")
            )
            stage_six_file_bytes = (
                size + PUBLISH_SIDECAR_BYTES + envelope_size + PUBLISH_SIDECAR_BYTES
            )
            projected_total = existing_bytes + stage_six_file_bytes + int(reserved_tail_bytes)
            if projected_total > int(max_total_bytes):
                partial_dst.unlink(missing_ok=True)
                raise StageExportOverflowError(
                    f"recursive export total {projected_total} B (existing={existing_bytes} B, "
                    f"stage_six_files={stage_six_file_bytes} B, "
                    f"reserved_tail={int(reserved_tail_bytes)} B) exceeds "
                    f"max_total_export_bytes={max_total_bytes}"
                )

    payload = publish(pt_path, _checked_writer)
    descriptor = verified_descriptor(Path(payload["path"]), identity)
    envelope = {
        "schema": SNAPSHOT_SCHEMA,
        "id": identity,
        "stage": int(stage),
        "snapshot": descriptor,
        "metadata": meta_copy,
    }
    info = publish(
        info_path, lambda path: path.write_text(json.dumps(envelope, sort_keys=True) + "\n")
    )
    if max_total_bytes is not None:
        observed_total = recursive_export_bytes(output) + int(reserved_tail_bytes)
        if observed_total > int(max_total_bytes):
            raise StageExportOverflowError(
                f"post-publish recursive export total {observed_total} B exceeds "
                f"max_total_export_bytes={max_total_bytes}"
            )
    return {"id": identity, "stage": int(stage), "exports": [descriptor, info]}


def load_ledger(
    path: Path,
    control: Mapping[str, Any],
    *,
    input_root: Path | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Materialize compact XZ inputs, validate ledger/sidecars/hashes, return verified stages."""
    payload = json.loads(path.read_text())
    expected_alloc = allocation(control)
    if (
        payload.get("schema") != LEDGER_SCHEMA
        or payload.get("namespace") != NAMESPACE
        or payload.get("allocation") != expected_alloc
        or not isinstance(payload.get("units"), list)
    ):
        raise ValueError("foreign C/T ledger namespace or allocation")

    compact_manifests: dict[str, dict[str, Any]] = {}
    if COMPACT_RESUME_FIELD in payload:
        if input_root is not None:
            compact_manifests = materialize_compact_entries(payload, input_root)
        else:
            validate_compact_resume(payload)

    rows: dict[int, dict[str, Any]] = {}
    prefix = f"{NAMESPACE}/stage-"
    for row in payload["units"]:
        if not isinstance(row, dict):
            raise ValueError("malformed verified stage unit")
        identity = row.get("id", "")
        if not isinstance(identity, str) or not identity.startswith(prefix):
            raise ValueError("foreign verified stage unit")
        try:
            stage = int(identity.removeprefix(prefix))
        except ValueError as err:
            raise ValueError("malformed stage index") from err
        if unit_id(stage) != identity or stage in rows:
            raise ValueError("duplicate or noncanonical stage unit")
        exports = row.get("exports")
        if not isinstance(exports, list) or len(exports) != 2:
            raise ValueError("stage unit requires payload and header exports")

        resolved_exports: list[dict[str, Any]] = []
        for exp in exports:
            if not _valid_descriptor_shape(exp):
                raise ValueError("malformed stage export descriptor")
            raw_p = Path(exp["path"])
            if input_root is not None:
                if raw_p.is_absolute() or ".." in raw_p.parts:
                    raise ValueError("unsafe relative stage export path")
                exp_path = (input_root / raw_p).resolve()
            else:
                exp_path = raw_p
            sidecar = Path(str(exp_path) + ".sha256")
            complete = Path(str(exp_path) + ".complete")
            require_sidecars = (
                input_root is None
                or sidecar.exists()
                or sidecar.is_symlink()
                or complete.exists()
                or complete.is_symlink()
            )
            if (
                exp_path.is_symlink()
                or not exp_path.is_file()
                or digest(exp_path) != exp["sha256"]
                or exp_path.stat().st_size != exp["size"]
                or (
                    require_sidecars
                    and (
                        sidecar.is_symlink()
                        or complete.is_symlink()
                        or not sidecar.is_file()
                        or not complete.is_file()
                        or sidecar.read_text().strip().split()[0] != exp["sha256"]
                    )
                )
            ):
                raise ValueError(f"incomplete or corrupt verified stage export: {exp_path.name}")
            resolved_exports.append({**exp, "path": str(exp_path)})

        headers = [Path(e["path"]) for e in resolved_exports if e["path"].endswith(".json")]
        if len(headers) != 1:
            raise ValueError("stage unit requires one JSON metadata header")
        header = json.loads(headers[0].read_text())
        meta = header.get("metadata", {})
        if (
            header.get("schema") != SNAPSHOT_SCHEMA
            or header.get("id") != identity
            or header.get("stage") != stage
            or meta.get("allocation") != expected_alloc
            or meta.get("source_commit") != control.get("source_commit")
        ):
            raise ValueError("stage snapshot metadata differs from frozen control")
        rows[stage] = {"id": identity, "stage": stage, "exports": resolved_exports}

    return rows, compact_manifests
