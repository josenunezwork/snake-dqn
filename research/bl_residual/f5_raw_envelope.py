"""Package-local transport for completed F5 timing units and their raw evidence.

The Colab input policy admits timing-unit roles but deliberately does not admit
arbitrary environment files.  This module is the narrow, application-level
bridge: it freezes the original broker exports inside one timing-unit envelope,
then reconstructs an isolated native output tree before asking the ordinary
resume scanner to grant skip authority.
"""

from __future__ import annotations

import base64
import hashlib
import json
import lzma
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.bl_ppo.exports import digest, export_bytes, export_json, verify_export
from research.bl_residual import f5_timing_resume as resume

SCHEMA = "bl-f5-timing-raw-envelope/v1"
MAX_ENVELOPE_BYTES = 45_397_643
MAX_DECODED_BYTES = 256 * 1024 * 1024
DECODE_MEMORY_BYTES = 128 * 1024 * 1024
_TRIPLE = ("payload", "sha256", "complete")


def _ref(value: Any, *, absolute: bool = False, allow_extra: bool = False) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or (set(value) != {"path", "size", "sha256"} and not allow_extra)
        or not {"path", "size", "sha256"} <= set(value)
        or not isinstance(value.get("path"), str)
        or not value["path"]
        or type(value.get("size")) is not int
        or value["size"] < 0
        or not isinstance(value.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
    ):
        raise resume.ResumeRefusal("raw envelope reference is malformed")
    path = Path(value["path"])
    if absolute and not path.is_absolute():
        raise resume.ResumeRefusal("raw envelope metadata reference is not absolute")
    if not absolute:
        resume._safe_relative(value["path"])
    return {name: value[name] for name in ("path", "size", "sha256")}


def _read_ref(
    reference: Mapping[str, Any], *, json_value: bool = False
) -> tuple[bytes, Any | None]:
    ref = _ref(reference, absolute=True)
    path = resume._safe_file(Path(ref["path"]))
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise resume.ResumeRefusal("raw envelope metadata is unavailable") from exc
    if len(raw) != ref["size"] or digest(path) != ref["sha256"]:
        raise resume.ResumeRefusal("raw envelope metadata hash or size differs")
    if not json_value:
        return raw, None
    try:
        return raw, json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise resume.ResumeRefusal("raw envelope metadata is not JSON") from exc


def _triple(path: Path, reference: Mapping[str, Any]) -> dict[str, str]:
    ref = _ref(reference)
    path = resume._safe_file(path)
    try:
        if verify_export(path) != ref["sha256"] or path.stat().st_size != ref["size"]:
            raise resume.ResumeRefusal("raw envelope export hash or size differs")
        parts = {"payload": path.read_bytes()}
        for label, suffix in (("sha256", ".sha256"), ("complete", ".complete")):
            parts[label] = Path(str(path) + suffix).read_bytes()
    except OSError as exc:
        raise resume.ResumeRefusal("raw envelope export is unavailable") from exc
    return {label: base64.b64encode(parts[label]).decode("ascii") for label in _TRIPLE}


def _decode_triple(value: Any, reference: Mapping[str, Any]) -> dict[str, bytes]:
    if not isinstance(value, Mapping) or set(value) != set(_TRIPLE):
        raise resume.ResumeRefusal("raw envelope triple is malformed")
    result: dict[str, bytes] = {}
    for label in _TRIPLE:
        item = value[label]
        if not isinstance(item, str):
            raise resume.ResumeRefusal("raw envelope triple encoding is malformed")
        try:
            result[label] = base64.b64decode(item.encode("ascii"), validate=True)
        except ValueError as exc:
            raise resume.ResumeRefusal("raw envelope triple is not base64") from exc
    ref = _ref(reference)
    if len(result["payload"]) != ref["size"] or digest_bytes(result["payload"]) != ref["sha256"]:
        raise resume.ResumeRefusal("raw envelope payload hash or size differs")
    expected_sidecar = f"{ref['sha256']}  {Path(ref['path']).name}\n".encode()
    if result["sha256"] != expected_sidecar or result["complete"] != b"complete\n":
        raise resume.ResumeRefusal("raw envelope sidecar bytes differ")
    return result


def digest_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _canonical_unit_path(entry: Mapping[str, Any]) -> str:
    ref = _ref(entry, allow_extra=True)
    path = ref["path"]
    key = entry["key"]
    name = "calibration" if key == resume.CALIBRATION_KEY else f"fixture-{key}"
    if path != f"timing-units/{name}.json":
        raise resume.ResumeRefusal("raw envelope unit path differs from its frozen key")
    return path


def _ledger(
    path: Path, binding: Mapping[str, str]
) -> tuple[bytes, dict[str, Any], dict[int | str, dict[str, Any]]]:
    try:
        resume._safe_file(path)
        verify_export(path)
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise resume.ResumeRefusal(
            "raw envelope ledger is not a verified immutable export"
        ) from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "unit_kind", "identity", "units"}
        or value.get("schema") != resume.SCHEMA
        or value.get("unit_kind") != resume.UNIT_KIND
        or value.get("identity") != dict(binding)
        or not isinstance(value.get("units"), list)
    ):
        raise resume.ResumeRefusal("raw envelope ledger identity differs")
    entries: dict[int | str, dict[str, Any]] = {}
    paths: set[str] = set()
    for item in value["units"]:
        if not isinstance(item, dict) or set(item) != {"key", "path", "sha256", "size"}:
            raise resume.ResumeRefusal("raw envelope ledger entry is malformed")
        key = item["key"]
        if (
            isinstance(key, bool)
            or not isinstance(key, (int, str))
            or key not in resume._keys()
            or key in entries
        ):
            raise resume.ResumeRefusal("raw envelope ledger key is foreign or duplicate")
        canonical = _canonical_unit_path(item)
        if canonical in paths:
            raise resume.ResumeRefusal("raw envelope ledger path is duplicate")
        paths.add(canonical)
        entries[key] = dict(item)
    return raw, value, entries


def _attempt(row: Mapping[str, Any], binding: Mapping[str, str]) -> tuple[Path, dict[str, Any]]:
    if not isinstance(row, Mapping) or set(row) != {
        "exports_root",
        "broker_receipt",
        "manifest",
        "control",
    }:
        raise resume.ResumeRefusal("raw envelope attempt row is malformed")
    if not isinstance(row["exports_root"], str) or not row["exports_root"]:
        raise resume.ResumeRefusal("raw envelope exports root is malformed")
    root = Path(row["exports_root"])
    if not root.is_absolute() or root.is_symlink():
        raise resume.ResumeRefusal("raw envelope exports root is unsafe")
    _, receipt = _read_ref(row["broker_receipt"], json_value=True)
    _, manifest = _read_ref(row["manifest"], json_value=True)
    _, control = _read_ref(row["control"], json_value=True)
    if not all(isinstance(item, dict) for item in (receipt, manifest, control)):
        raise resume.ResumeRefusal("raw envelope attempt metadata is malformed")
    by_path = _validate_attempt(
        receipt,
        manifest,
        control,
        manifest_ref=row["manifest"],
        control_ref=row["control"],
        binding=binding,
    )
    return root, {
        "receipt": dict(row["broker_receipt"]),
        "manifest": dict(row["manifest"]),
        "control": dict(row["control"]),
        "parsed": receipt,
        "exports": by_path,
    }


def _validate_attempt(
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    manifest_ref: Mapping[str, Any],
    control_ref: Mapping[str, Any],
    binding: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Verify original metadata and return its duplicate-free export inventory."""
    name = receipt.get("job_id")
    sealed = control.get("identity")
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(manifest.get("commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", manifest["commit"]) is None
        or manifest.get("job_id") != name
        or receipt.get("manifest_sha256") != _ref(manifest_ref, absolute=True)["sha256"]
        or receipt.get("commit") != manifest.get("commit")
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
        or receipt.get("status") not in {"PASS", "INCOMPLETE"}
        or (
            receipt.get("status") == "PASS"
            and (
                not isinstance(receipt.get("terminal"), Mapping)
                or receipt["terminal"].get("returncode") != 0
            )
        )
        or control.get("timing_attempt_id") != name
        or not isinstance(sealed, Mapping)
        or sealed.get("source_sha256") != binding["source_sha256"]
        or sealed.get("environment_sha256") != binding["environment_sha256"]
        or sealed.get("allocation_sha256") != binding["allocation"]
        or control.get("timing_input_sha256") != binding["input_sha256"]
        or not isinstance(manifest.get("inputs"), list)
        or not any(
            isinstance(item, Mapping)
            and item.get("dest") == "control.json"
            and item.get("sha256") == _ref(control_ref, absolute=True)["sha256"]
            for item in manifest.get("inputs", [])
        )
    ):
        raise resume.ResumeRefusal("raw envelope broker provenance differs")
    exports = receipt.get("exports")
    if not isinstance(exports, list):
        raise resume.ResumeRefusal("raw envelope receipt lacks exports")
    listed: dict[str, dict[str, Any]] = {}
    for item in exports:
        reference = _ref(item)
        if reference["path"] in listed:
            raise resume.ResumeRefusal("raw envelope receipt has duplicate exports")
        listed[reference["path"]] = reference
    return listed


def _validate_provenance(
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    receipt_ref: Mapping[str, Any],
    manifest_ref: Mapping[str, Any],
    control_ref: Mapping[str, Any],
    binding: Mapping[str, str],
    entry: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    """Apply the broker/export linkage identically before and after transport."""
    _ref(receipt_ref, absolute=True)
    listed = _validate_attempt(
        receipt,
        manifest,
        control,
        manifest_ref=manifest_ref,
        control_ref=control_ref,
        binding=binding,
    )
    if payload.get("timing_attempt_id") != receipt["job_id"]:
        raise resume.ResumeRefusal("raw envelope payload attempt provenance differs")
    for reference, label in (
        (entry, "timing unit"),
        (payload.get("environment_before"), "environment-before"),
        (payload.get("environment_after"), "environment-after"),
    ):
        wanted = _ref(reference, allow_extra=(label == "timing unit"))
        if listed.get(wanted["path"]) != wanted:
            raise resume.ResumeRefusal(f"raw envelope {label} is absent from broker exports")


def _transfer(path: Path, dest: str) -> list[dict[str, Any]]:
    result = []
    for suffix in ("", ".sha256", ".complete"):
        item = Path(str(path) + suffix)
        result.append(
            {
                "dest": dest + suffix,
                "path": str(item),
                "sha256": digest(item),
                "size_bytes": item.stat().st_size,
            }
        )
    return result


def _write_triple(
    root: Path, relative: str, triple: Mapping[str, Any], reference: Mapping[str, Any]
) -> Path:
    path = root / resume._safe_relative(relative)
    if path.exists() or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise resume.ResumeRefusal("raw envelope materialization path is unsafe or already exists")
    decoded = _decode_triple(triple, reference)
    export_bytes(path, decoded["payload"])
    if (
        Path(str(path) + ".sha256").read_bytes() != decoded["sha256"]
        or Path(str(path) + ".complete").read_bytes() != decoded["complete"]
    ):
        raise resume.ResumeRefusal("raw envelope materialization rewrote sidecar bytes")
    return path


def _envelope_metadata(block: Any, *, label: str) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    """Decode carried metadata without depending on its old Mac pathname."""
    if not isinstance(block, Mapping) or set(block) != {"reference", "text_utf8"}:
        raise resume.ResumeRefusal("raw envelope metadata block is malformed")
    reference = _ref(block["reference"], absolute=True)
    encoded = block.get("text_utf8")
    if not isinstance(encoded, str):
        raise resume.ResumeRefusal("raw envelope metadata encoding is malformed")
    try:
        raw = encoded.encode("utf8")
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise resume.ResumeRefusal("raw envelope metadata bytes are malformed") from exc
    if (
        len(raw) != reference["size"]
        or digest_bytes(raw) != reference["sha256"]
        or not isinstance(value, dict)
    ):
        raise resume.ResumeRefusal(f"raw envelope {label} metadata hash or shape differs")
    return reference, raw, value


def _blob_id(parts: Mapping[str, bytes]) -> str:
    wire = b"".join(len(parts[name]).to_bytes(8, "big") + parts[name] for name in _TRIPLE)
    return hashlib.sha256(wire).hexdigest()


def _triple_parts(path: Path, reference: Mapping[str, Any]) -> dict[str, bytes]:
    _triple(path, reference)  # preserves the existing verified-export checks
    return {
        "payload": path.read_bytes(),
        "sha256": Path(str(path) + ".sha256").read_bytes(),
        "complete": Path(str(path) + ".complete").read_bytes(),
    }


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)).encode()


def _decode_archive(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"codec", "decoded_sha256", "decoded_size", "data_base64"}
        or value.get("codec") != "xz-lzma2"
        or type(value.get("decoded_size")) is not int
        or not 0 <= value["decoded_size"] <= MAX_DECODED_BYTES
        or not isinstance(value.get("decoded_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", value["decoded_sha256"]) is None
        or not isinstance(value.get("data_base64"), str)
    ):
        raise resume.ResumeRefusal("raw envelope archive declaration is malformed")
    try:
        compressed = base64.b64decode(value["data_base64"].encode("ascii"), validate=True)
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=DECODE_MEMORY_BYTES)
        raw = decoder.decompress(compressed, max_length=value["decoded_size"] + 1)
    except (ValueError, lzma.LZMAError) as exc:
        raise resume.ResumeRefusal("raw envelope archive does not decompress") from exc
    if (
        len(raw) != value["decoded_size"]
        or len(raw) > MAX_DECODED_BYTES
        or hashlib.sha256(raw).hexdigest() != value["decoded_sha256"]
        or not decoder.eof
        or decoder.unused_data
    ):
        raise resume.ResumeRefusal("raw envelope archive is truncated trailing or oversized")
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise resume.ResumeRefusal("raw envelope archive JSON is malformed") from exc
    if not isinstance(decoded, dict) or set(decoded) != {
        "schema",
        "blobs",
        "payloads",
        "attempts",
        "units",
    }:
        raise resume.ResumeRefusal("raw envelope archive schema differs")
    if decoded.get("schema") != "bl-f5-timing-raw-envelope-archive/v1":
        raise resume.ResumeRefusal("raw envelope archive version differs")
    return decoded


def _archive_blob(value: Any, payloads: Mapping[str, bytes]) -> dict[str, bytes]:
    if not isinstance(value, Mapping) or set(value) != set(_TRIPLE):
        raise resume.ResumeRefusal("raw envelope archive blob is malformed")
    result: dict[str, bytes] = {}
    for name in _TRIPLE:
        if not isinstance(value[name], str):
            raise resume.ResumeRefusal("raw envelope archive blob encoding differs")
        if name == "payload":
            if value[name] not in payloads:
                raise resume.ResumeRefusal("raw envelope archive payload is absent")
            result[name] = payloads[value[name]]
        else:
            try:
                result[name] = base64.b64decode(value[name].encode("ascii"), validate=True)
            except ValueError as exc:
                raise resume.ResumeRefusal("raw envelope archive blob is not base64") from exc
    return result


def _carrier_path(entries: Mapping[int | str, Mapping[str, Any]]) -> str:
    return sorted((str(entry["path"]) for entry in entries.values()))[0]


def _require_attempt_closure(
    prepared: Sequence[tuple[Path, dict[str, Any]]],
    entries: Mapping[int | str, Mapping[str, Any]],
    binding: Mapping[str, str],
) -> dict[int | str, tuple[Path, dict[str, Any], dict[str, Any]]]:
    """Close each receipt over all timing-unit exports before selecting bytes."""
    by_path = {str(entry["path"]): (key, dict(entry)) for key, entry in entries.items()}
    selected: dict[int | str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for exports_root, attempt in prepared:
        _, manifest = _read_ref(attempt["manifest"], json_value=True)
        _, control = _read_ref(attempt["control"], json_value=True)
        for exported_path, exported in attempt["exports"].items():
            if not exported_path.startswith("timing-units/"):
                continue
            pair = by_path.get(exported_path)
            if pair is None:
                raise resume.ResumeRefusal("raw envelope receipt exported a foreign timing unit")
            key, entry = pair
            if exported != _ref(entry, allow_extra=True) or key in selected:
                raise resume.ResumeRefusal(
                    "raw envelope receipt timing-unit export is duplicate or altered"
                )
            unit_path = exports_root / resume._safe_relative(exported_path)
            read_key, payload = resume._read_unit(unit_path, binding, entry)
            if (
                type(read_key) is not type(key)
                or read_key != key
                or payload.get("canary_only") is True
            ):
                raise resume.ResumeRefusal("raw envelope refuses foreign or CANARY_ONLY unit")
            _validate_provenance(
                attempt["parsed"],
                manifest,
                control,
                receipt_ref=attempt["receipt"],
                manifest_ref=attempt["manifest"],
                control_ref=attempt["control"],
                binding=binding,
                entry=entry,
                payload=payload,
            )
            selected[key] = (exports_root, attempt, payload)
    if set(selected) != set(entries):
        raise resume.ResumeRefusal("raw envelope ledger omits a receipt-bound completed unit")
    return selected


def package_inputs(
    output: Path, ledger: Path, binding: Mapping[str, str], attempts: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Package all completed units into one compressed admitted carrier role."""
    if Path(ledger).name != "verified-ledger.json":
        raise resume.ResumeRefusal("raw envelope requires the canonical verified-ledger filename")
    raw_ledger, _, entries = _ledger(ledger, binding)
    inputs = Path(output) / "inputs"
    ledger_target = inputs / "verified-ledger.json"
    export_bytes(ledger_target, raw_ledger)
    if (
        Path(str(ledger_target) + ".sha256").read_bytes()
        != Path(str(ledger) + ".sha256").read_bytes()
        or Path(str(ledger_target) + ".complete").read_bytes()
        != Path(str(ledger) + ".complete").read_bytes()
    ):
        raise resume.ResumeRefusal("raw envelope ledger sidecar bytes changed")
    transfers = _transfer(ledger_target, "verified-ledger.json")
    if not entries:
        return transfers, []
    prepared = [_attempt(row, binding) for row in attempts]
    selected = _require_attempt_closure(prepared, entries, binding)
    blobs: dict[str, dict[str, str]] = {}
    payloads: dict[str, dict[str, str]] = {}

    def add_triple(path: Path, reference: Mapping[str, Any]) -> str:
        parts = _triple_parts(path, reference)
        identifier = _blob_id(parts)
        payload_sha = digest_bytes(parts["payload"])
        payloads.setdefault(payload_sha, {"text_utf8": parts["payload"].decode("utf8")})
        blobs.setdefault(
            identifier,
            {
                "payload": payload_sha,
                **{
                    name: base64.b64encode(parts[name]).decode("ascii")
                    for name in ("sha256", "complete")
                },
            },
        )
        return identifier

    attempt_rows: list[dict[str, Any]] = []
    attempt_index: dict[str, int] = {}
    for exports_root, attempt in prepared:
        name = attempt["parsed"]["job_id"]
        if name in attempt_index:
            raise resume.ResumeRefusal("raw envelope attempt identity is duplicate")
        refs = {name: attempt[name] for name in ("receipt", "manifest", "control")}
        data = {name: _read_ref(attempt[name])[0] for name in refs}
        attempt_index[name] = len(attempt_rows)
        attempt_rows.append(
            {
                "receipt": {
                    "reference": refs["receipt"],
                    "text_utf8": data["receipt"].decode("utf8"),
                },
                "manifest": {
                    "reference": refs["manifest"],
                    "text_utf8": data["manifest"].decode("utf8"),
                },
                "control": {
                    "reference": refs["control"],
                    "text_utf8": data["control"].decode("utf8"),
                },
            }
        )
    unit_rows: list[dict[str, Any]] = []
    for key, entry in sorted(entries.items(), key=lambda item: str(item[0])):
        exports_root, attempt, payload = selected[key]
        before, after = _ref(payload.get("environment_before")), _ref(
            payload.get("environment_after")
        )
        unit_rows.append(
            {
                "key": key,
                "unit": dict(entry),
                "attempt": attempt_index[payload["timing_attempt_id"]],
                "unit_blob": add_triple(
                    exports_root / resume._safe_relative(entry["path"]),
                    _ref(entry, allow_extra=True),
                ),
                "before": {
                    "reference": before,
                    "blob": add_triple(
                        exports_root / resume._safe_relative(before["path"]), before
                    ),
                },
                "after": {
                    "reference": after,
                    "blob": add_triple(exports_root / resume._safe_relative(after["path"]), after),
                },
            }
        )
    archive_raw = _canonical_json(
        {
            "schema": "bl-f5-timing-raw-envelope-archive/v1",
            "blobs": blobs,
            "payloads": payloads,
            "attempts": attempt_rows,
            "units": unit_rows,
        }
    )
    if len(archive_raw) > MAX_DECODED_BYTES:
        raise resume.ResumeRefusal("raw envelope decoded archive exceeds frozen bound")
    compressed = lzma.compress(
        archive_raw,
        format=lzma.FORMAT_XZ,
        filters=[{"id": lzma.FILTER_LZMA2, "dict_size": 64 * 1024 * 1024, "preset": 6}],
    )
    carrier = {
        "schema": SCHEMA,
        "identity": dict(binding),
        "units": [
            dict(entry) for _, entry in sorted(entries.items(), key=lambda item: str(item[0]))
        ],
        "broker_receipts": [row["receipt"]["reference"] for row in attempt_rows],
        "archive": {
            "codec": "xz-lzma2",
            "decoded_size": len(archive_raw),
            "decoded_sha256": hashlib.sha256(archive_raw).hexdigest(),
            "data_base64": base64.b64encode(compressed).decode("ascii"),
        },
    }
    carrier_relative = _carrier_path(entries)
    carrier_target = inputs / carrier_relative
    exported = export_json(carrier_target, carrier)
    if exported["size"] > MAX_ENVELOPE_BYTES:
        raise resume.ResumeRefusal("raw envelope exceeds frozen per-file transport bound")
    transfers.extend(_transfer(carrier_target, carrier_relative))
    inventory = [
        {
            "path": carrier_relative,
            "size": exported["size"],
            "sha256": exported["sha256"],
            "units": carrier["units"],
            "broker_receipts": carrier["broker_receipts"],
        }
    ]
    return transfers, inventory


def load_ledger(
    ledger: Path,
    binding: Mapping[str, str],
    inventory: list[Any] | None,
    materialization_root: Path,
) -> dict[int | str, dict[str, Any]]:
    """Rebuild isolated native roots from the compressed carrier and rescan them."""
    if inventory is None:
        return resume.load_ledger(ledger, binding)
    _, _, entries = _ledger(ledger, binding)
    if not entries:
        if inventory != []:
            raise resume.ResumeRefusal("empty timing ledger must not carry an envelope")
        return {}
    if not isinstance(inventory, list) or len(inventory) != 1:
        raise resume.ResumeRefusal("raw envelope inventory must contain one carrier")
    row = inventory[0]
    if not isinstance(row, Mapping) or set(row) != {
        "path",
        "size",
        "sha256",
        "units",
        "broker_receipts",
    }:
        raise resume.ResumeRefusal("raw envelope inventory row is malformed")
    reference = _ref(row, allow_extra=True)
    ordered = [dict(entry) for _, entry in sorted(entries.items(), key=lambda item: str(item[0]))]
    if (
        row.get("path") != _carrier_path(entries)
        or row.get("units") != ordered
        or not isinstance(row.get("broker_receipts"), list)
    ):
        raise resume.ResumeRefusal("raw envelope inventory does not exactly cover the ledger")
    if reference["size"] > MAX_ENVELOPE_BYTES:
        raise resume.ResumeRefusal("raw envelope carrier exceeds frozen bound")
    source = resume._safe_file(Path(ledger).parent / resume._safe_relative(reference["path"]))
    try:
        if (
            verify_export(source) != reference["sha256"]
            or source.stat().st_size != reference["size"]
        ):
            raise resume.ResumeRefusal("raw envelope carrier hash or size differs")
        carrier = json.loads(source.read_bytes())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise resume.ResumeRefusal("raw envelope carrier is invalid") from exc
    if (
        not isinstance(carrier, dict)
        or set(carrier) != {"schema", "identity", "units", "broker_receipts", "archive"}
        or carrier.get("schema") != SCHEMA
        or carrier.get("identity") != dict(binding)
        or carrier.get("units") != ordered
        or carrier.get("broker_receipts") != row["broker_receipts"]
    ):
        raise resume.ResumeRefusal("raw envelope carrier schema or inventory differs")
    archive = _decode_archive(carrier["archive"])
    if (
        not isinstance(archive["blobs"], Mapping)
        or not isinstance(archive["payloads"], Mapping)
        or not isinstance(archive["attempts"], list)
        or not isinstance(archive["units"], list)
    ):
        raise resume.ResumeRefusal("raw envelope archive collections are malformed")
    payloads: dict[str, bytes] = {}
    for identifier, encoded in archive["payloads"].items():
        if (
            not isinstance(identifier, str)
            or not isinstance(encoded, Mapping)
            or set(encoded) != {"text_utf8"}
            or not isinstance(encoded["text_utf8"], str)
        ):
            raise resume.ResumeRefusal("raw envelope payload pool is malformed")
        try:
            raw = encoded["text_utf8"].encode("utf8")
        except ValueError as exc:
            raise resume.ResumeRefusal("raw envelope payload pool is not base64") from exc
        if digest_bytes(raw) != identifier:
            raise resume.ResumeRefusal("raw envelope payload pool hash differs")
        payloads[identifier] = raw
    blobs: dict[str, dict[str, bytes]] = {}
    for identifier, value in archive["blobs"].items():
        if not isinstance(identifier, str) or re.fullmatch(r"[0-9a-f]{64}", identifier) is None:
            raise resume.ResumeRefusal("raw envelope blob identifier is malformed")
        parts = _archive_blob(value, payloads)
        if _blob_id(parts) != identifier:
            raise resume.ResumeRefusal("raw envelope blob identifier differs")
        blobs[identifier] = parts
    if Path(materialization_root).exists() or Path(materialization_root).is_symlink():
        raise resume.ResumeRefusal("raw envelope materialization root already exists or is unsafe")
    carried = []
    names: set[str] = set()
    for attempt in archive["attempts"]:
        if not isinstance(attempt, Mapping) or set(attempt) != {"receipt", "manifest", "control"}:
            raise resume.ResumeRefusal("raw envelope archive attempt is malformed")
        decoded = {
            label: _envelope_metadata(attempt[label], label=label)
            for label in ("receipt", "manifest", "control")
        }
        _validate_attempt(
            decoded["receipt"][2],
            decoded["manifest"][2],
            decoded["control"][2],
            manifest_ref=decoded["manifest"][0],
            control_ref=decoded["control"][0],
            binding=binding,
        )
        name = decoded["receipt"][2]["job_id"]
        if name in names:
            raise resume.ResumeRefusal("raw envelope carried attempt identity is duplicate")
        names.add(name)
        carried.append(decoded)
    if [item["receipt"][0] for item in carried] != carrier["broker_receipts"]:
        raise resume.ResumeRefusal("raw envelope broker receipt closure differs")
    results: dict[int | str, dict[str, Any]] = {}
    seen: set[int | str] = set()
    used_blobs: set[str] = set()
    for unit in archive["units"]:
        if not isinstance(unit, Mapping) or set(unit) != {
            "key",
            "unit",
            "attempt",
            "unit_blob",
            "before",
            "after",
        }:
            raise resume.ResumeRefusal("raw envelope archive unit is malformed")
        key = unit["key"]
        if (
            isinstance(key, bool)
            or not isinstance(key, (int, str))
            or key not in entries
            or key in seen
            or unit["unit"] != entries[key]
            or type(unit["attempt"]) is not int
            or not 0 <= unit["attempt"] < len(archive["attempts"])
        ):
            raise resume.ResumeRefusal("raw envelope archive unit identity differs")
        seen.add(key)
        root = Path(materialization_root) / f"unit-{key}"
        for block_name, relative, ref in (
            ("unit_blob", entries[key]["path"], _ref(entries[key], allow_extra=True)),
            ("before", None, None),
            ("after", None, None),
        ):
            block = unit[block_name]
            if block_name == "unit_blob":
                identifier = block
            else:
                if not isinstance(block, Mapping) or set(block) != {"reference", "blob"}:
                    raise resume.ResumeRefusal(
                        "raw envelope archive environment block is malformed"
                    )
                ref, relative, identifier = (
                    _ref(block["reference"]),
                    block["reference"]["path"],
                    block["blob"],
                )
            if not isinstance(identifier, str) or identifier not in blobs:
                raise resume.ResumeRefusal("raw envelope archive blob is absent")
            used_blobs.add(identifier)
            _write_triple(
                root,
                str(relative),
                {
                    name: base64.b64encode(blobs[identifier][name]).decode("ascii")
                    for name in _TRIPLE
                },
                ref,
            )
        unit_path = root / resume._safe_relative(entries[key]["path"])
        read_key, payload = resume._read_unit(unit_path, binding, entries[key])
        attempt = carried[unit["attempt"]]
        receipt_ref, receipt_raw, receipt = attempt["receipt"]
        manifest_ref, manifest_raw, manifest = attempt["manifest"]
        control_ref, control_raw, control = attempt["control"]
        _validate_provenance(
            receipt,
            manifest,
            control,
            receipt_ref=receipt_ref,
            manifest_ref=manifest_ref,
            control_ref=control_ref,
            binding=binding,
            entry=entries[key],
            payload=payload,
        )
        if type(read_key) is not type(key) or read_key != key or payload.get("canary_only") is True:
            raise resume.ResumeRefusal("raw envelope reconstructed unit is foreign")
        for name, raw in (
            ("receipt", receipt_raw),
            ("manifest", manifest_raw),
            ("control", control_raw),
        ):
            export_bytes(root / "metadata" / f"{name}.json", raw)
        results[key] = payload
    if seen != set(entries):
        raise resume.ResumeRefusal("raw envelope archive omitted a ledger unit")
    used_payloads = {archive["blobs"][identifier]["payload"] for identifier in used_blobs}
    if used_blobs != set(blobs) or used_payloads != set(payloads):
        raise resume.ResumeRefusal("raw envelope contains unreferenced foreign byte blobs")
    carried_refs: list[dict[str, Any]] = []
    attempt_names: set[str] = set()
    for index, attempt in enumerate(carried):
        receipt_ref, _, receipt = attempt["receipt"]
        carried_refs.append(receipt_ref)
        name = receipt.get("job_id")
        if not isinstance(name, str) or not name or name in attempt_names:
            raise resume.ResumeRefusal("raw envelope carried attempt identity is duplicate")
        attempt_names.add(name)
        exports = receipt.get("exports")
        if not isinstance(exports, list):
            raise resume.ResumeRefusal("raw envelope carried receipt lacks exports")
        observed: dict[str, dict[str, Any]] = {}
        for exported in exports:
            ref = _ref(exported)
            if ref["path"] in observed:
                raise resume.ResumeRefusal("raw envelope carried receipt has duplicate exports")
            observed[ref["path"]] = ref
        expected: dict[str, dict[str, Any]] = {}
        for unit in archive["units"]:
            if unit["attempt"] != index:
                continue
            key = unit["key"]
            entry = entries[key]
            expected[entry["path"]] = _ref(entry, allow_extra=True)
            if results[key].get("timing_attempt_id") != name:
                raise resume.ResumeRefusal(
                    "raw envelope carried receipt has foreign timing payload"
                )
        actual = {path: ref for path, ref in observed.items() if path.startswith("timing-units/")}
        if actual != expected:
            raise resume.ResumeRefusal("raw envelope carried receipt timing-unit closure differs")
    if carried_refs != carrier["broker_receipts"]:
        raise resume.ResumeRefusal("raw envelope broker receipt closure differs")
    return results
