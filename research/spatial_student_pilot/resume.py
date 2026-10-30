"""Immutable spatial units and outcome-blind, Mac-verified completion ledgers.

Only self-generated, hash-bound payloads are unpickled. Ledger construction reads
paths, bounded publication metadata and hashes, never a tensor or scientific
outcome. A supervisor receipt binds each scientific export and allocation time.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import torch

from research.spatial_student_pilot.config import PilotConfig

SCHEMA = "spatial-verified-ledger/v1"
UNIT_SCHEMA = "spatial-immutable-unit/v1"
MAX_UNIT_BYTES = 128 * 1024**2
MAX_EXPORT_BYTES = 2 * 1024**3
MAX_DECODED_BYTES = 512 * 1024**2
_UNIT_ID = re.compile(r"(?:collect|search|fit|assess|gameplay)-[0-9]{4,9}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _identity(cfg: PilotConfig) -> dict[str, Any]:
    return dict(
        namespace=cfg.namespace, allocation_index=cfg.allocation_index, config_sha256=cfg.sha256
    )


def _unit_path(cfg: PilotConfig, unit_id: str) -> str:
    if not isinstance(unit_id, str) or not _UNIT_ID.fullmatch(unit_id):
        raise ValueError("invalid logical unit identity")
    stage, raw_index = unit_id.split("-")
    index = int(raw_index)
    if stage == "fit":
        final = cfg.bc_updates + cfg.improvement_updates
        eligible = 0 < index <= final and (
            index % cfg.checkpoint_every == 0 or index in {cfg.bc_updates, final}
        )
        if len(raw_index) != 7 or not eligible:
            raise ValueError("ineligible fit trajectory snapshot")
    else:
        counts = {
            "collect": math.ceil(cfg.worlds / cfg.world_batch_size),
            "search": math.ceil((cfg.train_late + cfg.train_ordinary) / cfg.search_chunk),
            "assess": math.ceil((cfg.assessment_late + cfg.assessment_ordinary) / cfg.search_chunk),
            "gameplay": math.ceil(cfg.gameplay_worlds / cfg.world_batch_size),
        }
        if len(raw_index) != 4 or index >= counts[stage]:
            raise ValueError("ineligible frozen allocation unit")
    namespace = hashlib.sha256(cfg.namespace.encode()).hexdigest()[:16]
    return f"units/{namespace}-a{cfg.allocation_index:04d}-{unit_id}.pt.gz"


def _safe(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if not relative or rel.is_absolute() or ".." in rel.parts or str(rel) != relative:
        raise ValueError("unsafe unit path")
    root = Path(root)
    if root.is_symlink():
        raise ValueError("symlink unit root")
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink unit path")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("unit path escape")
    return path


def empty_ledger(cfg: PilotConfig) -> dict[str, Any]:
    """The bound first-attempt input, including zero prior allocated time."""
    return dict(schema=SCHEMA, **_identity(cfg), consumed_seconds=0.0, receipts=[], units=[])


def _validate_entry(entry: dict[str, Any], cfg: PilotConfig) -> None:
    if not isinstance(entry, dict) or entry.get("schema") != UNIT_SCHEMA:
        raise ValueError("invalid completed unit schema")
    if any(entry.get(k) != v for k, v in _identity(cfg).items()):
        raise ValueError("foreign namespace/allocation/config")
    if type(entry.get("allocation_index")) is not int:
        raise ValueError("invalid allocation index")
    if entry.get("path") != _unit_path(cfg, entry.get("unit_id")):
        raise ValueError("foreign deterministic unit path")
    if type(entry.get("size")) is not int or not 0 < entry["size"] <= MAX_UNIT_BYTES:
        raise ValueError("unit size exceeds bound")
    if not isinstance(entry.get("sha256"), str) or not _SHA.fullmatch(entry["sha256"]):
        raise ValueError("invalid unit digest")


def validate_ledger(ledger: dict[str, Any], cfg: PilotConfig) -> None:
    """Reject foreign identity, duplicate IDs, malformed accounting and bounds."""
    if not isinstance(ledger, dict) or ledger.get("schema") != SCHEMA:
        raise ValueError("invalid verified ledger schema")
    if any(ledger.get(k) != v for k, v in _identity(cfg).items()):
        raise ValueError("foreign ledger namespace/allocation/config")
    if type(ledger.get("allocation_index")) is not int:
        raise ValueError("invalid ledger allocation index")
    elapsed = ledger.get("consumed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (float, int)):
        raise ValueError("invalid consumed seconds")
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid consumed seconds")
    units, receipts = ledger.get("units"), ledger.get("receipts")
    if not isinstance(units, list) or not isinstance(receipts, list):
        raise ValueError("invalid ledger inventories")
    if len(receipts) > 3:
        raise ValueError("original plus two completion attempts exhausted")
    seen = set()
    total = 0
    for entry in units:
        _validate_entry(entry, cfg)
        if entry["unit_id"] in seen:
            raise ValueError("duplicate unit identity")
        seen.add(entry["unit_id"])
        total += entry["size"]
    if total > MAX_EXPORT_BYTES:
        raise ValueError("ledger exceeds total export bound")
    receipt_digests = set()
    receipt_generations = set()
    receipt_elapsed = 0.0
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise ValueError("invalid receipt binding")
        digest, seconds = receipt.get("sha256"), receipt.get("elapsed_seconds")
        if (
            not isinstance(digest, str)
            or not _SHA.fullmatch(digest)
            or digest in receipt_digests
            or not isinstance(receipt.get("generation"), str)
            or not receipt["generation"]
            or receipt["generation"] in receipt_generations
            or isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("invalid receipt binding/accounting")
        receipt_digests.add(digest)
        receipt_generations.add(receipt["generation"])
        receipt_elapsed += seconds
    if abs(receipt_elapsed - elapsed) > 1e-6:
        raise ValueError("ledger allocation accounting differs from receipt bindings")
    if cfg.mode == "pilot" and units and (not receipts or elapsed <= 0):
        raise ValueError("scientific ledger lacks verified allocation accounting")


def _verify(
    root: Path, cfg: PilotConfig, unit_id: str, expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    relative = _unit_path(cfg, unit_id)
    path = _safe(root, relative)
    marker = _safe(root, relative + ".complete")
    sidecar = _safe(root, relative + ".sha256")
    if not marker.is_file() or not 0 < marker.stat().st_size <= 4096:
        raise ValueError("missing or oversized completion marker")
    entry = json.loads(marker.read_text())
    _validate_entry(entry, cfg)
    if entry["unit_id"] != unit_id or (expected is not None and entry != expected):
        raise ValueError("completed unit differs from verified ledger")
    if not path.is_file() or path.stat().st_size != entry["size"]:
        raise ValueError("missing or wrong unit size")
    if not sidecar.is_file() or not 0 < sidecar.stat().st_size <= 1024:
        raise ValueError("missing or oversized digest sidecar")
    words = sidecar.read_text().split()
    if not words or words[0] != entry["sha256"] or _sha(path) != entry["sha256"]:
        raise ValueError("unit digest mismatch")
    return entry


def load_verified_unit(root: Path, cfg: PilotConfig, unit_id: str) -> Any:
    """Read a local self-generated unit only after publication/identity checks."""
    entry = _verify(Path(root), cfg, unit_id)
    path = _safe(root, entry["path"])
    with gzip.open(path, "rb") as handle:
        raw = handle.read(MAX_DECODED_BYTES + 1)
    if len(raw) > MAX_DECODED_BYTES:
        raise ValueError("decoded unit exceeds bound")
    envelope = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(envelope, dict) or envelope.get("identity") != {
        **_identity(cfg),
        "unit_id": unit_id,
        "schema": UNIT_SCHEMA,
    }:
        raise ValueError("payload identity differs from completion marker")
    return envelope["payload"]


def _publish_bytes(path: Path, content: bytes) -> None:
    """Atomic create-only publication; concurrent/existing writers cannot replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class UnitStore:
    """Skip exactly verified units; new output must be fresh for every attempt."""

    def __init__(
        self, output: Path, cfg: PilotConfig, ledger: dict[str, Any], input_dir: Path
    ) -> None:
        validate_ledger(ledger, cfg)
        self.output, self.cfg, self.input_dir = Path(output), cfg, Path(input_dir)
        self.ledger = ledger
        self._entries = {entry["unit_id"]: entry for entry in ledger["units"]}
        self._roots = {name: self.input_dir for name in self._entries}
        for entry in ledger["units"]:
            _verify(self.input_dir, cfg, entry["unit_id"], entry)
        if self.output.is_symlink():
            raise ValueError("symlink output root")
        # Existing completed output is forbidden: only the Mac-verified bound
        # input ledger may authorize reuse across attempts.
        if self.output.exists() and any(self.output.rglob("*.pt.gz.complete")):
            raise ValueError("nonfresh output; use a verified input ledger")
        self.output.mkdir(parents=True, exist_ok=True)

    def list_ids(self, prefix: str = "") -> list[str]:
        return sorted(name for name in self._entries if name.startswith(prefix))

    def latest_fit(self) -> str | None:
        """Latest eligible full-state trajectory snapshot, by completed updates."""
        snapshots = self.list_ids("fit-")
        return snapshots[-1] if snapshots else None

    def has(self, unit_id: str) -> bool:
        _unit_path(self.cfg, unit_id)
        if unit_id not in self._entries:
            return False
        _verify(self._roots[unit_id], self.cfg, unit_id, self._entries[unit_id])
        return True

    def get(self, unit_id: str) -> Any | None:
        if not self.has(unit_id):
            return None
        return load_verified_unit(self._roots[unit_id], self.cfg, unit_id)

    def publish(self, unit_id: str, payload: Any) -> Path:
        relative = _unit_path(self.cfg, unit_id)
        if unit_id in self._entries:
            raise ValueError("verified unit cannot be recomputed or replaced")
        raw = io.BytesIO()
        torch.save(
            dict(
                identity={**_identity(self.cfg), "unit_id": unit_id, "schema": UNIT_SCHEMA},
                payload=payload,
            ),
            raw,
        )
        if raw.tell() > MAX_DECODED_BYTES:
            raise ValueError("decoded unit exceeds bound")
        content = gzip.compress(raw.getvalue(), compresslevel=1, mtime=0)
        if len(content) > MAX_UNIT_BYTES:
            raise ValueError("unit exceeds export bound")
        entry = dict(
            schema=UNIT_SCHEMA,
            **_identity(self.cfg),
            unit_id=unit_id,
            path=relative,
            size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        # Budget payloads plus bounded digest/marker metadata, including verified
        # prior units. This is conservative for each fresh attempt export tree.
        current = sum(e["size"] + 5120 for e in self._entries.values())
        if current + len(content) + 5120 > MAX_EXPORT_BYTES:
            raise ValueError("total export bound exceeded")
        path = _safe(self.output, relative)
        _publish_bytes(path, content)
        _publish_bytes(
            _safe(self.output, relative + ".sha256"),
            (entry["sha256"] + "  " + path.name + "\n").encode(),
        )
        _publish_bytes(
            _safe(self.output, relative + ".complete"),
            (json.dumps(entry, sort_keys=True) + "\n").encode(),
        )
        self._entries[unit_id] = entry
        self._roots[unit_id] = self.output
        return path


def _root_entries(root: Path, cfg: PilotConfig) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("invalid export root")
    entries = []
    for marker in sorted(root.rglob("*.pt.gz.complete")):
        if marker.is_symlink() or not 0 < marker.stat().st_size <= 4096:
            raise ValueError("invalid completion marker")
        entry = json.loads(marker.read_text())
        _validate_entry(entry, cfg)
        if marker.relative_to(root).as_posix() != entry["path"] + ".complete":
            raise ValueError("foreign publication path")
        entries.append(_verify(root, cfg, entry["unit_id"], entry))
    return entries


def build_ledger(
    cfg: PilotConfig,
    export_roots: Iterable[Path],
    receipts: Iterable[Path] = (),
    *,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Verify union by hashes only; require genuine stopped supervisor receipts.

    Each receipt lives beside manifest.json and binds exports/<relative path>.
    Supply the spatial export root (normally output/exports/spatial) for each
    attempt; receipt roots may have no completed units but still charge time.
    """
    roots = [Path(p) for p in export_roots]
    ledger = empty_ledger(cfg)
    by_id = {}
    sources = {}
    for root in roots:
        for entry in _root_entries(root, cfg):
            if entry["unit_id"] in by_id:
                raise ValueError("duplicate unit identity across export roots")
            by_id[entry["unit_id"]] = entry
            sources[entry["unit_id"]] = root
    covered = set()
    seen_receipts = set()
    seen_generations = set()
    elapsed = 0.0
    for path in map(Path, receipts):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024**2:
            raise ValueError("invalid supervisor receipt")
        digest = _sha(path)
        if digest in seen_receipts:
            raise ValueError("duplicate supervisor receipt")
        seen_receipts.add(digest)
        receipt = json.loads(path.read_text())
        duration = receipt.get("elapsed_seconds")
        if (
            receipt.get("stop_verified") is not True
            or receipt.get("cleanup_error")
            or isinstance(duration, bool)
            or not isinstance(duration, (float, int))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError("receipt lacks verified stop/allocation elapsed time")
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.is_file() or _sha(manifest_path) != receipt.get("manifest_sha256"):
            raise ValueError("supervisor manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text())
        generation = receipt.get("generation")
        if (
            receipt.get("commit") != manifest.get("commit")
            or not isinstance(generation, str)
            or not generation
            or generation in seen_generations
        ):
            raise ValueError("supervisor receipt identity mismatch")
        seen_generations.add(generation)
        exports = receipt.get("exports")
        if not isinstance(exports, list):
            raise ValueError("missing verified export inventory")
        inventory = {entry["path"]: entry for entry in exports}
        if len(inventory) != len(exports):
            raise ValueError("duplicate supervisor export inventory")
        for name, entry in by_id.items():
            candidate = _safe(sources[name], entry["path"])
            try:
                rel = candidate.relative_to(path.parent / "exports").as_posix()
            except ValueError:
                continue
            exported = inventory.get(rel)
            if (
                not isinstance(exported, dict)
                or exported.get("sha256") != entry["sha256"]
                or exported.get("size") != entry["size"]
            ):
                raise ValueError("completed unit not bound by supervisor export receipt")
            covered.add(name)
        elapsed += duration
        ledger["receipts"].append(
            dict(
                path=str(path.resolve()),
                sha256=digest,
                elapsed_seconds=duration,
                generation=receipt["generation"],
            )
        )
    if cfg.mode == "pilot" and set(by_id) != covered:
        raise ValueError("scientific units lack supervisor receipt bindings")
    if elapsed_seconds is not None and (
        not math.isfinite(elapsed_seconds) or abs(elapsed_seconds - elapsed) > 1e-6
    ):
        raise ValueError("elapsed-seconds differs from cumulative supervisor receipts")
    ledger["units"] = [by_id[name] for name in sorted(by_id)]
    ledger["consumed_seconds"] = elapsed
    validate_ledger(ledger, cfg)
    return ledger


def upload_inputs(
    ledger: dict[str, Any], roots: Iterable[Path], cfg: PilotConfig
) -> list[dict[str, str]]:
    """Bind verified payload/metadata inputs using supervisor relative dest paths."""
    validate_ledger(ledger, cfg)
    roots = list(map(Path, roots))
    result = []
    for entry in ledger["units"]:
        found = [root for root in roots if _safe(root, entry["path"] + ".complete").is_file()]
        if len(found) != 1:
            raise ValueError("unit requires exactly one source export root")
        _verify(found[0], cfg, entry["unit_id"], entry)
        for suffix in ("", ".sha256", ".complete"):
            path = _safe(found[0], entry["path"] + suffix)
            result.append(
                dict(path=str(path.resolve()), dest=entry["path"] + suffix, sha256=_sha(path))
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--roots", nargs="+", required=True, type=Path)
    parser.add_argument("--receipts", nargs="*", default=[], type=Path)
    parser.add_argument("--elapsed-seconds", type=float, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = PilotConfig.from_dict(json.loads(args.config.read_text()))
    ledger = build_ledger(cfg, args.roots, args.receipts, elapsed_seconds=args.elapsed_seconds)
    with args.output.open("x") as handle:
        json.dump(ledger, handle, sort_keys=True, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
