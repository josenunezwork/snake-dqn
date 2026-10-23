"""Fail-closed, no-network release-asset metadata scanner for P1.

The scanner deliberately never calls ``torch.load`` or unpickles an object.  It
only parses container structure, pickle opcodes, NPY headers, and textual
metadata that could be made public with a release asset.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickletools
import re
import struct
import tarfile
import zipfile
from ast import literal_eval
from pathlib import Path
from typing import Any

import numpy as np

_MAX_MEMBERS = 4096
_MAX_MEMBER_BYTES = 2 * 1024**3
_TEXT_EXTENSIONS = {".json", ".jsonl", ".txt", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
_ABS_PATH = re.compile(r"(?<![\w.-])/(?:Users|home|private|var|tmp)/[^\s\"'\\]+")
_CONTENT_PATH = re.compile(r"(?<![\w.-])/content(?:/[^\s\"'\\]+)?")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_TOKEN_PATTERNS = (
    ("github_token", re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    (
        "oauth_token",
        re.compile(r"(?:ya29\.[A-Za-z0-9._-]{12,}|(?<![A-Za-z0-9_./-])1//[A-Za-z0-9._-]{30,})"),
    ),
    ("api_token", re.compile(r"(?:AIza[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,})")),
    ("private_pem", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
)
_CREDENTIAL_KEY = re.compile(
    r"(?:access[_-]?token|refresh[_-]?token|client[_-]?secret|api[_-]?key|"
    r"secret[_-]?key|password|credential|authorization|bearer)",
    re.I,
)
_ASSIGNMENT_SECRET = re.compile(
    r"(?:access[_-]?token|refresh[_-]?token|client[_-]?secret|api[_-]?key|"
    r"secret[_-]?key|password|authorization)\s*[=:]\s*[\"']?([^\s\"',}]{8,})",
    re.I,
)
_SECRET_KINDS = {
    "github_token",
    "aws_access_key",
    "oauth_token",
    "api_token",
    "private_pem",
    "credential_assignment",
    "credential_value",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _Scanner:
    def __init__(self) -> None:
        self.findings: list[dict[str, str]] = []
        self.methods: dict[str, int] = {}
        self.failures: list[str] = []

    def _count(self, method: str) -> None:
        self.methods[method] = self.methods.get(method, 0) + 1

    def _finding(self, kind: str, location: str, indicator: str) -> None:
        # Never retain a matched value: results are safe to place in the plan.
        row = {"kind": kind, "location": location, "indicator": indicator, "value": "[REDACTED]"}
        if row not in self.findings:
            self.findings.append(row)

    def text(self, value: str, location: str) -> None:
        self._count("text")
        for kind, pattern in _TOKEN_PATTERNS:
            if pattern.search(value):
                self._finding(kind, location, kind)
        if _ASSIGNMENT_SECRET.search(value):
            self._finding("credential_assignment", location, "credential assignment")
        for match in _ABS_PATH.finditer(value):
            self._finding("absolute_path", location, match.group(0))
        for match in _CONTENT_PATH.finditer(value):
            self._finding("content_path", location, match.group(0))
        for match in _EMAIL.finditer(value):
            self._finding("email", location, match.group(0))

    def json_value(self, value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_location = f"{location}.{key}"
                self.text(str(key), child_location)
                if _CREDENTIAL_KEY.search(str(key)):
                    self._finding("credential_field_indicator", child_location, str(key))
                    if isinstance(child, str) and len(child.strip()) >= 8:
                        self._finding("credential_value", child_location, str(key))
                self.json_value(child, child_location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self.json_value(child, f"{location}[{index}]")
        elif isinstance(value, str):
            self.text(value, location)

    def json_bytes(self, raw: bytes, location: str) -> None:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.failures.append(
                f"unsupported JSON metadata at {location}: {exc.__class__.__name__}"
            )
            return
        self._count("json")
        self.json_value(value, location)

    def pickle_bytes(self, raw: bytes, location: str, depth: int = 0) -> None:
        if depth > 4:
            self.failures.append(f"pickle nesting limit at {location}")
            return
        try:
            ops = list(pickletools.genops(raw))
        except Exception as exc:  # pickletools is parser-only, and failures block publication.
            self.failures.append(
                f"unsupported pickle metadata at {location}: {exc.__class__.__name__}"
            )
            return
        self._count("pickle")
        pending_credential_key: str | None = None
        for opcode, arg, position in ops:
            where = f"{location}:pickle@{position}"
            if isinstance(arg, str):
                self.text(arg, where)
                credential_key = _CREDENTIAL_KEY.search(arg)
                if credential_key:
                    indicator = credential_key.group(0)
                    self._finding("credential_field_indicator", where, indicator)
                    pending_credential_key = indicator
                elif pending_credential_key and len(arg.strip()) >= 8:
                    # Pickle emits mapping keys and values as independent
                    # opcodes; this records the relation without unpickling.
                    self._finding("credential_value", where, pending_credential_key)
                    pending_credential_key = None
            elif isinstance(arg, bytes):
                if pending_credential_key and len(arg.strip()) >= 8:
                    self._finding("credential_value", where, pending_credential_key)
                    pending_credential_key = None
                elif arg.startswith(b"\x80"):
                    self.pickle_bytes(arg, where, depth + 1)
                elif len(arg) <= 4 * 1024 * 1024:
                    try:
                        decoded = arg.decode("utf-8")
                    except UnicodeDecodeError:
                        self._count("binary_storage_skipped")
                    else:
                        if (
                            decoded
                            and sum(char.isprintable() or char.isspace() for char in decoded)
                            / len(decoded)
                            > 0.9
                        ):
                            self.text(decoded, where)
                        else:
                            self._count("binary_storage_skipped")

    def npy_bytes(self, raw: bytes, location: str) -> None:
        if not raw.startswith(b"\x93NUMPY") or len(raw) < 10:
            self.failures.append(f"unsupported NPY header at {location}")
            return
        major = raw[6]
        if major == 1:
            offset, length = 10, struct.unpack("<H", raw[8:10])[0]
        elif major in {2, 3}:
            offset, length = 12, struct.unpack("<I", raw[8:12])[0]
        else:
            self.failures.append(f"unsupported NPY version at {location}")
            return
        header = raw[offset : offset + length]
        try:
            header_text = header.decode("latin1")
            metadata = literal_eval(header_text)
            dtype = np.dtype(metadata["descr"])
        except (KeyError, SyntaxError, TypeError, ValueError):
            self.failures.append(f"unsupported NPY header at {location}")
            return
        self._count("npy_header")
        self.text(header_text, f"{location}:header")
        # Object arrays embed pickle data.  Refuse rather than execute or guess.
        if dtype.hasobject:
            self.failures.append(f"object NPY requires unsupported metadata decode at {location}")
        else:
            shape = metadata.get("shape")
            if not isinstance(shape, tuple) or not all(
                isinstance(item, int) and item >= 0 for item in shape
            ):
                self.failures.append(f"unsupported NPY shape at {location}")
                return
            count = 1
            for dimension in shape:
                count *= dimension
            payload = raw[offset + length :]
            if len(payload) != count * dtype.itemsize:
                self.failures.append(f"NPY payload size mismatch at {location}")
                return
            array = np.frombuffer(payload, dtype=dtype, count=count)
            string_names = [name for name in dtype.names or () if dtype[name].kind in {"S", "U"}]
            if dtype.kind in {"S", "U"}:
                for item in array:
                    self.text(str(item), f"{location}:string-data")
            elif string_names:
                for name in string_names:
                    for item in array[name]:
                        self.text(str(item), f"{location}:string-data:{name}")
            else:
                self._count("numeric_npy_storage_skipped")

    def archive_zip(self, raw: bytes | Path, location: str) -> None:
        try:
            archive = zipfile.ZipFile(io.BytesIO(raw) if isinstance(raw, bytes) else raw)
            entries = archive.infolist()
        except (OSError, zipfile.BadZipFile) as exc:
            self.failures.append(
                f"unsupported ZIP metadata at {location}: {exc.__class__.__name__}"
            )
            return
        if len(entries) > _MAX_MEMBERS:
            self.failures.append(f"ZIP member limit exceeded at {location}")
            return
        self._count("zip")
        self.text(archive.comment.decode("utf-8", "replace"), f"{location}:comment")
        for entry in entries:
            if entry.file_size > _MAX_MEMBER_BYTES:
                self.failures.append(
                    f"ZIP member size limit exceeded at {location}!{entry.filename}"
                )
                continue
            self.text(entry.filename, f"{location}!name")
            self.text(
                entry.comment.decode("utf-8", "replace"), f"{location}!{entry.filename}:comment"
            )
            if entry.is_dir():
                continue
            try:
                # PyTorch's ZIP serializer keeps tensor storage in numbered
                # data members.  It is numeric storage, not metadata.
                if re.search(r"(?:^|/)data/\d+$", entry.filename):
                    self._count("numeric_torch_storage_skipped")
                    continue
                member = archive.read(entry)
                if Path(entry.filename).name in {
                    "version",
                    "byteorder",
                    "serialization_id",
                    ".format_version",
                    ".storage_alignment",
                }:
                    self.text(member.decode("utf-8", "replace"), f"{location}!{entry.filename}")
                else:
                    self.payload(member, f"{location}!{entry.filename}")
            except RuntimeError as exc:
                self.failures.append(
                    f"unreadable ZIP member at {location}!{entry.filename}: "
                    f"{exc.__class__.__name__}"
                )

    def archive_tar(self, raw: bytes | Path, location: str) -> None:
        try:
            archive = (
                tarfile.open(fileobj=io.BytesIO(raw), mode="r:*")
                if isinstance(raw, bytes)
                else tarfile.open(raw, mode="r:*")
            )
            members = archive.getmembers()
        except (OSError, tarfile.TarError) as exc:
            self.failures.append(
                f"unsupported TAR metadata at {location}: {exc.__class__.__name__}"
            )
            return
        if len(members) > _MAX_MEMBERS:
            self.failures.append(f"TAR member limit exceeded at {location}")
            return
        self._count("tar")
        for member in members:
            self.text(member.name, f"{location}!name")
            if member.size > _MAX_MEMBER_BYTES:
                self.failures.append(f"TAR member size limit exceeded at {location}!{member.name}")
                continue
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is None:
                self.failures.append(f"unreadable TAR member at {location}!{member.name}")
                continue
            self.payload(handle.read(), f"{location}!{member.name}")

    def payload(self, raw: bytes, location: str) -> None:
        suffix = Path(location.split("!")[-1]).suffix.lower()
        if (
            raw.startswith(b"PK\x03\x04")
            or suffix in {".pt", ".npz", ".zip"}
            and raw.startswith(b"PK")
        ):
            self.archive_zip(raw, location)
        elif raw.startswith(b"\x1f\x8b") or location.endswith((".tar.gz", ".tgz")):
            self.archive_tar(raw, location)
        elif suffix == ".npy":
            self.npy_bytes(raw, location)
        elif suffix in _TEXT_EXTENSIONS or location.endswith((".sha256", ".complete")):
            if suffix in {".json", ".jsonl"}:
                self.json_bytes(raw, location)
            else:
                try:
                    self.text(raw.decode("utf-8"), location)
                except UnicodeDecodeError:
                    self.failures.append(f"unsupported text metadata at {location}")
        elif suffix in {".pt", ".pth", ".pkl", ".pickle"} or raw.startswith(b"\x80"):
            self.pickle_bytes(raw, location)
        else:
            self.failures.append(f"unsupported release metadata type at {location}")


def scan_asset(path: Path) -> dict[str, Any]:
    """Return a redacted, fail-closed publication scan for one immutable asset."""
    resolved = path.resolve()
    scanner = _Scanner()
    if not resolved.is_file():
        scanner.failures.append(f"asset is not a regular file: {resolved}")
        size = 0
    else:
        size = resolved.stat().st_size
        with resolved.open("rb") as handle:
            magic = handle.read(8)
        suffix = resolved.suffix.lower()
        if magic.startswith(b"PK"):
            scanner.archive_zip(resolved, str(resolved))
        elif magic.startswith(b"\x1f\x8b") or str(resolved).endswith((".tar.gz", ".tgz")):
            scanner.archive_tar(resolved, str(resolved))
        elif suffix in {".pt", ".pth", ".pkl", ".pickle"} or magic.startswith(b"\x80"):
            scanner.pickle_bytes(resolved.read_bytes(), str(resolved))
        else:
            scanner.payload(resolved.read_bytes(), str(resolved))
    secret_findings = [row for row in scanner.findings if row["kind"] in _SECRET_KINDS]
    blocked = bool(secret_findings or scanner.failures)
    disclosed = bool(scanner.findings)
    return {
        "path": str(resolved),
        "bytes": size,
        "sha256": _sha256(resolved) if resolved.is_file() else None,
        "status": (
            "BLOCKED"
            if blocked
            else (
                "PASS_NO_SECRETS_WITH_PATH_OR_PII_DISCLOSURES"
                if disclosed
                else "PASS_NO_SECRETS_OR_PII_FOUND"
            )
        ),
        "publication_blocked": blocked,
        "method_counts": dict(sorted(scanner.methods.items())),
        "findings": scanner.findings,
        "failures": scanner.failures,
        "scanner": "p1-release-scan/v1-no-unpickle-no-network",
    }
