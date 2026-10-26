"""Bounded F5 status projection; no real input is executed by this review package.

The historical converted NPZ has independent NPY members. Only eight metadata
members may be opened. Outcome/observation members are never opened or decoded.
The raw JSON lives omit selectability, so no raw-JSON adapter is provided.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import stat
import struct
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO, Callable

import numpy as np

SCHEMA = "bl-f5-status-extractor/v3"
STATUSES = frozenset({"EXACT", "NOT_EVALUATED", "UNKNOWN"})
NPZ_OPEN = frozenset(
    {
        "row_id.npy",
        "life_id.npy",
        "parent_world.npy",
        "ordinary.npy",
        "natural_stream.npy",
        "legal.npy",
        "action_mask.npy",
        "status.npy",
    }
)
NPZ_MEMBERS = NPZ_OPEN | frozenset(
    {
        "local.npy",
        "global.npy",
        "scalars.npy",
        "q.npy",
        "bare.npy",
        "died.npy",
        "mass.npy",
        "sampling_weight.npy",
        "sampling_prob.npy",
    }
)
NPZ_SOURCE_BYTES = 45397643
MAX_TOTAL_ROWS = 94152
MAX_ORDINARY_ROWS = 7752
MAX_OUTPUT_UNITS = 46512
MAX_WALL_SECONDS = 120.0
MAX_HEADER_BYTES = 4096
MAX_CENTRAL_DIRECTORY_BYTES = 16384
COPY_CHUNK = 65536
RULING_SHA = "d6e2f029da689bbbe44e6c9c337ad989202a36af1fc26bcadad2f6f3be7e4fa6"
SOURCE_PATH = (
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/"
    "d1024-pack-04/inputs/train-1024.npz"
)
SOURCE_SHA = "6906bffc7241de568965c1a91f4faebe799895ee0b4650e59b20e43db1dac494"


class Refused(ValueError):
    """A redacted refusal; source values must never appear in the error."""


def refuse(category: str) -> None:
    """Emit fixed categories only, without exception or payload formatting."""
    raise Refused("status extractor refused: " + category) from None


def canonical(value: object) -> bytes:
    """Encode allowed metadata only."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def safe_path(path: Path) -> None:
    """Reject exclusions before stat/open, and every symlink component."""
    if not path.is_absolute() or ".." in path.parts:
        refuse("source path")
    lowered = str(path).lower()
    if "4096" in lowered or "t34-round0-study-01" in lowered:
        refuse("protected source")
    if any(part.is_symlink() for part in (path, *path.parents)):
        refuse("symlink source")


def validate_bindings(
    frozen_lives: dict[str, Any], exclusions: frozenset[str]
) -> dict[str, dict[str, Any]]:
    """Validate frozen metadata without following any source reference."""
    if not isinstance(frozen_lives, dict) or not 0 < len(frozen_lives) <= 969:
        refuse("life list bound")
    result, orders = {}, set()
    for life, row in frozen_lives.items():
        if not isinstance(life, str) or not re.fullmatch(r"[^/]+/[^/]+/[0-9]+", life):
            refuse("frozen life identity")
        if not isinstance(row, dict):
            refuse("frozen life binding")
        parent, order, ordinary = (
            row.get("parent_world"),
            row.get("allocation_order"),
            row.get("ordinary_only"),
        )
        if (
            not isinstance(parent, str)
            or parent != life.split("/")[1]
            or parent in exclusions
            or type(order) is not int
            or order < 0
            or order in orders
            or type(ordinary) is not bool
        ):
            refuse("frozen lineage or order")
        if "original_fold" in row and row["original_fold"] not in (1, 2, 3, 4):
            refuse("excluded fold")
        orders.add(order)
        result[life] = {
            "parent_world": parent,
            "allocation_order": order,
            "ordinary_only": ordinary,
        }
    return result


class Budget:
    """One-process monotonic deadline; checked on every bounded read and row block."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.deadline = clock() + MAX_WALL_SECONDS

    def check(self) -> None:
        if self.clock() >= self.deadline:
            refuse("wall bound")

    def read(self, stream: BinaryIO, length: int) -> bytes:
        chunks, remaining = [], length
        while remaining:
            self.check()
            value = stream.read(min(remaining, COPY_CHUNK))
            if not value:
                refuse("truncated metadata member")
            chunks.append(value)
            remaining -= len(value)
        self.check()
        return b"".join(chunks)


def _zip_structure(stream: BinaryIO, budget: Budget) -> None:
    """Bound central-directory allocation before constructing ZipFile.

    np.savez uses ordinary EOCD (17 entries); ZIP64 per-member headers are fine,
    but archive-level ZIP64, comments, trailing data and multi-disk archives are
    not emitted by the historical writer and fail closed.
    """
    stream.seek(0, 2)
    size = stream.tell()
    if not 22 <= size <= NPZ_SOURCE_BYTES:
        refuse("source byte bound")
    stream.seek(-22, 2)
    tail = budget.read(stream, 22)
    if tail[:4] != b"PK\x05\x06":
        refuse("ZIP footer schema")
    disk, central_disk, disk_count, count, length, offset, comment = struct.unpack(
        "<4H2IH", tail[4:]
    )
    if (
        disk
        or central_disk
        or disk_count != 17
        or count != 17
        or comment
        or length > MAX_CENTRAL_DIRECTORY_BYTES
        or offset + length != size - 22
    ):
        refuse("ZIP directory bound")
    stream.seek(0)


def _header(
    archive: zipfile.ZipFile, name: str, widths: dict[str, int], row_limit: int, budget: Budget
) -> tuple[np.dtype, tuple[int, ...], int, int]:
    """Read structural NPY metadata; validate dtype and shape before allocation."""
    info = archive.getinfo(name)
    # All metadata members have at most row_limit rows and 6 fields of U13.
    max_payload = row_limit * max(6 * 13 * 4, max(widths.values()) * 4)
    if (
        info.flag_bits & 1
        or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
        or info.file_size > max_payload + MAX_HEADER_BYTES + 12
    ):
        refuse("metadata member bound")
    with archive.open(info) as stream:
        if budget.read(stream, 6) != b"\x93NUMPY":
            refuse("NPY magic")
        version = budget.read(stream, 2)
        if version == b"\x01\x00":
            length = struct.unpack("<H", budget.read(stream, 2))[0]
            prefix = 10
        elif version == b"\x02\x00":
            length = struct.unpack("<I", budget.read(stream, 4))[0]
            prefix = 12
        else:
            refuse("NPY version")
        if not 0 < length <= MAX_HEADER_BYTES:
            refuse("NPY header bound")
        try:
            header = ast.literal_eval(budget.read(stream, length).decode("ascii"))
        except Exception:
            refuse("NPY header encoding")
    if (
        not isinstance(header, dict)
        or set(header) != {"descr", "fortran_order", "shape"}
        or header["fortran_order"] is not False
    ):
        refuse("NPY header schema")
    descr, shape = header["descr"], header["shape"]
    if (
        not isinstance(descr, str)
        or not isinstance(shape, tuple)
        or not 1 <= len(shape) <= 2
        or any(type(x) is not int or x <= 0 for x in shape)
        or shape[0] > row_limit
    ):
        refuse("NPY shape bound")
    bool_role = name in {"ordinary.npy", "natural_stream.npy", "legal.npy", "action_mask.npy"}
    width = widths.get(name, 13)
    if bool_role:
        if descr != "|b1":
            refuse("NPY boolean dtype")
    elif not re.fullmatch(r"<U[1-9][0-9]{0,2}", descr) or int(descr[2:]) > width:
        # _table creates Unicode via np.asarray(str), never byte/object strings.
        refuse("NPY identity or status dtype")
    dtype = np.dtype(descr)
    payload = math.prod(shape) * dtype.itemsize
    if prefix + length + payload != info.file_size:
        refuse("NPY payload byte count")
    return dtype, shape, prefix + length, payload


def _project(
    stream: BinaryIO,
    frozen_lives: dict[str, Any],
    exclusions: frozenset[str],
    budget: Budget,
    require_both_strata: bool = True,
) -> dict[str, Any]:
    bindings = validate_bindings(frozen_lives, exclusions)
    widths = {
        "life_id.npy": max(map(len, bindings)),
        "parent_world.npy": max(len(row["parent_world"]) for row in bindings.values()),
        "row_id.npy": max(map(len, bindings)) + 6,
    }
    row_limit = min(MAX_TOTAL_ROWS, sum(8 if r["ordinary_only"] else 98 for r in bindings.values()))
    _zip_structure(stream, budget)
    values = {}
    with zipfile.ZipFile(stream) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != NPZ_MEMBERS:
            refuse("NPZ member schema")
        headers = {
            name: _header(archive, name, widths, row_limit, budget) for name in sorted(NPZ_OPEN)
        }
        n = headers["row_id.npy"][1][0]
        for name, (_, shape, _, _) in headers.items():
            expected = (n, 6) if name in {"legal.npy", "action_mask.npy", "status.npy"} else (n,)
            if shape != expected:
                refuse("NPY column shape")
        # Header checks for ALL columns precede ANY numeric/string value decoding.
        for name in sorted(NPZ_OPEN):
            dtype, shape, header_bytes, payload_bytes = headers[name]
            with archive.open(name) as member:
                budget.read(member, header_bytes)
                payload = budget.read(member, payload_bytes)
                if member.read(1):
                    refuse("metadata trailing bytes")
            if dtype.kind == "b" and any(byte not in (0, 1) for byte in payload):
                refuse("boolean byte encoding")
            values[name[:-4]] = np.frombuffer(payload, dtype=dtype).reshape(shape)
    status = values["status"]
    if not set(status.flat) <= STATUSES:
        refuse("unknown status")
    ordinary = values["ordinary"]
    if not np.array_equal(ordinary, values["natural_stream"]):
        refuse("contradictory stratum flags")
    ids, lives, parents = (values[k].tolist() for k in ("row_id", "life_id", "parent_world"))
    if len(set(ids)) != n:
        refuse("duplicate row identity")
    if set(lives) != set(bindings):
        refuse("frozen life coverage")
    per_life, per_stratum, rows, strata = Counter(), Counter(), [], Counter()
    for index in range(n):
        if index % 256 == 0:
            budget.check()
        life, parent, row_id = lives[index], parents[index], ids[index]
        binding = bindings[life]
        if parent != binding["parent_world"] or not row_id.startswith(life + "/"):
            refuse("row ancestry")
        decision = row_id[len(life) + 1 :]
        if not re.fullmatch(r"0|[1-9][0-9]{0,4}", decision) or int(decision) > 14400:
            refuse("row identity")
        stratum = "ordinary" if ordinary[index] else "doom"
        if binding["ordinary_only"] and stratum != "ordinary":
            refuse("ordinary-only sibling stratum")
        per_stratum[(life, stratum)] += 1
        if per_stratum[(life, stratum)] > (8 if stratum == "ordinary" else 90):
            refuse("per-life stratum bound")
        order = per_life[life]
        per_life[life] += 1
        strata[stratum] += 1
        masks = values["action_mask"][index]
        selectable = masks if masks.any() else np.array([True, True, True, False, False, False])
        for action in range(6):
            rows.append(
                {
                    "row_id": row_id,
                    "life_id": life,
                    "parent_world": parent,
                    "allocation_order": binding["allocation_order"],
                    "row_order": order,
                    "source_row_index": index,
                    "stratum": stratum,
                    "action": action,
                    "legal": bool(values["legal"][index, action]),
                    "served_selectable": bool(selectable[action]),
                    "status": "EXACT" if status[index, action] == "EXACT" else "NONEXACT",
                }
            )
    if require_both_strata and (not strata["ordinary"] or not strata["doom"]):
        refuse("both strata required")
    if strata["ordinary"] > MAX_ORDINARY_ROWS:
        refuse("ordinary inventory bound")
    rows.sort(key=lambda row: (row["allocation_order"], row["row_order"], row["action"]))
    missing = [
        r
        for r in rows
        if r["stratum"] == "ordinary"
        and r["legal"]
        and r["served_selectable"]
        and r["status"] == "NONEXACT"
    ]
    if len(missing) > MAX_OUTPUT_UNITS:
        refuse("U inventory bound")
    budget.check()
    result = {
        "schema": SCHEMA,
        "rows": n,
        "strata": {s: strata[s] for s in ("ordinary", "doom")},
        "status_inventory": rows,
        "status_inventory_sha256": hashlib.sha256(canonical(rows)).hexdigest(),
        "cov_missing_ordinary_units": missing,
        "count_U": len(missing),
        "U_inventory_sha256": hashlib.sha256(canonical(missing)).hexdigest(),
        "if_COV_selected": (
            "HOLD_NO_TARGET"
            if not missing
            else ("HOLD_COVERAGE_IMPLEMENTATION" if len(missing) > 46080 else "PENDING_R4_SELECTOR")
        ),
        "count_reconciliation": {
            "eligible_lives": len(bindings),
            "observed_lives": len(per_life),
            "rows": n,
            "rows_from_strata": sum(strata.values()),
            "action_units": len(rows),
            "expected_action_units": 6 * n,
            "U": len(missing),
        },
    }
    budget.check()
    return result


def project_synthetic_npz(
    path: Path,
    frozen_lives: dict[str, Any],
    *,
    exclusions: frozenset[str] = frozenset(),
    require_both_strata: bool = True,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Exercise the same projection on a bounded tempfile fixture only.

    Historical files live outside the OS temporary directory and cannot enter
    this helper. No path or source reference from a fixture is followed.
    """
    safe_path(path)
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if not path.is_relative_to(temporary_root):
        refuse("synthetic tempfile boundary")
    try:
        with path.open("rb") as stream:
            return _project(stream, frozen_lives, exclusions, Budget(clock), require_both_strata)
    except Refused:
        raise
    except Exception:
        refuse("metadata encoding")


def validate_acceptance(receipt: dict[str, Any], pins: dict[str, str]) -> None:
    """Offline-testable source-bound acceptance check; it grants no adoption."""
    expected = {"schema", "decision", "independent_reviewer", "ruling_sha256", "pins"}
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected
        or receipt["schema"] != "bl-f5-extractor-independent-acceptance/v1"
        or receipt["decision"] != "ACCEPT_STATUS_ONLY_EXTRACTOR"
        or not isinstance(receipt["independent_reviewer"], str)
        or not receipt["independent_reviewer"].strip()
        or receipt["ruling_sha256"] != RULING_SHA
        or receipt["pins"] != pins
        or set(pins) != {"source", "tests", "contract", "inventory"}
        or any(not re.fullmatch(r"[0-9a-f]{64}", p) for p in pins.values())
    ):
        refuse("independent acceptance required")


def open_source(path: Path) -> BinaryIO:
    """Open with no-follow dirfds at every path component, preventing escape."""
    safe_path(path)
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        final = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
        if not stat.S_ISREG(os.fstat(final).st_mode):
            os.close(final)
            refuse("regular source required")
        return os.fdopen(final, "rb")
    finally:
        os.close(descriptor)


def extract_accepted(acceptance: dict[str, Any]) -> dict[str, Any]:
    """Later operator API: exact acceptance and frozen identity precede real open.

    This function is NOT invoked during package preparation or by its CLI. The
    independent judge supplies the acceptance object; the implementer cannot
    grant it. The API performs no scientific work or mechanism selection.
    """
    root = Path(__file__).absolute().parents[2]
    folder = root / "research/bl_residual/f5_r10a_closure_20261026"
    paths = {
        "source": Path(__file__).absolute(),
        "tests": root / "tests/test_f5_status_extractor.py",
        "contract": folder / "extractor-contract.json",
        "inventory": folder / "extractor-input-inventory.json",
    }
    content = {}
    for name, path in paths.items():
        safe_path(path)
        if not path.is_file() or path.stat().st_size > 2_000_000:
            refuse("review artifact bound")
        content[name] = path.read_bytes()
    pins = {key: hashlib.sha256(value).hexdigest() for key, value in content.items()}
    validate_acceptance(acceptance, pins)
    try:
        inventory = json.loads(content["inventory"])
        if (
            inventory["schema"] != "bl-f5-status-input-inventory/v1"
            or inventory["annex_sha256"]
            != "8ee06291b23a1ef8c657b59a4f4990349ae4c524a680c47f06785a47d227f764"
            or inventory["counts"] != {"selected": 960, "ordinary_only_siblings": 9, "total": 969}
        ):
            refuse("frozen inventory schema")
        source = inventory["converted_container"]
        if (source["path"], source["sha256"], source["size"]) != (
            SOURCE_PATH,
            SOURCE_SHA,
            NPZ_SOURCE_BYTES,
        ):
            refuse("frozen source identity")
        records = inventory["original_life_sources"]
        if (
            len(records) != 969
            or len({r["life_id"] for r in records}) != 969
            or hashlib.sha256(canonical(records)).hexdigest()
            != inventory["original_life_sources_sha256"]
        ):
            refuse("frozen life list")
        mapping = {r["life_id"]: r for r in records}
        exclusions = frozenset(inventory["training_exclusions"])
        validate_bindings(mapping, exclusions)
    except Refused:
        raise
    except Exception:
        refuse("frozen inventory encoding")
    budget = Budget()
    try:
        with open_source(Path(SOURCE_PATH)) as stream:
            before = os.fstat(stream.fileno())
            if before.st_size != NPZ_SOURCE_BYTES:
                refuse("source size mismatch")
            digest = hashlib.sha256()
            for _ in range((NPZ_SOURCE_BYTES + COPY_CHUNK - 1) // COPY_CHUNK):
                budget.check()
                digest.update(stream.read(COPY_CHUNK))
            if digest.hexdigest() != SOURCE_SHA:
                refuse("source hash mismatch")
            stream.seek(0)
            result = _project(stream, mapping, exclusions, budget)
            after = os.fstat(stream.fileno())
            if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                refuse("source changed during projection")
    except Refused:
        raise
    except Exception:
        refuse("accepted source access")
    result["source_binding"] = {"path": SOURCE_PATH, "sha256": SOURCE_SHA, "size": NPZ_SOURCE_BYTES}
    result["accepted_review_pins"] = pins
    result["source_metadata_inventory_sha256"] = pins["inventory"]
    return result


def main(argv: list[str] | None = None) -> int:
    """This submission exposes no real-source CLI before independent acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    parser.error("refused: real-input scan is disabled; independent acceptance must precede access")


if __name__ == "__main__":
    raise SystemExit(main())
