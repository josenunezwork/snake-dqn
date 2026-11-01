"""Mac-only host accounting adapter for Watch6 V3 shard-00 never-allocated recovery.

This package lives strictly outside ``research.watch6_search_efficacy_v3.protocol.source_paths()``
and never enters the runtime import/source closure. Runtime jobs execute unchanged commit
``357e3504ab7f4b1e5c78209eee14391d16234c7a`` from the real V3 broker root
``/Users/josenunez/Projects/ml/snake-dqn-argon-v3-confirm``, whose actual bytes compute
``protocol.identity() == b26f7fa03927509342fc1adbbbd1d1878eb3511aab4f6698d103aa010a0e1420``.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import marshal
import math
import os
import pickletools
import re
import subprocess
import sys
import types
from collections.abc import Iterator
from pathlib import Path

from research.watch6_search_efficacy_v3 import protocol as p
from tools.ops.package_spec import digest

BASELINE_COMMIT = "357e3504ab7f4b1e5c78209eee14391d16234c7a"
BASELINE_SCIENCE = "b26f7fa03927509342fc1adbbbd1d1878eb3511aab4f6698d103aa010a0e1420"
BASELINE_NAMESPACE = "t34-watch6-exact-stop-efficacy-20261031-v3"
REGISTRATION_V3_SHA256 = "d9441b1d66285c570a3b9a67e9db87f60d4921a268579f65889612feb722ae55"
PROTOCOL_V3_SHA256 = "c2ec1781ce0f99f704214fc6951ddb6891b9dd650da8f80d5b2d24d1aa6422c4"

V3_BROKER_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-argon-v3-confirm")
MAIN_CHECKOUT_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn")

RULING_2_REL = "t34-sol/watch6-v3-ruling-2/RULING_V3.md"
RULING_2_SHA256 = "36a51041e86d9d55b1694f84a09cb0b444d5b7f9a38b4cf60bf9dddf7316ed51"

RULING_3_REL = "t34-sol/watch6-v3-ruling-3/RULING_V3.md"
RULING_3_SHA256 = "fc160389ccdbdb048a57aa436a2d7d93010982feef7a68bf364d8953b5b50176"

SHARD00_PATH_RULING_REL = "t34-sol/watch6-v3-shard00-ruling/RULING.md"
SHARD00_PATH_RULING_SHA256 = "80bb687aeb801fe5659f9bba9dcf185dd4dac136090b7845979871ef749c0308"

SHARD00_PROOF_REL = "ops/unallocated-watch6-v3-fix3-shard-00-proof-01.json"
SHARD00_PROOF_DISPATCH_REL = "ops/dispatch/watch6-v3-fix3-shard-00/unallocated-proof-01.json"
SHARD00_PROOF_SHA256 = "cd4ee9b5f654ad698ac997314d154484720b545109956d6db4197b0be9f1fc46"

SHARD00_RECEIPT_REL = (
    "t34-sol/argon-v3-confirm-fix3/package-v3-fix3-01/output-shard-00/receipt.json"
)
SHARD00_RECEIPT_SHA256 = "2dbb41da38b386fd5289b0ef6bec4b873a81afc2d479cacdff9fa9ff9ebc80ad"
SHARD00_EXEC_MANIFEST_SHA256 = "c83b0acc19b06b75e11a6c9d104aeb5516d40bf46385606939b2bf5cd1cb818b"
SHARD00_PENDING_SHA256 = "09d48c3de15c900834cae050bc8193d5a86d6e7dad2a3daa77f92e7bb0944c31"
SHARD00_NOTEBOOK_HASH = "f04a62f8d103480f8420adb74f9a34a2"

RETAINED_PENDING_CLEANUP_ERROR = "ALLOCATION_UNRESOLVED: retained pending record"
ADAPTER_PACKAGE_REL = "tools/watch6_v3_accounting"
ADAPTER_ROOT = Path(__file__).resolve().parents[2]

ADAPTER_FILES: tuple[str, ...] = (
    "tools/watch6_v3_accounting/__init__.py",
    "tools/watch6_v3_accounting/admission.py",
    "tools/watch6_v3_accounting/dispatch.py",
    "tools/watch6_v3_accounting/final_look.py",
    "tools/watch6_v3_accounting/prepare.py",
    "tools/watch6_v3_accounting/proof.py",
    "tools/watch6_v3_accounting/ratify.py",
    "tools/watch6_v3_accounting/recover.py",
    "tools/watch6_v3_accounting/resume.py",
    "tests/test_watch6_v3_accounting.py",
)


def artifacts_root(repo: Path | None = None) -> Path:
    root = (repo or ADAPTER_ROOT).resolve()
    return (root.parent / "snake-dqn-artifacts").resolve()


def default_artifact_dir(repo: Path | None = None) -> Path:
    return artifacts_root(repo) / "t34-sol/argon-v3-accounting-fix3"


def default_amendment_path(repo: Path | None = None) -> Path:
    return default_artifact_dir(repo) / "WATCH6_V3_ACCOUNTING_AMENDMENT_V4_20261031.md"


def default_ruling_json_path(repo: Path | None = None) -> Path:
    return default_artifact_dir(repo) / "astra-accounting-ruling.json"


def is_forbidden_main_checkout(target: Path) -> bool:
    """True iff target resolves to the main checkout (/Users/josenunez/Projects/ml/snake-dqn)."""
    resolved = target.resolve()
    forbidden = {MAIN_CHECKOUT_ROOT.resolve(), (ADAPTER_ROOT.parent / "snake-dqn").resolve()}
    return resolved in forbidden


def adapter_files_map(repo: Path | None = None) -> dict[str, str]:
    """Return exact SHA256 map of the complete 10-file adapter inventory and refuse extras."""
    root = (repo or ADAPTER_ROOT).resolve()
    pkg_dir = root / ADAPTER_PACKAGE_REL
    if not pkg_dir.is_dir() or pkg_dir.is_symlink():
        raise ValueError(f"missing adapter package directory: {pkg_dir}")

    expected_set = set(ADAPTER_FILES)
    for item in sorted(pkg_dir.rglob("*")):
        if "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        if item.is_symlink():
            raise ValueError(f"symlink forbidden inside adapter package: {item}")
        if item.is_file():
            rel = str(item.relative_to(root))
            if rel not in expected_set:
                raise ValueError(f"unexpected file in adapter package directory: {rel}")

    inventory: dict[str, str] = {}
    for rel in sorted(ADAPTER_FILES):
        target = root / rel
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"missing or unsafe adapter file: {target}")
        inventory[rel] = digest(target)
    return inventory


def adapter_inventory_sha256(repo: Path | None = None) -> str:
    """Canonical SHA256 digest binding the complete 10-file adapter inventory."""
    payload = json.dumps(adapter_files_map(repo), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def adapter_commit(
    repo: Path | None = None,
    *,
    require_clean: bool = True,
    expected_commit: str | None = None,
) -> str:
    """Resolve/verify adapter git commit and require all 10 adapter files to be clean & bound."""
    root = (repo or ADAPTER_ROOT).resolve()
    head = subprocess.check_output(
        ["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=root, text=True, timeout=10
    ).strip()
    if len(head) != 40 or any(ch not in "0123456789abcdef" for ch in head):
        raise ValueError(f"invalid adapter HEAD commit: {head!r}")

    target_commit = expected_commit if expected_commit is not None else head
    if len(target_commit) != 40 or any(ch not in "0123456789abcdef" for ch in target_commit):
        raise ValueError(f"invalid expected adapter commit: {target_commit!r}")

    if expected_commit is not None:
        cat = subprocess.run(
            ["git", "--no-optional-locks", "cat-file", "-t", target_commit],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if cat.returncode != 0 or cat.stdout.strip() != "commit":
            raise ValueError(f"REFUSE: bound adapter commit {target_commit} absent in {root}")
        anc = subprocess.run(
            ["git", "--no-optional-locks", "merge-base", "--is-ancestor", target_commit, head],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if anc.returncode != 0:
            raise ValueError(
                f"REFUSE: bound adapter commit {target_commit} is not an ancestor of HEAD {head}"
            )

    if require_clean:
        status = subprocess.check_output(
            [
                "git",
                "--no-optional-locks",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                ADAPTER_PACKAGE_REL,
                *ADAPTER_FILES,
            ],
            cwd=root,
            text=True,
            timeout=10,
        ).strip()
        if status:
            raise ValueError(f"REFUSE: uncommitted or dirty adapter files in {root}:\n{status}")
        live_map = adapter_files_map(root)
        check_commits = [target_commit] if target_commit == head else [target_commit, head]
        for ref_commit in check_commits:
            for rel, live_sha in live_map.items():
                committed_bytes = subprocess.check_output(
                    ["git", "--no-optional-locks", "show", f"{ref_commit}:{rel}"],
                    cwd=root,
                    timeout=10,
                )
                if hashlib.sha256(committed_bytes).hexdigest() != live_sha:
                    raise ValueError(
                        f"REFUSE: working tree adapter file {rel} differs from {ref_commit}"
                    )
    return target_commit


_PYC_CACHE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.(cpython-\d+)(?:\.opt-([12]))?\.pyc$")
_NUMBA_NBI_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)-(\d+)\.py\d+\.nbi$"
)
_NUMBA_NBC_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*-\d+\.py\d+)\.\d+\.nbc$"
)
_EXTENSION_MODULE_SUFFIXES: tuple[str, ...] = (".so", ".pyd", ".dylib")
_PACKAGING_DIR_SUFFIXES: tuple[str, ...] = (".egg-info", ".dist-info", ".egg")
_IMPORTABLE_HAZARD_SUFFIXES: tuple[str, ...] = (
    ".py",
    ".pyc",
    ".pyo",
    ".so",
    ".pyd",
    ".dylib",
    ".pth",
    ".egg",
)
_SAFE_NUMBA_PICKLE_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("builtins", "object"),
        ("copyreg", "_reconstructor"),
    }
)
_FORBIDDEN_PICKLE_OPCODES: frozenset[str] = frozenset(
    {"INST", "OBJ", "EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"}
)
_CACHED_PROTOCOL_MODULES: dict[tuple[str, str, str], types.ModuleType] = {}
_VERIFIED_NUMBA_STREAM_COUNTS: dict[str, int] = {}
_CACHED_NUMBA_FUNCS_BY_SRC_SHA: dict[bytes, frozenset[tuple[str, int]]] = {}
_VERIFIED_NUMBA_NBI_BY_SHA: dict[str, tuple[frozenset[bytes], tuple[str, ...]]] = {}
_CACHED_BASELINE_TREES: dict[tuple[str, str], tuple[set[str], dict[str, str]]] = {}
_LAST_P_SOURCE_FINGERPRINT: tuple[str, str] | None = None


def _closure_python_files(root: Path, source_paths: list[str]) -> list[Path]:
    py_files: set[Path] = set()
    for rel in source_paths:
        target = root / rel
        if target.is_dir():
            for cand in target.rglob("*.py"):
                if "__pycache__" not in cand.parts and cand.is_file():
                    py_files.add(cand)
        elif target.suffix == ".py" and target.is_file():
            py_files.add(target)
    return sorted(py_files)


def _assert_adapter_excluded_from_closure(root: Path, source_paths: list[str]) -> None:
    for entry in source_paths:
        if "watch6_v3_accounting" in entry or entry.startswith("tools/watch6_v3_accounting"):
            raise ValueError(f"REFUSE: adapter leaked into protocol.source_paths(): {entry}")

    for py_path in _closure_python_files(root, source_paths):
        source_text = py_path.read_text(encoding="utf-8")
        if "watch6_v3_accounting" not in source_text:
            continue
        tree = ast.parse(source_text, filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "watch6_v3_accounting" in alias.name:
                        raise ValueError(
                            f"REFUSE: runtime closure file {py_path} imports {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names = [a.name for a in node.names]
                if "watch6_v3_accounting" in mod or any(
                    "watch6_v3_accounting" in name for name in names
                ):
                    raise ValueError(
                        f"REFUSE: runtime closure file {py_path} imports watch6_v3_accounting"
                    )


def _safe_clear_protocol_caches(*, clear_source_paths: bool = True) -> None:
    if clear_source_paths and hasattr(p.source_paths, "cache_clear"):
        p.source_paths.cache_clear()
    if hasattr(p.identity, "cache_clear"):
        p.identity.cache_clear()


def _py_closure_tree_digest(root: Path) -> str:
    """Compute real-time actual-byte SHA-256 fingerprint of all *.py files read by source_paths."""
    hasher = hashlib.sha256()
    for rel_dir in ("src", "web/backend", "research"):
        base = root / rel_dir
        if base.is_symlink():
            raise ValueError(f"REFUSE: symlinked Python package tree in {root}: {base}")
        if not base.is_dir():
            hasher.update(f"MISSING:{rel_dir}\0".encode("utf-8"))
            continue
        entries: list[tuple[str, Path]] = []
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    full = Path(dirpath) / fn
                    if full.is_symlink():
                        raise ValueError(f"REFUSE: symlinked Python file in {root}: {full}")
                    entries.append((full.relative_to(root).as_posix(), full))
        for rel_posix, full_path in sorted(entries, key=lambda item: item[0]):
            hasher.update(rel_posix.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(digest(full_path).encode("ascii"))
            hasher.update(b"\n")
    return hasher.hexdigest()


def _load_root_protocol_module(root: Path, *, py_tree_sha: str | None = None) -> types.ModuleType:
    """Evaluate unmodified V3 protocol.py in-memory against root's actual bytes (zero .pyc)."""
    for rel_dir in (Path("research"), Path("research/watch6_search_efficacy_v3")):
        cand_dir = root / rel_dir
        if cand_dir.is_symlink() or not cand_dir.is_dir():
            raise ValueError(
                f"REFUSE: missing or symlinked V3 protocol directory in {root}: {cand_dir}"
            )
    proto_path = root / "research/watch6_search_efficacy_v3/protocol.py"
    if proto_path.is_symlink() or not proto_path.is_file():
        raise ValueError(f"REFUSE: missing or symlinked V3 protocol.py in {root}")
    proto_sha = digest(proto_path)
    if proto_sha != PROTOCOL_V3_SHA256:
        raise ValueError(
            f"REFUSE: V3 protocol.py modified in {root} ({proto_sha} != {PROTOCOL_V3_SHA256}); "
            "identity exclusions or edits to protocol.py are forbidden"
        )
    if hasattr(p, "ACCEPTED_PRIOR_SCIENCES"):
        raise ValueError("REFUSE: ACCEPTED_PRIOR_SCIENCES bypass is forbidden by Astra ruling")

    eff_py_sha = py_tree_sha if py_tree_sha is not None else _py_closure_tree_digest(root)
    cache_key = (str(root), proto_sha, eff_py_sha)
    cached = _CACHED_PROTOCOL_MODULES.get(cache_key)
    if cached is not None:
        if hasattr(cached, "ACCEPTED_PRIOR_SCIENCES"):
            raise ValueError("REFUSE: ACCEPTED_PRIOR_SCIENCES bypass is forbidden by Astra ruling")
        if hasattr(cached.identity, "cache_clear"):
            cached.identity.cache_clear()
        return cached

    module = types.ModuleType("_watch6_v3_broker_protocol")
    module.__file__ = str(proto_path)
    code = compile(proto_path.read_bytes(), str(proto_path), "exec")
    exec(code, module.__dict__)
    if Path(module.ROOT).resolve() != root:
        raise ValueError(f"REFUSE: protocol.py ROOT in {root} resolved to {module.ROOT}")
    if hasattr(module, "ACCEPTED_PRIOR_SCIENCES"):
        raise ValueError("REFUSE: ACCEPTED_PRIOR_SCIENCES bypass is forbidden by Astra ruling")
    _CACHED_PROTOCOL_MODULES[cache_key] = module
    return module


def _classify_foreign_closure_file(rel_posix: str) -> str:
    parts = Path(rel_posix).parts
    name = parts[-1] if parts else rel_posix
    lower = name.lower()
    if any(
        any(seg.lower().endswith(sfx) for sfx in _PACKAGING_DIR_SUFFIXES) for seg in parts
    ) or lower.endswith(".pth"):
        return "packaging metadata or path-hook (.pth/.egg-info/.dist-info/.egg)"
    if any(lower.endswith(sfx) for sfx in _EXTENSION_MODULE_SUFFIXES):
        return "native extension module (.so/.pyd/.dylib)"
    if lower.endswith((".py", ".pyc", ".pyo")):
        return "stray Python source or bytecode (.py/.pyc/.pyo)"
    return "untracked or ignored closure file"


def _const_structurally_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, types.CodeType) and isinstance(right, types.CodeType):
        return _code_structurally_equal(left, right)
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _const_structurally_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, frozenset) and isinstance(right, frozenset):
        return len(left) == len(right) and left == right
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left):
        return math.isnan(right)
    return left == right


def _code_structurally_equal(c1: types.CodeType, c2: types.CodeType) -> bool:
    code_attrs = (
        "co_argcount",
        "co_posonlyargcount",
        "co_kwonlyargcount",
        "co_nlocals",
        "co_stacksize",
        "co_flags",
        "co_code",
        "co_names",
        "co_varnames",
        "co_freevars",
        "co_cellvars",
        "co_name",
        "co_qualname",
        "co_firstlineno",
        "co_linetable",
        "co_exceptiontable",
    )
    if any(getattr(c1, attr) != getattr(c2, attr) for attr in code_attrs):
        return False
    return _const_structurally_equal(c1.co_consts, c2.co_consts)


def _verify_pyc_against_tracked_source(
    pyc_path: Path,
    src_py: Path,
    opt_level: int,
    root: Path,
) -> None:
    """Verify PEP-552 header (magic, flags, mtime+size or source_hash) AND compiled CodeType."""
    rel_pyc = pyc_path.relative_to(root).as_posix()
    rel_src = src_py.relative_to(root).as_posix()
    src_bytes = src_py.read_bytes()
    src_stat = src_py.stat()
    pyc_bytes = pyc_path.read_bytes()
    max_allowed = max(1_048_576, 16 * len(src_bytes) + 65_536)
    if len(pyc_bytes) <= 16 or len(pyc_bytes) > max_allowed:
        raise ValueError(f"REFUSE: malformed or oversized bytecode file in {root}: {rel_pyc}")
    if pyc_bytes[:4] != importlib.util.MAGIC_NUMBER:
        raise ValueError(f"REFUSE: foreign or mismatched Python magic number in {root}: {rel_pyc}")

    bit_field = int.from_bytes(pyc_bytes[4:8], "little")
    if bit_field == 0:
        mtime_u32 = int.from_bytes(pyc_bytes[8:12], "little")
        size_u32 = int.from_bytes(pyc_bytes[12:16], "little")
        expected_mtime = int(src_stat.st_mtime) & 0xFFFFFFFF
        expected_size = src_stat.st_size & 0xFFFFFFFF
        if mtime_u32 != expected_mtime or size_u32 != expected_size:
            raise ValueError(
                f"REFUSE: stale or mismatched timestamp .pyc header in {root}: {rel_pyc} "
                f"does not match tracked {rel_src}"
            )
    elif bit_field in (0b01, 0b11):
        expected_hash = importlib.util.source_hash(src_bytes)
        if pyc_bytes[8:16] != expected_hash:
            raise ValueError(
                f"REFUSE: mismatched PEP-552 source_hash in .pyc header in {root}: {rel_pyc} "
                f"does not match tracked {rel_src}"
            )
    else:
        raise ValueError(
            f"REFUSE: unsupported PEP-552 bit_field={bit_field:#x} in {root}: {rel_pyc}"
        )

    try:
        expected_code = compile(
            src_bytes,
            str(src_py),
            "exec",
            dont_inherit=True,
            optimize=opt_level,
        )
        loaded_code = marshal.loads(pyc_bytes[16:])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"REFUSE: corrupt marshalled bytecode in {root}: {rel_pyc}") from exc
    if not isinstance(loaded_code, types.CodeType) or not _code_structurally_equal(
        loaded_code, expected_code
    ):
        raise ValueError(
            f"REFUSE: bytecode body in {root}:{rel_pyc} does not match compiled tracked {rel_src}"
        )


def _inspect_safe_numba_pickle_streams(
    raw: bytes,
    target_path: Path,
    root: Path,
) -> list[list[tuple[str, object]]]:
    """Inspect Numba .nbi/.nbc pickle opcodes safely via pickletools (never unpickling)."""
    rel = target_path.relative_to(root).as_posix()
    if not raw:
        raise ValueError(f"REFUSE: empty Numba cache file in {root}: {rel}")
    buf = io.BytesIO(raw)
    streams: list[list[tuple[str, object]]] = []
    try:
        while buf.tell() < len(raw):
            memo: dict[int, object] = {}
            stack: list[object] = []
            next_memo = 0
            ops: list[tuple[str, object]] = []
            for op, arg, _ in pickletools.genops(buf):
                if op.name in _FORBIDDEN_PICKLE_OPCODES:
                    raise ValueError(
                        f"REFUSE: forbidden pickle opcode {op.name} in Numba cache {root}:{rel}"
                    )
                ops.append((op.name, arg))
                if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                    stack.append(arg)
                elif op.name == "MEMOIZE":
                    if stack:
                        memo[next_memo] = stack[-1]
                    next_memo += 1
                elif op.name in ("BINPUT", "LONG_BINPUT", "PUT"):
                    if stack and arg is not None:
                        memo[int(arg)] = stack[-1]
                elif op.name in ("BINGET", "LONG_BINGET", "GET"):
                    stack.append(memo.get(int(arg)) if arg is not None else None)
                elif op.name == "GLOBAL":
                    parts = str(arg).split(" ", 1)
                    pair = (parts[0], parts[1] if len(parts) > 1 else "")
                    if pair not in _SAFE_NUMBA_PICKLE_GLOBALS and not pair[0].startswith("numba."):
                        raise ValueError(
                            f"REFUSE: foreign pickle global {pair!r} in Numba cache {root}:{rel}"
                        )
                    stack.append(None)
                elif op.name == "STACK_GLOBAL":
                    attr_name = stack.pop() if stack else None
                    mod_name = stack.pop() if stack else None
                    if not isinstance(mod_name, str) or not isinstance(attr_name, str):
                        raise ValueError(
                            f"REFUSE: unresolved STACK_GLOBAL in Numba cache {root}:{rel}"
                        )
                    pair = (mod_name, attr_name)
                    if pair not in _SAFE_NUMBA_PICKLE_GLOBALS and not mod_name.startswith("numba."):
                        raise ValueError(
                            f"REFUSE: foreign pickle global {pair!r} in Numba cache {root}:{rel}"
                        )
                    stack.append(None)
                else:
                    stack.append(None)
            if not ops or ops[-1][0] != "STOP":
                raise ValueError(f"REFUSE: truncated Numba cache stream in {root}:{rel}")
            streams.append(ops)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"REFUSE: corrupt Numba cache stream in {root}:{rel}") from exc
    if buf.tell() != len(raw):
        raise ValueError(f"REFUSE: trailing bytes in Numba cache {root}:{rel}")
    return streams


def _cached_numba_functions_for_source(
    src_bytes: bytes,
    src_sha: bytes,
) -> frozenset[tuple[str, int]]:
    cached = _CACHED_NUMBA_FUNCS_BY_SRC_SHA.get(src_sha)
    if cached is not None:
        return cached
    tree = ast.parse(src_bytes.decode("utf-8"))
    found: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            first_line = min([node.lineno, *(dec.lineno for dec in node.decorator_list)])
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call):
                    for kw in dec.keywords:
                        if (
                            kw.arg == "cache"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                        ):
                            found.add((node.name, first_line))
    result = frozenset(found)
    _CACHED_NUMBA_FUNCS_BY_SRC_SHA[src_sha] = result
    return result


def _verify_pycache_directory(
    root: Path,
    cache_dir: Path,
    tracked_blobs: set[str],
) -> None:
    """Validate <pkg>/__pycache__ entries fail-closed against tracked baseline .py files."""
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise ValueError(f"REFUSE: unsafe __pycache__ directory in {root}: {cache_dir}")
    rel_cache = cache_dir.relative_to(root).as_posix()
    entries = sorted(cache_dir.iterdir(), key=lambda item: item.name)
    if not entries:
        return

    pkg_dir = cache_dir.parent
    if pkg_dir == root:
        raise ValueError(f"REFUSE: top-level __pycache__ files forbidden in {root}: {rel_cache}")
    pkg_rel = pkg_dir.relative_to(root).as_posix()
    if not any(
        tracked.startswith(pkg_rel + "/")
        and "/" not in tracked[len(pkg_rel) + 1 :]
        and tracked.endswith(".py")
        for tracked in tracked_blobs
    ):
        raise ValueError(
            f"REFUSE: __pycache__ in directory without tracked baseline .py sources "
            f"in {root}: {rel_cache}"
        )

    expected_tag = sys.implementation.cache_tag
    nbi_entries: list[tuple[Path, str, str, int]] = []
    nbc_by_name: dict[str, Path] = {}

    for item in entries:
        item_rel = item.relative_to(root).as_posix()
        if item.is_symlink():
            raise ValueError(f"REFUSE: symlink inside __pycache__ forbidden in {root}: {item_rel}")
        if item.is_dir() or not item.is_file():
            raise ValueError(
                f"REFUSE: non-file entry inside __pycache__ forbidden in {root}: {item_rel}"
            )
        pyc_match = _PYC_CACHE_RE.fullmatch(item.name)
        if pyc_match is not None:
            mod_name, cache_tag, opt_raw = pyc_match.groups()
            if cache_tag != expected_tag:
                raise ValueError(
                    f"REFUSE: foreign interpreter cache tag {cache_tag!r} != {expected_tag!r} "
                    f"in {root}: {item_rel}"
                )
            src_rel = f"{pkg_rel}/{mod_name}.py"
            src_py = root / src_rel
            if src_rel not in tracked_blobs or src_py.is_symlink() or not src_py.is_file():
                raise ValueError(
                    f"REFUSE: orphan .pyc without tracked baseline {src_rel} in {root}: {item_rel}"
                )
            opt_level = int(opt_raw) if opt_raw is not None else 0
            _verify_pyc_against_tracked_source(item, src_py, opt_level, root)
            continue

        nbi_match = _NUMBA_NBI_RE.fullmatch(item.name)
        if nbi_match is not None:
            nbi_entries.append(
                (item, nbi_match.group(1), nbi_match.group(2), int(nbi_match.group(3)))
            )
            continue

        nbc_match = _NUMBA_NBC_RE.fullmatch(item.name)
        if nbc_match is not None:
            nbc_by_name[item.name] = item
            continue

        raise ValueError(
            f"REFUSE: foreign file inside __pycache__ ({_classify_foreign_closure_file(item_rel)}) "
            f"in {root}: {item_rel}"
        )

    referenced_nbc: set[str] = set()
    src_funcs_cache: dict[str, frozenset[tuple[str, int]]] = {}
    src_sha_cache: dict[str, bytes] = {}

    for nbi_path, mod_name, func_name, lineno in nbi_entries:
        nbi_rel = nbi_path.relative_to(root).as_posix()
        src_rel = f"{pkg_rel}/{mod_name}.py"
        src_py = root / src_rel
        if src_rel not in tracked_blobs or src_py.is_symlink() or not src_py.is_file():
            raise ValueError(
                f"REFUSE: orphan Numba .nbi without tracked baseline {src_rel} "
                f"in {root}: {nbi_rel}"
            )
        if src_rel not in src_sha_cache:
            raw_src = src_py.read_bytes()
            raw_sha = hashlib.sha256(raw_src).digest()
            src_sha_cache[src_rel] = raw_sha
            src_funcs_cache[src_rel] = _cached_numba_functions_for_source(raw_src, raw_sha)
        if (func_name, lineno) not in src_funcs_cache[src_rel]:
            raise ValueError(
                f"REFUSE: Numba .nbi {nbi_rel} does not match any cache=True function "
                f"{func_name}@{lineno} in tracked {src_rel}"
            )
        nbi_raw = nbi_path.read_bytes()
        nbi_sha = hashlib.sha256(nbi_raw).hexdigest()
        nbi_meta = _VERIFIED_NUMBA_NBI_BY_SHA.get(nbi_sha)
        if nbi_meta is None:
            streams = _inspect_safe_numba_pickle_streams(nbi_raw, nbi_path, root)
            if len(streams) != 2:
                raise ValueError(f"REFUSE: malformed Numba .nbi stream count in {root}: {nbi_rel}")
            bytes_args = frozenset(
                arg for _, arg in streams[1] if isinstance(arg, (bytes, bytearray))
            )
            nbc_args = tuple(
                arg for _, arg in streams[1] if isinstance(arg, str) and arg.endswith(".nbc")
            )
            nbi_meta = (bytes_args, nbc_args)
            _VERIFIED_NUMBA_NBI_BY_SHA[nbi_sha] = nbi_meta
        bytes_stamps, nbc_refs = nbi_meta
        expected_stamp = src_sha_cache[src_rel]
        if expected_stamp not in bytes_stamps:
            raise ValueError(
                f"REFUSE: stale or mismatched SHA-256 source stamp in Numba .nbi "
                f"in {root}: {nbi_rel}"
            )
        nbi_stem = nbi_path.name[: -len(".nbi")]
        for arg in nbc_refs:
            if not arg.startswith(nbi_stem + ".") or _NUMBA_NBC_RE.fullmatch(arg) is None:
                raise ValueError(
                    f"REFUSE: cross-function Numba .nbc reference {arg!r} in {root}: {nbi_rel}"
                )
            referenced_nbc.add(arg)

    orphan_nbc = sorted(set(nbc_by_name) - referenced_nbc)
    if orphan_nbc:
        raise ValueError(
            f"REFUSE: orphan Numba .nbc file(s) not referenced by verified .nbi "
            f"in {root}:{rel_cache}: {orphan_nbc}"
        )
    missing_nbc = sorted(referenced_nbc - set(nbc_by_name))
    if missing_nbc:
        raise ValueError(
            f"REFUSE: missing Numba .nbc file(s) referenced by .nbi "
            f"in {root}:{rel_cache}: {missing_nbc}"
        )
    for nbc_name in sorted(nbc_by_name):
        nbc_path = nbc_by_name[nbc_name]
        nbc_raw = nbc_path.read_bytes()
        nbc_sha = hashlib.sha256(nbc_raw).hexdigest()
        stream_count = _VERIFIED_NUMBA_STREAM_COUNTS.get(nbc_sha)
        if stream_count is None:
            stream_count = len(_inspect_safe_numba_pickle_streams(nbc_raw, nbc_path, root))
            _VERIFIED_NUMBA_STREAM_COUNTS[nbc_sha] = stream_count
        if stream_count != 1:
            nbc_rel = nbc_path.relative_to(root).as_posix()
            raise ValueError(f"REFUSE: malformed Numba .nbc stream count in {root}: {nbc_rel}")


def _closure_scan_targets(root: Path, source_paths: list[str]) -> tuple[list[str], list[str]]:
    """Return (recursive_closure_dirs, tracked_top_files) for the full import/source closure."""
    recursive_dirs: set[str] = {
        "src",
        "web",
        "configs",
        "research",
        "tools",
        "docs/research/final_run_20261019",
    }
    top_files: set[str] = set()
    for entry in source_paths:
        clean = entry.strip("/")
        if not clean:
            continue
        candidate = root / clean
        if candidate.is_symlink():
            raise ValueError(f"REFUSE: symlinked V3 closure entry in {root}: {candidate}")
        if candidate.is_dir():
            recursive_dirs.add(clean)
        else:
            parent = Path(clean).parent.as_posix()
            if parent and parent != ".":
                recursive_dirs.add(parent)
            else:
                top_files.add(clean)

    pruned_dirs: list[str] = []
    for candidate_dir in sorted(recursive_dirs):
        if any(
            candidate_dir == existing or candidate_dir.startswith(existing + "/")
            for existing in pruned_dirs
        ):
            continue
        pruned_dirs.append(candidate_dir)
    return pruned_dirs, sorted(top_files)


def _baseline_tracked_entries(
    root: Path,
    baseline_commit: str,
    source_paths: list[str],
) -> tuple[set[str], dict[str, str]]:
    """Return (tracked_regular_blobs, tracked_symlink_git_oid_by_path) at baseline_commit."""
    cache_key = (str(root), baseline_commit)
    cached = _CACHED_BASELINE_TREES.get(cache_key)
    if cached is None:
        raw = subprocess.check_output(
            ["git", "--no-optional-locks", "ls-tree", "-r", "-z", baseline_commit],
            cwd=root,
            timeout=15,
        )
        tracked_blobs: set[str] = set()
        tracked_symlinks: dict[str, str] = {}
        for record in raw.split(b"\0"):
            if not record:
                continue
            header, rel_bytes = record.split(b"\t", 1)
            mode, obj_type, oid = header.decode("ascii").split(" ")
            rel_posix = rel_bytes.decode("utf-8")
            if mode == "120000" and obj_type == "blob":
                tracked_symlinks[rel_posix] = oid
                continue
            if obj_type != "blob":
                if rel_posix.startswith(("src/", "web/", "configs/", "research/", "tools/")):
                    raise ValueError(f"REFUSE: non-blob mode {mode} in baseline tree: {rel_posix}")
                continue
            tracked_blobs.add(rel_posix)
        cached = (tracked_blobs, tracked_symlinks)
        _CACHED_BASELINE_TREES[cache_key] = cached

    tracked_blobs, tracked_symlinks = cached
    direct_source_dirs = tuple(
        entry.rstrip("/") + "/" for entry in source_paths if (root / entry).is_dir()
    )
    direct_source_files = set(source_paths)
    for rel_posix in tracked_symlinks:
        if rel_posix in direct_source_files or rel_posix.startswith(direct_source_dirs):
            raise ValueError(
                f"REFUSE: symlink mode 120000 inside direct V3 source_paths: {rel_posix}"
            )
    return tracked_blobs, tracked_symlinks


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _git_untracked_and_ignored_paths(root: Path, scan_targets: list[str]) -> set[str]:
    """Enumerate untracked AND git-ignored paths via git status --ignored and git ls-files."""
    found: set[str] = set()
    status_raw = subprocess.check_output(
        [
            "git",
            "--no-optional-locks",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignored=matching",
            "--",
            *scan_targets,
        ],
        cwd=root,
        timeout=15,
    )
    tokens = status_raw.split(b"\0")
    idx = 0
    while idx < len(tokens):
        rec = tokens[idx]
        idx += 1
        if not rec:
            continue
        xy = rec[:2].decode("ascii", errors="replace")
        rel_path = rec[3:].decode("utf-8")
        if "R" in xy or "C" in xy:
            idx += 1
        if xy in ("??", "!!"):
            found.add(rel_path)
        else:
            raise ValueError(
                f"REFUSE: dirty tracked file ({xy}) inside V3 closure in {root}: {rel_path}"
            )

    for extra_args in (
        ["--others"],
        ["--others", "--ignored", "--exclude-standard"],
    ):
        ls_raw = subprocess.check_output(
            [
                "git",
                "--no-optional-locks",
                "ls-files",
                "-z",
                *extra_args,
                "--",
                *scan_targets,
            ],
            cwd=root,
            timeout=15,
        )
        for item in ls_raw.split(b"\0"):
            if item:
                found.add(item.decode("utf-8"))
    return found


def _verify_clean_tracked_closure(
    root: Path,
    source_paths: list[str],
    baseline_commit: str,
) -> None:
    """Enumerate ALL closure files (including git-ignored) via Git + fs walk and refuse foreigns."""
    tracked_blobs, tracked_symlinks = _baseline_tracked_entries(root, baseline_commit, source_paths)
    recursive_dirs, top_files = _closure_scan_targets(root, source_paths)

    for req_dir in (
        "src",
        "web/backend",
        "configs",
        "research/watch6_search_efficacy_v3",
        "tools/ops",
    ):
        target_dir = root / req_dir
        if target_dir.is_symlink() or not target_dir.is_dir():
            raise ValueError(
                f"REFUSE: missing or symlinked required V3 closure directory in {root}: {req_dir}"
            )

    for sp_entry in source_paths:
        sp_target = root / sp_entry
        if sp_target.is_symlink() or not sp_target.exists():
            raise ValueError(f"REFUSE: missing or symlinked V3 source_paths entry: {sp_target}")

    scan_targets = [*recursive_dirs, *top_files]
    git_untracked = _git_untracked_and_ignored_paths(root, scan_targets)

    # Top-level root (sys.path[0]) filesystem audit:
    git_marker = root / ".git"
    if git_marker.is_symlink():
        raise ValueError(f"REFUSE: symlinked .git forbidden in {root}")
    tracked_top_dirs = {rel.split("/", 1)[0] for rel in tracked_blobs if "/" in rel}
    recursive_top_set = {d.split("/", 1)[0] for d in recursive_dirs}
    modified_top_candidates: list[str] = []

    for top_entry in sorted(root.iterdir(), key=lambda item: item.name):
        if top_entry.name == ".git":
            continue
        top_rel = top_entry.name
        if top_entry.is_symlink():
            raise ValueError(f"REFUSE: symlinked top-level entry forbidden in {root}: {top_rel}")
        if any(top_rel.lower().endswith(sfx) for sfx in _PACKAGING_DIR_SUFFIXES):
            raise ValueError(f"REFUSE: foreign top-level packaging metadata ({top_rel}) in {root}")
        if top_entry.is_file():
            if top_rel not in tracked_blobs:
                kind = _classify_foreign_closure_file(top_rel)
                raise ValueError(
                    f"REFUSE: foreign top-level {kind} not tracked at {baseline_commit} "
                    f"in {root}: {top_rel}"
                )
            modified_top_candidates.append(top_rel)
            continue
        if not top_entry.is_dir():
            raise ValueError(f"REFUSE: irregular top-level filesystem entry in {root}: {top_rel}")
        if top_rel == "__pycache__":
            _verify_pycache_directory(root, top_entry, tracked_blobs)
            continue
        if top_rel in recursive_top_set or top_rel in tracked_top_dirs:
            continue

        # Untracked top-level directory (e.g. .pytest_cache or empty saved_snakes on live broker):
        is_hidden_dot_dir = top_rel.startswith(".")
        for dirpath, dirnames, filenames in os.walk(top_entry, followlinks=False):
            dp = Path(dirpath)
            for dn in sorted(dirnames):
                child_dir = dp / dn
                child_rel = child_dir.relative_to(root).as_posix()
                if child_dir.is_symlink():
                    raise ValueError(f"REFUSE: symlinked directory in {root}: {child_rel}")
                if dn == "__pycache__" or any(
                    dn.lower().endswith(sfx) for sfx in _PACKAGING_DIR_SUFFIXES
                ):
                    raise ValueError(
                        f"REFUSE: forbidden cache/packaging directory inside untracked "
                        f"top-level tree in {root}: {child_rel}"
                    )
            for fn in sorted(filenames):
                child_file = dp / fn
                child_rel = child_file.relative_to(root).as_posix()
                if child_file.is_symlink() or not child_file.is_file():
                    raise ValueError(f"REFUSE: unsafe file in {root}: {child_rel}")
                if not is_hidden_dot_dir or any(
                    fn.lower().endswith(sfx) for sfx in _IMPORTABLE_HAZARD_SUFFIXES
                ):
                    kind = _classify_foreign_closure_file(child_rel)
                    raise ValueError(
                        f"REFUSE: foreign {kind} inside untracked top-level directory "
                        f"in {root}: {child_rel}"
                    )

    combined_diff_targets = list(dict.fromkeys([*scan_targets, *modified_top_candidates]))
    diff_bytes = subprocess.check_output(
        ["git", "--no-optional-locks", "diff", baseline_commit, "--", *combined_diff_targets],
        cwd=root,
        timeout=15,
    )
    if diff_bytes:
        raise ValueError(
            f"REFUSE: V3 source closure in {root} differs from baseline commit {baseline_commit}"
        )

    # Full recursive filesystem walk across all closure trees (including git-ignored directories):
    fs_files: set[str] = set()
    pycache_dirs: set[Path] = set()

    for rel_dir in recursive_dirs:
        base_dir = root / rel_dir
        if base_dir.is_symlink():
            raise ValueError(f"REFUSE: symlinked closure directory in {root}: {rel_dir}")
        if not base_dir.exists():
            continue
        if not base_dir.is_dir():
            raise ValueError(f"REFUSE: expected directory for closure tree in {root}: {rel_dir}")

        stack: list[tuple[str, str, bool]] = [
            (str(base_dir), rel_dir, base_dir.name == "__pycache__")
        ]
        while stack:
            abs_dp, rel_dp, in_pycache = stack.pop()
            if in_pycache:
                parent_rel = rel_dp.rsplit("/", 1)[0] if "/" in rel_dp else ""
                if "__pycache__" in parent_rel.split("/"):
                    raise ValueError(f"REFUSE: nested __pycache__ forbidden in {root}: {rel_dp}")
                pycache_dirs.add(Path(abs_dp))

            with os.scandir(abs_dp) as it:
                entries = sorted(it, key=lambda e: e.name)
            next_dirs: list[tuple[str, str, bool]] = []
            for entry in entries:
                name = entry.name
                sub_rel = f"{rel_dp}/{name}"
                if entry.is_symlink():
                    if entry.is_dir(follow_symlinks=True):
                        raise ValueError(
                            f"REFUSE: symlinked directory inside V3 closure in {root}: {sub_rel}"
                        )
                    expected_oid = tracked_symlinks.get(sub_rel)
                    if expected_oid is not None and not in_pycache:
                        link_target_bytes = os.readlink(entry.path).encode("utf-8")
                        if _git_blob_sha1(link_target_bytes) == expected_oid:
                            continue
                    raise ValueError(
                        f"REFUSE: symlinked file inside V3 closure in {root}: {sub_rel}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    if name == ".git":
                        raise ValueError(
                            f"REFUSE: nested .git repository/worktree inside V3 closure "
                            f"in {root}: {sub_rel}"
                        )
                    lower_name = name.lower()
                    if lower_name.endswith(_PACKAGING_DIR_SUFFIXES):
                        raise ValueError(
                            f"REFUSE: foreign packaging metadata directory ({sub_rel}) "
                            f"inside V3 closure in {root}"
                        )
                    if in_pycache:
                        raise ValueError(
                            f"REFUSE: subdirectory inside __pycache__ forbidden "
                            f"in {root}: {sub_rel}"
                        )
                    next_dirs.append((entry.path, sub_rel, name == "__pycache__"))
                else:
                    if name == ".git" or not entry.is_file(follow_symlinks=False):
                        raise ValueError(
                            f"REFUSE: irregular or nested .git file inside V3 closure "
                            f"in {root}: {sub_rel}"
                        )
                    fs_files.add(sub_rel)
            for item in reversed(next_dirs):
                stack.append(item)

    # Expand any directory-level git status --ignored entries (trailing '/') and merge with fs walk:
    combined_candidates: set[str] = set(fs_files)
    for git_rel in sorted(git_untracked):
        clean_git_rel = git_rel.rstrip("/")
        if not clean_git_rel:
            continue
        git_target = root / clean_git_rel
        if git_target.is_symlink():
            raise ValueError(
                f"REFUSE: symlinked untracked/ignored closure entry in {root}: {clean_git_rel}"
            )
        if any(
            seg.lower().endswith(sfx)
            for seg in Path(clean_git_rel).parts
            for sfx in _PACKAGING_DIR_SUFFIXES
        ):
            raise ValueError(
                f"REFUSE: foreign ignored/untracked packaging metadata in {root}: {clean_git_rel}"
            )
        if git_target.is_dir():
            if git_target.name == "__pycache__":
                pycache_dirs.add(git_target)
            for sub in sorted(git_target.rglob("*")):
                if sub.is_symlink():
                    raise ValueError(
                        f"REFUSE: symlink inside ignored directory in {root}: "
                        f"{sub.relative_to(root).as_posix()}"
                    )
                if sub.is_dir() and sub.name == "__pycache__":
                    pycache_dirs.add(sub)
                elif sub.is_file():
                    combined_candidates.add(sub.relative_to(root).as_posix())
        elif git_target.exists():
            combined_candidates.add(clean_git_rel)

    untracked_closure_files = sorted(combined_candidates - tracked_blobs)
    for rel_posix in untracked_closure_files:
        parts = Path(rel_posix).parts
        if len(parts) >= 2 and parts[-2] == "__pycache__" and parts.count("__pycache__") == 1:
            pycache_dirs.add((root / rel_posix).parent)
            continue
        kind = _classify_foreign_closure_file(rel_posix)
        raise ValueError(
            f"REFUSE: foreign {kind} not tracked at baseline commit {baseline_commit} "
            f"in {root}: {rel_posix}"
        )

    for cache_dir in sorted(pycache_dirs):
        _verify_pycache_directory(root, cache_dir, tracked_blobs)


def compute_broker_protocol(broker_root: Path | None = None) -> dict[str, object]:
    """Compute source_paths, registration SHA256, and protocol.identity() from actual bytes."""
    raw = broker_root if broker_root is not None else V3_BROKER_ROOT
    if raw.is_symlink():
        raise ValueError(f"REFUSE: symlinked V3 broker root forbidden: {raw}")
    if not raw.exists() or not raw.is_dir():
        raise ValueError(f"REFUSE: missing V3 broker root directory: {raw}")
    root = raw.resolve()

    py_tree_sha = _py_closure_tree_digest(root)
    module = _load_root_protocol_module(root, py_tree_sha=py_tree_sha)
    source_paths = list(module.source_paths())
    for rel_name in source_paths:
        target = root / rel_name
        if target.is_symlink():
            raise ValueError(f"REFUSE: symlinked V3 closure entry in {root}: {target}")
        if target.is_dir():
            dir_stack = [str(target)]
            while dir_stack:
                cur_dir = dir_stack.pop()
                with os.scandir(cur_dir) as it:
                    for entry in it:
                        if entry.name == "__pycache__":
                            continue
                        if entry.is_symlink():
                            raise ValueError(
                                f"REFUSE: symlinked file inside V3 closure in {root}: {entry.path}"
                            )
                        if entry.is_dir(follow_symlinks=False):
                            dir_stack.append(entry.path)

    reg_path = Path(module.registration())
    if reg_path.is_symlink() or not reg_path.is_file():
        raise ValueError(f"REFUSE: missing or symlinked REGISTRATION_V3.md in {root}")
    reg_sha = digest(reg_path)
    if hasattr(module.identity, "cache_clear"):
        module.identity.cache_clear()
    actual_science = str(module.identity())
    return {
        "broker_root": str(root),
        "namespace": str(module.NAMESPACE),
        "science": actual_science,
        "source_paths": source_paths,
        "allocation": list(module.allocation()),
        "registration_path": str(reg_path),
        "registration_sha256": reg_sha,
        "_py_tree_sha256": py_tree_sha,
    }


def compute_broker_identity(broker_root: Path | None = None) -> str:
    """Compute unmodified V3 protocol.identity() directly from the actual bytes of broker_root."""
    return str(compute_broker_protocol(broker_root)["science"])


def verify_baseline_closure(
    broker_root: Path | None = None,
    *,
    baseline_commit: str = BASELINE_COMMIT,
    require_head_at_baseline: bool = True,
) -> dict[str, str]:
    """Verify actual bytes of broker_root compute protocol.identity() == b26f7fa0... at 357e3504."""
    global _LAST_P_SOURCE_FINGERPRINT
    raw = broker_root if broker_root is not None else V3_BROKER_ROOT
    if raw.is_symlink():
        raise ValueError(f"REFUSE: symlinked V3 broker root forbidden: {raw}")
    if not raw.exists() or not raw.is_dir():
        raise ValueError(f"REFUSE: missing V3 broker root directory: {raw}")
    root = raw.resolve()

    if is_forbidden_main_checkout(root):
        try:
            computed = compute_broker_identity(root)
        except Exception as exc:  # noqa: BLE001
            computed = f"<error:{exc}>"
        raise ValueError(
            f"REFUSE: main checkout {root} is forbidden for V3 baseline/broker verification "
            f"(computed identity={computed}, expected={BASELINE_SCIENCE}); "
            f"bind to {V3_BROKER_ROOT}"
        )

    proto_info = compute_broker_protocol(root)
    source_paths = [str(item) for item in proto_info["source_paths"]]  # type: ignore[union-attr]
    actual_science = str(proto_info["science"])
    reg_sha = str(proto_info["registration_sha256"])
    namespace = str(proto_info["namespace"])
    py_tree_sha = str(proto_info["_py_tree_sha256"])

    if hasattr(p, "ACCEPTED_PRIOR_SCIENCES"):
        raise ValueError("REFUSE: ACCEPTED_PRIOR_SCIENCES bypass is forbidden by Astra ruling")
    if p.NAMESPACE != BASELINE_NAMESPACE or namespace != BASELINE_NAMESPACE:
        raise ValueError(f"REFUSE: unexpected protocol.NAMESPACE in {root}: {namespace!r}")

    if prev_p_root := p.ROOT:
        is_patched_identity = getattr(p.identity, "__wrapped__", None) is None
        if prev_p_root.resolve() == root or is_patched_identity:
            prev_fingerprint = _LAST_P_SOURCE_FINGERPRINT
            target_fingerprint = (str(root), py_tree_sha)
            try:
                p.ROOT = root
                _safe_clear_protocol_caches(
                    clear_source_paths=(prev_fingerprint != target_fingerprint)
                )
                _LAST_P_SOURCE_FINGERPRINT = target_fingerprint
                bound_science = p.identity()
            finally:
                if prev_p_root.resolve() != root:
                    p.ROOT = prev_p_root
                    _safe_clear_protocol_caches(clear_source_paths=True)
                    _LAST_P_SOURCE_FINGERPRINT = None
                else:
                    p.ROOT = prev_p_root
                    _safe_clear_protocol_caches(clear_source_paths=False)
        else:
            bound_science = actual_science
    else:
        bound_science = actual_science

    _assert_adapter_excluded_from_closure(root, source_paths)
    if ADAPTER_ROOT.is_dir() and ADAPTER_ROOT.resolve() != root:
        _assert_adapter_excluded_from_closure(ADAPTER_ROOT.resolve(), source_paths)

    if reg_sha != REGISTRATION_V3_SHA256:
        raise ValueError(
            f"REFUSE: REGISTRATION_V3.md SHA256 rotated in {root} "
            f"({reg_sha} != {REGISTRATION_V3_SHA256})"
        )
    if actual_science != BASELINE_SCIENCE or bound_science != BASELINE_SCIENCE:
        observed = actual_science if actual_science != BASELINE_SCIENCE else bound_science
        raise ValueError(
            f"REFUSE: protocol.identity() rotated in {root} ({observed} != {BASELINE_SCIENCE})"
        )

    if not (root / ".git").exists():
        raise ValueError(f"REFUSE: V3 broker/baseline root {root} is not a git checkout/worktree")

    obj_type = subprocess.run(
        ["git", "--no-optional-locks", "cat-file", "-t", baseline_commit],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if obj_type.returncode != 0 or obj_type.stdout.strip() != "commit":
        raise ValueError(f"REFUSE: baseline commit {baseline_commit} absent in {root}")

    if require_head_at_baseline or root == V3_BROKER_ROOT.resolve():
        head = subprocess.check_output(
            ["git", "--no-optional-locks", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            timeout=10,
        ).strip()
        if head != baseline_commit:
            raise ValueError(
                f"REFUSE: V3 broker root {root} HEAD is {head} != baseline commit {baseline_commit}"
            )

    _verify_clean_tracked_closure(root, source_paths, baseline_commit)

    return {
        "broker_root": str(root),
        "baseline_commit": baseline_commit,
        "science": actual_science,
        "namespace": namespace,
        "registration_sha256": reg_sha,
    }


@contextlib.contextmanager
def bind_broker_protocol(
    broker_root: Path | None = None,
    *,
    baseline_commit: str = BASELINE_COMMIT,
    require_head_at_baseline: bool = True,
) -> Iterator[dict[str, str]]:
    """Verify broker_root from actual bytes and scope V3 protocol.ROOT to that verified broker."""
    global _LAST_P_SOURCE_FINGERPRINT
    closure_info = verify_baseline_closure(
        broker_root,
        baseline_commit=baseline_commit,
        require_head_at_baseline=require_head_at_baseline,
    )
    resolved_broker = Path(closure_info["broker_root"])
    host_proto = Path(p.__file__).resolve()
    if digest(host_proto) != PROTOCOL_V3_SHA256:
        raise ValueError(
            f"REFUSE: host V3 protocol.py modified ({host_proto}); edits/exclusions are forbidden"
        )
    py_tree_sha = _py_closure_tree_digest(resolved_broker)
    target_fp = (str(resolved_broker), py_tree_sha)
    prev_root = p.ROOT
    try:
        p.ROOT = resolved_broker
        _safe_clear_protocol_caches(clear_source_paths=(_LAST_P_SOURCE_FINGERPRINT != target_fp))
        _LAST_P_SOURCE_FINGERPRINT = target_fp
        live_science = p.identity()
        if live_science != closure_info["science"] or live_science != BASELINE_SCIENCE:
            raise ValueError(
                f"REFUSE: bound broker protocol.identity() mismatch in {resolved_broker} "
                f"({live_science} != {BASELINE_SCIENCE})"
            )
        yield closure_info
    finally:
        if prev_root.resolve() != resolved_broker:
            p.ROOT = prev_root
            _safe_clear_protocol_caches(clear_source_paths=True)
            _LAST_P_SOURCE_FINGERPRINT = None
        else:
            p.ROOT = prev_root
            _safe_clear_protocol_caches(clear_source_paths=False)
