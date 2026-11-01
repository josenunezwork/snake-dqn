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
import json
import subprocess
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
    return artifacts_root(repo) / "t34-sol/argon-v3-accounting-fix2"


def default_amendment_path(repo: Path | None = None) -> Path:
    return default_artifact_dir(repo) / "WATCH6_V3_ACCOUNTING_AMENDMENT_V3_20261031.md"


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
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
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


def _safe_clear_protocol_caches() -> None:
    if hasattr(p.source_paths, "cache_clear"):
        p.source_paths.cache_clear()
    if hasattr(p.identity, "cache_clear"):
        p.identity.cache_clear()


def _load_root_protocol_module(root: Path) -> types.ModuleType:
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
    module = types.ModuleType("_watch6_v3_broker_protocol")
    module.__file__ = str(proto_path)
    code = compile(proto_path.read_bytes(), str(proto_path), "exec")
    exec(code, module.__dict__)
    if Path(module.ROOT).resolve() != root:
        raise ValueError(f"REFUSE: protocol.py ROOT in {root} resolved to {module.ROOT}")
    if hasattr(module, "ACCEPTED_PRIOR_SCIENCES") or hasattr(p, "ACCEPTED_PRIOR_SCIENCES"):
        raise ValueError("REFUSE: ACCEPTED_PRIOR_SCIENCES bypass is forbidden by Astra ruling")
    return module


def compute_broker_protocol(broker_root: Path | None = None) -> dict[str, object]:
    """Compute source_paths, registration SHA256, and protocol.identity() from actual bytes."""
    raw = broker_root if broker_root is not None else V3_BROKER_ROOT
    if raw.is_symlink():
        raise ValueError(f"REFUSE: symlinked V3 broker root forbidden: {raw}")
    if not raw.exists() or not raw.is_dir():
        raise ValueError(f"REFUSE: missing V3 broker root directory: {raw}")
    root = raw.resolve()

    module = _load_root_protocol_module(root)
    source_paths = list(module.source_paths())
    for rel_name in source_paths:
        target = root / rel_name
        if target.is_symlink():
            raise ValueError(f"REFUSE: symlinked V3 closure entry in {root}: {target}")
        if target.is_dir():
            for item in target.rglob("*"):
                if "__pycache__" in item.parts:
                    continue
                if item.is_symlink():
                    raise ValueError(f"REFUSE: symlinked file inside V3 closure in {root}: {item}")

    reg_path = Path(module.registration())
    if reg_path.is_symlink() or not reg_path.is_file():
        raise ValueError(f"REFUSE: missing or symlinked REGISTRATION_V3.md in {root}")
    reg_sha = digest(reg_path)
    actual_science = str(module.identity())
    return {
        "broker_root": str(root),
        "namespace": str(module.NAMESPACE),
        "science": actual_science,
        "source_paths": source_paths,
        "allocation": list(module.allocation()),
        "registration_path": str(reg_path),
        "registration_sha256": reg_sha,
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

    prev_p_root = p.ROOT
    try:
        p.ROOT = root
        _safe_clear_protocol_caches()
        if hasattr(p, "ACCEPTED_PRIOR_SCIENCES"):
            raise ValueError("REFUSE: ACCEPTED_PRIOR_SCIENCES bypass is forbidden by Astra ruling")
        if p.NAMESPACE != BASELINE_NAMESPACE or namespace != BASELINE_NAMESPACE:
            raise ValueError(f"REFUSE: unexpected protocol.NAMESPACE in {root}: {namespace!r}")
        bound_science = p.identity()
    finally:
        p.ROOT = prev_p_root
        _safe_clear_protocol_caches()

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

    diff_bytes = subprocess.check_output(
        ["git", "--no-optional-locks", "diff", baseline_commit, "--", *source_paths],
        cwd=root,
        timeout=15,
    )
    if diff_bytes:
        raise ValueError(
            f"REFUSE: V3 source closure in {root} differs from baseline commit {baseline_commit}"
        )

    status_out = subprocess.check_output(
        [
            "git",
            "--no-optional-locks",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *source_paths,
        ],
        cwd=root,
        text=True,
        timeout=15,
    ).strip()
    if status_out:
        raise ValueError(
            f"REFUSE: dirty or untracked files inside V3 source closure in {root}:\n{status_out}"
        )

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
    prev_root = p.ROOT
    try:
        p.ROOT = resolved_broker
        _safe_clear_protocol_caches()
        live_science = p.identity()
        if live_science != closure_info["science"] or live_science != BASELINE_SCIENCE:
            raise ValueError(
                f"REFUSE: bound broker protocol.identity() mismatch in {resolved_broker} "
                f"({live_science} != {BASELINE_SCIENCE})"
            )
        yield closure_info
    finally:
        p.ROOT = prev_root
        _safe_clear_protocol_caches()
