"""Mac-only host accounting adapter for Watch6 V3 shard-00 never-allocated recovery.

This package lives strictly outside ``research.watch6_search_efficacy_v3.protocol.source_paths()``
and never enters the runtime import/source closure. Runtime jobs execute unchanged commit
``357e3504ab7f4b1e5c78209eee14391d16234c7a`` from the real V3 broker root
``/Users/josenunez/Projects/ml/snake-dqn-argon-v3-confirm``, whose actual bytes compute
``protocol.identity() == b26f7fa03927509342fc1adbbbd1d1878eb3511aab4f6698d103aa010a0e1420``.

Fail-closed cache rule (Astra Ruling 4 Points 3 & 4):
- Zero cache whitelist inside the broker closure: every ``__pycache__`` directory (even empty),
  every Numba ``.nbi``/``.nbc`` native cache payload, every CPython ``.pyc``/``.pyo`` bytecode
  cache, and every other untracked or git-ignored file or directory inside the broker closure is
  refused outright without opening, unmarshalling, or unpickling untrusted cache payloads.
- Every Mac adapter entrypoint executes with ``-B``, ``PYTHONDONTWRITEBYTECODE=1``,
  ``PYTHONPYCACHEPREFIX`` outside the broker tree, and ``NUMBA_CACHE_DIR`` set to a fresh, empty,
  verified-empty, single-use adapter-owned directory outside the broker tree (with
  ``NUMBA_CACHE_LOCATOR_CLASSES=UserProvidedCacheLocator``), so no cache is ever written back into
  the broker tree.
"""

from __future__ import annotations

import ast
import atexit
import contextlib
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

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

RULING_VERIFY_4_REL = "t34-sol/watch6-v3-accounting-verify-4/RULING_ACCOUNTING.md"
RULING_VERIFY_4_SHA256 = "16033cb89602260dd9557a02df9626358acd126686b84d658142e7f26367006d"

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


def _forbidden_cache_trees(extra_broker: Path | None = None) -> tuple[Path, ...]:
    candidates = [V3_BROKER_ROOT, ADAPTER_ROOT, MAIN_CHECKOUT_ROOT]
    if extra_broker is not None:
        candidates.append(extra_broker)
    resolved: list[Path] = []
    seen: set[Path] = set()
    for cand in candidates:
        try:
            target = cand.resolve()
        except OSError:
            target = Path(os.path.abspath(str(cand)))
        if target not in seen:
            seen.add(target)
            resolved.append(target)
    return tuple(resolved)


def _is_strictly_outside_trees(candidate: Path, forbidden_trees: tuple[Path, ...]) -> bool:
    resolved = candidate.resolve()
    for tree in forbidden_trees:
        if resolved == tree or resolved.is_relative_to(tree) or tree.is_relative_to(resolved):
            return False
    return True


def _select_external_temp_base(forbidden_trees: tuple[Path, ...]) -> Path:
    raw_candidates = [Path(tempfile.gettempdir()), Path("/private/tmp"), Path("/tmp")]
    for raw in raw_candidates:
        if not raw.exists() or not raw.is_dir():
            continue
        resolved = raw.resolve()
        if any(resolved == tree or resolved.is_relative_to(tree) for tree in forbidden_trees):
            continue
        return resolved
    raise RuntimeError(
        "REFUSE: unable to locate an external temporary base directory outside broker/repo trees"
    )


# Enforce zero in-tree bytecode/Numba cache emission before importing any project/broker module.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
_BOOTSTRAP_FORBIDDEN = _forbidden_cache_trees()
_BOOTSTRAP_ISOLATION_ROOT = Path(
    tempfile.mkdtemp(
        prefix="watch6-v3-acct-bootstrap-",
        dir=str(_select_external_temp_base(_BOOTSTRAP_FORBIDDEN)),
    )
).resolve()
os.chmod(_BOOTSTRAP_ISOLATION_ROOT, 0o700)
_BOOTSTRAP_NUMBA_CACHE_DIR = _BOOTSTRAP_ISOLATION_ROOT / "numba_cache"
_BOOTSTRAP_PYCACHE_DIR = _BOOTSTRAP_ISOLATION_ROOT / "cpython_pycache"
_BOOTSTRAP_NUMBA_CACHE_DIR.mkdir(mode=0o700, exist_ok=False)
_BOOTSTRAP_PYCACHE_DIR.mkdir(mode=0o700, exist_ok=False)
sys.pycache_prefix = str(_BOOTSTRAP_PYCACHE_DIR)
os.environ["PYTHONPYCACHEPREFIX"] = str(_BOOTSTRAP_PYCACHE_DIR)
os.environ["NUMBA_CACHE_DIR"] = str(_BOOTSTRAP_NUMBA_CACHE_DIR)
os.environ["NUMBA_CACHE_LOCATOR_CLASSES"] = "UserProvidedCacheLocator"
atexit.register(shutil.rmtree, _BOOTSTRAP_ISOLATION_ROOT, True)

from research.watch6_search_efficacy_v3 import protocol as p  # noqa: E402
from tools.ops.package_spec import digest  # noqa: E402

_ISOLATION_TLS = threading.local()
_USED_CACHE_LOCK = threading.Lock()
_USED_NUMBA_CACHE_DIRS: set[str] = set()


@dataclass(frozen=True)
class IsolatedBrokerEnv:
    """Fresh, single-use, verified-empty external cache isolation state for one invocation."""

    entrypoint: str
    invocation_id: str
    root_dir: Path
    numba_cache_dir: Path
    pycache_prefix_dir: Path
    bin_dir: Path
    python_wrapper: Path
    real_python: str
    verified_empty_before_use: bool

    def verify_outside(self, broker_root: Path | None = None) -> None:
        forbidden = _forbidden_cache_trees(broker_root)
        for target in (self.root_dir, self.numba_cache_dir, self.pycache_prefix_dir, self.bin_dir):
            if target.is_symlink() or not target.is_dir():
                raise ValueError(f"REFUSE: missing or symlinked isolated cache directory: {target}")
            if not _is_strictly_outside_trees(target, forbidden):
                raise ValueError(
                    f"REFUSE: isolated cache directory {target} overlaps broker/repo tree"
                )

    def subprocess_env(
        self,
        base_env: dict[str, str] | None = None,
        *,
        prepend_wrapper_to_path: bool = True,
    ) -> dict[str, str]:
        env = dict(os.environ if base_env is None else base_env)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPYCACHEPREFIX"] = str(self.pycache_prefix_dir)
        env["NUMBA_CACHE_DIR"] = str(self.numba_cache_dir)
        env["NUMBA_CACHE_LOCATOR_CLASSES"] = "UserProvidedCacheLocator"
        env["SNAKE_DQN_PYTHON"] = str(self.python_wrapper)
        if prepend_wrapper_to_path:
            existing_path = env.get("PATH", "")
            env["PATH"] = (
                f"{self.bin_dir}{os.pathsep}{existing_path}" if existing_path else str(self.bin_dir)
            )
        return env


def active_broker_isolation() -> IsolatedBrokerEnv | None:
    """Return the active per-invocation ``IsolatedBrokerEnv`` if inside an entrypoint context."""
    current = getattr(_ISOLATION_TLS, "active", None)
    return current if isinstance(current, IsolatedBrokerEnv) else None


def used_numba_cache_dirs() -> frozenset[str]:
    """Return all single-use ``NUMBA_CACHE_DIR`` paths allocated during this process lifetime."""
    with _USED_CACHE_LOCK:
        return frozenset(_USED_NUMBA_CACHE_DIRS)


def _sync_loaded_numba_cache_config(*, rebind_dispatchers: bool) -> None:
    """Reload ``numba.core.config`` and rebind already-loaded cached dispatchers if imported."""
    nb_cfg = sys.modules.get("numba.core.config")
    if nb_cfg is not None and hasattr(nb_cfg, "reload_config"):
        nb_cfg.reload_config()
    if not rebind_dispatchers:
        return
    nb_disp_mod = sys.modules.get("numba.core.dispatcher")
    nb_cache_mod = sys.modules.get("numba.core.caching")
    if nb_disp_mod is None or nb_cache_mod is None:
        return
    disp_cls = getattr(nb_disp_mod, "Dispatcher", None)
    null_cache_cls = getattr(nb_cache_mod, "NullCache", None)
    if disp_cls is None or null_cache_cls is None:
        return
    seen_ids: set[int] = set()
    for mod in list(sys.modules.values()):
        mod_dict = getattr(mod, "__dict__", None)
        if not isinstance(mod_dict, dict):
            continue
        for obj in list(mod_dict.values()):
            obj_id = id(obj)
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)
            if isinstance(obj, disp_cls):
                cur_cache = getattr(obj, "_cache", None)
                if cur_cache is not None and not isinstance(cur_cache, null_cache_cls):
                    obj.enable_caching()


@contextlib.contextmanager
def isolated_broker_execution(
    entrypoint: str,
    broker_root: Path | None = None,
) -> Iterator[IsolatedBrokerEnv]:
    """Run an adapter entrypoint with a fresh, verified-empty, single-use ``NUMBA_CACHE_DIR``."""
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    current = active_broker_isolation()
    if current is not None:
        current.verify_outside(broker_root)
        if (
            not sys.dont_write_bytecode
            or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
            or os.environ.get("NUMBA_CACHE_DIR") != str(current.numba_cache_dir)
            or os.environ.get("NUMBA_CACHE_LOCATOR_CLASSES") != "UserProvidedCacheLocator"
        ):
            raise RuntimeError("REFUSE: active broker cache isolation environment was mutated")
        yield current
        return

    forbidden = _forbidden_cache_trees(broker_root)
    temp_base = _select_external_temp_base(forbidden)
    clean_ep = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in entrypoint)[:32]
    inv_id = uuid.uuid4().hex[:12]
    inv_root = Path(
        tempfile.mkdtemp(
            prefix=f"watch6-v3-acct-{clean_ep}-{inv_id}-",
            dir=str(temp_base),
        )
    ).resolve()
    os.chmod(inv_root, 0o700)

    try:
        numba_cache_dir = inv_root / "numba_cache"
        pycache_prefix_dir = inv_root / "cpython_pycache"
        bin_dir = inv_root / "bin"
        numba_cache_dir.mkdir(mode=0o700, exist_ok=False)
        pycache_prefix_dir.mkdir(mode=0o700, exist_ok=False)
        bin_dir.mkdir(mode=0o700, exist_ok=False)

        for created_dir in (inv_root, numba_cache_dir, pycache_prefix_dir, bin_dir):
            if created_dir.is_symlink() or not created_dir.is_dir():
                raise ValueError(f"REFUSE: unsafe isolated cache directory: {created_dir}")
            if not _is_strictly_outside_trees(created_dir, forbidden):
                raise ValueError(
                    f"REFUSE: isolated cache directory {created_dir} lies inside broker/repo tree"
                )

        if any(numba_cache_dir.iterdir()):
            raise ValueError(f"REFUSE: NUMBA_CACHE_DIR not empty before use: {numba_cache_dir}")
        if any(pycache_prefix_dir.iterdir()):
            raise ValueError(
                f"REFUSE: PYTHONPYCACHEPREFIX not empty before use: {pycache_prefix_dir}"
            )

        numba_key = str(numba_cache_dir)
        with _USED_CACHE_LOCK:
            if numba_key in _USED_NUMBA_CACHE_DIRS:
                raise ValueError(f"REFUSE: NUMBA_CACHE_DIR reuse forbidden: {numba_cache_dir}")
            _USED_NUMBA_CACHE_DIRS.add(numba_key)

        real_python = os.path.abspath(sys.executable)
        wrapper_script = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "export PYTHONDONTWRITEBYTECODE=1\n"
            f"export PYTHONPYCACHEPREFIX={shlex.quote(str(pycache_prefix_dir))}\n"
            f"export NUMBA_CACHE_DIR={shlex.quote(str(numba_cache_dir))}\n"
            "export NUMBA_CACHE_LOCATOR_CLASSES=UserProvidedCacheLocator\n"
            f'exec {shlex.quote(real_python)} -B "$@"\n'
        )
        python_wrapper = bin_dir / "python"
        python3_wrapper = bin_dir / "python3"
        for wrap in (python_wrapper, python3_wrapper):
            wrap.write_text(wrapper_script, encoding="utf-8")
            wrap.chmod(0o700)

        iso = IsolatedBrokerEnv(
            entrypoint=entrypoint,
            invocation_id=inv_id,
            root_dir=inv_root,
            numba_cache_dir=numba_cache_dir,
            pycache_prefix_dir=pycache_prefix_dir,
            bin_dir=bin_dir,
            python_wrapper=python_wrapper,
            real_python=real_python,
            verified_empty_before_use=True,
        )
        iso.verify_outside(broker_root)
    except Exception:
        shutil.rmtree(inv_root, ignore_errors=True)
        raise

    tracked_env_keys = (
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPYCACHEPREFIX",
        "NUMBA_CACHE_DIR",
        "NUMBA_CACHE_LOCATOR_CLASSES",
        "SNAKE_DQN_PYTHON",
    )
    saved_env = {key: os.environ.get(key) for key in tracked_env_keys}
    try:
        sys.dont_write_bytecode = True
        sys.pycache_prefix = str(pycache_prefix_dir)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["PYTHONPYCACHEPREFIX"] = str(pycache_prefix_dir)
        os.environ["NUMBA_CACHE_DIR"] = str(numba_cache_dir)
        os.environ["NUMBA_CACHE_LOCATOR_CLASSES"] = "UserProvidedCacheLocator"
        os.environ["SNAKE_DQN_PYTHON"] = str(python_wrapper)
        _sync_loaded_numba_cache_config(rebind_dispatchers=True)
        _ISOLATION_TLS.active = iso
        yield iso
    finally:
        _ISOLATION_TLS.active = None
        sys.dont_write_bytecode = True
        for key, prev_val in saved_env.items():
            if prev_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev_val
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.pycache_prefix = str(_BOOTSTRAP_PYCACHE_DIR)
        restored_nb = os.environ.get("NUMBA_CACHE_DIR")
        need_safe_nb = True
        if restored_nb:
            cand_nb = Path(restored_nb)
            if (
                cand_nb.is_dir()
                and not cand_nb.is_symlink()
                and _is_strictly_outside_trees(cand_nb, forbidden)
            ):
                need_safe_nb = False
        if need_safe_nb:
            os.environ["NUMBA_CACHE_DIR"] = str(_BOOTSTRAP_NUMBA_CACHE_DIR)
            os.environ["NUMBA_CACHE_LOCATOR_CLASSES"] = "UserProvidedCacheLocator"
            _sync_loaded_numba_cache_config(rebind_dispatchers=False)
            if saved_env["NUMBA_CACHE_DIR"] is not None:
                os.environ["NUMBA_CACHE_DIR"] = saved_env["NUMBA_CACHE_DIR"]
            else:
                os.environ["NUMBA_CACHE_DIR"] = str(_BOOTSTRAP_NUMBA_CACHE_DIR)
        else:
            _sync_loaded_numba_cache_config(rebind_dispatchers=False)
        shutil.rmtree(inv_root, ignore_errors=True)


def ensure_cli_dash_b(module_name: str, argv: list[str] | None = None) -> None:
    """Guarantee CLI entrypoints execute under CPython ``-B`` + ``PYTHONDONTWRITEBYTECODE=1``."""
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if sys.flags.dont_write_bytecode:
        return
    if os.environ.get("_WATCH6_V3_ACCT_DASH_B_REEXEC") == "1":
        raise RuntimeError(
            "REFUSE: Python interpreter failed to enable -B (sys.flags.dont_write_bytecode == 0)"
        )
    eff_argv = list(sys.argv[1:] if argv is None else argv)
    child_env = dict(os.environ)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["_WATCH6_V3_ACCT_DASH_B_REEXEC"] = "1"
    child_env["PYTHONPYCACHEPREFIX"] = str(_BOOTSTRAP_PYCACHE_DIR)
    child_env["NUMBA_CACHE_DIR"] = str(_BOOTSTRAP_NUMBA_CACHE_DIR)
    child_env["NUMBA_CACHE_LOCATOR_CLASSES"] = "UserProvidedCacheLocator"
    real_python = os.path.abspath(sys.executable)
    completed = subprocess.run(
        [real_python, "-B", "-m", module_name, *eff_argv],
        env=child_env,
        check=False,
    )
    raise SystemExit(completed.returncode)


def artifacts_root(repo: Path | None = None) -> Path:
    root = (repo or ADAPTER_ROOT).resolve()
    return (root.parent / "snake-dqn-artifacts").resolve()


def default_artifact_dir(repo: Path | None = None) -> Path:
    return artifacts_root(repo) / "t34-sol/argon-v3-accounting-fix4"


def default_amendment_path(repo: Path | None = None) -> Path:
    return default_artifact_dir(repo) / "WATCH6_V3_ACCOUNTING_AMENDMENT_V5_20261031.md"


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


_NUMBA_CACHE_SUFFIXES: tuple[str, ...] = (".nbi", ".nbc")
_CPYTHON_BYTECODE_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo")
_EXTENSION_MODULE_SUFFIXES: tuple[str, ...] = (".so", ".pyd", ".dylib")
_PACKAGING_DIR_SUFFIXES: tuple[str, ...] = (".egg-info", ".dist-info", ".egg")
_IMPORTABLE_HAZARD_SUFFIXES: tuple[str, ...] = (
    ".py",
    ".pyc",
    ".pyo",
    ".nbi",
    ".nbc",
    ".pkl",
    ".pickle",
    ".so",
    ".pyd",
    ".dylib",
    ".pth",
    ".egg",
    ".o",
    ".a",
)
_CACHED_PROTOCOL_MODULES: dict[tuple[str, str, str], types.ModuleType] = {}
_CACHED_BASELINE_TREES: dict[tuple[str, str], tuple[set[str], dict[str, str], set[str]]] = {}
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
    if lower.endswith(_NUMBA_CACHE_SUFFIXES):
        return "Numba native cache payload (.nbi/.nbc)"
    if lower.endswith(_CPYTHON_BYTECODE_SUFFIXES):
        return "CPython bytecode cache / stray Python source or bytecode (.pyc/.pyo)"
    if any(
        any(seg.lower().endswith(sfx) for sfx in _PACKAGING_DIR_SUFFIXES) for seg in parts
    ) or lower.endswith(".pth"):
        return "packaging metadata or path-hook (.pth/.egg-info/.dist-info/.egg)"
    if any(lower.endswith(sfx) for sfx in _EXTENSION_MODULE_SUFFIXES):
        return "native extension module (.so/.pyd/.dylib)"
    if lower.endswith(".py"):
        return "stray Python source or bytecode (.py/.pyc/.pyo)"
    if "__pycache__" in parts or lower.endswith((".pkl", ".pickle", ".npy", ".npz", ".cache")):
        return "non-CPython cache payload"
    return "untracked or ignored closure file"


def _forbid_pycache_directory(root: Path, cache_dir: Path) -> None:
    """Unconditionally refuse any ``__pycache__`` directory or cache payload (never unpickling)."""
    rel_cache = cache_dir.relative_to(root).as_posix()
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise ValueError(
            f"REFUSE: forbidden __pycache__ entry inside V3 closure in {root}: {rel_cache}"
        )
    with os.scandir(cache_dir) as it:
        child_names = sorted(entry.name for entry in it)
    for child_name in child_names:
        if child_name.lower().endswith(_NUMBA_CACHE_SUFFIXES):
            raise ValueError(
                f"REFUSE: forbidden Numba native cache payload (.nbi/.nbc) inside V3 closure "
                f"in {root}: {rel_cache}/{child_name}"
            )
    for child_name in child_names:
        if child_name.lower().endswith(_CPYTHON_BYTECODE_SUFFIXES):
            raise ValueError(
                f"REFUSE: forbidden CPython bytecode cache (.pyc/.pyo) inside V3 closure "
                f"in {root}: {rel_cache}/{child_name}"
            )
    if child_names:
        first_rel = f"{rel_cache}/{child_names[0]}"
        raise ValueError(
            f"REFUSE: forbidden non-CPython cache payload "
            f"({_classify_foreign_closure_file(first_rel)}) inside __pycache__ "
            f"in {root}: {first_rel}"
        )
    raise ValueError(
        f"REFUSE: forbidden __pycache__ directory inside V3 closure in {root}: {rel_cache}"
    )


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
) -> tuple[set[str], dict[str, str], set[str]]:
    """Return (tracked_regular_blobs, tracked_symlinks, tracked_parent_dirs) at baseline_commit."""
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
        tracked_dirs: set[str] = set()
        for record in raw.split(b"\0"):
            if not record:
                continue
            header, rel_bytes = record.split(b"\t", 1)
            mode, obj_type, oid = header.decode("ascii").split(" ")
            rel_posix = rel_bytes.decode("utf-8")
            parts = rel_posix.split("/")
            for idx in range(1, len(parts)):
                tracked_dirs.add("/".join(parts[:idx]))
            if mode == "120000" and obj_type == "blob":
                tracked_symlinks[rel_posix] = oid
                continue
            if obj_type != "blob":
                if rel_posix.startswith(("src/", "web/", "configs/", "research/", "tools/")):
                    raise ValueError(f"REFUSE: non-blob mode {mode} in baseline tree: {rel_posix}")
                continue
            tracked_blobs.add(rel_posix)
        cached = (tracked_blobs, tracked_symlinks, tracked_dirs)
        _CACHED_BASELINE_TREES[cache_key] = cached

    tracked_blobs, tracked_symlinks, tracked_dirs = cached
    direct_source_dirs = tuple(
        entry.rstrip("/") + "/" for entry in source_paths if (root / entry).is_dir()
    )
    direct_source_files = set(source_paths)
    for rel_posix in tracked_symlinks:
        if rel_posix in direct_source_files or rel_posix.startswith(direct_source_dirs):
            raise ValueError(
                f"REFUSE: symlink mode 120000 inside direct V3 source_paths: {rel_posix}"
            )
    return tracked_blobs, tracked_symlinks, tracked_dirs


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
    """Enumerate ALL closure files/dirs (including git-ignored) and refuse any cache or foreign."""
    tracked_blobs, tracked_symlinks, tracked_dirs = _baseline_tracked_entries(
        root, baseline_commit, source_paths
    )
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
        if top_rel == "__pycache__":
            _forbid_pycache_directory(root, top_entry)
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
                if dn == "__pycache__":
                    _forbid_pycache_directory(root, child_dir)
                if any(dn.lower().endswith(sfx) for sfx in _PACKAGING_DIR_SUFFIXES):
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

    # Full recursive filesystem walk across all closure trees (zero cache whitelist):
    fs_files: set[str] = set()

    for rel_dir in recursive_dirs:
        base_dir = root / rel_dir
        if base_dir.is_symlink():
            raise ValueError(f"REFUSE: symlinked closure directory in {root}: {rel_dir}")
        if not base_dir.exists():
            continue
        if not base_dir.is_dir():
            raise ValueError(f"REFUSE: expected directory for closure tree in {root}: {rel_dir}")
        if base_dir.name == "__pycache__":
            _forbid_pycache_directory(root, base_dir)

        stack: list[tuple[str, str]] = [(str(base_dir), rel_dir)]
        while stack:
            abs_dp, rel_dp = stack.pop()
            with os.scandir(abs_dp) as it:
                entries = sorted(it, key=lambda e: e.name)
            next_dirs: list[tuple[str, str]] = []
            for entry in entries:
                name = entry.name
                sub_rel = f"{rel_dp}/{name}"
                if name == "__pycache__":
                    _forbid_pycache_directory(root, Path(entry.path))
                if entry.is_symlink():
                    if entry.is_dir(follow_symlinks=True):
                        raise ValueError(
                            f"REFUSE: symlinked directory inside V3 closure in {root}: {sub_rel}"
                        )
                    expected_oid = tracked_symlinks.get(sub_rel)
                    if expected_oid is not None:
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
                    if sub_rel not in tracked_dirs:
                        # Scan inside untracked dir first so file-specific payload reason surfaces:
                        next_dirs.append((entry.path, sub_rel))
                    else:
                        next_dirs.append((entry.path, sub_rel))
                else:
                    if name == ".git" or not entry.is_file(follow_symlinks=False):
                        raise ValueError(
                            f"REFUSE: irregular or nested .git file inside V3 closure "
                            f"in {root}: {sub_rel}"
                        )
                    fs_files.add(sub_rel)
            if rel_dp not in tracked_dirs and not entries:
                raise ValueError(
                    f"REFUSE: foreign untracked or ignored directory inside V3 closure "
                    f"in {root}: {rel_dp}"
                )
            for item in reversed(next_dirs):
                stack.append(item)

    # Expand any directory-level git status --ignored entries (trailing '/') and merge with fs walk:
    combined_candidates: set[str] = set(fs_files)
    for git_rel in sorted(git_untracked):
        clean_git_rel = git_rel.rstrip("/")
        if not clean_git_rel:
            continue
        git_target = root / clean_git_rel
        if git_target.name == "__pycache__" or "__pycache__" in Path(clean_git_rel).parts:
            parts = Path(clean_git_rel).parts
            pc_idx = parts.index("__pycache__")
            _forbid_pycache_directory(root, root / Path(*parts[: pc_idx + 1]))
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
            for sub in sorted(git_target.rglob("*")):
                if sub.name == "__pycache__":
                    _forbid_pycache_directory(root, sub)
                if sub.is_symlink():
                    raise ValueError(
                        f"REFUSE: symlink inside ignored directory in {root}: "
                        f"{sub.relative_to(root).as_posix()}"
                    )
                if sub.is_file():
                    combined_candidates.add(sub.relative_to(root).as_posix())
            if clean_git_rel not in tracked_dirs and not any(git_target.iterdir()):
                raise ValueError(
                    f"REFUSE: foreign untracked or ignored directory inside V3 closure "
                    f"in {root}: {clean_git_rel}"
                )
        elif git_target.exists():
            combined_candidates.add(clean_git_rel)

    untracked_closure_files = sorted(combined_candidates - tracked_blobs)
    for rel_posix in untracked_closure_files:
        kind = _classify_foreign_closure_file(rel_posix)
        raise ValueError(
            f"REFUSE: foreign {kind} not tracked at baseline commit {baseline_commit} "
            f"in {root}: {rel_posix}"
        )


def compute_broker_protocol(broker_root: Path | None = None) -> dict[str, object]:
    """Compute source_paths, registration SHA256, and protocol.identity() from actual bytes."""
    raw = broker_root if broker_root is not None else V3_BROKER_ROOT
    if raw.is_symlink():
        raise ValueError(f"REFUSE: symlinked V3 broker root forbidden: {raw}")
    if not raw.exists() or not raw.is_dir():
        raise ValueError(f"REFUSE: missing V3 broker root directory: {raw}")
    root = raw.resolve()

    with isolated_broker_execution("compute-broker-protocol", broker_root=root):
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
                            if entry.is_symlink():
                                raise ValueError(
                                    f"REFUSE: symlinked file inside V3 closure in {root}: "
                                    f"{entry.path}"
                                )
                            if entry.is_dir(follow_symlinks=False):
                                if entry.name != "__pycache__":
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

    with isolated_broker_execution("verify-baseline-closure", broker_root=root):
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
        raw_sp = proto_info["source_paths"]
        source_paths = [str(item) for item in raw_sp]  # type: ignore[union-attr]
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
            raise ValueError(
                f"REFUSE: V3 broker/baseline root {root} is not a git checkout/worktree"
            )

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
                    f"REFUSE: V3 broker root {root} HEAD is {head} != baseline commit "
                    f"{baseline_commit}"
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
    raw = broker_root if broker_root is not None else V3_BROKER_ROOT
    with isolated_broker_execution("bind-broker-protocol", broker_root=raw):
        closure_info = verify_baseline_closure(
            raw,
            baseline_commit=baseline_commit,
            require_head_at_baseline=require_head_at_baseline,
        )
        resolved_broker = Path(closure_info["broker_root"])
        host_proto = Path(p.__file__).resolve()
        if digest(host_proto) != PROTOCOL_V3_SHA256:
            raise ValueError(
                f"REFUSE: host V3 protocol.py modified ({host_proto}); "
                "edits/exclusions are forbidden"
            )
        py_tree_sha = _py_closure_tree_digest(resolved_broker)
        target_fp = (str(resolved_broker), py_tree_sha)
        prev_root = p.ROOT
        try:
            p.ROOT = resolved_broker
            _safe_clear_protocol_caches(
                clear_source_paths=(_LAST_P_SOURCE_FINGERPRINT != target_fp)
            )
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
