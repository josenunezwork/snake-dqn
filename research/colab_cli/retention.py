"""Narrow, source-bound A100 retention admission and immutable setup cache keys.

This module deliberately knows nothing about experiment outcomes.  It only
describes the transport boundary: a chain may reuse an immutable environment
when every link presents the same complete runtime fingerprint.  Scientific
inputs, outputs, credentials and job roots are excluded by construction.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

A100_RATE_UNITS_PER_HOUR = 10.0
PRODUCTION_MAX_MINUTES = 240.0
PRODUCTION_MAX_UNITS = 40.0
ACCEPTANCE_MAX_MINUTES = 288.0
ACCEPTANCE_MAX_UNITS = 48.0
MAX_HANDOFF_MINUTES = 15.0
_HEX = re.compile(r"[0-9a-f]{64}")
_CHAIN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


class RetentionError(ValueError):
    """The narrow retention contract was absent, stale, or ambiguous."""


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_inventory(repo: Path) -> dict[str, str]:
    """Hash every executable transport byte that can affect setup or release."""
    roots = (repo / "research" / "colab_cli", repo / "tools" / "ops")
    return {
        str(path.relative_to(repo)): _hash(path)
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".sh"}
    }


def pinned_tree_sha256(repo: Path, commit: str) -> str:
    """SHA-256 the raw pinned Git tree object, including all config/source bytes."""
    raw = subprocess.check_output(
        ["git", "cat-file", "tree", f"{commit}^{{tree}}"], cwd=repo, timeout=15
    )
    return hashlib.sha256(raw).hexdigest()


def runtime_source_identity(root: Path, repo: Path) -> tuple[str, str]:
    """Verify the generation-bound source proof, including sparse fallback delivery.

    Sparse archives contain only the exact verified selected inventory. Their
    full commit/tree identity is attested by the Mac broker, never reconstructed
    or claimed to have been independently derived from the sparse archive.
    """
    spec = json.loads((root / "spec.json").read_text())
    identity = spec.get("retention_source_identity", {})
    delivery = json.loads((root / "retention-source-delivery.json").read_text())
    if (
        spec.get("generation") != root.name
        or spec.get("root") != str(root)
        or delivery.get("generation") != spec["generation"]
        or not re.fullmatch(r"[0-9a-f]{40}", str(identity.get("commit")))
        or not re.fullmatch(r"[0-9a-f]{40}", str(identity.get("tree")))
        or not _HEX.fullmatch(str(identity.get("tree_sha256")))
        or delivery.get("commit") != identity["commit"]
        or delivery.get("tree") != identity["tree"]
        or delivery.get("tree_sha256") != identity["tree_sha256"]
    ):
        raise RetentionError("retention source identity or generation is absent or changed")
    if (repo / ".git").exists():
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=15
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True, timeout=15
        ).strip()
        if (
            delivery.get("transport") != "github"
            or head != identity["commit"]
            or tree != identity["tree"]
            or pinned_tree_sha256(repo, head) != identity["tree_sha256"]
        ):
            raise RetentionError("runtime Git source differs from Mac verified pin")
        return identity["tree_sha256"], "runtime_git_tree"
    if delivery.get("transport") != "fallback" or delivery.get("inventory_verified") is not True:
        raise RetentionError("sparse source lacks verified fallback inventory evidence")
    return identity["tree_sha256"], "mac_verified_sparse_inventory"


def runtime_cache(manifest: Mapping[str, Any], repo: Path) -> tuple[dict[str, str], str] | None:
    """Validate local immutable pins before they are copied into the remote spec."""
    if manifest.get("requirements", {}).get("gpu") != "A100" or "retention_chain" not in manifest:
        return None
    _shared, cache, _qualification = _chain(manifest)
    if cache["source_sha256"] != pinned_tree_sha256(repo, str(manifest["commit"])):
        raise RetentionError("retention cache source_sha256 differs from the pinned full Git tree")
    requirements = str(manifest.get("env", {}).get("COLAB_CLI_REQUIREMENTS", "requirements.txt"))
    if requirements.startswith("/") or ".." in requirements.split("/"):
        raise RetentionError("retention requirements path is unsafe")
    content = subprocess.check_output(
        ["git", "show", f"{manifest['commit']}:{requirements}"], cwd=repo
    )
    if hashlib.sha256(content).hexdigest() != cache["dependency_lock_sha256"]:
        raise RetentionError("retention dependency lock differs from the pinned requirements bytes")
    canary = manifest.get("canary", {})
    receipt_ref = canary.get("receipt", {}) if isinstance(canary, Mapping) else {}
    receipt_path = receipt_ref.get("path") if isinstance(receipt_ref, Mapping) else None
    receipt_digest = receipt_ref.get("sha256") if isinstance(receipt_ref, Mapping) else None
    if not isinstance(receipt_path, str) or not _HEX.fullmatch(str(receipt_digest)):
        raise RetentionError("retention main requires a hash-bound current-pin native canary")
    receipt = Path(receipt_path)
    if not receipt.is_file() or _hash(receipt) != receipt_digest:
        raise RetentionError("retention native canary receipt is missing or changed")
    canary_pins = json.loads(receipt.read_text()).get("cache_evidence", {}).get("pins")
    if canary_pins != cache:
        raise RetentionError("retention cache pins differ from the accepted native canary")
    return {key: str(value) for key, value in cache.items()}, cache_fingerprint(cache)


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RetentionError(f"retention {field} must be a nonempty string")
    return value


def cache_fingerprint(cache: Mapping[str, Any]) -> str:
    """Return a canonical, inspectable cache key for immutable binaries only."""
    required = (
        "python_abi",
        "torch_cuda",
        "dependency_lock_sha256",
        "installed_distributions_sha256",
        "source_sha256",
        "compiler",
        "compiler_flags",
        "sm",
    )
    if not isinstance(cache, Mapping) or set(cache) != set(required):
        raise RetentionError("retention cache must contain exactly the immutable runtime pins")
    normalized = {key: _require_text(cache[key], f"cache.{key}") for key in required}
    for key in (
        "dependency_lock_sha256",
        "installed_distributions_sha256",
        "source_sha256",
        "compiler",
    ):
        if not _HEX.fullmatch(normalized[key]):
            raise RetentionError(f"retention cache.{key} must be a SHA-256")
    try:
        flags = json.loads(normalized["compiler_flags"])
    except json.JSONDecodeError as exc:
        raise RetentionError("retention cache.compiler_flags must be canonical JSON") from exc
    if (
        not isinstance(flags, dict)
        or set(flags) != {"CXXFLAGS", "TORCH_CUDA_ARCH_LIST"}
        or any(not isinstance(value, str) for value in flags.values())
        or json.dumps(flags, sort_keys=True, separators=(",", ":")) != normalized["compiler_flags"]
    ):
        raise RetentionError(
            "retention cache.compiler_flags must pin CXXFLAGS and TORCH_CUDA_ARCH_LIST"
        )
    if not re.fullmatch(r"[0-9]+\.[0-9]+", normalized["sm"]):
        raise RetentionError("retention cache.sm must be an exact major.minor capability")
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _chain(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    value = manifest.get("retention_chain")
    if not isinstance(value, Mapping) or value.get("enabled") is not True:
        raise RetentionError("A100 retention requires retention_chain.enabled=true")
    shared, cache, qualification = (
        value.get("shared"),
        value.get("cache"),
        value.get("qualification"),
    )
    if (
        not isinstance(shared, dict)
        or not isinstance(cache, dict)
        or not isinstance(qualification, dict)
    ):
        raise RetentionError("retention_chain requires shared, cache, and qualification objects")
    if not _CHAIN.fullmatch(_require_text(shared.get("chain_id"), "shared.chain_id")):
        raise RetentionError("retention chain_id is invalid")
    for key in ("retention_minutes", "retention_units"):
        number = shared.get(key)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or number <= 0:
            raise RetentionError(f"retention shared.{key} must be positive")
    cache_fingerprint(cache)
    return dict(shared), dict(cache), dict(qualification)


def validate_a100_chain(
    manifests: list[Mapping[str, Any]],
    minutes: float,
    units: float,
) -> tuple[dict[str, Any], str]:
    """Validate packet-local A100 constraints before any allocation side effect."""
    if len(manifests) < 2:
        raise RetentionError("A100 retention requires at least two main packets")
    first_shared: dict[str, Any] | None = None
    first_key: str | None = None
    for manifest in manifests:
        requirements, budget = manifest.get("requirements", {}), manifest.get("budget", {})
        probe = manifest.get("canary")
        if requirements.get("gpu") != "A100" or (
            isinstance(probe, Mapping) and probe.get("mode") == "probe"
        ):
            raise RetentionError(
                "A100 retention is main-only; retained canary promotion is forbidden"
            )
        if float(budget.get("rate_units_per_hour", 0)) != A100_RATE_UNITS_PER_HOUR:
            raise RetentionError(
                "A100 retention requires the explicit 10 units/hour admission rate"
            )
        if requirements.get("allowed_gpus", ["A100"]) != ["A100"]:
            raise RetentionError("A100 retention forbids hardware substitution")
        shared, cache, _qualification = _chain(manifest)
        key = cache_fingerprint(cache)
        if first_shared is None:
            first_shared, first_key = shared, key
        elif shared != first_shared or key != first_key:
            raise RetentionError(
                "every retained link must share identical frozen cache and chain pins"
            )
    assert first_shared is not None and first_key is not None
    if minutes != float(first_shared["retention_minutes"]) or units != float(
        first_shared["retention_units"]
    ):
        raise RetentionError("chain arguments differ from frozen retention_chain.shared")
    if minutes > PRODUCTION_MAX_MINUTES or units > PRODUCTION_MAX_UNITS:
        raise RetentionError("production A100 retention exceeds 4 hours / 40 units")
    return first_shared, first_key


def _plan_manifest_hashes(plan: Mapping[str, Any]) -> dict[str, str]:
    rows = plan.get("manifests")
    if not isinstance(rows, list) or not rows:
        raise RetentionError("acceptance plan lacks exact manifest inventory")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RetentionError("acceptance plan manifest row is invalid")
        path, digest = row.get("path"), row.get("sha256")
        if not isinstance(path, str) or not _HEX.fullmatch(str(digest)):
            raise RetentionError("acceptance plan manifest binding is invalid")
        result[path] = str(digest)
    return result


def require_acceptance_plan(
    plan_path: Path, manifests: list[Mapping[str, Any]], minutes: float, units: float, repo: Path
) -> None:
    """Allow the one exact bounded pre-qualification native acceptance package.

    This is deliberately a file-bound package, never an environment flag.  Its
    result still cannot qualify production; only Codex-V's later native PASS can.
    """
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    if plan.get("schema") != "snake-dqn-a100-retention-acceptance-plan/v1":
        raise RetentionError("A100 pre-qualification requires the exact acceptance plan schema")
    if plan.get("source_inventory") != source_inventory(repo):
        raise RetentionError("A100 acceptance plan is not bound to current transport bytes")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if plan.get("commit") != commit:
        raise RetentionError("A100 acceptance plan commit differs from the broker")
    if minutes > ACCEPTANCE_MAX_MINUTES or units > ACCEPTANCE_MAX_UNITS:
        raise RetentionError("A100 acceptance exceeds 4.8 hours / 48 units")
    if plan.get("retention_minutes") != minutes or plan.get("retention_units") != units:
        raise RetentionError("A100 acceptance plan caps differ from invocation")
    final_only = plan.get("final_only_result")
    if not isinstance(final_only, Mapping):
        raise RetentionError("A100 native admission lacks the hash-bound final-only result")
    final_path, final_digest = final_only.get("path"), final_only.get("sha256")
    if not isinstance(final_path, str) or not _HEX.fullmatch(str(final_digest)):
        raise RetentionError("A100 native admission final-only binding is invalid")
    target = Path(final_path)
    if not target.is_file() or _hash(target) != final_digest:
        raise RetentionError("A100 native admission final-only result is missing or changed")
    final_result = json.loads(target.read_text())
    try:
        from research.colab_cli import acceptance, admission, retention_acceptance

        admission.require_final_only(final_result, repo)
        acceptance.verify_plan_sources(final_result["final_plan"], repo)
        retention_acceptance.verify_native_admission(plan_path, manifests)
    except (KeyError, ValueError, RuntimeError) as exc:
        raise RetentionError(
            "A100 native admission requires a source-bound final-only PASS"
        ) from exc
    expected = _plan_manifest_hashes(plan)
    actual = {
        hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for row in manifests
    }
    # Plans bind semantic frozen manifests; their paths need not be rewritten
    # into a live checkout, but every declared hash must occur exactly once.
    if sorted(expected.values()) != sorted(actual):
        raise RetentionError("A100 acceptance manifests differ from the reviewed exact package")


def require_native_qualification(qualification: Mapping[str, Any], repo: Path) -> None:
    """Require an independent, source-bound native A100 retention PASS."""
    path, digest = qualification.get("path"), qualification.get("sha256")
    if not isinstance(path, str) or not _HEX.fullmatch(str(digest)):
        raise RetentionError("A100 retention requires a hash-bound independent qualification")
    record = Path(path)
    if not record.is_absolute() or not record.is_file() or _hash(record) != digest:
        raise RetentionError("A100 retention qualification is missing or changed")
    review = json.loads(record.read_text())
    result_binding = review.get("acceptance_result", {}) if isinstance(review, Mapping) else {}
    if isinstance(result_binding, Mapping):
        result_path, result_digest = result_binding.get("path"), result_binding.get("sha256")
    else:
        result_path, result_digest = None, None
    if not isinstance(result_path, str) or not _HEX.fullmatch(str(result_digest)):
        raise RetentionError("A100 retention review lacks a hash-bound acceptance result")
    result_file = Path(result_path)
    if not result_file.is_file() or _hash(result_file) != result_digest:
        raise RetentionError("A100 retention acceptance result is missing or changed")
    result = json.loads(result_file.read_text())
    if (
        review.get("schema") != "snake-dqn-a100-retention-review/v1"
        or review.get("status") != "PASS"
        or review.get("independent_reviewer") != "Codex-V"
        or review.get("source_inventory") != source_inventory(repo)
        or result.get("schema") != "snake-dqn-a100-retention-acceptance-result/v1"
        or result.get("status") != "PASS"
        or result.get("source_inventory") != source_inventory(repo)
        or result.get("native_a100_two_packet") is not True
    ):
        raise RetentionError("A100 retention lacks an independent source-bound native PASS")
    plan = result.get("plan", {})
    try:
        from research.colab_cli.retention_acceptance import reduce_plan

        regenerated = reduce_plan(Path(plan["path"]))
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        raise RetentionError("A100 retention result cannot be independently regenerated") from exc
    if regenerated != result:
        raise RetentionError("A100 retention result differs from its immutable reducer output")
