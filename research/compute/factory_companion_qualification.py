"""Mandatory activation fence for Astra's bounded-concurrency acceptance evidence.

The process-counter and polling implementation is a prepared candidate, not an
escape-proof Darwin descendant tracker. No qualified tracking backend is shipped
here. In particular, neither a successful unit suite, a reviewer verdict, an
environment variable nor a hand-written PASS flag activates production. A future
implementation must supply and qualify that backend before this fence can open.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from research.compute.factory_companion_policy import POLICY_SHA256

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/" "factory-companions/qualification.json"
)
SOURCE_PATHS = (
    "research/bl_labels/mac_driver.py",
    "research/bl_labels/slowdown_guard.py",
    "research/bl_labels/slowdown_resume.py",
    "research/compute/factory_companion.py",
    "research/compute/factory_companion_argv.py",
    "research/compute/factory_companion_commands.py",
    "research/compute/factory_companion_native.py",
    "research/compute/factory_companion_policy.py",
    "research/compute/factory_companion_pytest.py",
    "research/compute/factory_companion_qualification.py",
    "tools/ops/factory_companion.sh",
    "tools/ops/factory_companion_monitor.sh",
    "tools/ops/factory_companion_monitor.py",
)


class QualificationError(ValueError):
    """Production concurrency lacks the required immutable native evidence."""


def source_closure_sha256(root: Path = ROOT) -> str:
    """Bind qualification to the exact operational implementation bytes."""
    digest = hashlib.sha256()
    for name in SOURCE_PATHS:
        digest.update(name.encode() + b"\0")
        digest.update(hashlib.sha256((root / name).read_bytes()).digest())
    return digest.hexdigest()


def qualification_status(path: Path = DEFAULT_EVIDENCE) -> dict[str, Any]:
    """Report evidence gaps without allocating, registering or touching a lease."""
    reasons = [
        "native_descendant_backend_unavailable: polling cannot prove ownership of "
        "a double-fork/setsid descendant born and exited between observations"
    ]
    evidence: dict[str, Any] = {}
    if path.exists():
        try:
            payload = path.read_bytes()
            expected = path.with_name(path.name + ".sha256").read_text().split()[0]
            complete = path.with_name(path.name + ".complete").read_text()
            if hashlib.sha256(payload).hexdigest() != expected or complete != "complete\n":
                raise ValueError("qualification triplet differs")
            evidence = json.loads(payload)
            if not isinstance(evidence, dict):
                raise ValueError("qualification must be an object")
        except (OSError, ValueError, IndexError):
            reasons.append("invalid_immutable_qualification_evidence")
    else:
        reasons.append("missing_immutable_qualification_evidence")
    if evidence.get("concurrency_policy_sha256") != POLICY_SHA256:
        reasons.append("qualification_policy_binding_missing_or_foreign")
    closure = source_closure_sha256()
    if evidence.get("operational_source_closure_sha256") != closure:
        reasons.append("qualification_source_binding_missing_or_foreign")
    acceptance = evidence.get("acceptance", {})
    if not isinstance(acceptance, dict):
        acceptance = {}
    for requirement in range(1, 8):
        item = acceptance.get(str(requirement))
        # Declarations are inspected for useful diagnostics, but are never an
        # authorization mechanism: the native backend fence above stays shut.
        if not isinstance(item, dict) or item.get("verdict") != "PASS":
            reasons.append(f"acceptance_{requirement}_not_qualified")
    return {
        "schema": "factory-companion-qualification-status/v1",
        "verdict": "HOLD",
        "concurrency_policy_sha256": POLICY_SHA256,
        "operational_source_closure_sha256": closure,
        "reasons": reasons,
    }


def require_activation() -> None:
    """Refuse every production entrypoint until the ruling can be enforced exactly."""
    report = qualification_status()
    raise QualificationError("factory concurrency activation HOLD: " + "; ".join(report["reasons"]))
