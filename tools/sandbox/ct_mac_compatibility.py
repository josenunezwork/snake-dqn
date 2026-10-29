"""Exact Mac admission-only source differences for the live C/T reseal-03.

Source staging still reads the frozen runtime commit/archive. This record only
lets the Mac broker run corrected admission code alongside those runtime bytes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from research.sandbox_a_resident import ct_json as json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "tools/sandbox/ct_mac_compatibility.json"
MAC_ONLY = frozenset(
    {
        "research/sandbox_a_resident/ct_cuda_observed.py",
        "research/sandbox_a_resident/ct_resume.py",
    }
)


def permits_mac_source(
    manifest: dict[str, Any], relative: str, pinned: bytes, current: bytes
) -> bool:
    """Require committed policy, exact old/new hashes and bounded sealed lineage."""
    from tools.sandbox.preflight import _assert_safe_path, verify_committed_manifest

    if relative not in MAC_ONLY:
        return False
    _assert_safe_path(POLICY, "C/T Mac compatibility policy")
    verify_committed_manifest(POLICY)
    policy = json.loads(POLICY.read_bytes())
    if (
        policy.get("schema") != "t34-cf-mac-source-compatibility/v1"
        or manifest.get("commit") != policy.get("runtime_commit")
        or set(policy.get("mac_only_sources", {})) != MAC_ONLY
    ):
        return False
    pair = policy["mac_only_sources"][relative]
    if pair != {
        "runtime_sha256": hashlib.sha256(pinned).hexdigest(),
        "broker_sha256": hashlib.sha256(current).hexdigest(),
    }:
        return False
    prior = manifest
    for _ in range(3):
        comp = prior.get("completion_attempt")
        if comp is None:
            admission = prior.get("sandbox_admission", {})
            sandbox = _assert_safe_path(admission.get("manifest_path", ""), "C/T root sandbox")
            # Reopen the original packet through its bound canonical sandbox.
            payload = json.loads(sandbox.read_bytes())
            bound = payload.get("outputs", {}).get("broker_manifest", {})
            path = _assert_safe_path(bound.get("path", ""), "C/T root manifest")
            raw = path.read_bytes()
            seal = _assert_safe_path(path.parent / "SEAL.json", "C/T root seal")
            return (
                json.loads(raw) == prior
                and hashlib.sha256(raw).hexdigest()
                == bound.get("sha256")
                == policy.get("root_manifest_sha256")
                and hashlib.sha256(seal.read_bytes()).hexdigest() == policy.get("root_seal_sha256")
            )
        if not isinstance(comp, dict) or comp.get("index") not in (1, 2):
            return False
        path = _assert_safe_path(comp.get("prior_manifest_path", ""), "C/T prior manifest")
        if not path.is_absolute() or path.name != "dispatch-manifest.json":
            return False
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != comp.get("prior_manifest_sha256"):
            return False
        prior = json.loads(raw)
    return False
