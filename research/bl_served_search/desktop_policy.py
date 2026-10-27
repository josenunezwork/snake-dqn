"""Prospective desktop intent bindings; no serving imports or live process controls."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from research.bl_paced_identity.events import canonical_bytes
from research.bl_paced_identity.record import read_export

POLICY = "classified-desktop-v1"
RULING_SHA256 = "0210543b502b6d1722ad631d9fc3be740d35c073591b2a02ff157ecc2c57905e"
RULING = Path(__file__).with_name("ASTRA_DESKTOP_RULING_20261027.md")
AMENDMENT = Path(__file__).with_name("DESKTOP_APPS_AMENDMENT_20261027.md")


def digest(value: Any) -> str:
    """Hash canonical finite evidence, identically to immutable export payloads."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def freeze_intent(registry_path: Path, source: dict, manifest: dict) -> dict[str, Any]:
    """Bind the prospective declaration before host observation or measured exposure."""
    from research.bl_served_search.desktop_apps import DesktopRegistry

    enrollment = DesktopRegistry(registry_path).read_export()
    registry = enrollment["registry"]
    ruling_hash = hashlib.sha256(RULING.read_bytes()).hexdigest()
    if ruling_hash != RULING_SHA256:
        raise ValueError("desktop ruling binding differs")
    return {
        "schema": "served-desktop-intent/v1",
        "policy": POLICY,
        "registry": registry,
        "registry_sha256": digest(registry),
        "registry_inputs": enrollment["registry_inputs"],
        "registry_input": {
            "path": str(registry_path.resolve(strict=True)),
            "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "format": "sealed registry extension/base JSON + SHA256 + completion published last",
        },
        "ruling_sha256": ruling_hash,
        "amendment_sha256": hashlib.sha256(AMENDMENT.read_bytes()).hexdigest(),
        "source": source,
        "execution_worktree": str(Path(__file__).resolve().parents[2]),
        "manifest_sha256": digest(manifest),
        "prior_attempts": {"07": "INVALID", "08": "INVALID"},
        "timing_authority": "all measured ticks and client receipts; no CPU subtraction",
    }


def observation_binding(path: Path, intent: Mapping[str, Any]) -> dict[str, Any]:
    """Require a sealed successful host observation of the same code and enrollment."""
    from research.bl_served_search.background_report import validate_desktop_host

    observation = read_export(path)
    previous = observation.get("desktop_intent", {})
    if (
        observation.get("schema") != "served-desktop-host-observation/v1"
        or observation.get("status") != "COMPLETE"
        or observation.get("load_policy") != POLICY
        or observation.get("source_before") != intent["source"]
        or observation.get("source_after") != intent["source"]
        or previous.get("registry_sha256") != intent["registry_sha256"]
        or previous.get("registry") != intent["registry"]
        or previous.get("policy") != POLICY
        or previous.get("amendment_sha256") != intent["amendment_sha256"]
        or previous.get("ruling_sha256") != intent["ruling_sha256"]
        or observation.get("host", {}).get("status") != "COMPLETE"
    ):
        raise ValueError("desktop observation is missing, invalid, or bound to different inputs")
    host = observation["host"]
    if (
        host.get("desktop_registry") != intent["registry"]
        or host.get("desktop_registry_sha256") != intent["registry_sha256"]
        or host.get("desktop_registry_evidence", {}).get("registry_inputs")
        != intent.get("registry_inputs")
    ):
        raise ValueError("desktop observation registry differs")
    validate_desktop_host(host)
    if not any(
        row.get("kind") in {"desktop_app", "desktop_renderer"}
        for sample in host["cpu_samples"]
        for row in sample["processes"]
    ):
        raise ValueError("desktop observation contains no actual admitted desktop app")
    # Check the separately sealed host evidence as well as the observation payload.
    if read_export(path.parent / "host.json") != host:
        raise ValueError("desktop observation host differs from original host export")
    if read_export(path.parent / "intent.json") != previous:
        raise ValueError("desktop observation intent differs from original intent export")
    return {"path": str(path.resolve(strict=True)), "sha256": digest(observation)}


def validate_intent(raw: Mapping[str, Any]) -> None:
    """Reconcile sealed policy/source inputs without re-reading current installed apps."""
    intent = raw.get("desktop_intent", {})
    host = raw.get("host", {})
    source = raw.get("source_before", {})
    if not all(isinstance(value, Mapping) for value in (intent, host, source)):
        raise ValueError("malformed desktop intent/source/host")
    hashes = source.get("source_sha256", {})
    prefix = "research/bl_served_search/"
    registry_input = intent.get("registry_input", {})
    if not isinstance(registry_input, Mapping):
        raise ValueError("malformed desktop registry input binding")
    worktree = intent.get("execution_worktree")
    if (
        intent.get("schema") != "served-desktop-intent/v1"
        or intent.get("policy") != POLICY
        or intent.get("source") != source
        or intent.get("manifest_sha256") != digest(raw.get("manifest"))
        or intent.get("ruling_sha256") != RULING_SHA256
        or hashes.get(prefix + RULING.name) != RULING_SHA256
        or not intent.get("amendment_sha256")
        or hashes.get(prefix + AMENDMENT.name) != intent.get("amendment_sha256")
        or not intent.get("registry")
        or digest(intent["registry"]) != intent.get("registry_sha256")
        or host.get("desktop_registry") != intent["registry"]
        or host.get("desktop_registry_sha256") != intent["registry_sha256"]
        or host.get("desktop_registry_evidence", {}).get("registry_inputs")
        != intent.get("registry_inputs")
        or not isinstance(worktree, str)
        or not worktree.startswith("/")
        or source.get("worktree") != worktree
        or not isinstance(registry_input.get("path"), str)
        or not registry_input["path"].startswith("/")
        or not isinstance(registry_input.get("sha256"), str)
        or len(registry_input["sha256"]) != 64
        or intent.get("prior_attempts") != {"07": "INVALID", "08": "INVALID"}
    ):
        raise ValueError("desktop intent/source/registry binding differs")
    registry_path = Path(registry_input["path"])
    if registry_path.is_relative_to(worktree):
        relative = str(registry_path.relative_to(worktree))
        if hashes.get(relative) != registry_input["sha256"]:
            raise ValueError("desktop registry differs from execution source closure")
    from research.bl_served_search.desktop_apps import DesktopRegistry

    enrollment = DesktopRegistry(registry_path).read_export()
    if (
        enrollment["registry"] != intent["registry"]
        or enrollment["registry_sha256"] != intent["registry_sha256"]
        or enrollment["registry_inputs"] != intent.get("registry_inputs")
        or hashlib.sha256(registry_path.read_bytes()).hexdigest() != registry_input["sha256"]
    ):
        raise ValueError("desktop sealed input chain differs")
    for item in enrollment["registry_inputs"]:
        input_path = registry_path.parent / item["path"]
        if input_path.is_relative_to(worktree):
            relative = str(input_path.relative_to(worktree))
            if hashes.get(relative) != item["sha256"]:
                raise ValueError("desktop base/extension input differs from source closure")
    observed = intent.get("host_observation")
    if (
        not isinstance(observed, Mapping)
        or observation_binding(Path(observed["path"]), intent) != observed
    ):
        raise ValueError("desktop panel lacks its original qualified host observation")
