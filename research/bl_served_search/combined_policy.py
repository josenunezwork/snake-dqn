"""Prospective composition of accepted desktop and supervision replay policies.

The original raw host evidence is never rewritten. Views exist solely to avoid
applying the ordinary 20% rule to independently authenticated supervision, and
to keep authenticated desktop CPU in its original shared background pool.
Live capture remains unavailable with the current direct-child collector.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from research.bl_served_search import background_report as background
from research.bl_served_search import remote_supervision as remote
from research.bl_served_search import supervision_cpu, supervision_identity

POLICY = "classified-desktop-supervised-v2"
SCHEMA = "served-tick-host/v5"
COMPONENTS = [background.DESKTOP_POLICY, remote.POLICY]
QUALIFICATION_SCHEMA = "served-combined-qualification/v1"


def _totals(sample: dict) -> None:
    """Reconcile a validation view; never alter the original CPU evidence."""
    rows = sample["processes"]
    categories = {
        "macos_service_cpu_percent": "macos_service",
        "desktop_renderer_cpu_percent": "desktop_renderer",
        "desktop_app_cpu_percent": "desktop_app",
        "workload_cpu_percent": "agent_workload",
        "unknown_cpu_percent": "unknown",
    }
    for field, kind in categories.items():
        sample[field] = sum(row["cpu_percent"] for row in rows if row["kind"] == kind)
    sample["background_cpu_percent"] = sum(sample[field] for field in list(categories)[:3])


def _points(host: dict) -> list[dict]:
    """Use the very same point inventory retained by the desktop policy."""
    result = []
    for observation in host["observations"]:
        screen = observation["process_screen"]
        result.append(
            {
                "time_ns": screen["observation_ns"],
                "processes": screen["classified_processes"],
            }
        )
    return result


def desktop_view(host: dict, lifecycles: dict) -> dict:
    """Build the desktop guard's validation view after supervision identity proof.

    The caller must first reconcile identity and exact CPU accounting. The raw
    host remains intact; only independently permitted supervision is removed
    from the desktop guard's ordinary workload inventory.
    """
    desktop = deepcopy(host)
    desktop.update(load_policy=background.DESKTOP_POLICY, schema="served-tick-host/v4")
    # Validate the original category identities/totals before deriving a view.
    background._samples(desktop)
    for sample in desktop["cpu_samples"]:
        a, b = sample["start_ns"], sample["end_ns"]
        sample["processes"] = [
            row
            for row in sample["processes"]
            if not remote._permitted_interval(row, a, b, lifecycles)
        ]
        _totals(sample)
    for observation in desktop["observations"]:
        screen = observation["process_screen"]
        screen["classified_processes"] = [
            row
            for row in screen["classified_processes"]
            if not remote._permitted(row, screen["observation_ns"], lifecycles)
        ]
    return desktop


def replay(host: dict[str, Any], *, final: bool = True) -> dict[str, Any]:
    """Compose offline proofs with explicit policy identity and no live claim."""
    try:
        return _replay(host, final=final)
    except (KeyError, TypeError, ValueError, AttributeError, RuntimeError) as exc:
        return {
            "policy": POLICY,
            "valid": False,
            "status": "INVALID",
            "errors": [str(exc)],
            "live_capture_capability": "NOT_ADMISSIBLE",
        }


def _replay(host: dict, *, final: bool) -> dict:
    if host.get("load_policy") != POLICY or host.get("schema") != SCHEMA:
        raise ValueError("combined policy/schema mismatch")
    registry = host["supervisor_registry"]
    registry_check = remote.validate_registry(registry)
    if not registry_check["valid"]:
        raise ValueError("invalid supervision registry: " + str(registry_check["errors"]))
    journal = host["supervision_journal"]
    cpu_evidence = host["supervision_cpu_evidence"]
    identity = supervision_identity.verify_identity(
        registry, journal, host["supervision_identity_observations"], final=final
    )
    cpu = supervision_cpu.replay_cpu(cpu_evidence, final=final)
    errors = [*identity["errors"], *cpu["errors"]]
    if identity["valid"] and cpu["valid"]:
        remote._crosscheck(identity, cpu_evidence, registry, errors)
    if errors:
        raise ValueError("; ".join(errors))
    # Validate every original desktop classification and category total before
    # building a guard view that excludes verified supervision only.
    desktop = desktop_view(host, identity["lifecycles"])
    points = _points(host)
    if host["supervision_point_screens"] != points:
        raise ValueError("combined point inventories differ")
    keys = set()
    for point in points:
        at = point["time_ns"]
        for row in point["processes"]:
            if row.get("kind") in {"desktop_app", "desktop_renderer"}:
                if not remote._permitted(row, at, identity["lifecycles"]):
                    keys.add(remote.digest([at, at, row]))
    for sample in host["cpu_samples"]:
        a, b = sample["start_ns"], sample["end_ns"]
        for row in sample["processes"]:
            permitted = remote._permitted_interval(row, a, b, identity["lifecycles"])
            if row.get("kind") in {"desktop_app", "desktop_renderer"}:
                if not permitted:
                    keys.add(remote.digest([a, b, row]))
    background.validate_desktop_host(desktop)
    ordinary = remote.replay_ordinary(
        points,
        host["cpu_samples"],
        identity["lifecycles"],
        cpu,
        desktop_background=keys,
        desktop_host=desktop,
    )
    errors.extend(ordinary["errors"])
    if final and not cpu["admission_ready"]:
        errors.append("60-second admission stream missing")
    if [(s["start_ns"], s["end_ns"]) for s in host["cpu_samples"]] != [
        (s["start_ns"], s["end_ns"]) for s in cpu["intervals"]
    ]:
        errors.append("combined ordinary/supervision coverage differs")
    return {
        **cpu,
        "policy": POLICY,
        "components": COMPONENTS,
        "valid": not errors,
        "status": "OFFLINE_VALID" if not errors else "INVALID",
        "errors": errors,
        "identity": identity,
        "registry_digest": remote.digest(registry),
        "ordinary_cpu_samples": ordinary["ordinary_cpu_samples"],
        "sampler_reconciliation": ordinary["sampler_reconciliation"],
        "desktop_summary": background.desktop_overall(desktop),
        "coverage_complete": not errors and cpu.get("complete", False),
        "live_capture_capability": "NOT_ADMISSIBLE",
    }


def summarize(
    result: dict, host: dict, proof: dict, *, production: bool = False, fps: int = 24
) -> dict:
    """Keep every tick and both component covariates in a combined offline report."""
    from research.bl_served_search.supervision_report import summarize_supervision

    checked = replay(host)
    if not checked["valid"] or proof != checked:
        raise ValueError("combined summary requires the exact independent replay")
    # The supervision reporter consumes the already reconciled ordinary CPU
    # projection. Its operation exclusion and complete client-gap rules remain
    # unchanged. This internal view cannot admit a historical artifact.
    supervision_host = {**host, "load_policy": remote.POLICY}
    supervision_proof = {**proof, "policy": remote.POLICY}
    report = summarize_supervision(
        result,
        supervision_host,
        supervision_proof,
        production=production,
        fps=fps,
        desktop_host=desktop_view(host, proof["identity"]["lifecycles"]),
    )
    report.update(
        policy=POLICY,
        components=COMPONENTS,
        desktop={
            "inventory": proof["desktop_summary"],
            "interpretation": "Desktop covariates; combined timing mask is authoritative.",
        },
        load_policy=POLICY,
        admission_scope="offline replay only; not live panel admission",
    )
    return report


def replay_host(raw: dict) -> dict:
    """Check combined intent bindings before the deliberately closed live gate.

    Pure replay() is a predicate over supplied evidence, like the accepted
    supervision replay. It cannot establish source-bound panel admission.
    """
    from research.bl_served_search import desktop_policy, supervision_io

    host = raw["host"]
    intent = raw["combined_intent"]
    supervision = raw["supervision_intent"]
    registry = host["supervisor_registry"]
    sources = supervision_io.policy_sources()
    if (
        raw.get("load_policy") != POLICY
        or host.get("load_policy") != POLICY
        or intent.get("schema") != "served-combined-intent/v1"
        or intent.get("load_policy") != POLICY
        or intent.get("components") != COMPONENTS
        or intent.get("source") != raw.get("source_before")
        or raw.get("source_before") != raw.get("source_after")
        or intent.get("manifest_sha256") != remote.digest(raw.get("manifest"))
        or intent.get("desktop_intent_sha256") != remote.digest(raw.get("desktop_intent"))
        or intent.get("supervision_intent_sha256") != remote.digest(supervision)
        or supervision.get("load_policy") != remote.POLICY
        or supervision.get("registry_sha256") != remote.digest(registry)
        or supervision.get("policy_source_sha256") != sources
        or registry.get("policy_source_hash") != remote.digest(sources)
        or registry.get("ruling_sha256") != supervision_io.RULING_SHA256
        or registry.get("ruling_hash") != supervision_io.RULING_SHA256
        or supervision.get("ruling_sha256") != supervision_io.RULING_SHA256
    ):
        raise ValueError("combined intent registry/ruling/policy/source mismatch")
    desktop_policy.validate_component_intent(raw)
    receipt = raw.get("qualification_receipt")
    observation_times = []
    captures = list(host["cpu_samples"])
    captures.extend(o["process_screen"] for o in host["observations"])
    if not host["cpu_samples"] or not host["observations"]:
        raise ValueError("combined intent has empty observation timestamps")
    for capture in captures:
        at = capture.get("capture_start_ns")
        if type(at) is not int or at < 0:
            raise ValueError("combined intent requires capture_start_ns on every observation")
        observation_times.append(at)
    observation_times.extend(s["start_ns"] for s in host["cpu_samples"])
    for observation in host["observations"]:
        screen = observation["process_screen"]
        observation_times.append(screen["observation_ns"])
    if any(type(at) is not int or at < 0 for at in observation_times):
        raise ValueError("combined intent has malformed observation timestamps")
    first_observation = min(observation_times)
    publications = [
        raw["desktop_intent"].get("published_ns"),
        supervision.get("published_ns"),
        intent.get("published_ns"),
        receipt.get("published_ns") if isinstance(receipt, dict) else None,
    ]
    if (
        not isinstance(receipt, dict)
        or not receipt
        or intent.get("qualification_receipt_sha256") != remote.digest(receipt)
        or any(type(at) is not int or not 0 <= at < first_observation for at in publications)
        or any(at > publications[2] for at in (publications[0], publications[1], publications[3]))
    ):
        raise ValueError("combined intent qualification/publication binding differs")
    # A qualification is a completed prior observation, bound to the same
    # source and both sealed registries. Duration comes from the covered clock
    # interval, never a declared nominal number of seconds.
    start, end = receipt.get("observation_start_ns"), receipt.get("observation_end_ns")
    if (
        receipt.get("schema") != QUALIFICATION_SCHEMA
        or receipt.get("load_policy") != POLICY
        or receipt.get("status") != "COMPLETE"
        or receipt.get("verdict") != "VALID"
        or receipt.get("coverage_complete") is not True
        or type(start) is not int
        or type(end) is not int
        or not 0 <= start <= end <= receipt["published_ns"]
        or end - start < 600_000_000_000
        or receipt.get("desktop_registry_sha256") != raw["desktop_intent"]["registry_sha256"]
        or receipt.get("supervisor_registry_sha256") != supervision["registry_sha256"]
        or receipt.get("ruling_sha256") != supervision["ruling_sha256"]
        or receipt.get("source_before") != raw["source_before"]
        or receipt.get("source_after") != raw["source_before"]
    ):
        raise ValueError("combined intent qualification completion/input binding differs")
    checked = replay(host)
    if not checked["valid"] or checked != host.get("supervision_replay"):
        raise ValueError("combined reported accounting differs from independent replay")
    # No invented capability flag or matching synthetic proof can replace the
    # missing live observer. Preserve both accepted policies' admission boundary.
    raise ValueError(supervision_io.BLOCKER)
