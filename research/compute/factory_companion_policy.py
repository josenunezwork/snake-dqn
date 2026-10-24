"""Pure prospective companion admission policy; no factory accounting or outcomes.

Times are monotonic within a boot. The monitor owns observation/history fields;
the factory alone publishes protected-period and eligible-window facts. Missing
facts close tests. All comparisons intentionally allow threshold equality.
"""

from __future__ import annotations

import math
from typing import Any

POLICY_SHA256 = "af117cd98d51f428cd83d6e33138fcd13def0b4d7c7cf63a8b2e2b83db99105f"


def number(value: Any, name: str) -> float:
    """Read a finite counter, refusing booleans and unknown observations."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing or invalid companion {name}")
    if not math.isfinite(value):
        raise ValueError(f"missing or invalid companion {name}")
    return float(value)


def check_admission(
    state: dict[str, Any], klass: str, now: float, policy_sha: str, lease_identity: str
) -> None:
    """Refuse before spawn unless every shared operational prerequisite holds."""
    if policy_sha != POLICY_SHA256 or not lease_identity:
        raise ValueError("companion policy or factory identity differs")
    if state.get("admission_closed") or state.get("exclusive_arm"):
        raise ValueError("companion tool admission is closed")
    age = now - number(state.get("heartbeat"), "monitor heartbeat")
    if age < 0 or age > 10 or not state.get("monitor"):
        raise ValueError("companion monitor unavailable")
    if state.get("unmanaged") or state.get("observation_failure"):
        raise ValueError("unmanaged local compute or missing measurement")
    factory = state.get("factory", {})
    if factory.get("identity") != lease_identity or factory.get("policy_sha256") != policy_sha:
        raise ValueError("companion active factory/lease binding differs")
    samples = [
        item
        for item in state.get("samples", [])
        if 0 <= now - number(item.get("at"), "sample") <= 60
    ]
    # Samples carry CPU seconds (preferred), or legacy rates over five seconds.
    light = sum(
        number(s.get("light_seconds", s.get("light_cpu", 0) * 5), "light CPU") for s in samples
    )
    total = sum(
        number(s.get("total_seconds", s.get("total_cpu", 0) * 5), "total CPU") for s in samples
    )
    if (
        light > 15
        or total > 75
        or any(number(s.get("rss", 0), "RSS") > 8 * 1024**3 for s in samples[-1:])
    ):
        raise ValueError("companion trailing resource window exceeds ceiling")
    if klass != "test":
        return
    if now < number(state.get("suspended_until", 0), "test suspension"):
        raise ValueError("test admission suspended")
    if now < number(state.get("history_hold_until", now + 1800), "test history hold"):
        raise ValueError("missing duty history requires 1800 s hold")
    since = number(state.get("light_only_since"), "light observation")
    observed = number(state.get("observed_since", since), "continuous observation")
    if min(now - since, now - observed) < 60:
        raise ValueError("test requires 60 s compliant light-only observation")
    if factory.get("protected") is not False:
        raise ValueError("factory baseline/rearming/W12 sanity protects test lane")
    streams = factory.get("streams")
    if not isinstance(streams, dict) or not streams:
        raise ValueError("factory stream protection is unknown")
    for stream in streams.values():
        if stream.get("baseline_ready") is not True or stream.get("fresh_window_ready") is not True:
            raise ValueError("new stream/baseline/resume rearming protects test lane")
    warning = state.get("warning")
    if warning:
        revoked = number(warning.get("revoked_at"), "warning epoch")
        if now - revoked < 900:
            raise ValueError("early warning test suspension has not elapsed")
        for stream in streams.values():
            if (
                stream.get("window_full") is not True
                or number(stream.get("window_started_at"), "fresh window start") < revoked
                or number(stream.get("median_ratio"), "window median ratio") > 1.15
            ):
                raise ValueError("early warning needs fresh full clean K-life window")
    duty = sum(
        number(item.get("cpu"), "test duty")
        for item in state.get("duty", [])
        if 0 <= now - number(item.get("at"), "test duty timestamp") <= 1800
    )
    reservations = sum(
        90
        for item in state.get("registrations", {}).values()
        if item.get("class") == "test"
        and not (item.get("wait4_reaped") is True and item.get("descendants_gone") is True)
    )
    if duty + reservations + 90 > 300:
        raise ValueError("test rolling CPU duty reserve exceeds 300")


def update_factory_state(
    state: dict[str, Any],
    identity: str,
    protected: bool,
    streams: dict[str, Any],
    now: float,
    owned: list[dict[str, Any]] | None = None,
) -> None:
    """Publish factory-only control facts without modifying its baselines/windows."""
    if not identity or type(protected) is not bool or not isinstance(streams, dict):
        raise ValueError("invalid factory companion control facts")
    state["factory"] = {
        "identity": identity,
        "policy_sha256": POLICY_SHA256,
        "protected": protected,
        "streams": streams,
        "owned": [] if owned is None else owned,
        "updated_at": now,
    }
