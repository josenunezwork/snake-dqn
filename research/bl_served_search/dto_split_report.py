"""Off-clock validation/reduction for the opt-in authoritative-game/DTO arm."""

from __future__ import annotations

from typing import Any, Mapping


def validate_transport(result: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse missing exports, stale replies, overflow or uncharged publication."""
    from research.bl_served_search.dto_split import MAX_EXPORT_BYTES, MAX_WIRE_BYTES
    from research.bl_served_search.served_tick_probe import _distribution

    data = result["dto_transport"]
    measured, warmup = result["measured_ticks"], result["warmup_ticks"]
    ticks, services, trace = data["ticks"], data["services"], result["trace"]
    if (
        data["capacity"] != 1
        or data["max_export_bytes"] != MAX_EXPORT_BYTES
        or data["max_wire_bytes"] != MAX_WIRE_BYTES
        or len(ticks) != measured
        or len(trace) != measured
        or data["busy"] != 0
        or data["crashes"] != 0
        or data["stale"] != 0
        or data["final_diagnostic_publication"] is not False
        or data["completed"] != warmup + measured
        or len(services) != warmup + measured
    ):
        raise ValueError("missing/overflow/fallback DTO exposure")
    for index, (tick, row) in enumerate(zip(ticks, trace)):
        if tick["sequence"] != warmup + index or tick["published"] is not True:
            raise ValueError("publication order/exposure")
        if not 0 < tick["export_bytes"] <= MAX_EXPORT_BYTES:
            raise ValueError("export size")
        for prefix in ("copy", "publish", "poll"):
            if (
                not row["start_ns"]
                <= tick[f"{prefix}_start_ns"]
                <= tick[f"{prefix}_end_ns"]
                <= row["end_ns"]
                or tick[f"{prefix}_wall_ns"]
                != tick[f"{prefix}_end_ns"] - tick[f"{prefix}_start_ns"]
                or type(tick[f"{prefix}_cpu_ns"]) is not int
                or tick[f"{prefix}_cpu_ns"] < 0
            ):
                raise ValueError("copy/publish/poll cost not charged to tick")
    for index, service in enumerate(services):
        if (
            service["sequence"] != index
            or service["generation"] != 0
            or service["stale"] != 0
            or not service["publish_ns"]
            <= service["receive_ns"]
            <= service["end_ns"]
            <= service["poll_ns"]
            or service["service_wall_ns"] != service["end_ns"] - service["receive_ns"]
            or service["service_latency_ns"] != service["end_ns"] - service["publish_ns"]
            or type(service["service_cpu_ns"]) is not int
            or service["service_cpu_ns"] < 0
            or not 0 < service["wire_bytes"] <= MAX_WIRE_BYTES
            or service["rss_bytes"] <= 0
        ):
            raise ValueError("renderer reply identity/clock/size/RSS")
    if not services or data["startup_ns"] <= 0:
        raise ValueError("missing cold service/startup evidence")
    return {
        "startup_ns": data["startup_ns"],
        "capacity": 1,
        "cold_first_service": services[0],
        "warm_service_wall_ns": _distribution([s["service_wall_ns"] for s in services[warmup:]]),
        "warm_service_cpu_ns": _distribution([s["service_cpu_ns"] for s in services[warmup:]]),
        "warm_service_latency_ns": _distribution(
            [s["service_latency_ns"] for s in services[warmup:]]
        ),
        "renderer_rss_bytes": _distribution([s["rss_bytes"] for s in services]),
        "game_peak_rss_bytes": data["game_peak_rss_bytes"],
        "overflow_fallback": {key: data[key] for key in ("busy", "crashes", "stale")},
        "costs": {
            key: _distribution([tick[key] for tick in ticks])
            for key in (
                "copy_wall_ns",
                "copy_cpu_ns",
                "publish_wall_ns",
                "publish_cpu_ns",
                "poll_wall_ns",
                "poll_cpu_ns",
                "export_bytes",
            )
        },
        "endpoint": data["endpoint"],
    }
