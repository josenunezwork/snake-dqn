"""Opt-in thermal guard for long CPU research runs (stdlib only; never plays episodes).

Two signals decide whether a run may admit its next episode:

1. macOS thermal state from ``pmset -g therm`` (:func:`parse_pmset_therm`). The parser reads
   the thermal warning level, the performance warning level and the ``CPU Power notify``
   block (``CPU_Speed_Limit``, ``CPU_Scheduler_Limit``, ``CPU_Available_CPUs``).
2. An episode-slowdown detector (:func:`slowdown_report`): per key (``arm/mix``) the
   median of the run's first ``baseline_k`` episode wall times is the baseline, and the key
   is flagged once the median of its latest ``window`` later episodes exceeds the baseline
   by more than ``slowdown`` (1.60 = 60 % slower). The defaults (12 / 1.60 / 8) are wide on
   purpose: episode wall time varies with game content (10-51 s, CV about 0.35 within a
   run), and the earlier 6 / 1.30 / 3 setting would have paused 4-24 times per past screen
   with no thermal cause; 12 / 1.60 / 8 paused 0 times on the same five screens.
   A pause answers the slowdown evidence seen so far. A key that is flagged again on fresh
   episodes after ``max_slowdown_backoffs`` consecutive slowdown pauses is a *persistent*
   slowdown, and :func:`admit_next_episode` stops at once (no further pause).

Fail-safe parsing (the guard fails CLOSED on darwin, where it can read the machine):

- ``Note: No thermal warning level has been recorded`` (and the performance twin) means
  nominal, level 0. A level line in pmset's own format (``Thermal Warning Level = %d``,
  ``Performance Warning Level = %d``) sets the level; the last occurrence wins. Any other
  line that names a level, including a negative level, is "unparsed".
- Any line starting with ``Error`` (pmset prints e.g. ``Error:Failed to get thermal warning
  level with error code 0x...``) is "unparsed"; it is never read as a level.
- ``Note: No CPU power status has been recorded`` means no limit is in force (nominal).
  ``CPU_Speed_Limit = N`` / ``CPU_Scheduler_Limit = N`` below ``limit_floor`` (100) is a
  throttle and not ok.
- Unknown state is not ok: a level with neither a number nor a "No ... recorded" note,
  CPU power with neither a value nor its note, any unparsed relevant line, or ``pmset``
  missing, failing or timing out on darwin.
- Off darwin there is no ``pmset``; the thermal part is skipped and recorded as such
  (``platform_not_gated``), and only the slowdown detector (and an AC reader, if given)
  can stop admission.

A not-ok check never relabels anything: the caller (``dev_screen --thermal-guard``) pauses
admission for a bounded backoff and, if the guard is still not ok, stops with a
``thermal: ...`` reason; the run's own decision rule then sees the missing episodes.
"""

from __future__ import annotations

import re
import statistics
import subprocess
import sys
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

DEFAULT_BASELINE_K = 12
DEFAULT_SLOWDOWN = 1.60
DEFAULT_WINDOW = 8
DEFAULT_MAX_SLOWDOWN_BACKOFFS = 2
DEFAULT_LIMIT_FLOOR = 100
PMSET_TIMEOUT_SECONDS = 10.0

_LEVELS = {
    "thermal_warning_level": "thermal warning level",
    "performance_warning_level": "performance warning level",
}
_LIMITS = ("CPU_Speed_Limit", "CPU_Scheduler_Limit", "CPU_Available_CPUs")
_CPU_POWER_NOTE = re.compile(r"no\s+cpu\s+power\s+status\s+has\s+been\s+recorded", re.I)


def parse_pmset_therm(text: str) -> Dict[str, Any]:
    """Parse ``pmset -g therm`` output into levels, limits and parse diagnostics.

    Returns ``thermal_warning_level`` / ``performance_warning_level`` (int, 0 when the
    "No ... recorded" note is present, ``None`` when unknown), ``cpu_power_status_note``
    (bool), each of :data:`_LIMITS` (int or ``None``), ``unparsed`` (relevant lines that
    could not be read) and ``recognized_lines``. Pure: no subprocess, no clock.
    """
    out: Dict[str, Any] = {key: None for key in _LEVELS}
    out.update({name: None for name in _LIMITS})
    out["cpu_power_status_note"] = False
    unparsed: List[str] = []
    recognized = 0
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("error"):
            unparsed.append(line)  # pmset's own failure line: fail closed, never guess
            continue
        if _CPU_POWER_NOTE.search(line):
            out["cpu_power_status_note"] = True
            recognized += 1
            continue
        matched = False
        for key, phrase in _LEVELS.items():
            if phrase not in lower:
                continue
            matched = True
            if re.search(rf"no\s+{phrase}\s+has\s+been\s+recorded", lower):
                if out[key] is None:
                    out[key] = 0
                recognized += 1
                continue
            number = re.fullmatch(rf"{phrase}\s*=\s*(\d+)", lower)
            if number is None:
                unparsed.append(line)
            else:
                out[key] = int(number.group(1))
                recognized += 1
        for name in _LIMITS:
            if name.lower() not in lower:
                continue
            matched = True
            number = re.search(rf"{name.lower()}\s*[=:]\s*(-?\d+)", lower)
            if number is None:
                unparsed.append(line)
            else:
                out[name] = int(number.group(1))
                recognized += 1
        if not matched and ("warning" in lower or "limit" in lower):
            unparsed.append(line)
    out["unparsed"] = unparsed
    out["recognized_lines"] = recognized
    return out


def thermal_reasons(parsed: Mapping[str, Any], limit_floor: int = DEFAULT_LIMIT_FLOOR) -> List[str]:
    """Reasons a parsed pmset reading is not ok (empty list = nominal); fails closed."""
    reasons: List[str] = []
    for key in _LEVELS:
        level = parsed.get(key)
        if level is None or level < 0:
            reasons.append(f"{key}_unknown")
        elif level > 0:
            reasons.append(f"{key}={level}")
    for name in ("CPU_Speed_Limit", "CPU_Scheduler_Limit"):
        value = parsed.get(name)
        if value is not None and value < limit_floor:
            reasons.append(f"{name}={value}<{limit_floor}")
    has_limit = parsed.get("CPU_Speed_Limit") is not None
    if not has_limit and not parsed.get("cpu_power_status_note"):
        reasons.append("cpu_power_status_unknown")
    if parsed.get("unparsed"):
        reasons.append(f"pmset_unparsed_lines={len(parsed['unparsed'])}")
    return reasons


def read_pmset_therm(
    runner: Callable[..., Any] = subprocess.run, platform: str | None = None
) -> Dict[str, Any]:
    """Run ``pmset -g therm`` (darwin only). Never raises; failures are recorded."""
    platform = sys.platform if platform is None else platform
    if platform != "darwin":
        return {"platform": platform, "gated": False, "available": False, "text": None}
    try:
        done = runner(
            ["pmset", "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=PMSET_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {
            "platform": platform,
            "gated": True,
            "available": False,
            "text": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"platform": platform, "gated": True, "available": True, "text": done.stdout}


def slowdown_report(
    times_by_key: Mapping[str, Sequence[float]],
    baseline_k: int = DEFAULT_BASELINE_K,
    slowdown: float = DEFAULT_SLOWDOWN,
    window: int = DEFAULT_WINDOW,
    acknowledged: Mapping[str, int] | None = None,
) -> Dict[str, Any]:
    """Per-key baseline vs recent median episode wall time; pure.

    ``times_by_key[key]`` is the key's wall times in run order. The baseline is the median
    of the first ``baseline_k``; the recent median is over the latest ``window`` times after
    the baseline, ignoring the first ``acknowledged[key]`` post-baseline times (evidence a
    backoff already answered). A key is flagged when ``recent > slowdown * baseline``.
    """
    if baseline_k < 1 or window < 1 or slowdown <= 1.0:
        raise ValueError("need baseline_k >= 1, window >= 1 and slowdown > 1")
    acknowledged = acknowledged or {}
    keys: Dict[str, Any] = {}
    flagged: List[str] = []
    for key, times in sorted(times_by_key.items()):
        values = [float(t) for t in times]
        row: Dict[str, Any] = {"episodes": len(values), "baseline_median": None}
        if len(values) >= baseline_k:
            baseline = statistics.median(values[:baseline_k])
            later = values[baseline_k + int(acknowledged.get(key, 0)) :]
            row["baseline_median"] = round(baseline, 6)
            row["fresh_after_baseline"] = len(later)
            if len(later) >= window and baseline > 0:
                recent = statistics.median(later[-window:])
                ratio = recent / baseline
                row.update({"recent_median": round(recent, 6), "ratio": round(ratio, 6)})
                if ratio > slowdown:
                    flagged.append(key)
        keys[key] = row
    return {
        "baseline_k": baseline_k,
        "window": window,
        "slowdown": slowdown,
        "flagged": flagged,
        "keys": keys,
    }


class ThermalGuard:
    """Combine pmset thermal state, the slowdown detector and an optional AC reader.

    ``pmset_reader`` returns :func:`read_pmset_therm`-shaped dicts (tests inject one);
    ``ac_reader`` returns True on AC power (``None`` = not checked). ``check`` never raises.
    ``slowdown_backoffs[key]`` counts the consecutive pauses taken while ``key`` was flagged;
    it resets when a fresh window for the key is not flagged. A key flagged again once that
    count reaches ``max_slowdown_backoffs`` is listed in ``readings.slowdown.persistent``.
    """

    def __init__(
        self,
        baseline_k: int = DEFAULT_BASELINE_K,
        slowdown: float = DEFAULT_SLOWDOWN,
        window: int = DEFAULT_WINDOW,
        limit_floor: int = DEFAULT_LIMIT_FLOOR,
        max_slowdown_backoffs: int = DEFAULT_MAX_SLOWDOWN_BACKOFFS,
        pmset_reader: Callable[[], Mapping[str, Any]] | None = None,
        ac_reader: Callable[[], bool] | None = None,
    ) -> None:
        slowdown_report({}, baseline_k, slowdown, window)  # validates the parameters
        if int(max_slowdown_backoffs) < 0:
            raise ValueError("need max_slowdown_backoffs >= 0")
        self.baseline_k = int(baseline_k)
        self.slowdown = float(slowdown)
        self.window = int(window)
        self.limit_floor = int(limit_floor)
        self.max_slowdown_backoffs = int(max_slowdown_backoffs)
        self.pmset_reader = pmset_reader or read_pmset_therm
        self.ac_reader = ac_reader
        self.times: Dict[str, List[float]] = {}
        self.acknowledged: Dict[str, int] = {}
        self.slowdown_backoffs: Dict[str, int] = {}
        self._flagged: List[str] = []

    def config(self) -> Dict[str, Any]:
        return {
            "baseline_k": self.baseline_k,
            "slowdown": self.slowdown,
            "window": self.window,
            "limit_floor": self.limit_floor,
            "max_slowdown_backoffs": self.max_slowdown_backoffs,
            "ac_checked": self.ac_reader is not None,
        }

    def record_episode(self, key: str, seconds: float) -> None:
        self.times.setdefault(str(key), []).append(float(seconds))

    def acknowledge_slowdown(self) -> None:
        """Mark every post-baseline time seen so far as answered (after a backoff).

        Keys flagged by the latest :meth:`check` gain one consecutive slowdown backoff.
        """
        for key in self._flagged:
            self.slowdown_backoffs[key] = self.slowdown_backoffs.get(key, 0) + 1
        self._flagged = []
        for key, values in self.times.items():
            self.acknowledged[key] = max(0, len(values) - self.baseline_k)

    def check(self) -> Dict[str, Any]:
        """``{ok, reasons, readings}`` for admitting the next episode."""
        reasons: List[str] = []
        try:
            raw = dict(self.pmset_reader())
        except Exception as exc:  # a reader bug must not crash the run; fail closed
            raw = {"gated": True, "available": False, "error": f"{type(exc).__name__}: {exc}"}
        pmset: Dict[str, Any] = {k: v for k, v in raw.items() if k != "text"}
        if not raw.get("gated", True):
            pmset["skipped"] = "platform_not_gated"
        elif not raw.get("available"):
            reasons.append("pmset_unavailable")
        else:
            parsed = parse_pmset_therm(raw.get("text") or "")
            pmset["parsed"] = parsed
            reasons.extend(thermal_reasons(parsed, self.limit_floor))
        slow = slowdown_report(
            self.times, self.baseline_k, self.slowdown, self.window, self.acknowledged
        )
        for key, row in slow["keys"].items():
            if "ratio" in row and key not in slow["flagged"]:
                self.slowdown_backoffs[key] = 0  # a fresh, unflagged window ends the streak
        self._flagged = list(slow["flagged"])
        persistent = [
            key
            for key in slow["flagged"]
            if self.slowdown_backoffs.get(key, 0) >= self.max_slowdown_backoffs
        ]
        slow["persistent"] = persistent
        slow["consecutive_backoffs"] = {
            key: count for key, count in sorted(self.slowdown_backoffs.items()) if count
        }
        reasons.extend(
            f"slowdown_persistent:{key}" if key in persistent else f"slowdown:{key}"
            for key in slow["flagged"]
        )
        readings: Dict[str, Any] = {"pmset": pmset, "slowdown": slow, "ac_power": None}
        if self.ac_reader is not None:
            try:
                readings["ac_power"] = bool(self.ac_reader())
            except Exception as exc:  # fail closed
                readings["ac_power_error"] = f"{type(exc).__name__}: {exc}"
                readings["ac_power"] = False
            if not readings["ac_power"]:
                reasons.append("on_battery")
        return {"ok": not reasons, "reasons": reasons, "readings": readings}


def admit_next_episode(
    guard: ThermalGuard,
    *,
    backoff_seconds: float,
    max_backoffs: int,
    seconds_left: Callable[[], float],
    budget_seconds: float,
    log: Callable[[Dict[str, Any]], None],
    sleep: Callable[[float], None],
) -> Tuple[bool, str | None, Dict[str, int]]:
    """Gate one episode: ``(admit, stop_reason, counts)``; bounded, never relabels.

    Checks the guard; while it is not ok, pauses ``backoff_seconds`` (at most
    ``max_backoffs`` times, and never past the point where ``seconds_left()`` would drop
    below ``budget_seconds``), acknowledging slowdown evidence after each pause, then
    re-checks. A persistent slowdown (see :class:`ThermalGuard`) stops at once, without a
    pause. Every check and pause is passed to ``log``. When the guard is still not ok
    the stop reason starts with ``"thermal: "``. ``counts`` has ``checks``,
    ``not_ok_checks``, ``backoffs`` and ``backoff_seconds``.
    """
    counts = {"checks": 0, "not_ok_checks": 0, "backoffs": 0, "backoff_seconds": 0}
    result: Dict[str, Any] = {"reasons": []}
    for attempt in range(int(max_backoffs) + 1):
        result = guard.check()
        counts["checks"] += 1
        log({"event": "thermal_guard_check", "attempt": attempt, **result})
        if result["ok"]:
            return True, None, counts
        counts["not_ok_checks"] += 1
        persistent = result.get("readings", {}).get("slowdown", {}).get("persistent") or []
        if persistent:
            reason = (
                f"thermal: persistent slowdown ({', '.join(persistent)}) still flagged after "
                f"{guard.max_slowdown_backoffs} consecutive slowdown backoff(s); "
                f"reasons: {', '.join(result['reasons'])}"
            )
            log({"event": "thermal_guard_stop", "reason": reason})
            return False, reason, counts
        if attempt == int(max_backoffs):
            break
        if seconds_left() - backoff_seconds < budget_seconds:
            reason = (
                f"thermal: not ok ({', '.join(result['reasons'])}) and a "
                f"{backoff_seconds:.0f} s backoff would cross the episode budget"
            )
            log({"event": "thermal_guard_stop", "reason": reason})
            return False, reason, counts
        log(
            {
                "event": "thermal_guard_backoff",
                "backoff": attempt + 1,
                "seconds": backoff_seconds,
                "reasons": result["reasons"],
            }
        )
        sleep(backoff_seconds)
        counts["backoffs"] += 1
        counts["backoff_seconds"] += int(round(backoff_seconds))
        guard.acknowledge_slowdown()
    reason = (
        f"thermal: still not ok ({', '.join(result['reasons'])}) after "
        f"{int(max_backoffs)} backoff(s) of {backoff_seconds:.0f} s"
    )
    log({"event": "thermal_guard_stop", "reason": reason})
    return False, reason, counts
