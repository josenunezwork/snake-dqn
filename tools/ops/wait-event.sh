#!/bin/bash
# Block until there are events in ops/events.log beyond the persisted read cursor (default 3 h base,
# automatically extended to outlast active job timeouts plus grace); print them and advance the cursor.
# Using a cursor (not "lines at start") means events written between two calls are never missed.
OPS=${OPS_DIR:-/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops}; mkdir -p "$OPS"; touch "$OPS/events.log"
HERE=$(cd "$(dirname "$0")" && pwd)
CONSUMER=${OPS_EVENT_CONSUMER:-operator}
CUR=$OPS/events.cursor.$CONSUMER; [ -f "$CUR" ] || echo 0 > "$CUR"
WAIT_SECONDS=${WAIT_EVENT_SECONDS:-10800}
GRACE_SECONDS=${WAIT_EVENT_GRACE_SECONDS:-600}
POLL_SECONDS=${WAIT_EVENT_POLL_SECONDS:-30}
start_ts=$(date +%s)

compute_deadline() {
  python3 - "$OPS" "$start_ts" "$WAIT_SECONDS" "$GRACE_SECONDS" <<'PY'
import json
import math
import os
import re
import sys
import time
from pathlib import Path

ops = Path(sys.argv[1])
start_ts = float(sys.argv[2])
wait_s = float(sys.argv[3])
grace_s = float(sys.argv[4])
now = time.time()
deadline = start_ts + wait_s


def file_start_ts(path: Path) -> float:
    st = path.stat()
    birth = getattr(st, "st_birthtime", 0.0) or 0.0
    if birth > 0:
        return float(min(birth, st.st_mtime))
    return float(st.st_mtime)


def manifest_timeout_seconds(data: dict) -> float:
    candidates = [0.0]
    for val in (
        data.get("timeout_minutes"),
        (data.get("budget") or {}).get("wall_minutes")
        if isinstance(data.get("budget"), dict)
        else None,
        (data.get("resume_exemption") or {}).get("timeout_minutes")
        if isinstance(data.get("resume_exemption"), dict)
        else None,
        (data.get("requirements") or {}).get("est_minutes")
        if isinstance(data.get("requirements"), dict)
        else None,
        data.get("est_minutes"),
        (data.get("bank_v2_screen") or {}).get("base_minutes")
        if isinstance(data.get("bank_v2_screen"), dict)
        else None,
    ):
        if (
            isinstance(val, (int, float))
            and not isinstance(val, bool)
            and math.isfinite(val)
            and val > 0
        ):
            candidates.append(float(val) * 60.0)
    t_sec = data.get("timeout_seconds")
    if (
        isinstance(t_sec, (int, float))
        and not isinstance(t_sec, bool)
        and math.isfinite(t_sec)
        and t_sec > 0
    ):
        candidates.append(float(t_sec))
    env = data.get("env")
    if isinstance(env, dict):
        for key in ("SCREEN_SESSION_SECONDS", "WATCH_REPLAY_SESSION_SECONDS", "SESSION_SECONDS"):
            raw = env.get(key)
            try:
                sec = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(sec) and sec > 0:
                candidates.append(sec)
    return max(candidates)


names: dict[str, Path | None] = {}
watch_logs = ops / "watch-logs.txt"
if watch_logs.is_file():
    for raw in watch_logs.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line:
            p = Path(line)
            names[p.stem[:-4] if p.name.endswith(".log") else p.stem] = p

manifests_dir = ops / "manifests"
if manifests_dir.is_dir():
    for mf in sorted(manifests_dir.glob("*.json")):
        if mf.name.endswith(".live.json"):
            continue
        names.setdefault(mf.stem, None)

for name, watched_log in names.items():
    if (ops / "state" / f"{name}.done").is_file():
        continue
    log_path = watched_log if watched_log is not None else (ops / "logs" / f"{name}.log")
    mf_path = manifests_dir / f"{name}.json"
    if not mf_path.is_file():
        mf_path = manifests_dir / f"{name}.live.json"
    timeout_s = 0.0
    job_start = 0.0
    if mf_path.is_file():
        try:
            job_start = file_start_ts(mf_path)
            data = json.loads(mf_path.read_text(errors="replace"))
            if isinstance(data, dict):
                timeout_s = manifest_timeout_seconds(data)
        except Exception:
            pass
    log_has_exit = False
    log_mtime = 0.0
    if log_path.is_file():
        try:
            log_mtime = float(log_path.stat().st_mtime)
            if job_start <= 0.0:
                job_start = file_start_ts(log_path)
            text = log_path.read_text(errors="replace")
            log_has_exit = re.search(r"^EXIT=\d+$", text, re.MULTILINE) is not None
        except Exception:
            pass
    if log_has_exit:
        if watched_log is not None and now <= log_mtime + grace_s:
            deadline = max(deadline, log_mtime + grace_s)
        continue
    if timeout_s > 0.0 and job_start > 0.0 and now <= job_start + timeout_s + grace_s:
        deadline = max(deadline, max(job_start, start_ts) + timeout_s + grace_s)

watch_pids = ops / "watch-pids.txt"
if watch_pids.is_file():
    for raw in watch_pids.read_text(errors="replace").splitlines():
        parts = raw.split()
        if len(parts) < 2:
            continue
        pid_str, name = parts[0], parts[1]
        if (ops / "state" / f"{name}.done").is_file():
            continue
        try:
            os.kill(int(pid_str), 0)
            deadline = max(deadline, now + grace_s)
        except Exception:
            pass

print(int(math.ceil(deadline)))
PY
}

check_events() {
  # Task notices are at-least-once until the consumer explicitly acknowledges
  # them with task_journal.py ack; printing is never an acknowledgment.
  notices=$(python3 "$HERE/task_journal.py" --ops "$OPS" deliver "$CONSUMER" 2>/dev/null)
  if [ -n "$notices" ] && [ "$notices" != '[]' ]; then printf '%s\n' "$notices"; exit 0; fi
  start=$(cat "$CUR"); now=$(wc -l < "$OPS/events.log" | tr -d ' ')
  if [ "$now" -gt "$start" ]; then tail -n +$((start+1)) "$OPS/events.log" | head -n $((now-start)); echo "$now" > "$CUR"; exit 0; fi
}

check_events
end=$(compute_deadline)
while [ "$(date +%s)" -lt "$end" ]; do
  sleep "$POLL_SECONDS"
  check_events
  end=$(compute_deadline)
done
check_events
echo "NO EVENT in 3 h"
