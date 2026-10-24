#!/bin/bash
# Persistent watcher daemon. Registration lines are: full log path + space + name.
OPS=${OPS_DIR:-/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops}
REPO=${OPS_REPO:-/Users/josenunez/Projects/ml/snake-dqn}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OPS"
touch "$OPS/watch-logs.txt" "$OPS/watch-pids.txt" "$OPS/seen.txt" "$OPS/events.log"
export PATH="$HOME/.local/bin:$PATH"
colab_low_since=""; mac_idle_since=""; qualification_last=""
ev() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$OPS/events.log"; }
bounded_capture() {
  # macOS lacks a portable timeout(1); keep external watcher probes below 15 s.
  python3 -c 'import subprocess,sys
try:
 p=subprocess.run(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=15)
 print(p.stdout, end=""); raise SystemExit(p.returncode)
except subprocess.TimeoutExpired as e:
 print((e.stdout or "") if isinstance(e.stdout, str) else "", end=""); print("WATCH_PROBE_TIMEOUT", file=sys.stderr); raise SystemExit(124)' "$@"
}
journal_wake() {
  # A watcher observation only wakes reconciliation; it cannot manufacture READY.
  python3 "$HERE/task_journal.py" --ops "$OPS" wake --source "$1" --detail "$2" >/dev/null 2>&1 || {
    ev "TASK_JOURNAL_WAKE_FAILED $1"; return 1
  }
}
journal_reconcile() {
  local notices
  notices=$(python3 "$HERE/task_journal.py" --ops "$OPS" reconcile 2>/dev/null) || {
    ev 'TASK_JOURNAL_RECONCILE_FAILED'; return
  }
  [ "$notices" = '[]' ] || ev "TASK_JOURNAL_RECONCILE $notices"
}
while true; do
  loop_started=$(date +%s)
  # Do this before a potentially slow external qualification refresh as well as
  # at loop end; normal watcher iterations remain bounded by the 60-second sleep.
  journal_reconcile
  # Fetch is bounded by qualification.py; a failed refresh also fails closed.
  qualification_args=(check --repo "$REPO" --record "$OPS/colab-cli-qualified.json" --ref origin/main)
  [ "${OPS_WATCH_FETCH:-1}" = 0 ] || qualification_args+=(--fetch)
  if qualification=$(bounded_capture python3 "$HERE/qualification.py" "${qualification_args[@]}" 2>&1); then
    qualification_last=""
  elif [ "$qualification" != "$qualification_last" ]; then
    ev "COLAB_CLI_UNQUALIFIED $qualification"
    qualification_last=$qualification
  fi
  while IFS= read -r entry; do
    # Names contain no whitespace; split at the last space to preserve log paths.
    case "$entry" in *' '*) f=${entry% *}; name=${entry##* };; *) continue;; esac
    if [ -n "$f" ] && grep -q '^EXIT=' "$f" 2>/dev/null && \
        ! grep -Fqx -- "log:$f" "$OPS/seen.txt"; then
      ev "JOB_ENDED $name $(grep '^EXIT=' "$f") $(grep -o 'INCOMPLETE[^;]*' "$f" | tail -1 | cut -c1-120)"
      journal_wake "log:$f" "JOB_ENDED $name" || continue
      printf '%s\n' "log:$f" >> "$OPS/seen.txt"
    fi
    journal_reconcile
  done < "$OPS/watch-logs.txt"
  while read -r pid name; do
    if [ -n "$pid" ] && ! ps -p "$pid" >/dev/null 2>&1 && \
        ! grep -Fqx -- "pid:$pid" "$OPS/seen.txt"; then
      ev "SOL_DONE $name"
      journal_wake "pid:$pid" "SOL_DONE $name" || continue
      printf '%s\n' "pid:$pid" >> "$OPS/seen.txt"
    fi
    journal_reconcile
  done < "$OPS/watch-pids.txt"
  if sessions=$(bounded_capture colab --auth oauth2 sessions 2>/dev/null); then
    n=$(printf '%s\n' "$sessions" | grep -c Hardware)
  else
    n=3; ev 'COLAB_SESSIONS_UNKNOWN'
  fi
  okuntil=$(cat "$OPS/colab-idle-ok-until" 2>/dev/null || echo 0)
  now=$(date +%s)
  if [ "${n:-0}" -lt 3 ] && [ "$now" -ge "${okuntil:-0}" ]; then
    [ -z "$colab_low_since" ] && colab_low_since=$now
    if [ $((now-colab_low_since)) -ge 600 ]; then
      ev "COLAB_UNDERUSED sessions=$n for >=10min"; colab_low_since=$now
    fi
  else colab_low_since=""; fi
  busy=$(ps -Ao pcpu,comm | awk '$2 ~ /[Pp]ython/ && $1>50' | wc -l | tr -d ' ')
  if [ "$busy" = 0 ]; then
    [ -z "$mac_idle_since" ] && mac_idle_since=$now
    if [ $((now-mac_idle_since)) -ge 900 ]; then
      ev 'MAC_IDLE no python worker >50% CPU for >=15min'; mac_idle_since=$now
    fi
  else mac_idle_since=""; fi
  # Events are hints; the durable journal scans unfinished obligations every loop.
  journal_reconcile
  [ "${OPS_WATCH_ONCE:-0}" = 1 ] && break
  # The external probes are bounded above; sleep only to the next 60-second
  # reconciliation deadline rather than adding a fixed minute after their work.
  next=$((loop_started + 60)); now=$(date +%s); delay=$((next-now)); [ "$delay" -gt 0 ] && sleep "$delay"
done
