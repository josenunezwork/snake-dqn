#!/usr/bin/env bash
# Persistent Watch server: runs the `snake-web` entry of .claude/launch.json
# (env + command) and restarts it if it exits. Meant to live in the detached
# tmux session `snake-web`, so serving no longer depends on a Claude app
# session (the app stops its preview servers when a session ends).
#
#   start:    tmux new-session -d -s snake-web "bash tools/ops/watch_server.sh"
#   stop:     tmux kill-session -t snake-web      (never kill-server)
#   rollback: edit .claude/launch.json, then stop + start (re-read on each start)
#   logs:     ../snake-dqn-artifacts/ops/watch-server/server.log
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$REPO/../snake-dqn-artifacts/ops/watch-server"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/server.log"
cd "$REPO" || exit 1
MAX_BYTES=$((100 * 1024 * 1024))  # release monitor logs ~10 lines/s; keep 3 x 100 MB

# Append stdin to $LOG, rotating to server.log.1/.2 when it passes MAX_BYTES.
rotate_log() {
  local n=0 line
  while IFS= read -r line; do
    printf '%s\n' "$line" >>"$LOG"
    n=$((n + 1))
    if ((n % 5000 == 0)) && [[ $(stat -f %z "$LOG") -gt $MAX_BYTES ]]; then
      mv -f "$LOG.1" "$LOG.2" 2>/dev/null
      mv -f "$LOG" "$LOG.1"
    fi
  done
}

while true; do
  # Re-read launch.json every start so a rollback edit takes effect on restart.
  ARGS=()
  while IFS= read -r a; do ARGS+=("$a"); done \
    < <(jq -r '.configurations[] | select(.name=="snake-web") | .runtimeArgs[]' .claude/launch.json)
  EXE="$(jq -r '.configurations[] | select(.name=="snake-web") | .runtimeExecutable' .claude/launch.json)"
  if [[ ${#ARGS[@]} -eq 0 || -z "$EXE" ]]; then
    echo "$(date '+%F %T %Z') snake-web entry missing from launch.json; retry in 60s" | tee -a "$LOG"
    sleep 60
    continue
  fi
  if lsof -ti tcp:8000 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "$(date '+%F %T %Z') port 8000 busy (another server?); retry in 30s" | tee -a "$LOG"
    sleep 30
    continue
  fi
  echo "$(date '+%F %T %Z') start: $EXE ${ARGS[*]}" | tee -a "$LOG"
  "$EXE" "${ARGS[@]}" 2>&1 | rotate_log
  echo "$(date '+%F %T %Z') server exited (code ${PIPESTATUS[0]}); restart in 5s" | tee -a "$LOG"
  sleep 5
done
