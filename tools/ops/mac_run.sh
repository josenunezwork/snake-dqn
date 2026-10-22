#!/bin/bash
# Launch a pinned local job from an isolated, detached worktree.
#
# Usage: tools/ops/mac_run.sh [--repo REPO] NAME COMMIT -- COMMAND [ARG ...]
#
# The worktree is deliberately retained at ../snake-dqn-run-NAME.noindex.  A job
# owns its .IN_USE_LOCK until its tmux command exits; the generated runner only
# removes a lock whose token it created.  A later invocation may reuse that
# worktree only when it is detached, clean, and at the requested commit.
set -euo pipefail

usage() {
  echo "usage: $0 [--repo REPO] NAME COMMIT -- COMMAND [ARG ...]" >&2
  exit 2
}

HERE="$(cd "$(dirname "$0")" && pwd -P)"
REPO="$(cd "$HERE/../.." && pwd -P)"
while [ "${1:-}" = "--repo" ]; do
  [ $# -ge 2 ] || usage
  REPO=$2
  shift 2
done
[ $# -ge 4 ] || usage
NAME=$1
REQUESTED_COMMIT=$2
shift 2
[ "${1:-}" = "--" ] || usage
shift
[ $# -gt 0 ] || usage

case "$NAME" in
  [A-Za-z0-9]* ) ;;
  * ) echo "REFUSE: NAME must start with an alphanumeric character" >&2; exit 2 ;;
esac
case "$NAME" in
  *[!A-Za-z0-9._-]* ) echo "REFUSE: NAME contains unsupported characters" >&2; exit 2 ;;
esac

REPO="$(cd "$REPO" && pwd -P)"
git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "REFUSE: --repo is not a Git worktree: $REPO" >&2
  exit 2
}
COMMIT=$(git -C "$REPO" rev-parse --verify "${REQUESTED_COMMIT}^{commit}") || {
  echo "REFUSE: not a commit: $REQUESTED_COMMIT" >&2
  exit 2
}
REPO_PARENT=$(cd "$REPO/.." && pwd -P)
WORKTREE="$REPO_PARENT/snake-dqn-run-$NAME.noindex"
OPS_DIR=${OPS_DIR:-"$REPO_PARENT/snake-dqn-artifacts/ops"}
OPS_DIR=$(mkdir -p "$OPS_DIR/logs" && cd "$OPS_DIR" && pwd -P)
LOCK="$WORKTREE/.IN_USE_LOCK"
SHORT_COMMIT=${COMMIT:0:12}
LOG="$OPS_DIR/logs/mac-$NAME-$SHORT_COMMIT.log"
RUNNER="$OPS_DIR/mac-run-$NAME-$SHORT_COMMIT.sh"
EXCLUDES="$OPS_DIR/mac-run-$NAME-$SHORT_COMMIT.gitignore"
TOKEN="mac-run:$NAME:$SHORT_COMMIT:$$:$(date +%s)"

clean_detached_at_commit() {
  [ "$(git -C "$WORKTREE" rev-parse --is-inside-work-tree 2>/dev/null || true)" = "true" ] || return 1
  [ "$(git -C "$WORKTREE" rev-parse HEAD)" = "$COMMIT" ] || return 1
  [ -z "$(git -C "$WORKTREE" symbolic-ref -q HEAD 2>/dev/null || true)" ] || return 1
  [ -z "$(git -C "$WORKTREE" status --porcelain --untracked-files=all)" ] || return 1
}

if [ -e "$WORKTREE" ]; then
  [ -e "$LOCK" ] && { echo "REFUSE: target worktree is in use: $LOCK" >&2; exit 3; }
  clean_detached_at_commit || {
    echo "REFUSE: existing worktree must be detached, clean, and pinned at $COMMIT: $WORKTREE" >&2
    exit 3
  }
else
  git -C "$REPO" worktree add --detach "$WORKTREE" "$COMMIT"
  clean_detached_at_commit || {
    echo "REFUSE: newly-created worktree did not verify clean/detached/pinned" >&2
    exit 3
  }
fi

[ ! -e "$LOG" ] || { echo "REFUSE: log already exists: $LOG" >&2; exit 3; }
[ ! -e "$RUNNER" ] || { echo "REFUSE: runner already exists: $RUNNER" >&2; exit 3; }
[ ! -e "$EXCLUDES" ] || { echo "REFUSE: private excludes file already exists: $EXCLUDES" >&2; exit 3; }

# A live lock must not make a pinned historical worktree appear dirty to the
# command it owns.  Do this per job rather than changing shared Git config or
# an old commit's tracked .gitignore.  Preserve the effective configured file
# first, then append only the root lock spelling.
CONFIG_EXCLUDES=$(git -C "$WORKTREE" config --path --get core.excludesFile 2>/dev/null || true)
[ -n "$CONFIG_EXCLUDES" ] || CONFIG_EXCLUDES="${XDG_CONFIG_HOME:-$HOME/.config}/git/ignore"
if [ -f "$CONFIG_EXCLUDES" ]; then
  cat "$CONFIG_EXCLUDES" > "$EXCLUDES"
else
  : > "$EXCLUDES"
fi
printf '\n/.IN_USE_LOCK\n' >> "$EXCLUDES"

# noclobber makes creation of the reservation atomic.  Never replace somebody
# else's lock, even if the worktree passed the earlier clean check.
if ! ( umask 077; set -C; printf '%s\n' "$TOKEN" > "$LOCK" ) 2>/dev/null; then
  echo "REFUSE: target worktree became in use: $LOCK" >&2
  exit 3
fi

cleanup_before_launch() {
  if [ "$LAUNCHED" = 0 ] && [ -f "$LOCK" ] && [ "$(cat "$LOCK" 2>/dev/null || true)" = "$TOKEN" ]; then
    rm -f "$LOCK"
  fi
}
LAUNCHED=0
trap cleanup_before_launch EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Keep the long shell syntax in a private generated runner.  printf %q preserves
# the caller's argument boundaries without evaluating command text a second time.
{
  printf '#!/bin/bash\nset -u\n'
  printf 'worktree=%q\nlog=%q\nlock=%q\ntoken=%q\nexcludes=%q\n' "$WORKTREE" "$LOG" "$LOCK" "$TOKEN" "$EXCLUDES"
  cat <<'RUNNER_BODY'
finish() {
  status=$?
  printf 'EXIT=%s\n' "$status" >> "$log"
  if [ -f "$lock" ] && [ "$(cat "$lock" 2>/dev/null || true)" = "$token" ]; then
    rm -f "$lock"
  fi
  exit "$status"
}
trap finish EXIT
set -C
exec 3>"$log"
exec 1>&3 2>&1
cd "$worktree" || exit 2
pmset -g batt | grep 'AC Power' >/dev/null || { echo 'REFUSE: on battery' >&2; exit 2; }
ioreg -r -k AppleClamshellState | grep '"AppleClamshellState" = No' >/dev/null || { echo 'REFUSE: lid closed' >&2; exit 2; }
config_count=${GIT_CONFIG_COUNT:-0}
case "$config_count" in *[!0-9]*|'') echo 'REFUSE: invalid GIT_CONFIG_COUNT' >&2; exit 2;; esac
export GIT_CONFIG_COUNT=$((config_count + 1))
export "GIT_CONFIG_KEY_${config_count}=core.excludesFile"
export "GIT_CONFIG_VALUE_${config_count}=$excludes"
RUNNER_BODY
  printf 'exec_args=( '
  printf '%q ' "$@"
  printf ')\n'
  cat <<'RUNNER_BODY'
exec_args=(caffeinate -dims env SNAKE_DQN_DEVICE="${SNAKE_DQN_DEVICE:-cpu}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "${exec_args[@]}")
"${exec_args[@]}"
RUNNER_BODY
} > "$RUNNER"
chmod 700 "$RUNNER"

SESSION="mac-$NAME-$SHORT_COMMIT"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "REFUSE: tmux session already exists: $SESSION" >&2; exit 3; }
RUNNER_COMMAND=$(printf 'exec %q' "$RUNNER")
# Once submission starts, a client interruption cannot prove that the server
# did not start the job. Preserve the lock on an uncertain launch failure.
LAUNCHED=1
if tmux new-session -d -s "$SESSION" "$RUNNER_COMMAND"; then
  :
else
  status=$?
  echo "REFUSE: tmux submission uncertain; inspect session $SESSION and lock $LOCK" >&2
  exit "$status"
fi
trap - EXIT INT TERM

touch "$OPS_DIR/watch-logs.txt"
printf '%s %s\n' "$LOG" "mac-$NAME" >> "$OPS_DIR/watch-logs.txt"
WATCHER_COMMAND=$(printf 'exec env OPS_DIR=%q OPS_REPO=%q %q' "$OPS_DIR" "$REPO" "$HERE/watchd.sh")
tmux has-session -t watchd 2>/dev/null || tmux new-session -d -s watchd "$WATCHER_COMMAND"
echo "MAC_DISPATCHED $NAME commit=$COMMIT worktree=$WORKTREE session=$SESSION log=$LOG"
