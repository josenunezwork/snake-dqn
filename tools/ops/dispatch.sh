#!/bin/bash
# Safe dispatch for Colab supervisor jobs. Usage:
#   tools/ops/dispatch.sh NAME MANIFEST [--dispatch-txt FILE] [--broker DIR] [--state-dir DIR] [--cwd DIR]
# Refuses on BLOCK/HOLD lines in DISPATCH.txt, validates, launches in tmux with EXIT= marker,
# registers the log with the watcher daemon and makes sure the daemon is running.
set -u
NAME=$1; MANIFEST=$2; shift 2
DTXT=""; CWD=/Users/josenunez/Projects/ml/snake-dqn; STATE=""; BROKER=/Users/josenunez/Projects/ml/snake-dqn
while [ $# -gt 0 ]; do case $1 in --dispatch-txt) DTXT=$2; shift 2;; --cwd) CWD=$2; shift 2;; --state-dir) STATE=$2; shift 2;; --broker) BROKER=$2; shift 2;; *) echo "unknown $1"; exit 2;; esac; done
[ -z "$DTXT" ] && DTXT="$(dirname "$MANIFEST")/DISPATCH.txt"
OPS=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops; mkdir -p $OPS
[ -f "$MANIFEST" ] || { echo "REFUSE: no manifest $MANIFEST"; exit 3; }
if [ -f "$DTXT" ]; then
  if grep -E -n '^[[:space:]]*(BLOCK|HOLD)\b|RUNTIME BLOCKED|do not run this|PREPARED_NOT_DISPATCHED' "$DTXT" | grep -v -i 'live inventory' ; then
    echo "REFUSE: $DTXT has BLOCK/HOLD lines above — discharge them first"; exit 4; fi
else echo "WARN: no DISPATCH.txt next to manifest"; fi
grep -q '"scientific_admission": *false' "$MANIFEST" && { echo "REFUSE: manifest scientific_admission=false"; exit 4; }
python3 - "$MANIFEST" <<'PY' || exit 4
import json,sys
m=json.load(open(sys.argv[1])); oa=m.get("budget",{}).get("other_assignments",{})
if not isinstance(oa,dict): print("REFUSE: budget.other_assignments must be a map"); sys.exit(1)
PY
[ -z "$STATE" ] && STATE="$OPS/state/$NAME"; mkdir -p "$(dirname $STATE)"
LOG=$OPS/logs/$NAME.log; mkdir -p $OPS/logs; [ -e $LOG ] && { echo "REFUSE: log $LOG exists (name reused)"; exit 5; }
export PATH=$HOME/.local/bin:$PATH
( cd "$CWD" && $BROKER/research/colab_cli/launch.sh --check --state-dir "$STATE" --colab $HOME/.local/bin/colab validate "$MANIFEST" ) > $OPS/logs/$NAME.validate 2>&1 || { echo "REFUSE: validate failed — see $OPS/logs/$NAME.validate"; tail -5 $OPS/logs/$NAME.validate; exit 6; }
tmux new-session -d -s "job-$NAME" "cd '$CWD' && export PATH=\$HOME/.local/bin:\$PATH && $BROKER/research/colab_cli/launch.sh --state-dir '$STATE' --colab \$HOME/.local/bin/colab run '$MANIFEST' > '$LOG' 2>&1; echo EXIT=\$? >> '$LOG'"
echo "$LOG $NAME" >> $OPS/watch-logs.txt
tmux has-session -t watchd 2>/dev/null || tmux new-session -d -s watchd /Users/josenunez/Projects/ml/snake-dqn/tools/ops/watchd.sh
echo "DISPATCHED $NAME log=$LOG"
