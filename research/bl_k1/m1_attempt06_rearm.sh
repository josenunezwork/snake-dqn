#!/usr/bin/env bash
# Measured host admission gate for the final PLAN-level M1 rerun.  This is an
# operator-side waiter; it deliberately does not alter the M1 harness.
set -euo pipefail

ARTIFACT_ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts
EXP="$ARTIFACT_ROOT/big-lever/exp"
OPS="$ARTIFACT_ROOT/ops"
WORKTREE=/Users/josenunez/Projects/ml/snake-dqn-m1run-fixed-06
FIXTURES="$EXP/bl-l2/ruler-fixtures-08-final/ruler-fixtures.json"
REFERENCE="$EXP/bl-l2/y1-native-mac/ruler-receipt.json"
PROOF=""
LEDGER=""
CONFIRMATION=""
DEADLINE='2026-10-24 09:00:00'
OUT="$EXP/bl-m1-attempt-06"
WINDOW="$EXP/bl-m1-attempt-06.quietwindow.json"
CANARY="$EXP/bl-m1-attempt-06.spotlight-canary.json"
FLAG="$OPS/mac-timing-armed"
CHAIN="$EXP/bl-m1-chain.log"
RUNNER=""

usage() {
  cat <<'EOF'
usage: m1_attempt06_rearm.sh --confirmation FILE --proof FILE --ledger FILE [options]

Waits for the JUDGMENT_M1_ATTEMPT6 section 3.1 clean window, then invokes the
unchanged run_m1.sh once.  CONFIRMATION is an existing operator-created file
recording the owner actions; this script never creates or infers it.

Options:
  --worktree DIR       frozen worktree (default: snake-dqn-m1run-fixed-06)
  --out DIR            attempt output directory
  --window FILE        create-only quiet-window receipt path
  --check-config       validate arguments only; do not arm, wait, or dispatch
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirmation) CONFIRMATION=${2:?}; shift 2 ;;
    --proof) PROOF=${2:?}; shift 2 ;;
    --ledger) LEDGER=${2:?}; shift 2 ;;
    --worktree) WORKTREE=${2:?}; shift 2 ;;
    --out) OUT=${2:?}; shift 2 ;;
    --window) WINDOW=${2:?}; shift 2 ;;
    --check-config) RUNNER=check; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$CONFIRMATION" && -s "$CONFIRMATION" ]] || {
  printf '%s\n' 'refusing: --confirmation must name an existing owner confirmation file' >&2
  exit 2
}
[[ -n "$PROOF" && -f "$PROOF" ]] || { printf '%s\n' 'refusing: existing --proof required' >&2; exit 2; }
[[ -n "$LEDGER" && -f "$LEDGER" ]] || { printf '%s\n' 'refusing: existing --ledger required' >&2; exit 2; }
[[ -d "$WORKTREE" ]] || { printf '%s\n' 'refusing: worktree does not exist' >&2; exit 2; }
[[ ! -e "$WINDOW" && ! -e "$WINDOW.sha256" && ! -e "$WINDOW.complete" ]] || {
  printf '%s\n' 'refusing: quiet-window receipt already exists' >&2; exit 2
}
[[ ! -e "$OUT" ]] || { printf '%s\n' 'refusing: attempt output already exists (started jobs never retry)' >&2; exit 2; }
[[ -x "$WORKTREE/research/bl_k1/run_m1.sh" ]] || { printf '%s\n' 'refusing: run_m1.sh missing' >&2; exit 2; }

deadline_epoch=$(TZ=America/Los_Angeles /bin/date -j -f '%Y-%m-%d %H:%M:%S' "$DEADLINE" '+%s' 2>/dev/null) || exit 2
if [[ "$RUNNER" == check ]]; then
  printf '%s\n' '{"status":"CHECK_ONLY","timed_cells_run":false}'
  exit 0
fi

now_epoch() { /bin/date '+%s'; }
before_deadline() { [[ $(now_epoch) -lt $deadline_epoch ]]; }

# The waiter stays a shell process so the process guard does not reject itself.
busy() {
  /usr/bin/pgrep -ix 'codex|python|python3|pytest|colab' >/dev/null ||
    /usr/bin/pgrep -f 'codex-handoff|codex exec|colab_cli|colab.*dispatch|[Pp]ython|pytest' >/dev/null
}

power_and_lid_ok() {
  local power lid
  power="$(/usr/bin/pmset -g batt 2>/dev/null || true)"
  lid="$(/usr/sbin/ioreg -r -k AppleClamshellState 2>/dev/null || true)"
  [[ "$power" == *'AC Power'* && "$lid" == *'"AppleClamshellState" = No'* ]]
}

# Emit one compact JSON sample and return nonzero unless every §3.1(B) predicate
# holds.  A literal process name is retained for later audit rather than hidden.
sample() {
  local raw=$1 out=$2
  local power lid
  power="$(/usr/bin/pmset -g batt 2>/dev/null || true)"
  lid="$(/usr/sbin/ioreg -r -k AppleClamshellState 2>/dev/null || true)"
  /bin/ps -ww -Ao pid=,ppid=,pcpu=,comm= > "$raw" || return 1
  M1_POWER="$power" M1_LID="$lid" /usr/bin/perl - "$raw" "$out" <<'PERL'
use strict;
use warnings;
use JSON::PP;
use Fcntl qw(O_WRONLY O_CREAT O_EXCL);
my ($raw, $out) = @ARGV;
open my $fh, '<', $raw or die "open $raw: $!";
my (@processes, @violations);
my $parsed = 0;
while (my $line = <$fh>) {
    chomp $line;
    unless ($line =~ /^\s*(\d+)\s+(\d+)\s+([0-9.]+)\s+(\S.*)$/) {
        push @violations, {rule => 'unparseable_process'};
        next;
    }
    $parsed++;
    my ($pid, $ppid, $cpu, $comm) = ($1, $2, 0 + $3, $4);
    my $args = $comm =~ /node/i ? qx{/bin/ps -ww -p $pid -o args=} : '';
    my $lower = lc $comm;
    my $daemon = $lower =~ /fileproviderd|corespotlightd|mds_stores|mdworker|mediaanalysisd|photoanalysisd|\bbird\b|\bcloudd\b|backupd|nsurlsessiond|onedrive|google drive|dfsfileproviderextension/;
    my $agent = $lower =~ /codex|colab_cli|colab.*dispatch|python|pytest/;
    my $other_node = $lower =~ /node/ && $comm !~ /claude code/i && $args !~ /claude code/i;
    my $limit = $daemon ? 5 : 20;
    push @violations, {pid => $pid, ppid => $ppid, pcpu => $cpu, comm => $comm,
                       rule => $agent || $other_node ? 'agent_alive' : ($daemon ? 'daemon_ge_5' : 'other_ge_20')}
        if $agent || $other_node || $cpu >= $limit;
    push @processes, {pid => $pid, ppid => $ppid, pcpu => $cpu, comm => $comm, args => $args}
        if $daemon || $agent || $other_node || $cpu >= 20;
}
close $fh;
push @violations, {rule => 'empty_process_screen'} unless $parsed;
my $ac_power = $ENV{M1_POWER} =~ /AC Power/ ? JSON::PP::true : JSON::PP::false;
my $lid_open = $ENV{M1_LID} =~ /"AppleClamshellState" = No/ ? JSON::PP::true : JSON::PP::false;
push @violations, {rule => 'power_or_lid'} unless $ac_power && $lid_open;
my $value = {timestamp_utc => scalar(gmtime) . 'Z', processes => \@processes,
             ac_power => $ac_power, lid_open => $lid_open,
             quiet => @violations ? JSON::PP::false : JSON::PP::true, violations => \@violations};
sysopen my $json, $out, O_WRONLY | O_CREAT | O_EXCL or die "create $out: $!";
print {$json} JSON::PP->new->canonical->encode($value), "\n";
close $json;
exit(@violations ? 1 : 0);
PERL
}

strict() {
  busy && return 1
  power_and_lid_ok || return 1
  # This is the existing rearm screen, deliberately separate from (and less
  # demanding than) the completed measured window above: named processes over
  # 20 percent refuse the screen.  It does not run Python, so it cannot make
  # its own process guard fail.
  /bin/ps -ww -Ao pcpu=,comm= | /usr/bin/awk '
    $1 > 20 && $2 ~ /fileproviderd|bird|cloudd|nsurlsessiond|mediaanalysisd|photoanalysisd|mds_stores|mdworker|spotlight|codex|node|[Pp]ython/ { bad = 1 }
    END { exit bad }
  '
}

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/m1-attempt06-window.XXXXXX")

# The owner requested this flag before the measured window.  It prevents new
# agent work while the waiter is collecting its admission evidence.
[[ ! -e "$FLAG" ]] || { printf '%s\n' 'refusing: mac timing is already armed' >&2; exit 2; }
( set -o noclobber; : > "$FLAG" ) || { printf '%s\n' 'refusing: could not create timing flag' >&2; exit 2; }
printf 'm1-attempt-06 %s\n' "$(/bin/date '+%F_%T')" >> "$FLAG"
trap 'rm -rf "$tmpdir" "$CANARY"; rm -f "$FLAG"' EXIT
rm -f "$CANARY"

consecutive=0
sample_no=0
while [[ $consecutive -lt 60 ]]; do
  before_deadline || { printf '%s\n' 'refusing: clean window did not finish before deadline' >&2; exit 3; }
  [[ $consecutive -ne 0 ]] || window_started_epoch=$(now_epoch)
  sample_no=$((sample_no + 1))
  raw="$tmpdir/sample-$sample_no.ps"
  json="$tmpdir/sample-$sample_no.json"
  if busy || ! sample "$raw" "$json"; then
    if [[ -e "$CANARY" && $consecutive -ge 30 ]] &&
      /usr/bin/grep -Eq '"comm":"[^"]*(corespotlightd|mdworker)[^"]*"[^\n]*"rule":"daemon_ge_5"' "$json"; then
      printf '%s\n' 'A2 not effective: Spotlight activity reached 5 percent after write canary; resetting window' >&2
    fi
    consecutive=0
    sample_no=0
    rm -f "$tmpdir"/*.ps "$tmpdir"/*.json
    rm -f "$CANARY"
    sleep 10
    continue
  fi
  consecutive=$((consecutive + 1))
  if [[ $consecutive -eq 30 ]]; then
    CANARY_SOURCE="$EXP/bl-m1-attempt-05/cpu-w1-k1.discarded-2.json"
    [[ -f "$CANARY_SOURCE" && $(/usr/bin/stat -f '%z' "$CANARY_SOURCE") -ge 5242880 ]] || {
      printf '%s\n' 'refusing: required >=5MB Spotlight canary source is unavailable' >&2; exit 2
    }
    /bin/cp "$CANARY_SOURCE" "$CANARY"
  fi
  sleep 10
done

# `sample` has evaluated the 30 post-canary samples.  Publish only the successful
# 60-sample receipt, then its sidecar and final completion marker.
/usr/bin/perl - "$tmpdir" "$WINDOW" "$CONFIRMATION" "$window_started_epoch" "$(now_epoch)" <<'PERL'
use strict;
use warnings;
use JSON::PP;
use Fcntl qw(O_WRONLY O_CREAT O_EXCL);
my ($dir, $out, $confirmation, $started, $finished) = @ARGV;
die "window shorter than 600 seconds" if $finished - $started < 600;
my @samples;
for my $number (1 .. 60) {
    open my $fh, '<', "$dir/sample-$number.json" or die "missing clean sample $number";
    local $/;
    push @samples, decode_json(<$fh>);
    close $fh;
}
my $value = {schema => 'bl-m1-attempt06-quiet-window/v1', duration_seconds => 0 + $finished - $started,
             started_epoch => 0 + $started, finished_epoch => 0 + $finished, required_duration_seconds => 600,
             sample_interval_seconds => 10, sample_count => 60,
             confirmation_path => $confirmation, spotlight_canary_sample => 30,
             samples => \@samples};
sysopen my $fh, $out, O_WRONLY | O_CREAT | O_EXCL or die "create $out: $!";
print {$fh} JSON::PP->new->canonical->pretty->encode($value);
close $fh;
PERL
[[ ! -e "$WINDOW.sha256" && ! -e "$WINDOW.complete" ]] || exit 2
( cd "$(/usr/bin/dirname "$WINDOW")"; set -o noclobber; /usr/bin/shasum -a 256 "$(/usr/bin/basename "$WINDOW")" > "$(/usr/bin/basename "$WINDOW").sha256" )
( set -o noclobber; printf 'complete\n' > "$WINDOW.complete" )
rm -f "$CANARY"

# Existing rearm screen remains three separated samples after the new receipt.
for _attempt in 1 2 3; do
  before_deadline || { printf '%s\n' 'refusing: strict screen crossed deadline' >&2; exit 3; }
  for strict_count in 1 2 3; do
    before_deadline || { printf '%s\n' 'refusing: strict screen crossed deadline' >&2; exit 3; }
    busy && { printf '%s\n' 'refusing: agent process became active during strict screen' >&2; exit 2; }
    strict || { printf '%s\n' 'refusing: strict screen failed; a fresh 600-second window is required' >&2; exit 2; }
    [[ $strict_count -ge 3 ]] || sleep 10
  done
  before_deadline || { printf '%s\n' 'refusing: dispatch crossed deadline' >&2; exit 3; }
  busy && { printf '%s\n' 'refusing: agent process became active before dispatch' >&2; exit 2; }
  cd "$WORKTREE"
  log="$EXP/bl-m1-attempt-06.try$_attempt.log"
  [[ ! -e "$log" ]] || { printf '%s\n' 'refusing: dispatch log already exists' >&2; exit 2; }
  ( set -o noclobber; : > "$log" ) || exit 2
  set +e
  research/bl_k1/run_m1.sh --run --fixtures "$FIXTURES" --reference "$REFERENCE" \
    --ledger "$LEDGER" --judged-recovery "$PROOF" --out "$OUT" >> "$log" 2>&1
  rc=$?
  set -e
  printf '%s EXIT=%s m1-06-try%s\n' "$(/bin/date '+%F %T')" "$rc" "$_attempt" >> "$CHAIN"
  [[ -e "$OUT" ]] && exit "$rc"  # Started: never retry.
  /usr/bin/grep -q 'idle Mac receipt' "$log" || exit "$rc"
  # A refused-before-start retry needs a wholly fresh 600-second window, so do
  # not reuse the already-published create-only receipt.
  printf '%s\n' 'refusing: idle receipt failure requires a new waiter invocation and window path' >&2
  exit "$rc"
done
