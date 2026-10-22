# DENSITY-2 completion operator handoff

Ruling (1) only; amendment 05 is the pre-data registration. Preparation never
allocates Colab. Run the skip-ledger canary first, recover its proof, then admit
and dispatch completion. Keep both remote per-attempt summaries sealed. The Mac
merge is the only scoring look. Do not open production episode JSON beforehand.

The verified original ledger has 408 units, SHA-256
`70b252ec52c7d61c62c419e77e4e837a8dc1c5d120c283a61439d186565f444c`.
The missing set is 84 episodes: all 72 C-P1 and B70p/mixed 022,025–035.
The ruling's 85 is a file count including summary.json; its 024–035 prose range
also differs from actual metadata. Allocation-minus-ledger is binding. Completion
runs these 84 plus A indices 000–003 in each mix as 12 non-scoring replays.

Completion: 35-minute estimate, 180-minute timeout, 5-unit cap including 1-unit
reserve. New skip canary: 10-minute estimate, 30-minute timeout, 1-unit cap including
0.2 reserve. Maximum two completion attempts. The second requires mechanical
failure/release evidence and a grown ledger, and new same-input canary proof.

Use the qualified broker only after `supervisor-telemetry-20261022` has merged.
The preparation pin must include that merge **and** this package's implementation.
If prepared before the transport merge, regenerate into a fresh directory with the
new origin/main SHA; never edit a prepared manifest or its input files in place.
Both package/source admission and ops qualification fail closed on stale transport.
The current account-rate map and exact acceptance-result path are operator inputs;
their values cannot be predicted while requalification and other jobs are running.

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-density
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
BASE=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density
cd "$REPO"
git pull origin main
# Must succeed before preparing the dispatch pin:
git merge-base --is-ancestor origin/supervisor-telemetry-20261022 origin/main
PIN=$(git rev-parse origin/main)
QUALIFICATION=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/telemetry-requalification-20261022/requalify-g4fekqn4/acceptance-1792700171/acceptance-result.json
# Choose a fresh preparation directory; never overwrite the delivered preparation.
PREP="$BASE/density2-completion-qualified-$PIN"
"$PY" -m research.bl_density.completion prepare --commit "$PIN" \
  --directory "$PREP" --ledger "$BASE/density2-completion-ledger-20261022-01.json" \
  --qualified --qualification "$QUALIFICATION"
# Set these to the reviewed live map and new final-only acceptance-result.json:
ACCOUNT_MAP=/absolute/reviewed-current-account-rates.json
"$PY" -m research.bl_density.completion admit \
  --template "$PREP/canary-template.json" --qualification "$QUALIFICATION" \
  --other-assignments "$ACCOUNT_MAP" --manifest "$PREP/canary-admitted.json"
"$REPO/tools/ops/dispatch.sh" "density2-skip-canary-${PIN:0:8}" \
  "$PREP/canary-admitted.json" --broker "$REPO" --cwd "$REPO" \
  --state-dir "$PREP/canary-state"
# Wait for terminal PASS and verified release; then recover proof offline:
"$PY" -m research.bl_density.canary --recovered-output "$PREP/canary-output" \
  --manifest "$PREP/canary-admitted.json" --out "$PREP/canary-proof.json"
"$PY" -m research.bl_density.completion admit \
  --template "$PREP/completion-template.json" --qualification "$QUALIFICATION" \
  --other-assignments "$ACCOUNT_MAP" --canary-proof "$PREP/canary-proof.json" \
  --canary-receipt "$PREP/canary-output/receipt.json" \
  --manifest "$PREP/completion-admitted.json"
"$REPO/tools/ops/dispatch.sh" "density2-completion-01-${PIN:0:8}" \
  "$PREP/completion-admitted.json" --broker "$REPO" --cwd "$REPO" \
  --state-dir "$PREP/completion-state"
```

If account assignments change, repeat offline admission into a new admitted filename
with the reviewed map; preserve the existing prepared templates and prior evidence.
Do not dispatch a completion before successful canary recovery. A CLI exit code alone
is insufficient. Transport and scientific admission are independent gates.
