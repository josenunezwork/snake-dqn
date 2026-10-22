# DENSITY-2 mechanical repin 04 — prepared, not dispatched

Authority: `docs/research/advice/2026-10-21_big_lever/JUDGMENT_DENSITY2_THIRD.md`
and its artifact original, read in full. The owner's instruction is a mechanical
repin only; science, caps, reserves, estimates, timeouts and thresholds stay fixed.
Pack:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/density2-completion-20261022-04`.

Source pin: `04b55162fabb761ccbd740ead8014e39177d9e67`, merged from origin/main.
Qualified commit: `f21a92edb86550df5263dbed4be7e2ff0bfa4c17`.
Qualified `research/colab_cli` tree: `553af3f692515b2bf66173356fe4d099acc96942`.
Acceptance result:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/canary-staging-requalification-20261022-b/requalify-gljfvzqa/acceptance-1792710644/acceptance-result.json`.
Acceptance SHA256:
`8b79a9bf264bf3f8900e8bf2e919882f8ff02e03be850f7954e32e9bff0e03c1`.
The handoff commit adds only this record and its inventory; executable source
remains pinned to the published commit above. Transport bytes are unchanged.

## Frozen inputs and attempt accounting

Three prior canary allocations, next/final canary **04**, production completion
attempts consumed **0**, next completion **01**, maximum completion attempts **2**.
The obsolete two-canary packaging counter in repin 03 is superseded by THIRD.
Any canary-04 gate, allocation, transport, supervisor or runtime failure is terminal;
no fifth canary or new judgment is authorized. The pack disables the ops wrapper's
optional accounting-allocation retry for canary-04, and atomically reserves its
single invocation before admission. Production retains the existing exact retry.

`skip-ledger.json` remains 408 entries, SHA256
`70b252ec52c7d61c62c419e77e4e837a8dc1c5d120c283a61439d186565f444c`.
`canary-ledger.json` remains
`1fac1b13ebe00a750e7896511967caf573832f6f2265e08385c75151db625aaa`;
`input-hashes.json` remains
`98acd6f9815f6e44dcaca0996f93775b78625dd374ee536bd4790c5c6205e3f5`.
All three files compare byte-identical to pack 03. Science baseline remains
`f1904777aded79366703c94afbf5dc13155bb0ee`, with 492 registered units,
84 missing primary units and 12 non-scoring identity replays. Frozen science
checks pass; no production outcomes or remote summaries were opened.

Operative artifacts are `canary-template-final.json`,
`completion-template-final.json`, and `preparation-final.json`. Earlier same-pack
unsuffixed templates/preparation are immutable, undispatched staging snapshots.
Final siblings add the eight current transitive source imports from the main
merge. This preserves the full executable import closure for both jobs. Other
template differences from pack 03 are commit/environment commit, job/output
identities, qualified receipt path/hash, local paths to identical input bytes,
and prior-spend accounting. Recursive validation rejects every other difference.

Existing owner-2x fields are retained: canary base 0.5 u / 0.1 reserve / 15 min,
effective 1.0 u / 0.2 reserve / 30 min; producer work limit 20 min, estimate 10 min.
Completion base 2.5 u / 0.5 reserve / 90 min, effective 5.0 u / 1.0 reserve /
180 min, estimate 35 min. Series envelope stays 16.50 u and L4 ceiling 1.54 u/h.
The literal THIRD canary prose repeats <=0.5 u / <=20 min; as explicitly requested,
this repin preserves existing effective owner-doubled fields, without presenting
those as a 0.5-u/20-min whole-session cap.

Prior spend before canary-04 is conservatively 4.90 u, above THIRD's 4.84-u
minimum. Admission adds the greater of canary-04 elapsed pricing and receipt debit
evidence; completion-02, if eligible, also adds completion-01 and its verified
accounting-allocation retry charge. Per-job and total caps remain unchanged.
Historical staging diagnostic re-score records 01 FAIL (150.12686575000407 s
transfers), 02 N/A, 03 PASS (21.243038500018883 s transfers, 177.7681104170042 s
whole staging). It is never substituted for new current-pin canary proof.

## Operator commands — not run

```bash
ACCOUNT_MAP=/absolute/path/to/reviewed-current-account-rates.json
/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/density2-completion-20261022-04/DISPATCH.sh canary "$ACCOUNT_MAP"

# Only after recovered live canary-04 PASS and verified release; refresh map.
/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/density2-completion-20261022-04/DISPATCH.sh completion "$ACCOUNT_MAP"

# Only after the registered mechanical failure/release condition; refresh map.
/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/density2-completion-20261022-04/DISPATCH.sh completion-02 "$ACCOUNT_MAP"
```

Names are `density2-skip-canary-04-04b55162`,
`density2-completion-01-04b55162` and `density2-completion-02-04b55162`.
Completion remains gated on a genuine recovered same-pin skip-ledger canary,
verified release and current reviewed account map. All outputs are create-only.
No scoring merge command is included; per-attempt production summaries stay sealed.

**Existing completion-02 limitation:** a grown ledger changes the exact bytes
bound by canary-04. Both existing density proof and qualified family gates reject
it, while THIRD prohibits another canary. The conditional command therefore
supports the case with no newly recovered primary units and an identical ledger;
it explicitly refuses a grown ledger before preparation/dispatch. It never
reuses a stale ledger, fabricates proof, or relaxes admission policy. The command
follows the one exact accounting-allocation retry only after verifying its receipt
and derived manifest. No claim is made that all possible completion-02 cases
are dispatchable under the existing gates.

## Local evidence

`OFFLINE_VALIDATION.py` ran real resume/science/qualification/source snapshot and
`launch.sh validate` admission in a temporary OPS_DIR. Submission was intercepted
at `dispatch.start`, before tmux/provider calls or watcher startup. Production
without real canary proof refused. Canary retry/failed/timeout observations each
refused after one observation; atomic reservation and terminal markers were tested.
Tiny real CPU H8 producer: **COMPLETE, 9 completed, 3 skipped**, 3.859915 total CPU
seconds for the synthetic canary. This is local producer evidence, not live L4 proof.
Five further synthetic cases cover completion-02 original/retry lineage and spend,
grown-ledger refusal on both paths, and foreign retry-manifest refusal.

Exact test commands and counts are recorded in the pack's `TEST_RESULTS.json`.
Black, isort, flake8, Bash syntax, qualification and Git whitespace checks passed.
The inventory in sibling `COMPLETION_REPIN_04.json` binds the final create-only
delivery and validation records. No Colab allocation/dispatch, production merge,
tmux modification, released checkpoint write, serving pin change, prior pack edit,
or hash-bound governance edit occurred.
