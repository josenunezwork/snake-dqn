# P1 Y3 snapshot byte-bound repair — 2026-10-22

Mechanical preparation only. No Colab job was dispatched. The scientific
recipe, seed, fit target, actor endpoint, sampling, gates, snapshot cadence and
maximum two outcome-blind completion retries are unchanged. New artifacts are
siblings; prior registrations, manifests, receipts and exports remain immutable.

`byte_bounds.py` derives base payload sizes from the frozen collector shape,
row layouts, reservoir capacities and transition maxima. The owner multiplier
is applied once: effective bytes = 2 × base bytes. `run.py` uses the primary
allowance at every periodic, pause/final, interrupted and tripwire snapshot;
`recovery.py` uses the immutable component allowances and verifies reused files.
The separate fixture bank is transported as an immutable component too.

The ordinary bank is a 10,000-row reservoir, not a 2M-row bank. The refit bank
retains at most 20,000 completed samples plus 2 × 128 × 9 open-life samples across the natural and supplement namespaces. The
primary contains the full simulator graph: body arrays alone occupy
128 × 6 × 10,400 × 2 × 8 = 127,795,200 bytes before serialization. Protocol-2
NumPy buffer expansion is budgeted. The prior 160 MiB primary allowance did not
cover the complete production shape.

For 128 worlds, IDs divisible by five give 26 held-out and 102 fit worlds.
The registered 4M fit ceiling therefore permits ceil(4,000,000 × 128 / 102)
+ 128 × 128 = 5,035,992 raw transitions. The 2M fit target corresponds to a
conservative 2,526,188 raw-transition maximum by the same formula. Tests evaluate
950,272, 2M raw, 2M fit and 4M fit sizes using integers only.

| Component | Base bytes | Effective bytes (2×) |
| --- | ---: | ---: |
| Ordinary immutable | 290,120,000 | 580,240,000 |
| Refit immutable | 91,357,184 | 182,714,368 |
| Fixture immutable | 91,357,184 | 182,714,368 |
| Full primary snapshot | 699,751,208 | 1,399,502,416 |
| One complete closure | 1,172,585,576 | 2,345,171,152 |

The existing qualified supervisor interprets `max_export_bytes` **per file**.
That field remains 2,147,483,648 bytes, and every component fits. The repair preserves the qualified per-file ceiling; the current transport
qualification is identified below. An additional manifest field,
`max_total_export_bytes`, is required to express an aggregate bound without
changing the qualified supervisor's field semantics. The P1 runtime exporter
enforces it before publication, including payloads, SHA sidecars and markers.

For the 300-minute source envelope and 900-second cadence, retain at most
floor(300 × 60 / 900) = 20 periodic captures, three terminal captures and three
transported closures: 26 closure allowances. Ancillary base allowance is
49,215,616 bytes. Aggregate base = 26 × 1,172,585,576 + 49,215,616 =
30,536,440,592; effective aggregate = 61,072,881,184 bytes. These are conservative
storage ceilings, not measured size or runtime forecasts.

The download catalog can require origin, source continuation and latest
operational closures: 3 × 2,345,171,152 + 65,536 = 7,035,578,992 bytes. Its base
rounded allowance is 4 GiB; the effective allowance is 8 GiB. Old catalogs retain
their 2 GiB aggregate default. New catalogs must explicitly declare
`max_total_bytes`, bind their digest in the control, and pass the control to
`p1_inputs --control`; individual downloads remain limited to 2 GiB.
The 6 MiB compressed / 64 MiB unpacked input bundle and 300-second upload
transport / 150-second normal forecast limits remain enforced. Large snapshot
payloads use runtime downloads, not Contents uploads. The new pack has no Drive
backup; adding `backup_dir` is refused without a separate staging allowance.

## Recovery verdict

The failed attempt at
`source-latebody-05-kernel-watchdog-repin-20261022/family-canary/source-output`
has five verified origin payloads totaling 152,518,321 bytes. It has **no full
snapshot and no operational snapshot**. Those partial banks cannot restore the
policy, optimizer, collector, counters or RNG. No exact recovery command exists
for this attempt. The replacement source must start fresh with an empty ledger.

The terminal receipt verifies stop and has no cleanup error. Failure occurred at
950,272 raw/hero transitions, 757,248 fit transitions, 58 updates and zero actor
transitions. This is incomplete mechanical evidence, not a scientific verdict.
The new debit includes all six paid predecessor terminal receipts: 5.55u conservatively charged; canary + source + replacement reservations bring liability to 21.03u, leaving 4.47u. Caps,
reserves, timeouts and the 25.50u shared envelope stay unchanged; actual residual
is recomputed before source admission.

Existing interrupted-refit and actor exact-resume tests still exercise complete
snapshots. For a future complete operational export, the offline ledger command
is (it does not itself resume or dispatch a job):

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_ppo.p1_resume \
  --exports /absolute/verified/exports --ledger /absolute/fresh/verified-ledger.json \
  --namespace bl_p1_y3_20261022 --allocation source_latebody_05
```

## Operator execution

The sibling preparation index records the exact executable commit, artifact
hashes, tests, independent review and upload qualification. Use that pin, not an
unreviewed later working tree. Refresh account inventory and confirm the frozen
rate before each dispatch. Claude/operator alone runs the following commands:

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-p1snap
PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-05-snapshot-bounds-staging-repin-20261022
ACCEPTANCE=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/canary-staging-requalification-20261022-b/requalify-gljfvzqa/acceptance-1792710644/acceptance-result.json
cd "$REPO"
"$REPO/tools/ops/dispatch.sh" p1-y3-05-snapbound-staging-canary-20261022 \
  "$PACK/family-canary/canary-manifest.json" --broker "$REPO" --cwd "$REPO" \
  --dispatch-txt "$PACK/DISPATCH-snapshot-bounds.txt"

# Continue only after a genuine current-pin passing canary, verified exports and stop.
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_ppo.p1_family_canary admit \
  --root "$PACK/family-canary" --receipt "$PACK/family-canary/canary-output/receipt.json" \
  --acceptance "$ACCEPTANCE" --review "$PACK/independent-artifact-review.json"
"$REPO/tools/ops/dispatch.sh" p1-y3-05-snapbound-staging-source-20261022 \
  "$PACK/family-canary/source-admitted.json" --broker "$REPO" --cwd "$REPO" \
  --dispatch-txt "$PACK/DISPATCH-snapshot-bounds.txt"
```

No tmux session, released checkpoint, serving pin, or pre-existing hash-bound
file was modified. No allocation, retry, runtime release, serving change or
promotion was performed by this repair.

Verification: whole-package `pytest -o addopts='' -q research/bl_ppo`: **235 passed in 270.99s**.
Dispatch/transport admission plus updated arithmetic/debit regression batch: **115 passed in 31.63s**.
Black with line length 100, isort with the Black profile, flake8 on all 18 changed
Python files, shell syntax and diff whitespace checks passed. The independent
Codex-V receipt records separate heavy replay, final arithmetic and lint evidence.

Prepared executable pin: `2a207aba5356d7aee79b2e6ce94a9f01d410b328`.
Both templates pin that published commit; subsequent handoff/index publication
changes no reviewed executable source.

A concurrent staging-accounting transport fix was merged from `origin/main`
and was already requalified by its operator. The staging-repin sibling uses
acceptance `acceptance-1792710644` and transport tree
`553af3f692515b2bf66173356fe4d099acc96942`. The earlier snapshot-bound pack and
preparation index remain immutable historical preparation. The new sibling
index ending `_02.json` supersedes only dispatch pointers and transport binding;
all 33 reviewed study source identities, byte bounds, ledger and debit match.

Post-merge staging checks: **6 passed, 72 deselected**; both warmup cases
**2 passed** after replacing a brittle exact duration comparison with
`pytest.approx(135)` (observed difference 2.84e-14 seconds). This test-only fix
does not alter executable source or transport qualification. Fresh normal
upload forecast is 67.736836s; fallback + fetch is 249.061567s < 300s.
