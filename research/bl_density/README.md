# DENSITY-2 — fresh natural-arm replication

Authority is [METRICS_AMENDMENT_04.md](METRICS_AMENDMENT_04.md), implementing the
full spare-slot judgment. The original METRICS and amendments 01–03 remain
immutable. The 90-episode pilot supplies sizing rates only and is excluded from
all replication statistics and decisions.

The fixed 492-episode allocation is A frozen/mixed/scripted 96/72/72; B70m 24 per
mix; B70p 36 per mix; C-P1 24 per mix. The namespace is
`bl-density-round14b-20261022`; twelve excluded canary episodes use
`bl-density-round14b-canary-20261022`. The allocator collision-checks all three
banks before gameplay. Production uses ten single-thread CPU workers on L4,
H5000, stride 200, max work 10,800 seconds, session <=180 min and cap <=5 units
including the 1-unit reserve. First cap wins. The tiny canary is <=.5 units
including .1 reserve and <=20 min. Setup and all attempts remain charged.

Scientific rule code in analysis.py and statistics.py is byte-identical to the
pilot. B8 bytes, CPU inference, features, prefilter, arm constructions and all
thresholds are unchanged. One terminal look. A mix with fewer than four new-A
doom-bearing episodes after the full allocation is INSUFFICIENT and closes
rows 9/9a for that mix this campaign. No third enlargement or pilot pooling.
Mechanical failure permits one newly prepared attempt within remaining caps;
no resume or automatic retry. An admission only permits registration of a
separate A-transfer test, never training, P1 or serving changes.

`manifests.py` prepares source-pinned family-canary and production templates.
The canary retains the complete actual input inventory and real entrypoint.
`upload_readiness.py --receipt ...` qualifies real pinned sparse source and
fallback sizes through Z-9 without provider calls. Both phases declare the
producer's `density/summary.json` export. Recovery additionally verifies all
twelve canary episode sidecars, stage CPU counters and backend release.
Production binds the genuine broker family receipt and recovered package proof;
its upload inventory remains identical to the canary's. No receipt is invented
by preparation. The validation report distinguishes schema/structure checks
from live admission prerequisites.

The family gate now scopes CUDA observations to jobs that declare CUDA. DENSITY-2
declares `SNAKE_DQN_DEVICE=cpu`, so its L4 canary still requires the real
entrypoint, full upload inventory, scanner-recovered export sidecars and verified
release, but not an invented CUDA decision. The transport code change invalidates
the old final-only acceptance. Until the exact changed source passes a new live
final-only acceptance, preparation is **PENDING_REQUALIFICATION**, never a PASS
and never scientific admission. No Colab calls or gameplay runs occur during
preparation. Commands in the delivery DISPATCH.txt are conditional operator
handoff commands, not evidence of execution.

Use the following create-only sequence after the gatefix branch is published.
It prepares both CPU-on-L4 templates while honestly retaining disabled runtime
admission. `SHA` is the published branch commit to requalify and `PACKET` must be
a fresh, absent artifact directory.

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
SHA=<published-gatefix-commit>
PACKET=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/density2-20261022-gatefix

"$PY" -m research.bl_density.manifests \
  --commit "$SHA" --phase canary --output-dir "$PACKET/canary-output" \
  --manifest "$PACKET/canary-pending.json" \
  --validation-report "$PACKET/canary-pending-validation.json" \
  --pending-requalification
"$PY" -m research.bl_density.manifests \
  --commit "$SHA" --phase production --output-dir "$PACKET/production-output" \
  --manifest "$PACKET/production-pending.json" \
  --validation-report "$PACKET/production-pending-validation.json" \
  --pending-requalification
```

After the exact commit is published on `origin/main` and Claude has completed the
fresh final-only acceptance described in `research/colab_cli/REQUALIFICATION.md`,
point admission at that real acceptance result. This is still offline validation;
it does not allocate. The canary command is the only operation that can set its
runtime guard to `BL_DENSITY_SCIENTIFIC_ADMISSION=1`:

```bash
ACCEPTANCE_RESULT=<fresh-final-only-acceptance-result.json>
"$PY" -m research.bl_density.manifests \
  --admit-template "$PACKET/canary-pending.json" \
  --manifest "$PACKET/canary-admitted.json" \
  --validation-report "$PACKET/canary-admitted-validation.json" \
  --source-qualification-receipt "$ACCEPTANCE_RESULT"
```

Only after a genuine canary run is recovered and its package proof and family
receipt are verified may production be finalized and admitted. The declared
`prior_spend_units` must cover the receipt's measured canary charge; production
remains disabled if any of those checks fail.

```bash
"$PY" -m research.bl_density.manifests \
  --admit-template "$PACKET/production-pending.json" \
  --manifest "$PACKET/production-admitted.json" \
  --validation-report "$PACKET/production-admitted-validation.json" \
  --source-qualification-receipt "$ACCEPTANCE_RESULT" \
  --canary-proof "$PACKET/canary-proof.json" \
  --canary-receipt "$PACKET/canary-output/receipt.json" \
  --prior-spend-units <measured-canary-charge-rounded-up>
```

Use `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`. Run only targeted
package tests with CPU and one numerical/Numba thread. Prepared artifacts live
under `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/`;
the DENSITY-2 delivery receipt identifies the fresh directory and exact hashes.
