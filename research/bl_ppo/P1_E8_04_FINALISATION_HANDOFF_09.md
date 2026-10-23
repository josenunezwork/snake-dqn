# P1 Y3 source finalisation repair — 2026-10-23

Verdict: **CODE FIXED; OFFLINE SOURCE COMPLETION UNAVAILABLE; DISPATCH HELD.**
The critic's existing PASS remains intact. No collection, refit, held-out gate,
actor replay on production exports, or Colab allocation was run for this repair.
The original supervisor receipt remains INCOMPLETE / CHILD_NONZERO_EXIT with
verified stop. A source completion receipt cannot be reconstructed from the
available evidence under the unchanged registration.

## Root cause and repair

The fresh source ledger is empty. Its origin phase writes the exact pre-actor
pause with `actor_enabled=true` and top-level `execution.phase="actor"`, but
without `operational` metadata. `run_train` restores that execution state and
previously used the phase alone to seed a periodic action suffix. The source
continuation consequently met the three-update/episode requirements after
49,152 actor transitions, before its 900-second periodic cadence. It wrote an
operational trace naming the exact origin, but no actor operational snapshot.
`p1_runtime` correctly rejected the empty operational snapshot list at finalisation.

`run.py` now seeds a resumed periodic suffix only from an actual OPERATIONAL
snapshot with the declared namespace/allocation and actor execution phase.
Ordinary exact-pause execution restoration is retained. The finaliser also
verifies the latest actor anchor's label, allocation, digest, trace binding,
transition interval, three subsequent updates and one episode boundary.

The regression delays the first actor periodic capture until update four. The
exact origin cannot satisfy the periodic requirement; after a genuine capture,
at least three further updates and the registered episode/action suffix must
complete. Existing interruption tests still cover refit and actor operational
resumes, full action tape equality and complete semantic state fingerprints.
Direct negative tests reject an exact pause and a refit anchor even when their
export hashes and trace metadata are internally consistent.

## Evidence and recovery boundary

Original pack:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-06-e8-04-review-20261022-08/final`.

The three periodic snapshots at 933,888 / 1,687,552 / 2,408,448 transitions are
all **refit** snapshots with zero actor transitions. The origin pause is exact
and unlabelled. The final continuation pause is a durability boundary with
49,152 actor transitions, three updates and 15 episode boundaries. Its saved
`operational-action-trace.json` names the unlabelled origin. These bytes cannot
prove the registered replay from an actor OPERATIONAL snapshot captured during
source continuation. Neither copying nor relabelling an origin/refit snapshot
can supply the missing evidence. A Mac replay would create new gameplay and
would not substitute for the required live L4 proof.

The create-only audit is:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-finalisation-09-20261023/offline-audit-02.json`.
It verifies the snapshot closures and eleven receipt identities/hashes, reports
`HOLD_MISSING_OPERATIONAL_PROOF`, retains `critic_qualified=true`, and explicitly
does not write a source receipt. SHA-256:
`738c4a5b959b81acfcbb2ea9cd0ebc881c1b090d70a9404454a53924a5427a33`.

To reproduce into a **fresh** sibling (the command refuses to overwrite):

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-p1snap
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_ppo.p1_source_audit \
  --pack /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-06-e8-04-review-20261022-08/final \
  --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-finalisation-09-20261023/offline-audit-claude.json
```

## Exact remaining liability and ruling required

| Item | Units |
| --- | ---: |
| Ten frozen predecessor charges | 7.64 |
| Failed source own elapsed charge | 2.372971089306343 |
| Failed source gross debit / conservative charge | 2.47 |
| Charged to date | **10.11** |
| Remaining from frozen 25.50 u | **15.39** |
| New exact-pin canary cap (30 min; reserve included) | 2.00 |
| Full source retry cap (300 min; reserve included) | 9.62 |
| Reserved replacement cap (120 min; reserve included) | 3.86 |
| All-in liability | **25.59** |
| Shortfall | **0.09** |

The rate ceiling remains 1.54 u/h with zero tolerance. Caps/timeouts already
include the owner's 2x rule and must not be doubled again. The source's original
75-minute estimate is not a measured estimate for a newly designed repair.
`WARNING_UNEXPLAINED_DEBIT` is carried forward; conservative charges are not
refunded by substituting own spend.

Even without the new canary, the child finalisation error after the critic
outcome became known is outside round-18 judgment §5's SUPERVISOR/transport,
outcome-blind retry permission. Amendment 05 also requires a fresh exact-pin
canary after executable changes. No full-cap retry is admitted. A judge must
rule on a separately priced mechanical continuation preserving the qualified
pause and completed science, rather than assume permission to rerun refit or
the critic gate. This handoff does not propose a new scientific look.

## Next-stage boundary

Replacement remains dependent on a genuine `COMPLETE_SOURCE_CONTINUATION`
with the actor operational anchor and its fixed suffix, verified source release,
and a distinct replacement L4 assignment. The 1M qualification remains dependent
on genuine full Y3 PASS plus separate judge pricing; its requested boundary is
1,000,000 and its actual whole-world boundary is **1,000,064**. Its old 4 u base
ceiling is not a newly ratified effective invoice. Neither stage is admitted by
this preparation. Missing prerequisites must be supplied through fresh sibling
artifacts; do not edit existing held packs or their sidecars to remove a fuse.

The held packs, qualified transport pin, wrapper commands, verification results
and independent review are bound by the sibling
`P1_E8_04_FINALISATION_CODEX_V_09.json` receipt. Claude must verify them before
any admission; a new judge ruling is still required for the repair and the
separately priced qualification. B8 serving, released checkpoints, serving pins,
live tmux sessions, prior immutable evidence, the protected study and the
`snake-dqn-l2mac` worktree remain untouched.

Executable pin: **`0e9788872fd2517468a77917e6c69dd2f78e8087`**, published to
`origin/main` and `origin/p1-snapbound-20261022`. The manifest source has the
qualified `research/colab_cli` tree
**`d0d92bdb572b770284a1fea6152417ffa5f06bea`**. The Mac qualification registry
names the same tree and kernel-UUID acceptance `acceptance-1792740334`.

Held pack root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-finalisation-09-20261023/next-stages-final-02`.
Each phase has a production-shaped control, input bundle, empty ledger, upload
preflight, preparation receipt, original manifest and a sibling
`held-manifest.json` with explicit `scientific_admission=false` and
`transport.dispatch_ready=false`. The 1M template's historical numerical
forecast is not an admitted or newly ratified effective price; the held record
requires separate judge pricing and one application of the 2x rule.

Reproduce into a fresh directory:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-p1snap
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_ppo.p1_next_stage prepare \
  --commit 0e9788872fd2517468a77917e6c69dd2f78e8087 \
  --root /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-finalisation-09-20261023/next-stages-claude
```

The following are the held wrapper commands. They intentionally refuse before
operational side effects. After the missing evidence, judge ruling and Claude
verification, create and review **new admitted siblings**; do not modify the
held manifests, commands or sidecars below.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-p1snap
tools/ops/dispatch.sh p1-y3-replacement-next-stage \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-finalisation-09-20261023/next-stages-final-02/y3-replacement/held-manifest.json \
  --broker /Users/josenunez/Projects/ml/snake-dqn-p1snap \
  --cwd /Users/josenunez/Projects/ml/snake-dqn-p1snap \
  --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-finalisation-09-20261023/next-stages-final-02/DISPATCH_NEXT_STAGE.txt
tools/ops/dispatch.sh p1-qual-1m-next-stage \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-finalisation-09-20261023/next-stages-final-02/qual-1m/held-manifest.json \
  --broker /Users/josenunez/Projects/ml/snake-dqn-p1snap \
  --cwd /Users/josenunez/Projects/ml/snake-dqn-p1snap \
  --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-finalisation-09-20261023/next-stages-final-02/DISPATCH_NEXT_STAGE.txt
```

## Verification scope

All pytest invocations use `-p no:xdist`, `SNAKE_DQN_DEVICE=cpu` and
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=VECLIB_MAXIMUM_THREADS=NUMBA_NUM_THREADS=1`.
The Python executable is `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`.

| Exact pytest selection after `-m pytest -p no:xdist -q` | Result |
| --- | --- |
| `research/bl_ppo/test_p1_y3_resume_entrypoint.py research/bl_ppo/test_p1_runtime.py research/bl_ppo/test_p1_operational_resume.py` | 28 passed, including source two-replay and interrupted-refit/actor integration |
| `research/bl_ppo/test_p1_source_audit.py research/bl_ppo/test_p1_runtime.py research/bl_ppo/test_p1_operational_resume.py` | 29 passed after adding direct anchor and accounting regression coverage |
| `tests/test_colab_cli_guard.py tests/test_colab_cli_schema_failures.py` | 66 passed |
| `tests/test_colab_cli_supervisor.py -k 'existing_output_directory_is_refused_as_nonfresh or allocation_cap_counts_existing_notebook_assignments or partial_local_export_without_completion_marker_is_rejected'` | 3 passed, 75 deselected |
| `research/bl_ppo/test_p1_next_stage.py` | 3 passed in 91.33 s; real pack/source upload preflights, no provider calls |

The initial broader transport selection (all three Colab modules above,
144 collected) was deliberately interrupted after 40 passed in 285.47 seconds
to keep verification bounded. It is **not** reported as a completed gate.
The complete scoped selections are reported separately. No transport source
was changed by this repair; no new live Colab qualification is claimed.
Black (100 columns), isort (Black profile), flake8 and `git diff --check` apply
to the changed Python files. The real preparation entrypoint succeeded. Offline
admission-fuse checks on both held manifests refused
`scientific_admission=false` before even creating an operations directory;
the operator shell wrapper was not dispatched. The create-only proof is
`source-finalisation-09-20261023/held-fuse-check-02.json`, beside the pack root.
These checks are bound by the Codex-V receipt.
