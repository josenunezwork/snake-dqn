# P1 E8 amendment 04 preparation handoff

Status: **HOLD — RETURN TO JUDGE / NO DISPATCH**, 2026-10-22 PT. Authority is the
round-18 **RATIFY-WITH-CHANGES** judgment at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-18/JUDGMENT.md`.
This package implements C1/C2/C3 prospectively on `5afeb8d9` in branch
`p1-snapbound-20261022`. All prior amendments and receipts retain their bytes.

`E8_LATE_BODY_AMENDMENT_04.md` freezes a float64 Helmert zero-sum solve on
`[1,zQ,h]`, unchanged fp32 PopArt installation, actor-phase fixed moments,
and fresh identity-bound collection offsets with burned-root exclusions.
The amendment-03 opt-in remains a separate historical recipe. Enabling both
03 and 04 refuses admission.

The new offline replay uses FIT criteria only. D1 held-out data
are consumed development evidence, never solver/tolerance selection evidence.
The first fixture's bias-shift check used precomputed features, making that
check vacuous. The first corrected full-forward fixture exceeded the installed
PopArt-scale displacement bound. The implementation lane then eased synthetic
targets and rewards and published `_02` as PASS; Codex-V rejects it as acceptance
evidence. Both receipts remain unchanged, and the original failed configuration
is retained for mechanical reproduction. No qualifying C2 PASS is claimed.

The sole replay reports FIT numerical acceptance, but its recorded source hash
predates the later rank-deficient report serialization fix and its numerical
thread environment is unset. It is not a final-code, single-thread qualification.
No second replay or held-out-driven variant search is authorized here.
The judgment's preparation kill rule therefore applies: no spend, return to
a judge with the independent receipt. A future critic PASS would establish
initialization adequacy, not late-body policy gain.

Mechanical failure reproduction `P1_E8_04_FIRST_STEP_FIXTURE_03.json`: all
three real actor-enabled updates and gradients remain finite, but cumulative
held-observation value displacements are 7.69897652, 14.62614822 and 20.38763809
against installed PopArt scale 5.53376055. The analytic common-mode shift is
0 and the actual full-forward uniform 1e-4 LayerNorm-bias shift is 2.861e-6
return units. Finiteness and common-mode checks pass; the stability bound fails.

C3 ledger evidence lives in `e8_04_allocation/`: source offset 3001000,
canary offset 3000000, both 128 worlds over episodes 0..16. Their audited
sets have zero burned-root and cross-lane collisions. Burned-root coverage
includes all 384 episode seeds observed in accessible source-05 operational
snapshots (zero missing). Historical source-02/03 episode state and source-04
status remain limited/unavailable as itemized in the coverage artifact;
conservative re-derivation is not a complete historical inventory. A full C3
acceptance PASS is therefore not claimed. The protected study was not opened.

`P1_E8_04_FORECAST.json` records the judgment's prices: canary 15min estimate,
30min effective timeout, 2.00u effective cap with .40u included reserve;
source 75min estimate, 300min effective timeout, 9.62u cap with 1.92u included
reserve. Both L4, `cuda:0`, rate 1.54u/h and zero tolerance. Owner 2x is already
applied once. Frozen 25.50u envelope, 7.26u charged, 11.62u liability leaves 6.62u,
or 2.76u holding the unadmitted 3.86u replacement. The 1M qualification needs
separate judge pricing after a genuine Y3 PASS.

Dispatch packs use the qualified Colab CLI tree
`553af3f692515b2bf66173356fe4d099acc96942`. Qualification registry
`snake-dqn-artifacts/ops/colab-cli-qualified.json` points to acceptance
`acceptance-1792710644`, receipt SHA256
`8b79a9bf264bf3f8900e8bf2e919882f8ff02e03be850f7954e32e9bff0e03c1`.
The receipt hash, PASS and qualified=true were checked locally. No transport
source changes are part of this package.

The operator must discharge every judgment section 6 condition in order.
Claude checks every independent receipt acceptance line and records that
check in CAMPAIGN_LOG.md. Current code/test evidence cannot substitute for
the future live canary, inventory, debit, rate and Mac-window checks.
The prior `WARNING_UNEXPLAINED_DEBIT` remains carried forward. An empty map
in manifest metadata is a requested admission constraint, not live inventory.

Software verification (distinct from the failed C2 acceptance): root's primary
venv pytest 9.0.1 run passed **108 tests in 19.36s**, single process with all
numerical thread caps 1:

```bash
env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -o addopts='' -p no:xdist -q research/bl_ppo/test_refit.py research/bl_ppo/test_offline_critic_convergence.py research/bl_ppo/test_p1_runtime.py research/bl_ppo/test_p1_collection_allocation.py research/bl_ppo/test_e8_04_preparation.py research/bl_ppo/test_p1_manifests.py research/bl_ppo/test_p1_canary_runtime.py research/bl_ppo/test_p1_operational_resume.py research/bl_ppo/test_recovery.py research/bl_ppo/test_model_learner.py research/bl_ppo/test_p1_entrypoints.py research/bl_ppo/test_p1_family_canary.py
```

Black check (100 columns), isort check (Black profile, 100 columns), flake8
over all 15 changed Python files, `bash -n research/bl_ppo/p1_colab.sh`, and
`git diff --check` pass. No full suite, CUDA run or Colab test was performed.
An initial root pytest command named nonexistent `test_p1_resume.py` and
`test_learner.py`; it collected no tests and was corrected to the command above.
The implementation lane's earlier pytest-7 environment report was not the
primary venv; root and reviewer verified the primary venv is pytest 9.0.1.

## Published execution pin and review-only packs

Executable pin: `f184c2c7f54e642b5870d6efd4a32b60ffe50496`, pushed to the
candidate branch and origin/main. Later ledger/review-only commits do not
change executable bytes or transport. The actual immutable pack directory is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-06-e8-04-review-20261022-02`.
`P1_E8_04_DISPATCH_PACKS.json` binds all payloads and sidecars. Both canary
and source have top-level and nested `scientific_admission=false` and HOLD
instructions. No `source-admitted.json` exists. Shared upload inputs are the
117386-byte archive plus 2409-byte public catalog; source/canary inventories
are byte-identical while their controls, offsets and empty ledgers differ.
The first partial preparation used a root-supplied abbreviated commit and
refused upload qualification. It is preserved at the sibling without `-02`.

## Judgment section 6 checklist

| Item | Status |
| --- | --- |
| 1. Code merged; independent receipt every line, PASS and no unresolved; Claude log check | Code published; **NOT MET**: C2 FAIL, review HOLD; Claude verification pending. |
| 2. Final-code FIT replay with sidecars published | Numerical FIT PASS published; **NOT MET**: stale learner/helper binding and unset thread environment. |
| 3. Allocation audit, zero collisions, coverage, distinct offsets | Enumerated E16 allocations clear and published; **LIMITED** historical inventory, full acceptance not met. |
| 4. Qualified proof/current/pin and cuda:0 | Qualified CLI tree/current/pin verified; cuda:0 declared; live family-canary proof remains pending. |
| 5. Fresh control, source06 namespace/allocation, ledger, artifact review | Review-only dual controls and empty ledgers built; independent review records HOLD. Admission false. |
| 6. Immediately-before-dispatch usage/sessions/rate/debit and warning | **PENDING** operator live check; no provider query by this task. Judgment7.26u is frozen history; inherited5.55u debit is explicitly stale. WARNING_UNEXPLAINED_DEBIT retained. |
| 7. Mac windows and never-idle | Local checks finished before Thu20:00 X5; any future timing/window admission remains operator work. No Colab reserved. |
| 8. Operator-only wrapper, owner release, protected boundaries | **PRESERVED**: no dispatch; B8 s3 unchanged, protected study closed, l2mac untouched. |

No executable dispatch command is authorized under this HOLD. A new judge
disposition and a fresh accepted preparation are required before an operator
may dispatch. The exact wrapper invocations below identify the reviewed
targets but **refuse now** on the immutable HOLD/admission fuses:

The independent `P1_E8_04_CODEX_V_REVIEW.json` (with SHA/completion sidecars)
records all 26 acceptance clauses and eight preconditions. Its status is
`HOLD_RETURN_TO_JUDGE`, with unresolved admission findings V1, V2 and V3;
artifact review passes only for the explicitly blocked review packs.

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-p1snap
PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-06-e8-04-review-20261022-02
"$REPO/tools/ops/dispatch.sh" p1-e8-04-06-canary "$PACK/canary-manifest.json" --broker "$REPO" --cwd "$REPO" --state-dir "$PACK/state-canary" --dispatch-txt "$PACK/DISPATCH.txt"
"$REPO/tools/ops/dispatch.sh" p1-e8-04-06-source "$PACK/source-template.json" --broker "$REPO" --cwd "$REPO" --state-dir "$PACK/state-source" --dispatch-txt "$PACK/DISPATCH.txt"
```

No Colab/provider operation, allocation, dispatch, tmux action, serving pin
or released-checkpoint change is authorized by this handoff. B8 s3 remains
served. The protected study stays closed and `snake-dqn-l2mac` is untouched.
