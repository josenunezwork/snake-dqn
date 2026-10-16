# T7: B8 fresh-bank, one-look Phase R

**RATIFIED 2026-10-16** by the owner's explicit instruction to implement and run
`docs/research/specs/t7_phase_r_b8_spec_2026-10-16.md` exactly, incorporating the binding
advisor choice `docs/research/advice/2026-10-16_phase_r_choice.md` and the single prospective
`docs/research/specs/guard_clause_amendment_2026-10-15.md`.

This registration is committed and pushed before any T7 gameplay. Rule tests and the
implementation receipt must pass before the frozen run plan is created. No smoke gameplay,
interim survival analysis, extension bank, training, serving qualification or release occurs
in this task. The existing served default remains frp3-s12 + v8.

## Fixed candidates and comparators

The candidate is arm B8, no decision-time veto, its five existing 20M-transition finals at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m4b-mac-20261013/B8_s<S>/final.pth`:

| Training seed | SHA256 |
| --- | --- |
| 0 | `bf6a42b387e6d643e943a03620a30128ec6032f984abb5a08559b9b0ac865f4f` |
| 1 | `a7b04a84c94ad6cf90f6461a2ab23b214a28c4c59ac9e7bc69c61ca3ae617753` |
| 2 | `5840ba615efd600325d5c5bcedf87b2cbb452fe0969d954d5797ab12f6917a0e` |
| 3 | `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8` |
| 4 | `6b6d3f37c6db129de941c3fb205ca6af77c6506db0793b1291656f767e4fb28f` |

Use the M4 serving-compatible evaluation path: GridBatchSim, `ego2s-bv8`, rowwise ego2s
and vector61 forwarding, the candidate advisory action mask, and the Python vector61
featurizer. The candidate has no veto; its v8 safety features remain model inputs.
Incumbent = served frp3-s12 + v8, SHA256
`eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723`.
Champion = the existing pinned `dev_screen.CHAMPION` champion_a5 + v8. Both references use
v8 lambda 8 on the hero only and are evaluated on **all five** candidate banks. All model,
opponent and deployment-profile identities are checked and SHA256-bound in the run plan.
Reuse and validate the existing per-final B8 P2-NV evidence (9/9 H5000 and 3/3 H10000 for
each seed) at `redesign-m4-p2-candidates/B8/`; do not rerun it as unregistered gameplay.

## Prospective bank and power choice

Namespace `redesign-t7-phase-r-b8/v1`, round 74, key `round74-seed<S>` for S=0,1,2,3,4.
Each ordered bank is `dev_screen.screen_seeds(192, namespace, key)`. All 960 identities must
be distinct. Scripted uses all 192, frozen and mixed use the first 96; each reference uses
exactly the same worlds, roster identities and full-bank roster construction as its hero.
The five training seeds' banks are independent. Within a bank, the two smaller mixes share
the scripted bank's first 96 seeds, as in the existing paired-bank design.

Before any gameplay, `SEED_AUDIT_T7.json` must pass and be committed. Audit the exact
candidate identities against prior registrations and visible repository/artifact JSON/JSONL
evidence, including dev rounds 50–73 (at least the first 1,024 identities per round), census,
M3/M3-C/M4 Phase R, training/distillation, T3 label and T4 return/snapshot collections.
Explicitly bind their source manifests/namespaces. Missing required evidence or a collision
blocks launch; there is no discretionary seed replacement. Record the audit's coverage and
limitations concerning unrecorded external experiments and opaque binary files.

The **192 scripted + 96 frozen + 96 mixed worlds per seed** are a prospective power choice,
not a threshold change. The reason is the **38.8%** historical combined survival-clause
failure-risk forecast in `docs/research/t5b_risk_and_fuse_2026-10-16.md`; the advisor estimated
about 27% at this size. Those forecasts use existing heterogeneous banks, contain uncertainty,
and are not T7 evidence. No previously observed outcome is re-scored as gate evidence.
There is no reserved extension or increase beyond this N.

## Horizons, one look and exact decision rule

Each of the 5,760 hero/reference episodes runs to H10000 with a terminal hero and an exact
H5000 prefix, using the M4 profiles (H5000 digest `d396d3ed93e3a264d050674887eb47e59de67d3c6696c2c41fa5f8b145ea0e8b`,
H10000 digest `c882e07fb7179c01feed5380aeae228113975442bb10ab8840bc58a0011f5b15`).
The first world of each hero × seed × mix also has a direct-H5000 prefix-identity control:
45 controls, excluded from all decision estimates. Compare M4's exact prefix fields, ignoring
only `probes.safety_veto` as in its existing identity check.

There is **one look, only at complete N**. No metrics or clauses are examined as units finish;
operational progress, errors and hashes may be monitored. All clauses below are required.
Report the exact estimates, SEs and thresholds. Survival comparison tolerance is M4's 1e-9.

| Clause | Fixed calculation / threshold |
| --- | --- |
| Efficacy | H5000 paired mass effect vs incumbent, Hartung–Knapp one-sided 90% lower bound > 0 |
| c1 mass point | Same HK pooled effect >= 20 |
| c2 scripted mass | Pooled paired H5000 scripted mass mean vs incumbent >= 0 |
| c2/c3 alive-frame survival | Pooled H5000 candidate minus incumbent survival_fraction >= -0.03 separately on scripted, frozen and mixed |
| c4 positive seeds | At least 4/5 seed effects strictly > 0 |
| c5 prefix controls | All 45 direct-H5000 records equal their nested prefixes |
| c5 mi10 | HK pooled H10000 mass effect vs incumbent >= 20 |
| c6 guard mass point | Original M4 guard: seed **0** bank only, equal-mix H5000 mass vs champion strictly > 0 |
| Tier 1 per-seed floors | Each of the five seeds' **own scripted** 192-world mean alive-frame difference vs incumbent >= -0.15 |
| Tier 1 absolute floors | Pooled H5000 candidate episode survival >= 0.50 on each mix |
| c7 feasibility | Original M4 formula, N_max <= 600 |

The five scripted floors follow the amendment/T5b definition of eleven survival checks:
three pooled alive-frame, five scripted per-seed, three absolute episode floors. Measuring
champion on all banks supplies diagnostics and Tier 2; it does not change the deciding
seed-0 guard mass cell. H5000 episode survival means `prefix_h5000.deaths == 0`, including a
hero that subsequently dies after H5000; it is not alive-frame survival or H10000 survival.

For mass pooling, each training seed contributes the equally weighted mean of its three mix
means and variance `sum(s_mix^2 / n_mix) / 9`. Use the existing M4 HK pool, including its
scale floor and df=4. Scripted gets no extra efficacy weight from its larger N. Positive
seed counts use these same equally mix-weighted effects. The unchanged feasibility formula
uses the largest per-mix SD of all primary paired mass deltas, MDE=max(HK LB90,0.6×HK mean),
per-mix alpha=0.05/3, power=0.90, iterative t-based `n_fixed`, and
`N_max=ceil(1.02*n_fixed-1e-9)`.

The existing opted-in mass futility at **G=50** is retained explicitly: compute the primary
HK UB90 and report `NO_GO_EARLY` as the failure reason when UB90 < 50 and the required pass
clauses are not met. With a single look this occurs only at full N; it creates no interim
look. A complete pass has precedence over this nonbinding futility. There is no additional
survival futility, KILL threshold, PARTIAL/UNGATEABLE acceptance or EXTEND. Final labels per
the T7 spec are **STRICT_PASS** if every deciding clause passes, otherwise **FAIL**, with
the failed clauses and futility reason retained. Invalid/incomplete evidence is not a
scientific decision and must not be relabeled as a pass or fail.

## Candidate seed, Tier 2 and disclosures

Fix candidate selection now: apply `example_rules.py`'s primary-cell mi5, confidence 0.80
rule only to seeds passing their own scripted floor. Use each eligible seed's two-sided
80% Welch–Satterthwaite interval for its equally mix-weighted paired mass effect. Choose
the largest lower endpoint; exact ties go to the lowest numeric seed. With no eligible
seed, candidate=null. Report selection even on FAIL; it grants no serving authorization.

**Tier 2 is reported, not deciding:** pooled scripted H5000 episode survival candidate minus
champion_a5 + v8 across all five fresh banks (960 pairs) >= -0.05. Report PASS/FAIL, estimate
and SE. Campaign completion requires both Tier 1 and Tier 2.

Publish fresh alive-frame/episode survival sensitivity against both references, and preserve
the amendment's historical sensitivity disclosures (historical evidence only):

| Historical comparison | Alive-frame difference | Episode difference |
| --- | --- | --- |
| B8 seed-0 M4 look 2 vs champion | -0.156 | -0.188 |
| Census vs champion | -0.039 | -0.157 |
| M4 pooled vs served incumbent | -0.003 | -0.050 |

The original M4 outcome remains **PARTIAL** and motivated this amendment. B8 was trained
with v8 labels as well as v8 input features; this is not a model trained without safety
information. Tier 2 remains open until demonstrated. Secondary descriptive evidence is
H5000 deaths per 1,000 decisions at length >=800 and length-matched hazards (pre-decision
length bins 0–199, 200–399, 400–799, >=800); these never decide the gate.

## Execution, evidence and Git

Mac CPU only, no RunPod and no MPS/CUDA use. At most six owned worker processes, each with
torch intra/inter-op and OMP/MKL/OPENBLAS/VECLIB/NUMEXPR/NUMBA thread limits of one, nice 10.
Run in a newly created `tmux t7-phase-r` under caffeinate. Refuse to reuse a pre-existing
session not owned by this task. Use the existing AC/open-lid/thermal guard before launch
and while running, pausing/resuming only owned children. Never interact with another
session or terminate, pause or reprioritize a process not started by this task.

Artifact root: `snake-dqn-artifacts/redesign-t7-phase-r-b8-20261016`. Create-only plan,
start marker, records and decision. Bind the clean implementation commit, registration,
rule, seed audit, source/config files, checkpoint bytes, engine modes, full ordered banks,
rosters and unit assignments. Resume only hash-validated records under the same identity;
never overwrite completed records or retry scientific failures. Hash/check all records,
require exact coverage, recompute prefix identity and the single decision in a separate
`python -I` audit before publication. Report execution wall time from start through decision.

Work on `t7-20261016` in the specified worktree. After every commit:
`git fetch origin && git merge origin/main`, then `git push origin HEAD:main`, then
`git push origin t7-20261016`. Freeze this worktree during gameplay. Commit and push
`docs/research/t7_phase_r_result_2026-10-16.md` and compact hash-bound decision/audit
evidence. Poll until the decision commit is pushed. No serving defaults change.

## After the result (outside this task)

After STRICT_PASS, a separate planner-approved step reuses the FRP-v3 Mac serving procedure
in `research/frp3_checkpoint_serving_20261005/protocol.md` and
`docs/research/frp3_serving_result_and_release_2026-10-06.md`: fresh hash-bound intent;
25 Watch episodes × H5000, 25 Play episodes up to H5000, two H5000 parity probes;
Mac CPU one thread, AC/open lid; identity, dispatch, counters, parity, default/refusal and
rollback checks S1–S6; independent serving audit. Adapt identities to the selected bare B8
without weakening those checks. A Phase R label does not itself qualify or release serving.
Any eventual release must preserve rollback to frp3-s12 + v8 using
`SNAKE_SERVE_CHECKPOINT=frp3-s12`, champion + v8 using `SNAKE_SERVE_CHECKPOINT=champion`,
and existing veto rollback semantics for `SNAKE_SERVE_VETO_VARIANT=v7/v5/v2` with no named
checkpoint; `SNAKE_SERVE_VETO_WATCH_HERO=0` remains the wrapper-off switch.

If T7 fails, record the failure, apply no second guard amendment, and leave frp3-s12 + v8
served. Finish this task with the decision, clause table, candidate seed, Tier 2, wall time
and pushed SHA.
