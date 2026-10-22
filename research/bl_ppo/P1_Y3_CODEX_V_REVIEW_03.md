# Independent Codex-V P1 round-14 review — source-latebody-03

Verdict: **PASS for software and source admission**, 2026-10-22. No unresolved
findings. This is independent review of the amendment-02 implementation and
prospective mechanical repairs, not natural support, critic accuracy, live
Y3 PASS, funding, dispatch, serving qualification or release.

Reviewer: read-only specialist `/root/p1_review`, separate from the original
amendment authors and from this turn's root preparer. Root independently
reconciles the artifacts and publishes the review receipt. Baseline after
`git pull origin main`: `cda1cf5c`, clean branch
`wp3-p1-ppo100m-20261021` at `/Users/josenunez/Projects/ml/snake-dqn-wp3`.
Authority: round-14 `JUDGMENT.md` ruling (c), digest
`95359ebf0e835ea3fc14ab138e94bdfa2f34a627a08b2381c23229ebac0a0314`.

## Findings, all resolved prospectively

1. **P1, false critic admission from inaccurate comparator.** The unscaled
   float32 default least-squares tolerance can discard intercept/time columns
   at 10,000 fit rows. Independent reproduction: target `10000+3000h+50L`,
   candidate `target+1000`, EV1, candidate MSE999999.875, erroneous baseline
   MSE2486126 => false PASS. `learner.fit_time_length_baseline` now fits the
   identical `(1,h,L)` columns on CPU float64 with explicit SVD `gelsd`;
   `refit.qualification` and `score_fixture_holdout` share it. Baseline prediction
   and MSE stay double; candidate MSE and registered EV computation stay
   unchanged. Reviewer reproduction recovers coefficients approximately
   `(10000.000007,2999.999936,50.00000002)`, baseline MSE.00000403, and refuses.
   `test_large_unscaled_fit_baseline_cannot_false_pass_biased_critic` uses
   disjoint fit/held-out rows and prevents regression.
2. **P2, price ceiling enforcement.** The default supervisor tolerance .05u/h
   would admit1.59 under a claimed1.54 ceiling. `accounting.admit_account`
   now permits an optional manifest tolerance only when tightening its default.
   This source sets0.0. Test proves1.54 admits,1.55 refuses and.06 cannot loosen
   the default. Existing manifest defaults are unchanged.
3. **P2, incomplete admission evidence.** Prefix handling formerly exempted
   unresolved Claude/independent TODOs. Only fresh live inventory remains an
   operational exception. Source preparation needs hash-bound independent
   PASS/no-unresolved-findings/no-new-CUDA-path review of its exact source and
   a recomputed Y3 debit. `p1_admission.reconcile` verifies released receipts,
   charges max(full elapsed ceiling,gross debit) rounded up and rejects a
   changed terminal-attempt inventory. Review/debit payloads and sidecars are
   bundled and bound in the control prerequisites.

## Ruling (c) review

| Clause | Reviewed implementation/evidence |
|---|---|
| 1 natural estimand | `refit.qualification`: natural completed held-out L>=800 only; comparator uses natural fit rows only. Separate fixture bank cannot fit PopArt/baseline or admit. |
| 2 support/continuation | `natural_support`: fit160 lives/640 rows; holdout40 lives/160 rows/8 per mix. `run_train` inspects support before accuracy, allows one bounded continuation to <=4M fit transitions, then refuses without actor updates if short. |
| 3 supplement | Triggered by natural fit below its minimum; not held-out-only shortage or accuracy failure. Distinct provenance and no natural support credit. |
| 4 retained evidence | Numeric refusal or explicit unavailability, support by provenance/mix/length/horizon, distinct lives, cumulative frame histograms, natural 257d features/targets in <=10MB chunks plus immutable row index/sidecars. |
| 5 replay | Natural and fixture bank, support receipt, histograms and algorithmic state persist through snapshots; full tiny source/replacement entrypoint replay is exercised. |
| 6 offline diagnostic | Registered head, interaction head and baseline fit natural fit rows, evaluate held-out rows by mix/length; descriptive only, no substitution. |
| 7 unchanged scope | Registration and amendment02 bytes unchanged. Seed,100M endpoint,reward,gamma/lambda,actor resets,opponents,bar,one look and release scope unchanged. |

No new CUDA path: model/network and collector dynamics are unchanged. The
comparator fix explicitly runs on CPU. Existing critic feature/head APIs,
optimizer and PPO math remain in use; no new kernel or actor curriculum.

Reviewer verification: initial E8/refusal/export/replay selection **34 passed
in83.71s**; after baseline correction **54 passed in1.10s**; final accounting,
transition accounting, admission, manifest and refit selection **72 passed
in.75s**, all CPU with one math thread. Root's final prepared-artifact receipt
records additional checks and exact source digests. Positive tiny replay uses
explicit synthetic natural-labelled support and controlled predictions inside
pytest; it proves software state transport, not actual support or accuracy.

Old evidence is preserved, never rescored. No protected t34 package, sealed
study, tmux job, released checkpoint or serving pin was modified or opened as
applicable. No Colab command was called. Claude owns the live usage/sessions
check, dispatch and campaign dispatch log. Final prepared-artifact evidence is
published separately after the executable source commit exists on origin/main.
