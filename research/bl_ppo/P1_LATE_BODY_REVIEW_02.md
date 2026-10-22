# P1 round-14 amendment 02 independent review — 2026-10-22

Scope: the complete diff from `7325fdb79395636e1a5e078a0d8fe2288d1e8810`
against the fully read round-14 `JUDGMENT.md`, especially ruling (c)1–7.
Independent reviewer: `/root/independent_review`, specialized read-only reviewer
agent, separate from all implementation lanes. The root also performed a
second careful review against each ruling clause. No reviewer changed code,
called Colab or opened the sealed study. This is a software/preparation review,
not scientific critic qualification, live Y3 PASS, funding or release.

## Clause-by-clause review

| Ruling | Evidence reviewed |
|---|---|
| (c)1 natural estimand | `refit.qualification` selects natural held-out L>=800; baseline uses all natural fit rows; `run_train` keeps fixtures in a separate bank; fixture scoring never calls admission or PopArt fitting. The registered fit-only supplement remains allowed critic training data, with distinct provenance. |
| (c)2 support and continuation | `natural_support` counts distinct completed world/life pairs and all three mixes, enforcing 160/640 fit and 40/160/8-per-mix holdout floors only for P1. `run_train` checks support before accuracy, continues once under the combined <=4M fit cap, then refuses without actor updates. |
| (c)3 supplement trigger | Only natural fit support below its minimum triggers the bounded 25%-root supplement; held-out shortage or poor accuracy alone does not. Supplement rows cannot count as natural support. |
| (c)4 retained evidence | Numeric refusal metrics include bias and availability labels; retained support tables separate provenance/mix/length/horizon and distinct lives. Natural length histograms persist in diagnostics and immutable monitor tick status files. Every retained natural fit/held-out row is exported in hash-bound NPZ chunks, <=10MB each, with a row index. This bound is per chunk, not a claim that combined fit-plus-held-out exports total <=10MB. |
| (c)5 replay | The exact source, flat transitive transport, replacement, 1M qualification and every tranche resume execute in `test_p1_replay_amendment02.py`; full algorithmic state fingerprints match. Fixture bank, support receipt and histogram persist through snapshots. |
| (c)6 offline diagnostic | Independent module fits `[z,h]`, `[z,h,h*z]` and `(1,h,L)` on natural fit rows only, evaluates natural held-out rows by mix/length, checks input sidecars and world disjointness, and exports a diagnostic receipt. No checkpoint or qualification substitution exists. |
| (c)7 unchanged scope | Original registration SHA remains `944aaf3798f9ad698698c5f7abe16e1f5325a034fb1b032ba3cc9387d9111a56`. Seed, endpoint, reward, gamma/lambda, actor resets, opponents, bar, one-look rule and serving scope remain unchanged. |

No new CUDA path: `model.py`, collector dynamics and PPO update math are
unchanged. The learner adds a bias metric; bank/row handling evaluates existing
critic feature/head APIs. The registered EV variance clamp is preserved;
offline diagnostics expose variance and whether EV is defined, without adding
a new admission criterion.

## Findings resolved before publication

The final review checked fixes for KILL precedence through support/row-export
failure, PAUSED/COMPLETE invalidation when required row exports fail, numeric
fallback on early/nonfinite refusal including a None critic receipt, actual
support counts on metric failure, and admission refusal after an incomplete or
failed required fit. It also required retaining the full successful replay
test alongside fixture-only refusal tests. No remaining product finding was
identified after these fixes.

## Verification and evidence limits

The integrated targeted selection passed **115 tests in 72.63s**, with
`nice -n 10`, CPU, one math thread per process and at most one collector child.
Exact test selection: `test_refit.py`, `test_late_body.py`,
`test_model_learner.py`, `test_diagnostics.py`, `test_monitor.py`,
`test_p1_manifests.py`, `test_offline_critic_diagnostic.py`,
`test_p1_entrypoints.py`, `test_p1_runtime.py`,
`test_p1_replay_amendment02.py`, `test_recovery.py`, `test_exports.py`,
`test_run.py`, and `test_p1_qualification.py`, all under `research/bl_ppo`.
Black, isort, flake8, shell syntax and diff checks pass on the changed files.
Retained log:
`big-lever/exp/bl-p1/verification-latebody02-01/pytest.log` under the artifact
root. The publication index binds final reviewed file digests and receipts.

The successful tiny replay test prospectively seeds explicitly synthetic,
natural-labelled completed bank rows inside pytest because a 16-frame world
cannot naturally grow to L800. It also controls critic predictions and reduces
execution dimensions. Main/run_train, real B8/PPO updates, collector, transport,
snapshot and recovery execute. The separate unseeded tests prove that the
constructed fixture bank alone refuses admission. These software controls
establish no natural support, learned accuracy, L4 identity or scientific PASS.

The source-02 predecessor's L4 transport evidence is the ruling's basis for
allowing source-latebody-02 as the first amended L4 run. New manifests must be
locally validated with recomputed pinned-source Z-9 forecasts and preserved
sidecars after the executable commit is published. They remain NOT DISPATCHED;
fresh inventory, real predecessor/Y3 receipts and transitive download catalogs
remain operator requirements. Old latebody-01 attempts receive create-only
quarantine siblings; old manifests, receipts and dispatch files stay immutable.
