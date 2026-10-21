# Catalog: research packages T5 through T13 (2026-10-16 .. 2026-10-17)

Read-only catalog, written 2026-10-20 from the package files, `research/INDEX.md`,
`docs/research/INDEX.md` and the result docs. No package was modified. There is no `t7*`
package under `research/`.

**Binding warning.** Every package below has its source files listed under
`preflight/source_hashes` in
[`research/t34_round0/REGISTRATION_ROUND0.json`](../../../research/t34_round0/REGISTRATION_ROUND0.json)
(also `VERIFICATION_A6*.json`), and several appear in other packages' `SEED_AUDIT*.json`
freshness boundaries. Moving, renaming or editing them would break those hash and seed
bindings. Treat them as frozen.

"Live imports" lists Python `import`/`from` statements outside the package itself
(found with grep over `*.py`). Path mentions inside JSON receipts are noted separately.

## Summary

| Package | Purpose (one line) | Verdict | Recorded in | Imported by (outside package) | Status |
|---|---|---|---|---|---|
| [t5_separability_20261016](../../../research/t5_separability_20261016/) | Restore T4e bank states, fit grouped separability of 24 per-action features | **FAIL** (death-AUC gain 0.0088, bar 0.05) | [t5_separability_2026-10-16.md](../t5_separability_2026-10-16.md) | t5b, t8, t10; tests | Reusable library (`fit`, `restore`) |
| [t5b_risk_and_fuse_20261016](../../../research/t5b_risk_and_fuse_20261016/) | B8 survival-guard risk model and fuse-layer probes on T5 states | Diagnostic: guard risk 38.81%, Tier 2 FAIL, all fuse AUC < 0.85 ("perception") | [t5b_risk_and_fuse_2026-10-16.md](../t5b_risk_and_fuse_2026-10-16.md) | t8, t10; tests | Reusable library (`fuse`) |
| [t6_restart_20261016](../../../research/t6_restart_20261016/) | Create-only launch of RESTART training lanes (train from late-game snapshots) | see t6_screen | [t6_preparation_2026-10-16.md](../t6_preparation_2026-10-16.md) | t6_screen, t11_local30; tests | Dead (launch-only) except `seed_audit` used by later audits |
| [t6_screen_20261017](../../../research/t6_screen_20261017/) | CPU probe screen of RESTART vs CTRL, rounds 72/73 | **KILL** | [t6_screen_result_2026-10-17.md](../t6_screen_result_2026-10-17.md) | t11_local30; tests | Dead; watcher/analyze pattern reused by t11_screen (copied, not imported) |
| [t8_probes_20261017](../../../research/t8_probes_20261017/) | Arrival-race features and raw-raster CNN classifier on T5 states | Diagnostic: CNN 0.742, arrival +0.014, no 0.85 | [t8_probes_2026-10-17.md](../t8_probes_2026-10-17.md); `docs/research/INDEX.md` combined T8-T11 row | t10 (`arrival_fit`); tests | Mostly dead; `arrival_fit` used by t10 |
| [t9_divergence_20261017](../../../research/t9_divergence_20261017/) | Replay 46 late T1 LADs, fatal vs surviving topology divergence | Diagnostic: no early separator; pocket builds over ~50-100 decisions | [t9_divergence_2026-10-17.md](../t9_divergence_2026-10-17.md) | t10 (`metrics`); tests | `metrics` reusable (flood/race topology) |
| [t10_certificate_horizon_20261017](../../../research/t10_certificate_horizon_20261017/) | Tail-reachability certificate audit and horizon-stratified GBM | Diagnostic: certificate 99.97% vs B8 99.92%; AUC ~0.92 at 30 to ~0.75 at 300 | [t10_certificate_horizon_2026-10-17.md](../t10_certificate_horizon_2026-10-17.md) (+ `_evidence/`) | tests only | Dead (terminal diagnostic) |
| [t11_local30_20261017](../../../research/t11_local30_20261017/) | LOCAL30 two-sided label bank, input binding, MPS smoke, 8M lane launch | see t11_screen | [t11_implementation_2026-10-17.md](../t11_implementation_2026-10-17.md) | t11_screen; tests | Dead |
| [t11_screen_20261017](../../../research/t11_screen_20261017/) | LOCAL30 probe screen, rounds 74/75 | **KILL before gameplay** (seed 13 halted, boost tripwire) | [t11_screen_result_2026-10-17.md](../t11_screen_result_2026-10-17.md) | tests only | Dead |
| [t12_b8_serving_20261017](../../../research/t12_b8_serving_20261017/) | B8 seed-3 Mac serving qualification through the real `GameSession` | **SERVING_FAIL / HOLD** (check 4 paced speed) | [t12_b8_serving_qualification_2026-10-17.md](../t12_b8_serving_qualification_2026-10-17.md) | t12b, t13, t13b, t15 (`seed_audit`, `measurements`, `parity`); tests | **Reusable / live library** for serving harnesses |
| [t12b_check4_20261017](../../../research/t12b_check4_20261017/) | Check-4-only rerun after serving-speed repair, plus pacing diagnosis | **SERVING_PASS** (check 4) | same T12 doc (amendment section); `results/` | tests only | Dead (`diagnose` reusable for pacing debugging) |
| [t12c_release_20261017](../../../research/t12c_release_20261017/) | Results only: release startup receipts for b8-s3 | **RELEASE** b8-s3 without veto | [b8_s3_release_2026-10-17.md](../b8_s3_release_2026-10-17.md) | none (no code) | Evidence only |
| [t13_product_roster_20261017](../../../research/t13_product_roster_20261017/) | Descriptive Watch roster measurement (all-B8 vs FRP3 vs mixed) | Descriptive: all-B8 hero worse, arena better; mixed roster helps hero | [t13_product_roster_2026-10-17.md](../t13_product_roster_2026-10-17.md) | t13b, t15 (`seed_audit` boundary); tests | Reusable roster harness (via t13b) |
| [t13b_six_snake_20261017](../../../research/t13b_six_snake_20261017/) | Six-snake default roster measurement + all-B8 eight supplement | **PLAN_MIXED_PENDING_ROSTER_QUALIFICATION** | [t13b_six_snake_roster_2026-10-17_results.md](../t13b_six_snake_roster_2026-10-17_results.md), `RULE_RESULT.json` | t15 (`seed_audit`); tests; `results/episodes.json.gz` is a declared data file of [t34_pathb/source_binding_v2.py](../../../research/t34_pathb/source_binding_v2.py) | **Live data input for T34 Path B** |

## Per package

### t5_separability_20261016 (T5)
- Purpose: CPU-only restoration of T4e bank states into a feature/label table and fixed grouped separability fits; never trains a policy (`__init__.py`).
- Entrypoints (`__main__`): `restore.py` (per-partition replay, byte-validated), `merge.py` (merge four partitions), `fit.py` (grouped fits), `t1_check.py` (T1 late-LAD action-area replay). `requirements.txt` holds task-local analysis deps.
- Verdict: FAIL, 19,976 states / 342 worlds. Recorded in the T5 doc and `docs/research/INDEX.md`.
- Live imports: `t5b.fuse`, `t8.{arrival,arrival_fit,cnn}`, `t10.{audit,horizon}`; tests `test_t5_fit.py`, `test_t5_restore.py`.
- Reusable: `fit` (fixed GBM/LR fold machinery) and `restore` (bank state reconstruction) are the base of T5b/T8/T10.

### t5b_risk_and_fuse_20261016 (T5b)
- Purpose: retrospective B8 survival-clause risk (`risk.py`) and fuse-representation classifiers on T5 states (`fuse.py`); never gate evidence.
- Entrypoints: `risk.py`, `fuse.py`.
- Verdict/recorded: see table; T5b doc plus `_evidence/`. Not a row in `docs/research/INDEX.md` (only referenced via T5 doc), unclear if intentional.
- Live imports: `fuse` by t8 (4 modules) and t10 (4 modules); tests `test_t5b_fuse.py`, `test_t5b_risk.py`.
- Reusable: `fuse`. `risk` is single-use.

### t6_restart_20261016 (T6 preparation)
- Purpose: fixed T6 RESTART command construction (`recipe.py`), create-only lanes (`launch.py --check`, `lane0.sh`, `lane1.sh`, `launch_inputs.json`), fail-closed freshness audit for rounds 72/73 (`seed_audit.py`).
- No `__init__.py`, no README.
- Verdict: training feeding the T6 screen (KILL). Prep recorded in `t6_preparation_2026-10-16.md`, `t6_seed_audit_2026-10-16.json`, `t6_smoke_validation_2026-10-16.json`.
- Live imports: `t6_screen.audit`, `t11_local30.seed_audit`; tests `test_t6_launch.py`, `test_t6_seed_audit.py`.
- Dead for launching; `seed_audit` stays live as a link in the freshness chain.

### t6_screen_20261017 (T6 screen)
- Purpose: registered CPU screen (96 scripted / 32 frozen / 32 mixed H5000 worlds per checkpoint) of RESTART vs CTRL vs incumbent.
- Entrypoints: `audit.py` (refresh audit), `watch.py` (receipt-only watcher, starts own CPU children), `probe.py`; `analyze.py` is a library (no `__main__`). Portable outputs in `results/`; raw logs external (`snake-dqn-artifacts/redesign-t6-probes-20261017`, per README).
- Verdict: KILL (survival delta 0.000).
- Live imports: `t11_local30.seed_audit`; tests `test_t6_screen_{analysis,audit,probe,watch}.py`.
- Dead.

### t8_probes_20261017 (T8)
- Purpose: supervised diagnostics; arrival-race features from restored T5 boundaries and a raw-raster D300 CNN with T5's exact OOF folds.
- Entrypoints: `arrival.py`, `merge.py`, `arrival_fit.py`, `cnn.py`, `audit.py` (independent recheck). No README.
- Verdict: no probe reaches 0.85 (CNN 0.742, arrival +0.014); points to the input-signal side per advisor rule.
- Live imports: `t10.horizon` imports `arrival_fit`; tests `test_t8_{arrival_fit,audit,cnn}.py`.
- Mostly dead; `cnn.py` is the only raw-raster classifier in this range.

### t9_divergence_20261017 (T9)
- Purpose: replay all 46 late T1 LADs on CPU through D100; static-world topology and optimistic arrival races.
- Entrypoints: `study.py`, `report.py`, `audit.py`; `metrics.py` library. `results.json` in package. Run commands in README.
- Verdict: artifact audit PASS; no early separator (see INDEX T8-T11 row).
- Live imports: `t10.certificate` imports `metrics`; tests `test_t9_{metrics,report,study}.py`.
- Reusable: `metrics` (four-neighbour TTL flood, race counts).

### t10_certificate_horizon_20261017 (T10)
- Purpose: optimistic TTL tail-reachability certificate audit on T3-dense and B8 trajectories; horizon-stratified refit of the T5 GBM.
- Entrypoints: `certificate.py`, `horizon.py`, `horizon_shards.py`, `audit.py`, `report.py` (all `__main__`).
- Verdict: certificate holds 99.970% vs B8 99.92%; horizon AUC decays with horizon. Doc has `<!-- verdicts:start -->` block.
- Live imports: none outside package; tests `test_t10_{audit,certificate,horizon,shards}.py`.
- Dead (terminal diagnostic of the closed label line).

### t11_local30_20261017 (T11 implementation)
- Purpose: LOCAL30 two-sided label bank from T4-prep v2 (`bank.py`), input binding (`bind_launch.py`), 200k MPS smoke (`smoke.py`), 8M lanes (`launch.py --check`, `lane{0,1}.sh`), rounds 74/75 seed audit.
- No `__init__.py`. Frozen receipts: `seed_audit.json` (README: do not overwrite), `smoke_receipt.json`, `completion.json`, `bank_manifest.json`.
- Verdict: feeds T11 screen (KILL). Recorded in `t11_implementation_2026-10-17.md`.
- Live imports: `t11_screen.audit`; tests `test_t11_{bank,seed_audit,smoke}.py`.
- Dead.

### t11_screen_20261017 (T11 screen)
- Purpose: registered LOCAL30 probe screen; `watch.py` reads training receipts first, any failure establishes KILL.
- Entrypoints: `audit.py`, `watch.py`, `probe.py`, `analyze.py`.
- Verdict: KILL before gameplay (seed 13 exited 1 at 300,160 transitions; INDEX: boost tripwire 1.07% at L < 700). Round 75 not started.
- Live imports: tests only (`test_t11_screen_{analysis,audit,probe,watch}.py`).
- Dead. Near-duplicate of t6_screen (same file set, sizes 389/390 lines for `watch.py`).

### t12_b8_serving_20261017 (T12)
- Purpose: pinned Mac serving qualification of b8-s3 driving `web.backend.session.GameSession` (not the simulator directly).
- Entrypoints: `serving_run.py` (coordinator/worker CLI), `serving_audit.py` (independent stdlib-only audit), `seed_audit.py`, `fill_strict_pin.py` (owner-only), `ui_server.py` (loopback UI, isolated score DB). Libraries: `episodes.py`, `measurements.py`, `parity.py`. Docs: `protocol.md`, `PREFLIGHT.md`; receipts `SEED_AUDIT_T12*.json`.
- Verdict: SERVING_FAIL / HOLD (checks 1-3, 5-8 pass; check 4 fail).
- Live imports: `t12b.{diagnose,run,seed_audit}`, `t13.seed_audit`, `t13b.seed_audit`, `t15_mixed_roster_20261017.{parity,seed_audit}`; 7 test files.
- **Reusable and live**: the shared serving-measurement and freshness-audit base for every later Watch/roster study.

### t12b_check4_20261017 (T12B)
- Purpose: create-only check-4 rerun (`run.py`) with fail-closed stdlib audit (`audit.py`), narrow seed registration (`seed_audit.py`), and pacing diagnosis replicating `Hub.engine_loop` order (`diagnose.py`). Protocol: `protocol_check4.md`.
- Verdict: SERVING_PASS (check 4), recorded in the T12 doc amendment and `results/`.
- Live imports: tests only (`test_t12b_check4_{audit,run}.py`).
- Dead; `diagnose.py` useful if serving pacing regresses.

### t12c_release_20261017 (T12C)
- Purpose: results only (14 files: startup frames, stderr logs, flake8 output, `manifest.json`, `startups.json`). No code, no `__init__.py`.
- Verdict: RELEASE b8-s3 without a veto. Recorded in `b8_s3_release_2026-10-17.md` (links into `results/`), `docs/research/INDEX.md` line for T12C, and `docs/codex-skills/snake-dqn-serving-release/SKILL.md`.
- Evidence only.

### t13_product_roster_20261017 (T13)
- Purpose: descriptive Watch-roster measurement through the real serving session; mixed roster is a harness-only mutation, no product default changed.
- Entrypoints: `seed_audit.py`, `run.py` (coordinator + at most four CPU subprocesses), `analyze.py` (paired seed bootstrap CIs). Library `episodes.py`. Docs: `protocol.md`, `REVIEW.md`, `FINAL_REVIEW.md`; `SMOKE.json`, `SEED_AUDIT.json`. No `__init__.py`.
- Verdict: descriptive (see table). Recorded in `t13_product_roster_2026-10-17.md`; no row found in `docs/research/INDEX.md` by grep for "T13".
- Live imports: `t13b.{analyze,episodes}`; `t15_mixed_roster.seed_audit` references the package path; tests `test_t13_{analysis,product_roster}.py`.
- Reusable via t13b (roster harness).

### t13b_six_snake_20261017 (T13b)
- Purpose: real Watch episodes at the six-snake default and research-only eight; temporarily patches T13's constructor count and invariant, restores after every episode.
- Entrypoints: `seed_audit.py`, `run.py`, `analyze.py`; library `episodes.py`. Docs as T13 plus `RULE_RESULT.json`. No `__init__.py`.
- Verdict: PLAN_MIXED_PENDING_ROSTER_QUALIFICATION (rule committed at `f36a20d3` before run). Recorded in results doc and `research/INDEX.md`.
- Live use: `results/episodes.json.gz` is a `DECLARED_DATA_FILES` entry in `research/t34_pathb/source_binding_v2.py` and appears in `REGISTRATION_RETENTION_V2*.json`/`RETENTION_OC_V2.json`. `t15_mixed_roster.seed_audit` references it. Tests `test_t13b_six_snake.py`.
- **Live (T34 data dependency)**.

## Cross-cutting observations
- Dependency spine: `t5.fit` -> `t5b.fuse` -> `t8` -> `t10`; `t9.metrics` -> `t10.certificate`; `t6_restart` -> `t6_screen` -> `t11_local30` (seed-audit chain); `t12` -> `t12b`, `t13`, `t13b`, `t15`.
- 41 test files under `tests/` target these packages (`test_t5*` .. `test_t13*`).
- README coverage: only t6_screen, t9, t10, t11_local30, t11_screen have README.md; others rely on docstrings or `protocol.md`.
- Packages without `__init__.py` (namespace packages, imported anyway): t6_restart, t11_local30, t12c, t13, t13b.
