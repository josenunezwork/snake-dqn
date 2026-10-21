# Catalog: research/t14* .. research/t24* packages (2026-10-20)

Read-only catalog. Every claim below comes from the package files or the docs they link to;
"unclear" means the files do not say. No package bytes were changed. Live-import data is from
`grep -rE "research[./](t1[4-9]|t2[0-4])"` over `src tests scripts web research .github verification`
(`*.py`, `*.sh`), excluding each package's references to itself.

Packages that do **not** exist: `t14*` (T14 is CI validation, recorded only in
[t14_ci_validation_2026-10-17.md](../t14_ci_validation_2026-10-17.md)), `t17*` other than `t17b`,
`t20*`, `t21*`.

## Summary

| package | kind | verdict | live importers outside package | status |
|---|---|---|---|---|
| [t15_mixed_roster_20261017](../../../research/t15_mixed_roster_20261017/) | code + evidence | SERVING_PASS | `research/t34_pathb/*`, `research/t34_shadow/*`, 5 tests | **reusable / live** |
| [t15b_watch_default_20261017](../../../research/t15b_watch_default_20261017/) | evidence only | RELEASE (mixed Watch default) | none | archive |
| [t16_colab](../../../research/t16_colab/) | code (Colab harness) | throughput evidence only | `research/t18_h100pod`, `research/colab_queue/worker.py`, `t23_r0_screen/*.sh`, 7 tests | **reusable / live** |
| [t17b_20261017](../../../research/t17b_20261017/) | evidence only | measurement, no verdict | none | archive |
| [t18_h100pod_20261017](../../../research/t18_h100pod_20261017/) | code (pod prepare step) | unclear (no result in package) | `research/perf_podbench_20261013/pod_bench.py`, 2 tests | live via pod_bench; RunPod-only |
| [t19_mixedseat_20261017](../../../research/t19_mixedseat_20261017/) | README only | none ("not a launch receipt") | n/a (code lives in t16_colab `n3` suite) | archive (doc) |
| [t22_frozen_pool_20261017](../../../research/t22_frozen_pool_20261017/) | benchmark + receipts | measurement; projection, 20,900 tr/s "not established" | none | archive (benchmark re-runnable) |
| [t23_r0_screen_20261017](../../../research/t23_r0_screen_20261017/) | code + prereg + results | **KILL** (-80.8 mass vs B8) | `research/q0q1_20261017/*`, `research/s1_mature_20261018/*`, `research/colab_queue/examples/t23_eval_shard.sh`, 6 tests | **reusable / live** (bank + evaluator) |
| [t24_v8_bodymap_20261017](../../../research/t24_v8_bodymap_20261017/) | benchmark + receipts | measurement; 26.5% long-body v8 reduction (CPU micro) | none | archive (benchmark re-runnable) |

## Per package

### t15_mixed_roster_20261017 — T15 mixed Watch roster serving qualification
- **Purpose:** pre-registered Mac serving qualification of bare b8-s3 hero vs FRP3-s12+v8 opponents
  (`SNAKE_SERVE_WATCH_ROSTER=mixed`); authority `docs/research/specs/t15_mixed_roster_qual_spec_2026-10-17.md`
  ([protocol.md](../../../research/t15_mixed_roster_20261017/protocol.md)).
- **Entrypoints (`__main__`):** `run.py` (producer; lazily imports `episodes`, `parity`, `measurements`,
  `seed_audit`), `audit.py`, `seed_audit.py`, `ui_server.py`. Library modules: `controls.py`, `episodes.py`,
  `measurements.py`, `parity.py`.
- **Verdict:** `SERVING_PASS` (`results/publication.json`); checks 1-5 pass (`results/summary.json`).
  Recorded in [t15_mixed_roster_qualification_2026-10-17.md](../t15_mixed_roster_qualification_2026-10-17.md),
  [INDEX.md](../INDEX.md) row 2026-10-17, and `docs/STATUS.md`. `results/UI_ENVELOPE_CORRECTION.md` /
  `UI_ENVELOPE_INVALID.json` preserve an invalid UI envelope (INDEX: "corrected evidence assembly").
- **Live imports:** `research/t34_pathb/{adapter_v2,live,run}.py`, `research/t34_shadow/{identity,common}.py`;
  tests `test_t15_{audit,controls,measurements,parity,run}.py`.
- **Reusable vs dead:** reusable — the active T34 campaign imports it. Do not edit (audit.py pins its own file hashes).

### t15b_watch_default_20261017 — T15B Watch default switch
- **Purpose:** final live smoke and focused validation for the owner-approved mixed Watch default
  (`results/manifest.json` description; spec `docs/research/specs/t15b_switch_watch_default_spec_2026-10-17.md`).
- **Entrypoints:** none (results only: frames/logs for `all_b8` and `default`, lint/test transcripts, SHA manifest).
- **Verdict:** RELEASE, mixed Watch with bare b8-s3 hero; recorded in
  [b8_s3_release_erratum_mixed_watch_2026-10-17.md](../b8_s3_release_erratum_mixed_watch_2026-10-17.md),
  INDEX.md, `docs/STATUS.md`.
- **Live imports:** none. **Dead code-wise; evidence archive.**

### t16_colab — T16 Colab throughput harness
- **Purpose:** bounded Colab throughput smoke (`run_smoke.py`, suites incl. `n3`), input bundle
  (`bundle.py`), runtime setup (`setup_runtime.sh`, `environment.py`), comparisons (`compare.py`, `sampler.py`),
  Mac N3 baseline (`mac_n3_baseline.sh`, `run_n3_bench_colab.sh`); `long_run.md` documents the
  owner-run `notebooks/colab_long_run.ipynb` recovery harness. Specs: `t16_colab_smoke_spec`, `t16b_python312_spec`.
- **Entrypoints:** `run_smoke.py`, `bundle.py`, `compare.py`, `sampler.py`; shell `setup_runtime.sh`,
  `mac_n3_baseline.sh`, `run_n3_bench_colab.sh`.
- **Verdict:** throughput evidence only, "not gate, quality or serving evidence"; recorded in
  [t16_colab_smoke_2026-10-17.md](../t16_colab_smoke_2026-10-17.md),
  [t16_colab_smoke02_results_2026-10-17.md](../t16_colab_smoke02_results_2026-10-17.md) (also holds T22/T24 appends),
  [t16_mac_baseline_2026-10-17.md](../t16_mac_baseline_2026-10-17.md).
- **Live imports:** `research/t18_h100pod_20261017/prepare.py` (`bundle`, `run_smoke`),
  `research/colab_queue/worker.py` (path), `research/t23_r0_screen_20261017/{run_t23_eval_colab,start_eval_colab}.sh`
  (`setup_runtime.sh`); tests `test_t16_colab_{bundle,compare,environment,sampler,smoke}.py`,
  `test_t18_h100pod.py`, `test_t19_compare.py`.
- **Reusable vs dead:** reusable infrastructure (Colab setup + bundle). Note the generic name has no date suffix.

### t17b_20261017 — T17b B5 Mac learner reference
- **Purpose:** local-file-mmap MPS learner reference (ms/update, host gather, device ms; raw/packed rings,
  fp32/bf16, bs2/bs4) — `results/b5_mac_2026-10-17.md`, `results/b2_gather_mac_2026-10-17.md`, JSON receipts,
  test transcripts. Spec `docs/research/specs/t17b_advisor_merge_spec_2026-10-17.md`.
- **Entrypoints:** none (results only; producing code is elsewhere — unclear which script from the package).
- **Verdict:** measurement; no pass/fail. Explicitly "not a quiet-host claim".
- **Live imports:** none. **Evidence archive.**

### t18_h100pod_20261017 — T18 H100 RunPod throughput smoke
- **Purpose:** archive-only RunPod substitute for the T16/T17 Colab smoke; `prepare.py` verifies and extracts
  the T16 input bundle on the pod and writes `prepare_receipt.json`; `commands.json` / `commands_n3.json` are
  the command manifests. Pod lifecycle stays in `research/perf_podbench_20261013/pod_bench.py`.
- **Entrypoints:** `prepare.py` (invoked by pod_bench on the pod).
- **Verdict:** unclear — the package holds no results; `h100_preruns_plan_2026-10-17.md` row 1 says
  "running (pod_bench T18, cap $8)" and README notes "old invalid receipts" are not rewritten.
- **Live imports:** `research/perf_podbench_20261013/pod_bench.py`; tests `test_t18_h100pod.py`, `test_perf_podbench.py`.
- **Reusable vs dead:** live only as a pod_bench step; T34 uses no RunPod (STATUS / memory), so dormant.

### t19_mixedseat_20261017 — T19 mixed-seat tooling status
- **Purpose:** README recording prepared tooling for step 4 of `h100_preruns_plan_2026-10-17.md`: the
  `n3` suite in `research/t16_colab/run_smoke.py` and the trainer's opt-in `--learner-seats {1,3,6}`.
- **Entrypoints:** none (README only).
- **Verdict:** none — "not a launch receipt, training result, promotion result, or serving change"; N3 is
  "measurement-only".
- **Live imports:** n/a. **Doc only.**

### t22_frozen_pool_20261017 — T22 frozen-pool CPU microbenchmark
- **Purpose:** measure the six-policy frozen pool (1 CPU thread, 128 worlds, 3 learner seats) before/after
  a pool optimization; `project_a100.py` projects A100 rate.
- **Entrypoints:** `bench_frozen_pool.py`, `project_a100.py`.
- **Verdict:** projection 13,578 tr/s (a4) / 14,833 tr/s (a6); "The 20,900 target is not established."
  Receipts `results_final_{reference,current}.json`, `a100_projection.json`, `verification.txt`. Recorded in
  [t16_colab_smoke02_results_2026-10-17.md](../t16_colab_smoke02_results_2026-10-17.md) section "T22 ...".
- **Live imports:** none. **Evidence archive; benchmark re-runnable.** (Parity fixture lives in `tests/fixtures/t22`.)

### t23_r0_screen_20261017 — T23 R0 N=3 static-pool 8M screen
- **Purpose:** pre-registered screen of B8 league fine-tune (5 seeds x 8M) vs B8
  (`PREREGISTRATION_T23.md`, `EVALUATION.md`); world bank (`bank.py`, `WORLD_BANK.json`), evaluator
  (`evaluate.py`), parity (`parity.py`, `parity/`), analysis (`analyze.py`), Colab launchers (`*.sh`).
- **Entrypoints:** `bank.py`, `evaluate.py`, `parity.py`, `analyze.py`; shell `run_t23_colab.sh`,
  `run_t23_eval_colab.sh`, `start_eval_colab.sh`.
- **Verdict:** **KILL** — mean equal-mix G = -80.8 mass vs B8, 0/5 seeds positive. Recorded in
  [RESULTS_T23.md](../../../research/t23_r0_screen_20261017/RESULTS_T23.md),
  `results/t23_verdict.{md,json}`, `docs/research/specs/after_t23_plan_2026-10-17.md`,
  `docs/research/advice/2026-10-17_after_t23_*.md`. `ALLOCATION_REJECTED_V1.json` keeps a rejected allocation.
- **Live imports:** `research/q0q1_20261017/*` (8 files), `research/s1_mature_20261018/{bank,evaluate}.py`,
  `research/colab_queue/examples/t23_eval_shard.sh`; tests `test_t23_{analysis,bank,decisions,evaluate,parity}.py`,
  `test_q0_episode_metrics.py`.
- **Reusable vs dead:** the hypothesis is dead (KILL); `bank.py` + `evaluate.py` are **live reusable
  evaluation infrastructure** (Q0/Q1/S1 build on them).

### t24_v8_bodymap_20261017 — T24 v8 body-map benchmark
- **Purpose:** CPU-only benchmark of shared per-frame v8 diagnostics (retained body map, shared learner/B8 call)
  in the N=3 six-policy league; `project_a100.py` projection.
- **Entrypoints:** `bench_bodymap.py`, `project_a100.py`.
- **Verdict:** measurement — combined v8 22.155 -> 16.280 ms at length 1200 (26.5% reduction); whole actor
  246.738 -> 234.482 ms/frame; "not A100 throughput or training performance evidence". Receipts
  `before_23c1ba73*.json`, `after_4b4723f*.json`, `VERIFICATION.md`, `verification_*.{txt,xml}`. Recorded in
  [t16_colab_smoke02_results_2026-10-17.md](../t16_colab_smoke02_results_2026-10-17.md) section "T24 ...".
- **Live imports:** none. **Evidence archive; benchmark re-runnable.**

## Gaps noticed
- [INDEX.md](../INDEX.md) has rows for T15/T15B but none matched for T16-T24 (grep for `T16..T24` in INDEX returned nothing).
- T22/T24 verdicts live only as appends inside the T16 smoke-02 results doc.
- T18 outcome is not recorded in its package.
