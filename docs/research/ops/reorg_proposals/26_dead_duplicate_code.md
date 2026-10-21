# 26 — Dead and duplicated code across `src/` and `research/` (proposal only)

Scope 26 of the 2026-10-20 org pass. **Read-only audit: nothing was changed.** Every item below is a
proposal. Ratified studies hash their source closure, so anything that is imported by
`t34_round0`, `t34_pathb`, `t34_exit_b8`, `t34_exit_b8_fit` or `colab_cli` (directly or
transitively) must stay byte-identical until that study closes.

## Method (reproducible)

- Duplicate names: `grep -rhoE '^def [a-zA-Z_0-9]+' --include='*.py' src research | sort | uniq -c`.
- Duplicate bodies: an `ast` pass over every top-level `def` in `src/` and `research/`, hashing the
  body with its docstring removed. "Identical group" = defs whose AST bodies are byte-equal.
- Dead modules: a `.py` file under `src/` or `research/` with **no code reference** (no dotted
  import, relative import, or `path/to/file.py` string in any `.py/.sh/.yml/.yaml/.toml/.ipynb`
  file or the Makefile). JSON manifests and Markdown were counted separately as "doc refs" because
  ratified manifests list every file they hash.
- Packages a ratified study depends on: `from research.<pkg>` lines in the five ratified packages
  plus the packages they import (one extra hop: `t34_shadow`, `t33_batched_teacher`,
  `t3_rollout_labels_20261015`, `phase0_search_20261019`).

## Headline findings

1. **No dead modules in `src/`.** Every `src/**/*.py` has at least one code importer or path
   reference. Two `src/scripts` files are referenced only from docs/manifests (leaf CLIs, not dead):
   `src/scripts/bench_simd.py` (blueprint P2 throughput bench) and
   `src/scripts/rebase_checkpoint.py` (curriculum warm-start metadata rewriter; documented usage).
2. **No truly dead modules in `research/`.** 81 research files have no code importer, but every
   one is cited by Markdown/JSON (3–30 doc refs) — they are leaf entrypoints (benches, reports,
   smokes, `merge.py`, `summarize*.py`), i.e. records of how a result was produced. Deleting them
   would break reproducibility of the cited result; do not delete.
3. **Heavy copy-paste of small provenance helpers** (hashing, canonical JSON, atomic/write-once
   writes, git info, AC-power check, seed digests). This is the real consolidation opportunity, but
   almost every copy lives in a frozen or ratified package, so the safe move is "new code imports a
   shared module", not "rewrite old packages".
4. **B8 loading is already mostly consolidated**: 21 research packages load B8 through
   `src.model.ego2s_network.load_ego2s_checkpoint`. Only small local wrappers are duplicated.
5. **A production path imports a research package**: `web/backend/pathb.py:13` does
   `from research.t34_pathb ...`. Everything else in `src/` and `web/` is research-free.

## Ranked consolidation opportunities

Rank = (number of copies × likelihood future code copies it again) ÷ risk.

### 1. Shared provenance helper module for NEW research code — High value, Low risk

Create one module (proposed `src/utils/provenance.py`; `src/utils/` today holds only
`nn_utils.py` and `tensor_utils.py`) with `sha256_file`, `sha256_bytes`, `canonical_json`,
`canonical_sha`, `atomic_write_json`, `write_once`, `git_head`, `on_ac_power`. Point new packages
at it; leave old copies untouched.

Evidence (top-level defs in `src/` + `research/`):

| helper | defs | distinct bodies | largest identical group (examples) |
|---|---|---|---|
| `sha256_file` | 55 | 19 | 8 — `apex_safety_20260926/dev_screen.py`, `apex_veto_serving_20261001/schema.py`, `apex_veto_v5_serving_20261001/schema.py`, `apex_veto_v7_serving_20261002/schema.py` |
| `sha256` | 48 | 15 | 20 — `redesign_t7_phase_r_20261016/seed_audit.py`, `t11_local30_20261017/{bind_launch,launch,seed_audit}.py` |
| `_sha` | 28 | 15 | 10 — `frp3_checkpoint_serving_20261005/{fill_strict_pin,serving_audit}.py`, `t12_b8_serving_20261017/...` |
| `_sha256` | 20 | 12 | 6 — `t11_screen_20261017/{analyze,audit}.py`, `t3_screen_20261015/banks.py` |
| `canonical` | 20 | 5 | 15 — `phase0_search_20261019/{analyze,bank}.py`, `q0q1_20261017/{a4_compatibility,audit_common}.py` |
| `canonical_sha` | 14 | 7 | 5 — `q0q1_20261017/*`, `s1_mature_20261018/bank.py` |
| `canonical_json` | 7 | 1 | 7 — all apex_veto serving `schema.py` + `apex_safety_20260926/dev_screen.py` |
| `_atomic_json` | 17 | 13 | 3 — `t3_diagnostics_20261016/replay.py`, `t4a_posthoc_20261016/*`; also `src/scripts/observation_value_diagnostic.py`, `research/colab_cli/runtime.py` |
| `write_once` | 6 | 6 | 1 — `t34_round0/common.py`, `colab_cli/supervisor.py`, `s1_mature_20261018/admission.py`, `t34_shadow`, `sequential_*` |
| `on_ac_power` | 10 | 6 | 4 — `apex_veto_{,v5_,v7_}strict*/strict_run.py`, `sequential_strict_template/sequential_runner.py` |
| `source_closure` | 9 | 7 | 3 — apex_veto strict runners; also `src/scripts/observation_value_diagnostic.py` |
| `source_manifest` | 8 | 8 | 1 — `q0q1`, `s1_mature`, `t23_r0_screen`, `redesign_t7_phase_r`, `colab_cli/acceptance.py` |

Note the bodies are *not* interchangeable: `sha256_file` alone has 19 variants (e.g.
`research/t34_exit_b8/make_manifest.py` streams with `hashlib.file_digest`;
`research/t34_exit_b8_fit/make_manifest.py` uses `path.read_bytes()`). The digests agree, but
`write_once`/`source_manifest` have 6–8 distinct semantics (different path filters, error types),
so a shared version must pick one contract and be tested; it must never be retrofitted into a
package whose manifest hashes are already bound.

Affected ratified studies: **none if done as "new code only"**. Retrofitting would touch
`t34_round0` (`common.py:28,54`, `runner.py:118`, `make_manifest.py:194`), `t34_pathb`
(`bank.py:22`, `cost_smoke_v2.py:46`, `source_binding_v2.py:61`, `registration_retention_v2.py:64`),
`t34_exit_b8` (`make_manifest.py:75`), `t34_exit_b8_fit` (`make_manifest.py:28`, `cache.py:41`,
`prepare_a3_rows.py:20`) and `colab_cli` (`acceptance.py:47,215`, `runtime.py:58,93`,
`supervisor.py:57,63`, `source.py:35`, `acceptance_probe.py:37`) — **must wait until those studies
close**.

### 2. One manifest builder for T34-era packages — High value, Medium risk

Five near-parallel `make_manifest.py` files (plus `t34_pathb/make_unpaced_manifest.py` and
`s1_mature_20261018/manifests.py`):
`research/t34_round0/make_manifest.py` (646 lines), `research/t34_exit_b8/make_manifest.py` (518),
`research/t34_exit_b8_fit/make_manifest.py` (276), `research/t34_round0d/make_manifest.py`,
`research/t34_claim_a/make_manifest.py`, `research/t33_batched_teacher/make_manifest.py`.
Each re-implements file hashing + source binding. Proposal: extract a `research/manifest_kit`
(or put it under the module from #1) for the *next* T34 study, then migrate the old builders only
after closure.

Affected: `t34_round0`, `t34_exit_b8`, `t34_exit_b8_fit` (and `t34_pathb` via
`make_unpaced_manifest.py`) — **wait for closure**. `t34_claim_a`, `t34_round0d`, `t33_batched_teacher`
are not in the ratified list, but `t33_batched_teacher` is imported by `t34_exit_b8`, so it is
frozen too.

### 3. Exact duplicate `_load_policy` inside `t34_exit_b8` — Medium value, Low effort, Blocked

`research/t34_exit_b8/benchmark.py:171` and `research/t34_exit_b8/label.py:52` are AST-identical
(the `--policy-factory module:callable` student-adapter loader). Move to one function in the same
package (e.g. a `policy_factory.py`). Affected: **t34_exit_b8 — wait for closure**.

### 4. B8 path / served-B8 environment helpers — Low-medium value, Low risk

- `b8_path(arm)` is identical in `research/b8_death_census_20261014/census.py:134` and
  `research/planner_feasibility_20261014/planner.py:605`
  (`B8_DIR / f"B8_s{int(arm[-1])}" / "final.pth"`).
- `_served_b8_environment()` exists twice with different semantics:
  `research/t12_b8_serving_20261017/measurements.py:502` pops the env keys and sets
  `CHECKPOINT_NAME`; `research/t12b_check4_20261017/run.py:303` saves with `.get` and sets
  `"b8-s3"`. A single helper keyed on the served checkpoint name would remove the drift risk.
- `research/t34_exit_b8_fit/runner.py:40` `_load_b8` wraps `load_ego2s_checkpoint` with an
  arch sniff; it is the only B8 loader that knows the exit-fit tower format.

Recommendation: add `b8_checkpoint_path(seed)` and a `served_checkpoint_env(name)` context
manager next to `load_ego2s_checkpoint` in `src/model/ego2s_network.py` (or a sibling) for new
code. Affected ratified studies: none (the duplicates are in closed 10-14/10-17 packages);
`t34_exit_b8_fit` only if `_load_b8` is moved — **wait**.

### 5. Seed derivation helpers — Medium value, High risk

`uint32_seed` (8 defs, 5 distinct bodies; identical group of 4 in `apex_safety_20260926/dev_screen.py`,
`apex_veto_{,v5_,v7_}strict*/strict_audit.py`) and `seed_digest` (8 defs, 4 bodies; identical group
of 5 in `t11_local30`, `t12_b8_serving`, `t12b_check4`, `t4f_return` `seed_audit.py`; also
`t34_claim_a`). Seed derivation defines preregistered seed banks, so different bodies may be
intentional. Proposal: document the canonical derivation for new studies in the #1 module; never
change existing ones (would silently re-key historic seed banks). No ratified-study impact unless
retrofitted.

### 6. Superseded strict/serving runner lineage — Low value now, High risk

- `research/apex_veto_strict_20260927/strict_run.py` (2138 lines),
  `apex_veto_v5_strict_20261001/strict_run.py` (2417), `apex_veto_v7_strict_20261002/strict_run.py`
  (2625), superseded by `research/sequential_strict_template/sequential_runner.py` (3524).
  v5 vs v7 differ by only 525 diff lines; `_git`, `read_json`, `on_ac_power`, `source_closure`,
  `_mean` are copied across all three.
- `research/frp3_checkpoint_serving_20261005/{fill_strict_pin,serving_audit}.py` vs
  `research/t12_b8_serving_20261017/` copies (279 / 1304 diff lines).
- `research/apex_veto_*_serving_*/schema.py` + `serving_run.py`: `canonical_json` identical ×7,
  `check_output_root` identical ×5.
- `earlier_namespaces` (13 defs, 4 bodies) is copied across every apex_veto v3–v8 screen/sweep.

These are **not dead**: each older package still has 4–11 importers outside itself (the veto
packages import each other in a chain, and `sequential_strict_template` has 13), and the v8/v7/v5
veto variants are still rollback targets for serving. Proposal: mark as "historical, superseded by
`sequential_strict_template`" in `research/INDEX.md`; no code moves. Not in any ratified closure as
far as the direct-import scan shows, except `apex_safety_20260926` and `apex_veto_serving_20261001`,
which ratified packages import (`t34_round0` imports `apex_safety_20260926`) — so those two are
frozen.

### 7. Near-duplicate small scripts — Low value

- `research/task_aligned_20260924/stage_cli.py` vs
  `research/task_aligned_continuation_20260924/stage_cli.py` (744 diff lines; both no-code-ref
  leaf CLIs).
- `research/t11_local30_20261017/bind_launch.py` vs `research/t4c_20261016/bind_launch.py`
  (126 diff lines on a 72-line file — mostly rewritten).
- `cpu_model` ×7 (identical ×4 in `runpod_fanout/{pod_agent,seed_agent,sls_handler}.py` and
  `sequential_strict_template/strict_sls_handler.py`).
- `environment()` watcher helper identical ×5 (`t11_screen_20261017/watch.py`,
  `t3_screen_20261015/watch_t3.py`, `watch_t4c.py`, `t4a_screen_20261016/watch_t4a.py`, ...).

Proposal: leave; fold into #1 only for new code.

### 8. `src/scripts` helper copies — Low value, Low risk

`src/scripts/observation_value_diagnostic.py` defines its own `canonical` (l.53), `sha256_file`
(l.57), `_atomic_json` (l.77), `source_closure` (l.99); `src/scripts/pqn_correctness_diagnostic.py:133`
another `sha256_file`. These are the only provenance-helper copies inside `src/` and are the safest
first users of the #1 module. Ratified impact: unknown whether any ratified manifest hashes these
two scripts — check the five studies' manifests before touching them (unclear from this scan).

### 9. Production import of research code — Process item

`web/backend/pathb.py:13` imports `research.t34_pathb`. When Path B closes, either promote the
used symbols into `src/` or `web/backend/`, or freeze the dependency explicitly. Affected:
**t34_pathb — wait for closure**.

## Frozen-closure map (do not touch until the listed study closes)

Direct `research.*` imports from the ratified packages and their first-hop dependencies:
`apex_safety_20260926`, `apex_veto_serving_20261001`, `b8_death_census_20261014`, `colab_cli`,
`compute`, `fixed_search`, `perf_sim_20261007`, `phase0_search_20261019`, `redesign_m2_20261008`,
`redesign_m3c_20261008`, `redesign_scope_20261007`, `redesign_t7_phase_r_20261016`,
`s1_mature_20261018`, `t15_mixed_roster_20261017`, `t1_commitment_20261015`, `t33_batched_teacher`,
`t34_exit_b8`, `t34_exit_b8_fit`, `t34_pathb`, `t34_round0`, `t34_shadow`,
`t3_rollout_labels_20261015`, `t4e_return_20261016`. Deeper transitive hops were not traced
(unclear); a full closure should be read from each study's own manifest before any edit.

## What NOT to do

- Do not delete any "no code importer" research file: all are cited by docs/manifests.
- Do not dedupe old packages in place: ratified/closed results bind exact source hashes.
- Do not unify seed derivation retroactively.
