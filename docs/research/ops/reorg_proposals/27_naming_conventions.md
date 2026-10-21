# 27 — Naming conventions: survey, inconsistencies, proposal (2026-10-20 PT)

Scope 27 of the org30 pass. **Read-only audit**: nothing was renamed. Sources are directory and branch
listings taken 2026-10-20: `research/`, `docs/research/` (+ `specs/`, `advice/`), all 430
entries in `~/Projects/ml/snake-dqn-artifacts/` (outside the repo),
`git branch -a` (391 lines) and `git worktree list` (177 entries). Mapping of packages to T-numbers comes from
the "T" column of [`research/INDEX.md`](../../../../research/INDEX.md).

## 1. De-facto conventions (what the tree actually does)

| Surface | Dominant pattern | Examples | Separator / date |
|---|---|---|---|
| `research/<pkg>/` (Python packages) | `<topic>_<YYYYMMDD>` until 2026-10-17; **undated** `t3N_<topic>` from T33 on | `apex_veto_v8_strict_20261003`, `t12c_release_20261017`; `t34_round0`, `t35_outcome_head` | snake_case (required for import), compact date |
| `docs/research/*.md` (results, designs) | `<topic>_<YYYY-MM-DD>.md`, suffixes `_evidence/`, `_results`, `_erratum…` | `t8_probes_2026-10-17.md`, `b8_s3_release_erratum_mixed_watch_2026-10-17.md` | snake_case, ISO date **last** |
| `docs/research/specs/` | before 10-19: `tN_<topic>_spec_<YYYY-MM-DD>.md` / `_prereg_`; from 10-19: `<YYYY-MM-DD>_<topic>.md` | `t4e_prereg_amendment1_2026-10-16.md`; `2026-10-20_t34_round0_prereg.md` | ISO date moved from **last** to **first** |
| `docs/research/advice/` | `<YYYY-MM-DD>_<topic>[_<persona>].md`, one subdir `2026-10-20_a100/` | `2026-10-20_compute_portfolio_astra_skeptic.md`, `…_FABLE_JUDGMENT.md`, `…_QUESTION.md` | ISO date first; role suffix mixed case |
| Inside a T34/T35 package | `PREREGISTRATION.md`, `RESULT.md`, `RESULT_PRIMARY.md`, `VERIFICATION_ROUND0D.md`, `BANK_ROUND0D.json` | `research/t34_round0d/` | UPPER_SNAKE status docs |
| Artifacts dirs (`snake-dqn-artifacts/`) | `<topic>-<YYYYMMDD>[-<attempt>]` | `apex-veto-v8-strict-20261003`, `redesign-m4a2-gpu-20261012-try9`, `t34-round0-20261020` | kebab-case, compact date |
| Branches | `<topic>-<YYYYMMDD>`; prefixes `codex/`, `claude/`, `worktree-agent-<hex>`, `worktree-wf_<id>-<n>` | `t34-retention-fixes3-20261020`, `codex/pqn-autoreset-b3t` | kebab-case, compact date |
| Worktrees | sibling dirs `snake-dqn-<short-topic>`; tool-made ones under `.claude/worktrees/` and `/private/tmp/…` | `snake-dqn-retention-fixes4`, `snake-dqn-portfolio-skeptic` | kebab, usually undated |

So the repo already has a stable *shape* — **topic + date, snake in Python/docs, kebab on disk/git** — but at
least four date spellings and two date positions.

## 2. Inconsistencies found

### 2.1 Dates
1. **Four spellings**: `YYYYMMDD` (research pkgs, artifacts, branches, `final_run_20261019/`,
   `q0_mass_mac_timing_20261018.json`), `YYYY-MM-DD` (docs, specs, advice), `MM-DD` (CAMPAIGN_LOG rows `10-20 ~18:59`),
   and **no date** (all `t34_*`/`t35_*` packages, `colab_cli`, `compute`, `runpod_fanout`, `sequential_phase_r`,
   `t16_colab`; artifacts `t33-reviews`, `t34-sol`, `t34-pathb-v2`, `t16-colab`, `org30`, `governance`).
2. **Two positions in the same folder**: `docs/research/specs/` holds both `t11_prereg_2026-10-17.md` (date last)
   and `2026-10-20_t34_round0_prereg.md` (date first). Sorting by name no longer sorts by time.
3. **Time-zone drift**: artifacts dir `t34-round0d-prep-20261021` was created **2026-10-20 18:49 PT** (mtime) —
   stamped with the UTC date. Project rule is Pacific time. Same work also appears dated 10-19 and 10-20 across
   surfaces: spec `2026-10-19_t34_v2_fitcache.md`, branch `t34-v2-fitcache-20261019`, artifacts
   `t34-v2-fitcache-20261020`.
4. Docs-level JSON evidence in `docs/research/` mixes `t4c_data_audit_2026-10-16.json` with
   `q0_mass_worker_parity_20261018.json`.

### 2.2 T-numbers
1. **Two "T35"s.** `research/t35_outcome_head` (outcome-head probe, KILL) and `research/t35_two_deviation`
   (two-deviation rescue, pooled KILL) are unrelated probes from different GPU-expert sources
   (`gpu_expert_3.md` vs `gpu_expert_2.md`) that share the label "T35"; `research/INDEX.md` lists both as
   plain `T35`. Earlier siblings got letter suffixes (`t4a`…`t4f`, `t12b/c`, `t13b`, `t15b`); T35 did not.
2. **Branch T-prefixes ≠ task T-numbers.** On branches the `tNN-` prefix behaves like a session/batch counter:
   `t31-` covers nine unrelated branches (`t31-colab-safety`, `t31-openmp-tests`, `t31-q0-init-fill`,
   `t31-q1-a5`, `t31-q1-strategy`, …); `t33-` covers both `t33-batched-teacher` and `t33-colab-cli`;
   `t32-` covers `t32-q1-a4` and `t32-search-teacher`; `t30-` covers `t30-colab-cli-advice` and `t30-keepalive`.
   Meanwhile research `t33_batched_teacher` is "T33" and the search-teacher campaign is "T34" (`t32-search-teacher`
   branch). No registry of T-numbers was found (grep for "T-number"/"T36" in docs/AGENTS: none).
3. **Sub-labels inside T34** are free-form: `L1`/`L2` (INDEX), `Phase 0` (`phase0_search_20261019` — no `t34_`
   prefix), `round0`, `round0d`, `pathb`, `pathb2`, `v2`, `claim_a`, `shadow`, `exit_b8`, `exit_b8_fit`.
4. Pre-T series reuse the letter: `redesign_t4_20261015`, `redesign_t7_phase_r_20261016`, artifacts
   `redesign-t3-mac-20261015` vs `t3-diagnostics-20261016` / `t3-rollout-labels-20261015` — same T3, two prefixes.

### 2.3 Attempts, versions and retries
Seven different retry spellings: `-try1`/`-try9`, `-r2`/`-r3`, `-a1`, `-attempt1-preflight-failed`, `-v2`/`-v3`,
`fixes1`…`fixes6`, `-<hash8>` (`t34-retention-oc-bare-unpaced-52dec74f`), plus outcome words in names
(`-refused1`, `-g2fail4`, `-nostock2`, `-oom3`, `-SMOKE2`, `-evidence-only`). Mixed case (`SMOKE`, `FABLE_JUDGMENT`,
`QUESTION`) appears only sporadically.

### 2.4 Worktrees and branches
1. Worktree name ≠ branch: `snake-dqn-adv-gpu` → `t30-keepalive-20261018`, `snake-dqn-adv-league` →
   `t31-q0-parallel-20261018`, `snake-dqn-adv-rl` → `adv3-overnight-20261018`, `snake-dqn-pactor` →
   `t25-colab-queue-20261017`. Many worktrees are detached with no hint of purpose
   (`snake-dqn-m4a2`, `snake-dqn-perf`, `snake-dqn-q1-baseline`).
2. Worktrees live in four places: repo siblings, `.claude/worktrees/`, `/private/tmp/…`, and **inside**
   `snake-dqn-artifacts/` (`s1-mature-20261018/barrier-fix-05/baseline474`, `t34-pathb-v2/verification-20261020/baseline`).
3. ~45 `worktree-agent-<hex>` and ~20 `worktree-wf_<id>-<n>` branches are tool-generated and opaque.
4. Old branches have no date (`frp-v3`, `m6-reward`, `veto-v9`, `codex/rl-repair-*`).

### 2.5 Loose files at artifacts root
~90 of the 430 artifacts entries are loose logs/scripts (`codex-t4e.err/.out`, `m4a2-pod-retry.sh`,
`*.console.log`, `HANDOFF_2026-10-05.md`, `decision_memo_training_bet_and_LH1_2026-10-04.json`) next to run dirs.

## 3. Proposed convention (new work only)

One identifier, spelled per surface, always derivable from the others.

**Work ID** = `t<NN><letter?>` + `_` + `<slug>`, e.g. `t36a_outcome_head`.
- `NN` = campaign/task number, allocated in **one place**: a new "T-number register" table (proposal: a section
  at the top of `research/INDEX.md`, owned by whoever updates that index). One row per *letter*, not per number.
- Independent probes under the same task get distinct letters (`t35a`, `t35b`), never a bare shared number.
- `<slug>`: 1–4 lowercase words, no version numbers; versions go in the attempt suffix.
- Non-task work (perf, ops, colab tooling, advice) keeps a plain slug with no `t` prefix — do **not** borrow the
  next T-number for a branch.

**Date** = Pacific calendar date, **`YYYYMMDD` in machine names** (dirs, packages, branches, artifacts, JSON) and
**`YYYY-MM-DD` only in prose/markdown filenames**; always the date the work was *started*, reused unchanged by
every surface for that work (no re-dating the artifacts dir to the run day).

| Surface | Pattern | Example |
|---|---|---|
| Research package | `research/<workid>_<YYYYMMDD>/` | `research/t36a_outcome_head_20261021/` |
| Package status docs | `PREREGISTRATION.md`, `AMENDMENT_<n>.md`, `RESULT.md`, `RESULT_<part>.md`, `VERIFICATION.md` | as today |
| Spec / prereg | `docs/research/specs/<YYYY-MM-DD>_<workid>_<kind>.md`, kind ∈ {spec, prereg, amendment<n>, fixes<n>} | `2026-10-21_t36a_outcome_head_prereg.md` |
| Advice | `docs/research/advice/<YYYY-MM-DD>_<topic>/` dir with `QUESTION.md`, `<persona>.md`, `JUDGMENT.md` | `2026-10-21_t36_next/astra_skeptic.md` |
| Result note | `docs/research/<YYYY-MM-DD>_<workid>_result.md` | |
| Artifacts | `snake-dqn-artifacts/<workid-kebab>-<YYYYMMDD>/<attempt>/` with attempt = `a01`, `a02`… and a `STATUS` file inside (ok / refused / oom / failed) instead of outcome words in the name; logs go *inside* the run dir | `t36a-outcome-head-20261021/a02/` |
| Branch | `<workid-kebab>-<YYYYMMDD>[-<part>]`; prefixes `codex/`, `claude/` only when the author tool matters | `t36a-outcome-head-20261021-fixes2` |
| Worktree | `snake-dqn-wt/<branch-name>` (one parent dir, name = branch); never inside `snake-dqn-artifacts/` | `~/Projects/ml/snake-dqn-wt/t36a-outcome-head-20261021` |

Rationale for date-last in machine names: it matches the 70+ existing `research/*_YYYYMMDD` packages and
artifacts dirs, so `ls` groups a work's attempts together. Date-first stays for specs/advice because those
folders are read chronologically and are already mostly date-first since 10-19.

## 4. Low-risk migration plan

Hard rule: **no renames of ratified or evidence paths** — research packages, artifacts referenced by receipts/
hashes, `final_run_20261019/`, governance and existing specs/advice keep their bytes and names forever.

1. **Register (docs only, now).** Add the T-number register to `research/INDEX.md` (owner of that file) listing every
   existing T-number with its packages, and disambiguate retroactively *in the index only*:
   `T35a = t35_outcome_head`, `T35b = t35_two_deviation`; note that branch prefixes `t30-`…`t33-` are batch
   counters, not task IDs. Directory names stay as they are.
2. **Document the convention** in one short section of `AGENTS.md` (owner-gated file — propose, don't edit here)
   and in the `codex-handoff`/Sol prompt template so new specs, packages and branches are named at birth.
3. **Apply to new work only**, starting with the next allocated number. First adopter: whatever follows T35.
4. **Fix-forward small leaks**: new artifacts dirs use the Pacific start date (avoid repeats of
   `t34-round0d-prep-20261021`); new run logs go inside their run dir rather than artifacts root.
5. **Optional, non-renaming cleanup** (separate scopes, needs owner OK): an artifacts-root `_loose/` index file
   that *lists* stray logs by run; a pruning pass for merged `worktree-agent-*`/`worktree-wf_*` branches
   (deletion is owner-only, see the worktree inventory in `../WORKTREE_INVENTORY_2026-10-20.md`).
6. **Lint (later, warn-only)**: a tiny script that checks new paths added in a commit against the patterns in §3
   and warns; never blocks, never touches existing paths.

## 5. Open / unclear
- Whether "T34 L1/L2" denote lanes or levels is not stated in the files read here.
- Why `phase0_search_20261019` lacks a `t34_` prefix while INDEX tags it "T34 Phase 0" is unclear (likely
  it predates the T34 label).
