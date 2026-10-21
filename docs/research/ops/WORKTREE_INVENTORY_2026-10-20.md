# Worktree / branch / tmux inventory — 2026-10-20

Snapshot taken 2026-10-20 ~19:17 PT from `/Users/josenunez/Projects/ml/snake-dqn` (`git fetch` first; merge checks against `origin/main` @ e301cfeb). **Read-only inventory: nothing was removed, pruned, deleted or moved.** The removal commands below are written out but were not run.

## Totals

| Class | Worktrees | Disk |
|---|---:|---:|
| PROTECTED | 12 | 6.69 GB |
| ACTIVE | 15 | 2.68 GB |
| NEEDS-OWNER | 24 | 3.94 GB |
| SAFE-TO-REMOVE | 94 | 12.17 GB |
| **All** | **145** | **25.47 GB** |

**Reclaimable by running the SAFE-TO-REMOVE commands: 12.17 GB** (94 worktrees). Sizes come from `du -sk` and include ignored files such as `saved_snakes/` and caches, which `git worktree remove` also deletes. Branches stay; removing a worktree does not delete its branch.

## Classification rules applied

- **PROTECTED**: never touch. Covers snake-dqn-round0-frozen-01, snake-dqn-round0-fixes2, snake-dqn-t35-2dev-frozen and snake-dqn-retention-fixes6 (until the Path B amendment lands). Also covers worktrees named in `research/t34_round0/README.md` or `research/t34_pathb/README_V2.md` (both still DRAFT_PENDING_OPERATOR_RATIFICATION, protected conservatively), the primary checkout (git-sync tmux), and snake-dqn-search-teacher (phase0-selection tmux, running runner).
- **ACTIVE**: in use by a running process, by one of tonight's review or advice agents, or by an agent worktree (`.claude/worktrees/agent-*`) whose HEAD is newer than 3 h or unmerged. Agent worktrees in this class are marked ACTIVE-AGENT.
- **NEEDS-OWNER**: dirty or unmerged and not active. Also covers clean, merged worktrees that are still referenced by `HANDOFF_2026-10-20.md` or cited as evidence baselines by committed verification docs, so they are held and not removable.
- **SAFE-TO-REMOVE**: clean (`status --porcelain` empty), HEAD is an ancestor of `origin/main`, not referenced by the three docs or by verification evidence, no process cwd, no tmux pane.

Reference check: substring search for the dir name, branch name and HEAD prefix in `docs/research/final_run_20261019/HANDOFF_2026-10-20.md`, `research/t34_round0/README.md` and `research/t34_pathb/README_V2.md`. False positives were cleared by hand: `s1-mature` is an artifacts path, `retention-fixes` is a prefix of `-fixes3/4/6`, and `baseline` matched a generic word. Process check: `ps`, `lsof -d cwd`, and `tmux` pane paths.

## Running processes and tmux

| tmux session | pane cwd | what | class of dir |
|---|---|---|---|
| git-sync | snake-dqn (main) | `snake-dqn-artifacts/git-sync/sync.sh`, pull-only ff of main | PROTECTED (never removable) |
| phase0-selection | snake-dqn-search-teacher | `research.phase0_search_20261019.runner run` (up ~30 h) | PROTECTED (never removable) |
| t34-round0-c000 | snake-dqn-round0-frozen-01 | `research.t34_round0.dispatch run` + colab_cli supervisor (chunk-000) | PROTECTED |

Non-tmux: Codex (Sol) `codex-handoff.sh` + pytest in **snake-dqn-test-hygiene** (ACTIVE). `snake-dqn-artifacts/git-sync/` is a plain directory, not a worktree, and must never be removed.

## PROTECTED (12, 6.69 GB)

| Worktree (under ~/Projects/ml/) | Branch | HEAD | Dirty | Merged | Last commit (PT) | Size | Doc refs | Reason |
|---|---|---|---:|---|---|---:|---|---|
| `snake-dqn` | `main` | e301cfeb | 0 | yes | 2026-10-20 19:01 | 4.73 GB | HANDOFF, R0-README, README_V2 | primary checkout (main); git-sync tmux pane cwd |
| `snake-dqn-engine-fanout` | `t34-engine-fanout-20261020` | 67f76aea | 2 | yes | 2026-10-20 14:49 | 183 MB | HANDOFF, R0-README | named in round0 README + HANDOFF (also dirty) |
| `snake-dqn-retention-fixes6` | `t34-retention-fixes6-20261020` | d1a3caec | 0 | yes | 2026-10-20 14:56 | 183 MB | HANDOFF, README_V2 | protected until Path B amendment lands (HANDOFF + README_V2) |
| `snake-dqn-retention-trace-97232d51-20261020` | `(detached)` | 97232d51 | 0 | yes | 2026-10-20 14:45 | 183 MB | HANDOFF | frozen source 97232d51 for Path B ratification (HANDOFF) |
| `snake-dqn-round0` | `t34-round0-prereg-20261020` | 9e241fb2 | 0 | yes | 2026-10-20 13:28 | 186 MB | HANDOFF, R0-README | Round 0 prereg branch; named in round0 README + HANDOFF |
| `snake-dqn-round0-fixes2` | `t34-round0-fixes2-20261020` | 652f2b1e | 0 | yes | 2026-10-20 16:45 | 183 MB | HANDOFF, R0-README | Round 0 authoring/ledger (named in round0 README) |
| `snake-dqn-round0-frozen-01` | `(detached)` | 882b0fa2 | 0 | yes | 2026-10-20 16:40 | 179 MB | HANDOFF, R0-README | Round 0 frozen dispatch source; tmux t34-round0-c000 running from it |
| `snake-dqn-round0-integrity` | `t34-round0-integrity-20261020` | 0c7810df | 0 | yes | 2026-10-20 16:03 | 183 MB | HANDOFF, R0-README | named in round0 README + HANDOFF |
| `snake-dqn-search-teacher` | `t32-search-teacher-20261019` | 002b82a6 | 0 | yes | 2026-10-19 12:58 | 177 MB | — | phase0-selection tmux session + running phase0 runner (never removable) |
| `snake-dqn-t35-2dev-frozen` | `(detached)` | 321ba7c7 | 0 | yes | 2026-10-20 18:40 | 183 MB | — | T35 two-deviation frozen source |
| `snake-dqn-unpaced-colab` | `t34-unpaced-colab-20261020` | 8ddf08aa | 0 | yes | 2026-10-20 13:53 | 182 MB | HANDOFF, README_V2 | named in README_V2 + HANDOFF |
| `snake-dqn-v2-pathb2` | `t34-v2-pathb2-20261019` | dafc78a4 | 0 | yes | 2026-10-20 02:36 | 184 MB | README_V2 | named in README_V2 |

## ACTIVE (15, 2.68 GB)

| Worktree (under ~/Projects/ml/) | Branch | HEAD | Dirty | Merged | Last commit (PT) | Size | Doc refs | Reason |
|---|---|---|---:|---|---|---:|---|---|
| `<session-scratch>/scratchpad/r0d` | `(detached)` | 8fc8e00e | 0 | yes | 2026-10-20 18:50 | 184 MB | — | this session's Round 0-D scratch worktree (0.5 h) |
| `snake-dqn-advice-a100-1` | `(detached)` | 6f1e751f | 0 | yes | 2026-10-20 17:26 | 180 MB | — | tonight's A100 advice worktree (gpu-expert agents, 1.9 h) |
| `snake-dqn-advice-a100-2` | `(detached)` | 6f1e751f | 0 | yes | 2026-10-20 17:26 | 180 MB | — | tonight's A100 advice worktree (gpu-expert agents, 1.9 h) |
| `snake-dqn-advice-a100-3` | `(detached)` | 6f1e751f | 0 | yes | 2026-10-20 17:26 | 180 MB | — | tonight's A100 advice worktree (gpu-expert agents, 1.9 h) |
| `snake-dqn-advice-a100-4` | `(detached)` | 6f1e751f | 0 | yes | 2026-10-20 17:26 | 180 MB | — | tonight's A100 advice worktree (gpu-expert agents, 1.9 h) |
| `snake-dqn-advice-a100-5` | `(detached)` | 6f1e751f | 0 | yes | 2026-10-20 17:26 | 180 MB | — | tonight's A100 advice worktree (gpu-expert agents, 1.9 h) |
| `snake-dqn-fable-review-bare-unpaced` | `(detached)` | 9de16df8 | 0 | no (+1) | 2026-10-20 18:22 | 184 MB | — | tonight's review worktree; HEAD unmerged (1 ahead) |
| `snake-dqn-rereview-bare-unpaced` | `(detached)` | 52dec74f | 0 | no (+3) | 2026-10-20 18:58 | 183 MB | — | tonight's re-review worktree; HEAD unmerged (3 ahead) |
| `snake-dqn-review-bare-unpaced` | `(detached)` | 9de16df8 | 1 | no (+1) | 2026-10-20 18:22 | 184 MB | — | tonight's review worktree (reviewer-bare-unpaced); dirty, unmerged |
| `snake-dqn-test-hygiene` | `t34-test-hygiene-20261020` | e301cfeb | 7 | yes | 2026-10-20 19:01 | 189 MB | HANDOFF | Codex (Sol) codex-handoff running in it; pytest procs |
| `snake-dqn/.claude/worktrees/agent-a1ac340a47c812487` | `worktree-agent-a1ac340a47c812487` | 182d85e2 | 0 | yes | 2026-10-20 18:59 | 184 MB | — | ACTIVE-AGENT: commit 0.3 h old |
| `snake-dqn/.claude/worktrees/agent-a20ad0e8cde207090` | `worktree-agent-a20ad0e8cde207090` | 52dec74f | 0 | no (+3) | 2026-10-20 18:58 | 184 MB | — | ACTIVE-AGENT: commit 0.3 h old, unmerged 3 ahead |
| `snake-dqn/.claude/worktrees/agent-a276d68378b46d40d` | `worktree-agent-a276d68378b46d40d` | 8fc8e00e | 0 | yes | 2026-10-20 18:50 | 185 MB | — | ACTIVE-AGENT: commit 0.5 h old |
| `snake-dqn/.claude/worktrees/agent-a8dd3118925e0c093` | `worktree-agent-a8dd3118925e0c093` | e301cfeb | 0 | yes | 2026-10-20 19:01 | 184 MB | — | ACTIVE-AGENT: commit 0.3 h old |
| `snake-dqn/.claude/worktrees/agent-abbec5873683a9168` | `worktree-agent-abbec5873683a9168` | 321ba7c7 | 0 | yes | 2026-10-20 18:40 | 183 MB | — | ACTIVE-AGENT: commit 0.6 h old |

## NEEDS-OWNER (24, 3.94 GB)

| Worktree (under ~/Projects/ml/) | Branch | HEAD | Dirty | Merged | Last commit (PT) | Size | Doc refs | Reason |
|---|---|---|---:|---|---|---:|---|---|
| `snake-dqn-a5-speed-astra` | `t34-a5-speed-astra-20261020` | 5417ffb6 | 0 | yes | 2026-10-20 13:52 | 179 MB | — | referenced: HANDOFF: "Astra speed answer pending" on this branch |
| `snake-dqn-advice-t34-product` | `t34-advice-product-20261019` | de43959b | 1 | yes | 2026-10-19 18:37 | 172 MB | — | dirty (1 untracked/modified) |
| `snake-dqn-advice-t34-rl` | `t34-advice-rl-20261019` | f7bafb7b | 1 | yes | 2026-10-19 18:37 | 172 MB | — | dirty (1 untracked/modified) |
| `snake-dqn-advice-t34-skeptic` | `t34-advice-skeptic-20261019` | e740bb94 | 1 | yes | 2026-10-19 18:37 | 172 MB | — | dirty (1 untracked/modified) |
| `snake-dqn-artifacts/s1-mature-20261018/barrier-fix-05/baseline474` | `(detached)` | 474b3c2e | 0 | yes | 2026-10-18 19:50 | 174 MB | — | referenced: evidence baseline cited by research/t34_round0/TABULAR_SEED_AUDIT.json |
| `snake-dqn-artifacts/t34-pathb-v2/verification-20261020/baseline` | `(detached)` | 45ecb926 | 0 | yes | 2026-10-20 01:33 | 180 MB | — (false positive) | referenced: evidence baseline cited by research/t34_pathb/VERIFICATION_V2.md + retention trace receipt |
| `snake-dqn-artifacts/t34-round0-integrity-20261020/baseline-4cced4f9` | `(detached)` | 4cced4f9 | 0 | yes | 2026-10-20 15:34 | 179 MB | — | referenced: evidence baseline cited by research/t34_round0/VERIFICATION_INTEGRITY_SIZING.md |
| `snake-dqn-carryover` | `colab-carryover-fix-20261019` | dab00823 | 0 | yes | 2026-10-19 22:15 | 176 MB | HANDOFF | referenced: HEAD dab00823 = Colab CLI qualification commit named in HANDOFF |
| `snake-dqn-colab-safety` | `t31-colab-safety-20261018` | bdccf3a6 | 0 | no (+19) | 2026-10-19 10:56 | 182 MB | — | HEAD unmerged (19 ahead of origin/main) |
| `snake-dqn-pathb` | `t34-pathb-20261019` | dc7cd64b | 2 | yes | 2026-10-19 20:53 | 289 MB | — | dirty (2 untracked/modified) |
| `snake-dqn-retention-fixes3` | `t34-retention-fixes3-20261020` | c40129bb | 0 | yes | 2026-10-20 14:06 | 182 MB | HANDOFF | referenced: named in HANDOFF |
| `snake-dqn-retention-fixes4` | `t34-retention-fixes4-20261020` | 96f93ee1 | 0 | yes | 2026-10-20 14:20 | 182 MB | HANDOFF | referenced: named in HANDOFF |
| `snake-dqn-round0-baseline-check-20261020` | `(detached)` | 4c19a83e | 0 | yes | 2026-10-20 12:09 | 178 MB | — | referenced: evidence baseline cited by research/t34_round0/VERIFICATION.md |
| `snake-dqn-round0-fixes1` | `t34-round0-fixes1-20261020` | 2e7dd0b1 | 0 | yes | 2026-10-20 16:29 | 183 MB | HANDOFF | referenced: named in HANDOFF |
| `snake-dqn-shadow-batch` | `t34-shadow-batch-20261020` | 60123c3a | 0 | yes | 2026-10-20 13:12 | 183 MB | HANDOFF | referenced: branch named in HANDOFF |
| `snake-dqn-t12` | `t12-20261017` | d751c96e | 1 | yes | 2026-10-17 01:53 | 215 MB | — | dirty (1 untracked/modified) |
| `snake-dqn-t13` | `t13-20261017` | b43998b8 | 1 | yes | 2026-10-17 06:28 | 114 MB | — | dirty (1 untracked/modified) |
| `snake-dqn-t14` | `t14-20261017` | 57eee56b | 1 | yes | 2026-10-17 08:27 | 252 MB | — | dirty (1 untracked/modified) |
| `snake-dqn-t15` | `t15-20261017` | 5ce43c09 | 1 | yes | 2026-10-17 09:25 | 252 MB | — | dirty (1 untracked/modified) |
| `snake-dqn-t1a` | `t1a-20261015` | d5f095db | 1 | yes | 2026-10-14 22:39 | 76 MB | — | dirty (1 untracked/modified) |
| `snake-dqn-t1c` | `t1c-20261015` | d45f5a21 | 1 | yes | 2026-10-14 22:33 | 76 MB | — | dirty (1 untracked/modified) |
| `snake-dqn-t2` | `t2-20261015` | 6a868b6d | 1 | yes | 2026-10-14 21:31 | 77 MB | — | dirty (1 untracked/modified) |
| `snake-dqn-t4p` | `t4p-20261016` | 0030c36d | 4 | yes | 2026-10-17 01:49 | 112 MB | — | dirty (4 untracked/modified) |
| `snake-dqn/.claude/worktrees/agent-a829dec67bfdb0262` | `worktree-agent-a829dec67bfdb0262` | 6db5e04d | 1 | yes | 2026-10-14 18:20 | 74 MB | — | dirty (1 untracked/modified) |

## SAFE-TO-REMOVE (94, 12.17 GB)

| Worktree (under ~/Projects/ml/) | Branch | HEAD | Dirty | Merged | Last commit (PT) | Size | Doc refs | Reason |
|---|---|---|---:|---|---|---:|---|---|
| `/private/tmp/s1-baseline-verify-35214489` | `(detached)` | 35214489 | 0 | yes | 2026-10-18 19:16 | 173 MB | — | clean, merged, unreferenced, no process |
| `/private/tmp/snake-dqn-baseline-ad33.S3prMJ` | `(detached)` | ad33f390 | 0 | yes | 2026-10-18 14:57 | 172 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-adv-gpu` | `t30-keepalive-20261018` | e0541e78 | 0 | yes | 2026-10-18 08:06 | 161 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-adv-league` | `t31-q0-parallel-20261018` | 3ee073ca | 0 | yes | 2026-10-18 12:24 | 166 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-adv-q1-skeptic` | `q1-next-skeptic-20261018` | 08d12ae1 | 0 | yes | 2026-10-18 18:40 | 170 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-adv-q1-strategy` | `t31-q1-strategy-20261018` | 6346e731 | 0 | yes | 2026-10-18 18:41 | 170 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-adv-rl` | `adv3-overnight-20261018` | cff0b298 | 0 | yes | 2026-10-18 00:41 | 161 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-advice-after-q1-rl-20261018` | `codex/advice-after-q1-rl-20261018` | 46af8ee8 | 0 | yes | 2026-10-18 19:10 | 170 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-after-q1-skeptic` | `t31-after-q1-skeptic-20261018` | b0474897 | 0 | yes | 2026-10-18 19:09 | 170 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-after-q1-strategy-20261018` | `codex/after-q1-strategy-20261018` | 5721e68b | 0 | yes | 2026-10-18 19:09 | 170 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-audit-eng` | `rules-audit-eng-20261019` | 807d3bf0 | 0 | yes | 2026-10-19 20:22 | 175 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-audit-ops` | `rules-audit-ops-20261019` | ba776b43 | 0 | yes | 2026-10-19 20:21 | 175 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-audit-research` | `rules-audit-research-20261019` | a40cf727 | 0 | yes | 2026-10-19 20:22 | 175 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-batched-teacher` | `t33-batched-teacher-20261019` | eeb46b72 | 0 | yes | 2026-10-19 16:11 | 191 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-colab-cli` | `t33-colab-cli-20261018` | c79ffa35 | 0 | yes | 2026-10-18 18:45 | 176 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-colab-cli-advice` | `t30-colab-cli-advice-20261018` | 55599743 | 0 | yes | 2026-10-18 17:47 | 169 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-exit-engine` | `t34-exit-engine-20261019` | 2482d3bc | 0 | yes | 2026-10-19 20:00 | 184 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-exit-fit` | `t34-exit-fit-20261019` | 00de27c0 | 0 | yes | 2026-10-19 19:43 | 184 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-m2m3-perf` | `t34-m2m3-advice-perf-20261020` | 3022c38f | 0 | yes | 2026-10-20 01:29 | 176 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-m4a` | `m4a` | f518763d | 0 | yes | 2026-10-12 20:51 | 72 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-m4a2` | `(detached)` | 8b5f5af9 | 0 | yes | 2026-10-13 01:20 | 73 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-m4gate` | `(detached)` | 0d9a974a | 0 | yes | 2026-10-13 05:44 | 73 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-m5` | `(detached)` | bb0fd75a | 0 | yes | 2026-10-13 08:52 | 74 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-m6` | `m6` | 91d0e8fa | 0 | yes | 2026-10-14 09:35 | 75 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-openmp` | `t31-openmp-tests-20261018` | cf5df857 | 0 | yes | 2026-10-18 20:39 | 178 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-pactor` | `t25-colab-queue-20261017` | 1030536c | 0 | yes | 2026-10-17 20:53 | 166 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-perf` | `(detached)` | 836646cb | 0 | yes | 2026-10-13 22:07 | 75 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-pleague` | `persona-league-20261017` | 72682c68 | 0 | yes | 2026-10-17 14:17 | 145 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-portfolio-astra` | `portfolio-advice-20261020` | 3cc293d5 | 0 | yes | 2026-10-20 13:35 | 178 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-portfolio-distill` | `portfolio-advice-distill-20261020` | 31e84788 | 0 | yes | 2026-10-20 13:36 | 178 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-portfolio-evalstats` | `portfolio-advice-evalstats-20261020` | 942a374c | 0 | yes | 2026-10-20 13:36 | 178 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-portfolio-gpuops` | `portfolio-advice-gpuops-20261020` | 767b65bc | 0 | yes | 2026-10-20 13:35 | 178 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-portfolio-product` | `portfolio-advice-product-20261020` | f85b9095 | 0 | yes | 2026-10-20 13:35 | 178 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-portfolio-search` | `portfolio-advice-search-20261020` | ba21fda3 | 0 | yes | 2026-10-20 13:36 | 178 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-portfolio-selfplay` | `portfolio-advice-selfplay-20261020` | 9ee907d0 | 0 | yes | 2026-10-20 13:35 | 178 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-portfolio-skeptic` | `portfolio-advice-skeptic-20261020` | f818b497 | 0 | yes | 2026-10-20 13:36 | 178 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-q0-init-fill` | `t31-q0-init-fill-20261018` | 8f7bd8a4 | 0 | yes | 2026-10-18 14:32 | 175 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-q0b-l4-a3` | `t31-q0b-l4-a3-20261018` | 17a927c9 | 0 | yes | 2026-10-18 18:05 | 169 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-q0q1-a3` | `q0q1-a3-20261018` | 105e9f37 | 0 | yes | 2026-10-18 17:13 | 175 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-q1-a4` | `t32-q1-a4-20261018` | 3f231b2e | 0 | yes | 2026-10-18 18:32 | 175 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-q1-a5` | `t31-q1-a5-20261018` | e602b489 | 0 | yes | 2026-10-18 19:04 | 176 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-q1-admission` | `t31-q1-admission-20261018` | ad33f390 | 0 | yes | 2026-10-18 14:57 | 175 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-q1-baseline` | `(detached)` | 8f7bd8a4 | 0 | yes | 2026-10-18 14:32 | 172 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-q1-next-rl-20261018` | `t31-q1-next-rl-20261018` | 912ca8ee | 0 | yes | 2026-10-18 18:40 | 170 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-retention-fixes` | `t34-retention-fixes-20261020` | f297e585 | 0 | yes | 2026-10-20 12:29 | 178 MB | — (false positive) | clean, merged, unreferenced, no process |
| `snake-dqn-retention-fixes2` | `t34-retention-fixes2-20261020` | a1dec26b | 0 | yes | 2026-10-20 13:03 | 182 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-retention-fixes5` | `t34-retention-fixes5-20261020` | cf44b81d | 0 | yes | 2026-10-20 14:31 | 182 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-retention-prereg` | `t34-retention-prereg-20261020` | 00a7e114 | 0 | yes | 2026-10-20 10:04 | 177 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-retention-resize` | `t34-retention-resize-20261020` | 764a6209 | 0 | yes | 2026-10-20 12:40 | 178 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-rules-code` | `rules-fixes-code-20261019` | 10a73528 | 0 | yes | 2026-10-19 21:50 | 183 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-s1-mature` | `s1-mature-20261018` | 3e6d2a84 | 0 | yes | 2026-10-18 20:47 | 179 MB | — (false positive) | clean, merged, unreferenced, no process |
| `snake-dqn-shadow` | `t34-shadow-measure-20261019` | f5b45365 | 0 | yes | 2026-10-19 23:57 | 185 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-speed-perf` | `t34-speed-advice-perf-20261019` | aa4f1124 | 0 | yes | 2026-10-19 19:40 | 172 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-speed-search` | `t34-speed-advice-search-20261019` | 0ea4d274 | 0 | yes | 2026-10-19 19:42 | 172 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-t11` | `t11-20261017` | a0b519ee | 0 | yes | 2026-10-16 22:34 | 104 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-t16` | `t16-20261017` | 58b461da | 0 | yes | 2026-10-17 11:53 | 148 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-t17` | `t20-selfdelete-key-20261017` | 5de8a759 | 0 | yes | 2026-10-17 15:36 | 149 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-t2p` | `t2p-20261015` | 814c22f3 | 0 | yes | 2026-10-15 07:23 | 77 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-t3` | `t3-20261015` | 92b0a5dd | 0 | yes | 2026-10-15 13:27 | 78 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-t3p` | `t3p-20261015` | 904476f4 | 0 | yes | 2026-10-15 21:40 | 83 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-t4a` | `t4a-20261016` | 949b4a18 | 0 | yes | 2026-10-15 19:26 | 79 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-t4b` | `t4b-20261016` | 7fd8e547 | 0 | yes | 2026-10-15 19:39 | 79 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-t4c` | `t4c-20261016` | 96f5e5a6 | 0 | yes | 2026-10-15 23:45 | 92 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-t4d` | `t4d-20261016` | ab385426 | 0 | yes | 2026-10-16 15:36 | 100 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-t4prep` | `t4prep-20261015` | a346a77c | 0 | yes | 2026-10-15 22:27 | 87 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-t5` | `t5-20261016` | 80ab6ed3 | 0 | yes | 2026-10-16 22:16 | 104 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-t6` | `t6-20261016` | f0e9008e | 0 | yes | 2026-10-16 17:10 | 100 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-t7` | `t7-20261016` | 45e3224a | 0 | yes | 2026-10-17 00:01 | 107 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-t9` | `t9-20261017` | d638887c | 0 | yes | 2026-10-16 23:17 | 104 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-v2-fitcache` | `t34-v2-fitcache-20261019` | 289d8808 | 0 | yes | 2026-10-20 00:12 | 179 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn-v2-teacher` | `t34-v2-teacher-20261019` | 97f367d7 | 0 | yes | 2026-10-20 02:12 | 185 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/agent-a43a4c0466e48c24b` | `v8-numba` | fd8c946e | 0 | yes | 2026-10-12 21:07 | 73 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/agent-a6a337321f925bb05` | `m4b` | 8b5f5af9 | 0 | yes | 2026-10-13 01:20 | 73 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/agent-a6d82e865cd27bd49` | `m4b-live` | 0d9a974a | 0 | yes | 2026-10-13 05:44 | 73 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/agent-aad6171926bbe53b4` | `perf-actor` | bb0fd75a | 0 | yes | 2026-10-13 08:52 | 74 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/agent-add75eb6ca748f0d3` | `m4-phase-r` | 188d1f29 | 0 | yes | 2026-10-13 05:28 | 73 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/agent-aeb62bf0764898cb2` | `perf-learner` | 09ef0781 | 0 | yes | 2026-10-13 15:47 | 74 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/keen-haibt-840882` | `claude/sweet-dirac-33e725` | db2ef7ab | 0 | yes | 2026-10-06 07:21 | 60 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wf_1f3ca2f7-9d3-1` | `perf-compile` | 8f625d50 | 0 | yes | 2026-10-13 21:20 | 74 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wf_1f3ca2f7-9d3-2` | `perf-podbench` | 372d42ac | 0 | yes | 2026-10-13 20:40 | 71 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wf_1f3ca2f7-9d3-3` | `perf-ram` | 3a9a332d | 0 | yes | 2026-10-13 21:47 | 73 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wf_1f3ca2f7-9d3-4` | `perf-eval` | cb000793 | 0 | yes | 2026-10-13 21:42 | 74 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wf_1f3ca2f7-9d3-5` | `perf-opp` | 8e177881 | 0 | yes | 2026-10-13 20:46 | 74 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wf_1f3ca2f7-9d3-6` | `perf-multiseed` | 8b124a91 | 0 | yes | 2026-10-13 21:06 | 73 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wf_1f3ca2f7-9d3-7` | `perf-profile` | 37c7bb21 | 0 | yes | 2026-10-13 20:59 | 74 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wf_2cdb93d3-f4b-1` | `perf-integrate` | 836646cb | 0 | yes | 2026-10-13 22:07 | 75 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wf_2ee9a0af-657-1` | `research-ensemble` | f9b942ab | 0 | yes | 2026-10-14 17:01 | 75 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wf_2ee9a0af-657-3` | `research-planner` | 5a315985 | 0 | yes | 2026-10-14 17:15 | 74 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wf_34c750b3-7ff-1` | `census-b8` | acd05dfc | 0 | yes | 2026-10-14 01:58 | 75 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wf_3e03f649-fe3-1` | `m6-reward` | a1246e79 | 0 | yes | 2026-10-14 02:59 | 75 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wf_3e03f649-fe3-2` | `m6-runner` | b5b68140 | 0 | yes | 2026-10-14 02:47 | 74 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wf_8af9a1a9-edb-2` | `m7-governor` | 5700c1b7 | 0 | yes | 2026-10-14 17:09 | 74 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wf_d05c9584-907-1` | `integrate-main` | 8297e62d | 0 | yes | 2026-10-14 16:37 | 78 MB | — | clean, merged, unreferenced, no process |
| `snake-dqn/.claude/worktrees/wizardly-nightingale-15f4f1` | `claude/wizardly-nightingale-15f4f1` | f90bea8f | 0 | yes | 2026-10-18 21:01 | 178 MB | — | clean, merged, unreferenced, no process |

## Branches

- Local branches merged into origin/main: 211. Remote-tracking branches merged: 87.
- Local branches **not** merged into origin/main (58): `census-frp3`, `claude/unruffled-albattani-f49854`, `codex/observation-probe-cli-20260912`, `codex/observation-probe-core-20260912`, `codex/observation-probe-docs-20260912`, `codex/observation-probe-stats-20260912`, `codex/pqn-ambient-objective-20260913`, `codex/pqn-autoreset-b1`, `codex/pqn-autoreset-b2`, `codex/pqn-autoreset-b3c`, `codex/pqn-autoreset-b3s`, `codex/pqn-autoreset-b3t`, `codex/pqn-decision-phase-20260912`, `codex/pqn-enemy-experience`, `codex/pqn-solo-completion-20260913`, `codex/pqn-solo-profile-20260913`, `codex/repair-serving-episode`, `codex/repair-user-docs`, `codex/rl-env-watch-action`, `codex/rl-repair-a0`, `codex/rl-repair-a1`, `codex/rl-repair-a2`, `codex/rl-repair-c0`, `codex/rl-repair-e0`, `codex/rl-repair-e1`, `codex/rl-repair-e2`, `codex/rl-repair-env`, `codex/rl-repair-env-watch`, `codex/rl-repair-h0`, `codex/rl-repair-n0`, `codex/rl-repair-obs`, `codex/rl-repair-off`, `codex/rl-repair-p1`, `codex/rl-repair-p2`, `codex/rl-repair-s0`, `codex/rl-repair-s1`, `diag-divergence`, `frp-v2`, `frp-v3`, `frp-v3-phaser-rp`, `frp-v4`, `frp-v5h`, `frp-v5h-phaser-rp`, `frp-v5s2-phaser-rp`, `growth-ceiling`, `lh1-screen`, `next-lever`, `oracle-rescue`, `rp-bank-long`, `rp-kill-robust`, `sight-widen`, `sls-batch-core`, `t31-colab-safety-20261018`, `task-aligned-subtest-report-boundary`, `train-smarter`, `train-smarter-e0`, `veto-v9`, `worktree-agent-a20ad0e8cde207090`.
- Remote branches not merged (4): `origin/claude/upbeat-allen-uvxqvd`, `origin/colab-slower-20261017`, `origin/t31-colab-safety-20261018`, `origin/t34-retention-bare-unpaced-20261020`.
- Of the unmerged local branches, only `t31-colab-safety-20261018` (NEEDS-OWNER) and `worktree-agent-a20ad0e8cde207090` (ACTIVE-AGENT) are checked out in a worktree. The others are older branch-only refs. This inventory proposes no branch deletions.

## SAFE-TO-REMOVE commands (written, NOT run)

Before running these, re-check `ps`/`tmux ls` and `git -C <path> status --porcelain`. A worktree can turn active between this snapshot and removal. Plain `git worktree remove` (without `--force`) refuses a dirty worktree, which is a built-in safety net.

```sh
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /private/tmp/s1-baseline-verify-35214489
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /private/tmp/snake-dqn-baseline-ad33.S3prMJ
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-adv-gpu
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-adv-league
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-adv-q1-skeptic
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-adv-q1-strategy
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-adv-rl
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-advice-after-q1-rl-20261018
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-after-q1-skeptic
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-after-q1-strategy-20261018
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-audit-eng
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-audit-ops
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-audit-research
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-batched-teacher
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-colab-cli
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-colab-cli-advice
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-exit-engine
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-exit-fit
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-m2m3-perf
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-m4a
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-m4a2
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-m4gate
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-m5
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-m6
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-openmp
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-pactor
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-perf
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-pleague
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-portfolio-astra
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-portfolio-distill
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-portfolio-evalstats
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-portfolio-gpuops
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-portfolio-product
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-portfolio-search
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-portfolio-selfplay
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-portfolio-skeptic
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-q0-init-fill
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-q0b-l4-a3
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-q0q1-a3
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-q1-a4
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-q1-a5
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-q1-admission
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-q1-baseline
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-q1-next-rl-20261018
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-retention-fixes
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-retention-fixes2
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-retention-fixes5
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-retention-prereg
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-retention-resize
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-rules-code
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-s1-mature
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-shadow
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-speed-perf
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-speed-search
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-t11
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-t16
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-t17
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-t2p
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-t3
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-t3p
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-t4a
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-t4b
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-t4c
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-t4d
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-t4prep
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-t5
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-t6
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-t7
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-t9
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-v2-fitcache
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn-v2-teacher
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/agent-a43a4c0466e48c24b
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/agent-a6a337321f925bb05
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/agent-a6d82e865cd27bd49
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/agent-aad6171926bbe53b4
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/agent-add75eb6ca748f0d3
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/agent-aeb62bf0764898cb2
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/keen-haibt-840882
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wf_1f3ca2f7-9d3-1
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wf_1f3ca2f7-9d3-2
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wf_1f3ca2f7-9d3-3
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wf_1f3ca2f7-9d3-4
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wf_1f3ca2f7-9d3-5
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wf_1f3ca2f7-9d3-6
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wf_1f3ca2f7-9d3-7
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wf_2cdb93d3-f4b-1
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wf_2ee9a0af-657-1
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wf_2ee9a0af-657-3
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wf_34c750b3-7ff-1
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wf_3e03f649-fe3-1
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wf_3e03f649-fe3-2
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wf_8af9a1a9-edb-2
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wf_d05c9584-907-1
git -C /Users/josenunez/Projects/ml/snake-dqn worktree remove /Users/josenunez/Projects/ml/snake-dqn/.claude/worktrees/wizardly-nightingale-15f4f1
```
