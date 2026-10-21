# Reorg proposals 25: repository hygiene (large, binary, duplicate, local-path, secret, .gitignore)

Written 2026-10-20 by an organizing agent. This was a read-only audit. Nothing below was moved, deleted or edited.
Measurements come from `git ls-files` at `6af08b0f` (3,752 tracked files, about 181 MB on disk by `du -k`).

**Binding rule used below.** Under `research/**` and `tests/fixtures/**`, studies and tests hash files by path. For example,
`research/t34_exit_b8/profile_shadow_batching.py` pins `engine_serial.py` by sha256, and
`research/t15_mixed_roster_20261017/results/publication.json` lists sha256 values for its outputs. Seed banks and audits
(`SEED_AUDIT*.json`, `bank.py`, `REGISTRATION_ROUND0.json`) also reference `WORLD_BANK.json` and parity files by path.
So **every file under `research/` is treated as bound by its study**. Moving or deleting one could break a receipt, so it
needs that study's owner to sign off. "Bound: no" means no hash or path reference turned up in this pass. It does not prove
the file is free.

## 1. Findings

### 1.1 Large tracked files (top of `du -k`)
| KB | Path | Bound? |
|---:|---|---|
| 31,304 | `research/t15_mixed_roster_20261017/results/evidence.tar.gz` | yes (t15 publication/intent sha256) |
| 10,792 | `research/q0q1_20261017/results/q0_mass_audit_receipt.json` | yes (Q0 accounting, `Q0_MASS_AUDIT.md`) |
| 5,032 | `research/t3_screen_20261015/t4c_results/analysis.json` | yes (t4c drift report) |
| 4,440 | `research/t23_r0_screen_20261017/WORLD_BANK.json` | yes (t23 bank/SEED_AUDIT/VALIDATION) |
| 3,792 | `research/t23_r0_screen_20261017/parity/mac.json.gz` | yes |
| 2,808 | `docs/research/t4a_posthoc_2026-10-16_evidence/posthoc-summary.json` | unclear (docs evidence bundle) |
| 2,432..1,040 | `research/{t4a,q0q1,redesign_t7_phase_r,t3,survival_band_v2,s1_mature,t12,t13,t13b,t12b,t6}/...` | yes |
| 860..384 | ~25 PNG learning-curve figures in `docs/experiments/*_2026-09-2x/` | no |

By area: `docs/experiments` is 46.5 MB (250 of the 252 tracked PNGs). `research/t15_mixed_roster_20261017` is 32 MB,
`research/q0q1_20261017` 14 MB and `research/t3_screen_20261015` 10 MB.

### 1.2 Tracked generated, binary or cache files
- **Caches and junk:** none. No `__pycache__`, `.pyc`, `.DS_Store`, `node_modules`, `.egg-info`, `.swp` or `.bak` files are tracked.
- **Archives:** 4 `evidence.tar.gz` (t12, t12b, t15) plus `postrun_owned_signals.patch.gz`, `episodes.json.gz` (x2) and `parity/mac.json.gz`. All live in `research/`, so all are bound.
- **Logs (29 `*.log`):** 8 in `docs/experiments/{pqn_sampler,state_system_audit}_2026-09-06/` and 21 in `research/*/results/`. Several are 0-byte (see 1.4).
- **`*.exit` and `*.complete` sentinel files:** 10 identical `*.json.complete` files in `research/t33_batched_teacher/`, plus `*.exit` and `*-exit.txt` files in t7, t11 and t6. These are run-control markers committed as evidence.
- **Notebooks:** 8 `notebooks/*.ipynb` files are tracked even though `.gitignore` has `*.ipynb`, so they were force-added.
- **Binary fixtures:** `tests/fixtures/raster31v2_golden.npz` and `tests/fixtures/t22/actions_seed741.npz` are intentional goldens and bound by tests.
- **Images in research:** `research/t12_b8_serving_20261017/results/{features,screenshot}.jpg` are bound (t12 evidence).

### 1.3 Secrets and local paths
- **Secrets:** no matches for common token shapes (`sk-`, `rpa_`, `AKIA`, `ghp_`, `AIza`, `ya29.`, PEM private keys). No tracked `.env`, `*.pem` or `*.key` files, and no filenames containing secret, token or credential. `.env` and `.env.local` are ignored. **Nothing to act on.**
- **Absolute `/Users/josenunez` paths:** found in 1,109 tracked files (638 in `research`, 425 in `docs`, 42 in `tests`, 2 in `web`, 2 in `verification`). Most are provenance records, which is expected. The ones that affect **runtime or portability**:
  - `web/backend/served_checkpoint.py` (lines ~98 and ~104) and `web/backend/served_checkpoint_pins.json` hard-code checkpoint roots under `snake-dqn-artifacts/`. These bind the **live served release** (B8 s3 and its frp3-s12 rollback).
  - `verification/s18_kl_contrast_outer_v1.py` and `verification/s18_kl_contrast_audit_outer_v1.py` hard-code `REPO=~/.codex/worktrees/m3d-autonomous/...`, `BASE` and `PYTHON`. They only run on this Mac, inside a Codex worktree that may no longer exist.
  - About 37 `tests/test_*.py` files, e.g. `tests/test_phase0_teacher.py:86` (an `s1-mature-20261018/inputs/B8_s3_final.pth` path), `tests/test_t34_claim_a.py` (`MAIN_SAVED = .../snake-dqn/saved_snakes`) and `tests/test_gate_calibration.py` (`ongoing-research-20260913/`). Unclear without reading each test whether it skips when the path is missing or fails.

### 1.4 Duplicate blobs (same git sha across paths): 17 groups
- **Empty file (e69de29), 33 paths.** 27 empty `research/*/__init__.py` (expected), plus empty `results/report.log`, `all_b8.stdout.log`, `default.stdout.log`, `flake8.txt` (x2) and `isort.txt`. The empty logs are "clean output" receipts.
- **`notebooks/colab_queue_worker{,_a,_b,_c}.ipynb`: 4 identical copies.**
- **`docs/experiments/state_system_audit_2026-09-06/`:** `pqn/objective_contract_probe.py` matches `verification/objective_contract_probe_recheck.py`, and `prior_audit/audit_reproductions.json` matches `audit_reproductions.log`.
- **~7 PNG pairs in `docs/experiments/` (2026-09-20..22):** one figure copied into a follow-up experiment folder. For example, `solo_food_pqn_h384_confirmation/historical-co-training-curves.png` matches `solo_food_pqn_loss_shape/training-curves.png` (828 KB), and one `learning-curves.png` exists in 4 controlled_encounter folders.
- **Bound duplicates (leave alone):** `research/t34_exit_b8/engine.py` matches `tests/fixtures/t34_shadow_batching/engine_serial.py` (the copy is a hash-pinned serial reference). `tests/fixtures/apex_veto_strict/records/{A,B}-frozen-2920083565.json` match `tests/fixtures/simd_parity_h5000/`. `research/frp3_strict_20261005/frozen_plan.json` matches `research/survival_band_v2_20261006/outputs/design_plan_pooled.json`. 4 identical `*.exit`/`*-exit.txt` files and 10 identical `*.json.complete` files.

### 1.5 `.gitignore` gaps and untracked clutter in the main checkout
`git status --ignored` could not be run against the main checkout from this isolated worktree, because the sandbox refused it.
Instead, the main checkout's filesystem was compared with the tracked list, skipping `venv`, `runs`, `logs`, `saved_snakes`,
`.claude`, caches and `node_modules`:
- Only 9 files are untracked: `.cursorignore`, `scores.db`, `snake_memories.db` and `src/snake_dqn.egg-info/*` (6 files). **All of them are already ignored.** No `.DS_Store` files were found.
- Ignored bulk on disk: `venv` 1.2 GB, `snake_memories.db` 176 MB, `runs/` 124 MB, `saved_snakes/` 60 MB and `logs/` 1 MB.
- `.claude/worktrees/` (51 worktrees) is ignored only through `.git/info/exclude` (`**/.claude/worktrees/`), not `.gitignore`. A fresh clone would show any worktrees created there as untracked.
- `.gitignore` has `*.ipynb`, which contradicts the 8 tracked notebooks. New notebooks are silently ignored unless someone uses `git add -f`.
- Missing patterns worth adding: `*.json.complete`, `*.exit`, `*.pth`/`*.pt` (checkpoints are only ignored via `saved_snakes/`, `runs/` and `checkpoints/`; a stray `*.pth` elsewhere would be picked up) and `*.npz` outside `tests/fixtures/`.

## 2. Proposed actions (none executed)

| # | Action | Risk | Bound by a ratified study? |
|---|---|---|---|
| H1 | **Add to `.gitignore`:** `*.pth`, `*.pt`, `*.json.complete`, `*.exit`, `/.claude/worktrees/`, and `!/notebooks/*.ipynb` after `*.ipynb`. Does not untrack anything; only stops future accidents. | Low. Check that no workflow writes new tracked `*.exit` or `.complete` receipts; if one does, scope the rule to the repo root. | No (`.gitignore` is owner-only under org30 RULES) |
| H2 | **Dedupe `notebooks/colab_queue_worker_{a,b,c}.ipynb`** to one notebook plus a lane parameter, or note in `notebooks/README` that they are deliberate per-lane copies. | Medium. The Colab skills and the legacy Drive queue (Q1) open them by name. Wait until Q1 is retired. | Q1 (legacy queue) operationally; not hash-bound as far as found |
| H3 | **Make the served checkpoint roots configurable** (env var or a repo-relative pins root) instead of absolute `/Users/josenunez/...` paths in `web/backend/served_checkpoint.py` and `served_checkpoint_pins.json`. | Medium. This touches the live Watch release, so it needs a release-style check and owner OK. | Yes: B8 s3 release (rollback frp3-s12) |
| H4 | **Retire or parameterize `verification/s18_kl_contrast_*_outer_v1.py`**, which hard-code a Codex worktree, `BASE` and `PYTHON`. Mark them HISTORICAL in a header, or read the paths from argv. | Low. These are M3-D diagnostics (closed 2026-10-12). | Possibly, as M3-D S18 receipts. Do not edit bytes; add a sibling note instead |
| H5 | **Tests with absolute artifact paths (~37):** audit whether each uses `pytest.skip` when the path is missing. Add a shared `SNAKE_ARTIFACTS_ROOT` fixture in a new conftest helper. | Medium. Tests guard ratified studies; changing collection could hide failures. | Yes for strict-gate tests (apex_veto v5/v7/v8, t34_*) |
| H6 | **Docs PNG duplicates (~7 pairs, ~4 MB):** replace the follow-up copy with a relative link to the original figure. | Low. Breaks nothing unless an evidence manifest hashes the copy; grep each folder first. | No (docs/experiments, Sept bundles) |
| H7 | **Move large `docs/experiments` PNG sets (46 MB) and `docs/research/t4a_posthoc_2026-10-16_evidence/posthoc-summary.json` (2.8 MB) to Git LFS or the artifacts tree**, keeping the markdown in the repo. | Medium. Needs LFS setup on every clone and Colab image, plus link rewrites. | No / unclear |
| H8 | **Large research evidence (`t15 evidence.tar.gz` 31 MB, `q0_mass_audit_receipt.json` 11 MB, `WORLD_BANK.json` files):** **keep as is.** Note in `research/INDEX.md` that these are sha-pinned receipts. Only consider LFS migration (same paths, same bytes) if repo size becomes a problem. | High if moved or rewritten: breaks sha receipts and seed audits. | **Yes** (t15, Q0/Q1, t23, s1, phase0) |
| H9 | **Empty `*.log` and `flake8.txt` files and identical `*.complete`/`*.exit` sentinels in `research/`:** keep. They are "clean run" receipts. | High if deleted (receipt completeness). | Yes |
| H10 | **`docs/experiments/state_system_audit_2026-09-06/` internal duplicates** (`objective_contract_probe.py` matches `..._recheck.py`; `.json` matches `.log`): add a line to that bundle's README saying the recheck reproduced byte-identically. Proposal 01 covers this bundle, so coordinate with it. | Low. | No |
| H11 | **Secrets:** no action. Optionally add `detect-secrets` or a `gitleaks` pre-commit hook. | Low (needs a `.pre-commit-config.yaml` change, owner-only). | No |

**Top proposal:** H1. It is a `.gitignore`-only change: un-ignore `notebooks/*.ipynb`, ignore `*.pth`/`*.pt` repo-wide,
and move the `.claude/worktrees/` ignore into `.gitignore`. It prevents the most likely future accidents (a stray
checkpoint commit, a silently ignored new notebook) without touching any bound file.
