# P follow-up accepted preparation — 2026-10-29

**INSUFFICIENT to fill the successor registration. Three engineering probes
prepared and independently accepted; none dispatched.** See
`FOLLOWUP_20261029.md` for the verified prior measurements and
`FOLLOWUP_REVISION_20261029.md` for the current probe scope and pricing. The
original DRAFT and the closed 09e scientific bank remain untouched.

Accepted executable source pin: `d7e5a1a51ea99ad3b8bffc3f10b9805dcf8bf7de`.
Publication branch: `astra-argon-falsifier-p-20261028`, never main.
This verification note changes no selected runtime source.

## Independent cross-review

Argon (`agy --model gemini-4-argon-high --mode plan --continue --print-timeout
900s --print ...`, outer 960s bound) reviewed the accepted source in its own
clean detached worktree `/Users/josenunez/Projects/ml/snake-dqn-p-successor4-review.noindex`.
**ACCEPT; no unresolved blockers; EXIT=0.** Static supplied-source review only:
no reviewer commands/tools, edits or scientific outcome access. It accepts
engineering preparation, not successor ratification or target execution proof.

Artifact base: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-p-successor4`.

- Final answer `argon-r3-answer.txt`: SHA256
  `b824ffa4bcb8f61b7b9e0f9364f428f9ad7d30d2b531ace69f51d476d0a87da4`.
- `VERIFICATION.json`, exact commands, checks, hashes and packet bindings:
  `4fcec615b1ecc5f123d6568ee28056a8deb6cca823ad91820269e0d5d589766a`.
- `FINAL_SEAL.json`, all three packet seals/commands/dry-runs and key evidence:
  `e02f88b536da796315ce8d6749969021d7862d4d1c4f2deea1ac78d059249a80`.
- Engineering measurements `analysis.json`:
  `c7eee791c187bc7361ded9d17667630caca91807c2ec3d067c0db7dea698f407`.
- The initial Argon rejection and `packet-v3` remain intact and superseded.
  `packet-v3.SUPERSEDED.md` says not to dispatch it. One earlier CLI invocation
  refused `--print-timeout 900` (duration required `900s`); no review ran in that
  invocation. The accepted review followed the sizing/publication/test fixes.

## Verified checks

All Python checks use `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`;
numerical commands use CPU plus all seven single-thread environment caps.

- `python -m pytest -q research/watch_lead_p_successor/test_profile.py
  research/watch_lead_p_successor/test_followup.py
  tools/sandbox/tests/test_source_delivery.py tests/test_colab_cli_canary.py`:
  **137 passed in 5.08 s** (`tests-v5.log`).
- Black (100 columns), isort (black profile) and flake8 on
  `research/watch_lead_p_successor`: **PASS**. Shell syntax checks and
  `bash tools/ops/tests/test_dispatch.sh`: **PASS**; `git diff --check`: **PASS**.
- Real local `followup --phase contention|bare|tail --smoke`: **all three PASS**,
  serialized under one CPU slot, AC/lid guard and caffeinate. Two ticks only;
  `smoke-v5.sh` and `smoke-v5.log` retain argv and output. This proves plumbing,
  not target speed or natural long-body support.
- `real-detail-check.py` under the same Mac guards: **PASS**, original excluded
  cached mature root hash verified first. Current restore/digest wrapper executes
  real H300 search twice for P0 and twice for P48; bare calls zero searches.
  Only timing/workload/opaque hashes are retained in `real-detail-check.json`.
- Original exported prior profiles: **93 A100 + 83 L4 verified JSON triples**,
  receipt size/hash agreement, no orphan components, individual/sweep agreement,
  recomputed overlap and opaque identity across completed sweeps.
- All three prepared `canary-command.sh` lines with `--dry-run` appended:
  **PASS_DRY_RUN; allocations=0; live_accounting=NOT_CHECKED**. Output and state
  directories remained absent. Full argv/stdout/stderr are in each `dry-run.json`.
- Current archive: **1,430,506 B compressed / 6,739,829 B unpacked; 320 files**.
  All three manifests use the same accepted source pin and closure. Per-packet
  seals, every selected source file and every exact checkpoint input were hashed.
  Native `research/colab_cli` executable bytes remain unchanged.

## Prepared operator commands — not executed

Each command contains its own thread/device environment and routes through the
existing ops selector, narrow host adapter and unchanged qualified supervisor.
Execute each at most once, after normal live admission/rate/slot checks; do not
assume three free account assignments or stop another job. These are independent
engineering probes, not an automatic chain or retry. The native account cap is
three including all other work.

```bash
bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-p-successor4/packet-v4/contention/canary-command.sh
bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-p-successor4/packet-v4/bare/canary-command.sh
bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-p-successor4/packet-v4/tail/canary-command.sh
```

Each probe: base/effective work **600/1200 s**, whole allocation **15/30 min**,
rate ceiling **1.54 u/h**, cap **0.39/0.78 u**, reserve **0.078/0.156 u**, priced
whole-timeout exposure **0.77 u**. Three original probes total **2.31 u exposure**;
cap guards total 2.34 u, reserves 0.468 u. Setup, bootstrap, exports, idle and
stop time are included. First cap wins. No dispatch, allocation, automatic retry,
unit purchase, serving change or main-study manifest has occurred.

## Limits and preserved state

The successor formula terms for long-body/full-search uplift, residual contention,
complete inclusive arm costs and shard scheduling remain UNQUALIFIED. No sample
size, threshold, bank outcome or efficacy inference was derived from closed P.
These small engineering probes still cannot establish full searched-session
coverage, worst late-branch costs or sustained high-worker tail throughput.
Missing/unsupported roots remain missing, never replaced; the first seed is fixed
prospectively. Future scientific shards must implement and offline-test the
DRAFT's immutable whole three-arm session ledger and complete-N one-look gate.

Preserved: original DRAFT, old P source/science and bank, old immutable manifests/
receipts and superseded packets, released checkpoint bytes, serving pins,
pre-existing untracked `saved_snakes`, all existing tmux sessions and other jobs.
No native Colab transport edit or main push. The added review worktree is clean
and retained for evidence; no existing worktree was reset or rebased.
