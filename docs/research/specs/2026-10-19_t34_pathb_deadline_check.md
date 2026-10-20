# T34 Path B deadline-feasibility check — Sol spec

Binding: [SUCCESS_CONTRACT.md](../final_run_20261019/SUCCESS_CONTRACT.md) (owner-ratified 2026-10-19)
§4 "Path B rule" and §5 "Sequencing change"; background [FINAL_PLAN.md](../final_run_20261019/FINAL_PLAN.md)
and the three advice notes in `docs/research/advice/2026-10-19_t34_success_*.md`.

Goal: build the REAL asynchronous hero-only lookahead controller over bare B8 s3 for mixed Watch6, and
the Mac harness that measures (a) efficacy under real deadlines and (b) the serving timing contract.
Development evidence only (not gate data). Package `research/t34_pathb/`; serving hook behind a
default-OFF flag in `web/backend/` (B8 s3 bare stays the default; nothing changes for users unless the
flag is set).

## Deliverables

1. **Frozen action rule (write it first, in `research/t34_pathb/PROTOCOL.md`, before any efficacy
   number exists):** trigger schedule (e.g. L>=600 / every k decisions / low Q-margin), verify
   horizon, and the complete replacement rule when the bare action fails verification (expand other
   legal actions in lockstep, pick highest-B8-Q survivor, ties → bare action, none survive →
   longest-surviving). Opponent/food handling must use only deployment-available state (current live
   sim snapshot; opponents rolled with their own served policies; food RNG cloned from the live
   state only if the live server legitimately has it — document exactly). No hindsight rewinds.
   Reuse the deterministic verify/expand kernel from T33/T34 (`research/t33_batched_teacher`,
   `research/t3_rollout_labels_20261015`, `research/serving_time_search_20261003`) — do not rewrite
   the sim.
2. **Async controller:** one extra worker thread/process (1 CPU thread), ≤1 outstanding request,
   bounded memory; request bound to (session, frame, hero life, checkpoint); result applied only if it
   arrives before that frame's action deadline and still matches; else bare action. Never blocks the
   tick. Log per decision: requested, applied, stale-discarded, deadline-miss, fallback reason,
   proposed vs executed action.
3. **Efficacy harness (Mac):** paced simulation at the real Watch6 cadence (24 fps ceiling, also the
   12 fps default) where the controller only gets wall-clock-real compute; paired vs bare B8 s3 on a
   fresh development bank (terminal-hero H5000, scripted/frozen/mixed, contract §2 metrics) AND
   continuous-respawn mixed Watch6 with FRP3+v8 opponents (contract §3 metrics). Also a "synchronous
   unlimited" arm on the same worlds to show how much gain deadlines cost. Make bank sizes and worker
   counts CLI flags; ship a tiny smoke config; I size and run the real one.
4. **Timing harness:** reuse T15's paced latency/cadence probes (`research/t15_mixed_roster_20261017`)
   with assistance ON vs OFF: update p95/p99 (contract: mixed-Watch6 p95 ≤ 11.688 ms), zero overruns at
   24 and 12 fps, delivered cadence, long-body and CPU-contention tails.
5. **UI identity:** when the flag is on, Watch shows the hero as "assisted" and the inspector shows
   interventions/fallbacks; rollback = flag off → bare B8 s3. Tests for both.

## Rules
- Mac only; no Colab/RunPod/paid services; no tmux; do not touch Phase 0 (tmux phase0-selection) or S1
  or the running web server. Keep tests/probes ≤2 CPU threads while Phase 0 runs.
- Never change the default served behaviour; flag defaults OFF.
- Commit often, push the branch, merge to main and push at the end (main moves; two T34 Sol jobs are
  also merging — resolve carefully). Run the full non-slow test suite before merging.
- Finish with `research/t34_pathb/README.md`: exact commands for me (smoke, efficacy run with sizing
  estimate in Mac core-hours, timing run), and an acceptance table vs contract §2–§4.
