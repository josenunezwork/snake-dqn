# Speed vs constraints — JUDGMENT (2026-10-20 ~22:30 PT)

Single independent Fable judge. I wrote none of the five Astra answers (`astra_{perf,gpu,validity,search,pm}.md`).
I checked their load-bearing claims against main at `0c48a770` and the saved receipts (§0). This is a ruling on
what to build and what to leave alone; it registers nothing and amends no running study. Colab use is pre-approved;
releases still need the owner.

## 1. Answer to the owner

1. **No — the restrictions are not why the next verdict is slow.** The calendar is a serial chain of Mac-only work
   that runs at wall-clock speed by definition: Phase 0 (6 Mac workers since Oct 19, ends Wed ~10:00–13:00) →
   Round 0 Mac replay (1–2 h) → Path B timing gate → pre-check (~6–7 h) → paced study (~18–19 h). A paced arm at
   24 fps sleeps ~75% of every tick; no engine speed-up shortens it.
2. **The engine is slow** (single-threaded Python; one L4 vCPU ≈ 1/6 of one Mac core on it), but that costs Colab
   units and label-round hours, not the date of the Path B verdict. The 64%-hashing profile is pre-fix; the fix
   (1.71×) is in. About **1.4–1.7× more** is available for ~4 engineer-days; a GPU port is not (sim is 6–12% of time).
3. **Do not stop.** Round 0, Phase 0, Path B operator-02, Claim A and 0-D continue unchanged; the speed program runs in
   parallel, off the Mac's quiet windows, with a 10-unit / 4-engineer-day cap and 48 h / 72 h kill checks (§5).
4. **Review loops are not the cost either:** six Path B FIX-FIRST rounds took ~2 h of wall clock (12:45→14:50) and
   each found a real defect. I shrink them to one executable contract + one reviewer + one judge, not to zero.
5. **The one restriction that idles the most hardware** is "one paced session at a time on a quiet Mac" (16 of 18
   cores idle for ~25 h). Keep it for the ratified run; design 4-session pacing now for the *next* paced Mac run.
6. Tonight: baseline harness (Opus), digest v4 (Sol), native v8 for branch rows (Opus), 4-session paced design (Opus).

## 0. Verified facts that decide the rulings

| # | Claim | Verified at | Status |
|---|---|---|---|
| F1 | Pre-fix Mac row-step 5.84 ms: `full_digest` 3.73 (64%), teacher.evaluate 0.93 incl. raster (forward 0.46 ≈ 8%), v8 hooks 0.76 (13%), sim 0.27 (5%) | `docs/research/advice/2026-10-20_t34_a5_speed_fable.md:8-26` | historical |
| F2 | Fix shipped: whole row-step 6.1605 → 3.5945 ms (1.7139×), exports byte-identical, 15,000-frame live/Grid/shadow identity 0 diffs; W1 fanout, CPU inference, lockstep bypass all in | `research/t34_exit_b8/ENGINE_FANOUT.md:12-60`; CAMPAIGN_LOG.md:57 | done — not to be re-proposed |
| F3 | Post-fix Mac diagnostic (6.02 s): digest 2.140 (35.5%), v8 1.338 (22.2%), inference 0.793 (13.2%), raster 0.573 (9.5%), sim 0.343 (5.7%), other ~13.9%; 208 active row-steps/s single process | `snake-dqn-artifacts/t34-engine-fanout-20261020/cpu-profile-smoke-02/profile-cpu.json` `/alone/exclusive_stage_seconds` | excluded diagnostic; prioritisation only |
| F4 | Post-fix **x86 CPU-only Colab** diagnostic (60 s): digest 35.4%, v8 19.9%, inference 13.3%, sim **12.3%**, raster 7.0%; 28 row-steps/s | `.../manifests-final-3ebf3991/cpu-profile-output/exports/profile-cpu.json` | this is the file astra_validity cited; different host, same conclusion |
| F5 | Production A5 on L4: 57.2 / 54.4 completed worlds/lane-h, **377 / 432 active row-steps/s with 12 procs** → 31–36 per vCPU | `.../a5-l4-s{0,1}-output/exports/benchmark.json` (`worlds_per_hour`, `active_row_steps_per_second`, `fanout/nproc`); CAMPAIGN_LOG.md:59 | verified; **one L4 vCPU ≈ 1/6 Mac core** (F3) |
| F6 | `full_digest` is control logic: `catch_up` stores a digest per predicted frame, `confirm` invalidates on mismatch; teacher clears pending labels/actions on invalidation; audit hashes already every 250 decisions + terminal | `research/t34_shadow/shadow.py:427,597,629`; `research/t34_exit_b8/teacher_v2.py:838-880,875-876` | sampling it = changing the controller |
| F7 | Path B adapter digests every transition and clones the world whenever the worker is idle | `research/t34_pathb/adapter_v2.py:153-160` | in the frozen controller's hot path |
| F8 | v7/v8 opponents run as live Python hook objects per row; `v8_nb` is decision-identical, nogil-capable, omits only per-episode counters and the `reference_lambda` diagnostic | `src/simd_env/vector61_policy.py:45-57,929-955`; `src/simd_env/v8_nb.py:1-44` | native swap is a behaviour-preserving engineering change |
| F9 | Sim keeps ordered food lists, per-env CPython RNG rejection loops and scalar collision resolution; parity CI-pinned to the live game | `src/simd_env/grid_sim.py:28-47` | GPU port must carry these; sim share small anyway |
| F10 | T3-dense: 54/64 vs 38/64, +92 pooled s0+s3 (s3 alone 26/32 vs 17/32, +123); 1.32 s/batch vs 208 ms → 6.3× | `docs/research/t3_ceiling_test_2026-10-15.md:117-129`; nogo plan E9 | search needs Path B's async design, not 1.6× |
| F11 | Serving: Watch6 update p95 8.398 ms (all-B8, T12) / 7.792 ms (mixed, T15); contract ceiling 11.688 ms; T12 does not certify real-time cadence | `t12_b8_serving_qualification_2026-10-17.md:71,184,285-291`; `t15_mixed_roster_qualification_2026-10-17.md:103`; `SUCCESS_CONTRACT.md:42-54` | headroom is for the tick, not for search |
| F12 | Path B operator-02 at `81c54731`: bare arm unpaced, paced arm 24 fps, chain 27–28 h → 18–19 h after admission (forecast); operator-01 VOID | `docs/research/specs/2026-10-20_t34_retention_amendment_bare_unpaced.md:157-159`; CAMPAIGN_LOG.md:81; STATUS.md:22 | "unchanged" = this identity |
| F13 | Mac order: Phase 0 end → Round 0 replay → timing gate → pre-check → study; Phase 0 = 6 one-thread Mac workers, 88.85 h forecast | nogo plan `:61-65`; `research/phase0_search_20261019/PREREGISTRATION.md:3,110`; HANDOFF:16 | the critical path |
| F14 | Six retention FIX-FIRST rounds 12:45 → ratify 14:50 (~2 h); 12/13 invalid Sep studies failed on tooling, not science | CAMPAIGN_LOG.md:43-56; `governance_tiers_2026-09-26.md:14` | review is agent-hours, not calendar |
| F15 | Mac↔x86 float ties flip actions ~1/50 H10000 episodes; cloud strict needs a new amendment with a tie-explaining identity check | `governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md:21-44` | Mac-only gate is a practical rule, not a statistical theorem |
| F16 | Claim A has two design blockers (free-Q flips 0.4% of decisive rows; ~1 step/epoch); two-deviation pooled KILL; outcome head KILL | CAMPAIGN_LOG.md:68,76; colab JUDGMENT.md:24-30 | faster labels fix none of these |
| F17 | Unit of inference = paired world for a fixed controller; comparator B8 s3 | `P7_STATISTICS.md:10-16` | five seeds already not required |
| F18 | Live now: Round 0 chunk 001 running (autochain to 002), Claim A L4 qualification running, Phase 0 at frame ~3–5k on frozen worlds, 0-D prep worktree exists, Claim A loss amendment not yet drafted | tmux `t34-round0-c001`, `t34-round0-autochain`, `t34-claima-l4q`, `phase0-selection`; `snake-dqn-artifacts/t34-round0-study-01/`; no spec file yet | §4 |

**Disagreements and how I resolved them.**
- *Simulator share 5.7% vs 12.3%.* Both numbers are real: F3 is the Mac M5 Pro, F4 the x86 CPU-only runtime
  astra_validity cited. Either way a free simulator caps at 1.06–1.14×. Ruling: sim rewrite and GPU port deferred.
- *Program size.* astra_pm wants a hard 2-day cap; perf/gpu scope 1–2 + 2–4 engineer-days. Ruling: **4 engineer-days
  total across two parallel owners**, with kill checks at 48 h (digest) and 72 h (v8). Two days was never enough
  for both lines; four days is still off the critical path.
- *Expected gain.* 1.3–1.6× (perf, gpu), 1.3–1.8× (search), 1.2–1.4× (pm). Ruling: plan at **1.4–1.6×**; adopt only
  what the matched harness measures (§5 D2/D3).
- *What none of the five said.* F5/F3 give the per-core ratio: 12 L4 vCPUs ≈ 2 Mac cores, so the Mac is ~9 L4 lanes
  and Colab caps us at 3 concurrent assignments. The Mac is therefore both our fastest CPU and the only legal host
  for paced work — which is why "quiet Mac, one paced session" is the costliest rule we have (§2, §3 item S4).
- *Claim A priority.* All five put the loss/dose repair ahead of bulk labels. Confirmed (§4).

## 2. Restrictions: KEEP / RELAX / DROP

| Restriction | Ruling | Reason (one line) |
|---|---|---|
| Bit-exact parity with the Mac reference sim | **KEEP** for the exact-shadow engine and action-tape replay; **RELAX** representation (any arrays/kernels/buffers may change if the parity harness passes) and allow *approximate* training sims as separately identified candidates that still get fresh Mac evaluation | Path B's confirm/invalidate logic is only sound if the shadow is the real game (F6); parity tests are also what makes every speed change safe to adopt (F2). |
| Per-step state hashing (`full_digest`) | **RELAX — redesign, never sample.** Explicit flat serializer of the known semantic state + once-per-run static fragments (config/checkpoint); keep v3 as the reference oracle in differential tests; audit hashes stay at every 250 decisions | It is the shadow's validity check, not logging (F6); the cost is the recursive traversal, not the function. |
| Full replayability / receipts | **KEEP** compact provenance (source/checkpoint/config/bank identities, seeds, action tapes, per-world outcomes, periodic hashes, terminal accounting); **DROP** any requirement to re-hash or re-review a registration for repository changes outside its import closure | Recomputing the conclusion needs the former; the latter is friction with no evidentiary value (F14). |
| Mac-only strict gate + serving qualification | **KEEP** this campaign | The claim is about the Mac-served controller including its deadlines; cloud needs a tie-explaining transfer protocol that does not exist (F15). Development, labels, fits, screens: Colab. |
| One paced session at a time on a quiet Mac | **KEEP for operator-02; RELAX for every later paced run**: register N≈4 concurrent paced sessions (each ≤ 2 threads on 18 cores) with per-session contention accounting and an interleaved single-vs-multi timing probe, pre-data | 16 idle cores for ~25 h is the largest resource waste in the project (F5, F13); but changing the ratified study now risks the only candidate that can reach strict this week. |
| Serial Python opponents | **DROP** as a constraint | Opponent *behaviour* (policy, carry, masks, ties, event order) is the requirement; Python execution is not; `v8_nb` already exists and is decision-identical (F8). |
| Freeze-before-data review loops | **KEEP** the freeze (candidate identity, metric, banks, sample size, selection, looks, stop rules); **RELAX** the loop to one executable acceptance contract (tests + binding/count/schema checks), one independent reviewer, delta-only re-review after fixes, one judge | The freeze is the statistical protection; repeated prose rounds are not (F14); today's six rounds cost ~2 h, so this is agent-hours, not the date. |
| Paced execution of the assisted arm | **KEEP** | Faster-than-real-time search would credit rescues that arrive too late in deployment (F10, F11); bare arm already unpaced (F12). |
| No Colab "strict" evidence; ≤ 3 concurrent Colab assignments | **KEEP** / platform limit | F15; account limit, not ours. |
| Five training replicates for a fixed checkpoint | already **DROPPED** (unit = paired world) | F17. |
| Spend-to-target on Colab units | **DROP** as a driver of lane design | Units buy decisions; unspent units are reported, not padded (colab JUDGMENT §4). |

## 3. Speed program (ordered; arithmetic on F3 shares; all gains are targets, not forecasts)

| # | Work | Expected gain | Eng. time | Builder | Start |
|---|---|---|---|---|---|
| **S0** | **Matched baseline harness**: one script that runs the excluded fixtures (3 mixes, short + long bodies ≥ 800, deaths/respawns, cold reforks) single-thread, reports exclusive stages, completed-world throughput and byte-identity of exports/sidecars vs the frozen reference; Mac (≤ 2 threads, nice 10, never in a quiet window) and one L4 (≤ 2 u) | none — the ruler | 2–4 h | Opus agent | **tonight** |
| **S1** | **Digest v4**: explicit canonical bytes for ordered bodies, food/corpse lists, mutable sim arrays by fixed name order, MT states, carry arrays; static fragments for config/checkpoint hashed once; remove the duplicate env-RNG encoding; schema `t34-shadow-full-digest/v4`; v3 kept as oracle; every field in `tests/test_t34_fast_digest.py` must flip v4 | digest 4× → **1/(1−0.355·0.75) = 1.36×** whole-step | 1–2 days | **Sol** (`codex-handoff.sh`, `CODEX_MODEL` set) | **tonight** |
| **S2** | **Native v8 for shadow/branch rows**: route `_apply_veto_live_hook` rows that are not scored roots through `v8_nb` (nogil), preserving carry and legal masks; keep the Python hook for root rows if counters are consumed, else reproduce counters; prove action identity incl. Q ties | v8 2× → +0.11; **with S1: 1/(1−0.266−0.111) = 1.61×** | 1–3 days | Opus agent (independent owner), Sol reviews | **tonight** |
| **S3** | Raster + clone/assembly buffer reuse (only if S0 shows ≥ 10% in assembly/raster) | ≤ +0.10 | ½–1 day | Opus agent | after S0 |
| **S4** | **4-session paced harness design** for the next Mac paced run (strict gate / L8 successor): N sessions × (tick + 1 worker thread), per-session contention ledger, interleaved 1-vs-N timing probe, pre-data registration text | paced calendar **~3–4×** (18–19 h → ~5–6 h) | 1 day design, 1–2 days build | Opus agent (design), Sol (build) | design **tonight**, build after Path B starts |
| S5 | GPU device-resident sim/search | — | 2–4 weeks | — | **deferred**; reopen trigger in §5 D6 |
| S6 | G4 / A100 for the teacher | — | — | — | **no**: 5.12× unit price for ≤ 4× vCPU, host-bound |

Throughput arithmetic at 1.6× (if F3 shares hold on L4): 55 → 88 worlds/lane-h; an 800-world bank 14.5 → 9.1
lane-h, 24.7 → 15.5 u; on 3 lanes 4.85 → 3.0 h wall. Over the ~500 u expected Colab plan that is roughly 100 u and
a day or two of lane time across the campaign — worth 4 engineer-days in parallel, not worth a pause.
Serving arithmetic: even 1.76× leaves T3-dense at 0.75 s/batch vs 208 ms (F10); search feasibility is Path B's
async design, not engine speed.

Caps: Colab ≤ 10 u total (8 + 2 reserve) through the qualified supervisor; Mac ≤ 2 threads, nice 10, no run while a
retention phase is admitted or running; nothing from S1–S3 enters a registered study except at that study's
registration.

## 4. In-flight work

| Item | Ruling | Why |
|---|---|---|
| Round 0 (Colab; chunk 001 running, autochain to 002; governing verdict = `amendment-boost-floor/decision.json`) | **continue unchanged** | ~13–33 u on one lane, verdict Wed; an engine swap saves < 1 h and would force re-registration. |
| Phase 0 (Mac, 6 workers, ends Wed ~10:00–13:00) | **continue unchanged** | 88.85 h registered job in its last ~12 h. |
| Round 0 Mac replay → Path B timing gate → pre-check → study, operator-02 at `81c54731` | **continue unchanged** (no multi-session amendment, no engine swap) | Paced duration is fixed by fps; any controller change (digest, hook, scheduling) is a new identity needing a new paced test; an amendment now (~2–3 h review + trace receipt + a new 1-vs-N timing qualification that itself needs the quiet Mac) risks the Wed start for ~10 h of gain. |
| Colab unpaced Path B arm (≤ 42.5 u, at `81c54731`) | **continue unchanged** | descriptive arm; cannot affect the verdict. |
| Claim A loss / update-dose amendment (not yet drafted) | **continue — and it takes Sol's calendar before S1 if they collide** (ruling target Oct 21 ~14:00) | F16: no engine speed fixes an objective that flips 0.4% of decisive rows; the colab JUDGMENT already orders it first. |
| Round 0-D rescope (draft + prep worktree; must be on main before the Round 0 sole analysis Wed midday) | **continue; bind the current engine** | It is the only off-ramp for a scientific Round 0 NO-GO; waiting for S1/S2 would miss its deadline; it may rebind a qualified faster engine only by its own pre-dispatch identity receipt. |
| Claim A L4 timing qualification (running, ≤ 5 u) | **continue unchanged** | needed to price fits; unaffected by this ruling. |

## 5. Decision points and kill rules

- **D1 (S0 done, ≤ Wed 09:00 PT):** the production-shape profile must show digest + v8 ≥ 40% of worker time on all
  mixes with long bodies. If not, S2 is cut and S1 continues only if digest alone ≥ 25%.
- **D2 (S1, 48 h after start):** adopt digest v4 only if (a) exports and sidecars byte-identical on the S0 fixtures,
  (b) an M1-style ≥ 15,000-frame live/Grid/shadow identity run shows 0 physical/carry/v8/action/shadow differences,
  (c) every mutation test flips v4, (d) ≥ 1.25× matched whole-workload on the same host. Otherwise revert to v3
  and close the digest line; no second attempt without a new profile.
- **D3 (S2, 72 h after start):** adopt only with exact action + carry identity on fixtures including Q ties, long
  bodies, deaths/respawns and contested food, and ≥ 10% additional whole-workload gain. Otherwise close.
- **D4 (program cap):** 4 engineer-days total, 10 Colab units, Mac ≤ 2 threads outside quiet windows. Any overrun
  stops the program; incomplete measurements are reported as incomplete.
- **D5 (no engine under a registered study):** a faster engine enters only at a registration (R1, 0-D before
  dispatch, L5, L8). A served controller that changes timing is a new identity and gets a fresh paced test.
- **D6 (GPU reopener):** only a registered lane with ≥ 10,000 dense worlds or ≥ 1 M searched decisions per round
  (gpu_expert_2 break-even), or a mechanism current compute cannot test. Not otherwise.
- **D7 (S4 multi-session pacing):** pre-data registration for the next paced Mac run only; kill if the interleaved
  1-vs-N probe shows update p95 > 1.1× single-session p95, any overrun, or cadence loss > 5%.
- **D8 (weekly):** report measured worlds/lane-hour against the 55 baseline and units saved; no gain claimed from an
  isolated kernel or the old 64% profile.

## 6. Tonight (start now, in parallel)

1. **Opus agent — S0 baseline harness** (2–4 h): fixtures, exclusive stages, identity check, one Mac run at nice 10
   ≤ 2 threads while Phase 0 runs (it is a contended-host number, labelled so), one L4 run ≤ 2 u.
2. **Sol — S1 digest v4** via `codex-handoff.sh` with `CODEX_MODEL` set; spec = §3 S1 + D2; v3 stays as oracle.
3. **Opus agent — S2 native v8 branch adapter** (independent owner; Sol reviews): feasibility + identity fixtures
   first, integration second; D3 governs.
4. **Opus agent — S4 design memo** for 4-session paced execution on the Mac (contention ledger, 1-vs-N probe,
   registration text) so the next paced run (strict gate or L8) does not idle 16 cores.
5. Nothing touches the Mac quiet windows, Round 0, Phase 0, or operator-02.
