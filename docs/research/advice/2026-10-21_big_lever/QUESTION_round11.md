# Big-lever research rounds — Round 11 question (RE-PLAN: the week-1 release track measured infeasible; what is the fastest credible route now?)

Judge: Fable (round-11 judge wrote this, 2026-10-21 ~14:35 PT, real wall clock). Read `../CHARTER.md` first, then `../PLAN.md`
(the round-10 product, now partly overtaken), `../round-10/JUDGMENT.md` §(a)/(c)/(d), then `../LIVE_FACTS.md` (authoritative, newest
last) and `../QUEUE.md`. Repo: `/Users/josenunez/Projects/ml/snake-dqn-biglever-ro` (detached read-only copy of `main`; main HEAD at
writing = `18a28ae3`; start with `AGENTS.md`, `docs/STATUS.md` "Campaign authority", `docs/research/final_run_20261019/CAMPAIGN_LOG.md`
lines 98–107 and 178). Never open `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01/`. Rounds 1–10's verified
facts and negative record hold with their corrections; cite them, do not re-derive. **This is a re-plan round, not a red-team of
PLAN.md's text: PLAN.md §0–§5's week-1 release track (Line S) rests on a latency premise that has now been measured false.**

## 1. Goal (unchanged)

A served snake on **this Mac** (Apple M5 Pro, 18 CPU cores, Apple GPU via MPS/Metal, no CUDA) that clearly beats bare B8 s3 — the
released Watch hero since 2026-10-17 (`docs/STATUS.md:29,77`) — on mass at equal-or-better survival, with a claim the owner can trust:
Mac strict gate (pooled ≥ +40 ∧ LB90 > 0, survival band v2, scripted no-death ≥ +.03, pre-data freezing) + serving qualification
(24 fps Watch6 / 12 fps Watch8/12, zero overruns, `SUCCESS_CONTRACT.md:47`) + owner release OK. These bars do not move
(`../round-10/JUDGMENT.md` §c1; `../PLAN.md:28-36`). Speed matters more than thrift (`../CHARTER.md:14`).

## 2. Verified facts (file:line / receipt path; newest measurements first)

**What died today (all verified against receipts):**
- **L1′ Mac latency — served exact-sim search is ~15–25× too slow at W ≤ 4.** Per-root dense search S p50/p99 on CPU 19.7/22.9 s (W1)
  → 23.7/27.2 s (W4); MPS 32–41 s (MPS still runs sim/features/vetoes/export on CPU); tick overruns in every trial (280/344 …
  390/533); vs S p99 ceilings 0.83–1.458 s. Formal verdict NOT_READY(INSTRUMENT_MISSING) (no production root provider; quiet host not
  achieved), but the gap is an order of magnitude and W ≤ 4 cannot close it.
  `../exp/bl-l1-20261021/RESULT_L1.md:3-5,29-44`; `../LIVE_FACTS.md:24`; `CAMPAIGN_LOG.md:107`.
- **Y1 Mac dense batch (6 × H300, CPU W1, contended):** frozen 10.6–26.3 s, mixed 7.6–11.0 s, **scripted 0.6 s** (`RESULT_L1.md:13-23`,
  receipt `../exp/bl-l1-20261021/y1-mac-01/mac-ruler.json`). **The cost is opponent-model-dominated**: the S0 ruler attributes 60–73 %
  of long-body worker time to digest (5–60 %) + v8 opponent veto rows (0–67 %), B8 forward ≈ 13 % (`../LIVE_FACTS.md:5`); S2 native v8
  gave 1.90× whole-workload (`../LIVE_FACTS.md:9`). Implied L4 R ≈ 0.5–1.5 batches/s < 2.18 ⇒ L2′ very likely NOT_DISPATCHABLE
  (formal L4 receipt pending; Y1 L4 failed at upload, `../LIVE_FACTS.md:24`).
- **E1b GPU engine (CUDA dtype fix 88adb3e2, L4): NO_CONTINUE/INCOMPLETE** — 1/18 decisions in the 10,820-s wall budget,
  `full_world_steps_per_second = 0.217` vs the frozen ≥ 20,000/s bar ⇒ engine thesis STOP this week by its own rule (no retry).
  `../exp/bl-e1b/cuda-dtype-fix-20261021-01/run.log` (gpu_spike_progress line; EXIT=0, benchmark.json exported); `../LIVE_FACTS.md:22`;
  `CAMPAIGN_LOG.md:103`. Lever 4 (GPU-resident exact sim) and GPU-scale search-in-the-loop are off.
- **Line R screen cannot reach power:** E9 BLOCKED (.27/.31 < .60); the replacement's best design (fixed 192/64/64) gives minimum joint
  funding **.36216** at true +40 (null-pass .066) ⇒ E2 not funded as a screen this week; R stays research-only.
  `../exp/bl-e2/e9-replacement/E9_REPLACEMENT.md:3-5,160-166`; `../LIVE_FACTS.md:16`; `CAMPAIGN_LOG.md:104`.
- **Claim A (distillation of the corrected-loss rescue teacher into B8) = KILL** — D0 registered rerun: "no corrected objective keeps
  the anchor guards for ≥ 512 updates" (`CAMPAIGN_LOG.md:100`; `../LIVE_FACTS.md:15`;
  `snake-dqn-artifacts/t34-claim-a-20261020/d0-l4-02/output/exports/D0_DECISION.json`). Earlier: T3 screen students −81…−88 vs CTRL
  (`docs/research/advice/2026-10-21_direction/FABLE.md:17-21`).
- **O1 = GO, O2 underpowered:** O1 census 35/48 fresh worlds (frozen 9 / mixed 13 / scripted 13) have a surviving single-action H500
  branch ≥ +5 from death-drop intake (`snake-dqn-artifacts/t34-o1-20261021/census-02-extracted/decision.json`, status GO;
  `CAMPAIGN_LOG.md:101`). O2 at N96 cannot have adequate full-GO power ("no N can promise 80 % at the +40 boundary";
  `snake-dqn-artifacts/t34-o2-20261021/OC.json` interpretation; `PREREGISTRATION_O2.md:127-132`); the V2 re-size needs **5,097 paired
  worlds (~150 runtime-h) for .80 power at +10 incremental over D**; fallback 1,260 worlds at .34 power; prepared, NOT dispatched;
  deferred behind L2′ (`CAMPAIGN_LOG.md:106,178`).
- **E4 COMPLETE:** ~1,440 active hero transitions/s per L4 (6 procs 1,403 vs 12 procs 1,444 — host-bound, not CPU-scaling);
  ~850/s allocated incl. setup ⇒ 8M ≈ 1.5 h, **100M ≈ 19 h per L4** (`../LIVE_FACTS.md:23`; `CAMPAIGN_LOG.md:107`; receipts under
  `../exp/bl-e2/z9-e4-recovery-01/e4-output/`).
- **Round 0 (offline teacher vs bare B8, 240 paired worlds) = KILL/INTEGRITY_FAILURE** (operator error, no outcome read; data sealed;
  mechanical re-registration ~15–40 u, ~3 h, only if a lane commits to consume it) (`CAMPAIGN_LOG.md:98-99`; `../LIVE_FACTS.md:14`).
- **Path B v2 smoke (serial, pre-W4):** 0/34 rescues at 24 fps; leads 224–433 frames (9–18 s) from doom detection to last avoidable
  decision; T3-dense continuous serving 6.3× over the one-thread tick budget (`../LIVE_FACTS.md:3,12`).

**Standing constraints and resources (verified):**
- **Serving stays on THIS Mac** (owner 00:30 PT, `../LIVE_FACTS.md:13`); GPU-host serving is out. Owner YES to W ≤ 4 / K ≤ 8 and to
  the probe-only grant (`../exp/week1-20261021/authority/serving-envelope.json`: answer "YES", W 4, K 8; `../LIVE_FACTS.md:21`).
- **Colab:** ~1,135 u, owner says spend it (`docs/STATUS.md:24-28`: "spend all … by ~2026-11-20"); **3 concurrent assignments
  measured** (4th refused), L4 1.54 u/h observed; T4 max 3 at 2 vCPU each; CPU-runtime cap unmeasured (`../LIVE_FACTS.md:15`;
  `CAMPAIGN_LOG.md:106`). **A100/H100 rule (07:20 PT):** allowed when the job's own L4 profile is GPU-bound (≳ 70 % busy) and
  expected wall gain ≥ 2×; current lanes do not qualify (`CAMPAIGN_LOG.md:105`). Three mechanical Colab failures in 24 h (upload cap,
  CUDA dtype, poll timeout) ⇒ Z-9 upload-path qualification is a readiness condition (`../PLAN.md:77`; fixed 07:10, `CAMPAIGN_LOG.md:104`).
- **Builders:** Sol (Codex) implements in hours-to-a-day per component; Codex-V review was the binding resource in PLAN (36–42 h,
  `../PLAN.md:70-73`); Claude operates; Fable judges. Zero students trained since 10-19 (`FABLE.md` direction memo, line 15-16).
- **Mac:** Phase 0 finished 12:46 PT; the Mac is otherwise free for quiet windows (`../LIVE_FACTS.md:23`); analyticsd/mediaanalysisd
  spikes defeated two quiet-host attempts (`RESULT_L1.md:11,56-57`).

**What is still alive (verified):**
- **The only lever that has ever clearly beaten B8 is exact-sim search over B8, offline and unpaced:** T3-dense ceiling pooled scripted
  survival 54/64 vs 38/64, mean mass 700.0 vs 608.1 (+92) (`docs/research/t3_ceiling_test_2026-10-15.md:115,123`); T3 s3 row +122.8
  (`CAMPAIGN_LOG.md:94`). Note the ceiling was measured on **scripted** census worlds — exactly the mix where search is cheap (0.6 s).
- **Harvest mechanism exists (O1 GO)** but needs a served search to deliver it and an unaffordable O2 to prove it.
- **Speed program:** S2 native v8 merged (1.90× whole-workload, byte-identical); S1 digest v4 bench exists
  (`snake-dqn-artifacts/speed-s1-digest-v4-20261021/bench-01.json`); perf-sim 2.1–2.4× already on main (memory `perf-sim-merged`).
- **PPO backbone (`research/bl_ppo/`, E8 horizon-aware PopArt critic landed, 101 tests; `../LIVE_FACTS.md:16`)** with a measured
  collection rate (E4). What it lacks is a screen with power — not compute: 100M ≈ 19 h per L4, 3 slots ⇒ ~300M per day of wall.
- **E3 (info-value / geometry residual) recovery-04 dispatched** (replay of Mac-verified action tapes; `../LIVE_FACTS.md:25`); E14/E16
  depend on it (week 2 at the earliest, `../PLAN.md:177-180`).
- **Negative record that still binds:** T23 league fine-tune −80.8 vs B8 (0/5 seeds); M3-C no-veto −250; B8 remaining deaths are
  self-enclosure at L ≥ 800 that v8 cannot help (memories `t23-r0-kill`, `b8-death-census`); E5-H horizon bundle hurt; FRP recipe
  saturated at +6.8 (`frp-v4-partial`).

**Arithmetic the advisors should start from (judge's, from the receipts above):** a 20× cut in per-root search cost on this Mac
(27 s → ~1.3 s p99 at W4) would land inside the K8 tail cap only for the *slowest* cell; the 24-fps per-decision budget is 41.7 ms,
so continuous per-tick search needs ~500×; a doom-triggered burst with a 9–18 s lead needs the *first improving* result inside ~5 s
and the certificate inside the lead. Scripted-opponent search already fits (0.6 s per 6×H300 batch) — the cost is the frozen/mixed
opponent models (v8 veto rows + digest), i.e. simulating the *opponents* exactly, not the hero.

## 3. What Round 10 left that this round must now overturn or re-found

PLAN.md's Stage 0–3 (Z-0, Y1, L2′, Z-2′, X7, L3/L4/strict/product/serving by Tue 23:00) was built on "L1′ K1 cell PASS at C p99 ≤ 333/542
ms" being *possible*; it measured ~20 s. Friday's admission (`../PLAN.md:123-132`) cannot be met by any cell. The R track was already
research-only; its screen (E9′ .36) cannot admit a 100M run by the registered rule; and the one GPU-scale idea (E1b) failed its bar by
five orders of magnitude. O2 is unaffordable at power. Nothing in the current portfolio is on a path to a *served* gain over B8 within
the week, and the 72-hour plan in PLAN §2–§3 is now mostly dead text. **Do not patch PLAN.md's gates; propose the route.**

## 4. The open question (one)

**Given these measurements, what is now the fastest credible route to a served gain over bare B8 s3 on this Mac with a claim the owner
can trust — and, concretely, what should the next 72 hours (Wed 15:00 → Sat 15:00 PT) do with 3 Colab slots, Sol, and the free Mac?**

Treat as open — argue from the receipts, not the list — whether the route runs through making search ~20× cheaper (shallower or sparser
search, learned-value truncation of rollouts, scripted-opponent-only or cheaper opponent proxies, caching / precomputed lookahead,
anytime search bounded by the measured 9–18 s lead), through distilling search into a bigger student that still serves at 24 fps
(where Claim A's loss failed and T3 students lost −81…−88), through PPO at 100M+ steps despite a screen that cannot fund it (and what
*would* count as evidence then), through a different lever entirely, or through accepting a longer timeline — and name the first
measurement, its cost, and its kill rule. Every proposal must state (i) what the owner would be able to trust at the end (which bar it
reaches, which it leaves unproven), (ii) what the Mac serial clock needs, (iii) what evidence would make *you* stop it.
