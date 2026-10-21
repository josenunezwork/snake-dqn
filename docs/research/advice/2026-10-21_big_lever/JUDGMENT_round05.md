# Round 5 judgment — big-lever research rounds

Judge: Fable (round-5 judge), 2026-10-21 ~03:30 PT. Inputs: `../CHARTER.md`, `../round-01..04/JUDGMENT.md`, `../LIVE_FACTS.md`
(newest entries 00:25 and **00:30 PT owner decision: serving stays on this Mac**), `../QUEUE.md`, `QUESTION.md`, `astra_01..10.md`
(advisors read snapshot `4857ac63`). Repo verified read-only at main `7122a8da` (three commits after the advisors' snapshot: the
E6/E7 development allocation is now bound, `7387fa84`). Sealed study not opened. No jobs run.

Repo state that matters (verified at `7122a8da`): **E8 still not landed** (`research/bl_ppo/model.py:101` `self.critic = nn.Linear(256, 1)`,
no horizon scalar, no PopArt; `grep horizon|popart research/bl_ppo/*.py` finds only the H5000 constant). **E10 text still absent**
(no `FINITE_MENU` anywhere in `research/bl_prefix/`). **No E16 harness and no `PARITY_RULE.md` on main** (`exp/bl-e16/` is empty;
QUEUE says Sol is building it). `screen.py:81-105` computes `Σ s²/(9 n_m)` — so the pilot's SE at paired SD 200 and 96 worlds/mix
is **11.8, not 18** (Round 4 carried the 96/32/32 number; 02, 04, 09 are right). New since the advisors wrote: **Round 0 printed
KILL / INTEGRITY_FAILURE by operator error — no pair record was read** (`CAMPAIGN_LOG.md:98`); the search-teacher premise test is
scientifically unread, not negative.

## (a) Bottom line

1. **All ten advisors answer the question the same way: no observable at 4M or 32M carries a *calibrated* probability of +40 at
   100M, and nothing this week can manufacture one.** Every history is a loser (sensitivity to winners is unidentified; even five
   independent failures leave a 90 % upper success rate of ~37 % — 02, 09), training checkpoints are different controllers rather
   than accumulating observations of one effect (04), and a saturating curve and a delayed-discovery curve agree on everything
   through 32M (02). The honest deliverable is a *falsifier*, not a forecast: freeze predictions from a 4M block, continue
   unselected arms a short distance, and see whether the prediction beats persistence (01, 02, 03, 04, 05, 08, 09, 10 — eight
   independent versions of the same ≤20-u design; registered once as **E17**, §f).
2. **Early *damage* is visible; early *promise* has never been seen.** Verified from the repo: T23 was already −92.9 at 2M and −77.3
   at 4M equal-mix vs B8 with food/alive-frame −12..−17 % in every mix (`Q0_MASS_AUDIT.md:7-23`; the 8M screen −80.8 was a different
   bank); T4c's decisive agreement crawled 10.0 → 16.5 → 22.7 → 27.6 → 43.7 % at 0.1/0.5/2/4/8M while counter agreement sat at
   99 % (`TWO_SIDED_T_s10/log.jsonl:1,5,20,40,80`); T6's restart stratum had **16** L ≥ 800 decisions in the 4M block against
   31,051 fresh (`RESTART_s10/log.jsonl:40`) — its "zero restart deaths" was a denominator artefact; S1 exported no mid-run weights
   (`PREREGISTRATION.md:40-43`). **M3-C was an M2b fine-tune, not a B8 one** (`PREREGISTRATION_M3C.md:10-13`) and its probes rose
   then reversed (151.9 → 212.1 → 187.7, `15m-probe-summary.json:77-85`); the Round 5 question was wrong to list it as a
   B8-warm-start curve. So a food-productivity / common-alive-loss tripwire at 2M has specificity; nothing has sensitivity.
3. **The frozen E16 admission rule is a funding screen, not a forecast, and it is kept as a screen.** Ruling (§c1): passing
   E16 buys a **32M tranche with a futility-only look**, never "admission to 100M"; the 100M decision is taken at 32M by
   04's two-look funding policy with 09's labelling: forecasts are model-conditional sensitivity ranges, and if plausible models
   disagree by > .20 the forecast is **UNCALIBRATED** and the remaining budget buys *balanced endpoints* (10's design), not the
   4M winner. Pre-data amendment to E15; the T34 contract is untouched.
4. **Two advisors' per-arm "signatures" tables are adopted as E16 reporting rows (0 u), never as gates** (§c4): geometry must show
   ordinary-state advantage before rescues (05's `D300` geometry-minus-control on fixed roots), search targets must show
   target-root → autonomous → natural-start transfer, archives restart → 32-decision-earlier ancestor → natural (07's
   acquisition / generalisation / discovery labels), pure PPO must show realised ≥ 2-decision departure runs. 06's held-out
   directional gradient `Z` is reported, not gated. Critic preparation gets a **finite deadline** (one repair attempt, 06's E19),
   closing 09's "immune mechanism" loophole.
5. **The owner's new question changes the round's arc, and I verified the facts it needs.** "Served search is the only lever
   that clearly beat B8" is true *offline, unpaced, scripted-only*: the T3-dense ceiling searched every 5th decision at
   L ≥ 600 (`trigger.py:20-27`), 6 actions × H300, on the 32 reused scripted census worlds for s0 and s3 — 54/64 vs 38/64, mass
   700 vs 608 (+92), **75 of 301,589 decisions changed**, 36,996 batches at **0.757 batches per worker-second (1.32 s per
   batch)**, 1,126 env-frames/worker-s (`dense/ceiling-result.json`). Nothing measures it on frozen or mixed worlds; if those
   gains were zero the strict-gate pooled value would be ≈ +31. Serving it continuously needs 4.8 batches/s = **6.34× one
   worker** (the 6.3× in LIVE_FACTS). The Phase 0 batched engine already measured **3.35–4.24× per search on Mac CPU and
   4.12–6.02× with the B8 forward on MPS**, fixture-identical (`PROFILE.md:64-66`); six worker *processes* gave 2.79× on six
   searches (`SPEED_REPORT.md`); the smoke's six in-process *threads* collapsed the tick to 5–16 fps (`RESULT.md` M3 pacing).
   The Mac has **6 performance + 12 efficiency cores** (`sysctl hw.perflevel0/1.physicalcpu` = 6/12), torch 2.9.1 with MPS.
6. **Latency alone is not what the smoke falsified.** From its per-doom table (N = 1, 68 dooms): the first fully evaluated
   improving candidate arrived *inside* the lead in **53/68** dooms (median 6.5 s vs median lead 10.1 s), yet 0/34 rescues and
   7 delays were delivered, because 475 improving evaluations completed for frames that had already passed and 41/75 of the
   T3 changes were delay-only. A k× faster engine moves 53 → 58/68 at k = 2 and no further. The binding constraint of the
   doom → LAD design is *which branch points are searched when*, plus the delay-only share — not raw search time. The T3-dense
   mechanism has no doom trigger and changes 0.025 % of decisions, so it is **speculatively pipelineable**: W worker processes
   can search decisions t+5, t+10, … from the exact shadow; an executed change invalidates the queue ~75 times per 300k
   decisions. On paper, W = 2 with the batched engine covers 4.8 batches/s; W = 7 at the current engine does not fit 6 P-cores.
7. **Round 6 therefore asks one question with two halves** (QUESTION.md): what binds between T3-dense's offline +92 and a
   served, strict-gate-passing gain on this Mac (engine latency, process parallelism/pipelining, the tick's zero-overrun budget,
   search scheduling, or the mechanism's unmeasured frozen/mixed gain), and how the big-lever push splits between a Mac
   served-search package and the Colab RL/representation line whose funding rules this round fixed. The three cheapest
   measurements are 0 u (smoke replay with evaluation times ÷ k and LAD-first ordering), a quiet-Mac serving-shape latency
   probe, and a ≤ 16-u dense ceiling on fresh frozen/mixed worlds — the last is the one that can kill the lever before any
   systems work.
8. **Shortlist:** served exact-sim search enters at the top as the owner's lever (40 % push / 25 % at bar, conditional on the
   frozen/mixed ceiling); representation 30 / 18; long-horizon PPO backbone 25 / 10; engine 30 % (Colab role unchanged);
   search-in-the-loop 15 / 10; archive 6; opponent add-on 4; world model / league 2. The week's calendar is set by the Mac,
   not by units; Colab's job is to kill or confirm the served lever's frozen/mixed value and to run the RL line in parallel.

## (b) Per-advisor verdicts

| ID | Persona | Verdict | Why (verified facts in **bold**) |
|---|---|---|---|
| 01 | Large-scale on-policy RL | **keep** | **E8 fields absent from persisted progress (`run.py:455-461,519-523` prints counters only; `model.py:97-101` critic has no clock)** verified. Correct that retention + repeatable discovery, not early mass, is the only defensible predictor, and that five rescues/100 (Wilson 90 % 2.6–10.3 %) is not one-third progress. Its persistence assay (one extra 4M block for A + ≤ 2 treatments, fresh 96/mix, predictions frozen before 8M, 16 u + 4 reserve) is one of the E17 forms; subjective arm priors (A 5 %, G 20 %, S 15 %, R 10 %, P 5 %) are labelled planning priors — accepted as such. |
| 02 | Learning-curve extrapolation | **keep** | Best statement of the identification problem (saturating vs delayed-breakthrough curves agree through 32M). **SE = 200/√288 = 11.8; Holm first threshold ≈ 19.4 not 29.6** (verified against `screen.py:88`). **T23 skipped its 2M/4M gameplay looks (`RESULTS_T23.md:5-6`)** verified. Minimum-information arithmetic (25 rescues → 20 % relative SE; 96/mix gives ~4.8 events) is the reason E16 cannot calibrate anything. Its 4/8/16 → 32M withheld-endpoint design is the most informative E17 variant but costs the whole 20 u; adopted at 4 → 12M with an 8M snapshot (§f). |
| 03 | Negative-results forensics | **keep** | The round's evidence: **T23 −92.9/−77.3 at 2M/4M; T4c decisive agreement 10.04 → 43.71 % vs counter 99.44 %; T6 16 restart L ≥ 800 decisions at 4M; S1 no intermediate weights; M3-C from M2b with 151.9 → 212.1 → 187.7** — all verified line by line. Right that T6's 1.65× was a conditional hazard ratio, not exposure. Its 4M → 12M transport test (5.22 h, 8–10.4 u) is E17's base form. Correctly refuses a point probability ("[0, 1]"). |
| 04 | Sequential design | **keep** | Formal two-look funding policy (nominate at `p_mass ≥ .30`, stop the pair at 32M below `.10`, efficacy spending (0, 0, .10)) with its own OC table (null admits 19 %, +40 pipeline 61 %, delayed 0 → 20 → 40 only 11.5 %) — adopted as the *structure* of E15's amendment (§c1). **SE correction verified.** Right that `p_joint ≤ p_mass` and that survival dynamics are unidentified. Its prior (`θ ~ N(0, 40²)`) is an assumption, labelled; 09's calibration demand applies. |
| 05 | Representation / latency | **keep** | **E3 encoder applies two stride-2 convs then pools (`students.py:26-31`)** — full-resolution input does not guarantee one-cell topology, verified; **E3 source mix 210/30/16 (`PREFLIGHT.md:38`)** verified; **E2 already adds a residual on B8 (`model.py:155-158`)** verified, so E16-G must preserve the *F parent's* residual, not reset to bare B8. The `D300` geometry-minus-control diagnostic on fixed roots is the right ordinary-state signal and is adopted as a reporting row. Its 20 % (5–40 %) is declared subjective. |
| 06 | Credit assignment | **keep** | GAE decomposition (first error term cancels, future action-correlated errors do not) is correct and important: mean bias ±5 can hide opposite-signed errors on straight vs escape actions. **Critic shares the actor core and optimizer (`model.py:96-101,155-158`; `learner.py:219-255`)** verified — freezing the encoder alone is not a frozen actor. Its one bounded critic-repair intervention (≤ 20 u) becomes **E19**, the finite deadline on "critic preparation" (§c5). `Z` is reporting-only. |
| 07 | Exploration / search targets | **keep** | Acquisition / generalisation / discovery labels and the "32 decisions before the first deviation" ancestor test are the right archive signals; held-out 48 source worlds (16/mix) outside training and E16's bank is correct adaptive-reuse hygiene. **`rollout.py` continues B8 / survival-first** and **T35 zero novel rescues** verified again. `τ ≥ .681` at q = .4, f = .6 and "five rescues/100 ⇒ τ ≈ .208: the gap is 3.3× execution, not 25× transitions" — checked. Its 20-u panel is conditional on R/S arms existing. |
| 08 | Distributed systems / Colab | **keep** | **`--checkpoint-every` default 120 s (`run.py:625`); identity binds `target_transitions` so 32M → 100M refuses exact resume (`run.py:380-399`, `recovery.py:88-90`); preflight restored only a completed 400-transition endpoint (`PREFLIGHT.md:18-21`); 25 % learner allowance unmeasured (`PREFLIGHT.md:54-56`)** — all verified. Cost table (two 32M prefixes 11.1 h; 70 u tranche incl. reserve; 160-u replication envelope) adopted for E15. Rule: predeclare target = 100M with a pause boundary at 32M; export cadence ≥ 10× measured transfer time. |
| 09 | Evaluation / statistics skeptic | **keep** | Three corrections adopted: SE 11.8; `J` measures improvement over a parent including ordinary PPO learning, so mechanism attribution needs **`D = arm − E16-A`**; "critic preparation" without a deadline makes a mechanism unfalsifiable. Intersection-union composite p-values with Holm across five reserved arm slots, Dwork burn rule, proper scoring of any forecast — adopted into the E13 family file. Its persistence baseline ("next G = current G") is E17's comparator. |
| 10 | Red team | **keep** | Honest retreat from the universal negative: **T23 is early-informative for damage**; the defensible claim is "no demonstrated predictor justifies concentrating the endpoint budget on the 4M winner" — ruled correct (§c1). **CTRL death-length cycling 86–1230 with a 1.28M replay period (`panel_review_summary:21`)** verified — 4M trends can encode replay phase. Its incremental-forecast-skill statistic `D` with a 400 mass² threshold is E18's primary; its "buy balanced endpoints if UNCALIBRATED" (control vs geometry, 3 seeds × 100M, 128–167 u) is the fallback branch of E15. |

Factual errors found: none that change a conclusion. Corrections to the Round 5 fact sheet: (i) SE at 96/mix is 11.8 (Round 4
carried 18); (ii) M3-C is not a B8 warm start; (iii) T23 *has* early measurements (Q0 audit) — the "nobody reported first blocks"
premise was wrong for T23 and right for S1/T6/T4c at the gameplay level; (iv) T6's 1.65× is a hazard ratio, not exposure.
Owner-input facts the advisors did not have: serving stays on the Mac (00:30 PT); Round 0 is an integrity KILL with no outcome.

## (c) Disagreements ruled

1. **What E16 admission buys — 04 (predictive-probability nomination), 09/10 (no efficacy-based selection until forecast skill is
   shown), 01/03/08 (keep the screen, fund 32M only).** Ruling, as a pre-data amendment to E15: E16's frozen rule stays and admits
   an arm to a **2 × 32M tranche with a futility-only look** (tripwires, survival band v2, UB90(G) ≤ 0, common-alive loss < −10,
   critic not scale-qualified ⇒ STOP). The **100M decision is taken at 32M** on 04's joint forecast of E15's two-seed nomination
   event, computed under ≥ 3 declared models (plateau / continued / forgetting, 07's forms; 04's Gaussian hierarchy), reported with
   09's labelling: if the models' `p_mass` disagree by > .20 the forecast is **UNCALIBRATED**, and the remainder is spent on
   *balanced endpoints* (control + the pre-nominated treatment, 2 seeds each to 100M, 10's branch) rather than on a forecast-selected
   arm; if they agree, nominate at `p ≥ .30`, stop the pair below `.10`. Efficacy α is spent only at 100M (E15's .05 reserved, Holm
   across endpoints). Selection among *treatments* at 4M is by prerequisites (E3/E14/E1/E10 availability) and by `D = arm − A`
   exceeding its simultaneous bound — never by `G` rank alone.
2. **Size of the short-horizon falsifier — 02 (4 → 32M with 8/16 snapshots, 12–16 u), 03/04/05/09 (4 → 12M, 8–11 u), 01 (+4M
   only, 16 u).** 03/09 win on cost-per-information: 4 → 12M for one pre-nominated treatment plus E16-A with an 8M snapshot,
   forecasts frozen before the 12M reveal, fresh 96/mix bank; a second 12 → 32M leg is admitted only if the 12M prediction beats
   persistence. 02's withheld-endpoint logic and 09's persistence comparator are both inside it (E17).
3. **SE and the Holm threshold — 02/04/09 vs Round 4.** Round 4 corrected: SE ≈ 11.8 at 96/mix (`screen.py:88` is the stratified
   formula). Consequences: Holm's first one-sided threshold at family .10 across two arms is ≈ 19.4; at 09's five reserved slots
   ≈ 27.4; power at true J = +20 is 26.5 %, at +40 85.7 % (09's table, checked). E9 must use the stratified formula and measured
   SDs, not 200.
4. **Mechanism signatures as gates (01, 05, 07 partly) vs as reports (02, 09, 10).** Reports. Each signature is logged and its
   falsifying next-block observation is pre-stated (02's, 04's, 09's tables agree), but no signature admits or kills an arm at
   4M — only the frozen E16 rule and tripwires do. The one exception is 05's identity precondition: an E16-G arm whose
   initialisation does not reproduce the F parent's logits is INVALID, not a null.
5. **Critic preparation without end (09) vs 06's repair.** 09 is right that "INCOMPLETE (critic)" without a deadline immunises a
   mechanism. Ruling: one registered repair attempt per arm (E19: refit the critic only against complete returns with every
   actor-affecting parameter frozen, then two matched 0.5M diagnostic continuations); a second E12 failure after repair is a
   **KILL of the recipe**, logged as "critic cannot be scale-qualified on these inputs", which is evidence *for* lever 3, not an
   excuse to re-run.
6. **A100/H100 — unanimous again: no.** Ten of ten repeat that a 5× price needs 5× useful throughput on a host-bound collector.
   Unchanged; serving-side GPU is now also ruled out by the owner (Mac only).
7. **The owner's hypothesis ("per-decision latency is the real unlock") — judged on the record, not by the advisors.** Partly right.
   Necessary: continuous T3-dense needs 6.34× one worker, and the serving contract's "one extra CPU thread" (`SUCCESS_CONTRACT.md`
   §4; `app.py:37`) makes that impossible by rule, which the direction judge already proposed lifting (`FABLE.md:24-26`). Not
   sufficient: the smoke shows improving evaluations inside the lead in 53/68 dooms with 0 rescues delivered (§a6), and the +92 is
   scripted-only on reused worlds. The lever's value is bounded first by the frozen/mixed ceiling (L2) and then by scheduling and
   process isolation (L0, L1) — Round 6's question.

## (d) Running shortlist (carried from Round 4, updated)

| # | Lever | Conf. push / beats B8 at bar | Cost to first signal | Most informative next experiment |
|---|---|---|---|---|
| new 1 | **Served exact-sim search on this Mac** (T3-dense mechanism: periodic 6-action H300 batches from an exact shadow, speculative pipelining across W worker processes, batched/MPS engine; Claim B identity, visibly labelled) — the owner's lever | 40 % / 25 % (conditional: frozen/mixed ceiling unmeasured; +92 is scripted-only; paced identity unproven) | L2 ≤ 16 u + L0/L1 0 u (Mac quiet window); systems package Sol 2–4 days | **L2 dense ceiling on fresh frozen/mixed worlds** (kills the lever at < +40 pooled-equivalent), then L1 serving-shape latency probe |
| 1→2 | **Representation: full-res geometry residual on the F parent** (`Q_F + Wz`, separate rich critic) | 30 % / 18 % (down 5/2: stride-2 stem may discard the topology it is meant to add; serving budget now shared with a planner) | E3 16 u + E14 0 u + E16-G | E3 → E14 (E3 architecture, on the same Mac core budget as the planner) → E16-G with the `D300` row |
| 2→3 | **Long-horizon on-policy RL from B8** (backbone; E8-amended objective) | 25 % / 10 % (down 5/2: no positive early block exists in any history; funding now staged 4M → 32M → 100M) | E8 0 u → E4 → E2 → E12 → E16 → E17 | **Land E8**, then E2; E17 continuation falsifier decides whether any early look has forecast value |
| 3→4 | **GPU-resident exact engine** (Colab only: 10^9 and E16-S branch rates; no serving role) | 30 % reaches ≥ 20k long-body world-steps/s (unchanged) | 20 u, E1b prepared, undispatched | E1b as briefed; latency mode dropped from its brief (serving GPU is out) |
| 4→5 | Search-in-the-loop (one-deviation current-policy targets) | 15 % / 10 % (down 5/2: its serving twin is now lever 1; its training form still waits on E1) | 20 u after E1 | E16-S |
| 5→6 | Restart archive / PLR | 6 % add-on | E7 (bank now bound) + E16-R | E10 U envelope first |
| 6→7 | Opponent-conditioned add-on | 4 % | after E3b | — |
| 7→8 | World model; league | 2 % / 2 % | — | not funded |
| 1b | Retention sub-arm (cumulative anchor / frozen trunk) | conditional | 36 u | on the collateral trigger |

## (e) Still unresolved

- **T3-dense's gain on frozen and mixed worlds** — the single number that decides whether lever 1 can reach a pooled +40; never
  measured (dense ceiling = scripted s0/s3 on 32 reused census worlds).
- **Paced identity of a served planner**: whether W worker processes on 6 P-cores keep Watch6 update p95 ≤ 11.688 ms with zero
  overruns while sustaining 4.8 batches/s; the smoke only shows that six in-process threads do not. Also whether the Watch
  opponents (FRP3 + v8, 56.8 % of worker time in S0) make a batch 1.5–2× dearer than the dense ceiling's scripted batches.
- **Scheduling, not latency**: what share of the smoke's 34 rescue-capable intervals a LAD-first or speculative-pipelined schedule
  would have delivered at k ∈ {1, 2, 3, 5, 10} engine speed (0-u replay, L0).
- **The serving contract clause** ("one extra CPU thread, ≤ 1 outstanding request") — a product rule the owner ratified 10-19; lifting
  it to a registered W is a legal pre-data amendment but needs the owner's word; the judge cannot grant it.
- **Whether any early look has forecast value at all** — E17 is the first prospective test; until it reports, the 32M decision
  runs on the UNCALIBRATED branch by default.
- E8, E10, E13, E16 harness, E9 with measured SDs — all still unwritten or unlanded on main; the RL line cannot collect until E8 lands.
- E4 rate, E5 cap, and T34's slot usage after the Round 0 integrity KILL (its replay and any re-analysis are the owner's call).

## (f) Experiments Claude can run now (≤ 20 u each; dispatch via Sol between rounds; report in LIVE_FACTS.md)

Mac serving line (owner-requested; all development evidence, none touches a registered study):

**L0 — Speed-up-to-delivery replay of the Path B smoke (0 u; Mac analysis only; Sol 0.5 day).** From the 80 retained
`lad-*.json` controller records (`t34-pathb-smoke-20261021/`), replay each doom's evaluation timeline with completion times divided by
k ∈ {1, 2, 3, 5, 10, 20} under (a) the recorded chronological schedule and (b) a LAD-first schedule (search the T1 LAD quartile
window 16–21 / 88–100 / 248–254 decisions before doom first), counting deliveries separately for the 34 rescue-capable intervals and
the 41 delay-only ones. Kill: if per-evaluation completion times or candidate frames are not in the records ⇒ INCOMPLETE (say so);
if even k = 20 under LAD-first delivers < 50 % of rescue-capable intervals ⇒ the doom → LAD design is the wrong serving form and only
the periodic T3-dense form goes forward. Report the k at which deliveries cross 50 % and 80 %.

**L1 — Serving-shape latency probe for T3-dense on this Mac (0 u; quiet Mac window ≥ 1 h, AC, lid open; Sol 1 day).** Wall time
per 6-action × H300 eps-0 batch from a single live-shaped world (L 800–1,500, Watch6 opponents FRP3 + v8, S2 native v8 on) using
(i) the T3 ceiling engine as run, (ii) the Phase 0 batched engine on CPU one thread, (iii) the same with the B8 and FRP3 forwards on
MPS; then throughput and tick impact with W ∈ {1, 2, 3, 4} *worker processes* (not threads) pinned by QoS while the Watch6 server runs
paced at 24 fps: report update p95/p99, overruns, delivered batches/s, and the speculative queue depth needed (`K = ceil(latency /
0.208 s)`). Kill: W = 4 processes cannot sustain ≥ 4.8 batches/s with zero overruns and p95 ≤ 11.688 ms ⇒ continuous T3-dense is
unservable on this Mac without a new engine; report the shortfall factor. Chosen actions must equal the reference engine's on all
fixtures (reuse `tests/fixtures/phase0_teacher_speed` parity harness).

**L2 — Dense ceiling on fresh frozen and mixed worlds (≤ 16 u on Colab CPU, or Mac 3 slots if the calendar allows; Sol 0.5 day for
bank wiring).** Run `ceiling.py --trigger-mode dense` (unchanged rule: every 5th decision at L ≥ 600, 6 actions × H300, B8 greedy
continuation, eps 0, no veto) with B8 s3 on 64 fresh collision-audited development worlds each for frozen and mixed (never the strict
banks), paired vs bare B8 s3 on identical seeds; also 32 fresh scripted worlds as a reused-vs-fresh control. Cost at the measured
0.757 batches/worker-s: ≈ 578 batches/world × 1.32 s ≈ 13 min/world/worker, 160 worlds ≈ 35 worker-h (frozen batches dearer) ⇒
cap 16 u / 8 L4-h on 12 vCPU, first cap wins; forecast overrun ⇒ NOT_DISPATCHABLE. Readout: per-mix paired mass and survival with
world-clustered LB90, pooled-equivalent `(scripted + frozen + mixed)/3`, changes per 1,000 decisions, delay-only vs horizon-surviving
changes, and the mass ledger. Decision (frozen): pooled-equivalent UB90 < 40 ⇒ **lever 1 cannot reach the gate even unpaced — KILL
the served-search push** (the RL/representation line becomes the week's only lever); LB90 > 0 in frozen and mixed and
pooled-equivalent point ≥ 40 ⇒ fund the Mac systems package. This is the first thing to dispatch.

Colab RL / representation line (pre-data amendments and one new falsifier):

**E17 — Short-horizon forecast falsifier (NEW; ≤ 20 u; Sol 0.5–1 day; after E16's 4M endpoints exist).** Pre-nominate, before any
E16 efficacy is read, E16-A plus one treatment (by prerequisite availability, not by score). Freeze, per arm, a 12M forecast of
`G`, collateral and productive rescues from the 4M block's logged signatures (02/04/09 tables) and the persistence forecast
("12M = 4M"). Continue both unchanged (exact resume, same schedule) to 12M with an 8M snapshot; evaluate 4M/8M/12M, parent and B8 on
one fresh 96/mix bank, one look. Cost: 16M transitions at the E4-measured rate (2.2 h at 2k/s) + 864 episodes (≤ 1.5 h at ≥ 800
world-steps/s) + 2 h setup ⇒ ≈ 6 h / 9–12 u; cap 10 h / 20 u incl. 4 reserve. Kill: integrity ⇒ INCOMPLETE; signature forecast no
better than persistence on held-out 12M (squared error, world-clustered) ⇒ the signatures lose any funding role and the 32M look
runs on the UNCALIBRATED branch; better than persistence ⇒ a 12 → 32M leg is admitted under the same rule. A positive result
licenses another short-horizon test, never a 100M probability.

**E18 — Zero-unit historical forecast audit (0 u; Sol 0.5 day; before E17).** Inventory every timestamped early snapshot / block log
and endpoint row (T23 Q0 2M/4M + 8M; T4c per-seed logs; T6 restart logs; S1 endpoint-only; M3-C labelled as M2b-start, kept
separate), with missingness explicit. Freeze four predictor families (common-alive loss / food per alive frame; decisive vs counter
agreement; occupancy denominators at L ≥ 800; realised departures) and compute 10's incremental skill `D` leave-one-recipe-out
against a recipe-only baseline. Kill: < 3 independent recipes with both early and endpoint rows ⇒ NOT CALIBRATABLE (expected);
any predictor with upper bound < 400 mass² is retired from funding roles. Negative-only histories can only reject predictors.

**E15 amendments (0 u; Sol 0.5 day; write before any 32M dispatch).** (1) Two-look funding per §c1 (32M futility-only; 100M
decision at 32M by joint forecast under ≥ 3 declared models; UNCALIBRATED ⇒ balanced endpoints: control + pre-nominated treatment,
2 seeds each; envelope 160 u incl. 32 reserve at 2k/s, 240 u at 1k/s, else reject the calendar). (2) Predeclare `target_transitions
= 100M` with a pause boundary at 32M (identity refuses a target change, `run.py:380-399`); never reset optimizer / PopArt / counters.
(3) Snapshot cadence from measured transfer time C: interval ≥ 10 C, ≤ 1 h lost work; local-only snapshots do not count.
(4) Efficacy α spent only at 100M; 09's intersection-union composite p-values with Holm across the five reserved arm slots;
mechanism superiority needs `D = arm − A`. (5) Burn rule: every world seen by E6/E7/E16/E17 is excluded from E15's bank.

**E16 reporting rows (0 u; add before the harness freezes).** Per arm at 0/2/4M: 05's `D300(s) = Q300^F(s, a_arm) − Q300^F(s,
a_control)` on 384 fixed outcome-independent roots (128/mix, six branches, shared F continuation and RNG), by mix × length; 07's
acquisition / generalisation / discovery labels with 48 held-out source worlds (16/mix) outside training and the ranking bank; realised
≥ 2 / ≥ 3-decision departure runs at L ≥ 800 per life and the mass of lives containing one; 06's `Z` (held-out directional gradient)
and E12 cells with unique-world support counts; re-death lag strata ≤ 100 / 101–300 / > 300. E16-G initialises as `logits_F + Wz`,
W = 0, with an actor-identity check (05) — mismatch ⇒ INVALID.

**E19 — Critic-repair deadline (conditional; ≤ 20 u; Sol 1 day).** On an E12 failure for a pre-nominated arm: refit the critic only
against complete returns from 192 fresh worlds/mix at 2M and 4M with every actor-affecting parameter frozen (the core is shared —
`learner.py:250-255`), verify actor outputs unchanged, then two matched 0.5M diagnostic continuations (original vs repaired critic)
evaluated on fresh 96/mix. 14.4M world-steps at ≥ 800/s + 1M transitions; cap 10 h / 20 u. Repaired EV passes E12 but execution
UB90 ≤ 0 ⇒ the critic was not the block; E12 fails again ⇒ **recipe KILL** ("not scale-qualifiable on these inputs").

**E9 — refine (0 u):** use `screen.py`'s stratified variance with the T23/S1 paired SDs by mix, not 200; simulate E16 + E17 +
the 32M look jointly under plateau / continued / forgetting trajectories and shared-row correlation 0 / .5 / .9; publish false
funding, false final efficacy, missed winners and expected units separately (04).

**E1b — brief change (0 u):** drop the "latency mode on this Mac" item from QUEUE's E1b follow-up (serving GPU is out by owner
decision); the Mac latency question is L1. E1b's throughput spike is unchanged.

**Dispatch order:** **L2** (first; it can kill lever 1 for ≤ 16 u) → E8 (commit) → D0 rerun → E4 → E5 → E10 (register) → E6 + E7
(allocation now bound) → L0 (0 u, any time) → L1 (first quiet Mac window after the Round 0 replay question is settled) → E1b →
E3 → E2 → E12 → E14 → E16 (+ reporting rows) → E17 → E13/E15 (amended) → 32M tranche. E18 and the E15/E9 amendments are 0 u and
run whenever Sol has a half day. Budget this inter-round: L2 16 + E1b 20 + E2 20 + E3 16 + E4 2 + E5 1 + E6 4 + E7 20 + E12 16 +
E16 20 + E17 20 ≈ 155 u; L0/L1/E8/E9/E10/E13/E15/E16-rows/E18 are 0 u. Deferred: E19 (on trigger), 07's R/S panels (on arm
existence), any A100/H100 use, any serving-contract change (owner's word required).
