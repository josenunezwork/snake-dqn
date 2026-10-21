# Round 11 judgment — RE-PLAN after Line S failed its Mac latency diagnostic

Judge: Fable (round-11 judge), 2026-10-21 **~14:45 PT** (wall clock `date` = 14:37 PDT at start). Inputs: `../CHARTER.md`,
`../round-01..10/JUDGMENT.md`, `../PLAN.md`, `../LIVE_FACTS.md` (newest 14:30 PT), `../QUEUE.md`, `QUESTION.md`, `astra_01..10.md`
(advisors read snapshot `a7bd01e9`; main HEAD `18a28ae3`). Repo read-only; artifact receipts under `../exp/` read with light
commands. Sealed study not opened. No jobs, Colab or tmux. Products: this file and `../PLAN_v2.md` (replaces PLAN §0–§5 week-1
release track; §6–§8 R-track rules and §7 reducer/signature texts stay in force where PLAN_v2 does not override them).

Facts newer than LIVE_FACTS or sharper than the question, verified in receipts/code and folded in as authoritative:
- **Y1 Mac receipt stage split (judge's extraction from `exp/bl-l1-20261021/y1-mac-01/mac-ruler.json`, medians of 30):**
  `opponent_v8` = 20.15 / 22.83 s (frozen-1200, 88 %), 19.18 / 21.26 (frozen-1500, 90 %), 9.09 / 10.60 (frozen-800, 86 %),
  9.06 / 10.87 (mixed-1200, 83 %), 8.91 / 10.38 (mixed-1500, 86 %), 5.97 / 7.65 (mixed-800, 78 %). Everything else — `sim`,
  `b8_forward`, `observation`, `vector61_frp3`, `hero_v8`, `clone_digest` — sums to **1.3–2.7 s per 6×H300 batch**; scripted
  batches (no v8 opponents) are 0.64–0.69 s. The S0 ruler's 56–67 % v8 share (`research/speed_s0_baseline/RESULT_MAC_CONTENDED_01.md:26-33`)
  understated the long-body branch case: on the Y1 fixtures the opponent veto hook is *the* cost.
- **The S2 native v8 path was NOT running in Y1 or L1′** (03's static finding, verified and extended): `research/bl_served_search/`
  and `research/bl_mac_latency/` contain no `native_scope`/`enable` call (grep: only the Boolean `s2_native_v8` at
  `ruler.py:134,214`); `research/t34_speed_s2/native_v8.py:116,140` + docstring: "the wrapper runs the native path only inside
  `native_scope`; everywhere else it calls the original method". The ruler operator builds stock `Vector61SimdPolicy` worlds and
  calls `ceiling._step` directly (`ruler_operator.py:130-136,176-186`). So the 15–25× gap was measured against the **stock Python
  hook**. S2's own D3 receipt is 1.8994× whole-workload on S0 where v8 was 56–67 % (`research/t34_speed_s2/RESULT_D3_MAC_CONTENDED_01.md:3-12`)
  ⇒ implied ≈ 5–6× on the hook term alone (1 − s + s/x = 1/1.9 at s ≈ .6). Applied to frozen-1500: ≈ 3.3 + 2.1 ≈ **5.4 s** per
  batch — ~4× better than measured, still ~4× short of the W4 tail cap. This is arithmetic, not a measurement; the kernel-only
  form (no Python counter replay; `v8_decide` is `nogil`, `v8_nb.py:930,1393,1491,1655`) is unmeasured and may be much faster.
- **Y1 L4 failed on the upload cap again** (`exp/bl-l2/run-y1b.log`: `total upload time cap reached at upload …/ruler-fixtures/
  fixture-frozen-800.root.json.sha256`, EXIT=1) — the fifth mechanical Colab failure of the upload class after Z-9 was declared
  fixed (`CAMPAIGN_LOG.md:104`). **L2′ still has no rate receipt and no dispatchable transport.**
- **E3 recovery-04 is live** (`exp/bl-e3/run-04.log` updating at 14:40; `recovery-04/colab-output-04/usage-*.json`), on the
  Mac-verified action-tape replay. One slot occupied.
- `rollout_all` evaluates frozen/mixed branches **serially** (`research/t3_rollout_labels_20261015/bank_engine.py:525-560`; the
  batched engine refuses stateful opponents, `engine.py:99-104`) — 01/03 right; the six branches never share a kernel call today.
- B8's dueling `value` is the mean of all six Q's (`src/model/ego2s_network.py:70-71`) — 02 right; it is not a horizon-mass value.
- T3 students' −81.2…−88.0 are **vs CTRL**, scripted only (`docs/research/t3_screen_result_2026-10-15.md:9-13`) — 05 right.
- E4: 1,403 / 1,444 active vs 821 / 867 allocated hero transitions/s at 6 / 12 procs (`e4-ruler.json`) — 04/09 right: 100M ≈
  19.2 h active, 32 h at the allocated short-run rate, neither a complete-training time.
- Paired SDs 364.683 / 262.732 / 212.360 (`research/bl_ppo/E9_REPLACEMENT.md:132-133`) — 06 right; 128/mix ⇒ pooled SE 14.65.
- SC §4 amendment W ≤ 4 / K ≤ 8 is in force (`SUCCESS_CONTRACT.md:119-125`); every other timing bar unchanged — 08 right.

## (a) Bottom line

1. **No lever can produce a trustworthy served gain over bare B8 s3 this week, and none can by Tue 23:00.** Ten of ten say so
   and the receipts agree: the only mechanism that ever beat B8 (exact search, +92 pooled / +122.8 s3, scripted, offline) is
   15–25× too slow to serve on this Mac (`RESULT_L1.md:29-44`), its frozen/mixed offline value has **never been measured**
   (L2′ never ran), the GPU engine missed its bar by five orders of magnitude, Claim A distillation is KILLed, the PPO screen
   cannot reach power, O2 needs 5,097 worlds. Even a candidate in hand today needs ≈ 80 serial Mac-hours of L3/L4/strict/
   product/serving (`PLAN.md:153-159`). **Ruling: the week-1 release track is withdrawn. B8 s3 stays served. No strict family opens.**
2. **The week's best use is two measurements that make week 2 decisive, plus one bounded backbone bet** — all chosen so that the
   Saturday/Tuesday disposition is a *decision*, not another NOT_READY:
   **(M1) L2′ — the offline frozen/mixed ceiling of exact search over B8** (64/64/32 fresh paired worlds, unpaced, Colab, one look;
   FUND_SYSTEMS / PACKAGE_KILL / INCONCLUSIVE as registered). It is the mechanism test for the *entire* search family (served
   search, harvest, search-labelled students). A served controller is a lossy projection of this ceiling (misses, sparsity,
   fallbacks), so the ceiling must sit well above +40 for anything downstream to matter; L2′ is powered for exactly that
   (≈ 80 % at +80, 20–30 % at +40; `PLAN.md:10-13`). 01 and 07 say "no new L2′ chunks"; I overrule them: the reason they give
   (no served route) is why the number is needed *before* another week of engineering.
   **(M2) K-1 — exact native opponent veto in branch rollouts on this Mac.** 78–90 % of every frozen/mixed batch is one Python
   hook whose compiled, decision-identical kernel already exists and was never switched on in the timed path (facts above).
   This is the only speed lever that is identity-preserving (actions byte-identical; counters are "bookkeeping only; never feeds
   a decision", `native_v8.py:12-17`). It decides whether exact served search is alive for week 2 at all: **≥ 5× on `opponent_v8`
   with action identity on the nine fixtures** keeps it alive (frozen-1500 → ≤ 6 s, within reach of a sparse/anytime burst
   controller); less closes exact serving and leaves only approximations (02/09/10), which are a 3–6-week program.
   **(P1) one fixed-endpoint 100M PPO run from B8** (F recipe, E8 critic, seed frozen, Y3 durability first, fresh 128/mix paired
   Colab evaluation vs B8 at the end) as a prospective *research registration*, not an E2 screen pass. Honest probability it beats
   B8 at bar ≈ 10 % (04's own number; T23 −80.8 at 8M binds); it costs no Mac time until a nominee exists (bare strict is
   unpaced, ≈ 7–10 h, `round-07 §a1`), it is the owner's requested use of credits, and it is the only lane that can nominate a
   *bare* artifact, whose serving qualification is simplest (08). One endpoint, not three (04): slots are needed for L2′, and
   three seeds of one recipe are correlated bets (04 concedes).
3. **Sparse/anytime burst (01, 08) is the right served *form* and is funded as a build-only package conditional on K-1.** Both
   advisors derive the same thing from the receipts: the present dense controller can never fit (27 s / 41.7 ms ≈ 650 tick
   budgets, 08), a future-root certificate delivered inside the 9–18 s lead is the only shape that can, and it is a **new
   identity** with no inherited T3 evidence and new bars (first valid H300 certificate ≤ 5 s on ≥ 50 % of dense-rescuable
   frozen/mixed opportunities, zero overruns, update p95 ≤ 11.688 ms). Build Thu–Fri only if K-1 passes; one 2-Mac-h fixture
   measurement Sat; no paced qualification this week.
4. **Not funded this week (ruled, with reasons):** learned-tail critic (02) — defer behind L2′ + K-1 (it is an approximation
   identity; its speed model ignores that the cost is per-opponent-frame, so truncation buys proportionally but the H300
   certificate is the thing being sold); bigger CNN student from search labels (05) — labels' frozen/mixed value is exactly
   L2′'s unknown; O2 at power (06/07/10 agree: no consumer); Round 0 re-registration (no consumer); E1b retry (frozen STOP);
   opponent-proxy fidelity test (10) — a good week-2 item if K-1 fails; A100/H100 — nothing is GPU-bound (unanimous, 16th time).
5. **"Stop and keep B8" criterion (explicit; PLAN_v2 §6):** KEEP_B8 is declared on the earlier of (i) Fri 23:59 — L2′ prints
   PACKAGE_KILL (pooled UB90 < 40), or (ii) Tue Oct 27 23:00 — no artifact has printed, on a fresh registered unpaced bank,
   pooled point ≥ 40 ∧ LB90 > 0 vs B8 s3 **and** K-1 < 5×. Under KEEP_B8: B8 stays served, no serving/strict machinery is built,
   Colab drops to ≤ 1 slot of research (E3 → E14 → E16 line) unless the owner re-opts into 07's 42–56-day program; any new release
   ambition needs a new charter. Partial outcomes are enumerated in PLAN_v2 §6.
6. **What the owner can trust at Sat 16:00 PT:** no served change; B8 s3 unchanged; three measured facts — (a) whether exact
   search has a T3-sized offline gain on frozen/mixed worlds (L2′ print, Fri night at the earliest; Sat if a chunk slips),
   (b) whether the opponent-veto cost can be removed exactly on this Mac (K-1), (c) P1 state (durability + 1M qualification
   receipts, 100M in flight) — plus E3's print if it lands. Nothing reaches strict, product or serving; no family is opened.
7. **Process rulings that bind the plan:** every Colab manifest needs a *real-transport* upload receipt (the Z-9 fix did not
   cover fixture directories — fifth failure); the Mac quiet-host protocol is "log and retain, never kill daemons, schedule at
   night" (03); a Mac Y1 rate may be used ×2 as a *dispatch forecast* for L2′ if Y1 L4 cannot land by Thu 12:00 (a sizing input,
   not evidence — the one-way "never copy Colab timing onto the Mac" rule is untouched); L2′ is re-chunked prospectively so that
   `2 × estimate ≤ timeout ≤ 12 h` per chunk (8 × 20-world chunks, 8/8/4, covering 160 exactly) and repriced to ≤ 200 u incl.
   2× liability and 20 % reserve — label map, one look, bank size and analyzer unchanged.

## (b) Per-advisor verdicts

| ID | Persona | Verdict | Why (verified facts in **bold**) |
|---|---|---|---|
| 01 | Anytime / search efficiency | **keep (conditional)** | **`rollout_all` serial for frozen/mixed; batched engine stateless-only (`bank_engine.py:525-560`, `engine.py:99-104`); Path B waits for all rows (`batch_v2.py:152-159`)** verified. Future-root targeting, resumable single-action continuations, first-certificate publication, "a shallow survivor is not a rescue" — adopted as WP-4's spec and the Sat fixture measurement. Its "no new L2′" is overruled (§c3). Its arithmetic (one H300 branch ≈ 3.3–4.5 s) is pre-K-1 and must be redone after it. |
| 02 | Learned-value truncation | **partly** | **`value` = mean over all six Q (`ego2s_network.py:70-71`)**; D0 non-transfer argument (action-relative losses vs a separate critic) is sound. But its speed model assumes a 5 % fixed overhead; the receipt shows the cost is per-opponent-frame (88 % in one hook), so the first lever is exact (K-1), not approximate. Deferred to week 2 behind L2′ + K-1; its P2 fidelity bars (≥ 90 % recall, ≤ 5 % harmful overrides, ≥ 80 % of O1 gain) are kept as the registered form if it is ever admitted. |
| 03 | Apple-silicon perf | **keep — the round's catch** | **No `native_scope` anywhere in `bl_served_search`/`bl_mac_latency`; `ruler.py:134,214` Boolean only; `native_v8.py:116,140` opt-in; `ruler_operator.py:130-136,176-186` stock policies; frozen-1500 veto 19.177 / 21.264 s** — all verified, and extended to all six frozen/mixed fixtures (78–90 %). Attest-then-integrate ordering, byte-identity proof list, "Metal is 3–5 builder-days, not this week", quiet-host rules — adopted. Its kill bar (< 1.5× ⇒ not a major lever) is replaced by the lever's own need (≥ 5× on `opponent_v8`). |
| 04 | Large-scale PPO | **partly** | **E4 1,403/1,444 active vs 821/867 allocated (`e4-ruler.json`)**; E2 blocked in code (`manifests.py:318-327`); 1 % departure arithmetic consistent with Round 4. Fund one endpoint (not three), as a prospective fixed-endpoint registration with Y3 + 1M qualification first; its nomination filter (band v2, boost, scripted no-death ≥ 0, pooled ≥ 20, UB90 ≥ 40, rank by LB90) adopted for the single endpoint; its ≈ 10 % at-bar estimate is the number the owner should hear. |
| 05 | Distillation / bigger student | **partly** | **75/301,589 decisions changed; −81…−88 vs CTRL not B8 s3 (`t3_screen_result:9-13`); Claim A D0 225 rows from 21 worlds** verified. Right that capacity was never isolated and that growth loss, not label fit, killed T3 students. Not funded this week: the labels' frozen/mixed value is L2′'s unknown and scripted-only labels buy +40.9 pooled at best (Round 6 §a6). Its 48-h design is the week-2 form if L2′ prints FUND. |
| 06 | Statistics / sequential design | **keep (as sizing)** | **SDs 364.683/262.732/212.360 (`E9_REPLACEMENT.md:132-133`)**; 385 worlds for 80 % at +40 one-sided .025; point ≥ 40 passes ≈ 50 % at true +40 regardless of N (consistent with Round 10 §a3). Adopted: P1's endpoint evaluation is 128/mix paired fresh (SE 14.65, ≈ 86 % for > 0 at +40), one look, Holm not needed (one artifact). Its 1,024/512/512 E2 redesign is for arms that do not exist; held for week 2. O2 deferral — adopted. |
| 07 | Red team: keep B8 | **keep (as the criterion)** | **Amdahl on S0 shares (2.5–3.7×)** correct for S0, superseded by the Y1 split (88 %, so the exact ceiling is ≈ 8–10× before the remainder is touched). Its "falsifiable 72-h challenge" is exactly M1 + K-1; its 42–56-day program is the fallback under KEEP_B8; its view-changing signal (pooled ≥ +80, LB90 > 0, frozen & mixed > 0) is adopted as the L2′ interpretation of "FUND means engineering is justified". Its "close today, no L2′" is overruled (§c3). |
| 08 | Product / serving | **keep** | **SC §4 W ≤ 4 / K ≤ 8 amendment (`SUCCESS_CONTRACT.md:119-125`)**; 27.2 s ≈ 652 tick budgets; validator checks checkpoint/veto only (`served_checkpoint.py:172-217`); rollback must disable assistance and drain. Burst identity (W4/K4, future root ≥ 144 frames ahead, three Q-ordered H300 alternatives, p95 first-result ≤ 5 s) adopted as WP-4's nominated cell; its "new bare checkpoint is the simplest serving path" is why P1 is funded. Two PPO endpoints → one (slots). |
| 09 | Colab throughput | **keep** | **E4 arithmetic; 44 kB/s and the 300-s staging cap; `supervisor.py:187-202` 2× rule** verified; the fifth upload failure (Y1 L4) confirms its admission checks. Complete-job PPO ruler folded into P1's 1M qualification; "no paid idle, release slots" adopted; 173-u portfolio ceiling replaced by PLAN_v2 §4's (≈ 330 u ceiling, ≈ 180 u expected). |
| 10 | slither.io strategy | **keep (week 2)** | **Hazard is late crowding, not length (`panel_mac_answers_2026-10-14.md:95-105`); O1 35/48 single-action H500 branches (`decision.json`)** verified. "Preservation (+30) > harvesting (+10)" is the right prior; proxy-opponent distortion test (≥ 80 % retained exact-positive advantage, ≤ 5 % false survival) is the registered form if K-1 fails and approximations are tried. Not this week. |

Factual errors that change a conclusion: none. One advisor-wide blind spot: nine of ten priced speed-ups off the S0 shares
(56–67 %) rather than the Y1 branch receipt (78–90 %); only 03 opened the receipt. Judge's corrections to Round 10/PLAN: L1′
was measured with the stock hook (PLAN §3 step 2 assumed S2 on); Z-9 is not done for fixture-carrying manifests.

## (c) Disagreements ruled

1. **Exact speed-up first (03) vs approximations first (01/02/09/10).** 03. The dominant term is one Python hook with an
   existing decision-identical compiled form; removing it changes no action and needs no fidelity study. Approximations each
   create a new identity and need their own regret/false-survival test; they come after K-1 and only if L2′ says the mechanism
   is worth it. Order: K-1 → (sparse burst) → approximations.
2. **PPO: three endpoints (04) / two (08) / one (09) / none (07, 10).** One. A fixed-endpoint research registration is lawful
   (it is not an E2 PASS); Y3 + 1M qualification precede the tranche; a second seed only on idle slots after L2′ is terminal.
3. **L2′ now (judge) vs never (01, 07).** Judge. The served form failed, but the mechanism's frozen/mixed value is the one
   number that decides whether *any* search-family work continues; it costs zero Mac hours and ≈ 60–100 u on slots that would
   otherwise idle. Re-chunk and reprice prospectively; bank, bars and one-look rule unchanged.
4. **"5-s first result" (01/08) vs the 1.458-s cap (PLAN).** Both: the cap belongs to the dense identity that failed; a sparse
   future-root identity gets its own bars, registered before its first timing, and inherits nothing. WP-4 is build + fixture
   measurement only.
5. **Release this week.** Unanimous no. Ruled: withdrawn; signature text B (PLAN §7) is the only one that can appear.
6. **A100/H100.** Unanimous no; nothing on the table is GPU-bound (P1 learner is B8-sized; search/collection is host-bound).
7. **O2 / Round 0 / distillation / E1b.** Deferred or closed as in §a4; each reopens only with a named consumer and the
   prerequisite print (L2′ FUND for harvest and labels).

## (d) Running shortlist (carried from Round 10, re-founded)

| # | Lever | Conf. push / beats B8 at bar | Cost to first signal | Most informative next experiment |
|---|---|---|---|---|
| 1 | **Exact-sim search over B8 — mechanism** (offline, unpaced) | 35 % that the frozen/mixed ceiling is ≥ +40 pooled; 15 % ≥ +80 | L2′ ≤ 200 u envelope (≈ 60–100 u expected), 0 Mac-h | **L2′ (M1)** — the gate for 1a/1b/1d/5 |
| 1a | Served sparse-burst identity on this Mac (01/08 form) after exact native veto | 12 % / 6 % (down 10/6: 15–25× gap measured on the stock hook; identity-preserving fix unmeasured) | K-1 0 u + 1–2 Mac-h; WP-4 build 1–2 Sol-d; week-2 L3/L4/strict ≈ 80 Mac-h | **K-1 (M2)**: ≥ 5× on `opponent_v8`, action-identical |
| 1b | Approximate search (learned tail 02 / proxy opponents 10 / H96 top-2 09) | 8 % / 4 % | 12–36 u each + Mac timing | only if L2′ FUND ∧ K-1 < 5×: 10's fidelity test first |
| 1d | Harvest-aware operator (O1 GO) | 3 % add-on | O2 ≥ 150 u at power | after L2′ FUND and a served identity |
| 2 | Representation: geometry residual (E3 → E14 → E16) | 20 % / 10 % (down 2/2: week 2+ by construction) | E3 running (16 u) → E14 3 Mac-h → E16 40 u | E3 print → E14 bare cells (Fri) |
| 3 | **Long-horizon PPO from B8 — fixed 100M endpoint (P1)** | 20 % / 10 % (up 5/3: funded as research, bare artifact, unpaced strict) | Y3 8 u + 1M qual 4 u + 100M ≤ 120 u + eval ≤ 10 u | **P1 endpoint + 128/mix paired eval** (Sun–Mon) |
| 4 | GPU-resident exact engine | **0 % this campaign** (E1b 0.217 steps/s vs 20k bar; frozen STOP) | — | none |
| 5 | Search-in-the-loop / search-labelled student (05) | 8 % / 4 % | 96 u week-2 design | only if L2′ FUND |
| 6–8 | Restart archive; opponent add-on; world model / league | 4 / 3 / 1 % | — | not funded |

## (e) Still unresolved

- **L4 core speed relative to this Mac for the ruler batches** (sets L2′ chunk count/timeouts); Y1 L4 has no transport.
- **Kernel-only native veto: action identity and speed** (K-1) — the single number that decides 1a vs 1b for week 2.
- **E3's print** (GO with architecture / GO-null / KILL) — gates E14 Friday and E16 next week.
- **P1's complete-training rate** (updates + probes, not E4's collection rate) — decides whether the 100M endpoint lands by Sat/Sun.
- **Colab upload class** — five failures; fixture directories need runtime reconstruction or a ≤ 6 MB bundle with a real-transport receipt.
- **Owner's appetite** for 07's 42–56-day program under KEEP_B8 (asked at Sat 16:00 with the three prints, not before).
- **Mac quiet-host protocol** — analyticsd/mediaanalysisd; night windows and retained stopped attempts, no daemon killing.

## (f) Experiments Claude can run now (dispatch via Sol; report in LIVE_FACTS.md) — detail and dates in PLAN_v2

**K-1 — exact native opponent veto in branch rollouts (0 u; Sol-K 0.5–1 d; 1–2 Mac-h Thu night).** (a) enter `native_scope` for
frozen/mixed branch rows in `rollout_all` and in the ruler operator, with native/fallback row counts attested; (b) a kernel-only
mode (no counter replay) audited action-identical against the stock hook on the nine Y1 fixtures × 30 batches + 24 fresh long-body
roots; (c) optional: pack the six branches' opponent rows into one kernel call. Same receipt schema as Y1 (`stage_seconds`). Kill:
any action mismatch ⇒ that mode INVALID; `opponent_v8` speed-up < 5× on frozen-1500 ⇒ exact served search closed for week 2 (1a → 1b).

**Y1-L4 ruler (≤ 4 u; after the fixture-transport fix; Thu AM).** As registered (`RULER_FIXTURES_08.md`); semantic match to the
Mac reference or INVALID; aggregate R per mix. If not landed by Thu 12:00: dispatch L2′ on the Mac rate × 2 as forecast (ruling §a7).

**L2′ — M1 (≤ 200 u envelope incl. 2× liability + 20 % reserve; ≈ 60–100 u expected; 2–3 slots; Thu 12:00 → Fri night).** Bank
`BANK_L2.json` unchanged (64/64/32); re-chunked 8 × 20 worlds (8/8/4), each `2 × estimate ≤ timeout ≤ 12 h`; per-assignment
parity inside the envelope; pinned analyzer merge after all chunks terminal; labels FUND_SYSTEMS / PACKAGE_KILL / INCONCLUSIVE via
the versioned map; one look. Kill: any chunk fault ⇒ INCONCLUSIVE (never partial); PACKAGE_KILL ⇒ KEEP_B8 for the search family.

**P1 — fixed-endpoint 100M PPO (Y3 ≤ 8 u → 1M qualification ≤ 4 u → 100M ≤ 120 u → eval ≤ 10 u; slot A after E3).** F recipe,
E8 critic (held-out late-body EV ≥ .1 gate), seed/recipe/endpoint frozen before dispatch, no checkpoint selection, E8 tripwires;
Y3 exact continuation on a replacement runtime before the tranche; evaluation = 128/mix fresh paired H5000 vs B8 s3 on Colab
(development evidence; E13 parity note). Kill: critic gate fail / tripwire / Y3 fail ⇒ lane NOT_READY; eval UB90 < 40 ⇒
ARTIFACT_REJECTED_AT_BAR; point ≥ 40 ∧ LB90 > 0 ∧ guards ⇒ Mac strict (unpaced, X5-priced) registered for week 2.

**X5 — bare strict timing receipt (0 u; 2 unpaced Mac-h; Wed night).** T7 harness, 48 disjoint dev worlds, B8 s3 ⇒ episodes/h
(prices P1's strict and any bare nominee).

**E14 bare cells (0 u; 3 Mac-h; Fri, only if E3 prints GO with an architecture).** As PLAN §3 step 4.

**WP-4 fixture measurement (0 u; 2 Mac-h; Sat AM; only if K-1 passes).** Constructed long-body fixtures 3 mixes × L800/1200/1500
× 10 reps, cold start, declared contention; first valid H300 certificate time, publication deadline hits, overruns, update p95.

Not funded this inter-round: O2, Round 0 re-registration, E1b, Claim A/T3-style distillation, E2 screen, A100/H100.
