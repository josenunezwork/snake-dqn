# Round 13 judgment — GPU-trained components to make the Mac search cheap: is it the lever, and what is the first measurement?

Judge: Fable (round-13 judge), 2026-10-21 **~22:50 PT** (wall clock `date` = 22:49 PDT at start). Inputs: `../CHARTER.md`,
`../round-01..12/JUDGMENT.md` (round-11 and round-12 in full; earlier rounds' bottom lines), `../PLAN_v2.md`, `../LIVE_FACTS.md`
(newest 21:00 PT), `../QUEUE.md`, `QUESTION.md`, `personas.txt`, `astra_01..10.md` (all read in full). Repo read-only at
`6e3c3245` (main HEAD, clean; the advisors' snapshot). Receipts under `../exp/` opened with light commands; one zero-unit
computation on `exp/bl-l1-20261021/y1-mac-01/mac-ruler.json` is mine (§0). Sealed study not opened. No jobs, Colab or tmux.
Owner's override for this round: **no round-14 inputs**; this file answers the owner's question and applies the round-12 process
rules (premise register, canary, execution attestation) to whatever it admits.

## 0. Facts verified tonight (newer or sharper than the question)

- **Every Y1 fixture has exactly four legal actions, and bare survives H300 in 30/30 batches on eight of the nine** (judge's
  extraction from `y1-mac-01/mac-ruler.json`, `records[].batches[].semantic`): frozen-1200/1500, mixed-800/1200/1500 and all three
  scripted fixtures choose bare 30/30 (`changed = 0`); **frozen-800 is the only "doom" fixture — bare dies 30/30 and the chosen action
  changes 30/30**. Median branch frames per batch are 1,200 (four full H300 branches) except mixed-1500 (939), scripted-1500 (1,015)
  and frozen-800 (968). Astra 05's "four legal actions, eight of nine choose bare" is exact.
- **The T3 ranking makes a bare-first exact stop decision-identical.** `_choose` returns `bare` whenever bare is among the best-scored
  outcomes (`research/bl_served_search/ceiling.py:139-145`); a surviving branch scores `DEPTH`, the maximum (`:128-137`). So if the bare
  branch survives H300, no alternative can change the action; if it dies, alternatives can be evaluated in descending Q with index ties
  (the same tie rule `_choose` uses) and stopped at the first H300 survivor. On the T3-dense distribution bare died in **440 of 36,996
  batches (1.19 %)** (`round-06/JUDGMENT.md:33`, from the dense receipts), so **98.8 % of dense batches would cost one branch instead of
  four to six** — a 0-u, training-free, exact saving that no advisor priced end-to-end (05 derived it; 07/09 adopted ordering; none
  computed the Y1 numbers). It helps the *common* root ≈ 4× and the *doom* root (bare dies — the only root where a rescue is possible)
  only through Q-ordered first-survivor stopping (≈ 2× expected, 1× when all die). The L2′ analyzer consumes no per-branch outcomes
  (`grep outcomes|frames_executed|survives research/bl_served_search/analyze.py` → none), so an L2′ amendment would change diagnostics
  footprints, not the estimand; `rollout_all` evaluates the branches serially (`bank_engine.py:541-569`), so the stop is one loop break.
- **K-1(a) is already in the branch path on main**: `rollout_all` enters `research.bl_k1.native.branch_scope(world.mix)`
  (`bank_engine.py:538-541`); the ruler operator still defaults `native_mode="stock"` (`ruler_operator.py:115-118`). K-1 identity:
  `ACTION_IDENTITY_PASS`, 294 completed, `timing_qualification: NOT_RUN` (`exp/bl-k1/identity.json`); timing `PREPARED_NOT_TIMED`,
  window hard-coded to **Thu 2026-10-22 20:00–22:00 PT** (`research/bl_k1/timing.py:40-43`), bar = warm-29 median ratio ≥ 5.0 on
  `opponent_v8` frozen-1500, kernel mode (`:22-37`); `wp4: BLOCKED_PENDING_MAC_TIMING_BAR`, six-branch packing `NOT_IMPLEMENTED`
  (`exp/bl-k1/handoff.json`). The K-1 adapter exports **row-level native/stock counters** (`native.py:27-34, 105-111`) — the
  execution-attestation pattern round 12 §4.3 requires.
- **The 294 K-1 cases are nine fixtures × 30 repeats + 24 constructed roots, "identity evidence, not efficacy"**
  (`research/bl_k1/README.md:29-31`). The question's proposed criterion — "kill if < ~95 % search decisions unchanged on the 294 K-1
  cases" — is **vacuous**: an always-bare controller agrees on 99.975 % of T3-dense decisions and 99.80 % of its searched batches
  (`t3_ceiling_test_2026-10-15.md:122`: 75 changed of 301,589 / 36,996) while discarding every rescue; on the Y1 fixtures it agrees
  on 8/9 fixtures × 30/30. All ten advisors reject it; so do I.
- **Residual floor**: frozen-1500 median whole batch 21.26 s, `opponent_v8` 19.18 s, `vector61_frp3` 0.43 s ⇒ **residual 2.09 s with
  veto only removed, 1.65 s with both opponent stages removed** (judge's recompute; 04/05/06/08/09/10 consistent). The dense W4 tail
  cap is **1.458 s** (`research/bl_mac_latency/RESULT_L1.md:44`: W2 0.830 / W3 1.250 / W4 1.458). So a *free* opponent model still
  misses the dense cap on a four-branch batch; only shortening the branches (fewer branches or fewer frames) can meet it.
- Static outcome predictors on this state: T10 D300 AUC 0.726–0.769 by input (`t10_certificate_horizon_2026-10-17.md:29-38`); T35
  frozen-trunk probe AUC 0.747, agreement 46.2 %, KILL (`research/t35_outcome_head/RESULT.md:27-31`, data-starved caveat at `:60-66`).
  B8's dueling `value` is a mean-of-Q baseline, not a horizon return (`src/model/ego2s_network.py:61-71`).
- **P1 is in recovery, not training**: `exp/bl-p1/y3/source-pausefix-03/receipt.json` and `qual-1m/prepare-pausefix-04/receipt.json`
  both `PREPARED_NOT_DISPATCHED`, `dispatch_ready: false` (10 right); **X5 NOT_READY, timing UNAVAILABLE** (`x5-disposition.json`);
  no `exp/mac-lease.json` exists yet (round-12 §4.7 not implemented). P1's own invoice ⇒ nominal complete-job ≤ 24 h
  (`REGISTRATION_P1.md:106-111`). Serving clause W ≤ 4 / K ≤ 8 and the probe-only grant stand (`SUCCESS_CONTRACT.md:119-125`).
- Arithmetic in the ten answers checked and correct: 0.999^1500 ≈ 22 %, 0.9999^1500 ≈ 86 %, 0.99^300 ≈ 5 %, 0.99^1500 ≈ 2.8e-7
  (illustrative, all flagged as such); 12,806 params for 128→64→64→6 (04); 12,443 bytes/leaf (08); Wilson LB 92.2 % for 280/294 (07);
  5× hook ⇒ 2.66–3.57× batch at p = .78–.90 (09/10); P1 142 u (10). Factual errors that change a conclusion: **none**.

## 1. Bottom line (for the owner)

1. **"GPU-train approximations so the Mac search gets cheap" is a legitimate research line but it is not the lever, and it is not
   this week's work.** The lever is unchanged: exact-sim search's *value* on frozen/mixed worlds (L2′, still unmeasured) and its
   *exact* speed on this Mac (K-1, Thu night). Learned components only matter if both of those come out alive, and the cheapest speed
   win left is not learned at all (point 3).
2. **Go / no-go: GO as a bounded, pre-registered research line for ONE component — a horizon-conditioned tail critic after a short
   exact prefix (H64) — with the first measurement costing 0 units; NO-GO on the opponent-model-first pilot as proposed, NO-GO on a
   pruning prior, NO-GO on any approximation as a serving route before L2′ prints, and NO-GO on "95 % decisions unchanged on 294
   K-1 cases" as an admission criterion (vacuous, §0).** Why the critic and not the opponent model: an opponent model attacks the
   78–90 % hook term that the exact kernel already attacks (it must beat the kernel, not the Python hook), cannot pass the 1.65–2.09 s
   residual floor on a full batch, and compounds up to 1,500 approximate decisions per branch where one wrong turn changes a closure;
   a tail critic shortens *every* per-frame cost ≈ 4.7× (300 → 64) with one approximation at the boundary, and keeps opponents exact
   in the prefix. Why not a pruning prior: the exact bare-first / Q-ordered first-survivor stop already removes 3 of 4 branches on
   98.8 % of dense roots at zero decision change; a learned order can only reorder ≤ 3 alternatives on the 1.2 % doom roots (≤ 2×)
   and hard pruning can delete the sole escape.
3. **Cheapest decisive first measurement = V0, a 0-unit Mac latency ladder on the K-1 fixtures, after K-1, under the Mac lease
   (§5).** Four exact/oracle cells on the same operator: kernel H300 all-branches (= K-1), kernel + exact stop, kernel + H64 prefix with
   an oracle leaf (cached exact H300 outcome, plus a fixed-shape dummy leaf call so inference overhead is counted), kernel + H64 +
   stop. Cap: 0 u, ≤ 1 leased Mac-hour, Sol-K 0.5–1 day + Codex-V 0.25 day. Kill: any chosen-action mismatch in a stop cell ⇒ INVALID;
   H64-oracle p99 > 1.458 s on frozen-1500 or frozen-800 ⇒ **the learned-cutoff route is closed without training anything**; H64-oracle
   < 2× faster than the best exact cell on the doom fixture (frozen-800) ⇒ no consumer for a critic, route closed. Only a PASS admits
   the first GPU fit (V1, ≤ 20 u, pre-specified in §5) — and V1 is research, not a controller.
4. **How a non-exact searcher is ever trusted:** as a **new controller identity** with no inherited T3/K-1 evidence — world-split
   fidelity bars (rescue recall ≥ 90 %, harmful accepted overrides ≤ 5 %, false-survival UB95 < 5 % with ≥ 80 predicted-surviving
   selected branches per mix), then fresh paired closed-loop H5000 outcomes on Colab (development), then the Mac strict gate (paced
   candidate), Watch product test, serving qualification (three oracles), owner release OK. Never action agreement, never a PASS string.
   Honest probabilities: critic passes its fidelity bar ≈ 20–25 % (T10/T35 say the information may not be in the state); this line
   yields a served, qualified gain over B8 by mid-November ≈ 5–8 %.
5. **Sequencing this week is unchanged and the new line touches none of it:** Y3 fix Thu AM; **K-1 Thu 20:00–22:00** (it is V0's
   baseline — nothing from this round enters that window); X5 Fri 09–11; native ruler Fri PM; L2′ placement ruling Sat 12:00; Sat 16:00
   handoff; Tue 23:00 KEEP_B8 reducer applied literally. V0 runs in the vacated WP-4 Sat 08:00–10:00 slot (or Fri 22:00–23:00) under the
   lease; V1 (if admitted) dispatches no earlier than Sat–Sun on an idle slot after the canary gate and supervisor requalification.
   **KEEP_B8 dates are untouched**: Fri 23:59 cannot fire (no L2′ print exists); Tue 23:00 fires iff no fresh-bank print at bar **and**
   measured K-1 < 5×. Under KEEP_B8, PLAN_v2 §6 closes every approximation lane including this one; it reopens only by the owner's
   §7 choice. Under K-1 ≥ 5× with no print, Tuesday is NOT_READY, nothing is funded for serving, and V0/V1 remain lawful research.
6. **One exact, zero-unit saving should be entered now where it changes a decision**: Sol-S adds the bare-first / Q-ordered stop as
   a mode of the ruler operator and Fri's native whole-workload rulers measure it (branches-evaluated counters attested); the Sat 12:00
   L2′ placement ruling then sees a Mac forecast that may fall under the 60-wall-h Mac-placement clause — a prospective amendment
   (chosen action identical; unevaluated branches recorded NOT_EVALUATED), reviewed by Codex-V before any bank is touched.
7. **The reverse direction the owner asked about (CPU sims generate data, GPUs fit) is already the pattern** of P1 and of every label
   job; nothing new is needed there, and nothing on the table is GPU-bound — A100/H100 remain unjustified (unanimous, 18th time).
8. B8 s3 stays served; no release question exists; the Sat handoff gains one line: "V0 (learned-cutoff feasibility) PASS/CLOSED/NOT_RUN".

## 2. Answers to the question's five parts

**(i) Is it the right lever for a served gain over B8 on this Mac?** No — it is a *speed add-on* to a mechanism whose value on the
worlds that matter (frozen/mixed) is unmeasured, and whose exact speed is being measured tomorrow. The order that makes any of it
decision-bearing is **L2′ value → K-1 exact speed → exact stopping → one learned approximation**, and the first two are already
scheduled. A learned component cannot rescue the calendar (10): nothing it produces reaches strict, product or serving this week.

**(ii) Which component first (or none)?** The **tail critic** (01, 02, 04, 08, 10), after an **exact stop** that is not learned
(05, 07, 09 ordering adopted in exact form). The opponent model (03, 06 would test it first) is ranked second-to-last: it has a mechanism
only if K-1 *fails* (the kernel cannot remove v8's feature computation but a clone could skip it entirely) — and that branch is closed
by PLAN_v2 §6 unless the owner re-opts. A pruning prior is last: its room is already taken by the exact rule.

**(iii) Cheapest decisive first measurement:** V0 (§5). It is decisive in the direction that matters first — it can close the route
for 0 u before a single label is generated — and it supplies the premise receipt (round-12 §4.2) that any fit would otherwise lack.

**(iv) Failure modes and how they are handled:** compounding error over long rollouts → the critic is applied once at H64 and opponents
stay exact in the prefix; distribution shift when the hero's behaviour changes → labels are collected on search-induced roots (both
ordinary and doom roots, all legal first actions), world-split, with one fixed DAgger-style relabel of critic-visited states registered
before fitting, and a closed-loop V2 before any controller; Claim-A-style failure → the critic is a separate net, B8 untouched, but
T10/T35 say the *information* may be missing from the state at D236 — hence the ≈ 20–25 % prior and the fidelity bars; identity/trust →
new identity, §1.4. The dense cap itself may still be unreachable (V0 decides); if only the sparse-burst form fits, that form needs its
own bars (round 11 §c4) and V0's cells are re-read against ≤ 5 s first-certificate, registered before the run.

**(v) Sequencing with K-1, P1, KEEP_B8:** §1.5 and §6. K-1 untouched; P1 untouched (Sol-R stays on Y3/1M; V1 uses no P1 slot); V0 is a
0-u Mac hour after K-1; V1 is conditional, ≤ 20 u, slot-idle only; KEEP_B8 text and dates unchanged.

## 3. Per-advisor verdicts

| ID | Persona | Verdict | Why (verified facts in **bold**) |
|---|---|---|---|
| 01 | Model-based RL | **keep** | Prefix-exact / suffix-approximate ordering, "closed-loop root-choice regret matters more than trajectory identity", **270 fixture batches + 24 fresh roots** in K-1 — right. Its P1 dummy-leaf latency probe is V0's core. Its 3–5 builder-day estimate is the honest calendar. |
| 02 | Value-truncation | **keep** | Best target spec of the round (`V_L`, `p_survive`, mass separate; **ranking at `ceiling.py:128-146`**; fatal-step convention `:134-137` verified; collector lacks leaf states, **`bank_engine.py:517-569`**). V0 → V1 → V2 ladder and bars adopted nearly verbatim (§5). Its "H300 prefix supplies H300 targets from one H500 continuation" is the right label economy. |
| 03 | Opponent cloning | **keep (as the reason not to start there)** | Right that scripted opponents must stay exact (`random_safe` is a hash), that a clone must predict the *final vetoed* action without recomputing v8 features, and that "99 % on ordinary states" is plausible while 99.99 % on crowded long-body branch states is not. Its ≤ 10-u pilot is deferred behind the critic: even a free clone cannot pass the residual floor on a four-branch batch. |
| 04 | Apple-silicon inference | **keep — the round's engineering catch** | **Residual 2.087 s / 5.923 s recomputed from paired samples**; MPS slab copies (`t33_batched_teacher/inference.py:136,161`); batching must be explicit (`batch_v2.py:50,139` keeps B8 rowwise); "an opponent proxy runs 300× per branch, a leaf model once". Its "test latency before training" is V0; its dummy-MLP batch-1/6/30 cells are folded into V0's oracle cells. |
| 05 | Search efficiency | **keep — the round's decisive finding** | **Four legal actions in every Y1 fixture; eight of nine choose bare 30/30** (verified tonight, §0); bare-first / Q-ordered exact stopping is decision-identical under `_choose`; "top-two buys ≈ 2× at four branches, not 3×". Adopted as the exact stop (§1.3, §6). Its learned-scheduler-first preference is overruled only because the exact rule exhausts most of the scheduler's room. |
| 06 | Distillation failure analyst | **partly** | Right that D0/T3 killed specific projections, not learning per se; **T3 LABELS 99.66 % bank agreement, −87.97 vs CTRL, 55.33/75.15 of the loss on common-alive frames** verified; table of what each component avoids/inherits is the clearest in the round. Overruled on *opponent surrogate first*: the residual floor and the kernel baseline make it a second-order lever; its closed-loop falsification design is kept for whichever component is tested. |
| 07 | Trust statistician | **keep** | **Wilson LB 92.2 % for 280/294**; false-survival definition and denominators; ≥ 80 predicted-surviving selected branches/mix for a Bonferroni UB95 < 5 %; "qualify as a new controller, not a nearly exact implementation"; 128/mix power table (SE 14.65, 92.6 % at +40 for LB90 > 0; point ≥ 40 ≈ 50 %) consistent with rounds 10–12. Adopted as the trust doctrine (§1.4). |
| 08 | Colab ML engineer | **keep** | H50/H100 oracle replay as the first measurement (= V0), canary that exercises the *real* pipeline (transport bytes, CUDA fit, resume, export, Mac reload) — exactly round-12 §4.1; **12,443 bytes/leaf**, 0.43 runtime-h label analogy flagged as an analogy. 10-u L4 envelope adopted for the fit portion of V1. |
| 09 | Red team | **keep (as the standing bar)** | **99.80 % / 99.975 % always-bare agreement; 294 = 9×30+24; residuals 2.013 / 1.662 s** verified. Its one view-changing measurement (fresh paired H5000 ≥ +40, LB90 > 0, controller acting under real Mac deadlines against exact opponents, guards satisfied) is the only thing that will ever count; its ordering-prior-with-exact-verification "least-bad first component" is the exact stop in §1.3. |
| 10 | Campaign manager | **keep** | **P1 holds `dispatch_ready: false`** in both pausefix receipts (verified); "GPU components cannot rescue this week's calendar"; Friday/Tuesday KEEP_B8 literal; "do not roll a failed week's deadline into a new proxy experiment". Its 96-root pre-fit diagnostic (removable time + consequential rank reversals, 2 Mac-h) is subsumed by V0 on the existing fixtures at ≤ 1 Mac-h. |

Shared blind spot: nobody computed the Y1 bare-survival census, so nine of ten priced the learned components against a four-branch
batch that an exact rule already cuts to one on the common root. Shared correct call: all ten reject the 95 %/294 criterion, all ten
protect K-1 and P1, all ten refuse A100/H100.

## 4. Disagreements ruled

1. **Tail critic first (01, 02, 04, 08, 10) vs opponent surrogate first (03, 06) vs ordering prior first (05, 07, 09).** Ruling:
   *exact* ordering/stopping first (0 u, no identity change), then the tail critic as the first *learned* component; opponent surrogate
   only on the K-1-FAIL branch and only by owner re-opt. The deciding facts: the residual floor (a free veto still misses 1.458 s on a
   full batch) and the Y1 census (the exact stop takes the scheduler's room).
2. **Fit now (03: ≤ 10 u diagnostic) vs wait for L2′ FUND (02, 04, 08, 09, 10) vs wait for K-1 + latency forecast (01, 06).** Ruling:
   V0 (0 u) needs only K-1; V1's *labels and fit* (≤ 20 u, research) are admitted on V0 PASS + K-1 receipt without waiting for L2′'s
   print — the owner's standing credits rule applies, the cost is bounded, and L2′ may not print before Tuesday; but **no controller,
   WP-4-class build or Mac paced work follows V1 until L2′ prints FUND** (round 12 §3 suspension stands). This is the round-12 premise
   rule applied: V1's premise (truncated latency fits) is measured by V0; the mechanism premise is explicitly flagged NOT_READY(L2′)
   on V1's registration, and V1's verdict can therefore only buy V2, never engineering.
3. **"Search stays exact because its prefix is exact" (question) vs "it is approximate Claim B" (02, 07, 09).** 02/07/09: a predicted
   survivor is not an H300 certificate; new identity, new evidence.
4. **Dense cap 1.458 s vs sparse-burst ≤ 5 s as V0's bar.** Both are reported; the registered *kill* uses the dense cap on frozen-1500
   and frozen-800 (the identity the question is about), and the burst reading is recorded for WP-4's owner — no bar is invented after
   the data.
5. **"Credits are plentiful, build now" vs "builder time is the scarce cost" (09, 10).** 09/10: Sol-K is the only builder free after
   Thu 22:00; V0 costs 0.5–1 of its days; nothing displaces Y3 (Sol-R), the native ruler (Sol-S) or Codex-V's canary/validator work due
   Fri 12:00.

## 5. The first measurement — frozen spec (V0), and the pre-specified conditional fit (V1)

### V0 — Mac latency ladder: exact stop and oracle-leaf truncation on the K-1 fixtures (0 u)

| Field | Spec |
|---|---|
| Platform / code path | **This Mac**, W1, one thread, under the Mac lease (`exp/mac-lease.json`, round-12 §4.7 — Sol writes the helper first), idle receipt first, load logged, no daemon killing. Operator = `ExactT3Operator` with `native_mode="kernel"` in every cell; new modes added by Sol-K only (`stop`, `prefix64_oracle`, `prefix64_oracle_stop`). Dense trigger semantics unchanged. |
| Fixtures | The nine Y1 fixtures (`RULER_FIXTURES_08.md`) × 30 batches per cell, warm-29 medians and nearest-rank p99 (first sample reported cold), cells alternated per batch as K-1 does. frozen-800 is the registered **doom** fixture (bare dies 30/30); frozen-1500 the registered **cap** fixture. |
| Cells | C1 kernel H300 all legal branches (K-1 kernel receipt may be reused if same window/host state; else re-timed). C2 kernel + exact stop (bare first; if bare survives → stop; else alternatives in descending Q, index ties, stop at first H300 survivor; all-die → all evaluated). C3 kernel + H64 exact prefix on all branches + oracle leaf = cached exact H300 outcome of that branch from C1, **plus one fixed-shape dummy leaf call per batch** (12.8k-param MLP, batch = legal count, CPU FP32, featurisation included) so inference overhead is charged. C4 = C3 + exact stop applied to the *oracle* outcomes. H32 cells optional, diagnostic only. |
| Attestation (round-12 §4.3) | Per batch: `kernel_rows > 0 ∧ stock_rows == 0` in branch rows, branches evaluated / skipped, frames per branch, chosen action, stage timers (`stage_seconds` schema), leaf-call wall. A receipt without these counters is INVALID for any speed claim. |
| Identity | C2's chosen action must equal C1's on every batch (exact claim); C3/C4 chosen actions must equal C1's by construction (oracle) — a mismatch is a harness bug ⇒ INVALID. |
| Cap | **0 Colab units; ≤ 1 leased Mac-hour** (forecast: 9 fixtures × 30 × 4 cells; C3/C4 ≈ ¼ of C1; C2 ≈ ¼ on eight fixtures — ≈ 35–50 min at K-1-kernel speeds, else stop at 60 min as INCOMPLETE); **Sol-K 0.5–1 day, Codex-V 0.25 day** (semantics of stop/tie/prefix/terminal, counters). Window: Sat 08:00–10:00 PT (vacated WP-4 slot) or Fri 22:00–23:00 after the native ruler; never inside K-1's or X5's window. |
| Kill rules (registered before the run) | (a) any C2 action mismatch ⇒ C2 INVALID (fix, re-run as a new attempt; no claim). (b) **C3 warm p99 > 1.458 s on frozen-1500 or on frozen-800 ⇒ LEARNED_CUTOFF_CLOSED** (no V1; dense learned-cutoff identity cannot fit even with a perfect leaf). (c) **C3 median < 2× C2 median on frozen-800 ⇒ NO_CONSUMER** (the exact stop already delivers what a critic would; no V1). (d) Incomplete cells ⇒ INCOMPLETE, never a smaller PASS. PASS = none of (b)–(d) ⇒ admits **only V1's registration**. Burst reading (C3/C4 ≤ 5 s on doom roots) is reported for the WP-4 owner, not gated here. |
| Premise register | claim: "H64 + leaf can meet the dense cap" → receipt: `exp/bl-v0/timing.json` (this run) → platform: Mac → code path: kernel + new modes, attested → falsifier: V0 itself. claim: "exact stop saves ≈ 4× on bare-survives roots" → same receipt. claim: "kernel ≥ 5×" → `exp/bl-k1/timing.json` (Thu). claim: "exact search has frozen/mixed value ≥ +40" → **NOT_READY(L2′)** — V1 inherits this flag. |

### V1 — first GPU-trained component: horizon-conditioned tail critic (conditional; ≤ 20 u)

Admitted only on V0 PASS ∧ a K-1 timing receipt (any value; it is the baseline) ∧ a canary receipt for the new manifest family
(round-12 §4.1: real transport bytes *and* file count ≤ 50 % of the 300-s cap, one CUDA forward/backward on L4 through the real
entrypoint, checkpoint + RNG resume, export sidecars by the real producer, Mac reload + batch-one prediction) ∧ supervisor
requalification landed (Fri 12:00). Registration carries the premise table with L2′ as NOT_READY.

- **Data:** 192 fresh development worlds (64/mix; fresh namespace, disjoint from sealed/strict/K-1/Y1/L2′ banks), split **by world**
  96 fit / 48 calibration / 48 challenge, all descendants together. Four roots per world from the exact bare shadow: two ordinary
  dense-trigger roots and two doom roots (bare continuation dies within H300), covering L 600–799 / 800–1,199 / ≥ 1,200 and late clocks;
  missing strata reported, not filled. All legal first actions, exact H300 continuations with **exact reacting opponents and carry**
  (kernel mode), leaf state captured at H64 (and H32/H100 diagnostics), targets `p_survive(236)`, remaining alive frames, remaining
  mass (diagnostic). ≤ 192×4×6×300 ≈ 1.38 M branch-steps + reconstruction. Collection on L4 CPU workers inside one assignment, fit on
  its GPU; shards stay runtime-local; nothing pushed through the staging path except the ≤ 6 MB manifest/catalog.
- **Model (one, frozen before fitting):** ≤ 40k-param head on frozen B8 256-d fused features + remaining horizon + deployment-visible
  roster/clock scalars; survival temperature per horizon fitted on calibration worlds; override rule frozen: accept a non-bare branch
  only at calibrated `p_survive ≥ .95` with positive lower-90 alive-frame advantage; otherwise abstain to bare (abstentions count as
  misses). Baselines: current mass, calibrated legal-greedy Q. No sweep; a second architecture is a new registration.
- **Cap:** 20 u total = canary ≤ 2 u + collection ≤ 8 allocated L4-h + fit/eval ≤ 1 GPU-h + 20 % reserve, first cap wins; Sol 1.5 d,
  Codex-V 0.5 d; ≤ 1 leased Mac-hour for exported-model batch-1/4/6 timing (C3's dummy leaf replaced by the real one — the dense cap
  re-checked). No A100/H100 (fit is seconds).
- **Kill rules:** INVALID (leakage, missing attestation, canary absent) before KILL. On the 48 challenge worlds, separately for frozen
  and mixed: **rescue recall < 90 %** of exact-search rescue choices, or **harmful accepted overrides > 5 %** (selected branch dies under
  exact H300 while an exact survivor existed), or real-leaf C3 p99 > 1.458 s ⇒ **KILL** for this component. **< 30 rescue-positive
  roots in a mix ⇒ INCONCLUSIVE** (no extension, no re-sampling). Pooled AUC and agreement are descriptive only. **PASS buys only V2**
  (fresh paired closed-loop 32/mix H5000, critic controller vs exact vs bare, ≤ 25 u) — never a controller build, never Mac paced time,
  until L2′ prints FUND.

Not funded this round: the opponent-model pilot (reopens only on K-1 FAIL by owner re-opt), a learned pruning prior, MPS/Core ML leaf
placement work (04's probe runs only if V1 passes and CPU batch-1 misses the cap), A100/H100, any six-branch kernel packing beyond
K-1's optional (c).

## 6. Sequencing Thu Oct 22 → Tue Oct 27 (round-12 §3 table unchanged; additions in bold)

| When (PT) | Action | Receipt / rule |
|---|---|---|
| Thu 09:00–12:00 | Y3 fix (Sol-R → Codex-V); D1 slip file for L2′. **Sol-S: add the exact-stop mode + branches-evaluated counters to `ruler_operator` (prospective amendment text to `AMENDMENT_08`-style file, Codex-V review).** | `exp/bl-p1/y3/source-03/…`; `research/bl_served_search/AMENDMENT_STOP.md` |
| Thu 20:00–22:00 | **K-1 timing** (untouched; V0 reuses its kernel receipt). Fable D2 ruling 22:30. | `exp/bl-k1/timing.json`; < 5× or mismatch ⇒ exact served search CLOSED for week 2 |
| Fri 09:00–11:00 | X5 under the lease. | `exp/week1-20261021/mac/x5.json` |
| Fri 12:00 | D1′ P1 (Y3 PASS ∧ 1M forecast ≤ 24 h nominal ∧ critic gate). **Canary mode + validator + failure classes + lease helper + ledger check land; supervisor requalified (≈ 1.5 u).** | per round 12 §4–5 |
| Fri 12:00–20:00 (D2 PASS) | Native whole-workload Y1 rerun on the Mac **in both all-branch and exact-stop modes**; one L4 native ruler ≤ 4 u. | `exp/bl-l2/y1-native-{mac,l4}/ruler-receipt.json` |
| Fri 22:00–23:00 **or** Sat 08:00–10:00 | **V0** under the lease (§5). Fable reads it Sat 11:00. | `exp/bl-v0/timing.json` (+ `idle.json`, counters) |
| Sat 12:00 | L2′ placement ruling — **now with the exact-stop forecast available** (Mac ≤ 60 wall-h clause). | `exp/bl-l2/placement-ruling.json` |
| Sat 16:00 | Handoff `INDEX_v2.json` + one line: V0 PASS / LEARNED_CUTOFF_CLOSED / NO_CONSUMER / NOT_RUN. | — |
| Sat–Sun (V0 PASS only) | V1 registration → canary → dispatch on an idle slot (never P1's); fit; Mac leaf timing ≤ 1 h. | `exp/bl-v1/{registration,canary,collect,fit,receipt}.json` |
| Tue 23:00 | KEEP_B8 reducer, literal. Under KEEP_B8 the V-line closes with every approximation lane (PLAN_v2 §6) and is offered back only through §7. | — |

Builder budget added by this round: Sol-K 0.5–1 d (V0), Sol-S 2–4 h (stop mode), Codex-V 0.25 d; conditional Sol 1.5 d + Codex-V 0.5 d
(V1). Units: 0 (V0); ≤ 20 (V1, conditional); 0 on A100/H100/RunPod.

## 7. Running shortlist (carried from round 11, updated)

| # | Lever | Conf. push / beats B8 at bar | Cost to first signal | Most informative next experiment |
|---|---|---|---|---|
| 1 | **Exact-sim search over B8 — mechanism** (offline, unpaced) | 35 % / 15 % (unchanged) | L2′ ≤ 200 u; placement unknown | **L2′ placement Sat 12:00**, now with the exact-stop forecast |
| 1a | Served sparse-burst **exact** identity on this Mac (K-1 kernel + exact stop) | **15 % / 7 %** (up 3/1: the exact stop removes 3 of 4 branches on 98.8 % of dense roots — a projection until V0) | K-1 0 u Thu; V0 0 u; WP-4 build suspended until L2′ prints | **K-1 (Thu) → V0 C2 cells** |
| 1b-i | Approximate search — **tail critic after H64 exact prefix** | 8 % / 4 % (unchanged; T10/T35 cap the prior) | V0 0 u → V1 ≤ 20 u → V2 ≤ 25 u | **V0 C3 cells**; V1 only on PASS |
| 1b-ii | Approximate search — opponent model | **3 % / 1 %** (down: residual floor; must beat the kernel; compounding) | ≤ 10 u pilot | none this campaign; K-1-FAIL branch only, owner re-opt |
| 1b-iii | Approximate search — learned pruning prior | **2 % / 1 %** (room taken by the exact stop) | — | none |
| 1d | Harvest-aware operator (O1 GO) | 3 % add-on | O2 ≥ 150 u at power | after L2′ FUND and a served identity |
| 2 | Representation (E3 → E14 → E16) | **0 % (E3 KILL)** | — | closed |
| 3 | Long-horizon PPO from B8 — P1 fixed 100M endpoint | **15 % / 8 %** (down 5/2: Y3 failed; lane in recovery; ≤ 24-h invoice binds) | Y3 ≤ 8 u → 1M ≤ 4 u → 100M ≤ 120 u → eval ≤ 10 u | Y3 source-03 (Thu) → D1′ Fri 12:00 |
| 4 | GPU-resident exact engine | 0 % this campaign | — | none |
| 5 | Search-labelled student | 5 % / 3 % | 96 u week-2 design | only if L2′ FUND |
| 6–8 | Restart archive; opponent add-on; world model / league | 4 / 3 / 1 % | — | not funded |

## 8. Still unresolved

K-1 timing (Thu); whether the exact stop's ≈ 4× projection survives measurement (V0 C2; per-batch fixed costs — clone, digest — are not
∝ frames); whether a perfect leaf at H64 meets 1.458 s on the doom fixture (V0 C3); the L2′ placement with the stop forecast (Sat);
P1's Y3/1M (Fri 12:00); X5 (Fri AM); the Mac lease helper and canary mode landing before any new dispatch; the owner's appetite under
KEEP_B8 for a 3–6-week approximation program (asked only at Sat 16:00 with V0 in hand); builder hours (start the per-WP time log Thu).

## 9. Experiments Claude can run now (dispatch via Sol; report in LIVE_FACTS.md)

- **V0 — Mac latency ladder (0 u; ≤ 1 leased Mac-h; Sol-K 0.5–1 d; Codex-V 0.25 d; Fri 22:00 or Sat 08:00).** Spec, attestation and
  kill rules in §5. Admits only V1's registration.
- **Exact-stop mode in the Fri native rulers (0 u; Sol-S 2–4 h; inside the already-scheduled Mac/L4 ruler windows).** Prospective
  amendment, Codex-V reviewed; branches-evaluated counters; chosen-action identity vs the all-branch mode on every batch. Feeds the
  Sat 12:00 L2′ placement ruling.
- **K-1 timing (0 u; Thu 20:00–22:00) — as registered; nothing added.**
- **V1 — tail-critic fit (≤ 20 u incl. canary + reserve; Sol 1.5 d; Codex-V 0.5 d; Sat–Sun, idle slot only; conditional on V0 PASS).**
  Spec and kill rules in §5; research only; buys V2 at most.
- **P1 Y3 fix → source-03 → 1M qualification (≤ 8 + 4 u; Thu–Fri) and X5 (0 u; Fri 09–11)** — as re-sequenced in round 12; untouched.
- Not funded: opponent-model pilot, pruning-prior fit, MPS/Core ML leaf work, O2, Round 0 re-registration, E1b, WP-4 build, A100/H100.
