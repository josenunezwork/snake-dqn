# Sequential Phase R for future Tier-1 screens -- design (2026-10-07)

**Status: design, committed before any code on branch `seq-phaser`.** The git log shows the
order. The operating characteristics (OC) in section 9 are filled in by a later commit, from the
committed simulator. The scratch exploration that chose the parameters below is disclosed in
section 8.

**Scope.** This is for future FRP-family Tier-1 screens only. Existing studies stay exactly as
they are: FRP-v2, v3, v4, v5-H, v5-S, v5-S2 and every running or merged Phase R keep their
fixed-N packages. Nothing here can promote. A Phase R GO still buys LH-1, then the strict
sequential gate, then serving qualification.

**User request (2026-10-07):** "a sequential Phase R that checks at about 1/3 and 2/3 and stops
early on a clear GO or KILL -- same approach as our strict gates; expected saving about half the
evaluation compute; current studies stay as they are."

**Headline (disclosed now, quantified in section 9).** In the exploration, a faithful
GO/KILL-only design saves about 1-15 % of Phase R compute when the true effect is between
-0 and +65. It saves about 35-40 % when the candidate is clearly worse (-20). It cannot reach
the expected ~50 %, for three reasons:

1. The FRP rule's full GO also needs gate feasibility (clause 7: a pooled point of about +66 or
   more at cap 600). So a clear GO exists only for large effects.
2. At the fixed rule's own precision, KILL (upper bound < +20) happens only about 45-60 % of the
   time even under a true null.
3. The labels most runs end with (GO_UNGATEABLE / RECIPE / PARTIAL) are judged only at the
   final look.

The optional **NO_GO** stop (section 4.4), a repeated upper bound below a pre-registered
"gateable size" G, reaches about 45-50 % saving under the null and about 20-25 % at +20. It does
this at the price of resolving PARTIAL / UNGATEABLE / KILL on fewer worlds. The owner chooses
whether a study pre-registers it; see the governance amendment draft.

## 1. What a fixed Phase R decides (the template being made sequential)

The FRP-family rule (FRP-v4 README, v5-H, v5-S2):

* **Pairing and estimator.** Paired by (seed, mix, world). Per seed, the effect is the
  mix-stratified mean of per-mix mean deltas, with variance `(1/M^2) sum_m s_m^2/n_m`. Seeds
  are pooled with DerSimonian-Laird + Hartung-Knapp (`screen_stats.random_effects_pool`,
  `hksj_floor`). A one-sided 90 % bound is an end of the two-sided 80 % interval.
* **GO** needs all of the following:
  * (1) pooled H5000 point >= +20 AND HK lower bound > 0;
  * (2) scripted mean >= 0 AND scripted survival >= -0.03;
  * (3) frozen / mixed survival >= -0.03;
  * (4) >= 4 of 5 (6 of 8) seeds positive;
  * (5) prefix identity controls pass AND pooled MI10 point >= +20;
  * (6) champion guard (point > 0, scripted survival >= -0.05 on the reference seed's 96
    episodes);
  * (7) gate feasibility (N_max <= cap);
  * (8) attribution contrast points (v5).
* **Secondary labels:** GO_UNGATEABLE (v4) or GO_RECIPE (v5).
* **KILL:**
  * v4: HK upper bound < +20;
  * v5: Welch upper bounds of the H - C contrast < +25 at H5000 AND < +40 at MI10.
* **Otherwise** PARTIAL.
* **INCOMPLETE / INVALID_ANALYSIS** come first.

## 2. Looks and the world schedule

* **Looks** are at information fractions 1/3, 2/3, 1 of the planned `N` worlds per (seed, mix).
  `look_sizes = ceil(f N)` (N = 32 gives 11 / 22 / 32). The fractions used for alpha spending
  are the actual `n_k / N`.
* **Look k uses the first `n_k` worlds of every seed's bank**, in the bank's pre-declared order
  (the same order in every mix), for **every cell**: incumbent, candidate, control arm,
  champion reference, descriptive tiers. Every look is therefore a balanced subset over
  seeds x mixes x heroes, and every delta stays paired.
* **The schedule is look-major.** The study plays all of look 0's units (every seed, mix and
  hero on world indices `[0, n_0)`), then analyses look 0, then plays `[n_0, n_1)`, and so on.
  A look's world groups (seed, mix, <= 8-world batch inside the look's index range) never
  straddle a look boundary, so the per-world platform rule (all heroes of a world on one pod)
  still holds.
* **Look units are whole nested episodes.** The H10000 records carry their exact H5000 prefix
  blocks, so every look has H5000 and MI10 on the same worlds, and no episode is ever cut at a
  look.
* **The prefix identity controls** (direct H5000 on world index 0) run in look 0. Their pass or
  fail is known from look 0 on and binds every later look.
* **Information time.** Spending uses the world fraction `t_k = n_k / N`. With between-seed
  heterogeneity (tau^2 > 0), the HK pooled estimate's variance is `(tau^2 + sigma^2/n)/k`, so
  the true information fraction is above `n_k / N`. The shared seed effects also raise the
  correlation between looks. For fixed per-look nominal levels the probability of crossing at
  any look decreases as correlation increases (Slepian), so world-fraction spending is
  conservative.

## 3. GO (efficacy): O'Brien-Fleming on the HK pooled one-sided lower bound

* **The efficacy statistic** is the rule's clause (1) HK pooled H5000 effect. It crosses at
  look k when `mu_k - t_{k_seeds-1}(p_k) se_k > 0`.
* **The nominal levels `p_k`** come from Lan-DeMets O'Brien-Fleming spending of
  `go_alpha = 0.10` one-sided, which is exactly the fixed rule's "HK one-sided 90 % lower bound".
  The boundaries are exact on the canonical joint normal (the strict gate's recursion,
  `screen_stats._crossing_recursion`).
* **The HK statistic is a t with k-1 df** (4 at five seeds). It is mapped by tail-area matching
  (Jennison & Turnbull 2000, section 3.8), the strict gate's z/t treatment.
* **N = 32 gives:**
  * nominal `p = 0.0050 / 0.0457 / 0.0860`;
  * at df 4 these are t multipliers of about 4.6 / 2.3 / 1.6, against the fixed 1.533.
* **Small-k HK.** With df = 4 the early look is very conservative. That is OBF's intent: a
  look-0 GO needs an overwhelming effect. The z/t mapping is exact at each single look, but the
  joint crossing probability under estimated variances is only approximate. The simulation
  (section 9) measures the GO type-I error directly, at tau^2 = 0 and 500, on resampled real
  deltas (skew included).
* **Every other GO clause** is judged at the stopping look (section 5). A GO at an interim look
  needs the **full** GO: every clause, including feasibility. GO_UNGATEABLE / RECIPE are never
  an early stop.
* **Futility is non-binding for GO.** The GO boundaries ignore KILL / NO_GO, so a KILL or NO_GO
  stop can only lower P(GO). An overridden KILL never inflates the GO error, but it is a
  protocol deviation that must be recorded.

## 4. KILL (and the optional NO_GO): repeated upper bounds

### 4.1 Choice: non-binding (for GO), followed (by protocol)

KILL boundaries are **not** used to relax the GO boundaries (non-binding), for three reasons:

* The owner has overridden or re-scoped screen outcomes before ("owner decides" is in every
  FRP rule).
* A binding futility would make the GO error depend on following every KILL.
* The power it buys (a slightly lower final GO boundary) is small next to clause (7), which
  dominates GO here.

KILL is still **followed by protocol**: a KILL look receipt ends the study, and `look_gate`
refuses the next look.

### 4.2 KILL error control

* **Mechanism.** KILL at look k when every KILL component's one-sided upper bound at the KILL
  nominal level `q_k` is below its threshold. The `q_k` spend `kill_alpha = 0.10` (the fixed
  rule's "one-sided 90 % UPPER bound") over the looks, so the upper bounds are **repeated
  confidence bounds** for the hypothesis "true effect >= threshold".
* **Guarantee.** A candidate exactly at a KILL threshold is KILLed with probability <= 0.10
  over all looks. That is the fixed rule's guarantee, kept under repeated looks.
* **Components are ANDed.** v5's two contrasts form an intersection, so the conjunction's
  error is <= each component's.

### 4.3 KILL spending: Pocock-type (Lan-DeMets)

* **Why not OBF.** At df 4, OBF's look-0 level (0.005, t of about 4.6) makes an early KILL
  nearly impossible: the bound needs a pooled point of about -60.
* **Pocock-type.** Lan-DeMets `alpha ln(1 + (e - 1) t)` spends about 0.046 / 0.051 / 0.053
  per look, giving t of about 2.4-2.5 at every look. In the exploration this gives:
  * a lower false-KILL rate than the fixed design at +20 and +30, because the final look's
    KILL is stricter;
  * 2-3x the early KILLs of OBF.
* **The cost** is a final-look KILL nominal of about 0.053 instead of 0.10. Some runs the fixed
  design would call KILL end PARTIAL instead (under the null, KILL 0.43 vs 0.59 at tau^2 = 0).
  That is the conservative direction for a "saturation shown" claim.
* HSD(gamma = 1) gave the same numbers.

### 4.4 NO_GO (optional, pre-registered per study): gate futility

* **The rule.** At an interim look, NO_GO when the HK pooled H5000 repeated upper bound (KILL's
  `q_k`) is below **G**, the smallest pooled effect for which a gateable GO is plausible.
* **Choosing G.** Clause (7) at cap 600 and SD* of about 290 needs a point of about +66-73, so
  the exploration used G = 40 and G = 50.
* **The claim and the action.** NO_GO is a repeated-bound claim "true effect < G" at error
  <= 0.10, and a truly gateable candidate (effect >= G) is stopped with probability <= 0.10.
  The action is "no strict gate for this candidate; the owner decides". The final-only labels
  are reported **descriptively** on the stopping look's worlds.
* **Precedence:** GO > KILL > NO_GO at an interim look.
* **What it costs.** A run the fixed design would have labelled GO_UNGATEABLE (a warm-start
  candidate) or PARTIAL is cut short with the label unresolved. At +40 the exploration lost
  about 10 % of the would-be UNGATEABLE outcomes this way at G = 50.
* **This is the only part that reaches ~50 % saving, and only near the null.** It is off unless
  a study's pre-registration declares it.

### 4.5 Precedence of final-only outcomes over an interim KILL / NO_GO

An interim KILL or NO_GO is suppressed (the study continues) while a higher-precedence
final-only outcome (UNGATEABLE / RECIPE) holds on the look's data. This keeps the fixed rule's
precedence: GO > final outcomes > KILL.

## 5. Survival floors, point clauses and the champion guard at an early stop

* **The interim margin.** Every point-threshold clause of GO is judged at the stopping look with
  the margin `m_k = z_m se_k (1 - sqrt(n_k / N))` against it. That covers (1)'s +20, (2), (3),
  (5)'s MI10 +20, (6)'s guard point and scripted survival, and (8)'s attribution points. The
  margin is zero at the final look, so the final look is the fixed point rule.
* **The property.** This is the strict gate's `block_at_stop` band rule
  (`sequential_gate.band_check`). For a world-level mean that truly sits D final-N standard
  errors on the wrong side of its floor, the interim pass probability is
  `Phi(-(D sqrt t + z_m (1 - sqrt t)))`. That is at most the fixed design's `Phi(-D)` whenever
  D <= z_m.
* **The value of `z_m`.** `z_m = 1.2816` (one-sided 90 %, the screens' convention). An early
  stop therefore never passes a floor-violating candidate more easily than the full run would,
  for violations up to 1.28 final SEs. Deeper violations pass with probability below 0.10
  either way.
* **Why not a full RCI on survival.** A full repeated confidence bound (the strict gate's band
  v2 `rci_obf`) would turn the fixed rule's *point* floors into *bound* floors, which is a
  different and much stricter rule at the final look. The margin rule is the RCI-style
  adjustment that reduces to the pre-registered point rule at N.
* **Count and flag clauses.** (4) positive seeds is a count. Interim noise mostly makes it
  harder to pass for a true effect, and under the null each seed is 50/50 at any look. (5)'s
  prefix controls are a flag.
* **Clause (7) feasibility** uses the stopping look's **own efficacy bound** (a repeated lower
  bound at `p_k`) for the MDE* lower-bound route, plus the 0.6 shrink route, and SD* from the
  look's worlds.
* **The champion guard** is on the reference seed's first `n_k` worlds, judged with the same
  margin.
* **Early-stop estimates are biased.** Point estimates after an early stop are biased upward
  (conditional on stopping). Phase R never promotes: LH-1 and the strict gate play fresh worlds,
  so the bias stays out of promotion. Candidate selection uses the stopping look's worlds, and
  that is disclosed.

## 6. INVALID precedence and protocol

* **Per look.** INVALID_ANALYSIS (not a scientific outcome) is decided first. The causes are:
  * missing or non-finite deltas;
  * a pooled estimate unavailable;
  * records beyond the look (prefix integrity);
  * an earlier look that already stopped or halted;
  * a plan mismatch.

  The look receipt's action is then HALT, and the owner decides. INCOMPLETE (a time box or cap
  reached before a look completes) stays the study's own outcome; a partial look is never
  analysed.
* **Precedence at a look:**
  * interim: INVALID > GO > KILL > NO_GO > CONTINUE;
  * final: INVALID > GO > final outcomes (rule order) > KILL > PARTIAL.
* **Create-only receipts.** Before any Phase R shard, `plan.json` is written: the plan dict
  with its sha256, the banks and the study binding (commit, ratification, pre-registration
  sha256). After look k's records are merged, `looks/look-<k>.json` is written. It holds:
  * the plan sha256;
  * the previous receipt's sha256;
  * every record path and its sha256;
  * the deltas digest and the flags;
  * the full decision and the action.

  Both are fsynced and hard-linked, so neither can ever be overwritten.
* **The look gate.** `look_gate(root, k)` refuses look k unless receipt k-1 exists, is valid and
  says CONTINUE. Its binding (plan sha256, previous receipt sha256, world index range) is
  recorded in every look-k shard's start marker, on the Mac or on a pod.
* **No resume or relabel.** A failed look stays failed. As in the strict template, nothing is
  relabelled.
* **Independent audit.** `research/sequential_phase_r/audit.py` is stdlib only and imports
  nothing from the repo. It recomputes:
  * the spending boundaries (its own recursion), the t quantiles, HK, Welch and the clause
    logic;
  * every look's deltas from the raw records (hash-checked);
  * the receipt chain and the prefix integrity.

  It must agree with every receipt.

## 7. Integration (opt-in; existing packages untouched)

* `src/evaluation/sequential_phase_r.py` holds the pure plan, statistics and decision.
* A study declares its rule as a JSON rule spec (`validate_rule`) in the pre-registration and
  pins the spec's sha256 by test.
* `research/sequential_phase_r/` holds the reusable pieces:
  * `hooks.py`: the look schedule (units per look), records to look deltas, and analysing a
    look into a receipt;
  * `receipts.py`;
  * `audit.py`;
  * `simulate.py`: the OC simulation;
  * `example_rules.py`: FRP-v4-like and FRP-v5-like specs.
* **Pod runner (`research/pod_phaser`, branch `frp-v5h-phaser-rp`).** Not modified here. A
  sequential study's stages are blocks **per look**. `podrun` must call `hooks.look_gate` before
  launching a look-k block, and the task agent must bind the returned dict into each shard's
  start marker. The Phase R merge runs per look, then `hooks.analyse_look`. Notes are in the
  package README.

## 8. Parameter choice (scratch exploration, disclosed)

**Inputs.** Before this document, a scratch numpy exploration (not committed) used the same
variance inputs as section 9. It resampled the per-world paired deltas of FRP-v2 Phase 2, FRP-v3
Phase R and FRP-v4 Phase R (480 world blocks of 3 mixes x {H5000 mass, H5000 survival, MI10}).
It compared:

* the fixed design;
* OBF/OBF;
* OBF/Pocock;
* OBF/HSD(1);
* margin z in {0, 1.28};
* NO_GO at G in {40, 50}.

**Chosen:**

* GO: OBF, alpha 0.10;
* KILL: Pocock-type, alpha 0.10;
* `z_m` = 1.2816;
* NO_GO optional, with G = 50 as the reference value.

**Why:** OBF and Pocock had the same GO power. Pocock had 2-3x the early KILLs at a lower
false-KILL rate than the fixed design. z 0 would add about 1.5 points of GO at +65, but would
give up the floor guarantee.

## 9. Operating characteristics (filled by the simulator commit)

To be reported from `research/sequential_phase_r/simulate.py`, for the fixed design, the
sequential design and the sequential design + NO_GO(G = 50):

* **Families:** FRP-v4-like (5 seeds, KILL on HK UB < 20, UNGATEABLE) and FRP-v5-like (8 seeds,
  H and C arms, KILL on the Welch contrast, RECIPE).
* **Grid:** effects -20 / 0 / +20 / +30 / +40 / +65 x tau^2 in {0, 500}, plus a
  survival-harm scenario.
* **Measures:**
  * P(GO), and P(efficacy crossed) = the error-controlled component, under the null;
  * the GO power loss at +40 / +65;
  * false KILL at +30 (and at the threshold);
  * survival-harm pass rates;
  * the expected fraction of evaluation compute and the saving;
  * the stop-look distribution, with Monte Carlo SEs.
