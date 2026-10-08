# Sequential Phase R for future Tier-1 screens -- design (2026-10-07)

**Status: design.** The first version (commit `463e905`) was committed before any code on
branch `seq-phaser`. This revision adds three things:

* the operating characteristics in section 9, from the committed simulator;
* the changes made during the build: protective-clause margins (section 5) and gate futility
  wording (section 4.5);
* the independent review's fixes (section 10).

The scratch exploration that chose the parameters is disclosed in section 8.

**Scope.** This is for future FRP-family Tier-1 screens only. Existing studies stay exactly as
they are: FRP-v2, v3, v4, v5-H, v5-S, v5-S2 and every running or merged Phase R keep their
fixed-N packages. Nothing here can promote. A Phase R GO still buys LH-1, then the strict
sequential gate, then serving qualification.

**User request (2026-10-07):** "a sequential Phase R that checks at about 1/3 and 2/3 and stops
early on a clear GO or KILL -- same approach as our strict gates; expected saving about half the
evaluation compute; current studies stay as they are."

**Headline (quantified in section 9).** Against the fixed design, the faithful GO/KILL-only
design keeps the error-controlled efficacy crossing rate at or below the fixed design's,
everywhere. It keeps GO power within about 2 points, and its false-KILL rate is at or below the
fixed design's.

**Saving, without NO_GO:**

| True effect | Saving |
|---|---|
| Null (0) | 12-23 % |
| +20 to +40 | 0-3 % |
| +50 | 3-6 % |
| +65 | 8-15 % |
| -20 (clearly worse) | 34-48 % |

It does **not** reach the expected ~50 %, for three reasons:

1. The FRP rule's full GO also needs gate feasibility (clause 7: a pooled point of about +66 or
   more at cap 600). A clear GO therefore exists only for large effects, and even then OBF at
   df 4 rarely stops at look 0.
2. KILL (upper bound < +20) at the fixed rule's own precision happens only about 45-85 % of the
   time, even under a true null.
3. The labels most runs end with (GO_UNGATEABLE / RECIPE / PARTIAL) are judged only at the
   final look.

**With the optional NO_GO stop** (section 4.4: a repeated upper bound below a pre-registered
"gateable size" G = 50), the saving is:

| True effect | Saving |
|---|---|
| Null (0) | 46-61 % |
| +20 | 23-38 % |
| -20 | 59-65 % |

Two prices come with it. Most runs the fixed design would label PARTIAL, UNGATEABLE, RECIPE or
KILL are resolved only descriptively. A candidate at G is stopped 2-4 % of the time (bound
0.10). The owner chooses whether a study pre-registers NO_GO; see the governance amendment
draft.

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
  * t multipliers of 4.60 / 2.21 / 1.66 at df 4, against the fixed 1.533;
  * t multipliers of 3.50 / 1.96 / 1.52 at df 7.
* **Clause (1) at the final look is slightly stricter than in the fixed rule** (nominal 0.086,
  not 0.10). This also applies to final-only labels that need the efficacy clause, such as
  GO_UNGATEABLE (`requires_efficacy`). The disclosed consequence: GO_UNGATEABLE is a slightly
  stricter label than FRP-v4's.
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

* **Why not OBF.** At df 4, OBF's look-0 level (0.005, t = 4.60) makes an early KILL nearly
  impossible: the bound needs a pooled point of about -60.
* **The Pocock-type rule.** Lan-DeMets `alpha ln(1 + (e - 1) t)` spends 0.046 / 0.032 / 0.022
  (cumulative 0.046 / 0.078 / 0.100). Its nominal levels are 0.0464 / 0.0508 / 0.0528:
  * t of 2.20 / 2.12 / 2.08 at df 4;
  * t of 1.95 / 1.88 / 1.86 at df 7.
* **What it gives** (section 9):
  * a false-KILL rate at and above the threshold that is *lower* than the fixed design's, because
    the final look's KILL is stricter;
  * 2-3x the early KILLs of OBF.
* **The cost** is a final-look KILL nominal of about 0.053 instead of 0.10. Some runs the fixed
  design would call KILL end PARTIAL instead (under the null, KILL 0.43 vs 0.59 for v4 at
  tau^2 = 0). That is the conservative direction for a "saturation shown" claim.
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
final-only outcome (UNGATEABLE / RECIPE) holds on the look's **point values**. Point values
means no interim margin is applied, so the suppression is never weakened by the margin. This
keeps the fixed rule's precedence: GO > final outcomes > KILL.

For v4 the suppression can never fire: KILL needs an upper bound below +20, and UNGATEABLE needs
clause (1)'s point at or above +20, so the two cannot hold together.

## 5. Survival floors, point clauses and the champion guard at an early stop

Every point-threshold clause of GO is judged at the stopping look with an interim margin against
it. The margin is zero at the final look, so the final look is the fixed point rule.

* **Protective clauses** (marked `"protective": true` in the rule): (2) scripted mean and
  scripted survival, (3) frozen / mixed survival, (6) guard point and guard scripted survival.
  * **Margin:** `m_k = 1.645 x se_k x sqrt(1 - n_k/N)`. The quantity `se_k^2 (1 - t)` is the
    variance of (interim mean - final mean), so an interim pass is one the full run would
    confirm with probability of about 0.95.
  * **The union problem.** A run that fails a floor at an interim look continues, so it gets
    another chance at a later look. A per-look guarantee therefore does not bound the
    probability of passing at *any* look.
  * **Exploration of other choices:**
    * the strict gate's `(1 - sqrt t)` shape at z 1.28 let survival-harmful candidates
      (+65 mass, a survival delta 0.6-2 final SEs below a floor) through up to +0.04 absolute
      more often than the fixed design;
    * judging the floors once, at the qualifying look (`block_at_stop`), cut GO power at +65
      by about 10 points.
  * **What the chosen shape gives** (section 9): survival-harm and guard-harm GO rates within
    **+0.004 absolute** of the fixed design's (paired SE 0.001). The review's independent grid
    over D = 0..2 SEs and effects 65 / 90 / 130 measured up to +0.004 as well.
  * **This is an empirical bound, not a theorem.** The remaining excess comes from the extra
    looks giving the *other* clauses more chances.
* **Other point clauses:** (1) the +20 point, (5) MI10 +20, and (8) the attribution points.
  * **Margin:** `m_k = 1.2816 x se_k x (1 - sqrt(n_k/N))`, the strict gate's `block_at_stop`
    band margin. For a world-level mean that truly sits D final-N standard errors on the wrong
    side of its threshold, the interim pass probability at a look is
    `Phi(-(D sqrt t + z (1 - sqrt t))) <= Phi(-D)` for D <= z.
* **Count and flag clauses.**
  * (4) positive seeds is a count. Interim noise mostly makes it harder to pass for a true
    effect; under the null each seed is 50/50 at any look.
  * (5)'s prefix controls are a flag, known from look 0 and identical in every receipt.
* **Clause (7) feasibility** uses three inputs: the stopping look's **own efficacy bound** (a
  repeated lower bound at `p_k`) for the MDE* lower-bound route, the 0.6 shrink route, and SD*
  from the look's worlds.
  * **Disclosed:** at an interim GO, 0.6 x an upward-biased point can size a strict gate that
    is underpowered. That costs strict-gate compute, not error: the strict gate controls its
    own error.
* **The champion guard** is on the reference seed's first `n_k` worlds, as a protective clause.
* **Early-stop estimates are biased.**
  * Point estimates after an early stop are biased upward (conditional on stopping).
  * Phase R never promotes: LH-1 and the strict gate play fresh worlds.
  * Candidate selection uses the stopping look's worlds, and that is disclosed.

## 6. INVALID precedence and protocol

* **Per look.** INVALID_ANALYSIS (not a scientific outcome) is decided first. The causes are:
  * missing or non-finite deltas;
  * a pooled estimate unavailable;
  * records beyond the look (prefix integrity);
  * an earlier look that already stopped or halted;
  * a plan mismatch.

  **Every invalid look is INVALID_ANALYSIS with action HALT, whatever its data say.** The
  would-be label is kept only as `status_if_valid`. The audit's verdict for a study that ended
  this way is `HALTED`, never PASS. INCOMPLETE (a time box or cap
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

  It also recomputes the prefix-identity flag from the control records and checks that the
  flags are identical in every receipt. It must agree with every receipt.

  **Not audited by it:** the shard start markers that carry the `look_gate` binding. They are
  the study runner's files, and the study's own audit checks them.

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
* margins: z 1.2816 (non-protective), changed during the build for protective clauses (see
  below);
* NO_GO optional, with G = 50 as the reference value.

**Why:** OBF and Pocock had the same GO power. Pocock had 2-3x the early KILLs at a lower
false-KILL rate than the fixed design. z 0 would add about 1.5 points of GO at +65, but would
weaken the floors.

**Changed during the build** (before the OC run in section 9, disclosed): the protective
clauses' margin became `1.645 sqrt(1 - t)`, after the union-over-looks probe in section 5.

## 9. Operating characteristics (committed simulator, `oc_results_20261007.json`)

**How it was run.** `python -m research.sequential_phase_r.simulate --reps 10000 --seed
20261007`: one thread, about 4 minutes, 10,000 replicates per scenario.

**What each replicate does.**

* Each seed resamples N = 32 world blocks, with replacement, from the variance pool: 480 blocks
  of FRP-v2 Phase 2, FRP-v3 Phase R and FRP-v4 Phase R per-world paired deltas, centred per
  (study, seed, mix, metric), so skew and heavy tails are kept. Then it adds the scenario:
  * an H5000 effect plus seed effects ~ N(0, tau^2);
  * MI10 = 1.5 x effect, with correlated seed effects;
  * survival + 0.00035 x effect;
  * an optional harm term.
* The guard (hero - champion) shares the hero's seed-0 episodes with the primary cell, at
  correlation 0.5 (FRP-v4 measured 0.54 for mass and 0.51 for survival), with a +75 gap.
* The v5 control arm shares the incumbent episodes, also at correlation 0.5.
* The prefix controls pass.

**Pairing.** All three designs see the same replicates (common random numbers), so seq - fixed
differences carry paired SEs.

**Plan shas (N = 32):**

| Rule | Plan sha | With NO_GO(50) |
|---|---|---|
| v4 | `cad427aaaf58a634...` | `f58507357a44c06e...` |
| v5 | `aa515abc0b6ee657...` | `6b989c823135b4fc...` |

**Cross-check.** The vectorised evaluator is checked against the pure decision function in
the tests, on replicates of both families, with and without NO_GO. The reviewer separately
checked 1,600 replicates: 0 mismatches.

**How to read the measures.**

* **"eff. crossing any look, non-binding"** is the error-controlled quantity. It is P(efficacy
  bound crosses at any look), with KILL / NO_GO / GO stops ignored.
* **P(GO)** under the null is 0 in both designs, because GO needs feasibility, which needs about
  +66.
* **"saving"** = 1 - E[fraction of the planned worlds played], for every cell (look-major
  schedule).

**V4 family** (5 seeds, KILL on HK UB < +20; 10,000 replicates per cell; paired SE of seq - fixed in parentheses)

| effect | tau^2 | P(GO) fixed / seq (diff, SE) | eff. crossing any look, non-binding: fixed / seq | P(KILL) fixed / seq | saving seq | saving seq+NO_GO(50) | P(NO_GO) |
|---|---|---|---|---|---|---|---|
| -20 | 0 | 0.000 / 0.000 (+0.000, 0.000) | 0.000 / 0.000 | 0.977 / 0.935 | 40% | 62% | 0.522 |
| +0 | 0 | 0.000 / 0.000 (+0.000, 0.000) | 0.054 / 0.047 | 0.589 / 0.430 | 13% | 52% | 0.811 |
| +20 | 0 | 0.001 / 0.001 (+0.000, 0.000) | 0.606 / 0.564 | 0.049 / 0.029 | 1% | 26% | 0.541 |
| +30 | 0 | 0.009 / 0.010 (+0.001, 0.001) | 0.876 / 0.852 | 0.006 / 0.004 | 0% | 13% | 0.278 |
| +40 | 0 | 0.079 / 0.070 (-0.009, 0.002) | 0.977 / 0.969 | 0.000 / 0.001 | 1% | 6% | 0.096 |
| +50 | 0 | 0.299 / 0.279 (-0.020, 0.002) | 0.998 / 0.998 | 0.000 / 0.000 | 3% | 4% | 0.021 |
| +65 | 0 | 0.733 / 0.713 (-0.020, 0.002) | 1.000 / 1.000 | 0.000 / 0.000 | 9% | 9% | 0.002 |
| harm_scripted-0.05 | 0 | 0.398 / 0.393 (-0.005, 0.001) | 1.000 / 1.000 | 0.000 / 0.000 | 2% | 2% | 0.002 |
| harm_scripted-0.07 | 0 | 0.205 / 0.205 (+0.000, 0.001) | 1.000 / 1.000 | 0.000 / 0.000 | 1% | 1% | 0.002 |
| harm_scripted-0.10 | 0 | 0.041 / 0.042 (+0.001, 0.000) | 1.000 / 1.000 | 0.000 / 0.000 | 0% | 0% | 0.002 |
| harm_frozen-0.07 | 0 | 0.220 / 0.221 (+0.001, 0.001) | 1.000 / 1.000 | 0.000 / 0.000 | 1% | 1% | 0.001 |
| harm_guard_scripted-0.10 | 0 | 0.396 / 0.390 (-0.006, 0.001) | 1.000 / 1.000 | 0.000 / 0.000 | 3% | 3% | 0.001 |
| -20 | 500 | 0.000 / 0.000 (+0.000, 0.000) | 0.003 / 0.003 | 0.870 / 0.779 | 34% | 59% | 0.550 |
| +0 | 500 | 0.000 / 0.000 (+0.000, 0.000) | 0.079 / 0.069 | 0.466 / 0.347 | 12% | 46% | 0.693 |
| +20 | 500 | 0.007 / 0.008 (+0.001, 0.001) | 0.464 / 0.429 | 0.075 / 0.050 | 2% | 23% | 0.437 |
| +30 | 500 | 0.029 / 0.030 (+0.000, 0.001) | 0.703 / 0.673 | 0.019 / 0.013 | 1% | 13% | 0.258 |
| +40 | 500 | 0.103 / 0.105 (+0.001, 0.002) | 0.874 / 0.850 | 0.004 / 0.002 | 2% | 7% | 0.108 |
| +50 | 500 | 0.259 / 0.259 (-0.001, 0.002) | 0.963 / 0.953 | 0.000 / 0.000 | 3% | 5% | 0.038 |
| +65 | 500 | 0.598 / 0.597 (-0.002, 0.002) | 0.996 / 0.994 | 0.000 / 0.000 | 8% | 8% | 0.005 |
| harm_scripted-0.05 | 500 | 0.349 / 0.347 (-0.002, 0.001) | 0.996 / 0.995 | 0.000 / 0.000 | 2% | 2% | 0.003 |
| harm_scripted-0.07 | 500 | 0.179 / 0.180 (+0.001, 0.001) | 0.995 / 0.993 | 0.000 / 0.000 | 1% | 1% | 0.005 |
| harm_scripted-0.10 | 500 | 0.039 / 0.040 (+0.001, 0.000) | 0.995 / 0.994 | 0.000 / 0.000 | 0% | 0% | 0.005 |
| harm_frozen-0.07 | 500 | 0.184 / 0.184 (-0.001, 0.001) | 0.996 / 0.994 | 0.000 / 0.000 | 1% | 1% | 0.004 |
| harm_guard_scripted-0.10 | 500 | 0.335 / 0.332 (-0.003, 0.001) | 0.996 / 0.995 | 0.000 / 0.000 | 3% | 3% | 0.005 |

**V5 family** (8 seeds + control arm C (+7), KILL on the H - C Welch UBs < +25 / +40; 10,000 replicates per cell; paired SE of seq - fixed in parentheses)

| effect | tau^2 | P(GO) fixed / seq (diff, SE) | eff. crossing any look, non-binding: fixed / seq | P(KILL) fixed / seq | saving seq | saving seq+NO_GO(50) | P(NO_GO) |
|---|---|---|---|---|---|---|---|
| -20 | 0 | 0.000 / 0.000 (+0.000, 0.000) | 0.000 / 0.000 | 0.997 / 0.985 | 48% | 65% | 0.423 |
| +0 | 0 | 0.000 / 0.000 (+0.000, 0.000) | 0.074 / 0.066 | 0.839 / 0.718 | 23% | 61% | 0.762 |
| +20 | 0 | 0.000 / 0.001 (+0.001, 0.000) | 0.819 / 0.797 | 0.196 / 0.124 | 3% | 38% | 0.654 |
| +30 | 0 | 0.010 / 0.012 (+0.002, 0.001) | 0.979 / 0.975 | 0.028 / 0.017 | 1% | 20% | 0.369 |
| +40 | 0 | 0.111 / 0.105 (-0.006, 0.002) | 1.000 / 1.000 | 0.002 / 0.002 | 2% | 9% | 0.141 |
| +50 | 0 | 0.431 / 0.412 (-0.019, 0.002) | 1.000 / 1.000 | 0.000 / 0.000 | 6% | 7% | 0.032 |
| +65 | 0 | 0.875 / 0.867 (-0.008, 0.001) | 1.000 / 1.000 | 0.000 / 0.000 | 15% | 15% | 0.001 |
| kill_threshold | 0 | 0.014 / 0.016 (+0.002, 0.001) | 0.991 / 0.987 | 0.019 / 0.011 | 1% | 17% | 0.312 |
| harm_scripted-0.05 | 0 | 0.425 / 0.429 (+0.004, 0.001) | 1.000 / 1.000 | 0.000 / 0.000 | 3% | 3% | 0.002 |
| harm_scripted-0.07 | 0 | 0.183 / 0.185 (+0.002, 0.001) | 1.000 / 1.000 | 0.000 / 0.000 | 1% | 1% | 0.001 |
| harm_scripted-0.10 | 0 | 0.019 / 0.019 (+0.001, 0.000) | 1.000 / 1.000 | 0.000 / 0.000 | 0% | 0% | 0.002 |
| harm_frozen-0.07 | 0 | 0.182 / 0.185 (+0.003, 0.001) | 1.000 / 1.000 | 0.000 / 0.000 | 1% | 1% | 0.001 |
| harm_guard_scripted-0.10 | 0 | 0.467 / 0.469 (+0.002, 0.001) | 1.000 / 1.000 | 0.000 / 0.000 | 5% | 5% | 0.001 |
| -20 | 500 | 0.000 / 0.000 (+0.000, 0.000) | 0.001 / 0.001 | 0.949 / 0.890 | 41% | 65% | 0.516 |
| +0 | 500 | 0.000 / 0.000 (+0.000, 0.000) | 0.095 / 0.088 | 0.610 / 0.471 | 16% | 58% | 0.773 |
| +20 | 500 | 0.002 / 0.003 (+0.000, 0.000) | 0.637 / 0.613 | 0.140 / 0.086 | 3% | 33% | 0.566 |
| +30 | 500 | 0.022 / 0.024 (+0.002, 0.001) | 0.884 / 0.868 | 0.042 / 0.025 | 1% | 18% | 0.327 |
| +40 | 500 | 0.112 / 0.107 (-0.004, 0.002) | 0.978 / 0.972 | 0.007 / 0.004 | 2% | 9% | 0.140 |
| +50 | 500 | 0.337 / 0.325 (-0.012, 0.002) | 0.997 / 0.996 | 0.001 / 0.001 | 5% | 7% | 0.044 |
| +65 | 500 | 0.749 / 0.736 (-0.014, 0.002) | 1.000 / 1.000 | 0.000 / 0.000 | 13% | 13% | 0.004 |
| kill_threshold | 500 | 0.034 / 0.034 (+0.001, 0.001) | 0.917 / 0.901 | 0.029 / 0.018 | 1% | 16% | 0.279 |
| harm_scripted-0.05 | 500 | 0.398 / 0.396 (-0.002, 0.001) | 1.000 / 1.000 | 0.000 / 0.000 | 3% | 3% | 0.003 |
| harm_scripted-0.07 | 500 | 0.166 / 0.167 (+0.001, 0.001) | 1.000 / 1.000 | 0.000 / 0.000 | 1% | 1% | 0.004 |
| harm_scripted-0.10 | 500 | 0.018 / 0.019 (+0.001, 0.000) | 1.000 / 1.000 | 0.000 / 0.000 | 0% | 0% | 0.004 |
| harm_frozen-0.07 | 500 | 0.174 / 0.175 (+0.001, 0.001) | 1.000 / 1.000 | 0.000 / 0.000 | 1% | 1% | 0.004 |
| harm_guard_scripted-0.10 | 500 | 0.407 / 0.403 (-0.004, 0.001) | 1.000 / 1.000 | 0.000 / 0.000 | 4% | 4% | 0.004 |


**Summary against the user's criteria:**

* **Type-I error of GO.** P(GO | null) = 0.000 for fixed and sequential alike. The
  error-controlled efficacy crossing under the null is, sequential vs fixed:

  | Family | tau^2 = 0 | tau^2 = 500 |
  |---|---|---|
  | v4 | 0.047 vs 0.054 | 0.069 vs 0.079 |
  | v5 | 0.066 vs 0.074 | 0.088 vs 0.095 |

  It is at or below the fixed design and at or below 0.10. The reviewer also checked
  tau^2 = 5000 and unequal seed variances: still at or below the fixed design.
* **Power.**

  | Effect | GO change, sequential vs fixed | Paired SE |
  |---|---|---|
  | +65 | -0.002 to -0.020 | 0.002 |
  | +50 | -0.001 to -0.020 | 0.002 |
  | +40 | at most -0.009 | 0.002 |

  This is a small loss.
* **False KILL at +30.** The sequential design is at or below the fixed design:

  | Family | tau^2 = 0 | tau^2 = 500 |
  |---|---|---|
  | v4 | 0.004 vs 0.006 | 0.013 vs 0.019 |
  | v5 | 0.017 vs 0.028 | 0.025 vs 0.042 |

  At the KILL threshold the sequential design is also lower:
  * v4 at +20: 0.029 / 0.050 vs 0.049 / 0.075;
  * v5 at a contrast of +25: 0.011 / 0.018 vs 0.019 / 0.029.
* **Expected saving (core GO/KILL design).**

  | True effect | Saving |
  |---|---|
  | -20 | 34-48 % |
  | Null | 12-23 % |
  | +20 | 1-3 % |
  | +40 | 1-2 % |
  | +65 | 8-15 % |

  The target of about 50 % is **not met**; see the headline for why.
* **Expected saving with NO_GO(50).**

  | True effect | Saving |
  |---|---|
  | Null | 46-61 % |
  | +20 | 23-38 % |
  | +40 | 6-9 % |
  | +65 | 8-15 % |

  P(NO_GO) at a true +50 is 0.021-0.044, within the 0.10 claim.
* **Survival and guard harm.** The sequential GO rate is at most +0.004 above the fixed
  design's (section 5).

**Validity limits.**

* The scenarios shift effects uniformly across mixes.
* MI10 is modelled as 1.5 x the H5000 effect.
* The pool mixes three studies' spreads.
* Seed heterogeneity is normal.
* A study whose rule, number of seeds, N or thresholds differ must re-run the simulator for its
  own spec (governance condition 7).

## 10. Independent review (2026-10-07) and fixes

* **Verdict:** GO-with-fixes. The reviewer's own simulations confirmed four things:
  * GO error control at tau^2 in {0, 500, 5000}, including unequal seed variances;
  * false-KILL control at the thresholds;
  * NO_GO at G within 0.10;
  * evaluator / decision identity.
* **Fixed:**
  * **B1:** an invalid look is now INVALID_ANALYSIS / HALT, keeping `status_if_valid`, and the
    audit verdict for it is HALTED, never PASS.
  * The non-binding efficacy metric is now reported.
  * This document's stale sections 0, 3, 4.3, 5 and 8.
  * The guard and control are now correlated in the simulator.
  * Added scenarios: NO_GO at G (+50) and guard harm.
  * The audit now recomputes the prefix-controls flag and checks that flags are consistent.
  * The unaudited shard markers are now stated.
  * Disclosed: the stricter UNGATEABLE efficacy and clause 7 at an interim GO.
  * Paired SEs.
  * Corrected t multipliers.
  * `validate_rule` now rejects hk / welch statistics on single-seed cells.
* **Open:** a cross-check of the simulator's fixed design against the frozen FRP-v4 / v5
  `phase_r_decision`. Those packages live on other branches, so a hermetic test on this branch
  cannot import them.
