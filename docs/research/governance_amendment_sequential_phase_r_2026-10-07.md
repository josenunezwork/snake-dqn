# Governance amendment (DRAFT, for the owner to ratify): sequential Phase R for Tier-1 screens

* **Date:** 2026-10-07.
* **Branch:** `seq-phaser`.
* **Status:** **Pending.** Nothing below applies until the owner ratifies it, and then only to
  studies whose pre-registration opts in.
* **Design:** `docs/research/sequential_phase_r_design_2026-10-07.md`.
* **Code:** `src/evaluation/sequential_phase_r.py` and `research/sequential_phase_r/`.

## 1. What it permits

A **future** FRP-family Tier-1 screen may pre-register a **sequential Phase R**
(method `sequential-phase-r-obf-hk-v1`) instead of the fixed-N Phase R. The design:

* **Looks:** at 1/3, 2/3 and 1 of the planned worlds per (seed, mix), on a look-major
  schedule of whole nested episodes. The prefix controls run in look 0.
* **Early GO:** only on the **full** GO rule. The efficacy clause uses O'Brien-Fleming spending
  of the fixed rule's one-sided 0.10 on the Hartung-Knapp pooled lower bound. Every other
  clause is judged at the stopping look with the pre-registered interim margins.
* **Early KILL:** on Pocock-type repeated upper bounds that spend the fixed rule's 0.10. KILL is
  non-binding for GO, but the protocol follows it.
* **Optional early NO_GO (gate futility):** only if the study declares it and its threshold G.
* **Final-only labels:** GO_UNGATEABLE, RECIPE and PARTIAL are judged only at the final look,
  with the fixed rule's precedence.
* **Evidence:** create-only plan and look receipts, a look gate at every shard start, and the
  independent audit, which must PASS.

## 2. What it does not change

* **Existing studies are untouched:** FRP-v2, v3, v4, v5-H, v5-S, v5-S2, every running or merged
  Phase R, and every strict gate.
* **Authority is unchanged:** Phase R stays Tier-1 and cannot promote. LH-1, the strict
  sequential gate (band v2) and Mac serving qualification are unchanged.
* **Fixed-N stays fixed:** a fixed-N pre-registered run is never stopped early. Stopping one
  invalidates it (the rule of 2026-10-02).

## 3. Conditions on a study that opts in

1. **The rule spec and plan are pinned before the first Phase R shard.** The pre-registration
   carries the rule spec JSON (`validate_rule` schema `sequential-phase-r-rule/v1`), with its
   sha256 and the plan sha256 pinned by test. It is committed before the code that runs it.
   The plan freezes:
   * `N` and the look sizes;
   * the GO / KILL alphas and spendings;
   * the margins;
   * the protective clauses;
   * the NO_GO threshold, if any.
2. **The plan is frozen before any Phase R shard.** `plan.json` is written create-only before any
   Phase R shard. Every look-k shard records the `look_gate` binding. No look-k+1 work starts
   before the look-k receipt says CONTINUE.
3. **Stops are followed.** A STOP is final:
   * a GO, KILL or NO_GO at an interim look ends the study;
   * an override is a recorded protocol deviation, which leaves the GO error unaffected because
     futility is non-binding;
   * there is no resume, rerun or relabel of a look.
4. **Halts and incomplete looks.** An INVALID_ANALYSIS (HALT) receipt ends the analysis, and the
   owner decides. An incomplete look is never analysed, and the study is then INCOMPLETE.
5. **Audit.** The audit (`python -I research/sequential_phase_r/audit.py --root ... --record-dirs
   ...`) must PASS before the study's result is reported. `UNCLOSED`, `HALTED` or `FAIL` makes the
   study INVALID_ANALYSIS. Every invalid look is INVALID_ANALYSIS / HALT, whatever its data
   say.
6. **Disclosure after an early stop.** The report must state:
   * point estimates are biased upward and are descriptive only;
   * final-only labels are unresolved;
   * candidate selection used the stopping look's worlds.

   The strict gate plays fresh worlds, so the bias does not reach promotion.
7. **The OC study.** The OC report (design doc section 9) is part of the pre-registration. A
   study whose design departs from the simulated families must re-run `simulate.py` for its own
   rule before ratification. Departures include a different number of seeds, different
   thresholds or a different `N`.

## 4. Decisions the owner must make

* **D1: ratify the core design** (GO / KILL only). In simulation it keeps the efficacy type-I
  error at or below the fixed design's and keeps GO power within a few points of it. The false
  KILL rate at +30 and at the KILL threshold is at or below the fixed design's. The saving is
  about **12-23 % near the null**, **~0-3 % at +20..+40**, **~8-15 % at +65** and **~34-48 %**
  for a clearly worse candidate (design doc section 9). The core design **does not reach the
  expected ~50 %**. The design doc explains why: clause (7) makes a clear GO rare, KILL is
  defined at the fixed rule's precision, and most runs end in final-only labels.
* **D2: allow NO_GO, and choose a default G.** The reference value is G = 50 at cap 600. NO_GO
  reaches roughly **46-61 % saving near the null** and **23-38 % at +20**. The price: runs that
  the fixed design would label GO_UNGATEABLE, RECIPE, KILL or PARTIAL end as NO_GO_EARLY, and
  their labels are reported only descriptively. A truly gateable candidate (effect >= G) is
  stopped with probability <= 0.10 (simulated: 0.02-0.04 at a true +50).
* **D3: the margins.**
  * protective clauses: z 1.645, shape sqrt(1 - t);
  * other point clauses: z 1.2816, shape 1 - sqrt t.

  Alternatives are in the design doc.
* **D4: the pod runner.** Sequential studies on RunPod run blocks per look, with the look gate
  enforced by `podrun`. `research/pod_phaser` is not changed by this amendment. Wiring it is a
  separate, reviewed change.

## 5. Ratification record (to be filled by the owner)

- Decision: (ratified / ratified with changes / rejected)
- Options: D2 NO_GO allowed (yes/no), default G = ...
- Changes, if any:
- By / date (UTC):
