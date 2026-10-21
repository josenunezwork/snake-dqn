# Colab lanes — JUDGMENT (2026-10-20 ~21:30 PT)

Single independent judge: an Opus stand-in (Fable is at its usage limit; the owner chose Opus). I wrote none of the six
Astra answers. Inputs: `QUESTION.md` and `astra_{crit,search,learn,eval,rl,skeptic}.md` in this folder. I checked their
load-bearing claims against main at `fe6e0830` (listed in §0). This is a plan and a set of rulings on *what to
register*. It does not ratify any study: each lane below still needs its own pre-data registration and one judge
approval (STATUS.md:24). Colab use itself is pre-approved. Releases and serving changes still need the owner.

## 1. Verdict

1. Path B stays the fastest route; Colab never delays its Mac order, and a PB GO goes straight to strict.
2. Claim A is a design problem first: repair the loss and update dose (L3 2×2) on excluded A5 data before any big bank.
3. Fund nine gated lanes, ceiling ~1,000 u (§2). Each opens only on the prior decision's GO.
4. No funds: two-deviation beam, outcome head, A100/G4, PQN/self-play, 5-replicate recipe studies (§6).
5. Honest forecast: ~500 u expected on R0 GO, ~200 u if both teacher premises fail. 38 u/day is met only if every
   Claim A gate passes. Unspent units are reported, not padded.
6. Order: loss ruling (Oct 21) → R1 labels (GO) or re-scoped 0-D (NO-GO) → projection after PB disposition → R2 and
   coverage → tower, R3 on trigger → finalist frozen Nov 9 → replication → Mac strict ~Nov 12–16.

## 0. Verified facts that decide the disagreements

| Claim | Verified at | Effect |
|---|---|---|
| Balance 1,153.84 u at 19:18 PT; L4 observed at 1.54 u/h; planned work ≈ 126–171 u (≈ 235 with extras); 920–1,030 u have no lane | `docs/research/ops/COLAB_UNITS_LEDGER.md:10-18, 96-118` | The gap is real |
| Frozen loss: KL, hinge and advantage-MSE share one `sample_weight`, so raising decisive-row weight cannot change the conflict between them | `research/t34_exit_b8_fit/loss.py:43-64` | Fix the loss, not the weights |
| Free-Q probe on 225 decisive rows: 0.4% flip to a\*, 97.8% stay bare. 21/39 worlds carry decisive rows. The default split leaves 0 held-out decisive rows | `snake-dqn-artifacts/fable-panel-claima/design.md:14-20` | Hard blocker; a held-out minimum is needed |
| 1:1 searched:unsearched sampler gives ~1 step/epoch on 39 worlds and ~12 steps/epoch (~60 per fit) on 800 worlds | `research/t34_exit_b8_fit/data.py:80-91`; `design.md:65-78` | Freeze an explicit update dose |
| Round 0-D's `analyze.py` imports the Round 0 `summarize`, which still has the absolute floor `0.005 <= boost` | `research/t34_round0d/analyze.py:19-27`; `research/t34_round0/analyze.py:170` | 0-D needs the relative floor before ratification |
| Dense R1 at 560 worlds: the draft gives 49–75 u against its own 30 u cap; the ops review estimates 110–225 u (r ≈ 5–10) | `research/t34_round0d/PREREGISTRATION_ROUND0D.md:257-270`; `fable-panel-round0d/ops.md` findings 1, 6 | Re-cap dense R1 by yield |
| Dense acting gives ~0.66 decisive rows/world, against 5.8 for teacher_v2 shadow | `fable-panel-round0d/science.md:60-61` | Dense labels are scarce; price them by yield |
| §5 amendment: R1 labels may follow "the published registered `decision.json` = GO only". The handoff says to report `amendment-boost-floor/decision.json`, "never the registered decision.json" | `SUCCESS_CONTRACT.md:80-88`; `HANDOFF_2026-10-20.md` update ~19:58 | Wording conflict; ruling needed (§7b) |
| T34 hard cap ≈ 270 u; R1 non-terminal; R3 only if R2 ≥ R1 + 15 mass or + 0.03 survival | `FINAL_PLAN.md:7, 22, 62-64, 89` | Raise the cap by amendment (§7d) |
| Unpaced retention arm: ≤ 42.5 u, at most 3 concurrent lanes including other jobs, dispatchable only at `81c54731` | ledger:93; `docs/research/specs/2026-10-20_t34_unpaced_arm_colab.md:7`; STATUS.md:22 | Keep one lane free Oct 22–25 |
| Outcome head KILL (AUC 0.747, 46.2%); two-deviation pooled KILL (0 novel rescues) | `research/t35_outcome_head/RESULT.md:25`; `research/t35_two_deviation/RESULT_EXTEND.md:3` | No successor funded |
| Old portfolio L1/L4 prices are obsolete (A5 v2 is ~0.036 u/teacher world); its L4 "no-rescue → two-deviation" branch is dead | `compute_portfolio_FABLE_JUDGMENT.md:34-39`; CAMPAIGN_LOG:59 | This plan replaces L2–L5 and R |

**Disagreements and how I resolved them.**
- **(a) Size of data scaling.** learn proposes 800→2,400→7,200 worlds and rl 800→2,400→5,600, against
  crit/skeptic/eval's "no scaling before projection works". I fund one learning-curve lane, gated on a viable
  projection: step 1 to 2,400 worlds, step 2 to 7,200 only on a frozen rule. Sparse labels cost ~0.036 u/world, so
  the curve is cheap, and it answers a real question: is the student coverage-limited?
- **(b) Five-replicate R2 or recipe study** (eval lane 4, learn D4). Not funded. The serving claim is about one fixed
  checkpoint, and P7 makes the paired world the unit for that (STATUS.md:17). Five replicates answer a recipe-science
  question that does not change what we serve. The fixed-finalist independent replication (L8) protects Mac time
  instead.
- **(c) Failure-window coverage** (search lane 2, ±30 decisions). Deferred to reserve. It needs a new entrypoint
  (Sol days) and ~72–83 u for an unmeasured expansion. The independent-world curve answers the cheaper version of
  the same question first. It can be released from reserve only if the curve shows held-out decisive agreement
  limited by coverage *near* failures.
- **(d) Search with the improved student as continuation** (search lane 4). Excluded this campaign. It creates a new
  controller identity that needs its own Mac paced test, which collides with the Nov 10–16 strict-gate window.
- **(e) Full-arena tower** (learn D3, rl lane 4). Funded as one controlled ablation without auxiliary heads, because
  the code already exists (`research/t34_exit_b8_fit/network.py:38-86`).
- **(f) Deadline successor cap** (60–200 u in the answers). Set at 100 u, only if attribution shows deadline or
  staleness loss.
- **(g) Round 0-D cap** (40 → 110/120 u). Re-scoped to ≤ 120 u with a frozen r→action table (§7e).

## 2. Lane table

Caps are total lane ceilings in priced units, including setup, exports, failed attempts and a ~20% internal reserve.
They replace the old portfolio's L2–L5 and R. They are not forecasts. Every lane uses L4 through the qualified
supervisor `research/colab_cli` (QUALIFICATION on dab00823). At most 3 assignments run at once across the account.
**Sol builds; Claude operates** (registration drafting, dispatch, publish, settle). "R0" means the *governing amended*
Round 0 verdict (§7b). "PB" means the Path B `decision.json` (GO/KILL/INCONCLUSIVE). INCONCLUSIVE routes as KILL for
Colab purposes, with no extension.

| Lane | Decision it produces | Runtime | Cap (u) | Earliest start (PT) | Depends on | Kill rule | Owner |
|---|---|---|---:|---|---|---|---|
| **L0 Existing deciders** (Round 0 remainder; Claim A L4 timing qualification; PB unpaced arm) | Round 0 verdict; whether fits are affordable; PB descriptive arm | L4 ×1–2 | existing: ≤ 43 / ≤ 5 / ≤ 42.5 | now; unpaced arm when the PB study starts (~Oct 22) | existing registrations | existing rules only; no retries beyond the registered mechanical rerun | Claude operates |
| **L1 Projection diagnostics (D0)** | Which corrected loss and update dose can flip decisive rows on *excluded* A5 data (free-Q ≥ 99% train flips, small-fit decisive agreement trend); frozen production dose and coefficients | L4 ×1 | **12** | Oct 21–22 | Claim A loss ruling (§7a) | no corrected objective reaches 70% decisive agreement even on the permissive diagnostic → close projection; Claim A then rests on the tower question only (L6) | Sol builds; Claude operates |
| **L2 R1 labels, sparse** | Label bank for the projection: 800 train + 200 whole-world held-out, fresh collision-audited seeds, β=1, teacher_v2 | L4 ×1–2 | **60** | ~Oct 22 (R0 GO published + R1 approved) | R0 GO; §7b/§7c rulings | held-out < 60 decisive-bearing worlds or < 300 decisive rows → PRECONDITION_FAILED (reported; no top-up without a new registration); integrity faults per the registration | Sol builds entrypoint; Claude operates |
| **L3 Projection 2×2 (D1)** | {frozen loss, corrected loss} × {trainable B8, frozen trunk + zero-init residual}: is there one viable design? Selected on 96/32/32 paired H5000 screens, Holm across arms | L4 ×1–2 (fits are GPU-bound) | **120** | ~Oct 24–25 | L1 pass; L2 bank (or dense R1); **PB disposition published** | per-fit FINAL_PLAN:89 tripwires (70% held-out decisive, ≤ 5% any length decile, boost ≤ 1.5×, …); the frozen-loss cells stop at their tripwire; no viable arm, or every arm has mass UB90 ≤ 0 → **close Claim A** (closes L4, L5, L6, L8) | Sol builds; Claude operates |
| **L4 R2 on-policy (+R3 on trigger)** | Does the selected student keep its gain on its own states (β=0.5)? R3 (β=0) only if R2 ≥ R1 + 15 mass or + 0.03 survival | L4 ×1–2 | **150** (R2 ≤ 80, R3 ≤ 70) | ~Oct 27 (R2); ~Nov 3 (R3) | L3 viable design | R2 scientific failure closes the recipe; no R3 without its trigger; no checkpoint hunting | Claude operates |
| **L5 Coverage learning curve** | Is the student limited by independent-world coverage? 800 → 2,400 (step 1), then → 7,200 (step 2) with the same objective, dose and evaluation | L4 ×1 | **280** (step 1 ≤ 100; step 2 ≤ 180) | ~Oct 27 | L3 viable; sparse teacher only (dense needs its own pricing) | step 2 only if step 1 gives ≥ +5 pp world-weighted held-out decisive agreement **or** ≥ +10 pooled mass with adjusted LB90 > 0, no guard regression; otherwise close data scaling | Claude operates |
| **L6 Full-arena tower ablation** | Is spatial information missing? Tower vs no-tower, same corrected loss, labels, seeds and dose; no auxiliary heads | L4 ×1 | **80** | ~Oct 30 | L3 viable, or L3 failing with high train / low held-out agreement | < +15 pooled mass with LB90 > 0 vs matched no-tower, or drift breach, or Mac p95 latency > 11.688 ms → close | Sol (fidelity check); Claude operates |
| **L7 Round 0-D (dense premise) + dense R1** | Does exact T3-dense acting meet the registered premise bars on fresh 3-mix worlds (240 pairs)? Then a yield-sized dense R1 bank | L4 ×2 (keep one lane for the unpaced arm) | **120** (study) + **200** (dense R1) | ~Oct 22 after rescope ratification | R0 **scientific** STOP/INCONCLUSIVE; §7e/§7f | conservative r forecast > cap or wall > 24 h, or r < 5 → KILL before production; fidelity fail → HOLD; scientific non-GO → **close the search-teacher Claim A family**; dense R1 only if the yield forecast meets L2's held-out minimum within 200 u | Sol rescopes; Claude operates |
| **L8 Deadline-delivery successor** | Can one frozen scheduling/branch-reuse change recover late or stale rescues? | L4 ×1 | **100** | ~Oct 25 | PB terminal **and** attribution shows late/stale delivery as a material loss; independent of R0 | attribution shows absent rescues → no launch; < +10 pp failure-cohort rescues or > 5% lost rescues → stop; 96-world 32/mix fresh screen mass UB90 vs v2 ≤ +20 → stop; any survivor still needs a new Mac paced test | Sol builds; Claude operates |
| **L9 Finalist replication + census** | Does one frozen finalist hold on an untouched bank (128 paired/mix), and what failures remain? | L4 ×1–2 | **100** (census-only ≤ 15) | Nov 6; freeze finalist by **Nov 9** | a finalist from L3–L6 or L8; never blocks a strict-eligible PB candidate | one fixed bank, one look, no extension or substitution; failure → no Mac gate slot for that candidate | Claude operates |
| **Reserve** | Only a named, priced, judge-approved packet (e.g. the window-coverage arm under rule (c), a mechanical rerun) | — | **100** | weekly review | — | never repeats a scientific KILL; never released to hit a daily debit | Claude |

Ceiling sum: GO branch ≈ 12+60+120+150+280+80+100+100+100 = **1,002 u** (plus ~90 u existing L0 liabilities).
On the NO-GO branch, L7 (≤ 320) replaces L2 (60), and L5 step 2 is then admissible only against the live balance at
measured dense prices. Every admission is checked against the **live** `colab usage` balance minus outstanding
liabilities.

## 3. Branch table

| | **PB GO** | **PB KILL / INCONCLUSIVE** |
|---|---|---|
| **R0 GO** | Mac: PB → strict gate → serving at once (release needs the owner's OK). Colab: L2 (during PB) → L3 → L4, L5, L6 → L9 for a Claim A finalist. **L8 not funded**, because PB passed and the Mac is reserved for strict. L7 is shelved. | Colab: L2 → L3 → L4, L5, L6 → L9. L8 only if attribution shows deadline loss. Mac strict is reserved for the Claim A finalist ~Nov 12–16. |
| **R0 scientific STOP/INCONCLUSIVE** | Mac: PB → strict → serving. Colab: L7 (0-D). On a 0-D GO: dense R1 (yield-gated) → L3 → L4/L6 (L5 only at measured dense prices). On a 0-D non-GO: Claim A search-teacher family closed; L9 census ≤ 15 u; **spending stops**. | Colab: L7. On a 0-D GO: as in the left cell. On a 0-D non-GO: L8 only on its attribution trigger, otherwise **no funded lanes remain**. Report the unspent balance and open a new advice round on a *new mechanism*. Do not enlarge killed studies. |
| **R0 controller fault / integrity INCONCLUSIVE** | Mechanical branch (NO-GO plan §3): fix, then re-register Round 0 on a fresh namespace (≤ ~45 u from reserve). No teacher substitution. PB → strict as usual. | Same mechanical re-run; L8 on its trigger. |

The re-scoped 0-D must be **ratified on main before the Round 0 sole analysis** (NO-GO plan rule), so that the
branch choice is mechanical.

## 4. Week-by-week units (target ≈ 38 u/day; max ≈ 111 u/day at 3 L4s × 24 h × 1.54)

| Week (PT) | Target | Ceiling if everything passes (R0 GO) | Expected (R0 GO, typical kills) | Both premises fail (R0 + 0-D non-GO, no L8) | Main work |
|---|---:|---:|---:|---:|---|
| Oct 21–27 | 266 | ~165 | ~120 | ~175 (0-D runs here) | R0 rest, qualification, L1, unpaced arm, L2 or L7, start of L3 |
| Oct 28–Nov 3 | 266 | ~310 | ~170 | ~15 | rest of L3, L4 R2, L5 step 1, L8 if triggered |
| Nov 4–10 | 266 | ~380 | ~120 | 0 | L4 R3 (trigger), L5 step 2 (rule), L6, freeze finalist Nov 9 |
| Nov 11–17 | 266 | ~120 | ~80 | 0 | L9 replication; Mac strict/serving |
| Nov 18–20 | 114 | ~10 | ~10 | ~5 | closeout, audit, settle receipts |
| **Total** | **~1,178** | **~985** | **~500** | **~195** | |

**Honest statement.** Spending the whole balance on decision-producing work needs Round 0 (or 0-D) GO, a viable
projection, and both scaling rules to fire. Expected spend is about 40–45% of the balance. If both teacher premises
fail, about 80% of the balance stays unspent, and that is the correct outcome. The weekly review reports actual
spend against both the target and this forecast. Unspent units are **not** moved into extra arms, worlds or epochs.

## 5. First three actions on Tuesday Oct 21

1. **Claude operates the deciders on the Mac's critical path.** Finish Round 0: publish chunk 001, dispatch and
   publish chunk 002 (as in the HANDOFF). In parallel, use a free L4 for the ≤ 5 u Claim A timing qualification.
   When Phase 0 ends (~10:00–13:00), run the **Round 0 Mac replay → sole analysis → condition-5 gate → amended
   `decide` first**, then the Path B timing gate (`SOURCE_COMMIT=81c54731`, operator-02). Dispatch the unpaced arm
   when the PB study starts.
2. **Claude drafts the Claim A L3 amendment; one independent Opus judge rules (target Oct 21 ~14:00).** The
   amendment covers rulings §7a–§7d: corrected loss with KL/adv-MSE only on non-decisive rows and CE + hinge on
   decisive rows; separate decisive/anchor streams; an explicit update dose frozen from excluded A5 data; whole-world
   held-out minimums; R1 = 800 + 200 worlds; binding "Round 0 GO" to the governing amended verdict; and the T34 cap
   raised to these lanes. **Then Sol** implements L1 (D0 diagnostics on the excluded A5 exports) and the R1
   labels-only entrypoint, so that a GO can dispatch Wed evening.
3. **Sol re-scopes Round 0-D (§7e–§7f) and Claude puts it through the Opus panel and judge before the Round 0 sole
   analysis (Wed midday).** Scope: relative boost floor, r→action table, A5-D resized, throughput estimator from
   completed process-seconds, replay timeout sized from A5-D, dense-R1 yield rule. Sol is run via
   `codex-handoff.sh` with `CODEX_MODEL` set. If this misses Wed, a scientific NO-GO waits for it. Nothing else
   substitutes.

## 6. Exclusions

- **Two-deviation beam or any wider/deeper beam**: pooled KILL, 0 novel rescues (`RESULT_EXTEND.md:3`).
- **Outcome head at larger scale or as an R1 arm**: KILL (AUC 0.747). Its data-starvation caveat does not earn a rerun.
- **A100 / H100 / G4**: host-bound; B8 forward is ~8–13% of the time (`a100/SYNTHESIS.md:6-7`). G4 would need ≥ 4.35–5.65× measured useful throughput for parity, and that is unmeasured. The only reopener is the existing A100 fit trigger (L4 fit ≥ 60 min, GPU leg ≥ 40%, ≥ 1 h saved), debited inside L3.
- **Five-replicate R2 or recipe-population studies**: they do not change what gets served (fixed checkpoint, P7 unit = paired world).
- **PQN / PPO / self-play / league restarts**: no B8-compatible implementation exists (`src/scripts/train_pqn.py` uses a separate raster stack). T23 league fine-tune was a KILL. These need a new campaign, not leftover units.
- **Search with the improved student as continuation**: needs a new Mac paced identity test that collides with the strict window. Not this campaign.
- **Failure-window (±30) coverage**: reserve only, under rule (c).
- **Scaling the frozen loss, or collecting a large bank before L1/L3**: predicted FIT_STOPPED that spends units and decides nothing (`design.md` finding 1).
- **Dense R1 at sparse prices** and **any 7,200-world step with dense labels** without a measured price.
- **Any spend made to meet a daily rate**: idle allocations, extra seeds or epochs, extra looks, sample extensions after an inconclusive result.

## 7. Registration and contract rulings needed (each prospective, pre-data, one independent judge)

- **(a) Claim A L3 amendment** (blocks any production fit). Adopt the 2×2 from portfolio L3. Corrected objective:
  KL(prev‖θ) + 0.01·adv-MSE on anchor rows only; CE(a\*) + forbidden-action hinge on decisive rows only.
  `all_die_delay` rows are excluded from the anchor quota. Draw anchors from the full shadow-confirm pool in a
  separate, normalized stream. Freeze an explicit optimizer-update dose (starting proposal 2,048 updates in 4 blocks,
  batch 1,024 at 128 decisive / 896 anchor), set from excluded A5 data only. Tripwires are unchanged; do **not**
  weaken the 70% held-out decisive bar. The frozen-loss cells remain controls and stop at their tripwire.
- **(b) "Round 0 GO" binding.** The §5 amendment says "the published registered `decision.json` = GO only". The
  governing amendment emits `amendment-boost-floor/decision.json`, and the registered file carries the superseded
  absolute floor. Ruling needed: for §5, "Round 0 GO" means the governing amended decision (merge 30b2739a), which
  is published only after the condition-5 gate passes. Without this ruling, R1 cannot be dispatched cleanly on a GO.
- **(c) R1 shape.** 800 training + 200 whole-world held-out (fresh, disjoint, collision-audited). Minimum held-out
  support: ≥ 60 decisive-bearing worlds and ≥ 300 decisive rows; otherwise PRECONDITION_FAILED, not a learning result.
- **(d) T34 cap.** FINAL_PLAN:7/64 hard cap ≈ 270 u is raised to the sum of the admitted lane caps above. Each lane
  keeps its own cap. Money from a killed lane does not transfer automatically.
- **(e) Round 0-D rescope.** Import the relative floor (0.5× paired bare ≤ candidate ≤ 0.10), not
  `t34_round0/analyze.py:170`. Replace the 40 u gate with a frozen r→action table: ratify if expected ≤ 120 u
  **and** forecast wall ≤ 24 h on the lanes available; KILL if r < 5. A5-D: timeout ~5 h, cap ≤ 10 u, sized from
  the slowest fidelity world, with a pre-registered HOLD path for a Mac↔L4 float32 tie flip. r is computed from
  completed process-seconds, not a 20–40-min window. Replay timeout ≥ max(estimate + 25 min, 2× estimate), or split
  the replay over two fresh runtimes. Ceiling formula per ops finding 5.
- **(f) Dense R1 cap.** Replace "≤ 30 u" (no plausible r meets it) with a yield rule: measure decisive rows per world
  on the sealed 240 worlds, set N_fresh = min(560, target / yield) so that the L2 held-out minimum is reachable,
  with ≤ 200 u and ≤ 24 h on 3 lanes. The §5 overlap permission covers only the sparse Round 0 GO route. Dense R1
  needs its own ordering clause (labels only until the PB disposition).
- **(g) Path B INCONCLUSIVE.** Routes as KILL for Colab lane selection (L8 attribution may still run). No extension.
- **(h) New-lane registrations** (L5 steps, L6, L8, L9). Each freezes before its data: banks, decision family,
  Holm/multiplicity, joint operating characteristics at +40/+60/+90 mass (eval's point: a +40 point bar passes
  ~50% at a true +40), and an explicit INCONCLUSIVE outcome.
- **Not needed:** any change to the Mac-only strict gate, serving qualification, survival band v2, or the owner's
  release authority.
