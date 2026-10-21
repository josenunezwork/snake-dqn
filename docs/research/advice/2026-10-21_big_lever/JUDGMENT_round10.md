# Round 10 judgment — big-lever research rounds (final: the plan under adversarial review → PLAN.md)

Judge: Fable (round-10 judge), 2026-10-21 **~03:10 PT** (wall clock `date` = 02:59 PDT at start). Inputs: `../CHARTER.md`,
`../round-01..09/JUDGMENT.md`, `../LIVE_FACTS.md` (newest 02:55 PT), `../QUEUE.md`, `QUESTION.md`, `astra_01..10.md` (advisors read
snapshot `4f15254f`, which is also main HEAD). Repo verified read-only at `4f15254f`. Artifact trees under `big-lever/exp/` read
(light commands). Sealed study not opened. No jobs, Colab or tmux. This judgment and `../PLAN.md` are the round's products.

Facts newer than LIVE_FACTS, found in the artifact trees and folded in as authoritative:
- **E4 FAILED mechanically at 02:54 PT** (`exp/bl-e2/run-e4.log`: `INCOMPLETE: TransferError(... total upload time cap reached at upload
  .../checkpoints/best_apex_stage1_fs.pth ... supervisor.py:1301)`, EXIT=1). Same failure class as E3 (upload-time cap). LIVE_FACTS'
  "E4 dispatched 02:46" is stale: **E4 is not running and the E2 sizing precondition is unmet.** Sol's E3 fix (`9f203c4f`, hash-bound
  runtime reconstruction, checkpoints fetched through the public-input transport) is the fix class E4 and every checkpoint-carrying
  manifest (E2, E16, O2, L2′ chunks, Y1) need. Three mechanical failures in ~24 h (E1b dtype, E3 upload, E4 upload) on three slots.
- **E1b relaunch is prepared, not launched** (`exp/bl-e1b/cuda-dtype-fix-20261021-01/preparation-receipt.json` 02:48, commit `88adb3e2`);
  the only L4 run (01:19) printed `Refused('remote terminal failed or foreign generation')`.
- **E3 recovery-02 is prepared** (`exp/bl-e3/recovery-02/DISPATCH.txt` 02:52; requires create-only terminal receipt + `stop_verified`).
- **Y1's fixture spec landed** (`4677dd02`, `research/bl_served_search/RULER_FIXTURES_08.md`: nine constructed long-body fixtures,
  L 800/1,200/1,500 × 3 mixes, FRP3 s12 + v8 opponents, ≥ 30 real H300 batches per root, 12 one-thread L4 workers; Mac half is a
  one-batch smoke that "cannot qualify throughput or latency"). `exp/bl-l2/ruler-allocation-08/` holds the seed audit (02:53).
- **O2 is not built** (`snake-dqn-artifacts/t34-o2-20261021/` empty). Its registered bar (`CAMPAIGN_LOG.md:101`): (D+O − D) LB90 > 0
  **and** +40 / LB90 > 0 vs B8 — not a positive point difference (10 right).
- **Colab cap 3 measured** (`colab-limit-test/result.log`: `MAX_CONCURRENT_L4=3`). With E3/E4 failed, E1b not relaunched and O1 done,
  slots are likely free now — but "likely" is not a receipt; Z-1 requires `colab usage` + terminal-export receipts before assignment.
- Phase 0 ETA Wed 10:00–13:00 PT (`CAMPAIGN_LOG.md:137`) — an ETA, not an idle receipt. Owner's two asks (serving W≤4/K≤8; probe-only)
  still PENDING; silence ⇒ W1/K1.

## (a) Bottom line

1. **The skeleton survives as structure and fails as an operator interface — ten of ten, from ten directions.** Round 9's three
   admissions, `C_strict` ceiling, two-cutoff fix, probe-only ask, Z-4, 320-u target, one-family-per-artifact and R-as-research all
   stand. What no advisor could sign is the *text*: no gate has a command, receipt path and failure print (01); the serial calendar
   has −4 h at Thursday's E14 if Y1 is accepted at its own deadline and −0.9..−2.9 h in the strict stage once the package's reserved
   8 replay + 2 semantic unpaced hours are charged after Friday (02, 07, 09; `PREFLIGHT.md:94-98` verified); Codex-V needs 36–42
   review-hours against two 8-h windows (03); the clock seam is *worse* than Round 9 said (04); the power sentence is false in one
   direction (05, 10); Z-4's option (b) cannot reach .60 by construction (06); the release conjunction does not bind a controller
   digest (07); Z-0 as "readiness code" would still ship a full-bank runner with `shard-count=1` (08); the §6 reducer has no total
   domain (09); §6's harvest and budget clauses mistake O1 for admission and a priority list for a ceiling (10). PLAN.md adopts all
   of it, item by item (§c), as executable text.
2. **Clock seam, corrected (04, verified):** `controller.py:60` checks the single `cutoff_ns` at *submission* (`event.time_ns >=
   request.cutoff_ns` in the admit path), `publish()` (`:118-138`) records `result_visible` with the worker's `completed_ns` and
   performs **no cutoff comparison**, and consumption (`:268-277`) compares commit / validated / **worker-completed** times against
   the same cutoff. Round 9's "checked at visibility" was wrong: there is no visibility timestamp at all. Splitting the field at
   consumption leaves publication unguarded. Z-2 becomes **Z-2′**: a visibility timestamp taken at the coordinator after IPC decode,
   `publish_cutoff = T(r−2)` enforced in `publish()`, `consume_cutoff = T(r+1)` at consumption, integer monotonic tick mapping, the
   four-case fixture run through **both** controller and the independent replay (`replay.py:241-281` also accepts visibility without a
   deadline), accepted **before any qualifying L1′ timing** — not by Friday (03, 04 vs the skeleton; §c3).
3. **Power statement, corrected (05, 10 right; Round 9 §a3 partly withdrawn).** "Cannot distinguish +40 from 0 on any single stage"
   is false: a mass > 0 test at α = .05 with 128 worlds/mix (SE 14.65) has ≈ 86 % power at +40, and L4's observed +40 gives LB90 ≈
   +2.4. What is true: each paced stage's **point ≥ 40** clause is a ≈ 50 % filter at a true +40 and ≈ 90 % at +80; stacking three
   such filters gives ≈ 12.5 % at +40; L2′ (unpaced, Colab, FUND needs frozen ≥ 42.6 ∧ mixed ≥ 34.3 ∧ pooled ≥ 40) is ≈ 20–30 % at a
   uniform +40 and rejects a (0, 0, 240) pooled +80 ≈ 99 % of the time (05's counter-example: cross-mix funding filters can discard a
   pooled winner by design, and that is the registered intent — scripted-only gains are not the lever). The 5–8 % / 60–65 %
   figures are **planning sensitivities, UNVALIDATED until Y6** runs 05's scenario set. PLAN.md §0 says exactly this and keeps every
   bar (Round 8 §c5 stands; nothing is loosened).
4. **Z-4 narrows to one live design (06, arithmetic checked against `E9_RULE.md:43-55`).** E2 and E6 already have separate .05 routes
   (funding = union), so "E6 reported, not Holm-coupled" (option b) removes a route and cannot raise funding; and the scripted-survival
   sign guard caps funding near .57 at survival Δ = 0 under the .9-correlation scenario regardless of N. Therefore **.60 at +40 with
   survival Δ = 0 is unreachable by any N**; the only lawful form is 192/64/64 fixed (SE 16.6, first-Holm mass threshold 32.5,
   mass-only power .67) with the survival alternative declared **pre-data** at the contract's scripted bar (+.03), where E9's own
   table shows .42–.61 at 128/48/48. Sequential (c) saves cost, never power. Z-4 ends in a ruling that may be a refusal; at 03:00
   with no judge there is no E2. And since **E4 failed**, E2 has no measured rate either: both preconditions are open.
5. **Z-0 is not "readiness code"; it is the chunk execution path (08, verified).** `manifests.py:212-214` selects `shard-count=1`,
   `ceiling.py:380-404` requires the full 64/64/32 bank, `readiness.py:29-41` checks one plan and one ≤ 384-min timeout, runtime
   parity binds node/machine/runtime root (`ceiling.py:74-87`) so a parity receipt from one assignment cannot admit another chunk's,
   and `analyze.py:217,223` still emits `KILL_SERVED_SEARCH_PUSH` / `FUND_MAC_SYSTEMS` (01, 08: a literal-string Friday predicate
   would reject valid evidence). Z-0 = full-bank hash + four explicit 40-world selections + per-assignment parity inside the
   parity/chunk invoice + a versioned producer→admission label map + pinned analyzer merge after four valid terminal chunks.
6. **The upload cap is a Stage-0 blocker in its own right (judge, from the artifacts).** E3 and E4 both died at the supervisor's
   total-upload-time cap on checkpoint uploads; the Mac preflight that "passed" exercised neither the real byte sizes nor the real
   transport. PLAN.md adds **Z-9**: no manifest dispatches unless its preflight exercised the real supervisor upload path at real
   sizes or the hash-bound runtime-reconstruction path (`9f203c4f`), with the upload-time forecast printed and ≤ 50 % of the cap.
   This is the owner's standing rule ("such errors must never happen") made executable; it costs 0 u and saves three slot-hours
   per failure.
7. **Calendar (02, 03, 07, 09, arithmetic checked):** `C_strict = 29.33 − R − B − A − D` where R = post-Friday replay/semantic hours
   (reserved 8 + 2, unmeasured), B = bare-comparator/control hours, A = other audit/packet hours, D = lost reservation. With R = 8..10
   and a 384-world candidate-paced strict (22.22 h), slack is −0.9..−2.9 h *before* B and A. So the honest Friday invoice must
   **prepay replay/semantic qualification on fixtures before Friday where the package allows it** (fixture qualification is
   pre-paced; trace replay of Saturday's observations is not — 02 right that these are different items), price B and A as rows, and
   let the sized bank be ≤ what remains — or defer. Thursday: L1′ must **start by Thu 16:00** for E14 to keep its Thursday window
   (02); otherwise E14 is NOT_READY(calendar) and moves to the next priced pre-S gap. Builders: Codex-V is the binding resource
   (36–42 h); Y9 must name hours, not roles (03).
8. **Shortlist (final):** served search **22 % / 12 %** (down 3/2: clock seam unguarded at publication, L1′ earliest Thursday,
   Codex-V binding, upload-cap failure class live; upside O1 GO unchanged); representation **22 / 12** (E3 relaunch prepared, E14
   −4 h at risk); PPO backbone **15 / 7** (E4 failed, Z-4(b) dead by arithmetic, survival-guard ceiling); engine **25 %** (E1b
   relaunch prepared, first L4 attempt Refused); search-in-the-loop 15 / 10; harvest operator 5 add-on (O2 unbuilt); archive 6;
   opponent 4; world model / league 2. A100/H100: no, fifteenth time — nothing this week is GPU-bound; Colab is not even the binding
   resource (142–202 u planned against ≈ 1,140 available; the Mac and the builders bind).

## (b) Per-advisor verdicts

| ID | Persona | Verdict | Why (verified facts in **bold**) |
|---|---|---|---|
| 01 | Operator at 03:00 | **keep** | **`write_receipt` is a library call (`receipt.py:458-460`); `audit.py` accepts only `--preflight`; `analyze.py:217,223` emit the old labels; `AMENDMENT_08.md:14-17` calls FUND_SYSTEMS an abbreviation** — verified. Its gate registry (`research/bl_week1/gates.py` + frozen `contract.json`, `g <gate>` validating evidence and writing create-only receipts, never dispatching) with the nine-row command/receipt/failure table is adopted as PLAN §4's instrument; `NOT_READY(INSTRUMENT_MISSING)` when the registry is absent is the right 03:00 behaviour. Its pre-strict replacement (sizing policy frozen pre-L3; `NOT_READY(SIZING_UNSPECIFIED)`) adopted. |
| 02 | Calendar red team | **keep** | **Thu 20:00 Y1 acceptance ⇒ L1′ 20–23, X5 to 01:00, E14 to 04:00 — Thursday missed by 4 h** (checked); **8 replay + 2 semantic unpaced hours reserved and unmeasured (`PREFLIGHT.md:94-98`, `README.md:138-150`)** verified; `29⅓ − R − B − A − D` is the correct ceiling form. Latest-start rule (L1′ by Thu 16:00), priced-invoice Friday predicate, and Tue 21:00 as the evidence seal (21:00–23:00 review only) adopted. Right that pre-Friday fixture replay cannot prepay Saturday's trace replay. |
| 03 | Builder-capacity red team | **keep** | Capacity table (Sol-S 16–22 h, Sol-C 28–36 h, Sol-R 16 h, Codex-I 26–30 h, **Codex-V 36–42 h vs two 8-h windows**) is labelled planning and is the only such invoice in the round; **`events.py:75-89`, `replay.py:218-220` still one cutoff; `transport.py:66-71` forces CPU; `run.py:429-435, 943-951` resume but no pause target** verified. Slip-file schema (`stage0/slips/<item>-<gate>.json` with `next_authorized_action`) adopted verbatim; Z-5 before the first 32M (as Round 9 Z-5 wrote, not "before 100M") — ruled with 08. Z-2 before L1′ — ruled with 04. |
| 04 | Apple-silicon latency / paced identity | **keep** | The round's code catch: **`controller.py:60` is the submission check; `publish()` has no cutoff comparison and no visibility timestamp; consumption uses worker `completed_ns`; `replay.py:241-281` accepts visibility without a deadline** — all verified; its in-memory reproduction (on-time publication rejected under a publication cutoff; late visibility accepted under a consumption cutoff) is exactly the ambiguity Round 9 named, now shown to be a missing check rather than a field split. Integer tick mapping `T(r)=t0+⌊r·10⁹/24⌋`, equality-is-late, four-case table through controller **and** replay, scoped probe authority (`authorization.json` with verbatim grant + allowed cells; negative tests both ways), MPS adapter attestation, and **K1 CPU PASS alone satisfies Friday's timing conjunct** — adopted. |
| 05 | Evaluation / statistics skeptic | **keep** | **Strict illustrative 128/mix SE 14.65; P(point ≥ 40 | +80) = 99.7 %; mass > 0 power ≈ 86 % at +40 with the point filter halving admission; (0,0,240) fails L2′'s frozen/mixed LB90 clauses ≈ 99 %** (checked). Its replacement power text, Y6 scenario set (40/40/40, 80/80/80, 0/0/240, below-bar, partial nulls — ADEMP), L4 precedence text (CLOSED_BELOW_BAR > BAR_MET > INCONCLUSIVE; FILTER separate; nonfinite ⇒ block), sizing-policy freeze pre-L3, and the per-family (not campaign-wide) alpha disclosure are adopted. Its α = .025 sizing example (135/mix = 23.4 h) shows an alpha split is *not automatically* over the ceiling — noted; Round 9's no-split ruling stands for the stated reason (no retry, one family per artifact), not for the hours. |
| 06 | Large-scale on-policy RL | **keep** | **E2/E6 are separate .05 routes with funding = union (`E9_RULE.md:5-13`, `oc.py:189-203`); `oc.py:268-289` hardcodes legacy counts and `dispatch=false`** verified; the survival sign-guard ceiling (≤ .5718 at Δ = 0 under ρ = .9) is model-conditional and correct in form — it means Z-4 must declare its survival alternative. Print-licence table (E2 GO ⇒ planning; E6 NEEDS_SCALE ⇒ bridge; E7 ⇒ separate archive registration; E12 ⇒ critic only, one E19 repair) and the next-week E16 registration text (A/G only, hash-bound, DRAFT_CONDITIONAL, earliest Oct 28, 40 u / 20 h with 4 reserve, ≥ 800 eval world-steps/s) adopted. Right that E4 measures collection rate, not evaluation rate. |
| 07 | Release / serving qualification | **keep** | **`strict_pin_problems()` checks checkpoint/veto/evidence-shape only (`served_checkpoint.py:172-217`); `pathb.eligible()` + `SNAKE_SERVE_PATHB` is opt-in infrastructure (`pathb.py:19-34,189-206`), not deployment evidence; frontend badge carries no controller digest (`types.ts:141-148`, `App.tsx:869-876`); X7 BUILD_VERIFIED_NOT_QUALIFIED** verified. Typed evidence chain bound to controller digest C under `exp/claim-b-release-20261024/<C>/`, repair ⇒ C′ with a fresh L3→L4→strict→product→serving chain, the rollback matrix (`SNAKE_SERVE_CHECKPOINT=b8-s3` alone is insufficient — the assistance flag must be disabled and pending work drained), and the two verbatim signature texts are adopted. "Skill-qualified", not "qualified", for an opened-and-passed S. Monitoring (7 days AND 100k decisions) cannot complete by Tuesday — the packet must say so. |
| 08 | Distributed systems / Colab parity | **keep** | **`manifests.py:212-214` shard-count=1; `ceiling.py:380-404` full bank; `ceiling.py:74-87` parity binds node/machine; `supervisor.py:981-986` retained vs verified stops; `recovery.py:47-76` sibling-bank snapshot references; `PARITY_RULE.md:118-131` E13 = DEVELOPMENT_TRANSFER** verified. Ready-job Z-1 (not FIFO; E4 preserved — now moot, E4 failed), Z-0 as execution path, Y3 transitive snapshot set on a replacement runtime, Z-5 before 32M, **130 u → 142 u base / 202 u conditional**, typed E13 index (execution_platform from the bound manifest, never the directory) — all adopted. |
| 09 | Print interpreter / day-7 reducer | **keep** | **`events.py:144-157` single Boolean; `receipt.py:346-354,412-417` base vs full; `PROTOCOL.md:63-76` GO without architecture; `admission.py:165-182` A-only refused; `FINAL_FAMILY_REGISTRATION.md:11-28,50-55` p = 1 members, look awaits skill AND product** verified. Authorization normalization (probe/serving as separate receipts; expanded PASS under probe-only ≠ Friday eligibility), the 6 × 4 S × R route product with per-study precedence (validity first), "opened-but-incomplete stays OPENED, excluded from automatic week-2 readmission", and the signature rules (R cannot veto or rescue S; owner refusal ⇒ RELEASE_HOLD(OWNER_DECLINED) preserving qualification) are adopted as PLAN §7. |
| 10 | Owner-trust red team | **keep** | **O2's registered bar is (D+O − D) LB90 > 0 ∧ +40/LB90 > 0 vs B8 (`CAMPAIGN_LOG.md:101`)**; **O1 branches require surviving bare H500 continuations and cannot be summed (`PREREGISTRATION_O1.md:46-53,85-93`)**; **O2 unbuilt (empty artifact dir)** verified. Sealed `handoff/INDEX.json` at Tue 21:00 as the object signed; harvest admission through O2's governing receipt and a new S identity with its own ruler/L2′-class/paced chain (tentative 72 u, not authorized); `resources.json` separating proposed / ratified / reserved / debited; per-family alpha disclosure — adopted. Correctly withdraws its own alpha split while keeping the disclosure. |

Factual errors found: none in the advisors that change a conclusion. Stale-at-snapshot: none (snapshot = HEAD). Corrections to the
Round 10 fact sheet (judge's own, verified): (i) E4 is not running — it failed at 02:54 on the upload cap; (ii) Round 9's "checked at
visibility" is wrong — publication is unguarded; (iii) "cannot distinguish +40 from 0 on any single stage" is withdrawn in favour of
05's precise statement; (iv) "Z-5 before any 100M" in the skeleton contradicts Round 9's Z-5 — before the first 32M stands; (v) the
130-u week-1 figure is a priority list, not a ceiling (142 base / 202 conditional); (vi) O2 is unbuilt and its bar is interval-based.

## (c) Disagreements ruled

1. **Power statement — 05/10 vs Round 9.** 05/10 win on the sentence; Round 9 wins on the bars. PLAN §0 states: per-stage point ≥ 40 is a
   ≈ 50 % filter at a true +40 (≈ 90 % at +80); L2′'s cross-mix clauses reject scripted-only winners by design; unconditional
   qualification is a planning sensitivity (UNVALIDATED until Y6 with 05's scenarios); no bar changes.
2. **Z-2 timing — skeleton (Fri 12:00/20:00) vs 03/04 (before any qualifying L1′).** 03/04. Z-2′ (with the visibility timestamp,
   §a2) is accepted through controller **and** replay before any L1′ cell can print PASS; a ruler run before it reports micro-timing
   only. Fri 20:00 is Z-2′'s latest acceptance, not its schedule; Sol-C targets Wed.
3. **Friday timing conjunct — 04 (K1 CPU PASS suffices) and 09 (authorization axis).** Both adopted, compatible: under serving silence
   the eligible cell is W1/K1 CPU; its PASS satisfies the conjunct; expanded/MPS cells are NOT_MEASURED unless probe-authorized and
   never veto or substitute; a probe-only YES never enlarges serving authority; a serving YES selects which *measured* cell counts.
4. **Z-5 — skeleton "before any 100M" vs 03/08 "before the first 32M".** 03/08 (= Round 9 Z-5). Also the E15 invoice must be reconciled
   as proposed / ratified / reserved / debited (10) before any tranche.
5. **Z-4 — three designs (Round 9) vs one live design (06).** 06. Option (b) is structurally incapable; (c) is a cost saver under (a)'s
   final test. The funding judge rules on (a) with the survival alternative declared at +.03 pre-data and the Δ = 0 ceiling reported;
   no judge at 03:00 ⇒ no E2. E4 must also be relaunched (after Z-9) before any E2 forecast exists.
6. **Z-0 — readiness-only (skeleton) vs execution path (08).** 08. Codex-V tests duplicate/missing worlds, wrong runtime, mismatched
   environment and legacy labels without gameplay; a chunk that cannot be executed by the pinned runner is NOT_DISPATCHABLE.
7. **Z-1 — ordered queue (skeleton) vs ready-job priority (08).** 08, with the judge's addition: **Z-9 upload-path qualification is a
   readiness condition for every job**, and the priority is Y1 → parity → E3 recovery-02 → E1b relaunch → E4 relaunch (after Z-9) →
   L2′ chunks (≤ 2 slots) → O2 → Y3 → admitted E2. Verify slots by `colab usage` + terminal-export receipts; never from stdout.
8. **Week-2 alpha — 05/09/10 (disclose per-family) vs Round 9 (no split).** Both: no split, no retry, one family per artifact, and the
   handoff prints "per-family .05; campaign union bound min(1, Σα); no campaign-wide 5 % guarantee" (05's ledger).
9. **Harvest — skeleton "if O2 D+O > D" vs 10.** 10. Admission only through O2's registered interval bar and a new S identity with its
   own ruler, L2′-class evidence, latency, oracles, L4, strict, product and serving; O1 GO admits nothing.
10. **Thursday E14 — skeleton vs 02.** 02: L1′ latest start Thu 16:00 (after Z-2′ + Y1-Mac acceptance + Phase 0 idle receipt); else
    E14 is NOT_READY(calendar), scheduled in the next priced pre-S gap; never displaces S; never substitutes an architecture.
11. **A100/H100.** Unanimous no; and the round's own arithmetic shows Colab units are not binding this week at all.

## (d) Running shortlist (final; carried from Round 9, updated)

| # | Lever | Conf. push / beats B8 at bar | Cost to first signal | Most informative next experiment |
|---|---|---|---|---|
| 1 | **Served exact-sim search on this Mac** (T3 certificate controller, reachable-root EDF burst, W/K-bound Claim B identity C, three oracles) — the week's only possible release-track artifact | 22 % / 12 % (down 3/2: publication cutoff unguarded, L1′ Thu at earliest, Codex-V binding, upload-cap class live) | Z-0 + Z-9 + Y1 ≤ 4 u → L2′ 4 × 16 u + parity 4 u → L1′ 3 Mac-h → L3 9 + L4 32 paced h → strict ≤ C_strict | **Y1 ruler** (R ≥ 2.18/s decides L2′) ∥ **Z-2′ → L1′ K1** (C p99 ≤ 333/542 ms decides serving) |
| 1d | Harvest-aware served operator (O1 GO upside) | 5 % add-on | O2 ≤ 24 u (unbuilt) | O2 print against its interval bar; week-2 identity only |
| 1c | RL endpoint as planner continuation (E20) | conditional | 8 u | week 2 |
| 2 | **Representation: geometry residual on the F parent** | 22 % / 12 % (down 3/2) | E3 recovery-02 16 u → E14 3 Mac-h → E16 40 u (week 2) | E3 relaunch → E14 bare cells (Thu if L1′ starts by 16:00) |
| 3 | **Long-horizon on-policy RL from B8** (backbone) | 15 % / 7 % (down 3/1: E4 failed; Z-4(b) dead; survival-guard ceiling) | Z-9 → E4 2 u → Z-4′ ruling → E2 20 u → E6/E7 24 u → E12 16 u | **E4 relaunch after Z-9; Z-4′ with declared survival alternative** |
| 4 | **GPU-resident exact engine** (Colab only) | 25 % (down 3: first L4 attempt Refused; relaunch prepared) | E1b 20 u | E1b relaunch (prepared at `88adb3e2`) |
| 5 | Search-in-the-loop | 15 % / 10 % | after E1b | E16-S (week 2) |
| 6 | Restart archive / PLR | 6 % | E6/E7 (validation v2 done, undispatched) + E10 | E10 envelope |
| 7 | Opponent-conditioned add-on | 4 % | after E3b | — |
| 8 | World model; league | 2 % / 2 % | — | not funded |
| 1b | Retention sub-arm | conditional | 36 u | on the collateral trigger |

## (e) Still unresolved (carried into PLAN.md as explicit branches or blockers)

- **Owner's serving clause (W ≤ 4 / K ≤ 8) and probe-only authorization** — both PENDING; PLAN §1 branches on each.
- **S per batch on this Mac (L1′) and R on L4 (Y1)** — unmeasured; the two numbers that decide L2′ dispatch and Friday timing.
- **Whether Z-2′ (visibility timestamp + two cutoffs through controller and replay) lands before Thursday 16:00** — decides whether
  E14 keeps Thursday and whether L1′ is Wednesday night or Thursday.
- **Upload-path qualification (Z-9) for every checkpoint-carrying manifest** — three failures already; E4 is the next test.
- **Post-Friday replay/semantic hours** (reserved 8 + 2) — the strict stage is insolvent at 384 worlds unless fixture qualification is
  prepaid before Friday and B/A are priced; the Friday invoice must show it.
- **Codex-V capacity** — 36–42 review-hours; Y9 must show named hours or items print NOT_READY.
- **Z-4′ ruling** (192/64/64 with survival alternative +.03) and **E4's rate** — both needed before any E2; E16 earliest Oct 28.
- **O2 build and print** — the only thing that could raise lever 1's ceiling; week-2 identity.
- **Phase 0 idle receipt** — ETA Wed 10:00–13:00; not assumed.

## (f) Experiments Claude can run now (≤ 20 u each unless stated; dispatch via Sol; report in LIVE_FACTS.md)

All of these are in PLAN.md with owners, dates and receipts; this list is the dispatch-priority view.

**Z-9 — Upload-path qualification (0 u; Sol 0.25 day per manifest family; today; before E4/E3/E1b/Y1/L2′/O2 dispatch).** Every
manifest's Mac preflight must exercise the real supervisor upload path at real byte sizes (or the `9f203c4f` hash-bound runtime
reconstruction) and print the forecast upload time ≤ 50 % of the supervisor's cap; receipt `exp/<job>/upload-preflight.json`. Kill:
no receipt ⇒ NOT_DISPATCHABLE. Relaunch **E4** (≤ 2 u) first — it is the cheapest real test of the fix and the E2 precondition.

**Y1 — L2′/L1′ ruler, L4 half (≤ 4 u; fixtures frozen at `4677dd02`; after Z-0 code + Z-9).** Nine constructed fixtures, ≥ 30 real
H300 batches per root, 12 one-thread workers, aggregate R; semantic outputs must match the Mac reference or the fixture is INVALID;
timing is platform-labelled. Kill: R < 2.18/s ⇒ L2′ NOT_DISPATCHABLE at 64 u (funding judge may reprice prospectively; bank never
shrinks); its slot goes to O2.

**E3 recovery-02 (16 u) and E1b relaunch (20 u)** — prepared; dispatch when slots and Z-9 receipts exist; terminal create-only
receipts + `stop_verified` required (CLI exit 0 is insufficient).

**L2′ chunks 1–4 (4 × 16 u incl. 3.2 reserve each; + per-assignment parity inside a 4-u envelope; after Z-0 code, Y1 ≥ 2.18/s, Z-9).**
Immutable 16/16/8 chunks, ≤ 384 timeout-min each, ≤ 2 chunks in parallel, one look after four valid terminal chunks under the pinned
analyzer with the versioned label map (`FUND_MAC_SYSTEMS` → FUND_SYSTEMS; `KILL_SERVED_SEARCH_PUSH` → PACKAGE_KILL). Chunk fault ⇒
study INCONCLUSIVE. Latest dispatch of the last chunk for a Fri 23:59 print: **Thu 22:23** (25.6-h liability on one slot) or Fri 10:00
on two slots.

**O2 (≤ 24 u; after Sol builds it + Z-9).** Registered bar as `CAMPAIGN_LOG.md:101`; unpaced; T34 lane; week-2 identity.

**Z-2′ — Clock seam, corrected (0 u; Sol-C 0.5 day + Codex-I 0.25 replay + Codex-V 0.25; before any qualifying L1′).** As §a2 and
04's table; receipt `exp/bl-l1/preflight/<attempt>/clock.json`. Kill: any mis-dispositioned case ⇒ `l3_identity_prerequisites`
cannot print PASS; L1′ reports micro-timing only; L3 blocked.

**Z-3 — L1′ K1 cells (0 u; 3 exclusive Mac-h; after Phase 0 idle receipt + Y1-Mac acceptance + Z-2′; latest start Thu 16:00).**
Bars: C p99 ≤ 333 ms (2.4 roots/s) / 542 ms (1.6) with ≥ 29/20 (12 s) and ≥ 36/24 (15 s) valid consumptions, zero overruns, update
p95 ≤ 11.688 ms; expanded/MPS cells only under a recorded probe-only grant (`authority/probe.json`) and the device adapter. Kill:
eligible cell FAIL ⇒ S's reservation released (NOT_READY, timing FAIL); unmeasured ⇒ NOT_MEASURED, S deferred.

**E14 bare cells (0 u; 3 Mac-h; Thu after L1′ + X5 if L1′ started by 16:00; else next priced pre-S gap).** E3's exact architecture
(`admission.py:259-267` binds it); p95 > 11.688 ms or overrun ⇒ G unservable.

**Z-4′ — E9 replacement (0 u; Sol-R 0.5 day + funding judge).** 192/64/64 fixed, survival alternative +.03 declared pre-data, Δ = 0
ceiling reported; (c) only as a futility look under (a)'s final test; receipts `exp/week1-20261021/R/e9-prime/{design,oc,approval}.json`.
Kill: no signed passing ruling ⇒ E2 BLOCKED; R's week = E4 + E3 + E14 + Y3 + Z-5.

**Y6 — OC with 05's scenarios (0 u; Codex-I 0.5 + Codex-V 0.25).** (40,40,40), (80,80,80), (0,0,240), below-bar, partial nulls;
ADEMP; day-7 power field UNVALIDATED until accepted.

**Z-6′ — Gate registry (0 u; Codex-I 1 day + Codex-V 0.5; Thu 20:00).** 01's `gates.py` + `contract.json`, the nine gates, the
reducer with 09's route product, the slip schema (03), the typed E13 index (08), 07's signature texts; Codex-V rehearses the nine
synthetic failure cases without gameplay.

**X5 (0 u; 2 unpaced Mac-h), Y3 (≤ 8 u; before any 32M/E17), Y7, Y5/Claim-B chain (07), Z-5 (before first 32M), Z-7/INDEX (10) —
unchanged in content, re-dated in PLAN.md.**

Budget this inter-round (units): Y1 4 + parity 4 + E4 2 + E3 16 + E1b 20 + L2′ 64 + O2 24 + Y3 8 = **142 u** base; + E2 20 + E6/E7 24
+ E12 16 = 202 u conditional. Everything else 0 u. A100/H100: none.
