RULING: CORRECT_AMENDMENT — L2′'s registered opponents (historical T3 strict-gate rosters, NO opponent veto) are the lawful and correct estimand for L2′'s question; my JUDGMENT_MAC_AMENDMENT P1(c) "frozen/mixed must show kernel rows" rule was a judge error that transplanted a ruler-only pricing choice into production. Pre-data correction: delete the native-activity requirement, require all-zero veto-hook counters plus an opponent-veto attestation, native_mode=stock; bank, bars, one look, seeds, analyzer labels unchanged. Native/exact-stop kernel work is NOT needed for L2′ (production has no opponent veto to accelerate). Measured production rate (P3, no v8 opponents) gives mode B `all` ≈ 22–25 wall-h (30 h with reserve) ≤ 60-h clause; L2′ can launch after a ~3 h Sol fix + Codex-V review + a ≤ 2 h P3′ identity rerun. Sol/Codex-V refusal was correct. Fri 23:59: almost certainly NOT_READY(L2′) (mid-run), per §4 of the prior ruling — clause 1 cannot fire.

# L2′ opponent premise — independent Fable judgment

Judge: Fable (one independent judge), 2026-10-22 ~PT afternoon. Read-only; nothing run, nothing committed; the sealed
`t34-round0-study-01` directory was not opened. Supersedes JUDGMENT_MAC_AMENDMENT.md §3 P1(c) and the matching lines of
AMENDMENT_MAC_L2.md; everything else in that ruling stands unless named below.

## 0. Verified facts (bytes read)

Production opponent construction (snake-dqn-l2mac @ 1ebdc7f2, main):
- `research/bl_served_search/ceiling.py:318-321` — `run_arm` builds the world via historical
  `research/t3_rollout_labels_20261015/bank_engine.build_world`.
- `bank_engine.py:399-447` — `build_world` captures the world through `eval_engine.run_simd_eval(...)` with
  `hero_ego2s=<B8>`, `vector61=True`, `opponent_specs_by_world={seed: roster}`; it passes **no** `hero_safety_veto`.
- `src/simd_env/eval_engine.py:900-916` — `policy_for(spec, hero=False)` builds every checkpoint opponent as
  `Vector61SimdPolicy(..., veto_slots=(0,) if hero_veto else (), veto_variant=... else "v2")` ⇒ opponents get
  `veto_slots=()`; comment at :901-902: "The hero's veto must never reach a same-checkpoint opponent".
- `src/simd_env/vector61_policy.py:798-799` — `if self.veto_slots: actions = self._apply_veto(...)` ⇒ with empty slots
  the veto hook is never called, so `bl_k1.native.configure` has nothing to route (`research/bl_k1/native.py:1-5`:
  "Only rollout_all enters native_scope"; the s2 kernel is "native v8 **opponent** veto", `t34_speed_s2/native_v8.py:1-8`).
- The ruler alone adds the veto: `ruler_operator.py:137-142` loops `world.opponents` and calls
  `policy._configure_veto(tuple(range(1, 6)), "v8", 8.0, None)`. Documented intent, `RULER_FIXTURES_08.md:15-21`:
  "checkpoint opponent slots use FRP3 s12 with v8 assistance (lambda 8) … This prices the expensive Watch6 opponent
  case prospectively; **it does not replace the original L2 bank's historical T3 rosters or operator**."

Registration of L2′:
- `PREREGISTRATION_L2.md:8-16` — question "can the unchanged T3-dense mechanism plausibly clear +40 equal-mix H5000
  mass against bare B8 s3"; "Same seeds, ordered **historical T3 scripted/frozen/mixed opponent rosters**, profile, RNG
  and opponents across arms. No hero veto, eps=0"; "Reuse the historical T3 world construction and branch engine
  read-only." Authority round-05 §(f); T3 ceiling test `docs/research/t3_ceiling_test_2026-10-15.md:17,117` (same
  rosters, "no veto").
- `BANK_L2.json` rows and the P3 allocation (`mac-prerequisites-01/p3-identity-02/allocation.json`): frozen =
  {champion_a5_freespace, best_apex_fs, best_apex_stage1_fs, best_apex_pre_fs} × 5 slots; mixed = three of those +
  2 × `random_safe`; scripted = `greedy_food` × 5. No FRP3 s12 anywhere in the bank.

Served product (snake-dqn main, 3c2ae889):
- `web/backend/served_checkpoint.py:74-83` — released default checkpoint `b8-s3` (2026-10-17), Watch roster default
  `mixed` (T15 SERVING_PASS); `served_checkpoint_pins.json` b8-s3 `veto_variant: "none"`.
- `web/backend/session.py:622-705` — `_install_mixed_watch_roster`: "Replace Watch opponents with pinned FRP3+v8"; every
  `game.snakes[1:]` gets the FRP3 s12 `ApexPolicy` (:677-680) and `install_watch_opponent_vetoes` (:682); any wrapper
  failure falls back to all-B8 (:684-694).
- `web/backend/safety_veto_serving.py:508-527` — "Install the exact v8 wrapper on mixed-Watch opponents only … every
  vector61 FRP3 opponent is bound to the v8 receipt."
- `docs/research/final_run_20261019/SUCCESS_CONTRACT.md:27-37` — §2 skill bars: "fresh paired banks, opponents held
  fixed … equal-weighted scripted / frozen / mixed, vs bare B8 s3"; §3 product bars: "continuous-respawn mixed Watch6,
  **FRP3+v8 opponents fixed**". Two different games, two different bars.

P3 run (`mac-prerequisites-01/p3-identity-02/receipt.json`, sha `8995f839…`; `review-03.json` REFUSED): W2, one thread,
nice 10/caffeinate; two frozen `all` worlds completed concurrently: frozen/1 491 batches, 2,542 branches, 744,765 rollout
frames, `wall_seconds` 1,095.4; frozen/0 575 batches, 3,085 branches, 905,053 frames, `wall_seconds` 1,323.8. Both
`native_v8` counters all zero; analyzer error "missing native branch rows" (my P1(c) rule). Elapsed 2,014.7 s to SIGTERM;
no exact_stop row completed; bank untouched; 0 units.

## 1. Question 1 — served vs registered opponent configuration

| | Served Watch6 (product, SC §3) | L2′ registered bank / T3 census (skill, SC §2) | Y1/K-1 rulers |
|---|---|---|---|
| Hero | bare B8 s3, no veto | bare B8 s3 (+ dense T3 overlay in arm 2), no veto | bare B8 s3 (+ overlay) |
| Opponents | 5 × FRP3 s12, **each wrapped with v8 λ=8** (`session.py:677-682`) | historical Apex checkpoints / `random_safe` / `greedy_food`, **veto_slots=()** (`eval_engine.py:910`) | FRP3 s12 + v8 on slots 1–5 (`ruler_operator.py:142`), scripted kept |
| Why | T15-qualified product roster | T3/strict-gate roster family, registered pre-data | worst-case pricing of the Watch6 opponent case |

So the parent's premise "L2′ production constructs opponents with veto_slots=()" is **true and intended**; "Y1/K-1 explicitly
ENABLED opponent v8" is also true and was a deliberate, documented pricing choice — not the registered L2′ game. The
discrepancy is between the ruler's fixture world and the bank's world, and my amendment ruling (P1(c)) wrongly demanded that
the bank *execute like the ruler*. Sol's wiring matched the registration; Codex-V's refusal of my rule was correct.

## 2. Question 2 — is the registered configuration right? What is the lawful correction?

**Yes, for the question L2′ registered.** L2′ is the development ceiling of the T3-dense mechanism (round-05 §(f) → round-13
§0: "operational ceiling of one dense controller"), scored on the SC §2 skill-bar game: fresh paired banks, equal-weighted
scripted/frozen/mixed, opponents held fixed, vs bare B8 s3 — the same game as the T3 ceiling test it extends and as the
strict gate that released B8 s3. Its labels (FUND_SYSTEMS / PACKAGE_KILL / INCONCLUSIVE, PLAN_v2 §3) are planning labels
for whether the search family can plausibly reach the §2 bar at all. The phrase in this question — "frozen/mixed opponents
of the served game" — names the §3 product game (Watch6, FRP3+v8). L2′ never registered that game, no bank exists for it,
and it is not a "frozen/mixed" bank at all (one roster, continuous respawn, deaths/1k). Swapping the bank's opponents to
FRP3+v8 now would be a **new registration** (new rosters, new audit, new estimand, new bar family), not a correction, and
would make L2′ incomparable with T3-dense (36,996 batches, 64 worlds) whose prior it is meant to extend.

Honest limit recorded: a FUND_SYSTEMS print from L2′ says nothing about the Watch6 product bar; any later served claim must
still pass SC §3 on the product roster (where the opponents *are* vetoed and the native kernel *does* matter, cf. §3 below).
The product-roster question is a separate, later bank (PLAN_v2 §5 route), not a reason to delay or rewrite L2′.

**Lawful correction (pre-data; no bank world has run; execution-only, same pattern as AMENDMENT_STOP rule 5):**

- P1(c)′ replaces P1(c): the analyzer **refuses** any row of any mix whose `native_v8` counters are not all zero
  (`calls`, `kernel_rows`, `stock_rows`, `fallback_rows`, `noncandidate_rows`, `python_calls`, `root_stock_rows` — a
  nonzero value means a veto reached an opponent and the row is not the registered game); any row whose exported
  `opponent_veto_slots` (read from each constructed `Vector61SimdPolicy.veto_slots` of the world, never from a flag) is not
  `[]` for every checkpoint opponent; any row whose roster digests differ from the plan; any row with `search_mode` /
  `native_mode` outside the registered pair; survivor branch frames ≠ 300 (unchanged).
- Registered pair becomes (`search_mode` ∈ {`exact_stop`, `all`} fixed for all 8 chunks, `native_mode = stock`). Kernel
  mode is not forbidden (it is a no-op here) but must not be *required*; `stock` is the honest default and removes a
  misleading "kernel" label from receipts.
- `AMENDMENT_MAC_L2.md:29-31` ("Frozen/mixed fired batches must use the kernel …") is replaced by the above; add one
  paragraph stating the opponents are the registered historical T3 rosters with no opponent veto, that the K-1/Y1 rulers
  priced a *different* (Watch6 FRP3+v8) opponent case, and that no Mac timing from those rulers is used for the forecast.
- **Unchanged:** `BANK_L2.json` (sha `0a8d3059…`) and audit, `chunks.py` 8/8/4 order, analyzer label map and bars, one look,
  H5000, `DenseTrigger(600,5)`, `_choose`, exact-stop semantics, B8 bytes, served hero. Bank, bars and one-look are **not**
  changed by this correction.

## 3. Question 3 — true Mac forecast on the correct configuration; is native/exact-stop needed; can L2′ run now?

Measured production rate (P3, W2 concurrent, one thread, nice 10): frozen `all` = **0.448 and 0.434 batches per worker-s**
(full H5000 worlds, 491/575 batches ≈ the 578 assumption). Compare: ruler `all` kernel frozen 0.4267 (JUDGMENT_MAC_AMENDMENT
§2), old stock-with-v8-opponents frozen **0.055** (round-12 QUESTION F7). Reading: the opponent v8 veto was 78–90 % of every
ruler batch; the production path never had it, so production `all` without any kernel already runs at the "kernel" speed.
**The native kernel buys nothing for L2′**; it stays relevant only to the served/product game (Path B serving latency, K-1's
actual purpose) and to any future product-roster bank. The exact-stop wiring remains a lawful 2–3× saver (ruler ratio
1.2221/0.4267 = 2.86×, opponent-independent since it only reorders/stops branch evaluation) but is **optional**: mode B
(`all`) is inside the clause on its own.

Forecast, mode B `all`, W2, no Colab setup term (measured world walls, not fixture rates):
- frozen 64 × 1,324 s (worst of two) = 84,736 worker-s; mixed 64 × ≤ 1,324 s (bound: 3/5 vector61 opponents + 2 scripted,
  expected ≈ 0.7×) = ≤ 84,736 (expected ≈ 59,300); scripted 32 × ≈ 370 s (ruler `all` scripted 1.5683 b/s, never vetoed)
  = 11,840. Total ≤ 181,312 worker-s = 50.4 worker-h ⇒ two lanes **≤ 25.2 h**, expected ≈ **21.7 h**; × 1.2 reserve ⇒
  **≤ 30.2 h (expected 26 h) ≤ 60 h** — eligible. Sensitivity at 1,000 batches/world: ≤ 43.6 h × 1.2 = 52 h, still ≤ 60.
- Per chunk (8/8/4): ≤ 22,664 worker-s ⇒ two lanes ≤ 3.15 h; **chunk timeout 6.3 h**; whole-run lease = launch + 60 h
  (mode B cap, unchanged). Mode A (if P3′ proves identity): ≈ 1/2.9 ⇒ ≈ 8–9 h wall; lease launch + 28.3 h as before.

**Can L2′ run now? No — one receipt is missing, not a capability.** P3 is INCONCLUSIVE under *either* rule: zero exact_stop
rows completed, so production-path identity of mode A is unproven, and mode B rows have never been accepted by a correct
analyzer. After the correction, P3′ (five dev worlds, both modes) costs ≈ 1 h at W2 (≈ 6,500 worker-s) — inside the ≤ 2 h
leased budget — and then launch is lawful the same day.

## 4. Question 4 — next steps, reviewers, launch conditions, KEEP_B8

1. **Sol-S (≈ 2–3 h, pre-data, one branch, merge to main):** (a) `analyze.py` P1(c)′ rule as in §2 (refuse nonzero veto
   counters / nonempty opponent `veto_slots` / roster drift); (b) row export of `opponent_veto_slots` and roster digests read
   from the constructed world in `_worker`; (c) default and registered `native_mode = stock` (keep `--native-mode kernel`
   accepted but not required; `mac_l2`/launcher admission updated); (d) AMENDMENT_MAC_L2.md paragraph swap per §2 with this
   file cited; (e) regenerate the Mac runtime-parity reference from the amended source; (f) focused tests (the 45 + new
   refusal cases: a row with `kernel_rows>0` must now be refused; a row with all-zero counters and `[]` slots accepted).
2. **Codex-V (independent, review-04.json):** diff scope limited to the files above; confirm `eval_engine.py:910` /
   `vector61_policy.py:798` are untouched; confirm BANK_L2 / chunks / label map / bars byte-unchanged.
3. **Claude operates P3′ (`p3-identity-03/`, ≤ 2 leased Mac-h, 0 u):** the same five round-1 dev worlds may be reused (they
   are development-only, non-bank, audited CLEAR; the two completed `all` rows are retained, not reused as receipts); both
   modes; require identical chosen action per fired batch and identical ledgers after stripping timing/diagnostics, all-zero
   veto counters, `_row_error is None` for every row. Any mode-A difference ⇒ mode B only (its rows must pass the same
   checks); mode-B failure ⇒ NOT_READY.
4. **Launch (P4 host gate unchanged):** no lease, QUIET idle receipt, clean checkout at the new commit = origin/main,
   lease owner `claude-l2prime-mac`, chunk timeout 142 min (A) / 6.3 h (B), W2, one thread, nice/caffeinate — all other run
   conditions C1–C10 of JUDGMENT_MAC_AMENDMENT.md stand. The one look is unchanged.
5. **Fri Oct 23 23:59 KEEP_B8 clause:** earliest realistic launch is Thu evening / Fri; mode B is ≥ 22 h, mode A ≈ 9 h.
   A print by Fri 23:59 is possible only with mode A launched Thu night; otherwise L2′ is mid-run and §4 of the prior ruling
   governs verbatim: no print ⇒ clause 1 **cannot fire** ⇒ NOT_READY(L2′), not PACKAGE_KILL; the run is not hurried,
   shrunk or peeked at; the look is taken when the eight chunks are terminal and D3 consequences apply from then. Tue 23:00
   clause 2 still cannot fire (K-1 20.13× stands for its own, product-game purpose). A faulted run is INCONCLUSIVE, never
   PACKAGE_KILL.

## 5. Accountability and honest limits

- The error was mine: JUDGMENT_MAC_AMENDMENT §3 P1(c) and its carry-over into AMENDMENT_MAC_L2.md:29-31 assumed the bank
  world was the ruler world. The placement forecast itself (rates ≈ 0.43 b/s frozen `all`) was accidentally right because the
  production path lacks the very cost the kernel removes; the clause test (≤ 60 h) therefore stands on the new measurement.
- Mixed-world cost is bounded, not measured (P3 stopped before any mixed row); C4's 2× chunk timeout absorbs it.
- Nothing here is efficacy, a served claim, or a release. B8 s3 remains the served hero; the product game (Watch6 FRP3+v8)
  remains unmeasured for search and is a separate bank if L2′ ever prints FUND_SYSTEMS.
