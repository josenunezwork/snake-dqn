# T34 ExIt-B8 campaign log (decisions, rulings, status)

Single running record of the T34 campaign. Times are Pacific. Binding documents are listed in
[docs/STATUS.md](../../STATUS.md) "Campaign authority"; this log explains how they came about and what is running.
Newest entries at the bottom of each section. Update this file whenever a decision or status changes.

## Process rules adopted during this campaign

- **Advisors:** facts + goal + an open question, never a menu. Default two advisors (one domain, one critic);
  expand only for a named unresolved question. When Astra disagrees with Fable, ask Fable whether it agrees with
  Astra's opposing view, then decide (owner rule, 2026-10-19).
- **Speed review:** any job estimated > ~4 h wall, > ~10 Colab units or > 1 day of Sol work gets the five-minute
  self-check first, then an advisor panel if needed (`~/.claude/skills/speed-review`). Long is fine when the work
  is irreducible, cheap and blocks no decision.
- **Freeze before data:** every design change is written to main (design amendment or pre-registration) before
  any data it could influence. Registered gates are never relabelled after results (e.g. M2 stays NO-GO).
- **Review loop before ratifying a registration:** independent reviewer → `REVIEW: RATIFY` or `FIX-FIRST` with
  file:line findings → operator rulings written as a Sol spec (pre-data) → Sol fixes + tests → same reviewer
  re-checks in code, not docs → repeat until RATIFY → operator ratifies, binding the final source.
- **Colab acceptance:** any `research/colab_cli` code change requires a fresh final-only acceptance (prepare →
  independent review → live run → record in `research/colab_cli/QUALIFICATION.md`). Fix known defects before
  paying for acceptance.
- **Roles:** Sol (Codex gpt-6.1-sol) implements; Claude plans, reviews, operates every live VM run and makes
  the next-step calls (owner delegated, 2026-10-20 00:15); releases/serving changes still need the owner.
- **Compute:** Colab credits are fine when they speed up a lane (owner, 2026-10-20 12:15); L4 by default, priced
  per useful result. Path B timing/paced tests, the strict gate and serving qualification stay on the Mac.

## Decisions (chronological)

| When | Decision | Record |
|---|---|---|
| 10-19 ~18:00 | Fable final-run panel recovered after a session limit; FINAL_PLAN "ExIt-B8" funded for prep + ~2 u L4 benchmark | [FINAL_PLAN.md](FINAL_PLAN.md) |
| 10-19 ~18:30 | No new checkpoint publication: all six inputs already public in release `s1-inputs-20261019` | `docs/research/specs/2026-10-19_exit_inputs_addendum.md` |
| 10-19 ~19:20 | Success contract ratified: Claim A bare student / Claim B labelled assisted hero; bare wins ties; Watch6 first | [SUCCESS_CONTRACT.md](SUCCESS_CONTRACT.md) |
| 10-19 ~21:15 | Design v2: shadow trajectory + LAD search (speed panel: cold H300 verify can never fit a tick) | [DESIGN_V2_SHADOW.md](DESIGN_V2_SHADOW.md) |
| 10-19 ~22:15 | Amendment 1: stride-10 + stride-1 teacher label rule (bisection non-monotone); L4-only hardware | DESIGN_V2 Amendment 1 |
| 10-19 ~23:20 | Rules audit (Astra ×3, Fable judge), owner "yes all": campaign authority block, P7 statistics | [P7_STATISTICS.md](P7_STATISTICS.md), `docs/research/advice/2026-10-19_rules_audit_*` |
| 10-19 ~22:00 | Colab CLI re-qualified: final-only acceptance PASS on `dab00823` after two independent reviews | `research/colab_cli/QUALIFICATION.md` |
| 10-20 01:00 | After M2/M3 NO-GO: Amendment 2 — ranked labels (decisive = horizon rescues, delay weight 0), earliest-first batched Path B, paced retention test as the decider | DESIGN_V2 Amendment 2, `speed_advice/m2m3_*` |
| 10-20 12:30 | A5 v2 benchmark on 2 L4 lanes (owner OK'd Colab credits) | `snake-dqn-artifacts/t34-a5-20261020/` |
| 10-20 12:45 | Retention prereg FIX-FIRST #1 rulings: efficacy overruns diagnostic; near-zero exception only if mass is the sole failure; p95 at 24 fps; timing gate first; runner-enforced quiet host; `decision.json` sole verdict | `docs/research/specs/2026-10-20_t34_retention_prereg_fixes.md` |
| 10-20 13:10 | A5 failed by implementation (serial shadow fill) → batch shadow/root/branch steps across worlds, rerun A5 | `docs/research/specs/2026-10-20_t34_shadow_batching.md` |
| 10-20 13:30 | Pre-data resize after power review: 48 worlds/mix, 32 Watch6 sessions, caps 40 h / 60 CPU-h (joint GO 0.27–0.32 → 0.80–0.83) | `docs/research/specs/2026-10-20_t34_retention_resize.md`, Amendment 2 addendum |
| 10-20 13:50 | FIX-FIRST #2 rulings: import-closure source binding + frozen worktree; guard never sleeps mid-row, pauses only between rows, outstanding requests dropped; contention counted only >50% CPU >60 s, ≤2% of elapsed allowed; pre-check own 8 h cap; unpaced-arm cost smoke | `docs/research/specs/2026-10-20_t34_retention_fixes2.md` |
| 10-20 14:15 | FIX-FIRST #2 merged (154 tests, 180 bindings). Cost smoke 38.45 h vs 40 h cap (17.7 h unpaced) → unpaced descriptive arm moved to Colab lanes (standing Colab rule); Mac runs bare + paced only (~20.7 h) | `docs/research/specs/2026-10-20_t34_unpaced_arm_colab.md` |
| 10-20 14:30 | Shadow batching merged (byte-identical; Mac −19%, L4 untested). A5 v2 GO re-based pre-run: ≥ 50 completed worlds/h per L4 lane (v1 3k row-steps/s floor priced 0.4M rows/world). A5 #2 dispatched | DESIGN_V2 Amendment 2 addendum B |

## Measured results so far

| Result | Outcome |
|---|---|
| S1 (bare B8 fine-tunes, Colab) | KILL both arms (−12.5 / −14.9 pooled mass vs B8); audit PASS |
| M1 shadow identity | GO — 15,000 frames, 3 mixes, zero differences |
| M2 cheap LAD search vs T3 rescues | NO-GO (registered) — bisection 39/75, stride rule 37/75; on the 42 truth-capable rescues 39/42 and 37/42; the other 33 were all-die delays (+1–4 frames) |
| M3 serial live LAD search | NO-GO — 30–63 s per doom vs 12.5–24.9 s of lead (engine, not idea: 6.6 ms/row-step serial) |
| M4 B8 runtime | ONNX 0.14 ms batch-1 (≈3–5 % of the serial path end to end) |
| A5 v2 L4 benchmark #1 | ~50 active row-steps/s/lane vs 3,000 floor — serial shadow fill bug; 0 decisions; ~0.7 u |
| Retention power (development-only OC) | joint GO 0.80–0.83 at T3-like effects after resize; protected false-pass 0.026 ≤ 0.06 |

## Current status (update in place)

_As of 2026-10-20 ~14:00:_
- **Phase 0** (Mac, 6 workers): ~80/160 worlds; done ≈ Wed 10-21 ~13:00. Holds the Mac until then.
- **Sol running:** teacher shadow batching (`t34-shadow-batch-20261020`); unpaced arm → Colab
  (`t34-unpaced-colab-20261020`); Round 0 pre-registration on Colab (`t34-round0-prereg-20261020`).
- **Next, in order:** re-review retention → RATIFY; merge shadow batching → rerun A5 on L4 (~2 u) → ratify and
  dispatch Round 0 on Colab lanes; after Phase 0: quiet-Mac timing gate → T3-bank pre-check (≥ 50/64, 8 h cap) →
  registered paced retention study (≤ 40 h).
- **Colab:** qualified on `dab00823`; ~1,160.6 units; nothing running.
