# Step −1 world-step ruler: pre-registered protocol (frozen before data)

Status: **pre-data**. Authority: `snake-dqn-artifacts/big-lever/exp/JUDGMENT_GPU_LABEL_PATH_20261024.md`
§3 step 2 (SHA-256 `ef488af238e1f60818dc5baea9fbc17001ea9fe09c1590867f3ce00cc08f4c00`). The definition and
speed-up formula come from `ADVICE_GPU_LABEL_PATH_20261024.md` §5
(`7566eb9c3054643ec43de1354d0eae3aa0837719129d8634f1d1ed5878007aa8`).

This is a premise test, not a speed row. It makes no label, gate, fit, serving or release claim. The F0/D1024
factory, its leases and `exp/mac-lease.json*` are untouched. `t34-round0-study-01/` is never opened.

## 1. Question

Is `T_gpu,eff / T_fleet` ≥ 4 in world-steps/s of real label-factory work? Here one world-step is one
`GridBatchSim.step` of one six-seat world, the same unit as the R2/R6 GPU rates.

## 2. Work per parent (dev unit, Colab CPU runtime)

- **Collector.** The unchanged F0 `collect_unit` (`research/bl_labels/factory.py`) runs at the measurement pin.
  - D1024 runs the OLD collector `61900af2`. The diff `61900af2..HEAD` in `factory.py`/`b2_engine.py` changes only
    how the bare decision and Q are computed (prepared packet, an optional shadow check), ring-pruning code and one
    more bare-audit compared field. It changes no step, restore, replay or branch control flow, so world-step counts
    per parent have the same distribution.
  - `prepared_decision_shadow` is left absent, matching the OLD behaviour.
- **Counters** (`ticks.py`), all observational:
  - every `WatchDriver.step` (= one world-step) is attributed to a phase: **play** (natural trajectory),
    **replay** (inside `factory._restore_root`) or **branch** (inside `factory._branch`: alternatives, bare audits,
    ordinary roots and H1440, deliberately not split);
  - restore and snapshot counts per phase.
- **Work per parent** is `W_p = play + replay + branch` ticks.
- **What is never written.** Label rows, outcomes, decisive/rescue/H1440/reversal counts. `publish_life` is a
  no-op. The unit exports keep only work counters (deaths, roots, digest checks, bare rerolls) and timing.
- **Allocation.** Dev namespace `bl-step-ruler-<commit>-dev`, 12 parents in this order:
  - 4 Watch6 (H14400);
  - 3 gate-frozen, 3 gate-mixed, 2 gate-scripted (H5000).
  - Seeds are `sha256("<namespace>/<stream>/<i>")[:4]`. Gate plans are frozen at preparation by the F0
    `materialize_gate_plans`.
  - A literal whole-word seed audit runs over this repo and the artifacts tree, excluding
    `t34-round0-study-01/` (never opened). Any hit refuses preparation.
- **Platform.** Colab CPU runtime (x86, nproc 2), one worker per vCPU, one numerical thread per worker. Never the
  leased Mac. World-step counts are platform-invariant in distribution; the x86 per-worker steps/s is recorded only
  for the L4 proxy in §4.
- **Canary first.** An excluded same-entrypoint probe uses namespace `…-transport-probe`, the first parent of each
  stream at horizon 3, plus a real-code counter self-check: 2 replay ticks/1 restore and one H300 branch. Its
  outputs never enter a ledger.

## 3. Estimators (decision statistic = point estimate)

- **Steps per parent.** `k_s` = mean `W_p` over COMPLETE verified dev units of stream
  s ∈ {watch6, gate-frozen, gate-mixed, gate-scripted}.
- **Fleet rate for Mac width w**, from metadata only, at analysis time:
  - Sources: start/terminal JSON in
    - `bl-labels-f0-speed/d1024-old-w6-equiv-02-pack-01/d1024-state-mac-01`;
    - `bl-labels-f0/mac-pack-02/d256-state-mac-01`.
  - Eligible attempts:
    - start `source_commit` ∈ {`61900af2…` (D256: the collector pin itself), `111c4781…` (D1024
      science/admission pin; per `research/bl_labels/D1024_RESUME_HANDOFF_20261024.md` it runs the OLD collector
      `61900af2`, W6)};
    - a terminal receipt, when present, must have `collector_source_commit` =
      `61900af28932e47e22e332840fc4406559adb062`;
    - darwin-arm64;
    - no width fallback, and start workers = terminal workers.
  - Terminal attempt: use its `completed_parents` and `elapsed_seconds`.
  - Live attempt (no terminal): use the `lives/*--session.json.complete` markers in its output dir, by **stat
    mtime only**. A marker counts if its mtime is ≥ `started_epoch`. Elapsed is the latest such mtime minus
    `started_epoch`. Session payloads are never opened.
  - Formula: `T_fleet,w = Σ_{attempts at w} Σ_{completed parents p} k_{stream(p)} / Σ_{attempts at w} elapsed`
    (world-steps/s). This equals steps-per-parent × parents-per-hour over the actual mix.
  - Parents/h per width is also reported.
- **GPU rate.**
  - `R_best` = maximum, over eligible cells, of the cell's median continuous world-steps/s.
  - Eligible cells:
    - R2 `R2_ANALYSIS.json` (`0b253f33a0f59fd0…`, verdict R2_ADVANCE): E1024 I′ per roster, median of 3;
    - R6 `gpu-r6-escaling-20261024/R6_ANALYSIS.json`, only if, at analysis time, its sha256/complete sidecars verify,
      its verdict ∈ {KEEP_E4096, DROP_E4096, INCOMPLETE}, and the cell has `exact` true. Cells at
      E ∈ {1024, 2048, 4096}, by `median_rate`.
  - Extrapolated E values never count.
  - `T_gpu,eff(f) = f × R_best` for f = 0.35 (pessimistic) and f = 0.65 (optimistic).

## 4. Ratios and pre-registered decision

- **Primary fleet: Mac W6** (the judgment's "today: Mac W6").
- `ratio(f) = T_gpu,eff(f) / T_fleet,W6`.
- **Stop rule:**
  - `ratio(0.65) < 4` → **CLOSE_NO_BOUND**. The exact-engine route closes and S0 is not built.
  - `ratio(0.65) ≥ 4` → **S0_ADMITTED**.
- **Pessimistic record:**
  - `ratio(0.35) ≥ 4` → `PESSIMISTIC_PASS`;
  - otherwise `PESSIMISTIC_BELOW_4`: any later S2 must measure f directly in its canary and stop below 4×.
- **Sensitivities (recorded, never change the verdict):**
  - Mac W12 (owner-approved and calibrated, ADOPT_W12). A ratio below 4 is flagged `W12_BELOW_4`.
  - Each width plus 3 L4 slots. **PROXY:** no measured L4 F0 rate exists, because F1 ended NO_PRODUCTION_PLATFORM.
    One L4 slot = 12 × (x86 per-worker world-steps/s measured by this run), an upper bound with no contention.
  - Per-roster GPU rates (watch6 shown against every roster) and per-stream ratios.
- **Validity → INCOMPLETE (no verdict):**
  - fewer than 6 COMPLETE verified dev units;
  - a stream with zero COMPLETE units;
  - no eligible W6 attempt;
  - no eligible GPU cell.
- **INVALID:** any INVALID unit (digest or bare-audit mismatch), or a foreign or duplicate unit.
- **One look.** The analyzer runs once, after the series ends (all 12 verified, or the original plus at most
  2 outcome-blind completion attempts used). It publishes create-only `STEP_M1_RULER.json` (+ `.sha256`,
  `.complete`).

## 5. Envelopes (owner 2× rule; ruling guard ≤ 1 u; CPU 0.08 u/h)

| Stage | Base → effective minutes | Base → effective units | Base → effective reserve |
|---|---:|---:|---:|
| Transport probe (canary) | 10 → 20 | 0.02 → 0.04 | 0.004 → 0.008 |
| Main (original + ≤ 2 completions share it) | 130 → 260 | 0.18 → 0.36 | 0.036 → 0.072 |
| Series total | 140 → 280 | **0.20 → 0.40 (≤ 1 u guard)** | |

- Forecast:
  - 4 Watch6 × ~30–35 x86 worker-min, plus 8 gate × ~3 min, on 2 workers ≈ 80 min;
  - plus setup, export and margin.
- First cap wins. No automatic retry and no cap raise mid-series.
- Completion attempts need a verified stop and missing units, use a fresh manifest and output, and skip the
  ledger-verified units.
