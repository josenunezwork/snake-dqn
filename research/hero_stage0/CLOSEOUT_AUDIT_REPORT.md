# T34 Pre-Registered Closeout Audit — Readout & Verdict

- **Registration:** `/Users/josenunez/Projects/ml/snake-dqn/docs/research/final_run_20261019/CAMPAIGN_LOG.md` row `2026-11-04 PT 17:40`
- **Design:** `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-after-1b-advice/ADVICE.md` §3
- **Dev bank (Part B):** `t34-hero-probe-20261103-v1` (`48` worlds, `310` certified Stage 1B rescue roots, `10,080` ordinary roots; sealed gate bank `t34-openstop-gate-20261104-v1` untouched: `sealed_gate_namespace_touched = False`)
- **Part A Verdict (`b8-head-closedloop` receipt):** **`INCONCLUSIVE`** on full `SUCCESS_CONTRACT.md` §2 schema (`receipt.json` is a supervisor transport receipt containing no gameplay metrics directly, and its receipt-bound batch exports omit `boost_share`, plan §8 tripwires, and Watch6 v2), and **`0/4 PASS` (`KILL` under Stop Rule A)** on all measured `SUCCESS_CONTRACT.md` §2 bars present in the receipt-bound batch exports.
- **Part B Verdict (Counterfactual Live `open-stop` Regimes 1 & 2):** **`CLOSE T34`** (`0` regimes pass the `>= 50.0%` rescue retention, `LB90 >= 40.0%`, and `<= 1.0%` ordinary false-override bar).
- **Single Overall Pre-Registered Verdict:** **`CLOSE T34`**

---

## 1. Part A — `b8-head-closedloop` (`M0a`) Receipt & `SUCCESS_CONTRACT.md` §2 Audit

### 1.1 Exact Audit of `receipt.json` and Receipt-Bound Exports

1. **Supervisor receipt (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/b8-head-closedloop/main-01/output/receipt.json`, SHA-256 `afd7022c114538ae…`):**
   - `status = 'PASS'`, `stop_verified = True`, `cleanup_error = None`, `commit = '591d94c3847b45aff77558da8a0ebb0acc802f33'`, `manifest_sha256_matches_receipt = True`.
   - `receipt.json` is a Colab CLI transport/accounting receipt; it contains zero gameplay metrics directly and binds four exported files under `exports/closedloop/` (`summary.json` and `units/batch-0000..0002.json`).
   - Furthermore, `closedloop.py`'s `statistics(rows)` helper (`closedloop_readout.py`) only computes `{worlds, mass, mass_delta, lb90, ci95, deaths, alive_fraction, overrides, decisions, override_rate}`, omitting even the scripted no-death gain (`>= +0.03`) and survival band v2 `LB95` non-inferiority bounds (`> -0.05` pooled, `> -0.075` per mix), though those two are computable directly from the `deaths` and `alive_fraction` columns in `batch-0000..0002.json`.
2. **Missing `SUCCESS_CONTRACT.md` §2 fields in the receipt-bound batch exports (`exports/closedloop/units/batch-0000..0002.json`):**
   - All three batch units pass byte-size, SHA-256, and `.complete` sidecar verification (`960` rows = `192` paired v1 terminal-hero `H=5,000` worlds × `5` arms `b8, m0, m10, m25, m50`).
   - Each exported row contains **only** `['alive_fraction', 'arm', 'deaths', 'decisions', 'initial_sha256', 'mass', 'mix', 'overrides', 'world_id']`.
   - **Unrecorded bars required by `SUCCESS_CONTRACT.md` §2 (`P7_STATISTICS.md` §4):**
     - **Boost share within `[0.5%, 10%]`:** Not recorded in `receipt.json` or `batch-0000..0002.json` (`closedloop.py` did not accumulate boost frames or boost decisions).
     - **Plan §8 tripwires:** Unconditional action-collapse (`> 90%` one-action share in any 100k-decision window) and fit-drift tripwires on the 32-world probe bank (`disagreement vs previous student <= 5%` at every length decile, `boost share <= 1.5x` previous student, `held-out decisive agreement >= 70%`) are not recorded.
     - **Simulator version:** `closedloop.py:170` ran `PilotCTDeviceSim` with `mechanics_version=1` on `64` disjoint worlds per mix rather than Watch6 `mechanics_version=2` (`t34-sol/b8-head-closedloop/SUMMARY.md:45`).
   - Because these `SUCCESS_CONTRACT.md` §2 bars are absent from the receipt and its exports, **Part A is formally reported as `INCONCLUSIVE` on the full §2 contract schema** (no unlogged numbers are invented).

### 1.2 Measured `SUCCESS_CONTRACT.md` §2 & `ADVICE.md` §3.1 Readout on the Receipt-Bound Batch Exports

Using `research/b8_head_test/closedloop.py`'s exact stratified paired-parent bootstrap (`10,000` draws, `seed = 343014`, equal mix weights across `64` scripted + `64` frozen + `64` mixed worlds = `192` worlds):

| Arm | Pooled Mass | Pooled Δ Mass (`>= +40`) | Pooled `LB90` (`> 0`) | Pooled `CI95` | Scripted No-Death Rate (`Δ >= +0.03`) | Pooled Alive Frac (`Δ / t-LB95 > -0.05`) | Worst Mix Alive `t-LB95` (`> -0.075`) | Living Override Rate (`<= 15%`) | Boost Share `[0.5%, 10%]` & Tripwires | Measured Bars Pass |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| `b8` (baseline) | `306.152` | `+0.000` | `+0.000` | `[+0.000, +0.000]` | `0.5625` (`36/64`, `+0.0000`) | `0.7293` (`+0.0000` / `+0.0000`) | `+0.0000` | `0.00%` | Not in receipt/exports | Baseline |
| `m0` | `1.057` | **`-305.095`** | **`-321.688`** | `[-330.178, -280.012]` | `0.3281` (`21/64`, **`-0.2344`**) | `0.7301` (`+0.0008` / **`-0.0533`**) | **`-0.3206`** | `60.21%` | `INCONCLUSIVE` (missing) | **`FAIL`** |
| `m10` | `5.724` | **`-300.428`** | **`-316.952`** | `[-325.322, -275.401]` | `0.2656` (`17/64`, **`-0.2969`**) | `0.5686` (`-0.1607` / **`-0.2094`**) | **`-0.3273`** | `36.97%` | `INCONCLUSIVE` (missing) | **`FAIL`** |
| `m25` | `193.073` | **`-113.079`** | **`-132.236`** | `[-141.900, -84.022]` | `0.0469` (`3/64`, **`-0.5156`**) | `0.6171` (`-0.1122` / **`-0.1530`**) | **`-0.2556`** | `3.86%` | `INCONCLUSIVE` (missing) | **`FAIL`** |
| `m50` | `279.936` | **`-26.216`** | **`-45.057`** | `[-55.588, +3.056]` | `0.3594` (`23/64`, **`-0.2031`**) | `0.7017` (`-0.0276` / **`-0.0652`**) | **`-0.1393`** | `0.20%` | `INCONCLUSIVE` (missing) | **`FAIL`** |

#### Per-Mix Breakdown (`batch-0000..0002.json`)

| Arm | Mix | Worlds | Mean Mass | Paired Δ Mass | `LB90` | `CI95` | Deaths | No-Death Rate (`Δ vs b8`) | Alive Frac (`Δ vs b8`) | Alive `t-LB95` / `boot-LB95` | Overrides / Decisions (`%`) |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `b8` | `scripted` | `64` | `387.470` | `+0.000` | `+0.000` | `[+0.000, +0.000]` | `28` | `0.5625` (`+0.0000`) | `0.7772` (`+0.0000`) | `+0.0000` / `+0.0000` | `0 / 248743` (`0.00%`) |
| `b8` | `frozen` | `64` | `226.290` | `+0.000` | `+0.000` | `[+0.000, +0.000]` | `50` | `0.2188` (`+0.0000`) | `0.6491` (`+0.0000`) | `+0.0000` / `+0.0000` | `0 / 207757` (`0.00%`) |
| `b8` | `mixed` | `64` | `304.696` | `+0.000` | `+0.000` | `[+0.000, +0.000]` | `38` | `0.4062` (`+0.0000`) | `0.7616` (`+0.0000`) | `+0.0000` / `+0.0000` | `0 / 243758` (`0.00%`) |
| `b8` | `all` | `192` | `306.152` | `+0.000` | `+0.000` | `[+0.000, +0.000]` | `116` | `0.3958` (`+0.0000`) | `0.7293` (`+0.0000`) | `+0.0000` / `+0.0000` | `0 / 700258` (`0.00%`) |
| `m0` | `scripted` | `64` | `0.813` | `-386.657` | `-419.180` | `[-436.981, -334.521]` | `43` | `0.3281` (`-0.2344`) | `0.5597` (`-0.2175`) | `-0.3206` / `-0.3183` | `119215 / 179158` (`66.54%`) |
| `m0` | `frozen` | `64` | `1.219` | `-225.071` | `-250.200` | `[-263.140, -187.418]` | `19` | `0.7031` (`+0.4844`) | `0.8442` (`+0.1952`) | `+0.1047` / `+0.1064` | `156097 / 270178` (`57.78%`) |
| `m0` | `mixed` | `64` | `1.140` | `-303.556` | `-329.924` | `[-344.009, -263.726]` | `26` | `0.5938` (`+0.1875`) | `0.7864` (`+0.0248`) | `-0.0652` / `-0.0646` | `146735 / 251677` (`58.30%`) |
| `m0` | `all` | `192` | `1.057` | `-305.095` | `-321.688` | `[-330.178, -280.012]` | `88` | `0.5417` (`+0.1458`) | `0.7301` (`+0.0008`) | `-0.0533` / `-0.0535` | `422047 / 701013` (`60.21%`) |
| `m10` | `scripted` | `64` | `3.778` | `-383.691` | `-416.142` | `[-433.922, -331.472]` | `47` | `0.2656` (`-0.2969`) | `0.5495` (`-0.2278`) | `-0.3273` / `-0.3243` | `76389 / 175878` (`43.43%`) |
| `m10` | `frozen` | `64` | `4.984` | `-221.307` | `-246.456` | `[-259.275, -183.571]` | `49` | `0.2344` (`+0.0156`) | `0.5876` (`-0.0615`) | `-0.1451` / `-0.1427` | `73379 / 188073` (`39.02%`) |
| `m10` | `mixed` | `64` | `8.409` | `-296.287` | `-322.224` | `[-336.025, -257.055]` | `49` | `0.2344` (`-0.1719`) | `0.5687` (`-0.1930`) | `-0.2626` / `-0.2624` | `52059 / 182019` (`28.60%`) |
| `m10` | `all` | `192` | `5.724` | `-300.428` | `-316.952` | `[-325.322, -275.401]` | `145` | `0.2448` (`-0.1510`) | `0.5686` (`-0.1607`) | `-0.2094` / `-0.2079` | `201827 / 545970` (`36.97%`) |
| `m25` | `scripted` | `64` | `211.720` | `-175.750` | `-213.768` | `[-233.857, -116.513]` | `61` | `0.0469` (`-0.5156`) | `0.6014` (`-0.1758`) | `-0.2556` / `-0.2519` | `8048 / 192519` (`4.18%`) |
| `m25` | `frozen` | `64` | `162.122` | `-64.168` | `-93.544` | `[-109.851, -19.541]` | `63` | `0.0156` (`-0.2031`) | `0.5873` (`-0.0617`) | `-0.1317` / `-0.1302` | `6871 / 188013` (`3.65%`) |
| `m25` | `mixed` | `64` | `205.376` | `-99.319` | `-128.219` | `[-143.239, -55.339]` | `63` | `0.0156` (`-0.3906`) | `0.6626` (`-0.0990`) | `-0.1622` / `-0.1601` | `7971 / 212108` (`3.76%`) |
| `m25` | `all` | `192` | `193.073` | `-113.079` | `-132.236` | `[-141.900, -84.022]` | `187` | `0.0260` (`-0.3698`) | `0.6171` (`-0.1122`) | `-0.1530` / `-0.1522` | `22890 / 592640` (`3.86%`) |
| `m50` | `scripted` | `64` | `324.176` | `-63.294` | `-99.022` | `[-118.376, -7.759]` | `41` | `0.3594` (`-0.2031`) | `0.7028` (`-0.0745`) | `-0.1393` / `-0.1372` | `270 / 224924` (`0.12%`) |
| `m50` | `frozen` | `64` | `210.350` | `-15.940` | `-44.322` | `[-58.661, +28.029]` | `52` | `0.1875` (`-0.0312`) | `0.6255` (`-0.0236`) | `-0.0885` / `-0.0865` | `528 / 200200` (`0.26%`) |
| `m50` | `mixed` | `64` | `305.280` | `+0.585` | `-33.220` | `[-51.416, +53.001]` | `42` | `0.3438` (`-0.0625`) | `0.7768` (`+0.0152`) | `-0.0522` / `-0.0497` | `532 / 248613` (`0.21%`) |
| `m50` | `all` | `192` | `279.936` | `-26.216` | `-45.057` | `[-55.588, +3.056]` | `135` | `0.2969` (`-0.0990`) | `0.7017` (`-0.0276`) | `-0.0652` / `-0.0649` | `1330 / 673737` (`0.20%`) |

- **Stop Rule A Application:** `0 / 4` candidate arms (`m0, m10, m25, m50`) pass the measured `SUCCESS_CONTRACT.md` §2 bars (`H=5,000` pooled mass `>= +40`, `LB90 > 0`, scripted no-death gain `>= +0.03`, survival band v2). Every arm loses mass (`-305.10` to `-26.22`), has negative `LB90` (`-321.69` to `-45.06`), loses scripted no-death rate (`-20.31 pp` to `-51.56 pp`), and fails survival band v2 (`pooled t-LB95 <= -0.0533 < -0.050`, `scripted t-LB95 <= -0.1393 < -0.075`).

---

## 2. Part B — Counterfactual Live `open-stop` Audit (`48` Dev Worlds, `310` Rescue Roots, `10,080` Ordinary Roots)

### 2.1 Primary Decision Table (Pre-Registered Stop Rule B)

- **Pre-Registered Bar (`CAMPAIGN_LOG.md` row `2026-11-04 PT 17:40`, `ADVICE.md` §3.2):** Retain **`>= 50.0%` (`>= 155 / 310`)** of certified rescue roots with **world-cluster bootstrap `LB90 >= 40.0%`** AND **`<= 1.0%` (`<= 100 / 10,080`)** ordinary false overrides.

| Regime / Variant | Triggered Overrides on `310` Rescues | `H=32` Rescue Retention (`suf0` Certified) | World-Cluster `LB90` (`CI95`) | `H=32` Rescue Retention (`suf0` Cert & `suf4..7 n_diff==0`) | `H=32` Rescue Retention (`suf4..7 n_diff==0`) | Exact Teacher Match | Ordinary False Overrides (`/ 10,080`) | Lethal Ord Flips (`suf4..7 Cert / Raw`) | Passes Stop Rule B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Regime 1 (`H=12, <=2 branches`, `exact_stop`)** | `12 / 310` (`3.87%`) | **`5 / 310` (`1.61%`)** | **`0.72%`** (`[0.35%, 3.02%]`) | `5 / 310` (`1.61%`, `LB90 0.72%`) | `12 / 310` (`3.87%`, `LB90 2.38%`) | `11 / 310` (`3.55%`) | **`4 / 10080` (`0.04%`)** | `1` / `0` | **`FAIL`** |
| Regime 1 (`H=12, <=2 branches`, certified-only) | `5 / 310` (`1.61%`) | **`5 / 310` (`1.61%`)** | **`0.72%`** (`[0.35%, 3.02%]`) | `5 / 310` (`1.61%`, `LB90 0.72%`) | `5 / 310` (`1.61%`, `LB90 0.72%`) | `5 / 310` (`1.61%`) | **`4 / 10080` (`0.04%`)** | `1` / `0` | **`FAIL`** |
| **Regime 2A (`Async d=3`, `t-3` override -> step-`t` action)** | `38 / 310` (`12.26%`) | **`20 / 310` (`6.45%`)** | **`4.78%`** (`[3.98%, 9.12%]`) | `20 / 310` (`6.45%`, `LB90 4.78%`) | `37 / 310` (`11.94%`, `LB90 9.74%`) | `18 / 310` (`5.81%`) | **`13 / 10080` (`0.13%`)** | `2` / `0` | **`FAIL`** |
| **Regime 2B (`Async d=3`, `t-3` override -> root action at `t`)** | `77 / 310` (`24.84%`) | **`40 / 310` (`12.90%`)** | **`10.44%`** (`[9.14%, 16.83%]`) | `39 / 310` (`12.58%`, `LB90 10.18%`) | `75 / 310` (`24.19%`, `LB90 21.23%`) | `38 / 310` (`12.26%`) | **`22 / 10080` (`0.22%`)** | `4` / `0` | **`FAIL`** |
| Regime 2C (`Async d=3`, unconditional `t-3` step-`t` action) | `40 / 310` (`12.90%`) | **`21 / 310` (`6.77%`)** | **`5.07%`** (`[4.23%, 9.45%]`) | `21 / 310` (`6.77%`, `LB90 5.07%`) | `39 / 310` (`12.58%`, `LB90 10.38%`) | `19 / 310` (`6.13%`) | **`73 / 10080` (`0.72%`)** | `2` / `0` | **`FAIL`** |

### 2.2 Length-Stratified (`L < 600` vs `L >= 600`) & Raw-Survival Diagnostics

| Regime / Variant | `L < 600` Certified Rescues (`111` roots) | `L >= 600` Certified Rescues (`199` roots) | `H=32` Raw Survival on `310` Rescues (`suf0`) | Raw `342` Rescue Set (`suf0` Certified) | Ordinary False Override `CI95` | Ordinary Flips Failing `suf0 H=32` |
|---|---:|---:|---:|---:|---:|---:|
| **Regime 1 (`H=12, <=2 branches`, `exact_stop`)** | `2 / 111` (`1.80%`, `LB90 0.00%`) | `3 / 199` (`1.51%`, `LB90 0.51%`) | `5 / 310` (`1.61%`, `LB90 0.72%`) | `5 / 342` (`1.46%`, `LB90 0.65%`) | `[0.01%, 0.08%]` | `2` |
| Regime 1 (`H=12, <=2 branches`, certified-only) | `2 / 111` (`1.80%`, `LB90 0.00%`) | `3 / 199` (`1.51%`, `LB90 0.51%`) | `5 / 310` (`1.61%`, `LB90 0.72%`) | `5 / 342` (`1.46%`, `LB90 0.65%`) | `[0.01%, 0.08%]` | `2` |
| **Regime 2A (`Async d=3`, `t-3` override -> step-`t` action)** | `6 / 111` (`5.41%`, `LB90 2.74%`) | `14 / 199` (`7.04%`, `LB90 5.03%`) | `32 / 310` (`10.32%`, `LB90 8.18%`) | `21 / 342` (`6.14%`, `LB90 4.46%`) | `[0.06%, 0.21%]` | `6` |
| **Regime 2B (`Async d=3`, `t-3` override -> root action at `t`)** | `13 / 111` (`11.71%`, `LB90 7.69%`) | `27 / 199` (`13.57%`, `LB90 10.46%`) | `67 / 310` (`21.61%`, `LB90 18.96%`) | `45 / 342` (`13.16%`, `LB90 10.86%`) | `[0.13%, 0.33%]` | `9` |
| Regime 2C (`Async d=3`, unconditional `t-3` step-`t` action) | `7 / 111` (`6.31%`, `LB90 3.39%`) | `14 / 199` (`7.04%`, `LB90 5.03%`) | `34 / 310` (`10.97%`, `LB90 8.82%`) | `22 / 342` (`6.43%`, `LB90 4.73%`) | `[0.57%, 0.89%]` | `7` |

### 2.3 Structural Mechanism Breakdown on the `310` Certified Rescue Roots

1. **Why Regime 1 (`H=12, <= 2 branches`) fails:**
   - On **`236 / 310` (`76.13%`)** of certified `H=32` rescue roots, `bare` is **still alive and un-enclosed at step `12`** (`H=12` certified). Consequently, `H=12` terminates at Branch 1 (`bare`) without inspecting any alternate action.
   - On the remaining `74 / 310` (`23.87%`) roots where `bare` fails at `H=12` (`9` raw deaths within 12 steps and `65` step-12 enclosures), the `<= 2` branch cap evaluates only Branch 2 (second-highest `q`): Branch 2 is certified at `H=12` on only `5 / 310` (`1.61%`) roots, and under `exact_stop` tie-breaking (`frames_survived` at `H=12`) overrides `bare` on only `12 / 310` (`3.87%`) roots, retaining **`5 / 310` (`1.61%`, `LB90 = 0.72%`)** on `suf0` (`12 / 310 = 3.87%`, `LB90 = 2.38%` on `suf4..7`).
   - Even removing the `<= 2` branch cap completely (`H=12, <= 6 branches` diagnostic), `H=12` retains only **`11 / 310` (`3.55%`, `LB90 = 2.15%`)** on `suf0` (`19 / 310 = 6.13%` on `suf4..7`), because self-enclosure traps seal between steps `13` and `32`.
2. **Why Regime 2 (`Async d=3-tick / 125 ms delay`, launched at `t - 3`) fails:**
   - On `309 / 310` rescue roots where the hero is in the same life at `t - 3`, `open-stop-h32-nb` (`K=1, H=32`) launched at `t - 3` sees `bare` **still certified safe over `[t - 3, t + 29]` on `156 / 310` (`50.32%`)** of roots (`154 / 310 = 49.68%` Branch-1 failures at `t - 3`), overriding `bare` at `t - 3` on **`103 / 310` (`33.23%`)** of roots.
   - When `t - 3` overrides (`chosen_{t-3} != bare_{t-3}`), its winning rollout diverges at step `t - 3` from the actual world (which stepped `t - 3 -> t` under `bare` and 5 live opponents). Delivering either the winning branch's planned step-`t` action (Regime 2A: `20 / 310 = 6.45%`, `LB90 = 4.78%` on `suf0`; `37 / 310 = 11.94%`, `LB90 = 9.74%` on `suf4..7`) or the delayed root action `chosen_{t-3}` at `t` (Regime 2B: `40 / 310 = 12.90%`, `LB90 = 10.44%` on `suf0`; `75 / 310 = 24.19%`, `LB90 = 21.23%` on `suf4..7`) fails the `>= 50.0%` (`LB90 >= 40.0%`) retention bar by a wide margin.

---

## 3. Pre-Registered Stop Rules & Final Verdict

- **Stop Rule A (`Part A`):** `INCONCLUSIVE` on the full `SUCCESS_CONTRACT.md` §2 schema (missing `boost_share` and plan §8 tripwires in `receipt.json` / batch exports); `0 / 4` arms pass the measured `SUCCESS_CONTRACT.md` §2 bars (`KILL` on measured bars).
- **Stop Rule B (`Part B`):** Neither Regime 1 (`H=12, <= 2 branches`) nor Regime 2 (`Async d=3-tick delay` from `t - 3`) retains `>= 50.0%` (`>= 155 / 310`, `LB90 >= 40.0%`) of certified rescues with `<= 1.0%` ordinary false overrides.
- **Single Verdict:** **`CLOSE T34`**
