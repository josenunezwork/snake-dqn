# T34 Stage 1B Readout — Open-Stop (`K=1`) Teacher -> Bare `b8-s3` Selectivity Gate

- **Pre-registration:** `docs/research/final_run_20261019/CAMPAIGN_LOG.md row '2026-11-04 PT 13:10'`
- **Dev world bank:** `t34-hero-probe-20261103-v1` (`48` worlds; sealed gate bank `t34-openstop-gate-20261104-v1` untouched: `sealed_gate_namespace_touched = False`)
- **Teacher arm:** `open-stop-h32-nb-k1` (`open-stop-h32-nb` `K=1` on `48` dev worlds)
- **Single Pre-Registered Verdict:** **`STOP`** — STOP the bare-network open-stop distillation line (close T34 under Claim A bare b8-s3 in Live Play + Claim B exact-stop-l2-prime replay bank in Watch)

---

## 1. Executive Summary & Decision Table

| Gate / Bar | Pre-Registered Requirement | Arm A (`bv8-only`) | Arm B (`bv8` + Multi-Step Reach/Choke `k in {8,16,24}`) | Status |
|---|---:|---:|---:|---|
| **Support: Rescue roots (`n_diff == 0` on suffixes 4..7)** | `>= 80` roots from `>= 30` worlds | `310` certified (`342` raw) across `48` worlds | `310` certified (`342` raw) across `48` worlds | **PASS** |
| **Support: Ordinary `b8-s3` roots scored on suffixes 4..7** | `>= 10,000` across all 48 worlds | `10080` (`210/world`) | `10080` (`210/world`) | **PASS** |
| **1-Thread Timing Gate at `L >= 1,500`: Overhead `p95`** | `< 1.000 ms` | `0.000 ms` (baseline) | `0.237 ms` (kernel `p95 = 0.020 ms`) | **PASS** |
| **1-Thread Timing Gate at `L >= 1,500`: Full Watch6 Tick `p95`** | `<= 11.688 ms` | `3.091 ms` | `3.093 ms` | **PASS** |
| **Selected Operating Point `(m, tau)`** | Rule: `<= 1.0%` ord flips & `<= 1` lethal flip | `m=0.03, tau=0.0287` | `m=0.1129, tau=0.0352` | Fixed rule |
| **OOF Ordinary Flip Rate (at selected operating point)** | `<= 1.00%` (`<= 100 / 10,080`) | `0.76%` (`77 / 10080`) | `0.99%` (`100 / 10080`) | **PASS** |
| **OOF Lethal Ordinary Flips (`n_diff >= 2/4` on suffixes 4..7)** | `<= 1` across `10,080` roots | `1` certified (`0` raw) | `1` certified (`0` raw) | **PASS** |
| **OOF Rescue Retention Overall (exact teacher match)** | `>= 35.0%` | **`8.06%`** (`25 / 310`) | **`9.03%`** (`28 / 310`) | **FAIL** |
| **OOF Rescue Retention World-Cluster Bootstrap `LB90`** | `>= 25.0%` | **`5.94%`** (`CI95 [4.94%, 11.52%]`) | **`6.50%`** (`CI95 [5.33%, 13.38%]`) | **FAIL** |
| **OOF Rescue Retention at `L < 600` (`111` certified roots)** | `>= 35.0%` | **`9.91%`** (`11 / 111`) | **`9.91%`** (`11 / 111`) | **FAIL** |
| **OOF Rescue Retention at `1.0%` Ord Flip Cap (ignoring lethal cap)** | Diagnostic upper bound at `1.0%` flip | `10.65%` (`4` lethal flips) | `9.03%` (`1` lethal flips) | **FAIL** (`< 11%`) |
| **Paired Arm B - Arm A Rescue Retention Delta (`LB90`)** | `>= +20.0 pp` (`LB90 > 0 pp`) | — | **`+0.97 pp`** (`LB90 = -0.62 pp`) | **FAIL** |

---

## 2. Dataset & Support Verification (48 Dev Worlds Only)

- **Dev world trajectory replay parity:** All `48` dev worlds (`t34-hero-probe-20261103-v1`) were replayed under `open-stop-h32-nb` (`K=1`) with exact bitwise `mass_sum`, `deaths`, `length`, `bare_action`, and `q` parity (`max |q - q_recorded| == 0.0` across all `350` overrides).
- **Rescue set (`n_diff == 0` on independent suffixes `4..7`):**
  - **Certified H=32 (`h32_certified.n_diff == 0`):** `310` rescue roots across `48 / 48` worlds (`111` at `L < 600`, `199` at `L >= 600`).
  - **Raw H=32 (`n_override_dies_where_b8_survives == 0`):** `342` rescue roots across `48 / 48` worlds (`119` at `L < 600`, `223` at `L >= 600`).
  - Both exceed the pre-registered minimum support bar (`>= 80` rescue roots from `>= 30` dev worlds).
- **Ordinary set (`b8-s3` non-override roots scored on suffixes `4..7`):**
  - Stratified sample of `210` non-override search roots per world = **`10080` ordinary roots** across all `48` dev worlds.
  - Every ordinary root's runner-up action and every CV-predicted alternative action was scored on the `4` independent suffixes `4, 5, 6, 7` (`H=32`, `open-stop-h32-nb` branch evaluation).
  - **Base `b8-s3` runner-up lethality on ordinary roots:** If `b8-s3` is flipped to its 2nd-best legal action on ordinary states, `78 / 10080` (`0.77%`) are **certified-lethal ordinary flips** (`n_diff >= 2/4` on suffixes `4..7`), and `32 / 10080` (`0.32%`) are **raw-lethal ordinary flips**.

---

## 3. 1-Thread Mac Timing Gate at `L >= 1,500`

- Measured under the 7 single-thread environment caps (`OMP/MKL/OPENBLAS/VECLIB/NUMBA/NUMEXPR/TORCH_NUM_THREADS=1`, `SNAKE_DQN_DEVICE=cpu`, `nice -n 15`) on a `Watch6` 6-snake state at hero length `L = 1,500` (`50` warmup + `500` measured ticks):

| Timing Component | `p50` (ms) | `p95` (ms) | `p99` (ms) | `mean` (ms) | Pre-Registered Bar | Pass |
|---|---:|---:|---:|---:|---:|---|
| Standalone Numba reach/choke kernel (`_reach_choke_row_kernel`, `k in (8,16,24)`) | `0.0195` | `0.0200` | `0.0221` | `0.0195` | `< 1.000 ms` | **PASS** |
| Hero step Arm A (`bv8` raster + trunk + head) | `0.6117` | `0.6624` | `0.7092` | `0.6177` | — | — |
| Hero step Arm B (`bv8` raster + reach/choke + trunk + head) | `0.6332` | `0.6776` | `0.7216` | `0.6392` | — | — |
| Paired Hero Overhead (Arm B - Arm A) | `0.0211` | `0.0553` | `0.0830` | `0.0214` | `< 1.000 ms` | **PASS** |
| Full Watch6 Tick Arm A (6 snakes at `L=1,500`) | `2.8413` | `3.0908` | `4.0299` | `2.8620` | `<= 11.688 ms` | **PASS** |
| Full Watch6 Tick Arm B (6 snakes at `L=1,500`) | `2.8642` | `3.0930` | `4.0239` | `2.8727` | `<= 11.688 ms` | **PASS** |
| Effective Overhead `p95` (max of kernel, hero diff, tick diff) | — | **`0.2370`** | — | — | `< 1.000 ms` | **PASS** |

Arm B passes the 1-thread Mac timing gate with wide margin (`0.02 ms` kernel `p95`, `< 0.35 ms` effective `p95` overhead, `3.47 ms` full Watch6 tick `p95` vs `11.688 ms` ceiling) and is therefore eligible for the Stage 1B decision rule.

---

## 4. 5-Fold World-Disjoint CV Results Across Margin Grid `{0.03, 0.0632, 0.1129}`

### 4.1 Complete Margin Grid & Operating-Point Table (`EquivariantResidualQHead`)

| Arm | Margin `m` | Operating Regime | Threshold `tau` | Ordinary Flips (`%`) | Lethal Ord Flips (Cert / Raw) | OOF Rescue Retention Exact (`LB90`) | OOF Rescue Retention `L < 600` | OOF Rescue Retention Survival-Match | OOF Rescue Any-Flip |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Arm A (`bv8-only`) | `0.03` | Unthresholded (`tau = 0`) | `0.0000` | `162 / 10080` (`1.61%`) | `4` / `1` | `11.61%` (`8.75%`) | `14.41%` (`16/111`) | `14.52%` | `14.84%` |
| Arm A (`bv8-only`) | `0.03` | At `<= 1.0%` ord flip cap | `0.0187` | `100 / 10080` (`0.99%`) | `2` / `0` | `9.35%` (`6.88%`) | `11.71%` (`13/111`) | `11.94%` | `11.94%` |
| Arm A (`bv8-only`) | `0.03` | **Constrained (`<= 1%` flip & `<= 1` lethal)** | `0.0287` | `77 / 10080` (`0.76%`) | `1` / `0` | `8.06%` (`5.94%`) | `9.91%` (`11/111`) | `10.65%` | `10.65%` |
| Arm A (`bv8-only`) | `0.0632` | Unthresholded (`tau = 0`) | `0.0000` | `154 / 10080` (`1.53%`) | `6` / `1` | `11.94%` (`9.57%`) | `13.51%` (`15/111`) | `15.48%` | `16.13%` |
| Arm A (`bv8-only`) | `0.0632` | At `<= 1.0%` ord flip cap | `0.0253` | `100 / 10080` (`0.99%`) | `4` / `0` | `10.65%` (`8.26%`) | `11.71%` (`13/111`) | `13.55%` | `13.87%` |
| Arm A (`bv8-only`) | `0.0632` | **Constrained (`<= 1%` flip & `<= 1` lethal)** | `0.0751` | `47 / 10080` (`0.47%`) | `1` / `0` | `7.10%` (`5.10%`) | `9.01%` (`10/111`) | `9.35%` | `9.68%` |
| Arm A (`bv8-only`) | `0.1129` | Unthresholded (`tau = 0`) | `0.0000` | `143 / 10080` (`1.42%`) | `5` / `5` | `11.94%` (`9.34%`) | `12.61%` (`14/111`) | `15.16%` | `15.48%` |
| Arm A (`bv8-only`) | `0.1129` | At `<= 1.0%` ord flip cap | `0.0500` | `100 / 10080` (`0.99%`) | `3` / `2` | `8.71%` (`6.35%`) | `8.11%` (`9/111`) | `11.29%` | `11.29%` |
| Arm A (`bv8-only`) | `0.1129` | **Constrained (`<= 1%` flip & `<= 1` lethal)** | `0.0802` | `76 / 10080` (`0.75%`) | `1` / `1` | `7.74%` (`5.42%`) | `8.11%` (`9/111`) | `10.32%` | `10.32%` |
| Arm B (`bv8+reach`) | `0.03` | Unthresholded (`tau = 0`) | `0.0000` | `175 / 10080` (`1.74%`) | `7` / `3` | `10.97%` (`8.71%`) | `12.61%` (`14/111`) | `17.10%` | `19.03%` |
| Arm B (`bv8+reach`) | `0.03` | At `<= 1.0%` ord flip cap | `0.0218` | `100 / 10080` (`0.99%`) | `5` / `3` | `8.71%` (`6.44%`) | `11.71%` (`13/111`) | `12.26%` | `12.90%` |
| Arm B (`bv8+reach`) | `0.03` | **Constrained (`<= 1%` flip & `<= 1` lethal)** | `0.0581` | `39 / 10080` (`0.39%`) | `1` / `1` | `6.13%` (`4.45%`) | `6.31%` (`7/111`) | `8.39%` | `8.39%` |
| Arm B (`bv8+reach`) | `0.0632` | Unthresholded (`tau = 0`) | `0.0000` | `183 / 10080` (`1.82%`) | `10` / `2` | `11.94%` (`9.48%`) | `13.51%` (`15/111`) | `17.74%` | `18.06%` |
| Arm B (`bv8+reach`) | `0.0632` | At `<= 1.0%` ord flip cap | `0.0397` | `100 / 10080` (`0.99%`) | `5` / `1` | `8.71%` (`6.23%`) | `10.81%` (`12/111`) | `12.26%` | `12.26%` |
| Arm B (`bv8+reach`) | `0.0632` | **Constrained (`<= 1%` flip & `<= 1` lethal)** | `0.0610` | `69 / 10080` (`0.68%`) | `1` / `0` | `8.39%` (`5.94%`) | `9.91%` (`11/111`) | `11.61%` | `11.61%` |
| Arm B (`bv8+reach`) | `0.1129` | Unthresholded (`tau = 0`) | `0.0000` | `136 / 10080` (`1.35%`) | `2` / `0` | `10.00%` (`7.34%`) | `9.91%` (`11/111`) | `13.23%` | `13.55%` |
| Arm B (`bv8+reach`) | `0.1129` | At `<= 1.0%` ord flip cap | `0.0352` | `100 / 10080` (`0.99%`) | `1` / `0` | `9.03%` (`6.50%`) | `9.91%` (`11/111`) | `11.61%` | `11.94%` |
| Arm B (`bv8+reach`) | `0.1129` | **Constrained (`<= 1%` flip & `<= 1` lethal)** | `0.0352` | `100 / 10080` (`0.99%`) | `1` / `0` | `9.03%` (`6.50%`) | `9.91%` (`11/111`) | `11.61%` | `11.94%` |

### 4.2 Secondary Head-Family Sensitivity Check (`FlatMLPResidualQHead` & `LinearResidualQHead`)

| Head Family | Arm | Constrained Rescue Retention (`<= 1%` flip & `<= 1` lethal) | Rescue Retention at `<= 1.0%` Ord Flip Cap (ignoring lethal cap) | Lethal Flips at `1.0%` Ord Flip Cap |
|---|---|---:|---:|---:|
| `FlatMLPResidualQHead` | Arm A (`bv8-only`) | `8.71%` (`L<600: 11.71%`) | `9.68%` (`any-flip: 12.26%`) | `2` cert / `1` raw |
| `FlatMLPResidualQHead` | Arm B (`bv8+reach`) | `10.97%` (`L<600: 10.81%`) | `12.26%` (`any-flip: 15.48%`) | `2` cert / `1` raw |
| `LinearResidualQHead` | Arm A (`bv8-only`) | `3.23%` (`L<600: 4.50%`) | `7.42%` (`any-flip: 10.00%`) | `8` cert / `4` raw |
| `LinearResidualQHead` | Arm B (`bv8+reach`) | `2.58%` (`L<600: 2.70%`) | `8.71%` (`any-flip: 12.26%`) | `7` cert / `4` raw |

---

## 5. Diagnostic Mechanism Analysis: Why Single-Frame Selectivity Fails

1. **Large Base Q-Deficits and Deep Action Rankings on Rescue Roots:**
   - Across the `310` certified rescue roots, `b8-s3`'s base Q-deficit `Q_b8(s, a_bare) - Q_b8(s, a_teacher)` has quantiles: `p10 = 0.0744`, `p25 = 0.1531`, **`p50 = 0.2427`**, **`p75 = 0.4893`**, **`p90 = 1.1006`**, `max = 6.2578`.
   - Base `b8-s3` Q-rank of the teacher's chosen escape action across the `310` certified rescue roots: Rank 2 = `151`, Rank 3 = `64`, Rank 4 = `49`, Rank 5 = `19`, Rank 6 = `27` (`49.4%` of rescue actions are ranked 3rd through 6th by `b8-s3`).
2. **184:1 Base-Rate Asymmetry Between Ordinary and Rescue Roots:**
   - On the 48 dev worlds, `open-stop-h32-nb` (`K=1`) evaluates `57,417` dense search roots (`L >= 1`, `interval = 5`) and overrides on only `350` roots (`0.61%`), of which `310` (`0.54%`) are certified rescues (`184:1` ratio of ordinary roots to rescue roots).
   - Because `open-stop-h32-nb` overrides only when `bare` fails all `K=1` 32-step coupled multi-agent rollouts (while `5` `frp3-s12+v8` opponents actively steer and coil), at `t=0` the `bare` action almost always still has open local and `k in {8, 16, 24}` static/opponent-eroded reachability.
   - Consequently, any single-forward residual head strong enough to overcome `0.24–1.10` base Q-gaps on rescue roots triggers false overrides on ordinary roots where `bare` is already safe, and `18%–35%` of those false ordinary flips are **lethal** (`n_diff >= 2/4` on suffixes `4..7`).
   - Enforcing the pre-registered selectivity constraints (`<= 1.0%` ordinary flips AND `<= 1` lethal ordinary flip across `10,080` roots) caps OOF rescue retention far below the `>= 35.0%` (`LB90 >= 25.0%`) bar on both Arm A and Arm B.

---

## 6. Verdict

- **`STOP`**: STOP the bare-network open-stop distillation line (close T34 under Claim A bare b8-s3 in Live Play + Claim B exact-stop-l2-prime replay bank in Watch).
- Per the pre-registered protocol (`CAMPAIGN_LOG.md` row `2026-11-04 PT 13:10`), no Colab Stage 2 training is launched (`0` Colab units spent).
