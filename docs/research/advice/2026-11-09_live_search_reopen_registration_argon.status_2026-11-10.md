# Status & Partial Supersession Note (2026-11-10 PT) for `2026-11-09_live_search_reopen_registration_argon.md`

- **Target registration (unedited, SHA-256-bound):** [`2026-11-09_live_search_reopen_registration_argon.md`](2026-11-09_live_search_reopen_registration_argon.md) (`Registration ID: t34-live-l1os-reopen-20261109-v2`, SHA-256 `76aece10bc4a334aeede962841ba549264b897c38531dbe1342e2de96ce90387`, pinned in `research/gpu_sitl/prepare.py:35` and `tests/test_gpu_sitl_b1b.py:1847`).
- **Authority:** Owner ruling of `2026-11-10 ~19:20 PT`, recorded in [`AGENTS.md`](../../../AGENTS.md), [`docs/STATUS.md`](../../STATUS.md), [`docs/REGISTERED_ITEMS.md`](../../REGISTERED_ITEMS.md), and [`CAMPAIGN_LOG.md`](../final_run_20261019/CAMPAIGN_LOG.md) (`2026-11-10 PT 19:25`).
- **Why this sibling file exists:** `2026-11-09_live_search_reopen_registration_argon.md` is SHA-256-bound by `research/gpu_sitl/prepare.py` and test guards and must never be edited in place. This status note records the exact clauses superseded by **Items 2 and 3** of the `2026-11-10` owner ruling (plus goal/track context from **Items 1, 4–6**).

---

## 1. Clauses Superseded by Owner Ruling Item 2 (Exploration Lane, Any GPU incl. `L4`, Runaway-Guard Caps)

**Ruling Item 2 (`2026-11-10`):** Step B1b smoke and pilot (`research/gpu_sitl`) are **exploration lane**: any GPU type may be used (`L4` included), and wall-time and unit caps are runaway guards the operator may raise and log with **no judge amendment needed**. Bars, held-out namespaces, and seeds are unchanged.

- **Change Log (`L3`, item `5/C/D`):**
  - *Superseded clause:* ``bound Step B1b's A100 cap to throughput measured in a `<= 3.0 A100 u` smoke as a runaway guard only``
  - *Superseding rule:* `Step B1b-Smoke` and `Step B1b` pilot are exploration lane on **any GPU (`L4` included)**; wall-time and unit caps may be raised and logged by the operator without a judge amendment.
- **§3.2 (`L122`, item `1`):**
  - *Superseded clause:* ``1. `t34-gpu-sitl-preflight-20261109-v1` (`E = 4` worlds, $H = 32$): Step B1a Mac preflight and Step B1b-Smoke A100 throughput ruler.``
  - *Superseding rule:* Only the `A100`-only hardware restriction on `Step B1b-Smoke` is superseded (`L4` or any GPU allowed); namespace `t34-gpu-sitl-preflight-20261109-v1`, `E = 4`, and $H = 32$ are unchanged.
- **§3.3 (`L139`, `Step B1b` header):**
  - *Superseded clause:* ``- **Step B1b — Target-GPU Throughput Smoke (`<= 3.0 A100 u`) + Single-A100 Online SITL Pilot (`B1-A100-Pilot`):**``
  - *Superseding rule:* Neither smoke nor pilot is restricted to `A100` or a judge-locked `<= 3.0 A100 u` cap; both run in the exploration lane on any GPU (`L4` included).
- **§3.3 (`L140`, `Step B1b-Smoke` bullet):**
  - *Superseded clause:* ``- **Runaway-Guard Calibration Smoke (`Step B1b-Smoke`, `<= 3.0 A100 u` cap = `1.5 A100 u` base / `3.0 A100 u` 2× under `short-canary <= 25 min`):**``
  - *Superseding rule:* Any GPU (`L4` included) may be used; wall-time and unit caps are runaway guards the operator may raise and log with no judge amendment.
- **§4.1 (`L190`, `Step B1b-Smoke` row):**
  - *Superseded clause:* ``| **Step B1b-Smoke:** Target-GPU Throughput Smoke (`<= 25 min` `short-canary`) | `1 A100` | `0.22 h` | `0.42 h` (`25 min`) | `1.5 u` (`A100`) | **`3.0 u` (`A100`)** |``
  - *Superseding rule:* Host/tier may be any GPU (`L4` included); wall-time and unit caps are exploration-lane runaway guards raised/logged by the operator (no judge amendment).
- **§4.1 (`L191`, `Step B1b` row):**
  - *Superseded clause:* ``| **Step B1b:** `B1-A100-Pilot` (`>= 5,000` mined rescues, `1,000` updates) | `1 A100` | *Set from B1b-Smoke* (`~2.2 h` prior) | *2× B1b-Smoke* (`~4.4 h` prior) |``
  - *Superseding rule:* Host/tier may be any GPU (`L4` included); wall-time and unit caps are exploration-lane runaway guards raised/logged by the operator (`>= 5,000` rescues, `1,000` updates, and `> ~1 h` checkpoint/resume stay intact).
- **§5 (`L212`, item `3`):**
  - *Superseded clause:* ``measured in `Step B1b-Smoke` (`<= 3.0 A100 u`) in §3.3 to set `Step B1b`'s runaway guard.``
  - *Superseding rule:* `Step B1b-Smoke` may run on any GPU (`L4` included), and both smoke and pilot caps are operator-adjustable runaway guards requiring no judge amendment.

---

## 2. Clauses Superseded by Owner Ruling Item 3 (Recipe-Scoped Kill Rules Only)

**Ruling Item 3 (`2026-11-10`):** Kill rules close **only the recipe that was tested**. Clauses such as `"permanently closes all learned-student work on ego2s-bv8 inputs"`, `"KILL_BRANCH_B ... 0 further Colab units and no sweeps"`, and whole-line `STOP_AND_MAINTAIN` triggers now mean: **that recipe stops; a different recipe may be tried after an advisor design note**. The served agents stay as they are until a release.

- **Change Log (`L3`, item `B`):**
  - *Superseded clause:* ``stopping **every** learned student on `ego2s-bv8` inputs``
  - *Superseding rule:* Stops only the tested `Step B1b` recipe (`unfrozen Ego2sNet` online `open-stop-h32` pure-hinge SITL), not all `ego2s-bv8` students; item `1/A`'s exclusive routing to Branch `(b)` likewise does not bar new advisor-designed recipes.
- **§1.3 (`L70`, Branch `(b)` row):**
  - *Superseded clause:* ``Live search remains permanently closed; execute **Branch (b)** (§3, Gate B1) exclusively.``
  - *Superseding rule:* Only the tested Branch `(a)` live-search recipe stays closed on timing; other recipes (after an advisor design note) and the new from-scratch GPU track (Item 5) are permitted.
- **§2 (`L93`, status callout):**
  - *Superseded clause:* ``**Branch (a) is permanently closed without running Stages A1–A5**``
  - *Superseding rule:* Closes only the tested Branch `(a)` recipe; a different recipe may be tried after an advisor design note.
- **§3.1 (`L118`, item `3` — scope bullet):**
  - *Superseded clause:* ``**Scope of `"inputs insufficient"` Stop:** This stop applies to **every learned student on `ego2s-bv8` inputs**, not just this pilot.``
  - *Superseding rule:* Whole-family `ego2s-bv8` ban (including the next sentence on `L118`) is superseded: `STOP_EGO2S_BV8_INPUTS_INSUFFICIENT` stops only this tested pilot recipe; a different recipe may follow an advisor design note.
- **§3.4 (`L169`, `Observational Aliasing Kill`):**
  - *Superseded clause:* ``**KILL Branch (b) and close all learned-student work on `ego2s-bv8` inputs (`"inputs insufficient"`).**``
  - *Superseding rule:* Diagnostic threshold (`< 80.0%` rescue agreement at `<= 2.0%` ordinary flips) is unchanged, but triggering it stops only this tested recipe, not all `ego2s-bv8` student work.
- **§3.4 (`L170`, `General Gate B1 Kill`):**
  - *Superseded clause:* ``If Gate B1 fails **any** of the 4 bars above on its single look $\Rightarrow$ **KILL Branch (b) immediately (`0` further Colab units; no model-scale or hyperparameter sweeps).**``
  - *Superseding rule:* Failing any of Gate B1 Bars 1–4 stops this tested recipe; a different recipe may be tried on Colab after an advisor design note.
- **§4.1 (`L189`, `L192`, `L193`, disposition cells):**
  - *Superseded clause:* ``any of Bars 1–4 fail $\Rightarrow$ **KILL (b) / STOP LINE**.``
  - *Superseding rule:* Every `STOP LINE` / `KILL (b) / STOP LINE` cell in `§4.1` (`L189`, `L192`, `L193`) stops only the tested recipe, not the entire research line.
- **§4.2 (`L196`, whole-line stop lead-in):**
  - *Superseded clause:* ``the entire live-hero research line **stops permanently (`STOP_AND_MAINTAIN`)** on the earliest of the following **scientific** triggers``
  - *Superseding rule:* Triggers `1–4` (`L197–L200`) stop only this tested recipe; they do not permanently close the live-hero research line.
- **§4.2 (`L198`, trigger `2`):**
  - *Superseded clause:* ``**`STOP_EGO2S_BV8_INPUTS_INSUFFICIENT`**, which permanently closes **every learned student on `ego2s-bv8` inputs** (not only this pilot).``
  - *Superseding rule:* Stops only this tested pilot recipe; a different recipe may be tried after an advisor design note.
- **§4.2 (`L202`, post-stop funding clause):**
  - *Superseded clause:* ``and no further live-search or `ego2s-bv8` bare-student training runs are funded.``
  - *Superseding rule:* Post-stop funding ban is superseded (new recipes may be funded after an advisor design note). The first clause of `L202` keeping **Watch** (`exact-stop-l1-os` bank) and **Play** (bare `b8-s3`) served until a release **remains in force**.

---

## 3. Related Context (`Items 1, 4, 5, 6`) & Unchanged Bindings

- **Goal & release-only Mac bars (`Header L11`, `§0.2 L37–L38` — Ruling Items 1 & 5):** Project goal is to train the strongest possible slither.io-style agent. Mac serving bars (`24 fps`, `GameState.update p95 <= 11.688 ms`, Mac strict gate, serving qualification) gate **release** only and do not limit what is trained; a large teacher distilled to a Mac-size student is allowed. A new main track — large-scale from-scratch training on the exact GPU simulator using all Colab slots — is open (*design pending, advisor round 2026-11-10*).
- **Scarce GPU stock polling (Ruling Item 4):** Scarce GPU stock is polled about once a minute (`~1/min`), not every `20` minutes.
- **Unchanged by this ruling (Ruling Items 2 & 6):**
  - All `Reg-20261109-v2` pass bars, tripwires, and thresholds for this recipe (`Step B1a`, `Step B1a-ext`, `Probe32` tripwires, `< 80.0%` / `<= 2.0%` diagnostic threshold, `Gate B1` Bars 1–4, and `Stage B2` confirmation bars).
  - All held-out, sealed, design-contaminated, and seed-audited namespaces and seeds (`§3.2`) — held-out/test data are never used for training or tuning.
  - Honest reporting, verified stops and cleanup, `L4` shake-out before new GPU code runs on registered hardware, checkpoint/resume for runs `> ~1 h`, and confirmation-lane evidence + explicit owner OK for any release.
