# GPU sim spike result — operator template

Excluded development evidence. No serving or registered verdict authority.
Baseline and kill rule were frozen at `179d6044` before device code execution.

Status: **E1 PENDING L4 OPERATOR RUN — NOT DISPATCHED BY SOL**.
E1 rule additions frozen before E1 data at `896310e3` (KILL_RULE.md).
The optional 5 u D8 probe cannot qualify the full D300 rule.
The Mac preflight is a tiny syntax/shape/runtime check, not a GPU cost result.

## Frozen rule (verbatim)

CONTINUE (recommend a full port) only if **both**:
- end-to-end GPU cost per dense-search decision ≤ **1/10** of the L4 CPU baseline on the long-body mixes, **and**
- first-action agreement ≥ **99 %** of searched decisions on the S0 fixtures, with 0 unexplained (bug) disagreements.
Otherwise STOP: write `RESULT.md` with the numbers and the reason; no second attempt without a new brief.

## Fill from immutable receipts

| Platform | Sample / shape | End-to-end seconds / decision | Actual u/h | Units / decision | Host seconds |
| --- | --- | --- | --- | --- | --- |
| Mac CPU reference | tiny preflight; see PREFLIGHT.md | | local / no Colab debit | | |
| L4 CPU reference | 12 processes, S0, D300 | | | | |
| L4 candidate | six S0 cases, D300 | | | | |
| Optional L4 probe | six S0 cases, D8; diagnostic only | | measured provider rate | | |

Include account/provider rate, bootstrap, allocation, host construction/upload,
Python dispatch, CPU label rescoring, readback, export and shutdown. Do not add
process CPU launch time to rollout wall time: they overlap. Report per long mix.
`BASELINE.json` marks extrapolated S0 estimates separately from matched CPU12 data.
The frozen approximate conversion is 0.0006 u/dense decision (0.036 u/world / 60
eligible dense batches), hence a prospective ceiling of 0.00006 u/decision.
Neither linear 12-process scaling nor six-branch row-step extrapolation is a
measurement. Preserve them as projections when presenting the measured comparison.

| S0 case | Decisions | First-action agreement | Branch mass MAE | Float tie | RNG-stream difference | Bug / unexplained |
| --- | --- | --- | --- | --- | --- | --- |
| short-scripted | | | | | | |
| short-frozen | | | | | | |
| short-mixed | | | | | | |
| long-scripted (hero 900) | | | | | | |
| long-frozen (hero 900) | | | | | | |
| long-mixed (hero 900) | | | | | | |

List every disagreement with decision, branch values, Q margins, first divergent
state/event, classification and evidence. Counter RNG alone does not establish
an RNG explanation: record matched-event replay evidence. Unknowns are bugs.
Every generated label must retain its independent CPU rescoring result.

## E1 full-world throughput and decisive audit

**<20,000 full world-steps/s on L4 => STOP engine thesis this week.** Apply to
each long-body mix. Record synchronized rollout and pipeline rates from the
benchmark, six branch worlds × depth, all six snakes and real opponents.
Disclose scheduled dead/illegal lanes. Warmup excluded only from rollout rate;
bootstrap/export/allocation costs come from supervisor receipts.

| Long mix | Full world-steps | Rollout seconds | Full world-steps/s | Pipeline rate | Floor status |
| --- | --- | --- | --- | --- | --- |
| scripted | | | | | |
| frozen | | | | | |
| mixed | | | | | |

| CPU-defined stratum | Roots | Agreement | Branch mass MAE | Tie | RNG | Unexplained |
| --- | --- | --- | --- | --- | --- | --- |
| Decisive (best survival > bare-greedy survival) | | | | | | |
| Other | | | | | | |

Zero decisive roots is NO_COVERAGE. Include every mismatch's root and evidence.
No threshold replaces the original ≥99% overall rule; audit does not confer
registered verdict or release authority. `BASELINE.json` retains historical
A100 wording; E1's owner brief authorizes an L4 spike + optional L4 probe only.

## Port boundary and decision

Fill the actual device and host stage inventories and measured host seconds.
If any rollout component remains on the host, identify the device-portion speed
as an **upper bound**, and do not report it as end-to-end GPU success. Missing
provider receipts, incomplete fixtures, or partial runs cannot support CONTINUE.

Decision: [CONTINUE / STOP / INCOMPLETE pending first operator runs].
Reason: [both prospective bars, with receipt paths, source/model hashes, wall/unit
caps, and terminal supervisor stop/disappearance evidence]. No automatic retry.
