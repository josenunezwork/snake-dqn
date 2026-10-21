# GPU sim spike result — operator template

Excluded development evidence. No serving or registered verdict authority.
Baseline and kill rule were frozen at `179d6044` before device code execution.

Status: **PENDING L4/A100 OPERATOR RUNS — NOT DISPATCHED BY SOL**.
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
| A100 candidate | six S0 cases, D300 | | measured provider rate | | |

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

## Port boundary and decision

Fill the actual device and host stage inventories and measured host seconds.
If any rollout component remains on the host, identify the device-portion speed
as an **upper bound**, and do not report it as end-to-end GPU success. Missing
provider receipts, incomplete fixtures, or partial runs cannot support CONTINUE.

Decision: [CONTINUE / STOP / INCOMPLETE pending first operator runs].
Reason: [both prospective bars, with receipt paths, source/model hashes, wall/unit
caps, and terminal supervisor stop/disappearance evidence]. No automatic retry.
