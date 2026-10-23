# F0 whole-session caps — mechanical amendment, 2026-10-23

Authority: the owner's explicit 2026-10-23 request to resize every F0 stage
for setup + runtime + export + margin, applying the generous 2x rule once.
This supersedes only the execution envelopes in INTENT.md and
FAMILY_PROBE_EXECUTION.md for the fresh sibling pack. Those frozen files,
pack-04, its log and refusal receipt remain unchanged. No scientific result
was used to size this repair; N, horizons, audit rates, labels, decision rules,
CPU placement and maximum two manual completion attempts remain unchanged.

Failure: `bl-f0-family-probe-f9bbae8c` was refused at the whole-session wall
cap. Its 270-second independent runtime budget was incompatible with a
300-second session, including setup and export. The preserved log still
shows setup at 177.8 seconds (public input installation), followed by factory
startup at 195.6 seconds. We use 180 seconds as a rounded setup planning
observation, not an exact setup duration or a useful-work throughput estimate.
The 270/3150-second work allowances below are retained planning allowances,
not new timing measurements. Export and margin are explicit planning reserves.

| Stage | Setup s | Work s | Export s | Margin s | Base minutes | Effective minutes | Base units | Effective units | Base reserve u | Effective reserve u |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Excluded family probe | 180 | 270 | 120 | 30 | 10 | 20 | 0.02 | 0.04 | 0.004 | 0.008 |
| Full CPU canary | 180 | 3150 | 120 | 150 | 60 | 120 | 0.10 | 0.20 | 0.020 | 0.040 |
| Series | | | | | **70** | **140** | **0.12** | **0.24** | **0.024** | **0.048** |

Arithmetic: probe (180+270+120+30)/60 = 10 base minutes; science
(180+3150+120+150)/60 = 60 base minutes. Multiply complete base caps,
unit caps and reserves by two exactly once. At the measured 0.08 u/hour,
20 minutes costs 0.026667 u; plus 0.008 reserve fits 0.04 u. Science's
120 minutes costs 0.16 u; plus 0.040 reserve fits 0.20 u. Total priced wall
cost 0.186667 u + 0.048 reserve fits 0.24 u. Do not double measured rates,
work counts, estimates, or retry maximum. The earlier failed probe remains
charged to the campaign history; it is not erased or relabelled successful.

The new stage/series envelope applies to this newly pinned repair series;
this is the owner's explicit replacement envelope, not a mid-attempt edit
or an automatic budget reset. Original plus at most two manual mechanical
completion attempts must share the new stage and series limits. Operator
must subtract elapsed/charged prior repair attempts before preparing any
completion manifest. No automatic retry or extension is introduced.

The manifest now supplies the whole-session cap and effective finalization
reserve: 300 s probe, 540 s science. At actual runtime entry, read the
qualified supervisor's `spec.json` and generation-matching `state.json`.
The spec timeout already deducts pre-watchdog setup and supervisor guards.
Collection allowance = spec timeout - measured elapsed since watchdog start
- finalization reserve. Thus uv/Python installation, dependencies, uploads
and public-input setup consume the same budget as collection. Slower setup
reduces collection time; exhausted setup refuses collection. There is no
independent `F0_RUNTIME_SECONDS` default. Export `session-budget.json` with
all operands. The watchdog and supervisor remain the final hard bounds.

No qualified transport implementation changes are made: tree
`d0d92bdb572b770284a1fea6152417ffa5f06bea` remains mandatory. A clean published
origin/main source, fresh Python 3.12 preflight, supervisor validate, and an
actual same-pin family PASS receipt are required. Prepared science templates
remain unadmitted until binding that receipt. Mac timing and replay audit
remain separate; no Colab allocation is performed by this repair.
