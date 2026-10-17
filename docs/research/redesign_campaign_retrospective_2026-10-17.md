# Redesign campaign retrospective — 2026-10-17

## Outcome and standing boundary

The redesign goal was to **beat frp3-s12 + v8 without a veto**. It was met:
**B8 seed 3** was released on 2026-10-17 after the T7 fresh-bank Phase R
`STRICT_PASS` under Tier 1, Tier 2 PASS, and T12 serving qualification,
including the T12B check-4 repair. The served release claim is **more mass at
equal survival against the served agent, with no veto**; it is not a safety
claim. See the [release record](b8_s3_release_2026-10-17.md), the [research
index's 2026-10-17 B8 Phase R under the Tier 1 guard entry](INDEX.md#later-results-recorded-outside-docsresearch),
and its [2026-10-17 T12C B8 seed 3 serving release entry](INDEX.md#release-update--2026-10-17).

The bare-network late-death line is closed. It may be reopened only with new
evidence of a different mechanism or a new observation/architecture. This
retrospective preserves the closure evidence; it does not amend any frozen
protocol, result, or receipt.

## Closed late-death arms

The following one-line outcomes reproduce the 2026-10-14 M6 through 2026-10-17
T8/T9/T10/T11 diagnostics and LOCAL30 entries in the [research index](INDEX.md#later-results-recorded-outside-docsresearch).
These were development screens or diagnostics, not promotion evidence.

| Arm | Recorded result |
| --- | --- |
| M6 reward levers | [NULL / not adopted](m6_result_2026-10-14.md): tightness cost and length-scaled death did not close the scripted-survival gap. |
| Huber T2 | [INCOMPLETE / no adoption](t2_screen_status_2026-10-15.md): the available Huber survival gain carried a mass tax, ANCHOR0 exhausted its extension, and HUBER_ANCHOR0 was KILL. |
| Huber T2b | [Both delta-10 and delta-30 extensions exhausted](t2b_delta_sweep_2026-10-15.md): the survival/mass trade-off did not meet adoption criteria. |
| T3 rollout labels | [No GO](t3_screen_result_2026-10-15.md): labels were learned, but the screen did not improve the primary survival endpoint and reduced mass. |
| T4 TWO_SIDED | [KILL](t4a_screen_result_2026-10-16.md): two-sided bank labels left scripted episode survival below CTRL. |
| T4 TWO_SIDED_T | [KILL after extension](t4c_screen_result_2026-10-16.md): the tighter labels fixed spillover but did not buy survival. |
| T4 RETURN | [Halted](t4e_bugcheck_2026-10-16.md): boost rate reached 34%; the retained check found no collector/trainer implementation bug. |
| T4 RETURN_C | [KILL](specs/t4f_prereg_addendum2_2026-10-16.md): the contrast arm reached its preregistered boost tripwire. |
| T5 input features | [FAIL](t5_separability_2026-10-16.md): the 24 added per-action features did not meet the death-AUC separability thresholds. |
| T6 RESTART | [KILL](t6_screen_result_2026-10-17.md): late-game death-window exposure did not change the scripted episode-survival endpoint or the held-out decisive Q-gap. |
| T11 LOCAL30 | [KILL before gameplay evaluation](t11_screen_result_2026-10-17.md): seed 13 exceeded the low-length boost tripwire, leaving no five-seed final roster. |

## T8–T10 mechanism record

The diagnostics describe a failure mode that is not separated early by the
tested inputs. The [T9 divergence record](t9_divergence_2026-10-17.md) found
that the fatal path builds a self-bounded pocket over roughly 50–100 decisions;
there is no clean, consistent early separator among the requested quantities.

The [T10 audit](t10_certificate_horizon_2026-10-17.md) found no certificate:
lookahead held the measured tail-reachability flag about as often as B8. The
[research index's 2026-10-17 T8 / T9 / T10 / T11 diagnostics and LOCAL30
entry](INDEX.md#later-results-recorded-outside-docsresearch) summarizes horizon
AUC as about 0.92 at 30 decisions and about 0.75 at 300 for every input. The
detailed T10 record reports input-dependent decay rather than a universal
ceiling. The [T8 probes](t8_probes_2026-10-17.md) supply the associated
raw-raster and arrival-race diagnostic results. That index entry records the
T8–T11 findings together: B8 is near the bare-network ceiling for this
late-game question, while the release result remains governed by its own strict
and serving evidence.
