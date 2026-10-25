# R30 probe-03: certificate deadline; narrow amendment (c)

2026-10-25 · Astra independent T34 judge · operational/timing evidence only.

**Verdict: INCOMPLETE / CERTIFICATE_DEADLINE. Choose (c): one prospective,
bounded certificate-time amendment. Reject (a) as unproved and (b) as unsupported:
no specific implementation defect is established. G5/main and R30-2 remain HOLD.**

## Verified evidence

Runtime receipt: round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-03/runtime-output/receipt.json
SHA256: d8db81533fa079f931515c570b4d3bba833ae7ea83731826f731b2ce244c382c
Runtime pin e18cdbcae140f3b25d7439a5d78cd9fc85b404f0; inspected clean worktree
sol-r30-post-f4-20261025 at 3a830e711c1851c1258912408d520ce1faf248ce.
Receipt status INCOMPLETE, failure_class CHILD_NONZERO_EXIT, elapsed
1121.8882933750283 s, stop_verified=true. Stop is receipt-verified, not live rechecked.

All 36 contiguous telemetry records were independently checked against their SHA
sidecars, complete markers, and receipt export size/hash inventory. Final record
exports/telemetry/000035.json SHA256: 147eda2fdc449dd1546c2dbc8d5b7b9efe8fd45a273a8e19c246e383a50d8bb5
It records phase certificate, exception_type TimeoutError, diagnostic_code
CERTIFICATE_DEADLINE, phase elapsed 600.0158037599995 s; worker telemetry elapsed
744.3109331650003 s. This is the certificate's internal cap, not the 1500-second
worker cutoff or the 1800-second session limit. Certificate did not complete;
600.016 s is censored elapsed time, never an admissible completion estimate.

Completed durations (seconds):
- adam_0_10: 0.052443735
- adam_10_20: 0.047157890
- adam_20_30: 0.046088820
- adam_setup: 1.088833640
- b8_masked_action_decision: 0.097253431
- bookkeeping_probe: 0.890277735
- closure_0: 0.494325053
- closure_10: 0.514804015
- closure_30: 0.509012893
- equality: 7.219233589
- feature_cache: 102.576564410
- imports: 2.469010137
- input_checks_load: 14.389584899
- plan_coefficients: 12.655036501
- solver_auxiliary: 2.436702843
- unpack: 0.807790493

The three ten-update blocks completed all 30 scheduled Adam updates in 0.145690445 s.
Closures at updates 0/10/30 completed. Certificate, subsequent diagnostics,
bindings, fusion/serialization, main-export I/O and cleanup completion are absent.

## Arithmetic and decision

The frozen timing_forecast.py formula gives 2000 * max(Adam-block)/10 =
10.488747 s and 500 * max(closure) = 257.402007 s. These are planning
extrapolations, not execution of 2000 updates or 500 solves, and not certified
upper bounds. The separate solver-bookkeeping allowance is 300 s. Known fixed
work plus these terms and the censored certificate duration do not establish a
3180-second overrun. Missing completion costs forbid a full forecast or G5 ACCEPT.
The original 600-second certificate requirement is not met by this calibration;
that does not establish main-wide INFEASIBLE or optimizer/scientific failure.

Prior G5 calibration authority explicitly uses CPU Adam/solve/certificate and
CUDA features. Fast measured Adam/closures do not justify GPU Adam/solve or an
unmeasured A100 substitution. Decimal50 certificate semantics must be preserved;
a GPU floating-point substitute is not admitted as an equivalent certificate.

## Narrow prospective amendment and next action

Amend only the certificate wall cap from 600 to 1200 seconds for one new timing
calibration and prospective R30-1 main. This extends time for the same certificate;
it does not relax precision, enclosure, certificate target, objective, batch plan,
solver bounds, endpoint selection, scientific look or main allowance. The evidence
is timing-only; no certificate values or scientific outcomes were inspected.
The 1200-second cap is a finite diagnostic envelope, not a completion forecast.

Authorize preparation of exactly one L4 timing-only packet, preserving the
continuous 30-update schedule, closures 0/10/30, predetermined update-30 certificate,
full frozen inputs, single-thread CPU settings and redaction. Freeze certificate
base/effective 600/1200 s and session base/effective 20/40 minutes before execution;
worker cutoff 2100 s, cleanup/export reserve 300 s, and earlier remaining-session
cutoff still apply. Preserve transfer <=150 s and staging <=300 s. Add 40 minutes
of nontransferable calibration liability to the prior 502-minute envelope: 542.
No prior liability is reclaimed. R30-1/R30-2 remain 144/288 minutes; main estimate
ceiling 53 minutes and timeout ceiling 106 minutes are unchanged.

Sol prepares/tests the cap-only implementation, source bindings, fresh paths,
redaction and timeout tests; independent packet review must ACCEPT before operator
dispatch through the qualified supervisor. No automatic retry. This ruling is not
an executable packet ACCEPT. Preserve original records and failed attempts.
A completed calibration still requires all remaining phases, reconciled overhead,
complete inventory, and independent G4/G5 review. No incomplete forecast admission.

## R30-1 / R30-2 / F5 sequencing

Pursue F5 preparation now; completing R30-1 remains the preferred information step
before selecting F5's change. Bound that dependency to this single calibration:
if it completes and fits <=53 minutes, finish the gated R30-1 diagnosis first.
If it times out again or its complete forecast exceeds 53 minutes, close the
current R30-1 route as operationally unresolved/not admitted under this envelope,
not as a scientific KILL or a proof of universal L4 infeasibility. Do not fund an
open-ended calibration chain. A distinct F5 mechanism may then be proposed and
prospectively registered with explicit disclosure that optimizer adequacy remains
unknown; R30-1 is no indefinite veto on the owner's requested successor work.
R30-2's diagnosis/control and eligible ordinary-archive requirements remain binding;
F5 cannot silently inherit an unqualified R30-2 control or bypass those requirements.
F4 remains closed. No successor fit, gameplay, release or serving change is admitted.

Only operational telemetry, receipt/log, governing rulings and timing orchestration/
forecast source were inspected. No scientific units, D4096 or protected study
payloads were opened. No tests, commits, edits to existing files, dispatch, Colab
calls or tmux were performed. Only this create-only ruling and its sidecar were written.
