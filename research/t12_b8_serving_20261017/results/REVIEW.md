# Independent evidence review — T12, 2026-10-17

The read-only `harness_review` reviewer independently confirmed **SERVING_FAIL / HOLD**:
checks 1–3 and 5–8 pass; check 4 fails. No simulation, timing retry, test, source edit,
process action, or tmux mutation was performed during this final evidence review.

The reviewer checked all 61 receipt-listed file hashes, all 1,024 frozen source hashes,
the raw P2 record comparisons, both 5,000-frame digest chains, episode termination and
AI-only dispatch, rollback identities/logs, confidence vectors, and browser sidecars.

Watch12 has exactly two paced attempts, each with one overrun in 120 ticks:

| Request | Budget | Maximum full tick | Result |
|---|---:|---:|---|
| 24 fps | 41.666667 ms | 55.911709 ms | Initial overrun |
| 17 fps | 58.823529 ms | 60.943292 ms | Final permitted retry fails |

The latter overrun is sample index 49, 2.119763 ms (3.6036%) over budget. No third
attempt occurred. All four update-p95 values are below 66.667 ms. The full-tick data
does not establish whether the outlier came from update work, snapshot work, scheduling,
or another cause. It does not support attributing the failure to the observed OS load.

Interpret the original audit's wording, `paced retry count exceeds the one permitted
retry`, as **the final permitted retry still overruns**. Its branch uses inaccurate
wording for this case; its FAIL verdict is correct and its bytes are preserved.

The raw per-case `estimated_achieved_fps` field remains based on the initial request
after a pacing retry. Watch12's 17.969350 uses 24 fps; at the final 17 fps the same
formula gives 13.734909. Play12's 20.677077 uses 26 fps; at the final 15 fps it gives
13.060309. The report uses each final attempt's measured cadence instead. These summary
defects do not alter the raw timings or FAIL. `shared_ui_speed_fps: 15` is not a
validated global ceiling because Watch12 failed. Frozen evidence was not rewritten.

The independently aggregated Watch baseline is 53 deaths / 125,000 decisions =
0.424 deaths per 1,000. The logical-length histogram has 125,000 samples, mean
208.219784, median 172, p95 528, minimum 1 and maximum 867. Final episode lengths
have mean 209.84, median 123, minimum 4 and maximum 868. The monitoring numbers are
descriptive; they do not override the HOLD.

The reviewer recommends preserving this run and investigating the long-tick cause
under a separately registered diagnostic plan before any serving fix or new qualification.
No further timing attempt is justified by this result alone. The released default
remains `NAME_FRP3_S12`.
