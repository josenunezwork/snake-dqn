# A-3-03 host admission runtime correction

The first dispatch dry-run allocated nothing and exceeded its fixed 60-second
validation subprocess bound. Running the exact broker validator directly passed
all gates in 63.79 seconds. The host adapter repeats the shared and resident
checks; its timeout and the qualified transport remain unchanged.

The duplicate tiny real-aot collection test and real-cache lifetime test now
share one real E1 assay. The combined test preserves TF32=True warmup, all five
compiled blocks, populated Adam, three measured cycles with no compiler work
or added validation, E2 cache reuse, and first-assay/simulator/optimizer weakref
collection. Production cap1024 four-bucket parity/purity and distinct TF32 mode
checks remain separate. This removes repeated graph lowering without skipping
assertion categories. Final full suite: 78 passed in 10.73 seconds (nice15,
all numerical/compiler threads1). Earlier 79-test evidence is preserved.

Registry admission still generates and checks all 22,457,142 seed values. It
now searches every normalized uint64 excluded value in the already sorted
unique array, with explicit bounds before indexing. This avoids a second large
sort inside numpy.isin. Zero, alias/duplicate and excluded-seed checks remain
exact. Tests retain 64-bit neighbor/alias cases and add both out-of-range ends.

Only host preparation/tests changed after the runtime seal. The 101-file shipped
closure still exactly matches published runtime pin
d12bf047c6c954c657d15e104e28a2a6ab7d5c32; neither packet's bytes changed.
No A100 allocation or execution occurred. Final source-bound preflight/dry-run
evidence is recorded in VERIFICATION_A3_RESEAL03.json.
