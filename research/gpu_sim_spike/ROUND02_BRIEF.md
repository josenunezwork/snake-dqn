# E1b Round 02 implementation brief

Prospective implementation of the owner's Round 02 E1 rebrief, before E1b
preflight or device data. Authority: external `big-lever/round-02/JUDGMENT.md`,
section (f), E1. Development evidence only; no registered verdict or release.
The frozen `KILL_RULE.md` and `BASELINE.json` bytes are unchanged.

The three contracts are separate:

1. Uncapped ego connectivity uses tiled local connected components, boundary
   unions, pointer compression and integer component-size reduction on CUDA.
   Convergence is device terminated and an exhausted watchdog is a visible error.
2. Count-capped v2/v5/v7 searches use a CUDA block per world/slot/action with
   sparse frontiers, atomic deduplication and device termination at cap or empty
   frontier. A release-rejected cell stays unseen, and queue storage covers
   discovered cells rather than just consumed cells.
3. Reach-time retains its existing 24 synchronous rounds and tail-wait semantics.

CPU implementations remain independent oracles. Mac fallback/shape coverage
cannot establish CUDA correctness or a speed improvement. CUDA parity is required
before interpreting CUDA performance; failures are preserved, without automatic
retry. CPU-defined decisive-root reporting remains part of the dense audit.

REALISTIC accounting uses concurrent resident batches of 256, 1,024 and 4,096
worlds, long bodies spanning 800–1,500 cells, all three opponent mixes, one real
B8 hero and five real opponents per world, simulator events/resets and collection
writes. Report three warmed repeats, full world-steps/s and p95 full tick time.
The 20,000/s L4 floor must be assessed at production shape. Serial-root dense
rates remain descriptive and cannot satisfy that floor. Missing shapes, failed
kernels, OOM, parity failure or timeout cannot qualify CONTINUE. The original
cost <=0.1x and agreement >=99%, zero unexplained disagreements, are unchanged.

Prepare one L4 spike within 15 units including reserve and an optional separate
L4 probe within 5 units including reserve; combined envelope <=20 units. First
wall/unit cap wins. The probe is diagnostic and cannot qualify D300 cost or
agreement. This task makes no Colab/provider calls and dispatches neither job.
Every exported payload has create-only `.sha256` and `.complete` sidecars.
