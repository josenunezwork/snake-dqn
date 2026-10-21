# E3 recovery-03: kernel headroom and bounded poll traffic

Status: **Mac SMOKE_COMPLETE; preparation only, NOT DISPATCHED**.
This sibling supersedes recovery-02 execution instructions only. No Colab calls.
The frozen protocol, roots, splits, students, optimizer recipe and decision rule
are unchanged. PROTOCOL.md SHA-256 remains
`4e84360fc1686ed4b9ad937f89fd1a72c9df8a8ef89c6c1e71f75283f9e8037d`.

Attempt02's saved receipt records INCOMPLETE after 1,445.425586 seconds,
a final 51.1388-second exec timeout, earlier 120-second exec failures,
`stop_verified=true` and no terminal result. Last observed reconstruction was
33/256 worlds. CPU starvation is plausible, not established by that evidence.
Both failed attempts and their manifests/receipts remain untouched.

`resources.py` counts Linux affinity CPUs (Mac fallback: cpu_count), reserves
two, caps requested workers, and refuses hosts with fewer than three usable
CPUs. Shell, parent and reconstruction workers use nice >=10. All native math
thread caps remain one. The manifest requires >=8 vCPUs for six workers; the
runtime independently refuses insufficient affinity for its priced worker count
before unpacking input. Smaller direct invocations cap6 to4 on six usable CPUs.
A small allocation can therefore be refused rather than run underpriced.

Per-world scratch payloads and sidecars now live in the work tree outside the
supervisor export inventory. The final dataset and timing report retain every
array and world metric. This avoids hundreds of exports and repeated download /
lease-renewal RPCs. Parent logs retain first/final records plus updates at most
once per120 seconds; individual worker messages are limited to once per300
seconds. Ordered task merging and all target computations remain unchanged.

Existing Z-9 qualification is retained: <=1 MiB local catalog, size/hash-bound
public checkpoint download, and a create-only upload-preflight receipt over the
exact manifest and pinned source fallback before job.json is published. Full
input remains256 worlds/1536 roots,192/32/32 worlds,768 roots per class.
Catalog360,881 bytes SHA-256:
`54eaa6a8f62fd427252d6468df720123e5c47a7c481b55e838c358be58a68fbc`.
Checkpoint archive51,787,546 bytes SHA-256:
`dfdef34f99aa73a21f5b9b3a23a7fd27ab4c8a4c388e5a73bfc4b5b13b520c37`.

Real Mac job.sh smoke, same compact smoke input as preflight02, one math thread
per process, serial plus two-worker byte parity, two-epoch smoke fits:

| Measurement | Seconds |
| --- | ---: |
| Three worlds / six roots, serial reconstruction |35.008672375 |
| Two-worker reconstruction parity |22.915699792 |
| Fits / batch benchmarks |1.934318833 |
| Complete entrypoint |60.726798833 |

Parent and observed reconstruction workers nice15 (inherited parent nice5 plus
launcher nice10). All13 final dataset arrays match preflight02 by dtype, shape
and exact bytes. First/final progress records are1/3 and3/3. The independent
supervisor inventory sees8 exports before the local verification receipt is
added, with no world scratch exports. Evidence:
`big-lever/exp/bl-e3/preflight-03/{timing,terminal,recovery-verification}.json`
and `preflight-03.log`; scratch is `preflight-work-03/worlds/`.
AC attached, lid open, no thermal warning; git-sync and phase0-selection tmux
sessions observed only. Timing is from a contended Mac, not serving latency.

Forecast: six workers on >=8 usable CPUs,2.585439584 hours /5.170879167 units
at2 units/hour, including20 minutes setup/export. Reconstruction2.172747832h;
CPU fit upper projection0.079358418h; no CUDA speedup credited. Retain the4x
slower-host assumption; no recovery03 L4 throughput/responsiveness measured.
Estimate155.126375min, timeout311min (>=2x), lease300sec, export cap256 MiB.
Attempt01 reserve0.2u plus attempt02 ceiling charge0.803014215u is rounded up to
**1.1u prior cost**, leaving**11.7u session cap +3.2u unchanged reserve** inside
16u. Timeout at2u/hour costs10.3667u; first wall/unit cap wins.

Targeted baseline checks:46 passed/64 deselected across the package plus
supervisor validation/public-input tests. Follow-up transport coverage adds
priced-worker affinity refusal and progress/scratch export regression tests.
Black100, isort, flake8, Bash syntax and diff whitespace checks pass.
Independent read-only review found no remaining actionable code issue; the
Mac preflight is not proof of Colab kernel responsiveness.

Final published source pin, manifest validation/readiness hashes and absolute
commands are in `big-lever/exp/bl-e3/recovery-03/DISPATCH.txt`.
The planned output is fresh `big-lever/exp/bl-e3/colab-output-03`.
Account inventory remains an empty placeholder: refresh reviewed assignments
and regenerate/requalify a new immutable manifest before future dispatch if
needed. No automatic retry or allocation. No serving authority.
