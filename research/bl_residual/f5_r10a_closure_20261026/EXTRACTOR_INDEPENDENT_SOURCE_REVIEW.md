# Independent source review of the F5 status extractor

Reviewer: `/root/extractor_design_review`, read-only reviewer role, separate from
the implementer. Disposition: **no remaining P1/P2 implementation findings**.
This is review evidence only. It issues no independent acceptance receipt,
scientific adoption or permission to scan real input. The required STOP remains.

| Artifact | SHA-256 |
|---|---|
| Extractor source | `cff47212a96be17a1d1636085f5f8d97bc8efe17b702dd346b31238c62fb4bb9` |
| Extractor tests | `297bc46a82925667bb23ce911f5f6c9f2aaac9f835b964bb678606098544f6c6` |
| Fixed access contract | `64b9489679503f9a952e12da67e34e05f337bb791536981dbc661d21081a8104` |
| Frozen 969-life inventory | `ed30150f323fe38f830da2233319ea4f17da0a26a7547a159e75701143f0a59e` |
| Review packet | `8461e16a3e549001f21891ea668587137bb5ce015893c113418c81440003b2c9` |

The reviewer inspected the complete source and execution paths, tests, review
pins, frozen metadata and corrected packet, and directly verified these hashes.
Bounded ZIP/NPY headers precede all value decoding. Only eight metadata members
open; nine observation/outcome members remain unopened. Primitive dtype, shape,
byte-count, lineage, exclusions, original order, per-life limits and complete
life coverage checks are present. Full metadata and U have separate hashes and
count reconciliation. The served all-false mask fallback is preserved.

The operator API checks exact independent-acceptance pins before the fixed real
source can open. No-follow directory descriptors, exact source size/hash on the
same descriptor and before/after identity checks protect that later access.
The CLI refuses all inputs. U=0 and U>46,080 have the registered consequences.

The reviewer inspected 34 extractor tests covering forbidden-value and nonexact
subtype invariance with identical member/decode traces, malformed headers before
decoding, duplicate and foreign schemas, identity/ancestry contradictions,
exclusions, symlinks, redacted refusals, pin mismatches and all 46,512 missing
units above the work ceiling. The root reported 79 passing tests across four
synthetic suites; the reviewer did not independently rerun them.

Residual limits: no real scan, no actual row/stratum counts or U, and no measured
real-input forecast. The 120-second deadline is a prospective runaway guard.
Known status literals are decoded for schema validation and EXACT/nonexact
projection; no nonexact subtype is exported or counted.
