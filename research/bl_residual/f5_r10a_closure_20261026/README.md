# F5 R10a sibling closure preparation

This package implements the bounded metadata preparation in the source-bound
[Astra ruling](JUDGMENT_F5_R10A_GAP_CLOSURE_20261026T003745Z.md). The original
`F5_R10A_METADATA_ANNEX_20261025` remains immutable (JSON SHA-256
`8ee06291b23a1ef8c657b59a4f4990349ae4c524a680c47f06785a47d227f764`).

The status extractor is prepared and tested only with synthetic inputs. Its
production inventory is copied from the admitted metadata, never discovered by
opening training label containers. No real-input extraction is performed by this
task. Independent acceptance must bind the exact source, test, contract and frozen
inventory hashes before any later machine extraction. A local code review is
verification evidence; it does not substitute for the required adoption decision.

The runtime and collision reports distinguish recovered historical metadata from
missing evidence. A source pin or current installed package version cannot fill a
historical runtime gap. An allocation that is clear against an incomplete catalog
cannot become an accepted allocation. A custodian completeness attestation must
come from a custodian; this package cannot issue one on their behalf.

No outcome payload, D4096 material or `t34-round0-study-01/` content may be opened.
No F4 post-mortem, F5 mechanism selection, coverage completion, fit, replay, canary,
Colab, dispatch or serving change is authorized by this package.

All test commands use `nice -n 15 ./venv/bin/python`, one thread per numeric
library, disabled plugin autoload, and `-p no:xdist`. Tests are synthetic and run
sequentially with no test-worker pool. See `synthetic-verification.json` for the
exact final commands and results.
