# Frozen experiment bundling contract

An allocation may contain several experiments only when their source pins,
input digests, output namespaces, and acceptance gates are frozen before
allocation. Setup is shared by the allocation; scientific acceptance remains
independent per experiment. A failed experiment closes its own branch and
cannot turn another experiment's result into a PASS.

Each experiment publishes payload bytes, a SHA-256 sidecar, and a completion
marker in that order. The marker is authoritative only after the Mac verifies
the payload and sidecar. The first attempt uploads an empty verified ledger;
later attempts skip exactly verified units in the same namespace and
allocation index, reject foreign namespaces and duplicate IDs, and never
recompute a verified unit into the result set.

The retention packet probe writes this contract into each manifest's bundle
and emits immutable fixture exports and ledgers. Supervisor receipts,
allocation accounting, and fault evidence are still required for independent
acceptance. The bundling mechanism does not change the transport qualification
registry or permit a release before the retention acceptance is PASS.

## Bundle API and Mac ledger

`python -m research.colab_cli.bundle validate PLAN` verifies a frozen plan;
`empty-ledger PLAN LEDGER` creates its outcome-blind input ledger. Plans pin a
40-hex source commit plus source/toolchain/dependency SHA-256 values. Cells use
unique `experiment/variant/cell/repeat` namespaces, argv commands, and owned
relative payload paths. Each cell may specify its own `experiment`, overriding
the plan's default, so I0, R1 and Q5 can share one allocation without sharing
resume identities or acceptance verdicts. Runtime supplies `COLAB_BUNDLE_NAMESPACE` and
`COLAB_BUNDLE_CELL_DIR`, then publishes payload, `.sha256`, and `.complete`.

After export, the Mac runs `python -m research.colab_cli.bundle verify-ledger
PLAN BUNDLE_OUTPUT --input-ledger OLD --output NEXT`. It admits only
path/size/hash/final-marker facts, binds the full plan digest, and rejects
foreign or duplicate namespaces; remote output cannot self-promote.

`depends_on` needs a frozen SHA-bound `dependency_admissions` Mac verdict with
`status: PASS` for a parent namespace already in the verified ledger. A hash
completion alone never opens a scientific branch. For
`continue_after_isolation`, a cell freezes restore argv and isolation-verifier
argv; the verifier must exit zero after restore before unrelated continuation.
The caller defines the isolated-state contract. Adoption still needs I0 to
show at least 20% fewer allocated minutes per completed valid cell while all
isolation, accounting, export, and independent-acceptance gates hold. Package
tests provide no numerical acceptance evidence.

Dependency admission bindings also declare `dest`, a safe relative uploaded
input destination. On Colab the verifier opens `COLAB_INPUT_DIR/dest`; the
absolute `path` is only the Mac preparation source. Upload these bindings as
ordinary SHA-bound manifest inputs. A Mac verdict must bind `status: PASS`, the
parent `namespace`, its exact `payload_sha256` from the verified ledger and the
same frozen `source` object. A PASS for another payload/source cannot open a
branch. Mac ledger union rejects duplicate completed units across attempts and
foreign completion markers, even when duplicate bytes happen to match.
