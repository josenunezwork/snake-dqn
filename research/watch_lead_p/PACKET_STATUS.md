# P packet status

Use **packet-03**, identity `sandbox-p-watch-lead-20261028-03`. Its runtime pin is
`f97838ebdc5d3315a95f173c623f717fc115d223`. No packet has been dispatched and none
of the 32 scientific sessions has been run. Packet numbering here records
pre-dispatch packaging revisions, not completion attempts or extra data looks.

Packet-01 is preserved unchanged and **must not be dispatched**. Its offline
dispatch dry-run passed, but an independent extracted-sparse-bundle check found
that the root requirements file and a dynamically loaded serving schema had not
been included. The final dependency inventory includes both; gameplay, seeds,
roster, scheduler, statistics, N and the measured fixture engine did not change.

An uncommitted packet-02 draft had an incorrectly transcribed source pin. It was
caught before commit, checkout or dispatch and archived unchanged outside Git at
`snake-dqn-artifacts/t34-watch-lead-1028/engineering-01/invalid-uncommitted-draft-02/`.
The sealer now rejects an invalid or absent full commit pin before creating any
packet files, and packet-03 takes its pin directly from Git.

The source and script/input digests in packet-03 are authoritative. Operator
dispatch and fresh account/hardware checks remain separate from local tests,
review, packet sealing and an allocation-free dry-run.
