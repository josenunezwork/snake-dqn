# T23 illustrative queue jobs

T23 R0 closed **KILL** on 2026-10-17. These files document how a single train
seed and an evaluation shard fit the queue contract; they do not authorize a
relaunch. Both adapters fail unless an operator explicitly sets
`T23_QUEUE_EXAMPLE_ALLOW=1`, which the checked-in JSON deliberately disables.

The enqueue tool must pre-create each job's private `output_dir`. The adapters
never create a Drive directory: they stage their working tree under `/content`,
then publish a validated tarball, its SHA-256 sidecar, and their log into the
provided `QUEUE_OUTPUT_DIR`.

`t23_eval_shard.sh` reads completed queue output from `T23_ROOT`, which defaults
to `/content/drive/MyDrive/t16_colab/t23-queue`. It validates and stages each
`seed-<n>/t23-seed-<n>.tar.gz` plus SHA-256 sidecar locally. The five train-job
IDs in the shard's dependency list illustrate the required fan-in; instantiate
the train sample once per seed by changing its id, `T23_SEED`, and `output_dir`.
It also accepts the original completed `seed-<n>` directories as a read-only
fallback. The registered evaluator needs all five seeds. It writes parity and
evaluation receipts locally before packaging them, so it cannot reproduce the
prior Drive duplicate-directory failure.

The commit fields pin the published queue adapters and repository setup. When
adapting these examples for an authorized study, use its published full commit
and registered launcher parameters; these guarded specs are not T23 relaunch requests.
