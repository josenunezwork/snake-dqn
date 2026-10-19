# Published manifest validation

Ten real CLI-schema review manifests are in `jobs_review/`, five per arm. All
passed `research.colab_cli.supervisor.validate` locally with zero Colab
allocations. `MANIFEST_VALIDATION.json` binds their exact bytes and inputs.
They pin the published source commit
`9a18e66120c0b869044f609a9c48dd5df338a281` and retain runtime DRAFT flags;
passing schema validation is not study admission.

Each job requests only L4, at least 12 vCPUs, a 180-second lease, and at most
2 GiB per immutable export. N3 estimate/timeout is 30/60 minutes; N1 is 68/136.
Input paths are Mac-side supervisor uploads, not runtime references to Mac files.
The actual seven input files total 62,552,352 bytes and are bound by `INPUTS.json`.

Registration remains DRAFT. `ADMISSION_STATUS_FINAL.json` remains REFUSED for
the failed inherited counter-bound preflight and unqualified complete full-size
transport liability. These remain blockers even if live supervisor acceptance
passes. No production launch command should be executed before they are cleared
prospectively within the existing caps and Claude approves the complete package.

After all gates pass, generate fresh approved production manifests and perform
the one-command launch using the exact sequence in
`IMPLEMENTATION_RECEIPT.md`. Do not edit or execute these DRAFT review manifests
as a way around admission.

The latest focused checks pass: 48 study tests, Black/isort/flake8, Bash syntax,
and execution of the absolute launcher with `--help` from `/tmp`. The updated
supervisor plus study check passed 134 tests before the final transport-layout
test was added; that additional test and the complete 48-test S1 suite pass.
The known full-suite baseline failures are documented in the implementation
receipt. No checkpoint, serving pin, Q1 file, live tmux session, or protected
supervisor source was modified by S1.
