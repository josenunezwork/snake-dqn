# Bounded implementation review

Read-only reviewer lane `/root/review_reconciliation` inspected the source, synthetic tests, metadata contract/inventory and historical `_table` implementation. No real data or outcomes were opened; no tests were run by the reviewer. This is implementation review, not independent scientific acceptance or renewed scan authorization.

Initial findings: P2 direct API could bypass CLI-only reservation; P2 output ancestor substitution between safe_path and ordinary file creation. Both were fixed and re-reviewed. The API now requires/consumes an empty exclusive reservation at the fixed inode before source access; the CLI creates that file through no-follow directory descriptors. Tests cover repeated API invocation and ancestor substitution.

Final reviewer conclusion: no remaining findings within the bounded re-review. Core reconciliation preserves the frozen IDs, refuses missing selected or extra IDs, and permits missing ordinary-only siblings solely with complete independently accepted count metadata matching all observed counts. A null metadata witness remains unproven and refuses. A provenance string and syntactic schema validation cannot authenticate a count attestation; independent review must verify its provenance before accepting changed inventory pins.

Reviewed source SHA-256: `49f7cb0fcadc2b95569aabe6d2e1613d2f46c5ed5aec6c24f1cbeebf60a157df`.
