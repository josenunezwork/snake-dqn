# Frozen T34 full-digest oracle

The oracle is `research/t34_shadow/shadow.py` at commit
`c97e6d7d7add2a7bb4253983d9069c7117327553`, loaded by
`tests/test_t34_fast_digest.py` with `git show` under a private module name.
Its SHA-256 is
`08b0ed1864c8f7c79049f80b61adf775c6faf889a9ebcf58b9eae2263c363838`.

Do not replace it with an import from the candidate worktree. The fixture
checks whether the new digest preserves the old digest's divergence decisions;
the digest values themselves intentionally changed in schema v2.
