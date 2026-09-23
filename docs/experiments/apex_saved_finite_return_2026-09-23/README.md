# Saved H64 finite-return comparison

The saved-record screen found `NO_STRONG_FINITE_RETURN_DISCORDANCE`. All 30 seed × bank × arm cells are in [table.md](table.md); the fixed representative records are copied to [representatives.json](representatives.json). The independent audit passed after validating 32 pinned game files, 1,536 saved games, and 98,304 native frames.

This was a saved-record analysis of completed artifacts, with no new gameplay or training: zero new games, model forwards, or optimizer updates. Each H64 record contains at most 64 native frames. The report computes finite discounted return with γ = 0.99 and no bootstrap. The fixed decision required the same arm to have lower food than its teacher, mean policy-minus-teacher G64 of at least 1.0, and at least 31 return wins out of 48 in both banks, in at least two of three seeds. No arm met that rule.

That result does not prove reward alignment or absence of a reward problem. It only says this screen did not find the predeclared strong discordance pattern in these finite policy-conditioned paths. The 48 teacher games per bank are reused across all three lineages and five arms; they are not 144 independent teacher games. The separate three-reward window after first food is exploratory and descriptive only. It did not affect the decision branch or define TD targets.

The compared policies are fixed source, control, epsilon-0.1, mixture, and uniform endpoints at the exact marks and checkpoint identities recorded by the frozen design. Source means the saved mark-5000 control checkpoint. The known adaptive and previously fresh/adaptive H64 banks are existing evaluation worlds; “fresh” is a historical bank label, not new confirmation. No reward change, training continuation, checkpoint selection, promotion, or reopening of a prior experiment is authorized by this result.

## Provenance

- Frozen design: `apex-saved-return-design/design-v2.json`, SHA-256 `11b6bc490cb93fc5768a587904eb5f607fcc8a85f894ea391352bae779a311a7`.
- Frozen repository HEAD: `66270442b7cd27e763200fa133894c4d187ca64c`.
- Frozen analyzer `analyze.py`: SHA-256 `0efcacd076be28ca22b88f8c66212503e6716f6618711a8cf2d4a86bef6b40e0`.
- Independent auditor `audit.py`: SHA-256 `1310a226ef9e8806680ffdb6ae71cf1a9be321f70cb6bcd6e89717fcbe0aa6e5`.
- Saved analysis report: SHA-256 `3303a10eb31a38886a6ae8b527e152064775a70af8bf598a078edf8433fe39ad`.
- Independent audit report: SHA-256 `d45afddadad936f499cd87e80b500bb6db5f94f1b49bfeb43794e708776303b1` (PASS).
- Sealed closeout: SHA-256 `89bfd03d4b3f7f3cec6309d06808453e92e23a1a8f1b3de161d47fed2870656d`; two natural-success receipts, 3.745063333 guarded seconds, peak child RSS 160,923,648 bytes, minimum available memory 35,162,849,280 bytes.

No new learning curve applies because this analysis performed zero updates. For historical context only, [the prior uniform-replay learning curves](../apex_uniform_replay_continuation_2026-09-22/learning-curves.png) and [representative paths](../apex_uniform_replay_continuation_2026-09-22/representative-paths.png) show an earlier policy study; they are not evidence from this finite-return screen.
