# E16 pre-data exact seed recipe

Prospective supplement to REGISTRATION.md, committed before allocation. Domain
`bl-e16-20261021/v1`; phases `critic`, `actor`, `timing`, `preflight`, `evaluation`;
mixes scripted/frozen/mixed. For each phase/mix/index:

```
seed = 2**48 + int.from_bytes(
    SHA256(f'{DOMAIN}/round-{round}/{phase}/{mix}|{index}').digest()[:6], 'big')
```

Counts per mix: critic4096, actor4096, timing3, preflight1, evaluation96.
Round0 first; deterministic consecutive whole-round candidates through100, rejecting
and preserving an entire candidate on any collision. No gameplay during allocation.
All seeds fit signed64; test full-int EnvRng upperbit preservation before gameplay.
All treatment arms consume the same critic seed vector. Finite per-phase training
allocation exhaustion refuses execution; no hidden uint32 seed derivation or cycling.
