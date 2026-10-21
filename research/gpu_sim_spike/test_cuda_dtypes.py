"""PyTorch 2.9.1 CUDA dtype contracts, enforced even on CPU and meta tensors.

Fake/meta dispatch validates shapes, not CUDA kernel dtype availability. These
rules mirror the subset used by E1 of PyTorch v2.9.1's ScatterGatherKernel.cu,
IndexKernel.cu and ReduceArgMinMaxKernel.cu. They deliberately reject unknown
scatter/index overloads until reviewed. Native kernels have their own typed ABI.
This is an offline regression guard, not CUDA compilation/execution evidence.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from research.gpu_sim_spike.v8_device import _paint

# AT_DISPATCH_ALL_TYPES_AND2(Half, BFloat16): notably no Bool or UInt64.
NUMERIC = frozenset(
    (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
)
DATA = NUMERIC | {torch.bool}
INDEX = {torch.int32, torch.int64}


class CudaDtypeRules(TorchDispatchMode):
    """Check actual dispatched operands across every package test/kernel."""

    def __torch_dispatch__(
        self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None
    ) -> Any:
        kwargs = kwargs or {}
        name = func._schema.name.removeprefix("aten::")
        overload = func._schema.overload_name
        bound = {arg.name: value for arg, value in zip(func._schema.arguments, args)}
        bound.update(kwargs)
        tensor = bound.get("self")
        if name.startswith(("scatter", "gather", "index", "_index")):
            if name in {"scatter_reduce", "scatter_reduce_", "scatter_add", "scatter_add_"}:
                assert overload in {"two", "default", ""}, f"unreviewed CUDA op {func}"
                assert tensor.dtype in NUMERIC, f"CUDA {name} does not admit {tensor.dtype}"
                source, index = bound["src"], bound["index"]
                assert source.dtype == tensor.dtype, f"CUDA {name} src/self dtype mismatch"
                assert index.dtype in INDEX, f"CUDA {name} index must be int32/int64"
                if "reduce" in bound:
                    assert bound["reduce"] in {"sum", "prod", "mean", "amax", "amin"}
            elif name in {"scatter", "scatter_", "gather"}:
                assert overload in {"src", "value", "default", ""}, f"unreviewed CUDA op {func}"
                assert tensor.dtype in DATA
                assert bound["index"].dtype in INDEX, f"CUDA {name} index must be int32/int64"
                assert "reduce" not in bound, f"unreviewed CUDA scatter reduction {func}"
                if "src" in bound:
                    assert bound["src"].dtype == tensor.dtype
            elif name in {"index", "index_put", "index_put_", "_index_put_impl_"}:
                assert overload in {"Tensor", "default", ""}, f"unreviewed CUDA op {func}"
                assert tensor.dtype in DATA
                assert all(i is None or i.dtype in INDEX | {torch.bool} for i in bound["indices"])
                if "values" in bound:
                    assert bound["values"].dtype == tensor.dtype
            elif name == "index_select":
                assert tensor.dtype in DATA
                assert bound["index"].dtype in INDEX
            else:
                raise AssertionError(f"unreviewed CUDA indexing op {func}")
        elif name in {"argmax", "argmin", "amin", "amax", "min", "max", "cummin", "cummax"}:
            assert tensor.dtype in NUMERIC, f"unreviewed CUDA {name} dtype {tensor.dtype}"
        elif name in {"sum", "cumsum", "sort"}:
            assert tensor.dtype in DATA, f"unreviewed CUDA {name} dtype {tensor.dtype}"
        elif name in {"prod", "cumprod", "topk"}:
            assert tensor.dtype in NUMERIC, f"unreviewed CUDA {name} dtype {tensor.dtype}"
        return func(*args, **kwargs)


@pytest.mark.parametrize("device", ("cpu", "meta"))
def test_guard_rejects_original_bool_amax_on_cpu_and_meta(device: str) -> None:
    """Prove the guard catches the L4 failure which meta/FakeTensor missed."""
    with CudaDtypeRules(), pytest.raises(AssertionError, match="does not admit torch.bool"):
        torch.zeros((1, 4), dtype=torch.bool, device=device).scatter_reduce_(
            1,
            torch.zeros((1, 2), dtype=torch.long, device=device),
            torch.ones((1, 2), dtype=torch.bool, device=device),
            "amax",
        )


def test_paint_duplicate_invalid_and_empty_rows_match_independent_union() -> None:
    cells = torch.tensor(
        [[[1, 1], [1, 1], [-1, 0], [9, 9], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]
    )
    valid = torch.tensor([[True, False, True, True, False], [False] * 5])
    expected = torch.zeros((2, 3, 4), dtype=torch.bool)
    expected[0, 1, 1] = True
    got = _paint(cells, valid, 3, 4)
    assert got.dtype == torch.bool
    assert torch.equal(got, expected)
    assert _paint(cells[:, :0], valid[:, :0], 3, 4).sum() == 0


def test_guard_allows_bool_assignment_gather_and_numeric_amax() -> None:
    with CudaDtypeRules():
        bits = torch.zeros((1, 3), dtype=torch.bool)
        idx = torch.tensor([[1, 1]])
        bits.scatter_(1, idx, True)
        assert bits.gather(1, idx).all()
        for dtype in (torch.uint8, torch.int8, torch.int32, torch.int64):
            out = torch.zeros((1, 3), dtype=dtype)
            out.scatter_reduce_(1, idx, torch.ones((1, 2), dtype=dtype), "amax")
            assert out[0, 1] == 1


def test_guard_rejects_bad_index_width_and_bool_argmax() -> None:
    with CudaDtypeRules(), pytest.raises(AssertionError, match="index must be"):
        torch.ones((1, 2)).gather(1, torch.zeros((1, 1), dtype=torch.int16))
    with CudaDtypeRules(), pytest.raises(AssertionError, match="argmax dtype"):
        torch.ones(2, dtype=torch.bool).argmax()


def test_native_wrapper_abi_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate both native wrappers against the checked C++ pointer types.

    Meta tensors never compile or execute CUDA. The stub checks the ABI, not
    the numerical result; real CUDA tests remain separate and skip on Mac.
    """
    from pathlib import Path

    from research.gpu_sim_spike import flood_cuda

    contracts = {
        "flood": (("start", "Long"), ("blocked", "Bool"), ("cap", "Long"), ("release", "Long")),
        "components": (
            ("allowed", "Bool"),
            ("heads", "Long"),
            ("direction", "Long"),
            ("alive", "Bool"),
            ("rows", "Long"),
        ),
    }
    cpp = Path(flood_cuda.__file__).with_suffix(".cu").read_text()
    calls = []

    class Extension:
        def __getattr__(self, name: str) -> Callable:
            def call(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
                for tensor, (arg, dtype) in zip(args, contracts[name]):
                    # Bind this test to the native checks so ABI drift is visible.
                    assert f'check_tensor({arg}, device, torch::k{dtype}, "{arg}")' in cpp
                    assert tensor.dtype == {"Long": torch.int64, "Bool": torch.bool}[dtype]
                    assert tensor.is_contiguous()
                calls.append(name)
                shape = args[0].shape[:2] if name == "flood" else (args[4].shape[0], 3)
                return torch.empty(shape, dtype=torch.long, device="meta"), torch.zeros(
                    (), dtype=torch.int32, device="meta"
                )

            return call

    monkeypatch.setattr(flood_cuda, "_load", lambda: Extension())
    start = torch.zeros((2, 3, 2), dtype=torch.long, device="meta")
    blocked = torch.zeros((2, 3, 4, 5), dtype=torch.bool, device="meta")
    cap = torch.ones((2, 3), dtype=torch.long, device="meta")
    assert flood_cuda.flood(start, blocked, cap, None).shape == (2, 3)
    assert flood_cuda.flood(
        start, blocked, cap, torch.zeros_like(blocked, dtype=torch.long)
    ).shape == (2, 3)
    assert flood_cuda.components(
        blocked[:, 0].contiguous(),
        start,
        cap,
        torch.ones_like(cap, dtype=torch.bool),
        start[:, 0].contiguous(),
    ).shape == (2, 3)
    with pytest.raises(AssertionError):
        flood_cuda.flood(start.int(), blocked, cap, None)
    assert calls == ["flood", "flood", "components"]
