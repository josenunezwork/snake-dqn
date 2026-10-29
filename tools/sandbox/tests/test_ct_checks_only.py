"""C/T receipt reuse skips tests only after all packet checks pass."""

from pathlib import Path

import pytest

from research.sandbox_a_resident import ct_preflight


def test_checks_only_revalidates_packet_before_skipping_tests(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(ct_preflight.sys, "argv", ["ct_preflight", "/packet", "--checks-only"])
    monkeypatch.setattr(
        ct_preflight,
        "validate_packet",
        lambda path: calls.append(path) or {"count": 42},
    )
    monkeypatch.setattr(
        ct_preflight.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("checks-only must not run a test subprocess"),
    )
    assert ct_preflight.main() == 0
    assert calls == [Path("/packet")]
    assert "PASS_CT1028_CHECKS" in capsys.readouterr().out


def test_checks_only_cannot_bypass_invalid_packet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ct_preflight.sys, "argv", ["ct_preflight", "/packet", "--checks-only"])

    def refuse(path: Path) -> dict[str, int]:
        raise ValueError("invalid committed packet")

    monkeypatch.setattr(ct_preflight, "validate_packet", refuse)
    with pytest.raises(ValueError, match="invalid committed packet"):
        ct_preflight.main()


@pytest.mark.parametrize(
    "args", [["--checks-only"], ["/packet", "--checks-only", "--cuda-semantics"]]
)
def test_checks_only_requires_packet_and_excludes_semantics(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    monkeypatch.setattr(ct_preflight.sys, "argv", ["ct_preflight", *args])
    with pytest.raises(SystemExit) as error:
        ct_preflight.main()
    assert error.value.code == 2


def test_full_tests_do_not_inherit_allocation_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ct_preflight.sys, "argv", ["ct_preflight", "/packet"])
    monkeypatch.setattr(ct_preflight, "validate_packet", lambda path: {"count": 42})
    keys = (
        "SNAKE_SANDBOX_TEST_RECEIPT",
        "SNAKE_SANDBOX_ALLOCATION_LEASE",
        "SNAKE_SANDBOX_REQUIRE_TEST_RECEIPT",
    )
    for key in keys:
        monkeypatch.setenv(key, "foreign")
    observed = []
    monkeypatch.setattr(
        ct_preflight.subprocess, "run", lambda *args, **kwargs: observed.append(kwargs["env"])
    )
    assert ct_preflight.main() == 0
    assert observed and all(key not in observed[0] for key in keys)
