"""Pure-Python freshness search contracts, independent of executable PATH."""

from pathlib import Path

import pytest

from research.fixed_search import fixed_string_hits


def test_fixed_search_lists_files_once_and_matches_literal_bytes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "")
    nested = tmp_path / "nested"
    nested.mkdir()
    match = nested / "match.txt"
    match.write_bytes(b"\xff literal[1].* literal[1].* 101 303\n")
    (tmp_path / "miss.txt").write_bytes(b"literal1zzz 1101 2020 _303 303x")
    assert fixed_string_hits(tmp_path, ["literal[1].*"]) == {match: {"literal[1].*"}}
    assert fixed_string_hits(tmp_path, ["absent"]) == {}
    assert fixed_string_hits(tmp_path, []) == {}
    assert fixed_string_hits(tmp_path, ["101", "202", "303"], word_regexp=True) == {
        match: {"101", "303"}
    }


def test_fixed_search_word_boundaries_are_unicode(tmp_path: Path) -> None:
    path = tmp_path / "words.txt"
    path.write_text("é101 101é _101 101_ 1101 101\u0301 \u200d101", encoding="utf-8")
    assert fixed_string_hits(tmp_path, ["101"], word_regexp=True) == {}
    assert fixed_string_hits(tmp_path, ["101"]) == {path: {"101"}}


def test_fixed_search_missing_root_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not a directory"):
        fixed_string_hits(tmp_path / "missing", ["needle"])


def test_fixed_search_read_error_fails_closed(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "receipt").write_bytes(b"needle")

    def denied(_self, *_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(RuntimeError, match="denied"):
        fixed_string_hits(tmp_path, ["needle"])


def test_fixed_search_skips_ignored_hidden_binary_and_symlink_files(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("cache/\n*.log\n!keep.log\n")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "receipt").write_bytes(b"needle")
    for name in ("skip.log", "keep.log", ".hidden"):
        (tmp_path / name).write_bytes(b"needle")
    (tmp_path / "binary").write_bytes(b"\0needle")
    (tmp_path / "link").symlink_to(tmp_path / "keep.log")
    assert fixed_string_hits(tmp_path, ["needle"]) == {tmp_path / "keep.log": {"needle"}}
