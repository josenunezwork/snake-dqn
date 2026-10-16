"""Dependency-free fixed-string searches for research freshness audits."""

from __future__ import annotations

import fnmatch
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable


def fixed_string_hits(
    root: Path, needles: Iterable[str], *, word_regexp: bool = False
) -> dict[Path, set[str]]:
    """Return matching files and their literal matches; no hits returns an empty dict.

    Walk visible regular files without following symlinks. Read bytes so invalid
    UTF-8 cannot abort an audit. Whole-word matching uses Unicode word boundaries,
    as in ripgrep. I/O failures raise rather than reporting a false clean scan.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise RuntimeError(f"search root is not a directory: {root}")
    patterns = {needle: re.compile(re.escape(needle)) for needle in needles}

    def word_character(char: str) -> bool:
        return (
            char.isalnum()
            or unicodedata.category(char) in {"Mn", "Mc", "Me", "Pc", "Nl"}
            or char in "\u200c\u200d"
        )

    def whole_word(pattern: re.Pattern[str], text: str) -> bool:
        return any(
            (match.start() == 0 or not word_character(text[match.start() - 1]))
            and (match.end() == len(text) or not word_character(text[match.end()]))
            for match in pattern.finditer(text)
        )

    hits: dict[Path, set[str]] = {}
    if not patterns:
        return hits

    def fail(error: OSError) -> None:
        raise RuntimeError(f"fixed-string search failed under {root}: {error}") from error

    # Carry directory-local ignore rules down the walk, with later rules winning.
    rules_by_dir: dict[Path, list[tuple[Path, str]]] = {}
    inherited: list[tuple[Path, str]] = []
    for parent in reversed(root.parents):
        for filename in (".gitignore", ".ignore", ".rgignore"):
            ignore = parent / filename
            if ignore.is_file():
                inherited.extend((parent, line) for line in ignore.read_text().splitlines())

    def ignored(path: Path, rules: list[tuple[Path, str]], is_dir: bool) -> bool:
        result = False
        for base, raw in rules:
            pattern = raw.rstrip()
            if not pattern or pattern.startswith("#"):
                continue
            negate = pattern.startswith("!")
            if negate:
                pattern = pattern[1:]
            directory_only = pattern.endswith("/")
            pattern = pattern.rstrip("/")
            if directory_only and not is_dir:
                continue
            relative = path.relative_to(base).as_posix()
            anchored = pattern.startswith("/") or "/" in pattern
            pattern = pattern.lstrip("/")
            candidate = relative if anchored else path.name
            if fnmatch.fnmatchcase(candidate, pattern):
                result = not negate
        return result

    for directory, dirs, files in os.walk(root, onerror=fail, followlinks=False):
        folder = Path(directory)
        rules = list(rules_by_dir.get(folder, inherited))
        for filename in (".gitignore", ".ignore", ".rgignore"):
            ignore = folder / filename
            if ignore.is_file():
                try:
                    rules.extend((folder, line) for line in ignore.read_text().splitlines())
                except OSError as error:
                    fail(error)
        dirs[:] = sorted(
            name
            for name in dirs
            if not name.startswith(".")
            and not (folder / name).is_symlink()
            and not ignored(folder / name, rules, True)
        )
        for name in dirs:
            rules_by_dir[folder / name] = rules
        for name in sorted(files):
            path = folder / name
            if (
                name.startswith(".")
                or path.is_symlink()
                or not path.is_file()
                or ignored(path, rules, False)
            ):
                continue
            try:
                with path.open("rb") as stream:
                    data = stream.read(8192)
                    if b"\0" in data:
                        continue
                    data += stream.read()
            except OSError as error:
                fail(error)
            if b"\0" in data:
                continue
            text = data.decode("utf-8", errors="replace") if word_regexp else ""
            matches = {
                needle
                for needle, pattern in patterns.items()
                if (whole_word(pattern, text) if word_regexp else needle.encode("utf-8") in data)
            }
            if matches:
                hits[path] = matches
    return hits
