#!/usr/bin/env python3
"""Convert single-escaped inline LaTeX delimiters to double-escaped form.

This script rewrites occurrences like `\(` and `\)` into `\\(` and `\\)` so that
the delimiters survive additional Markdown or template processing stages that
eat the first backslash. Existing double-escaped delimiters are left untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import re


OPEN_PATTERN = re.compile(r"(?<!\\)\\\(")
CLOSE_PATTERN = re.compile(r"(?<!\\)\\\)")


def convert_text(text: str) -> str:
    """Return text with single-escaped inline LaTeX delimiters doubled."""
    text = OPEN_PATTERN.sub(r"\\\\(", text)
    text = CLOSE_PATTERN.sub(r"\\\\)", text)
    return text


def process_file(path: Path, dry_run: bool) -> bool:
    """Convert delimiters in a single file. Returns True if modified."""
    original = path.read_text(encoding="utf-8")
    converted = convert_text(original)
    if original == converted:
        return False
    if not dry_run:
        path.write_text(converted, encoding="utf-8")
    return True


def iter_target_files(paths: list[Path], extensions: set[str]) -> list[Path]:
    """Yield files to process given paths and allowed extensions."""
    for path in paths:
        if path.is_dir():
            yield from (
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and (not extensions or candidate.suffix.lower() in extensions)
            )
        elif path.is_file():
            if not extensions or path.suffix.lower() in extensions:
                yield path
        else:
            raise FileNotFoundError(f"Path not found: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Escape inline LaTeX delimiters: \\( -> \\\\(, \\) -> \\\\)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to process.",
    )
    parser.add_argument(
        "--ext",
        "-e",
        action="append",
        default=[".md"],
        help="File extensions to include (default: .md). Use multiple times for more.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would change without writing them.",
    )

    args = parser.parse_args(argv)

    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.ext}
    changed = []
    for file_path in sorted(set(iter_target_files(args.paths, extensions))):
        try:
            if process_file(file_path, dry_run=args.dry_run):
                changed.append(file_path)
        except Exception as exc:  # pragma: no cover - surface errors clearly.
            print(f"[ERROR] {file_path}: {exc}", file=sys.stderr)
            return 1

    if changed:
        header = "Would update" if args.dry_run else "Updated"
        print(f"{header} {len(changed)} file(s):")
        for path in changed:
            print(f"  {path}")
    else:
        print("No changes needed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
