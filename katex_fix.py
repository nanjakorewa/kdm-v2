#!/usr/bin/env python3
"""
katex_fix.py
------------
Scan a Hugo site's `content/` directory and convert math delimiters:
  - \( ... \)  ->  $ ... $
  - \[ ... \]  ->  $$ ... $$
This helps KaTeX auto-render pick up math when Markdown (Goldmark) eats backslashes.

The script skips fenced code blocks (``` or ~~~) and inline code spans (`...`).

Usage:
  python katex_fix.py                   # process ./content recursively (in-place)
  python katex_fix.py --path content    # specify root (default: ./content)
  python katex_fix.py --dry-run         # show changes without writing
  python katex_fix.py --ext md markdown # add/override extensions (default: md, markdown)

Notes:
- Creates a .bak backup next to modified files unless --no-backup is set.
- Counts and prints a summary of conversions.
"""
from __future__ import annotations
import argparse
import os
import re
from typing import Dict, List, Tuple

FENCE_RE = re.compile(
    r'(^```.*?$[\s\S]*?^```$)|(^~~~.*?$[\s\S]*?^~~~$)',
    re.MULTILINE
)
INLINE_CODE_RE = re.compile(r'`+[^`]*`+')

# Math delimiters (non-greedy; dotall so newlines inside are allowed)
INLINE_MATH_RE = re.compile(r'\\\((.+?)\\\)', re.DOTALL)   # \( ... \)
DISPLAY_MATH_RE = re.compile(r'\\\[(.+?)\\\]', re.DOTALL)  # \[ ... \]

def mask_regions(text: str, pattern: re.Pattern, token_prefix: str) -> Tuple[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    def repl(m):
        key = f'__{token_prefix}_{len(mapping)}__'
        mapping[key] = m.group(0)
        return key
    masked = pattern.sub(repl, text)
    return masked, mapping

def unmask(text: str, mapping: Dict[str, str]) -> str:
    # Replace tokens in insertion order
    for key, val in mapping.items():
        text = text.replace(key, val)
    return text

def convert_math(text: str) -> Tuple[str, int, int]:
    """Convert \(..\) -> $..$ and \[..] -> $$..$$, return new text and counts."""
    inline_count = 0
    display_count = 0

    def inline_repl(m):
        nonlocal inline_count
        inline_count += 1
        inner = m.group(1).strip()
        return f'${inner}$'

    def display_repl(m):
        nonlocal display_count
        display_count += 1
        inner = m.group(1).strip()
        # Keep on one line to avoid introducing unintended paragraphs
        return f'$$ {inner} $$'

    text = INLINE_MATH_RE.sub(inline_repl, text)
    text = DISPLAY_MATH_RE.sub(display_repl, text)
    return text, inline_count, display_count

def process_file(path: str, dry_run: bool = False, backup: bool = True) -> Tuple[bool, int, int]:
    """Return (modified?, inline_count, display_count)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            original = f.read()
    except UnicodeDecodeError:
        # Try fallback
        with open(path, 'r', encoding='utf-8-sig') as f:
            original = f.read()

    text = original

    # 1) Mask fenced code blocks
    text, fence_map = mask_regions(text, FENCE_RE, 'FENCE')

    # 2) Mask inline code spans
    text, inline_map = mask_regions(text, INLINE_CODE_RE, 'CODE')

    # 3) Convert math in the remaining regions
    text, ic, dc = convert_math(text)

    # 4) Unmask
    text = unmask(text, inline_map)
    text = unmask(text, fence_map)

    modified = (text != original)
    if modified and not dry_run:
        if backup:
            try:
                with open(path + '.bak', 'w', encoding='utf-8') as b:
                    b.write(original)
            except Exception as e:
                print(f'[warn] Could not write backup for {path}: {e}')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

    return modified, ic, dc

def main():
    ap = argparse.ArgumentParser(description='Fix KaTeX delimiters in Hugo content.')
    ap.add_argument('--path', default='content', help='Root directory to scan (default: ./content)')
    ap.add_argument('--dry-run', action='store_true', help='Show changes without writing files')
    ap.add_argument('--no-backup', action='store_true', help="Don't create .bak backups")
    ap.add_argument('--ext', nargs='*', default=['md', 'markdown'], help='File extensions to include')
    args = ap.parse_args()

    root = args.path
    if not os.path.isdir(root):
        print(f'[error] Not a directory: {root}')
        return

    exts = {e.lower().lstrip('.') for e in args.ext}
    total_files = 0
    changed_files = 0
    total_inline = 0
    total_display = 0

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
            if ext not in exts:
                continue
            path = os.path.join(dirpath, name)
            total_files += 1
            modified, ic, dc = process_file(path, dry_run=args.dry_run, backup=not args.no_backup)
            if modified:
                changed_files += 1
                total_inline += ic
                total_display += dc
                print(f'[fix] {path}  (+{ic} inline, +{dc} display)')
    print('—' * 72)
    print(f'Files scanned:  {total_files}')
    print(f'Files changed:  {changed_files}')
    print(f'Inline math:    +{total_inline}  (\\( .. \\) -> $..$)')
    print(f'Display math:   +{total_display} (\\[ .. \\] -> $$..$$)')
    if args.dry_run:
        print('[dry-run] No files were modified.')

if __name__ == "__main__":
    main()
