"""Generate basic chapter assets by executing code blocks from Markdown.

Run:
    python scripts/generate_basic_assets.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import japanize_matplotlib as jam

jam.japanize()

plt.style.use("scripts/k_dm.mplstyle")

CODE_BLOCK_PATTERN = re.compile(r"```python\s*(.*?)```", re.DOTALL)


def slugify(name: str) -> str:
    slug = name.replace(".ja", "").replace(".md", "")
    slug = slug.replace("_", "-").replace(" ", "-").lower()
    slug = re.sub(r"[^a-z0-9\-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "figure"


def extract_python_blocks(text: str) -> list[str]:
    return [dedent(match.group(1)).strip() for match in CODE_BLOCK_PATTERN.finditer(text)]


def iter_targets(content_root: Path) -> list[tuple[Path, list[str], list[str], str]]:
    targets: list[tuple[Path, list[str], list[str], str]] = []
    for md_path in sorted(content_root.rglob("*.ja.md")):
        text = md_path.read_text(encoding="utf-8")
        blocks = extract_python_blocks(text)
        if not blocks:
            continue
        rel_parts = md_path.relative_to(content_root).parts[:-1]
        rel_slugs = [slugify(part) for part in rel_parts]
        file_slug = slugify(md_path.stem)
        targets.append((md_path, blocks, rel_slugs, file_slug))
    return targets


def save_figures(
    figures: list[int],
    repo_root: Path,
    rel_slugs: list[str],
    file_slug: str,
    block_idx: int,
) -> None:
    output_dir = repo_root / "static" / "images" / "basic"
    if rel_slugs:
        output_dir = output_dir.joinpath(*rel_slugs)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fig_idx, fig_num in enumerate(figures, start=1):
        fig = plt.figure(fig_num)
        suffix = "" if fig_idx == 1 else f"_fig{fig_idx:02d}"
        filename = f"{file_slug}_block{block_idx:02d}{suffix}.svg"
        output_path = output_dir / filename
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved {output_path.relative_to(repo_root)}")
        plt.close(fig)


def main(_: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    content_root = repo_root / "content" / "basic"

    for md_path, blocks, rel_slugs, file_slug in iter_targets(content_root):
        print(f"Processing {md_path.relative_to(repo_root)}")
        namespace: dict[str, object] = {}
        for block_idx, code in enumerate(blocks, start=1):
            existing = set(plt.get_fignums())
            jam.japanize()
            try:
                exec(compile(code, str(md_path), "exec"), namespace)  # noqa: S102
            except Exception as exc:  # noqa: BLE001
                print(f"  Failed block {block_idx}: {exc}")
                plt.close("all")
                continue

            new_figs = [num for num in plt.get_fignums() if num not in existing]
            if not new_figs:
                print(f"  Block {block_idx}: no figures captured")
                continue

            save_figures(new_figs, repo_root, rel_slugs, file_slug, block_idx)

        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(parser.parse_args())
