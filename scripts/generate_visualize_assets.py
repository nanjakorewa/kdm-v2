"""Generate visualization assets by executing code blocks from Markdown.

Run:
    python scripts/generate_visualize_assets.py
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

TARGET_MARKDOWN = [
    "content/visualize/line/lineplot-multi.ja.md",
    "content/visualize/line/area-chart.ja.md",
    "content/visualize/correlation/corr-heatmap.ja.md",
    "content/visualize/correlation/scatter-matrix.ja.md",
    "content/visualize/advanced/hexbin.ja.md",
]

CODE_BLOCK_PATTERN = re.compile(r"```python\n(.*?)```", re.DOTALL)


def extract_python_blocks(text: str) -> list[str]:
    return [dedent(match.group(1)).strip() for match in CODE_BLOCK_PATTERN.finditer(text)]


def run_snippet(code: str, file_path: Path) -> None:
    namespace: dict[str, object] = {}
    exec(compile(code, str(file_path), "exec"), namespace)  # noqa: S102
    plt.close("all")


def main(_: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for relative in TARGET_MARKDOWN:
        md_path = repo_root / relative
        code_blocks = extract_python_blocks(md_path.read_text(encoding="utf-8"))
        for idx, block in enumerate(code_blocks, start=1):
            run_snippet(block, md_path)
            print(f"Executed block {idx} from {relative}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(parser.parse_args())
