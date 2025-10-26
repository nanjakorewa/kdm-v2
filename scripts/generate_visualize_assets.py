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
import matplotlib.pyplot as plt
import japanize_matplotlib

plt.style.use("scripts/k_dm.mplstyle")


TARGET_MARKDOWN = [
    "content/visualize/advanced/bump-chart.ja.md",
    "content/visualize/advanced/dumbbell-chart.ja.md",
    "content/visualize/advanced/hexbin.ja.md",
    "content/visualize/advanced/polar-area.ja.md",
    "content/visualize/advanced/radar-chart.ja.md",
    "content/visualize/bar/diverging-bar.ja.md",
    "content/visualize/bar/grouped-bar.ja.md",
    "content/visualize/bar/horizontal-bar.ja.md",
    "content/visualize/bar/lollipop-chart.ja.md",
    "content/visualize/bar/simple-bar.ja.md",
    "content/visualize/bar/stacked-bar.ja.md",
    "content/visualize/bar/waterfall-chart.ja.md",
    "content/visualize/bar/waffle-chart.ja.md",
    "content/visualize/distribution/boxplot.ja.md",
    "content/visualize/distribution/densityplot.ja.md",
    "content/visualize/distribution/ecdf.ja.md",
    "content/visualize/distribution/histogram.ja.md",
    "content/visualize/distribution/kde-2d.ja.md",
    "content/visualize/distribution/qqplot.ja.md",
    "content/visualize/distribution/ridgelineplot.ja.md",
    "content/visualize/distribution/rugplot.ja.md",
    "content/visualize/distribution/swarmplot.ja.md",
    "content/visualize/distribution/violinplot.ja.md",
    "content/visualize/line/area-chart.ja.md",
    "content/visualize/line/line-dual-axis.ja.md",
    "content/visualize/line/line-forecast-band.ja.md",
    "content/visualize/line/line-highlight-range.ja.md",
    "content/visualize/line/line-rolling-average.ja.md",
    "content/visualize/line/line-sparkline.ja.md",
    "content/visualize/line/line-step.ja.md",
    "content/visualize/line/lineplot-multi.ja.md",
    "content/visualize/scatter/scatter-basic.ja.md",
    "content/visualize/scatter/scatter-bubble.ja.md",
    "content/visualize/scatter/scatter-category-annotation.ja.md",
    "content/visualize/scatter/scatter-jointplot.ja.md",
    "content/visualize/scatter/scatter-lm.ja.md",
    "content/visualize/scatter/scatter-marginal-hist.ja.md",
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
        
        try:
            for idx, block in enumerate(code_blocks, start=1):
                jam.japanize()
                run_snippet(block, md_path)
                print(f"Executed block {idx} from {relative}")
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(parser.parse_args())
