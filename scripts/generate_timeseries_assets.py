"""Generate time series assets by executing code blocks from Markdown files.

Run:
    python scripts/generate_timeseries_assets.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import japanize_matplotlib  # noqa: E402

plt.style.use("scripts/k_dm.mplstyle")


TARGET_MARKDOWN_TS = [
    "content/basic/timeseries/Anomaly_Zscore.ja.md",
    "content/basic/timeseries/AR_Model.ja.md",
    "content/basic/timeseries/ARMA_Model.ja.md",
    "content/basic/timeseries/Autocorrelation.ja.md",
    "content/basic/timeseries/Autocorrelation_Heatmap.ja.md",
    "content/basic/timeseries/Calendar_Heatmap.ja.md",
    "content/basic/timeseries/Change_Point.ja.md",
    "content/basic/timeseries/Differencing_Comparison.ja.md",
    "content/basic/timeseries/ETS_Model.ja.md",
    "content/basic/timeseries/Granger_Causality_Bar.ja.md",
    "content/basic/timeseries/Lag_Plot_Grid.ja.md",
    "content/basic/timeseries/MA_Model.ja.md",
    "content/basic/timeseries/Missing_Data_Highlight.ja.md",
    "content/basic/timeseries/Monthly_Boxplot.ja.md",
    "content/basic/timeseries/Monthly_Subseries.ja.md",
    "content/basic/timeseries/Partial_Autocorrelation.ja.md",
    "content/basic/timeseries/Percent_Change.ja.md",
    "content/basic/timeseries/Power_Spectrum.ja.md",
    "content/basic/timeseries/Resample_Trend.ja.md",
    "content/basic/timeseries/Rolling_Correlation.ja.md",
    "content/basic/timeseries/Rolling_Forecast_Origin.ja.md",
    "content/basic/timeseries/Rolling_Statistics.ja.md",
    "content/basic/timeseries/SARIMAX_Exogenous.ja.md",
    "content/basic/timeseries/Seasonal_Polar.ja.md",
    "content/basic/timeseries/STL_Forecast.ja.md",
    "content/basic/timeseries/Time_Series_Overview.ja.md",
    "content/basic/timeseries/Train_Test_Split.ja.md",
    "content/basic/timeseries/UCM_Model.ja.md",
    "content/basic/timeseries/VAR_Forecast.ja.md",
    "content/basic/timeseries/Weekday_Averages.ja.md",
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
    for relative in TARGET_MARKDOWN_TS:
        md_path = repo_root / relative
        code_blocks = extract_python_blocks(md_path.read_text(encoding="utf-8"))

        for idx, block in enumerate(code_blocks, start=1):
            japanize_matplotlib.japanize()
            try:
                run_snippet(block, md_path)
                print(f"Executed block {idx} from {relative}")
            except FileNotFoundError:
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(parser.parse_args())
