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


TIMESERIES_DIR = Path("content") / "timeseries"

TARGET_MARKDOWN_TS = [
    TIMESERIES_DIR / "Anomaly_Zscore.ja.md",
    TIMESERIES_DIR / "AR_Model.ja.md",
    TIMESERIES_DIR / "ARMA_Model.ja.md",
    TIMESERIES_DIR / "Autocorrelation.ja.md",
    TIMESERIES_DIR / "Autocorrelation_Heatmap.ja.md",
    TIMESERIES_DIR / "Calendar_Heatmap.ja.md",
    TIMESERIES_DIR / "Change_Point.ja.md",
    TIMESERIES_DIR / "Differencing_Comparison.ja.md",
    TIMESERIES_DIR / "ETS_Model.ja.md",
    TIMESERIES_DIR / "Granger_Causality_Bar.ja.md",
    TIMESERIES_DIR / "Lag_Plot_Grid.ja.md",
    TIMESERIES_DIR / "MA_Model.ja.md",
    TIMESERIES_DIR / "Missing_Data_Highlight.ja.md",
    TIMESERIES_DIR / "Monthly_Boxplot.ja.md",
    TIMESERIES_DIR / "Monthly_Subseries.ja.md",
    TIMESERIES_DIR / "Partial_Autocorrelation.ja.md",
    TIMESERIES_DIR / "Percent_Change.ja.md",
    TIMESERIES_DIR / "Power_Spectrum.ja.md",
    TIMESERIES_DIR / "Resample_Trend.ja.md",
    TIMESERIES_DIR / "Rolling_Correlation.ja.md",
    TIMESERIES_DIR / "Rolling_Forecast_Origin.ja.md",
    TIMESERIES_DIR / "Rolling_Statistics.ja.md",
    TIMESERIES_DIR / "SARIMAX_Exogenous.ja.md",
    TIMESERIES_DIR / "Seasonal_Polar.ja.md",
    TIMESERIES_DIR / "STL_Forecast.ja.md",
    TIMESERIES_DIR / "Time_Series_Overview.ja.md",
    TIMESERIES_DIR / "Train_Test_Split.ja.md",
    TIMESERIES_DIR / "UCM_Model.ja.md",
    TIMESERIES_DIR / "VAR_Forecast.ja.md",
    TIMESERIES_DIR / "Weekday_Averages.ja.md",
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
    for relative_path in TARGET_MARKDOWN_TS:
        md_path = repo_root / relative_path
        if not md_path.exists():
            print(f"Skipping missing file: {relative_path.as_posix()}")
            continue

        code_blocks = extract_python_blocks(md_path.read_text(encoding="utf-8"))

        for idx, block in enumerate(code_blocks, start=1):
            japanize_matplotlib.japanize()
            try:
                run_snippet(block, md_path)
                print(f"Executed block {idx} from {relative_path.as_posix()}")
            except:
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(parser.parse_args())
