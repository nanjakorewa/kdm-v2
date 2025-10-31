"""Utility script inventory for the `scripts/` directory.

Each entry explains what the helper does, when to run it, the important inputs
and outputs, and notable implementation details that affect usage.
"""

SCRIPT_DETAILS: dict[str, str] = {
    "escape_inline_latex.py": """
    Purpose
        Normalize inline LaTeX delimiters so Markdown processors do not strip
        the escaping backslash.

    How it works
        * Walks files or directories provided on the command line.
        * Limits processing to user-supplied extensions (default: .md).
        * Replaces every single-escaped `\\(` / `\\)` with double-escaped forms
          `\\\\(` / `\\\\)`, leaving already double-escaped text untouched.
        * Offers `--dry-run` to preview changes without modifying files.

    Typical usage
        python scripts/escape_inline_latex.py docs/ --ext .md --ext .qmd

    Notes
        The regex guards against over-escaping sequences that are already
        doubled or intentionally raw. Errors surface immediately with the file
        path that failed.
    """,
    "generate_basic_assets.py": """
    Purpose
        Execute Python code blocks embedded in Markdown under `content/basic`
        and capture the Matplotlib figures they produce as SVG assets stored in
        `static/images/basic/...`.

    How it works
        * Uses the Matplotlib Agg backend and a shared house style
          (`scripts/k_dm.mplstyle`).
        * Ensures Japanese text renders by delegating to japanize_matplotlib
          when available; otherwise applies a curated font fallback list.
        * Provides a stub implementation of several scikit-learn APIs
          (`LinearRegression`, `StandardScaler`, `PolynomialFeatures`,
          `make_pipeline`, `mean_squared_error`) so snippets run even if
          scikit-learn is not installed.
        * Scans Markdown recursively, orders multilingual variants with a
          language priority, and executes fenced ```python``` blocks one by one.
        * Captures any new figures opened during a block, saves them to PNG
          files named `<slug>_blockXX[_figYY]_lang.png` (例:
          `_ja`, `_en`, `_es`, `_id`) under `static/images/basic/...`, then
          closes the figures.

    Typical usage
        python scripts/generate_basic_assets.py

    Notes
        Each Markdown file shares a persistent namespace across its blocks, so
        earlier snippets can define helpers reused later in the same file. Any
        execution error is logged and that block is skipped.
    """,
    "generate_timeseries_assets.py": """
    Purpose
        Run preselected Markdown notebooks in `content/timeseries` and execute
        every ```python``` block so that the code can generate the necessary
        plots or side effects (e.g., saving images referenced by the chapter).

    How it works
        * Targets a curated list of `.ja.md` files covering the time-series
          chapter.
        * Uses the Agg backend, applies the shared Matplotlib style, and calls
          `japanize_matplotlib.japanize()` before each block to ensure CJK
          fonts load.
        * Executes blocks in isolated namespaces; after each block it closes
          all open figures so subsequent blocks start fresh.
        * Silently skips blocks that raise exceptions (they simply continue to
          the next block) but reports successful execution.

    Typical usage
        python scripts/generate_timeseries_assets.py

    Notes
        This helper does not itself save figures; the Markdown snippets are
        responsible for writing outputs (e.g., via `plt.savefig`). Missing
        Markdown files are logged and skipped.
    """,
    "generate_visualize_assets.py": """
    Purpose
        Batch-execute visualization examples under `content/visualize/**` so the
        Python code embedded in the documentation runs and produces its assets.

    How it works
        * Maintains an explicit whitelist of Markdown files across visualization
          categories (bar, distribution, line, scatter, advanced).
        * Uses the Agg backend, shared Matplotlib style, and applies
          japanize_matplotlib before every block to support Japanese labels.
        * Extracts ```python``` fenced blocks, executes them in a fresh namespace,
          and closes figures immediately afterward.
        * Reports each successfully executed block; on error it swallows the
          exception and continues so one failure does not halt the batch.

    Typical usage
        python scripts/generate_visualize_assets.py

    Notes
        Output management (saving figures) is expected to occur inside the
        Markdown code. Duplicate import lines in the script are harmless but
        retained from the original implementation.
    """,
}

# The module is intended as human-readable documentation. Importing it gives
# access to SCRIPT_DETAILS for programmatic inspection if desired.
