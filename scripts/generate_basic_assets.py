"""Generate basic chapter assets by executing code blocks from Markdown.

Run:
    python scripts/generate_basic_assets.py
"""

from __future__ import annotations

import argparse
import re
import sys
import types
from pathlib import Path
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from itertools import combinations_with_replacement

try:
    import japanize_matplotlib as jam
except ImportError:  # pragma: no cover - optional dependency
    jam = None

plt.style.use("scripts/k_dm.mplstyle")
plt.rcParams["axes.unicode_minus"] = False


def _noop_show(*_: object, **__: object) -> None:
    """Matplotlib show() stub to keep figures open under Agg backend."""


plt.show = _noop_show  # type: ignore[assignment]

if jam is None:
    jam_stub = types.ModuleType("japanize_matplotlib")

    def _noop(*_: object, **__: object) -> None:
        """Fallback japanize that does nothing."""

    jam_stub.japanize = _noop  # type: ignore[attr-defined]
    sys.modules.setdefault("japanize_matplotlib", jam_stub)
else:
    sys.modules.setdefault("japanize_matplotlib", jam)


def apply_font_settings() -> None:
    if jam is not None:
        jam.japanize()
    else:
        plt.rcParams["font.family"] = [
            "IPAexGothic",
            "MS Gothic",
            "Yu Gothic",
            "Meiryo",
            "DejaVu Sans",
            "sans-serif",
        ]
    plt.rcParams["axes.unicode_minus"] = False


def install_sklearn_stub() -> None:
    if "sklearn" in sys.modules:
        return

    sklearn_module = types.ModuleType("sklearn")
    sklearn_module.__path__ = []  # type: ignore[attr-defined]
    linear_model_module = types.ModuleType("sklearn.linear_model")
    preprocessing_module = types.ModuleType("sklearn.preprocessing")
    pipeline_module = types.ModuleType("sklearn.pipeline")
    metrics_module = types.ModuleType("sklearn.metrics")

    class LinearRegression:
        def __init__(self) -> None:
            self.coef_: np.ndarray | None = None
            self.intercept_: float = 0.0

        def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
            X_arr = np.asarray(X, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, np.newaxis]
            ones = np.ones((X_arr.shape[0], 1), dtype=float)
            design = np.hstack([X_arr, ones])
            beta, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
            self.coef_ = beta[:-1]
            self.intercept_ = float(beta[-1])
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            if self.coef_ is None:
                raise RuntimeError("LinearRegression is not fitted yet.")
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, np.newaxis]
            return X_arr @ self.coef_ + self.intercept_

    class StandardScaler:
        def __init__(self, *, with_mean: bool = True, with_std: bool = True) -> None:
            self.with_mean = with_mean
            self.with_std = with_std
            self.mean_: np.ndarray | None = None
            self.scale_: np.ndarray | None = None

        def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "StandardScaler":
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, np.newaxis]
            self.mean_ = X_arr.mean(axis=0) if self.with_mean else np.zeros(X_arr.shape[1])
            if self.with_std:
                scale = X_arr.std(axis=0, ddof=0)
                scale[scale == 0] = 1.0
                self.scale_ = scale
            else:
                self.scale_ = np.ones(X_arr.shape[1])
            return self

        def transform(self, X: np.ndarray) -> np.ndarray:
            if self.mean_ is None or self.scale_ is None:
                raise RuntimeError("StandardScaler is not fitted yet.")
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, np.newaxis]
            return (X_arr - (self.mean_ if self.with_mean else 0)) / self.scale_

        def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
            return self.fit(X, y).transform(X)

    class PolynomialFeatures:
        def __init__(self, degree: int = 2, include_bias: bool = True) -> None:
            self.degree = degree
            self.include_bias = include_bias
            self.n_features_in_: int | None = None
            self._combinations: list[tuple[int, ...]] = []

        def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PolynomialFeatures":
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, np.newaxis]
            self.n_features_in_ = X_arr.shape[1]
            combos: list[tuple[int, ...]] = []
            for degree in range(1, self.degree + 1):
                combos.extend(combinations_with_replacement(range(self.n_features_in_), degree))
            self._combinations = combos
            return self

        def transform(self, X: np.ndarray) -> np.ndarray:
            if self.n_features_in_ is None:
                raise RuntimeError("PolynomialFeatures is not fitted yet.")
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, np.newaxis]
            features: list[np.ndarray] = []
            if self.include_bias:
                features.append(np.ones(X_arr.shape[0]))
            for combo in self._combinations:
                features.append(np.prod(X_arr[:, combo], axis=1))
            return np.vstack(features).T

        def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
            return self.fit(X, y).transform(X)

    class SimplePipeline:
        def __init__(self, steps: list[object]) -> None:
            self.steps = steps

        def fit(self, X: np.ndarray, y: np.ndarray) -> "SimplePipeline":
            Xt = X
            for step in self.steps[:-1]:
                if hasattr(step, "fit_transform"):
                    Xt = step.fit_transform(Xt, y)
                else:
                    Xt = step.fit(Xt, y).transform(Xt)  # type: ignore[attr-defined]
            final = self.steps[-1]
            if hasattr(final, "fit"):
                final.fit(Xt, y)  # type: ignore[arg-type]
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            Xt = X
            for step in self.steps[:-1]:
                if hasattr(step, "transform"):
                    Xt = step.transform(Xt)  # type: ignore[attr-defined]
            final = self.steps[-1]
            if hasattr(final, "predict"):
                return final.predict(Xt)  # type: ignore[return-value]
            raise RuntimeError("Final pipeline step does not implement predict().")

    def make_pipeline(*steps: object) -> SimplePipeline:
        return SimplePipeline(list(steps))

    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        return float(np.mean((y_true_arr - y_pred_arr) ** 2))

    linear_model_module.LinearRegression = LinearRegression
    preprocessing_module.StandardScaler = StandardScaler
    preprocessing_module.PolynomialFeatures = PolynomialFeatures
    pipeline_module.make_pipeline = make_pipeline
    metrics_module.mean_squared_error = mean_squared_error

    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.linear_model"] = linear_model_module
    sys.modules["sklearn.preprocessing"] = preprocessing_module
    sys.modules["sklearn.pipeline"] = pipeline_module
    sys.modules["sklearn.metrics"] = metrics_module

CODE_BLOCK_PATTERN = re.compile(r"```python\s*(.*?)```", re.DOTALL)


LANG_SUFFIXES = (".ja", ".en", ".es", ".id")
LANG_PRIORITY = {"ja": 0, "es": 1, "id": 2, "en": 3}


def slugify(name: str) -> str:
    for suffix in LANG_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    slug = name.replace(".md", "")
    slug = slug.replace("_", "-").replace(" ", "-").lower()
    slug = re.sub(r"[^a-z0-9\-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "figure"


def extract_python_blocks(text: str) -> list[str]:
    return [dedent(match.group(1)).strip() for match in CODE_BLOCK_PATTERN.finditer(text)]


def detect_language(md_path: Path) -> str:
    parts = md_path.name.split(".")
    if len(parts) >= 3 and parts[-2] in LANG_PRIORITY:
        return parts[-2]
    return ""


def iter_targets(
    content_root: Path,
) -> list[tuple[Path, list[str], list[str], str, str]]:
    unsorted_targets: list[
        tuple[tuple[str, ...], int, Path, list[str], list[str], str, str]
    ] = []
    for md_path in sorted(content_root.rglob("*.md")):
        if md_path.suffix != ".md":
            continue
        if md_path.name.endswith(".bak"):
            continue
        text = md_path.read_text(encoding="utf-8")
        blocks = extract_python_blocks(text)
        if not blocks:
            continue
        rel_parts = tuple(md_path.relative_to(content_root).parts[:-1])
        rel_slugs = [slugify(part) for part in rel_parts]
        file_slug = slugify(md_path.stem)
        lang = detect_language(md_path)
        base_key = tuple(rel_slugs + [file_slug])
        priority = LANG_PRIORITY.get(lang, len(LANG_PRIORITY))
        unsorted_targets.append(
            (base_key, priority, md_path, blocks, rel_slugs, file_slug, lang)
        )

    unsorted_targets.sort(key=lambda item: (item[0], item[1], str(item[2])))
    return [
        (md_path, blocks, rel_slugs, file_slug, lang)
        for _, _, md_path, blocks, rel_slugs, file_slug, lang in unsorted_targets
    ]


def save_figures(
    figures: list[int],
    repo_root: Path,
    rel_slugs: list[str],
    file_slug: str,
    block_idx: int,
    lang: str,
) -> None:
    output_dir = repo_root / "static" / "images" / "basic"
    if rel_slugs:
        output_dir = output_dir.joinpath(*rel_slugs)
    output_dir.mkdir(parents=True, exist_ok=True)

    lang_suffix_map = {"ja": "_ja", "en": "_en", "es": "_es", "id": "_id"}
    lang_suffix = lang_suffix_map.get(lang or "ja", "_ja")

    for fig_idx, fig_num in enumerate(figures, start=1):
        fig = plt.figure(fig_num)
        suffix = "" if fig_idx == 1 else f"_fig{fig_idx:02d}"
        filename = f"{file_slug}_block{block_idx:02d}{suffix}{lang_suffix}.png"
        output_path = output_dir / filename
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved {output_path.relative_to(repo_root)}")
        plt.close(fig)


def main(_: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    content_root = repo_root / "content" / "basic"

    install_sklearn_stub()

    for md_path, blocks, rel_slugs, file_slug, lang in iter_targets(content_root):
        print(f"Processing {md_path.relative_to(repo_root)}")
        namespace: dict[str, object] = {}
        for block_idx, code in enumerate(blocks, start=1):
            existing = set(plt.get_fignums())
            apply_font_settings()
            namespace.setdefault("np", np)
            namespace.setdefault("plt", plt)
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

            save_figures(new_figs, repo_root, rel_slugs, file_slug, block_idx, lang)

        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(parser.parse_args())
