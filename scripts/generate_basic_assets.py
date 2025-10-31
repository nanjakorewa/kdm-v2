"""Generate basic chapter assets by executing code blocks from Markdown.

Run:
    python scripts/generate_basic_assets.py
"""

from __future__ import annotations

import argparse
import importlib
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
    try:
        importlib.import_module("sklearn")
        return
    except ModuleNotFoundError:
        pass

    sklearn_module = types.ModuleType("sklearn")
    sklearn_module.__path__ = []  # type: ignore[attr-defined]
    linear_model_module = types.ModuleType("sklearn.linear_model")
    preprocessing_module = types.ModuleType("sklearn.preprocessing")
    datasets_module = types.ModuleType("sklearn.datasets")
    pipeline_module = types.ModuleType("sklearn.pipeline")
    metrics_module = types.ModuleType("sklearn.metrics")
    tree_module = types.ModuleType("sklearn.tree")
    neighbors_module = types.ModuleType("sklearn.neighbors")

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

        def score(self, X: np.ndarray, y: np.ndarray) -> float:
            predictions = self.predict(X)
            y_arr = np.asarray(y)
            if predictions.shape != y_arr.shape:
                y_arr = y_arr.reshape(predictions.shape)
            return float(np.mean(predictions == y_arr))

    def make_pipeline(*steps: object) -> SimplePipeline:
        return SimplePipeline(list(steps))

    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        return float(np.mean((y_true_arr - y_pred_arr) ** 2))

    def make_regression(
        *,
        n_samples: int = 100,
        n_features: int = 1,
        noise: float = 0.0,
        random_state: int | None = None,
        effective_rank: int | None = None,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)
        X = rng.normal(size=(n_samples, n_features))
        coef = rng.normal(size=n_features)
        y = X @ coef
        if effective_rank:
            y += 0.1 * rng.normal(size=n_samples)
        if noise:
            y += rng.normal(scale=noise, size=n_samples)
        return X, y

    def make_blobs(
        *,
        n_samples: int = 100,
        centers: int | list[tuple[float, ...]] | np.ndarray = 3,
        cluster_std: float | list[float] | tuple[float, ...] | np.ndarray = 1.0,
        random_state: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)
        if isinstance(centers, int):
            centers_array = rng.uniform(-5.0, 5.0, size=(centers, 2))
        else:
            centers_array = np.asarray(centers, dtype=float)
        centers_array = centers_array.reshape(centers_array.shape[0], -1)
        n_features = centers_array.shape[1]

        if isinstance(cluster_std, (list, tuple, np.ndarray)):
            std_array = np.asarray(cluster_std, dtype=float)
            if std_array.size == 1:
                std_array = np.full(centers_array.shape[0], float(std_array[0]))
            elif std_array.size != centers_array.shape[0]:
                std_array = np.resize(std_array, centers_array.shape[0])
        else:
            std_array = np.full(centers_array.shape[0], float(cluster_std))

        counts = np.full(centers_array.shape[0], n_samples // centers_array.shape[0])
        counts[: n_samples % centers_array.shape[0]] += 1

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for idx, (center, std, count) in enumerate(zip(centers_array, std_array, counts)):
            if count <= 0:
                continue
            cov = np.eye(n_features) * float(std) ** 2
            samples = rng.multivariate_normal(center, cov, size=int(count))
            X_parts.append(samples)
            y_parts.append(np.full(int(count), idx))

        X_all = np.vstack(X_parts)
        y_all = np.concatenate(y_parts)
        return X_all, y_all

    class DecisionTreeRegressor:
        def __init__(self, **params: object) -> None:
            self.params = params
            self._is_fitted = False
            self.n_features_in_: int | None = None
            self.feature_importances_: np.ndarray | None = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
            X_arr = np.asarray(X, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, np.newaxis]
            if y_arr.ndim > 1:
                y_arr = np.mean(y_arr, axis=1)
            self.n_features_in_ = X_arr.shape[1]
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)
            self._target_mean = float(np.mean(y_arr))
            self._target_std = float(np.std(y_arr)) if y_arr.size else 0.0
            self._is_fitted = True
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            if not self._is_fitted:
                raise RuntimeError("DecisionTreeRegressor is not fitted yet.")
            X_arr = np.asarray(X, dtype=float)
            n_samples = X_arr.shape[0]
            return np.full(n_samples, getattr(self, "_target_mean", 0.0), dtype=float)

    class KNeighborsClassifier:
        def __init__(self, n_neighbors: int = 5, *, weights: str = "uniform") -> None:
            self.n_neighbors = max(1, int(n_neighbors))
            self.weights = weights
            self._X: np.ndarray | None = None
            self._y: np.ndarray | None = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> "KNeighborsClassifier":
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, np.newaxis]
            y_arr = np.asarray(y)
            if X_arr.shape[0] != y_arr.shape[0]:
                raise ValueError("X and y have inconsistent lengths.")
            self._X = X_arr
            self._y = y_arr
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            if self._X is None or self._y is None:
                raise RuntimeError("KNeighborsClassifier is not fitted yet.")
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, np.newaxis]

            train_X = self._X
            if train_X.ndim == 1:
                train_X = train_X[:, np.newaxis]

            distances = np.linalg.norm(train_X[None, :, :] - X_arr[:, None, :], axis=2)
            predictions: list[object] = []
            neighbor_count = min(self.n_neighbors, train_X.shape[0])

            for row_idx in range(distances.shape[0]):
                row = distances[row_idx]
                neighbor_idx = np.argsort(row)[:neighbor_count]
                neighbor_labels = self._y[neighbor_idx]
                if self.weights == "distance":
                    neighbor_distances = row[neighbor_idx]
                    weights = 1.0 / np.maximum(neighbor_distances, 1e-12)
                else:
                    weights = np.ones_like(neighbor_labels, dtype=float)
                class_votes: dict[object, float] = {}
                for label, weight in zip(neighbor_labels, weights):
                    class_votes[label] = class_votes.get(label, 0.0) + float(weight)
                best_label = max(class_votes.items(), key=lambda item: (item[1], item[0]))[0]
                predictions.append(best_label)

            return np.asarray(predictions, dtype=self._y.dtype)

        def score(self, X: np.ndarray, y: np.ndarray) -> float:
            preds = self.predict(X)
            y_arr = np.asarray(y)
            if preds.shape != y_arr.shape:
                y_arr = y_arr.reshape(preds.shape)
            return float(np.mean(preds == y_arr))

    linear_model_module.LinearRegression = LinearRegression
    preprocessing_module.StandardScaler = StandardScaler
    preprocessing_module.PolynomialFeatures = PolynomialFeatures
    pipeline_module.make_pipeline = make_pipeline
    metrics_module.mean_squared_error = mean_squared_error
    datasets_module.make_regression = make_regression
    datasets_module.make_blobs = make_blobs
    tree_module.DecisionTreeRegressor = DecisionTreeRegressor
    neighbors_module.KNeighborsClassifier = KNeighborsClassifier

    sys.modules["sklearn"] = sklearn_module
    sklearn_module.linear_model = linear_model_module  # type: ignore[attr-defined]
    sklearn_module.preprocessing = preprocessing_module  # type: ignore[attr-defined]
    sklearn_module.datasets = datasets_module  # type: ignore[attr-defined]
    sklearn_module.pipeline = pipeline_module  # type: ignore[attr-defined]
    sklearn_module.metrics = metrics_module  # type: ignore[attr-defined]
    sklearn_module.tree = tree_module  # type: ignore[attr-defined]
    sklearn_module.neighbors = neighbors_module  # type: ignore[attr-defined]

    sys.modules["sklearn.linear_model"] = linear_model_module
    sys.modules["sklearn.preprocessing"] = preprocessing_module
    sys.modules["sklearn.datasets"] = datasets_module
    sys.modules["sklearn.pipeline"] = pipeline_module
    sys.modules["sklearn.metrics"] = metrics_module
    sys.modules["sklearn.tree"] = tree_module
    sys.modules["sklearn.neighbors"] = neighbors_module

    if "dtreeviz" not in sys.modules:
        dtreeviz_module = types.ModuleType("dtreeviz")
        dtreeviz_trees_module = types.ModuleType("dtreeviz.trees")

        def _rtreeviz_bivar_3d(
            estimator: object,
            X: np.ndarray,
            y: np.ndarray,
            *,
            ax: object | None = None,
            feature_names: list[str] | None = None,
            target_name: str = "",
            **__: object,
        ) -> object:
            import matplotlib.pyplot as _plt  # noqa: WPS433

            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, np.newaxis]
            if X_arr.shape[1] == 1:
                X_arr = np.column_stack([X_arr, np.zeros_like(X_arr[:, 0])])
            y_arr = np.asarray(y, dtype=float).reshape(-1)

            if ax is None:
                fig = _plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(
                X_arr[:, 0],
                X_arr[:, 1],
                y_arr,
                c=y_arr,
                cmap="viridis",
                alpha=0.6,
            )
            ax.set_xlabel((feature_names or ["x1", "x2"])[0])
            ax.set_ylabel((feature_names or ["x1", "x2"])[1])
            ax.set_zlabel(target_name or "y")
            ax.set_title(target_name or "Decision Tree")
            if hasattr(ax, "figure"):
                ax.figure.colorbar(scatter, ax=ax, shrink=0.6)
            return ax

        def _dtreeviz(*_: object, **__: object) -> None:
            return None

        dtreeviz_trees_module.dtreeviz = _dtreeviz  # type: ignore[attr-defined]
        dtreeviz_trees_module.rtreeviz_bivar_3D = _rtreeviz_bivar_3d  # type: ignore[attr-defined]
        dtreeviz_module.trees = dtreeviz_trees_module  # type: ignore[attr-defined]

        sys.modules["dtreeviz"] = dtreeviz_module
        sys.modules["dtreeviz.trees"] = dtreeviz_trees_module

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
    np.random.seed(777)
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
