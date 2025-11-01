---
title: "ロバスト回帰 | Huber損失で外れ値に強くする"
linkTitle: "ロバスト回帰"
seo_title: "ロバスト回帰 | Huber損失で外れ値に強くする"
pre: "2.1.3 "
weight: 3
title_suffix: "Huber損失で外れ値に強くする"
---

{{% summary %}}
- 最小二乗法は外れ値の影響を強く受けるため、観測ミスが混ざると推定が大きく歪みやすい。
- Huber損失は小さな誤差には二乗誤差、大きな誤差には線形誤差を適用し、外れ値の影響を自動的に抑える。
- しきい値 \\(\delta\\) と L2 正則化 \\(\alpha\\) を調整することで、外れ値への頑健性とバイアスのバランスを取れる。
- 特徴量のスケーリングと交差検証を組み合わせると、現実のデータでも安定した学習が行える。
{{% /summary %}}

## 直感
外れ値はセンサー異常や入力ミス、分布の変化などで発生し、最小二乗法では二乗誤差が極端に大きくなるため推定全体が引きずられてしまいます。ロバスト回帰は大きな残差に対して緩やかな罰則を与え、典型的なデータ点は従来どおり扱うことで、支配的な傾向を保ちながら頑健性を確保します。Huber損失はこの目的に適した代表的な損失関数で、ゼロ付近では二乗損失、遠方では絶対値損失として振る舞います。

## 具体的な数式
残差 \\(r = y - \hat{y}\\) とし、しきい値 \\(\delta > 0\\) を用いると Huber損失は

$$
\ell_\delta(r) =
\begin{cases}
\dfrac{1}{2} r^2, & |r| \le \delta, \\\\
\delta \left(|r| - \dfrac{1}{2}\delta \right), & |r| > \delta.
\end{cases}
$$

小さな残差では通常の二乗誤差と同じ挙動、大きな残差では線形に増加するため外れ値による極端な影響を抑えられます。導関数（影響関数）も折れ線状になり、外れ値の寄与が自動的に頭打ちになります。

## Pythonを用いた実験や説明
Huber損失の形状と、外れ値を含むデータに対する回帰結果を `scikit-learn` で確認します。

```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

### Huber損失と関連損失の比較

```python
def huber_loss(r: np.ndarray, delta: float = 1.5):
    half_sq = 0.5 * np.square(r)
    lin = delta * (np.abs(r) - 0.5 * delta)
    return np.where(np.abs(r) <= delta, half_sq, lin)

delta = 1.5
r_vals = np.arange(-2, 2, 0.01)
h_vals = huber_loss(r_vals, delta=delta)

plt.figure(figsize=(8, 6))
plt.plot(r_vals, np.square(r_vals), "red", label=r"二乗損失 $r^2$")
plt.plot(r_vals, np.abs(r_vals), "orange", label=r"絶対値損失 $|r|$")
plt.plot(r_vals, h_vals, "green", label=fr"Huber損失 ($\delta={delta}$)")
plt.axhline(0, color="k", linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlabel("残差 $r$")
plt.ylabel("損失値")
plt.title("二乗・絶対値・Huber損失の比較")
plt.show()
```

### 外れ値を含むデータでの挙動

```python
np.random.seed(42)

N = 30
x1 = np.arange(N)
x2 = np.arange(N)
X = np.c_[x1, x2]
epsilon = np.random.rand(N)
y = 5 * x1 + 10 * x2 + epsilon * 10

y[5] = 500  # 外れ値を1点だけ挿入

plt.figure(figsize=(8, 6))
plt.plot(x1, y, "ko", label="data")
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("外れ値を含むデータ")
plt.show()
```

```python
from sklearn.linear_model import HuberRegressor, Ridge, LinearRegression

plt.figure(figsize=(8, 6))

huber = HuberRegressor(alpha=0.0, epsilon=3.0)
huber.fit(X, y)
plt.plot(x1, huber.predict(X), "green", label="Huber回帰")

ridge = Ridge(alpha=1.0, random_state=0)
ridge.fit(X, y)
plt.plot(x1, ridge.predict(X), "orange", label="リッジ回帰 (α=1.0)")

ols = LinearRegression()
ols.fit(X, y)
plt.plot(x1, ols.predict(X), "r-", label="最小二乗法 (OLS)")

plt.plot(x1, y, "kx", alpha=0.7)
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("外れ値を含むときの推定曲線の違い")
plt.grid(alpha=0.3)
plt.show()
```

### 実行結果の読み方
- Huber損失は小さな残差では最小二乗法と同じ挙動、大きな残差では絶対値損失に近い挙動を示す。
- 外れ値があっても、Huber回帰はリッジ回帰や通常の線形回帰よりも影響を受けにくい。
- ハイパーパラメータ `epsilon` と `alpha` を調整し、交差検証で最適値を探すと安定したモデルになる。

## 参考文献
{{% references %}}
<li>Huber, P. J. (1964). Robust Estimation of a Location Parameter. <i>The Annals of Mathematical Statistics</i>, 35(1), 73–101.</li>
<li>Hampel, F. R. et al. (1986). <i>Robust Statistics: The Approach Based on Influence Functions</i>. Wiley.</li>
<li>Huber, P. J., &amp; Ronchetti, E. M. (2009). <i>Robust Statistics</i> (2nd ed.). Wiley.</li>
{{% /references %}}
