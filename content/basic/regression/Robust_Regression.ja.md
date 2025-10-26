---
title: "外れ値と頑健性"
pre: "2.1.3 "
weight: 3
title_suffix: "に対応できるHuber回帰について解説！"
---

{{% youtube "CrN5Si0379g" %}}


<div class="pagetop-box">
  <p><b>外れ値</b>とは、他の多くのデータ点から大きく外れた値（極端に大きい・小さいなど）の総称です。何が外れ値かは、問題設定・データの分布・目的に依存します。</p>
  <p>このページでは、外れ値のあるデータに対して「二乗誤差（最小二乗法）」で回帰した場合と「Huber損失」を用いた回帰の違いを、数式とコードで確認します。</p>
</div>

```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

---

## 1. なぜ最小二乗法は外れ値に弱いのか

最小二乗法（通常の線形回帰）は、残差の二乗和
$$
\text{RSS} = \sum_{i=1}^n \big(y_i - \hat{y}_i\big)^2
$$
を最小化します。残差を<b>二乗</b>するため、わずかな外れ値でも損失が急激に大きくなり、<b>直線（モデル）が外れ値の方向へ強く引っ張られる</b>という問題が起きます。

---

## 2. Huber損失：二乗と絶対値の「いいとこ取り」

**Huber損失**は、残差が小さいときは二乗、大きいときは絶対値で扱う損失です。  
残差 \\(r = y - \hat{y}\\) とし、しきい値を \\(\delta > 0\\) とすると

$$
\ell_\delta(r) =
\begin{cases}
\dfrac{1}{2}r^2, & |r| \le \delta \\
\delta\left(|r| - \dfrac{1}{2}\delta\right), & |r| > \delta
\end{cases}
$$

- **小さな誤差**は二乗で滑らかに最適化（最小二乗の良さを活かす）  
- **大きな誤差**は絶対値相当で抑制（外れ値の影響を弱める）

勾配（影響度）は
$$
\psi_\delta(r) = \frac{d}{dr}\ell_\delta(r) =
\begin{cases}
r, & |r|\le \delta \\
\delta\,\mathrm{sign}(r), & |r|>\delta
\end{cases}
$$
となり、外れ値に対して<b>勾配がクリップ</b>されるのがポイントです。

> 用語メモ：  
> scikit-learn の `HuberRegressor` では、このしきい値をパラメータ `epsilon` で指定します（上式の \\(\delta\\) に対応）。

---

## 3. Huber損失の形を可視化する

下のコードは、二乗誤差・絶対誤差・Huber損失を一緒に描いて比較します。

```python
def huber_loss(r: np.ndarray, delta: float = 1.5):
    half_sq = 0.5 * np.square(r)
    lin = delta * (np.abs(r) - 0.5 * delta)
    return np.where(np.abs(r) <= delta, half_sq, lin)

delta = 1.5
r_vals = np.arange(-2, 2, 0.01)
h_vals = huber_loss(r_vals, delta=delta)

plt.figure(figsize=(8, 6))
plt.plot(r_vals, np.square(r_vals), "red",   label=r"二乗誤差 $r^2$")
plt.plot(r_vals, np.abs(r_vals),    "orange",label=r"絶対誤差 $|r|$")
plt.plot(r_vals, h_vals,            "green", label=fr"Huber損失（$\delta={delta}$）")
plt.axhline(0, color="k", linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlabel("残差 $r$")
plt.ylabel("損失")
plt.title("二乗・絶対・Huber損失の比較")
plt.show()
```

![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_5_0.png)

---

## 4. 外れ値があると何が起こるか（データ作成）

単純な 2 変数（\\(x_1, x_2\\)）の線形モデルに、意図的に<b>1点だけ非常に大きい外れ値</b>を混ぜます。

```python
np.random.seed(42)

N = 30
x1 = np.arange(N)
x2 = np.arange(N)
X = np.c_[x1, x2]                      # 形状 (N, 2)
epsilon = np.random.rand(N)            # 0~1 の雑音
y = 5 * x1 + 10 * x2 + epsilon * 10    # 真の関係 + 雑音

y[5] = 500  # 1点だけ極端に大きい外れ値

plt.figure(figsize=(8, 6))
plt.plot(x1, y, "ko", label="data")
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("外れ値を1点含むデータ")
plt.show()
```

![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_7_0.png)

---

## 5. 最小二乗法 vs. Ridge vs. Huber を比較

- **最小二乗法（OLS）**：二乗誤差 → 外れ値に弱い  
- **Ridge（L2正則化）**：係数を縮める → 少し安定するが、外れ値の影響は残る  
- **Huber 回帰**：外れ値の影響をクリップ → 直線が外れ値に引っ張られにくい

{{% notice document %}}
- [sklearn.linear_model.HuberRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor)
{{% /notice %}}

{{% notice seealso %}}
[前処理の方法：外れ値にラベルを付与①](https://k-dm.work/ja/prep/numerical/add_label_to_anomaly/)
{{% /notice %}}

```python
from sklearn.linear_model import HuberRegressor, Ridge, LinearRegression

plt.figure(figsize=(8, 6))

# Huber回帰（epsilon=3 で外れ値の影響を抑制）
huber = HuberRegressor(alpha=0.0, epsilon=3.0)
huber.fit(X, y)
plt.plot(x1, huber.predict(X), "green", label="Huber回帰")

# Ridge（L2正則化）。alpha を 0 にすると OLS と実質同じになるので注意
ridge = Ridge(alpha=1.0, random_state=0)
ridge.fit(X, y)
plt.plot(x1, ridge.predict(X), "orange", label="リッジ回帰（α=1.0）")

# OLS
lr = LinearRegression()
lr.fit(X, y)
plt.plot(x1, lr.predict(X), "r-", label="最小二乗法（OLS）")

# 元データ
plt.plot(x1, y, "kx", alpha=0.7)

plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("外れ値があるときの各回帰直線の違い")
plt.grid(alpha=0.3)
plt.show()
```

![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_9_0.png)

**読み取り方**：  
- OLS（赤）は外れ値に強く引っ張られています。  
- Ridge（橙）は少し緩和しますが、依然として影響を受けます。  
- Huber（緑）は、外れ値の影響が抑えられ、データ全体の傾向をより素直に捉えています。

---

## 6. パラメータのコツ（epsilon と alpha）

- `epsilon`（= \\(\delta\\)）  
  - しきい値を大きくすると<b>OLS寄り</b>に、小さくすると<b>絶対値損失寄り</b>に。  
  - 目安は残差のスケールに依存します。スケールが大きく変わる場合は標準化やロバストなスケール推定を併用してください。  
- `alpha`（L2 ペナルティ）  
  - 係数の暴れを抑える効果。相関が強い特徴や少データでは安定化に有効。

**epsilon の感度を見る例**：

```python
from sklearn.metrics import mean_squared_error

for eps in [1.2, 1.5, 2.0, 3.0]:
    h = HuberRegressor(alpha=0.0, epsilon=eps).fit(X, y)
    mse = mean_squared_error(y, h.predict(X))
    print(f"epsilon={eps:>3}: MSE={mse:.3f}")
```

---

## 7. 実務での注意点

- **スケーリング**：特徴量・目的変数のスケールが大きく異なると、`epsilon` の意味合いが変わります。標準化やロバストスケール推定を検討。  
- **レバレッジ点には弱い**：Huber は主に \\(y\\) 側の外れ値（垂直方向）に頑健です。<b>\\(X\\) 側の強い外れ値（レバレッジ点）</b>には依然として脆弱なことに注意。  
- **閾値の選び方**：`GridSearchCV` で `epsilon` と `alpha` を同時に探索するのが安全。  
- **モデル比較はCVで**：訓練データの当てはまりだけで判断せず、交差検証で汎化性能を評価しましょう。

---

## 8. まとめ

- OLS は外れ値に弱く直線が引っ張られやすい。  
- Huber 損失は「小さな誤差＝二乗」「大きな誤差＝絶対値」で、外れ値の影響を<b>勾配クリップ</b>的に抑える。  
- `epsilon` と `alpha` のチューニングで、頑健性と当てはまりのバランスを取れる。  
- レバレッジ点には注意。必要なら検出・可視化・前処理（例：外れ値ラベル付与）と併用。

---
