---
title: "勾配ブースティング（基礎）"
pre: "2.4.5 "
weight: 5
title_suffix: "の直感・数式・実装"
---

{{< katex />}}
{{% youtube "ZgssfFWQbZ8" %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.ensemble import GradientBoostingRegressor
```

# 勾配ブースティング（基礎）

<div class="pagetop-box">
  <p><b>勾配ブースティング（Gradient Boosting）</b>は、AdaBoost を一般化した手法で、「損失関数を最小化するための勾配情報」を利用して弱学習器を逐次追加するアンサンブル学習です。</p>
  <p>柔軟に損失関数を選べるため、回帰・分類・分位点予測など幅広い問題に適用できます。</p>
</div>

---

## 1. 直感：残差を埋めていくモデル

1. まずは「単純な予測器」からスタート（平均値など）。  
2. 残差（予測と実際の差）を次のモデルで学習。  
3. 新しいモデルを加えて予測を改善。  
4. これを繰り返し → 「残差をどんどん埋めていく」。  

> AdaBoost が「誤分類した点に重みをかけ直す」のに対し、勾配ブースティングは **「損失関数の勾配（擬似残差）」に基づいて補正する」** という発想です。

---

## 2. 数式で理解する勾配ブースティング

目的関数：
$$
L = \sum_{i=1}^n \ell\big(y_i, F(x_i)\big)
$$
を最小化したい。ここで $F(x)$ が最終予測関数。

### アルゴリズム（回帰の場合）
1. **初期モデル**
$$
F_0(x) = \arg\min_c \sum_i \ell(y_i, c)
$$
（例：二乗誤差なら平均値）

2. **反復 $m=1,\dots,M$**
   - 勾配（擬似残差）を計算：
     $$
     r_{im} = - \left[\frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}
     $$
   - $r_{im}$ を目的変数として弱学習器 $h_m(x)$ を学習
   - 最適なステップ幅を計算：
     $$
     \rho_m = \arg\min_\rho \sum_i \ell\big(y_i, F_{m-1}(x_i) + \rho \, h_m(x_i)\big)
     $$
   - モデル更新：
     $$
     F_m(x) = F_{m-1}(x) + \nu \, \rho_m \, h_m(x)
     $$
     ここで $\nu$ は **学習率 (learning rate)**

最終モデル：
$$
F_M(x) = \sum_{m=0}^M \nu \rho_m h_m(x)
$$

---

## 3. 実装例：非線形データを近似

```python
# 訓練データ（三角関数にノイズを加える）
X = np.linspace(-10, 10, 500)[:, np.newaxis]
noise = np.random.rand(X.shape[0]) * 10
y = (
    (np.sin(X).ravel() + np.cos(4 * X).ravel()) * 10
    + 10
    + np.linspace(-10, 10, 500)
    + noise
)

# 勾配ブースティング回帰
reg = GradientBoostingRegressor(
    n_estimators=50,      # 弱学習器の数
    learning_rate=0.5,    # 学習率（小さくすると安定だが多く必要）
    max_depth=3,          # 弱学習器の木の深さ
    random_state=42,
)
reg.fit(X, y)
y_pred = reg.predict(X)

# 可視化
plt.figure(figsize=(10, 5))
plt.scatter(X, y, c="k", marker="x", label="訓練データ", alpha=0.5)
plt.plot(X, y_pred, c="r", label="予測", linewidth=1.2)
plt.xlabel("x"); plt.ylabel("y")
plt.title("勾配ブースティング回帰のフィッティング")
plt.legend(); plt.show()
```

![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_5_0.png)

---

## 4. 損失関数の違いと外れ値への挙動

scikit-learn の `GradientBoostingRegressor` では `loss` を切り替えられます。

- **squared_error**（二乗誤差）：外れ値に敏感  
- **absolute_error**（絶対誤差）：外れ値にロバスト  
- **huber**：小さい誤差は二乗、大きい誤差は絶対値（ハイブリッド）  
- **quantile**：分位点回帰（予測区間や下限・上限予測に有用）

```python
# 外れ値を混ぜたデータ
X = np.linspace(-10, 10, 500)[:, np.newaxis]
noise = np.random.rand(X.shape[0]) * 10
for i in range(0, X.shape[0], 80):
    noise[i] = 70 + np.random.randint(-10, 10)  # 外れ値を挿入
y = (
    (np.sin(X).ravel() + np.cos(4 * X).ravel()) * 10
    + 10
    + np.linspace(-10, 10, 500)
    + noise
)

for loss in ["squared_error", "absolute_error", "huber", "quantile"]:
    reg = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.5,
        loss=loss,
        random_state=42,
    )
    reg.fit(X, y)
    y_pred = reg.predict(X)

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, c="k", marker="x", label="データ", alpha=0.5)
    plt.plot(X, y_pred, c="r", label=f"予測 (loss={loss})", linewidth=1.2)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(f"損失関数の違いによる予測の変化: loss={loss}")
    plt.legend(); plt.show()
```

![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_7_0.png)
![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_7_1.png)

---

## 5. 実務でのコツ

- **学習率 × 弱学習器数**：小さめの学習率（例 0.05〜0.1）＋多めの器（数百〜千）が安定。  
- **木の深さ**：浅い（max_depth=3〜5）方が汎化性能が良いケースが多い。  
- **損失関数**：データ特性に応じて選択。外れ値が多いなら `huber` や `absolute_error`。  
- **early_stopping**：バリデーションスコアが改善しなくなった時点で打ち切ると過学習防止に有効。  
- **近年は**：`HistGradientBoosting`（scikit-learn）、XGBoost、LightGBM、CatBoost などの実装が高速で大規模データに適している。  

---

## まとめ

- 勾配ブースティングは「損失関数の勾配」を利用して弱学習器を逐次追加し、残差を埋めていく仕組み。  
- 損失関数を自由に選べるのが強み。回帰・分類・分位点予測まで対応可能。  
- ハイパーパラメータ調整（学習率・弱学習器数・木の深さ）が性能に直結する。  
- 実務では早期停止・外れ値対応・大規模実装（XGBoost/LightGBMなど）を組み合わせると強力。  

---
