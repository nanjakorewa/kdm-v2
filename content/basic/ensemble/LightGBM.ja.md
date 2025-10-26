---
title: "LightGBM"
pre: "2.4.7 "
weight: 7
title_suffix: "の仕組みとチューニングのコツ"
---



<div class="pagetop-box">
  <p><b>LightGBM</b> は Microsoft が開発した高速な勾配ブースティングライブラリです。ヒストグラム近似と葉優先の木構築により、大規模データや高次元特徴でも <b>学習を高速化しながら高精度を維持</b> できます。</p>
  <p>GPU 学習やカテゴリ変数のネイティブ処理、分散学習など実務で嬉しい機能が揃っており、Kaggle でも定番のアンサンブル手法です。</p>
</div>

---

## 1. LightGBM の特徴

- **葉優先 (leaf-wise) 成長**：分割で最も損失が改善する葉を貪欲に展開。深い木になりやすいが、`max_depth` や `num_leaves` で制御可能。
- **ヒストグラム近似**：連続値を 256 などのビンにまとめて分割候補を探索。計算量削減とメモリ節約につながる。
- **勾配・ヘッセ行列の再利用**：バッチごとに勾配統計をまとめることで CPU キャッシュヒット率を向上。
- **ネイティブなカテゴリサポート**：ワンホット化不要で、カテゴリ集合を自動的に最適な二分割へ並べ替える。
- **機能豊富**：早期打ち切り、重み付きデータ、モノトニック制約、GPU 対応など業務要件に柔軟に対応。

---

## 2. 仕組みを式で眺める

LightGBM も一般的な勾配ブースティングと同様に、損失関数の負勾配を使って弱学習器を逐次追加します。

$$
\mathcal{L} = \sum_{i=1}^n \ell\big(y_i, F_{m-1}(x_i) + f_m(x_i)\big) + \Omega(f_m)
$$

ここで \\(f_m\\) は勾配ブースティング木 (GBT)。LightGBM は勾配 \\(g_i\\) とヘッセ行列 \\(h_i\\) をヒストグラム単位で集計し、

$$
\text{Gain} = \frac{1}{2} \left( \frac{\left(\sum_{i \in L} g_i\right)^2}{\sum_{i \in L} h_i + \lambda} + \frac{\left(\sum_{i \in R} g_i\right)^2}{\sum_{i \in R} h_i + \lambda} - \frac{\left(\sum_{i \in (L \cup R)} g_i\right)^2}{\sum_{i \in (L \cup R)} h_i + \lambda} \right) - \gamma
$$

のような分割利得を葉ごとに計算します。\\(\lambda\\) は L2 正則化、\\(\gamma\\) はノード分割のペナルティです。

---

## 3. Python で二値分類を試す

```python
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

X, y = make_classification(
    n_samples=30_000,
    n_features=40,
    n_informative=12,
    n_redundant=8,
    weights=[0.85, 0.15],
    random_state=42,
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

params = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "min_data_in_leaf": 30,
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)

gbm = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, valid_data],
    valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
)

y_pred = gbm.predict(X_valid)
print("ROC-AUC:", roc_auc_score(y_valid, y_pred))
print(classification_report(y_valid, (y_pred > 0.5).astype(int)))
```

早期打ち切りが有効になると、`num_boost_round` を大きめに指定しても過学習を抑えながら学習を止められます。

---

## 4. 主なハイパーパラメータ

| パラメータ | 役割・ポイント |
| --- | --- |
| `num_leaves` | 葉の総数。モデル容量を直接決める。`2 ** max_depth` を超えない範囲で徐々に拡大させる。 |
| `max_depth` | 木の深さ制限。`-1` で無制限だが、浅めにすると過学習防止になる。 |
| `min_data_in_leaf` / `min_sum_hessian_in_leaf` | 葉に必要なサンプル数/ヘッセ値。大きくするほど滑らかに。 |
| `learning_rate` | 学習率。小さくするほど安定だが、多数の木が必要。 |
| `feature_fraction` | 特徴量サブサンプリング。ランダム性を入れて汎化を狙う。 |
| `bagging_fraction` & `bagging_freq` | 行方向のサンプリング。`bagging_freq > 0` で有効化。 |
| `lambda_l1`, `lambda_l2` | L1/L2 正則化。複雑さを抑えたいときに調整。 |
| `min_gain_to_split` | 分割するために必要な利得のしきい値。ノイズ分割を抑える。 |

---

## 5. カテゴリ変数と欠損値

- `categorical_feature` 引数に列番号や列名を渡すだけで OK。ターゲットエンコーディング不要で高精度を得やすい。
- 欠損は内部で別枝に振り分けられるため、`NaN` のままでも動作する。ただし大量の欠損がある場合は補完との併用を検討。
- ADHD (Additive Decision Tree) 的に単調制約を課したい場合は `monotone_constraints` でサポート。

```python
categorical_cols = [0, 2, 5]
train_data = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=categorical_cols,
)
```

---

## 6. チューニングの流れ例

1. **ベースライン**：`learning_rate=0.1`、`num_leaves=31`、`feature_fraction=1.0` などでざっくり測定。
2. **容量調整**：`num_leaves` と `min_data_in_leaf` をセットで増減し、過学習兆候 (valid ロス上昇) を確認。
3. **ランダム化**：`feature_fraction` や `bagging_fraction` を 0.6〜0.9 に設定して汎化性能を上げる。
4. **正則化**：必要に応じて `lambda_l1`/`lambda_l2`、`min_gain_to_split` を追加。
5. **学習率を下げる**：性能が頭打ちなら `learning_rate` を 0.01 などに下げ、`num_boost_round` を増やして再学習。
6. **アンサンブル**：学習率・初期化を変えた LightGBM 模型を複数積み重ねると更に安定する。

---

## 7. まとめ

- LightGBM はヒストグラム近似と葉優先分割で高速かつ高精度な Boosting を実現する。
- `num_leaves` と `learning_rate` のバランス調整が性能の要。過学習には正則化とサブサンプリングで対処する。
- カテゴリ変数・欠損・GPU 対応など実務で役立つ機能が一通り揃っており、他モデルとのスタッキングでも強力なベースとなる。

---
