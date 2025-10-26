---
title: "k近傍法 (k-NN)"
pre: "2.2.6 "
weight: 6
title_suffix: "距離に基づくシンプルな分類器"
---

{{< lead >}}
k-NN は学習時にパラメータを推定せず、予測時に近いサンプルを多数決するだけの「怠惰学習」モデルです。直感的で、高次元すぎないデータであれば堅実に機能します。
{{< /lead >}}

---

## 1. アルゴリズム

1. 学習データを丸ごと記憶
2. 予測したい点に対して距離を計算
3. 最も近い \\(k\\) 個のラベルを集計し、多数決または重み付き投票

距離にはユークリッド距離がよく使われますが、コサイン距離やマンハッタン距離に変えることも可能です。

---

## 2. Python 実装

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

X, y = make_classification(
    n_samples=600,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    n_classes=3,
    random_state=7,
)

ks = [1, 3, 5, 7, 11]
for k in ks:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights="distance"))
    scores = cross_val_score(model, X, y, cv=5)
    print(f"k={k}: CV accuracy={scores.mean():.3f} ± {scores.std():.3f}")
```

![ダミー図: k-NN の決定境界例](/images/placeholder_regression.png)

---

## 3. ハイパーパラメータ

- \\(k\\): 小さいほど境界が細かくなるがノイズに弱い。奇数にすると多数決がタイになりにくい
- `weights="distance"`: 近い点に大きな票を与える設定。密度が異なるデータで有効
- 距離関数: `metric` パラメータで指定。スケール差がある場合は標準化必須

---

## 4. 長所と欠点

| 長所 | 欠点 |
| ---- | ---- |
| 実装が非常に簡単 | 予測時に全データとの距離を計算するため遅い |
| 非線形境界を自然に扱える | 次元の呪いで高次元になるほど距離の差が縮む |
| 学習が不要 | ノイズに敏感で、外れ値が投票結果を乱す |

---

## 5. まとめ

- k-NN は「近くのものは似ている」という仮定をそのまま利用する基本手法
- 前処理（標準化、特徴選択）と \\(k\\) の探索だけで堅実なベースラインを構築できる
- 大規模データでは kd-tree や ball-tree、近似最近傍探索を検討しましょう

---
