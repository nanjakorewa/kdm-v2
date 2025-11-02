---

title: "ワッサースタイン距離（Earth Mover's Distance）"

pre: "4.4.3 "

weight: 6

title_suffix: "分布の形と位置を同時に比較"

---



{{< lead >}}

ワッサースタイン距離（Wasserstein Distance）、別名 Earth Mover's Distance は、確率分布間の距離を「土砂を運ぶコスト」として捉える指標です。分布の位置だけでなく形の違いも反映し、生成モデルやヒストグラムの比較に適しています。

{{< /lead >}}



---



## 1. 定義



一次元の離散分布では、累積分布関数 (CDF) の差の積分として表せます。



$$

W_1(P, Q) = \int_{-\infty}^{\infty} |F_P(x) - F_Q(x)| \, dx

$$



多次元の場合は最適輸送問題として定式化され、分布間で「質量」を移動させる最小コストを求めます。



---



## 2. Python で計算



```python

import numpy as np

from scipy.stats import wasserstein_distance



x = np.random.normal(0, 1, size=1000)

y = np.random.normal(1, 1.5, size=1000)



dist = wasserstein_distance(x, y)

print("Wasserstein distance:", round(dist, 3))

```



`scipy.stats.wasserstein_distance` は 1 次元の距離を計算します。多次元の場合は `ot`（POT: Python Optimal Transport）ライブラリなどを使います。



---



## 3. 特徴



- **形の違いに敏感**：平均が同じでも分散が異なる分布は距離が大きくなる。

- **ロバスト**：KL ダイバージェンスのようにサポートがずれて無限大になることがない。

- **計算コスト**：高次元では最適輸送の計算が重いため、Sinkhorn 距離などの近似を使うことが多い。



---



## 4. 実務での応用



- **生成モデル評価**：GAN や VAE などで生成分布と実データ分布の差を測る。

- **品質検査**：ヒストグラム全体の違いを捉えたい場合に有効。

- **時系列比較**：期間ごとの分布変化をトレースし、ワッサースタイン距離が閾値を超えたら異常とみなす。



---



## 5. 注意点



- 多次元データでは正規化や次元削減を行ってから距離を計算すると安定しやすい。

- 見かけの距離が小さくても、平均や分散など他指標も合わせて比較すると理解が深まる。

- Sinkhorn 距離（正則化付き最適輸送）は計算が高速で、大規模データセット向き。



---



## まとめ



- ワッサースタイン距離は分布間の移動コストとして距離を測り、位置と形の両方を反映できる。

- `wasserstein_distance` で一次元の距離を簡単に計算でき、多次元では最適輸送ライブラリを活用する。

- KL や JSD と併用し、分布の差異を多角的に評価しよう。



---

