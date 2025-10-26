---
title: "StandardScaler による標準化"
pre: "3.3.2 "
weight: 2
title_suffix: "平均0・分散1でスケールをそろえる"
---

{{< lead >}}
StandardScaler は平均 0、分散 1 に正規化することで、特徴量のスケールを揃える定番の前処理です。距離計算や線形モデルで重みのバランスを取りやすくなります。
{{< /lead >}}

---

## 1. いつ使う？

- 距離ベース手法（k-means、SVM、PCA など）
- 勾配降下を使うモデル（線形回帰、ロジスティック回帰、ニューラルネット）
- 正則化を入れるモデル（Lasso、Ridge）で特徴量の単位がバラバラなとき

標準化を行わないと、単位の大きい特徴量が距離や重みを支配し、学習が偏る恐れがあります。

---

## 2. Python 実装

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.array(
    [[180, 80],
     [165, 60],
     [170, 72],
     [150, 50]]
)

scaler = StandardScaler(with_mean=True, with_std=True)
scaled = scaler.fit_transform(data)

print("平均:", scaler.mean_)
print("分散:", scaler.var_)
print("変換後:\n", scaled)
```

`fit_transform` で学習と変換を同時に行います。学習済みのスケーラーを保存し、テストデータには `transform` のみを適用する点に注意してください。

---

## 3. ハイパーパラメータと直感

| パラメータ | 役割 | 変えるとどうなるか |
| --- | --- | --- |
| `with_mean` | 平均を引くかどうか | `False` にすると中心をずらさず分散のみ調整。疎行列に対して有効 |
| `with_std` | 分散で割るかどうか | `False` にすると単なる中心化。標準偏差が極端に小さい列で安定させたいときに使う |
| `copy` | 元データをコピーするか | `False` にするとインプレース変換。大規模データでメモリ節約が必要なときに選択 |

疎行列の場合、平均を引くと疎性が崩れ計算コストが増します。そのときは `with_mean=False` を設定するか、`MaxAbsScaler` を利用するとよいでしょう。

---

## 4. 逆変換で元のスケールに戻す

```python
original = scaler.inverse_transform(scaled)
```

モデルの出力を解釈したいときは逆変換で元の単位に戻すと納得感が生まれます。

---

## まとめ

- StandardScaler は平均 0、分散 1 に正規化して特徴量の影響力を均等にする。
- `with_mean` や `with_std` を調整することで疎行列や特殊なデータにも対応可能。
- 学習と推論で同じスケーラーを使い回し、逆変換を活用すると実務で扱いやすい。

---
