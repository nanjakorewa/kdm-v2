---
title: "RobustScaler で外れ値に強い標準化"
pre: "3.3.4 "
weight: 4
title_suffix: "中央値と四分位範囲でスケーリング"
---

{{< lead >}}
RobustScaler は中央値と四分位範囲（IQR）を用いてスケーリングすることで、外れ値の影響を抑えます。金融データやセンサー値など、極端な値が混在するケースで重宝します。
{{< /lead >}}

---

## 1. 変換の仕組み

$$
x' = \frac{x - \text{median}(x)}{\text{IQR}(x)}
$$

ここで `IQR = Q3 - Q1`。中央値でセンタリングし、IQR でスケールするため、外れ値があっても中心やスケールが大きく変動しません。

---

## 2. Python 実装

```python
import numpy as np
from sklearn.preprocessing import RobustScaler

data = np.array(
    [[10], [12], [9], [11], [300]]  # 外れ値を含む
)

scaler = RobustScaler(
    with_centering=True,
    with_scaling=True,
    quantile_range=(25.0, 75.0),
)
scaled = scaler.fit_transform(data)

print("中央値:", scaler.center_)
print("IQR:", scaler.scale_)
print("変換後:\n", scaled.T)
```

外れ値の 300 があっても、他の値が大きく押しつぶされずに済みます。

---

## 3. ハイパーパラメータの直感

| パラメータ | 役割 | 変えるとどうなるか |
| --- | --- | --- |
| `quantile_range` | IQR の上下パーセンタイル | `(10, 90)` のように広げるとスケールが大きく、狭めると外れ値の影響がさらに減る |
| `with_centering` | 中央値を引くか | `False` にするとスケーリングのみ。疎行列や既に中心化されたデータで有効 |
| `with_scaling` | IQR で割るか | `False` にすると中央値調整だけを行う。スケールを保ちつつ外れ値対策したいとき |
| `unit_variance` | IQR を 1 に合わせるか | `True` で IQR が 1 になるよう再スケーリングし、StandardScaler に近い感覚で扱える |

`quantile_range` はデータの分布に合わせて調整するのがポイント。極端な外れ値が多い場合は `(20, 80)` や `(30, 70)` まで狭めると頑健性が増します。

---

## 4. 他スケーラーとの使い分け

- **StandardScaler**：外れ値が少なく、ガウス分布に近い場合。
- **MinMaxScaler**：値域を揃えたい場合。
- **RobustScaler**：外れ値が多い、または分布が歪んでいる場合の第一候補。

---

## まとめ

- RobustScaler は中央値と IQR を用いて外れ値に強いスケーリングを実現する。
- `quantile_range` を調整すると外れ値耐性とスケール感をトレードオフできる。
- 他のスケーラーと状況に応じて使い分けることで、前処理の品質が向上する。

---
