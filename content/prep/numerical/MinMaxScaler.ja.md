---
title: "MinMaxScaler でレンジ正規化"
pre: "3.3.3 "
weight: 3
title_suffix: "0〜1以外の範囲にも柔軟に対応"
---

{{< lead >}}
MinMaxScaler はデータを指定した最小値・最大値の範囲に線形変換します。ニューラルネットのシグモイド出力や画像ピクセルなど、値域を制限したいケースで便利です。
{{< /lead >}}

---

## 1. 変換の仕組み

$$
x' = \frac{x - \min(x)}{\max(x) - \min(x)} (b - a) + a
$$

`feature_range=(a, b)` を指定すると、各特徴量が `[a, b]` にスケールされます。既定値は `(0, 1)` です。

---

## 2. Python 実装

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.array(
    [[20, 400],
     [30, 600],
     [25, 550],
     [35, 800]]
)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(data)

print("レンジ:", scaler.data_min_, scaler.data_max_)
print("変換後:\n", scaled)
```

`feature_range=(-1, 1)` のように指定すれば、双方向に圧縮できます。ニューラルネットに入力する場合、活性化関数との相性を考えて設定しましょう。

---

## 3. ハイパーパラメータの直感

| パラメータ | 役割 | 変えるとどうなるか |
| --- | --- | --- |
| `feature_range` | 変換後のレンジ (a, b) | `(-1, 1)` で双方向、`(0, 1)` がデフォルト。`(0, 255)` にするとピクセル値へ戻せる |
| `clip` | レンジ外の値をクリップするか | `True` で指定範囲に収める。外れ値があるときの安全策。`False` だと線形に外へはみ出す |
| `copy` | コピーするか | `False` でインプレース変換。大量データでメモリを節約したいとき |

外れ値が混ざると、全体が極端に押しつぶされる点に注意。必要ならば外れ値を事前に処理するか、`clip=True` で上限を制限しましょう。

---

## 4. 逆変換と応用

```python
original = scaler.inverse_transform(scaled)
```

モデルの予測結果を元の単位に戻すことができます。オートエンコーダで 0〜1 の範囲に収めた特徴量を、後段のアプリに渡す際にも役立ちます。

---

## まとめ

- MinMaxScaler は特徴量を任意のレンジに線形変換し、特定の範囲に収めたい用途に適している。
- `feature_range` と `clip` の調整で、活性化関数との相性や外れ値の影響をコントロールできる。
- 逆変換を使えばビジネス側に元の単位で結果を説明できるため、実運用でも扱いやすい。

---
