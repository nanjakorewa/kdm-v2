---
title: "異常検知①"
pre: "2.9.1 "
weight: 1
---

{{< katex />}}

<div class="pagetop-box">
  <p><b>異常検知（Anomaly Detection）</b>とは、通常のパターンから外れたデータを見つける手法です。  
  機械の故障検知、不正アクセスの発見、売上の異常変動の検出など、多くの実務に役立ちます。</p>
</div>

---

## 1. 実験データの準備
ここでは [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) のデータを使い、  
[ADTK (Anomaly Detection Toolkit)](https://adtk.readthedocs.io/en/stable/index.html) で異常検知を試します。

```python
import pandas as pd
from adtk.data import validate_series

s_train = pd.read_csv(
    "./training.csv", index_col="timestamp", parse_dates=True, squeeze=True
)
s_train = validate_series(s_train)
print(s_train.head())
```

```python
from adtk.visualization import plot
plot(s_train)
```

![png](/images/basic/anomaly/adtk1_files/adtk1_2_1.png)

---

## 2. 代表的な異常検知手法

ADTKには様々な検出器が用意されています。ここでは代表的な5種類を比較します。

```python
import matplotlib.pyplot as plt
from adtk.detector import (
    AutoregressionAD,
    InterQuartileRangeAD,
    LevelShiftAD,
    PersistAD,
    SeasonalAD,
)

model_dict = {
    "LevelShiftAD": LevelShiftAD(window=5),
    "SeasonalAD": SeasonalAD(),
    "PersistAD": PersistAD(c=3.0, side="positive"),
    "InterQuartileRangeAD": InterQuartileRangeAD(c=1.5),
    "AutoregressionAD": AutoregressionAD(n_steps=14, step_size=24, c=3.0),
}

for model_name, model in model_dict.items():
    anomalies = model.fit_detect(s_train)
    plot(s_train, anomaly=anomalies, anomaly_color="red", anomaly_tag="marker")
    plt.title(model_name)
    plt.show()
```

---

## 3. 各手法の直感と数式

### (1) LevelShiftAD（水準の変化）
データの平均値が急に変化するかどうかを見る。  
例：センサー値がある時点で急に上がった/下がった。  

$$
\Delta_t = \bar{x}_{t:t+w} - \bar{x}_{t-w:t} 
$$

もし $|\Delta_t|$ が大きければ異常と判定。

---

### (2) SeasonalAD（季節性とのズレ）
周期性を学習し、そのパターンから外れるデータを検知。  
例：毎日のアクセス数が急に増えすぎた。  

$$
e_t = x_t - \hat{x}_t^{(\text{seasonal})}
$$

誤差 $e_t$ が大きければ異常。

---

### (3) PersistAD（値の持続）
「過去の値から大きく乖離したか」を判定。  

$$
|x_t - x_{t-1}| > c \cdot \sigma
$$

閾値 $c$ を超えると異常。

---

### (4) InterQuartileRangeAD（四分位範囲）
統計的に外れ値を検出。  
データ分布の第1四分位 $Q1$、第3四分位 $Q3$、四分位範囲 $IQR=Q3-Q1$ を用いる。  

$$
x_t < Q1 - c \cdot IQR \quad \text{または} \quad x_t > Q3 + c \cdot IQR
$$

---

### (5) AutoregressionAD（自己回帰）
過去の値から未来を予測し、誤差が大きい点を異常とする。  

$$
x_t \approx \sum_{i=1}^p \phi_i x_{t-i} + \epsilon_t
$$

残差 $\epsilon_t$ が大きいと異常。

---

## 4. 可視化結果

![png](/images/basic/anomaly/adtk1_files/adtk1_4_0.png)
![png](/images/basic/anomaly/adtk1_files/adtk1_4_1.png)
![png](/images/basic/anomaly/adtk1_files/adtk1_4_2.png)
![png](/images/basic/anomaly/adtk1_files/adtk1_4_3.png)
![png](/images/basic/anomaly/adtk1_files/adtk1_4_4.png)

---

## まとめ
- 異常検知は「通常パターンから外れたデータ」を見つける技術。  
- ADTKでは **統計的手法・季節性・自己回帰** など多様な検出器を利用可能。  
- 使い分けのポイント：
  - **水準変化検知**：平均が変化した時  
  - **季節性検知**：周期パターンから外れた時  
  - **自己回帰検知**：過去データで予測できない時  

---
