---
title: "異常検知②"
pre: "2.9.2 "
weight: 2
---

{{< katex />}}

<div class="pagetop-box">
  <p><b>異常検知（Anomaly Detection）</b>の2回目では、多次元データに対して異常検知を行います。  
  複数の特徴量を同時に扱うことで、より高度な異常（単変量では見えない異常）を見つけられます。</p>
</div>

---

## 1. 実験データの準備
前回のセンサーデータに新しい列を追加して、2次元のデータにします。

```python
import numpy as np
import pandas as pd
from adtk.data import validate_series

s_train = pd.read_csv("./training.csv", index_col="timestamp", parse_dates=True)
s_train = validate_series(s_train)

# 新しい特徴を追加（元の値から sin と cos を組み合わせたもの）
s_train["value2"] = s_train["value"].apply(lambda v: np.sin(v) + np.cos(v))
s_train.head()
```

```python
from adtk.visualization import plot
plot(s_train)
```

![png](/images/basic/anomaly/adtk2_files/adtk2_2_1.png)

---

## 2. 多次元での異常検知の考え方
- **1次元**では「その特徴だけの外れ値」を検知。  
- **多次元**では「各特徴の組み合わせから外れる点」を検知できる。  

例：  
- `value` も `value2` も正常範囲にある → 正常  
- しかし「両者の関係」から外れている → 異常  

---

## 3. 使用する手法

ここでは3種類の異常検知を試します。

### (1) OutlierDetector（局所外れ値因子）
[Local Outlier Factor (LOF)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) を利用。  
局所的な密度の低さを利用して外れ値を判定する。

$$
LOF_k(x) = \frac{\sum_{o \in N_k(x)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(x)}}{|N_k(x)|}
$$

ここで \\(\text{lrd}\\) は局所到達可能密度。  
LOFが大きいほど「周囲に比べて浮いている」と判断。

---

### (2) RegressionAD（回帰による予測残差）
ある特徴（ここでは `value2`）を他の特徴から回帰予測し、その誤差を異常とする。  

$$
e_t = y_t - \hat{y}_t, \quad |e_t| > c \cdot \sigma \Rightarrow \text{異常}
$$

---

### (3) PcaAD（主成分分析）
複数特徴を主成分分析し、低次元空間で表現できない部分を異常とみなす。  

- データ行列 \\(X\\) を固有分解  
- 上位の主成分で再構成したときの誤差  
  $$
  \| X - X_{\text{reconstructed}} \| > \text{threshold}
  $$

---

## 4. 実験コード

```python
import matplotlib.pyplot as plt
from adtk.detector import OutlierDetector, PcaAD, RegressionAD
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor

model_dict = {
    "OutlierDetector": OutlierDetector(LocalOutlierFactor(contamination=0.05)),
    "RegressionAD": RegressionAD(regressor=LinearRegression(), target="value2", c=3.0),
    "PcaAD": PcaAD(k=2),
}

for model_name, model in model_dict.items():
    anomalies = model.fit_detect(s_train)

    plot(
        s_train,
        anomaly=anomalies,
        ts_linewidth=1,
        ts_markersize=3,
        anomaly_color="red",
        anomaly_alpha=0.3,
        curve_group="all",
    )
    plt.title(model_name)
    plt.show()
```

---

## 5. 結果の可視化

![png](/images/basic/anomaly/adtk2_files/adtk2_4_1.png)
![png](/images/basic/anomaly/adtk2_files/adtk2_4_3.png)
![png](/images/basic/anomaly/adtk2_files/adtk2_4_5.png)

---

## まとめ
- 多次元の異常検知では「単一の値」だけでなく「特徴同士の関係」から外れたデータを見つけられる。  
- **LOF**：局所密度から外れた点を検知。  
- **RegressionAD**：回帰の誤差で異常を検知。  
- **PCA**：主成分で説明できない部分を異常とみなす。  
- 実務では「複数特徴の関係」まで考慮することで、より精度の高い異常検知が可能になる。  

---
