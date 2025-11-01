---
title: "RuleFit | Aturan + Linear dengan L1"
linkTitle: "RuleFit"
seo_title: "RuleFit | Aturan + Linear dengan L1"
pre: "2.3.4 "
weight: 4
title_suffix: "Aturan + Linear dengan L1"
---

{{% notice ref %}}
Friedman, Jerome H., dan Bogdan E. Popescu. “Predictive learning via rule ensembles.” The Annals of Applied Statistics (2008). ([pdf](https://jerryfriedman.su.domains/ftp/RuleFit.pdf))
{{% /notice %}}

<div class="pagetop-box">
  <p><b>RuleFit</b> menggabungkan <b>aturan</b> dari ansambel pohon (mis., boosting, random forest) dan fitur asli dalam model <b>linear</b>, dengan <b>regularisasi L1</b> untuk sparsitas dan interpretabilitas.</p>
</div>

---

## 1. Gagasan (dengan rumus)

1) <b>Ekstrak aturan</b> dari jalur ke daun, ubah ke fitur biner $r_j(x)\in\{0,1\}).  
2) <b>Tambahkan</b> istilah linear terstandar \\(z_k(x)$.  
3) <b>Fit linear dengan L1</b>:

$$
\hat y(x) = \beta_0 + \sum_j \beta_j r_j(x) + \sum_k \gamma_k z_k(x)
$$

L1 mempertahankan aturan/istilah penting saja.

---

## 2. Dataset (OpenML: house_sales)

{{% notice document %}}
- [fetch_openml](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)
{{% /notice %}}

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dataset = fetch_openml(data_id=42092, as_frame=True)
X = dataset.data.select_dtypes("number").copy()
y = dataset.target.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 3. Jalankan RuleFit

{{% notice document %}}
- Implementasi Python: <a href="https://github.com/christophM/rulefit" target="_blank" rel="noopener">christophM/rulefit</a>
{{% /notice %}}

```python
from rulefit import RuleFit
import warnings
warnings.simplefilter("ignore")

rf = RuleFit(max_rules=200, tree_random_state=27)
rf.fit(X_train.values, y_train.values, feature_names=list(X_train.columns))

pred_tr = rf.predict(X_train.values)
pred_te = rf.predict(X_test.values)

def report(y_true, y_pred, name):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{name:8s}  RMSE={rmse:,.0f}  MAE={mae:,.0f}  R2={r2:.3f}")

report(y_train, pred_tr, "Train")
report(y_test,  pred_te, "Test")
```

---

## 4. Tinjau aturan teratas

```python
rules = rf.get_rules()
rules = rules[rules.coef != 0].sort_values(by="importance", ascending=False)
rules.head(10)
```

- <b>rule</b>: kondisi if–then (atau `type=linear`)  
- <b>coef</b>: koefisien (unit sama dengan target)  
- <b>support</b>: proporsi sampel yang memenuhi aturan  
- <b>importance</b>: pentingnya aturan

Tips: `type=linear` untuk efek marginal; `type=rule` untuk interaksi/ambang; pilih yang koefisien besar dan support wajar.

---

## 5. Validasi aturan dengan visualisasi

Gunakan scatter/boxplot untuk memeriksa konsistensi aturan terhadap target.

---

## 6. Catatan praktis

- Skala dan tangani outlier  
- Kategori: rapikan level langka sebelum one-hot  
- Transformasi target (mis. `log(y)`) untuk target miring  
- Atur jumlah/kedalaman aturan; tuning dengan CV  
- Hindari kebocoran; fit praproses pada train saja

---

## 7. Ringkasan

- RuleFit = <b>aturan</b> dari pohon + <b>istilah linear</b> + <b>L1</b>.  
- Seimbang antara nonlinier & interpretabilitas.  
- Tuning granularitas dan validasi visual.

---

