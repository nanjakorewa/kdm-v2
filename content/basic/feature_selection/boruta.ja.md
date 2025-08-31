---
title: "Boruta"
pre: "2.7.2 "
weight: 2
searchtitle: "Borutaを使い特徴選択を実行する"
---

{{< katex />}}
{{% youtube "xOkKnsqhUgw" %}}

<div class="pagetop-box">
  <p><b>Boruta（ボルタ）</b>は、特徴選択（Feature Selection）のアルゴリズムのひとつです。<br>
  データに含まれる多数の特徴量の中から「本当に有用な特徴量」だけを選び出し、不要な特徴を削除してくれます。  
  モデルの精度向上・解釈性の改善・計算効率化に役立つ重要な手法です。</p>
</div>

---

## 1. なぜ特徴選択が必要か？
- **高次元データの問題**  
  特徴が多すぎると「ノイズ」が増え、過学習しやすくなる。  

- **計算コスト**  
  無駄な特徴を減らせば学習も予測も速くなる。  

- **解釈性の向上**  
  モデルの判断に本当に寄与している要素を特定できる。  

---

## 2. Borutaの仕組み（直感）
1. すべての特徴を使ってランダムフォレストを学習する。  
2. 各特徴の「重要度」を計算する。  
3. データをランダムにシャッフルした「影の特徴（shadow features）」を作り、比較基準にする。  
4. 本物の特徴の重要度が「影の特徴」より高ければ **有用**、低ければ **不要** と判断する。  
5. これを繰り返して、安定的に有用な特徴を決定する。  

---

## 3. 実装例（CSVデータ）

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# データを読み込み
X = pd.read_csv("examples/test_X.csv", index_col=0).values
y = pd.read_csv("examples/test_y.csv", header=None, index_col=0).values.ravel()

# ランダムフォレスト
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)

# Boruta
feat_selector = BorutaPy(rf, n_estimators="auto", verbose=2, random_state=1)
feat_selector.fit(X, y)

print("選ばれた特徴:", feat_selector.support_)
print("特徴のランキング:", feat_selector.ranking_)

# 有用な特徴だけ残す
X_filtered = feat_selector.transform(X)
```

---

## 4. 人工データでの実験

Borutaが「必要な特徴は残し、不要な特徴は削除できるか」を確認します。

### すべて有用な特徴（削除しない）

```python
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

X, y = make_classification(
    n_samples=1000, n_features=10,
    n_informative=10, n_redundant=0, n_classes=2,
    random_state=0, shuffle=False
)
model = XGBClassifier(max_depth=4)

feat_selector = BorutaPy(model, n_estimators="auto", verbose=2, random_state=1)
feat_selector.fit(X, y)
X_filtered = feat_selector.transform(X)

print(f"{X.shape[1]} --> {X_filtered.shape[1]}")
```

> 不要な特徴がなければ削除されず、そのまま残る。

---

### 不要な特徴が多い場合（削除する）

```python
X, y = make_classification(
    n_samples=2000, n_features=100,
    n_informative=10, n_redundant=0, n_classes=2,
    random_state=0, shuffle=False
)
model = XGBClassifier(max_depth=5)

feat_selector = BorutaPy(model, n_estimators="auto", verbose=2, random_state=1)
feat_selector.fit(X, y)
X_filtered = feat_selector.transform(X)

print(f"{X.shape[1]} --> {X_filtered.shape[1]}")
```

> 100個中10個だけ有用 → Borutaは不要な特徴を削除し、有用な10個を残す。

---

## 5. 実務でのポイント
- **ランダムフォレスト/XGBoost** などツリー系モデルと相性が良い。  
- データに含まれる「ノイズ特徴」をしっかり落とせる。  
- ただし計算コストはやや高い（特徴数が多い場合は注意）。  
- 得られた「重要特徴」をもとにモデルの解釈や可視化が可能。  

---

## まとめ
- Borutaは「影の特徴」と比較することで、有用な特徴を安定して選択できる手法。  
- すべて必要なら削除せず、不要ならしっかり削除してくれる。  
- 実務では前処理として使うことで、精度・効率・解釈性が改善する。  

---
