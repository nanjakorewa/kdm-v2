---
title: "AdaBoost（分類）"
pre: "2.4.3 "
weight: 3
title_suffix: "の直感・数式・実装"
---

{{< katex />}}
{{% youtube "1K-h4YzrnsY" %}}

<div class="pagetop-box">
  <p><b>AdaBoost</b> は、前の弱学習器が<b>誤分類したサンプルに重み</b>を置き直し、次の弱学習器を順に学習していく <b>Boosting</b> の代表手法です。浅い決定木（切り株～ごく浅い木）を多数積み重ね、全体として強力な分類器を作ります。</p>
</div>

{{% notice document %}}
- [AdaBoostClassifier — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
{{% /notice %}}

---

## 1. 直感：重み付き学習で「難しい点」を重点強化
1個目の弱学習器は全データを同じ重みで学習します。  
↓  
間違えた点の重みを増やす（= 次はそこをより重視）  
↓  
2個目を学習 … という流れを <b>反復</b>します。  
最終的には「多数決」ですが、各弱学習器には <b>信頼度に応じた重み</b>（後述の \(\alpha_t\)）が付きます。

---

## 2. 数式でみる AdaBoost（二値分類）

二値分類 \(y_i \in \{-1,+1\}\)、弱学習器 \(h_t(x)\in\{-1,+1\}\) とします（SAMME.R の連続出力版は後述）。

### 2.1 反復 \(t=1,\dots,T\)
- 現在の重み付き誤り率
  $$
  \varepsilon_t
  = \frac{\sum_i w_i^{(t)} \,\mathbf{1}\!\big[y_i \ne h_t(x_i)\big]}{\sum_i w_i^{(t)}}
  $$
- 弱学習器の重み（信頼度）
  $$
  \alpha_t = \tfrac{1}{2}\ln\frac{1-\varepsilon_t}{\varepsilon_t}
  $$

- サンプル重みの更新（正規化あり）
  $$
  w_i^{(t+1)}
  = w_i^{(t)}\exp\!\big(\alpha_t\,\mathbf{1}[y_i \ne h_t(x_i)]\big)
  $$

> 直感：\(\varepsilon_t\) が小さい（よく当たる）ほど \(\alpha_t\) は大きく、逆に外しまくる弱学習器は小さく扱われます。

### 2.2 予測
$$
H(x) = \mathrm{sign}\!\left(\sum_{t=1}^T \alpha_t\, h_t(x)\right)
$$

---

## 3. （もう一歩）指数損失とマージンの視点

AdaBoost は、分類マージン \(m_i = y_i\,F(x_i)\)（ここで \(F(x)=\sum_t \alpha_t h_t(x)\)）について **指数損失** を最小化する逐次最適化と解釈できます：
$$
\mathcal{L}
= \sum_{i=1}^n \exp(-m_i)
= \sum_{i=1}^n \exp\big(-y_i \sum_t \alpha_t h_t(x_i)\big)
$$
マージンが大きい（= 自信を持って正しく分類）ほど損失は小さく、誤分類や自信のない正解には大きなペナルティがかかります。  
この性質により、AdaBoost は「難しい点（マージンが小さい点）」へ自然に注目していきます。

> scikit-learn の `AdaBoostClassifier` には **SAMME**（離散出力）と **SAMME.R**（連続出力・確率的）があり、既定は後者（`algorithm="SAMME.R"`）。多クラスでも同様の考え方で動作します。

---

## 4. 実験：データ作成と学習・可視化

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# データ作成
X, y = make_classification(
    n_samples=2500,
    n_features=20,
    n_informative=10,
    n_redundant=4,
    n_clusters_per_class=5,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# 弱学習器：浅い決定木（切り株～深さ2程度が定番）
base = DecisionTreeClassifier(max_depth=2, random_state=117117)

ab_clf = AdaBoostClassifier(
    base_estimator=base,      # 新しめのバージョンでは 'estimator' 引数に変わる場合があります
    n_estimators=50,          # 反復回数（多すぎは過学習に注意）
    learning_rate=0.5,        # 学習率（下げれば安定だが多くの推定器が必要）
    algorithm="SAMME.R",      # 連続出力（確率）を活用
    random_state=117117,
).fit(X_train, y_train)

# 予測確率で AUC（ハード予測ではなく proba を使う）
y_score = ab_clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_score)
print(f"ROC-AUC: {auc:.3f}")

# ROC 曲線
RocCurveDisplay.from_predictions(y_test, y_score)
plt.grid(alpha=0.3); plt.show()
```

![png](/images/basic/ensemble/Adaboost_Classification_files/Adaboost_Classification_10_0.png)

> **補足**：AUC は確率出力に対して評価するのが一般的です（`predict` の 0/1 より情報量が多い）。

---

## 5. ハイパーパラメータの影響と挙動

### 5.1 `learning_rate`（学習率）
小さいほど一歩ずつ学ぶ（安定）反面、より多くの弱学習器が必要。  
`learning_rate × n_estimators` は概ねトレードオフ。

```python
scores = []
learning_rate_list = np.linspace(0.01, 1, 40)
for lr in learning_rate_list:
    clf = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=2, random_state=0),
        n_estimators=50,
        learning_rate=lr,
        algorithm="SAMME.R",
        random_state=0,
    ).fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]
    scores.append(roc_auc_score(y_test, y_score))

plt.figure(figsize=(6,4))
plt.plot(learning_rate_list, scores)
plt.xlabel("learning_rate"); plt.ylabel("ROC-AUC"); plt.grid(True); plt.show()
```

![png](/images/basic/ensemble/Adaboost_Classification_files/Adaboost_Classification_13_0.png)

### 5.2 `n_estimators`（弱学習器の数）
多くすると表現力は増すが、過学習や計算コストに注意。  
`learning_rate` と併せてグリッドで調整。

```python
scores = []
n_estimators_list = [int(ne) for ne in np.linspace(10, 200, 20)]
for ne in n_estimators_list:
    clf = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=2, random_state=0),
        n_estimators=ne,
        learning_rate=0.4,
        algorithm="SAMME.R",
        random_state=0,
    ).fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]
    scores.append(roc_auc_score(y_test, y_score))

plt.figure(figsize=(6,4))
plt.plot(n_estimators_list, scores)
plt.xlabel("n_estimators"); plt.ylabel("ROC-AUC"); plt.grid(True); plt.show()
```

![png](/images/basic/ensemble/Adaboost_Classification_files/Adaboost_Classification_16_0.png)

### 5.3 `base_estimator`（弱学習器）
弱学習器は「弱い」ほど Boosting の恩恵が出やすいのが経験則。  
深い木にすると各器が強すぎ、Boosting の積み重ね効果が薄まることがあります。

```python
scores = []
base_list = [DecisionTreeClassifier(max_depth=md, random_state=0) for md in [1,2,3,4,5]]
for base in base_list:
    clf = AdaBoostClassifier(
        base_estimator=base,
        n_estimators=60,
        learning_rate=0.3,
        algorithm="SAMME.R",
        random_state=0,
    ).fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]
    scores.append(roc_auc_score(y_test, y_score))

plt.figure(figsize=(7,4))
idx = range(len(base_list))
plt.bar(list(idx), scores)
plt.xticks(list(idx), [f"max_depth={b.max_depth}" for b in base_list])
plt.xlabel("base_estimator (DecisionTree)"); plt.ylabel("ROC-AUC")
plt.tight_layout(); plt.show()
```

---

## 6. 実務で効くコツ

- **確率校正**：`SAMME.R` は確率っぽい出力を返すが完全校正ではないことも。必要なら `CalibratedClassifierCV`。  
- **クラス不均衡**：`sample_weight` や `class_weight`、適切な評価指標（ROC-AUC, PR-AUC）を。  
- **特徴量前処理**：決定木ベースなのでスケーリングは不要だが、欠損処理やノイズ列の整理は有効。  
- **早期停止の工夫**：`n_estimators` を増やしつつ、検証スコアが頭打ちになったら止める（自前で実装）。  
- **過学習の兆候**：学習曲線で Train が上がるのに Test が下がる／`learning_rate` を下げる・`max_depth` を浅くする。

---

## 7. まとめ
- AdaBoost は「誤り点の重み付け」によって難しいサンプルを重点的に学ぶ Boosting。  
- 数式的には指数損失・マージン最大化の観点で理解できる。  
- 実装では `learning_rate × n_estimators` と弱学習器の強さのバランス調整が肝。  
- 確率出力で評価（AUC など）し、実務では校正や不均衡対応もあわせて検討すると良い。

---
