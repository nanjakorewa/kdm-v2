---
title: "CatBoost | カテゴリ処理に強い勾配ブースティング"
linkTitle: "CatBoost"
seo_title: "CatBoost | カテゴリ処理に強い勾配ブースティング"
pre: "2.4.8 "
weight: 8
title_suffix: "カテゴリ処理に強い勾配ブースティング"
---

<div class="pagetop-box">
  <p><b>CatBoost</b> は Yandex が開発した <b>カテゴリ変数に強い勾配ブースティング木</b> です。ターゲット統計量のオンライン変換と対称木 (Oblivious Tree) により、前処理を最小限にしながら高精度を実現します。</p>
  <p>少ない学習率でも収束が速く、欠損値やシフトした分布へのロバスト性も高いため、実務・コンペ双方で広く使われています。</p>
</div>

---

## 1. CatBoost の仕組み

- **カテゴリ変数のエンコーディング**  
  ターゲット統計量をリークなく扱うために、データをランダムに並び替えて逐次的に平均・頻度を更新する <b>Ordered Target Statistics</b> を採用。ワンホット化や Label Encoding の欠点を回避。

- **対称木 (Oblivious Tree)**  
  木の各階層で同じ特徴量と閾値を使う制約を課し、木の深さごとに 2^d の葉を生成。GPU フレンドリーで、推論時の分岐が少なく高速。

- **Ordered Boosting**  
  過学習を抑えるため、各反復で別の順序付けを用いて勾配を推定。標準的な勾配ブースティングより安定した学習が可能。

- **リッチな機能**  
  失敗ラベルの重み付け、テキスト特徴のサポート、対称木の深さ指定、メトリクスをブレンドした評価など多機能。

---

## 2. Python で分類タスクを学習

```python
from catboost import CatBoostClassifier, Pool
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# クレジット申請データ (カテゴリ・数値が混在)
data = fetch_openml(name="credit-g", version=1, as_frame=True)
X = data.data
y = (data.target == "good").astype(int)

categorical_features = X.select_dtypes(include="category").columns.tolist()

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_pool = Pool(
    X_train,
    label=y_train,
    cat_features=categorical_features,
)
valid_pool = Pool(X_valid, label=y_valid, cat_features=categorical_features)

model = CatBoostClassifier(
    depth=6,
    iterations=1000,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    early_stopping_rounds=50,
    verbose=100,
)
model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

proba = model.predict_proba(X_valid)[:, 1]
pred = (proba >= 0.5).astype(int)
print("ROC-AUC:", roc_auc_score(y_valid, proba))
print(classification_report(y_valid, pred, digits=3))
```

`Pool` にカテゴリ列のインデックス/名前を渡すだけでターゲットエンコーディングと並び替えが内部的に適用されます。

---

## 3. 主なハイパーパラメータ

| パラメータ | 役割・調整のポイント |
| --- | --- |
| `depth` | 木の深さ。対称木なので `depth=6` で葉は 64。深くすると表現力が高まるが過学習リスク増。 |
| `iterations` | ブースト回数。`early_stopping_rounds` を併用して過学習を抑制。 |
| `learning_rate` | 学習率。小さくすると精度が上がる傾向、代わりに反復を増やす。 |
| `l2_leaf_reg` | 葉の正則化項。大きいほど滑らかになりノイズに強くなる。 |
| `border_count` | 数値特徴のビン数。デフォルト 254。少なくすると高速だが分割精度が落ちる。 |
| `bagging_temperature` | 行方向サンプリングの温度。0 に近いと決定的、値を上げるとランダム性増。 |
| `class_weights` | 不均衡データのクラス重みを直接指定可能。 |

---

## 4. 特徴量重要度と SHAP

```python
importance = model.get_feature_importance(type="PredictionValuesChange")
for name, score in sorted(zip(X.columns, importance), key=lambda x: -x[1])[:10]:
    print(f"{name}: {score:.3f}")

shap_values = model.get_feature_importance(valid_pool, type="ShapValues")
# shap_values[:, -1] はベースライン。matplotlib で要約プロットを描画可能。
```

- `PredictionValuesChange` は予測値変化に基づく重要度。正負で寄与方向を解釈できる。
- SHAP 値は `type="ShapValues"` で取得でき、各サンプルの寄与度解析に便利。

---

## 5. CatBoost ならではのテクニック

- **カテゴリの組み合わせ**：`one_hot_max_size` や `combinations_ctypes` を使うと、複数カテゴリの交互作用を自動で生成。
- **テキスト特徴**：`text_features` に列名を渡し、`text_processing` パラメータで TF-IDF や BM25 の設定が可能。
- **モノトニック制約**：`monotone_constraints` で特定特徴に単調性を課すことができ、料金体系などに有用。
- **オンザフライのクロス検証**：`cv` 関数を使うと Ordered Booster に最適化された CV が実行できる。

---

## 6. 他手法との使い分け

- 前処理を抑えたい、カテゴリ主体のデータ ⇒ **CatBoost** が第一候補。
- 数値主体で高速に大量トライしたい ⇒ LightGBM の Hist ベースが有利。
- 線形・スパース行列で大規模 ⇒ XGBoost の DMatrix がメモリ効率良い場合も。
- アンサンブルでは CatBoost を混ぜることで、カテゴリの扱いに強いモデルを追加できパフォーマンスが安定しやすい。

---

## 7. まとめ

- CatBoost はカテゴリ変数を自動処理し、Ordered Boosting で安定性を確保する高性能 GBDT。
- `depth`・`iterations`・`learning_rate` のバランスと `l2_leaf_reg` の正則化が性能の鍵。
- SHAP やテキスト対応など機能が充実しており、他のブースティング手法と組み合わせたスタッキングでも効果を発揮する。

---
