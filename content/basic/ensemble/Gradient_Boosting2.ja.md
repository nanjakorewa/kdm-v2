---
title: "勾配ブースティングの可視化 | 段階的な改善の見える化"
linkTitle: "勾配ブースティングの可視化"
seo_title: "勾配ブースティングの可視化 | 段階的な改善の見える化"
pre: "2.4.6 "
weight: 6
title_suffix: "段階的な改善の見える化"
---

{{< katex />}}
{{% youtube "ZgssfFWQbZ8" %}}

<div class="pagetop-box">
  <p>勾配ブースティング回帰では、弱学習器（小さな決定木）を<b>1本ずつ順番に追加</b>しながらモデルを改良していきます。<br>
  このページでは、<b>「各木がどのように予測に寄与しているか」</b>を可視化し、段階的に改善していくイメージを掴みます。</p>
</div>

{{% notice document %}}
- [GradientBoostingRegressor — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
{{% /notice %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.ensemble import GradientBoostingRegressor
```

---

## 1. 学習と最終予測の確認

まずは簡単なデータに対して勾配ブースティング回帰を学習し、最終的な予測曲線を確認します。

```python
n_samples = 500
X = np.linspace(-10, 10, n_samples)[:, np.newaxis]
noise = np.random.rand(X.shape[0]) * 10
y = (np.sin(X).ravel()) * 10 + 10 + noise

# モデル作成
n_estimators = 10
learning_rate = 0.5
reg = GradientBoostingRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
)
reg.fit(X, y)

# 予測
y_pred = reg.predict(X)

# 可視化
plt.figure(figsize=(20, 10))
plt.scatter(X, y, c="k", marker="x", label="訓練データ")
plt.plot(X, y_pred, c="r", label="最終予測", linewidth=1)
plt.axhline(y=np.mean(y), color="gray", linestyle=":", label="初期モデル（平均値）")
plt.xlabel("x"); plt.ylabel("y")
plt.title("勾配ブースティングの最終予測")
plt.legend(); plt.show()
```

![gradient-boosting2 block 2](/images/basic/ensemble/gradient-boosting2_block02.svg)

**解説**  
- 灰色の破線は「初期モデル」（平均値のみの予測）。  
- 赤い線が **10本の木を足し合わせた最終予測**。  
- 初期値から出発して、木を追加するごとに予測が改良されていきます。

---

## 2. 木ごとの寄与を積み上げて表示

次に「各木がどれだけ予測を修正したか」を棒グラフで積み上げます。

```python
fig, ax = plt.subplots(figsize=(20, 10))
temp = np.zeros(n_samples) + np.mean(y)  # 初期モデルは平均値

for i in range(n_estimators):
    # i本目の木の予測値 × learning_rate が寄与部分
    res = reg.estimators_[i][0].predict(X) * learning_rate
    ax.bar(X.flatten(), res, bottom=temp, label=f"{i+1} 本目の木", alpha=0.05)
    temp += res  # 累積して次へ

# データと最終予測を重ねる
plt.scatter(X.flatten(), y, c="k", marker="x", label="訓練データ")
plt.plot(X, y_pred, c="r", label="最終予測", linewidth=1)
plt.xlabel("x"); plt.ylabel("y")
plt.title("木ごとの寄与を積み上げた可視化")
plt.legend(); plt.show()
```

![gradient-boosting2 block 3](/images/basic/ensemble/gradient-boosting2_block03.svg)

**解説**  
- 薄い棒が「各木がどれだけ予測を修正したか」を示します。  
- それらを累積すると、最終的に赤い予測曲線になります。  
- 学習率 `learning_rate` を掛けることで、一歩ずつ慎重に修正しています。

---

## 3. 途中までの積み上げ（段階的な改善）

さらに「木を5本まで足した時点」での予測を順に可視化します。

```python
for i in range(5):
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.title(f"{i+1} 本目までの寄与で作られた予測")
    temp = np.zeros(n_samples) + np.mean(y)

    for j in range(i + 1):
        res = reg.estimators_[j][0].predict(X) * learning_rate
        ax.bar(X.flatten(), res, bottom=temp, label=f"{j+1} 本目", alpha=0.05)
        temp += res

    # データと予測を描画
    plt.scatter(X.flatten(), y, c="k", marker="x", label="訓練データ")
    plt.plot(X, temp, c="r", linewidth=1.2, label="途中の予測")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend(); plt.show()
```

![gradient-boosting2 block 4](/images/basic/ensemble/gradient-boosting2_block04.svg)

![gradient-boosting2 block 4 fig 1](/images/basic/ensemble/gradient-boosting2_block04_fig01.svg)

![gradient-boosting2 block 4 fig 2](/images/basic/ensemble/gradient-boosting2_block04_fig02.svg)

![gradient-boosting2 block 4 fig 3](/images/basic/ensemble/gradient-boosting2_block04_fig03.svg)

![gradient-boosting2 block 4 fig 4](/images/basic/ensemble/gradient-boosting2_block04_fig04.svg)

![gradient-boosting2 block 4 fig 5](/images/basic/ensemble/gradient-boosting2_block04_fig05.svg)

**解説**  
- 1本目：大まかに残差を補正  
- 2〜3本目：細かいパターンに対応  
- 5本目：だいぶ曲線に近づいてくる  
- さらに木を追加すると、最終的に赤い曲線（完成版）になります

---

## まとめ

- 勾配ブースティングは「初期値からスタートして、木を少しずつ積み上げる」手法。  
- 各木は「残差を補正する役割」を持ち、累積すると複雑な予測が可能になる。  
- 可視化すると、<b>「段階的に改良していくプロセス」</b>が直感的に理解できる。  

> 💡 ポイント：  
> - 木ごとの寄与を意識すると「学習率」「木の数」の意味が理解しやすくなる。  
> - 学習率が小さいと一歩ずつ修正（多くの木が必要）。大きいと大胆に修正（少ない木で済むが過学習リスク）。  

---
