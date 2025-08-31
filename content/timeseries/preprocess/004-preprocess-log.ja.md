---
title: Box-Cox変換
weight: 5
pre: "<b>5.1.4 </b>"
searchtitle: "Box-Cox変換でデータの分布を正規分布に近づける"
---

{{% youtube "sMKsXqFWo-Q" %}}


<div class="pagetop-box">
    <p>モデルを使って時系列データを分析するためには前処理が必要な場合があります。時系列モデルはどんなデータでも分析可能というわけではなく、「分散が常に一定」「正規分布に従っている」などの仮定をおいていることが多いからです。</p>
    <p>ここではBox-Cox変換を用いて、少し偏りのあるデータを正規分布に近い形に変換し、それがモデルの出力（正解と予測値の誤差の分布）にどのような影響があるかを見てみます。</p>
</div>


```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

```python
from scipy import stats
import numpy as np

plt.figure(figsize=(12, 5))
data_wb = np.random.weibull(2.0, size=50000)
plt.hist(data_wb, bins=30, rwidth=0.9)
plt.show()


plt.figure(figsize=(12, 5))
data_lg = stats.loggamma.rvs(2.0, size=50000)
plt.hist(data_lg, bins=30, rwidth=0.9)
plt.show()
```


    
![png](/images/timeseries/preprocess/004-preprocess-log_files/004-preprocess-log_4_0.png)
    



    
![png](/images/timeseries/preprocess/004-preprocess-log_files/004-preprocess-log_4_1.png)
    


{{% notice document %}}
[scipy.stats.boxcox — SciPy v1.8.0 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html)
{{% /notice %}}

```python
from scipy.stats import boxcox

plt.figure(figsize=(12, 5))
plt.hist(boxcox(data_wb), bins=30, rwidth=0.9)
plt.show()
```


    
![png](/images/timeseries/preprocess/004-preprocess-log_files/004-preprocess-log_5_0.png)
    

```python
try:
    plt.figure(figsize=(12, 5))
    plt.hist(boxcox(data_lg), bins=30, rwidth=0.9)
    plt.show()
except ValueError as e:
    print(f"エラーの内容： ValueError {e.args}")
```

    エラーの内容： ValueError ('Data must be positive.',)



    <Figure size 864x360 with 0 Axes>




{{% notice document %}}
[scipy.stats.yeojohnson — SciPy v1.8.0 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html)
{{% /notice %}}

```python
from scipy.stats import yeojohnson

plt.figure(figsize=(12, 5))
plt.hist(yeojohnson(data_lg), bins=30, rwidth=0.9)
plt.show()
```


    
![png](/images/timeseries/preprocess/004-preprocess-log_files/004-preprocess-log_7_0.png)
    


## リッジ回帰をフィッティングしてみる

ｙの分布を正規分布に近づけずにリッジ回帰を適用した場合は、残差の分布に偏りがあることがわかります。


```python
from sklearn.linear_model import Ridge

N = 1000
rng = np.random.RandomState(0)
y = np.random.weibull(2.0, size=N)

X = rng.randn(N, 5)
X[:, 0] = np.sqrt(y) + np.random.rand(N) / 10

plt.figure(figsize=(12, 5))
plt.hist(y, bins=20, rwidth=0.9)
plt.title("yの分布")
plt.show()
```


    
![png](/images/timeseries/preprocess/004-preprocess-log_files/004-preprocess-log_9_0.png)
    



```python
clf = Ridge(alpha=1.0)
clf.fit(X, y)
pred = clf.predict(X)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title("正解と出力の分布")
plt.scatter(y, pred)
plt.plot([0, 2], [0, 2], "r")
plt.xlabel("正解")
plt.ylabel("出力")
plt.xlim(0, 2)
plt.ylim(0, 2)
plt.grid()
plt.subplot(122)
plt.title("残差の分布")
plt.hist(y - pred)
plt.xlim(-0.5, 0.5)
plt.show()
```


    
![png](/images/timeseries/preprocess/004-preprocess-log_files/004-preprocess-log_10_0.png)
    



```python
clf = Ridge(alpha=1.0)
clf.fit(X, yeojohnson(y)[0])
pred = clf.predict(X)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title("正解と出力の分布")
plt.scatter(yeojohnson(y)[0], pred)
plt.plot([0, 2], [0, 2], "r")
plt.xlabel("正解")
plt.ylabel("出力")
plt.xlim(0, 2)
plt.ylim(0, 2)
plt.grid()
plt.subplot(122)
plt.title("残差の分布")
plt.hist(yeojohnson(y)[0] - pred)
plt.xlim(-0.15, 0.15)
plt.show()
```


    
![png](/images/timeseries/preprocess/004-preprocess-log_files/004-preprocess-log_11_0.png)
    
