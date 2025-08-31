---
title: "RuleFit"
pre: "2.3.4 "
weight: 4
---

{{% notice ref %}}
Friedman, Jerome H and Bogdan E Popescu. “Predictive learning via rule ensembles.” The Annals of Applied Statistics. JSTOR, 916–54. (2008).([pdf](https://jerryfriedman.su.domains/ftp/RuleFit.pdf))
{{% /notice %}}

### 実験用のデータを取得する
openmlで公開されている CC0 Public Domain のデータセット[house_sales
](https://www.openml.org/d/42092) データセットを使用して回帰モデルを作成します。

{{% notice info %}}
上記openmlページではデータの出典が不明ですが自分が調べた限りではデータの提供元は[ここ](https://gis-kingcounty.opendata.arcgis.com/datasets/zipcodes-for-king-county-and-surrounding-area-shorelines-zipcode-shore-area/explore?location=47.482924%2C-121.477600%2C8.00&showTable=true)のようです。
{{% /notice %}}

{{% notice document %}}
[sklearn.datasets.fetch_openml](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)
{{% /notice %}}

```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

dataset = fetch_openml(data_id=42092)
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
X = X.select_dtypes("number")
y = dataset.target
```
### データの中身を確認する


```python
X.head(10)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>1.00</td>
      <td>1180.0</td>
      <td>5650.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>1180.0</td>
      <td>0.0</td>
      <td>1955.0</td>
      <td>0.0</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340.0</td>
      <td>5650.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>2.25</td>
      <td>2570.0</td>
      <td>7242.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>2170.0</td>
      <td>400.0</td>
      <td>1951.0</td>
      <td>1991.0</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690.0</td>
      <td>7639.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>1.00</td>
      <td>770.0</td>
      <td>10000.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>770.0</td>
      <td>0.0</td>
      <td>1933.0</td>
      <td>0.0</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720.0</td>
      <td>8062.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>3.00</td>
      <td>1960.0</td>
      <td>5000.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>1050.0</td>
      <td>910.0</td>
      <td>1965.0</td>
      <td>0.0</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360.0</td>
      <td>5000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>2.00</td>
      <td>1680.0</td>
      <td>8080.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>1680.0</td>
      <td>0.0</td>
      <td>1987.0</td>
      <td>0.0</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800.0</td>
      <td>7503.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.0</td>
      <td>4.50</td>
      <td>5420.0</td>
      <td>101930.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>3890.0</td>
      <td>1530.0</td>
      <td>2001.0</td>
      <td>0.0</td>
      <td>47.6561</td>
      <td>-122.005</td>
      <td>4760.0</td>
      <td>101930.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.0</td>
      <td>2.25</td>
      <td>1715.0</td>
      <td>6819.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>1715.0</td>
      <td>0.0</td>
      <td>1995.0</td>
      <td>0.0</td>
      <td>47.3097</td>
      <td>-122.327</td>
      <td>2238.0</td>
      <td>6819.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.0</td>
      <td>1.50</td>
      <td>1060.0</td>
      <td>9711.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>1060.0</td>
      <td>0.0</td>
      <td>1963.0</td>
      <td>0.0</td>
      <td>47.4095</td>
      <td>-122.315</td>
      <td>1650.0</td>
      <td>9711.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3.0</td>
      <td>1.00</td>
      <td>1780.0</td>
      <td>7470.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>1050.0</td>
      <td>730.0</td>
      <td>1960.0</td>
      <td>0.0</td>
      <td>47.5123</td>
      <td>-122.337</td>
      <td>1780.0</td>
      <td>8113.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3.0</td>
      <td>2.50</td>
      <td>1890.0</td>
      <td>6560.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>1890.0</td>
      <td>0.0</td>
      <td>2003.0</td>
      <td>0.0</td>
      <td>47.3684</td>
      <td>-122.031</td>
      <td>2390.0</td>
      <td>7570.0</td>
    </tr>
  </tbody>
</table>
</div>



### RuleFitを実行する
[Python implementation of the rulefit algorithm - GitHub](https://github.com/christophM/rulefit)の実装を使用してRuleFitを動かしてみます。

※実行する際は `import warnings;warnings.simplefilter('ignore')` は外してください


```python
from rulefit import RuleFit
import warnings

warnings.simplefilter("ignore")  ## ConvergenceWarning

rf = RuleFit(max_rules=100)
rf.fit(X.values, y, feature_names=list(X.columns))
```




    RuleFit(max_rules=100,
            tree_generator=GradientBoostingRegressor(learning_rate=0.01,
                                                     max_depth=100,
                                                     max_leaf_nodes=5,
                                                     n_estimators=28,
                                                     random_state=27,
                                                     subsample=0.04543939429397564))



### 作成されたルールを確認する


```python
rules = rf.get_rules()
rules = rules[rules.coef != 0].sort_values(by="importance", ascending=False)
rules.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rule</th>
      <th>type</th>
      <th>coef</th>
      <th>support</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>grade</td>
      <td>linear</td>
      <td>6.199314e+04</td>
      <td>1.000000</td>
      <td>66184.725645</td>
    </tr>
    <tr>
      <th>29</th>
      <td>sqft_living &gt; 9475.0</td>
      <td>rule</td>
      <td>1.927942e+06</td>
      <td>0.001018</td>
      <td>61491.753935</td>
    </tr>
    <tr>
      <th>43</th>
      <td>grade &gt; 8.5 &amp; sqft_living &gt; 3405.0 &amp; long &lt;= -...</td>
      <td>rule</td>
      <td>3.570118e+05</td>
      <td>0.024440</td>
      <td>55126.384264</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sqft_living</td>
      <td>linear</td>
      <td>5.532732e+01</td>
      <td>1.000000</td>
      <td>46347.924165</td>
    </tr>
    <tr>
      <th>11</th>
      <td>yr_built</td>
      <td>linear</td>
      <td>-1.522004e+03</td>
      <td>1.000000</td>
      <td>44393.726859</td>
    </tr>
    <tr>
      <th>15</th>
      <td>sqft_living15</td>
      <td>linear</td>
      <td>5.344916e+01</td>
      <td>1.000000</td>
      <td>34501.058499</td>
    </tr>
    <tr>
      <th>62</th>
      <td>lat &lt;= 47.516000747680664 &amp; sqft_living &lt;= 3920.0</td>
      <td>rule</td>
      <td>-6.549757e+04</td>
      <td>0.361507</td>
      <td>31467.457947</td>
    </tr>
    <tr>
      <th>103</th>
      <td>sqft_basement &lt;= 3660.0 &amp; grade &gt; 9.5</td>
      <td>rule</td>
      <td>1.240216e+05</td>
      <td>0.068228</td>
      <td>31270.434139</td>
    </tr>
    <tr>
      <th>48</th>
      <td>sqft_living &lt;= 9475.0 &amp; grade &gt; 9.5 &amp; long &gt; -...</td>
      <td>rule</td>
      <td>-1.473030e+05</td>
      <td>0.040733</td>
      <td>29117.596559</td>
    </tr>
    <tr>
      <th>67</th>
      <td>sqft_living &lt;= 4695.0 &amp; waterfront &gt; 0.5 &amp; sqf...</td>
      <td>rule</td>
      <td>3.936285e+05</td>
      <td>0.005092</td>
      <td>28016.079499</td>
    </tr>
  </tbody>
</table>
</div>



### ルールが正しいか確認してみる

`sqft_above	linear	8.632149e+01	1.000000	66243.550192` のルールに基づいて、`sqft_above` が増加すると y(`price`)が増える傾向にあるかどうか確認します。

{{% notice document %}}
[matplotlib.pyplot.boxplot — Matplotlib 3.5.1 documentation](https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.boxplot.html)
{{% /notice %}}


```python
plt.figure(figsize=(6, 6))
plt.scatter(X["sqft_above"], y, marker=".")
plt.xlabel("sqft_above")
plt.ylabel("price")
```

    
![png](/images/basic/tree/RuleFit_files/RuleFit_12_1.png)
    


`sqft_living <= 3935.0 & lat <= 47.5314998626709	rule	-8.271074e+04	0.377800	40101.257833` のルールに該当するデータのみ抽出してみます。
係数がマイナスになっているので、このルールに該当するデータのy(`price`)は低い傾向にあるはずです。
log(y)を箱髭図で確認すると、確かにルールに該当しているデータのyはルールに該当しないデータのyと比較すると低くなっています。


```python
applicable_data = np.log(
    y[X.query("sqft_living <= 3935.0 & lat <= 47.5314998626709").index]
)
not_applicable_data = np.log(
    y[X.query("not(sqft_living <= 3935.0 & lat <= 47.5314998626709)").index]
)

plt.figure(figsize=(10, 6))
plt.boxplot([applicable_data, not_applicable_data], labels=["ルールに該当", "ルールに該当しない"])
plt.ylabel("log(price)")
plt.grid()
plt.show()
```


    
![png](/images/basic/tree/RuleFit_files/RuleFit_14_0.png)
    

