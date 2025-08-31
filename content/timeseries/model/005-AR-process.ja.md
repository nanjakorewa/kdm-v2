---
title: AR過程
weight: 5
pre: "<b>5.3.1 </b>"
---

{{% youtube "J418ZrK5p08" %}}

<div class="pagetop-box">
ARモデルは自己回帰モデルとも呼ばれるものです。
自己回帰という名の通り、時点 t におけるモデルの出力は時点 t 以前の自分自身の出力に依存する確率過程のことを指しています。
</div>

```python
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
```

## AR過程のデータを作成する
データを生成するための関数を用意します


```python
def create_ARdata(phis=[0.1], N=500, init=1, c=1, sigma=0.3):
    """AR過程のデータを作成する"""
    print(f"==AR({len(phis)})過程の長さ{N}のデータを作成==")
    data = np.zeros(N)
    data[0] = init + np.random.normal(0, sigma)

    for t in range(2, N):
        res = c + np.random.normal(0, sigma)
        for j, phi_j in enumerate(phis):
            res += phi_j * data[t - j - 1]
        data[t] = res
    return data
```

### 係数が1より小さい場合


```python
plt.figure(figsize=(12, 6))
phis = [0.1]
ar1_1 = create_ARdata(phis=phis)
plt.plot(ar1_1)
plt.title(f"係数が{phis[0]}である場合", fontsize=15)
plt.show()
```

    ==AR(1)過程の長さ500のデータを作成==



    
![png](/images/timeseries/model/005-AR-process_files/005-AR-process_4_1.png)
    


### 係数が１である場合


```python
plt.figure(figsize=(12, 6))
phis = [1]
ar1_2 = create_ARdata(phis=phis)
plt.plot(ar1_2)
plt.title(f"係数が{phis[0]}である場合", fontsize=15)
plt.show()
```

    ==AR(1)過程の長さ500のデータを作成==



    
![png](/images/timeseries/model/005-AR-process_files/005-AR-process_6_1.png)
    


### 係数が１より大きい場合


```python
plt.figure(figsize=(12, 6))
phis = [1.04]
ar1_2 = create_ARdata(phis=phis)
plt.plot(ar1_2)
plt.title(f"係数が{phis[0]}である場合", fontsize=15)
plt.show()
```

    ==AR(1)過程の長さ500のデータを作成==



    
![png](/images/timeseries/model/005-AR-process_files/005-AR-process_8_1.png)
    


## AR(2)


```python
plt.figure(figsize=(12, 6))
phis = [0.1, 0.3]
ar2_1 = create_ARdata(phis=phis, N=100)
plt.plot(ar2_1)
plt.title(f"係数が{phis}である場合", fontsize=15)
plt.show()
```

    ==AR(2)過程の長さ100のデータを作成==



    
![png](/images/timeseries/model/005-AR-process_files/005-AR-process_10_1.png)
    



```python
plt.figure(figsize=(12, 6))
phis = [0.1, -1.11]
ar2_1 = create_ARdata(phis=phis)
plt.plot(ar2_1)
plt.title(f"係数が{phis}である場合", fontsize=15)
plt.show()
```

    ==AR(2)過程の長さ500のデータを作成==



    
![png](/images/timeseries/model/005-AR-process_files/005-AR-process_11_1.png)
    


## モデルの推定


```python
from statsmodels.tsa.ar_model import AutoReg

res = AutoReg(ar1_1, lags=1).fit()

out = "AIC: {0:0.3f}, HQIC: {1:0.3f}, BIC: {2:0.3f}"
print(out.format(res.aic, res.hqic, res.bic))
```

    AIC: 231.486, HQIC: 236.445, BIC: 244.124



```python
print(res.params)
print(res.sigma2)
res.summary()
```

    [1.03832755 0.07236388]
    0.09199676371696269





<table class="simpletable">
<caption>AutoReg Model Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>y</td>        <th>  No. Observations:  </th>    <td>500</td>  
</tr>
<tr>
  <th>Model:</th>            <td>AutoReg(1)</td>    <th>  Log Likelihood     </th> <td>-112.743</td>
</tr>
<tr>
  <th>Method:</th>         <td>Conditional MLE</td> <th>  S.D. of innovations</th>   <td>0.303</td> 
</tr>
<tr>
  <th>Date:</th>          <td>Sat, 13 Aug 2022</td> <th>  AIC                </th>  <td>231.486</td>
</tr>
<tr>
  <th>Time:</th>              <td>01:55:17</td>     <th>  BIC                </th>  <td>244.124</td>
</tr>
<tr>
  <th>Sample:</th>                <td>1</td>        <th>  HQIC               </th>  <td>236.445</td>
</tr>
<tr>
  <th></th>                      <td>500</td>       <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>    1.0383</td> <td>    0.052</td> <td>   20.059</td> <td> 0.000</td> <td>    0.937</td> <td>    1.140</td>
</tr>
<tr>
  <th>y.L1</th>  <td>    0.0724</td> <td>    0.045</td> <td>    1.621</td> <td> 0.105</td> <td>   -0.015</td> <td>    0.160</td>
</tr>
</table>
<table class="simpletable">
<caption>Roots</caption>
<tr>
    <td></td>   <th>            Real</th>  <th>         Imaginary</th> <th>         Modulus</th>  <th>        Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>          13.8190</td> <td>          +0.0000j</td> <td>          13.8190</td> <td>           0.0000</td>
</tr>
</table>


