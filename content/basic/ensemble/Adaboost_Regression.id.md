---
title: "Adaboost (Regresi)"
pre: "2.4.4 "
weight: 4
title_suffix: "Penjelasan Mekanisme"
---

{{% youtube "1K-h4YzrnsY" %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
```


```python
# NOTE: Model yang dibuat untuk memeriksa sample_weight model
class DummyRegressor:
    def __init__(self):
        self.model = DecisionTreeRegressor(max_depth=5)
        self.error_vector = None
        self.X_for_plot = None
        self.y_for_plot = None

    def fit(self, X, y):
        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        # Bobot dihitung berdasarkan kesalahan regresi
        # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_weight_boosting.py#L1130
        self.error_vector = np.abs(y_pred - y)
        self.X_for_plot = X.copy()
        self.y_for_plot = y.copy()
        return self.model

    def predict(self, X, check_input=True):
        return self.model.predict(X)

    def get_params(self, deep=False):
        return {}

    def set_params(self, deep=False):
        return {}
```

## Memasukkan Model Regresi ke Data Pelatihan

```python
# Data pelatihan
X = np.linspace(-10, 10, 500)[:, np.newaxis]
y = (np.sin(X).ravel() + np.cos(4 * X).ravel()) * 10 + 10 + np.linspace(-2, 2, 500)

## Membuat model regresi
reg = AdaBoostRegressor(
    DummyRegressor(),
    n_estimators=100,
    random_state=100,
    loss="linear",
    learning_rate=0.8,
)
reg.fit(X, y)
y_pred = reg.predict(X)

# Memeriksa kecocokan model pada data pelatihan
plt.figure(figsize=(10, 5))
plt.scatter(X, y, c="k", marker="x", label="Data pelatihan")
plt.plot(X, y_pred, c="r", label="Prediksi model akhir", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Kecocokan pada Data Pelatihan")
plt.legend()
plt.show()
```

    
![png](/images/basic/ensemble/Adaboost_Regression_files/Adaboost_Regression_6_0.png)
    


## Memvisualisasikan Bobot Sampel (ketika `loss='linear'`)
Dalam Adaboost, bobot ditentukan berdasarkan kesalahan regresi. Di bawah pengaturan `loss='linear'`, kita akan memvisualisasikan besarnya bobot. Data dengan bobot yang lebih besar memiliki kemungkinan lebih tinggi untuk dipilih selama pelatihan.

> loss{‘linear’, ‘square’, ‘exponential’}, default=’linear’
> The loss function to use when updating the weights after each boosting iteration.

{{% notice document %}}
[sklearn.ensemble.AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor)
{{% /notice %}}


```python
def visualize_weight(reg, X, y, y_pred):
    """Fungsi untuk memplot nilai yang sesuai dengan bobot sampel (jumlah kemunculan data yang disampling)

    Parameters
    ----------
    reg : sklearn.ensemble._weight_boosting
        Model boosting
    X : numpy.ndarray
        Data pelatihan
    y : numpy.ndarray
        Data target
    y_pred:
        Prediksi model
    """
    assert reg.estimators_ is not None, "len(reg.estimators_) > 0"

    for i, estimators_i in enumerate(reg.estimators_):
        if i % 100 == 0:
            # Hitung jumlah kemunculan data yang digunakan dalam pembuatan model ke-i
            weight_dict = {xi: 0 for xi in X.ravel()}
            for xi in estimators_i.X_for_plot.ravel():
                weight_dict[xi] += 1

            # Plot jumlah kemunculan dengan lingkaran oranye (semakin besar lingkaran, semakin sering muncul)
            weight_x_sorted = sorted(weight_dict.items(), key=lambda x: x[0])
            weight_vec = np.array([s * 100 for xi, s in weight_x_sorted])

            # Plot grafik
            plt.figure(figsize=(10, 5))
            plt.title(f"Visualisasi Sampel Berbobot Setelah Model ke-{i}, loss={reg.loss}")
            plt.scatter(X, y, c="k", marker="x", label="Data Pelatihan")
            plt.scatter(
                estimators_i.X_for_plot,
                estimators_i.y_for_plot,
                marker="o",
                alpha=0.2,
                c="orange",
                s=weight_vec,
            )
            plt.plot(X, y_pred, c="r", label="Prediksi Model Akhir", linewidth=2)
            plt.legend(loc="upper right")
            plt.show()


## Membuat model regresi dengan loss="linear"
reg = AdaBoostRegressor(
    DummyRegressor(),
    n_estimators=101,
    random_state=100,
    loss="linear",
    learning_rate=1,
)
reg.fit(X, y)
y_pred = reg.predict(X)
visualize_weight(reg, X, y, y_pred)
```


    
![png](/images/basic/ensemble/Adaboost_Regression_files/Adaboost_Regression_8_0.png)
    



    
![png](/images/basic/ensemble/Adaboost_Regression_files/Adaboost_Regression_8_1.png)
    



```python
## Membuat Model Regresi dengan `loss="square"`
reg = AdaBoostRegressor(
    DummyRegressor(),
    n_estimators=101,
    random_state=100,
    loss="square",
    learning_rate=1,
)
reg.fit(X, y)
y_pred = reg.predict(X)
visualize_weight(reg, X, y, y_pred)
```


    
![png](/images/basic/ensemble/Adaboost_Regression_files/Adaboost_Regression_9_0.png)
    



    
![png](/images/basic/ensemble/Adaboost_Regression_files/Adaboost_Regression_9_1.png)
    

