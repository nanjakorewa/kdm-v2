---

title: "ROC-AUC | Panduan menyesuaikan ambang keputusan dan membandingkan model"

linkTitle: "ROC-AUC"

seo_title: "ROC-AUC | Panduan menyesuaikan ambang keputusan dan membandingkan model"

pre: "4.3.3 "

weight: 1

---



{{< lead >}}

Kurva ROC menggambarkan hubungan antara true positive rate dan false positive rate ketika ambang keputusan berubah, sedangkan AUC (Area Under the Curve) merangkum kemampuan pemeringkatan model. Dengan Python 3.13 kita dapat memvisualkannya dan menggunakan hasilnya untuk memilih ambang yang tepat.

{{< /lead >}}



---



## 1. Makna kurva ROC dan AUC



Kurva ROC menempatkan **False Positive Rate (FPR)** pada sumbu x dan **True Positive Rate (TPR)** pada sumbu y sambil menggeser ambang prediksi dari 0 ke 1. Nilai AUC berada di kisaran 0.5 (acak) hingga 1.0 (pemisahan sempurna).



- AUC ≈ 1.0 → model sangat baik dalam membedakan kelas.

- AUC ≈ 0.5 → serupa dengan tebakan acak.

- AUC < 0.5 → mungkin tanda bahwa keputusan perlu dibalik atau ambang perlu disesuaikan signifikan.



---



## 2. Implementasi dan visualisasi di Python 3.13



Pastikan interpreter dan paket yang dibutuhkan telah tersedia:



```bash

python --version        # contoh: Python 3.13.0

pip install scikit-learn matplotlib

```



Kode di bawah ini melatih regresi logistik pada dataset Breast Cancer, menampilkan kurva ROC, dan menyimpan gambar ke static/images/eval/classification/rocauc. Alur ini kompatibel dengan generate_eval_assets.py untuk regenerasi otomatis.



```python

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import RocCurveDisplay, roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, random_state=42, stratify=y

)



pipeline = make_pipeline(

    StandardScaler(),

    LogisticRegression(max_iter=2000, solver="lbfgs"),

)

pipeline.fit(X_train, y_train)

proba = pipeline.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, proba)

print(f"ROC-AUC: {auc:.3f}")



fig, ax = plt.subplots(figsize=(5, 5))

RocCurveDisplay.from_predictions(

    y_test,

    proba,

    name="Logistic Regression",

    ax=ax,

)

ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5, label="Acak")

ax.set_xlabel("False Positive Rate")

ax.set_ylabel("True Positive Rate")

ax.set_title("Kurva ROC (Breast Cancer Dataset)")

ax.legend(loc="lower right")

fig.tight_layout()

output_dir = Path("static/images/eval/classification/rocauc")

output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(output_dir / "roc_curve.png", dpi=150)

plt.close(fig)

```



{{< figure src="/images/eval/classification/rocauc/roc_curve.png" alt="Kurva ROC" caption="AUC adalah luas di bawah kurva ROC; makin besar nilainya, makin baik kemampuan pemeringkatan model." >}}



---



## 3. Menggunakan ROC-AUC untuk menyesuaikan ambang



- **Kasus sensitif terhadap recall** (kesehatan, fraud): pilih titik pada kurva yang menjaga TPR tinggi dengan FPR yang masih dapat diterima.

- **Menjaga keseimbangan precision–recall**: model dengan AUC tinggi cenderung stabil di berbagai ambang.

- **Perbandingan model**: AUC memberikan ukuran tunggal sebelum menentukan ambang operasional.



Sertakan analisis Precision–Recall agar dampak perubahan ambang terhadap kualitas prediksi semakin jelas.



---



## 4. Checklist operasional



1. **Periksa ketidakseimbangan data** – walau AUC hanya sekitar 0.5, mungkin ada ambang lain yang menyelamatkan kasus penting.

2. **Uji bobot kelas** – lihat apakah memberi bobot berbeda meningkatkan AUC.

3. **Bagikan visualisasinya** – tampilkan kurva ROC di dashboard agar tim mudah memahami trade-off.

4. **Notebook Python 3.13** – simpan evaluasi agar dapat dijalankan ulang setiap kali model diperbarui.



---



## Ringkasan



- ROC-AUC menilai kemampuan model mengurutkan kelas positif di depan kelas negatif pada seluruh rentang ambang.

- Di Python 3.13, gunakan RocCurveDisplay dan 

oc_auc_score untuk menghitung serta memvisualisasikannya dengan ringkas.

- Kombinasikan dengan metrik precision–recall guna memilih ambang yang sejalan dengan kebutuhan bisnis.

---

