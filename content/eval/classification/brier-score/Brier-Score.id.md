---

title: "Brier Score | Mengukur kalibrasi probabilitas"

linkTitle: "Brier Score"

seo_title: "Brier Score | Mengukur kalibrasi probabilitas"

pre: "4.3.10 "

weight: 10

---



{{< lead >}}

Brier Score adalah selisih kuadrat antara probabilitas yang diprediksi dan label aktual (0/1). Metrik ini menunjukkan seberapa baik model mengkalibrasi probabilitasnya—penting ketika keputusan bisnis bergantung pada nilai peluang tersebut. Berikut cara menghitung dan memvisualisasikannya di Python 3.13.

{{< /lead >}}



---



## 1. Definisi



Untuk klasifikasi biner, Brier Score ditulis sebagai





\mathrm{Brier} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2,





dengan \(p_i\) adalah probabilitas kelas positif dan \(y_i\) label aktual (0 atau 1). Pada kasus multikelas, error kuadrat dihitung per kelas lalu dirata-rata.



---



## 2. Implementasi dan visualisasi di Python 3.13



```bash

python --version        # contoh: Python 3.13.0

pip install scikit-learn matplotlib

```



Contoh berikut melatih regresi logistik pada dataset Breast Cancer, menghitung Brier Score, dan menggambar diagram reliabilitas. Gambar disimpan di static/images/eval/classification/brier-score/reliability_curve.png sehingga dapat diregenerasi lewat generate_eval_assets.py.



```python

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.calibration import CalibrationDisplay

from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import brier_score_loss

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, stratify=y, random_state=42

)



pipeline = make_pipeline(

    StandardScaler(),

    LogisticRegression(max_iter=2000, solver="lbfgs"),

)

pipeline.fit(X_train, y_train)

proba = pipeline.predict_proba(X_test)[:, 1]



score = brier_score_loss(y_test, proba)

print(f"Brier Score: {score:.3f}")



fig, ax = plt.subplots(figsize=(5, 5))

CalibrationDisplay.from_predictions(y_test, proba, n_bins=10, ax=ax)

ax.set_title("Reliability Diagram (Breast Cancer Dataset)")

fig.tight_layout()

output_dir = Path("static/images/eval/classification/brier-score")

output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(output_dir / "reliability_curve.png", dpi=150)

plt.close(fig)

```



{{< figure src="/images/eval/classification/brier-score/reliability_curve.png" alt="Diagram reliabilitas" caption="Penyimpangan dari garis 45° menandakan probabilitas yang terlalu percaya diri atau terlalu ragu." >}}



---



## 3. Cara membaca skor



- Kalibrasi sempurna menghasilkan **0**.

- Model yang selalu memprediksi 0.5 (pada data seimbang) mencapai **0.25**.

- Nilai semakin kecil semakin baik; probabilitas yang jauh dari hasil aktual akan menambah penalti secara kuadrat.



---



## 4. Diagnostik kalibrasi dengan reliabilitas diagram



Diagram reliabilitas mengelompokkan probabilitas ke dalam beberapa bin, lalu menampilkan rata-rata probabilitas yang diprediksi versus frekuensi aktual.



- Titik **di bawah** garis diagonal → model terlalu percaya diri (probabilitas terlalu tinggi).

- Titik **di atas** garis diagonal → model kurang percaya diri.

- Setelah menerapkan teknik kalibrasi (Platt scaling, isotonic regression, dsb.), hitung kembali Brier Score dan diagram ini untuk memastikan adanya perbaikan.



---



## Ringkasan



- Brier Score mengukur kesalahan kuadrat rata-rata dari probabilitas; semakin kecil semakin baik.

- Dengan Python 3.13, rier_score_loss dan diagram reliabilitas memberikan inspeksi kalibrasi secara cepat.

- Gunakan bersama ROC-AUC serta Precision/Recall agar evaluasi mencakup kualitas ranking sekaligus akurasi probabilitas.

---

