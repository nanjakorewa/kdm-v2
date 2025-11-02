---

title: "Log Loss | Menilai kualitas probabilitas prediksi"

linkTitle: "Log Loss"

seo_title: "Log Loss | Menilai kualitas probabilitas prediksi"

pre: "4.3.4 "

weight: 4

---



{{< lead >}}

Log Loss (cross-entropy) menghukum probabilitas yang salah secara eksponensial. Ini adalah metrik pilihan ketika probabilitas terkalibrasi sangat penting. Berikut cara menghitungnya di Python 3.13 dan penjelasan tentang kurva penalti.

{{< /lead >}}



---



## 1. Definisi



Untuk klasifikasi biner:





\mathrm{LogLoss} = -\frac{1}{n} \sum_{i=1}^{n} \left[y_i \log(p_i) + (1 - y_i) \log(1 - p_i)\right],





dengan \(p_i\) probabilitas kelas positif dan \(y_i\) label sebenarnya (0 atau 1). Pada multikelas, probabilitas masing-masing kelas dijumlahkan dengan label one-hot.



---



## 2. Perhitungan di Python 3.13



```bash

python --version        # contoh: Python 3.13.0

pip install scikit-learn matplotlib

```



```python

from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, log_loss

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, stratify=y, random_state=42

)



model = make_pipeline(

    StandardScaler(),

    LogisticRegression(max_iter=2000, solver="lbfgs"),

)

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)



print(classification_report(y_test, model.predict(X_test), digits=3))

print("Log Loss:", log_loss(y_test, proba))

```



Cukup oper probabilitas dari predict_proba ke log_loss.



---



## 3. Memahami penalti log loss



{{< figure src="/images/eval/classification/log-loss/log_loss_curves.png" alt="Kurva penalti log loss" caption="Probabilitas yang jauh dari label sebenarnya dikenai penalti tinggi." >}}



- Memberi probabilitas kecil pada sampel positif (misal 0.1) menghasilkan penalti besar.

- Mengembalikan 0.5 untuk semua sampel juga dihukum karena model tidak membedakan kelas.



---



## 4. Kapan menggunakan Log Loss



- **Memeriksa kalibrasi** – setelah Platt scaling atau isotonic regression, pastikan Log Loss turun.

- **Kompetisi dan leaderboard** – banyak lomba machine learning menggunakan Log Loss sebagai metrik evaluasi.

- **Perbandingan tanpa ambang** – berbeda dari Accuracy, Log Loss menilai distribusi probabilitas secara penuh.



Parameter labels, eps, dan 

ormalize pada log_loss membantu menangani label yang hilang atau masalah stabilitas numerik.



---



## Ringkasan



- Log Loss mengukur seberapa jauh probabilitas prediksi dari hasil nyata; semakin kecil nilainya semakin baik.

- Di Python 3.13 cukup gunakan predict_proba dan log_loss dari scikit-learn.

- Kombinasikan dengan ROC-AUC atau kurva PR untuk mengevaluasi kemampuan diskriminasi sekaligus kualitas probabilitas.

---

