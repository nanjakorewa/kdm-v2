---

title: "Average Precision (AP) | Menilai kurva precision–recall"

linkTitle: "Average Precision"

seo_title: "Average Precision (AP) | Menilai kurva precision–recall"

pre: "4.3.9 "

weight: 9

---



{{< lead >}}

Average Precision (AP) merangkum kurva precision–recall dengan memberi bobot pada precision sesuai kenaikan recall. Metrik ini menggambarkan perilaku model pada seluruh rentang ambang dan sangat membantu saat data tidak seimbang. Berikut cara menghitungnya di Python 3.13 dan mengapa ia melengkapi F1 serta ROC-AUC.

{{< /lead >}}



---



## 1. Definisi



Jika kurva PR berisi titik \((R_n, P_n)\), maka average precision didefinisikan sebagai:





\mathrm{AP} = \sum_{n}(R_n - R_{n-1}) P_n





Kenaikan recall menjadi bobot, sehingga AP mencerminkan rata-rata precision saat ambang diturunkan dari tinggi ke rendah.



---



## 2. Perhitungan di Python 3.13



```bash

python --version        # contoh: Python 3.13.0

pip install scikit-learn matplotlib

```



Memanfaatkan probabilitas proba dari contoh precision–recall, kita menghitung AP dengan scikit-learn:



```python

from sklearn.metrics import precision_recall_curve, average_precision_score



precision, recall, thresholds = precision_recall_curve(y_test, proba)

ap = average_precision_score(y_test, proba)

print(f"Average Precision: {ap:.3f}")

```



Kurva PR yang terkait adalah pr_curve.png yang dibuat pada langkah sebelumnya.



{{< figure src="/images/eval/classification/precision-recall/pr_curve.png" alt="Kurva precision–recall" caption="AP mengintegrasikan kurva PR dengan bobot kenaikan recall." >}}



---



## 3. AP vs. PR-AUC



- verage_precision_score menggunakan integrasi bertingkat seperti pada sistem pencarian informasi.

- sklearn.metrics.auc(recall, precision) memakai aturan trapesium dan menghasilkan PR-AUC klasik.

- AP cenderung lebih stabil di data tidak seimbang karena menekankan bagian kurva saat recall meningkat.



---



## 4. Catatan praktis



- **Penentuan ambang** – AP tinggi berarti model menjaga precision tinggi dalam jangkauan recall yang luas.

- **Tugas ranking** – Pada rekomendasi dan pencarian, Mean Average Precision (MAP) dipakai dengan merata-rata AP per kueri.

- **Pelengkap F1** – F1 mengukur satu titik operasi, sedangkan AP mengevaluasi seluruh spektrum ambang.



---



## Ringkasan



- Average Precision mengevaluasi keseluruhan kurva precision–recall dan cocok untuk data tidak seimbang.

- Dengan scikit-learn di Python 3.13, verage_precision_score memudahkan perhitungan AP.

- Gunakan bersamaan dengan F1, ROC-AUC, dan visualisasi PR untuk membandingkan model serta memilih ambang yang tepat.

---

