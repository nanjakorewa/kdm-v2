---
title: "SVD"
pre: "2.6.2 "
weight: 2
title_suffix: "Penjelasan Mekanisme"
---

{{% youtube "ni1atKuoAcE" %}}

<div class="pagetop-box">
    <p><b>SVD (Singular Value Decomposition)</b> adalah metode untuk mendekomposisi matriks menjadi perkalian antara matriks ortogonal dan matriks diagonal. SVD juga dapat dianggap sebagai salah satu algoritme untuk reduksi dimensi. Di sini, kita akan mencoba merepresentasikan data gambar menggunakan data berdimensi lebih rendah dengan bantuan SVD.</p>
</div>


```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy import linalg
from PIL import Image
```

## Data untuk Eksperimen
Kita akan menggunakan gambar karakter "実験" sebagai data yang diperlakukan sebagai serangkaian vektor (karena dalam skala abu-abu, data akan berupa kumpulan vektor seperti [0, 0, 1, 1, 0 ...]). Kemudian, data tersebut akan didekomposisi menggunakan Singular Value Decomposition (SVD).

{{% notice document %}}
[https://docs.scipy.org/doc/scipy/tutorial/linalg.html](https://docs.scipy.org/doc/scipy/tutorial/linalg.html)
{{% /notice %}}


```python
img = Image.open("./sample.png").convert("L").resize((163, 372)).rotate(90, expand=True)
img
```




    
![png](/images/basic/dimensionality_reduction/SVD_files/SVD_4_0.png)
    



## Eksekusi Singular Value Decomposition (SVD) pada matriks \( A \).


```python
## Matriks \( A \), yang berisi data karakter, akan didekomposisi menggunakan Singular Value Decomposition (SVD).
X = np.asarray(img)
U, Sigma, VT = linalg.svd(X, full_matrices=True)

print(f"A: {X.shape}, U: {U.shape}, Σ:{Sigma.shape}, V^T:{VT.shape}")
```

    A: (163, 372), U: (163, 163), Σ:(163,), V^T:(372, 372)


## Dekati gambar menggunakan representasi peringkat rendah.


```python
#Semakin Banyak Rank, Semakin Presisi Restorasi Gambar Asli
for rank in [1, 2, 3, 4, 5, 10, 20, 50]:
    # Ekstraksi elemen hingga peringkat tertentu
    U_i = U[:, :rank]
    Sigma_i = np.matrix(linalg.diagsvd(Sigma[:rank], rank, rank))
    VT_i = VT[:rank, :]

    # Rekonstruksi gambar
    temp_image = np.asarray(U_i * Sigma_i * VT_i)
    Image.fromarray(np.uint8(temp_image))
    plt.title(f"rank={rank}")
    plt.imshow(temp_image, cmap="gray")
    plt.show()
```

## Isi Matriks \( V \)

```python
total = np.zeros((163, 372))

for rank in [1, 2, 3, 4, 5]:
    # Ekstraksi elemen hingga peringkat tertentu
    U_i = U[:, :rank]
    Sigma_i = np.matrix(linalg.diagsvd(Sigma[:rank], rank, rank))
    VT_i = VT[:rank, :]

    # Setel semua nilai singular selain peringkat saat ini ke nol, hanya menyisakan elemen peringkat tertentu
    if rank > 1:
        for ri in range(rank - 1):
            Sigma_i[ri, ri] = 0

    # Rekonstruksi gambar
    temp_image = np.asarray(U_i * Sigma_i * VT_i)
    Image.fromarray(np.uint8(temp_image))

    # Tambahkan hanya elemen peringkat tertentu
    total += temp_image

    # Bandingkan gambar yang direkonstruksi menggunakan elemen hingga peringkat tertentu 
    # dengan plot nilai di kolom peringkat tertentu pada matriks \( V \)
    plt.figure(figsize=(5, 5))
    plt.suptitle(f"$u_{rank}$")
    plt.subplot(211)
    plt.imshow(temp_image, cmap="gray")
    plt.subplot(212)
    plt.plot(VT[0])
    plt.show()
```


```python
# Periksa bahwa dengan menjumlahkan elemen dari peringkat 1 hingga 5, gambar asli dapat direkonstruksi sepenuhnya
plt.imshow(total)
```
