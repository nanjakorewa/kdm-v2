---
title: "Singular Value Decomposition | Matrix Factorization for Compression"
linkTitle: "SVD"
seo_title: "Singular Value Decomposition | Matrix Factorization for Compression"
pre: "2.6.2 "
weight: 2
title_suffix: "の仕組みの説明"
---

{{% youtube "ni1atKuoAcE" %}}

<div class="pagetop-box">
    <p>SVD (Singular Value Decomposition) is a method of decomposing a matrix into a product of an orthogonal matrix and a diagonal matrix, and can be viewed as a type of dimensionality reduction algorithm. Here, we will try to use SVD to represent image data in a lower dimensional data.</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy import linalg
from PIL import Image
```

## Data for experiment

Let's consider the image of the word "実験" as a series of data (grayscale, so [0, 0, 1, 1, 0 ...]) ) and decompose it into singular values.

{{% notice document %}}
[https://docs.scipy.org/doc/scipy/tutorial/linalg.html](https://docs.scipy.org/doc/scipy/tutorial/linalg.html)
{{% /notice %}}


```python
img = Image.open("./sample.png").convert("L").resize((163, 372)).rotate(90, expand=True)
img
```


    
![png](/images/basic/dimensionality_reduction/SVD_files/SVD_4_0.png)
    


## Perform singular value decomposition


```python
## Decompose matrix X into singular values
X = np.asarray(img)
U, Sigma, VT = linalg.svd(X, full_matrices=True)

print(f"A: {X.shape}, U: {U.shape}, Σ:{Sigma.shape}, V^T:{VT.shape}")
```

    A: (163, 372), U: (163, 163), Σ:(163,), V^T:(372, 372)


## Approximate image with low rank


```python
# Increasing the rank restores the original characters more precisely
for rank in [1, 2, 3, 4, 5, 10, 20, 50]:
    # Extract elements up to the rank-th element
    U_i = U[:, :rank]
    Sigma_i = np.matrix(linalg.diagsvd(Sigma[:rank], rank, rank))
    VT_i = VT[:rank, :]

    # Restore image
    temp_image = np.asarray(U_i * Sigma_i * VT_i)
    Image.fromarray(np.uint8(temp_image))
    plt.title(f"rank={rank}")
    plt.imshow(temp_image, cmap="gray")
    plt.show()
```

## Contents of matrix V


```python
total = np.zeros((163, 372))

for rank in [1, 2, 3, 4, 5]:
    # Extract elements up to the rank-th element
    U_i = U[:, :rank]
    Sigma_i = np.matrix(linalg.diagsvd(Sigma[:rank], rank, rank))
    VT_i = VT[:rank, :]

    # Set all values except the rank-th singular value to 0, leaving only the rank-th element.
    if rank > 1:
        for ri in range(rank - 1):
            Sigma_i[ri, ri] = 0

    # Restore image
    temp_image = np.asarray(U_i * Sigma_i * VT_i)
    Image.fromarray(np.uint8(temp_image))

    # Add only the rank-th element
    total += temp_image

    plt.figure(figsize=(5, 5))
    plt.suptitle(f"$u_{rank}$")
    plt.subplot(211)
    plt.imshow(temp_image, cmap="gray")
    plt.subplot(212)
    plt.plot(VT[0])
    plt.show()
```


```python
# Confirm that adding the first through the fifth elements together properly restores the original image
plt.imshow(total)
```
