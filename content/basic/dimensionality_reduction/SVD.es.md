---
title: "SVD"
pre: "2.6.2 "
weight: 2
title_suffix: "Explicación del funcionamiento"
---

{{% youtube "ni1atKuoAcE" %}}

<div class="pagetop-box">
    <p><b>SVD (Descomposición en Valores Singulares)</b> es un método que descompone una matriz en el producto de matrices ortogonales y una matriz diagonal. También puede considerarse un tipo de algoritmo de reducción de dimensionalidad. Aquí, intentaremos representar datos de imágenes utilizando datos de menor dimensionalidad mediante SVD.</p>
</div>


```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy import linalg
from PIL import Image
```

## Datos para experimentos
Tomaremos una imagen con el texto "実験" y la trataremos como una serie de datos (en escala de grises, representados como una colección de vectores como [0, 0, 1, 1, 0 ...]) para realizar la descomposición en valores singulares.


{{% notice document %}}
[https://docs.scipy.org/doc/scipy/tutorial/linalg.html](https://docs.scipy.org/doc/scipy/tutorial/linalg.html)
{{% /notice %}}


```python
img = Image.open("./sample.png").convert("L").resize((163, 372)).rotate(90, expand=True)
img
```




    
![png](/images/basic/dimensionality_reduction/SVD_files/SVD_4_0.png)
    

## Realizar la descomposición en valores singulares

```python
## Realizar la descomposición en valores singulares de la matriz A que contiene los datos de la imagen del texto
X = np.asarray(img)
U, Sigma, VT = linalg.svd(X, full_matrices=True)

print(f"A: {X.shape}, U: {U.shape}, Σ:{Sigma.shape}, V^T:{VT.shape}")
```

    A: (163, 372), U: (163, 163), Σ:(163,), V^T:(372, 372)


## Aproximación de la imagen con rango reducido

```python
# A medida que se aumenta el rango, el texto original se restaura con mayor precisión
for rank in [1, 2, 3, 4, 5, 10, 20, 50]:
    # Extraer los elementos hasta el rango especificado
    U_i = U[:, :rank]
    Sigma_i = np.matrix(linalg.diagsvd(Sigma[:rank], rank, rank))
    VT_i = VT[:rank, :]

    # Restaurar la imagen
    temp_image = np.asarray(U_i * Sigma_i * VT_i)
    Image.fromarray(np.uint8(temp_image))
    plt.title(f"rango={rank}")
    plt.imshow(temp_image, cmap="gray")
    plt.show()
```

## Contenido de V

```python
total = np.zeros((163, 372))

for rank in [1, 2, 3, 4, 5]:
    # Extraer los elementos hasta el rango especificado
    U_i = U[:, :rank]
    Sigma_i = np.matrix(linalg.diagsvd(Sigma[:rank], rank, rank))
    VT_i = VT[:rank, :]

    # Poner a 0 todos los valores excepto el singular del rango especificado, dejando solo el elemento del rango actual
    if rank > 1:
        for ri in range(rank - 1):
            Sigma_i[ri, ri] = 0

    # Restaurar la imagen
    temp_image = np.asarray(U_i * Sigma_i * VT_i)
    Image.fromarray(np.uint8(temp_image))

    # Sumar solo el elemento del rango actual
    total += temp_image

    # Comparar la imagen restaurada con los elementos hasta el rango actual y los valores de la columna correspondiente de la matriz V
    plt.figure(figsize=(5, 5))
    plt.suptitle(f"$u_{rank}$")
    plt.subplot(211)
    plt.imshow(temp_image, cmap="gray")
    plt.subplot(212)
    plt.plot(VT[0])
    plt.show()

```


```python
# Confirmar que al sumar los elementos del rango 1 al 5 se puede restaurar correctamente la imagen original
plt.imshow(total)
```
