---
title: Introducción
weight: 1
pre: "<b>1. </b>"
chapter: true
not_use_colab: true
not_use_twitter: true
---

### Sección 1
# ¿Qué es esta página?

Este sitio es una página donde _k (creador del sitio) deja un vlog y un cuaderno jupyter de lo que estudió.


## Cómo ejecutar el código

{{< tabs >}}
{{% tab name="Google Colaboratory(Colab)" %}}
Google Colaboratory(**Colab**) es el nombre del entorno de ejecución de Python proporcionado por Google que se ejecuta en el navegador. Colab permite ejecutar y compartir fácilmente el código sin necesidad de construir ningún entorno virtual. Se requiere una cuenta de Google para utilizar este servicio.

Haz clic en el botón<a href="#" class="btn colab-btn-border in-text-button">Colab</a> para ejecutar el código de esa página en Google Colaboratory.

![](2022-03-30-20-15-24.png)

{{% /tab %}}
{{% tab name="Anaconda" %}}

Después de crear un entorno virtual con Anaconda, instale los paquetes de bibliotecas necesarios para cada cuaderno y código. Sin embargo, tenga en cuenta que este sitio no utiliza Anaconda en absoluto y no ha sido probado.

```
conda create -n py38_env python=3.8
conda install *library_name*
```

{{% /tab %}}
{{< /tabs >}}