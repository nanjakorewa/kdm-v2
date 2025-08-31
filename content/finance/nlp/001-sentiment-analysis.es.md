---
title: "Análisis del sentimiento del texto"
pre: "7.3.1 "
weight: 1
searchtitle: "Análisis del sentimiento del texto en python"
---


Utilizaremos el modelo de『[SiEBERT - English-Language Sentiment Classification](https://www.sciencedirect.com/science/article/pii/S0167811622000477?via%3Dihub)』 para clasificar el sentimiento de las frases en inglés. Me gustaría clasificar el sentimiento de cada frase en inglés por positivo y negativo.

{{% notice ref %}}
Hartmann, Jochen and Heitmann, Mark and Siebert, Christian and Schamp, Christina,
"More than a feeling: Accuracy and Application of Sentiment Analysis", International Journal of Research in Marketing(2022)
{{% /notice %}}

Aquí usamos el modelo [siebert/sentiment-roberta-large-english](https://huggingface.co/siebert/sentiment-roberta-large-english?text=I+like+you.+I+love+you) en huggingface, si quieres usar transformadores en Google Colab necesitas instalar `transformers` de antemano.

```python
import numpy as np
from transformers import pipeline
from IPython.display import HTML

sentiment_pipeline = pipeline(
    "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
)
```

## Analizar el sentimiento de cada frase del texto

Todo el texto está separado por "." para separar cada frase.

En este caso, se utilizan los datos de la transcripción [Petrobras Webcast - Resultados del 3er Trimestre 2022 5 de noviembre de 2022](https://www.investidorpetrobras.com.br/en/results-and-announcements/results-center/).


```python
transcript = """Hello!Hello!Hello!Hello!Hello!"""
ts_list = [ts for ts in transcript.split(".") if len(ts) > 20]
scores = sentiment_pipeline(ts_list)
```

## Visualizar los resultados

Visualice los resultados utilizando etiquetas positivas y negativas y sus puntuaciones.


```python
for t, s in zip(ts_list, scores):
    score = np.round(float(s["score"]), 4)  #  puntuación de sentimiento
    font_weight = "bold" if score > 0.995 else "normal"  # espesor del texto

    if s["label"] == "NEGATIVE":
        r = 255 - 10 * int(1000 - score * 1000)
        display(
            HTML(
                f"[score={score}] <span style='color:rgb({r},100,100);font-weight:{font_weight};'>{t}</span>"
            )
        )
    elif s["label"] == "POSITIVE":
        g = 255 - 10 * int(1000 - score * 1000)
        display(
            HTML(
                f"[score={score}] <span style='color:rgb(100,{g},100);font-weight:{font_weight};'>{t}</span>"
            )
        )
```


[score=0.9976] <span style='color:rgb(100,235,100);font-weight:bold;'>Hello!Hello!Hello!Hello!Hello!</span>

