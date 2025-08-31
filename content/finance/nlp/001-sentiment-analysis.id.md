---
title: "Analisis sentimen teks"
pre: "7.3.1 "
weight: 1
searchtitle: "Analisis sentimen teks dalam python"
---


Kami akan menggunakan model dalam 『[SiEBERT - English-Language Sentiment Classification](https://www.sciencedirect.com/science/article/pii/S0167811622000477?via%3Dihub)』 untuk mengklasifikasikan sentimen kalimat bahasa Inggris. Saya ingin mengklasifikasikan sentimen setiap kalimat bahasa Inggris dengan positif dan negatif.

{{% notice ref %}}
Hartmann, Jochen and Heitmann, Mark and Siebert, Christian and Schamp, Christina,
"More than a feeling: Accuracy and Application of Sentiment Analysis", International Journal of Research in Marketing(2022)
{{% /notice %}}

Di sini kami menggunakan model [siebert/sentiment-roberta-large-english](https://huggingface.co/siebert/sentiment-roberta-large-english?text=I+like+you.+I+love+you) pada huggingface, jika Anda ingin menggunakan transformer pada Google Colab, Anda perlu menginstal `transformers` terlebih dahulu.

```python
import numpy as np
from transformers import pipeline
from IPython.display import HTML

sentiment_pipeline = pipeline(
    "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
)
```

## Menganalisis sentimen dari setiap kalimat dalam teks

Seluruh teks dipisahkan dengan "." untuk memisahkan setiap kalimat.

Di sini, [Petrobras Webcast - 3rd Quarter Results 2022 November 5, 2022](https://www.investidorpetrobras.com.br/en/results-and-announcements/results-center/) data transkripsi digunakan.


```python
transcript = """Hello!Hello!Hello!Hello!Hello!"""
ts_list = [ts for ts in transcript.split(".") if len(ts) > 20]
scores = sentiment_pipeline(ts_list)
```

## Visualize the results

Visualize the results using positive and negative labels and their scores.


```python
for t, s in zip(ts_list, scores):
    score = np.round(float(s["score"]), 4)  # skor sentimen
    font_weight = "bold" if score > 0.995 else "normal"  # ketebalan teks

    # tampilan warna untuk setiap sentimen
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

