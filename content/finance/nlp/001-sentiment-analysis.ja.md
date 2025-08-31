---
title: "テキストの感情分析"
pre: "7.3.1 "
weight: 1
searchtitle: "pythonでテキストの感情分析をしてみよう！"
---

{{% youtube "aEFJUihC81k" %}}

『[SiEBERT - English-Language Sentiment Classification](https://www.sciencedirect.com/science/article/pii/S0167811622000477?via%3Dihub)』のモデルを使って英文の感情分類をします。英文の各文章の感情をポジティブ・ネガティブで分類してみたいと思います。

{{% notice ref %}}
Hartmann, Jochen and Heitmann, Mark and Siebert, Christian and Schamp, Christina,
"More than a feeling: Accuracy and Application of Sentiment Analysis", International Journal of Research in Marketing(2022)
{{% /notice %}}

ここではhuggingface上の[siebert/sentiment-roberta-large-english](https://huggingface.co/siebert/sentiment-roberta-large-english?text=I+like+you.+I+love+you)のモデルを使用しています。Google Colab上でtransformersを使用する場合は事前に`transformers`をインストールする必要があります。

```python
import numpy as np
from transformers import pipeline
from IPython.display import HTML

sentiment_pipeline = pipeline(
    "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
)
```

## テキストの各文の感情を分析してみる

テキスト全体を「.」で区切ることで、一文ごとに分けています。

ここでは[Petrobras Webcast – 3rd Quarter Results 2022 November 5, 2022](https://www.investidorpetrobras.com.br/en/results-and-announcements/results-center/)の文字起こしデータを使用しています。


```python
transcript = """Hello!Hello!Hello!Hello!Hello!"""
ts_list = [ts for ts in transcript.split(".") if len(ts) > 20]
scores = sentiment_pipeline(ts_list)
```

## 結果を可視化

ポジティブ・ネガティブのラベルと、そのスコアを用いて結果を可視化してみます。


```python
for t, s in zip(ts_list, scores):
    score = np.round(float(s["score"]), 4)  # 感情スコア
    font_weight = "bold" if score > 0.995 else "normal"  # 表示する文字の太さ

    # 感情ごとに色を分けて表示
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

