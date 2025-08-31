---
title: "Sentiment analysis of text"
pre: "7.3.1 "
weight: 1
searchtitle: "Text sentiment analysis in python"
---


We will use the model in『[SiEBERT - English-Language Sentiment Classification](https://www.sciencedirect.com/science/article/pii/S0167811622000477?via%3Dihub)』 to classify the sentiment of English sentences. I would like to classify the sentiment of each English sentence by positive and negative.

{{% notice ref %}}
Hartmann, Jochen and Heitmann, Mark and Siebert, Christian and Schamp, Christina,
"More than a feeling: Accuracy and Application of Sentiment Analysis", International Journal of Research in Marketing(2022)
{{% /notice %}}

Here we use the [siebert/sentiment-roberta-large-english](https://huggingface.co/siebert/sentiment-roberta-large-english?text=I+like+you.+I+love+you) model on huggingface, if you want to use transformers on Google Colab you need to install `transformers` beforehand.

```python
import numpy as np
from transformers import pipeline
from IPython.display import HTML

sentiment_pipeline = pipeline(
    "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
)
```

## Analyze the sentiment  of each sentence in the text

The entire text is separated by "." to separate each sentence.

Here, [Petrobras Webcast – 3rd Quarter Results 2022 November 5, 2022](https://www.investidorpetrobras.com.br/en/results-and-announcements/results-center/) transcription data is used.


```python
transcript = """Hello!Hello!Hello!Hello!Hello!"""
ts_list = [ts for ts in transcript.split(".") if len(ts) > 20]
scores = sentiment_pipeline(ts_list)
```

## Visualize the results

Visualize the results using positive and negative labels and their scores.


```python
for t, s in zip(ts_list, scores):
    score = np.round(float(s["score"]), 4)  # sentiment score
    font_weight = "bold" if score > 0.995 else "normal"  # thickness of the text

    # color display for each sentiment
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

