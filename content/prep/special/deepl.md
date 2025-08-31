---
title: "deeplで論文を翻訳"
pre: "3.6.1 "
weight: 1
title_replace: "pythonでdeeplのAPIを使って論文を翻訳してみる"
---

{{% youtube "_5j8KMIb0qc" %}}

## arXivから論文を取得する
以下の論文をarXivからダウンロードして、概要を翻訳します。
断りがない限り、コード中に出現する英文は以下の論文の「Abstract」の英文の一部です。

Vaswani, Ashish, et al. "[Attention is all you need.](https://arxiv.org/pdf/1706.03762.pdf)" Advances in neural information processing systems 30 (2017).


```python
import arxiv

search = arxiv.Search(id_list=["1706.03762"])
paper = next(search.results())
print(f"論文タイトル：{paper.title}")
```

    論文タイトル：Attention Is All You Need



```python
pdf_path = paper.download_pdf()
print(f"pdf保存先：{pdf_path}")
```

    pdf保存先：./1706.03762v5.Attention_Is_All_You_Need.pdf


## pdfからテキストを抽出する


```python
import fitz

abstract_text = ""
with fitz.open(pdf_path) as pages:
    first_page = pages[0]
    text = first_page.get_text().replace("\n", "")
    print(f'アブスト開始位置：{text.find("Abstract")}, イントロ開始位置：{text.find("Introduction")}')
    abstract_text = text[text.find("Abstract") + 8 : text.find("Introduction") - 1]

print(f"{abstract_text[:400]}...")
```

    アブスト開始位置：394, イントロ開始位置：1528
    The dominant sequence transduction models are based on complex recurrent orconvolutional neural networks that include an encoder and a decoder. The bestperforming models also connect the encoder and decoder through an attentionmechanism. We propose a new simple network architecture, the Transformer,based solely on attention mechanisms, dispensing with recurrence and convolutionsentirely. Experimen...


## deepl-pythonを使って英語を翻訳する


```python
import deepl
import os

translator = deepl.Translator(os.getenv("DEEPL_AUTH_KEY"))
result = translator.translate_text("Good morning!", source_lang="EN", target_lang="JA")
print(result)
```

    おはようございます。



```python
result = translator.translate_text(abstract_text, source_lang="EN", target_lang="JA")
print(result)
```

    優性配列変換モデルは、エンコーダとデコーダを含む複雑なリカレントニューラルネットワークまたは畳み込みニューラルネットワークをベースにしています。また、最も優れたモデルでは、エンコーダとデコーダをアテンションメカニズムで接続しています。本研究では、再帰や畳み込みを必要とせず、注目メカニズムのみに基づいた新しいシンプルなネットワークアーキテクチャ「Transformer」を提案する。2つの機械翻訳タスクを用いた実験では、これらのモデルが優れた品質を持ち、並列化が可能で学習時間が大幅に短縮されることが示された。WMT 2014の英独翻訳タスクにおいて、我々のモデルは28.4 BLEUを達成し、アセンブルを含む既存の最良の結果よりも2 BLEU以上向上した。WMT 2014の英仏翻訳タスクでは、8つのGPUを用いて3.5日間の学習を行った結果、41.8という最新のBLEUスコアを達成しましたが、これは文献に掲載されている最高のモデルの学習コストのごく一部です。また、大規模および限定的な学習データを用いた英語の構文解析に適用することで、Transformerが他のタスクにもよく適応することを示しました。

