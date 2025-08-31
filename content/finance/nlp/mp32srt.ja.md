---
title: "動画からの音声認識"
pre: "7.3.2.1 "
weight: 2
not_use_colab: true
searchtitle: "whisperで動画からの音声認識をする"
---

{{% youtube "1-dsqPNcE2Y" %}}

<div class="pagetop-box">
    <p>Whisperとは、OpenAIが開発しgithub上で公開されている”ウェブから収集した68万時間に及ぶ多言語・マルチタスク教師付きデータで学習させた自動音声認識（ASR）”モデルであり、多言語での書き起こし・多言語から英語への翻訳が可能です。</p>
    <p>このページでは、海外企業の決算動画を例として、python上にてwhisperを使った音声認識をします。その後、HuggingFace上の英語翻訳モデルを用いて決算動画を日本語にしてみます。</p>
</div>

## 仮想環境の作成

Anacondaを使用している場合は、以下のコマンドを順番に実行することで `py39-whisper` という名前のpython3.9実行環境が作成されます。

    conda create -n py39-whisper python=3.9 anaconda
    conda activate py39-whisper
    pip install git+https://github.com/openai/whisper.git
    conda install ffmpeg -c conda-forge
    conda install jupyter ipykernel pandas
    conda install transformers[sentencepiece]

## whisperを使って音声認識を実行する

{{% notice info %}}
音声データは『[Flex LNG Q3 2022 Key Takeaways](https://www.youtube.com/watch?v=tsU0jebpux0)』の音声を使用しています。
{{% /notice %}}

```python
import warnings
import whisper
from transformers import pipeline


model = whisper.load_model("base")
result = model.transcribe("FlexLNG_Q3_2022_Key_Takeaways.mp3")
print(result["text"])
```

```text
Hi and welcome to FlexLng's TURD Quater Highlights. Revenue's 4D Quater came in at 91 million in line with previous guidance of approximately 90 million. Ernings was strong, net income and adjusted net income was 47.42 million, translating into Ernings per share and adjusted Ernings per share of 88.79 respectively. Freight market during the quarter boomed and this affected both short term and long term rates positively. During the quarter we had three ships...
```

## 日本語を英語に翻訳する

翻訳の正しさは保証されておらず、また本コードも動作や出力に関しては一切の責任を負えません。

- [openai/whisper](https://github.com/openai/whisper)
- 翻訳に使用するモデル
  - [ニューラル機械翻訳モデルFuguMT](https://staka.jp/wordpress/?p=413)
  - [staka/fugumt-en-ja - Hugging Face](https://huggingface.co/staka/fugumt-en-ja)


```python
MAX_LENGTH = 400  # モデルに入力することができる文字数の上限
translator = pipeline("translation", model="staka/fugumt-en-ja")
translated_text = []

for t in result["text"].split(". "):
    translated_text.append(translator(t[:MAX_LENGTH])[0]["translation_text"])

print(translated_text)
```

```
['Flexingの第3四半期ハイライトへようこそ。',
 'Revenueの4D Quaterは9100万で、前回のガイダンスで約9000万だった。',
 'Erningsは好調で、純利益と調整済み純利益は47.42百万で、それぞれ1株当たりErningsに翻訳され、調整済みErningsは88.79だった。',
 '第4四半期の貨物市場は急成長し、短期と長期の両方にプラスの影響を与えました。',
 '四半期中は3隻の船が新船のチャーターを開始し',
 '6月にフレックスと価格とフレックスアンバーの両方の7年間のチャーターを発表し、これらの船は短期契約の代わりに7月にこれらの新しい長期チャーターを開始しました。',
 'また、第4四半期末には、シェニエとの契約により、最終第5船としてフレクサーオーラをチェニエに納入しました。',
 'CFOの四半期中は、バランスシート最適化プログラムフェーズ2の下で再融資に忙しかったため、追加の1億ドルの現金を調達する目標がありました。この6億3000万の資金を4隻の船舶に調達することで、すでに1億1000万の現金リリースを確保しています。',
 'また、新たに3隻の船舶をリファイナンスし、バランスシート最適化プログラムの目標を3億に引き上げることはできません。',
 'フェーズ1では1億3700万ドルを',
 '本日 110を発売すると発表しました',
```