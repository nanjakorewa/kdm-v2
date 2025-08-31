---
title: "英語から日本語字幕データを作成"
pre: "7.3.2.2 "
weight: 3
not_use_colab: true
searchtitle: "pythonで英語字幕データを日本語に翻訳する"
---

{{% youtube "1-dsqPNcE2Y" %}}

<div class="pagetop-box">
    <p>Whisperとは、OpenAIが開発しgithub上で公開されている”ウェブから収集した68万時間に及ぶ多言語・マルチタスク教師付きデータで学習させた自動音声認識（ASR）”モデルであり、多言語での書き起こし・多言語から英語への翻訳が可能です。</p>
    <p>このページでは、python上にてwhisperによる音声認識結果を日本語に翻訳し、日本語の字幕データ( .srtフォーマット)を作成します。さらにそれをずんだもんにしゃべらせてみます。</p>
</div>


{{% notice seealso %}}
[VOICEVOX -- 無料で使える中品質なテキスト読み上げソフトウェア](https://voicevox.hiroshiba.jp/)
{{% /notice %}}

## whisperから作成した字幕ファイルを読み込む

whisperはデフォルトで `.srt`ファイルを書き出します。このファイルの英語の部分を日本語に翻訳します。
翻訳の正しさは保証されておらず、また本コードも動作や出力に関しては一切の責任を負えません。

- [openai/whisper](https://github.com/openai/whisper)
- 翻訳に使用するモデル
  - [ニューラル機械翻訳モデルFuguMT](https://staka.jp/wordpress/?p=413)
  - [staka/fugumt-en-ja - Hugging Face](https://huggingface.co/staka/fugumt-en-ja)


{{% notice info %}}
音声データは『[Flex LNG Q3 2022 Key Takeaways](https://www.youtube.com/watch?v=tsU0jebpux0)』の音声を使用しています。
{{% /notice %}}


```python
import warnings
from transformers import pipeline


MAX_LENGTH = 400  # モデルに入力することができる文字数の上限
translator = pipeline("translation", model="staka/fugumt-en-ja")
file = open("FlexLNG_Q3_2022_Key_Takeaways.mp3.srt", "r")
lines = [l.replace("\n", "") for l in file.readlines()]
lines[:15]
```

```
 ['1',
 '00:00:00,000 --> 00:00:13,240',
 "Hi and welcome to FlexLNG's third quarter highlights.",
 '',
 '2',
 '00:00:13,240 --> 00:00:18,120',
 'Revenues for the quarter came in at 91 million in line with previous guidance of approximately',
 '',
 '3',
 '00:00:18,120 --> 00:00:19,520',
 '90 million.',
 '',
 '4',
 '00:00:19,520 --> 00:00:26,020',
 'Earnings were strong, net income and adjusted net income was 47 and 42 million, translating']
```

## 前処理

テキストに前処理を加えたいので、ファイルを保存する前にテキストを書き換えます。

```python
def zundamonize(text: str) -> str:
    """

        語尾をそれらしくする

        Args:
            text (str): 日本語テキスト

        Returns:
            str: 前処理済みの日本語テキスト

    """
    # TODO: 辞書を使って英語をカタカナに変換する
    #       https://github.com/KEINOS/google-ime-user-dictionary-ja-en など
    text = text.replace("Flex", "フレックス")

    if text.endswith("ありがとうございました"):
        return text.replace("ありがとうございました", "ありがとうなのだ。")
    elif text.endswith("しました。"):
        return text.replace("しました。", "したのだ。")
    elif text.endswith("れました。"):
        return text.replace("れました。", "れたのだ。")
    elif text.endswith("れました。"):
        return text.replace("れました。", "れたのだ。")
    elif text.endswith("れます。"):
        return text.replace("れます。", "れるのだ。")
    elif text.endswith("できます。"):
        return text.replace("できます。", "できるのだ。")
    elif text.endswith("あります。"):
        return text.replace("あります。", "あるのだ。")
    elif text.endswith("ようこそ。"):
        return text.replace("ようこそ。", "ようこそなのだ。")
    elif text.endswith("ています。"):
        return text.replace("ています。", "ているのだ。")
    elif text.endswith("ている。"):
        return text.replace("ている。", "ているのだ。")
    elif text.endswith("できる。"):
        return text.replace("できる。", "できるのだ。")
    elif text.endswith("できま。"):
        return text.replace("できます。", "できるのだ。")
    elif text.endswith("できた。"):
        return text.replace("できた。", "できたのだ。")
    elif text.endswith("えました。"):
        return text.replace("えました。", "えたのだ。")
    elif text.endswith("しました。"):
        return text.replace("しました。", "したのだ。")
    elif text.endswith("ました。"):
        return text.replace("ました。", "たのだ。")
    elif text.endswith("った。"):
        return text.replace("った。", "ったのだ。")
    elif text.endswith("した。"):
        return text.replace("した。", "したのだ。")
    elif text.endswith("する。"):
        return text.replace("する。", "するのだ。")
    elif text.endswith("です。"):
        return text.replace("です。", "なのだ。")
    return text
```

## 翻訳したテキストを出力する

字幕用のフォーマット(`.srt`)とテキストファイルの両方を出力します。
途中、タイムスタンプをいろいろ書き換えているのは字幕のタイミングを調整するためです。

{{% notice document %}}
[SubRip file format(.srtファイルの仕様)](https://en.wikipedia.org/wiki/SubRip#SubRip_file_format)
<br />ファイル一行目に改行を入れるとAdobe Premiere Pro等で読み取りに失敗するので注意
{{% /notice %}}

作成された `.srt` ファイルをAdobe Premiere Proに読み込むと字幕を表示できます。
また、`.txt`ファイルをVOICEVOXに読み込めば音声データを作成できます。


```python
cnt = 0
result = []
text_only_result = []
temp_timestamp_start = "00:00:00,000"
temp_timestamp_end = "00:00:00,000"
temp_text = ""
centense_continue = False

for line in lines:
    if len(line) == 0:
        # 空白行はスキップ
        continue
    elif line[0].isdigit():
        # 会話以外の出力はスキップ
        if "-->" in line:
            if centense_continue:
                temp_timestamp_end = line.split(" --> ")[1]
            else:
                temp_timestamp_start, temp_timestamp_end = line.split(" --> ")
            continue
        elif len(line) < 5:
            continue

    # 会話が続いているかの判定
    if line.endswith("."):
        centense_continue = False
    else:
        centense_continue = True

    # 翻訳
    temp_text += line
    if not centense_continue:
        cnt += 1
        translation_text = translator(temp_text)[0]["translation_text"]
        translation_text = zundamonize(translation_text)
        temp_text = ""

        print("")
        print(cnt)
        print(f"{temp_timestamp_start} --> {temp_timestamp_end}")
        print(translation_text)

        if cnt > 1:
            result.append("")
        result.append(cnt)
        result.append(f"{temp_timestamp_start} --> {temp_timestamp_end}")
        result.append(translation_text)
        text_only_result.append(f"{translation_text}")
    else:
        continue

with open("日本語字幕.srt", "w", encoding="utf-8-sig") as f:
    for line in result:
        f.write(f"{line}\n")

with open("日本語テキスト.txt", "w", encoding="utf-8-sig") as f:
    for line in text_only_result:
        f.write(f"{line}\n")
```

```
1
00:00:00,000 --> 00:00:13,240
フレックスLNGの第3四半期ハイライトへようこそなのだ。

2
00:00:13,240 --> 00:00:19,520
この四半期の売上は9100万ドルで、前回のガイダンスでは約9000万ドルだったのだ。

3
00:00:19,520 --> 00:00:33,480
利益は好調で、純利益と調整済み純利益は47と4200万ドルで、1株当たり利益と調整済み1株当たり利益はそれぞれ88セントと79セントだったのだ。

4
00:00:33,480 --> 00:00:41,520
四半期中の貨物市場は急成長し、これは短期と長期の両方にプラスの影響を与えたのだ。

5
00:00:41,520 --> 00:00:45,100
この四半期には3隻の船が新造船のチャーターを開始したのだ。

6
00:00:45,100 --> 00:00:57,720
6月には、フレックス Enterpriseとフレックス Amberの両方の7年間のチャーターを発表し、これらの船は、短期契約の代わりに、7月にこれらの新しい長期チャーターを開始したのだ。
```