---
title: "カタカナ・ひらがなを統一する"
pre: "3.7.2 "
weight: 2
title_suffix: "音声検索や名寄せ向けに片仮名⇄平仮名を変換"
---

ふりがなやフードメニューなど、片仮名と平仮名が混在すると集計がずれます。用途に合わせて片仮名に揃えたり、平仮名に落としたりするユーティリティを用意しておきましょう。

```python
import unicodedata

HIRA_START = ord("ぁ")
HIRA_END = ord("ゖ")
KATA_START = ord("ァ")

def kata_to_hira(text: str) -> str:
    chars = []
    for ch in text:
        code = ord(ch)
        if KATA_START <= code <= ord("ヺ"):
            chars.append(chr(code - 0x60))
        else:
            chars.append(ch)
    return "".join(chars)

def hira_to_kata(text: str) -> str:
    chars = []
    for ch in text:
        code = ord(ch)
        if HIRA_START <= code <= HIRA_END:
            chars.append(chr(code + 0x60))
        else:
            chars.append(ch)
    return "".join(chars)

samples = ["ﾌｧﾐﾘｰﾏｰﾄ", "ファミリーマート", "ふぁみりーまーと"]
for s in samples:
    s_nfkc = unicodedata.normalize("NFKC", s)
    print(s_nfkc, "→", kata_to_hira(s_nfkc))
```

NFKC 正規化を挟むと半角カタカナも自動的に全角へ揃えられます。業務システムではカタカナで統一することが多いので、`hira_to_kata` を通してから保存する運用が定番です。

### 追加の工夫
- `ヴ` や `ヵ` など拗音・促音の揺れは別処理が必要です。辞書ベースで例外を作るか、案件に応じた置換テーブルを併用してください。
- 送り仮名付きの漢字（例: `申し込み`）から読みを取り出したい場合、機械学習の前処理として `fugashi` や `SudachiPy` の出力を平仮名へ変換すると安定します。
