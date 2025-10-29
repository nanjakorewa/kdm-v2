---
title: "々・ゝなどの踊り字を展開する"
pre: "3.7.3 "
weight: 3
title_suffix: "表記揺れを抑えてキーワードマッチの漏れを防ぐ"
---

「時々」「人々」などの踊り字は、機械的な検索ではヒットしにくい表記です。観測値をすべて漢字に展開しておくと、自然言語処理のステップや単純なフィルタでも漏れを減らせます。

```python
def expand_iteration_marks(text: str) -> str:
    result = []
    prev_char = ""
    for ch in text:
        if ch in {"\u3005", "\u303B"} and prev_char:
            result.append(prev_char)
        elif ch in {"\u309D", "\u309E"} and prev_char:
            # ひらがなの繰り返し
            base = prev_char
            if ch == "\u309E":  # ゞ
                base = chr(ord(base) + 1)
            result.append(base)
        elif ch in {"\u30FD", "\u30FE"} and prev_char:
            base = prev_char
            if ch == "\u30FE":  # ヾ
                base = chr(ord(base) + 1)
            result.append(base)
        else:
            result.append(ch)
            prev_char = ch
    return "".join(result)

samples = [
    "時々のことなので人々にも共有する",
    "くつゝいた靴下",
    "サンバゝ←古い仮名遣い",
]

for s in samples:
    print(expand_iteration_marks(s))
```

### 注意点
- `ゝ/ゞ` や `ヽ/ヾ` は歴史的仮名遣いでは濁点を伴う場合があるため、厳密な変換が必要なら辞書を参照して補正します。
- 地名や固有名詞に使われる `々` は必ずしも同じ漢字が繰り返されるとは限りません（例: `佐々木`）。信頼できる正規化には、辞書やドメイン知識と組み合わせることをおすすめします。
