---
title: "漢数字をアラビア数字へ変換する"
pre: "3.7.4 "
weight: 4
title_suffix: "万・億・兆までを安全にパースして数値化"
---

帳票やアンケートでは「三千五百二十円」や「約２億円」のように漢数字が混在します。機械学習で数値として扱うには、漢数字をアラビア数字へ変換するユーティリティを用意しておくと便利です。

```python
KANJI_DIGITS = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}
KANJI_UNITS = {"十": 10, "百": 100, "千": 1000}
KANJI_BIG_UNITS = {"万": 10_000, "億": 100_000_000, "兆": 1_000_000_000_000}


def kanji_to_int(text: str) -> int:
    text = text.replace("約", "").replace("およそ", "")
    total = 0
    section = 0
    number = 0

    for ch in text:
        if ch in KANJI_DIGITS:
            number = KANJI_DIGITS[ch]
        elif ch in KANJI_UNITS:
            unit = KANJI_UNITS[ch]
            section += max(number, 1) * unit
            number = 0
        elif ch in KANJI_BIG_UNITS:
            unit = KANJI_BIG_UNITS[ch]
            section += number
            total += max(section, 1) * unit
            section = 0
            number = 0
        elif ch.isdigit():
            number = number * 10 + int(ch)
        else:
            continue

    return total + section + number


samples = ["三千五百二十円", "１２億３０００万", "4兆5000億", "約五万二千"]
for s in samples:
    print(s, "→", kanji_to_int(s))
```

### 運用メモ
- 「億」より大きい単位（京、垓など）が出る業種では `KANJI_BIG_UNITS` を拡張してください。
- 「数十」「数百」といった曖昧表現は 0 〜 9 の仮の値で置き換えるか、別カラムとして保持する運用が安全です。
- カンマ付きのアラビア数字 (`1,234`) と漢数字が混在している場合は、正規表現でアラビア数字を先に数値化すると処理がシンプルになります。
