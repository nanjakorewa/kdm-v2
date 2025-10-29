---
title: "氏名から敬称やカッコ書きを除去する"
pre: "3.7.7 "
weight: 7
title_suffix: "顧客名寄せに向けたクリーニング"
---

顧客名や担当者名には「田中太郎 様」「佐藤花子（新担当）」のように敬称や補足情報が付くことがよくあります。名寄せや照合をスムーズに行うために、敬称・注釈を取り除きましょう。

```python
import re

HONORIFICS = ["様", "さま", "さん", "ちゃん", "殿", "どの", "氏"]
PAREN_PATTERN = re.compile(r"（.*?）|\(.*?\)|\[.*?\]")

def clean_name(name: str) -> str:
    name = PAREN_PATTERN.sub("", name)
    for honorific in HONORIFICS:
        if name.endswith(honorific):
            name = name[: -len(honorific)]
            break
    name = name.replace("　", " ").strip()
    return re.sub(r"\s+", " ", name)

samples = [
    "田中太郎 様",
    "佐藤花子（新担当）",
    "山本-次郎さん",
    "㈱ABC　営業本部　鈴木次長",
]

for s in samples:
    print(s, "→", clean_name(s))
```

### 運用上の注意
- 役職（「部長」「課長」など）を残したい場合はリストを分けておき、別カラムへ保持します。
- カタカナやローマ字で記載された敬称（例: `SAMA`）が混じる場合は、全角・半角正規化と合わせてルールを拡張します。
- 企業名と担当者名が同じカラムに入っている場合は、敬称除去 → 正規表現で人名っぽさを判定 → 人名だけ抽出、という流れでフィルタするのが実務では定番です。
