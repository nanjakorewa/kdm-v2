---
title: "和暦を西暦へ変換する"
pre: "3.7.5 "
weight: 5
title_suffix: "令和・平成などの年号を年次データに統合"
---

自治体のオープンデータや社内の申請書では「令和5年」「平成28年度」のように和暦が多く登場します。西暦に直しておくとタイムライン解析や季節性の把握が楽になります。

```python
import re

ERA_TABLE = {
    "令和": 2018,  # 令和元年 = 2019
    "平成": 1988,
    "昭和": 1925,
    "大正": 1911,
    "明治": 1867,
}

ERA_PATTERN = re.compile(r"(明治|大正|昭和|平成|令和)(元|\d{1,2})年?")


def wareki_to_western(text: str) -> int | None:
    match = ERA_PATTERN.search(text)
    if not match:
        return None
    era, year = match.groups()
    base_year = ERA_TABLE[era]
    year_num = 1 if year == "元" else int(year)
    return base_year + year_num


records = ["令和5年度決算", "平成28年4月入社", "昭和60年生まれ", "西暦2020年"]
for r in records:
    converted = wareki_to_western(r)
    print(r, "→", converted if converted else "西暦表記")
```

### 実務向けの補足
- 和暦と西暦が混在する場合は、変換後も元の表記を保持するカラムを作り、監査ログで追えるようにします。
- 明治以前の年号が必要な場合は `ERA_TABLE` に追記してください（古い戸籍や歴史資料の処理で利用します）。
- 「R05」「H28」のような略記には追加の正規表現が必要です。部署やシステムごとのクセをヒアリングして、バリエーションを網羅するテーブルを育てていきましょう。
