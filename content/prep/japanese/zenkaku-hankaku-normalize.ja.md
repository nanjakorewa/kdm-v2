---
title: "全角・半角のゆれを正規化する"
pre: "3.7.1 "
weight: 1
title_suffix: "NFKC と置換テーブルで日本語入力のばらつきを抑える"
---

日本語の入力フォームから集めたデータには、`ＡＢＣ` のような全角英数字や `－` と `ー` が混在するケースがよくあります。テキストを正規化しておくと、検索やグルーピングのヒット率が大きく向上します。

```python
import unicodedata

TRANSLATION_TABLE = str.maketrans(
    {
        "ー": "-",  # 長音記号をハイフンへ統一
        "～": "~",
        "―": "-",
        "’": "'",
        "”": '"',
    }
)

def normalize_ja(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(TRANSLATION_TABLE)
    text = text.replace("\u3000", " ").replace("　", " ")  # 全角スペース
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()

samples = [
    "ＡＢＣ　株式会社―営業部",
    "abc㈱‐営業部",
    "ＡＢＣ−営業部",
]

for s in samples:
    print(normalize_ja(s))
```

NFKC 正規化だけでは吸収できない長音やダッシュ類は、追加の変換テーブルで吸収します。ブランド名などでハイフンが重要な場合は、翻訳テーブルを業務のルールに合わせて調整してください。

### 運用のヒント
- 全角スペースは `strip()` だけでは落ちないので、明示的に半角へ統一しておきます。
- 機種依存文字（例: `①` や `㈱`）は NFKC で通常の数字・括弧に展開されるため、識別子に組み込める形へ整形できます。
- 日付や数値カラムでも全角数字が混ざることがあるので、数値変換前にこの関数を噛ませるとエラーが減ります。
