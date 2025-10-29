---
title: "ルビ付きテキストから読み仮名を抽出・除去する"
pre: "3.7.9 "
weight: 9
title_suffix: "括弧や HTML のルビタグを正規表現で処理"
---

ニュース記事や議事録では「データ（データ）」のように読み仮名が併記されていることがあります。読みだけを別カラムに保存したり、文章から削って形態素解析を安定させたりする用途で活用できます。

```python
import re

PAREN_RUBY = re.compile(r"([一-龥々〆ヵヶ]+)（([ぁ-んァ-ン]+)）")
HTML_RUBY = re.compile(r"<ruby>(.+?)<rt>(.+?)</rt></ruby>")

def extract_ruby(text: str) -> tuple[str, list[tuple[str, str]]]:
    readings: list[tuple[str, str]] = []

    def replace_paren(match: re.Match[str]) -> str:
        kanji, ruby = match.groups()
        readings.append((kanji, ruby))
        return kanji

    text = PAREN_RUBY.sub(replace_paren, text)

    def replace_html(match: re.Match[str]) -> str:
        kanji, ruby = match.groups()
        readings.append((kanji, ruby))
        return kanji

    text = HTML_RUBY.sub(replace_html, text)
    return text, readings


sample = "機械学習（きかいがくしゅう）を支える<ruby>教師あり学習<rt>きょうしあり</rt></ruby>の基礎"
cleaned, ruby_list = extract_ruby(sample)
print(cleaned)
print(ruby_list)
```

### 応用アイデア
- 読み仮名を別フィールドに分離したうえで、検索時には漢字・かな・ローマ字すべてに対応する逆引き辞書を作ると UX が向上します。
- 電子書籍の EPUB などでは `<rb>` `<rt>` `<rp>` タグの組み合わせが使われるため、XPath や BeautifulSoup でタグを抽出する方法も検討してください。
- 文書全体の読み仮名比率を算出し、ルビの密度が高いページを優先翻訳する、といったワークフローにも応用できます。
